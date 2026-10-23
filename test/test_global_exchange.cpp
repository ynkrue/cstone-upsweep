/*! @file
 * @brief test globalUpsweep against a single-rank upsweep of the global tree
 *
 * Usage: test_global_exchange [L=3] [n=32] [reps=10]
 */

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

#include <mpi.h>

#include "cstone/domain/domaindecomp.hpp"
#include "cstone/sfc/sfc.hpp"
#include "cstone/tree/cs_util.hpp"
#include "cstone/tree/csarray.hpp"

#include "upsweep_p2p.hpp"

using namespace upsweep_p2p;

int main(int argc, char** argv)
{
    using KeyType = unsigned;

    MPI_Init(&argc, &argv);
    int myRank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    int      numLevels  = argc > 1 ? std::atoi(argv[1]) : 3;  // L, 8^L global leaves
    unsigned numPerCell = argc > 2 ? std::atoi(argv[2]) : 32; // particles per global leaf
    int      numReps    = argc > 3 ? std::atoi(argv[3]) : 10; // timed calls after one warm-up call

    //! regular global tree with L levels
    std::vector<KeyType> gLeaves = cstone::makeNLevelGrid<KeyType>(numLevels);
    gLeaves.push_back(cstone::nodeRange<KeyType>(0));
    std::vector<unsigned> gCounts(cstone::nNodes(gLeaves), numPerCell);

    //! global tree, distributed across ranks
    cstone::OctreeData<KeyType, cstone::execution::Cpu> globalTree;
    globalTree.resize(cstone::nNodes(gLeaves));
    cstone::updateInternalTree<KeyType>(gLeaves, globalTree.data());
    auto gOctree   = globalTree.cdata();
    gOctree.leaves = gLeaves.data();

    auto                           assignment = cstone::makeSfcAssignment(numRanks, gCounts, gLeaves.data());
    std::span<const TreeNodeIndex> globDispl  = assignment.treeOffsetsConst();

    //! leaf quantity with distinct values
    std::vector<unsigned> gLeafQ(cstone::nNodes(gLeaves));
    std::iota(gLeafQ.begin(), gLeafQ.end(), 1u);

    auto upsweep = [](auto levelRange, auto childOffsets, auto Q)
    { cstone::upsweep(levelRange, childOffsets, Q, cstone::NodeCount<unsigned>{}); };

    //! reference upsweep of all global leaves
    std::vector<unsigned> globQRef(gOctree.numNodes);
    cstone::scatter(gOctree.leafToInternalSpan(), gLeafQ.data(), globQRef.data());
    upsweep(gOctree.levelRangeSpan(), gOctree.childOffsets, globQRef.data());

    TreeNodeIndex             globNumNodes = globDispl[myRank + 1] - globDispl[myRank];
    std::span<const unsigned> gLeafQLoc(gLeafQ.data() + globDispl[myRank], globNumNodes);
    std::vector<unsigned>     globQ(gOctree.numNodes);
    cstone::scatter(gOctree.leafToInternalSpan().subspan(globDispl[myRank], gLeafQLoc.size()), gLeafQLoc.data(),
                    globQ.data());

    //! p2p per call: flags, lists, upsweep + downsweep; call 0 is the warm-up
    auto                 parent = [&](TreeNodeIndex c) { return gOctree.parents[(c - 1) / 8]; };
    std::vector<uint8_t> flags(gOctree.numNodes);
    double               times[4] = {0, 0, 0, 0}; // flags, lists, upsweep, allgather [s per call]
    for (int r = 0; r <= numReps; ++r)
    {
        //! flags, overlap case: the ancestors of my leaves
        double t0 = MPI_Wtime();
        std::fill(flags.begin(), flags.end(), 0);
        for (TreeNodeIndex i = globDispl[myRank]; i < globDispl[myRank + 1]; ++i) // walk through my leaves
        {
            for (TreeNodeIndex n = gOctree.leafToInternalSpan()[i]; n != 0; n = parent(n)) // walk through ancestors
            {
                flags[parent(n)] = 1;
            }
        }
        double t1 = MPI_Wtime();
        auto [upLists, downLists]   = computeSweepLists(gOctree, assignment, myRank);
        double t2 = MPI_Wtime();
        globalUpsweep(gOctree, upLists, flags, assignment, std::span<unsigned>(globQ), MPI_COMM_WORLD, upsweep);
        double t3 = MPI_Wtime();

        if (r == 0) { continue; }
        times[0] += (t1 - t0) / numReps;
        times[1] += (t2 - t1) / numReps;
        times[2] += (t3 - t2) / numReps;
    }

    //! reference: Allgatherv of the global leaves + upsweep of the whole tree, as in FocusedOctree::globalExchange
    std::vector<int> globCounts(numRanks);
    for (int r = 0; r < numRanks; ++r)
    {
        globCounts[r] = globDispl[r + 1] - globDispl[r];
    }
    std::vector<unsigned> gLeafQAll(gOctree.numLeafNodes), globQAll(gOctree.numNodes);
    for (int r = 0; r <= numReps; ++r)
    {
        double t0 = MPI_Wtime();
        mpiAllgatherv(gLeafQLoc.data(), globNumNodes, gLeafQAll.data(), globCounts.data(), globDispl.data(),
                      MPI_COMM_WORLD);
        cstone::scatter(gOctree.leafToInternalSpan(), gLeafQAll.data(), globQAll.data());
        upsweep(gOctree.levelRangeSpan(), gOctree.childOffsets, globQAll.data());
        if (r > 0) { times[3] += (MPI_Wtime() - t0) / numReps; }
    }

    //! check: every node I own and every child of a flagged node equals the single-rank reference
    int numErrors = 0;
    for (TreeNodeIndex i = 0; i < gOctree.numNodes; ++i)
    {
        bool held = nodeOwner(gOctree, assignment, i) == myRank || (i > 0 && flags[parent(i)]);
        if (held && globQ[i] != globQRef[i]) { ++numErrors; }
    }
    MPI_Allreduce(MPI_IN_PLACE, &numErrors, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    //! times in ms per call, max over ranks
    MPI_Allreduce(MPI_IN_PLACE, times, 4, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (myRank == 0)
    {
        std::printf("upsweep: %s (%d errors)\n", numErrors == 0 ? "PASS" : "FAIL", numErrors);
        std::printf("time [ms] L %d P %d: flags %.3f | lists %.3f | upsweep %.3f || allgather %.3f\n", numLevels,
                    numRanks, 1e3 * times[0], 1e3 * times[1], 1e3 * times[2], 1e3 * times[3]);
    }

    MPI_Finalize();
    return numErrors != 0;
}
