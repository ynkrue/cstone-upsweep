/*! @file
 * @brief test globalUpsweep against a single-rank upsweep of the global tree
 *
 * Usage: test_global_exchange [L=3] [theta=0.5] [n=32] [reps=10]
 */

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

#include <mpi.h>

#include "cstone/domain/domaindecomp.hpp"
#include "cstone/focus/source_center.hpp"
#include "cstone/sfc/sfc.hpp"
#include "cstone/traversal/macs.hpp"
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

    int      numLevels  = argc > 1 ? std::atoi(argv[1]) : 3;      // L, 8^L global leaves
    float    theta      = argc > 3 ? std::atof(argv[2]) : 0.5f;   // MAC opening angle
    unsigned numPerCell = argc > 2 ? std::atoi(argv[3]) : 32;     // particles per global leaf
    int      numReps    = argc > 4 ? std::atoi(argv[4]) : 10;     // timed calls after one warm-up call

    //! regular global tree with L levels
    std::vector<KeyType> gLeaves = cstone::makeNLevelGrid<KeyType>(numLevels);
    gLeaves.push_back(cstone::nodeRange<KeyType>(0));
    std::vector<unsigned> gCounts(cstone::nNodes(gLeaves), 32);

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

    //! upsweep lists
    double     tSetup  = MPI_Wtime();
    SweepLists upLists = computeUpsweepLists(gOctree, assignment, myRank);

    //! flags: global nodes that fail the minMAC against one of my leaves, as in FocusedOctree::updateMacs
    auto                         parent = [&](TreeNodeIndex c) { return gOctree.parents[(c - 1) / 8]; };
    cstone::Box<T>               box(0, 1);
    std::vector<cstone::Vec4<T>> centers(gOctree.numNodes);
    cstone::geoMacSpheres(std::span<const KeyType>(gOctree.prefixes, gOctree.numNodes), centers.data(),
                          cstone::invThetaMinMac(theta), box);
    std::vector<uint8_t> flags(gOctree.numNodes, 0);
    cstone::markMacs(gOctree.prefixes, gOctree.childOffsets, gOctree.parents, centers.data(), box,
                     gLeaves.data() + globDispl[myRank], globDispl[myRank + 1] - globDispl[myRank], false, flags.data());
    tSetup = MPI_Wtime() - tSetup;

    //! point-to-point upsweep + downsweep
    TreeNodeIndex             globNumNodes = globDispl[myRank + 1] - globDispl[myRank];
    std::span<const unsigned> gLeafQLoc(gLeafQ.data() + globDispl[myRank], globNumNodes);
    std::vector<unsigned>     globQ(gOctree.numNodes);
    cstone::scatter(gOctree.leafToInternalSpan().subspan(globDispl[myRank], gLeafQLoc.size()), gLeafQLoc.data(), globQ.data());
    for (int r = 0; r <= numReps; ++r)
    {
        if (r == 1) { upsweepTimes = {}; } // call 0 is the warm-up
        globalUpsweep(gOctree, upLists, flags, assignment, std::span<unsigned>(globQ), MPI_COMM_WORLD, upsweep);
    }

    //! comparison: Allgatherv of the global leaves + upsweep of the whole tree, as in FocusedOctree::globalExchange
    std::vector<int> globCounts(numRanks);
    for (int r = 0; r < numRanks; ++r)
    {
        globCounts[r] = globDispl[r + 1] - globDispl[r];
    }
    std::vector<unsigned> gLeafQAll(gOctree.numLeafNodes), globQAll(gOctree.numNodes);
    double                tAllgatherv = 0, tFullUpsweep = 0;
    for (int r = 0; r <= numReps; ++r)
    {
        if (r == 1) { tAllgatherv = tFullUpsweep = 0; } // call 0 is the warm-up
        double t0 = MPI_Wtime();
        mpiAllgatherv(gLeafQLoc.data(), globNumNodes, gLeafQAll.data(), globCounts.data(), globDispl.data(),
                              MPI_COMM_WORLD);
        double t1 = MPI_Wtime();
        cstone::scatter(gOctree.leafToInternalSpan(), gLeafQAll.data(), globQAll.data());
        upsweep(gOctree.levelRangeSpan(), gOctree.childOffsets, globQAll.data());
        tAllgatherv += t1 - t0;
        tFullUpsweep += MPI_Wtime() - t1;
    }

    //! check: every node I own and every child of a flagged node equals the single-rank reference
    int numErrors = 0;
    for (TreeNodeIndex i = 0; i < gOctree.numNodes; ++i)
    {
        bool held = nodeOwner(gOctree, assignment, i) == myRank || (i > 0 && flags[parent(i)]);
        if (held && globQ[i] != globQRef[i]) { ++numErrors; }
    }
    MPI_Allreduce(MPI_IN_PLACE, &numErrors, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (myRank == 0) { std::printf("upsweep: %s (%d errors)\n", numErrors == 0 ? "PASS" : "FAIL", numErrors); }

    //! times in ms, max over ranks, per call
    double times[7] = {tSetup,
                       upsweepTimes.request / numReps,
                       upsweepTimes.upExchange / numReps,
                       upsweepTimes.upCompute / numReps,
                       upsweepTimes.downExchange / numReps,
                       tAllgatherv / numReps,
                       tFullUpsweep / numReps};
    MPI_Allreduce(MPI_IN_PLACE, times, 7, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (myRank == 0)
    {
        std::printf("time [ms] L %-3d P %-3d:  total %.3f | setup %.3f | request %.3f | up exchange %.3f | up compute %.3f | "
                    "down exchange %.3f || allgather total %.3f | allgatherv %.3f | full upsweep %.3f\n",
                    numLevels, numRanks, 1e3 * (times[0] + times[1] + times[2] + times[3] + times[4]), 1e3 * times[0], 1e3 * times[1], 1e3 * times[2], 1e3 * times[3],
                    1e3 * times[4], 1e3 * (times[5] + times[6]), 1e3 * times[5], 1e3 * times[6]);
    }

    MPI_Finalize();
    // return numErrors != 0;
    return 0;
}
