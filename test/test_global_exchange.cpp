/*! @file
 * @brief test globalUpsweep against a single-rank upsweep of the global tree
 *
 * Usage: test_global_exchange [L=3] [n=32]
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
    using KeyType = uint64_t;

    MPI_Init(&argc, &argv);
    int myRank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    int      numLevels  = argc > 1 ? std::atoi(argv[1]) : 3;      // L, 8^L global leaves
    unsigned numPerCell = argc > 2 ? std::atoi(argv[2]) : 32;     // particles per global leaf

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

    //! upsweep lists
    SweepLists upLists = computeUpsweepLists(gOctree, assignment, myRank);

    //! flags, overlap case: the ancestors of my leaves
    auto                 parent = [&](TreeNodeIndex c) { return gOctree.parents[(c - 1) / 8]; };
    std::vector<uint8_t> flags(gOctree.numNodes, 0);
    for (TreeNodeIndex i = globDispl[myRank]; i < globDispl[myRank + 1]; ++i) // walk through my leaves
    {
        for (TreeNodeIndex n = gOctree.leafToInternalSpan()[i]; n != 0; n = parent(n)) // walk through ancestors
        {
            flags[parent(n)] = 1;
        }
    }

    //! point-to-point upsweep + downsweep
    TreeNodeIndex             globNumNodes = globDispl[myRank + 1] - globDispl[myRank];
    std::span<const unsigned> gLeafQLoc(gLeafQ.data() + globDispl[myRank], globNumNodes);
    std::vector<unsigned>     globQ(gOctree.numNodes);
    cstone::scatter(gOctree.leafToInternalSpan().subspan(globDispl[myRank], gLeafQLoc.size()), gLeafQLoc.data(), globQ.data());
    globalUpsweep(gOctree, upLists, flags, assignment, std::span<unsigned>(globQ), MPI_COMM_WORLD, upsweep);

    //! check: every node I own and every child of a flagged node equals the single-rank reference
    int numErrors = 0, numHeld = 0;
    for (TreeNodeIndex i = 0; i < gOctree.numNodes; ++i)
    {
        bool held = nodeOwner(gOctree, assignment, i) == myRank || (i > 0 && flags[parent(i)]);
        numHeld += held;
        if (held && globQ[i] != globQRef[i]) { ++numErrors; }
    }
    std::printf("rank %d holds %d of %d global nodes\n", myRank, numHeld, gOctree.numNodes);

    MPI_Allreduce(MPI_IN_PLACE, &numErrors, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (myRank == 0) { std::printf("upsweep: %s (%d errors)\n", numErrors == 0 ? "PASS" : "FAIL", numErrors); }

    MPI_Finalize();
    return numErrors != 0;
}
