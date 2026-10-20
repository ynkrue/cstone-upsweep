/*! @file
 * @brief test globalUpsweep against a single-rank upsweep of the global tree
 *
 * Usage: test_global_exchange [L=3] [n=32] [theta=1.0]
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
    using T       = double;

    MPI_Init(&argc, &argv);
    int myRank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    int      numLevels  = argc > 1 ? std::atoi(argv[1]) : 3;      // L, 8^L global leaves
    unsigned numPerCell = argc > 2 ? std::atoi(argv[2]) : 32;     // particles per global leaf
    float    theta      = argc > 3 ? std::atof(argv[3]) : 1.0;
    cstone::Box<T> box(0, 1);

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

    //! compute node owners and exchange lists
    std::vector<int> nodeOwner = computeNodeOwners(gOctree, assignment);
    // UpsweepLists upSweepLists =
    //     computeMacLists(gOctree, std::span<const KeyType>(gLeaves), globDispl, std::span<const int>(nodeOwner), box,
    //                     cstone::invThetaMinMac(theta), myRank);
    UpsweepLists upSweepLists = computeOverlapLists(gOctree, globDispl, std::span<const int>(nodeOwner), myRank);

    //! check recv lists via the ancestors of my leaves
    //  expected: c is not mine and parent(c) is an ancestor of one of my leaves
    auto parent = [&](TreeNodeIndex c) { return gOctree.parents[(c - 1) / 8]; };
    std::vector<char> overlapsMe(gOctree.numNodes, 0);
    for (TreeNodeIndex i = globDispl[myRank]; i < globDispl[myRank + 1]; ++i) // walk through my leaves
    {
        for (TreeNodeIndex n = gOctree.leafToInternalSpan()[i]; n != 0; n = parent(n)) // walk trough ancestors
        { 
            overlapsMe[parent(n)] = 1;
        }
    }
    int numSetErrors = 0;
    std::vector<char> inRecv(gOctree.numNodes, 0);
    for (TreeNodeIndex c : upSweepLists.recvIdx)
    {
        numSetErrors += inRecv[c]; // check duplicates
        inRecv[c] = 1;
    }
    for (TreeNodeIndex c = 0; c < gOctree.numNodes; ++c) // check if each node is expected
    {
        bool expected = c > 0 && overlapsMe[parent(c)] && nodeOwner[c] != myRank;
        if (expected != bool(inRecv[c])) { ++numSetErrors; }
    }

    MPI_Allreduce(MPI_IN_PLACE, &numSetErrors, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (myRank == 0)
    {
        std::printf("recv list: %s (%d errors)\n", numSetErrors == 0 ? "PASS" : "FAIL", numSetErrors);
    }

    //! point-to-point upsweep
    TreeNodeIndex             globNumNodes = globDispl[myRank + 1] - globDispl[myRank];
    std::span<const unsigned> gLeafQLoc(gLeafQ.data() + globDispl[myRank], globNumNodes);
    std::vector<unsigned>     globQ(gOctree.numNodes);
    cstone::scatter(gOctree.leafToInternalSpan().subspan(globDispl[myRank], gLeafQLoc.size()), gLeafQLoc.data(), globQ.data());
    globalUpsweep(gOctree, upSweepLists, std::span<unsigned>(globQ), MPI_COMM_WORLD, upsweep);

    //! check: every node I own or receive equals the single-rank reference
    std::vector<char> held(gOctree.numNodes, 0);
    for (TreeNodeIndex i = 0; i < gOctree.numNodes; ++i) { held[i] = nodeOwner[i] == myRank; }
    for (TreeNodeIndex i : upSweepLists.recvIdx) { held[i] = 1; }

    int numGlobErrors = 0, numHeld = 0;
    for (TreeNodeIndex i = 0; i < gOctree.numNodes; ++i)
    {
        numHeld += held[i];
        if (held[i] && globQ[i] != globQRef[i]) { ++numGlobErrors; }
    }
    std::printf("rank %d holds %d of %d global nodes\n", myRank, numHeld, gOctree.numNodes);

    MPI_Allreduce(MPI_IN_PLACE, &numGlobErrors, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (myRank == 0)
    {
        std::printf("upsweep: %s (%d errors)\n", numGlobErrors == 0 ? "PASS" : "FAIL", numGlobErrors);
    }

    MPI_Finalize();
    return numGlobErrors != 0 || numSetErrors != 0;
}
