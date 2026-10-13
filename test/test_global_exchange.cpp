/*! @file
 * @brief test globalExchangeP2p against FocusedOctree::globalExchange
 *
 * Usage: test_global_exchange [L=4] [n=4] [theta=1.0]
 */

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <mpi.h>

#include "cstone/domain/domaindecomp.hpp"
#include "cstone/focus/octree_focus_mpi.hpp"
#include "cstone/sfc/sfc.hpp"
#include "cstone/traversal/macs.hpp"
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

    int      numLevels       = argc > 1 ? std::atoi(argv[1]) : 4;   // L, 8^L global leaves
    unsigned numPerCell      = argc > 2 ? std::atoi(argv[2]) : 4;   // particles per leaf
    float    theta           = argc > 3 ? std::atof(argv[3]) : 1.0; // mac
    unsigned bucketSize      = 4 * numPerCell;                      // n <= bucketSize < 8n
    unsigned bucketSizeLocal = numPerCell;                          // LET leaf size inside the focus

    cstone::Box<T> box(0, 1);

    //! particles: n per cell of the 2^L grid
    int                               cellsPerDim  = 1 << numLevels;
    std::size_t                       numParticles = std::size_t(cellsPerDim) * cellsPerDim * cellsPerDim * numPerCell;
    std::vector<T>                    x(numParticles), y(numParticles), z(numParticles);
    std::mt19937                      gen(42);
    std::uniform_real_distribution<T> uniform(0.0, 1.0);

    std::size_t idx = 0;
    for (int ix = 0; ix < cellsPerDim; ++ix)
    {
        for (int iy = 0; iy < cellsPerDim; ++iy)
        {
            for (int iz = 0; iz < cellsPerDim; ++iz)
            {
                for (unsigned k = 0; k < numPerCell; ++k)
                {
                    x[idx] = (ix + uniform(gen)) / cellsPerDim;
                    y[idx] = (iy + uniform(gen)) / cellsPerDim;
                    z[idx] = (iz + uniform(gen)) / cellsPerDim;
                    ++idx;
                }
            }
        }
    }

    std::vector<KeyType> keys(numParticles);
    cstone::computeSfcKeys(x.data(), y.data(), z.data(), cstone::sfcKindPointer(keys.data()), numParticles, box);
    std::sort(keys.begin(), keys.end());

    //! global tree, distributed across ranks
    auto [gLeaves, gCounts] = cstone::computeOctree<KeyType>(keys, bucketSize);
    cstone::OctreeData<KeyType, cstone::execution::Cpu> globalTree;
    globalTree.resize(cstone::nNodes(gLeaves));
    cstone::updateInternalTree<KeyType>(gLeaves, globalTree.data());
    auto gOctree   = globalTree.cdata();
    gOctree.leaves = gLeaves.data();

    auto                           assignment = cstone::makeSfcAssignment(numRanks, gCounts, gLeaves.data());
    std::span<const TreeNodeIndex> globDispl  = assignment.treeOffsetsConst();

    std::vector<int> nodeOwner(gOctree.numNodes, -1);
    computeNodeOwners(gOctree, globDispl, nodeOwner);

    //! LET, focused per rank
    auto                 first = std::lower_bound(keys.begin(), keys.end(), assignment[myRank]);
    auto                 last  = std::lower_bound(keys.begin(), keys.end(), assignment[myRank + 1]);
    std::vector<KeyType> myKeys(first, last);

    cstone::FocusedOctree<KeyType, T> focusTree(cstone::execution::cpu, myRank, numRanks, bucketSizeLocal,
                                                MPI_COMM_WORLD);
    focusTree.converge(box, myKeys, assignment, gLeaves, gCounts, cstone::invThetaMinMac(theta));

    //! peerRanges_ (temporary: is private in focusTree)
    std::vector<int> peerFlags = cstone::focusPeersAcc<KeyType>(
        cstone::execution::cpu, globDispl, focusTree.assignment(), myRank, gLeaves, focusTree.treeLeaves());
    std::vector<int> peers;
    cstone::peerFlagsToList(peerFlags, peers, cstone::PeerMask::focus);
    std::vector<cstone::TreeIndexPair> peerRanges;
    cstone::extractPeerRanges(peers, myRank, focusTree.assignment(), peerRanges);

    //! LET counts
    auto                  letOctree = focusTree.octreeViewAcc();
    std::vector<unsigned> letQIn(focusTree.countsAcc().begin(), focusTree.countsAcc().end());
    auto farLeaves = cstone::enumerateRanges(cstone::invertRanges(0, peerRanges, letOctree.numLeafNodes));
    for (TreeNodeIndex j : farLeaves)
    {
        letQIn[letOctree.leafToInternal[letOctree.numInternalNodes + j]] = 0xDEAD; // sentinel values
    }

    auto upsweep = [](auto levelRange, auto childOffsets, auto Q)
    { cstone::upsweep(levelRange, childOffsets, Q, cstone::NodeCount<unsigned>{}); };
    std::vector<int> scratch;

    //! reference: Allgatherv version
    std::vector<unsigned> letQRef = letQIn, globQRef(gOctree.numNodes);
    focusTree.globalExchange(gOctree, std::span<unsigned>(letQRef), std::span<unsigned>(globQRef), scratch, upsweep);

    //! reimplementation: point-to-point version
    std::vector<unsigned> letQ = letQIn, globQ(gOctree.numNodes);
    globalExchangeP2p<unsigned>(focusTree, gOctree, std::span<unsigned>(letQ), std::span<unsigned>(globQ), scratch,
                                globDispl, nodeOwner, peerRanges, myRank, MPI_COMM_WORLD, upsweep);

    //! check
    int numGlobErrors = 0;
    for (TreeNodeIndex i = 0; i < gOctree.numNodes; ++i)
    {
        bool ownedOrChildOfOwned =
            nodeOwner[i] == myRank || (i > 0 && nodeOwner[gOctree.parents[(i - 1) / 8]] == myRank);
        if (ownedOrChildOfOwned && globQ[i] != globQRef[i]) { ++numGlobErrors; }
    }

    int numLetErrors = 0;
    for (std::size_t i = 0; i < letQ.size(); ++i)
    {
        if (letQ[i] != letQRef[i]) { ++numLetErrors; }
    }

    MPI_Allreduce(MPI_IN_PLACE, &numGlobErrors, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &numLetErrors, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (myRank == 0)
    {
        std::printf("upsweep (%ld far leaves): %s (%d errors)\n", farLeaves.size(), numGlobErrors == 0 ? "PASS" : "FAIL", numGlobErrors);
        std::printf("LET (%ld far leaves):     %s (%d errors)\n", farLeaves.size(), numLetErrors == 0 ? "PASS" : "FAIL", numLetErrors);
    }

    MPI_Finalize();
    return numGlobErrors != 0 || numLetErrors != 0;
}
