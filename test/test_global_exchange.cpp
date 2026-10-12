/*! @file
 * @brief p2p global exchange (upward pass + delivery) on a real FocusedOctree (LET)
 *
 * Particles on the uniform 2^L grid, n per cell, identical pool on all ranks → uniform global tree
 * with n particles per leaf. Each rank keeps its slice and builds a LET with FocusedOctree::converge.
 * Quantity: particle counts (exact). Usage: test_global_exchange [L=4] [n=4] [theta=1.0]
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

    //! particles: synthetically uniform
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

    //! global tree from the particles
    auto [gLeaves, gCounts] = cstone::computeOctree<KeyType>(keys, bucketSize);
    cstone::OctreeData<KeyType, cstone::execution::Cpu> globalTree;
    globalTree.resize(cstone::nNodes(gLeaves));
    cstone::updateInternalTree<KeyType>(gLeaves, globalTree.data());
    auto gOctree = globalTree.cdata();

    //! distribute the global leaves across ranks
    auto                           assignment = cstone::makeSfcAssignment(numRanks, gCounts, gLeaves.data());
    std::span<const TreeNodeIndex> globDispl  = assignment.treeOffsetsConst();

    std::vector<int> nodeOwner(gOctree.numNodes, -1);
    computeNodeOwners(gOctree, globDispl, nodeOwner);

    KeyType              focusStart = assignment[myRank];
    KeyType              focusEnd   = assignment[myRank + 1];
    auto                 first      = std::lower_bound(keys.begin(), keys.end(), focusStart);
    auto                 last       = std::lower_bound(keys.begin(), keys.end(), focusEnd);
    std::vector<KeyType> myKeys(first, last);

    //! Build LET
    cstone::FocusedOctree<KeyType, T> focusTree(cstone::execution::cpu, myRank, numRanks, bucketSizeLocal,
                                                MPI_COMM_WORLD);
    focusTree.converge(box, myKeys, assignment, gLeaves, gCounts, cstone::invThetaMinMac(theta));

    //! reference: serial upsweep of the exact global leaf counts
    auto upsweep = [](auto levelRange, auto childOffsets, auto Q)
    { cstone::upsweep(levelRange, childOffsets, Q, cstone::NodeCount<unsigned>{}); };

    std::vector<unsigned> globQRef(gOctree.numNodes, 0);
    cstone::scatter(gOctree.leafToInternalSpan(), gCounts.data(), globQRef.data());
    upsweep(gOctree.levelRangeSpan(), gOctree.childOffsets, globQRef.data());

    //! @brief Reimplementation of globalExchange
    //! p2p upward pass (replace gatherGlobalLeaves, scatter and upsweep)
    std::span<const unsigned> gLeafQLoc =
        std::span<const unsigned>(gCounts).subspan(globDispl[myRank], globDispl[myRank + 1] - globDispl[myRank]);
    std::vector<unsigned> globQ(gOctree.numNodes, 0);
    globalUpsweep<unsigned>(gOctree, globDispl, nodeOwner, gLeafQLoc, std::span<unsigned>(globQ), upsweep, myRank,
                            MPI_COMM_WORLD);

    //! p2p extract (replace extractGlobal)
    // TODO: peerRanges as in updateTree (octree_focus_mpi.hpp:133-138): focusPeersAcc → peerFlagsToList
    //       → extractPeerRanges, with focusTree.assignment(), focusTree.treeLeaves()
    // TODO: reference letRef via focusTree.extractGlobal(globQRef), ours letQ via extractGlobalP2p(globQ)
    //       → compare at far leaves, print PASS/FAIL

    //! check exchange
    int numErrors = 0;
    for (TreeNodeIndex i = 0; i < gOctree.numNodes; ++i)
    {
        bool ownedOrChildOfOwned =
            nodeOwner[i] == myRank || (i > 0 && nodeOwner[gOctree.parents[(i - 1) / 8]] == myRank);
        if (ownedOrChildOfOwned && globQ[i] != globQRef[i]) { ++numErrors; }
    }
    MPI_Allreduce(MPI_IN_PLACE, &numErrors, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (myRank == 0) { std::printf("upward pass: %s (%d errors)\n", numErrors == 0 ? "PASS" : "FAIL", numErrors); }

    MPI_Finalize();
    return numErrors != 0;
}
