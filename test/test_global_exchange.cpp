/*! @file
 * @brief test globalUpsweep against a single-rank upsweep of the global tree
 *
 * Usage: test_global_exchange [L=4] [n=4]
 */

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
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

    int      numLevels  = argc > 1 ? std::atoi(argv[1]) : 4; // L, 8^L global leaves
    unsigned numPerCell = argc > 2 ? std::atoi(argv[2]) : 4; // particles per leaf
    unsigned bucketSize = 4 * numPerCell;                    // n <= bucketSize < 8n

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

    auto upsweep = [](auto levelRange, auto childOffsets, auto Q)
    { cstone::upsweep(levelRange, childOffsets, Q, cstone::NodeCount<unsigned>{}); };

    //! reference: upsweep of all global leaves on one rank
    std::vector<unsigned> globQRef(gOctree.numNodes);
    cstone::scatter(gOctree.leafToInternalSpan(), gCounts.data(), globQRef.data());
    upsweep(gOctree.levelRangeSpan(), gOctree.childOffsets, globQRef.data());

    //! point-to-point upsweep of the global tree, Q = particle counts of my global leaves
    std::vector<int> nodeOwner(gOctree.numNodes, -1);
    computeNodeOwners(gOctree, globDispl, nodeOwner);
    TreeNodeIndex             globNumNodes = globDispl[myRank + 1] - globDispl[myRank];
    std::span<const unsigned> gLeafQLoc(gCounts.data() + globDispl[myRank], globNumNodes);
    std::vector<unsigned>     globQ(gOctree.numNodes);
    globalUpsweep<unsigned>(gOctree, globDispl, nodeOwner, gLeafQLoc, std::span<unsigned>(globQ), myRank,
                            MPI_COMM_WORLD, upsweep);

    //! check
    int numGlobErrors = 0;
    for (TreeNodeIndex i = 0; i < gOctree.numNodes; ++i)
    {
        bool ownedOrChildOfOwned =
            nodeOwner[i] == myRank || (i > 0 && nodeOwner[gOctree.parents[(i - 1) / 8]] == myRank);
        if (ownedOrChildOfOwned && globQ[i] != globQRef[i]) { ++numGlobErrors; }
    }

    MPI_Allreduce(MPI_IN_PLACE, &numGlobErrors, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (myRank == 0)
    {
        std::printf("upsweep (%d global nodes): %s (%d errors)\n", gOctree.numNodes, numGlobErrors == 0 ? "PASS" : "FAIL",
                    numGlobErrors);
    }

    MPI_Finalize();
    return numGlobErrors != 0;
}
