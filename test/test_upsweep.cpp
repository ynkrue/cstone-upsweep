// Step 1: reference global upsweep (Allgatherv) on a uniform tree, Q = 1 per leaf
#include <cstdio>
#include <cstdint>
#include <cstdlib>

#include <mpi.h>

#include "cstone/tree/cs_util.hpp"
#include "cstone/tree/octree.hpp"

#include "upsweep_p2p.hpp"

using namespace upsweep_p2p;

int main(int argc, char** argv)
{
    using KeyType = uint64_t;
    using Q       = double;

    MPI_Init(&argc, &argv);
    int myRank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    // uniform global tree with 8^numLevels leaves, all ranks have the same tree
    int numLevels = argc > 1 ? std::atoi(argv[1]) : 2;
    std::vector<KeyType> gLeaves = cstone::makeUniformNLevelTree<KeyType>(std::size_t(1) << (3 * numLevels), 1);

    cstone::OctreeData<KeyType, cstone::execution::Cpu> globalTree;
    globalTree.resize(cstone::nNodes(gLeaves));
    // createUnsortedLayoutCpu (space of internal and compute prefix) + sort_by_key (level ordering)
    // + getLevelRangeCpu (start index of each level) + linkTreeCpu (childOffsets and parents)
    cstone::updateInternalTree<KeyType>(gLeaves, globalTree.data());
    auto gOctree = globalTree.cdata();

    // distribute global leaves across ranks
    std::vector<TreeNodeIndex> globNumNodes, globDispl;
    assignGlobalLeaves(gOctree.numLeafNodes, numRanks, globNumNodes, globDispl);

    // each rank fills its own global leaves with 1.0 (placeholder for FocusedOctree::populateGlobal)
    std::vector<Q> gLeafQLoc(globNumNodes[myRank], 1.0);

    auto upsweep = [](auto levelRange, auto childOffsets, auto M)
    { cstone::upsweep(levelRange, childOffsets, M, cstone::SumCombination<Q>{}); };

    std::vector<Q> globQ(gOctree.numNodes, 0.0);
    globalExchange<Q>(gOctree, globNumNodes, globDispl, std::span<const Q>(gLeafQLoc), std::span<Q>(globQ), upsweep,
                      myRank, MPI_COMM_WORLD);

    // check: a node at level l holds 8^(numLevels - l)
    int numErrors = 0;
    for (TreeNodeIndex i = 0; i < gOctree.numNodes; ++i)
    {
        int level  = cstone::decodePrefixLength(gOctree.prefixes[i]) / 3;
        Q expected = Q(std::size_t(1) << (3 * (numLevels - level)));
        if (globQ[i] != expected) { numErrors++; }
    }

    std::printf("rank %d: global leaves [%d, %d)\n", myRank, globDispl[myRank], globDispl[myRank + 1]);

    // debug: first and last node of every level (root, internal, leaves), same on all ranks after exchange
    if (myRank == 0)
    {
        for (int l = 0; l <= numLevels; ++l)
        {
            for (TreeNodeIndex i : {gOctree.levelRange[l], gOctree.levelRange[l + 1] - 1})
            {
                Q expected = Q(std::size_t(1) << (3 * (numLevels - l)));
                std::printf("level %d  node %3d  prefix %6lo  firstChild %3d  Q %4g  expected %4g\n", l, i,
                            gOctree.prefixes[i], gOctree.childOffsets[i], globQ[i], expected);
            }
        }
    }

    int globalErrors = 0;
    MPI_Reduce(&numErrors, &globalErrors, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (myRank == 0) { std::printf("%s\n", globalErrors == 0 ? "PASS" : "FAIL"); }

    // step 2: owner of every node, same on all ranks
    std::vector<int> nodeOwner(gOctree.numNodes, -1);
    computeNodeOwners<KeyType>(gOctree, globDispl, nodeOwner);

    // debug: owners of root + level-1 nodes, then cut parents (children with more than one owner)
    if (myRank == 0)
    {
        std::printf("owner root %d | level 1:", nodeOwner[0]);
        for (TreeNodeIndex i = gOctree.levelRange[1]; i < gOctree.levelRange[2]; ++i)
        {
            std::printf(" %d", nodeOwner[i]);
        }
        std::printf("\n");

        for (TreeNodeIndex p = 0; p < gOctree.numInternalNodes; ++p)
        {
            TreeNodeIndex c = gOctree.childOffsets[p];
            bool isCut      = false;
            for (int k = 1; k < 8; ++k) { isCut |= nodeOwner[c + k] != nodeOwner[c]; }
            if (!isCut) { continue; }

            std::printf("cut parent node %3d prefix %lo owner %d, child owners:", p, gOctree.prefixes[p], nodeOwner[p]);
            for (int k = 0; k < 8; ++k) { std::printf(" %d", nodeOwner[c + k]); }
            std::printf("\n");
        }
    }

    MPI_Finalize();
    return globalErrors != 0;
}
