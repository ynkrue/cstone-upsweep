#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <numeric>

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

    //! uniform global tree, same on all ranks
    int numLevels = argc > 1 ? std::atoi(argv[1]) : 2;
    std::vector<KeyType> gLeaves = cstone::makeUniformNLevelTree<KeyType>(std::size_t(1) << (3 * numLevels), 1);

    //! build global linked tree from leaves, same on all ranks
    cstone::OctreeData<KeyType, cstone::execution::Cpu> globalTree;
    globalTree.resize(cstone::nNodes(gLeaves));
    cstone::updateInternalTree<KeyType>(gLeaves, globalTree.data());
    auto gOctree = globalTree.cdata();

    //! distribute global leaves across ranks
    std::vector<TreeNodeIndex> globNumNodes, globDispl;
    assignGlobalLeaves(gOctree.numLeafNodes, numRanks, globNumNodes, globDispl);

    //! Generate leaf quantities Q
    std::vector<Q> gLeafQAll(gOctree.numLeafNodes);
    std::iota(gLeafQAll.begin(), gLeafQAll.end(), Q(1.0));
    // each rank takes its local slice from global Q (placeholder for FocusedOctree::populateGlobal)
    std::span<const Q> gLeafQLoc = std::span<const Q>(gLeafQAll).subspan(globDispl[myRank], globNumNodes[myRank]);

    auto upsweep = [](auto levelRange, auto childOffsets, auto M)
    { cstone::upsweep(levelRange, childOffsets, M, cstone::SumCombination<Q>{}); };

    //! sequential ref upsweep on global tree
    std::vector<Q> globQRef(gOctree.numNodes, 0.0);
    cstone::scatter(gOctree.leafToInternalSpan(), gLeafQAll.data(), globQRef.data());
    
    upsweep(gOctree.levelRangeSpan(), gOctree.childOffsets, globQRef.data());

    //! collective global upsweep from local data, (mirrors FocusedOctree::globalExchange)
    std::vector<Q> globQ(gOctree.numNodes, 0.0);
    
    globalExchange<Q>(gOctree, globNumNodes, globDispl, gLeafQLoc, std::span<Q>(globQ), upsweep,
                      myRank, MPI_COMM_WORLD);
    
    //! p2p upsweep from local data
    std::vector<Q> globQp2p(gOctree.numNodes, 0.0);
    std::vector<int> nodeOwner(gOctree.numNodes, -1);
    computeNodeOwners(gOctree, globDispl, nodeOwner);

    globalExchangeP2p<Q>(gOctree, globNumNodes, globDispl, nodeOwner, gLeafQLoc, std::span<Q>(globQp2p), upsweep,
                         myRank, MPI_COMM_WORLD);

    //! check Q data
    auto check = [&](std::span<const Q> testQ, const char* label) {
        int numErrors = 0;
        for (TreeNodeIndex i = 0; i < gOctree.numNodes; ++i)
        {
            Q expected = globQRef[i];
            if (testQ[i] != expected)
            {
                if (numErrors < 5)
                {
                    std::printf("rank %d: node %3d  prefix %6lo  firstChild %3d  Q %.0f  expected %.0f (%s)\n", myRank,
                                i, gOctree.prefixes[i], gOctree.childOffsets[i], testQ[i], expected, label);
                }
                ++numErrors;
            }
        }
        return numErrors;
    };

    int numErrorsQ = check(globQ, "allgather");
    int globalErrors = 0;
    MPI_Reduce(&numErrorsQ, &globalErrors, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (myRank == 0) { std::printf("Upsweep allgather: %s (%d errors)\n", globalErrors == 0 ? "PASS" : "FAIL", globalErrors); }
    
    // int numErrorsP2p = check(globQp2p, "p2p");
    // int globalErrorsP2p = 0;
    // MPI_Reduce(&numErrorsP2p, &globalErrorsP2p, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    // if (myRank == 0) { std::printf("Upsweep p2p: %s (%d errors)\n", globalErrorsP2p == 0 ? "PASS" : "FAIL", globalErrorsP2p); }

    MPI_Finalize();
    return globalErrors != 0;
}
