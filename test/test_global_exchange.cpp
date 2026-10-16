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

    int      numLevels  = argc > 1 ? std::atoi(argv[1]) : 3; // L, 8^L global leaves
    unsigned numPerCell = argc > 2 ? std::atoi(argv[2]) : 4; // particles per leaf
    float    theta      = argc > 3 ? std::atof(argv[3]) : 1.0; // MAC opening angle
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
    std::vector<int> nodeOwner = computeNodeOwners(gOctree, assignment);

    UpsweepLists lists = computeUpsweepLists(gOctree, std::span<const KeyType>(gLeaves), globDispl,
                                             std::span<const int>(nodeOwner), box, cstone::invThetaMinMac(theta), myRank);

    //! debug: lists of one rank, compare with the atlas (L=3, P=3, theta=1, view as rank 1)
    if (myRank == 1)
    {
        for (std::size_t l = 0; l < lists.recv.size(); ++l)
        {
            for (auto m : lists.recv[l])
            {
                std::printf("recv l=%zu peer=%d count=%d nodes %d..%d\n", l, m.peer, m.count,
                            lists.recvIdx[m.offset], lists.recvIdx[m.offset + m.count - 1]);
            }
            for (auto m : lists.send[l])
            {
                std::printf("send l=%zu peer=%d count=%d nodes %d..%d\n", l, m.peer, m.count,
                            lists.sendIdx[m.offset], lists.sendIdx[m.offset + m.count - 1]);
            }
        }
    }

    //! check: per level, every send message matches the receiver's recv message (count, node indices, order)
    int numListErrors = 0;
    for (std::size_t l = 0; l < lists.recv.size(); ++l)
    {
        //! (a) counts per pair: what each peer sends me vs. what my recv list expects
        std::vector<int> sendCounts(numRanks, 0), recvCounts(numRanks, 0), expected(numRanks, 0);
        for (auto m : lists.send[l]) { sendCounts[m.peer] = m.count; }
        for (auto m : lists.recv[l]) { expected[m.peer] = m.count; }
        MPI_Alltoall(sendCounts.data(), 1, MPI_INT, recvCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        int countErrors = 0;
        for (int r = 0; r < numRanks; ++r) { countErrors += recvCounts[r] != expected[r]; }
        numListErrors += countErrors;
        int anyCountErrors = countErrors;
        MPI_Allreduce(MPI_IN_PLACE, &anyCountErrors, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if (anyCountErrors) { continue; } // exchanging indices with mismatched counts could hang

        //! (b) node indices: the sender's sendIdx slice must equal my recvIdx slice
        std::vector<TreeNodeIndex> got(lists.recvIdx.size());
        std::vector<MPI_Request>   requests;
        for (auto m : lists.recv[l])
        {
            MPI_Irecv(got.data() + m.offset, m.count, MPI_INT, m.peer, int(l), MPI_COMM_WORLD, &requests.emplace_back());
        }
        for (auto m : lists.send[l])
        {
            MPI_Isend(lists.sendIdx.data() + m.offset, m.count, MPI_INT, m.peer, int(l), MPI_COMM_WORLD,
                      &requests.emplace_back());
        }
        MPI_Waitall(int(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
        for (auto m : lists.recv[l])
        {
            for (TreeNodeIndex i = m.offset; i < m.offset + m.count; ++i) { numListErrors += got[i] != lists.recvIdx[i]; }
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &numListErrors, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (myRank == 0) { std::printf("lists send/recv match: %s (%d errors)\n", numListErrors == 0 ? "PASS" : "FAIL", numListErrors); }

    // TreeNodeIndex             globNumNodes = globDispl[myRank + 1] - globDispl[myRank];
    // std::span<const unsigned> gLeafQLoc(gCounts.data() + globDispl[myRank], globNumNodes);
    // std::vector<unsigned>     globQ(gOctree.numNodes);
    // globalUpsweep<unsigned>(gOctree, globDispl, nodeOwner, gLeafQLoc, std::span<unsigned>(globQ), myRank,
    //                         MPI_COMM_WORLD, upsweep);

    // //! check
    // int numGlobErrors = 0;
    // for (TreeNodeIndex i = 0; i < gOctree.numNodes; ++i)
    // {
    //     bool ownedOrChildOfOwned =
    //         nodeOwner[i] == myRank || (i > 0 && nodeOwner[gOctree.parents[(i - 1) / 8]] == myRank);
    //     if (ownedOrChildOfOwned && globQ[i] != globQRef[i]) { ++numGlobErrors; }
    // }

    // MPI_Allreduce(MPI_IN_PLACE, &numGlobErrors, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    // if (myRank == 0)
    // {
    //     std::printf("upsweep (%d global nodes): %s (%d errors)\n", gOctree.numNodes, numGlobErrors == 0 ? "PASS" : "FAIL",
    //                 numGlobErrors);
    // }

    MPI_Finalize();
    return 0;
    // return numGlobErrors != 0;
}
