/*! @file
 * @brief point-to-point upsweep of the global tree, replaces the Allgatherv in FocusedOctree::globalExchange
 */

#pragma once

#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include <mpi.h>

#include "cstone/sfc/common.hpp"
#include "cstone/domain/domaindecomp.hpp"
#include "cstone/domain/index_ranges.hpp"
#include "cstone/focus/source_center.hpp"
#include "cstone/primitives/gather.hpp"
#include "cstone/primitives/mpi_wrappers.hpp"
#include "cstone/tree/octree.hpp"

namespace upsweep_p2p
{

using namespace cstone;

// MPI Tag§
constexpr int upsweepTag = 6500;

/*! @brief owner rank of every global tree node: the rank holding the node's first leaf
 *
 * @param[in]  gOctree     global octree, same on all ranks
 * @param[in]  assignment  SFC key range per rank
 *
 * @return owner vector with owner[i] = rank that owns global node i
 */
template<class KeyType>
std::vector<int> computeNodeOwners(OctreeView<const KeyType> gOctree, const cstone::SfcAssignment<KeyType>& assignment)
{
    std::vector<int> owner(gOctree.numNodes);
    for (TreeNodeIndex i = 0; i < gOctree.numNodes; ++i)
    {
        owner[i] = assignment.findRank(decodePlaceholderBit(gOctree.prefixes[i]));
    }
    return owner;
}

/*! @brief send/recv lists of the upsweep per parent level
 *
 * send[l]: children I own of parents at level l that fail the MAC for peer
 * recv[l]: children owned by peer of parents at level l that fail the MAC for me
 * Message m covers the node indices sendIdx/recvIdx[m.offset : m.offset + m.count],
 * grouped by level, then by peer
 */
struct UpsweepLists
{
    struct Message
    {
        int           peer;
        TreeNodeIndex offset, count;
    };
    std::vector<std::vector<Message>> send, recv;
    std::vector<TreeNodeIndex>        sendIdx, recvIdx;
};

/*! @brief send/recv lists of the upsweep
 *
 * @param[in]  gOctree      global octree, same on all ranks
 * @param[in]  gLeaves      global leaf keys, length gOctree.numLeafNodes + 1
 * @param[in]  globDispl    first global leaf per rank, length numRanks + 1
 * @param[in]  nodeOwner    owner rank per node
 * @param[in]  box          global bounding box
 * @param[in]  invThetaEff  minMAC parameter, invThetaMinMac(theta) = 1/theta + 0.5
 * @param[in]  myRank       executing rank
 * @return                  send/recv lists per parent level
 */
template<class KeyType, class T>
UpsweepLists computeUpsweepLists(OctreeView<const KeyType> gOctree, std::span<const KeyType> gLeaves,
                                 std::span<const TreeNodeIndex> globDispl, std::span<const int> nodeOwner,
                                 const Box<T>& box, float invThetaEff, int myRank)
{
    int  numRanks        = int(globDispl.size()) - 1;
    auto levelRange      = gOctree.levelRangeSpan();
    int  numParentLevels = maxDepth(levelRange.data(), levelRange.size()); // deepest level has only leaves

    //! compute cell centers and mac flags
    std::vector<Vec4<T>> centers(gOctree.numNodes);
    geoMacSpheres(std::span<const KeyType>(gOctree.prefixes, gOctree.numNodes), centers.data(), invThetaEff, box);
    std::vector<uint8_t> macs(std::size_t(numRanks) * gOctree.numNodes, 0);
    for (int r = 0; r < numRanks; ++r)
    {
        markMacs(gOctree.prefixes, gOctree.childOffsets, gOctree.parents, centers.data(), box,
                 gLeaves.data() + globDispl[r], globDispl[r + 1] - globDispl[r], false,
                 macs.data() + std::size_t(r) * gOctree.numNodes);
    }
    auto failsMac = [&](int r, TreeNodeIndex p) { return macs[std::size_t(r) * gOctree.numNodes + p] != 0; };

    UpsweepLists lists;
    lists.send.resize(numParentLevels);
    lists.recv.resize(numParentLevels);

    //! consecutive nodes with the same peer extend the last message
    auto addNode = [](std::vector<UpsweepLists::Message>& msgs, std::vector<TreeNodeIndex>& idx, int peer, TreeNodeIndex c)
    {
        if (msgs.empty() || msgs.back().peer != peer) { msgs.push_back({peer, TreeNodeIndex(idx.size()), 0}); } // new message
        idx.push_back(c);
        msgs.back().count++;
    };
    auto parent = [&](TreeNodeIndex c) { return gOctree.parents[(c - 1) / 8]; };


    for (int l = 0; l < numParentLevels; ++l)
    {
        //! children of level-l parents are all nodes of level l + 1
        TreeNodeIndex firstChild = levelRange[l + 1], lastChild = levelRange[l + 2];

        //! recv: children of parents that fail the MAC for me, from their owner
        for (TreeNodeIndex c = firstChild; c < lastChild; ++c)
        {
            if (nodeOwner[c] != myRank && failsMac(myRank, parent(c))) { addNode(lists.recv[l], lists.recvIdx, nodeOwner[c], c); }
        }
        //! send: my children of parents that fail the MAC for q
        for (int q = 0; q < numRanks; ++q)
        {
            for (TreeNodeIndex c = firstChild; c < lastChild; ++c)
            {
                if (q != myRank && nodeOwner[c] == myRank && failsMac(q, parent(c))) { addNode(lists.send[l], lists.sendIdx, q, c); }
            }
        }
    }

    return lists;
}

// /*! @brief p2p upsweep of the global tree
//  *
//  * @param[in]  gOctree          global octree, same on all ranks
//  * @param[in]  globDispl        first global leaf per rank, length numRanks + 1
//  * @param[in]  nodeOwner        owner rank per node
//  * @param[in]  gLeafQLoc        quantities of my global leaves
//  * @param[out] globQ            quantities of the global tree, valid for nodes I own and their children
//  * @param[in]  myRank           executing rank
//  * @param[in]  comm             MPI communicator
//  * @param[in]  upsweepFunction  upsweep of one level
//  * @param[in]  upsweepArgs      extra upsweep arguments, e.g. expansion centers
//  */
// template<class Q, class KeyType, class F, class... UArgs>
// void globalUpsweep(cstone::OctreeView<const KeyType> gOctree, std::span<const TreeNodeIndex> globDispl,
//                    std::span<const int> nodeOwner, // temporary: nodeOwner_ in cstone
//                    std::span<const Q> gLeafQLoc, std::span<Q> globQ, int myRank, MPI_Comm comm, F&& upsweepFunction,
//                    UArgs&&... upsweepArgs)
// {
//     auto levelRange     = gOctree.levelRangeSpan();
//     auto leafToInternal = gOctree.leafToInternalSpan();

//     UpsweepLists lists           = computeUpsweepLists(gOctree, std::span<const int>(nodeOwner), myRank);
//     int          numParentLevels = int(lists.send.size());

//     //! scatter local leaf quantities
//     cstone::scatter(leafToInternal.subspan(globDispl[myRank], gLeafQLoc.size()), gLeafQLoc.data(), globQ.data());

//     for (int l = numParentLevels - 1; l >= 0; --l)
//     {
//         const auto&                 send = lists.send[l];
//         const auto&                 recv = lists.recv[l];
//         std::vector<std::vector<Q>> sendBuf(send.size()), recvBuf(recv.size());
//         std::vector<MPI_Request>    requests;
//         int                         tag = upsweepTag + l;

//         //! exchange child quantities with their owners
//         for (std::size_t m = 0; m < recv.size(); ++m)
//         {
//             const auto& [peer, ranges] = recv[m];
//             recvBuf[m].resize(ranges.totalCount());
//             mpiRecvAsync(recvBuf[m].data(), int(ranges.totalCount()), peer, tag, requests, comm);
//         }

//         for (std::size_t m = 0; m < send.size(); ++m)
//         {
//             const auto& [peer, ranges] = send[m];
//             sendBuf[m].resize(ranges.totalCount());
//             std::size_t writeIdx = 0;
//             for (std::size_t k = 0; k < ranges.nRanges(); ++k)
//             {
//                 for (TreeNodeIndex i = ranges.rangeStart(k); i < ranges.rangeEnd(k); ++i)
//                 {
//                     sendBuf[m][writeIdx++] = globQ[i];
//                 }
//             }
//             mpiSendAsync(sendBuf[m].data(), int(ranges.totalCount()), peer, tag, requests, comm);
//         }

//         MPI_Waitall(int(requests.size()), requests.data(), MPI_STATUSES_IGNORE);

//         //! place received child quantities
//         for (std::size_t m = 0; m < recv.size(); ++m)
//         {
//             const auto& ranges  = recv[m].second;
//             std::size_t readIdx = 0;
//             for (std::size_t k = 0; k < ranges.nRanges(); ++k)
//             {
//                 for (TreeNodeIndex i = ranges.rangeStart(k); i < ranges.rangeEnd(k); ++i)
//                 {
//                     globQ[i] = recvBuf[m][readIdx++];
//                 }
//             }
//         }

//         //! accumulate this parent level
//         upsweepFunction(levelRange.subspan(l, 2), gOctree.childOffsets, globQ.data(), upsweepArgs...);
//     }
// }

} // namespace upsweep_p2p
