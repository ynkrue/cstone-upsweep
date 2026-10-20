/*! @file
 * @brief point-to-point upsweep of the global tree, replaces the Allgatherv in FocusedOctree::globalExchange
 */

#pragma once

#include <cstdint>
#include <span>
#include <utility>
#include <vector>
#include <algorithm>

#include <mpi.h>

#include "cstone/sfc/common.hpp"
#include "cstone/domain/domaindecomp.hpp"
#include "cstone/focus/source_center.hpp"
#include "cstone/primitives/gather.hpp"
#include "cstone/primitives/mpi_wrappers.hpp"
#include "cstone/tree/octree.hpp"

namespace upsweep_p2p
{

using namespace cstone;

// MPI Tag
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

/*! @brief send/recv lists of the upsweep on a regular global tree: children of all parents that overlap the receiver
 *
 * @param[in] gOctree  global octree, same on all ranks, regular
 * @return             send/recv lists per parent level
 * 
 * Regular tree: all global leaves on level L, so parent j of level l covers the global leaves [j * span, (j+1) * span)
 * with span = 8^(L-l). The parents overlapping a rank are then one index range per level.
 */
template<class KeyType>
UpsweepLists computeOverlapLists(OctreeView<const KeyType> gOctree,
                                 std::span<const TreeNodeIndex> globDispl_, // becomes member
                                 std::span<const int>           nodeOwner_, // becomes member
                                 int                            myRank_)    // becomes member
{
    auto levelRange = gOctree.levelRangeSpan();
    int  numLevels  = maxDepth(levelRange.data(), levelRange.size()); // L
    auto leafOwner  = nodeOwner_.subspan(levelRange[numLevels]);      // owner of global leaf i

    UpsweepLists lists;
    lists.send.resize(numLevels);
    lists.recv.resize(numLevels);

    //! consecutive nodes with the same peer extend the last message
    auto addNode = [](std::vector<UpsweepLists::Message>& msgs, std::vector<TreeNodeIndex>& idx, int peer, TreeNodeIndex c)
    {
        if (msgs.empty() || msgs.back().peer != peer) { msgs.push_back({peer, TreeNodeIndex(idx.size()), 0}); }
        idx.push_back(c);
        msgs.back().count++;
    };

    TreeNodeIndex myFirstLeaf = globDispl_[myRank_], myEndLeaf = globDispl_[myRank_ + 1];
    if (myFirstLeaf == myEndLeaf) { return lists; } // no leaves: overlaps nothing, owns nothing

    for (int l = 0; l < numLevels; ++l)
    {
        TreeNodeIndex span   = TreeNodeIndex(1) << 3 * (numLevels - l); // global leaves per parent
        TreeNodeIndex jFirst = myFirstLeaf / span, jLast = (myEndLeaf - 1) / span; // parents overlapping me
        auto children = [&](TreeNodeIndex j) { return levelRange[l + 1] + 8 * j; }; // first child of parent j

        //! recv: children of the parents overlapping me, except mine
        for (TreeNodeIndex c = children(jFirst); c < children(jLast + 1); ++c)
        {
            if (nodeOwner_[c] != myRank_) { addNode(lists.recv[l], lists.recvIdx, nodeOwner_[c], c); }
        }

        //! send: my children to every other rank q overlapping their parent, one rank after the other
        int qFirst = leafOwner[jFirst * span], qLast = leafOwner[(jLast + 1) * span - 1];
        for (int q = qFirst; q <= qLast; ++q)
        {
            if (q == myRank_ || globDispl_[q] == globDispl_[q + 1]) { continue; }
            //! parents overlapping both me and q
            TreeNodeIndex qjFirst = std::max(jFirst, globDispl_[q] / span);
            TreeNodeIndex qjLast  = std::min(jLast, (globDispl_[q + 1] - 1) / span);
            for (TreeNodeIndex c = children(qjFirst); c < children(qjLast + 1); ++c)
            {
                if (nodeOwner_[c] == myRank_) { addNode(lists.send[l], lists.sendIdx, q, c); }
            }
        }
    }

    return lists;
}

/*! @brief send/recv lists of the upsweep
 *
 * @param[in]  gOctree      global octree, same on all ranks
 * @param[in]  gLeaves      global leaf keys, length gOctree.numLeafNodes + 1
 * @param[in]  invThetaEff  minMAC parameter, invThetaMinMac(theta) = 1/theta + 0.5
 * @return                  send/recv lists per parent level, becomes member: upsweepLists_
 */
template<class KeyType, class T>
UpsweepLists computeMacLists(OctreeView<const KeyType> gOctree, std::span<const KeyType> gLeaves,
                                 std::span<const TreeNodeIndex> globDispl_, // becomes member
                                 std::span<const int>           nodeOwner_, // becomes member
                                 const Box<T>&                  box_,       // becomes member
                                 float                          invThetaEff,
                                 int                            myRank_) // becomes member
{
    int  numRanks        = int(globDispl_.size()) - 1;
    auto levelRange      = gOctree.levelRangeSpan();
    int  numParentLevels = maxDepth(levelRange.data(), levelRange.size());

    //! compute cell centers and mac flags
    std::vector<Vec4<T>> centers(gOctree.numNodes);
    geoMacSpheres(std::span<const KeyType>(gOctree.prefixes, gOctree.numNodes), centers.data(), invThetaEff, box_);
    std::vector<uint8_t> macs(std::size_t(numRanks) * gOctree.numNodes, 0);
    for (int r = 0; r < numRanks; ++r)
    {
        markMacs(gOctree.prefixes, gOctree.childOffsets, gOctree.parents, centers.data(), box_,
                 gLeaves.data() + globDispl_[r], globDispl_[r + 1] - globDispl_[r], false,
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
            if (nodeOwner_[c] != myRank_ && failsMac(myRank_, parent(c)))
            {
                addNode(lists.recv[l], lists.recvIdx, nodeOwner_[c], c);
            }
        }
        //! send: my children of parents that fail the MAC for q
        for (int q = 0; q < numRanks; ++q)
        {
            for (TreeNodeIndex c = firstChild; c < lastChild; ++c)
            {
                if (q != myRank_ && nodeOwner_[c] == myRank_ && failsMac(q, parent(c)))
                {
                    addNode(lists.send[l], lists.sendIdx, q, c);
                }
            }
        }
    }

    return lists;
}

/*! @brief p2p upsweep of the global tree
 *
 * @param[in]    gOctree          global octree, same on all ranks
 * @param[in]    lists            send/recv lists, once per tree (computeMacLists)
 * @param[inout] globQ            in: my global leaf quantities at their node positions
 *                                out: valid for nodes I own and nodes in lists.recvIdx
 * @param[in]    comm             MPI communicator
 * @param[in]    upsweepFunction  upsweep of one level
 * @param[in]    upsweepArgs      extra upsweep arguments, e.g. expansion centers
 */
template<class Q, class KeyType, class F, class... UArgs>
void globalUpsweep(cstone::OctreeView<const KeyType> gOctree, const UpsweepLists& lists,
                   std::span<Q> globQ, MPI_Comm comm, F&& upsweepFunction, UArgs&&... upsweepArgs)
{
    auto levelRange     = gOctree.levelRangeSpan();
    int numParentLevels = int(lists.send.size());

    //! buffers indexed like sendIdx / recvIdx
    std::vector<Q> sendBuf(lists.sendIdx.size()), recvBuf(lists.recvIdx.size());

    //! the messages of one level are one contiguous slice of sendIdx / recvIdx: (first, count)
    auto levelSlice = [](const std::vector<UpsweepLists::Message>& msgs)
    {
        if (msgs.empty()) { return std::pair<TreeNodeIndex, TreeNodeIndex>{0, 0}; }
        return std::pair<TreeNodeIndex, TreeNodeIndex>{msgs.front().offset,
                                                       msgs.back().offset + msgs.back().count - msgs.front().offset};
    };

    //! Exchange messages per level
    for (int l = numParentLevels - 1; l >= 0; --l)
    {
        std::vector<MPI_Request> requests;
        int tag = upsweepTag + l;
        const auto& send = lists.send[l];
        const auto& recv = lists.recv[l];

        // receive values
        for (const auto& m : recv)
        {
            mpiRecvAsync(recvBuf.data() + m.offset, m.count, m.peer, tag, requests, comm);
        }

        // pack the whole level once, then send each message
        auto [sFirst, sCount] = levelSlice(send);
        gather(std::span<const TreeNodeIndex>(lists.sendIdx).subspan(sFirst, sCount), globQ.data(), sendBuf.data() + sFirst);
        for (const auto& m : send)
        {
            mpiSendAsync(sendBuf.data() + m.offset, m.count, m.peer, tag, requests, comm);
        }
        MPI_Waitall(int(requests.size()), requests.data(), MPI_STATUSES_IGNORE);

        // unpack the whole level once
        auto [rFirst, rCount] = levelSlice(recv);
        scatter(std::span<const TreeNodeIndex>(lists.recvIdx).subspan(rFirst, rCount), recvBuf.data() + rFirst,
                globQ.data());

        //! compute the parents of level l from their children
        upsweepFunction(levelRange.subspan(l, 2), gOctree.childOffsets, globQ.data(), upsweepArgs...);
    }
}

} // namespace upsweep_p2p
