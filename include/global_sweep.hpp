/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief GROMACS global exchange isolated for CPU/GPU experiments
 *
 * @author Yannik Rüfenacht <yannik.rue@gmail.com>
 */

#pragma once

#include <algorithm>
#include <array>
#include <cstdio>
#include <span>
#include <vector>

#include <mpi.h>

#include "cstone/focus/global_exchange_nvtx.hpp"
#include "cstone/primitives/primitives_acc.hpp"
#include "cstone/sfc/common.hpp"
#include "cstone/domain/domaindecomp.hpp"
#include "cstone/primitives/mpi_wrappers.hpp"
#include "cstone/tree/octree.hpp"

namespace global_sweep
{
using namespace cstone;

//! @brief owner rank of global node i (owner of it's first leaf)
template<class KeyType>
int nodeOwner(OctreeView<const KeyType> gOctree, const SfcAssignment<KeyType>& assignment, TreeNodeIndex i)
{ return assignment.findRank(decodePlaceholderBit(gOctree.prefixes[i])); }

//! @brief level of global node i
template<class KeyType>
int nodeLevel(OctreeView<const KeyType> gOctree, TreeNodeIndex i)
{ return decodePrefixLength(gOctree.prefixes[i]) / 3; }

//! @brief check wheter global node i is inside the subtree of node p
template<class KeyType>
bool containsNode(OctreeView<const KeyType> gOctree, TreeNodeIndex p, TreeNodeIndex i)
{
    KeyType sp = decodePlaceholderBit(gOctree.prefixes[p]);
    int     lp = nodeLevel(gOctree, p);
    KeyType si = decodePlaceholderBit(gOctree.prefixes[i]);
    return nodeLevel(gOctree, i) >= lp && sp <= si && si < sp + nodeRange<KeyType>(lp);
}

//! @brief which child of node p contains i
template<class KeyType>
TreeNodeIndex childContaining(OctreeView<const KeyType> gOctree, TreeNodeIndex p, TreeNodeIndex i)
{
    assert(gOctree.childOffsets[p] != 0);          // p is not a leaf
    assert(i != p && containsNode(gOctree, p, i)); // i strictly below p
    KeyType si = decodePlaceholderBit(gOctree.prefixes[i]);
    return gOctree.childOffsets[p] + TreeNodeIndex(octalDigit(si, nodeLevel(gOctree, p) + 1));
}

//! @brief nodes exchanged with one peer rank on one level
struct PeerNodes
{
    int                        peer;
    std::vector<TreeNodeIndex> send, recv;
};

//! @brief entry of @p peer in @p level, appended if not present
inline PeerNodes& peerNodes(std::vector<PeerNodes>& level, int peer)
{
    auto it = std::find_if(level.begin(), level.end(), [peer](const PeerNodes& e) { return e.peer == peer; });
    if (it != level.end()) { return *it; }
    return level.emplace_back(PeerNodes{peer, {}, {}});
}

//! @brief per parent level: one PeerNodes per partner rank
using LevelPairs = std::vector<std::vector<PeerNodes>>;

//! @brief upsweep communication per parent level, children 1-7 go to the parent owner
template<class KeyType>
LevelPairs basePairs(OctreeView<const KeyType> gOctree, const SfcAssignment<KeyType>& assignment, int myRank)
{
    GlobalExchangeRange functionRange("fmm:basePairs");
    auto                levelRange = gOctree.levelRangeSpan();
    int                 depth      = maxDepth(levelRange.data(), levelRange.size());

    auto owner = [&](TreeNodeIndex i) { return nodeOwner(gOctree, assignment, i); };

    // per level communication pairs
    LevelPairs pairs(depth);
    for (int l = 0; l < depth; ++l)
    {
        for (TreeNodeIndex p = levelRange[l]; p < levelRange[l + 1]; ++p)
        {
            TreeNodeIndex firstChild = gOctree.childOffsets[p];
            if (firstChild == 0) { continue; }
            int parentOwner = owner(p);
            for (TreeNodeIndex c = firstChild + 1; c < firstChild + 8; ++c)
            {
                int childOwner = owner(c);
                if (childOwner == parentOwner) { continue; }
                if (childOwner == myRank) { peerNodes(pairs[l], parentOwner).send.push_back(c); }
                if (parentOwner == myRank) { peerNodes(pairs[l], childOwner).recv.push_back(c); }
            }
        }
    }
    return pairs;
}

//! @brief route requests for @p targets over the base pairs, append the mirrored value hops
template<class KeyType>
void routeRequests(OctreeView<const KeyType> gOctree, const SfcAssignment<KeyType>& assignment,
                   std::vector<TreeNodeIndex> targets, int myRank, MPI_Comm comm, LevelPairs& upPeers,
                   LevelPairs& downPeers)
{
    GlobalExchangeRange functionRange("fmm:routeRequests");
    using NodeRequest = std::array<TreeNodeIndex, 2>; // {target, position at the receiver}
    constexpr int tag = static_cast<int>(P2pTags::globalRequests);

    int  depth    = int(upPeers.size());
    auto owner    = [&](TreeNodeIndex i) { return nodeOwner(gOctree, assignment, i); };
    auto parent   = [&](TreeNodeIndex i) { return gOctree.parents[(i - 1) / 8]; };
    auto holdNode = [&](TreeNodeIndex i) { return owner(i) == myRank || (i != 0 && owner(parent(i)) == myRank); };
    auto nextNode = [&](TreeNodeIndex target, TreeNodeIndex pos)
    { return containsNode(gOctree, pos, target) ? childContaining(gOctree, pos, target) : parent(pos); };

    // requests per edge level, waiting for their up or down round
    std::vector<std::vector<NodeRequest>> upRequests(depth), downRequests(depth);

    //! walk locally while the path is mine, then queue for the round of the edge to the next owner
    auto queueRequest = [&](TreeNodeIndex target, TreeNodeIndex pos)
    {
        // nothing to route if I own target or get it as a child of my own node in the upsweep
        if (holdNode(target)) { return; }

        while (owner(nextNode(target, pos)) == myRank)
        {
            pos = nextNode(target, pos);
        }
        TreeNodeIndex next = nextNode(target, pos);

        // the level of an edge is the level of its upper node
        if (containsNode(gOctree, pos, target)) { downRequests[nodeLevel(gOctree, pos)].push_back({target, next}); }
        else
        {
            upRequests[nodeLevel(gOctree, next)].push_back({target, next});
        }
    };

    /*! exchange the requests of edge level l with the base partners of that level
     *
     * Requests travel opposite to their values: an up request records a down value hop and vice versa.
     * Every partner gets a message, also an empty one, so each rank knows how many messages it receives.
     */
    auto exchangeRequests = [&](int l, bool up)
    {
        std::vector<NodeRequest>& requests = up ? upRequests[l] : downRequests[l];
        LevelPairs&               hops     = up ? downPeers : upPeers;

        // up requests go child owner to parent owner, down requests the reverse
        std::vector<int> sendPeers, recvPeers;
        for (const PeerNodes& e : upPeers[l])
        {
            if (!(up ? e.send : e.recv).empty()) { sendPeers.push_back(e.peer); }
            if (!(up ? e.recv : e.send).empty()) { recvPeers.push_back(e.peer); }
        }

        std::vector<std::vector<NodeRequest>> sendBuffers(sendPeers.size());
        std::vector<MPI_Request>              mpiRequests;
        for (std::size_t i = 0; i < sendPeers.size(); ++i)
        {
            for (auto [target, next] : requests)
            {
                if (owner(next) != sendPeers[i]) { continue; }
                sendBuffers[i].push_back({target, next});
                peerNodes(hops[l], sendPeers[i]).recv.push_back(target);
            }
            mpiSendAsync(sendBuffers[i].data(), sendBuffers[i].size(), sendPeers[i], tag, mpiRequests, comm);
        }

        for (int peer : recvPeers)
        {
            MPI_Status status;
            int        numInts;
            MPI_Probe(peer, tag, comm, &status);
            MPI_Get_count(&status, MpiType<TreeNodeIndex>{}, &numInts);

            std::vector<NodeRequest> received(numInts / 2);
            mpiRecvSync(received.data(), int(received.size()), peer, tag, MPI_STATUS_IGNORE, comm);
            for (auto [target, pos] : received)
            {
                peerNodes(hops[l], peer).send.push_back(target);
                queueRequest(target, pos);
            }
        }

        MPI_Waitall(int(mpiRequests.size()), mpiRequests.data(), MPI_STATUSES_IGNORE);
        requests.clear();
    };

    // sorted and unique targets
    std::sort(targets.begin(), targets.end());
    targets.erase(std::unique(targets.begin(), targets.end()), targets.end());

    // start at the end of my range that faces the target
    TreeNodeIndex firstLeaf = gOctree.leafToInternalSpan()[assignment.treeOffsetsConst()[myRank]];
    TreeNodeIndex lastLeaf  = gOctree.leafToInternalSpan()[assignment.treeOffsetsConst()[myRank + 1] - 1];
    for (TreeNodeIndex target : targets)
    {
        TreeNodeIndex start =
            decodePlaceholderBit(gOctree.prefixes[target]) < assignment[myRank] ? firstLeaf : lastLeaf;
        queueRequest(target, start);
    }

    // up rounds deepest edge level first, then down rounds top edge level first
    for (int l = depth - 1; l >= 0; --l)
    {
        exchangeRequests(l, true);
    }
    for (int l = 0; l < depth; ++l)
    {
        exchangeRequests(l, false);
    }
}

/*! @brief send/recv lists of one sweep (up or down) per parent level
 *
 * Message m covers the node indices sendIdx/recvIdx[m.offset : m.offset + m.count],
 * the messages of one level are contiguous
 */
struct SweepMessage
{
    int           peer;
    TreeNodeIndex offset, count;
};
template<class IdxVec = std::vector<TreeNodeIndex>>
struct SweepLists
{
    using Message = SweepMessage;
    std::vector<std::vector<Message>> send, recv;
    IdxVec                            sendIdx, recvIdx;
};

/*! @brief sort + unique the nodes of each peer and flatten them level by level
 *
 * Both sides of a pair sort the same node set, so sender and receiver agree on the order.
 */
inline SweepLists<> toSweepLists(LevelPairs& pairs)
{
    GlobalExchangeRange functionRange("fmm:toSweepLists");
    SweepLists<>        lists;
    lists.send.resize(pairs.size());
    lists.recv.resize(pairs.size());

    //! append the sorted, unique nodes of one peer as one message, skip empty ones
    auto addMessage = [](std::vector<TreeNodeIndex>& nodes, int peer, std::vector<SweepMessage>& msgs,
                         std::vector<TreeNodeIndex>& idx)
    {
        std::sort(nodes.begin(), nodes.end());
        nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
        if (nodes.empty()) { return; }
        msgs.push_back({peer, TreeNodeIndex(idx.size()), TreeNodeIndex(nodes.size())});
        idx.insert(idx.end(), nodes.begin(), nodes.end());
    };

    for (std::size_t l = 0; l < pairs.size(); ++l)
    {
        for (PeerNodes& e : pairs[l])
        {
            addMessage(e.send, e.peer, lists.send[l], lists.sendIdx);
            addMessage(e.recv, e.peer, lists.recv[l], lists.recvIdx);
        }
    }
    return lists;
}

//! Post receives into disjoint per-level slices before starting a sweep.
template<class Q, class IdxVec>
auto postSweepReceives(const SweepLists<IdxVec>& lists, int tag, std::span<Q> recvBuf, MPI_Comm comm)
{
    assert(lists.recvIdx.size() <= recvBuf.size());
    std::vector<std::vector<MPI_Request>> requests(lists.recv.size());
    for (std::size_t l = 0; l < lists.recv.size(); ++l)
    {
        for (const auto& m : lists.recv[l])
        {
            mpiRecvAsync(recvBuf.data() + m.offset, m.count, m.peer, tag + int(l), requests[l], comm);
        }
    }
    return requests;
}

/*! @brief send this level's values and consume its preposted receives
 */
template<class Exec, class Q, class IdxVec>
void exchangeLevel(Exec exec, const SweepLists<IdxVec>& lists, int l, int tag, std::span<Q> globQ, std::span<Q> sendBuf,
                   std::span<Q> recvBuf, std::vector<MPI_Request>& recvRequests, std::vector<MPI_Request>& sendRequests,
                   MPI_Comm comm)
{
    const auto& send = lists.send[l];
    const auto& recv = lists.recv[l];

    //! the messages of one level are one contiguous slice of sendIdx / recvIdx, starting at sFirst / rFirst
    auto levelNodes = [](const auto& msgs, const IdxVec& idx)
    {
        if (msgs.empty()) { return std::span<const TreeNodeIndex>{}; }
        return std::span<const TreeNodeIndex>(rawPtr(idx) + msgs.front().offset,
                                              msgs.back().offset + msgs.back().count - msgs.front().offset);
    };
    auto          sendNodes = levelNodes(send, lists.sendIdx), recvNodes = levelNodes(recv, lists.recvIdx);
    TreeNodeIndex sFirst = send.empty() ? 0 : send.front().offset, rFirst = recv.empty() ? 0 : recv.front().offset;
    assert(sFirst + sendNodes.size() <= sendBuf.size());
    assert(rFirst + recvNodes.size() <= recvBuf.size());

    //! Gather on the active execution space and send directly from the resulting buffer.
    if (!send.empty())
    {
        GlobalExchangeRange packRange("fmm:global-exchange/p2p/send");
        gather(exec, sendNodes, globQ.data(), sendBuf.data() + sFirst);
        if constexpr (execution::HaveGpu<Exec>{}) { syncGpu(exec); }
        for (const auto& m : send)
        {
            mpiSendAsync(sendBuf.data() + m.offset, m.count, m.peer, tag, sendRequests, comm);
        }
    }
    // wait for needed nodes
    if (!recv.empty())
    {
        GlobalExchangeRange waitRange("fmm:global-exchange/p2p/wait");
        MPI_Waitall(int(recvRequests.size()), recvRequests.data(), MPI_STATUSES_IGNORE);
        GlobalExchangeRange scatterRange("fmm:global-exchange/p2p/scatter");
        scatter(exec, recvNodes, recvBuf.data() + rFirst, globQ.data());
    }
}

//! Owned parent ranges, as in FocusedOctree::computeOwnedParentRanges.
template<class KeyType>
auto computeOwnedParentRanges(OctreeView<const KeyType> tree, const SfcAssignment<KeyType>& assignment, int rank)
{
    std::vector<std::array<TreeNodeIndex, 2>> owned;
    auto                                      levels = tree.levelRangeSpan();
    int                                       depth  = maxDepth(levels.data(), levels.size());
    for (int l = 0; l < depth; ++l)
    {
        auto lessKey = [](KeyType prefix, KeyType key) { return decodePlaceholderBit(prefix) < key; };
        auto first =
            std::lower_bound(tree.prefixes + levels[l], tree.prefixes + levels[l + 1], assignment[rank], lessKey);
        auto last = std::lower_bound(first, tree.prefixes + levels[l + 1], assignment[rank + 1], lessKey);
        owned.push_back({TreeNodeIndex(first - tree.prefixes), TreeNodeIndex(last - tree.prefixes)});
    }
    return owned;
}

/*! @brief lists of both sweeps and the owned parent ranges, built once in setup
 *
 * Message metadata (peer, offset, count) and owned ranges stay on the host, the node indices
 * sendIdx/recvIdx live in IdxVec (device for GPU), as the globUpLists_/globDownLists_ members in GROMACS.
 */
template<class IdxVec>
struct GlobalExchangeLists
{
    SweepLists<IdxVec>                        up, down;
    std::vector<std::array<TreeNodeIndex, 2>> ownedParents;
};

/*! @brief FocusedOctree::createGlobalExchangeLists with the wanted global nodes passed in
 *
 * @param[in] exec        execution policy
 * @param[in] gOctree     global octree
 * @param[in] assignment  global SFC assignment
 * @param[in] targets     global nodes this rank wants
 * @param[in] myRank      executing rank
 * @param[in] comm        communicator
 */
template<class IdxVec, class Exec, class KeyType>
GlobalExchangeLists<IdxVec> createGlobalExchangeLists(Exec exec, OctreeView<const KeyType> gOctree,
                                                      const SfcAssignment<KeyType>& assignment,
                                                      std::vector<TreeNodeIndex> targets, int myRank, MPI_Comm comm)
{
    GlobalExchangeRange                 setupRange("fmm:global-exchange/setup");
    OctreeData<KeyType, execution::Cpu> globalTreeHost;
    OctreeView<const KeyType>           globalTreeView = gOctree;
    if constexpr (execution::HaveGpu<Exec>{})
    {
        //! List construction and request routing on CPU
        downloadOctreeFromGpu(exec, globalTreeHost, gOctree);
        syncGpu(exec);
        globalTreeView = globalTreeHost.cdata();
    }

    //! get the basic upsweep communication pairs
    auto       upPeers = basePairs(globalTreeView, assignment, myRank);
    LevelPairs downPeers(upPeers.size());

    //! append the wanted nodes, routed over upsweep pairs
    routeRequests(globalTreeView, assignment, std::move(targets), myRank, comm, upPeers, downPeers);

    //! flatten into per-level send/recv lists, indices into the execution space
    auto toAcc = [](SweepLists<> host)
    {
        SweepLists<IdxVec> acc;
        acc.send    = std::move(host.send);
        acc.recv    = std::move(host.recv);
        acc.sendIdx = host.sendIdx;
        acc.recvIdx = host.recvIdx;
        return acc;
    };

    GlobalExchangeLists<IdxVec> lists;
    lists.up           = toAcc(toSweepLists(upPeers));
    lists.down         = toAcc(toSweepLists(downPeers));
    lists.ownedParents = computeOwnedParentRanges(globalTreeView, assignment, myRank);
    return lists;
}

//! FocusedOctree::globalSweep extracted without changing the execution order or synchronization.
template<class Exec, class KeyType, class Q, class IdxVec, class F, class... UArgs>
void globalSweep(Exec exec, OctreeView<const KeyType> gOctree, const GlobalExchangeLists<IdxVec>& lists,
                 std::span<Q> globQ, std::span<Q> sendBuf, std::span<Q> recvBuf, MPI_Comm comm, F&& upsweepFunction,
                 UArgs&&... upsweepArgs)
{
    GlobalExchangeRange functionRange("fmm:globalSweep");
    const auto&         up              = lists.up;
    const auto&         down            = lists.down;
    const auto&         ownedParents    = lists.ownedParents;
    int                 numParentLevels = int(up.send.size());

    // Keep all messages in separate slices so later receives and earlier sends can remain active.
    auto upSend       = sendBuf.first(up.sendIdx.size());
    auto downSend     = sendBuf.subspan(up.sendIdx.size());
    auto upRecv       = recvBuf.first(up.recvIdx.size());
    auto downRecv     = recvBuf.subspan(up.recvIdx.size());
    auto upRequests   = postSweepReceives(up, static_cast<int>(P2pTags::globalUp), upRecv, comm);
    auto downRequests = postSweepReceives(down, static_cast<int>(P2pTags::globalDown), downRecv, comm);

    //! send requests don't need to finsh by level (only at the end)
    std::vector<MPI_Request> sendRequests;
    std::size_t              numSends = 0;
    for (const auto& msgs : up.send)
    {
        numSends += msgs.size();
    }
    for (const auto& msgs : down.send)
    {
        numSends += msgs.size();
    }
    sendRequests.reserve(numSends);

    //! upsweep
    for (int l = numParentLevels - 1; l >= 0; --l)
    {
        char upRangeName[64];
        std::snprintf(upRangeName, sizeof(upRangeName), "fmm:global-exchange/p2p/up-level-%d", l);
        GlobalExchangeRange upRange(upRangeName);

        exchangeLevel(exec, up, l, static_cast<int>(P2pTags::globalUp) + l, globQ, upSend, upRecv, upRequests[l],
                      sendRequests, comm);

        //! compute the parents Q values, only for parents this rank owns
        auto owned = ownedParents[l];
        if (owned[0] < owned[1])
        {
            GlobalExchangeRange          parentRange("fmm:global-exchange/p2p/sweep");
            std::array<TreeNodeIndex, 2> ownedLevelRange{owned[0], owned[1]};
            upsweepFunction(std::span<const TreeNodeIndex>(ownedLevelRange), gOctree.childOffsets, globQ.data(),
                            upsweepArgs...);
        }
    }

    //! downsweep
    for (int l = 0; l < numParentLevels; ++l)
    {
        char downRangeName[64];
        std::snprintf(downRangeName, sizeof(downRangeName), "fmm:global-exchange/p2p/down-level-%d", l);
        GlobalExchangeRange upRange(downRangeName);
        exchangeLevel(exec, down, l, static_cast<int>(P2pTags::globalDown) + l, globQ, downSend, downRecv,
                      downRequests[l], sendRequests, comm);
    }

    // wait for all sends to complete
    GlobalExchangeRange sendWaitRange("fmm:global-exchange/p2p/send-wait");
    MPI_Waitall(int(sendRequests.size()), sendRequests.data(), MPI_STATUSES_IGNORE);
}

} // namespace global_sweep
