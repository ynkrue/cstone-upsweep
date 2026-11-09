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
    GlobalExchangeRange functionRange("cstone:basePairs");
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

//! @brief exchange graph with one node per combine or supplied value
struct ExchangeGraph
{
    std::vector<TreeNodeIndex> cellIds;       //!< global cell of each graph node
    std::vector<int>           computeRanks;  //!< rank producing each node
    std::vector<TreeNodeIndex> inputOffsets;
    std::vector<TreeNodeIndex> inputIds;      //!< graph node IDs of the inputs
    std::vector<TreeNodeIndex> targetOffsets;
    std::vector<int>           targetRanks;   //!< ranks that need the complete value

    TreeNodeIndex numNodes() const { return TreeNodeIndex(cellIds.size()); }
};

/*! @brief base exchange graph with one node per global cell
 */
template<class KeyType>
ExchangeGraph buildExchangeGraph(OctreeView<const KeyType> gOctree, const SfcAssignment<KeyType>& assignment)
{
    GlobalExchangeRange functionRange("cstone:buildExchangeGraph");
    ExchangeGraph       graph;
    auto owner = [&](TreeNodeIndex i) { return nodeOwner(gOctree, assignment, i); };

    graph.inputOffsets = {0};
    graph.targetOffsets = {0};
    for (TreeNodeIndex g = 0; g < gOctree.numNodes; ++g)
    {
        graph.cellIds.push_back(g);
        TreeNodeIndex firstChild = gOctree.childOffsets[g];
        if (firstChild == 0)
        {
            graph.computeRanks.push_back(owner(g));
        }
        else
        {
            graph.computeRanks.push_back(-1);
            for (int k = 0; k < 8; ++k)
            {
                 graph.inputIds.push_back(firstChild + k);
            }
        }
        graph.inputOffsets.push_back(graph.inputIds.size());
        graph.targetRanks.push_back(owner(g));
        graph.targetOffsets.push_back(graph.targetRanks.size());
    }
    return graph;
}

using TargetRequest = std::array<TreeNodeIndex, 3>; // {cell, position, requester}

template<class KeyType>
void addGraphTargets(OctreeView<const KeyType> gOctree, const SfcAssignment<KeyType>& assignment, ExchangeGraph& graph, std::span<const TreeNodeIndex> localTargets, const LevelPairs& setupPairs, int myRank, MPI_Comm comm)
{
    GlobalExchangeRange functionRange("cstone:addGraphTargets");
    constexpr int       tag = static_cast<int>(P2pTags::globalRequests);

    auto owner  = [&](TreeNodeIndex i) { return nodeOwner(gOctree, assignment, i); };
    auto parent = [&](TreeNodeIndex i) { return gOctree.parents[(i - 1) / 8]; };

    //! next node on the tree path from pos to cell
    auto nextNode = [&](TreeNodeIndex cell, TreeNodeIndex pos)
    { return containsNode(gOctree, pos, cell) ? childContaining(gOctree, pos, cell) : parent(pos); };

    std::vector<std::array<TreeNodeIndex, 2>> registered; // {cell, requester}

    // requests per edge level, waiting for their up or down round
    int                                     depth = int(setupPairs.size());
    std::vector<std::vector<TargetRequest>> upRequests(depth), downRequests(depth);

    //! register if I am the contact of cell, else walk while the path is mine and queue the next hop
    auto queueRequest = [&](TreeNodeIndex cell, TreeNodeIndex pos, TreeNodeIndex requester)
    {
        if (owner(cell) == myRank)
        {
            registered.push_back({cell, requester});
            return;
        }

        while (owner(nextNode(cell, pos)) == myRank)
        {
            pos = nextNode(cell, pos);
        }
        TreeNodeIndex next = nextNode(cell, pos);

        // the level of an edge is the level of its upper node
        if (containsNode(gOctree, pos, cell)) { downRequests[nodeLevel(gOctree, pos)].push_back({cell, next, requester}); }
        else { upRequests[nodeLevel(gOctree, next)].push_back({cell, next, requester}); }
    };

    //! exchange the requests of edge level l with the base partners of that level, one message per partner
    auto exchangeRequests = [&](int l, bool up)
    {
        std::vector<TargetRequest>& requests = up ? upRequests[l] : downRequests[l];

        // up requests go child owner to parent owner, down requests the reverse
        std::vector<int> sendPeers, recvPeers;
        for (const PeerNodes& e : setupPairs[l])
        {
            if (!(up ? e.send : e.recv).empty()) { sendPeers.push_back(e.peer); }
            if (!(up ? e.recv : e.send).empty()) { recvPeers.push_back(e.peer); }
        }

        std::vector<std::vector<TargetRequest>> sendBuffers(sendPeers.size());
        std::vector<MPI_Request>                mpiRequests;
        for (std::size_t i = 0; i < sendPeers.size(); ++i)
        {
            for (const TargetRequest& request : requests)
            {
                if (owner(request[1]) == sendPeers[i]) { sendBuffers[i].push_back(request); }
            }
            mpiSendAsync(sendBuffers[i].data(), sendBuffers[i].size(), sendPeers[i], tag, mpiRequests, comm);
        }

        for (int peer : recvPeers)
        {
            MPI_Status status;
            int        numRequests;
            MPI_Probe(peer, tag, comm, &status);
            mpiGetCount<TargetRequest>(&status, &numRequests);

            std::vector<TargetRequest> received(numRequests);
            mpiRecvSync(received.data(), numRequests, peer, tag, MPI_STATUS_IGNORE, comm);
            for (auto [cell, pos, requester] : received)
            {
                queueRequest(cell, pos, requester);
            }
        }

        MPI_Waitall(int(mpiRequests.size()), mpiRequests.data(), MPI_STATUSES_IGNORE);
        requests.clear();
    };

    // start at the end of my range that faces the cell
    TreeNodeIndex firstLeaf = gOctree.leafToInternalSpan()[assignment.treeOffsetsConst()[myRank]];
    TreeNodeIndex lastLeaf  = gOctree.leafToInternalSpan()[assignment.treeOffsetsConst()[myRank + 1] - 1];
    for (TreeNodeIndex cell : localTargets)
    {
        TreeNodeIndex start = decodePlaceholderBit(gOctree.prefixes[cell]) < assignment[myRank] ? firstLeaf : lastLeaf;
        queueRequest(cell, start, myRank);
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

    // merge the graph node on the old targets plus the registered requesters of its cell
    std::sort(registered.begin(), registered.end());
    std::vector<TreeNodeIndex> targetOffsets{0};
    std::vector<int>           targetRanks;
    auto                       reg = registered.begin();
    for (TreeNodeIndex g = 0; g < graph.numNodes(); ++g)
    {
        auto first = targetRanks.size();
        targetRanks.insert(targetRanks.end(), graph.targetRanks.begin() + graph.targetOffsets[g],
                           graph.targetRanks.begin() + graph.targetOffsets[g + 1]);
        for (; reg != registered.end() && (*reg)[0] == graph.cellIds[g]; ++reg)
        {
            targetRanks.push_back((*reg)[1]);
        }
        std::sort(targetRanks.begin() + first, targetRanks.end());
        targetRanks.erase(std::unique(targetRanks.begin() + first, targetRanks.end()), targetRanks.end());
        targetOffsets.push_back(TreeNodeIndex(targetRanks.size()));
    }
    graph.targetOffsets = std::move(targetOffsets);
    graph.targetRanks   = std::move(targetRanks);
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
    GlobalExchangeRange functionRange("cstone:toSweepLists");
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
        GlobalExchangeRange packRange("cstone:global-exchange/p2p/send");
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
        GlobalExchangeRange waitRange("cstone:global-exchange/p2p/wait");
        MPI_Waitall(int(recvRequests.size()), recvRequests.data(), MPI_STATUSES_IGNORE);
        GlobalExchangeRange scatterRange("cstone:global-exchange/p2p/scatter");
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

} // namespace global_sweep
