/*! @file
 * @brief point-to-point upsweep of the global tree, replaces the Allgatherv in FocusedOctree::globalExchange
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <span>
#include <vector>

#include <mpi.h>

#include "cstone/sfc/common.hpp"
#include "cstone/domain/domaindecomp.hpp"
#include "cstone/primitives/gather.hpp"
#include "cstone/primitives/mpi_wrappers.hpp"
#include "cstone/tree/octree.hpp"

namespace upsweep_p2p
{

using namespace cstone;

// MPI tags
constexpr int upsweepTag     = 6500;
constexpr int requestUpTag   = 6600;
constexpr int downsweepTag   = 6700;
constexpr int requestDownTag = 6800;
constexpr int upFlaggedTag   = 6900;

/*! @brief owner rank of global node i: the rank holding the node's first leaf
 *
 * @param[in]  gOctree     global octree, same on all ranks
 * @param[in]  assignment  SFC key range per rank
 * @param[in]  i           global node index
 */
template<class KeyType>
int nodeOwner(OctreeView<const KeyType> gOctree, const SfcAssignment<KeyType>& assignment, TreeNodeIndex i)
{ return assignment.findRank(decodePlaceholderBit(gOctree.prefixes[i])); }
{ return assignment.findRank(decodePlaceholderBit(gOctree.prefixes[i])); }

/*! @brief send/recv lists of one sweep (up or down) per parent level
 *
 * Message m covers the node indices sendIdx/recvIdx[m.offset : m.offset + m.count],
 * the messages of one level are contiguous
 */
struct SweepLists
{
    struct Message
    {
        int           peer;
        TreeNodeIndex offset, count;
    };
    std::vector<std::vector<Message>> send, recv;
    std::vector<TreeNodeIndex>        sendIdx, recvIdx;
};

//! @brief append node i for @p peer, consecutive nodes with the same peer extend the last message
inline void addNode(std::vector<SweepLists::Message>& msgs, std::vector<TreeNodeIndex>& idx, int peer, TreeNodeIndex i)
{
    if (msgs.empty() || msgs.back().peer != peer) { msgs.push_back({peer, TreeNodeIndex(idx.size()), 0}); }
    idx.push_back(i);
    msgs.back().count++;
}

//! @brief exchange the values of one level: send globQ[sendIdx], receive into globQ[recvIdx]
template<class Q>
void exchangeLevel(const SweepLists& lists, int l, int tag, std::span<Q> globQ, MPI_Comm comm)
{
    const auto& send = lists.send[l];
    const auto& recv = lists.recv[l];

    //! the messages of one level are one contiguous slice of sendIdx / recvIdx, starting at sFirst / rFirst
    auto levelNodes = [](const std::vector<SweepLists::Message>& msgs, const std::vector<TreeNodeIndex>& idx)
    {
        if (msgs.empty()) { return std::span<const TreeNodeIndex>{}; }
        return std::span<const TreeNodeIndex>(idx).subspan(msgs.front().offset, msgs.back().offset + msgs.back().count -
                                                                                    msgs.front().offset);
    };
    auto           sendNodes = levelNodes(send, lists.sendIdx), recvNodes = levelNodes(recv, lists.recvIdx);
    TreeNodeIndex  sFirst = send.empty() ? 0 : send.front().offset, rFirst = recv.empty() ? 0 : recv.front().offset;
    std::vector<Q> sendBuf(sendNodes.size()), recvBuf(recvNodes.size());

    std::vector<MPI_Request> mpiRequests;
    for (const auto& m : recv)
    {
        mpiRecvAsync(recvBuf.data() + m.offset - rFirst, m.count, m.peer, tag, mpiRequests, comm);
    }
    gather(sendNodes, globQ.data(), sendBuf.data());
    for (const auto& m : send)
    {
        mpiSendAsync(sendBuf.data() + m.offset - sFirst, m.count, m.peer, tag, mpiRequests, comm);
    }
    MPI_Waitall(int(mpiRequests.size()), mpiRequests.data(), MPI_STATUSES_IGNORE);
    scatter(recvNodes, recvBuf.data(), globQ.data());
}

/*! @brief upsweep lists between siblings: children 1-7 go to the owner of child 0, which owns the parent
 *
 * @param[in] gOctree  global octree, same on all ranks
 * @return             {up, down} send/recv lists per parent level
 *
 * Up lists, children 1-7 go to the owner of child 0, which owns the parent.
 * Down lists, the same pairs reversed, the parent owner sends back the children of the parent and of its ancestors,
 * until the child owner has its next pair. Overlap flags only, no communication.
 */
template<class KeyType>
std::pair<SweepLists, SweepLists> computeSweepLists(OctreeView<const KeyType>     gOctree,
                                                    const SfcAssignment<KeyType>& assignment, // becomes member
                                                    int                           myRank_)    // becomes member
{
    auto levelRange = gOctree.levelRangeSpan();
    int  numLevels  = maxDepth(levelRange.data(), levelRange.size()); // L
    auto globDispl  = assignment.treeOffsetsConst();
    auto parent     = [&](TreeNodeIndex i) { return gOctree.parents[(i - 1) / 8]; };
    auto owner      = [&](TreeNodeIndex i) { return nodeOwner(gOctree, assignment, i); };

    //! children of i and its ancestors that go back down to the receiver, stop at the receiver's next pair
    auto addDown = [&](std::vector<SweepLists::Message>& msgs, std::vector<TreeNodeIndex>& idx, int peer, int receiver,
                       TreeNodeIndex i)
    {
        for (TreeNodeIndex n = i;; n = parent(n))
        {
            for (TreeNodeIndex ch = gOctree.childOffsets[n]; ch < gOctree.childOffsets[n] + 8; ++ch)
            {
                if (owner(ch) != receiver) { addNode(msgs, idx, peer, ch); }
            }
            if (n == 0) { break; } // root

            // receiver owns a sibling of n, so it has its own pair one level up
            TreeNodeIndex firstSibling = gOctree.childOffsets[parent(n)];
            bool          hasPair      = false;
            for (TreeNodeIndex s = firstSibling; s < firstSibling + 8; ++s)
            {
                hasPair |= owner(s) == receiver;
            }
            if (hasPair) { break; }
        }
    };

    //! ancestors of the first leaf of every rank > 0
    std::vector<TreeNodeIndex> boundaryParents;
    for (int r = 1; r < assignment.numRanks(); ++r)
    {
        if (globDispl[r] == globDispl[r + 1]) { continue; } // empty rank
        for (TreeNodeIndex n = gOctree.leafToInternalSpan()[globDispl[r]]; n != 0; n = parent(n))
        {
            boundaryParents.push_back(parent(n));
        }
    }
    std::sort(boundaryParents.begin(), boundaryParents.end());
    boundaryParents.erase(std::unique(boundaryParents.begin(), boundaryParents.end()), boundaryParents.end());

    SweepLists up, down;
    up.send.resize(numLevels);
    up.recv.resize(numLevels);
    down.send.resize(numLevels);
    down.recv.resize(numLevels);

    int l = 0; // level of parent i
    for (TreeNodeIndex i : boundaryParents)
    {
        while (i >= levelRange[l + 1])
        {
            ++l;
        }
        int           parentOwner = owner(i);
        TreeNodeIndex firstChild  = gOctree.childOffsets[i];
        for (TreeNodeIndex c = firstChild + 1; c < firstChild + 8; ++c) // child 0 has the parent's owner
        {
            int childOwner = owner(c);
            if (childOwner == parentOwner) { continue; }
            bool firstOfOwner = owner(c - 1) != childOwner; // only the first child of each owner adds the down message
            if (childOwner == myRank_)
            {
                addNode(up.send[l], up.sendIdx, parentOwner, c);
                if (firstOfOwner) { addDown(down.recv[l], down.recvIdx, parentOwner, childOwner, i); }
            }
            if (parentOwner == myRank_)
            {
                addNode(up.recv[l], up.recvIdx, childOwner, c);
                if (firstOfOwner) { addDown(down.send[l], down.sendIdx, childOwner, childOwner, i); }
            }
        }
    }
    return {up, down};
}

/*! @brief route the requests for my flagged nodes over the sibling pairs, before the upsweep
 *
 * @param[in] gOctree     global octree, same on all ranks
 * @param[in] up          upsweep lists from computeUpsweepLists
 * @param[in] flags_      rank-local, length numNodes: I need the children of every flagged node
 * @param[in] assignment  SFC key range per rank
 * @param[in] comm        MPI communicator
 * @return                {upFlagged, downFlagged}: children of flagged nodes to send/receive per level in the
 *                        upsweep / downsweep of globalUpsweep
 *
 * A request (n, a): I need the children of n, the request is at node a. It climbs from my first leaf until a contains n
 * (up loop), then descends to owner(n) (down loop); it changes rank only on an up pair, or on a reversed one going
 * down. Every hop is recorded in reverse: up hop -> downFlagged, down hop -> upFlagged.
 */
template<class KeyType>
std::pair<SweepLists, SweepLists> computeExchangeLists(OctreeView<const KeyType> gOctree, const SweepLists& up,
                                                       std::span<const uint8_t>      flags_,     // becomes member
                                                       const SfcAssignment<KeyType>& assignment, // becomes member
                                                       MPI_Comm                      comm)
{
    int myRank;
    MPI_Comm_rank(comm, &myRank);
    auto globDispl       = assignment.treeOffsetsConst();
    int  numParentLevels = int(up.send.size());

    auto owner    = [&](TreeNodeIndex i) { return nodeOwner(gOctree, assignment, i); };
    auto parent   = [&](TreeNodeIndex i) { return gOctree.parents[(i - 1) / 8]; };
    auto level    = [&](TreeNodeIndex i) { return int(decodePrefixLength(gOctree.prefixes[i])) / 3; };
    auto ancestor = [&](TreeNodeIndex n, int lev) -> TreeNodeIndex
    {
        while (level(n) > lev)
        {
            n = parent(n);
        }
        return n;
    };
    auto contains = [&](TreeNodeIndex a, TreeNodeIndex n) -> bool
    { return level(n) >= level(a) && ancestor(n, level(a)) == a; };
    auto addChildren = [&](std::vector<SweepLists::Message>& msgs, std::vector<TreeNodeIndex>& idx, int peer,
                           std::span<const TreeNodeIndex> nodes)
    {
        for (TreeNodeIndex n : nodes)
        {
            TreeNodeIndex firstChild = gOctree.childOffsets[n];
            for (TreeNodeIndex c = firstChild; c < firstChild + 8; ++c)
            {
                addNode(msgs, idx, peer, c);
            }
        }
    };

    SweepLists upFlagged, downFlagged;
    upFlagged.send.resize(numParentLevels);
    upFlagged.recv.resize(numParentLevels);
    downFlagged.send.resize(numParentLevels);
    downFlagged.recv.resize(numParentLevels);

    if (globDispl[myRank] == globDispl[myRank + 1]) { return {upFlagged, downFlagged}; } // empty rank

    //! my requests (n, a): I need the children of n, the request is at node a
    auto                                                 firstLeaf = gOctree.leafToInternalSpan()[globDispl[myRank]];
    std::vector<std::pair<TreeNodeIndex, TreeNodeIndex>> upRequests, downRequests;
    for (TreeNodeIndex i = 0; i < gOctree.numInternalNodes; ++i)
    {
        if (flags_[i] && owner(i) != myRank) { upRequests.push_back({i, firstLeaf}); }
    }

    //! up loop, deepest level first
    for (int l = numParentLevels - 1; l >= 0; --l)
    {
        std::vector<TreeNodeIndex> sendNodes;
        for (auto& [n, a] : upRequests)
        {
            while (true)
            {
                if (contains(a, n))
                {
                    downRequests.push_back({n, a}); // turn around
                    n = -1;
                    break;
                }

                // climb one level
                TreeNodeIndex p = parent(a);
                if (owner(p) == myRank)
                {
                    a = p;
                    continue;
                }

                if (level(p) == l) // hop to owner of p
                {
                    sendNodes.push_back(n);
                    n = -1;
                }
                break;
            }
        }
        std::erase_if(upRequests, [](const auto& req) { return req.first == -1; });

        std::vector<MPI_Request> mpiRequests;
        for (const auto& m : up.send[l])
        {
            mpiSendAsync(sendNodes.data(), sendNodes.size(), m.peer, requestUpTag + l, mpiRequests, comm);
            addChildren(downFlagged.recv[l], downFlagged.recvIdx, m.peer, sendNodes);
        }
        for (const auto& m : up.recv[l])
        {
            MPI_Status status;
            MPI_Probe(m.peer, requestUpTag + l, comm, &status);
            int count;
            mpiGetCount<TreeNodeIndex>(&status, &count);

            std::vector<TreeNodeIndex> recvBuf(count);
            mpiRecvSync(recvBuf.data(), count, m.peer, requestUpTag + l, MPI_STATUS_IGNORE, comm);
            addChildren(downFlagged.send[l], downFlagged.sendIdx, m.peer, recvBuf);
            TreeNodeIndex a = parent(up.recvIdx[m.offset]);
            for (TreeNodeIndex n : recvBuf)
            {
                upRequests.push_back({n, a});
            }
        }
        MPI_Waitall(int(mpiRequests.size()), mpiRequests.data(), MPI_STATUSES_IGNORE);
    }

    //! requests received in round 0 are at the root, which contains every node
    downRequests.insert(downRequests.end(), upRequests.begin(), upRequests.end());

    //! down loop, root first
    for (int l = 0; l < numParentLevels; ++l)
    {
        std::vector<std::pair<int, TreeNodeIndex>> sendNodes; // (peer, n)
        for (auto& [n, a] : downRequests)
        {
            while (true)
            {
                // arrived: I own n and supply its children
                if (owner(n) == myRank)
                {
                    n = -1;
                    break;
                }

                // descend one level, to the child of a that contains n
                TreeNodeIndex c = ancestor(n, level(a) + 1);
                if (owner(c) == myRank)
                {
                    a = c;
                    continue;
                }

                if (level(a) == l) // hop to the owner of c
                {
                    sendNodes.push_back({owner(c), n});
                    n = -1;
                }
                break;
            }
        }
        std::erase_if(downRequests, [](const auto& req) { return req.first == -1; });

        //! one message to every child owner of this level, also when empty, the peer probes for it
        std::vector<std::vector<TreeNodeIndex>> sendBufs(up.recv[l].size());
        std::vector<MPI_Request>                mpiRequests;
        for (size_t k = 0; k < up.recv[l].size(); ++k)
        {
            int peer = up.recv[l][k].peer;
            for (const auto& [toPeer, n] : sendNodes)
            {
                if (toPeer == peer) { sendBufs[k].push_back(n); }
            }
            mpiSendAsync(sendBufs[k].data(), sendBufs[k].size(), peer, requestDownTag + l, mpiRequests, comm);
            addChildren(upFlagged.recv[l], upFlagged.recvIdx, peer, sendBufs[k]);
        }
        for (const auto& m : up.send[l])
        {
            MPI_Status status;
            MPI_Probe(m.peer, requestDownTag + l, comm, &status);
            int count;
            mpiGetCount<TreeNodeIndex>(&status, &count);

            std::vector<TreeNodeIndex> recvBuf(count);
            mpiRecvSync(recvBuf.data(), count, m.peer, requestDownTag + l, MPI_STATUS_IGNORE, comm);
            addChildren(upFlagged.send[l], upFlagged.sendIdx, m.peer, recvBuf);
            for (TreeNodeIndex n : recvBuf)
            {
                downRequests.push_back({n, ancestor(n, l + 1)});
            } // at my child
        }
        MPI_Waitall(int(mpiRequests.size()), mpiRequests.data(), MPI_STATUSES_IGNORE);
    }
    return {upFlagged, downFlagged};
}

/*! @brief p2p upsweep + downsweep of the global tree
 *
 * @param[in]    gOctree          global octree, same on all ranks
 * @param[in]    up               upsweep lists from computeUpsweepLists
 * @param[in]    upFlagged        children of flagged nodes for the upsweep, from computeExchangeLists
 * @param[in]    downFlagged      children of flagged nodes for the downsweep, from computeExchangeLists
 * @param[inout] globQ            in: my global leaf quantities at their node positions
 *                                out: valid for the nodes I own and the children of my flagged nodes
 * @param[in]    comm             MPI communicator
 * @param[in]    upsweepFunction  upsweep of one level
 * @param[in]    upsweepArgs      extra upsweep arguments, e.g. expansion centers
 *
 * Upsweep, deepest level first: children to the parent owner, compute the parents, then the flagged children of this
 * round. Downsweep, root first: the flagged children.
 */
template<class Q, class KeyType, class F, class... UArgs>
void globalUpsweep(cstone::OctreeView<const KeyType> gOctree,
                   const SweepLists&                 up,          // becomes member
                   const SweepLists&                 upFlagged,   // becomes member
                   const SweepLists&                 downFlagged, // becomes member
                   std::span<Q> globQ, MPI_Comm comm, F&& upsweepFunction, UArgs&&... upsweepArgs)
{
    auto levelRange      = gOctree.levelRangeSpan();
    int  numParentLevels = int(up.send.size());

    //! upsweep
    for (int l = numParentLevels - 1; l >= 0; --l)
    {
        exchangeLevel(up, l, upsweepTag + l, globQ, comm);

        //! compute the parents Q values
        upsweepFunction(levelRange.subspan(l, 2), gOctree.childOffsets, globQ.data(), upsweepArgs...);

        exchangeLevel(upFlagged, l, upFlaggedTag + l, globQ, comm);
    }

    //! downsweep
    for (int l = 0; l < numParentLevels; ++l)
    {
        exchangeLevel(downFlagged, l, downsweepTag + l, globQ, comm);
    }
}

} // namespace upsweep_p2p
