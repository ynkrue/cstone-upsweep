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
constexpr int downsweepTag   = 6600;
constexpr int requestUpTag   = 6700;
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

/*! @brief up lists between siblings: children 1-7 go to the owner of child 0, which owns the parent
 *
 * @param[in] gOctree  global octree, same on all ranks
 * @return             send/recv lists per parent level
 */
template<class KeyType>
SweepLists computeSweepLists(OctreeView<const KeyType>     gOctree,
                             const SfcAssignment<KeyType>& assignment, // becomes member
                             int                           myRank_)    // becomes member
{
    auto levelRange = gOctree.levelRangeSpan();
    int  numLevels  = maxDepth(levelRange.data(), levelRange.size()); // L
    auto globDispl  = assignment.treeOffsetsConst();
    auto globDispl  = assignment.treeOffsetsConst();
    auto parent     = [&](TreeNodeIndex i) { return gOctree.parents[(i - 1) / 8]; };
    auto owner      = [&](TreeNodeIndex i) { return nodeOwner(gOctree, assignment, i); };

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

//! @brief append the 8 children of each node in @p nodes for @p peer
inline void addChildren(const TreeNodeIndex* childOffsets, std::vector<SweepLists::Message>& msgs,
                        std::vector<TreeNodeIndex>& idx, int peer, std::span<const TreeNodeIndex> nodes)
{
    for (TreeNodeIndex n : nodes)
    {
        for (TreeNodeIndex c = childOffsets[n]; c < childOffsets[n] + 8; ++c)
        {
            addNode(msgs, idx, peer, c);
        }
    }
}

/*! @brief route the requests for my flagged nodes over the sibling pairs
 *
 * @return {upFlagged, downFlagged}: children of flagged nodes, sent/received per level in the upsweep / downsweep
 *
 * A request for n (I need the children of n) climbs from my first leaf until the current node contains n (up loop),
 * then descends to owner(n) (down loop). It changes rank only on an up pair, or on a reversed one going down. Every hop
 * is recorded in reverse: up hop -> downFlagged, down hop -> upFlagged.
 */
template<class KeyType>
std::pair<SweepLists, SweepLists> computeExchangeLists(OctreeView<const KeyType>     gOctree,
                                                       const SfcAssignment<KeyType>& assignment, const SweepLists& up,
                                                       std::span<const uint8_t> flags, MPI_Comm comm)
{
    int myRank;
    MPI_Comm_rank(comm, &myRank);
    auto globDispl       = assignment.treeOffsetsConst();
    int  numParentLevels = int(up.send.size());

    auto owner    = [&](TreeNodeIndex i) { return nodeOwner(gOctree, assignment, i); };
    auto parent   = [&](TreeNodeIndex i) { return gOctree.parents[(i - 1) / 8]; };
    auto level    = [&](TreeNodeIndex i) { return int(decodePrefixLength(gOctree.prefixes[i])) / 3; };
    auto keyStart = [&](TreeNodeIndex i) { return decodePlaceholderBit(gOctree.prefixes[i]); };
    //! true if node a is n or an ancestor of n
    auto contains = [&](TreeNodeIndex a, TreeNodeIndex n)
    {
        return level(n) >= level(a) && keyStart(n) >= keyStart(a) &&
               keyStart(n) < keyStart(a) + nodeRange<KeyType>(level(a));
    };
    //! child of a that contains n, a contains n and a != n
    auto childToward = [&](TreeNodeIndex a, TreeNodeIndex n)
    { return gOctree.childOffsets[a] + TreeNodeIndex(octalDigit(keyStart(n), level(a) + 1)); };
    auto sortUnique = [](auto& v)
    {
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
    };
    //! variable length request message from a known peer
    auto recvRequests = [&](int peer, int tag)
    {
        MPI_Status status;
        int        count;
        MPI_Probe(peer, tag, comm, &status);
        mpiGetCount<TreeNodeIndex>(&status, &count);
        std::vector<TreeNodeIndex> nodes(count);
        mpiRecvSync(nodes.data(), count, peer, tag, MPI_STATUS_IGNORE, comm);
        return nodes;
    };

    SweepLists upFlagged, downFlagged;
    upFlagged.send.resize(numParentLevels);
    upFlagged.recv.resize(numParentLevels);
    downFlagged.send.resize(numParentLevels);
    downFlagged.recv.resize(numParentLevels);
    if (globDispl[myRank] == globDispl[myRank + 1]) { return {upFlagged, downFlagged}; } // empty rank

    //! buckets per round: n to send up to my up peer, (peer, n) to send down to a child owner
    std::vector<std::vector<TreeNodeIndex>>                 upSend(numParentLevels);
    std::vector<std::vector<std::pair<int, TreeNodeIndex>>> downSend(numParentLevels);

    //! from node a (mine, contains n): descend while I own the child, then wait for the hop round level(a)
    auto descend = [&](TreeNodeIndex n, TreeNodeIndex a)
    {
        while (owner(n) != myRank)
        {
            TreeNodeIndex c = childToward(a, n);
            if (owner(c) != myRank)
            {
                downSend[level(a)].push_back({owner(c), n});
                return;
            }
            a = c;
        }
    };
    //! from node a (mine): climb while I own the parent, then wait for the hop round level(parent); turn where a
    //! contains n
    auto climb = [&](TreeNodeIndex n, TreeNodeIndex a)
    {
        while (!contains(a, n))
        {
            TreeNodeIndex p = parent(a);
            if (owner(p) != myRank)
            {
                upSend[level(p)].push_back(n);
                return;
            }
            a = p;
        }
        descend(n, a);
    };

    //! my requests start at my first leaf
    TreeNodeIndex firstLeaf = gOctree.leafToInternalSpan()[globDispl[myRank]];
    for (TreeNodeIndex n = 0; n < gOctree.numNodes; ++n)
    {
        if (flags[n] && gOctree.childOffsets[n] != 0 && owner(n) != myRank) { climb(n, firstLeaf); }
    }

    //! up loop, deepest level first
    for (int l = numParentLevels - 1; l >= 0; --l)
    {
        auto& out = upSend[l];
        sortUnique(out);
        std::vector<MPI_Request> mpiRequests;
        for (const auto& m : up.send[l]) // sent also when empty, the peer probes for it
        {
            mpiSendAsync(out.data(), out.size(), m.peer, requestUpTag + l, mpiRequests, comm);
            addChildren(gOctree.childOffsets, downFlagged.recv[l], downFlagged.recvIdx, m.peer, out);
        }
        for (const auto& m : up.recv[l])
        {
            auto in = recvRequests(m.peer, requestUpTag + l);
            addChildren(gOctree.childOffsets, downFlagged.send[l], downFlagged.sendIdx, m.peer, in);
            TreeNodeIndex a = parent(up.recvIdx[m.offset]); // all children a peer sends at one level share this parent
            for (TreeNodeIndex n : in)
            {
                climb(n, a);
            }
        }
        MPI_Waitall(int(mpiRequests.size()), mpiRequests.data(), MPI_STATUSES_IGNORE);
    }

    //! down loop, root first
    for (int l = 0; l < numParentLevels; ++l)
    {
        sortUnique(downSend[l]);
        std::vector<std::vector<TreeNodeIndex>> out(up.recv[l].size());
        std::vector<MPI_Request>                mpiRequests;
        for (size_t k = 0; k < up.recv[l].size(); ++k) // to every child owner, also when empty
        {
            int peer = up.recv[l][k].peer;
            for (auto [p, n] : downSend[l])
            {
                if (p == peer) { out[k].push_back(n); }
            }
            mpiSendAsync(out[k].data(), out[k].size(), peer, requestDownTag + l, mpiRequests, comm);
            addChildren(gOctree.childOffsets, upFlagged.recv[l], upFlagged.recvIdx, peer, out[k]);
        }
        for (const auto& m : up.send[l])
        {
            auto in = recvRequests(m.peer, requestDownTag + l);
            addChildren(gOctree.childOffsets, upFlagged.send[l], upFlagged.sendIdx, m.peer, in);
            TreeNodeIndex p = parent(up.sendIdx[m.offset]); // the peer owns p, the n arrive at my child of p
            for (TreeNodeIndex n : in)
            {
                descend(n, childToward(p, n));
            }
        }
        MPI_Waitall(int(mpiRequests.size()), mpiRequests.data(), MPI_STATUSES_IGNORE);
    }
    return {upFlagged, downFlagged};
}

/*! @brief p2p upsweep + downsweep of the global tree
 *
 * @param[in]    gOctree          global octree, same on all ranks
 * @param[in]    up               up lists from computeSweepLists
 * @param[in]    flags            length numNodes, I need the children of every flagged node
 * @param[inout] globQ            in: my global leaf quantities at their node positions
 *                                out: valid for the nodes I own and the children of my flagged nodes
 * @param[in]    comm             MPI communicator
 * @param[in]    upsweepFunction  upsweep of one level
 * @param[in]    upsweepArgs      extra upsweep arguments, e.g. expansion centers
 *
 * Pass 1: computeExchangeLists routes the requests. Pass 2, upsweep: per level the children to the parent owner,
 * compute the parents, then the flagged children of this round. Downsweep: per level the flagged children.
 */
template<class Q, class KeyType, class F, class... UArgs>
void globalUpsweep(cstone::OctreeView<const KeyType> gOctree,
                   const SfcAssignment<KeyType>&     assignment, // becomes member
                   const SweepLists&                 up,         // becomes member
                   std::span<const uint8_t> flags, std::span<Q> globQ, MPI_Comm comm, F&& upsweepFunction,
                   UArgs&&... upsweepArgs)
{
    auto levelRange               = gOctree.levelRangeSpan();
    int  numParentLevels          = int(up.send.size());
    auto [upFlagged, downFlagged] = computeExchangeLists(gOctree, assignment, up, flags, comm);

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
