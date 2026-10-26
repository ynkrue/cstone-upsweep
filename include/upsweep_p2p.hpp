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
constexpr int upsweepTag   = 6500; // child owner to parent owner
constexpr int downsweepTag = 6600; // parent owner to child owner

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

//! @brief append the 8 children of every flagged node for @p peer
inline void addFlagged(const TreeNodeIndex* childOffsets, std::vector<SweepLists::Message>& msgs,
                       std::vector<TreeNodeIndex>& idx, int peer, std::span<const uint8_t> flags)
{
    for (TreeNodeIndex n = 0; n < TreeNodeIndex(flags.size()); ++n)
    {
        for (TreeNodeIndex c = childOffsets[n]; flags[n] && c < childOffsets[n] + 8; ++c)
        {
            addNode(msgs, idx, peer, c);
        }
    }
}

/*! @brief lists that bring the children of my flagged nodes to me
 *
 * @param[in] gOctree     global octree, same on all ranks
 * @param[in] assignment  SFC key range per rank
 * @param[in] up          up lists from computeSweepLists
 * @param[in] flags       length numNodes, I need the children of every flagged node
 * @param[in] comm        MPI communicator
 * @return                {upFlagged, downFlagged} lists for globalUpsweep
 *
 * Pass 1. The flags go up the up pairs until a rank owns the node, the rest to the root and down the same pairs to
 * the owner. Every hop is recorded reversed, a hop up becomes a down hop of the children, a hop down an up hop.
 */
template<class KeyType>
std::pair<SweepLists, SweepLists> computeFlagLists(OctreeView<const KeyType>     gOctree,
                                                   const SfcAssignment<KeyType>& assignment, const SweepLists& up,
                                                   std::span<const uint8_t> flags, MPI_Comm comm)
{
    int myRank;
    MPI_Comm_rank(comm, &myRank);
    TreeNodeIndex numNodes  = gOctree.numNodes;
    int           numLevels = int(up.send.size());
    auto          owner     = [&](TreeNodeIndex n) { return nodeOwner(gOctree, assignment, n); };
    auto          parent    = [&](TreeNodeIndex n) { return gOctree.parents[(n - 1) / 8]; };
    auto          level     = [&](TreeNodeIndex n) { return int(decodePrefixLength(gOctree.prefixes[n])) / 3; };

    // extra nodes that go up / down during the sweeps
    SweepLists upFlagged, downFlagged;
    upFlagged.send.resize(numLevels);
    upFlagged.recv.resize(numLevels);
    downFlagged.send.resize(numLevels);
    downFlagged.recv.resize(numLevels);

    //! requests on my rank, flagged nodes owned by another rank
    std::vector<uint8_t> held(numNodes);
    for (TreeNodeIndex n = 0; n < numNodes; ++n)
    {
        held[n] = flags[n] && gOctree.childOffsets[n] != 0 && owner(n) != myRank;
    }

    //! up phase, send requests to the parent owner level by level
    for (int l = numLevels - 1; l >= 0; --l)
    {
        // receive the requests of my child owners
        std::vector<uint8_t>     recvFlags(up.recv[l].size() * numNodes);
        std::vector<MPI_Request> mpiRequests;
        for (size_t i = 0; i < up.recv[l].size(); ++i)
        {
            mpiRecvAsync(recvFlags.data() + i * numNodes, numNodes, up.recv[l][i].peer, upsweepTag + l, mpiRequests,
                         comm);
        }
        // send my requests to my parent owner, it sends the children back in the downsweep
        for (const auto& m : up.send[l])
        {
            mpiSendAsync(held.data(), numNodes, m.peer, upsweepTag + l, mpiRequests, comm);
            addFlagged(gOctree.childOffsets, downFlagged.recv[l], downFlagged.recvIdx, m.peer, held);
        }
        MPI_Waitall(int(mpiRequests.size()), mpiRequests.data(), MPI_STATUSES_IGNORE);

        if (!up.send[l].empty()) { std::fill(held.begin(), held.end(), 0); } // my parent owner has them now

        // send the children back in the downsweep, keep requests for nodes I don't own
        for (size_t i = 0; i < up.recv[l].size(); ++i)
        {
            std::span<const uint8_t> peerFlags(recvFlags.data() + i * numNodes, numNodes);
            addFlagged(gOctree.childOffsets, downFlagged.send[l], downFlagged.sendIdx, up.recv[l][i].peer, peerFlags);
            for (TreeNodeIndex n = 0; n < numNodes; ++n)
            {
                held[n] |= peerFlags[n] && owner(n) != myRank;
            }
        }
    }

    //! down phase, root first, the owner of the next node toward n gets the flag
    for (int l = 0; l < numLevels; ++l)
    {
        std::vector<std::vector<uint8_t>> sendFlags(up.recv[l].size(), std::vector<uint8_t>(numNodes));
        for (TreeNodeIndex n = 0; n < numNodes; ++n)
        {
            if (!held[n] || level(n) <= l) { continue; }
            TreeNodeIndex ancestor = n; // of n at level l + 1
            while (level(ancestor) > l + 1)
            {
                ancestor = parent(ancestor);
            }
            for (size_t i = 0; i < up.recv[l].size(); ++i)
            {
                if (up.recv[l][i].peer == owner(ancestor))
                {
                    sendFlags[i][n] = 1;
                    held[n]       = 0;
                }
            }
        }

        // receive the requests of my parent owner
        std::vector<uint8_t>     recvFlags(up.send[l].size() * numNodes);
        std::vector<MPI_Request> mpiRequests;
        for (size_t i = 0; i < up.send[l].size(); ++i)
        {
            mpiRecvAsync(recvFlags.data() + i * numNodes, numNodes, up.send[l][i].peer, downsweepTag + l, mpiRequests,
                         comm);
        }
        // send the requests to my child owners, they send the children up in the upsweep
        for (size_t i = 0; i < up.recv[l].size(); ++i)
        {
            mpiSendAsync(sendFlags[i].data(), numNodes, up.recv[l][i].peer, downsweepTag + l, mpiRequests, comm);
            addFlagged(gOctree.childOffsets, upFlagged.recv[l], upFlagged.recvIdx, up.recv[l][i].peer, sendFlags[i]);
        }
        MPI_Waitall(int(mpiRequests.size()), mpiRequests.data(), MPI_STATUSES_IGNORE);

        // send the children up in the upsweep, pass the requests on one level down
        for (size_t i = 0; i < up.send[l].size(); ++i)
        {
            std::span<const uint8_t> peerFlags(recvFlags.data() + i * numNodes, numNodes);
            addFlagged(gOctree.childOffsets, upFlagged.send[l], upFlagged.sendIdx, up.send[l][i].peer, peerFlags);
            for (TreeNodeIndex n = 0; n < numNodes; ++n)
            {
                held[n] |= peerFlags[n];
            }
        }
    }
    return {upFlagged, downFlagged};
}

/*! @brief merge two lists into one message per level and peer, the nodes of @p a first, then those of @p b
 *
 * Sender and receiver merge in the same order, so the merged send and recv lists still match.
 */
inline SweepLists mergeLists(const SweepLists& a, const SweepLists& b)
{
    using Messages = std::vector<SweepLists::Message>;
    using Indices  = std::vector<TreeNodeIndex>;

    auto mergeLevel = [](const Messages& aMsgs, const Indices& aIdx, const Messages& bMsgs, const Indices& bIdx,
                         Messages& msgs, Indices& idx)
    {
        //! peers of the level in order of first appearance
        std::vector<int> peers;
        for (const auto* src : {&aMsgs, &bMsgs})
        {
            for (const auto& m : *src)
            {
                if (std::find(peers.begin(), peers.end(), m.peer) == peers.end()) { peers.push_back(m.peer); }
            }
        }
        auto append = [&](int peer, const Messages& srcMsgs, const Indices& srcIdx)
        {
            for (const auto& m : srcMsgs)
            {
                for (TreeNodeIndex k = m.offset; m.peer == peer && k < m.offset + m.count; ++k)
                {
                    addNode(msgs, idx, peer, srcIdx[k]);
                }
            }
        };
        for (int peer : peers)
        {
            append(peer, aMsgs, aIdx);
            append(peer, bMsgs, bIdx);
        }
    };

    SweepLists merged;
    merged.send.resize(a.send.size());
    merged.recv.resize(a.recv.size());
    for (size_t l = 0; l < a.send.size(); ++l)
    {
        mergeLevel(a.send[l], a.sendIdx, b.send[l], b.sendIdx, merged.send[l], merged.sendIdx);
        mergeLevel(a.recv[l], a.recvIdx, b.recv[l], b.recvIdx, merged.recv[l], merged.recvIdx);
    }
    return merged;
}

/*! @brief p2p upsweep + downsweep of the global tree
 *
 * @param[in]    gOctree          global octree, same on all ranks
 * @param[in]    up               up lists merged with the upFlagged lists of computeFlagLists
 * @param[in]    down             downFlagged lists from computeFlagLists
 * @param[inout] globQ            in: my global leaf quantities at their node positions
 *                                out: valid for the nodes I own and the children of my flagged nodes
 * @param[in]    comm             MPI communicator
 * @param[in]    upsweepFunction  upsweep of one level
 * @param[in]    upsweepArgs      extra upsweep arguments, e.g. expansion centers
 *
 * Pass 2. Upsweep per level, up lists to the parent owners, then I compute my parents.
 * Downsweep per level, down lists to the child owners, root first. 2L rounds.
 */
template<class Q, class KeyType, class F, class... UArgs>
void globalUpsweep(cstone::OctreeView<const KeyType> gOctree,
                   const SweepLists&                 up,   // becomes member
                   const SweepLists&                 down, // becomes member
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
    }

    //! downsweep
    for (int l = 0; l < numParentLevels; ++l)
    {
        exchangeLevel(down, l, downsweepTag + l, globQ, comm);
    }
}

} // namespace upsweep_p2p
