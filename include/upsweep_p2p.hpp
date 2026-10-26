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
constexpr int downsweepTag = 6600; // parent owner to child owner, NBX replies
constexpr int requestTag   = 6700; // and 6701, NBX requests to owners

/*! @brief owner rank of global node i: the rank holding the node's first leaf
 *
 * @param[in]  gOctree     global octree, same on all ranks
 * @param[in]  assignment  SFC key range per rank
 * @param[in]  i           global node index
 */
template<class KeyType>
int nodeOwner(OctreeView<const KeyType> gOctree, const SfcAssignment<KeyType>& assignment, TreeNodeIndex i)
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

    SweepLists up;
    up.send.resize(numLevels);
    up.recv.resize(numLevels);

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
            if (childOwner == myRank_) { addNode(up.send[l], up.sendIdx, parentOwner, c); }
            if (parentOwner == myRank_) { addNode(up.recv[l], up.recvIdx, childOwner, c); }
        }
    }
    return up;
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

/*! @brief get the children of my held nodes from their owners with NBX (Hoefler et al. 2010)
 *
 * Owners don't know who asks. Every rank sends one request per owner with MPI_Issend and probes for requests from any
 * rank. Once its own requests are matched it enters the MPI_Ibarrier and keeps probing, when the barrier completes
 * every request has arrived. Then the owners send the children back, they have them after the upsweep.
 */
template<class Q, class KeyType>
void fetchFromOwners(OctreeView<const KeyType> gOctree, const SfcAssignment<KeyType>& assignment,
                     std::span<const uint8_t> held, std::span<Q> globQ, MPI_Comm comm)
{
    // the tag alternates, ranks are at most one call apart and a probe must not match the next call
    static int call = 0;
    int        tag  = requestTag + (call++ & 1);

    //! my requests by owner, one message per owner
    std::vector<std::pair<int, TreeNodeIndex>> byOwner;
    for (TreeNodeIndex n = 0; n < gOctree.numNodes; ++n)
    {
        if (held[n]) { byOwner.push_back({nodeOwner(gOctree, assignment, n), n}); }
    }
    std::sort(byOwner.begin(), byOwner.end());
    std::vector<int>                        owners;
    std::vector<std::vector<TreeNodeIndex>> nodesPerOwner;
    for (auto [owner, n] : byOwner)
    {
        if (owners.empty() || owners.back() != owner)
        {
            owners.push_back(owner);
            nodesPerOwner.emplace_back();
        }
        nodesPerOwner.back().push_back(n);
    }

    // the owners send the children back
    SweepLists replies;
    replies.send.resize(1);
    replies.recv.resize(1);

    //! send my requests
    std::vector<MPI_Request> sends(owners.size());
    for (size_t i = 0; i < owners.size(); ++i)
    {
        MPI_Issend(nodesPerOwner[i].data(), int(nodesPerOwner[i].size()), MpiType<TreeNodeIndex>{}, owners[i], tag,
                   comm, &sends[i]);
        addChildren(gOctree.childOffsets, replies.recv[0], replies.recvIdx, owners[i], nodesPerOwner[i]);
    }

    //! receive requests from any rank until the barrier completes, enter it once my requests are matched
    MPI_Request barrier = MPI_REQUEST_NULL;
    for (int done = 0; !done;)
    {
        int        haveRequest;
        MPI_Status status;
        MPI_Iprobe(MPI_ANY_SOURCE, tag, comm, &haveRequest, &status);
        if (haveRequest)
        {
            int count;
            mpiGetCount<TreeNodeIndex>(&status, &count);
            std::vector<TreeNodeIndex> nodes(count);
            mpiRecvSync(nodes.data(), count, status.MPI_SOURCE, tag, MPI_STATUS_IGNORE, comm);
            addChildren(gOctree.childOffsets, replies.send[0], replies.sendIdx, status.MPI_SOURCE, nodes);
        }
        if (barrier != MPI_REQUEST_NULL) { MPI_Test(&barrier, &done, MPI_STATUS_IGNORE); }
        else
        {
            int sent;
            MPI_Testall(int(sends.size()), sends.data(), &sent, MPI_STATUSES_IGNORE);
            if (sent) { MPI_Ibarrier(comm, &barrier); }
        }
    }

    //! send the children of the requested nodes back
    exchangeLevel(replies, 0, downsweepTag, globQ, comm);
}

/*! @brief p2p upsweep + downsweep of the global tree, the requests travel with the upsweep
 *
 * @param[in]    gOctree          global octree, same on all ranks
 * @param[in]    assignment       SFC key range per rank
 * @param[in]    up               up lists from computeSweepLists
 * @param[in]    flags            length numNodes, I need the children of every flagged node
 * @param[inout] globQ            in: my global leaf quantities at their node positions
 *                                out: valid for the nodes I own and the children of my flagged nodes
 * @param[in]    comm             MPI communicator
 * @param[in]    upsweepFunction  upsweep of one level
 * @param[in]    upsweepArgs      extra upsweep arguments, e.g. expansion centers
 *
 * Upsweep per level, values and requests to the parent owner, the parent owner answers what it owns in the downsweep.
 * A request stops climbing at the level of its node, after the upsweep it goes to the owner with NBX.
 * Downsweep per level, the children of the requests that went up, root first.
 */
template<class Q, class KeyType, class F, class... UArgs>
void globalUpsweep(cstone::OctreeView<const KeyType> gOctree,
                   const SfcAssignment<KeyType>&     assignment, // becomes member
                   const SweepLists&                 up,         // becomes member
                   std::span<const uint8_t> flags, std::span<Q> globQ, MPI_Comm comm, F&& upsweepFunction,
                   UArgs&&... upsweepArgs)
{
    int myRank;
    MPI_Comm_rank(comm, &myRank);
    auto levelRange      = gOctree.levelRangeSpan();
    int  numParentLevels = int(up.send.size());
    auto owner           = [&](TreeNodeIndex n) { return nodeOwner(gOctree, assignment, n); };

    // extra nodes that go down during the downsweep
    SweepLists downFlagged;
    downFlagged.send.resize(numParentLevels);
    downFlagged.recv.resize(numParentLevels);

    //! requests on my rank, flagged nodes owned by another rank
    std::vector<uint8_t> held(gOctree.numNodes);
    for (TreeNodeIndex n = 0; n < gOctree.numNodes; ++n)
    {
        held[n] = flags[n] && gOctree.childOffsets[n] != 0 && owner(n) != myRank;
    }

    //! upsweep, values and requests to the parent owner level by level
    for (int l = numParentLevels - 1; l >= 0; --l)
    {
        // requests up to level l go up, deeper ones stay here and come from their owner with NBX
        TreeNodeIndex prefix = levelRange[l + 1];

        // receive the requests of my child owners
        std::vector<uint8_t>     recvFlags(up.recv[l].size() * prefix);
        std::vector<MPI_Request> mpiRequests;
        for (size_t i = 0; i < up.recv[l].size(); ++i)
        {
            mpiRecvAsync(recvFlags.data() + i * prefix, prefix, up.recv[l][i].peer, upsweepTag + l, mpiRequests, comm);
        }
        // send my requests to my parent owner, it sends the children back in the downsweep
        for (const auto& m : up.send[l])
        {
            mpiSendAsync(held.data(), prefix, m.peer, upsweepTag + l, mpiRequests, comm);
            addFlagged(gOctree.childOffsets, downFlagged.recv[l], downFlagged.recvIdx, m.peer,
                       {held.data(), size_t(prefix)});
        }
        // send the values of my children to my parent owner
        exchangeLevel(up, l, upsweepTag + l, globQ, comm);
        MPI_Waitall(int(mpiRequests.size()), mpiRequests.data(), MPI_STATUSES_IGNORE);

        if (!up.send[l].empty()) { std::fill(held.begin(), held.begin() + prefix, 0); } // my parent owner has them now

        // send the children back in the downsweep, keep requests for nodes I don't own
        for (size_t i = 0; i < up.recv[l].size(); ++i)
        {
            std::span<const uint8_t> peerFlags(recvFlags.data() + i * prefix, prefix);
            addFlagged(gOctree.childOffsets, downFlagged.send[l], downFlagged.sendIdx, up.recv[l][i].peer, peerFlags);
            for (TreeNodeIndex n = 0; n < prefix; ++n)
            {
                held[n] |= peerFlags[n] && owner(n) != myRank;
            }
        }

        //! compute the parents Q values
        upsweepFunction(levelRange.subspan(l, 2), gOctree.childOffsets, globQ.data(), upsweepArgs...);
    }

    //! the requests nobody on the way owns
    fetchFromOwners(gOctree, assignment, std::span<const uint8_t>(held), globQ, comm);

    //! downsweep
    for (int l = 0; l < numParentLevels; ++l)
    {
        exchangeLevel(downFlagged, l, downsweepTag + l, globQ, comm);
    }
}

} // namespace upsweep_p2p
