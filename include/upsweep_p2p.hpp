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
constexpr int upsweepTag   = 6500;
constexpr int downsweepTag = 6600;
constexpr int flagTag      = 6700;
constexpr int requestTag   = 6800; // and 6801, alternating per call
constexpr int replyTag     = 6900;

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

/*! @brief get the children of my held nodes from their owners
 *
 * Every owner of a held node gets my held flags. Owners don't know who asks: requests go out with MPI_Issend, owners
 * receive them with MPI_Iprobe until the MPI_Ibarrier, entered once my requests are matched, completes (NBX).
 */
template<class Q, class KeyType>
void fetchFromOwners(OctreeView<const KeyType> gOctree, const SfcAssignment<KeyType>& assignment,
                     std::span<const uint8_t> held, std::span<Q> globQ, MPI_Comm comm)
{
    static int call = 0;
    int        tag  = requestTag + (call++ & 1); // a probe must not match a request of the next call
    int        myRank;
    MPI_Comm_rank(comm, &myRank);
    auto owner   = [&](TreeNodeIndex n) { return nodeOwner(gOctree, assignment, n); };
    auto ownedBy = [&](std::span<const uint8_t> f, int rank)
    {
        std::vector<uint8_t> o(f.size());
        for (size_t n = 0; n < f.size(); ++n)
        {
            o[n] = f[n] && owner(n) == rank;
        }
        return o;
    };

    std::vector<int> owners;
    for (TreeNodeIndex n = 0; n < gOctree.numNodes; ++n)
    {
        if (held[n]) { owners.push_back(owner(n)); }
    }
    std::sort(owners.begin(), owners.end());
    owners.erase(std::unique(owners.begin(), owners.end()), owners.end());

    SweepLists replies; // one level
    replies.send.resize(1);
    replies.recv.resize(1);
    std::vector<MPI_Request> sends(owners.size());
    for (size_t i = 0; i < owners.size(); ++i)
    {
        MPI_Issend(held.data(), gOctree.numNodes, MPI_UINT8_T, owners[i], tag, comm, &sends[i]);
        addFlagged(gOctree.childOffsets, replies.recv[0], replies.recvIdx, owners[i], ownedBy(held, owners[i]));
    }

    std::vector<uint8_t> request(gOctree.numNodes);
    MPI_Request          barrier = MPI_REQUEST_NULL;
    for (int done = 0; !done;)
    {
        int        haveRequest;
        MPI_Status status;
        MPI_Iprobe(MPI_ANY_SOURCE, tag, comm, &haveRequest, &status);
        if (haveRequest)
        {
            MPI_Recv(request.data(), gOctree.numNodes, MPI_UINT8_T, status.MPI_SOURCE, tag, comm, MPI_STATUS_IGNORE);
            addFlagged(gOctree.childOffsets, replies.send[0], replies.sendIdx, status.MPI_SOURCE,
                       ownedBy(request, myRank));
        }
        if (barrier != MPI_REQUEST_NULL) { MPI_Test(&barrier, &done, MPI_STATUS_IGNORE); }
        else
        {
            int sent;
            MPI_Testall(int(sends.size()), sends.data(), &sent, MPI_STATUSES_IGNORE);
            if (sent) { MPI_Ibarrier(comm, &barrier); }
        }
    }
    exchangeLevel(replies, 0, replyTag, globQ, comm);
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
 * held: flagged internal nodes I don't own, of mine or of the ranks below me.
 * Upsweep: per level l the values and the held flags of levels <= l go to the parent owner, then I compute my parents.
 * What is still held after the upsweep comes from the owners. Downsweep: per level l the pairs reversed, the children
 * of the flags that went up in round l.
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
    auto needed          = [&](TreeNodeIndex n)
    { return gOctree.childOffsets[n] != 0 && nodeOwner(gOctree, assignment, n) != myRank; };

    std::vector<uint8_t> held(gOctree.numNodes);
    for (TreeNodeIndex n = 0; n < gOctree.numNodes; ++n)
    {
        held[n] = flags[n] && needed(n);
    }
    SweepLists down;
    down.send.resize(numParentLevels);
    down.recv.resize(numParentLevels);

    //! upsweep
    for (int l = numParentLevels - 1; l >= 0; --l)
    {
        //! held flags of levels <= l = the first levelRange[l + 1] nodes
        TreeNodeIndex            prefix = levelRange[l + 1];
        std::vector<uint8_t>     recvFlags(up.recv[l].size() * prefix);
        std::vector<MPI_Request> flagRequests;
        for (size_t i = 0; i < up.recv[l].size(); ++i)
        {
            mpiRecvAsync(recvFlags.data() + i * prefix, prefix, up.recv[l][i].peer, flagTag + l, flagRequests, comm);
        }
        for (const auto& m : up.send[l])
        {
            mpiSendAsync(held.data(), prefix, m.peer, flagTag + l, flagRequests, comm);
            addFlagged(gOctree.childOffsets, down.recv[l], down.recvIdx, m.peer, {held.data(), size_t(prefix)});
        }
        exchangeLevel(up, l, upsweepTag + l, globQ, comm);
        MPI_Waitall(int(flagRequests.size()), flagRequests.data(), MPI_STATUSES_IGNORE);

        if (!up.send[l].empty()) { std::fill(held.begin(), held.begin() + prefix, 0); } // they come back down
        for (size_t i = 0; i < up.recv[l].size(); ++i)
        {
            std::span<const uint8_t> f(recvFlags.data() + i * prefix, prefix);
            addFlagged(gOctree.childOffsets, down.send[l], down.sendIdx, up.recv[l][i].peer, f);
            for (TreeNodeIndex n = 0; n < prefix; ++n)
            {
                held[n] |= f[n] && needed(n);
            }
        }
        //! compute the parents Q values
        upsweepFunction(levelRange.subspan(l, 2), gOctree.childOffsets, globQ.data(), upsweepArgs...);
    }

    fetchFromOwners(gOctree, assignment, std::span<const uint8_t>(held), globQ, comm);

    //! downsweep
    for (int l = 0; l < numParentLevels; ++l)
    {
        exchangeLevel(down, l, downsweepTag + l, globQ, comm);
    }
}

} // namespace upsweep_p2p
