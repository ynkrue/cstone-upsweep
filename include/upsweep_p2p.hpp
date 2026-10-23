/*! @file
 * @brief point-to-point upsweep of the global tree, replaces the Allgatherv in FocusedOctree::globalExchange
 */

#pragma once

#include <algorithm>
#include <array>
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

//! TEMPORARY profiling: accumulated times [s], reset by the caller
struct ProfileTimes
{
    double reqStart = 0, reqUpWait = 0, reqUp = 0, reqDownWait = 0, reqDown = 0; // computeExchangeLists
    double upEx = 0, upCompute = 0, upFlaggedEx = 0, downEx = 0;                 // globalUpsweep
    double exWait = 0;                                                           // Waitall in exchangeLevel
};
inline ProfileTimes profile;

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

/*! @brief exchange the values of one level: send globQ[sendIdx], receive into globQ[recvIdx]
 *
 * With @p childOffsets, each index i stands for the 8 children of i, sent in octant order.
 */
template<class Q>
void exchangeLevel(const SweepLists& lists, int l, int tag, std::span<Q> globQ, MPI_Comm comm,
                   const TreeNodeIndex* childOffsets = nullptr)
{
    int         stride = childOffsets ? 8 : 1;
    const auto& send   = lists.send[l];
    const auto& recv   = lists.recv[l];

    //! the messages of one level are one contiguous slice of sendIdx / recvIdx, starting at sFirst / rFirst
    auto levelNodes = [](const std::vector<SweepLists::Message>& msgs, const std::vector<TreeNodeIndex>& idx)
    {
        if (msgs.empty()) { return std::span<const TreeNodeIndex>{}; }
        return std::span<const TreeNodeIndex>(idx).subspan(msgs.front().offset, msgs.back().offset + msgs.back().count -
                                                                                    msgs.front().offset);
    };
    auto           sendNodes = levelNodes(send, lists.sendIdx), recvNodes = levelNodes(recv, lists.recvIdx);
    TreeNodeIndex  sFirst = send.empty() ? 0 : send.front().offset, rFirst = recv.empty() ? 0 : recv.front().offset;
    std::vector<Q> sendBuf(sendNodes.size() * stride), recvBuf(recvNodes.size() * stride);

    std::vector<MPI_Request> mpiRequests;
    for (const auto& m : recv)
    {
        mpiRecvAsync(recvBuf.data() + (m.offset - rFirst) * stride, m.count * stride, m.peer, tag, mpiRequests, comm);
    }
    if (childOffsets)
    {
        for (size_t j = 0; j < sendNodes.size(); ++j)
        {
            std::copy_n(globQ.data() + childOffsets[sendNodes[j]], 8, sendBuf.data() + 8 * j);
        }
    }
    else
    {
        gather(sendNodes, globQ.data(), sendBuf.data());
    }
    for (const auto& m : send)
    {
        mpiSendAsync(sendBuf.data() + (m.offset - sFirst) * stride, m.count * stride, m.peer, tag, mpiRequests, comm);
    }
    double tw = MPI_Wtime();
    MPI_Waitall(int(mpiRequests.size()), mpiRequests.data(), MPI_STATUSES_IGNORE);
    profile.exWait += MPI_Wtime() - tw;
    if (childOffsets)
    {
        for (size_t j = 0; j < recvNodes.size(); ++j)
        {
            std::copy_n(recvBuf.data() + 8 * j, 8, globQ.data() + childOffsets[recvNodes[j]]);
        }
    }
    else
    {
        scatter(recvNodes, recvBuf.data(), globQ.data());
    }
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
 * @return                {upFlagged, downFlagged}: flagged nodes whose 8 children are sent/received per level in
 *                        the upsweep / downsweep of globalUpsweep
 *
 * A request for n (I need the children of n) climbs from my first leaf until the current node contains n (up loop),
 * then descends to owner(n) (down loop). It changes rank only on an up pair, or on a reversed one going down. Locally a
 * request is moved straight to its next hop and waits in the bucket of that round. Every hop is recorded in reverse:
 * up hop -> downFlagged, down hop -> upFlagged. Each message is sorted and free of duplicates.
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
    //! one message with @p peer that holds @p nodes
    auto addMessage = [](std::vector<SweepLists::Message>& msgs, std::vector<TreeNodeIndex>& idx, int peer,
                         std::span<const TreeNodeIndex> nodes)
    {
        if (nodes.empty()) { return; }
        msgs.push_back({peer, TreeNodeIndex(idx.size()), TreeNodeIndex(nodes.size())});
        idx.insert(idx.end(), nodes.begin(), nodes.end());
    };
    auto sortUnique = [](auto& v)
    {
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
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
        while (owner(n) != myRank) // owner(n) == me: arrived, I supply the children of n
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
    //! from node a (mine): climb while I own the parent, then wait for the hop round level(parent); turn at a contains
    //! n
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

    double tp0 = MPI_Wtime();
    //! my requests start at my first leaf
    TreeNodeIndex firstLeaf = gOctree.leafToInternalSpan()[globDispl[myRank]];
    for (TreeNodeIndex i = 0; i < gOctree.numInternalNodes; ++i)
    {
        if (flags_[i] && owner(i) != myRank) { climb(i, firstLeaf); }
    }

    double tp1 = MPI_Wtime();
    profile.reqStart += tp1 - tp0;
    double waitUp = 0;
    //! up loop, deepest level first
    for (int l = numParentLevels - 1; l >= 0; --l)
    {
        std::vector<TreeNodeIndex>& sendNodes = upSend[l];
        sortUnique(sendNodes);

        std::vector<MPI_Request> mpiRequests;
        for (const auto& m : up.send[l]) // at most one pair per level, sent also when empty
        {
            mpiSendAsync(sendNodes.data(), sendNodes.size(), m.peer, requestUpTag + l, mpiRequests, comm);
            addMessage(downFlagged.recv[l], downFlagged.recvIdx, m.peer, sendNodes);
        }
        for (const auto& m : up.recv[l])
        {
            MPI_Status status;
            double     tw = MPI_Wtime();
            MPI_Probe(m.peer, requestUpTag + l, comm, &status);
            int count;
            mpiGetCount<TreeNodeIndex>(&status, &count);

            std::vector<TreeNodeIndex> recvBuf(count);
            mpiRecvSync(recvBuf.data(), count, m.peer, requestUpTag + l, MPI_STATUS_IGNORE, comm);
            waitUp += MPI_Wtime() - tw;
            addMessage(downFlagged.send[l], downFlagged.sendIdx, m.peer, recvBuf);

            TreeNodeIndex a = parent(up.recvIdx[m.offset]); // all children a peer sends at one level share this parent
            for (TreeNodeIndex n : recvBuf)
            {
                climb(n, a);
            }
        }
        double tw = MPI_Wtime();
        MPI_Waitall(int(mpiRequests.size()), mpiRequests.data(), MPI_STATUSES_IGNORE);
        waitUp += MPI_Wtime() - tw;
    }

    double tp2 = MPI_Wtime();
    profile.reqUpWait += waitUp;
    profile.reqUp += tp2 - tp1 - waitUp;
    double waitDown = 0;
    //! down loop, root first
    for (int l = 0; l < numParentLevels; ++l)
    {
        auto& sendPairs = downSend[l];
        sortUnique(sendPairs); // by peer, then n

        //! one message to every child owner of this level, also when empty, the peer probes for it
        std::vector<std::vector<TreeNodeIndex>> sendBufs(up.recv[l].size());
        std::vector<MPI_Request>                mpiRequests;
        for (size_t k = 0; k < up.recv[l].size(); ++k)
        {
            int peer      = up.recv[l][k].peer;
            auto [lo, hi] = std::equal_range(sendPairs.begin(), sendPairs.end(), std::pair{peer, TreeNodeIndex(0)},
                                             [](const auto& x, const auto& y) { return x.first < y.first; });
            for (auto it = lo; it != hi; ++it)
            {
                sendBufs[k].push_back(it->second);
            }
            mpiSendAsync(sendBufs[k].data(), sendBufs[k].size(), peer, requestDownTag + l, mpiRequests, comm);
            addMessage(upFlagged.recv[l], upFlagged.recvIdx, peer, sendBufs[k]);
        }
        for (const auto& m : up.send[l])
        {
            MPI_Status status;
            double     tw = MPI_Wtime();
            MPI_Probe(m.peer, requestDownTag + l, comm, &status);
            int count;
            mpiGetCount<TreeNodeIndex>(&status, &count);

            std::vector<TreeNodeIndex> recvBuf(count);
            mpiRecvSync(recvBuf.data(), count, m.peer, requestDownTag + l, MPI_STATUS_IGNORE, comm);
            waitDown += MPI_Wtime() - tw;
            addMessage(upFlagged.send[l], upFlagged.sendIdx, m.peer, recvBuf);

            TreeNodeIndex p = parent(up.sendIdx[m.offset]); // the peer owns p, the n arrive at my child of p
            for (TreeNodeIndex n : recvBuf)
            {
                descend(n, childToward(p, n));
            }
        }
        double tw = MPI_Wtime();
        MPI_Waitall(int(mpiRequests.size()), mpiRequests.data(), MPI_STATUSES_IGNORE);
        waitDown += MPI_Wtime() - tw;
    }
    profile.reqDownWait += waitDown;
    profile.reqDown += MPI_Wtime() - tp2 - waitDown;
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
                   const SfcAssignment<KeyType>&     assignment,  // becomes member
                   std::span<Q> globQ, MPI_Comm comm, F&& upsweepFunction, UArgs&&... upsweepArgs)
{
    int myRank;
    MPI_Comm_rank(comm, &myRank);
    auto levelRange      = gOctree.levelRangeSpan();
    int  numParentLevels = int(up.send.size());

    //! my nodes of level l: within a level the nodes are sorted by key, mine start in [assignment[myRank], +1)
    auto ownedRange = [&](int l)
    {
        auto first = [&](KeyType key) // first node of level l with start key >= key
        {
            TreeNodeIndex lo = levelRange[l], hi = levelRange[l + 1];
            while (lo < hi)
            {
                TreeNodeIndex mid = (lo + hi) / 2;
                if (decodePlaceholderBit(gOctree.prefixes[mid]) < key) { lo = mid + 1; }
                else
                {
                    hi = mid;
                }
            }
            return lo;
        };
        return std::array<TreeNodeIndex, 2>{first(assignment[myRank]), first(assignment[myRank + 1])};
    };

    //! upsweep
    for (int l = numParentLevels - 1; l >= 0; --l)
    {
        double ts0 = MPI_Wtime();
        exchangeLevel(up, l, upsweepTag + l, globQ, comm);
        double ts1 = MPI_Wtime();

        //! compute the Q values of my parents of this level
        auto owned = ownedRange(l);
        upsweepFunction(std::span<const TreeNodeIndex>(owned), gOctree.childOffsets, globQ.data(), upsweepArgs...);
        double ts2 = MPI_Wtime();

        exchangeLevel(upFlagged, l, upFlaggedTag + l, globQ, comm, gOctree.childOffsets);
        profile.upEx += ts1 - ts0;
        profile.upCompute += ts2 - ts1;
        profile.upFlaggedEx += MPI_Wtime() - ts2;
    }

    //! downsweep
    double ts3 = MPI_Wtime();
    for (int l = 0; l < numParentLevels; ++l)
    {
        exchangeLevel(downFlagged, l, downsweepTag + l, globQ, comm, gOctree.childOffsets);
    }
    profile.downEx += MPI_Wtime() - ts3;
}

} // namespace upsweep_p2p
