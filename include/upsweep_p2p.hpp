/*! @file
 * @brief point-to-point upsweep of the global tree, replaces the Allgatherv in FocusedOctree::globalExchange
 */

#pragma once

#include <span>
#include <utility>
#include <vector>

#include <mpi.h>

#include "cstone/domain/index_ranges.hpp"
#include "cstone/primitives/gather.hpp"
#include "cstone/primitives/mpi_wrappers.hpp"
#include "cstone/tree/octree.hpp"

namespace upsweep_p2p
{

using cstone::TreeNodeIndex;

// MPI Tag§
constexpr int upsweepTag = 6500;

/*! @brief owner rank of every global tree node
 *
 * @param[in]  gOctree    global octree, same on all ranks
 * @param[in]  globDispl  first global leaf per rank, length numRanks + 1
 * @param[out] nodeOwner  owner rank per node, length gOctree.numNodes
 */
template<class KeyType>
void computeNodeOwners(cstone::OctreeView<const KeyType> gOctree, std::span<const TreeNodeIndex> globDispl,
                       std::span<int> nodeOwner)
{
    int numRanks = int(globDispl.size()) - 1;

    auto leafToInternal = gOctree.leafToInternalSpan();
    //! leaves: compute owner for each global leaf
    for (int r = 0; r < numRanks; ++r)
    {
        for (TreeNodeIndex i = globDispl[r]; i < globDispl[r + 1]; ++i)
        {
            TreeNodeIndex nodeIdx = leafToInternal[i];
            nodeOwner[nodeIdx]    = r;
        }
    }

    //! internal nodes: compute owner for each internal node
    for (TreeNodeIndex i = gOctree.numNodes - 1; i >= 0; --i)
    {
        TreeNodeIndex c = gOctree.childOffsets[i];
        if (c == 0) { continue; }
        nodeOwner[i] = nodeOwner[c];
    }
}

/*! @brief send/recv lists of the upsweep per parent level, (peer rank, child node ranges)
 *
 * send[l]: children I own, parent owned by peer
 * recv[l]: children owned by peer, parent owned by me
 */
struct UpsweepLists
{
    using Messages = std::vector<std::pair<int, cstone::IndexRanges<TreeNodeIndex>>>;
    std::vector<Messages> send, recv;
};

/*! @brief send/recv lists of the upsweep
 *
 * @param[in]  gOctree    global octree, same on all ranks
 * @param[in]  nodeOwner  owner rank per node
 * @param[in]  myRank     executing rank
 * @return                send/recv lists per parent level
 */
template<class KeyType>
UpsweepLists computeUpsweepLists(OctreeView<const KeyType> gOctree, std::span<const int> nodeOwner, int myRank)
{
    auto levelRange     = gOctree.levelRangeSpan();
    int numParentLevels = maxDepth(levelRange.data(), levelRange.size()); // deepest level has only leaves

    UpsweepLists lists;
    lists.send.resize(numParentLevels);
    lists.recv.resize(numParentLevels);

    //! owners only increase along a level, so a peer's children always come in a row
    auto addMessage = [](UpsweepLists::Messages& msgs, std::vector<TreeNodeIndex>& idx, int peer, TreeNodeIndex first,
                         TreeNodeIndex last)
    {
        if (msgs.empty() || msgs.back().peer != peer) { msgs.push_back({peer, TreeNodeIndex(idx.size()), 0}); }
        for (TreeNodeIndex i = first; i < last; ++i)
        {
            idx.push_back(i);
        }
        msgs.back().count += last - first;
    };

    for (int l = 0; l < numParentLevels; ++l)
    {
        TreeNodeIndex firstParent = levelRange[l];
        TreeNodeIndex lastParent  = levelRange[l + 1];

        for (TreeNodeIndex p = firstParent; p < lastParent; ++p)
        {
            TreeNodeIndex firstChild = gOctree.childOffsets[p];
            TreeNodeIndex lastChild  = firstChild + 8;
            if (firstChild == 0) { continue; }

            //! children of p grouped by owner
            for (TreeNodeIndex c = firstChild; c < lastChild;)
            {
                int owner            = nodeOwner[c];
                TreeNodeIndex runEnd = c;
                while (runEnd < lastChild && nodeOwner[runEnd] == owner)
                {
                    ++runEnd;
                }

                if (owner == myRank && nodeOwner[p] != myRank)
                {
                    addMessage(lists.send[l], lists.sendIdx, nodeOwner[p], c, runEnd);
                }
                if (owner != myRank && nodeOwner[p] == myRank)
                {
                    addMessage(lists.recv[l], lists.recvIdx, owner, c, runEnd);
                }

                c = runEnd;
            }
        }
    }

    return lists;
}

/*! @brief p2p upsweep of the global tree
 *
 * @param[in]  gOctree          global octree, same on all ranks
 * @param[in]  globDispl        first global leaf per rank, length numRanks + 1
 * @param[in]  nodeOwner        owner rank per node
 * @param[in]  gLeafQLoc        quantities of my global leaves
 * @param[out] globQ            quantities of the global tree, valid for nodes I own and their children
 * @param[in]  myRank           executing rank
 * @param[in]  comm             MPI communicator
 * @param[in]  upsweepFunction  upsweep of one level
 * @param[in]  upsweepArgs      extra upsweep arguments, e.g. expansion centers
 */
template<class Q, class KeyType, class F, class... UArgs>
void globalUpsweep(cstone::OctreeView<const KeyType> gOctree, std::span<const TreeNodeIndex> globDispl,
                   std::span<const int> nodeOwner, // temporary: nodeOwner_ in cstone
                   std::span<const Q> gLeafQLoc, std::span<Q> globQ, int myRank, MPI_Comm comm, F&& upsweepFunction,
                   UArgs&&... upsweepArgs)
{
    auto levelRange     = gOctree.levelRangeSpan();
    auto leafToInternal = gOctree.leafToInternalSpan();

    UpsweepLists lists           = computeUpsweepLists(gOctree, std::span<const int>(nodeOwner), myRank);
    int          numParentLevels = int(lists.send.size());

    //! scatter local leaf quantities
    cstone::scatter(leafToInternal.subspan(globDispl[myRank], gLeafQLoc.size()), gLeafQLoc.data(), globQ.data());

    for (int l = numParentLevels - 1; l >= 0; --l)
    {
        const auto&                 send = lists.send[l];
        const auto&                 recv = lists.recv[l];
        std::vector<std::vector<Q>> sendBuf(send.size()), recvBuf(recv.size());
        std::vector<MPI_Request>    requests;
        int                         tag = upsweepTag + l;

        //! exchange child quantities with their owners
        for (std::size_t m = 0; m < recv.size(); ++m)
        {
            const auto& [peer, ranges] = recv[m];
            recvBuf[m].resize(ranges.totalCount());
            mpiRecvAsync(recvBuf[m].data(), int(ranges.totalCount()), peer, tag, requests, comm);
        }

        for (std::size_t m = 0; m < send.size(); ++m)
        {
            const auto& [peer, ranges] = send[m];
            sendBuf[m].resize(ranges.totalCount());
            std::size_t writeIdx = 0;
            for (std::size_t k = 0; k < ranges.nRanges(); ++k)
            {
                for (TreeNodeIndex i = ranges.rangeStart(k); i < ranges.rangeEnd(k); ++i)
                {
                    sendBuf[m][writeIdx++] = globQ[i];
                }
            }
            mpiSendAsync(sendBuf[m].data(), int(ranges.totalCount()), peer, tag, requests, comm);
        }

        MPI_Waitall(int(requests.size()), requests.data(), MPI_STATUSES_IGNORE);

        //! place received child quantities
        for (std::size_t m = 0; m < recv.size(); ++m)
        {
            const auto& ranges  = recv[m].second;
            std::size_t readIdx = 0;
            for (std::size_t k = 0; k < ranges.nRanges(); ++k)
            {
                for (TreeNodeIndex i = ranges.rangeStart(k); i < ranges.rangeEnd(k); ++i)
                {
                    globQ[i] = recvBuf[m][readIdx++];
                }
            }
        }

        //! accumulate this parent level
        upsweepFunction(levelRange.subspan(l, 2), gOctree.childOffsets, globQ.data(), upsweepArgs...);
    }
}

} // namespace upsweep_p2p
