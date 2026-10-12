/*! @file
 * @brief point-to-point replacement of the Allgatherv in cstone's global upsweep
 *        (FocusedOctree::globalExchange, cstone/focus/octree_focus_mpi.hpp)
 *
 * Once per global tree: computeNodeOwners, computeUpsweepPlan
 * Per call:             globalUpsweep (upward pass), extractGlobalP2p (distribution)
 */

#pragma once

#include <algorithm>
#include <span>
#include <utility>
#include <vector>

#include <mpi.h>

#include "cstone/domain/index_ranges.hpp"
#include "cstone/primitives/gather.hpp"
#include "cstone/primitives/mpi_wrappers.hpp"
#include "cstone/tree/octree.hpp"
#include "cstone/domain/layout.hpp"
#include "cstone/sfc/common.hpp"


namespace upsweep_p2p
{

using cstone::TreeNodeIndex;

/*! @brief owner rank of every global tree node
 *
 * @param[in]  gOctree    global octree, same on all ranks
 * @param[in]  globDispl  index of first global leaf per rank, length numRanks + 1
 * @param[out] nodeOwner  owner rank of each node (level-sorted layout), length gOctree.numNodes
 *
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

/*! @brief point-to-point message lists of the global upsweep
 *
 * Indexed [parentLevel], each entry a list of (peer rank, child node index ranges); only peers with
 * messages are stored. Children are at parentLevel + 1.
 *  send[l]: children I own, whose parent the peer owns
 *  recv[l]: children the peer owns, whose parent I own
 * Ranges are in ascending node index, so sender and receiver agree on the value order.
 * Per level a rank sends to at most 1 peer and receives from at most 7.
 */
struct UpsweepPlan
{
    using Messages = std::vector<std::pair<int, cstone::IndexRanges<TreeNodeIndex>>>;
    std::vector<Messages> send, recv;
};

/*! @brief compute the messages of the p2p global upsweep
 *
 * @param[in]  gOctree    global octree, same on all ranks
 * @param[in]  nodeOwner  owner rank of each node
 * @param[in]  myRank     executing rank
 * @return                send and receive lists per parent level
 *
 */
template<class KeyType>
UpsweepPlan computeUpsweepPlan(cstone::OctreeView<const KeyType> gOctree, std::span<const int> nodeOwner, int myRank)
{
    auto levelRange      = gOctree.levelRangeSpan();
    int  numParentLevels = cstone::maxDepth(levelRange.data(), levelRange.size()); // deepest level has only leaves

    UpsweepPlan plan;
    plan.send.resize(numParentLevels);
    plan.recv.resize(numParentLevels);

    //! owners only increase along a level, so a peer's ranges always come in a row
    auto addMessage = [](UpsweepPlan::Messages& msgs, int peer, TreeNodeIndex first, TreeNodeIndex last)
    {
        if (msgs.empty() || msgs.back().first != peer)
        {
            msgs.emplace_back(peer, cstone::IndexRanges<TreeNodeIndex>{});
        }
        msgs.back().second.addRange(first, last);
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
                int           owner  = nodeOwner[c];
                TreeNodeIndex runEnd = c;
                while (runEnd < lastChild && nodeOwner[runEnd] == owner)
                {
                    ++runEnd;
                }

                if (owner == myRank && nodeOwner[p] != myRank) { addMessage(plan.send[l], nodeOwner[p], c, runEnd); }
                if (owner != myRank && nodeOwner[p] == myRank) { addMessage(plan.recv[l], owner, c, runEnd); }

                c = runEnd;
            }
        }
    }

    return plan;
}

/*! @brief point-to-point global upsweep, replaces the Allgatherv + redundant upsweep in globalExchange
 *
 * @param[in]  gOctree          global octree, identical on all ranks
 * @param[in]  globDispl        index of first global leaf per rank, length numRanks + 1
 * @param[in]  plan             send/recv lists from computeUpsweepPlan
 * @param[in]  gLeafQLoc        quantities of the global leaves owned by the executing rank
 * @param[out] globQ            quantities of global tree nodes (level-sorted layout), length gOctree.numNodes
 * @param[in]  upsweepFunction  callable(levelRange, childOffsets, Q*) combining the levels of the given span
 * @param[in]  myRank           executing rank
 * @param[in]  comm             MPI communicator
 *
 * One exchange round per parent level, deepest first: children go to the owner of their parent,
 * then the parent level is combined.
 * Post-condition on rank r: globQ[i] is final for every node i owned by r and every child of an
 * internal node owned by r. All other entries are unspecified.
 */
template<class Q, class KeyType, class F>
void globalUpsweep(cstone::OctreeView<const KeyType> gOctree, std::span<const TreeNodeIndex> globDispl,
                   std::span<const int>     nodeOwner, // temporary: becomes nodeOwner_ in cstone
                   std::span<const Q> gLeafQLoc, std::span<Q> globQ, F&& upsweepFunction, int myRank, MPI_Comm comm)
{
    auto levelRange     = gOctree.levelRangeSpan();
    auto leafToInternal = gOctree.leafToInternalSpan();

    UpsweepPlan plan            = computeUpsweepPlan(gOctree, std::span<const int>(nodeOwner), myRank);
    int         numParentLevels = int(plan.send.size());

    cstone::scatter(leafToInternal.subspan(globDispl[myRank], gLeafQLoc.size()), gLeafQLoc.data(), globQ.data());

    for (int l = numParentLevels - 1; l >= 0; --l)
    {
        const auto&                 send = plan.send[l];
        const auto&                 recv = plan.recv[l];
        std::vector<std::vector<Q>> sendBuf(send.size()), recvBuf(recv.size());
        std::vector<MPI_Request>    requests;
        int                         tag = upsweepP2pTag + l;

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

        upsweepFunction(levelRange.subspan(l, 2), gOctree.childOffsets, globQ.data());
    }
}

/*! @brief Copy far LET leaves from their owning ranks.
 *
 * @param[in]  globalNodeKeys    global node keys referenced from the LET
 * @param[in]  globalLevelRange  level ranges of the global tree used to locate nodes by index
 * @param[in]  globalQuantities  quantities stored on the owning ranks
 * @param[out] localQuantities   values written into the local LET slots that reference remote nodes
 * @param[in]  comm              MPI communicator
 */
template<class T, class KeyType>
void extractGlobalP2p(cstone::OctreeView<const KeyType>      let,        // temporary: octreeAcc_ in cstone
                      std::span<const cstone::TreeIndexPair> peerRanges, // temporary: peerRanges_ in cstone
                      const KeyType* globalNodeKeys, const TreeNodeIndex* globalLevelRange,
                      std::span<const int> nodeOwner, // temporary: becomes nodeOwner_ in cstone
                      std::span<const T> globalQuantities, std::span<T> localQuantities, MPI_Comm comm)
{
    // TODO part 1 (copy CPU branch of FocusedOctree::extractGlobal, octree_focus_mpi.hpp:301)
    // - far LET leaves: enumerateRanges(invertRanges(0, peerRanges, let.numLeafNodes))
    // - leaf → LET node ℓ: let.leafToInternal[let.numInternalNodes + j]
    // - ℓ → global node g: locateNode(let.prefixes[ℓ], globalNodeKeys, globalLevelRange)
    // - check: print #far leaves per rank (P = 3, L = 4: 854 / 869 / 855)

    // TODO part 2a: sort (g, ℓ) pairs by (nodeOwner[g], g), group per owner → my g list + ℓ list per owner
    // TODO part 2b: sendCounts[owner] = list size → MPI_Alltoall → recvCounts (copy computeExtractPlan)
    // TODO part 2c: Irecv g lists from requesters, Isend my g lists to owners, Waitall (extractRequestTag)
    // TODO part 2d: owner packs globalQuantities[g] per requester → Isend; Irecv per owner → Waitall
    //               → localQuantities[ℓ] = value (extractValueTag)
    // TODO after: delete ExtractPlan/computeExtractPlan, (void) leftovers, complete the @param list above
    (void) let;
    (void) peerRanges;
    (void) globalNodeKeys;
    (void) globalLevelRange;
    (void) nodeOwner;
    (void) globalQuantities;
    (void) localQuantities;
    (void) comm;

}

} // namespace upsweep_p2p
