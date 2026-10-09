#pragma once

#include <cstdio>
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

/*! @brief contiguous split of the global leaves across ranks
 *
 * @param[in]  numGlobalLeaves  number of global tree leaves
 * @param[in]  numRanks         number of ranks
 * @param[out] globNumNodes     number of global leaves per rank, length numRanks
 * @param[out] globDispl        index of first global leaf per rank, length numRanks + 1
 *
 * Mirrors FocusedOctree::globNumNodes_ and globDispl_, which cstone copies from the
 * SfcAssignment (numNodesPerRank, treeOffsets).
 */
inline void assignGlobalLeaves(TreeNodeIndex numGlobalLeaves,
                               int numRanks,
                               std::vector<TreeNodeIndex>& globNumNodes,
                               std::vector<TreeNodeIndex>& globDispl)
{
    globNumNodes.resize(numRanks);
    globDispl.resize(numRanks + 1);
    for (int r = 0; r <= numRanks; ++r)
    {
        globDispl[r] = TreeNodeIndex((long(numGlobalLeaves) * r) / numRanks);
    }
    for (int r = 0; r < numRanks; ++r)
    {
        globNumNodes[r] = globDispl[r + 1] - globDispl[r];
    }
}

/*! @brief owner rank of every global tree node
 *
 * @param[in]  gOctree    global octree, identical on all ranks
 * @param[in]  globDispl  index of first global leaf per rank, length numRanks + 1
 * @param[out] nodeOwner  owner rank of each node (level-sorted layout), length gOctree.numNodes
 *
 * leaf: the rank whose leaf range contains it
 * internal node: the owner of its first (lowest-key) child
 */
template<class KeyType>
void computeNodeOwners(cstone::OctreeView<const KeyType> gOctree,
                       std::span<const TreeNodeIndex> globDispl,
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
UpsweepPlan computeUpsweepPlan(cstone::OctreeView<const KeyType> gOctree,
                               std::span<const int> nodeOwner,
                               int myRank)
{
    auto levelRange     = gOctree.levelRangeSpan();
    int numParentLevels = cstone::maxDepth(levelRange.data(), levelRange.size()); // deepest level has only leaves

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

            //! children of p grouped by owner: owners are sorted within a parent, e.g. 0 0 0 1 1 1 2 2
            for (TreeNodeIndex c = firstChild; c < lastChild;)
            {
                int owner            = nodeOwner[c];
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

/*! @brief collective global upsweep, mirrors FocusedOctree::globalExchange
 *
 * @param[in]  gOctree          global octree, identical on all ranks
 * @param[in]  globNumNodes     number of global leaves per rank
 * @param[in]  globDispl        index of first global leaf per rank
 * @param[in]  gLeafQLoc        quantities of the global leaves owned by the executing rank
 * @param[out] globQ            quantities of all global tree nodes (level-sorted layout), length gOctree.numNodes
 * @param[in]  upsweepFunction  callable(levelRange, childOffsets, Q*) performing the upsweep on the global tree
 *
 * Allgatherv of all leaves, then every rank does the full upsweep redundantly. In cstone,
 * gLeafQLoc comes from populateGlobal and globQ is passed on to extractGlobal
 */
template<class Q, class KeyType, class F>
void globalExchange(cstone::OctreeView<const KeyType> gOctree,
                    std::span<const TreeNodeIndex> globNumNodes,
                    std::span<const TreeNodeIndex> globDispl,
                    std::span<const Q> gLeafQLoc,
                    std::span<Q> globQ,
                    F&& upsweepFunction,
                    int myRank,
                    MPI_Comm comm)
{
    std::vector<Q> gLeafQAll(gOctree.numLeafNodes);

    //! exchange global leaves
    mpiAllgatherv(gLeafQLoc.data(), globNumNodes[myRank], gLeafQAll.data(), globNumNodes.data(),
                  globDispl.data(), comm);

    cstone::scatter(gOctree.leafToInternalSpan(), gLeafQAll.data(), globQ.data());
    //! upsweep with the global tree
    upsweepFunction(gOctree.levelRangeSpan(), gOctree.childOffsets, globQ.data());
}

//! @brief MPI tag base of the p2p upsweep, tag = base + parent level
constexpr int upsweepP2pTag = 6500;

/*! @brief point-to-point global upsweep
 *
 * @param[in]  gOctree          global octree, same on all ranks
 * @param[in]  globNumNodes     number of global leaves per rank
 * @param[in]  globDispl        index of first global leaf per rank
 * @param[in]  plan             send/recv lists
 * @param[in]  gLeafQLoc        quantities of the global leaves owned by the executing rank
 * @param[out] globQ            quantities of global tree nodes
 * @param[in]  upsweepFunction  callable object that will be used to compute internal cell properties of the
 *                              global tree based on global leaf quantities
 *
 * One exchange round per parent level (deepest first)
 * Post-condition on rank r: globQ[i] is final for every node i owned by r and every child of an
 * internal node owned by r. All other entries are unspecified.
 * 
 * Open questions:
 * - Is it enough that rank r only receives the global nodes extractGlobal reads?
 *   -> nodes matching ranks LET leaves outside its own and peer ranges
 * - How to get the results back to the ranks? bcast or ask/reply?
 * - One communication round per level (latency?)
 */
template<class Q, class KeyType, class F>
void globalExchangeP2p(cstone::OctreeView<const KeyType> gOctree,
                       std::span<const TreeNodeIndex> globNumNodes,
                       std::span<const TreeNodeIndex> globDispl,
                       const UpsweepPlan& plan,
                       std::span<const Q> gLeafQLoc,
                       std::span<Q> globQ,
                       F&& upsweepFunction,
                       int myRank,
                       MPI_Comm comm)
{
    auto levelRange     = gOctree.levelRangeSpan();
    auto leafToInternal = gOctree.leafToInternalSpan();
    int numParentLevels = int(plan.send.size());

    cstone::scatter(leafToInternal.subspan(globDispl[myRank], globNumNodes[myRank]), gLeafQLoc.data(), globQ.data());

    for (int l = numParentLevels - 1; l >= 0; --l)
    {
        const auto& send = plan.send[l];
        const auto& recv = plan.recv[l];
        std::vector<std::vector<Q>> sendBuf(send.size()), recvBuf(recv.size());
        std::vector<MPI_Request> requests;
        int tag = upsweepP2pTag + l;

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

} // namespace upsweep_p2p
