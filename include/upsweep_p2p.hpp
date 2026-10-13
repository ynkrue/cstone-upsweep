/*! @file
 * @brief point-to-point global exchange, replaces the Allgatherv in FocusedOctree::globalExchange
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

#include <mpi.h>

#include "cstone/domain/index_ranges.hpp"
#include "cstone/primitives/gather.hpp"
#include "cstone/primitives/mpi_wrappers.hpp"
#include "cstone/tree/octree.hpp"
#include "cstone/domain/layout.hpp"
#include "cstone/focus/octree_focus_mpi.hpp"
#include "cstone/sfc/common.hpp"

namespace upsweep_p2p
{

using cstone::TreeNodeIndex;

//! @brief MPI tag of the upsweep and far leaf exchange (temporary: added to P2pTags in definitions.h)
constexpr int upsweepTag        = 6500;
constexpr int extractRequestTag = 6600;
constexpr int extractValueTag   = 6601;

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
UpsweepLists computeUpsweepLists(cstone::OctreeView<const KeyType> gOctree, std::span<const int> nodeOwner, int myRank)
{
    auto levelRange      = gOctree.levelRangeSpan();
    int  numParentLevels = cstone::maxDepth(levelRange.data(), levelRange.size()); // deepest level has only leaves

    UpsweepLists lists;
    lists.send.resize(numParentLevels);
    lists.recv.resize(numParentLevels);

    //! owners only increase along a level, so a peer's ranges always come in a row
    auto addMessage = [](UpsweepLists::Messages& msgs, int peer, TreeNodeIndex first, TreeNodeIndex last)
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

                if (owner == myRank && nodeOwner[p] != myRank) { addMessage(lists.send[l], nodeOwner[p], c, runEnd); }
                if (owner != myRank && nodeOwner[p] == myRank) { addMessage(lists.recv[l], owner, c, runEnd); }

                c = runEnd;
            }
        }
    }

    return lists;
}

/*! @brief gather with distributed src, src[gmap[k]] fetched from its owner rank
 *
 * @param[in]  gmap       global node per entry
 * @param[in]  nodeOwner  owner rank per global node
 * @param[in]  src        quantities of the global tree, valid on the owner
 * @param[out] out        out[k] = src[gmap[k]], length gmap.size()
 * @param[in]  comm       MPI communicator
 */
template<class T>
void gatherFromOwners(std::span<const TreeNodeIndex> gmap, std::span<const int> nodeOwner, const T* src, T* out,
                      MPI_Comm comm)
{
    int numRanks = 0;
    MPI_Comm_size(comm, &numRanks);
    std::size_t n = gmap.size();

    //! round 1: global nodes I need, grouped by owner, counts of round 1 (become recv counts in round 2)
    std::vector<int>           sendCounts(numRanks, 0), sendDispl(numRanks + 1, 0);
    std::vector<TreeNodeIndex> sendIdx(n);

    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b)
              { return std::pair(nodeOwner[gmap[a]], gmap[a]) < std::pair(nodeOwner[gmap[b]], gmap[b]); });

    //! send buffer and counts per owner
    for (std::size_t m = 0; m < n; ++m)
    {
        TreeNodeIndex g = gmap[order[m]];
        sendIdx[m]      = g;
        ++sendCounts[nodeOwner[g]];
    }
    std::inclusive_scan(sendCounts.begin(), sendCounts.end(), sendDispl.begin() + 1);

    //! number of requests each rank sends to me
    std::vector<int> recvCounts(numRanks, 0), recvDispl(numRanks + 1, 0);
    MPI_Alltoall(sendCounts.data(), 1, MPI_INT, recvCounts.data(), 1, MPI_INT, comm);
    std::inclusive_scan(recvCounts.begin(), recvCounts.end(), recvDispl.begin() + 1);

    //! exchange global node indices, sendIdx to owners, requesters to recvIdx
    std::vector<TreeNodeIndex> recvIdx(recvDispl[numRanks]);
    std::vector<MPI_Request>   requests;

    for (int r = 0; r < numRanks; ++r)
    {
        if (recvCounts[r] > 0)
        {
            mpiRecvAsync(recvIdx.data() + recvDispl[r], recvCounts[r], r, extractRequestTag, requests, comm);
        }
    }

    for (int r = 0; r < numRanks; ++r)
    {
        if (sendCounts[r] > 0)
        {
            mpiSendAsync(sendIdx.data() + sendDispl[r], sendCounts[r], r, extractRequestTag, requests, comm);
        }
    }

    MPI_Waitall(int(requests.size()), requests.data(), MPI_STATUSES_IGNORE);

    //! round 2: values of the requested nodes
    std::vector<T> sendQ(recvIdx.size()), recvQ(n);
    for (std::size_t i = 0; i < recvIdx.size(); ++i)
    {
        sendQ[i] = src[recvIdx[i]];
    }

    requests.clear();
    for (int r = 0; r < numRanks; ++r)
    {
        if (sendCounts[r] > 0)
        {
            mpiRecvAsync(recvQ.data() + sendDispl[r], sendCounts[r], r, extractValueTag, requests, comm);
        }
    }

    for (int r = 0; r < numRanks; ++r)
    {
        if (recvCounts[r] > 0)
        {
            mpiSendAsync(sendQ.data() + recvDispl[r], recvCounts[r], r, extractValueTag, requests, comm);
        }
    }

    MPI_Waitall(int(requests.size()), requests.data(), MPI_STATUSES_IGNORE);

    //! recvQ is in sendIdx order, order[m] is the original position
    for (std::size_t m = 0; m < n; ++m)
    {
        out[order[m]] = recvQ[m];
    }
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

/*! @brief p2p fetch of the far LET leaves from their owners
 *
 * @param[in]  let               LET
 * @param[in]  peerRanges        LET leaf ranges of my peers and me
 * @param[in]  globalNodeKeys    global node keys
 * @param[in]  globalLevelRange  global level ranges
 * @param[in]  nodeOwner         owner rank per global node
 * @param[in]  globalQuantities  quantities of the global tree
 * @param[out] localQuantities   quantities of the LET, far leaves written
 * @param[-]   letIdxBuf         temp buffer, size let.numLeafNodes
 * @param[-]   letToGlobBuf      temp buffer, size let.numLeafNodes
 * @param[in]  comm              MPI communicator
 */
template<class T, class KeyType>
void extractGlobalP2p(cstone::OctreeView<const KeyType>      let,        // temporary: octreeAcc_ in cstone
                      std::span<const cstone::TreeIndexPair> peerRanges, // temporary: peerRanges_ in cstone
                      const KeyType* globalNodeKeys, const TreeNodeIndex* globalLevelRange,
                      std::span<const int> nodeOwner, // temporary: nodeOwner_ in cstone
                      std::span<const T> globalQuantities, std::span<T> localQuantities, TreeNodeIndex* letIdxBuf,
                      TreeNodeIndex* letToGlobBuf, MPI_Comm comm)
{
    //! list of leaf cell indices in the locally focused tree that need global information
    using namespace cstone;
    auto idxFromGlob = enumerateRanges(invertRanges(0, peerRanges, let.numLeafNodes));
    // const TreeNodeIndex* toInternal = leafToInternal(octreeAcc_).data();
    const TreeNodeIndex* toInternal = let.leafToInternal + let.numInternalNodes; // temporary
    std::span            letIdx{letIdxBuf, idxFromGlob.size()};
    std::span            letToGlob{letToGlobBuf, idxFromGlob.size()};
    // if constexpr (execution::HaveGpu<Exec>{})
    // {
    //     memcpyH2DAsync(exec_, idxFromGlob.data(), idxFromGlob.size(), letIdx.data());
    //     gather(exec_, letIdx.data(), idxFromGlob.size(), toInternal, letIdx.data());

    //     locateNodesGpu(exec_, let.prefixes.data(), letIdx.data(), idxFromGlob.size(), globalNodeKeys,
    //                     globalLevelRange, letToGlob.data());
    //     gatherScatter(exec_, letToGlob.data(), letIdx.data(), idxFromGlob.size(), globalQuantities.data(),
    //                     localQuantities.data());
    // }
    // else
    {
        gather<TreeNodeIndex>(idxFromGlob, toInternal, idxFromGlob.data());
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < idxFromGlob.size(); ++i)
        {
            letToGlob[i] = locateNode(let.prefixes[idxFromGlob[i]], globalNodeKeys, globalLevelRange);
        }
        // gatherScatter<TreeNodeIndex>(letToGlob, idxFromGlob, globalQuantities.data(), localQuantities.data());
        std::vector<T> globValues(idxFromGlob.size());
        gatherFromOwners<T>(letToGlob, nodeOwner, globalQuantities.data(), globValues.data(), comm);
        scatter<TreeNodeIndex>(idxFromGlob, globValues.data(), localQuantities.data());
    }
}

/*! @brief p2p version of FocusedOctree::globalExchange
 *
 * @param[in]    focusTree        LET
 * @param[in]    gOctree          global octree, same on all ranks
 * @param[inout] quantities       quantities of the LET
 * @param[out]   globQOut         quantities of the global tree, can be empty
 * @param[-]     scratch          temp buffer
 * @param[in]    globDispl        first global leaf per rank, length numRanks + 1
 * @param[in]    nodeOwner        owner rank per global node
 * @param[in]    peerRanges       LET leaf ranges of my peers and me
 * @param[in]    myRank           executing rank
 * @param[in]    comm             MPI communicator
 * @param[in]    upsweepFunction  upsweep of one level
 * @param[in]    upsweepArgs      extra upsweep arguments, e.g. expansion centers
 */
template<class Q, class KeyType, class RealType, class F, class Vector, class... UArgs>
void globalExchangeP2p(const cstone::FocusedOctree<KeyType, RealType>& focusTree, // temporary: this in cstone
                       cstone::OctreeView<const KeyType> gOctree, std::span<Q> quantities, std::span<Q> globQOut,
                       Vector&                                scratch,
                       std::span<const TreeNodeIndex>         globDispl,  // temporary: globDispl_ in cstone
                       std::span<const int>                   nodeOwner,  // temporary: nodeOwner_ in cstone
                       std::span<const cstone::TreeIndexPair> peerRanges, // temporary: peerRanges_ in cstone
                       int myRank, MPI_Comm comm,                         // temporary: myRank_, comm_ in cstone
                       F&& upsweepFunction, UArgs&&... upsweepArgs)
{
    using namespace cstone;

    assert(gOctree.leaves != nullptr);
    TreeNodeIndex globNumNodes = globDispl[myRank + 1] - globDispl[myRank]; // globNumNodes_[myRank_] in cstone
    std::size_t   numLetIdx    = focusTree.octreeViewAcc().numLeafNodes;
    auto          s            = scratch.size();
    auto [gLeafQLoc, globQ, gmap, letIdx, letToGlob] = util::packAllocBuffer(
        scratch, util::TypeList<Q, Q, TreeNodeIndex, TreeNodeIndex, TreeNodeIndex>{},
        {size_t(globNumNodes), size_t(gOctree.numNodes), size_t(globNumNodes), numLetIdx, numLetIdx}, 128);
    focusTree.template populateGlobal<Q>(gOctree.leafSpan(), quantities, gLeafQLoc, gmap);

    auto globQuse = globQOut.data() ? globQOut : globQ;
    //! upsweep with the global tree, replaces gatherGlobalLeaves + scatter + upsweep
    globalUpsweep<Q>(gOctree, globDispl, nodeOwner, std::span<const Q>(gLeafQLoc), globQuse, myRank, comm,
                     upsweepFunction, upsweepArgs...);

    //! from the global tree, extract the part that the executing rank was missing
    extractGlobalP2p<Q>(focusTree.octreeViewAcc(), peerRanges, gOctree.prefixes, gOctree.d_levelRange, nodeOwner,
                        std::span<const Q>(globQuse), quantities, letIdx.data(), letToGlob.data(), comm);
    reallocate(scratch, s, 1.0);
}

} // namespace upsweep_p2p
