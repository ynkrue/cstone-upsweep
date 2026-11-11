/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief GROMACS isolated global exchange
 *
 * @author Yannik Rüfenacht <yannik.rue@gmail.com>
 */

#pragma once

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstdint>
#include <span>
#include <vector>
#include <mpi.h>

#include "cstone/cuda/cuda_utils.hpp"
#include "cstone/primitives/mpi_wrappers.hpp"
#include "cstone/primitives/primitives_acc.hpp"
#include "cstone/traversal/macs.hpp"

namespace global_sweep
{
using namespace cstone;

/*! @brief message to one peer rank
 *
 * Message m covers the node indices sendIdx/recvIdx[m.offset : m.offset + m.count]
 */
struct SweepMessage
{
    int           peer;
    TreeNodeIndex offset, count;
};

/*! @brief find the far nodes of every rank
 *
 * @tparam KeyType         SFC key type
 * @tparam T               floating point type of the box and node centers
 * @param[in]  gOctree     global octree
 * @param[in]  gLeaves     global cornerstone leaves
 * @param[in]  treeOffsets first global leaf of each rank
 * @param[in]  centers     MAC centers and squared radii of all global nodes
 * @param[in]  box         global coordinate bounding box
 * @param[out] farOffsets  P + 1 offsets into @p farNodes
 * @param[out] farNodes    far nodes of rank r in [farOffsets[r], farOffsets[r + 1])
 */
template<class KeyType, class T>
void computeFarNodes(OctreeView<const KeyType> gOctree,
                     std::span<const KeyType> gLeaves,
                     std::span<const TreeNodeIndex> treeOffsets,
                     std::span<const Vec4<T>> centers,
                     const Box<T>& box,
                     std::vector<TreeNodeIndex>& farOffsets,
                     std::vector<TreeNodeIndex>& farNodes)
{
    int numRanks = int(treeOffsets.size()) - 1;
    farOffsets.assign(numRanks + 1, 0);
    farNodes.clear();

    for (int r = 0; r < numRanks; ++r)
    {
        KeyType focusStart = gLeaves[treeOffsets[r]];
        KeyType focusEnd   = gLeaves[treeOffsets[r + 1]];

        std::vector<uint8_t> flags(gOctree.numNodes);
        markMacs(gOctree.prefixes, gOctree.childOffsets, gOctree.parents, centers.data(), box,
                 gLeaves.data() + treeOffsets[r], treeOffsets[r + 1] - treeOffsets[r], false, flags.data());

        // parent of node c is stored at parents[(c - 1) / 8]
        for (TreeNodeIndex c = 1; c < gOctree.numNodes; ++c)
        {
            if (flags[gOctree.parents[(c - 1) / 8]] && !flags[c] &&
                !containedIn(gOctree.prefixes[c], focusStart, focusEnd))
            {
                farNodes.push_back(c);
            }
        }
        farOffsets[r + 1] = farNodes.size();
    }
}

//! @brief number of rank index bits owned by rounds 1..@p round (1-based), for 1, 2 or 4 leaves per rank
inline int lowBits(int leavesPerRank, int round) { return 3 * round - std::countr_zero(unsigned(leavesPerRank)); }

/*! @brief true if @p rank computes its level lam - 1 node in @p round, lam = L - round + 1
 *
 * True iff that node or one of its ancestors is a far node of a rank that has the same lowest
 * lowBits(round) rank index bits as @p rank.
 *
 * @param[in] gOctree,gLeaves,treeOffsets,farOffsets,farNodes  see computeFarNodes
 * @param[in] rank        rank that computes
 * @param[in] round       1-based round, 1..L
 */
template<class KeyType>
bool computesParent(OctreeView<const KeyType> gOctree,
                    std::span<const KeyType> gLeaves,
                    std::span<const TreeNodeIndex> treeOffsets,
                    std::span<const TreeNodeIndex> farOffsets,
                    std::span<const TreeNodeIndex> farNodes,
                    int rank,
                    int round)
{
    int numRanks = int(treeOffsets.size()) - 1;
    int level    = maxDepth(gOctree.levelRange, maxTreeLevel<KeyType>{} + 2) - round;
    int bits     = lowBits(treeOffsets[1] - treeOffsets[0], round);

    KeyType cell = encodePlaceholderBit(gLeaves[treeOffsets[rank]], 3 * level);
    for (int t = rank & ((1 << bits) - 1); t < numRanks; t += 1 << bits)
    {
        for (TreeNodeIndex i = farOffsets[t]; i < farOffsets[t + 1]; ++i)
        {
            auto [nodeStart, nodeEnd] = decodePlaceholderBit2K(gOctree.prefixes[farNodes[i]]);
            if (containedIn(cell, nodeStart, nodeEnd)) { return true; }
        }
    }
    return false;
}

/*! @brief nodes that rank @p src sends to rank @p dst in @p round
 *
 * Far nodes of all ranks behind @p dst that lie inside the level lam cell of @p src (lam = L - round + 1;
 * for lam == L the leaves of src), plus that cell itself (all leaves of src for lam == L) if @p dst
 * computes its parent. This is the only place the exchange policy lives.
 *
 * @param[in] gOctree,gLeaves,treeOffsets,farOffsets,farNodes  see computeFarNodes
 * @param[in] src         sending rank
 * @param[in] dst         receiving rank
 * @param[in] round       1-based round, 1..L
 * @return                sorted node indices without duplicates
 */
template<class KeyType>
std::vector<TreeNodeIndex> messageNodes(OctreeView<const KeyType> gOctree,
                                        std::span<const KeyType> gLeaves,
                                        std::span<const TreeNodeIndex> treeOffsets,
                                        std::span<const TreeNodeIndex> farOffsets,
                                        std::span<const TreeNodeIndex> farNodes,
                                        int src,
                                        int dst,
                                        int round)
{
    int numRanks = int(treeOffsets.size()) - 1;
    int level    = maxDepth(gOctree.levelRange, maxTreeLevel<KeyType>{} + 2) - round + 1;
    int bits     = lowBits(treeOffsets[1] - treeOffsets[0], round);

    // level lam cell of src; for lam == L the leaves of src, which may be several level L nodes
    KeyType cellStart = enclosingBoxCode(gLeaves[treeOffsets[src]], level);
    KeyType cellEnd   = std::max(gLeaves[treeOffsets[src + 1]], cellStart + nodeRange<KeyType>(level));

    std::vector<TreeNodeIndex> nodes;
    for (int t = dst & ((1 << bits) - 1); t < numRanks; t += 1 << bits)
    {
        for (TreeNodeIndex i = farOffsets[t]; i < farOffsets[t + 1]; ++i)
        {
            if (containedIn(gOctree.prefixes[farNodes[i]], cellStart, cellEnd)) { nodes.push_back(farNodes[i]); }
        }
    }

    if (computesParent(gOctree, gLeaves, treeOffsets, farOffsets, farNodes, dst, round))
    {
        for (KeyType key = cellStart; key < cellEnd; key += nodeRange<KeyType>(level))
        {
            nodes.push_back(locateNode(encodePlaceholderBit(key, 3 * level), gOctree.prefixes, gOctree.levelRange));
        }
    }

    std::sort(nodes.begin(), nodes.end());
    nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
    return nodes;
}

//! @brief send, receive and compute lists of one rank for all rounds
struct SweepLists
{
    std::vector<SweepMessage>  send, recv;               // messages, grouped by round
    std::vector<TreeNodeIndex> sendOffsets, recvOffsets; // numRounds + 1, into send / recv
    std::vector<TreeNodeIndex> sendIdx, recvIdx;         // node indices of all messages, see SweepMessage
    std::vector<TreeNodeIndex> computeOffsets;           // numRounds + 1, into computeNodes
    std::vector<TreeNodeIndex> computeNodes;             // parents to compute after the receives of each round
};

/*! @brief build the send, receive and compute lists of @p myRank
 *
 * Requires P a power of two and a complete tree with 1, 2 or 4 leaves per rank.
 *
 * @param[in] gOctree,gLeaves,treeOffsets,farOffsets,farNodes  see computeFarNodes
 * @param[in] myRank      rank to build the lists for
 */
template<class KeyType>
SweepLists createSweepLists(OctreeView<const KeyType> gOctree,
                            std::span<const KeyType> gLeaves,
                            std::span<const TreeNodeIndex> treeOffsets,
                            std::span<const TreeNodeIndex> farOffsets,
                            std::span<const TreeNodeIndex> farNodes,
                            int myRank)
{
    int lpr       = treeOffsets[1] - treeOffsets[0];
    int numRounds = maxDepth(gOctree.levelRange, maxTreeLevel<KeyType>{} + 2);
    assert(std::has_single_bit(treeOffsets.size() - 1) && (lpr == 1 || lpr == 2 || lpr == 4));

    SweepLists lists;
    lists.sendOffsets = lists.recvOffsets = lists.computeOffsets = {0};

    for (int round = 1; round <= numRounds; ++round)
    {
        // partners differ from myRank only in the rank index bits [first, lowBits(round)) of this round
        int first = std::max(0, lowBits(lpr, round - 1));
        int group = (1 << lowBits(lpr, round)) - (1 << first);
        int base  = myRank & ~group;

        for (int peer = base; peer <= (base | group); peer += 1 << first)
        {
            if (peer == myRank) { continue; }

            auto sendNodes = messageNodes(gOctree, gLeaves, treeOffsets, farOffsets, farNodes, myRank, peer, round);
            auto recvNodes = messageNodes(gOctree, gLeaves, treeOffsets, farOffsets, farNodes, peer, myRank, round);
            if (!sendNodes.empty())
            {
                lists.send.push_back({peer, TreeNodeIndex(lists.sendIdx.size()), TreeNodeIndex(sendNodes.size())});
                lists.sendIdx.insert(lists.sendIdx.end(), sendNodes.begin(), sendNodes.end());
            }
            if (!recvNodes.empty())
            {
                lists.recv.push_back({peer, TreeNodeIndex(lists.recvIdx.size()), TreeNodeIndex(recvNodes.size())});
                lists.recvIdx.insert(lists.recvIdx.end(), recvNodes.begin(), recvNodes.end());
            }
        }
        lists.sendOffsets.push_back(lists.send.size());
        lists.recvOffsets.push_back(lists.recv.size());

        if (computesParent(gOctree, gLeaves, treeOffsets, farOffsets, farNodes, myRank, round))
        {
            KeyType cell = encodePlaceholderBit(gLeaves[treeOffsets[myRank]], 3 * (numRounds - round));
            lists.computeNodes.push_back(locateNode(cell, gOctree.prefixes, gOctree.levelRange));
        }
        lists.computeOffsets.push_back(lists.computeNodes.size());
    }
    return lists;
}

/*! @brief run the sweep: for each round send, receive, then compute parents
 *
 * @tparam Exec               execution policy
 * @tparam Q                  node value type
 * @tparam ComputeParents     callable computeParents(std::span<const TreeNodeIndex>, childOffsets, Q*)
 * @param[in]    exec         execution policy
 * @param[in]    lists        send, receive and compute lists, see createSweepLists
 * @param[in]    sendIdx      lists.sendIdx in the execution space of @p exec
 * @param[in]    recvIdx      lists.recvIdx in the execution space of @p exec
 * @param[in]    childOffsets global tree child offsets in the execution space of @p exec
 * @param[inout] globQ        values of all global nodes, own leaves filled by the caller
 * @param[-]     sendBuf      send buffer, size sendIdx.size()
 * @param[-]     recvBuf      receive buffer, size recvIdx.size()
 * @param[in]    computeParents  computes the values of the given nodes from their children
 * @param[in]    tag          MPI tag
 * @param[in]    comm         MPI communicator
 */
template<class Exec, class Q, class ComputeParents>
void globalSweep(Exec exec,
                 const SweepLists& lists,
                 std::span<const TreeNodeIndex> sendIdx,
                 std::span<const TreeNodeIndex> recvIdx,
                 const TreeNodeIndex* childOffsets,
                 std::span<Q> globQ,
                 std::span<Q> sendBuf,
                 std::span<Q> recvBuf,
                 ComputeParents&& computeParents,
                 int tag,
                 MPI_Comm comm)
{
    int numRounds = int(lists.sendOffsets.size()) - 1;

    // entries of round e of an array grouped by round, e.g. lists.send with lists.sendOffsets
    auto roundSlice = [](const auto& v, const auto& offsets, int e)
    { return std::span(v.data() + offsets[e], v.data() + offsets[e + 1]); };

    // mpiRecvAsync appends one request per call, so recvRequests[i] belongs to lists.recv[i]
    std::vector<MPI_Request> recvRequests, sendRequests;
    for (const auto& m : lists.recv)
    {
        mpiRecvAsync(recvBuf.data() + m.offset, m.count, m.peer, tag, recvRequests, comm);
    }

    for (int e = 0; e < numRounds; ++e)
    {
        // the messages of one round are contiguous in sendIdx / recvIdx, so one gather / scatter serves them all
        auto sends = roundSlice(lists.send, lists.sendOffsets, e);
        if (!sends.empty())
        {
            TreeNodeIndex first = sends.front().offset;
            TreeNodeIndex count = sends.back().offset + sends.back().count - first;
            gather(exec, sendIdx.subspan(first, count), globQ.data(), sendBuf.data() + first);
            if constexpr (execution::HaveGpu<Exec>{}) { syncGpu(exec); }
            for (const auto& m : sends)
            {
                mpiSendAsync(sendBuf.data() + m.offset, m.count, m.peer, tag, sendRequests, comm);
            }
        }

        auto recvs = roundSlice(lists.recv, lists.recvOffsets, e);
        if (!recvs.empty())
        {
            TreeNodeIndex first = recvs.front().offset;
            TreeNodeIndex count = recvs.back().offset + recvs.back().count - first;
            MPI_Waitall(int(recvs.size()), recvRequests.data() + lists.recvOffsets[e], MPI_STATUSES_IGNORE);
            scatter(exec, recvIdx.subspan(first, count), recvBuf.data() + first, globQ.data());
        }

        computeParents(roundSlice(lists.computeNodes, lists.computeOffsets, e), childOffsets, globQ.data());
    }

    // the last scatter may still read recvBuf, which the receives of the next sweep overwrite
    if constexpr (execution::HaveGpu<Exec>{}) { syncGpu(exec); }
    MPI_Waitall(int(sendRequests.size()), sendRequests.data(), MPI_STATUSES_IGNORE);
}

} // namespace global_sweep
