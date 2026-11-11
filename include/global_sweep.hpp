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

#include <cstdint>
#include <span>
#include <vector>

#include "cstone/sfc/box.hpp"
#include "cstone/traversal/boxoverlap.hpp"
#include "cstone/traversal/macs.hpp"
#include "cstone/tree/definitions.h"
#include "cstone/tree/octree.hpp"

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

        // children of ascending parents are ascending, see linkTreeCpu
        for (TreeNodeIndex i = 0; i < gOctree.numNodes; ++i)
        {
            if (!flags[i] || gOctree.childOffsets[i] == 0) { continue; }
            for (TreeNodeIndex c = gOctree.childOffsets[i]; c < gOctree.childOffsets[i] + 8; ++c)
            {
                if (!flags[c] && !containedIn(gOctree.prefixes[c], focusStart, focusEnd)) { farNodes.push_back(c); }
            }
        }
        farOffsets[r + 1] = farNodes.size();
    }
}

} // namespace global_sweep
