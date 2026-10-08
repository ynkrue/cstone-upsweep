#pragma once

#include <span>
#include <vector>

#include <mpi.h>

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
 * gLeafQLoc comes from populateGlobal and globQ is passed on to extractGlobal; both are LET glue.
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

template<class Q, class KeyType, class F>
void globalExchangeP2p(cstone::OctreeView<const KeyType> gOctree,
                        std::span<const TreeNodeIndex> globNumNodes,
                        std::span<const TreeNodeIndex> globDispl,
                        std::span<const int> nodeOwner,
                        std::span<const Q> gLeafQLoc,
                        std::span<Q> globQ,
                        F&& upsweepFunction,
                        int myRank,
                        MPI_Comm comm)
{
    // TODO
    (void)gOctree;
    (void)globNumNodes;
    (void)globDispl;
    (void)nodeOwner;
    (void)gLeafQLoc;
    (void)globQ;
    (void)upsweepFunction;
    (void)myRank;
    (void)comm;
}

} // namespace upsweep_p2p
