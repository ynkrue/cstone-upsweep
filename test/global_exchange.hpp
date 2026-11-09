#pragma once

/*! @file
 * @brief FocusedOctree::globalExchange without the LET: artificial wanted nodes, compared with Allgather + full upsweep
 */
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>
#include <mpi.h>

#include "cstone/domain/domaindecomp.hpp"
#include "cstone/focus/source_center.hpp"
#include "cstone/traversal/macs.hpp"
#include "cstone/tree/csarray.hpp"
#include "cstone/primitives/mpi_cuda.cuh"
#include "global_sweep.hpp"

using namespace global_sweep;
using Q = unsigned;

//! @brief copy the host tree into the execution space, stands in for buildOctreeGpu in GROMACS DomainFixed
template<class Exec, class KeyType>
OctreeData<KeyType, Exec> uploadOctree(const OctreeData<KeyType, execution::Cpu>& host)
{
    OctreeData<KeyType, Exec> acc;
    acc.resize(host.numLeafNodes);
    acc.prefixes       = host.prefixes;
    acc.childOffsets   = host.childOffsets;
    acc.parents        = host.parents;
    acc.levelRange     = host.levelRange;
    acc.d_levelRange   = host.d_levelRange;
    acc.internalToLeaf = host.internalToLeaf;
    acc.leafToInternal = host.leafToInternal;
    return acc;
}

/*! @brief artificial demand, replaces the LET to global mapping: the global nodes that fail the MAC
 *
 * @param[in] gOctree    global octree on the host
 * @param[in] myLeaves   SFC keys of my leaves, length numLeaves + 1
 * @param[in] numLeaves  number of leaves of this rank
 * @param[in] theta      opening angle
 * @return               flag per global node, 1 = wanted
 */
template<class KeyType>
std::vector<uint8_t> flagNodes(OctreeView<const KeyType> gOctree, const KeyType* myLeaves, TreeNodeIndex numLeaves,
                               double theta)
{
    std::vector<uint8_t>      flags(gOctree.numNodes);
    std::vector<Vec4<double>> centers(gOctree.numNodes);
    Box<double>               box(0, 1);
    geoMacSpheres(std::span<const KeyType>(gOctree.prefixes, gOctree.numNodes), centers.data(), invThetaMinMac(theta),
                  box);
    markMacs(gOctree.prefixes, gOctree.childOffsets, gOctree.parents, centers.data(), box, myLeaves, numLeaves, false,
             flags.data());
    return flags;
}

//! @brief leaf values of repetition @p rep, different in every repetition so stale values fail the check
inline void populate(int rep, std::span<Q> leafQ)
{
    for (std::size_t i = 0; i < leafQ.size(); ++i)
    {
        leafQ[i] = Q(i) + Q(rep) + 1;
    }
}

/*! @brief FocusedOctree::globalExchange without populateGlobal / extractGlobal
 *
 * @param[in]  gOctree    global octree in the execution space
 * @param[in]  lists      from createGlobalExchangeLists
 * @param[in]  firstLeaf  index of my first global leaf
 * @param[in]  localQ     values of my leaves
 * @param[out] globQ      global node values: owned + wanted nodes are valid afterwards
 */
template<class Exec, class KeyType, class IdxVec, class F>
void globalExchange(Exec exec, OctreeView<const KeyType> gOctree, const GlobalExchangeLists<IdxVec>& lists,
                    TreeNodeIndex firstLeaf, std::span<const Q> localQ, std::span<Q> globQ, std::span<Q> sendBuf,
                    std::span<Q> recvBuf, MPI_Comm comm, F&& upsweep)
{
    fill(exec, globQ.data(), globQ.data() + globQ.size(), Q{});
    scatter(exec, gOctree.leafToInternalSpan().subspan(firstLeaf, localQ.size()), localQ.data(), globQ.data());
    globalSweep(exec, gOctree, lists, globQ, sendBuf, recvBuf, comm, upsweep);
}

template<template<class...> class Vector, class Exec, class Upsweep>
int testGlobalExchange(Exec exec, Upsweep upsweep, int rank, int numRanks)
{
    using KeyType          = unsigned;
    constexpr double theta = 1.1;
    constexpr int    reps  = 10;

    // global tree
    std::vector<KeyType> boundaries(numRanks + 1);
    KeyType              stride = nodeRange<KeyType>(0) / numRanks;
    for (int r = 0; r < numRanks; ++r)
    {
        boundaries[r] = KeyType(r) * stride;
    }
    boundaries.back()                          = nodeRange<KeyType>(0);
    auto                                leaves = computeSpanningTree<KeyType>(boundaries);
    std::vector<unsigned>               counts(nNodes(leaves), 1);
    OctreeData<KeyType, execution::Cpu> tree;
    tree.resize(nNodes(leaves));
    updateInternalTree<KeyType>(leaves, tree.data());
    auto hostTree = tree.cdata();

    // global tree in the execution space, as globalOctreeAcc_ in GROMACS DomainFixed
    auto gOctreeAcc = uploadOctree<Exec>(tree);
    auto gOctree    = gOctreeAcc.cdata();

    auto          assignment = makeSfcAssignment(numRanks, counts, leaves.data());
    auto          displ      = assignment.treeOffsetsConst();
    TreeNodeIndex firstLeaf  = displ[rank];
    TreeNodeIndex numLeaves  = displ[rank + 1] - displ[rank];

    // the Allgather baseline needs the same number of leaves on every rank
    for (int r = 0; r < numRanks; ++r)
    {
        if (displ[r + 1] - displ[r] == numLeaves) { continue; }
        if (rank == 0) { std::printf("unequal leaves per rank at P %d, Allgather not possible\n", numRanks); }
        return 1;
    }

    // wanted nodes
    auto                       flags = flagNodes(hostTree, leaves.data() + firstLeaf, numLeaves, theta);
    std::vector<TreeNodeIndex> targets;
    for (TreeNodeIndex n = 0; n < hostTree.numNodes; ++n)
    {
        if (flags[n]) { targets.push_back(n); }
    }

    // setup: lists once, includes the tree download as in GROMACS
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    auto   lists = createGlobalExchangeLists<Vector<TreeNodeIndex>>(exec, gOctree, assignment, std::move(targets), rank,
                                                                    MPI_COMM_WORLD);
    double setupTime = MPI_Wtime() - start;

    Vector<Q> globQ(hostTree.numNodes), allLeaves(hostTree.numLeafNodes), allQ(hostTree.numNodes);
    Vector<Q> sendBuf(lists.up.sendIdx.size() + lists.down.sendIdx.size());
    Vector<Q> recvBuf(lists.up.recvIdx.size() + lists.down.recvIdx.size());

    auto download = [](const Vector<Q>& v)
    {
        if constexpr (execution::HaveGpu<Exec>{}) { return toHost(v); }
        else
        {
            return v;
        }
    };

    std::vector<Q>      leafQ(hostTree.numLeafNodes), reference(hostTree.numNodes);
    std::vector<double> sweepTime(reps), allgatherTime(reps);
    int                 errors = 0;
    for (int rep = 0; rep < reps; ++rep)
    {
        populate(rep, leafQ);
        cstone::scatter(hostTree.leafToInternalSpan(), leafQ.data(), reference.data());
        cstone::upsweep(hostTree.levelRangeSpan(), hostTree.childOffsets, reference.data(), NodeCount<Q>{});
        Vector<Q> localQ(leafQ.data() + firstLeaf, leafQ.data() + firstLeaf + numLeaves);
        if constexpr (execution::HaveGpu<Exec>{}) { syncGpu(exec); }

        // point-to-point
        MPI_Barrier(MPI_COMM_WORLD);
        start = MPI_Wtime();
        globalExchange(exec, gOctree, lists, firstLeaf, std::span<const Q>(localQ.data(), localQ.size()),
                       std::span<Q>(globQ.data(), globQ.size()), std::span<Q>(sendBuf.data(), sendBuf.size()),
                       std::span<Q>(recvBuf.data(), recvBuf.size()), MPI_COMM_WORLD, upsweep);
        if constexpr (execution::HaveGpu<Exec>{}) { syncGpu(exec); }
        sweepTime[rep] = MPI_Wtime() - start;

        // Allgather + full upsweep
        MPI_Barrier(MPI_COMM_WORLD);
        start = MPI_Wtime();
        mpiAllgatherGpuDirect(exec, localQ.data(), allLeaves.data(), numLeaves, MPI_COMM_WORLD);
        scatter(exec, gOctree.leafToInternalSpan(), allLeaves.data(), allQ.data());
        upsweep(gOctree.levelRangeSpan(), gOctree.childOffsets, allQ.data());
        if constexpr (execution::HaveGpu<Exec>{}) { syncGpu(exec); }
        allgatherTime[rep] = MPI_Wtime() - start;

        // check: owned + wanted nodes from p2p, all nodes from Allgather
        auto p2pQ = download(globQ), fullQ = download(allQ);
        for (TreeNodeIndex i = 0; i < hostTree.numNodes; ++i)
        {
            bool required = nodeOwner(hostTree, assignment, i) == rank || flags[i];
            if (required && p2pQ[i] != reference[i]) { ++errors; }
            if (fullQ[i] != reference[i]) { ++errors; }
        }
    }

    // maximum over ranks
    MPI_Allreduce(MPI_IN_PLACE, &errors, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &setupTime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, sweepTime.data(), reps, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, allgatherTime.data(), reps, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0)
    {
        std::printf("tree: P %d, %d leaves, %d leaves/rank, Q %zu B\n", numRanks, hostTree.numLeafNodes, numLeaves,
                    sizeof(Q));
        std::printf("setup [ms]: %.3f\n", setupTime * 1e3);
        for (int rep = 0; rep < reps; ++rep)
        {
            std::printf("rep %d [ms]: p2p %.3f | allgather %.3f\n", rep+1, sweepTime[rep] * 1e3,
                        allgatherTime[rep] * 1e3);
        }
        std::printf("upsweep: %s (%d errors)\n", errors ? "FAIL" : "PASS", errors);
    }
    return errors != 0;
}
