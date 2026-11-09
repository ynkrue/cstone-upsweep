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

//! @brief artificial demand, replaces the LET to global mapping: the global nodes that fail the MAC
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

//! @brief FocusedOctree::createGlobalExchangeLists with the wanted global nodes passed in
template<class Exec, class KeyType>
ExchangeGraph createGlobalExchangeLists(Exec exec, OctreeView<const KeyType> gOctree,
                                        const SfcAssignment<KeyType>& assignment,
                                        std::span<const TreeNodeIndex> targets, int myRank,
                                        MPI_Comm comm)
{
    GlobalExchangeRange                 setupRange("cstone:global-exchange/setup");
    OctreeData<KeyType, execution::Cpu> globalTreeHost;
    OctreeView<const KeyType>           globalTreeView = gOctree;
    if constexpr (execution::HaveGpu<Exec>{})
    {
        //! List construction and request routing on CPU
        downloadOctreeFromGpu(exec, globalTreeHost, gOctree);
        syncGpu(exec);
        globalTreeView = globalTreeHost.cdata();
    }

    //! base graph, identical on all ranks
    auto graph = buildExchangeGraph(globalTreeView, assignment);

    //! setup connections, used from registerTargets on
    auto setupPairs = basePairs(globalTreeView, assignment, myRank);

    //! add target node indices to graph
    addGraphTargets(globalTreeView, assignment, graph, targets, setupPairs, myRank, comm);

    return graph;
}

//! @brief debug print of all graph nodes: id, cell, compute rank, inputs, targets
inline void printExchangeGraph(const ExchangeGraph& graph)
{
    std::printf("graph: %d nodes, %zu inputs, %zu targets\n", graph.numNodes(), graph.inputIds.size(),
                graph.targetRanks.size());
    if (graph.inputOffsets.size() != graph.cellIds.size() + 1 || graph.targetOffsets.size() != graph.cellIds.size() + 1)
    {
        std::printf("  offsets not of length N + 1, skip node print\n");
        return;
    }
    for (TreeNodeIndex g = 0; g < graph.numNodes(); ++g)
    {
        std::printf("  %3d cell %3d rank %3d | in", g, graph.cellIds[g], graph.computeRanks[g]);
        for (TreeNodeIndex i = graph.inputOffsets[g]; i < graph.inputOffsets[g + 1]; ++i)
        {
            std::printf(" %d", graph.inputIds[i]);
        }
        std::printf(" | tgt");
        for (TreeNodeIndex t = graph.targetOffsets[g]; t < graph.targetOffsets[g + 1]; ++t)
        {
            std::printf(" %d", graph.targetRanks[t]);
        }
        std::printf("\n");
    }
}

template<template<class...> class Vector, class Exec, class Upsweep>
int testGlobalExchange(Exec exec, Upsweep upsweep, int rank, int numRanks)
{
    using KeyType               = unsigned;
    constexpr double theta      = 1.1;
    constexpr int    reps       = 10;
    constexpr int    maxPrinted = 80;

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

    // setup: once, includes the tree download as in GROMACS
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    auto   graph = createGlobalExchangeLists(exec, gOctree, assignment, targets, rank, MPI_COMM_WORLD);
    double setupTime = MPI_Wtime() - start;

    // base graph against its definition, on every rank
    if (rank == 0 && hostTree.numNodes <= maxPrinted) { printExchangeGraph(graph); }

    // Vector<Q> globQ(hostTree.numNodes);
    Vector<Q> allLeaves(hostTree.numLeafNodes);
    Vector<Q> allQ(hostTree.numNodes);

    auto download = [](const Vector<Q>& v)
    {
        if constexpr (execution::HaveGpu<Exec>{}) { return toHost(v); }
        else
        {
            return v;
        }
    };

    std::vector<Q>      leafQ(hostTree.numLeafNodes), reference(hostTree.numNodes);
    // std::vector<double> sweepTime(reps);
    std::vector<double> allgatherTime(reps);
    int                 errors = 0;
    for (int rep = 0; rep < reps; ++rep)
    {
        populate(rep, leafQ);
        cstone::scatter(hostTree.leafToInternalSpan(), leafQ.data(), reference.data());
        cstone::upsweep(hostTree.levelRangeSpan(), hostTree.childOffsets, reference.data(), NodeCount<Q>{});
        Vector<Q> localQ(leafQ.data() + firstLeaf, leafQ.data() + firstLeaf + numLeaves);
        if constexpr (execution::HaveGpu<Exec>{}) { syncGpu(exec); }

        // // point-to-point
        // MPI_Barrier(MPI_COMM_WORLD);
        // start = MPI_Wtime();
        // globalExchange(exec, gOctree, lists, firstLeaf, std::span<const Q>(localQ.data(), localQ.size()),
        //                std::span<Q>(globQ.data(), globQ.size()), std::span<Q>(sendBuf.data(), sendBuf.size()),
        //                std::span<Q>(recvBuf.data(), recvBuf.size()), MPI_COMM_WORLD, upsweep);
        // if constexpr (execution::HaveGpu<Exec>{}) { syncGpu(exec); }
        // sweepTime[rep] = MPI_Wtime() - start;

        // Allgather + full upsweep
        MPI_Barrier(MPI_COMM_WORLD);
        start = MPI_Wtime();
        mpiAllgatherGpuDirect(exec, localQ.data(), allLeaves.data(), numLeaves, MPI_COMM_WORLD);
        scatter(exec, gOctree.leafToInternalSpan(), allLeaves.data(), allQ.data());
        upsweep(gOctree.levelRangeSpan(), gOctree.childOffsets, allQ.data());
        if constexpr (execution::HaveGpu<Exec>{}) { syncGpu(exec); }
        allgatherTime[rep] = MPI_Wtime() - start;

        // check owned + wanted nodes from p2p and all nodes from Allgather
        // auto p2pQ = download(globQ);
        auto fullQ = download(allQ);
        for (TreeNodeIndex i = 0; i < hostTree.numNodes; ++i)
        {
            // bool required = nodeOwner(hostTree, assignment, i) == rank || flags[i];
            // if (required && p2pQ[i] != reference[i]) { ++errors; }
            if (fullQ[i] != reference[i]) { ++errors; }
        }
    }

    // sum of errors, maximum of times over ranks
    MPI_Allreduce(MPI_IN_PLACE, &errors, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &setupTime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    // MPI_Allreduce(MPI_IN_PLACE, sweepTime.data(), reps, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, allgatherTime.data(), reps, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0)
    {
        std::printf("tree: P %d, %d leaves, %d leaves/rank, Q %zu B\n", numRanks, hostTree.numLeafNodes, numLeaves,
                    sizeof(Q));
        std::printf("setup [ms]: %.3f\n", setupTime * 1e3);
        for (int rep = 0; rep < reps; ++rep)
        {
            std::printf("rep %d [ms]: allgather %.3f\n", rep + 1, allgatherTime[rep] * 1e3);
        }
        std::printf("allgather upsweep: %s (%d errors)\n", errors ? "FAIL" : "PASS", errors);
    }
    return errors != 0;
}
