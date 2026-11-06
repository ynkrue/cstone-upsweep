#pragma once

/*! @file
 * @brief Isolated GROMACS global exchange, compared with Allgather + full upsweep.
 */
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
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

template<template<class...> class Vector, class Exec, class Upsweep>
int testGlobalExchange(Exec exec, Upsweep upsweep, int rank, int numRanks)
{
    using KeyType          = unsigned;
    constexpr double theta = 1.1;
    constexpr int    reps  = 50;

    // Global tree: same equal SFC boundaries as GROMACS.
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
    auto hostTree   = tree.cdata();
    hostTree.leaves = leaves.data();
    auto assignment = makeSfcAssignment(numRanks, counts, leaves.data());
    auto displ      = assignment.treeOffsetsConst();
    int  localCount = displ[rank + 1] - displ[rank];

    // Rank-local demand: children of MAC-flagged cells.
    MPI_Barrier(MPI_COMM_WORLD);
    double                    start = MPI_Wtime();
    std::vector<uint8_t>      flags(hostTree.numNodes);
    std::vector<Vec4<double>> centers(hostTree.numNodes);
    Box<double>               box(0, 1);
    geoMacSpheres(std::span<const KeyType>(hostTree.prefixes, hostTree.numNodes), centers.data(), invThetaMinMac(theta),
                  box);
    markMacs(hostTree.prefixes, hostTree.childOffsets, hostTree.parents, centers.data(), box,
             leaves.data() + displ[rank], localCount, false, flags.data());
    std::vector<TreeNodeIndex> targets;
    for (TreeNodeIndex n = 0; n < hostTree.numNodes; ++n)
    {
        if (!flags[n] || !hostTree.childOffsets[n]) { continue; }
        for (TreeNodeIndex c = hostTree.childOffsets[n]; c < hostTree.childOffsets[n] + 8; ++c)
        {
            targets.push_back(c);
        }
    }
    double times[4] = {MPI_Wtime() - start, 0, 0, 0};
    // Build the communication lists once.
    MPI_Barrier(MPI_COMM_WORLD);
    start              = MPI_Wtime();
    auto       upPeers = basePairs(hostTree, assignment, rank);
    LevelPairs downPeers(upPeers.size());
    routeRequests(hostTree, assignment, std::move(targets), rank, MPI_COMM_WORLD, upPeers, downPeers);
    auto upHost = toSweepLists(upPeers), downHost = toSweepLists(downPeers);
    auto owned = computeOwnedParentRanges(hostTree, assignment, rank);

    // Put indices and values in the execution space.
    auto                              activeTree = hostTree;
    SweepLists<Vector<TreeNodeIndex>> up, down;
    up.send      = std::move(upHost.send);
    up.recv      = std::move(upHost.recv);
    up.sendIdx   = upHost.sendIdx;
    up.recvIdx   = upHost.recvIdx;
    down.send    = std::move(downHost.send);
    down.recv    = std::move(downHost.recv);
    down.sendIdx = downHost.sendIdx;
    down.recvIdx = downHost.recvIdx;
    Vector<TreeNodeIndex> children(hostTree.childOffsets, hostTree.childOffsets + hostTree.numNodes);
    Vector<TreeNodeIndex> leafMap(hostTree.leafToInternalSpan().data(),
                                  hostTree.leafToInternalSpan().data() + hostTree.numLeafNodes);
    activeTree.childOffsets = children.data();
    Vector<Q> localQ(localCount), globQ(hostTree.numNodes), allLeaves(hostTree.numLeafNodes), allQ(hostTree.numNodes);
    Vector<Q> sendBuf(up.sendIdx.size() + down.sendIdx.size()), recvBuf(up.recvIdx.size() + down.recvIdx.size());
    if constexpr (execution::HaveGpu<Exec>{}) { syncGpu(exec); }
    times[1] = MPI_Wtime() - start;
    std::span<const TreeNodeIndex> leafIndices(leafMap.data(), hostTree.numLeafNodes);
    std::vector<int>               rankCounts(numRanks);
    for (int r = 0; r < numRanks; ++r)
    {
        rankCounts[r] = displ[r + 1] - displ[r];
    }
    bool equalCounts = std::all_of(rankCounts.begin(), rankCounts.end(), [&](int n) { return n == localCount; });
    std::vector<Q> leafValues(hostTree.numLeafNodes), reference(hostTree.numNodes), result(hostTree.numNodes);
    int            errors = 0;

    // One warm-up, then repeated exchanges with fresh values.
    for (int rep = 0; rep <= reps; ++rep)
    {
        for (size_t i = 0; i < leafValues.size(); ++i)
        {
            leafValues[i] = unsigned(i) + unsigned(rep) + 1;
        }
        cstone::scatter(hostTree.leafToInternalSpan(), leafValues.data(), reference.data());
        cstone::upsweep(hostTree.levelRangeSpan(), hostTree.childOffsets, reference.data(), NodeCount<unsigned>{});
        localQ = std::vector<Q>(leafValues.begin() + displ[rank], leafValues.begin() + displ[rank + 1]);
        // Clear remote values before each exchange.
        fill(exec, globQ.data(), globQ.data() + globQ.size(), Q{});
        if constexpr (execution::HaveGpu<Exec>{}) { syncGpu(exec); }
        MPI_Barrier(MPI_COMM_WORLD);
        start = MPI_Wtime();
        // Point-to-point sweep.
        scatter(exec, leafIndices.subspan(displ[rank], localCount), localQ.data(), globQ.data());
        globalUpsweep(exec, activeTree, up, down, owned, std::span<Q>(globQ.data(), globQ.size()),
                      std::span<Q>(sendBuf.data(), sendBuf.size()), std::span<Q>(recvBuf.data(), recvBuf.size()),
                      MPI_COMM_WORLD, upsweep);
        if constexpr (execution::HaveGpu<Exec>{}) { syncGpu(exec); }
        if (rep > 0) { times[2] += (MPI_Wtime() - start) / reps; }

        MPI_Barrier(MPI_COMM_WORLD);
        start = MPI_Wtime();
        // Collective baseline.
        if (equalCounts) { mpiAllgatherGpuDirect(exec, localQ.data(), allLeaves.data(), localCount, MPI_COMM_WORLD); }
        else
        {
            mpiAllgathervGpuDirect(exec, localQ.data(), localCount, allLeaves.data(), rankCounts.data(), displ.data(),
                                   MPI_COMM_WORLD);
        }
        scatter(exec, leafIndices, allLeaves.data(), allQ.data());
        upsweep(hostTree.levelRangeSpan(), activeTree.childOffsets, allQ.data());
        if constexpr (execution::HaveGpu<Exec>{}) { syncGpu(exec); }
        if (rep > 0) { times[3] += (MPI_Wtime() - start) / reps; }
        // Check both paths against the CPU reference.
        result = toHost(globQ);
        for (TreeNodeIndex i = 0; i < hostTree.numNodes; ++i)
        {
            bool required =
                nodeOwner(hostTree, assignment, i) == rank || (i > 0 && flags[hostTree.parents[(i - 1) / 8]]);
            if (required && result[i] != reference[i]) { ++errors; }
        }
        result = toHost(allQ);
        for (TreeNodeIndex i = 0; i < hostTree.numNodes; ++i)
        {
            if (result[i] != reference[i]) { ++errors; }
        }
    }
    // Report correctness and maximum-rank timings.
    MPI_Allreduce(MPI_IN_PLACE, &errors, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, times, 4, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0)
    {
        auto [minLeaves, maxLeaves] = std::minmax_element(rankCounts.begin(), rankCounts.end());
        std::printf("tree: %d ranks, %d leaves, %d-%d leaves/rank\n", numRanks, hostTree.numLeafNodes, *minLeaves,
                    *maxLeaves);
        std::printf("upsweep: %s (%d errors)\n", errors ? "FAIL" : "PASS", errors);
        std::printf("time [ms] P %d Q %zu B: flags-once %.3f | lists/buffers-once %.3f | upsweep %.3f || %s %.3f\n",
                    numRanks, sizeof(Q), times[0] * 1e3, times[1] * 1e3, times[2] * 1e3,
                    equalCounts ? "allgather" : "allgatherv", times[3] * 1e3);
    }
    return errors != 0;
}
