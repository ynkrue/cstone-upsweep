#pragma once

/*! @file
 * @brief Allgather + full upsweep reference for FocusedOctree::globalExchange
 */
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <span>
#include <type_traits>
#include <vector>
#include <mpi.h>

#include "cstone/cuda/cuda_utils.hpp"
#include "cstone/domain/domaindecomp.hpp"
#include "cstone/focus/source_center.hpp"
#include "cstone/primitives/gather.hpp"
#include "cstone/primitives/mpi_cuda.cuh"
#include "cstone/primitives/primitives_acc.hpp"
#include "cstone/sfc/common.hpp"
#include "cstone/traversal/macs.hpp"
#include "cstone/tree/csarray.hpp"
#include "cstone/tree/octree.hpp"
#include "global_sweep.hpp"

using namespace global_sweep;
using Q = util::array<double, 98>;
// using Q = unsigned;

//! @brief node value from its 8 children, always added in the order 0..7; NodeCount for counts, plain sum otherwise
struct SumChildren
{
    template<class T>
    HOST_DEVICE_FUN T operator()(TreeNodeIndex nodeIdx, TreeNodeIndex c, const T* q) const
    {
        if constexpr (std::is_arithmetic_v<T>) { return NodeCount<T>{}(nodeIdx, c, q); }
        else
        {
            T sum = q[c];
            for (int k = 1; k < 8; ++k)
            {
                sum += q[c + k];
            }
            return sum;
        }
    }
};

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
                               double theta, const Box<double>& box)
{
    std::vector<uint8_t>      flags(gOctree.numNodes);
    std::vector<Vec4<double>> centers(gOctree.numNodes);
    geoMacSpheres(std::span<const KeyType>(gOctree.prefixes, gOctree.numNodes), centers.data(), invThetaMinMac(theta),
                  box);
    markMacs(gOctree.prefixes, gOctree.childOffsets, gOctree.parents, centers.data(), box, myLeaves, numLeaves, false,
             flags.data());
    return flags;
}

//! @brief the global nodes a LET would use as sources outside my range
template<class KeyType>
std::vector<uint8_t> flagLetNodes(OctreeView<const KeyType> gOctree, const KeyType* myLeaves, TreeNodeIndex numLeaves,
                                  double theta, const Box<double>& box)
{
    auto macFlags = flagNodes(gOctree, myLeaves, numLeaves, theta, box);

    std::vector<uint8_t> flags(gOctree.numNodes, 0);
    for (TreeNodeIndex i = 0; i < gOctree.numNodes; ++i)
    {
        if (!macFlags[i]) { continue; }
        TreeNodeIndex firstChild = gOctree.childOffsets[i];
        if (firstChild == 0) { continue; }
        for (TreeNodeIndex c = firstChild; c < firstChild + 8; ++c)
        {
            if (!macFlags[c] && !containedIn(gOctree.prefixes[c], myLeaves[0], myLeaves[numLeaves])) { flags[c] = 1; }
        }
    }
    return flags;
}

//! @brief leaf values of repetition @p rep, different in every repetition so stale values fail the check
template<class T>
void populate(int rep, std::span<T> leafQ)
{
    for (std::size_t i = 0; i < leafQ.size(); ++i)
    {
        // small integers, so that sums over the tree are exact; components differ by the factor k + 1
        if constexpr (std::is_arithmetic_v<T>) { leafQ[i] = T(i + rep + 1); }
        else
        {
            for (std::size_t k = 0; k < leafQ[i].size(); ++k)
            {
                leafQ[i][k] = double((i + rep + 1) * (k + 1));
            }
        }
    }
}

template<template<class...> class Vector, class Exec, class Upsweep, class ComputeParents>
int testGlobalExchange(Exec exec, Upsweep upsweep, ComputeParents computeParents, int rank, int numRanks)
{
    using KeyType = unsigned;
    constexpr double theta = 1.1;
    constexpr int reps = 10;

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

    /// p2p setup, once
    Box<double> box(0, 1, BoundaryType::periodic);
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    std::vector<Vec4<double>> centers(hostTree.numNodes);
    geoMacSpheres(std::span<const KeyType>(hostTree.prefixes, hostTree.numNodes), centers.data(),
                  invThetaMinMac(theta), box);
    std::vector<TreeNodeIndex> farOffsets, farNodes;
    computeFarNodes<KeyType, double>(hostTree, leaves, displ, centers, box, farOffsets, farNodes);
    auto   lists     = createSweepLists<KeyType>(hostTree, leaves, displ, farOffsets, farNodes, rank);
    double setupTime = MPI_Wtime() - start;

    /// p2p buffers
    Vector<TreeNodeIndex> sendIdx(lists.sendIdx), recvIdx(lists.recvIdx);
    Vector<Q>             globQ(hostTree.numNodes), sendBuf(lists.sendIdx.size()), recvBuf(lists.recvIdx.size());
    constexpr int         tag = 0;

    /// Allgather setup
    Vector<Q> allLeaves(hostTree.numLeafNodes);
    Vector<Q> allQ(hostTree.numNodes);

    auto download = [](const Vector<Q>& v)
    {
        if constexpr (execution::HaveGpu<Exec>{}) { return toHost(v); }
        else { return v; }
    };

    std::vector<Q>      leafQ(hostTree.numLeafNodes), reference(hostTree.numNodes);
    std::vector<double> allgatherTime(reps);
    std::vector<double> p2pTime(reps);
    int                errors = 0;
    int                p2pErrors = 0;
    for (int rep = 0; rep < reps; ++rep)
    {
        populate(rep, std::span<Q>(leafQ));
        cstone::scatter(hostTree.leafToInternalSpan(), leafQ.data(), reference.data());
        cstone::upsweep(hostTree.levelRangeSpan(), hostTree.childOffsets, reference.data(), SumChildren{});
        Vector<Q> localQ(leafQ.data() + firstLeaf, leafQ.data() + firstLeaf + numLeaves);
        if constexpr (execution::HaveGpu<Exec>{}) { syncGpu(exec); }

        /// Allgather + full upsweep
        MPI_Barrier(MPI_COMM_WORLD);
        start = MPI_Wtime();
        mpiAllgatherGpuDirect(exec, localQ.data(), allLeaves.data(), numLeaves, MPI_COMM_WORLD);
        scatter(exec, gOctree.leafToInternalSpan(), allLeaves.data(), allQ.data());
        upsweep(gOctree.levelRangeSpan(), gOctree.childOffsets, allQ.data());
        if constexpr (execution::HaveGpu<Exec>{}) { syncGpu(exec); }
        allgatherTime[rep] = MPI_Wtime() - start;

        // check all nodes from Allgather
        auto fullQ = download(allQ);
        for (TreeNodeIndex i = 0; i < hostTree.numNodes; ++i)
        {
            if (fullQ[i] != reference[i]) { ++errors; }
        }

        /// distributed p2p upsweep
        MPI_Barrier(MPI_COMM_WORLD);
        start = MPI_Wtime();
        scatter(exec, gOctree.leafToInternalSpan().subspan(firstLeaf, numLeaves), localQ.data(), globQ.data());
        globalSweep(exec, lists, {sendIdx.data(), sendIdx.size()}, {recvIdx.data(), recvIdx.size()},
                    gOctree.childOffsets, std::span<Q>(globQ.data(), globQ.size()), {sendBuf.data(), sendBuf.size()},
                    {recvBuf.data(), recvBuf.size()}, computeParents, tag, MPI_COMM_WORLD);
        p2pTime[rep] = MPI_Wtime() - start;

        // check the far nodes of this rank from the p2p sweep
        auto sweptQ = download(globQ);
        for (TreeNodeIndex i = farOffsets[rank]; i < farOffsets[rank + 1]; ++i)
        {
            if (sweptQ[farNodes[i]] != reference[farNodes[i]]) { ++p2pErrors; }
        }
    }

    // sum of errors, maximum of times over ranks
    MPI_Allreduce(MPI_IN_PLACE, &errors, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &p2pErrors, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &setupTime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    std::vector<double> gatheredTimes(std::size_t(numRanks) * reps), gatheredP2pTimes(std::size_t(numRanks) * reps);
    MPI_Allgather(allgatherTime.data(), reps, MPI_DOUBLE, gatheredTimes.data(), reps, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgather(p2pTime.data(), reps, MPI_DOUBLE, gatheredP2pTimes.data(), reps, MPI_DOUBLE, MPI_COMM_WORLD);

    // max, mean and median over ranks of repetition rep
    auto rankStats = [&](const std::vector<double>& gathered, int rep, double& max, double& mean, double& median)
    {
        std::vector<double> rankTimes(numRanks);
        mean = 0;
        for (int r = 0; r < numRanks; ++r)
        {
            rankTimes[r] = gathered[std::size_t(r) * reps + rep];
            mean += rankTimes[r];
        }
        mean /= numRanks;
        std::sort(rankTimes.begin(), rankTimes.end());
        max    = rankTimes.back();
        median = numRanks % 2 ? rankTimes[numRanks / 2] : (rankTimes[numRanks / 2 - 1] + rankTimes[numRanks / 2]) / 2;
    };

    if (rank == 0)
    {
        std::printf("tree: P %d, %d leaves, %d leaves/rank, Q %zu B\n", numRanks, hostTree.numLeafNodes, numLeaves,
                    sizeof(Q));
        std::printf("p2p setup [ms]: %.3f\n", setupTime * 1e3);
        std::printf("rep | allgather max [ms] | mean [ms] | median [ms] |      p2p max [ms] | mean [ms] | median [ms]\n");
        for (int rep = 0; rep < reps; ++rep)
        {
            double max, mean, median, p2pMax, p2pMean, p2pMedian;
            rankStats(gatheredTimes, rep, max, mean, median);
            rankStats(gatheredP2pTimes, rep, p2pMax, p2pMean, p2pMedian);
            std::printf("%3d | %18.3f | %9.3f | %11.3f | %16.3f | %9.3f | %11.3f\n", rep + 1, max * 1e3, mean * 1e3,
                        median * 1e3, p2pMax * 1e3, p2pMean * 1e3, p2pMedian * 1e3);
        }
        std::printf("allgather upsweep: %s (%d errors)\n", errors ? "FAIL" : "PASS", errors);
        std::printf("p2p sweep: %s (%d errors)\n", p2pErrors ? "FAIL" : "PASS", p2pErrors);
    }
    return errors != 0 || p2pErrors != 0;
}
