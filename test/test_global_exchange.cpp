/*! @file
 * @brief test globalUpsweep against a single-rank upsweep of the global tree
 *
 * Usage: test_global_exchange [L=3] [theta=0.5] [n=32] [reps=10]
 */

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <tuple>
#include <vector>

#include <mpi.h>

#include "cstone/domain/domaindecomp.hpp"
#include "cstone/focus/source_center.hpp"
#include "cstone/sfc/sfc.hpp"
#include "cstone/traversal/macs.hpp"
#include "cstone/tree/cs_util.hpp"
#include "cstone/tree/csarray.hpp"

#include "upsweep_p2p.hpp"

using namespace upsweep_p2p;

int main(int argc, char** argv)
{
    using KeyType = uint64_t;
    using T       = double;

    MPI_Init(&argc, &argv);
    int myRank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    int      numLevels  = argc > 1 ? std::atoi(argv[1]) : 3;    // L, 8^L global leaves
    float    theta      = argc > 2 ? std::atof(argv[2]) : 0.5f; // MAC opening angle
    unsigned numPerCell = argc > 3 ? std::atoi(argv[3]) : 32;   // particles per global leaf
    int      numReps    = argc > 4 ? std::atoi(argv[4]) : 10;   // timed calls after one warm-up call

    //! regular global tree with L levels
    std::vector<KeyType> gLeaves = cstone::makeNLevelGrid<KeyType>(numLevels);
    gLeaves.push_back(cstone::nodeRange<KeyType>(0));
    std::vector<unsigned> gCounts(cstone::nNodes(gLeaves), 32);

    //! global tree, distributed across ranks
    cstone::OctreeData<KeyType, cstone::execution::Cpu> globalTree;
    globalTree.resize(cstone::nNodes(gLeaves));
    cstone::updateInternalTree<KeyType>(gLeaves, globalTree.data());
    auto gOctree   = globalTree.cdata();
    gOctree.leaves = gLeaves.data();

    auto                           assignment   = cstone::makeSfcAssignment(numRanks, gCounts, gLeaves.data());
    std::span<const TreeNodeIndex> globDispl    = assignment.treeOffsetsConst();
    TreeNodeIndex                  globNumNodes = globDispl[myRank + 1] - globDispl[myRank];

    //! leaf quantity with distinct values
    std::vector<unsigned> gLeafQ(cstone::nNodes(gLeaves));
    std::iota(gLeafQ.begin(), gLeafQ.end(), 1u);

    auto upsweep = [](auto levelRange, auto childOffsets, auto Q)
    { cstone::upsweep(levelRange, childOffsets, Q, cstone::NodeCount<unsigned>{}); };

    //! reference upsweep of all global leaves
    std::vector<unsigned> globQRef(gOctree.numNodes);
    cstone::scatter(gOctree.leafToInternalSpan(), gLeafQ.data(), globQRef.data());
    upsweep(gOctree.levelRangeSpan(), gOctree.childOffsets, globQRef.data());

    std::span<const unsigned> gLeafQLoc(gLeafQ.data() + globDispl[myRank], globNumNodes);
    std::vector<unsigned>     globQ(gOctree.numNodes);
    cstone::scatter(gOctree.leafToInternalSpan().subspan(globDispl[myRank], globNumNodes), gLeafQLoc.data(),
                    globQ.data());

    //! p2p, everything per call as in FocusedOctree::globalExchange; call 0 is the warm-up
    cstone::Box<T>               box(0, 1);
    std::vector<cstone::Vec4<T>> centers(gOctree.numNodes);
    std::vector<uint8_t>         flags(gOctree.numNodes);
    SweepLists                   upLists, upFlagged, downFlagged;
    double                       times[4] = {0, 0, 0, 0}; // flags, lists, sweep, allgather [s per call]
    double                       tUpLists = 0;            // TEMPORARY profiling: computeUpsweepLists
    for (int r = 0; r <= numReps; ++r)
    {
        //! flags: global nodes that fail the minMAC against one of my leaves (stand-in for the LET flags)
        double t0 = MPI_Wtime();
        cstone::geoMacSpheres(std::span<const KeyType>(gOctree.prefixes, gOctree.numNodes), centers.data(),
                              cstone::invThetaMinMac(theta), box);
        std::fill(flags.begin(), flags.end(), 0);
        cstone::markMacs(gOctree.prefixes, gOctree.childOffsets, gOctree.parents, centers.data(), box,
                         gLeaves.data() + globDispl[myRank], globNumNodes, false, flags.data());
        MPI_Barrier(MPI_COMM_WORLD); // the stand-in flags are unbalanced, lists must not measure waiting for them
        double t1 = MPI_Wtime();

        if (r == 1) { profile = {}; } // TEMPORARY profiling, call 0 is the warm-up
        upLists                          = computeUpsweepLists(gOctree, assignment, myRank);
        double tUp                       = MPI_Wtime();
        std::tie(upFlagged, downFlagged) = computeExchangeLists(gOctree, upLists, flags, assignment, MPI_COMM_WORLD);
        double t2                        = MPI_Wtime();
        if (r > 0) { tUpLists += (tUp - t1) / numReps; }

        globalUpsweep(gOctree, upLists, upFlagged, downFlagged, assignment, std::span<unsigned>(globQ), MPI_COMM_WORLD,
                      upsweep);
        double t3 = MPI_Wtime();

        if (r == 0) { continue; }
        times[0] += (t1 - t0) / numReps;
        times[1] += (t2 - t1) / numReps;
        times[2] += (t3 - t2) / numReps;
    }

    //! comparison: Allgatherv of the global leaves + upsweep of the whole tree, as in FocusedOctree::globalExchange
    std::vector<int> globCounts(numRanks);
    for (int r = 0; r < numRanks; ++r)
    {
        globCounts[r] = globDispl[r + 1] - globDispl[r];
    }
    std::vector<unsigned> gLeafQAll(gOctree.numLeafNodes), globQAll(gOctree.numNodes);
    for (int r = 0; r <= numReps; ++r)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        mpiAllgatherv(gLeafQLoc.data(), globNumNodes, gLeafQAll.data(), globCounts.data(), globDispl.data(),
                      MPI_COMM_WORLD);
        cstone::scatter(gOctree.leafToInternalSpan(), gLeafQAll.data(), globQAll.data());
        upsweep(gOctree.levelRangeSpan(), gOctree.childOffsets, globQAll.data());
        if (r > 0) { times[3] += (MPI_Wtime() - t0) / numReps; }
    }

    //! check: every node I own and every child of a flagged node equals the single-rank reference
    auto parent    = [&](TreeNodeIndex c) { return gOctree.parents[(c - 1) / 8]; };
    int  numErrors = 0;
    for (TreeNodeIndex i = 0; i < gOctree.numNodes; ++i)
    {
        bool held = nodeOwner(gOctree, assignment, i) == myRank || (i > 0 && flags[parent(i)]);
        if (held && globQ[i] != globQRef[i]) { ++numErrors; }
    }
    MPI_Allreduce(MPI_IN_PLACE, &numErrors, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    //! times in ms per call, max over ranks
    MPI_Allreduce(MPI_IN_PLACE, times, 4, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (myRank == 0)
    {
        std::printf("upsweep: %s (%d errors)\n", numErrors == 0 ? "PASS" : "FAIL", numErrors);
        std::printf("time [ms] L %d P %d: p2p %.3f (flags %.3f | lists %.3f | sweep %.3f) || allgather %.3f\n",
                    numLevels, numRanks, 1e3 * (times[0] + times[1] + times[2]), 1e3 * times[0], 1e3 * times[1],
                    1e3 * times[2], 1e3 * times[3]);
    }

    //! TEMPORARY profiling, ms per call, each value max over ranks
    double prof[11] = {
        tUpLists,     profile.reqStart,  profile.reqUp,       profile.reqUpWait, profile.reqDown, profile.reqDownWait,
        profile.upEx, profile.upCompute, profile.upFlaggedEx, profile.downEx,    profile.exWait};
    for (int k = 1; k < 11; ++k)
    {
        prof[k] /= numReps;
    }
    MPI_Allreduce(MPI_IN_PLACE, prof, 11, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (myRank == 0)
    {
        std::printf("profile [ms]: lists (upsweepLists %.3f | start %.3f | up %.3f wait %.3f | down %.3f wait %.3f) "
                    "sweep (up ex %.3f | compute %.3f | upFlagged ex %.3f | down ex %.3f | ex wait %.3f)\n",
                    1e3 * prof[0], 1e3 * prof[1], 1e3 * prof[2], 1e3 * prof[3], 1e3 * prof[4], 1e3 * prof[5],
                    1e3 * prof[6], 1e3 * prof[7], 1e3 * prof[8], 1e3 * prof[9], 1e3 * prof[10]);
    }

    MPI_Finalize();
    return numErrors != 0;
}
