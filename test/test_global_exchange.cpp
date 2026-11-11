#include "global_exchange.hpp"

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    auto upsweep = [](std::span<const TreeNodeIndex> ranges, const TreeNodeIndex* children, Q* q)
    { cstone::upsweep(ranges, children, q, SumChildren{}); };
    auto computeParents = [](std::span<const TreeNodeIndex> nodes, const TreeNodeIndex* children, Q* q)
    {
        for (TreeNodeIndex n : nodes)
        {
            q[n] = SumChildren{}(n, children[n], q);
        }
    };
    int status = testGlobalExchange<std::vector>(execution::cpu, upsweep, computeParents, rank, numRanks);
    MPI_Finalize();
    return status;
}
