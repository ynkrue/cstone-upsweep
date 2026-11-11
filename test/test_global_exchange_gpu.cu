#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/cuda/stream_holder.cuh"
#include "global_exchange.hpp"

namespace
{
__global__ void sumParents(TreeNodeIndex first, TreeNodeIndex last, const TreeNodeIndex* children, Q* q)
{
    TreeNodeIndex i = first + blockIdx.x * blockDim.x + threadIdx.x;
    if (i < last && children[i]) { q[i] = SumChildren{}(i, children[i], q); }
}
} // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    MPI_Comm shared;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shared);
    int localRank, devices;
    MPI_Comm_rank(shared, &localRank);
    MPI_Comm_free(&shared);
    checkGpuErrors(cudaGetDeviceCount(&devices));
    if (!devices) { MPI_Abort(MPI_COMM_WORLD, 1); }
    checkGpuErrors(cudaSetDevice(localRank % devices));
    gpuAwareMpiEnabled = true;
    int status;
    {
        StreamHolder stream;
        auto         exec    = stream.exec();
        auto         upsweep = [&](std::span<const TreeNodeIndex> ranges, const TreeNodeIndex* children, Q* q)
        {
            for (int l = int(ranges.size()) - 2; l >= 0; --l)
            {
                int n = ranges[l + 1] - ranges[l];
                if (n > 0)
                {
                    sumParents<<<(n + 255) / 256, 256, 0, exec>>>(ranges[l], ranges[l + 1], children, q);
                    checkGpuErrors(cudaGetLastError());
                }
            }
        };
        auto computeParents = [&](std::span<const TreeNodeIndex> nodes, const TreeNodeIndex* children, Q* q)
        {
            for (TreeNodeIndex n : nodes)
            {
                sumParents<<<1, 1, 0, exec>>>(n, n + 1, children, q);
                checkGpuErrors(cudaGetLastError());
            }
        };
        status = testGlobalExchange<DeviceVector>(exec, upsweep, computeParents, rank, numRanks);
    }
    MPI_Finalize();
    return status;
}
