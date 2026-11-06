# Standalone hierarchical exchange

We are trying to replace allgather with a custom hierarchical point-to-point algorithm.
```sh
cmake -S . -B build -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build -j 8

OMP_NUM_THREADS=1 srun -n 8 build/test_global_exchange
OMP_NUM_THREADS=1 MPICH_GPU_SUPPORT_ENABLED=1 srun -n 4 --gpus-per-task=1 build/test_global_exchange_gpu
```