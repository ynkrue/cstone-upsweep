# source env.sh — toolchain for upsweep-p2p (Kez)
module use /scratch/yrfenach/.kez/modulefiles
module load gcc-13.4.0 openmpi-5.0.10-gcc-13.4.0
export CC=mpicc CXX=mpicxx
