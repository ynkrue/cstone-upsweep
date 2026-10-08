# source env.sh — toolchain for upsweep-p2p (Daint)
# Sets CC/CXX and, if not already active, starts uenv prgenv-gnu/24.11:v1 (view default).
# Note: `uenv start` opens a new shell (like `ldcs`); exit it to leave the uenv.
export CC=mpicc CXX=mpicxx
if [[ "$(command -v mpicxx)" != /user-environment/* ]]; then
    uenv image pull prgenv-gnu/24.11:v1 >/dev/null 2>&1
    uenv start prgenv-gnu/24.11:v1 --view=default
fi
