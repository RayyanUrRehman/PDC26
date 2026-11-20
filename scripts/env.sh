# Source this from the other scripts. It prefers a user-local OpenMPI
# build when the system has no mpicc on PATH.
if [ -x "${HOME}/.local/openmpi/bin/mpicc" ]; then
    export PATH="${HOME}/.local/openmpi/bin:${PATH}"
    libdir="${HOME}/.local/openmpi/lib"
    if [ -d "${libdir}" ]; then
        export LD_LIBRARY_PATH="${libdir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
fi
