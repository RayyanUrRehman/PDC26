#!/usr/bin/env bash
# One hybrid run. --cores K uses exactly K cores. See README.md.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck source=env.sh
source "${ROOT}/scripts/env.sh"
# shellcheck source=lib.sh
source "${ROOT}/scripts/lib.sh"

size=1024
seed=42
schedule=static
trace=0
verify=0
procs=""
threads=""
cores=""
allow=0

usage() {
    echo "Usage: $0 [--cores K | --procs P --threads T] [--size N] [--seed S]" >&2
    echo "          [--schedule SPEC] [--trace] [--verify] [--allow-oversubscribe]" >&2
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --size) size=$2; shift 2 ;;
        --seed) seed=$2; shift 2 ;;
        --schedule) schedule=$2; shift 2 ;;
        --procs) procs=$2; shift 2 ;;
        --threads) threads=$2; shift 2 ;;
        --cores) cores=$2; shift 2 ;;
        --allow-oversubscribe) allow=1; shift ;;
        --trace) trace=1; shift ;;
        --verify) verify=1; shift ;;
        --help|-h) usage ;;
        *) echo "unknown argument $1" >&2; usage ;;
    esac
done

host_cores=$(nproc)
if [[ -z "${cores}" && -z "${procs}" && -z "${threads}" ]]; then
    cores=${host_cores}
fi

if [[ -n "${cores}" && -z "${procs}" && -z "${threads}" ]]; then
    read -r procs threads < <(choose_factors "${cores}")
elif [[ -n "${cores}" && -n "${procs}" && -z "${threads}" ]]; then
    if ((cores % procs != 0)); then
        echo "--cores ${cores} is not divisible by --procs ${procs}" >&2
        exit 2
    fi
    threads=$((cores / procs))
elif [[ -n "${cores}" && -z "${procs}" && -n "${threads}" ]]; then
    if ((cores % threads != 0)); then
        echo "--cores ${cores} is not divisible by --threads ${threads}" >&2
        exit 2
    fi
    procs=$((cores / threads))
elif [[ -z "${procs}" && -n "${threads}" ]]; then
    procs=1
elif [[ -n "${procs}" && -z "${threads}" ]]; then
    threads=1
fi

if [[ -n "${cores}" && $((procs * threads)) -ne ${cores} ]]; then
    echo "procs x threads (${procs} x ${threads}) must equal --cores ${cores}" >&2
    exit 2
fi

used=$((procs * threads))
over=0
if ((used > host_cores)); then
    if ((allow == 0)); then
        echo "requested ${used} cores but nproc is ${host_cores}; pass --allow-oversubscribe" >&2
        exit 2
    fi
    over=1
fi

echo "using ${used} cores as ${procs} processes x ${threads} threads (host nproc=${host_cores})"

cd "${ROOT}"
make -s hybrid

extra=()
if ((trace == 1)); then
    extra+=(--trace)
fi
if ((verify == 1)); then
    extra+=(--verify)
fi

# core:PE=T is rejected: a core has one cpu, so it cannot host T>1 threads.
# socket:PE=T gives each rank T cores on the socket. Oversubscribe cannot pin that way.
if ((over == 1)); then
    map_args=(--bind-to none --oversubscribe)
else
    map_args=(--map-by "socket:PE=${threads}" --bind-to core)
fi

omp_schedule="${OMP_SCHEDULE:-static}"
mpirun "${map_args[@]}" -np "${procs}" \
    -x OMP_NUM_THREADS="${threads}" \
    -x OMP_PROC_BIND=close \
    -x OMP_PLACES=cores \
    -x OMP_SCHEDULE="${omp_schedule}" \
    ./hybrid --size "${size}" --threads "${threads}" --seed "${seed}" \
    --schedule "${schedule}" --oversubscribe "${over}" "${extra[@]}"
