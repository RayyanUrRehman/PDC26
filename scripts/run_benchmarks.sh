#!/usr/bin/env bash
# Strong-scaling sweep driven by --cores (default: nproc). No core count is compiled in.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck source=env.sh
source "${ROOT}/scripts/env.sh"
# shellcheck source=lib.sh
source "${ROOT}/scripts/lib.sh"

cores=""
seed=42
quick=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cores) cores=$2; shift 2 ;;
        --seed) seed=$2; shift 2 ;;
        --quick) quick=1; shift ;;
        *) echo "unknown argument $1" >&2; exit 2 ;;
    esac
done

if [[ -z "${cores}" ]]; then
    cores=$(nproc)
fi

cd "${ROOT}"
make -s sequential hybrid
rm -rf results/logs
mkdir -p results/logs results/figures

csv="results/timings.csv"
echo "N,P,threads,t_scatter,t_bcast,t_compute,t_gather,t_total,gflops,max_abs_err,checksum,schedule,program,oversubscribe" > "${csv}"

append_csv() {
    local log=$1
    if grep -q '^csv,' "${log}"; then
        grep '^csv,' "${log}" | tail -n 1 | sed 's/^csv,//' >> "${csv}"
    fi
}

run_seq() {
    local n=$1
    local log="results/logs/seq_N${n}.txt"
    local rc=0
    echo "=== sequential N=${n} ==="
    if ./sequential --size "${n}" --seed "${seed}" | tee -a "${log}"; then
        rc=0
    else
        rc=$?
    fi
    append_csv "${log}"
    return "${rc}"
}

run_hyb() {
    local n=$1 p=$2 t=$3 sched=$4 over=$5
    local safe_sched=${sched//,/_}
    local log="results/logs/hyb_N${n}_P${p}_T${t}_${safe_sched}_o${over}.txt"
    local rc=0
    echo "=== hybrid N=${n} P=${p} T=${t} schedule=${sched} over=${over} ==="
    local -a allow=()
    if [[ "${over}" == "1" ]]; then
        allow=(--allow-oversubscribe)
    fi
    if [[ "${sched}" == "runtime" ]]; then
        if OMP_SCHEDULE=static "${ROOT}/scripts/run.sh" --size "${n}" --procs "${p}" --threads "${t}" \
            --seed "${seed}" --schedule "${sched}" "${allow[@]}" | tee -a "${log}"; then
            rc=0
        else
            rc=$?
        fi
    elif "${ROOT}/scripts/run.sh" --size "${n}" --procs "${p}" --threads "${t}" \
        --seed "${seed}" --schedule "${sched}" "${allow[@]}" | tee -a "${log}"; then
        rc=0
    else
        rc=$?
    fi
    append_csv "${log}"
    return "${rc}"
}

full_sweep() {
    local n=$1
    local reps=$2
    local r c p t
    for ((r = 0; r < reps; r++)); do
        local seen=" "
        run_seq "${n}"
        for ((t = 1; t <= cores; t++)); do
            run_hyb "${n}" 1 "${t}" static 0
            seen+=" 1,${t},static "
        done
        for ((p = 1; p <= cores; p++)); do
            if [[ "${seen}" != *" ${p},1,static "* ]]; then
                run_hyb "${n}" "${p}" 1 static 0
                seen+=" ${p},1,static "
            fi
        done
        for ((c = 1; c <= cores; c++)); do
            read -r p t < <(choose_factors "${c}")
            if [[ "${seen}" != *" ${p},${t},static "* ]]; then
                run_hyb "${n}" "${p}" "${t}" static 0
                seen+=" ${p},${t},static "
            fi
        done
    done
}

if ((quick == 1)); then
    full_sweep 256 1
    exit 0
fi

full_sweep 1024 3
full_sweep 2048 1

read -r hp ht < <(choose_factors "${cores}")
run_hyb 1024 "${hp}" $((ht + 1)) static 1

run_seq 4096
run_hyb 4096 1 "${cores}" static 0
if [[ "${hp}" != "1" || "${ht}" != "${cores}" ]]; then
    run_hyb 4096 "${hp}" "${ht}" static 0
fi

for sched in static "static,1" "dynamic,1" "guided,1" runtime; do
    run_hyb 1024 "${hp}" "${ht}" "${sched}" 0
done

echo "=== attempting N=8192 ==="
set +e
run_seq 8192
seq_rc=$?
run_hyb 8192 "${hp}" "${ht}" static 0
hyb_rc=$?
set -e
{
    echo "sequential_exit=${seq_rc}"
    echo "hybrid_exit=${hyb_rc}"
    echo "hybrid_pair=${hp}x${ht}"
} > results/n8192_status.txt

echo "timings written to ${csv}"
