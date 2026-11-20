#!/usr/bin/env bash
# Correctness checks from the project plan. Exits non-zero on VERIFY FAIL.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck source=env.sh
source "${ROOT}/scripts/env.sh"
cd "${ROOT}"
make -s sequential hybrid

checksum_of() {
    grep '^csv,' "$1" | tail -n 1 | awk -F, '{print $12}'
}

pass_of() {
    grep '^VERIFY ' "$1" | tail -n 1
}

run_case() {
    local name=$1
    shift
    local log="results/logs/${name}.txt"
    mkdir -p results/logs
    echo "=== ${name} ==="
    "$@" | tee "${log}"
    if [[ "${name}" == seq_* ]]; then
        return 0
    fi
    local line
    line=$(pass_of "${log}")
    echo "${line}"
    [[ "${line}" == "VERIFY PASS" ]]
}

run_case seq_256 ./sequential --size 256 --seed 42
run_case seq_128 ./sequential --size 128 --seed 42

run_case hyb_128_2x2 "${ROOT}/scripts/run.sh" --size 128 --procs 2 --threads 2 --allow-oversubscribe --verify
run_case hyb_512_2x2 "${ROOT}/scripts/run.sh" --size 512 --procs 2 --threads 2 --allow-oversubscribe --verify
run_case hyb_128_1x1 "${ROOT}/scripts/run.sh" --size 128 --procs 1 --threads 1 --verify
run_case hyb_1024_spot "${ROOT}/scripts/run.sh" --size 1024 --procs 1 --threads 1 --verify
run_case hyb_100_scatterv "${ROOT}/scripts/run.sh" --size 100 --procs 3 --threads 1 --verify

c_seq=$(checksum_of results/logs/seq_128.txt)
c_hyb=$(checksum_of results/logs/hyb_128_1x1.txt)
python3 - << PY
seq = float("${c_seq}")
hyb = float("${c_hyb}")
scale = abs(seq) + abs(hyb) + 1e-12
rel = abs(seq - hyb) / scale
print(f"checksum sequential={seq:.6e} hybrid_1x1={hyb:.6e} rel={rel:.3e}")
if rel > 1e-9:
    raise SystemExit("1x1 hybrid checksum does not match the sequential baseline")
PY

echo "all correctness checks passed"
