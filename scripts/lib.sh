# Factor K as P x T. Closest to a square; ties prefer the smaller P
# so B is replicated fewer times.
choose_factors() {
    local k=$1
    local best_p=1
    local best_t=$k
    local best_diff=$k
    local p t diff
    for ((p = 1; p <= k; p++)); do
        if ((k % p == 0)); then
            t=$((k / p))
            if ((p > t)); then
                diff=$((p - t))
            else
                diff=$((t - p))
            fi
            if ((diff < best_diff || (diff == best_diff && p < best_p))); then
                best_diff=$diff
                best_p=$p
                best_t=$t
            fi
        fi
    done
    echo "${best_p} ${best_t}"
}
