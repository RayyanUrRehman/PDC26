#!/usr/bin/env python3
"""Median timings, five figures, and a markdown summary for the report."""

import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "results" / "timings.csv"
FIG = ROOT / "results" / "figures"
SUMMARY = ROOT / "results" / "summary.md"


def choose_factors(k: int) -> tuple[int, int]:
    best = (1, k, k)
    for p in range(1, k + 1):
        if k % p == 0:
            t = k // p
            diff = abs(p - t)
            if diff < best[2] or (diff == best[2] and p < best[0]):
                best = (p, t, diff)
    return best[0], best[1]


def load_rows() -> list[dict]:
    rows = []
    with CSV_PATH.open(newline="") as handle:
        for rec in csv.DictReader(handle):
            rec["N"] = int(rec["N"])
            rec["P"] = int(rec["P"])
            rec["threads"] = int(rec["threads"])
            rec["oversubscribe"] = int(rec["oversubscribe"])
            for key in (
                "t_scatter",
                "t_bcast",
                "t_compute",
                "t_gather",
                "t_total",
                "gflops",
                "max_abs_err",
                "checksum",
            ):
                rec[key] = float(rec[key])
            rows.append(rec)
    return rows


def median_groups(rows: list[dict]) -> list[dict]:
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for rec in rows:
        key = (
            rec["N"],
            rec["P"],
            rec["threads"],
            rec["schedule"],
            rec["program"],
            rec["oversubscribe"],
        )
        buckets[key].append(rec)
    grouped = []
    for key, items in buckets.items():
        n, p, threads, schedule, program, over = key
        def med(field: str) -> float:
            return statistics.median(item[field] for item in items)

        grouped.append(
            {
                "N": n,
                "P": p,
                "threads": threads,
                "schedule": schedule,
                "program": program,
                "oversubscribe": over,
                "repeats": len(items),
                "t_scatter": med("t_scatter"),
                "t_bcast": med("t_bcast"),
                "t_compute": med("t_compute"),
                "t_gather": med("t_gather"),
                "t_total": med("t_total"),
                "gflops": med("gflops"),
                "max_abs_err": max(item["max_abs_err"] for item in items),
                "cores": p * threads,
            }
        )
    return grouped


def baseline(grouped: list[dict], n: int) -> float | None:
    hits = [
        g["t_total"]
        for g in grouped
        if g["N"] == n and g["program"] == "sequential" and g["P"] == 1 and g["threads"] == 1
    ]
    return hits[0] if hits else None


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    grouped = median_groups(load_rows())
    sizes = sorted({g["N"] for g in grouped})

    def hybrid_static(n: int | None = None) -> list[dict]:
        return [
            g
            for g in grouped
            if g["program"] == "hybrid"
            and g["schedule"] == "static"
            and g["oversubscribe"] == 0
            and (n is None or g["N"] == n)
        ]

    # Time vs threads, one process.
    plt.figure(figsize=(7, 4.5))
    for n in sizes:
        pts = sorted(
            [g for g in hybrid_static(n) if g["P"] == 1],
            key=lambda g: g["threads"],
        )
        if len(pts) >= 2:
            plt.plot(
                [g["threads"] for g in pts],
                [g["t_total"] for g in pts],
                marker="o",
                label=f"N={n}",
            )
    plt.xlabel("OpenMP threads (1 MPI process)")
    plt.ylabel("Total time (s)")
    plt.title("Execution time vs threads")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "time_vs_threads.png", dpi=140)
    plt.close()

    # Time vs processes, one thread.
    plt.figure(figsize=(7, 4.5))
    for n in sizes:
        pts = sorted(
            [g for g in hybrid_static(n) if g["threads"] == 1],
            key=lambda g: g["P"],
        )
        if len(pts) >= 2:
            plt.plot(
                [g["P"] for g in pts],
                [g["t_total"] for g in pts],
                marker="o",
                label=f"N={n}",
            )
    plt.xlabel("MPI processes (1 thread each)")
    plt.ylabel("Total time (s)")
    plt.title("Execution time vs processes")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "time_vs_processes.png", dpi=140)
    plt.close()

    fig_s, ax_s = plt.subplots(figsize=(7, 4.5))
    fig_e, ax_e = plt.subplots(figsize=(7, 4.5))
    for n in sizes:
        base = baseline(grouped, n)
        pts = hybrid_static(n)
        if base is None or not pts:
            continue
        pts = sorted(pts, key=lambda g: (g["cores"], g["P"]))
        cores = [g["cores"] for g in pts]
        speedup = [base / g["t_total"] if g["t_total"] > 0 else 0 for g in pts]
        efficiency = [s / c if c else 0 for s, c in zip(speedup, cores)]
        ax_s.plot(cores, speedup, marker="o", linestyle="none", label=f"N={n}")
        ax_e.plot(cores, efficiency, marker="o", linestyle="none", label=f"N={n}")
    for ax, ylabel, title, name in (
        (ax_s, "Speedup  T(1) / T(P,T)", "Speedup vs cores used", "speedup.png"),
        (ax_e, "Efficiency  speedup / (P x T)", "Efficiency vs cores used", "efficiency.png"),
    ):
        ax.set_xlabel("Cores used (processes x threads)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig_s.tight_layout()
    fig_e.tight_layout()
    fig_s.savefig(FIG / "speedup.png", dpi=140)
    fig_e.savefig(FIG / "efficiency.png", dpi=140)
    plt.close(fig_s)
    plt.close(fig_e)

    # Stacked compute vs communication for the largest size that has several hybrid points.
    breakdown_n = None
    for n in sorted(sizes, reverse=True):
        if len(hybrid_static(n)) >= 2:
            breakdown_n = n
            break
    if breakdown_n is not None:
        pts = sorted(hybrid_static(breakdown_n), key=lambda g: (g["cores"], g["P"], g["threads"]))
        labels = [f"{g['P']}x{g['threads']}" for g in pts]
        compute = [g["t_compute"] for g in pts]
        comm = [g["t_scatter"] + g["t_bcast"] + g["t_gather"] for g in pts]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(labels, compute, label="compute")
        ax.bar(labels, comm, bottom=compute, label="communication")
        ax.set_xlabel("Processes x threads")
        ax.set_ylabel("Time (s)")
        ax.set_title(f"Compute and communication, N={breakdown_n}")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG / "time_breakdown.png", dpi=140)
        plt.close(fig)

    lines = ["# Timing summary", ""]
    lines.append("Medians. Speedup uses the sequential binary for the same N.")
    lines.append("")
    lines.append("| N | P | T | schedule | over | cores | repeats | t_compute (s) | t_comm (s) | t_total (s) | GFLOP/s | speedup | efficiency | max abs err |")
    lines.append("|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for g in sorted(grouped, key=lambda item: (item["N"], item["program"], item["oversubscribe"], item["cores"], item["P"], item["schedule"])):
        base = baseline(grouped, g["N"])
        comm = g["t_scatter"] + g["t_bcast"] + g["t_gather"]
        if g["program"] == "sequential" or base is None or g["t_total"] <= 0:
            speedup = 1.0 if g["program"] == "sequential" else float("nan")
        else:
            speedup = base / g["t_total"]
        cores = g["cores"] if g["program"] == "hybrid" else 1
        efficiency = speedup / cores if cores else float("nan")
        lines.append(
            f"| {g['N']} | {g['P']} | {g['threads']} | {g['schedule']} | {g['oversubscribe']} | {cores} | {g['repeats']} | "
            f"{g['t_compute']:.4f} | {comm:.4f} | {g['t_total']:.4f} | {g['gflops']:.3f} | "
            f"{speedup:.3f} | {efficiency:.3f} | {g['max_abs_err']:.3e} |"
        )
        g["speedup"] = speedup
        g["efficiency"] = efficiency
        g["t_comm"] = comm

    lines.append("")
    lines.append("## Amdahl estimate")
    lines.append("")
    lines.append("Serial fraction fitted as s = (C * T(C) - T(1)) / (T(1) * (C - 1)), where C = P x T and T is total time.")
    lines.append("Negative s means the parallel total time beat a pure 1/C split (noise or a tiny problem).")
    lines.append("")
    for g in grouped:
        if g["program"] != "hybrid" or g["schedule"] != "static" or g["oversubscribe"] != 0:
            continue
        if g["cores"] < 2:
            continue
        base = baseline(grouped, g["N"])
        if base is None:
            continue
        s = (g["cores"] * g["t_total"] - base) / (base * (g["cores"] - 1))
        comm_share = g["t_comm"] / g["t_total"] if g["t_total"] else 0
        if s < 0:
            bound = float("inf")
        else:
            bound = 1.0 / (s + (1.0 - s) / g["cores"])
        lines.append(
            f"- N={g['N']} {g['P']}x{g['threads']}: s={s:.3f}, communication share={comm_share:.3f}, "
            f"Amdahl bound={bound:.2f}, measured speedup={g['speedup']:.2f}"
        )
    lines.append("")
    lines.append("Gustafson speedup is P - s(P-1). This study keeps N fixed, so it is strong scaling and Amdahl is the model that applies. Gustafson would require growing N with the core count.")
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {FIG} and {SUMMARY}")


if __name__ == "__main__":
    main()
