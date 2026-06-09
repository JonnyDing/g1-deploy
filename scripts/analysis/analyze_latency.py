#!/usr/bin/env python3
"""Analyze a latency trace CSV produced by ``LatencyTraceRecorder``.

Usage::

    python scripts/analysis/analyze_latency.py latency_trace.csv
    python scripts/analysis/analyze_latency.py latency_trace.csv --plot
    python scripts/analysis/analyze_latency.py latency_trace.csv --new-only

Columns (per-step):
    step           – policy step index
    t_frame_arrival – pico_bridge receive_time_s (T0)
    t_retarget_done – GMR retarget + timeline append complete (T1)
    t_ref_sampled   – reference window sampled (T2)
    t_obs_built     – 166D observation built (T3)
    t_action_done   – ONNX inference + get_target_dof_pos (T4)
    t_cmd_sent      – SDK send_positions() (T5)
    frame_seq       – pico_bridge frame sequence number
    is_new_frame    – bool: whether this step received a new input frame

Derived latencies (all in milliseconds):
    retarget_ms      = T1 - T0
    ref_processing_ms = T2 - T1
    obs_build_ms     = T3 - T2
    onnx_infer_ms    = T4 - T3
    safety_send_ms   = T5 - T4
    host_total_ms    = T5 - T0
    compute_pipe_ms  = T5 - T1   (host pipeline excluding network)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def load_trace(csv_path: str) -> dict[str, np.ndarray]:
    """Load latency trace CSV, returning a dict of column arrays."""
    data = np.genfromtxt(
        csv_path,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    if data is None or len(data) == 0:
        raise ValueError(f"No data found in {csv_path}")
    return {name: data[name] for name in data.dtype.names}


def compute_latency_stats(values_ms: np.ndarray, label: str) -> dict[str, float]:
    """Compute summary statistics for a latency array (in ms)."""
    finite = values_ms[np.isfinite(values_ms)]
    if len(finite) == 0:
        return {"label": label, "count": 0}
    return {
        "label": label,
        "count": len(finite),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(np.max(finite)),
        "min": float(np.min(finite)),
    }


def print_stats_table(all_stats: list[dict], *, unit: str = "ms") -> None:
    """Print a formatted statistics table."""
    header = f"{'Stage':<28} {'Count':>7} {'Mean':>8} {'Std':>8} {'P50':>8} {'P95':>8} {'P99':>8} {'Max':>8} {'Min':>8}"
    print(header)
    print("-" * len(header))
    for s in all_stats:
        if s["count"] == 0:
            print(f"{s['label']:<28} {'—':>7}")
            continue
        print(
            f"{s['label']:<28} {s['count']:>7d} "
            f"{s['mean']:>8.3f} {s['std']:>8.3f} {s['p50']:>8.3f} "
            f"{s['p95']:>8.3f} {s['p99']:>8.3f} {s['max']:>8.3f} {s['min']:>8.3f}"
        )


def analyze(csv_path: str, *, new_only: bool = False, plot: bool = False) -> None:
    """Load, analyze, and optionally plot a latency trace."""
    trace = load_trace(csv_path)

    t0 = np.asarray(trace["t_frame_arrival"], dtype=np.float64)
    t1 = np.asarray(trace["t_retarget_done"], dtype=np.float64)
    t2 = np.asarray(trace["t_ref_sampled"], dtype=np.float64)
    t3 = np.asarray(trace["t_obs_built"], dtype=np.float64)
    t4 = np.asarray(trace["t_action_done"], dtype=np.float64)
    t5 = np.asarray(trace["t_cmd_sent"], dtype=np.float64)
    is_new = np.asarray(trace["is_new_frame"], dtype=bool)

    if new_only:
        mask = is_new
        print(f"Filtering to {np.sum(mask)} new-frame steps (out of {len(t0)} total)\n")
    else:
        mask = np.ones(len(t0), dtype=bool)

    def _ms(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (b[mask] - a[mask]) * 1000.0

    stages = [
        ("retarget (T1-T0)", _ms(t0, t1)),
        ("ref_processing (T2-T1)", _ms(t1, t2)),
        ("obs_build (T3-T2)", _ms(t2, t3)),
        ("onnx_infer (T4-T3)", _ms(t3, t4)),
        ("safety_send (T5-T4)", _ms(t4, t5)),
        ("host_total (T5-T0)", _ms(t0, t5)),
        ("compute_pipe (T5-T1)", _ms(t1, t5)),
    ]

    all_stats = [compute_latency_stats(values, label) for label, values in stages]

    # Print frame-level info
    total_steps = len(t0)
    new_frame_steps = int(np.sum(is_new))
    print(f"Trace: {csv_path}")
    print(f"Total policy steps: {total_steps}")
    print(f"New-frame steps:    {new_frame_steps}")
    print(f"Input FPS (est):    {new_frame_steps / max((t0[-1] - t0[0]), 1e-6):.1f}")
    print()

    print_stats_table(all_stats)
    print()

    # Per-step cost breakdown (pie-like summary)
    host_total_ms = _ms(t0, t5)
    finite_total = host_total_ms[np.isfinite(host_total_ms)]
    if len(finite_total) > 0:
        median_total = float(np.median(finite_total))
        print(f"Median host end-to-end latency: {median_total:.2f} ms")
        print()

        retarget_pct = np.median(_ms(t0, t1)) / median_total * 100 if median_total > 0 else 0
        ref_pct = np.median(_ms(t1, t2)) / median_total * 100 if median_total > 0 else 0
        obs_pct = np.median(_ms(t2, t3)) / median_total * 100 if median_total > 0 else 0
        onnx_pct = np.median(_ms(t3, t4)) / median_total * 100 if median_total > 0 else 0
        safety_pct = np.median(_ms(t4, t5)) / median_total * 100 if median_total > 0 else 0

        print("Median latency breakdown (% of host_total):")
        print(f"  retarget:       {retarget_pct:.1f}%")
        print(f"  ref_processing: {ref_pct:.1f}%")
        print(f"  obs_build:      {obs_pct:.1f}%")
        print(f"  onnx_infer:     {onnx_pct:.1f}%")
        print(f"  safety_send:    {safety_pct:.1f}%")

    if plot:
        _plot_timeline(trace, mask)


def _plot_timeline(trace: dict[str, np.ndarray], mask: np.ndarray) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[plot] matplotlib not available; install with: pip install matplotlib")
        return

    t0 = np.asarray(trace["t_frame_arrival"], dtype=np.float64)[mask]
    t1 = np.asarray(trace["t_retarget_done"], dtype=np.float64)[mask]
    t2 = np.asarray(trace["t_ref_sampled"], dtype=np.float64)[mask]
    t3 = np.asarray(trace["t_obs_built"], dtype=np.float64)[mask]
    t4 = np.asarray(trace["t_action_done"], dtype=np.float64)[mask]
    t5 = np.asarray(trace["t_cmd_sent"], dtype=np.float64)[mask]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Top: cumulative latency stack
    base = t0
    axes[0].fill_between(range(len(t0)), 0, (t1 - base) * 1000, alpha=0.7, label="retarget (T1-T0)")
    bottom = t1
    axes[0].fill_between(range(len(t0)), (bottom - base) * 1000, (t2 - base) * 1000, alpha=0.7, label="ref_processing (T2-T1)")
    bottom = t2
    axes[0].fill_between(range(len(t0)), (bottom - base) * 1000, (t3 - base) * 1000, alpha=0.7, label="obs_build (T3-T2)")
    bottom = t3
    axes[0].fill_between(range(len(t0)), (bottom - base) * 1000, (t4 - base) * 1000, alpha=0.7, label="onnx_infer (T4-T3)")
    bottom = t4
    axes[0].fill_between(range(len(t0)), (bottom - base) * 1000, (t5 - base) * 1000, alpha=0.7, label="safety_send (T5-T4)")
    axes[0].set_ylabel("Latency (ms)")
    axes[0].set_title("Per-step host pipeline latency breakdown")
    axes[0].legend(loc="upper right", fontsize=8)

    # Bottom: end-to-end host latency
    host_total = (t5 - t0) * 1000
    axes[1].plot(host_total, linewidth=0.8, alpha=0.7)
    axes[1].axhline(np.median(host_total), color="red", linestyle="--", linewidth=0.8, label=f"median = {np.median(host_total):.2f} ms")
    axes[1].set_xlabel("Policy step")
    axes[1].set_ylabel("Host total (ms)")
    axes[1].set_title("Host end-to-end latency (T5-T0)")
    axes[1].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze sim2real latency trace CSV")
    parser.add_argument("csv", type=str, help="Path to latency_trace CSV file")
    parser.add_argument("--new-only", action="store_true", help="Only analyze steps with new input frames")
    parser.add_argument("--plot", action="store_true", help="Show matplotlib timeline plots")
    args = parser.parse_args()

    if not Path(args.csv).is_file():
        print(f"ERROR: file not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    analyze(args.csv, new_only=args.new_only, plot=args.plot)


if __name__ == "__main__":
    main()
