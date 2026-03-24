"""Analyze autoresearch experiment results.

Usage:
    python autoresearch/analyze.py [results.tsv]
"""

from __future__ import annotations

import sys
from pathlib import Path


def load_results(path: str = "results.tsv") -> list[dict]:
    """Load results.tsv into a list of dicts."""
    rows = []
    with open(path) as f:
        header = f.readline().strip().split("\t")
        for line in f:
            vals = line.strip().split("\t")
            if len(vals) != len(header):
                continue
            row = dict(zip(header, vals))
            # Parse numeric fields
            for key in ["val_kl_divergence", "val_w1"]:
                if key in row:
                    try:
                        row[key] = float(row[key])
                    except ValueError:
                        row[key] = None
            for key in ["epochs", "n_params"]:
                if key in row:
                    try:
                        row[key] = int(row[key])
                    except ValueError:
                        row[key] = None
            rows.append(row)
    return rows


def print_summary(rows: list[dict]) -> None:
    """Print experiment summary."""
    n_total = len(rows)
    n_keep = sum(1 for r in rows if r.get("status") == "keep")
    n_discard = sum(1 for r in rows if r.get("status") == "discard")
    n_crash = sum(1 for r in rows if r.get("status") == "crash")
    n_baseline = sum(1 for r in rows if r.get("status") == "baseline")

    print("=" * 70)
    print("AUTORESEARCH RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total experiments: {n_total}")
    print(f"  Baseline:  {n_baseline}")
    print(f"  Kept:      {n_keep}")
    print(f"  Discarded: {n_discard}")
    print(f"  Crashed:   {n_crash}")
    print(f"  Keep rate: {n_keep / max(n_total - n_baseline, 1) * 100:.1f}%")
    print()

    # Baseline and best
    baseline_rows = [r for r in rows if r.get("status") == "baseline" and r.get("val_kl_divergence") is not None]
    kept_rows = [r for r in rows if r.get("status") == "keep" and r.get("val_kl_divergence") is not None]
    all_valid = [r for r in rows if r.get("val_kl_divergence") is not None]

    if baseline_rows:
        baseline_val = baseline_rows[0]["val_kl_divergence"]
        print(f"Baseline val_kl_divergence: {baseline_val:.6f} (|{abs(baseline_val):.6f}|)")

    if all_valid:
        best = min(all_valid, key=lambda r: abs(r["val_kl_divergence"]))
        print(f"Best val_kl_divergence:     {best['val_kl_divergence']:.6f} (|{abs(best['val_kl_divergence']):.6f}|)  ({best.get('description', 'N/A')})")

        if baseline_rows:
            improvement = abs(baseline_val) - abs(best["val_kl_divergence"])
            pct = improvement / abs(baseline_val) * 100 if abs(baseline_val) > 0 else 0
            print(f"Total improvement:   {improvement:.6f}  ({pct:.1f}%)")

    print()

    # Top experiments (sorted by val_kl_divergence)
    if kept_rows:
        print("TOP KEPT EXPERIMENTS (best first):")
        print("-" * 70)
        kept_sorted = sorted(kept_rows, key=lambda r: abs(r["val_kl_divergence"]))
        for i, r in enumerate(kept_sorted[:15]):
            print(
                f"  {i+1:2d}. val_kl_divergence={r['val_kl_divergence']:.6f} | "
                f"active_dims={r.get('active_dims', '?')} | "
                f"{r.get('description', 'N/A')}"
            )
        print()

    # Progress over time (running minimum)
    if len(all_valid) >= 2:
        print("PROGRESS (running best):")
        print("-" * 70)
        running_min = float("inf")
        for i, r in enumerate(all_valid):
            if abs(r["val_kl_divergence"]) < running_min:
                running_min = abs(r["val_kl_divergence"])
                print(
                    f"  Experiment {i+1}: val_kl_divergence={r['val_kl_divergence']:.6f} ← NEW BEST "
                    f"({r.get('description', 'N/A')})"
                )


def plot_progress(rows: list[dict], save_path: str = "progress.png") -> None:
    """Plot experiment progress chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    valid = [r for r in rows if r.get("val_kl_divergence") is not None]
    if len(valid) < 2:
        print("Not enough data points for a plot")
        return

    x = list(range(1, len(valid) + 1))
    y = [abs(r["val_kl_divergence"]) for r in valid]
    statuses = [r.get("status", "unknown") for r in valid]

    # Running minimum
    running_min = []
    best = float("inf")
    for v in y:
        best = min(best, v)
        running_min.append(best)

    fig, ax = plt.subplots(figsize=(12, 5))

    # Color by status
    colors = {
        "baseline": "black",
        "keep": "green",
        "discard": "red",
        "crash": "orange",
    }
    for i in range(len(x)):
        c = colors.get(statuses[i], "gray")
        marker = "o" if statuses[i] == "keep" else "x" if statuses[i] == "discard" else "s"
        ax.scatter(x[i], y[i], c=c, marker=marker, s=40, zorder=3, alpha=0.7)

    # Running minimum line
    ax.plot(x, running_min, color="blue", linewidth=2, alpha=0.8, label="Running best")

    ax.set_xlabel("Experiment #")
    ax.set_ylabel("|val_kl_divergence|")
    ax.set_title("Autoresearch: Distribution VAE Experiments")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Legend for status markers
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="green", label="Keep", markersize=8),
        Line2D([0], [0], marker="x", color="red", label="Discard", markersize=8),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="orange", label="Crash", markersize=8),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="black", label="Baseline", markersize=8),
        Line2D([0], [0], color="blue", label="Running best", linewidth=2),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Progress chart saved to {save_path}")


if __name__ == "__main__":
    tsv_path = sys.argv[1] if len(sys.argv) > 1 else "results.tsv"
    if not Path(tsv_path).exists():
        print(f"No results file found at {tsv_path}")
        sys.exit(1)

    rows = load_results(tsv_path)
    print_summary(rows)
    plot_progress(rows)
