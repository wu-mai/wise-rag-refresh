#!/usr/bin/env python3
"""Plot the label-efficiency curve and save as PDF + PNG.

Reads: results/label_efficiency.json
Writes: figures/fig_label_efficiency.{pdf,png}
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_PATH = PROJECT_ROOT / "results" / "label_efficiency.json"
FIG_PDF = PROJECT_ROOT / "figures" / "fig_label_efficiency.pdf"
FIG_PNG = PROJECT_ROOT / "figures" / "fig_label_efficiency.png"


def main():
    with open(RESULTS_PATH) as f:
        r = json.load(f)

    K_VALUES = r["k_values"]
    SEEDS = r["seeds"]
    summary = r["summary"]
    zs = r["zero_shot"]

    means = [summary[str(k)]["mean_f1"] for k in K_VALUES]
    stds = [summary[str(k)]["std_f1"] for k in K_VALUES]
    zs_f1 = zs["f1"]

    fig, ax = plt.subplots(figsize=(5.5, 3.6))

    # Main curve: F1 vs k with 3-seed std error bars
    ax.errorbar(K_VALUES, means, yerr=stds,
                marker="o", linewidth=2, capsize=4, color="#1976D2",
                markerfacecolor="white", markeredgewidth=2,
                label="Adapted (this paper, mean ± std of 3 seeds)")

    # Zero-shot baseline as horizontal line
    ax.axhline(zs_f1, color="#FF9800", linestyle="--", linewidth=1.8,
               label=f"Zero-shot baseline (F1 = {zs_f1:.3f})")

    # In-paper full-adaptation reference (RQ3 in-sample on 60 = 0.685) — distinct
    # from the entertainment hold-out, included only as orientation for the reader.
    ax.axhline(0.685, color="#9E9E9E", linestyle=":", linewidth=1.5,
               label="In-sample full-adaptation (0.685, RQ3 reference)")

    ax.set_xscale("log", base=2)
    ax.set_xticks(K_VALUES)
    ax.set_xticklabels([str(k) for k in K_VALUES])
    ax.set_xlabel("Number of TimeVarying target labels $k$ (log scale)", fontsize=10)
    ax.set_ylabel("Hold-out F1 (entertainment, $n{=}11$)", fontsize=10)
    ax.set_title("Label-efficiency of Fact-Type Router adaptation", fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)

    # Annotate the k=49 point with its value
    ax.annotate(f"k=49\nF1={means[-1]:.3f}",
                xy=(K_VALUES[-1], means[-1]),
                xytext=(8, 12), textcoords="offset points",
                fontsize=8, color="#1976D2",
                arrowprops=dict(arrowstyle="->", color="#1976D2", lw=0.8))

    plt.tight_layout()
    FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_PDF, dpi=200, bbox_inches="tight")
    plt.savefig(FIG_PNG, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIG_PDF} and {FIG_PNG}")

    # Also print a compact summary
    print("\n--- Label-efficiency summary ---")
    print(f"Zero-shot (entertainment hold-out): F1 = {zs_f1:.3f}")
    for k, m, s in zip(K_VALUES, means, stds):
        print(f"  k={k:>2}: mean F1 = {m:.3f} ± {s:.3f}")


if __name__ == "__main__":
    main()