"""
utils/visualization.py
======================
All matplotlib plots for the self-pruning experiment.

Generates:
  1. gate_histograms.png          — side-by-side gate distributions per λ
  2. lambda_tradeoff.png          — λ vs Accuracy & Sparsity (dual-axis)
  3. training_curves.png          — accuracy + sparsity vs epoch
  4. accuracy_vs_sparsity.png     — scatter plot
  5. gate_buckets.png             — bar chart of gate activity buckets
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe on Mac/Linux/cloud
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# ── Dark theme ────────────────────────────────────────────────────────────────
BG    = "#0f1117"
PANEL = "#1a1d27"
TEXT  = "#d4d8e8"
GRID  = "#2a2d3e"
COLS  = ["#4f8ef7", "#f7b84f", "#e05c5c"]


def _style(fig, *axes):
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)
        ax.tick_params(colors=TEXT)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        ax.grid(True, color=GRID, lw=0.5, alpha=0.8)


# ── 1. Gate histograms ────────────────────────────────────────────────────────

def plot_gate_histograms(results: List[Dict[str, Any]], out_dir: Path) -> None:
    """
    Side-by-side histogram of gate values for each λ.
    A successful pruning shows a spike near 0 AND a cluster near 1
    (bimodal distribution: pruned gates vs active gates).
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5), facecolor=BG)
    if n == 1:
        axes = [axes]
    fig.suptitle("Gate Value Distributions — Self-Pruning Network",
                 color=TEXT, fontsize=13, y=1.02)

    for ax, r, col in zip(axes, results, COLS):
        g   = r["gates"]
        sp  = (g < 0.01).mean() * 100
        ax.hist(g, bins=100, range=(0, 1), color=col, alpha=0.85, edgecolor="none")
        ax.axvline(0.01, color="#e05c5c", lw=1.6, ls="--", label="threshold=0.01")
        ax.set_title(f"λ = {r['lam']}\nSparsity = {sp:.1f}%", pad=8)
        ax.set_xlabel("gate value  σ(gate_score)")
        ax.set_ylabel("count" if ax is axes[0] else "")
        ax.legend(fontsize=7.5, framealpha=0.3)
        _style(fig, ax)
        ax.text(0.97, 0.97,
                f"pruned: {int((g<0.01).sum()):,}\ntotal:  {len(g):,}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color=TEXT,
                bbox=dict(fc=PANEL, ec=GRID, alpha=0.85, boxstyle="round,pad=0.35"))

    fig.tight_layout()
    p = out_dir / "gate_histograms.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    logger.info(f"  ✓ {p}")


# ── 2. Lambda trade-off ───────────────────────────────────────────────────────

def plot_lambda_tradeoff(results: List[Dict[str, Any]], out_dir: Path) -> None:
    """Dual-axis line plot: λ (log scale) vs Accuracy and Sparsity."""
    lams = [r["lam"] for r in results]
    accs = [r["accuracy"] * 100 for r in results]
    sprs = [r["sparsity"] * 100 for r in results]

    fig, ax1 = plt.subplots(figsize=(7, 4.5), facecolor=BG)
    ax2 = ax1.twinx()

    l1, = ax1.plot(lams, accs, color="#4f8ef7", lw=2.4, marker="o", ms=9,
                   label="Test Accuracy (%)")
    l2, = ax2.plot(lams, sprs, color="#e05c5c", lw=2.4, marker="s", ms=9,
                   ls="--", label="Sparsity (%)")

    for lam, a, s in zip(lams, accs, sprs):
        ax1.annotate(f"{a:.1f}%", (lam, a), textcoords="offset points",
                     xytext=(0, 11), ha="center", fontsize=9,
                     color="#4f8ef7", fontweight="bold")
        ax2.annotate(f"{s:.1f}%", (lam, s), textcoords="offset points",
                     xytext=(0, -16), ha="center", fontsize=9,
                     color="#e05c5c", fontweight="bold")

    ax1.set_xscale("log")
    ax1.set_xlabel("λ  (log scale)")
    ax1.set_ylabel("Test Accuracy (%)", color="#4f8ef7")
    ax2.set_ylabel("Sparsity (%)",      color="#e05c5c")
    ax1.tick_params(axis="y", labelcolor="#4f8ef7")
    ax2.tick_params(axis="y", labelcolor="#e05c5c")
    ax1.set_title("Accuracy & Sparsity Trade-off vs λ", color=TEXT, pad=10)
    _style(fig, ax1); ax2.set_facecolor(PANEL)
    ax1.legend([l1, l2], [l.get_label() for l in [l1, l2]],
               framealpha=0.3, loc="upper right")

    fig.tight_layout()
    p = out_dir / "lambda_tradeoff.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    logger.info(f"  ✓ {p}")


# ── 3. Training curves ────────────────────────────────────────────────────────

def plot_training_curves(results: List[Dict[str, Any]], out_dir: Path) -> None:
    """Accuracy and sparsity vs epoch for all λ values."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=BG)

    for r, col in zip(results, COLS):
        h   = r["history"]
        eps = [e["epoch"] for e in h]
        a1.plot(eps, [e["val_accuracy"] * 100 for e in h],
                color=col, lw=2, label=f"λ={r['lam']}")
        a2.plot(eps, [e["val_sparsity"] * 100 for e in h],
                color=col, lw=2, label=f"λ={r['lam']}")

    a1.set_title("Test Accuracy vs Epoch");  a1.set_ylabel("Accuracy (%)")
    a2.set_title("Sparsity Level vs Epoch"); a2.set_ylabel("Sparsity (%)")
    for ax in (a1, a2):
        ax.set_xlabel("Epoch"); ax.legend(framealpha=0.3)
    _style(fig, a1, a2)
    fig.suptitle("Training Dynamics — Self-Pruning Network", color=TEXT, fontsize=13)
    fig.tight_layout()
    p = out_dir / "training_curves.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    logger.info(f"  ✓ {p}")


# ── 4. Accuracy vs Sparsity scatter ──────────────────────────────────────────

def plot_accuracy_vs_sparsity(results: List[Dict[str, Any]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=BG)
    for r, col in zip(results, COLS):
        ax.scatter(r["sparsity"] * 100, r["accuracy"] * 100,
                   color=col, s=220, zorder=5, edgecolors="white", lw=0.8)
        ax.annotate(f"λ={r['lam']}",
                    (r["sparsity"] * 100, r["accuracy"] * 100),
                    textcoords="offset points", xytext=(9, 5),
                    fontsize=9.5, color=col, fontweight="bold")
    ax.set_xlabel("Sparsity (%)"); ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Accuracy vs Sparsity", color=TEXT, pad=10)
    _style(fig, ax)
    fig.tight_layout()
    p = out_dir / "accuracy_vs_sparsity.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    logger.info(f"  ✓ {p}")


# ── 5. Gate activity buckets ──────────────────────────────────────────────────

def plot_gate_buckets(results: List[Dict[str, Any]], out_dir: Path) -> None:
    """Bar chart: fraction of gates in pruned / weak / active buckets."""
    labels = ["< 0.01\n(pruned)", "0.01–0.5\n(weak)", "≥ 0.5\n(active)"]
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=BG)
    x, w = np.arange(3), 0.25

    for i, (r, col) in enumerate(zip(results, COLS)):
        g = r["gates"]
        fracs = [
            (g < 0.01).mean() * 100,
            ((g >= 0.01) & (g < 0.5)).mean() * 100,
            (g >= 0.5).mean() * 100,
        ]
        bars = ax.bar(x + i * w, fracs, w, color=col, alpha=0.85,
                      label=f"λ={r['lam']}", edgecolor="none")
        for b, v in zip(bars, fracs):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                    f"{v:.1f}%", ha="center", va="bottom",
                    fontsize=7.5, color=col)

    ax.set_xticks(x + w); ax.set_xticklabels(labels)
    ax.set_ylabel("% of gates")
    ax.set_title("Gate Activity Buckets", color=TEXT, pad=10)
    ax.legend(framealpha=0.3)
    _style(fig, ax)
    fig.tight_layout()
    p = out_dir / "gate_buckets.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    logger.info(f"  ✓ {p}")


# ── Master function ───────────────────────────────────────────────────────────

def generate_all_plots(results: List[Dict[str, Any]], out_dir: Path) -> None:
    """Generate the full plot suite and save to out_dir."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating plots…")
    plot_gate_histograms(results, out_dir)
    plot_lambda_tradeoff(results, out_dir)
    plot_training_curves(results, out_dir)
    plot_accuracy_vs_sparsity(results, out_dir)
    plot_gate_buckets(results, out_dir)
    logger.info(f"All plots saved → {out_dir}")
