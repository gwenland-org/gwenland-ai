#!/usr/bin/env python3
"""Generate the M2-completion charts embedded in ArchGLML_Done.md.

All numbers are measured on a Tesla T4 (sm_75) via glcuda/examples/bench and the
7B run in glcuda_t4_validation.ipynb. Colab GPUs are shared, so absolute values
carry run-to-run jitter (~a few %); the deltas below are stable across runs.

    python gen_done_graphs.py     # writes done_*.png next to this file
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# --- validated palette (dataviz skill, light surface) ---
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
BLUE = "#2a78d6"   # series 1 — "after" / SoA
AQUA = "#1baf7a"   # series 2
MUTED = "#b0afa8"  # neutral baseline — "before" / AoS
GRID = "#e6e5e0"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 11,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.edgecolor": GRID,
})
HERE = os.path.dirname(os.path.abspath(__file__))


def _clean(ax, x=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)
    if x:
        ax.xaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# 1 — 7B decode: AoS -> SoA vs hardware bandwidth ceiling
def decode_chart():
    labels = ["AoS Q8_0\n(before)", "SoA Q8_0\n(after)", "HW ceiling\n(100% BW)"]
    vals = [24.4, 29.2, 35.0]
    colors = [MUTED, BLUE, "none"]
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    bars = ax.barh(labels, vals, color=colors, height=0.62,
                   edgecolor=[MUTED, BLUE, MUTED],
                   linewidth=[0, 0, 1.4], hatch=[None, None, "////"])
    ax.set_xlim(0, 39)
    ax.set_xlabel("decode throughput (tok/s) — Qwen2.5-7B-Q8_0, Tesla T4")
    ax.invert_yaxis()
    _clean(ax)
    for b, v in zip(bars, vals):
        ax.text(v + 0.5, b.get_y() + b.get_height() / 2,
                f"{v:.1f}", va="center", ha="left", color=INK, fontweight="bold")
    ax.text(29.2 / 2, 1, "+20%", va="center", ha="center",
            color="white", fontweight="bold")
    save(fig, "done_decode.png")


# 2 — Q8_0 GEMV bandwidth: AoS vs SoA per shape (% of 264 GB/s achievable)
def gemv_bw_chart():
    shapes = ["gate_up", "down", "lm_head"]
    aos = [205, 196, 217]
    soa = [232, 248, 220]
    ceil = 264
    y = range(len(shapes))
    h = 0.36
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    b1 = ax.barh([i + h / 2 for i in y], aos, height=h, color=MUTED, label="AoS (before)")
    b2 = ax.barh([i - h / 2 for i in y], soa, height=h, color=BLUE, label="SoA (after)")
    ax.axvline(ceil, color=INK2, lw=1.2, ls=(0, (4, 3)))
    ax.text(ceil - 3, -0.72, f"{ceil} GB/s achievable", ha="right", va="center",
            color=INK2, fontsize=9)
    ax.set_yticks(list(y))
    ax.set_yticklabels(shapes)
    ax.set_xlim(0, 300)
    ax.set_xlabel("Q8_0 GEMV bandwidth (GB/s) — Tesla T4")
    ax.invert_yaxis()
    _clean(ax)
    for bars, vals, col in ((b1, aos, INK2), (b2, soa, INK)):
        for b, v in zip(bars, vals):
            ax.text(v + 3, b.get_y() + b.get_height() / 2, f"{v}",
                    va="center", ha="left", color=col,
                    fontweight="bold" if col == INK else "normal", fontsize=9.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.005, 1.0), frameon=False, fontsize=9.5)
    save(fig, "done_gemv_bw.png")


# 3 — prefill: batched GEMM vs sequential per-token GEMV (32 tokens)
def prefill_chart():
    labels = ["Sequential\n32x GEMV\n(today)", "Batched GEMM\ngl_gemm_q8_0_soa\n(T=4)"]
    vals = [1393, 429]
    colors = [MUTED, AQUA]
    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    bars = ax.barh(labels, vals, color=colors, height=0.6)
    ax.set_xlim(0, 1560)
    ax.set_xlabel("time for 32 tokens, gate/up shape (us) — lower is better")
    ax.invert_yaxis()
    _clean(ax)
    for b, v in zip(bars, vals):
        ax.text(v + 18, b.get_y() + b.get_height() / 2, f"{v} us",
                va="center", ha="left", color=INK, fontweight="bold")
    ax.text(429 / 2, 1, "3.25x", va="center", ha="center",
            color="white", fontweight="bold", fontsize=13)
    save(fig, "done_prefill.png")


if __name__ == "__main__":
    decode_chart()
    gemv_bw_chart()
    prefill_chart()
    print("done.")
