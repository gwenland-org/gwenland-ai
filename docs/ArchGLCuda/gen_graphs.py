#!/usr/bin/env python3
"""Generate the benchmark graphs for BENCHMARK_ArchGLCuda.md.

All numbers are REAL measurements from a Google Colab Tesla T4 (sm_75,
driver 13.0, 15 GB), running Qwen2.5-0.5B-Instruct Q8_0 through the glcuda
engine. Re-run after each optimization to refresh the images.

    python docs/ArchGLCuda/gen_graphs.py

Design: colorblind-safe categorical palette (validated, dataviz skill),
one y-axis per chart, thin recessive axes, direct labels over legends where
a single series, values in ink not the series color.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

OUT = os.path.dirname(os.path.abspath(__file__))

# --- validated categorical palette (light surface) ---
BLUE   = "#2a78d6"
AQUA   = "#1baf7a"
YELLOW = "#eda100"
RED    = "#e34948"
INK    = "#0b0b0b"
INK2   = "#52514e"
GRID   = "#e6e6e3"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": INK2,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "font.size": 11,
    "figure.dpi": 130,
    "savefig.bbox": "tight",
})

def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

# ==========================================================================
# MEASURED DATA (Tesla T4, Qwen2.5-0.5B Q8_0)
# ==========================================================================

# Two 10-run passes, post attention-fusion (commit fbd6f51). Decode tok/s
# warms up over the first runs then settles; prefill is stable.
run_idx = list(range(1, 11))
decode_a = [107.7, 110.8, 105.4, 102.0, 101.8, 102.1, 102.7, 98.5, 102.7, 111.3]  # earlier session
decode_b = [128.0, 135.4, 136.0, 137.1, 134.6, 136.0, 142.6, 149.5, 151.0, 149.5]  # warmed session
prefill_b = [183.8, 229.4, 226.9, 228.2, 223.0, 227.5, 227.2, 227.6, 230.2, 227.1]

# ==========================================================================
# GRAPH 1 — decode tok/s across runs (warmup curve)
# ==========================================================================
def graph_decode_runs():
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    style(ax)
    ax.plot(run_idx, decode_b, color=BLUE, linewidth=2, marker="o", markersize=6,
            zorder=3, label="warmed session")
    ax.plot(run_idx, decode_a, color=YELLOW, linewidth=2, marker="s", markersize=5,
            zorder=2, label="cold session")
    # direct labels at the last point of each
    ax.annotate(f"{decode_b[-1]:.0f}", (run_idx[-1], decode_b[-1]),
                textcoords="offset points", xytext=(8, 2), color=BLUE, fontweight="bold")
    ax.annotate(f"{decode_a[-1]:.0f}", (run_idx[-1], decode_a[-1]),
                textcoords="offset points", xytext=(8, -4), color="#b87c00")
    ax.set_xlabel("run #")
    ax.set_ylabel("decode tok/s")
    ax.set_title("Decode throughput per run (Tesla T4, Qwen2.5-0.5B Q8_0)",
                 fontweight="bold", loc="left", color=INK)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.set_ylim(90, 165)
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(os.path.join(OUT, "benchmark_img1.png"))
    plt.close(fig)
    print("wrote benchmark_img1.png")

# ==========================================================================
# GRAPH 2 — prefill vs decode (steady-state medians)
# ==========================================================================
def graph_prefill_vs_decode():
    import statistics as st
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    style(ax)
    labels = ["prefill", "decode"]
    vals = [st.median(prefill_b), st.median(decode_b)]
    colors = [AQUA, BLUE]
    bars = ax.bar(labels, vals, color=colors, width=0.55, zorder=3)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.0f} tok/s", (b.get_x() + b.get_width()/2, v),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    color=INK, fontweight="bold")
    ax.set_ylabel("tok/s (median of 10 runs)")
    ax.set_title("Prefill vs decode throughput", fontweight="bold", loc="left", color=INK)
    ax.set_ylim(0, 260)
    fig.savefig(os.path.join(OUT, "benchmark_img2.png"))
    plt.close(fig)
    print("wrote benchmark_img2.png")

# ==========================================================================
# GRAPH 3 — the ceiling: measured decode vs bandwidth-implied maximum
# ==========================================================================
def graph_ceiling():
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    style(ax)
    # weights streamed per token ~ 0.5 GB (Q8_0 0.5B). T4 peak BW 320 GB/s,
    # achievable typically ~0.8x. Decode = weight stream once per token.
    measured = 150.0                     # best steady decode
    bw_peak_max = 320 / 0.5              # 640 tok/s at peak bandwidth
    bw_real_max = 256 / 0.5             # ~512 tok/s at ~80% achievable
    labels = ["measured\n(now)", "achievable-BW\nceiling (~80%)", "peak-BW\nceiling"]
    vals = [measured, bw_real_max, bw_peak_max]
    colors = [BLUE, YELLOW, "#cccccc"]
    bars = ax.barh(labels[::-1], vals[::-1], color=colors[::-1], height=0.6, zorder=3)
    for b, v in zip(bars, vals[::-1]):
        ax.annotate(f"{v:.0f} tok/s", (v, b.get_y() + b.get_height()/2),
                    textcoords="offset points", xytext=(6, 0), va="center",
                    color=INK, fontweight="bold")
    ax.set_xlabel("decode tok/s")
    ax.set_title("Headroom: decode vs memory-bandwidth ceiling",
                 fontweight="bold", loc="left", color=INK)
    ax.set_xlim(0, 720)
    pct = measured / bw_real_max * 100
    ax.annotate(f"~{pct:.0f}% of achievable bandwidth",
                (measured, 2), textcoords="offset points", xytext=(6, 14),
                color=INK2, fontsize=10)
    fig.savefig(os.path.join(OUT, "benchmark_img3.png"))
    plt.close(fig)
    print("wrote benchmark_img3.png")

if __name__ == "__main__":
    graph_decode_runs()
    graph_prefill_vs_decode()
    graph_ceiling()
    print("done ->", OUT)
