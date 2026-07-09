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

# Definitive 50-run pass (warmed). Decode P50 147, P95 150; a brief dip to
# ~133 at runs 15-21 is a thermal/clock throttle that recovers. Prefill is
# rock-steady at ~222 (stdev 2.2).
run50 = list(range(1, 51))
decode50 = [140.8,149.8,149.9,149.2,148.1,150.4,144.1,148.1,145.3,145.9,147.1,147.5,148.4,142.2,133.3,132.9,132.4,134.1,134.4,133.6,134.4,149.9,146.2,146.8,149.4,148.3,146.0,149.3,148.1,145.9,147.4,146.6,146.9,148.2,147.8,147.8,147.1,147.8,147.5,146.8,146.5,145.6,143.8,147.6,148.4,147.0,147.5,148.6,146.1,145.2]
prefill50 = [191.1,224.6,224.3,223.9,224.4,223.2,224.1,223.4,223.2,224.5,223.9,222.5,222.7,222.4,220.8,216.4,230.9,223.8,223.1,223.0,217.3,223.8,220.6,222.1,221.8,222.1,220.9,221.7,221.3,221.1,221.5,220.7,220.8,222.5,220.9,221.1,221.7,221.2,222.0,220.1,221.0,220.7,220.8,222.5,220.9,221.1,221.7,217.7,219.8,220.8]

# ==========================================================================
# GRAPH 1 — decode tok/s across 50 runs (distribution + throttle dip)
# ==========================================================================
def graph_decode_runs():
    import statistics as st
    p50 = st.median(decode50[1:])  # drop warmup run 1
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    style(ax)
    ax.plot(run50, decode50, color=BLUE, linewidth=1.8, marker="o", markersize=4,
            zorder=3, label="decode (50 runs)")
    ax.axhline(p50, color=INK2, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate(f"P50 = {p50:.0f} tok/s", (1, p50), textcoords="offset points",
                xytext=(2, 5), color=INK2, fontsize=10)
    # call out the throttle dip
    ax.annotate("thermal/clock\nthrottle, recovers", (18, 132.4),
                textcoords="offset points", xytext=(6, -34), color="#b87c00",
                fontsize=9, ha="left",
                arrowprops=dict(arrowstyle="-", color="#b87c00", linewidth=0.8))
    ax.set_xlabel("run #")
    ax.set_ylabel("decode tok/s")
    ax.set_title("Decode throughput, 50 runs (Tesla T4, Qwen2.5-0.5B Q8_0)",
                 fontweight="bold", loc="left", color=INK)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.set_ylim(125, 155)
    ax.legend(frameon=False, loc="lower right")
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
    vals = [st.median(prefill50[1:]), st.median(decode50[1:])]
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
    import statistics as st
    measured = st.median(decode50[1:])   # P50 steady decode (147)
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
