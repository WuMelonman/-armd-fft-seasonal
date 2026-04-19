#!/usr/bin/env python3
"""
Grouped bar chart: MTMD vs MTMD_EX (MSE / MAE) on ETTh1 & ETTm1.
Saves PNG under ``forecasting_exp/mtmd_vs_mtmd_ex_barplot.png`` (repo root).
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "forecasting_exp"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "mtmd_vs_mtmd_ex_barplot.png"

# =========================
# Data
# =========================
datasets = ["ETTh1", "ETTm1"]

mtmd_mse = [0.384, 0.305]
mtmd_mae = [0.427, 0.326]

mtmd_ex_mse = [0.980, 0.799]
mtmd_ex_mae = [0.729, 0.673]

# =========================
# Style
# =========================
plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.linewidth"] = 1.1

color_mtmd = "#4C78A8"
color_mtmd_ex = "#F58518"

# =========================
# Layout
# =========================
x = np.arange(len(datasets)) * 2.2
bar_w = 0.28

mse_pos_mtmd = x - 0.45
mse_pos_mtmd_ex = x - 0.17
mae_pos_mtmd = x + 0.17
mae_pos_mtmd_ex = x + 0.45

fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

# =========================
# Bars
# =========================
bars1 = ax.bar(
    mse_pos_mtmd,
    mtmd_mse,
    width=bar_w,
    color=color_mtmd,
    edgecolor="black",
    linewidth=0.8,
    label="MTMD",
)
bars2 = ax.bar(
    mse_pos_mtmd_ex,
    mtmd_ex_mse,
    width=bar_w,
    color=color_mtmd_ex,
    edgecolor="black",
    linewidth=0.8,
    label=r"MTMD$_{EX}$",
)
bars3 = ax.bar(
    mae_pos_mtmd,
    mtmd_mae,
    width=bar_w,
    color=color_mtmd,
    edgecolor="black",
    linewidth=0.8,
)
bars4 = ax.bar(
    mae_pos_mtmd_ex,
    mtmd_ex_mae,
    width=bar_w,
    color=color_mtmd_ex,
    edgecolor="black",
    linewidth=0.8,
)

# =========================
# X-axis: dataset labels
# =========================
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=13)
ax.set_xlabel("Datasets", fontsize=14)
ax.set_ylabel("Error", fontsize=14)

for i in range(len(datasets)):
    ax.text(
        (mse_pos_mtmd[i] + mse_pos_mtmd_ex[i]) / 2,
        -0.06,
        "MSE",
        ha="center",
        va="top",
        fontsize=11,
        transform=ax.get_xaxis_transform(),
    )
    ax.text(
        (mae_pos_mtmd[i] + mae_pos_mtmd_ex[i]) / 2,
        -0.06,
        "MAE",
        ha="center",
        va="top",
        fontsize=11,
        transform=ax.get_xaxis_transform(),
    )

# =========================
# Legend / Grid / Limits
# =========================
ax.legend(loc="upper right", fontsize=11, frameon=True)
ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.set_axisbelow(True)
ax.set_ylim(0, 1.1)


def add_labels(bars):
    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            h + 0.015,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )


add_labels(bars1)
add_labels(bars2)
add_labels(bars3)
add_labels(bars4)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_PATH}")
