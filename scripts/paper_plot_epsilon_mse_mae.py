#!/usr/bin/env python3
"""
论文级折线图：MSE / MAE 随 epsilon 变化。
版式与 paper_plot_alpha_c_mse_mae.py 一致（半栏宽、小字号、无纵轴标题）。

用法:
  python scripts/paper_plot_epsilon_mse_mae.py
  python scripts/paper_plot_epsilon_mse_mae.py -o forecasting_exp/epsilon_metrics.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLOR_MSE = "#4C78A8"
COLOR_MAE = "#F58518"

SINGLE_COLUMN_IN = 3.4
PANEL_W = SINGLE_COLUMN_IN / 2.0
PANEL_H = PANEL_W * 0.52


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot MSE & MAE vs epsilon (publication style).")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("forecasting_exp/epsilon_mse_mae.png"),
        help="Output file (.png / .pdf / .svg). Default: forecasting_exp/epsilon_mse_mae.png",
    )
    p.add_argument("--dpi", type=int, default=300, help="Raster DPI (PNG). Default: 300")
    return p


def main() -> None:
    args = build_parser().parse_args()

    eps = np.linspace(0.0, 0.6, 7)
    mse = np.array([0.362, 0.327, 0.341, 0.372, 0.419, 0.479, 0.551], dtype=np.float64)
    mae = np.array([0.419, 0.385, 0.394, 0.416, 0.451, 0.492, 0.539], dtype=np.float64)

    _fs_tick = 5.5
    _fs_label = 6.25
    _fs_title = 6.75
    _fs_leg = 5.5
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Nimbus Sans"],
            "axes.linewidth": 0.9,
            "axes.edgecolor": "0.15",
            "axes.labelsize": _fs_label,
            "axes.titlesize": _fs_title,
            "xtick.labelsize": _fs_tick,
            "ytick.labelsize": _fs_tick,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.fontsize": _fs_leg,
            "axes.unicode_minus": False,
        }
    )

    fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H), dpi=args.dpi, constrained_layout=False)

    ax.set_facecolor("#ececec")
    fig.patch.set_facecolor("white")

    lw = 1.15
    ms = 4.0
    mew = 0.85

    ax.plot(
        eps,
        mse,
        color=COLOR_MSE,
        linewidth=lw,
        marker="o",
        markersize=ms,
        markeredgecolor="0.15",
        markeredgewidth=mew,
        label="MSE",
        clip_on=False,
        zorder=3,
    )
    ax.plot(
        eps,
        mae,
        color=COLOR_MAE,
        linewidth=lw,
        marker="o",
        markersize=ms,
        markeredgecolor="0.15",
        markeredgewidth=mew,
        label="MAE",
        clip_on=False,
        zorder=3,
    )

    ax.set_xlim(-0.02, 0.62)
    y_min = min(mse.min(), mae.min())
    y_max = max(mse.max(), mae.max())
    pad = (y_max - y_min) * 0.08
    ax.set_ylim(y_min - pad, y_max + pad)

    ax.set_xticks(np.arange(0.0, 0.61, 0.1))
    ax.set_xlabel(r"$\varepsilon$", fontsize=_fs_label)
    ax.tick_params(axis="both", which="major", labelsize=_fs_tick, length=3.0, width=0.65)

    ax.set_title(
        r"MSE and MAE vs.\ $\varepsilon$",
        fontweight="bold",
        fontsize=_fs_title,
        pad=3,
    )

    ax.grid(True, which="major", linestyle="-", linewidth=0.5, color="white", alpha=0.95, zorder=0)
    ax.grid(True, which="minor", linestyle="-", linewidth=0.35, color="0.92", alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.minorticks_on()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    leg = ax.legend(
        loc="upper left",
        frameon=True,
        fancybox=False,
        edgecolor="0.72",
        facecolor="white",
        framealpha=1.0,
        fontsize=_fs_leg,
        handlelength=1.35,
        handletextpad=0.45,
        borderpad=0.35,
        labelspacing=0.25,
    )
    leg.get_frame().set_linewidth(0.5)

    fig.subplots_adjust(left=0.19, right=0.98, top=0.90, bottom=0.20)

    out: Path = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[paper_plot_epsilon_mse_mae] saved: {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
