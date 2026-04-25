import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple


def build_series(n_points: int = 160) -> Tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0, 11.5, n_points)

    # Build a "paper-like" signal: low-frequency backbone + dense small ripples.
    y = (
        0.58 * np.sin(3.25 * x + 0.25)
        + 0.24 * np.sin(6.8 * x - 0.7)
        + 0.10 * np.cos(10.5 * x + 0.55)
        + 0.060 * np.sin(15.0 * x - 0.20)
        + 0.045 * np.cos(19.5 * x + 1.10)
        + 0.036 * np.sin(24.5 * x - 0.85)
        + 0.028 * np.cos(30.0 * x + 0.35)
        + 0.022 * np.sin(36.5 * x - 1.20)
    )

    # Small deterministic roughness to enhance the scratchy texture.
    y += 0.008 * np.sin(52 * x + 0.3) + 0.006 * np.cos(64 * x - 0.5)

    # Add gentle trend and normalization for a clean display range.
    y += 0.02 * x
    y = (y - y.min()) / (y.max() - y.min())
    y = 0.18 + 0.78 * y
    return x, y


def pick_markers(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Pick quasi-periodic marker locations (not uniform) to avoid exact replication.
    idx = np.array([7, 18, 28, 39, 50, 61, 73, 85, 97, 110, 124, 139, 153])
    return x[idx], y[idx]


def main() -> None:
    x, y = build_series()
    mx, my = pick_markers(x, y)
    baseline = np.full_like(mx, 0.02)

    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=140)

    # Main curve.
    ax.plot(x, y, color="black", linewidth=2.2, zorder=3)

    # Vertical dashed connectors.
    for xi, yi, bi in zip(mx, my, baseline):
        ax.plot(
            [xi, xi],
            [bi, yi],
            linestyle="--",
            color="#1f5a9d",
            linewidth=1.0,
            alpha=0.95,
            zorder=1,
        )

    # Top and bottom marker points.
    ax.scatter(mx, my, s=58, facecolors="none", edgecolors="#1f5a9d", linewidths=1.3, zorder=4)
    ax.scatter(mx, baseline, s=56, facecolors="#b9d4ef", edgecolors="#1f5a9d", linewidths=1.1, zorder=4)

    # Remove title, legend, and axis labels as requested.
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    # Keep a clean look, close to the provided style.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="both", which="both", length=0, labelbottom=False, labelleft=False)

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(-0.08, 1.06)
    fig.tight_layout()

    out_path = Path("forecasting_exp/similar_wave_custom.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved figure to: {out_path.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
