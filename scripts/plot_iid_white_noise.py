import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def build_white_noise(n_points: int = 190, seed: int = 7):
    rng = np.random.RandomState(seed)
    x = np.arange(n_points)
    y = rng.normal(loc=0.0, scale=1.0, size=n_points)
    return x, y


def pick_markers(x: np.ndarray):
    idx = np.array([12, 32, 52, 72, 92, 112, 132, 152, 172])
    return x[idx], idx


def main():
    x, y = build_white_noise()
    mx, midx = pick_markers(x)
    my = y[midx]
    baseline = np.full_like(mx, y.min() - 0.9, dtype=float)

    fig, ax = plt.subplots(figsize=(10.2, 5.9), dpi=140)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Main white-noise series.
    ax.plot(x, y, color="black", linewidth=2.0, zorder=3)

    # Vertical dashed lines and marker circles.
    for xi, yi, bi in zip(mx, my, baseline):
        ax.plot(
            [xi, xi],
            [bi, yi],
            linestyle="--",
            color="#3b6ea5",
            linewidth=1.2,
            alpha=0.95,
            zorder=1,
        )

    ax.scatter(mx, my, s=155, facecolors="none", edgecolors="#3b6ea5", linewidths=2.0, zorder=4)
    ax.scatter(mx, baseline, s=155, facecolors="#d3e4f6", edgecolors="#3b6ea5", linewidths=2.0, zorder=4)

    # Add iid annotation between neighboring sampled points.
    for i in range(len(mx) - 1):
        x0, x1 = mx[i], mx[i + 1]
        y_arrow = baseline[i] + 0.35
        ax.annotate(
            "",
            xy=(x1 - 1.4, y_arrow),
            xytext=(x0 + 1.4, y_arrow),
            arrowprops=dict(arrowstyle="<->", color="#5f88b3", lw=1.2),
            zorder=2,
        )
        ax.text(
            (x0 + x1) / 2.0,
            y_arrow + 0.23,
            "iid",
            color="#1f1f1f",
            fontsize=14,
            ha="center",
            va="bottom",
        )

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(baseline.min() - 0.35, y.max() + 0.6)

    # Clean look: no title, legend, axis labels, ticks, or spines.
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="both", which="both", length=0, labelbottom=False, labelleft=False)
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(False)

    fig.tight_layout()

    out_path = Path("forecasting_exp/iid_white_noise_custom.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved figure to: {out_path.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
