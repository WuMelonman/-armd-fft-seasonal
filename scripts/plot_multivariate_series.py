import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def build_multivariate_series(n_steps: int = 240, seed: int = 21):
    rng = np.random.RandomState(seed)
    t = np.arange(n_steps)

    common = 0.35 * np.sin(2 * np.pi * t / 48.0)

    s1 = common + 0.55 * np.sin(2 * np.pi * t / 18.0 + 0.3) + 0.12 * rng.randn(n_steps)
    s2 = 0.85 * common + 0.45 * np.cos(2 * np.pi * t / 25.0 - 0.7) + 0.11 * rng.randn(n_steps)
    s3 = 0.95 * common + 0.40 * np.sin(2 * np.pi * t / 33.0 + 1.0) + 0.10 * rng.randn(n_steps)

    data = np.vstack([s1, s2, s3]).T
    return t, data


def zscore_by_column(data: np.ndarray) -> np.ndarray:
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (data - mean) / std


def main():
    t, data = build_multivariate_series()
    data = zscore_by_column(data)

    n_series = 3
    # Vertical spacing between "folds" (ridge layers).
    layer_gap = 2.8
    # Amplitude of each wiggle within its band.
    amp = 0.85
    colors = ["#1f5f9d", "#c2410c", "#15803d"]
    fill_alphas = [0.16, 0.14, 0.12]
    label_text = [r"$X_{1}$", r"$X_{2}$", r"$X_{3}$"]

    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Draw from back (bottom) to front (top) for a stacked-paper / folded look.
    for k in range(n_series):
        i = n_series - 1 - k
        base = i * layer_gap
        y_line = data[:, i] * amp + base
        t_shadow = t + 0.6
        y_shadow = y_line - 0.10

        ax.fill_between(
            t,
            base,
            y_line,
            color=colors[i],
            alpha=fill_alphas[i],
            zorder=10 + 3 * i,
            linewidth=0,
        )
        # Soft "shadow" behind the fold (gives a bit of 3D depth).
        ax.plot(
            t_shadow,
            y_shadow,
            color="0.45",
            linewidth=1.4,
            alpha=0.18,
            solid_capstyle="round",
            zorder=9 + 3 * i,
        )
        ax.plot(
            t,
            y_line,
            color=colors[i],
            linewidth=1.9,
            solid_capstyle="round",
            zorder=11 + 3 * i,
        )
        # Baseline of this fold.
        ax.axhline(
            base,
            color="0.55",
            linewidth=0.6,
            linestyle="-",
            alpha=0.22,
            zorder=1,
        )
        # Per-sequence label on the left (not a single legend).
        y_mid = base + 0.35
        ax.text(
            t[0] - 14,
            y_mid,
            label_text[i],
            fontsize=13,
            color="0.2",
            ha="right",
            va="center",
            zorder=20,
        )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.grid(False)
    ax.set_xlim(t.min() - 20, t.max() + 2)
    stacked = data * amp + np.arange(n_series) * layer_gap
    y_min = stacked.min()
    y_max = stacked.max()
    ax.set_ylim(y_min - 0.35, y_max + 0.35)
    fig.tight_layout()

    out_path = Path("forecasting_exp/multivariate_series_folded.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved figure to: {out_path.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
