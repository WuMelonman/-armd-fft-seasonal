"""
trend_conv 固定 MA 分解的可视化：Original / Trend (MA) / Seasonal / Reconstructed.
与参考图布局一致，纵排、共享 x 轴。
"""

import matplotlib
matplotlib.use("Agg")
import math

import matplotlib.pyplot as plt
import numpy as np
import torch

from trend_utils.trend_conv import moving_average_btc


def plot_trend_decomposition(
    loader,
    kernel_size: int = 25,
    channel_idx: int = 0,
    save_path: str = "trend_decomposition.png",
):
    """
    从 loader 取第一个 batch 的第一个样本，用固定 MA 分解画图。
    不依赖可训练的 trend_conv，直接使用 moving_average_btc。
    """
    print("[Plot] preparing first batch for MA decomposition...", flush=True)
    batch = next(iter(loader))
    x = batch[0] if isinstance(batch, (list, tuple)) else batch  # (B, T, C)
    x = x[:1]  # (1, T, C)

    with torch.no_grad():
        trend, seasonal = moving_average_btc(x, kernel_size=kernel_size)

    original = x.squeeze(0).cpu().numpy()       # (T, C)
    trend_np = trend.squeeze(0).cpu().numpy()   # (T, C)
    seasonal_np = seasonal.squeeze(0).cpu().numpy()  # (T, C)
    reconstructed = trend_np + seasonal_np      # 应等于 original

    t = range(original.shape[0])
    fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(t, original[:, channel_idx])
    axes[0].set_title("Original")
    axes[0].set_ylabel("")
    axes[1].plot(t, trend_np[:, channel_idx])
    axes[1].set_title("Trend (MA)")
    axes[1].set_ylabel("")
    axes[2].plot(t, seasonal_np[:, channel_idx])
    axes[2].set_title("Seasonal (= Original − Trend)")
    axes[2].set_ylabel("")
    axes[3].plot(t, reconstructed[:, channel_idx])
    axes[3].set_title("Reconstructed (Trend + Seasonal)")
    axes[3].set_ylabel("")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Plot] saved to {save_path}", flush=True)


def plot_forecast_fit(
    pred: np.ndarray,
    real: np.ndarray,
    save_path: str = "forecast_fit.png",
    channel_idx: int = 0,
    sample_idx: int = 0,
    scaler=None,
    dataset=None,
    title: str = None,
    gt_label: str = "Ground Truth",
    pred_label: str = "MTMD Prediction",
    figsize=None,
    max_timesteps: int = 200,
    concat_windows: int = None,
):
    """
    单条曲线：真实 vs 预测（蓝=真值、橙=MTMD 预测、浅灰虚线网格）。

    pred, real: (N, pred_len, C) 与 sample_forecast 输出一致。
    从 ``sample_idx`` 起沿时间拼接若干预测窗，再**只取前 ``max_timesteps`` 个点**画图（默认约 200，横轴不会拉到上千）。
    若 ``concat_windows`` 为 None，则自动取「至少能覆盖 max_timesteps」的最少段数；若设为正整数，则与上述下限取较大者再截断。
    若传入 ``dataset`` 或 ``scaler``，对 (T, C) 做 ``inverse_transform``。
    ``figsize``：默认正方形 ``(8, 8)``；可自定义。
    """
    p = np.asarray(pred, dtype=np.float64)
    r = np.asarray(real, dtype=np.float64)
    if p.shape != r.shape:
        raise ValueError(f"pred/real shape mismatch: {p.shape} vs {r.shape}")
    n_avail = p.shape[0] - sample_idx
    if n_avail <= 0:
        raise ValueError(f"sample_idx={sample_idx} out of range, N={p.shape[0]}")
    t_win = p.shape[1]
    k_need = max(1, int(math.ceil(max_timesteps / t_win)))
    if concat_windows is None:
        k = min(n_avail, k_need)
    else:
        k = min(n_avail, max(k_need, int(concat_windows)))
    sl = p[sample_idx : sample_idx + k]  # (k, T, C)
    sr = r[sample_idx : sample_idx + k]
    c_dim = sl.shape[2]
    sl = sl.reshape(-1, c_dim)
    sr = sr.reshape(-1, c_dim)

    def _to_original(y_norm_2d):
        """(T, C) Z-score -> 原始量级（与 CustomDataset 中 StandardScaler 一致）"""
        sc = scaler
        if sc is None and dataset is not None and getattr(dataset, "scaler", None) is not None:
            sc = dataset.scaler
        if sc is not None:
            t, c = y_norm_2d.shape
            return sc.inverse_transform(y_norm_2d.reshape(-1, c)).reshape(t, c)
        return y_norm_2d

    sl = _to_original(sl)
    sr = _to_original(sr)
    n_plot = min(max_timesteps, sl.shape[0])
    sl = sl[:n_plot]
    sr = sr[:n_plot]
    y_gt = sr[:, channel_idx]
    y_pr = sl[:, channel_idx]
    t = np.arange(len(y_gt))

    # 参考色：matplotlib 默认蓝 / 浅橙
    c_gt = "#1f77b4"
    c_pr = "#ffbb78"

    if figsize is None:
        figsize = (8.0, 8.0)
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    ax.plot(t, y_gt, color=c_gt, linewidth=1.1, label=gt_label, zorder=2)
    ax.plot(t, y_pr, color=c_pr, linewidth=1.1, label=pred_label, zorder=2)
    ax.grid(True, linestyle="--", alpha=0.5, color="0.75", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=10)
    ax.margins(x=0.008)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("0.15")

    if title:
        ax.set_title(title, fontsize=12, pad=8)
    leg = ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=True,
        fancybox=False,
        edgecolor="0.7",
        fontsize=11,
    )
    leg.get_frame().set_linewidth(0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] forecast fit saved to {save_path}", flush=True)


__all__ = ["plot_trend_decomposition", "plot_forecast_fit"]
