"""
trend_conv 固定 MA 分解的可视化：Original / Trend (MA) / Seasonal / Reconstructed.
与参考图布局一致，纵排、共享 x 轴。
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


__all__ = ["plot_trend_decomposition"]
