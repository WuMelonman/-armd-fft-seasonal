import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import STL
import torch


def plot_trend_decomposition(
    loader,
    trend_conv,
    kernel_size: int,
    seasonal_period: int = 24,
    save_path: str = "trend_decomposition.png",
):
    """Take the first sample, decompose the trend output, and save a plot."""

    trend_conv.eval()
    print("[Plot] preparing first batch for decomposition...", flush=True)
    batch = next(iter(loader))
    x = batch[0] if isinstance(batch, (list, tuple)) else batch  # (B, T, C)
    x = x[:1]  # only first sample
    x_ch_first = x.permute(0, 2, 1)  # (1, C, T)

    trend = trend_conv(x_ch_first).detach().cpu().squeeze(0)  # (C, T)
    original = x.squeeze(0).detach().cpu()  # (T, C)

    channel_idx = 0  # plot the first channel by default
    series = trend[channel_idx].numpy()
    stl = STL(series, period=seasonal_period, robust=True).fit()

    t = range(series.shape[0])
    fig, axes = plt.subplots(5, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(t, original[:, channel_idx].numpy())
    axes[0].set_title('Original')
    axes[1].plot(t, series)
    axes[1].set_title('TrendConv output')
    axes[2].plot(t, stl.trend)
    axes[2].set_title('Trend (STL)')
    axes[3].plot(t, stl.seasonal)
    axes[3].set_title('Seasonal (STL)')
    axes[4].plot(t, stl.resid)
    axes[4].set_title('Residual (STL)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'[Plot] saved plot to {save_path}', flush=True)


__all__ = ["plot_trend_decomposition"]
