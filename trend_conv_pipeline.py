import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")  # ensure non-GUI backend
import matplotlib.pyplot as plt

from typing import List
from statsmodels.tsa.seasonal import STL
from Data.build_dataloader import build_dataloader, build_dataloader_cond
from Utils.io_utils import load_yaml_config


def set_seed(seed: int = 2025):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TrendConvNet(nn.Module):
    """Depthwise 1D conv used to learn a trend smoother."""

    def __init__(self, feature_size: int, kernel_size: int = 5):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=kernel_size,
            padding=padding,
            groups=feature_size,
            bias=True,
        )
        nn.init.normal_(self.conv.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        return self.conv(x)


def moving_average_target(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Fixed moving average as pseudo target for trend.

    Args:
        x: (B, C, T)
        kernel_size: window length
    Returns:
        (B, C, T) smoothed series
    """
    b, c, _ = x.shape
    weight = torch.ones(c, 1, kernel_size, device=x.device, dtype=x.dtype) / kernel_size
    padding = kernel_size // 2
    return F.conv1d(x, weight, padding=padding, groups=c)


def train_trend_conv(model: TrendConvNet, dataloader, device, epochs: int = 3, kernel_size: int = 5, lr: float = 1e-3):
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"[TrendConv] start training: epochs={epochs}, lr={lr}, kernel={kernel_size}", flush=True)
    for epoch in range(epochs):
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            x = x.to(device)  # (B, T, C)
            x_ch_first = x.permute(0, 2, 1)  # (B, C, T)
            target = moving_average_target(x_ch_first, kernel_size)
            pred = model(x_ch_first)
            loss = F.mse_loss(pred, target)
            optim.zero_grad()
            loss.backward()
            optim.step()
        print(f"[TrendConv] epoch {epoch+1}/{epochs} loss={loss.item():.6f}", flush=True)
    model.eval()


def build_trend_loader(original_loader, model: TrendConvNet, device, batch_size: int, shuffle: bool):
    """Run the trained conv over all batches and return a new DataLoader of trend tensors."""
    trend_batches: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in original_loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            x = x.to(device)
            trend = model(x.permute(0, 2, 1)).permute(0, 2, 1)  # back to (B, T, C)
            trend_batches.append(trend.cpu())
    if not trend_batches:
        raise RuntimeError("No data found to build trend loader")
    all_trend = torch.cat(trend_batches, dim=0)

    class _TrendDataset(torch.utils.data.Dataset):
        def __init__(self, data: torch.Tensor):
            self.data = data
        def __len__(self):
            return self.data.shape[0]
        def __getitem__(self, idx):
            return self.data[idx]

    dataset = _TrendDataset(all_trend)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)


def plot_trend_decomposition(loader, trend_conv: TrendConvNet, kernel_size: int, seasonal_period: int = 24, save_path: str = 'trend_decomposition.png'):
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


def parse_arguments():
    parser = argparse.ArgumentParser(description="Trend conv + ARMD trainer")
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='./forecasting_exp')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--trend_epochs', type=int, default=3)
    parser.add_argument('--trend_lr', type=float, default=1e-3)
    parser.add_argument('--trend_kernel', type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_arguments()
    set_seed(2025)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    configs = load_yaml_config(args.config_path)

    # 1) Build original train loader (plotting只需要train)
    train_info = build_dataloader(configs, args)
    train_loader = train_info['dataloader']
    try:
        train_len = len(train_loader.dataset)
    except Exception:
        train_len = 'unknown'
    print(f"[Data] train batches={len(train_loader)}, train samples={train_len}", flush=True)

    feature_size = configs['model']['params']['feature_size']
    trend_conv = TrendConvNet(feature_size=feature_size, kernel_size=args.trend_kernel).to(device)

    print('Training trend conv...')
    train_trend_conv(trend_conv, train_loader, device, epochs=args.trend_epochs, kernel_size=args.trend_kernel, lr=args.trend_lr)

    # Freeze conv for downstream
    for p in trend_conv.parameters():
        p.requires_grad = False

    # Plot decomposition on the first sample after freezing weights
    seasonal_period = configs['dataloader']['train_dataset']['params'].get('seasonal_period', 24)
    plot_trend_decomposition(train_loader, trend_conv, kernel_size=args.trend_kernel, seasonal_period=seasonal_period, save_path='trend_decomposition.png')


if __name__ == '__main__':
    main()
