"""
固定分解（MA 或 STL），不参与训练。
趋势：MA 核大小 25；季节/残差：x - trend，供 FFT top-k 预测。
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple

try:
    from statsmodels.tsa.seasonal import STL
except ImportError:
    STL = None


# ---------- MA 分解（核大小 25），趋势 = MA(x)，季节 = x - trend ----------
def moving_average_btc(
    x: torch.Tensor,
    kernel_size: int = 25,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对 (B, T, C) 做固定 MA 分解，不参与训练。
    trend = MA(x), seasonal = x - trend。
    """
    B, T, C = x.shape
    device = x.device
    dtype = x.dtype
    padding = kernel_size // 2
    # (B, T, C) -> (B, C, T)
    x_bct = x.permute(0, 2, 1)
    weight = torch.ones(C, 1, kernel_size, device=device, dtype=dtype) / kernel_size
    trend_bct = F.conv1d(x_bct, weight, padding=padding, groups=C)
    trend = trend_bct.permute(0, 2, 1)  # (B, T, C)
    seasonal = x - trend
    return trend, seasonal


class MADecompose(torch.nn.Module):
    """固定 MA 分解，无参数。forward(x) -> (trend, seasonal)，核大小默认 25。"""

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return moving_average_btc(x, self.kernel_size)


# ---------- STL 分解（可选保留） ----------
def stl_decompose_1d(series: np.ndarray, seasonal_period: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    单条 1D 序列的 STL 分解，返回 trend 与 seasonal（不包含 resid）。
    series: (T,), 返回 trend (T,), seasonal (T,)。
    """
    if STL is None:
        raise ImportError("statsmodels is required: pip install statsmodels")
    # period 须为奇数且 >= 7，且小于序列长度
    period = max(7, seasonal_period)
    if period % 2 == 0:
        period += 1
    period = min(period, len(series) - 1)
    if period < 7:
        return series.copy(), np.zeros_like(series)
    res = STL(series, period=period, robust=True).fit()
    return res.trend, res.seasonal


def stl_decompose_btc(
    x: torch.Tensor,
    seasonal_period: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对 (B, T, C) 张量做固定 STL 分解，不参与训练。
    x: (B, T, C)
    返回: trend (B, T, C), seasonal (B, T, C)
    """
    device = x.device
    x_np = x.detach().cpu().numpy()
    B, T, C = x_np.shape
    trend_np = np.zeros_like(x_np)
    seasonal_np = np.zeros_like(x_np)
    for b in range(B):
        for c in range(C):
            tr, se = stl_decompose_1d(x_np[b, :, c], seasonal_period)
            trend_np[b, :, c] = tr
            seasonal_np[b, :, c] = se
    trend = torch.from_numpy(trend_np).to(device=device, dtype=x.dtype)
    seasonal = torch.from_numpy(seasonal_np).to(device=device, dtype=x.dtype)
    return trend, seasonal


class STLDecompose(torch.nn.Module):
    """
    固定 STL 分解模块，无参数、不参与训练。
    forward(x) -> (trend, seasonal)，x 为 (B, T, C)。
    """

    def __init__(self, seasonal_period: int = 24):
        super().__init__()
        self.seasonal_period = seasonal_period

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return stl_decompose_btc(x, self.seasonal_period)


__all__ = ["MADecompose", "moving_average_btc", "STLDecompose", "stl_decompose_btc", "stl_decompose_1d"]
