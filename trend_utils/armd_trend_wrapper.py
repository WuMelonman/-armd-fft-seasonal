"""
ARMD + 固定分解（趋势用 MA 核 25，季节用 FFT top-k）：
- 趋势 → ARMD 原模型（训练与预测）
- 季节 → FFT top-k 外推（仅预测时加回）
"""

import torch
import torch.nn as nn

from trend_utils.trend_conv import moving_average_btc


def fft_topk_forecast(
    seasonal: torch.Tensor,
    pred_len: int,
    topk: int = 5,
    exclude_dc: bool = True,
) -> torch.Tensor:
    """
    用 FFT 取 top-k 主频，外推得到未来 pred_len 步的季节分量。
    seasonal: (B, T, C)，历史季节分量
    返回: (B, pred_len, C)
    注意：实序列 rfft 重建时需用归一化 2/T（DC 与 Nyquist 用 1/T），否则幅值尺度错误。
    """
    B, T, C = seasonal.shape
    device = seasonal.device
    dtype = seasonal.dtype
    nf = (T // 2) + 1  # rfft 长度
    # (B, T, C) -> (B, C, T)
    s = seasonal.permute(0, 2, 1)  # (B, C, T)
    spec = torch.fft.rfft(s, dim=-1)  # (B, C, nf)
    mag = torch.abs(spec)
    if topk >= nf - (1 if exclude_dc else 0):
        topk = max(1, nf - (1 if exclude_dc else 0))
    mag_sel = mag.clone()
    if exclude_dc:
        mag_sel[:, :, 0] = -1
    topk_mag, topk_idx = torch.topk(mag_sel, topk, dim=-1)  # (B, C, topk)

    # 实序列 rfft：bin 0 为 DC，bin nf-1 为 Nyquist，其余为 2/T 倍幅值
    t_future = torch.arange(pred_len, device=device, dtype=dtype) + T
    t_future = t_future.unsqueeze(0).unsqueeze(0)  # (1, 1, pred_len)
    forecast = torch.zeros(B, C, pred_len, device=device, dtype=dtype)
    for k in range(topk):
        idx = topk_idx[:, :, k]  # (B, C)
        amp_raw = torch.abs(spec).gather(-1, idx.unsqueeze(-1)).squeeze(-1)  # (B, C)
        # 归一化：DC 和 Nyquist 用 1/T，其余用 2/T
        scale = torch.where(
            (idx == 0) | (idx == nf - 1),
            torch.full_like(amp_raw, 1.0 / T, device=device, dtype=dtype),
            torch.full_like(amp_raw, 2.0 / T, device=device, dtype=dtype),
        )
        amp = amp_raw * scale
        phase = torch.angle(spec).gather(-1, idx.unsqueeze(-1)).squeeze(-1)  # (B, C)
        k_float = idx.to(dtype)
        angle = 2 * 3.141592653589793 * k_float.unsqueeze(-1) * t_future / T + phase.unsqueeze(-1)
        forecast += amp.unsqueeze(-1) * torch.cos(angle)
    return forecast.permute(0, 2, 1)  # (B, pred_len, C)


class ARMDTrendWrapper(nn.Module):
    """
    包装 ARMD：趋势用固定 MA（核大小 25），季节 = x - trend，用 FFT top-k 外推。
    - 趋势 → ARMD 原模型（训练与预测）
    - 季节 → FFT top-k 外推（仅预测时加回）
    """

    def __init__(
        self,
        armd: nn.Module,
        feature_size: int = None,
        ma_kernel_size: int = 25,
        fft_topk: int = 5,
    ):
        super().__init__()
        self.armd = armd
        if not hasattr(self.armd, "pred_len"):
            raise ValueError("ARMD model must expose pred_len")
        self.pred_len = self.armd.pred_len
        feat = feature_size if feature_size is not None else getattr(self.armd, "feature_size", None)
        if feat is None:
            raise ValueError("ARMDTrendWrapper expects armd to have feature_size or pass feature_size.")
        self.feature_size = feat
        self.ma_kernel_size = ma_kernel_size
        self.fft_topk = fft_topk

    def forward(self, data: torch.Tensor, **kwargs):
        # 固定 MA 分解，不参与训练
        trend, seasonal = moving_average_btc(data, kernel_size=self.ma_kernel_size)
        H = self.pred_len
        real_target_trend = trend[:, H:, :]
        kwargs.pop("target", None)
        return self.armd(trend, target=real_target_trend, **kwargs)

    @torch.no_grad()
    def generate_mts(self, x: torch.Tensor, **kwargs):
        trend, seasonal = moving_average_btc(x, kernel_size=self.ma_kernel_size)
        trend_pred = self.armd.generate_mts(trend, **kwargs)  # (B, pred_len, C)
        seasonal_pred = fft_topk_forecast(seasonal, self.pred_len, topk=self.fft_topk)  # (B, pred_len, C)
        return trend_pred + seasonal_pred

    @property
    def fast_sampling(self):
        return self.armd.fast_sampling

    @fast_sampling.setter
    def fast_sampling(self, v: bool):
        self.armd.fast_sampling = v

    def __getattr__(self, name):
        if name in {"armd", "__getstate__", "__setstate__"}:
            return super().__getattr__(name)
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.armd, name)
