"""
ARMD + 固定分解（MA 得 trend / seasonal = x - trend）：
- 季节 → ARMD（训练与条件生成）
- 趋势 → FFT top-k 谐波外推（与原先 trend/season 分工对调）
"""

import torch
import torch.nn as nn

from trend_utils.trend_conv import moving_average_btc


def fft_topk_forecast(
    hist: torch.Tensor,
    pred_len: int,
    topk: int = 5,
    exclude_dc: bool = True,
) -> torch.Tensor:
    """
    用 FFT 取 top-k 主频，外推未来 pred_len 步（可用于季节或趋势等序列）。
    hist: (B, T, C)
    返回: (B, pred_len, C)
    注意：实序列 rfft 重建时需用归一化 2/T（DC 与 Nyquist 用 1/T），否则幅值尺度错误。
    """
    B, T, C = hist.shape
    device = hist.device
    dtype = hist.dtype
    nf = (T // 2) + 1  # rfft 长度
    # (B, T, C) -> (B, C, T)
    s = hist.permute(0, 2, 1)  # (B, C, T)
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
    包装 ARMD：固定 MA 分解后，季节 = x - trend。
    - 季节 → ARMD（训练与 generate_mts）
    - 趋势 → FFT top-k 外推，再与季节预测相加
    """

    def __init__(
        self,
        armd: nn.Module,
        feature_size: int = None,
        ma_kernel_size: int = 25,
        fft_topk: int = 5,
        use_nlinear: bool = False,
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
        self.use_nlinear = use_nlinear

    def forward(self, data: torch.Tensor, **kwargs):
        H = self.pred_len
        trend, seasonal = moving_average_btc(data, kernel_size=self.ma_kernel_size)
        kwargs.pop("target", None)
        if self.use_nlinear:
            # NLinear：对季节分量 last 去中心，仅历史窗进模型
            seasonal_full = seasonal
            _, seasonal_hist = moving_average_btc(data[:, :H, :], kernel_size=self.ma_kernel_size)
            last = seasonal_hist[:, -1:, :]
            seasonal_centered_input = seasonal_hist - last
            real_target_seasonal = seasonal_full[:, H:, :] - last
            return self.armd(seasonal_centered_input, target=real_target_seasonal, **kwargs)
        else:
            real_target_seasonal = seasonal[:, H:, :]
            return self.armd(seasonal, target=real_target_seasonal, **kwargs)

    @torch.no_grad()
    def generate_mts(self, x: torch.Tensor, **kwargs):
        H = self.pred_len
        if self.use_nlinear:
            x_hist = x[:, :H, :]
            trend_hist, seasonal_hist = moving_average_btc(x_hist, kernel_size=self.ma_kernel_size)
            last = seasonal_hist[:, -1:, :]
            seasonal_centered_hist = seasonal_hist - last
            seasonal_pred = self.armd.generate_mts(seasonal_centered_hist, **kwargs) + last
            trend_pred = fft_topk_forecast(trend_hist, self.pred_len, topk=self.fft_topk)
        else:
            trend, seasonal = moving_average_btc(x, kernel_size=self.ma_kernel_size)
            seasonal_pred = self.armd.generate_mts(seasonal, **kwargs)
            trend_pred = fft_topk_forecast(trend, self.pred_len, topk=self.fft_topk)
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
