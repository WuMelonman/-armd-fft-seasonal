"""
ARMD + adaptive MA decomposition:
- Trend → ARMD diffusion (unchanged)
- High-frequency residual → FFT top-k (default) or Residual TCN (optional)

Tensor layout: [B, T, C]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce

from trend_utils.trend_conv import moving_average_btc
from Models.autoregressive_diffusion.residual_tcn import ResidualTCNPredictor


def estimate_adaptive_kernels(
    x: torch.Tensor,
    min_kernel: int = 5,
    max_kernel: int = 49,
    exclude_dc: bool = True,
) -> torch.Tensor:
    """
    Per-variable dominant-period → odd MA kernel in [min_kernel, max_kernel].
    x: [B, T, C]  →  kernels: [C] long
    """
    B, T, C = x.shape
    device = x.device
    kernels = []
    for j in range(C):
        s = x[:, :, j]
        spec = torch.fft.rfft(s, dim=-1)
        mag_avg = torch.abs(spec).mean(dim=0)
        if exclude_dc and mag_avg.numel() > 0:
            mag_avg = mag_avg.clone()
            mag_avg[0] = -1
        peak_idx = int(torch.argmax(mag_avg).item())
        if peak_idx <= 0:
            kernel = min_kernel
        else:
            period = round(T / peak_idx)
            kernel = max(min_kernel, min(max_kernel, period))
        if kernel % 2 == 0:
            kernel += 1
        kernels.append(kernel)
    return torch.tensor(kernels, device=device, dtype=torch.long)


def apply_ma_with_kernels(x: torch.Tensor, kernels: torch.Tensor):
    """
    Apply per-variable MA with given kernels.
    x: [B, T, C], kernels: [C] → trend, residual [B, T, C]
    """
    B, T, C = x.shape
    if kernels.numel() != C:
        raise ValueError(f"kernels length {kernels.numel()} != C={C}")
    trend_parts = []
    for j in range(C):
        k = int(kernels[j].item())
        trend_j, _ = moving_average_btc(x[:, :, j : j + 1], kernel_size=k)
        trend_parts.append(trend_j)
    trend = torch.cat(trend_parts, dim=-1)
    residual = x - trend
    return trend, residual


def adaptive_moving_average_btc(
    x: torch.Tensor,
    min_kernel: int = 5,
    max_kernel: int = 49,
    exclude_dc: bool = True,
    kernels: torch.Tensor = None,
):
    """
    对每个变量按主频自适应 MA 核大小分解。
    x: (B, T, C)，返回 trend, seasonal/residual, kernels (C,)。
    若传入 kernels，则跳过主频估计并复用该核（用于 history/target 一致分解）。
    """
    if kernels is None:
        kernels = estimate_adaptive_kernels(
            x, min_kernel=min_kernel, max_kernel=max_kernel, exclude_dc=exclude_dc
        )
    trend, residual = apply_ma_with_kernels(x, kernels)
    return trend, residual, kernels


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
    包装 ARMD：
    - 趋势：自适应/固定 MA → ARMD 扩散（训练与预测）
    - 高频残差：FFT top-k（默认，兼容旧行为）或 Residual TCN
    """

    def __init__(
        self,
        armd: nn.Module,
        feature_size: int = None,
        ma_kernel_size: int = 25,
        fft_topk: int = 5,
        use_nlinear: bool = False,
        adaptive_ma: bool = True,
        ma_min_kernel: int = 5,
        ma_max_kernel: int = 49,
        residual_predictor: str = "fft",
        residual_tcn_hidden_dim: int = 32,
        residual_tcn_dilations=(1, 2, 4, 8),
        residual_tcn_kernel_size: int = 3,
        residual_tcn_dropout: float = 0.1,
        final_loss_weight: float = 1.0,
        final_mae_weight: float = 0.5,
        trend_loss_weight: float = 0.2,
        residual_loss_weight: float = 0.1,
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
        self.adaptive_ma = adaptive_ma
        self.ma_min_kernel = ma_min_kernel
        self.ma_max_kernel = ma_max_kernel

        self.residual_predictor = str(residual_predictor).lower()
        if self.residual_predictor not in {"fft", "tcn"}:
            raise ValueError(f"Unsupported residual predictor: {residual_predictor}")

        if isinstance(residual_tcn_dilations, list):
            residual_tcn_dilations = tuple(residual_tcn_dilations)

        self.final_loss_weight = float(final_loss_weight)
        self.final_mae_weight = float(final_mae_weight)
        # trend_loss_weight: FFT 模式忽略（保持原 armd loss）；TCN 模式乘在 diffusion trend loss 上
        self.trend_loss_weight = float(trend_loss_weight)
        self.residual_loss_weight = float(residual_loss_weight)

        if self.residual_predictor == "tcn":
            self.residual_tcn = ResidualTCNPredictor(
                feature_size=self.feature_size,
                hidden_dim=int(residual_tcn_hidden_dim),
                dilations=residual_tcn_dilations,
                kernel_size=int(residual_tcn_kernel_size),
                dropout=float(residual_tcn_dropout),
                input_layout="BTN",
            )
        else:
            self.residual_tcn = None

        self._residual_debug_printed = False
        self._kernel_debug_printed = False
        self._last_loss_stats = None

    # ------------------------------------------------------------------ #
    # Unified adaptive / fixed MA decompose (train / sample / plot)
    # ------------------------------------------------------------------ #
    def estimate_kernels(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate per-variable MA kernels from x ([B,T,C])."""
        if self.adaptive_ma:
            return estimate_adaptive_kernels(
                x, min_kernel=self.ma_min_kernel, max_kernel=self.ma_max_kernel
            )
        C = x.shape[-1]
        k = int(self.ma_kernel_size)
        if k % 2 == 0:
            k += 1
        return torch.full((C,), k, device=x.device, dtype=torch.long)

    def decompose(self, x: torch.Tensor, kernels=None, return_kernel_sizes: bool = False):
        """
        Public decompose API shared by train / sample / plot.
        x: [B, T, C]
        """
        if kernels is None:
            kernels = self.estimate_kernels(x)
        trend, residual = apply_ma_with_kernels(x, kernels)
        if return_kernel_sizes:
            return trend, residual, kernels
        return trend, residual

    def predict_residual(self, residual_history: torch.Tensor) -> torch.Tensor:
        """Predict future residual from history residual [B, H, C] → [B, H, C]."""
        if self.residual_predictor == "fft":
            return fft_topk_forecast(
                residual_history, self.pred_len, topk=self.fft_topk
            )
        if self.residual_predictor == "tcn":
            return self.residual_tcn(residual_history)
        raise ValueError(f"Unsupported residual predictor: {self.residual_predictor}")

    # ------------------------------------------------------------------ #
    # Trend diffusion helper (TCN mode only; mirrors ARMD._train_loss)
    # ------------------------------------------------------------------ #
    def _trend_pred_and_loss(self, trend_full: torch.Tensor, target_trend: torch.Tensor):
        """One diffusion step: return (trend_pred [B,H,C], scalar loss)."""
        armd = self.armd
        b = trend_full.shape[0]
        device = trend_full.device
        t = torch.randint(0, armd.num_timesteps, (1,), device=device).repeat(b).long()
        x = armd.q_sample(x_start=trend_full, t=t)
        x_start_base = armd.output(x, t, training=True)
        x_start_coupled, *_ = armd.coupled_trend_from_xstart(x, t, x_start_base)
        trend_pred = x_start_base + armd.couple_alpha * (x_start_coupled - x_start_base)
        loss_pred = armd.loss_fn(trend_pred, target_trend, reduction="none")
        loss_trend = reduce(loss_pred, "b ... -> b (...)", "mean").mean()
        return trend_pred, loss_trend

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def forward(self, data: torch.Tensor, **kwargs):
        H = self.pred_len
        kwargs.pop("target", None)

        # -------- FFT mode: keep legacy behavior exactly (incl. full-window MA) --------
        if self.residual_predictor == "fft":
            if self.adaptive_ma:
                trend, _, kernels = adaptive_moving_average_btc(
                    data, self.ma_min_kernel, self.ma_max_kernel
                )
            else:
                trend, _ = moving_average_btc(data, kernel_size=self.ma_kernel_size)
                kernels = None
            if kernels is not None and not self._kernel_debug_printed:
                print(
                    "[Adaptive MA] kernel sizes:",
                    kernels.detach().cpu().tolist(),
                    flush=True,
                )
                self._kernel_debug_printed = True

            if self.use_nlinear:
                trend_full = trend
                if self.adaptive_ma:
                    trend_hist, _, _ = adaptive_moving_average_btc(
                        data[:, :H, :], self.ma_min_kernel, self.ma_max_kernel
                    )
                else:
                    trend_hist, _ = moving_average_btc(
                        data[:, :H, :], kernel_size=self.ma_kernel_size
                    )
                last = trend_hist[:, -1:, :]
                trend_centered_input = trend_hist - last
                real_target_trend = trend_full[:, H:, :] - last
                return self.armd(trend_centered_input, target=real_target_trend, **kwargs)

            real_target_trend = trend[:, H:, :]
            return self.armd(trend, target=real_target_trend, **kwargs)

        # -------- TCN mode: leak-free kernels from history only --------
        hist = data[:, :H, :]
        target = data[:, H:, :]
        kernels = self.estimate_kernels(hist)
        if not self._kernel_debug_printed:
            print(
                "[Adaptive MA] kernel sizes:",
                kernels.detach().cpu().tolist(),
                flush=True,
            )
            self._kernel_debug_printed = True

        hist_trend, hist_residual = apply_ma_with_kernels(hist, kernels)
        # same kernels for target labels (no re-estimation on target)
        tgt_trend, tgt_residual = apply_ma_with_kernels(target, kernels)

        if self.use_nlinear:
            last = hist_trend[:, -1:, :]
            trend_input = hist_trend - last  # [B, H, C]
            real_target_trend = tgt_trend - last
            trend_pred_c, loss_trend = self._trend_pred_and_loss(trend_input, real_target_trend)
            trend_pred = trend_pred_c + last
        else:
            trend_full = torch.cat([hist_trend, tgt_trend], dim=1)  # [B, 2H, C]
            trend_pred, loss_trend = self._trend_pred_and_loss(trend_full, tgt_trend)

        residual_pred = self.predict_residual(hist_residual)
        if not self._residual_debug_printed:
            print(
                "[Residual Predictor]\n"
                f"mode: tcn\n"
                f"input shape: {tuple(hist_residual.shape)}\n"
                f"output shape: {tuple(residual_pred.shape)}\n"
                f"adaptive kernel sizes: {kernels.detach().cpu().tolist()}\n"
                f"residual target shape: {tuple(tgt_residual.shape)}",
                flush=True,
            )
            self._residual_debug_printed = True

        final_pred = trend_pred + residual_pred
        loss_residual = F.l1_loss(residual_pred, tgt_residual)
        loss_final = F.mse_loss(final_pred, target) + self.final_mae_weight * F.l1_loss(
            final_pred, target
        )
        loss_trend_mae = F.l1_loss(trend_pred, tgt_trend)

        # L = L_trend-diffusion + λ_T * L_trend-MAE + λ_F * L_final + λ_R * L_residual
        loss = (
            loss_trend
            + self.trend_loss_weight * loss_trend_mae
            + self.final_loss_weight * loss_final
            + self.residual_loss_weight * loss_residual
        )

        self._last_loss_stats = {
            "loss_total": float(loss.detach().item()),
            "loss_trend": float(loss_trend.detach().item()),
            "loss_residual": float(loss_residual.detach().item()),
            "loss_final": float(loss_final.detach().item()),
            "residual_pred_abs_mean": float(residual_pred.detach().abs().mean().item()),
            "target_residual_abs_mean": float(tgt_residual.detach().abs().mean().item()),
        }
        return loss

    # ------------------------------------------------------------------ #
    # Sampling / forecasting
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate_mts(self, x: torch.Tensor, **kwargs):
        H = self.pred_len

        # -------- FFT mode: preserve legacy (possibly leaky) path --------
        if self.residual_predictor == "fft":
            if self.use_nlinear:
                x_hist = x[:, :H, :]
                if self.adaptive_ma:
                    trend_hist, seasonal_hist, _ = adaptive_moving_average_btc(
                        x_hist, self.ma_min_kernel, self.ma_max_kernel
                    )
                else:
                    trend_hist, seasonal_hist = moving_average_btc(
                        x_hist, kernel_size=self.ma_kernel_size
                    )
                last = trend_hist[:, -1:, :]
                trend_centered_hist = trend_hist - last
                trend_pred = self.armd.generate_mts(trend_centered_hist, **kwargs) + last
                seasonal_pred = fft_topk_forecast(
                    seasonal_hist, self.pred_len, topk=self.fft_topk
                )
            else:
                if self.adaptive_ma:
                    trend, seasonal, _ = adaptive_moving_average_btc(
                        x, self.ma_min_kernel, self.ma_max_kernel
                    )
                else:
                    trend, seasonal = moving_average_btc(x, kernel_size=self.ma_kernel_size)
                trend_pred = self.armd.generate_mts(trend, **kwargs)
                seasonal_pred = fft_topk_forecast(
                    seasonal, self.pred_len, topk=self.fft_topk
                )
            return trend_pred + seasonal_pred

        # -------- TCN mode: kernels / MA from history only --------
        hist = x[:, :H, :]
        kernels = self.estimate_kernels(hist)
        hist_trend, hist_residual = apply_ma_with_kernels(hist, kernels)

        if self.use_nlinear:
            last = hist_trend[:, -1:, :]
            trend_pred = self.armd.generate_mts(hist_trend - last, **kwargs) + last
        else:
            # generate_mts only consumes first pred_len of its input
            trend_pred = self.armd.generate_mts(hist_trend, **kwargs)

        residual_pred = self.predict_residual(hist_residual)
        return trend_pred + residual_pred

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
