import torch
import torch.nn as nn
import torch.nn.functional as F

# 你的 TrendConvNet 和 moving_average_target 直接复用
# from your_file import TrendConvNet, moving_average_target
from trend_utils.trend_conv import TrendConvNet, moving_average_target
def second_diff_smoothness(trend_btc: torch.Tensor) -> torch.Tensor:
    """
    ||Δ^2 trend||^2
    trend_btc: (B, T, C)
    """
    d1 = trend_btc[:, 1:] - trend_btc[:, :-1]
    d2 = d1[:, 1:] - d1[:, :-1]
    return (d2 ** 2).mean()
class ARMDTrendWrapper(nn.Module):
    """
    Wrap your ARMD so Trainer can keep doing:
        loss = model(data, target=data)

    data is expected to be (B, seq_length, C) where seq_length = pred_len * 2 in your code.
    """

    def __init__(
        self,
        armd: nn.Module,
        trend_conv: nn.Module,
        lambda_smooth: float = 0.0,
        lambda_ma_init: float = 0.0,
        ma_kernel: int = 25,
        detach_trend: bool = False,
        season_top_k: int = 5,
    ):
        super().__init__()
        self.armd = armd
        self.trend_conv = trend_conv

        self.lambda_smooth = float(lambda_smooth)
        self.lambda_ma_init = float(lambda_ma_init)
        self.ma_kernel = int(ma_kernel)

        # 如果你想先让 ARMD 稳定、暂时不回传梯度到 trend，可以 True
        # 端到端训练就 False
        self.detach_trend = bool(detach_trend)

        # Top-K FFT seasonal forecasting hyperparameter.
        # season_top_k <= 0 will fall back to simple persistence.
        self.season_top_k = int(season_top_k)

    def _trend(self, x_btc: torch.Tensor) -> torch.Tensor:
        # x: (B,T,C) -> (B,C,T) -> conv -> (B,T,C)
        tau = self.trend_conv(x_btc.permute(0, 2, 1)).permute(0, 2, 1)
        if self.detach_trend:
            tau = tau.detach()
        return tau

    def _seasonal_fft_predict(self, s_hist: torch.Tensor, pred_len: int) -> torch.Tensor:
        """
        Top-K FFT seasonal forecasting.

        s_hist: (B, H, C) seasonal / residual component from history
        pred_len: number of future steps to predict (H)
        """
        top_k = getattr(self, "season_top_k", 0)
        if top_k is None or top_k <= 0:
            # Fallback to simple persistence
            return s_hist[:, -pred_len:, :]

        B, H, C = s_hist.shape
        if H <= 0:
            raise ValueError("History length must be > 0 for FFT seasonal forecasting.")

        # (B, H, C) -> (B, C, H)
        x_bch = s_hist.permute(0, 2, 1)

        # Real FFT along time dimension
        freqs = torch.fft.rfft(x_bch, dim=-1)  # (B, C, F)

        # Keep only Top-K frequencies by magnitude for each (B, C)
        F_dim = freqs.shape[-1]
        if top_k < F_dim:
            mag = freqs.abs()
            # indices: (B, C, top_k)
            topk_idx = mag.topk(top_k, dim=-1).indices
            mask = torch.zeros_like(freqs, dtype=torch.bool)
            mask.scatter_(-1, topk_idx, True)
            freqs = freqs * mask

        # Inverse FFT to reconstruct smoothed seasonal history of length H
        s_smooth_bch = torch.fft.irfft(freqs, n=H, dim=-1)
        s_smooth = s_smooth_bch.permute(0, 2, 1)  # (B, H, C)

        # Periodic extension to future horizon
        if pred_len <= H:
            s_pred = s_smooth[:, -pred_len:, :]
        else:
            repeat = (pred_len + H - 1) // H
            s_tiled = s_smooth.repeat(1, repeat, 1)
            s_pred = s_tiled[:, :pred_len, :]

        return s_pred

    def forward(self, data: torch.Tensor, **kwargs):
        """
        Return a scalar total loss = armd_loss + reg_loss
        kwargs will pass into ARMD.forward -> _train_loss
        """
        # 1) online trend decomposition
        tau = self._trend(data)

        # 2) ARMD trains on trend-domain sequence only
        #    IMPORTANT: your ARMD ignores external target and uses x_start[:, pred_len:,:] internally,
        #    so this will make it predict future trend from trend history.
        armd_loss = self.armd(tau, **kwargs)

        # 3) trend regularization (must-have)
        reg = 0.0
        if self.lambda_smooth > 0:
            reg = reg + self.lambda_smooth * second_diff_smoothness(tau)

        # 4) optional MA "soft anchor" (warm-up, not a hard pretrain)
        if self.lambda_ma_init > 0:
            x_bct = data.permute(0, 2, 1)
            ma = moving_average_target(x_bct, self.ma_kernel).permute(0, 2, 1)
            reg = reg + self.lambda_ma_init * F.mse_loss(tau, ma)

        return armd_loss + reg

    @torch.no_grad()
    def generate_mts(self, x: torch.Tensor, oracle_season: bool = False):
        print("WRAPPER generate_mts called")
        H = self.pred_len if hasattr(self, "pred_len") else x.shape[1] // 2

        x_hist = x[:, :H, :]
        x_fut = x[:, H:, :]  # future ground truth (ONLY for oracle/debug)

        # 1) history decomposition
        tau_hist = self._trend(x_hist)
        s_hist = x_hist - tau_hist

        # 2) trend prediction (no-leak) on trend-domain only
        tau_pred = self.armd.generate_mts(tau_hist)  # (B, H, C)

        # 3) season prediction
        if oracle_season:
            # ⚠️ ORACLE: season constructed from FUTURE truth
            # Use the same trend filter on future (still leakage)
            tau_fut = self._trend(x_fut)
            s_pred = x_fut - tau_fut
        else:
            # ✅ NO-LEAK: Top-K FFT seasonal forecasting from HISTORY only
            s_pred = self._seasonal_fft_predict(s_hist, pred_len=H)

        # 4) full prediction: y = trend + seasonal
        y_pred = tau_pred + s_pred
        return y_pred

    # 下面两个属性让你 main 里 model.fast_sampling = True 仍然工作
    @property
    def fast_sampling(self):
        return self.armd.fast_sampling

    @fast_sampling.setter
    def fast_sampling(self, v: bool):
        self.armd.fast_sampling = v

    def __getattr__(self, name):
        """
        Delegate missing attributes to the wrapped ARMD.
        This makes Trainer's self.model.betas / self.model.num_timesteps etc. work.
        """
        if name in {"armd", "trend_conv", "__getstate__", "__setstate__"}:
            return super().__getattr__(name)
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.armd, name)
