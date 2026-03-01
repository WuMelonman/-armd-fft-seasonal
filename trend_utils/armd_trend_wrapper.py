import torch
import torch.nn as nn


class ResidualStripper(nn.Module):
    """
    从原序列中剥离 ARMD 不擅长建模的高频部分。
    高通核固定为二阶差分 [-1, 2, -1]，仅训练 alpha；z = x - alpha * high。
    """

    def __init__(self, feature_size: int):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            padding=1,
            groups=feature_size,
            bias=False,
        )
        # 固定为二阶差分核 [-1, 2, -1]，每个通道相同
        with torch.no_grad():
            w = self.conv.weight  # (C, 1, 3)
            w.zero_()
            w[:, 0, 0] = -1.0
            w[:, 0, 1] = 2.0
            w[:, 0, 2] = -1.0
        self.conv.weight.requires_grad = False
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        x_bct = x.permute(0, 2, 1)   # (B, C, T)
        high = self.conv(x_bct)       # (B, C, T)
        high_btc = high.permute(0, 2, 1)  # (B, T, C)
        z = x - self.alpha * high_btc
        return z


class ARMDTrendWrapper(nn.Module):
    """
    Wrap ARMD with ResidualStripper: strip high-frequency then pass to ARMD.
    Trainer keeps: loss = model(data, target=data)
    data: (B, seq_length, C), seq_length = pred_len * 2.
    """

    def __init__(self, armd: nn.Module, feature_size: int = None):
        super().__init__()
        self.armd = armd

        if not hasattr(self.armd, "pred_len"):
            raise ValueError("ARMD model must expose pred_len")
        self.pred_len = self.armd.pred_len

        feat = feature_size if feature_size is not None else getattr(self.armd, "feature_size", None)
        if feat is None:
            raise ValueError("ARMDTrendWrapper expects `armd` to have `feature_size` or pass feature_size.")
        self.feature_size = feat

        self.stripper = ResidualStripper(self.feature_size)

    def forward(self, data: torch.Tensor, **kwargs):
        H = self.pred_len
        z = self.stripper(data)
        real_target = data[:, H:, :]
        kwargs.pop("target", None)
        return self.armd(z, target=real_target, **kwargs)

    @torch.no_grad()
    def generate_mts(self, x: torch.Tensor, **kwargs):
        x = self.stripper(x)
        return self.armd.generate_mts(x, **kwargs)

    @property
    def fast_sampling(self):
        return self.armd.fast_sampling

    @fast_sampling.setter
    def fast_sampling(self, v: bool):
        self.armd.fast_sampling = v

    def __getattr__(self, name):
        if name in {"armd", "stripper", "__getstate__", "__setstate__"}:
            return super().__getattr__(name)
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.armd, name)
