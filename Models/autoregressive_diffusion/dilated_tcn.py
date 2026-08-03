"""
Lightweight dilated TCN residual branch for long-range temporal dependency.

Input / output layout matches ARMD backbone tensors: [B, T, C].
Internally uses Conv1d layout [B, C, T].
"""

import torch
import torch.nn as nn


class DilatedResidualBlock(nn.Module):
    """Depthwise temporal conv + gated pointwise mix, with residual."""

    def __init__(self, channels: int, dilation: int, dropout: float = 0.1, kernel_size: int = 3):
        super().__init__()
        # symmetric padding keeps length: L_out = L + 2*pad - dil*(k-1) = L when pad = dil
        padding = dilation * (kernel_size - 1) // 2
        if padding != dilation:
            # for kernel_size=3, pad = dilation exactly
            padding = dilation

        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels * 2, kernel_size=1)
        self.norm = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        residual = x
        h = self.depthwise(x)
        if h.shape[-1] != residual.shape[-1]:
            # safety crop/pad (should not trigger with correct padding)
            L = residual.shape[-1]
            if h.shape[-1] > L:
                h = h[..., :L]
            else:
                h = nn.functional.pad(h, (0, L - h.shape[-1]))
        value, gate = self.pointwise(h).chunk(2, dim=1)
        h = value * torch.sigmoid(gate)
        h = self.dropout(h)
        h = self.norm(h)
        return residual + h


class LongRangeTCNBranch(nn.Module):
    """
    Dilated TCN residual branch with lightweight FiLM time-step conditioning.

    Args:
        feature_size: variable count C
        timesteps: diffusion steps T_diff (for t normalization)
        hidden_dim: TCN hidden channels
        dilations: dilation schedule
        dropout: residual-block dropout
        out_init_std: if >0, output proj ~ N(0, std); else zeros (strict zero residual)
    """

    def __init__(
        self,
        feature_size: int,
        timesteps: int,
        hidden_dim: int = 64,
        dilations=(1, 2, 4, 8, 16, 32),
        dropout: float = 0.1,
        out_init_std: float = 1e-3,
    ):
        super().__init__()
        self.feature_size = feature_size
        self.timesteps = max(int(timesteps) - 1, 1)
        dilations = tuple(dilations)

        self.input_projection = nn.Conv1d(feature_size, hidden_dim, kernel_size=1)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )
        self.blocks = nn.ModuleList(
            [
                DilatedResidualBlock(hidden_dim, dilation=d, dropout=dropout)
                for d in dilations
            ]
        )
        self.output_projection = nn.Conv1d(hidden_dim, feature_size, kernel_size=1)

        if out_init_std is None or out_init_std <= 0:
            nn.init.zeros_(self.output_projection.weight)
            nn.init.zeros_(self.output_projection.bias)
        else:
            nn.init.normal_(self.output_projection.weight, mean=0.0, std=float(out_init_std))
            nn.init.zeros_(self.output_projection.bias)

        self._debug_printed = False

    def _normalize_t(self, t, batch: int, device, dtype) -> torch.Tensor:
        """Return t_norm as [B, 1] float in [0, 1]."""
        if isinstance(t, int):
            t = torch.full((batch,), t, device=device, dtype=torch.long)
        elif not torch.is_tensor(t):
            t = torch.as_tensor(t, device=device)
        t = t.to(device=device)
        if t.ndim == 0:
            t = t.view(1).expand(batch)
        elif t.ndim == 2 and t.shape[-1] == 1:
            t = t.view(-1)
        if t.shape[0] == 1 and batch > 1:
            t = t.expand(batch)
        t_norm = (t.float() / float(self.timesteps)).view(batch, 1)
        return t_norm.to(dtype=dtype)

    def forward(self, x: torch.Tensor, t) -> torch.Tensor:
        """
        x: [B, T, C]
        t: int | 0-d tensor | [B] | [B, 1]
        returns: [B, T, C]
        """
        if x.ndim != 3:
            raise ValueError(f"LongRangeTCNBranch expects [B, T, C], got {tuple(x.shape)}")
        B, T, C = x.shape
        if C != self.feature_size:
            raise ValueError(
                f"feature_size mismatch: got C={C}, expected {self.feature_size}"
            )

        h = x.permute(0, 2, 1).contiguous()  # [B, C, T]
        h = self.input_projection(h)  # [B, H, T]

        t_norm = self._normalize_t(t, B, x.device, h.dtype)
        scale, shift = self.time_mlp(t_norm).chunk(2, dim=-1)  # [B, H] each
        h = h * (1.0 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)

        for block in self.blocks:
            h = block(h)

        out = self.output_projection(h)  # [B, C, T]
        out = out.permute(0, 2, 1).contiguous()  # [B, T, C]

        if out.shape != x.shape:
            raise RuntimeError(
                f"TCN residual shape {tuple(out.shape)} does not match input {tuple(x.shape)}"
            )
        return out
