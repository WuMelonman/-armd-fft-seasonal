"""
Lightweight Residual TCN for high-frequency residual forecasting.

Not part of the diffusion loop; no diffusion-step conditioning.
Tensor layout matches ARMDTrendWrapper: [B, T, C] (BTN).
"""

import torch
import torch.nn as nn


class ResidualTCNBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        dilation: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(
                "Residual TCN kernel_size must be odd "
                "to preserve sequence length with symmetric padding."
            )
        padding = dilation * (kernel_size - 1) // 2
        self.depthwise = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(
            in_channels=channels,
            out_channels=channels * 2,
            kernel_size=1,
        )
        self.norm = nn.GroupNorm(num_groups=1, num_channels=channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.depthwise(x)
        if h.shape[-1] != residual.shape[-1]:
            L = residual.shape[-1]
            h = h[..., :L] if h.shape[-1] > L else nn.functional.pad(h, (0, L - h.shape[-1]))
        h = self.pointwise(h)
        value, gate = h.chunk(2, dim=1)
        h = value * torch.sigmoid(gate)
        h = self.dropout(h)
        h = self.norm(h)
        if h.shape != residual.shape:
            raise RuntimeError(
                f"Residual block output shape {tuple(h.shape)} "
                f"does not match input shape {tuple(residual.shape)}"
            )
        return residual + h


class ResidualTCNPredictor(nn.Module):
    """
    Hist residual [B, T, C] -> predicted future residual [B, T, C].
    """

    def __init__(
        self,
        feature_size: int,
        hidden_dim: int = 32,
        dilations=(1, 2, 4, 8),
        kernel_size: int = 3,
        dropout: float = 0.1,
        input_layout: str = "BTN",
    ):
        super().__init__()
        self.feature_size = feature_size
        self.input_layout = input_layout
        self.dilations = tuple(dilations)

        self.input_projection = nn.Conv1d(
            in_channels=feature_size,
            out_channels=hidden_dim,
            kernel_size=1,
        )
        self.blocks = nn.ModuleList(
            [
                ResidualTCNBlock(
                    channels=hidden_dim,
                    dilation=d,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
                for d in self.dilations
            ]
        )
        self.output_projection = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=feature_size,
            kernel_size=1,
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        nn.init.xavier_uniform_(self.output_projection.weight, gain=0.1)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected a 3D tensor, got {tuple(x.shape)}")
        original_shape = x.shape

        if self.input_layout == "BTN":
            x_conv = x.transpose(1, 2).contiguous()  # [B, C, T]
        elif self.input_layout == "BNT":
            x_conv = x
        else:
            raise ValueError(f"Unsupported input layout: {self.input_layout}")

        if x_conv.size(1) != self.feature_size:
            raise ValueError(
                f"Expected feature dimension {self.feature_size}, got {x_conv.size(1)}"
            )

        h = self.input_projection(x_conv)
        for block in self.blocks:
            h = block(h)
        output = self.output_projection(h)

        if self.input_layout == "BTN":
            output = output.transpose(1, 2).contiguous()

        if output.shape != original_shape:
            raise RuntimeError(
                f"Residual TCN output shape {tuple(output.shape)} "
                f"does not match input shape {tuple(original_shape)}"
            )
        return output
