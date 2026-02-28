import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List


class TrendConvNet(nn.Module):
    """Depthwise 1D convolution used to learn a smooth trend component."""

    def __init__(self, feature_size: int, kernel_size: int = 25):  # ✅ default 25
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


def moving_average_target(x: torch.Tensor, kernel_size: int = 25) -> torch.Tensor:  # ✅ default 25
    """Fixed moving average as the pseudo target for the trend smoother."""

    b, c, _ = x.shape
    weight = torch.ones(c, 1, kernel_size, device=x.device, dtype=x.dtype) / kernel_size
    padding = kernel_size // 2
    return F.conv1d(x, weight, padding=padding, groups=c)


def train_trend_conv(
    model: TrendConvNet,
    dataloader,
    device,
    epochs: int = 3,
    kernel_size: int = 25,  # ✅ default 25
    lr: float = 1e-3,
):
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

            # ✅ target MA uses same kernel_size=25
            target = moving_average_target(x_ch_first, kernel_size=kernel_size)

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
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True
    )


__all__ = ["TrendConvNet", "moving_average_target", "train_trend_conv", "build_trend_loader"]
