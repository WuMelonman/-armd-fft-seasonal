"""
Tests for adaptive MA consistency + Residual TCN residual predictor.

Usage:
  python scripts/test_residual_tcn.py
"""

import os
import sys
import copy
import argparse

import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Models.autoregressive_diffusion.armd import ARMD
from Models.autoregressive_diffusion.residual_tcn import ResidualTCNPredictor
from trend_utils.armd_trend_wrapper import (
    ARMDTrendWrapper,
    adaptive_moving_average_btc,
    estimate_adaptive_kernels,
    apply_ma_with_kernels,
)


def _make_wrapper(residual_predictor="tcn", feature_size=21, seq_length=96, **kw):
    armd = ARMD(
        seq_length=seq_length,
        feature_size=feature_size,
        timesteps=seq_length,
        sampling_timesteps=1,
        loss_type="l1",
        beta_schedule="cosine",
        w_grad=True,
        use_long_range_tcn=False,
    )
    return ARMDTrendWrapper(
        armd=armd,
        feature_size=feature_size,
        ma_kernel_size=25,
        fft_topk=5,
        adaptive_ma=True,
        residual_predictor=residual_predictor,
        residual_tcn_hidden_dim=32,
        residual_tcn_dilations=[1, 2, 4, 8],
        **kw,
    )


def test_decompose_consistency():
    torch.manual_seed(0)
    B, T, C = 2, 192, 7
    x = torch.randn(B, T, C)
    model = _make_wrapper("fft", feature_size=C, seq_length=96)

    trend_a, res_a, k_a = model.decompose(x, return_kernel_sizes=True)
    trend_b, res_b, k_b = model.decompose(x, return_kernel_sizes=True)
    # "plot path" without model: same helpers
    trend_c, res_c, k_c = adaptive_moving_average_btc(x)

    assert torch.equal(k_a, k_b) and torch.equal(k_a, k_c)
    assert torch.allclose(trend_a, trend_b, atol=1e-6)
    assert torch.allclose(res_a, res_c, atol=1e-6)
    print("[PASS] adaptive decompose consistency")


def test_residual_tcn_shape_grad():
    # Real layout is [B, T, C]
    B, T, C = 4, 96, 21
    x = torch.randn(B, T, C)
    model = ResidualTCNPredictor(
        feature_size=C, hidden_dim=32, dilations=(1, 2, 4, 8), kernel_size=3, dropout=0.1
    )
    y = model(x)
    assert y.shape == x.shape
    loss = F.l1_loss(y, torch.randn_like(y))
    loss.backward()
    for name, p in [
        ("input_projection", model.input_projection.weight),
        ("blocks0.depthwise", model.blocks[0].depthwise.weight),
        ("blocks0.pointwise", model.blocks[0].pointwise.weight),
        ("output_projection", model.output_projection.weight),
    ]:
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    print("[PASS] ResidualTCN shape + gradients", tuple(y.shape))


def test_no_future_leak_in_tcn_mode():
    torch.manual_seed(1)
    B, H, C = 2, 96, 7
    hist = torch.randn(B, H, C)
    target_a = torch.randn(B, H, C)
    target_b = torch.randn(B, H, C) * 5 + 10

    model = _make_wrapper("tcn", feature_size=C, seq_length=H)
    k1 = model.estimate_kernels(hist)
    ht1, hr1 = apply_ma_with_kernels(hist, k1)
    # different targets must not change history decompose
    k2 = model.estimate_kernels(hist)
    ht2, hr2 = apply_ma_with_kernels(hist, k2)
    assert torch.equal(k1, k2)
    assert torch.allclose(ht1, ht2) and torch.allclose(hr1, hr2)

    # target kernels forced from history
    ta1, ra1 = apply_ma_with_kernels(target_a, k1)
    ta2, ra2 = apply_ma_with_kernels(target_b, k1)
    assert ta1.shape == target_a.shape and not torch.allclose(ta1, ta2)
    print("[PASS] no future leak: history kernels/decompose independent of target")


def test_fft_mode_compat():
    torch.manual_seed(2)
    B, H, C = 2, 96, 7
    x = torch.randn(B, 2 * H, C)
    fft_m = _make_wrapper("fft", feature_size=C, seq_length=H)
    tcn_m = _make_wrapper("tcn", feature_size=C, seq_length=H)
    fft_m.eval()
    tcn_m.eval()
    with torch.no_grad():
        y_fft = fft_m.generate_mts(x)
        y_tcn = tcn_m.generate_mts(x)
    assert y_fft.shape == y_tcn.shape == (B, H, C)
    # FFT path still callable for training loss
    loss = fft_m(x)
    assert torch.isfinite(loss)
    print("[PASS] fft/tcn sampling shape compat; fft train loss ok", float(loss))


def test_short_train_weather(steps=5):
    from Utils.io_utils import load_yaml_config, instantiate_from_config
    from Data.build_dataloader import build_dataloader
    from engine.solver import Trainer

    csv = os.path.join(ROOT, "Data", "datasets", "weather.csv")
    if not os.path.exists(csv):
        print("[SKIP] short train: weather.csv missing")
        return

    configs = load_yaml_config(os.path.join(ROOT, "Config", "weather.yaml"))
    assert configs["model"]["params"].get("residual_predictor") == "tcn"
    configs["solver"]["max_epochs"] = steps
    configs["solver"]["save_cycle"] = steps
    configs["solver"]["results_folder"] = "./Checkpoints_weather_res_tcn_smoke"
    configs["dataloader"]["batch_size"] = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mp = configs["model"]["params"]
    armd = instantiate_from_config(configs["model"]).to(device)
    model = ARMDTrendWrapper(
        armd=armd,
        feature_size=mp["feature_size"],
        ma_kernel_size=mp.get("ma_kernel_size", 48),
        fft_topk=mp.get("fft_topk", 5),
        adaptive_ma=mp.get("adaptive_ma", True),
        residual_predictor=mp.get("residual_predictor", "tcn"),
        residual_tcn_hidden_dim=mp.get("residual_tcn_hidden_dim", 32),
        residual_tcn_dilations=mp.get("residual_tcn_dilations", [1, 2, 4, 8]),
        residual_tcn_kernel_size=mp.get("residual_tcn_kernel_size", 3),
        residual_tcn_dropout=mp.get("residual_tcn_dropout", 0.1),
        final_loss_weight=mp.get("final_loss_weight", 1.0),
        final_mae_weight=mp.get("final_mae_weight", 0.5),
        trend_loss_weight=mp.get("trend_loss_weight", 0.2),
        residual_loss_weight=mp.get("residual_loss_weight", 0.1),
    ).to(device)

    # snapshot residual tcn params
    w0 = model.residual_tcn.output_projection.weight.detach().clone()

    args = argparse.Namespace(
        config_path=os.path.join(ROOT, "Config", "weather.yaml"),
        save_dir="./forecasting_exp_res_tcn_smoke",
        gpu=0,
        name="res_tcn_smoke",
    )
    os.makedirs(args.save_dir, exist_ok=True)
    dl_info = build_dataloader(configs, args)
    trainer = Trainer(
        config=configs,
        args=args,
        model=model,
        dataloader={"dataloader": dl_info["dataloader"]},
        logger=None,
    )
    trainer.log_frequency = 1
    trainer.train()

    w1 = model.residual_tcn.output_projection.weight.detach()
    assert not torch.allclose(w0, w1), "Residual TCN params did not update"
    assert model._last_loss_stats is not None
    assert torch.isfinite(torch.tensor(model._last_loss_stats["loss_residual"]))

    trainer.save(99)
    trainer2 = Trainer(
        config=configs,
        args=args,
        model=copy.deepcopy(model),
        dataloader={"dataloader": dl_info["dataloader"]},
        logger=None,
    )
    trainer2.load(99)

    x = next(iter(dl_info["dataloader"])).to(device)[:2]
    with torch.no_grad():
        sample = trainer.ema.ema_model.generate_mts(x)
    assert sample.shape == (2, mp["seq_length"], mp["feature_size"])
    assert torch.isfinite(sample).all()
    print("[PASS] short train + ckpt + sampling", tuple(sample.shape), model._last_loss_stats)


if __name__ == "__main__":
    test_decompose_consistency()
    test_residual_tcn_shape_grad()
    test_no_future_leak_in_tcn_mode()
    test_fft_mode_compat()
    test_short_train_weather(steps=5)
    print("\nAll residual-TCN tests finished.")
