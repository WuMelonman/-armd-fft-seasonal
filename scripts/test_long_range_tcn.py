"""
Minimal tests for Dilated TCN residual branch in ARMD.

Usage:
  python scripts/test_long_range_tcn.py
"""

import os
import sys
import copy
import math

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Models.autoregressive_diffusion.armd import ARMD


def _make_armd(use_tcn: bool, feature_size: int = 21, seq_length: int = 96, **extra):
    return ARMD(
        seq_length=seq_length,
        feature_size=feature_size,
        timesteps=seq_length,
        sampling_timesteps=1,
        loss_type="l1",
        beta_schedule="cosine",
        w_grad=True,
        use_long_range_tcn=use_tcn,
        tcn_hidden_dim=64,
        tcn_dilations=[1, 2, 4, 8, 16, 32],
        tcn_dropout=0.1,
        tcn_gamma_init=extra.get("tcn_gamma_init", 1e-3),
        tcn_out_init_std=extra.get("tcn_out_init_std", 1e-3),
    )


def test_shape():
    # Real ARMD layout is [B, T, C], not [B, C, T]
    B, T, C = 4, 96, 21
    model = _make_armd(True, feature_size=C, seq_length=T)
    model.eval()
    x = torch.randn(B, T, C)
    t = torch.randint(0, T, (B,))
    with torch.no_grad():
        out = model.output(x, t, training=False)
    assert out.shape == x.shape, f"shape mismatch: {out.shape} vs {x.shape}"
    print("[PASS] shape:", tuple(out.shape))


def test_initial_equivalence():
    B, T, C = 2, 96, 21
    torch.manual_seed(0)
    base = _make_armd(False, feature_size=C, seq_length=T)
    torch.manual_seed(0)
    tcn = _make_armd(True, feature_size=C, seq_length=T, tcn_gamma_init=0.0, tcn_out_init_std=0.0)
    # force gamma=0 and zero output proj for strict degeneration
    with torch.no_grad():
        tcn.tcn_gamma.fill_(0.0)
        tcn.long_range_tcn.output_projection.weight.zero_()
        tcn.long_range_tcn.output_projection.bias.zero_()
        # copy shared backbone / coupling weights
        base_sd = base.state_dict()
        tcn_sd = tcn.state_dict()
        for k, v in base_sd.items():
            if k in tcn_sd and tcn_sd[k].shape == v.shape:
                tcn_sd[k].copy_(v)
        tcn.load_state_dict(tcn_sd)

    base.eval()
    tcn.eval()
    x = torch.randn(B, T, C)
    t = torch.randint(0, T, (B,))
    with torch.no_grad():
        out_base = base.output(x, t, training=False)
        out_tcn = tcn.output(x, t, training=False)
    diff = torch.max(torch.abs(out_base - out_tcn)).item()
    print(f"[INFO] max_abs_difference={diff:.3e}")
    assert diff < 1e-6, f"equivalence failed: {diff}"
    print("[PASS] initial equivalence (gamma=0, out_proj=0)")


def test_grad():
    B, T, C = 2, 96, 21
    model = _make_armd(True, feature_size=C, seq_length=T, tcn_gamma_init=1e-3, tcn_out_init_std=1e-3)
    model.train()
    x = torch.randn(B, T, C)
    t = torch.randint(0, T, (B,))
    out = model.output(x, t, training=True)
    loss = out.abs().mean()
    loss.backward()

    g_gamma = model.tcn_gamma.grad
    g_out = model.long_range_tcn.output_projection.weight.grad
    g_block = model.long_range_tcn.blocks[0].depthwise.weight.grad
    print(
        f"[INFO] gamma.grad={None if g_gamma is None else float(g_gamma.abs().mean()):.3e}, "
        f"out_proj.grad={None if g_out is None else float(g_out.abs().mean()):.3e}, "
        f"block0.grad={None if g_block is None else float(g_block.abs().mean()):.3e}"
    )
    assert g_gamma is not None and torch.isfinite(g_gamma).all()
    assert g_out is not None and torch.isfinite(g_out).all()
    # with gamma~1e-3 and small out init, block grads should be finite (may be tiny)
    assert g_block is not None and torch.isfinite(g_block).all()
    print("[PASS] gradients")


def test_short_train_weather(steps: int = 3):
    from Utils.io_utils import load_yaml_config, instantiate_from_config
    from Data.build_dataloader import build_dataloader
    from trend_utils.armd_trend_wrapper import ARMDTrendWrapper
    from engine.solver import Trainer
    import argparse

    cfg_path = os.path.join(ROOT, "Config", "weather.yaml")
    if not os.path.exists(os.path.join(ROOT, "Data", "datasets", "weather.csv")):
        print("[SKIP] short train: weather.csv not found")
        return

    configs = load_yaml_config(cfg_path)
    assert configs["model"]["params"].get("use_long_range_tcn", False) is True
    configs["solver"]["max_epochs"] = steps
    configs["solver"]["save_cycle"] = steps  # force one save
    configs["solver"]["results_folder"] = "./Checkpoints_weather_tcn_smoke"
    configs["dataloader"]["batch_size"] = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    armd = instantiate_from_config(configs["model"]).to(device)
    assert armd.use_long_range_tcn
    model = ARMDTrendWrapper(
        armd=armd,
        feature_size=configs["model"]["params"]["feature_size"],
        ma_kernel_size=configs["model"]["params"].get("ma_kernel_size", 25),
        fft_topk=configs["model"]["params"].get("fft_topk", 5),
    ).to(device)

    args = argparse.Namespace(
        config_path=cfg_path,
        save_dir="./forecasting_exp_tcn_smoke",
        gpu=0,
        name="tcn_smoke",
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
    trainer.train()

    # finite loss already implied by train loop; check TCN param in optimizer / EMA
    tcn_params = [n for n, _ in model.named_parameters() if "long_range_tcn" in n or n.endswith("tcn_gamma")]
    assert tcn_params, "TCN params missing from model"
    print(f"[INFO] TCN param tensors: {len(tcn_params)}")

    # save + reload
    trainer.save(999)
    trainer2 = Trainer(
        config=configs,
        args=args,
        model=copy.deepcopy(model),
        dataloader={"dataloader": dl_info["dataloader"]},
        logger=None,
    )
    trainer2.load(999)
    print("[PASS] short train + checkpoint save/load")

    # sampling path
    x = next(iter(dl_info["dataloader"])).to(device)[:2]
    with torch.no_grad():
        sample = trainer.ema.ema_model.generate_mts(x)
    assert sample.shape[0] == x.shape[0]
    assert sample.shape[1] == configs["model"]["params"]["seq_length"]
    assert sample.shape[2] == configs["model"]["params"]["feature_size"]
    assert torch.isfinite(sample).all()
    print("[PASS] sampling with TCN, sample shape", tuple(sample.shape))


if __name__ == "__main__":
    test_shape()
    test_initial_equivalence()
    test_grad()
    test_short_train_weather(steps=3)
    print("\nAll TCN tests finished.")
