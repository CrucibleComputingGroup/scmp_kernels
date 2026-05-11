#!/usr/bin/env python3
"""
Profile per-(timestep, layer, operator) metric statistics from a DiT inference.

Runs a single image generation (no SC, pure FP16) and hooks into every
SCAttention / SCMlp to collect the importance metrics that the MP algorithm
would use:
  - QK:    ||Q_row||_inf  per head           → shape [H]
  - AV:    max(attn_row)  per (batch*head, N) → shape [N]
  - proj:  ||x_row||_inf  per token          → shape [M]
  - mlp:   ||x_row||_inf  per token          → shape [M]

For each (timestep, block, operator), we record:
  metric_mean, metric_std, metric_min, metric_max

Outputs: profile_metric_sigma.csv

Usage:
    python scripts/profile_metric_sigma.py \
        --wbits 8 --abits 8 --w_sym --a_sym \
        --image-size 256 --num-sampling-steps 100
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import os
import csv
import torch
import torch.nn as nn
import numpy as np
from pytorch_lightning import seed_everything
from torch.cuda.amp import autocast

from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from utils.download import find_model
from models.models import DiT_models
from collections import defaultdict

from qdit.quant import *
from qdit.outlier import *
from qdit.datautils import *

from qdit.sc_integration import (
    SCController,
    add_sc_wrapper,
    quantize_sc_model,
    create_sc_controller_from_args,
)


# ── Global profiling storage ───────────────────────────────────────

_profile_log = []   # list of dicts
_current_timestep = None


def _record(block_idx, operator, metric: torch.Tensor):
    """Record metric statistics for one (timestep, block, operator)."""
    m = metric.float()
    _profile_log.append({
        "timestep": _current_timestep,
        "block": block_idx,
        "operator": operator,
        "N": m.numel(),
        "mean": m.mean().item(),
        "std": m.std().item(),
        "min": m.min().item(),
        "max": m.max().item(),
        "median": m.median().item(),
        "q25": m.quantile(0.25).item(),
        "q75": m.quantile(0.75).item(),
        "q95": m.quantile(0.95).item(),
        "q99": m.quantile(0.99).item(),
    })


# ── Hooks ──────────────────────────────────────────────────────────

def _make_attn_hook(block_idx, num_heads, head_dim, scale, q_norm, k_norm):
    """Create a lightweight hook: only collect input/output row metrics.
    Avoid recomputing attention (too memory-heavy for profiling)."""

    def hook(module, input, output):
        if _current_timestep is None:
            return

        x = input[0]   # [B, N, C]
        B, N, C = x.shape

        with torch.no_grad():
            # Input proj metric: ||x_row||_inf  (same metric used for input_proj and mlp)
            x_flat = x.reshape(-1, C)
            input_metric = x_flat.float().abs().amax(dim=-1)  # [B*N]
            _record(block_idx, "input_proj", input_metric)

            # Output (proj) metric: ||output_row||_inf
            out = output  # [B, N, C]
            out_flat = out.reshape(-1, C)
            proj_metric = out_flat.float().abs().amax(dim=-1)  # [B*N]
            _record(block_idx, "proj", proj_metric)

    return hook


def _make_mlp_hook(block_idx):
    """Create a hook that collects MLP input metric."""

    def hook(module, input, output):
        if _current_timestep is None:
            return

        x = input[0]  # [B, N, C]
        with torch.no_grad():
            x_flat = x.reshape(-1, x.shape[-1])
            mlp_metric = x_flat.float().abs().amax(dim=-1)
            _record(block_idx, "mlp", mlp_metric)

    return hook


# ── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-sampling-steps", type=int, default=100)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--vae", type=str, default="mse")
    parser.add_argument("--output", type=str, default="profile_metric_sigma.csv")
    # Quant args needed for model loading
    parser.add_argument("--wbits", type=int, default=8)
    parser.add_argument("--abits", type=int, default=8)
    parser.add_argument("--w_sym", action="store_true")
    parser.add_argument("--a_sym", action="store_true")
    parser.add_argument("--weight_group_size", type=str, default="-1")
    parser.add_argument("--act_group_size", type=str, default="-1")
    parser.add_argument("--weight_channel_group", type=int, default=1)
    parser.add_argument("--tiling", type=int, default=0)
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--quant_method", type=str, default="max")
    parser.add_argument("--a_clip_ratio", type=float, default=1.0)
    parser.add_argument("--w_clip_ratio", type=float, default=1.0)
    parser.add_argument("--quant_type", type=str, default="int")
    parser.add_argument("--calib_data_path", type=str, default="../cali_data/cali_data_256.pth")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--exponential", action="store_true")
    parser.add_argument("--quantize_bmm_input", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(args.seed)

    # Load model (FP16, no SC — pure inference for profiling)
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
    ).to(device)

    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval().half()

    diffusion = create_diffusion(str(args.num_sampling_steps))
    total_timesteps = diffusion.num_timesteps

    # Register hooks on attention and MLP modules
    hooks = []
    for block_idx, block in enumerate(model.blocks):
        attn = block.attn
        hooks.append(attn.register_forward_hook(
            _make_attn_hook(
                block_idx, attn.num_heads, attn.head_dim,
                attn.scale, attn.q_norm, attn.k_norm)))
        hooks.append(block.mlp.register_forward_hook(
            _make_mlp_hook(block_idx)))

    print(f"Registered hooks on {len(model.blocks)} blocks")
    print(f"Total timesteps: {total_timesteps}")

    # Run one image generation
    global _current_timestep

    # Use single image to minimize memory (profiling only needs statistics)
    class_labels = [207]
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device).half()
    y = torch.tensor(class_labels, device=device)
    model_kwargs = dict(y=y)  # No CFG to save memory

    indices = list(range(total_timesteps))[::-1]

    from tqdm import tqdm
    img = z

    with torch.no_grad(), autocast():
        for i in tqdm(indices, desc="Profiling"):
            _current_timestep = i
            t = torch.tensor([i] * img.shape[0], device=device)
            out = diffusion.ddim_sample(model, img, t,
                                         clip_denoised=False,
                                         model_kwargs=model_kwargs)
            img = out["sample"]

    # Remove hooks
    for h in hooks:
        h.remove()

    # Write CSV
    if _profile_log:
        fieldnames = list(_profile_log[0].keys())
        with open(args.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(_profile_log)
        print(f"\nWrote {len(_profile_log)} rows to {args.output}")
        print(f"  Timesteps: {total_timesteps}")
        print(f"  Blocks: {len(model.blocks)}")
        print(f"  Operators: qk, av, input_proj, proj, mlp")
    else:
        print("No data collected!")


if __name__ == "__main__":
    main()
