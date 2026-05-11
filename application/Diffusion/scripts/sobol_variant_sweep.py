"""Standalone benchmark for Sobol-prefix variants in fixed-level SC mode.

Goal: fixed sc_prec=8 (int8 quant), varying stoc_len (stream length).
Current baseline (first 64 of Sobol-8) yields values in {0,4,...,252}, which
biases small boundaries. This sweep compares variants designed to break the
stratification artifact.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SC_ROOT = REPO_ROOT / "SC"
if str(SC_ROOT) not in sys.path:
    sys.path.insert(0, str(SC_ROOT))

import sc_triton
from scmp_kernels.sc.sc_triton import (
    clear_rng_cache,
    sc_matmul_enable_batched_bipolar,
    sc_matmul_enable_triton,
    sc_matmul_grouped_enable_triton,
)
from scmp_kernels.sc.config_helpers import make_sobol_simple_config


def rel_err(pred: torch.Tensor, target: torch.Tensor) -> float:
    num = (pred - target).pow(2).mean().sqrt()
    den = target.pow(2).mean().sqrt().clamp_min(1e-8)
    return float((num / den).item())


# ---------- Variant prefix builders --------------------------------------


def v_baseline(rng: torch.Tensor, stoc_len: int) -> torch.Tensor:
    return rng[:, :stoc_len].contiguous()


def _xor_scramble(rng: torch.Tensor, stoc_len: int, mask: int) -> torch.Tensor:
    prefix = rng[:, :stoc_len]
    return (prefix.to(torch.int64) ^ mask).to(rng.dtype).contiguous()


def v_scramble_2(rng, stoc_len):
    return _xor_scramble(rng, stoc_len, 2)


def v_scramble_3(rng, stoc_len):
    return _xor_scramble(rng, stoc_len, 3)


def v_scramble_rand(rng, stoc_len, seed=12345):
    D = rng.shape[0]
    g = torch.Generator(device=rng.device).manual_seed(seed)
    masks = torch.randint(0, 256, (D, 1), generator=g, device=rng.device)
    prefix = rng[:, :stoc_len]
    return (prefix.to(torch.int64) ^ masks).to(rng.dtype).contiguous()


def v_stride_k(rng, stoc_len, k):
    full = rng.shape[1]
    if stoc_len * k > full:
        # fall back
        return v_baseline(rng, stoc_len)
    idx = torch.arange(0, stoc_len * k, k, device=rng.device)[:stoc_len]
    return rng[:, idx].contiguous()


def v_stride_2(rng, stoc_len):
    return v_stride_k(rng, stoc_len, 2)


def v_stride_4(rng, stoc_len):
    return v_stride_k(rng, stoc_len, 4)


def v_reverse_bits(rng, stoc_len):
    import math

    full = rng.shape[1]
    if stoc_len & (stoc_len - 1) != 0:
        return v_baseline(rng, stoc_len)
    bits = int(math.log2(stoc_len))
    base = torch.arange(stoc_len, device=rng.device)
    rev = torch.zeros_like(base)
    for b in range(bits):
        rev |= ((base >> b) & 1) << (bits - 1 - b)
    # Map to full-length positions: spread first `stoc_len` into `full` by stride `full/stoc_len`
    scale = full // stoc_len
    idx = rev * scale
    return rng[:, idx].contiguous()


def v_stratum_mid(rng, stoc_len):
    """Analytic stratum midpoint: for each of stoc_len bins of width full/stoc_len,
    pick the bin midpoint. Independent of Sobol direction vectors."""
    full = rng.shape[1]
    step = full // stoc_len
    mid = step // 2
    vals = torch.arange(stoc_len, device=rng.device) * step + mid
    vals = vals.clamp_max(full - 1).to(rng.dtype)
    # Broadcast D dims
    D = rng.shape[0]
    return vals.unsqueeze(0).expand(D, -1).contiguous()


VARIANTS = {
    "baseline":       v_baseline,
    "scramble_2":     v_scramble_2,
    "scramble_3":     v_scramble_3,
    "scramble_rand":  v_scramble_rand,
    "stride_2":       v_stride_2,
    "stride_4":       v_stride_4,
    "reverse_bits":   v_reverse_bits,
    "stratum_mid":    v_stratum_mid,
}


# ---------- Monkey-patch scaffolding -------------------------------------


def run_variant(variant_fn, stoc_len, seed=0):
    torch.manual_seed(seed)
    orig_prepare = sc_triton._prepare_rng_prefix

    def patched(rng, sc_prec, stoc_len_inner, rng_levels):
        grid = sc_triton._resolve_rng_levels(sc_prec, rng_levels)
        base = 2 ** sc_prec
        # Only override on fixed-level path (grid == base)
        if grid != base:
            return orig_prepare(rng, sc_prec, stoc_len_inner, rng_levels)
        return variant_fn(rng, stoc_len_inner)

    sc_triton._prepare_rng_prefix = patched
    clear_rng_cache()
    try:
        # Linear: N=32, D=1152, M=512
        n, d, m = 32, 1152, 512
        x = torch.randn(n, d, device="cuda", dtype=torch.float32)
        w = torch.randn(m, d, device="cuda", dtype=torch.float32)
        fp_lin = x @ w.t()
        cfg_lin = make_sobol_simple_config(d, d, 8)
        sc_lin = sc_matmul_enable_triton(
            x, w, x.max().item(), x.min().item(), w.max().item(), w.min().item(),
            mode="bipolar", sc_prec=8, config=cfg_lin, stoc_len=stoc_len, rng_levels=None,
        )
        lin_err = rel_err(sc_lin, fp_lin)

        # AV: N=128, D=72 (softmax attn @ v)
        nn, dd = 128, 72
        attn = torch.softmax(torch.randn(nn, nn, device="cuda"), dim=-1)
        v = torch.randn(nn, dd, device="cuda")
        fp_av = attn @ v
        cfg_av = make_sobol_simple_config(nn, nn, 8)
        sc_av = sc_matmul_grouped_enable_triton(
            attn, v.t().contiguous(), group_a=nn, group_b=dd,
            mode="bipolar", sc_prec=8, config=cfg_av,
            stoc_len=stoc_len, rng_levels=None,
        )
        av_err = rel_err(sc_av, fp_av)

        # QK: B=2, N=128, D=72 (batched bipolar)
        bh, N, D = 2, 128, 72
        q = torch.randn(bh, N, D, device="cuda")
        k = torch.randn(bh, N, D, device="cuda")
        fp_qk = q @ k.transpose(-1, -2)
        qmax = q.amax(dim=(1, 2)); qmin = q.amin(dim=(1, 2))
        kmax = k.amax(dim=(1, 2)); kmin = k.amin(dim=(1, 2))
        cfg_qk = make_sobol_simple_config(D, D, 8)
        sc_qk = sc_matmul_enable_batched_bipolar(
            q, k, qmax, qmin, kmax, kmin, 8, cfg_qk,
            stoc_len=stoc_len, rng_levels=None,
        )
        qk_err = rel_err(sc_qk, fp_qk)

        return lin_err, av_err, qk_err
    finally:
        sc_triton._prepare_rng_prefix = orig_prepare
        clear_rng_cache()


def main():
    assert torch.cuda.is_available(), "CUDA required"
    torch.set_default_dtype(torch.float32)

    levels = [32, 48, 64, 96, 128]
    header = f"{'variant':<18} " + " ".join(f"{op}_sl{sl:>3d}".rjust(14) for op in ("lin", "av", "qk") for sl in levels)
    print(header)
    print("-" * len(header))

    for name, fn in VARIANTS.items():
        row = [f"{name:<18}"]
        # Collect (lin, av, qk) for each level
        results = {sl: run_variant(fn, sl, seed=0) for sl in levels}
        for op_idx, op in enumerate(("lin", "av", "qk")):
            for sl in levels:
                row.append(f"{results[sl][op_idx]:>14.4f}")
        print(" ".join(row))


if __name__ == "__main__":
    main()
