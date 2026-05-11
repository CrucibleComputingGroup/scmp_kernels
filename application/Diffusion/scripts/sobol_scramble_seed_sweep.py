"""Test stability of scramble_rand across different seeds.

We want to confirm the per-D random XOR mask variant isn't just lucky at one
particular seed. Run 5 different mask-seeds and check variance.
"""

from __future__ import annotations

import torch

from scmp_kernels.sc import sc_triton
from scmp_kernels.sc.sc_triton import (
    clear_rng_cache,
    sc_matmul_enable_batched_bipolar,
    sc_matmul_enable_triton,
    sc_matmul_grouped_enable_triton,
)
from scmp_kernels.sc.config_helpers import make_sobol_simple_config


def rel_err(pred, target):
    num = (pred - target).pow(2).mean().sqrt()
    den = target.pow(2).mean().sqrt().clamp_min(1e-8)
    return float((num / den).item())


def scramble_rand_with_seed(seed):
    def fn(rng, stoc_len):
        D = rng.shape[0]
        g = torch.Generator(device=rng.device).manual_seed(seed)
        masks = torch.randint(0, 256, (D, 1), generator=g, device=rng.device)
        prefix = rng[:, :stoc_len]
        return (prefix.to(torch.int64) ^ masks).to(rng.dtype).contiguous()
    return fn


def run_suite(prefix_fn, seed=0):
    torch.manual_seed(seed)
    orig_prepare = sc_triton._prepare_rng_prefix

    def patched(rng, sc_prec, stoc_len_inner, rng_levels):
        grid = sc_triton._resolve_rng_levels(sc_prec, rng_levels)
        base = 2 ** sc_prec
        if grid != base:
            return orig_prepare(rng, sc_prec, stoc_len_inner, rng_levels)
        return prefix_fn(rng, stoc_len_inner)

    sc_triton._prepare_rng_prefix = patched
    clear_rng_cache()
    try:
        out = {}
        for stoc_len in [32, 48, 64, 96, 128]:
            # Linear
            n, d, m = 32, 1152, 512
            x = torch.randn(n, d, device="cuda")
            w = torch.randn(m, d, device="cuda")
            fp_lin = x @ w.t()
            cfg_lin = make_sobol_simple_config(d, d, 8)
            sc_lin = sc_matmul_enable_triton(
                x, w, x.max().item(), x.min().item(), w.max().item(), w.min().item(),
                mode="bipolar", sc_prec=8, config=cfg_lin,
                stoc_len=stoc_len, rng_levels=None,
            )
            # AV
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
            # QK
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
            out[stoc_len] = (rel_err(sc_lin, fp_lin),
                             rel_err(sc_av, fp_av),
                             rel_err(sc_qk, fp_qk))
        return out
    finally:
        sc_triton._prepare_rng_prefix = orig_prepare
        clear_rng_cache()


def main():
    assert torch.cuda.is_available()
    seeds = [0, 7, 42, 12345, 99999]
    levels = [32, 48, 64, 96, 128]

    print("Stability across 5 mask-seeds\n")
    header = f"{'seed':<8} " + " ".join(f"{op}_sl{sl:>3d}".rjust(11) for op in ("lin", "av", "qk") for sl in levels)
    print(header)
    print("-" * len(header))

    all_runs = []
    for s in seeds:
        out = run_suite(scramble_rand_with_seed(s), seed=0)
        all_runs.append(out)
        row = [f"{s:<8d}"]
        for op_idx in range(3):
            for sl in levels:
                row.append(f"{out[sl][op_idx]:>11.4f}")
        print(" ".join(row))

    # Mean/std
    print()
    header2 = f"{'stat':<8} " + " ".join(f"{op}_sl{sl:>3d}".rjust(11) for op in ("lin", "av", "qk") for sl in levels)
    print(header2)
    print("-" * len(header2))
    for stat_name, stat_fn in (("mean", lambda xs: sum(xs) / len(xs)),
                               ("min",  min),
                               ("max",  max)):
        row = [f"{stat_name:<8}"]
        for op_idx in range(3):
            for sl in levels:
                vals = [run[sl][op_idx] for run in all_runs]
                row.append(f"{stat_fn(vals):>11.4f}")
        print(" ".join(row))


if __name__ == "__main__":
    main()
