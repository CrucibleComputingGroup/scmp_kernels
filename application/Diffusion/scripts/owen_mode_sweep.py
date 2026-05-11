"""Sweep SC kernel relative error across stoc_len ∈ {16,32,48,64,96,128} for
each Owen mask mode in {off, random, counter, bitrev}.

Reuses the shapes from debug_fixed_level_sanity.py (linear / av / qk).
Prints a table per op so you can A/B the modes at a glance.
"""
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SC_ROOT = REPO_ROOT / "SC"
if str(SC_ROOT) not in sys.path:
    sys.path.insert(0, str(SC_ROOT))


LEVELS = [16, 32, 48, 64, 96, 128]
MODES = ["off", "random", "counter", "bitrev"]


def rel_err(pred: torch.Tensor, target: torch.Tensor) -> float:
    num = (pred - target).pow(2).mean().sqrt()
    den = target.pow(2).mean().sqrt().clamp_min(1e-8)
    return float((num / den).item())


def make_inputs(seed: int = 0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    # linear: x [32,1152] @ w^T [512,1152]
    n_l, d_l, m_l = 32, 1152, 512
    x_l = torch.randn(n_l, d_l, device="cuda", dtype=torch.float32, generator=g)
    w_l = torch.randn(m_l, d_l, device="cuda", dtype=torch.float32, generator=g)
    # av: softmax(randn[128,128]) @ v[128,72]
    n_av, d_av = 128, 72
    attn = torch.softmax(torch.randn(n_av, n_av, device="cuda", generator=g), dim=-1)
    v = torch.randn(n_av, d_av, device="cuda", generator=g)
    # qk: q,k [2,128,72]
    bh, n_qk, d_qk = 2, 128, 72
    q = torch.randn(bh, n_qk, d_qk, device="cuda", generator=g)
    k = torch.randn(bh, n_qk, d_qk, device="cuda", generator=g)
    return dict(linear=(x_l, w_l), av=(attn, v), qk=(q, k))


def run_mode(mode: str, inputs):
    """Set Owen mode, clear caches so new mask is honored, run all three ops."""
    os.environ["SC_OWEN_MODE"] = mode
    # Importing only after env var is set is not strictly required (mode is read
    # per-call inside _owen_scramble), but we MUST clear cached enable tables
    # and RNG sequences so the new mask is used. Cache key does not include
    # the mode, so without clear we'd see stale results.
    import sc_triton as sct
    sct.clear_rng_cache()
    from config_helpers import make_sobol_simple_config

    out = {}

    # linear
    x, w = inputs["linear"]
    fp = x @ w.t()
    config = make_sobol_simple_config(x.shape[1], x.shape[1], 8)
    out["linear"] = []
    for sl in LEVELS:
        sc = sct.sc_matmul_enable_triton(
            x, w, x.max().item(), x.min().item(),
            w.max().item(), w.min().item(),
            mode="bipolar", sc_prec=8, config=config, stoc_len=sl,
        )
        out["linear"].append(rel_err(sc, fp))

    # av
    attn, v = inputs["av"]
    fp = attn @ v
    config = make_sobol_simple_config(attn.shape[1], attn.shape[1], 8)
    out["av"] = []
    for sl in LEVELS:
        sc = sct.sc_matmul_grouped_enable_triton(
            attn, v.t().contiguous(),
            group_a=attn.shape[0], group_b=v.shape[1],
            mode="bipolar", sc_prec=8, config=config, stoc_len=sl,
        )
        out["av"].append(rel_err(sc, fp))

    # qk
    q, k = inputs["qk"]
    fp = q @ k.transpose(-1, -2)
    qmax = q.amax(dim=(1, 2))
    qmin = q.amin(dim=(1, 2))
    kmax = k.amax(dim=(1, 2))
    kmin = k.amin(dim=(1, 2))
    config = make_sobol_simple_config(q.shape[2], q.shape[2], 8)
    out["qk"] = []
    for sl in LEVELS:
        sc = sct.sc_matmul_enable_batched_bipolar(
            q, k, qmax, qmin, kmax, kmin, 8, config, stoc_len=sl,
        )
        out["qk"].append(rel_err(sc, fp))

    return out


def main():
    assert torch.cuda.is_available(), "CUDA is required"
    torch.manual_seed(0)
    inputs = make_inputs(seed=0)

    results = {}  # mode -> {op -> [rel_err per stoc_len]}
    for mode in MODES:
        results[mode] = run_mode(mode, inputs)

    # Print per-op tables
    for op in ("linear", "av", "qk"):
        print(f"\n=== op = {op} ===")
        header = f"{'mode':>10}  " + "  ".join(f"sl={sl:>3}" for sl in LEVELS)
        print(header)
        for mode in MODES:
            row = "  ".join(f"{results[mode][op][i]:7.4f}" for i in range(len(LEVELS)))
            print(f"{mode:>10}  {row}")

    # Print delta vs random (Owen baseline) — negative means better than random
    print("\n=== rel_err improvement vs SC_OWEN_MODE=random (negative = better) ===")
    for op in ("linear", "av", "qk"):
        print(f"  op={op}")
        for mode in MODES:
            if mode == "random":
                continue
            deltas = [results[mode][op][i] - results["random"][op][i]
                      for i in range(len(LEVELS))]
            row = "  ".join(f"{d:+7.4f}" for d in deltas)
            print(f"    {mode:>10}  {row}")


if __name__ == "__main__":
    main()
