"""Capture sc_matmul outputs across a grid of configurations.

Run twice — once before the refactor (to seed reference), once after (to verify
bit-exact equivalence). Driven by PHASE env var:

    PHASE=ref python capture_reference.py    # writes reference.pt
    PHASE=verify python capture_reference.py # diffs against reference.pt
"""
from __future__ import annotations

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scmp_kernels import sc_matmul
from scmp_kernels.sc import clear_rng_cache

REF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference.pt")


def _make_inputs(BH, N, M, D, dim, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)
    if dim == 2:
        a = torch.randn(N, D, generator=g, device=device)
        b = torch.randn(M, D, generator=g, device=device)
    else:
        a = torch.randn(BH, N, D, generator=g, device=device)
        b = torch.randn(BH, M, D, generator=g, device=device)
    return a, b


def _cases():
    # (name, BH, N, M, D, dim, granularity, mode, sc_prec, stoc_len, chunk_d)
    return [
        ("pt_2d_bip",   1, 16, 24, 32, 2, "per_tensor", "bipolar",   8,  256, 0),
        ("pt_2d_uni",   1, 16, 24, 32, 2, "per_tensor", "unipolar",  8,  256, 0),
        ("pr_2d_bip",   1, 16, 24, 32, 2, "per_row",    "bipolar",   8,  256, 0),
        ("pr_2d_uni",   1, 16, 24, 32, 2, "per_row",    "unipolar",  8,  256, 0),
        ("pr_2d_chunk", 1, 16, 24, 64, 2, "per_row",    "bipolar",   8,  256, 16),
        ("pr_3d_bip",   2, 8,  12, 32, 3, "per_row",    "bipolar",   8,  256, 0),
        ("ph_3d_bip",   2, 8,  12, 32, 3, "per_head",   "bipolar",   8,  256, 0),
        ("pr_2d_lowprec", 1, 16, 24, 32, 2, "per_row",  "bipolar",   4,  16,  0),
    ]


def run_grid():
    device = torch.device("cuda")
    out = {}
    for (name, BH, N, M, D, dim, gran, mode, prec, slen, ch) in _cases():
        clear_rng_cache()
        a, b = _make_inputs(BH, N, M, D, dim, seed=1234, device=device)
        y = sc_matmul(a, b,
                      granularity=gran, mode=mode,
                      sc_prec=prec, stoc_len=slen, chunk_d=ch)
        out[name] = y.detach().cpu()
        print(f"  {name:15s} -> shape={tuple(y.shape)} "
              f"sum={y.sum().item():+.6f}  abs.mean={y.abs().mean().item():.6f}")
    return out


def time_case(repeat=5):
    """Single representative timing for regression sanity."""
    device = torch.device("cuda")
    clear_rng_cache()
    a, b = _make_inputs(1, 128, 128, 128, 2, seed=7, device=device)
    # warmup
    for _ in range(3):
        y = sc_matmul(a, b, granularity="per_row", mode="bipolar",
                      sc_prec=8, stoc_len=256)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        y = sc_matmul(a, b, granularity="per_row", mode="bipolar",
                      sc_prec=8, stoc_len=256)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000 / repeat
    print(f"per_row 128x128x128 sc_prec=8 stoc_len=256 -> {ms:.3f} ms/it")
    return ms


def main():
    phase = os.environ.get("PHASE", "ref")
    print(f"[capture_reference] phase={phase}")
    print(f"[capture_reference] torch={torch.__version__} cuda={torch.cuda.is_available()}")
    print(f"[capture_reference] scmp_kernels at "
          f"{__import__('scmp_kernels').__file__}")

    out = run_grid()
    ms = time_case()

    if phase == "ref":
        torch.save({"outputs": out, "ms": ms}, REF_PATH)
        print(f"[capture_reference] saved reference to {REF_PATH}")
        return 0

    # verify
    if not os.path.exists(REF_PATH):
        print(f"[capture_reference] ERROR: reference {REF_PATH} not found. "
              "Run PHASE=ref first.")
        return 2

    ref = torch.load(REF_PATH, weights_only=False)
    fail = 0
    for name, cur in out.items():
        r = ref["outputs"][name]
        if cur.shape != r.shape:
            print(f"  {name}: SHAPE MISMATCH {cur.shape} vs {r.shape}")
            fail += 1
            continue
        if torch.equal(cur, r):
            print(f"  {name}: PASS (bit-exact)")
        else:
            diff = (cur - r).abs()
            print(f"  {name}: FAIL  max|Δ|={diff.max().item():.3e}  "
                  f"mean|Δ|={diff.mean().item():.3e}")
            fail += 1
    ms_ref = ref["ms"]
    pct = (ms - ms_ref) / ms_ref * 100
    print(f"timing: {ms:.3f} ms (before {ms_ref:.3f} ms, {pct:+.1f}%)")
    if fail:
        print(f"[capture_reference] FAIL: {fail} case(s) mismatched")
        return 1
    print("[capture_reference] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
