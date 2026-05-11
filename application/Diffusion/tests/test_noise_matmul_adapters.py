"""
Unit test: compare real SC Triton kernels vs noisy surrogate adapters.

For each of the 4 kernel signatures used in sc_attention.py / sc_mlp.py,
call the real kernel and the noisy adapter with IDENTICAL inputs, then
compare:
  1. shape, dtype
  2. mean, std (distribution similarity)
  3. RMSE vs exact float matmul (noise level)
  4. wall-clock time

Shapes chosen to match DiT-XL/2 inference at 256x256:
    B=2, H=16, N=256, D=72       (head_dim = 72 in DiT-XL/2)
    mlp dim 1152 → 4608          (typical Q-DiT weight shapes)

Run:
    python -m pytest tests/test_noise_matmul_adapters.py -sv
  or:
    python tests/test_noise_matmul_adapters.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "SC"))

from qdit.sc_integration.noise_matmul import (
    noisy_sc_matmul,
    noisy_sc_matmul_mlp,
    noisy_sc_matmul_grouped,
    noisy_sc_matmul_enable_batched_bipolar,
)
from scmp_kernels.sc.sc_triton import (
    sc_matmul_enable_triton,
    sc_matmul_enable_triton_mlp,
    sc_matmul_grouped_enable_triton,
    sc_matmul_enable_batched_bipolar,
)
from scmp_kernels.sc.config_helpers import make_sobol_simple_config


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32


def _stats(name, ref, surrogate, exact):
    ref_rmse = (ref - exact).pow(2).mean().sqrt().item()
    sur_rmse = (surrogate - exact).pow(2).mean().sqrt().item()
    return (
        f"{name:42s} "
        f"ref_rmse={ref_rmse:>9.4f}  sur_rmse={sur_rmse:>9.4f}  "
        f"ratio={sur_rmse/max(ref_rmse,1e-9):>5.2f}x  "
        f"ref:mean={ref.mean().item():>+8.3f} std={ref.std().item():>8.3f}  "
        f"sur:mean={surrogate.mean().item():>+8.3f} std={surrogate.std().item():>8.3f}"
    )


def test_sc_matmul_enable_triton_vs_noisy():
    """2D input projection: (BN, D_in) @ (D_out, D_in)^T"""
    torch.manual_seed(0)
    M, D_in, D_out = 128, 1152, 1152  # DiT-XL/2 hidden_size=1152
    a = torch.randn(M, D_in, device=DEVICE, dtype=DTYPE)
    w = torch.randn(D_out, D_in, device=DEVICE, dtype=DTYPE) * 0.05  # weight scale

    config = make_sobol_simple_config(D_in, D_in, 8)
    exact = a @ w.transpose(-2, -1)

    print()
    for L in [256, 128, 64, 32, 16]:
        sc_prec = {256:8, 128:7, 64:6, 32:5, 16:4}[L]
        # Real
        torch.cuda.synchronize() if DEVICE == "cuda" else None
        t0 = time.time()
        ref = sc_matmul_enable_triton(
            a, w,
            a.max().item(), a.min().item(), w.max().item(), w.min().item(),
            mode="bipolar", sc_prec=sc_prec, config=config, stoc_len=L,
        )
        torch.cuda.synchronize() if DEVICE == "cuda" else None
        t_ref = time.time() - t0

        # Surrogate
        torch.cuda.synchronize() if DEVICE == "cuda" else None
        t0 = time.time()
        sur = noisy_sc_matmul(
            a, w,
            a.max().item(), a.min().item(), w.max().item(), w.min().item(),
            mode="bipolar", sc_prec=sc_prec, config=config, stoc_len=L,
        )
        torch.cuda.synchronize() if DEVICE == "cuda" else None
        t_sur = time.time() - t0

        assert ref.shape == sur.shape == exact.shape
        assert sur.dtype == torch.float32
        assert torch.isfinite(sur).all()
        print(
            _stats(f"sc_matmul_enable_triton L={L:>4d}", ref, sur, exact),
            f"t_ref={t_ref*1000:>6.1f}ms  t_sur={t_sur*1000:>6.1f}ms  speedup={t_ref/max(t_sur,1e-6):>5.1f}x"
        )


def test_sc_matmul_enable_triton_mlp_vs_noisy():
    """MLP fc1: (M, D) @ (D_hidden, D)^T where D_hidden = 4*D."""
    torch.manual_seed(1)
    M, D_in, D_out = 128, 1152, 4608
    a = torch.randn(M, D_in, device=DEVICE, dtype=DTYPE)
    w = torch.randn(D_out, D_in, device=DEVICE, dtype=DTYPE) * 0.05

    config = make_sobol_simple_config(D_in, D_in, 8)
    exact = a @ w.transpose(-2, -1)

    print()
    for L in [256, 64, 16]:
        sc_prec = {256:8, 64:6, 16:4}[L]
        torch.cuda.synchronize() if DEVICE == "cuda" else None
        t0 = time.time()
        ref = sc_matmul_enable_triton_mlp(
            a, w,
            a.max().item(), a.min().item(), w.max().item(), w.min().item(),
            mode="bipolar", sc_prec=sc_prec, config=config,
            group_a=1, group_b=1, stoc_len=L,
        )
        torch.cuda.synchronize() if DEVICE == "cuda" else None
        t_ref = time.time() - t0

        torch.cuda.synchronize() if DEVICE == "cuda" else None
        t0 = time.time()
        sur = noisy_sc_matmul_mlp(
            a, w,
            a.max().item(), a.min().item(), w.max().item(), w.min().item(),
            mode="bipolar", sc_prec=sc_prec, config=config,
            group_a=1, group_b=1, stoc_len=L,
        )
        torch.cuda.synchronize() if DEVICE == "cuda" else None
        t_sur = time.time() - t0

        assert ref.shape == sur.shape == exact.shape
        print(
            _stats(f"sc_matmul_enable_triton_mlp L={L:>4d}", ref, sur, exact),
            f"t_ref={t_ref*1000:>6.1f}ms  t_sur={t_sur*1000:>6.1f}ms  speedup={t_ref/max(t_sur,1e-6):>5.1f}x"
        )


def test_sc_matmul_grouped_enable_triton_vs_noisy():
    """AV single-head: attn (N, N) @ v^T (D, N) → (N, D).

    attn is softmax output (non-negative, row-sum=1).
    """
    torch.manual_seed(2)
    N, D = 256, 72
    v = torch.randn(N, D, device=DEVICE, dtype=DTYPE)
    # Softmax-like attn
    logits = torch.randn(N, N, device=DEVICE, dtype=DTYPE)
    attn = torch.softmax(logits, dim=-1)
    v_t = v.transpose(-2, -1).contiguous()  # (D, N)

    config = make_sobol_simple_config(N, N, 8)
    exact = attn @ v

    print()
    for L in [256, 64, 16]:
        sc_prec = {256:8, 64:6, 16:4}[L]
        torch.cuda.synchronize() if DEVICE == "cuda" else None
        t0 = time.time()
        ref = sc_matmul_grouped_enable_triton(
            attn, v_t,
            group_a=1, group_b=1,
            mode="bipolar", sc_prec=sc_prec, config=config, stoc_len=L,
        )
        torch.cuda.synchronize() if DEVICE == "cuda" else None
        t_ref = time.time() - t0

        torch.cuda.synchronize() if DEVICE == "cuda" else None
        t0 = time.time()
        sur = noisy_sc_matmul_grouped(
            attn, v_t,
            group_a=1, group_b=1,
            mode="bipolar", sc_prec=sc_prec, config=config, stoc_len=L,
        )
        torch.cuda.synchronize() if DEVICE == "cuda" else None
        t_sur = time.time() - t0

        assert ref.shape == sur.shape == exact.shape, f"ref {ref.shape}, sur {sur.shape}, exact {exact.shape}"
        print(
            _stats(f"sc_matmul_grouped_enable_triton L={L:>4d}", ref, sur, exact),
            f"t_ref={t_ref*1000:>6.1f}ms  t_sur={t_sur*1000:>6.1f}ms  speedup={t_ref/max(t_sur,1e-6):>5.1f}x"
        )


def test_sc_matmul_enable_batched_bipolar_vs_noisy():
    """QK batched: (BH, N, D) @ (BH, N, D)^T → (BH, N, N)"""
    torch.manual_seed(3)
    B, H, N, D = 2, 16, 256, 72
    q = torch.randn(B*H, N, D, device=DEVICE, dtype=DTYPE)
    k = torch.randn(B*H, N, D, device=DEVICE, dtype=DTYPE)

    config = make_sobol_simple_config(D, D, 8)
    exact = q @ k.transpose(-2, -1)

    print()
    for L in [256, 64, 16]:
        sc_prec = {256:8, 64:6, 16:4}[L]
        q_maxs = q.amax(dim=(1,2)); q_mins = q.amin(dim=(1,2))
        k_maxs = k.amax(dim=(1,2)); k_mins = k.amin(dim=(1,2))

        torch.cuda.synchronize() if DEVICE == "cuda" else None
        t0 = time.time()
        ref = sc_matmul_enable_batched_bipolar(
            q, k, q_maxs, q_mins, k_maxs, k_mins,
            sc_prec, config, stoc_len=L,
        )
        torch.cuda.synchronize() if DEVICE == "cuda" else None
        t_ref = time.time() - t0

        torch.cuda.synchronize() if DEVICE == "cuda" else None
        t0 = time.time()
        sur = noisy_sc_matmul_enable_batched_bipolar(
            q, k, q_maxs, q_mins, k_maxs, k_mins,
            sc_prec, config, stoc_len=L,
        )
        torch.cuda.synchronize() if DEVICE == "cuda" else None
        t_sur = time.time() - t0

        assert ref.shape == sur.shape == exact.shape, f"ref {ref.shape}, sur {sur.shape}, exact {exact.shape}"
        print(
            _stats(f"sc_matmul_enable_batched_bipolar L={L:>4d}", ref, sur, exact),
            f"t_ref={t_ref*1000:>6.1f}ms  t_sur={t_sur*1000:>6.1f}ms  speedup={t_ref/max(t_sur,1e-6):>5.1f}x"
        )


if __name__ == "__main__":
    print(f"Running on device: {DEVICE}")
    print("=" * 180)
    print("Legend: ref = real SC kernel, sur = noisy surrogate adapter")
    print("        ratio = sur_rmse / ref_rmse (should be ~1.0 if surrogate is calibrated correctly)")
    print("        surrogate should be MUCH faster than real SC")
    print("=" * 180)
    test_sc_matmul_enable_triton_vs_noisy()
    test_sc_matmul_enable_triton_mlp_vs_noisy()
    test_sc_matmul_grouped_enable_triton_vs_noisy()
    test_sc_matmul_enable_batched_bipolar_vs_noisy()
    print("\nAll tests passed.")
