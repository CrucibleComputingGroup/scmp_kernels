"""Basic correctness test for SmoothQuant integration in scmp_kernels.

Covers:
  1. Calibration aggregator: running per-channel max-abs across batches.
  2. compute_smooth_scales: shape / positivity / finiteness.
  3. apply_smoothing math identity: (a/s) @ (b*s).T ≈ a @ b.T.
  4. Outlier MSE improvement under per-row absmax int8 quant.
  5. sc_matmul(..., smooth_scales=s) wiring  (CUDA-gated).

Run:
    python arch_impl/test_smoothquant.py
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from scmp_kernels.quant.smoothquant import (
    accumulate_act_scales,
    compute_smooth_scales,
    apply_smoothing,
    apply_smoothing_offline,
)


def _per_row_absmax_int8_quant(x: torch.Tensor, q_max: int = 127) -> torch.Tensor:
    """Per-row symmetric absmax fake-quant matching the SC bipolar setup."""
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5) / q_max
    return (x / scale).round().clamp(-q_max, q_max) * scale


def _per_col_absmax_int8_quant(w: torch.Tensor, q_max: int = 127) -> torch.Tensor:
    """Per-row absmax on the weight (M, D) — applied along the M axis,
    mirroring SC's per-row quant on operand b."""
    scale = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5) / q_max
    return (w / scale).round().clamp(-q_max, q_max) * scale


def test_accumulate_act_scales():
    torch.manual_seed(0)
    D = 16
    x1 = torch.randn(8, D)
    x2 = torch.randn(8, D) * 3.0
    running = accumulate_act_scales(x1)
    running = accumulate_act_scales(x2, running)
    expected = torch.maximum(x1.abs().amax(dim=0), x2.abs().amax(dim=0))
    assert torch.allclose(running, expected), \
        f"calibration mismatch: max diff {(running - expected).abs().max():.3e}"
    assert running.shape == (D,)


def test_compute_smooth_scales_basic():
    torch.manual_seed(1)
    D, M = 16, 32
    act_scales = torch.rand(D) + 0.1
    weight = torch.randn(M, D)
    s = compute_smooth_scales(act_scales, weight, alpha=0.5)
    assert s.shape == (D,)
    assert torch.all(s > 0)
    assert torch.all(torch.isfinite(s))
    # alpha=0 → s = 1 / w_max ; alpha=1 → s = act_scales.
    s0 = compute_smooth_scales(act_scales, weight, alpha=0.0)
    s1 = compute_smooth_scales(act_scales, weight, alpha=1.0)
    assert torch.allclose(s0, 1.0 / weight.abs().amax(dim=0).clamp(min=1e-5))
    assert torch.allclose(s1, act_scales.clamp(min=1e-5))


def test_apply_smoothing_math_identity():
    torch.manual_seed(2)
    N, M, D = 4, 6, 32
    a = torch.randn(N, D, dtype=torch.float64)
    b = torch.randn(M, D, dtype=torch.float64)
    s = (torch.rand(D, dtype=torch.float64) + 0.5)  # in [0.5, 1.5]
    a_s, b_s = apply_smoothing(a, b, s)
    y_ref = a @ b.T
    y_smooth = a_s @ b_s.T
    diff = (y_ref - y_smooth).abs().max().item()
    assert diff < 1e-10, f"math identity violated: max diff {diff:.3e}"


def test_apply_smoothing_3d():
    torch.manual_seed(3)
    BH, N, M, D = 2, 4, 6, 32
    a = torch.randn(BH, N, D, dtype=torch.float64)
    b = torch.randn(BH, M, D, dtype=torch.float64)
    s = (torch.rand(D, dtype=torch.float64) + 0.5)
    a_s, b_s = apply_smoothing(a, b, s)
    y_ref = a @ b.transpose(-1, -2)
    y_smooth = a_s @ b_s.transpose(-1, -2)
    diff = (y_ref - y_smooth).abs().max().item()
    assert diff < 1e-10, f"3D math identity violated: max diff {diff:.3e}"


def _per_tensor_absmax_int8_quant(x: torch.Tensor, q_max: int = 127) -> torch.Tensor:
    """Per-tensor symmetric absmax fake-quant (one scale for the whole tensor)."""
    scale = x.abs().max().clamp(min=1e-5) / q_max
    return (x / scale).round().clamp(-q_max, q_max) * scale


def _per_head_absmax_int8_quant(x: torch.Tensor, q_max: int = 127) -> torch.Tensor:
    """Per-batch-slice symmetric absmax fake-quant for 3D (BH, *, *) tensors."""
    BH = x.shape[0]
    flat = x.reshape(BH, -1)
    scale = flat.abs().amax(dim=-1).clamp(min=1e-5) / q_max     # (BH,)
    shape = [BH] + [1] * (x.dim() - 1)
    s = scale.view(*shape)
    return (x / s).round().clamp(-q_max, q_max) * s


def _outlier_act_2d(N, D, outlier_channels, scale=60.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(N, D, generator=g) * 0.5
    a[:, outlier_channels] *= scale
    return a


def _outlier_act_3d(BH, N, D, outlier_channels, scale=60.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(BH, N, D, generator=g) * 0.5
    a[:, :, outlier_channels] *= scale
    return a


def test_outlier_mse_improvement_per_row():
    """Headline SmoothQuant claim under per-row activation quant."""
    torch.manual_seed(4)
    N, M, D = 128, 64, 256
    weight = torch.randn(M, D) * 0.5
    outlier_channels = torch.tensor([3, 17, 42, 88, 159])
    a = _outlier_act_2d(N, D, outlier_channels, scale=60.0, seed=4)

    y_ref = a @ weight.T

    y_plain = _per_row_absmax_int8_quant(a) @ _per_col_absmax_int8_quant(weight).T
    mse_plain = (y_plain - y_ref).pow(2).mean().item()

    s = compute_smooth_scales(accumulate_act_scales(a), weight, alpha=0.5)
    a_s, w_s = apply_smoothing(a, weight, s)
    y_smooth = _per_row_absmax_int8_quant(a_s) @ _per_col_absmax_int8_quant(w_s).T
    mse_smooth = (y_smooth - y_ref).pow(2).mean().item()

    print(f"  per_row : plain={mse_plain:.3e}  smooth={mse_smooth:.3e}  "
          f"{mse_plain / max(mse_smooth, 1e-30):6.1f}x")
    assert mse_smooth < mse_plain * 0.5


def test_outlier_mse_improvement_per_tensor():
    """SmoothQuant under per-tensor activation quant (expected: biggest win,
    because one global absmax was being set by the outlier channels)."""
    torch.manual_seed(4)
    N, M, D = 128, 64, 256
    weight = torch.randn(M, D) * 0.5
    outlier_channels = torch.tensor([3, 17, 42, 88, 159])
    a = _outlier_act_2d(N, D, outlier_channels, scale=60.0, seed=4)

    y_ref = a @ weight.T

    y_plain = _per_tensor_absmax_int8_quant(a) @ _per_tensor_absmax_int8_quant(weight).T
    mse_plain = (y_plain - y_ref).pow(2).mean().item()

    s = compute_smooth_scales(accumulate_act_scales(a), weight, alpha=0.5)
    a_s, w_s = apply_smoothing(a, weight, s)
    y_smooth = _per_tensor_absmax_int8_quant(a_s) @ _per_tensor_absmax_int8_quant(w_s).T
    mse_smooth = (y_smooth - y_ref).pow(2).mean().item()

    print(f"  per_tensor: plain={mse_plain:.3e}  smooth={mse_smooth:.3e}  "
          f"{mse_plain / max(mse_smooth, 1e-30):6.1f}x")
    assert mse_smooth < mse_plain * 0.5


def test_outlier_mse_improvement_per_head():
    """SmoothQuant under per-head (3D, per-batch-slice) activation quant.
    Outliers live in the D axis, so per-head absmax is still polluted."""
    torch.manual_seed(4)
    BH, N, M, D = 4, 32, 64, 256
    weight = torch.randn(BH, M, D) * 0.5
    outlier_channels = torch.tensor([3, 17, 42, 88, 159])
    a = _outlier_act_3d(BH, N, D, outlier_channels, scale=60.0, seed=4)

    y_ref = a @ weight.transpose(-1, -2)

    a_q = _per_head_absmax_int8_quant(a)
    w_q = _per_head_absmax_int8_quant(weight)
    y_plain = a_q @ w_q.transpose(-1, -2)
    mse_plain = (y_plain - y_ref).pow(2).mean().item()

    # Calibrate per-channel over (BH, N) — one (D,) vector shared across heads.
    # Weight max for s also takes the worst-case head: flatten (BH, M, D) → (BH*M, D).
    s = compute_smooth_scales(
        accumulate_act_scales(a),
        weight.reshape(BH * M, D),
        alpha=0.5,
    )
    a_s, w_s = apply_smoothing(a, weight, s)
    y_smooth = (_per_head_absmax_int8_quant(a_s)
                @ _per_head_absmax_int8_quant(w_s).transpose(-1, -2))
    mse_smooth = (y_smooth - y_ref).pow(2).mean().item()

    print(f"  per_head  : plain={mse_plain:.3e}  smooth={mse_smooth:.3e}  "
          f"{mse_plain / max(mse_smooth, 1e-30):6.1f}x")
    assert mse_smooth < mse_plain * 0.5


def test_apply_smoothing_offline_round_trip():
    torch.manual_seed(5)
    N, M, D = 4, 6, 32
    a = torch.randn(N, D, dtype=torch.float64)
    weight = torch.randn(M, D, dtype=torch.float64)
    s = (torch.rand(D, dtype=torch.float64) + 0.5)

    w_baked = apply_smoothing_offline(weight, s)
    y_offline = (a / s) @ w_baked.T
    y_ref = a @ weight.T
    diff = (y_offline - y_ref).abs().max().item()
    assert diff < 1e-10


def test_invalid_args():
    D = 8
    # alpha out of [0, 1]
    try:
        compute_smooth_scales(torch.ones(D), torch.ones(4, D), alpha=1.5)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for alpha=1.5")
    # shape mismatch
    try:
        compute_smooth_scales(torch.ones(D), torch.ones(4, D + 1))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for D mismatch")
    # smooth_scales not 1D
    try:
        apply_smoothing(torch.zeros(2, D), torch.zeros(3, D), torch.zeros(2, D))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for non-1D smooth_scales")


def test_sc_matmul_kwarg_wiring_cuda():
    """End-to-end: sc_matmul(a, b, smooth_scales=s) == sc_matmul(a/s, b*s).

    Requires CUDA (Triton kernels are GPU-only)."""
    if not torch.cuda.is_available():
        print("  [skip] no CUDA")
        return
    from scmp_kernels.sc.matmul import sc_matmul

    torch.manual_seed(6)
    dev = "cuda"
    N, M, D = 16, 16, 64
    a = torch.randn(N, D, device=dev) * 0.5
    a[:, 7] *= 30.0
    b = torch.randn(M, D, device=dev) * 0.5
    act_scales = accumulate_act_scales(a)
    s = compute_smooth_scales(act_scales, b, alpha=0.5)

    y_kwarg = sc_matmul(a, b, granularity="per_row", sc_prec=8,
                        smooth_scales=s)
    a_s, b_s = apply_smoothing(a, b, s)
    y_manual = sc_matmul(a_s, b_s, granularity="per_row", sc_prec=8)
    assert torch.equal(y_kwarg, y_manual), \
        f"kwarg path differs from manual smoothing: max diff " \
        f"{(y_kwarg - y_manual).abs().max():.3e}"


def main():
    tests = [
        test_accumulate_act_scales,
        test_compute_smooth_scales_basic,
        test_apply_smoothing_math_identity,
        test_apply_smoothing_3d,
        test_outlier_mse_improvement_per_row,
        test_outlier_mse_improvement_per_tensor,
        test_outlier_mse_improvement_per_head,
        test_apply_smoothing_offline_round_trip,
        test_invalid_args,
        test_sc_matmul_kwarg_wiring_cuda,
    ]
    failed = []
    for t in tests:
        name = t.__name__
        try:
            t()
            print(f"[PASS] {name}")
        except Exception as e:
            failed.append((name, e))
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
    if failed:
        print(f"\n{len(failed)} of {len(tests)} tests failed.")
        sys.exit(1)
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    main()
