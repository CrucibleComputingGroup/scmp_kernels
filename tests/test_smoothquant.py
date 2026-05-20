"""SmoothQuant pre-quantization transform tests.

Covers the four public helpers in :mod:`scmp_kernels.quant.smoothquant` and
the ``smooth_scales`` kwarg on :func:`scmp_kernels.sc_matmul`:

  1. ``accumulate_act_scales`` — running per-channel max-abs.
  2. ``compute_smooth_scales`` — alpha=0 / alpha=1 closed forms + shape sanity.
  3. ``apply_smoothing`` — math identity ``(a/s) @ (b*s).T == a @ b.T``
     (2D and 3D).
  4. Outlier MSE improvement under per-row / per-tensor / per-head simulated
     int8 quant — confirms the headline claim holds for every granularity.
  5. ``sc_matmul(..., smooth_scales=s)`` wiring (CUDA-only).

The helpers themselves are pure PyTorch, so most tests run on CPU. The
``sc_matmul`` kwarg test self-skips without CUDA.
"""
from __future__ import annotations

import pytest
import torch

from scmp_kernels.quant.smoothquant import (
    accumulate_act_scales,
    compute_smooth_scales,
    apply_smoothing,
    apply_smoothing_offline,
)


# ---------------- helpers: CPU-emulated absmax fake-quant ----------------

def _per_row_absmax_int8(x: torch.Tensor, q_max: int = 127) -> torch.Tensor:
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5) / q_max
    return (x / scale).round().clamp(-q_max, q_max) * scale


def _per_tensor_absmax_int8(x: torch.Tensor, q_max: int = 127) -> torch.Tensor:
    scale = x.abs().max().clamp(min=1e-5) / q_max
    return (x / scale).round().clamp(-q_max, q_max) * scale


def _per_head_absmax_int8(x: torch.Tensor, q_max: int = 127) -> torch.Tensor:
    BH = x.shape[0]
    scale = x.reshape(BH, -1).abs().amax(dim=-1).clamp(min=1e-5) / q_max
    return (x / scale.view(BH, *([1] * (x.dim() - 1)))).round().clamp(
        -q_max, q_max) * scale.view(BH, *([1] * (x.dim() - 1)))


def _outlier_act_2d(N, D, channels, scale=60.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(N, D, generator=g) * 0.5
    a[:, channels] *= scale
    return a


def _outlier_act_3d(BH, N, D, channels, scale=60.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(BH, N, D, generator=g) * 0.5
    a[:, :, channels] *= scale
    return a


# ---------------- pure helper tests (CPU) ----------------

def test_accumulate_act_scales():
    torch.manual_seed(0)
    D = 16
    x1 = torch.randn(8, D)
    x2 = torch.randn(8, D) * 3.0
    running = accumulate_act_scales(x1)
    running = accumulate_act_scales(x2, running)
    expected = torch.maximum(x1.abs().amax(dim=0), x2.abs().amax(dim=0))
    assert running.shape == (D,)
    assert torch.allclose(running, expected)


def test_compute_smooth_scales_closed_forms():
    torch.manual_seed(1)
    D, M = 16, 32
    act_scales = torch.rand(D) + 0.1
    weight = torch.randn(M, D)

    s = compute_smooth_scales(act_scales, weight, alpha=0.5)
    assert s.shape == (D,)
    assert torch.all(s > 0) and torch.all(torch.isfinite(s))

    # alpha=0 → s = 1 / w_max ; alpha=1 → s = act_scales
    s0 = compute_smooth_scales(act_scales, weight, alpha=0.0)
    s1 = compute_smooth_scales(act_scales, weight, alpha=1.0)
    assert torch.allclose(s0, 1.0 / weight.abs().amax(dim=0).clamp(min=1e-5))
    assert torch.allclose(s1, act_scales.clamp(min=1e-5))


def test_apply_smoothing_math_identity_2d():
    torch.manual_seed(2)
    N, M, D = 4, 6, 32
    a = torch.randn(N, D, dtype=torch.float64)
    b = torch.randn(M, D, dtype=torch.float64)
    s = torch.rand(D, dtype=torch.float64) + 0.5
    a_s, b_s = apply_smoothing(a, b, s)
    assert (a @ b.T - a_s @ b_s.T).abs().max().item() < 1e-10


def test_apply_smoothing_math_identity_3d():
    torch.manual_seed(3)
    BH, N, M, D = 2, 4, 6, 32
    a = torch.randn(BH, N, D, dtype=torch.float64)
    b = torch.randn(BH, M, D, dtype=torch.float64)
    s = torch.rand(D, dtype=torch.float64) + 0.5
    a_s, b_s = apply_smoothing(a, b, s)
    y_ref = a @ b.transpose(-1, -2)
    y_smooth = a_s @ b_s.transpose(-1, -2)
    assert (y_ref - y_smooth).abs().max().item() < 1e-10


def test_apply_smoothing_offline_round_trip():
    torch.manual_seed(5)
    N, M, D = 4, 6, 32
    a = torch.randn(N, D, dtype=torch.float64)
    weight = torch.randn(M, D, dtype=torch.float64)
    s = torch.rand(D, dtype=torch.float64) + 0.5
    w_baked = apply_smoothing_offline(weight, s)
    diff = ((a / s) @ w_baked.T - a @ weight.T).abs().max().item()
    assert diff < 1e-10


def test_invalid_args():
    D = 8
    with pytest.raises(ValueError):
        compute_smooth_scales(torch.ones(D), torch.ones(4, D), alpha=1.5)
    with pytest.raises(ValueError):
        compute_smooth_scales(torch.ones(D), torch.ones(4, D + 1))
    with pytest.raises(ValueError):
        apply_smoothing(torch.zeros(2, D), torch.zeros(3, D), torch.zeros(2, D))


# ---------------- headline claim: outlier MSE improvement ----------------

@pytest.mark.parametrize(
    "granularity,act_quant,b_quant,a_factory,b_factory",
    [
        ("per_row",
         _per_row_absmax_int8, _per_row_absmax_int8,
         lambda: _outlier_act_2d(128, 256, torch.tensor([3, 17, 42, 88, 159]),
                                 60.0, 4),
         lambda: torch.randn(64, 256, generator=torch.Generator().manual_seed(40)) * 0.5),
        ("per_tensor",
         _per_tensor_absmax_int8, _per_tensor_absmax_int8,
         lambda: _outlier_act_2d(128, 256, torch.tensor([3, 17, 42, 88, 159]),
                                 60.0, 4),
         lambda: torch.randn(64, 256, generator=torch.Generator().manual_seed(40)) * 0.5),
    ],
)
def test_outlier_mse_improvement_2d(granularity, act_quant, b_quant,
                                    a_factory, b_factory):
    a = a_factory()
    weight = b_factory()
    y_ref = a @ weight.T

    mse_plain = (act_quant(a) @ b_quant(weight).T - y_ref).pow(2).mean().item()

    s = compute_smooth_scales(accumulate_act_scales(a), weight, alpha=0.5)
    a_s, w_s = apply_smoothing(a, weight, s)
    mse_smooth = (act_quant(a_s) @ b_quant(w_s).T - y_ref).pow(2).mean().item()

    assert mse_smooth < mse_plain * 0.5, (
        f"{granularity}: plain={mse_plain:.3e} smooth={mse_smooth:.3e}")


def test_outlier_mse_improvement_per_head():
    BH, N, M, D = 4, 32, 64, 256
    weight = torch.randn(BH, M, D,
                         generator=torch.Generator().manual_seed(40)) * 0.5
    a = _outlier_act_3d(BH, N, D, torch.tensor([3, 17, 42, 88, 159]), 60.0, 4)

    y_ref = a @ weight.transpose(-1, -2)
    mse_plain = (
        _per_head_absmax_int8(a) @ _per_head_absmax_int8(weight).transpose(-1, -2)
        - y_ref
    ).pow(2).mean().item()

    s = compute_smooth_scales(
        accumulate_act_scales(a),
        weight.reshape(BH * M, D),
        alpha=0.5,
    )
    a_s, w_s = apply_smoothing(a, weight, s)
    mse_smooth = (
        _per_head_absmax_int8(a_s)
        @ _per_head_absmax_int8(w_s).transpose(-1, -2)
        - y_ref
    ).pow(2).mean().item()

    assert mse_smooth < mse_plain * 0.5, (
        f"per_head: plain={mse_plain:.3e} smooth={mse_smooth:.3e}")


# ---------------- sc_matmul kwarg wiring (CUDA only) ----------------

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="SC kernels require CUDA + Triton.")
def test_sc_matmul_smooth_scales_kwarg_equivalence():
    """sc_matmul(a, b, smooth_scales=s) must equal sc_matmul(a/s, b*s)."""
    from scmp_kernels import sc_matmul

    torch.manual_seed(6)
    N, M, D = 16, 16, 64
    a = torch.randn(N, D, device="cuda") * 0.5
    a[:, 7] *= 30.0
    b = torch.randn(M, D, device="cuda") * 0.5
    s = compute_smooth_scales(accumulate_act_scales(a), b, alpha=0.5)

    y_kwarg = sc_matmul(a, b, granularity="per_row", sc_prec=8,
                        smooth_scales=s)
    a_s, b_s = apply_smoothing(a, b, s)
    y_manual = sc_matmul(a_s, b_s, granularity="per_row", sc_prec=8)
    assert torch.equal(y_kwarg, y_manual)
