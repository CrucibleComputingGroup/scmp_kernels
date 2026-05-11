"""Smoke tests for scmp_kernels.sc — verify imports succeed and the unified
sc_matmul dispatches correctly for each granularity.

Numerical correctness is checked by downstream application benchmarks
(see application/Diffusion). Here we only verify:

  1. The package imports cleanly.
  2. ``sc_matmul`` runs on CUDA for each granularity and returns the
     expected output shape.
  3. ``chunk_d`` validation raises on unsupported combinations.
  4. ``granularity="per_head"`` shape gating works.

Requires a CUDA-capable Triton install.
"""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="SC kernels require CUDA + Triton.")


def test_import_only():
    from scmp_kernels import sc_matmul                 # noqa: F401
    from scmp_kernels.sc import sc_matmul  # noqa: F401


def test_per_tensor_2d():
    from scmp_kernels import sc_matmul
    a = torch.randn(8, 64, device="cuda")
    b = torch.randn(16, 64, device="cuda")
    y = sc_matmul(a, b, granularity="per_tensor", sc_prec=8)
    assert y.shape == (8, 16)
    assert y.dtype == torch.float32


def test_per_row_2d():
    from scmp_kernels import sc_matmul
    a = torch.randn(8, 64, device="cuda")
    b = torch.randn(16, 64, device="cuda")
    y = sc_matmul(a, b, granularity="per_row", sc_prec=8)
    assert y.shape == (8, 16)


def test_per_row_3d_batched():
    from scmp_kernels import sc_matmul
    a = torch.randn(4, 8, 64, device="cuda")
    b = torch.randn(4, 16, 64, device="cuda")
    y = sc_matmul(a, b, granularity="per_row", sc_prec=8)
    assert y.shape == (4, 8, 16)


def test_per_head_bipolar():
    from scmp_kernels import sc_matmul
    q = torch.randn(16, 197, 64, device="cuda")
    k = torch.randn(16, 197, 64, device="cuda")
    y = sc_matmul(q, k, granularity="per_head", sc_prec=8)
    assert y.shape == (16, 197, 197)


def test_per_row_mlp_chunk_d():
    from scmp_kernels import sc_matmul
    # chunk_d > 0 requires per_row + bipolar + 2D.
    a = torch.randn(64, 1024, device="cuda")
    b = torch.randn(128, 1024, device="cuda")
    y = sc_matmul(a, b, granularity="per_row", chunk_d=72, sc_prec=8)
    assert y.shape == (64, 128)


# ---------------------------------------------------------------------------
# Validation gates — must raise ValueError on unsupported combinations.
# ---------------------------------------------------------------------------

def test_chunk_d_rejects_per_tensor():
    from scmp_kernels import sc_matmul
    a = torch.randn(64, 1024, device="cuda")
    b = torch.randn(128, 1024, device="cuda")
    with pytest.raises(ValueError, match="chunk_d"):
        sc_matmul(a, b, granularity="per_tensor", chunk_d=72)


def test_chunk_d_rejects_unipolar():
    from scmp_kernels import sc_matmul
    a = torch.randn(64, 1024, device="cuda")
    b = torch.randn(128, 1024, device="cuda")
    with pytest.raises(ValueError, match="chunk_d"):
        sc_matmul(a, b, granularity="per_row", mode="unipolar", chunk_d=72)


def test_chunk_d_rejects_3d():
    from scmp_kernels import sc_matmul
    a = torch.randn(4, 64, 1024, device="cuda")
    b = torch.randn(4, 128, 1024, device="cuda")
    with pytest.raises(ValueError, match="chunk_d"):
        sc_matmul(a, b, granularity="per_row", chunk_d=72)


def test_per_head_rejects_2d():
    from scmp_kernels import sc_matmul
    a = torch.randn(64, 1024, device="cuda")
    b = torch.randn(128, 1024, device="cuda")
    with pytest.raises(ValueError, match="per_head"):
        sc_matmul(a, b, granularity="per_head")


def test_per_head_rejects_unipolar():
    from scmp_kernels import sc_matmul
    q = torch.randn(4, 8, 64, device="cuda")
    k = torch.randn(4, 8, 64, device="cuda")
    with pytest.raises(ValueError, match="per_head"):
        sc_matmul(q, k, granularity="per_head", mode="unipolar")


def test_unknown_granularity():
    from scmp_kernels import sc_matmul
    a = torch.randn(8, 64, device="cuda")
    b = torch.randn(16, 64, device="cuda")
    with pytest.raises(ValueError, match="granularity"):
        sc_matmul(a, b, granularity="per_block")


def test_unknown_mode():
    from scmp_kernels import sc_matmul
    a = torch.randn(8, 64, device="cuda")
    b = torch.randn(16, 64, device="cuda")
    with pytest.raises(ValueError, match="mode"):
        sc_matmul(a, b, mode="ternary")
