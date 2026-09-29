"""CPU correctness tests for the enable-signal SC paths.

The Triton GPU kernels in ``scmp_kernels.sc.kernels`` cannot run on CPU, but the
pure-Python port in ``scmp_kernels.sc.sc_enable`` does. These tests verify:

  1. ``cycle_by_cycle`` (compact, no tables) and ``k_shortcut`` (table-based)
     produce **bit-exact** AND-counts. They are mathematically equivalent —
     k_shortcut is just a vectorized lookup of the same prefix sums
     cycle_by_cycle accumulates — so the float outputs must match exactly.
  2. Both methods approximate a real matmul: normalized RMSE vs ``a @ b.T``
     under 0.1, in both bipolar and unipolar modes.
  3. ``sc_matmul(..., method="compact"|"table")`` dispatches to the CPU path
     and matches direct ``sc_matmul_enable`` calls.
  4. Scope gates: non-per_tensor granularity with non-auto method raises.

No CUDA needed.
"""
from __future__ import annotations

import pytest
import torch


# ---------------------------------------------------------------------------
# Direct sc_matmul_enable: cycle_by_cycle (compact) == k_shortcut (table)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["bipolar", "unipolar"])
def test_compact_matches_table_exactly_2d(mode):
    from scmp_kernels.sc.sc_enable import sc_matmul_enable

    torch.manual_seed(0)
    N, D, M = 4, 32, 6
    if mode == "bipolar":
        a = torch.randn(N, D) * 3.0
        b = torch.randn(M, D) * 3.0
        max_fp, min_fp = float(max(a.abs().max(), b.abs().max())), None
        # symmetric per-tensor range
        rng_max = max(a.abs().max().item(), b.abs().max().item())
        max_fp_a, min_fp_a = rng_max, -rng_max
    else:
        a = torch.rand(N, D)
        b = torch.rand(M, D)
        max_fp_a, min_fp_a = 1.0, 0.0

    y_compact = sc_matmul_enable(
        a, b, max_fp_a, min_fp_a,
        mode=mode, sc_prec=6, method="cycle_by_cycle",
    )
    y_table = sc_matmul_enable(
        a, b, max_fp_a, min_fp_a,
        mode=mode, sc_prec=6, method="k_shortcut",
    )
    # Same integer counts, same float decode → must be bit-exact.
    assert torch.equal(y_compact, y_table), (
        f"compact vs table differ: max diff "
        f"{(y_compact - y_table).abs().max().item()}"
    )


def test_compact_matches_table_exactly_3d_bipolar():
    from scmp_kernels.sc.sc_enable import sc_matmul_enable

    torch.manual_seed(1)
    B, N, D, M = 2, 3, 16, 4
    a = torch.randn(B, N, D) * 2.0
    b = torch.randn(B, M, D) * 2.0
    rng_max = max(a.abs().max().item(), b.abs().max().item())

    y_compact = sc_matmul_enable(
        a, b, rng_max, -rng_max,
        mode="bipolar", sc_prec=5, method="cycle_by_cycle",
    )
    y_table = sc_matmul_enable(
        a, b, rng_max, -rng_max,
        mode="bipolar", sc_prec=5, method="k_shortcut",
    )
    assert y_compact.shape == (B, N, M)
    assert torch.equal(y_compact, y_table)


# ---------------------------------------------------------------------------
# Both methods are reasonable matmul approximations
# ---------------------------------------------------------------------------

def test_bipolar_rmse_vs_ground_truth():
    from scmp_kernels.sc.sc_enable import sc_matmul_enable

    torch.manual_seed(42)
    N, D, M = 8, 64, 8
    a = torch.randn(N, D) * 3.0
    b = torch.randn(M, D) * 3.0
    max_fp = max(a.abs().max().item(), b.abs().max().item())

    gt = a @ b.T
    y = sc_matmul_enable(a, b, max_fp, -max_fp, mode="bipolar",
                         sc_prec=8, method="k_shortcut")

    max_dot = D * max_fp * max_fp
    rmse = ((y - gt) ** 2).mean().sqrt().item() / max_dot
    assert rmse < 0.1, f"bipolar SC RMSE too high: {rmse}"


def test_unipolar_rmse_vs_ground_truth():
    from scmp_kernels.sc.sc_enable import sc_matmul_enable

    torch.manual_seed(42)
    N, D, M = 8, 64, 8
    a = torch.rand(N, D)
    b = torch.rand(M, D)

    gt = a @ b.T
    y = sc_matmul_enable(a, b, 1.0, 0.0, mode="unipolar",
                         sc_prec=8, method="k_shortcut")

    rmse = ((y - gt) ** 2).mean().sqrt().item() / float(D)
    assert rmse < 0.1, f"unipolar SC RMSE too high: {rmse}"


# ---------------------------------------------------------------------------
# High-level sc_matmul flag dispatches to CPU sc_enable
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["compact", "table"])
@pytest.mark.parametrize("mode", ["bipolar", "unipolar"])
def test_sc_matmul_method_flag_cpu(method, mode):
    from scmp_kernels import sc_matmul
    from scmp_kernels.sc.sc_enable import sc_matmul_enable

    torch.manual_seed(7)
    if mode == "bipolar":
        a = torch.randn(4, 16) * 2.0
        b = torch.randn(6, 16) * 2.0
    else:
        a = torch.rand(4, 16)
        b = torch.rand(6, 16)

    # High-level call with method flag
    y_high = sc_matmul(a, b, granularity="per_tensor", mode=mode,
                       sc_prec=6, method=method)

    # Reference: call sc_matmul_enable directly with the same per-tensor range
    cpu_method = "cycle_by_cycle" if method == "compact" else "k_shortcut"
    y_ref = sc_matmul_enable(
        a, b,
        max_fp_a=a.max().item(), min_fp_a=a.min().item(),
        max_fp_b=b.max().item(), min_fp_b=b.min().item(),
        mode=mode, sc_prec=6, method=cpu_method,
    )
    assert torch.equal(y_high, y_ref)


def test_sc_matmul_compact_and_table_agree_high_level():
    from scmp_kernels import sc_matmul

    torch.manual_seed(11)
    a = torch.randn(4, 32) * 2.0
    b = torch.randn(5, 32) * 2.0

    y_compact = sc_matmul(a, b, granularity="per_tensor",
                          sc_prec=6, method="compact")
    y_table = sc_matmul(a, b, granularity="per_tensor",
                        sc_prec=6, method="table")
    assert torch.equal(y_compact, y_table)


# ---------------------------------------------------------------------------
# Scope gates
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["compact", "table"])
def test_method_rejects_per_row(method):
    from scmp_kernels import sc_matmul
    a = torch.randn(4, 16)
    b = torch.randn(5, 16)
    with pytest.raises(ValueError, match="per_tensor"):
        sc_matmul(a, b, granularity="per_row", method=method)


@pytest.mark.parametrize("method", ["compact", "table"])
def test_method_rejects_per_head(method):
    from scmp_kernels import sc_matmul
    a = torch.randn(2, 4, 16)
    b = torch.randn(2, 5, 16)
    with pytest.raises(ValueError, match="per_tensor"):
        sc_matmul(a, b, granularity="per_head", method=method)


@pytest.mark.parametrize("method", ["compact", "table"])
def test_method_rejects_chunk_d(method):
    from scmp_kernels import sc_matmul
    a = torch.randn(4, 64)
    b = torch.randn(5, 64)
    with pytest.raises(ValueError, match="chunk_d"):
        sc_matmul(a, b, granularity="per_tensor", chunk_d=8, method=method)


def test_unknown_method_raises():
    from scmp_kernels import sc_matmul
    a = torch.randn(4, 16)
    b = torch.randn(5, 16)
    with pytest.raises(ValueError, match="method"):
        sc_matmul(a, b, granularity="per_tensor", method="lookup")
