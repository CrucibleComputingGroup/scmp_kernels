"""
Enable-signal stochastic computing matrix multiplication (pure-Python, CPU path).

Implements the enable-signal (conditional BSG) mechanism from UnarySim's FSUMul:
operand B's RNG index advances only when operand A's current bit is 1. This
preserves Sobol low-discrepancy properties for the bits that actually contribute
to the AND result, improving multiplication accuracy.

Two methods are provided:

- ``k_shortcut`` — table-based. Builds a ``cum_indicator(D, stoc_len+1, V)``
  prefix-sum table for B's RNG sequence and a popcount ``k_table(D, V)`` for A's,
  then gathers AND-counts in O(1) per element. Faster, but memory scales with
  ``D · (stoc_len + 1) · (2^sc_prec + 1)``.
- ``cycle_by_cycle`` — compact, table-free. Simulates the bitstream
  tick-by-tick (advancing B's RNG index only when A's bit is 1), accumulating
  AND-bits. Slower but only carries the RNG sequences themselves.

Both are mathematically equivalent — k_shortcut is just a vectorised lookup of
the same prefix sums cycle_by_cycle accumulates. The pair is ported verbatim
from ``scmp_llm_old/SC/sc_enable.py`` so the cycle_by_cycle reference can serve
as the oracle for the new table-free path in ``scmp_kernels``.

Supports:

- Bipolar mode (sign-magnitude): exact sign from integer, unipolar AND+enable
  on magnitudes.
- Unipolar mode (asymmetric): direct AND+enable with zero-point correction.

Quantization is per-tensor (one ``max_fp / min_fp`` for the whole operand).
Other granularities are not supported here — the table-based Triton path in
``kernels.py`` handles per-row / per-head.
"""

from __future__ import annotations

from typing import Optional

import torch

from .sng import RNGPool, SNGBank


# =============================================================================
# Helper functions
# =============================================================================


def _compute_boundary(mag: torch.Tensor, max_val: float, max_rng_val: int) -> torch.Tensor:
    """Integer boundary for stochastic comparison.

    ``boundary = round(mag * max_rng_val / max_val)`` mapping
    ``[0, max_val] -> [0, max_rng_val]``.
    """
    return (mag.float() * max_rng_val / max_val).round().long()


def _build_cum_indicator_table(rng_b: torch.Tensor, max_rng_val: int) -> torch.Tensor:
    """Cumulative indicator table for B's RNG sequence.

    ``cum_indicator[d, k, v] = |{i < k : v > rng_b[d, i]}|`` — for each dim ``d``
    and prefix length ``k``, how many of the first ``k`` RNG values are strictly
    less than each possible boundary ``v``.

    Returns: ``(D, stoc_len + 1, max_rng_val + 1)`` int32.
    """
    D, stoc_len = rng_b.shape
    device = rng_b.device

    v_range = torch.arange(max_rng_val + 1, device=device)  # (V,)
    rng_b_exp = rng_b.unsqueeze(2)  # (D, stoc_len, 1)
    delta = (v_range.unsqueeze(0).unsqueeze(0) > rng_b_exp).int()  # (D, stoc_len, V)
    cum_inner = delta.cumsum(dim=1)  # (D, stoc_len, V)
    zeros = torch.zeros(D, 1, max_rng_val + 1, dtype=torch.int32, device=device)
    cum = torch.cat([zeros, cum_inner], dim=1)  # (D, stoc_len+1, V)
    return cum


def _compute_k_table(rng_a: torch.Tensor, max_rng_val: int) -> torch.Tensor:
    """Popcount table for A's RNG sequence.

    ``k_table[d, v] = |{t : v > rng_a[d, t]}|`` — how many 1-bits A would have
    for boundary ``v``.

    Returns: ``(D, max_rng_val + 1)`` int32.
    """
    D, stoc_len = rng_a.shape
    device = rng_a.device

    v_range = torch.arange(max_rng_val + 1, device=device)
    indicator = (v_range.unsqueeze(0).unsqueeze(0) > rng_a.unsqueeze(2))  # (D, stoc_len, V)
    k_table = indicator.sum(dim=1).int()  # (D, V)
    return k_table


# =============================================================================
# Core multiplication functions
# =============================================================================


def _enable_mul_cycle_by_cycle(
    mag_a: torch.Tensor,
    mag_b: torch.Tensor,
    rng_a: torch.Tensor,
    rng_b: torch.Tensor,
    max_val: float,
    sc_prec: int,
    return_per_dim: bool = False,
) -> torch.Tensor:
    """Compact (no-table) enable-signal multiplication via exact cycle-by-cycle simulation.

    Matches UnarySim FSUMul semantics: B's RNG index advances only when A's
    current bit is 1 (AND gate with enable signal).

    This is the **compact** path — it carries only the RNG sequences themselves,
    no ``cum_indicator`` table. Use as the reference oracle for ``k_shortcut``.
    """
    N, D = mag_a.shape
    M = mag_b.shape[0]
    stoc_len = 2 ** sc_prec
    max_rng_val = 2 ** sc_prec
    device = mag_a.device

    boundary_a = _compute_boundary(mag_a, max_val, max_rng_val)  # (N, D)
    boundary_b = _compute_boundary(mag_b, max_val, max_rng_val)  # (M, D)

    enable_idx = torch.zeros(N, D, dtype=torch.long, device=device)

    if return_per_dim:
        counts = torch.zeros(N, M, D, dtype=torch.long, device=device)
    else:
        counts = torch.zeros(N, M, dtype=torch.long, device=device)

    d_indices = torch.arange(D, device=device).unsqueeze(0).expand(N, D)  # (N, D)

    for t in range(stoc_len):
        rng_a_t = rng_a[:, t]  # (D,)
        a_bits = (boundary_a > rng_a_t.unsqueeze(0)).long()  # (N, D)

        rng_b_at_idx = rng_b[d_indices, enable_idx]  # (N, D)
        b_bits = (boundary_b.unsqueeze(0) > rng_b_at_idx.unsqueeze(1)).long()  # (N, M, D)

        output_bits = a_bits.unsqueeze(1) * b_bits  # (N, M, D)
        if return_per_dim:
            counts += output_bits
        else:
            counts += output_bits.sum(dim=2)

        enable_idx += a_bits

    return counts


def _enable_mul_k_shortcut(
    mag_a: torch.Tensor,
    mag_b: torch.Tensor,
    rng_a: torch.Tensor,
    rng_b: torch.Tensor,
    max_val: float,
    sc_prec: int,
    return_per_dim: bool = False,
) -> torch.Tensor:
    """Table-based enable-signal multiplication via k-shortcut (prefix-sum lookup).

    Mathematically equivalent to cycle-by-cycle but eliminates the sequential
    dependency by precomputing lookup tables. If ``k_a = popcount(A)``, then
    ``count = |{i < k_a : boundary_b > rng_b[i]}|`` — directly readable from
    ``cum_indicator[d, k_a, boundary_b]``.
    """
    N, D = mag_a.shape
    M = mag_b.shape[0]
    max_rng_val = 2 ** sc_prec
    device = mag_a.device

    boundary_a = _compute_boundary(mag_a, max_val, max_rng_val)  # (N, D)
    boundary_b = _compute_boundary(mag_b, max_val, max_rng_val)  # (M, D)

    cum_indicator = _build_cum_indicator_table(rng_b, max_rng_val)  # (D, stoc_len+1, V)
    k_table = _compute_k_table(rng_a, max_rng_val)                  # (D, V)

    if return_per_dim:
        d_idx = torch.arange(D, device=device)
        k_a = k_table[d_idx.unsqueeze(0), boundary_a]  # (N, D)
        d_exp = d_idx.unsqueeze(0).unsqueeze(0).expand(N, M, D)
        k_exp = k_a.unsqueeze(1).expand(N, M, D)
        b_exp = boundary_b.unsqueeze(0).expand(N, M, D)
        counts = cum_indicator[d_exp, k_exp, b_exp].long()
    else:
        d_idx = torch.arange(D, device=device)
        k_a = k_table[d_idx.unsqueeze(0), boundary_a]  # (N, D)
        counts = torch.zeros(N, M, dtype=torch.long, device=device)
        for d in range(D):
            k_idx = k_a[:, d].long()
            b_idx = boundary_b[:, d].long()
            row_indexed = cum_indicator[d][k_idx]      # (N, V)
            counts += row_indexed[:, b_idx].long()     # (N, M)

    return counts


# =============================================================================
# Bipolar and unipolar wrappers
# =============================================================================


def _sc_matmul_enable_bipolar(
    a, b, max_fp_a, min_fp_a, max_fp_b, min_fp_b, sc_prec,
    rand_seqs_a_t, rand_seqs_b_t, N, D, M, stoc_len, method,
):
    """Bipolar enable-signal SC matmul using sign-magnitude decomposition."""
    q_max = 2 ** (sc_prec - 1) - 1
    q_min = -(2 ** (sc_prec - 1))

    abs_max_a = max(abs(max_fp_a), abs(min_fp_a), 1e-5)
    abs_max_b = max(abs(max_fp_b), abs(min_fp_b), 1e-5)
    scale_a = abs_max_a / q_max
    scale_b = abs_max_b / q_max

    a_int = (a / scale_a).round().clamp(q_min, q_max)
    b_int = (b / scale_b).round().clamp(q_min, q_max)

    sign_a = torch.sign(a_int)
    sign_b = torch.sign(b_int)
    mag_a = a_int.abs()
    mag_b = b_int.abs()

    mul_fn = _enable_mul_cycle_by_cycle if method == "cycle_by_cycle" else _enable_mul_k_shortcut

    counts = mul_fn(mag_a, mag_b, rand_seqs_a_t, rand_seqs_b_t,
                    float(q_max), sc_prec, return_per_dim=True)  # (N, M, D)

    decoded = counts.float() * float(q_max * q_max) / float(stoc_len)
    signed = decoded * sign_a.unsqueeze(1) * sign_b.unsqueeze(0)
    result = signed.sum(dim=2) * (scale_a * scale_b)

    return result


def _sc_matmul_enable_unipolar(
    a, b, max_fp_a, min_fp_a, max_fp_b, min_fp_b, sc_prec,
    rand_seqs_a_t, rand_seqs_b_t, N, D, M, stoc_len, method,
):
    """Unipolar enable-signal SC matmul with asymmetric quantization + zero-point correction."""
    q_max = 2 ** sc_prec - 1
    q_min = 0

    range_a = max(max_fp_a - min_fp_a, 1e-5)
    scale_a = range_a / q_max
    zp_a = round(-min_fp_a / scale_a)
    zp_a = max(q_min, min(q_max, zp_a))

    range_b = max(max_fp_b - min_fp_b, 1e-5)
    scale_b = range_b / q_max
    zp_b = round(-min_fp_b / scale_b)
    zp_b = max(q_min, min(q_max, zp_b))

    a_int = (a / scale_a + zp_a).round().clamp(q_min, q_max)
    b_int = (b / scale_b + zp_b).round().clamp(q_min, q_max)

    mul_fn = _enable_mul_cycle_by_cycle if method == "cycle_by_cycle" else _enable_mul_k_shortcut

    counts = mul_fn(a_int, b_int, rand_seqs_a_t, rand_seqs_b_t,
                    float(q_max), sc_prec, return_per_dim=False)  # (N, M)

    sc_raw = counts.float() * float(q_max * q_max) / float(stoc_len)

    zp_a_f = float(zp_a)
    zp_b_f = float(zp_b)
    a_sum = a_int.sum(dim=-1, keepdim=True)            # (N, 1)
    b_sum = b_int.sum(dim=-1, keepdim=True)            # (M, 1)

    correction = -zp_b_f * a_sum - zp_a_f * b_sum.transpose(-2, -1) + D * zp_a_f * zp_b_f
    corrected = sc_raw + correction

    result = corrected * (scale_a * scale_b)
    return result


# =============================================================================
# Public API
# =============================================================================


@torch.no_grad()
def sc_matmul_enable(
    a: torch.Tensor,
    b: torch.Tensor,
    max_fp_a: float,
    min_fp_a: float,
    max_fp_b: Optional[float] = None,
    min_fp_b: Optional[float] = None,
    mode: str = "bipolar",
    sc_prec: int = 8,
    config: Optional[dict] = None,
    method: str = "k_shortcut",
) -> torch.Tensor:
    """Enable-signal stochastic computing matrix multiplication: ``a @ b.T``.

    Pure-Python CPU reference. FP-in, FP-out. Uses the enable-signal
    (conditional BSG) mechanism where B's RNG index advances only when A's
    current bit is 1.

    Args:
        a: Left operand, shape ``(N, D)`` or ``(B, N, D)``.
        b: Right operand, shape ``(M, D)`` or ``(B, M, D)``.
        max_fp_a, min_fp_a: per-tensor range for ``a``.
        max_fp_b, min_fp_b: per-tensor range for ``b``. Default to A's.
        mode: ``"bipolar"`` (sign-magnitude) or ``"unipolar"`` (asymmetric AND).
        sc_prec: SC precision. ``stoc_len = 2 ** sc_prec``.
        config: optional Sobol RNG/SNG config dict. Auto-built when ``None``.
        method: ``"k_shortcut"`` (table-based, faster) or ``"cycle_by_cycle"``
            (compact, no tables — the reference oracle).

    Returns:
        FP tensor, shape ``(N, M)`` or ``(B, N, M)``.
    """
    if max_fp_b is None:
        max_fp_b = max_fp_a
    if min_fp_b is None:
        min_fp_b = min_fp_a

    if a.dim() == 3:
        return _sc_matmul_enable_batched(
            a, b, max_fp_a, min_fp_a, max_fp_b, min_fp_b, mode, sc_prec, config, method,
        )

    if a.dim() != 2 or b.dim() != 2:
        raise ValueError(f"Expected 2D tensors, got a:{a.dim()}D, b:{b.dim()}D")
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"Embedding dim mismatch: a={a.shape[1]}, b={b.shape[1]}")
    if method not in ("k_shortcut", "cycle_by_cycle"):
        raise ValueError(f"Unknown method: {method!r}. Expected 'k_shortcut' or 'cycle_by_cycle'.")

    N, D = a.shape
    M = b.shape[0]
    stoc_len = 2 ** sc_prec

    if config is None:
        from .config_helpers import make_sobol_simple_config
        config = make_sobol_simple_config(D, D, sc_prec)

    device = a.device
    a = a.float()
    b = b.float()

    rng_pool = RNGPool(config["rng_pool"], sc_prec)
    sng_a = SNGBank(rng_pool, config["sng"]["q"])
    sng_b = SNGBank(rng_pool, config["sng"]["k"])

    rand_seqs_a = sng_a.get_all_sequences(stoc_len)
    rand_seqs_b = sng_b.get_all_sequences(stoc_len)
    rand_seqs_a_t = torch.as_tensor(rand_seqs_a, dtype=torch.long, device=device)
    rand_seqs_b_t = torch.as_tensor(rand_seqs_b, dtype=torch.long, device=device)

    if mode == "bipolar":
        return _sc_matmul_enable_bipolar(
            a, b, max_fp_a, min_fp_a, max_fp_b, min_fp_b, sc_prec,
            rand_seqs_a_t, rand_seqs_b_t, N, D, M, stoc_len, method,
        )
    if mode == "unipolar":
        return _sc_matmul_enable_unipolar(
            a, b, max_fp_a, min_fp_a, max_fp_b, min_fp_b, sc_prec,
            rand_seqs_a_t, rand_seqs_b_t, N, D, M, stoc_len, method,
        )
    raise ValueError(f"Unknown mode: {mode!r}. Expected 'bipolar' or 'unipolar'.")


def _sc_matmul_enable_batched(
    a, b, max_fp_a, min_fp_a, max_fp_b, min_fp_b, mode, sc_prec, config, method,
):
    """Per-batch loop wrapper for 3D inputs."""
    B = a.shape[0]
    results = []
    for i in range(B):
        results.append(sc_matmul_enable(
            a[i], b[i], max_fp_a, min_fp_a, max_fp_b, min_fp_b,
            mode=mode, sc_prec=sc_prec, config=config, method=method,
        ))
    return torch.stack(results, dim=0)


__all__ = ["sc_matmul_enable"]
