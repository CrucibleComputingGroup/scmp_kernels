"""SmoothQuant pre-quantization transform.

Mathematically equivalent diagonal rescaling along the shared inner dim ``D``
that migrates per-channel activation outliers into the weight, leaving the
matmul output unchanged but making both operands easier to quantize:

    Y = A @ B.T = (A * s^-1) @ (B * s).T,    s_j > 0

The smoothing vector ``s`` is built once from calibration statistics:

    s_j = act_max[j] ** alpha  /  weight_max[j] ** (1 - alpha)

``act_max`` is the per-channel max-abs over a few hundred representative
activations (collect with :func:`accumulate_act_scales`); ``weight_max`` is
the per-input-channel max-abs of the weight ``B`` (shape ``(M, D)``).

This module ships only the transform — the int quantization happens in the
existing fused / grouped kernels. The transform is wired into
``sc_matmul`` via the ``smooth_scales=`` kwarg; callers using the lower-level
kernels can apply :func:`apply_smoothing` themselves.

Reference: Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training
Quantization for Large Language Models", ICML 2023.
"""
from __future__ import annotations

from typing import Optional

import torch


@torch.no_grad()
def accumulate_act_scales(
    x: torch.Tensor,
    running: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-channel max-abs accumulator for SmoothQuant calibration.

    Args:
        x: ``(..., D)`` float tensor — the activation entering a linear layer.
        running: previous ``(D,)`` running max, or ``None`` for the first call.

    Returns:
        Updated ``(D,)`` running max-abs (on ``x``'s device/dtype).
    """
    flat = x.detach().abs().reshape(-1, x.shape[-1])
    cur = flat.amax(dim=0)
    if running is None:
        return cur.clone()
    return torch.maximum(running.to(device=cur.device, dtype=cur.dtype), cur)


@torch.no_grad()
def compute_smooth_scales(
    act_scales: torch.Tensor,
    weight: torch.Tensor,
    alpha: float = 0.5,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Build the SmoothQuant per-channel scale vector.

    Args:
        act_scales: ``(D,)`` per-channel max-abs of the activation.
        weight: ``(M, D)`` weight tensor of the linear layer (the operand
            ``b`` in :func:`sc_matmul`). For multi-tensor sharing (e.g. q/k/v
            projections in upstream), feed the column-wise stacked weight
            so ``weight_max`` covers all sharing layers.
        alpha: migration strength in ``[0, 1]``. Larger ``alpha`` pushes more
            difficulty into the weight. Upstream defaults: 0.5 for OPT,
            0.8–0.9 for Llama.
        eps: numerical floor to avoid division by zero.

    Returns:
        ``(D,)`` smoothing vector on ``weight``'s device/dtype.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if act_scales.dim() != 1:
        raise ValueError(
            f"act_scales must be 1D (D,), got shape {tuple(act_scales.shape)}")
    if weight.dim() != 2:
        raise ValueError(
            f"weight must be 2D (M, D), got shape {tuple(weight.shape)}")
    if weight.shape[-1] != act_scales.shape[0]:
        raise ValueError(
            f"D mismatch: weight.shape[-1]={weight.shape[-1]} vs "
            f"act_scales.shape[0]={act_scales.shape[0]}")

    device, dtype = weight.device, weight.dtype
    a = act_scales.to(device=device, dtype=dtype).clamp(min=eps)
    w_max = weight.abs().amax(dim=0).clamp(min=eps)
    s = (a.pow(alpha) / w_max.pow(1.0 - alpha)).clamp(min=eps)
    return s


@torch.no_grad()
def apply_smoothing(
    a: torch.Tensor,
    b: torch.Tensor,
    smooth_scales: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply ``s`` along the last (``D``) axis of both operands.

    Args:
        a: ``(..., D)`` activation.
        b: ``(..., D)`` weight.
        smooth_scales: ``(D,)`` smoothing vector from
            :func:`compute_smooth_scales`.

    Returns:
        ``(a / s, b * s)`` — mathematically equivalent to ``(a, b)`` for the
        matmul ``a @ b.T``, since the diagonal factors cancel along the
        contracted dimension.
    """
    if smooth_scales.dim() != 1:
        raise ValueError(
            f"smooth_scales must be 1D (D,), got shape "
            f"{tuple(smooth_scales.shape)}")
    if a.shape[-1] != smooth_scales.shape[0] or b.shape[-1] != smooth_scales.shape[0]:
        raise ValueError(
            f"D mismatch: a.shape[-1]={a.shape[-1]}, b.shape[-1]={b.shape[-1]}, "
            f"smooth_scales.shape[0]={smooth_scales.shape[0]}")

    sa = smooth_scales.to(device=a.device, dtype=a.dtype)
    sb = smooth_scales.to(device=b.device, dtype=b.dtype)
    return a / sa, b * sb


@torch.no_grad()
def apply_smoothing_offline(
    weight: torch.Tensor,
    smooth_scales: torch.Tensor,
) -> torch.Tensor:
    """Bake ``s`` into a weight tensor; returns a new ``weight * s``.

    Use this once at load time when you want every subsequent call to only
    pay the activation-side ``a / s`` divide.
    """
    if smooth_scales.dim() != 1:
        raise ValueError(
            f"smooth_scales must be 1D (D,), got shape "
            f"{tuple(smooth_scales.shape)}")
    if weight.shape[-1] != smooth_scales.shape[0]:
        raise ValueError(
            f"D mismatch: weight.shape[-1]={weight.shape[-1]} vs "
            f"smooth_scales.shape[0]={smooth_scales.shape[0]}")
    return weight * smooth_scales.to(device=weight.device, dtype=weight.dtype)


__all__ = [
    "accumulate_act_scales",
    "compute_smooth_scales",
    "apply_smoothing",
    "apply_smoothing_offline",
]
