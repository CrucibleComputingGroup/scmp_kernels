# SmoothQuant integration in `scmp_kernels/quant/`

## Key idea (recap from upstream `smoothquant`)
Migrate per-channel activation outliers into the weight via a mathematically
equivalent diagonal rescaling along the shared inner dim `D`:

    Y = A @ B.T = (A * s.reciprocal()) @ (B * s).T,   s_j > 0

Pick s from calibrated statistics:

    s_j = act_max[j] ** alpha  /  weight_max[j] ** (1 - alpha)

This shrinks A's per-token absmax (activations easier to quantize) at the cost
of widening B's per-row absmax (weights — already easy — slightly harder).
The actual int-quant kernel is unchanged.

## Interface

New module: `scmp_kernels/quant/smoothquant.py` — pure PyTorch.

```python
accumulate_act_scales(x, running=None) -> (D,)             # calibration
compute_smooth_scales(act_scales, weight, alpha=0.5) -> (D,)
apply_smoothing(a, b, smooth_scales) -> (a/s, b*s)
apply_smoothing_offline(weight, smooth_scales) -> weight*s  # bake into weight
```

Integration with the SC matmul:
- `sc_matmul(a, b, ..., smooth_scales=s)` — new optional kwarg, applies
  `apply_smoothing` to `(a, b)` before dispatching to the existing kernels.
  `None` (default) → no change in behavior.

## Constraints
- Pure PyTorch math; no new Triton kernels.
- `smooth_scales` shape: `(D,)`. Works for both 2D `(N, D)/(M, D)` and 3D
  `(BH, N, D)/(BH, M, D)` inputs — broadcasts along leading dims.
- Default `None` preserves byte-for-byte legacy behavior.

## Test strategy (CPU-friendly; CUDA test is gated)
1. **Math identity (no quant)** — `(a/s) @ (b*s).T ≈ a @ b.T` to fp32 tolerance.
2. **Calibration aggregator** — running per-channel max-abs is correct across
   2 batches of synthetic activations.
3. **Outlier MSE improvement** — construct activation with one outlier channel
   (≈100× others); simulate per-row absmax int8 quant in PyTorch; assert
   smoothed-and-quantized MSE < plain-quantized MSE by a wide margin.
4. **Scale formula** — sanity-check `s` shape, positivity, finite.
5. **`sc_matmul` kwarg wiring** — assert `sc_matmul(a, b, smooth_scales=s)`
   equals `sc_matmul(a/s, b*s)` with a fixed seed. Skipped if `cuda` absent.
