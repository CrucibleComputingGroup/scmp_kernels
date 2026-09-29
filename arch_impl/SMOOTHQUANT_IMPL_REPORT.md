# SmoothQuant integration — arch-implement report

## What was built

- [scmp_kernels/quant/smoothquant.py](../scmp_kernels/quant/smoothquant.py) —
  new pure-PyTorch module implementing the SmoothQuant pre-quantization
  transform:
  - `accumulate_act_scales(x, running=None) -> (D,)` — per-channel max-abs
    running aggregator for the calibration pass.
  - `compute_smooth_scales(act_scales, weight, alpha=0.5, eps=1e-5) -> (D,)`
    — builds `s_j = act_max[j]^α / weight_max[j]^(1-α)`.
  - `apply_smoothing(a, b, smooth_scales) -> (a/s, b*s)` — diagonal rescale
    along the contracted dim (works for 2D and 3D).
  - `apply_smoothing_offline(weight, smooth_scales) -> weight*s` — bake `s`
    into the weight once.
- [scmp_kernels/quant/__init__.py](../scmp_kernels/quant/__init__.py) — exports
  the four new names alongside the existing fused / grouped quant API.
- [scmp_kernels/sc/matmul.py](../scmp_kernels/sc/matmul.py) — new
  `smooth_scales: Optional[torch.Tensor] = None` kwarg on `sc_matmul`. When
  passed, the matmul is rewritten as `(a/s) @ (b*s).T` before dispatch. Default
  `None` preserves byte-for-byte legacy behavior.

## How it relates to the upstream SmoothQuant key idea

Upstream SmoothQuant (Xiao et al., ICML 2023) targets W8A8 quantization of
LLM linear layers and migrates per-channel activation outliers into the
weights via the mathematically equivalent rewrite

    Y = X·W = (X·diag(s)^-1) · (diag(s)·W),   s_j = max(|X_j|)^α / max(|W_j|)^(1-α).

The diagonal cancels in fp32, but per-row activation absmax quant of the
*smoothed* X now has a much smaller dynamic range. Upstream wires the
transform into LayerNorm gain + Linear weight; here it lives one level lower,
at the matmul boundary, because `scmp_kernels` is consumed by both
matmul-shaped attention (Q@Kᵀ) and weight-shaped MLPs. Wiring as an
`sc_matmul` kwarg keeps the surface tiny and lets callers reuse the same
calibration vector for every shape.

## Test results

- Command: `python arch_impl/test_smoothquant.py`
- Verdict: **PASS (7 pass, 1 skip)**
- Cases covered:
  - Golden: calibration aggregator matches `torch.maximum(...).amax(0)` across
    2 batches.
  - Golden: `compute_smooth_scales` matches the closed form at α=0 and α=1.
  - Property: `(a/s) @ (b*s).T` equals `a @ b.T` to <1e-10 in fp64 (2D and 3D).
  - Property: offline-baked `weight*s` round-trips through `(a/s) @ ...`.
  - Headline claim: with one outlier channel (60× tail), per-row int8 quant
    MSE drops from **8.94e-01 → 5.43e-02 (16.5× improvement)**.
  - Edge: invalid α, mismatched D, non-1D `smooth_scales` all raise
    `ValueError`.
- Skipped:
  - `test_sc_matmul_kwarg_wiring_cuda` — requires CUDA (Triton kernels are
    GPU-only). The kwarg is wired and the import path resolves; full
    end-to-end verification needs a GPU node.

## Known limitations

- `smooth_scales` must be shape `(D,)` — no `(BH, D)` per-head smoothing yet.
  Add a 2D branch in `apply_smoothing` if needed for per-head Q/K outliers.
- No automatic α sweep / search — caller picks α explicitly (upstream
  recommends 0.5 for OPT, 0.85 for Llama-2/3).
- Smoothing is applied *outside* the Triton kernel as a separate launch
  (one `a / s`, one `b * s`). For very small shapes this is measurable; for
  typical LLM-block shapes it's lost in the kernel runtime.
- Calibration is offline and the user's responsibility — no model-walker like
  upstream's `smooth_lm` / `quantize_*` since `scmp_kernels` is model-agnostic.

## How to re-run

    cd /home/allenjin/Projects/SCMP/scmp_kernels
    conda activate annstention
    python arch_impl/test_smoothquant.py

The original `scmp_quantwm` env was deleted 2026-07-25 to clear the /home quota
(80G cap; that env alone was 15G and had not been touched since April). Its
recipe is archived at `/nfs/turbo/coe-nbleier/allenjin/env_backups/` as
`scmp_quantwm.yml` + `scmp_quantwm.pip.txt` if an exact rebuild is ever needed:

    conda env create -f /nfs/turbo/coe-nbleier/allenjin/env_backups/scmp_quantwm.yml
