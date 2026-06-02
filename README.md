# scmp_kernels

Shared stochastic-computing kernels, FP→int quantization, and mixed-precision
dispatch — factored out of `scmp_llm` and `vit_sc` so the LLM, diffusion, and
speculative-decoding repos all build on one kernel package.

## Layout

```
scmp_kernels/
├── sc/             # Stochastic-computing matmul kernels (Triton)   ← migrated
├── quant/          # FP → int quantization for the SC kernels       ← migrated
├── mp/             # Mixed-precision config + row/group classifiers  ← migrated
├── qwt/            # QwT compensation                                (placeholder)
└── sensitivity/    # Per-(op, block) sensitivity tools               (placeholder)
```

## Install

```bash
pip install -e .          # needs a CUDA GPU + Triton for the SC kernels
```

## SC quickstart

```python
import torch
from scmp_kernels import sc_matmul     # or: from scmp_kernels.sc import sc_matmul

a = torch.randn(128, 1024, device="cuda")
b = torch.randn(512, 1024, device="cuda")

# Per-row quantization (most common — used by all linear/MLP paths)
y = sc_matmul(a, b, granularity="per_row", sc_prec=8)

# Per-tensor quantization
y = sc_matmul(a, b, granularity="per_tensor", sc_prec=8)

# Per-head batched (QK attention pattern); softmax·V may be asymmetric (N≠M)
q = torch.randn(16, 196, 64, device="cuda")   # (BH, N, D)
k = torch.randn(16, 196, 64, device="cuda")
y = sc_matmul(q, k, granularity="per_head", sc_prec=8)

# MLP fast path: per-row + chunk_d on wide D
y = sc_matmul(a, b, granularity="per_row", chunk_d=72, sc_prec=8)
```

### API

```
sc_matmul(a, b,
    granularity: "per_tensor" | "per_row" | "per_head" = "per_row",
    *,
    mode: "bipolar" | "unipolar" = "bipolar",
    sc_prec: int = 8,
    stoc_len: int | None = None,            # default 2 ** sc_prec
    chunk_d: int = 0,                        # per_row + bipolar only
    group_a: int = 1,                        # row-group size on operand a
    group_b: int = 1,                        # row-group size on operand b
    rng_levels: int | None = None,           # mixed-precision stream lengths
    config: dict | None = None,              # Sobol/Owen config; auto-built if None
    halve_bipolar_stoc_len: bool = False,    # bipolar: if stoc_len/rng_levels are None, default to 2 ** (sc_prec - 1)
    smooth_scales: torch.Tensor | None = None,  # SmoothQuant per-channel scales
) -> torch.Tensor
```

Computes `a @ b.T`, all-float32 in/out — quantization happens inside the Triton
kernels. `chunk_d > 0` requires `granularity="per_row"` and `mode="bipolar"`;
`per_head` requires 3D input and `mode="bipolar"`. Invalid combinations raise
`ValueError`.

Also exported from `scmp_kernels.sc`:

* `clear_rng_cache()` — drop cached RNG sequences (call after changing
  Sobol/Owen env vars or rotating seeds).
* `det_kernel_tuning()` — context manager opting into det-tuned tile sizes on
  the batched grouped path.

## Quantization (`scmp_kernels.quant`)

FP→int quantization split out of the SC matmul so quant strategies can evolve
independently. Produces the SC-domain integer representation the matmul kernels
consume (bipolar: `(boundary, sign, scale)`; unipolar: `(boundary, scale, zp[, row_sum])`).

* `.fused` — Triton-fused per-tensor / per-row quant (one launch):
  `fused_quantize_bipolar`, `fused_quantize_bipolar_perrow`,
  `fused_quantize_unipolar`.
* `.grouped` — pure-PyTorch row-group quant for the per-row matmul path:
  `_grouped_symmetric_quant`, `_grouped_asymmetric_quant`,
  `_grouped_symmetric_quant_batched`.
* `.smoothquant` — SmoothQuant pre-quantization transform:
  `accumulate_act_scales`, `compute_smooth_scales`, `apply_smoothing`,
  `apply_smoothing_offline`. Pass the resulting per-channel scales to
  `sc_matmul(..., smooth_scales=...)`.

## Mixed precision (`scmp_kernels.mp`)

Config objects + row/group classifiers shared by the application repos to drive
per-row / per-group `stoc_len` assignment: `MPConfig`, `AdaptiveMPConfig`,
`RangeMPConfig`, `RowAssignment`, `classify_rows_by_metric`,
`adaptive_classify_rows`, `classify_groups_by_range`, plus the
`MPDistributionLogger` / `MetricProfiler` instrumentation helpers.

## QwT / Sensitivity

Not yet migrated. The empty `qwt/` and `sensitivity/` packages reserve the
namespace.

## Tests

```bash
pytest tests/                 # test_sc_smoke.py, test_smoothquant.py
```
