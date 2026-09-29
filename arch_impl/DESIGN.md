# arch-implement DESIGN

## Goal
Split FP→int **quantization** out of the SC matmul kernels into a new
`scmp_kernels/quant/` sub-package, so quant strategies can be explored
independently of the SC kernels. Public API and speed must be unchanged.

## Constraints
- **Speed unchanged**: keep the existing fused Triton quant kernels intact;
  no new launches, no new copies.
- **External API unchanged**: `scmp_kernels.sc_matmul` keeps the same signature
  and behavior. `scmp_kernels.sc.kernels.fused_quantize_bipolar_perrow` etc.
  must still be importable (used by `scmp_diffusion/optimization_workspace/*`).

## Interface
`scmp_kernels.quant` exports the same names that currently live in
`scmp_kernels.sc.kernels`:

- Triton kernels: `fused_quant_kernel`, `fused_quant_bipolar_batched_kernel`
- Fused host wrappers: `fused_quantize_bipolar`, `fused_quantize_bipolar_perrow`,
  `fused_quantize_unipolar`, `_quant_dummy`
- Grouped (PyTorch) variants: `_grouped_symmetric_quant`,
  `_grouped_asymmetric_quant`, `_grouped_symmetric_quant_batched`

`scmp_kernels.sc.kernels` re-exports them under the same names for backward
compatibility — existing callers in `scmp_diffusion` and `scmp_llm` keep
working without edits.

## Layout
```
scmp_kernels/quant/
  __init__.py     — re-exports all symbols
  fused.py        — Triton fused_quant_kernel, fused_quant_bipolar_batched_kernel,
                    fused_quantize_{bipolar,bipolar_perrow,unipolar}, _quant_dummy
  grouped.py      — _grouped_{symmetric,asymmetric}_quant, _grouped_symmetric_quant_batched
```

`_resolve_rng_levels` stays in `sc/kernels.py` (it's about the SC RNG grid, not
quant per se) and is imported by `quant/fused.py`.

## Algorithm
Strictly mechanical move — cut + paste. No logic changes. Imports rewritten:
- `quant/fused.py` imports `_resolve_rng_levels` from `..sc.kernels`
- `sc/kernels.py` imports all moved symbols from `..quant` and re-exports

## Test strategy
1. **Golden vectors**: capture `sc_matmul` output before refactor for a fixed
   seed × {granularity, mode, sc_prec, stoc_len} grid. After refactor, re-run
   and assert `torch.equal` (bit-exact).
2. **Import surface**: confirm `from scmp_kernels.sc.kernels import
   fused_quantize_bipolar_perrow, fused_quant_kernel,
   _grouped_symmetric_quant_batched, ...` still works.
3. **New surface**: confirm `from scmp_kernels.quant import ...` works.
4. **Speed sanity**: warmup + 5-iter timing of `sc_matmul`, before vs after,
   one representative shape. Expect ±5% noise; flag >10% regression.
