# arch-implement report

## What was built

New sub-package `scmp_kernels/quant/` that owns FP → int quantization:

- [scmp_kernels/quant/__init__.py](../scmp_kernels/quant/__init__.py) — re-exports
- [scmp_kernels/quant/fused.py](../scmp_kernels/quant/fused.py) — Triton fused
  kernels `fused_quant_kernel`, `fused_quant_bipolar_batched_kernel` and host
  wrappers `fused_quantize_bipolar` / `_bipolar_perrow` / `_unipolar`,
  plus `_quant_dummy`
- [scmp_kernels/quant/grouped.py](../scmp_kernels/quant/grouped.py) — pure-PyTorch
  `_grouped_symmetric_quant`, `_grouped_asymmetric_quant`,
  `_grouped_symmetric_quant_batched`

`scmp_kernels/sc/kernels.py` keeps the SC computation paths intact and now
imports the quant symbols from `scmp_kernels.quant` and re-exports them under
their original names, so existing callers (`scmp_diffusion/optimization_workspace/`,
`scmp_llm/calibrate_mp_thresholds.py`, etc.) keep working unchanged.

## Test results

- **Test command (GPU, on `gl1810`):**
  ```
  PYTHONPATH=/home/allenjin/Projects/SCMP/scmp_kernels PHASE=ref    python arch_impl/capture_reference.py   # before refactor
  # (apply refactor)
  PYTHONPATH=/home/allenjin/Projects/SCMP/scmp_kernels PHASE=verify python arch_impl/capture_reference.py   # after refactor
  PYTHONPATH=/home/allenjin/Projects/SCMP/scmp_kernels python -m pytest /home/allenjin/Projects/SCMP/scmp_llm/kernels/tests/test_sc_smoke.py -v
  ```

- **Verdict: PASS**

- **Cases covered (bit-exact, all PASS):**
  - golden: `per_tensor` bipolar 2D
  - golden: `per_tensor` unipolar 2D
  - golden: `per_row` bipolar 2D
  - golden: `per_row` unipolar 2D
  - golden: `per_row` bipolar 2D + `chunk_d=16`
  - golden: `per_row` bipolar 3D (batched)
  - golden: `per_head` bipolar 3D (batched-per-head)
  - low-prec: `per_row` bipolar `sc_prec=4 stoc_len=16`
  - import surface: new `scmp_kernels.quant.*` imports
  - import surface: legacy `scmp_kernels.sc.kernels.*` imports (must still work)
  - identity: legacy name and new name resolve to the *same* object
  - external import: simulates `scmp_diffusion/optimization_workspace/sc_patch.py`
    and `optimized_kernels.py` import lists
  - pytest smoke (`test_sc_smoke.py`): 13/13 PASS

- **Speed:** representative `sc_matmul` per_row 128×128×128 sc_prec=8 stoc_len=256
  → 0.413 ms/it after vs 0.406 ms/it before (+1.6%, within shared-GPU noise).
  No code paths in the matmul launch sequence changed; the fused Triton quant
  kernels were moved as-is.

- **Cases NOT covered (and why):**
  - Large-D MLP path (D≥1024) with `chunk_d > 0` for unipolar — `chunk_d` is
    bipolar-only by design (raises in `sc_matmul`).
  - End-to-end LLM correctness (e.g. Llama check_mse) — out of scope for a
    refactor; covered by the in-repo smoke + bit-exact reference grid above.
  - 4080-class compact-kernel forcing (`SC_FORCE_COMPACT=1`) — same code path,
    behavior unchanged by construction.

## Known limitations / TODOs

- No `TODO(arch-implement)` markers were left in code.
- `_resolve_rng_levels` stays in `sc/kernels.py` because it's about the SC RNG
  grid, not the quant grid. `quant/fused.py` reaches it via a lazy import to
  avoid a circular-import-at-load cost; this is the only structural wart.

## How to re-run

```bash
# On gl-login*, with an existing gpu-rtx6000 allocation reachable via ssh gl<NNNN>:
ssh gl1810 'source ~/.bashrc && conda activate annstention && \
  PYTHONPATH=/home/allenjin/Projects/SCMP/scmp_kernels \
  PHASE=verify python /home/allenjin/Projects/SCMP/scmp_kernels/arch_impl/capture_reference.py'

ssh gl1810 'source ~/.bashrc && conda activate annstention && \
  PYTHONPATH=/home/allenjin/Projects/SCMP/scmp_kernels \
  python -m pytest /home/allenjin/Projects/SCMP/scmp_llm/kernels/tests/test_sc_smoke.py -v'
```

Logs from this run are in `arch_impl/logs/`:
- `00_capture_ref.log` — pre-refactor reference values
- `02_verify.log` — bit-exact PASS post-refactor
- `01_import_smoke.log` — both new and legacy import paths work
- `03_pytest_smoke.log` — 13/13 in-repo smoke tests
- `04_external_imports.log` — sibling-repo import surface intact
