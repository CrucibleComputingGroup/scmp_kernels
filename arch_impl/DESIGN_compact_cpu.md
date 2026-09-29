# DESIGN: bring compact (table-free) enable-signal SC path into scmp_kernels (CPU)

## Source

In `scmp_llm_old/SC/sc_enable.py`, the pure-Python enable-signal SC matmul exposes
two equivalent methods:

- `k_shortcut` — table-based: builds `cum_indicator(D, stoc_len+1, V)` and looks up
  AND-count per dim in O(1). Fast on CPU/GPU but ~D·(stoc_len+1)·V·4B memory.
- `cycle_by_cycle` — compact: simulates the AND+enable bitstream tick-by-tick,
  no cum_indicator table. ~D·stoc_len·8B memory, slower per call but no table.

`scmp_kernels` only carried over the Triton GPU kernels (`kernels.py`); there is
no pure-Python CPU path. GPU is occupied, so we port both methods to CPU for
correctness checking.

## Interface

New module `scmp_kernels/sc/sc_enable.py` — port of the old file, imports rewritten
to `.sng` / `.config_helpers`. Public surface:

```
sc_matmul_enable(
    a, b,
    max_fp_a, min_fp_a, max_fp_b=None, min_fp_b=None,
    mode="bipolar"|"unipolar",
    sc_prec=8,
    config=None,
    method="k_shortcut"|"cycle_by_cycle",
) -> Tensor (N,M) or (B,N,M)
```

`sc_matmul` (the unified dispatcher in `matmul.py`) gains a `method` kwarg:

- `method="auto"` (default) — unchanged Triton path. Existing behaviour preserved.
- `method="table"`   — pure-Python `k_shortcut`.    CPU.
- `method="compact"` — pure-Python `cycle_by_cycle`. CPU. (the new one)

When `method` is `"table"` or `"compact"`, dispatch goes through
`sc_matmul_enable` regardless of device. Supported scope: `granularity="per_tensor"`
in 2D/3D, both bipolar and unipolar. Other granularities raise `ValueError` when
combined with non-auto `method` (the old CPU path only ever supported per-tensor).

## Algorithm (port, not redesign)

1. Quantize FP→int with per-tensor scales (bipolar: symmetric; unipolar:
   asymmetric with zero-point).
2. Build per-dim Sobol RNG sequences via `RNGPool` + `SNGBank` (already in
   `scmp_kernels.sc`).
3. Compute AND counts with the chosen method:
   - **k_shortcut**: build `cum_indicator(D, stoc_len+1, V)` and `k_table(D, V)`
     via `cumsum`; gather `counts = cum_indicator[d, k_a[n,d], boundary_b[m,d]]`.
   - **cycle_by_cycle**: loop t in [0, stoc_len); at each tick advance B's
     enable index only where A's bit is 1; accumulate AND-bits.
4. Decode `counts -> raw = counts * q_max^2 / stoc_len`, apply signs (bipolar)
   or zero-point correction (unipolar), scale, sum over D.

## Test strategy

Pure-CPU, no CUDA needed. `tests/test_sc_enable_cpu.py`:

- `cycle_by_cycle == k_shortcut` (within 1e-3) on small N,D,M, both modes —
  proves the two paths are mathematically equivalent.
- Both methods are close to ground truth `a @ b.T` (normalized RMSE < 0.1) in
  bipolar and unipolar — proves the SC machinery actually approximates matmul.
- `sc_matmul(..., method="compact")` and `sc_matmul(..., method="table")` work
  on CPU and produce the same numbers as direct `sc_matmul_enable` calls.
- `sc_matmul(method="compact", granularity="per_row")` raises ValueError
  (scope gate documented in the design).

Reference oracle: `_enable_mul_cycle_by_cycle` IS the spec for enable-signal
SC. The `k_shortcut` path is checked against it.
