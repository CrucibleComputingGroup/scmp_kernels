# arch-implement report: compact (table-free) enable-signal SC path (CPU)

## What was built

- `scmp_kernels/sc/sc_enable.py` (new, ~290 lines) — pure-Python CPU port of
  `scmp_llm_old/SC/sc_enable.py`. Public: `sc_matmul_enable(...)`. Two
  methods exposed via `method=`:
  - `"k_shortcut"` — table-based (precomputes `cum_indicator(D, stoc_len+1, V)`).
  - `"cycle_by_cycle"` — compact (no tables; bitstream simulation).
- `scmp_kernels/sc/matmul.py:118-126,167-197` — added `method` kwarg to the
  high-level `sc_matmul(...)`:
  - `"auto"` (default) → unchanged Triton GPU path.
  - `"table"` → routes to `sc_matmul_enable(method="k_shortcut")`.
  - `"compact"` → routes to `sc_matmul_enable(method="cycle_by_cycle")`.
  Scope-gated to `granularity="per_tensor"` + `chunk_d == 0` (raises otherwise).
- `tests/test_sc_enable_cpu.py` (new, 17 tests) — CPU-only correctness.

## Test results

- Test command: `python -m pytest tests/test_sc_enable_cpu.py -v`
- Full suite:   `python -m pytest tests/ -v`
- Verdict: **PASS** — 17/17 new tests pass; full suite 26 passed, 14 pre-existing
  CUDA tests skipped (no GPU available, as expected).

### Cases covered

- **Golden / equivalence** (4 tests): `cycle_by_cycle` and `k_shortcut` produce
  **bit-exact** outputs (`torch.equal`) in bipolar + unipolar, 2D + 3D.
- **Property — matmul approximation** (2 tests): normalized RMSE vs `a @ b.T`
  < 0.1 for both bipolar and unipolar at `sc_prec=8`.
- **High-level dispatch** (5 tests): `sc_matmul(..., method=...)` matches
  direct `sc_matmul_enable` calls; compact and table agree at the high level.
- **Scope gates** (6 tests): non-`per_tensor` granularity, `chunk_d > 0`,
  and unknown `method` all raise `ValueError` with the expected message.

### Cases NOT covered (and why)

- Triton GPU paths — no GPU available this session (per user instruction).
  The `method="auto"` default leaves Triton dispatch untouched; the existing
  CUDA-gated `tests/test_sc_smoke.py` will exercise it on next GPU run.

## Known limitations

- CPU paths only support `granularity="per_tensor"`. Per-row / per-head would
  require porting `_sc_matmul_per_row*` etc. as Python, which is outside the
  port scope. The flag explicitly gates this so callers get a clear error.
- `cycle_by_cycle` is O(N·M·D·stoc_len) memory in the 3D AND-product
  intermediate — fine for the small N/M used in tests but not for production
  matmul sizes. That's expected: it exists as the reference oracle.
- No new `TODO(arch-implement)` markers.

## How to re-run

```bash
cd /home/allenjin/Projects/SCMP/scmp_kernels
/home/allenjin/.conda/envs/vit_sc/bin/python -m pytest tests/test_sc_enable_cpu.py -v
```
