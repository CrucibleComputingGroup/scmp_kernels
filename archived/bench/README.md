# archived/bench

Benchmark, comparison, and ad-hoc test scripts that pre-date the move into `scmp_kernels`. **Not on the runtime path.** Kept as a starting point for regression benchmarks if/when needed.

## Contents

- `bench_table_vs_compact.py` — kernel-level timing: table-based vs compact (on-the-fly) enable-signal SC matmul, across attention shapes (D=72) and MLP shapes (D=1152). Has an `--e2e` mode that drives `quant_sc_main.py`.
- `compare_cbsg.py` — accuracy + speed comparison between UnarySim's C-BSG and the independent-RNG approach.
- `compare_enable.py` — accuracy comparison: enable-signal SC vs standard SC vs FP ground truth (RMSE / Pearson / max abs err). Imports the CPU `sc_matmul_enable` from `archived/origin_cpu/sc_enable.py`.
- `compare_matmul.py` — UnarySim vs Triton matmul on attention-sized operations.
- `compare_unarysim.py` — UnarySim's C-BSG vs independent generation, using UnarySim's `FSUMul` directly.
- `test_kernel_opt.py` — scratch tests for kernel optimizations (BLOCK_K tiling, fused quant, enable-table caching, etc.).

## Imports

Same caveat as `archived/origin_cpu/`: bare imports from the original `SC/` layout. Won't run as-is inside the `scmp_kernels` package — fix imports or set `PYTHONPATH` before resurrecting.
