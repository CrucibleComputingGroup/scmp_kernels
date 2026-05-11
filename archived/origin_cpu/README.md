# archived/origin_cpu

Frozen reference implementations of stochastic-computing primitives that pre-date the Triton GPU rewrite. **Not on the runtime path.** Kept for:

- Historical reference.
- Cross-checking the Triton kernels in `scmp_kernels/sc/sc_triton.py` produce numerically equivalent results.

## Contents

- `sc_enable.py` — original NumPy/PyTorch-CPU implementation of enable-signal (conditional BSG) SC matmul.
  - `sc_matmul_enable(...)` is the public entrypoint (cycle-by-cycle and k-shortcut paths).
  - Superseded in production by `sc_matmul_enable_triton`, `sc_matmul_enable_triton_mlp`, `sc_matmul_grouped_enable_triton`, `sc_matmul_enable_batched_bipolar` in `scmp_kernels.sc.sc_triton`.

## Imports

Files in this folder use bare imports (`from sng import RNGPool, SNGBank`) inherited from the original scmp_llm `SC/` layout. They will **not** run as-is inside the `scmp_kernels` package. To resurrect for a comparison run, either:

- Run with `PYTHONPATH` pointing at `scmp_kernels/scmp_kernels/sc/`, or
- Rewrite to `from scmp_kernels.sc.sng import RNGPool, SNGBank`.
