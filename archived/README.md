# archived/

Code from the original `scmp_llm/SC/` tree that is intentionally **not** on the runtime path of `scmp_kernels`. Kept as historical reference and as a starting point if any of these need to be resurrected.

| Subfolder | What's inside |
|---|---|
| `origin_cpu/` | CPU/NumPy reference implementations of SC matmul (`sc.py`, `sc_enable.py`). Superseded by Triton kernels in `scmp_kernels/sc/sc_triton.py`. |
| `bench/` | Benchmark, comparison, and ad-hoc test scripts. |
| `tools/` | Standalone utilities (DSE harness, noise-model calibration). |

All files in here use **bare imports** (`from sng import ...`) inherited from the old layout. They will not run unmodified inside the `scmp_kernels` package — see each subfolder's README.
