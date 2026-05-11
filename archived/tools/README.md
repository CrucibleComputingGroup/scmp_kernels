# archived/tools

Standalone utilities that were in `scmp_llm/SC/` but had zero in-repo callers at migration time. **Not on the runtime path.**

## Contents

- `dse.py` — generic Design Space Exploration helpers for SC operations, working over the RNG/SNG architecture. Loads tap configurations from `lfsr_taps_data.json` (run `python lfsr_taps.py <bits>` to regenerate). Useful starting point for sweeping configurations against a custom op.
- `noise_model_calibration.py` — calibrates a noise-injection surrogate against the real SC matmul to decide whether per-element error scales with output magnitude or is a flat floor.

## Imports

Same caveat as `archived/origin_cpu/`: bare imports from the original `SC/` layout. Won't run as-is inside the `scmp_kernels` package — fix imports or set `PYTHONPATH` before resurrecting.
