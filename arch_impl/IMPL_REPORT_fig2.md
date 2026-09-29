# arch-implement report — Fig 2 property check on scmp_kernels

**Date:** 2026-05-27
**Hardware:** gl1802 (NVIDIA RTX PRO 6000 Blackwell, CUDA 12.8, torch 2.10.0, conda env `annstention`)
**Paper:** Kim et al., DAC '16 — *Dynamic Energy-Accuracy Trade-off Using Stochastic Computing in DNNs*
**Kernel settings:** `granularity="per_tensor"`, `mode="bipolar"`, `sc_prec=8`, `halve_bipolar_stoc_len=True`. With halve on, `stoc_len = 2^(sc_prec-1) = 128`. This is the production setting used in `scmp_llm` / `vit_sc`. The iid-XNOR baseline uses the same `stoc_len=128` for an apples-to-apples comparison.

## Verdict (one line)

**The property described in Figure 2 does NOT hold for `scmp_kernels`.** The paper's "random error of XNOR-based bipolar SC peaks near `(X=0, Y=0)`" pattern is **inverted** in the new kernel: low-|xy|/high-|xy| error ratio is **0.70×** vs the paper-reproducing iid-XNOR baseline at **1.80×** at the same stream length. The absolute error magnitude is also ~24× lower than the iid baseline.

## What was built

- [arch_impl/DESIGN_fig2.md](DESIGN_fig2.md) — Phase 1 sketch.
- [arch_impl/fig2_reproduce.py](fig2_reproduce.py) — three experiments, GPU.
- [arch_impl/figures/fig2a_baseline_iid_surface.png](figures/fig2a_baseline_iid_surface.png) &
  [arch_impl/figures/fig2a_baseline_iid_heatmap.png](figures/fig2a_baseline_iid_heatmap.png) — paper's setup (iid Bernoulli XNOR, 1024-bit stream, 8-seed mean |error|).
- [arch_impl/figures/fig2a_surface.png](figures/fig2a_surface.png) &
  [arch_impl/figures/fig2a_heatmap.png](figures/fig2a_heatmap.png) — `scmp_kernels` error surface under the same conditions.
- [arch_impl/figures/fig2b_near_zero.png](figures/fig2b_near_zero.png) — per-weight |error| for w ~ N(0, 0.1²) at fixed input x = +0.5.
- [arch_impl/logs/fig2_run.log](logs/fig2_run.log) — raw stdout.

## Test results

| Experiment | What it measures | Result | Verdict |
|---|---|---|---|
| **E0** — iid Bernoulli XNOR multiplier (paper's setup, our own implementation, `stoc_len=128`) | tent property: low-\|xy\| band vs high-\|xy\| band | mean(\|xy\|<0.10) = 7.12e-2, mean(\|xy\|>0.70) = 3.95e-2, ratio = **1.80×** | **PASS** — paper's property reproduces. Max error is at (+0.05, −0.55) — exactly on the X-axis where the tent peaks. |
| **E1** — `sc_matmul(granularity="per_tensor", mode="bipolar", sc_prec=8, halve_bipolar_stoc_len=True)` → `stoc_len=128`; grid on the quantisation lattice (`k / q_max`, `q_max=127`) | tent property under scmp_kernels | mean(\|xy\|<0.10) = 2.48e-3, mean(\|xy\|>0.70) = 3.51e-3, ratio = **0.70×** | **FAIL** — pattern is inverted; error is slightly higher near the corners than near the axes. Absolute magnitude ~24× smaller than E0. |
| **E2** — per-weight \|error\| for w ~ N(0, 0.1²), x = +0.5 fixed, 20 000 weights, 8-seed mean, same kernel settings as E1 | does the error track the histogram-of-weights peak (paper's Fig 2(b))? | mean(\|w\|<0.05) = 6.50e-3, mean(\|w\|>0.30) = 6.71e-3 — band means flat across \|w\| ∈ [0, 0.4] (range 6.5e-3 to 7.6e-3) | **FAIL** — error is essentially flat in \|w\|; no peaking near zero. |

### Test command (re-runnable)

```bash
ssh gl1802 'bash -lc "source ~/.bashrc && conda activate annstention && \
  cd /home/allenjin/Projects/SCMP/scmp_kernels && \
  python arch_impl/fig2_reproduce.py 2>&1 | tee arch_impl/logs/fig2_run.log"'
```

## Why the property does not hold — mechanism

The paper analyses a **correlation-free XNOR multiplier driven by iid Bernoulli streams**. For bit-streams with `P(a=1) = (1+x)/2`, `P(b=1) = (1+y)/2`, the XNOR output has bipolar mean `xy` and per-bit variance `(1 − x²y²)`. Averaging over `N` cycles gives variance `(1 − x²y²)/N`, which peaks along the axes (`xy → 0`) and vanishes at the corners (`|xy| → 1`). That's the tent. (`N=128` here; paper used `N=1024` — same shape, different magnitude.)

`scmp_kernels` uses a **different SC scheme**:

1. **Enable-signal (UnarySim FSUMul-style) multiplier**, not free-running XNOR. The B-stream RNG only advances on A-bits that are 1, so the count of ones in the output stream is deterministically bounded by `popcount(A)`. When `mag(A) = 0` the enable signal never fires and the output is **exactly 0** — there is no random fluctuation at all on the X = 0 (or Y = 0) line. This alone breaks Fig 2(a)'s tent.
2. **Sobol-scrambled SNGs** (low-discrepancy quasi-Monte-Carlo) instead of iid Bernoulli. The error structure is bounded by the QMC discrepancy and is largely deterministic + ~25× lower in magnitude than iid at the same stream length. The residual is dominated by Sobol-sequence-specific structure, not by `(1 − x²y²)/N`.

Project memory matches: `scmp_kernels` runs Sobol with Owen scramble OFF (`SC_DISABLE_OWEN=1` in practice) and uses q+k-independent Sobol axes; the CPU/Triton kernels match cycle-by-cycle. Both confirm the enable-signal + QMC picture.

The Fig 2(b) "error proportional to weight histogram" follows directly from Fig 2(a) — it's the accumulator integrating `(1 − w²x²)/N` per weight, scaled by the count at each weight value. Once Fig 2(a)'s tent goes away, Fig 2(b)'s peak-at-zero pattern goes too. E2 confirms this: band means stay within a 1.5× envelope across the full `|w| ∈ [0, 0.4]` range.

## Practical implication for the kernel user

The paper's design choices — pruning near-zero weights, weight-scaling, and integrating the activation function into the accumulator — were all motivated by Fig 2's high-near-zero error. **None of these motivations apply to `scmp_kernels`** as-is:

- Per-product error is ~24× smaller than iid-XNOR at the same stream length (`stoc_len=128`).
- The error is roughly uniform in `(x, y)` and in `|w|` (slightly *lower* near the axes).
- Near-zero weights are NOT a privileged failure mode in this kernel.

This is also worth checking before lifting paper-derived guidance into the SC training pipeline; the Sobol enable-signal path with halved cycles may already be doing the job that "remove near-zero weights" was meant to do.

## Caveats

- E1 isolates the stochastic component by placing the (X, Y) grid exactly on the quantisation lattice (`k / q_max`, `q_max=127` for `sc_prec=8`). Without this, the rounding-to-lattice residual at the corners would mask the stochastic shape.
- E2 deviates from the paper's `x = 0` setup. With `x = 0` the enable-signal kernel returns exactly 0, so all-zero errors trivially conceal the per-weight pattern. Using `x = +0.5` makes the enable signal fire and surfaces the actual per-weight error.
- The iid-XNOR baseline (E0) is matched to `stoc_len=128` so the comparison is apples-to-apples. At the paper's original 1024-bit stream the iid SD would be ~3× smaller but the tent's *shape* is unchanged — the property is qualitative, not magnitude-dependent.
- E0 averages 8 seeds; a single shot is too noisy to see the tent.
- Result holds for the auto Triton path. The CPU `method="table"` / `method="compact"` paths share the same enable-signal + Sobol math and would behave identically.

## How to re-run

```bash
# on gl-login3
ssh gl1802 'bash -lc "source ~/.bashrc && conda activate annstention && \
  cd /home/allenjin/Projects/SCMP/scmp_kernels && \
  python arch_impl/fig2_reproduce.py 2>&1 | tee arch_impl/logs/fig2_run.log"'
```

No `TODO(arch-implement)` markers left in the source.
