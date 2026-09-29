# Fig 2 reproduction with scmp_kernels — design

## Property under test (Kim et al., DAC '16, Figure 2)

**Fig 2(a)** — absolute random error of a 1024-bit bipolar SC multiplier
(XNOR-based) is largest near `(X=0, Y=0)` and decreases as `|X|` or `|Y|`
approaches 1; the error surface is a "tent" centred at the origin.

**Fig 2(b)** — when synaptic weights are concentrated near zero (regularised
DNN) and inputs are also small, the sum-of-products error is dominated by
the near-zero-weight contributions: histogram of 20 000 N(0, σ²) weights vs.
the absolute error of `Σ wᵢ · 0` is bimodal — error peaks where weights are
near zero, drops off as `|w|` grows.

## Why this is non-trivial for `scmp_kernels`

The kernel uses Sobol-scrambled SNGs (quasi-Monte Carlo) instead of plain
LFSR/Bernoulli streams. The variance argument `|err| ∝ √((1-x²y²)/N)` only
holds for true iid Bernoulli; QMC streams have a structured residual.
Fig 2(a)'s "tent" depends only on the `(1-x²)(1-y²)`-style factor in the
ideal-bit-variance, so the shape *should* survive — we want to verify.

User-memory cross-checks:
- `stoc_len ≤ 2**sc_prec` (else wrap). Production setting in `scmp_llm`/`vit_sc`
  is `sc_prec=8` with `halve_bipolar_stoc_len=True`, which sets `stoc_len = 2^7 = 128`
  and keeps the bipolar magnitude grid at `q_max = 2^7 − 1 = 127`.
- `SC_DISABLE_OWEN=1` in practice → leave Owen scramble off (env var, GPU path).
- `scmp_kernels` does q+k-independent Sobol axes (≠ UnarySim) → leave default.

## Interface used

```python
sc_matmul(a, b,
          granularity="per_tensor",
          mode="bipolar",
          sc_prec=8,
          halve_bipolar_stoc_len=True,   # → stoc_len = 128
          method="auto")                  # Triton GPU path
```

Choosing per-tensor with `a.max=1, a.min=-1, b.max=1, b.min=-1` makes the
bipolar scale exactly `1/q_max` so `(X,Y)∈[-1,1]²` map cleanly onto the
quant grid. The iid-XNOR baseline (E0) uses the same `stoc_len=128` for an
apples-to-apples comparison; the paper used 1024 but the *shape* of the
tent is independent of stream length.

## Experiments

### E1 — Fig 2(a) error surface (single XNOR, D=1)

- Grid: X, Y ∈ linspace(-1, 1, 41).
- `a = X.reshape(41,1)`, `b = Y.reshape(41,1)`, shape `(41, 1)`.
- Compute `y_sc = sc_matmul(a, b, …)`; reference `y_ref = X·Yᵀ`.
- Error[i,j] = |y_sc[i,j] - y_ref[i,j]|.
- Plot 3D surface + 2D heatmap of Error vs (X, Y).
- **Pass criterion:** max(Error) is attained inside the `|X|<0.2 ∧ |Y|<0.2`
  centre square AND mean(Error) on the centre row/col is at least 2× the
  mean(Error) on the `|X|>0.9 ∨ |Y|>0.9` border.

### E2 — Fig 2(b) near-zero-weight sum-of-products

- 20 000 weights `w ~ N(0, σ²)` with σ chosen to match
  Fig 2(b) range ([-0.4, 0.4]) → σ ≈ 0.1.
- Inputs are constant 0 (per paper footnote: products taken with 0 input,
  to surface accumulator error).
- For each weight value bin, multiply by 0 (in SC) and record |output|.
- 200-bin histogram of weights + per-bin mean(|error|).
- **Pass criterion:** error peaks at the |w|≈0 bins and falls off monotonically
  with |w|.

### Test oracle

- E1: closed-form `X·Y`. Error is exact element-wise abs diff.
- E2: closed-form `w · 0 = 0`. Error is `|SC_output|`.

### Constraints / non-goals

- Single Sobol config (deterministic). Paper used one stream too. Multi-seed
  averaging is reported as a sanity check but not required for the verdict.
- We do **not** run a full DNN; the property is a per-multiplier statement.

## Files

```
arch_impl/fig2_reproduce.py    # main script
arch_impl/figures/fig2a_*.png  # surface, heatmap
arch_impl/figures/fig2b_*.png  # histogram-overlay
arch_impl/logs/fig2_run.log    # raw stdout
arch_impl/IMPL_REPORT_fig2.md  # verdict
```
