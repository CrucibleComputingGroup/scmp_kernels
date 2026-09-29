# Presentation summary — does Fig 2 of Kim et al. (DAC '16) hold for `scmp_kernels`?

**Settings** — `granularity="per_tensor"`, `mode="bipolar"`, `sc_prec=8`, `halve_bipolar_stoc_len=True` (production setting in `scmp_llm`/`vit_sc`). Bipolar magnitude grid `q_max = 127`; stream and RNG grid `max_rng_val = stoc_len = 128`. The iid-XNOR baseline is matched to the same `stoc_len = 128`.

## Headline figures

- [arch_impl/figures/fig2_presentation_compare.png](figures/fig2_presentation_compare.png) — side-by-side: paper's iid-Bernoulli XNOR (left) vs `scmp_kernels` enable-signal + Sobol QMC (right). Same axes, same colormap; note that the right z-axis maxes around 1.3×10⁻² while the left tops out at 1.4×10⁻¹.
- [arch_impl/figures/fig2_presentation_scmp.png](figures/fig2_presentation_scmp.png) — paper-Fig-2-style composite for `scmp_kernels`: (a) the same error surface and (b) the weight histogram overlaid with per-weight |error| (analog of paper's Fig 2(b)).

## One-paragraph summary (drop-in for slides / write-up)

> Following Kim et al. (DAC '16, Fig 2), we measured the absolute error of a single bipolar SC multiplier across `(X, Y) ∈ [-1, 1]²` at our production settings (`sc_prec=8`, `halve_bipolar_stoc_len=True`, `stoc_len=128`). The paper's correlation-free iid-Bernoulli XNOR multiplier reproduces the canonical *tent* pattern (left): error peaks along `X = 0` / `Y = 0` (Bernoulli variance `(1 − x²y²)/N` is maximal there) and vanishes at the corners, with mean |error| in the low-`|xy|` band 1.80× the mean in the high-`|xy|` band. `scmp_kernels` (right) behaves fundamentally differently: (i) the absolute error magnitude is ~24× smaller at the same stream length (`~3×10⁻³` vs `~7×10⁻²`); (ii) the error is **systematic, not stochastic** — Sobol-driven enable-signal streams are deterministic, so every `(X, Y)` produces a repeatable, structured residual rather than Gaussian noise around the truth; and (iii) the locus is **inverted** relative to the paper — the axes are exactly zero (when one operand has `mag = 0`, the enable signal never fires and the output is identically zero), and the remaining residual lies in the body of the surface, giving a low-`|xy|` / high-`|xy|` ratio of 0.70×. The accumulator consequence (paper's Fig 2(b)) does not carry over either: with weights `~ N(0, 0.1²)` and a fixed `x = 0.5`, the per-weight |error| stays in the `6.5–7.6 × 10⁻³` band across `|w| ∈ [0, 0.4]` rather than peaking near zero. The paper-derived design rules — pruning near-zero weights, weight-scaling, fused activation — were aimed at a stochastic failure mode that the Sobol enable-signal scheme appears to have already designed away.

## Direct answer to the audience's likely question

> **"So is the error random or systematic?"** — Systematic. The Sobol sequences are fixed; the enable-signal FSM is deterministic. Same `(X, Y)` → same output → same error, every run. What looks "noisy" in the right surface is structured QMC-discrepancy residual, not stochastic fluctuation. This is a qualitative shift from the paper's analysis, which assumed correlation-free iid-Bernoulli streams.
