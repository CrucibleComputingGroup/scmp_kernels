#!/usr/bin/env python3
"""
Analytic 2D adaptive mixed-precision for SC with Gaussian importance model.

Token importance ~ N(μ, σ²).  Precision assignment via log2 interpolation:
  - z = (importance - μ) / σ   (z-score)
  - z ≤ 0        → L_MIN  (lowest precision)
  - 0 < z < 2    → log2-interpolate between L_MIN and L_MAX
  - z ≥ 2        → L_MAX  (highest precision)

The 2D modulation (timestep × layer) controls σ_eff:
  - Early t (noisy):  large σ_eff → few tokens reach high z → aggressive compression
  - Late  t (clean):  small σ_eff → more tokens reach high z → conservative
  - Late  layers amplify the effect (more sensitivity to importance)

Grid: T=100 timesteps × 28 layers.
"""

import argparse
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats

# ── Parameters ──────────────────────────────────────────────────────
T = 100          # total diffusion timesteps
N_LAYERS = 28    # DiT blocks
L_MIN = 16       # shortest bitstream
L_MAX = 256      # longest  bitstream

# Available discrete SC precision levels (must be powers of 2)
SC_LEVELS = np.array([16, 32, 64, 128, 256])

# ── Core: z-score → stoc_len mapping (log2 interpolation) ──────────

LOG2_MIN = np.log2(L_MIN)  # 4
LOG2_MAX = np.log2(L_MAX)  # 8
Z_LOW = 0.0                # z-score threshold: below → L_MIN
Z_HIGH = 2.0               # z-score threshold: above → L_MAX


def z_to_stoc_len(z):
    """Map z-score to stoc_len via log2-linear interpolation.

    z ≤ 0:  L_MIN
    0 < z < 2:  2^(log2(L_MIN) + (log2(L_MAX) - log2(L_MIN)) * z / 2)
    z ≥ 2:  L_MAX

    This gives exponentially-spaced precision: each +0.5σ doubles the precision.
    """
    z = np.asarray(z, dtype=float)
    t = np.clip((z - Z_LOW) / (Z_HIGH - Z_LOW), 0.0, 1.0)  # [0, 1]
    log2_len = LOG2_MIN + (LOG2_MAX - LOG2_MIN) * t
    return np.power(2.0, log2_len)


def z_to_discrete_stoc_len(z):
    """Map z-score to nearest available SC level (powers of 2)."""
    continuous = z_to_stoc_len(z)
    # Snap to nearest level in log2 space
    log2_cont = np.log2(continuous)
    log2_levels = np.log2(SC_LEVELS)
    # For each value, find nearest level
    idx = np.argmin(np.abs(log2_cont[..., None] - log2_levels), axis=-1)
    return SC_LEVELS[idx].astype(float)


# ── 2D modulation: (timestep, layer) → σ_eff ───────────────────────
# σ_eff controls how "spread out" the importance distribution is.
# Large σ_eff → most tokens have z < 0 → low precision (aggressive).
# Small σ_eff → more tokens have z > 0 → higher precision (conservative).
#
# Intuition: σ_eff is how many raw-σ of importance it takes to reach
# the "deserves high precision" region.

# We model σ_eff as a function of (p_t, p_l):
#   p_t ∈ [0,1]: timestep progress (1=noisy, 0=clean)
#   p_l ∈ [0,1]: layer progress (0=early, 1=late)

SIGMA_MIN = 0.6   # σ_eff at late t + late layer  (conservative, Q4)
SIGMA_MAX = 3.0   # σ_eff at early t + early layer (aggressive, Q1)


def sigma_eff(p_t, p_l):
    """Effective sigma: controls aggressiveness of precision allocation.

    Large σ_eff → aggressive (most tokens get low precision).
    Small σ_eff → conservative (more tokens get high precision).

    Design:
      Q1 (early t, early L): σ_eff ≈ SIGMA_MAX  → very aggressive
      Q2 (early t, late  L): σ_eff ≈ moderate    → outlier protection
      Q3 (late  t, early L): σ_eff ≈ moderate    → medium precision
      Q4 (late  t, late  L): σ_eff ≈ SIGMA_MIN   → conservative, bimodal
    """
    # Time: noisy → large σ, clean → small σ
    time_factor = p_t  # 0..1

    # Layer: late layers reduce σ (more sensitive to importance)
    layer_factor = 1.0 - 0.4 * p_l

    # Interaction: late t + late layer → extra reduction
    interaction = -0.3 * (1.0 - p_t) * p_l

    s = SIGMA_MIN + (SIGMA_MAX - SIGMA_MIN) * (time_factor * layer_factor + interaction)
    return np.clip(s, SIGMA_MIN, SIGMA_MAX)


def importance_to_z(importance_raw, sig_eff):
    """Convert raw importance (assumed ~ N(0,1)) to effective z-score.

    z_eff = importance_raw / σ_eff

    When σ_eff is large, even tokens with high raw importance get
    mapped to low z → low precision (aggressive).
    """
    return importance_raw / sig_eff


# ── Analytic expected stoc_len under Gaussian ──────────────────────

def expected_stoc_len_gaussian(sig_eff, n_samples=10000):
    """E[stoc_len] when importance ~ N(0,1) and σ_eff modulates the mapping.

    Computes the integral ∫ stoc_len(x / σ_eff) · φ(x) dx  numerically.
    """
    # Sample from standard normal
    x = np.linspace(-4, 4, n_samples)
    pdf = stats.norm.pdf(x)
    z = x / sig_eff
    sl = z_to_stoc_len(z)
    return np.trapezoid(sl * pdf, x)


def fraction_per_level_gaussian(sig_eff):
    """Fraction of tokens assigned to each SC level under N(0,1) importance."""
    # Boundaries in z-score space for each level transition
    log2_levels = np.log2(SC_LEVELS).astype(float)
    # Midpoints between adjacent levels in log2 space → z boundaries
    z_boundaries = []
    for i in range(len(log2_levels) - 1):
        mid_log2 = (log2_levels[i] + log2_levels[i + 1]) / 2.0
        t = (mid_log2 - LOG2_MIN) / (LOG2_MAX - LOG2_MIN)
        z_b = Z_LOW + t * (Z_HIGH - Z_LOW)
        z_boundaries.append(z_b)

    # Convert z boundaries to raw importance thresholds: raw = z * σ_eff
    fracs = []
    # Below lowest boundary → level 0 (L_MIN)
    raw_boundaries = [z * sig_eff for z in [Z_LOW] + z_boundaries + [Z_HIGH]]

    for i in range(len(SC_LEVELS)):
        lo = raw_boundaries[i]
        hi = raw_boundaries[i + 1]
        if i == 0:
            # Everything below the first boundary
            f = stats.norm.cdf(hi)
        elif i == len(SC_LEVELS) - 1:
            # Everything above the last boundary
            f = 1.0 - stats.norm.cdf(lo)
        else:
            f = stats.norm.cdf(hi) - stats.norm.cdf(lo)
        fracs.append(f)
    return np.array(fracs)


# ── Load profiled σ from CSV ────────────────────────────────────────

def load_profiled_sigma(csv_path, operator="input_proj"):
    """Load profiled metric stats and build σ_eff grid from real data.

    The CSV has columns: timestep, block, operator, N, mean, std, min, max, ...
    We use the 'std' column as a proxy for importance spread at each (t, block).

    Returns:
        sigma_grid: [N_LAYERS, T] array of σ_eff values
        (or None if loading fails)
    """
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["operator"] == operator:
                rows.append(row)

    if not rows:
        print(f"[WARN] No rows for operator={operator} in {csv_path}")
        return None

    # Build grid: σ_eff[layer, timestep_index]
    # Timesteps in CSV are raw diffusion timesteps (99..0)
    timestep_set = sorted(set(int(r["timestep"]) for r in rows), reverse=True)
    layer_set = sorted(set(int(r["block"]) for r in rows))

    t_map = {t: i for i, t in enumerate(timestep_set)}
    l_map = {l: i for i, l in enumerate(layer_set)}

    grid = np.full((len(layer_set), len(timestep_set)), np.nan)
    for r in rows:
        t_idx = t_map[int(r["timestep"])]
        l_idx = l_map[int(r["block"])]
        grid[l_idx, t_idx] = float(r["std"])

    # Normalize to [SIGMA_MIN, SIGMA_MAX] range
    valid = grid[~np.isnan(grid)]
    if len(valid) == 0:
        return None
    g_min, g_max = valid.min(), valid.max()
    # Invert: high raw std → low σ_eff (high std means more spread → tokens
    # naturally differentiate → need less artificial spreading)
    # Low raw std → high σ_eff (uniform → need aggressive compression)
    normalized = (grid - g_min) / (g_max - g_min + 1e-8)
    sigma_grid = SIGMA_MIN + (SIGMA_MAX - SIGMA_MIN) * (1.0 - normalized)

    print(f"[Profile] Loaded {len(rows)} rows from {csv_path} (op={operator})")
    print(f"[Profile] Raw std range: [{g_min:.4f}, {g_max:.4f}]")
    print(f"[Profile] σ_eff range: [{sigma_grid[~np.isnan(sigma_grid)].min():.2f}, "
          f"{sigma_grid[~np.isnan(sigma_grid)].max():.2f}]")

    return sigma_grid


# ── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analytic 2D MP visualization")
    parser.add_argument("--profile-csv", type=str, default=None,
                        help="Path to profile_metric_sigma.csv (from MetricProfiler). "
                             "If provided, uses profiled σ instead of analytic model.")
    parser.add_argument("--operator", type=str, default="input_proj",
                        help="Operator to visualize when using profiled data.")
    parser.add_argument("--output", type=str, default="analytic_mp_2d.png")
    args = parser.parse_args()

    # Build discrete grid
    timesteps = np.arange(T - 1, -1, -1)
    layers = np.arange(N_LAYERS)
    p_t_arr = timesteps / max(T - 1, 1)
    p_l_arr = layers / max(N_LAYERS - 1, 1)
    PT, PL = np.meshgrid(p_t_arr, p_l_arr)  # [28, 100]

    # Determine σ_eff source
    if args.profile_csv:
        profiled = load_profiled_sigma(args.profile_csv, operator=args.operator)
        if profiled is not None:
            SIGMA_EFF = profiled
            source_label = f"Profiled ({args.operator})"
        else:
            print("Falling back to analytic model.")
            SIGMA_EFF = sigma_eff(PT, PL)
            source_label = "Analytic (hand-tuned)"
    else:
        SIGMA_EFF = sigma_eff(PT, PL)
        source_label = "Analytic (hand-tuned)"

    # ════════════════════════════════════════════════════════════════
    # Plot
    # ════════════════════════════════════════════════════════════════

    fig = plt.figure(figsize=(22, 14))
    fig.suptitle(
        f"2D Adaptive SC Mixed Precision — Gaussian Model [{source_label}]\n"
        f"(T={T}, Layers={N_LAYERS}, levels={list(SC_LEVELS)}, "
        f"σ_eff ∈ [{SIGMA_EFF.min():.2f}, {SIGMA_EFF.max():.2f}])",
        fontsize=14, y=0.99,
    )

    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35)
    norm = Normalize(vmin=L_MIN, vmax=L_MAX)

    # ── (0,0): z-score → stoc_len mapping curve ───────────────────
    ax = fig.add_subplot(gs[0, 0])
    z_arr = np.linspace(-1, 3, 500)
    ax.plot(z_arr, z_to_stoc_len(z_arr), "b-", linewidth=2.5, label="continuous")
    ax.plot(z_arr, z_to_discrete_stoc_len(z_arr), "r--", linewidth=1.5,
            alpha=0.7, label="snapped to levels")
    for sl in SC_LEVELS:
        ax.axhline(y=sl, color="gray", ls=":", alpha=0.3)
    ax.axvline(x=0, color="gray", ls="--", alpha=0.4)
    ax.axvline(x=2, color="gray", ls="--", alpha=0.4)
    ax.set_xlabel("z-score  (= importance / σ_eff)")
    ax.set_ylabel("stoc_len")
    ax.set_title("z → stoc_len mapping\n(log2-linear interpolation)")
    ax.legend(fontsize=9)
    ax.set_xlim(-1, 3)
    ax.set_ylim(0, L_MAX + 20)
    ax.grid(True, alpha=0.2)
    ax.annotate("z<0: all L_MIN", xy=(-0.5, L_MIN), fontsize=8, color="blue")
    ax.annotate("z>2: all L_MAX", xy=(2.1, L_MAX), fontsize=8, color="blue")

    # ── (0,1): σ_eff heatmap ──────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(
        SIGMA_EFF, origin="lower", aspect="auto", cmap="coolwarm_r",
        extent=[T - 1, 0, 0, N_LAYERS - 1],
    )
    ax.set_xlabel("Diffusion timestep t\n(← noisy    clean →)")
    ax.set_ylabel("Layer index")
    ax.set_title(f"σ_eff [{source_label}]\nLarger = more aggressive")
    ax.set_yticks(np.arange(0, N_LAYERS, 4))
    ax.set_xticks(np.arange(0, T, 10))
    for (tx, ty, label) in [
        (80, 3, "Q1"), (80, 24, "Q2"),
        (15, 3, "Q3"), (15, 24, "Q4"),
    ]:
        ax.text(tx, ty, label, ha="center", va="center",
                fontsize=12, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.08)
    plt.colorbar(im, cax=cax, label="σ_eff")

    # ── (0,2): Gaussian PDF with z-boundaries for 4 quadrant corners
    ax = fig.add_subplot(gs[0, 2])
    x_plot = np.linspace(-4, 4, 500)
    # Pick 4 corner points from the actual SIGMA_EFF grid
    quadrant_configs = [
        ("Q1: noisy+early", SIGMA_EFF[2, 90] if SIGMA_EFF.shape[1] > 90 else SIGMA_EFF[2, -10], "C3"),
        ("Q2: noisy+late",  SIGMA_EFF[25, 90] if SIGMA_EFF.shape[1] > 90 else SIGMA_EFF[-3, -10], "C1"),
        ("Q3: clean+early", SIGMA_EFF[2, 5] if SIGMA_EFF.shape[1] > 5 else SIGMA_EFF[2, 0], "C0"),
        ("Q4: clean+late",  SIGMA_EFF[25, 5] if SIGMA_EFF.shape[1] > 5 else SIGMA_EFF[-3, 0], "C2"),
    ]
    for label, s, color in quadrant_configs:
        pdf = stats.norm.pdf(x_plot)
        ax.plot(x_plot, pdf, color=color, linewidth=1.5, label=f"{label} (σ={s:.2f})")
        ax.axvline(x=0, color="gray", ls="--", alpha=0.2)
        ax.axvline(x=2 * s, color=color, ls=":", alpha=0.6)

    ax.set_xlabel("Raw importance  (~ N(0, 1))")
    ax.set_ylabel("Probability density")
    ax.set_title("N(0,1) with z=2 boundaries (dotted)\nRight of dotted = L_MAX")
    ax.legend(fontsize=8)
    ax.set_xlim(-4, 4)
    ax.grid(True, alpha=0.2)

    # ── (1,0)-(1,2): stoc_len heatmaps for z=0σ, 1σ, 2σ tokens ──
    z_vals = [0.0, 1.0, 2.0]
    z_titles = [
        "Token at mean (z=0)\n→ all get L_MIN",
        "Token at +1σ",
        "Token at +2σ\n→ all get L_MAX",
    ]
    for idx, (z_val, title) in enumerate(zip(z_vals, z_titles)):
        ax = fig.add_subplot(gs[1, idx])
        Z_EFF = z_val / SIGMA_EFF
        SL = z_to_stoc_len(Z_EFF)

        im = ax.imshow(
            SL, origin="lower", aspect="auto", cmap="RdYlGn", norm=norm,
            extent=[T - 1, 0, 0, N_LAYERS - 1],
        )
        ax.set_xlabel("Diffusion timestep t\n(← noisy    clean →)")
        ax.set_ylabel("Layer index")
        ax.set_title(title, fontsize=11)
        ax.set_yticks(np.arange(0, N_LAYERS, 4))
        ax.set_xticks(np.arange(0, T, 10))
        for (tx, ty, qlabel) in [
            (80, 3, "Q1"), (80, 24, "Q2"),
            (15, 3, "Q3"), (15, 24, "Q4"),
        ]:
            ax.text(tx, ty, qlabel, ha="center", va="center",
                    fontsize=12, fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.08)
        plt.colorbar(im, cax=cax, label="stoc_len")

    # ── (2,0): Per-level fraction bar chart ───────────────────────
    ax = fig.add_subplot(gs[2, 0])
    x_pos = np.arange(len(SC_LEVELS))
    width = 0.18
    for i, (label, s, color) in enumerate(quadrant_configs):
        fracs = fraction_per_level_gaussian(s)
        bars = ax.bar(x_pos + i * width, fracs, width, label=label, color=color, alpha=0.8)
    ax.set_xticks(x_pos + 1.5 * width)
    ax.set_xticklabels([str(sl) for sl in SC_LEVELS])
    ax.set_xlabel("stoc_len level")
    ax.set_ylabel("Fraction of tokens")
    ax.set_title("Token distribution per precision level")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")

    # ── (2,1): Expected stoc_len heatmap ──────────────────────────
    ax = fig.add_subplot(gs[2, 1])
    E_SL = np.vectorize(expected_stoc_len_gaussian)(SIGMA_EFF)
    im = ax.imshow(
        E_SL, origin="lower", aspect="auto", cmap="RdYlGn", norm=norm,
        extent=[T - 1, 0, 0, N_LAYERS - 1],
    )
    ax.set_xlabel("Diffusion timestep t")
    ax.set_ylabel("Layer index")
    ax.set_title("E[stoc_len] under N(0,1)\n= expected compute budget")
    ax.set_yticks(np.arange(0, N_LAYERS, 4))
    ax.set_xticks(np.arange(0, T, 10))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.08)
    plt.colorbar(im, cax=cax, label="E[stoc_len]")

    # ── (2,2): stoc_len vs raw importance curves ──────────────────
    ax = fig.add_subplot(gs[2, 2])
    raw_arr = np.linspace(-2, 4, 500)
    for label, s, color in quadrant_configs:
        sl = z_to_stoc_len(raw_arr / s)
        ax.plot(raw_arr, sl, label=f"{label} (σ={s:.1f})", linewidth=2, color=color)
    ax.axvline(x=0, color="gray", ls="--", alpha=0.3, label="mean")
    ax.set_xlabel("Raw token importance (σ units from mean)")
    ax.set_ylabel("stoc_len")
    ax.set_title("Precision vs raw importance")
    ax.legend(fontsize=8)
    ax.set_xlim(-2, 4)
    ax.set_ylim(0, L_MAX + 20)
    ax.grid(True, alpha=0.2)

    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output}")

    # ── Summary table ─────────────────────────────────────────────
    print(f"\n{'=' * 78}")
    print(f"{'Quadrant Summary (Gaussian Model)':^78}")
    print(f"{'=' * 78}")
    print(f"  {'Quadrant':<22s}  {'σ_eff':>6s}  {'E[SL]':>7s}  ", end="")
    for sl in SC_LEVELS:
        print(f"  SL={sl:<3d}", end="")
    print()
    print(f"  {'-' * 74}")
    for label, s, _ in quadrant_configs:
        e_sl = expected_stoc_len_gaussian(s)
        fracs = fraction_per_level_gaussian(s)
        print(f"  {label:<22s}  {s:6.2f}  {e_sl:7.1f}  ", end="")
        for f in fracs:
            print(f"  {f:6.1%}", end="")
        print()
    print(f"{'=' * 78}")

    full = L_MAX * T * N_LAYERS
    actual = E_SL.sum()
    savings = 1.0 - actual / full
    print(f"\nE[total compute]: {actual:.0f} / {full} = {actual / full:.1%}  "
          f"(savings: {savings:.1%})")


if __name__ == "__main__":
    main()
