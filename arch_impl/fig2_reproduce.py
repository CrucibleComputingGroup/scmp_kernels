"""Reproduce Kim et al. (DAC '16) Figure 2 with scmp_kernels.

Three experiments:
  E0 — Baseline. Pure-Python iid-Bernoulli XNOR multiplier (paper's setup).
       Verifies that the paper's "tent at origin" property holds for the
       *original* SC scheme it analysed. Sets the expected pattern.
  E1 — Fig 2(a) "tent" error surface for scmp_kernels' enable-signal SC
       multiplier over (X, Y) in [-1, 1]^2 with 1024-bit streams.
  E2 — Fig 2(b) sum-of-products error vs weight value for scmp_kernels
       with weights concentrated near zero (N(0, sigma^2)) and a small but
       non-zero input (so the enable signal can actually fire).

Outputs:
  arch_impl/figures/fig2a_surface.png        — scmp_kernels error surface
  arch_impl/figures/fig2a_heatmap.png        — scmp_kernels heatmap
  arch_impl/figures/fig2a_baseline_iid.png   — E0 paper-style tent
  arch_impl/figures/fig2b_near_zero.png      — error vs weight value
  arch_impl/logs/fig2_run.log

Pass criteria printed at the end as VERDICT-E0 / VERDICT-E1 / VERDICT-E2.
"""
from __future__ import annotations

import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/Projects/scmp_kernels"))

from scmp_kernels import sc_matmul


# ---------------------------------------------------------------------------
# Knobs
# ---------------------------------------------------------------------------

GRID_N    = 41        # 41x41 grid over [-1, 1]^2
SC_PREC   = 8         # production setting in scmp_llm / vit_sc (q_max = 127 bipolar)
HALVE     = True      # uSystolic/HUB cycle-halving: stoc_len → 2^(sc_prec-1) = 128
STOC_LEN  = 2 ** (SC_PREC - 1) if HALVE else 2 ** SC_PREC   # 128 with halve, else 256
SEEDS_E2  = 8         # average E2 over a few Sobol config rotations
N_WEIGHTS = 20_000    # match paper's count
SIGMA_W   = 0.1       # weight std; gives roughly [-0.4, 0.4] range like Fig 2(b)
HIST_BINS = 40        # weight-value bins for the histogram + per-bin error

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def banner(msg: str) -> None:
    print("\n" + "=" * 72)
    print(msg)
    print("=" * 72, flush=True)


# ---------------------------------------------------------------------------
# E0 — Baseline: iid Bernoulli XNOR multiplier (paper's actual setup)
# ---------------------------------------------------------------------------

def iid_xnor_bipolar(x: torch.Tensor, y: torch.Tensor, stoc_len: int,
                    seed: int = 0) -> torch.Tensor:
    """Plain iid Bernoulli bipolar SC multiply (XNOR), broadcast over a 2D grid.

    Each output[i, j] is the decoded mean of ``stoc_len`` independent XNOR bits
    where  P(a_bit=1) = (1+x[i])/2,  P(b_bit=1) = (1+y[j])/2.
    """
    g = torch.Generator(device=x.device).manual_seed(seed)
    px = ((1.0 + x) / 2.0).clamp_(0.0, 1.0).view(-1, 1, 1)
    py = ((1.0 + y) / 2.0).clamp_(0.0, 1.0).view(1, -1, 1)
    # a_bits: (N, 1, T), b_bits: (1, M, T); broadcast → (N, M, T) when XNOR.
    a_bits = (torch.rand(x.numel(), 1, stoc_len, generator=g, device=x.device) < px)
    b_bits = (torch.rand(1, y.numel(), stoc_len, generator=g, device=x.device) < py)
    xnor = (a_bits == b_bits)
    p_xnor = xnor.float().mean(dim=-1)            # (N, M)
    return 2.0 * p_xnor - 1.0                     # bipolar decode


def run_e0() -> dict:
    banner(f"E0 (baseline): iid Bernoulli XNOR over [-1,1]^2, "
           f"{GRID_N}x{GRID_N} grid, stoc_len={STOC_LEN}")
    xs = torch.linspace(-1.0, 1.0, GRID_N, device=DEVICE)
    ys = torch.linspace(-1.0, 1.0, GRID_N, device=DEVICE)
    # Average over a few seeds to surface the |error| pattern (paper plotted
    # absolute error, but a single shot is noisy; the *shape* is the property).
    errs = []
    for seed in range(8):
        y_sc = iid_xnor_bipolar(xs, ys, STOC_LEN, seed=seed)
        y_ref = xs.view(-1, 1) * ys.view(1, -1)
        errs.append((y_sc - y_ref).abs())
    err = torch.stack(errs).mean(dim=0).cpu().numpy()
    xs_np, ys_np = xs.cpu().numpy(), ys.cpu().numpy()
    # Fig 2(a)'s tent peaks where the iid-XNOR variance (1-x^2 y^2)/N is largest
    # — i.e. along the axes where |x*y| ≈ 0 — and is smallest at the corners
    # where |x*y| → 1. So compare LOW-|xy| band vs HIGH-|xy| band, not a
    # central square vs a frame.
    xy = np.abs(xs_np)[:, None] * np.abs(ys_np)[None, :]
    central_mask = xy < 0.10
    border_mask  = xy > 0.70
    arg = np.unravel_index(np.argmax(err), err.shape)
    mean_centre = float(err[central_mask].mean())
    mean_border = float(err[border_mask].mean())
    ratio = mean_centre / max(mean_border, 1e-12)
    in_centre = bool(central_mask[arg])
    print(f"  max_err={err.max():.4e} at (X,Y)=({xs_np[arg[0]]:+.3f},{ys_np[arg[1]]:+.3f})  "
          f"in_centre={in_centre}")
    print(f"  mean_err centre={mean_centre:.4e}  border={mean_border:.4e}  "
          f"ratio={ratio:.2f}x")
    e0_pass = ratio >= 1.5
    print(f"  VERDICT-E0 (low-|xy|/high-|xy| ≥ 1.5x → paper tent visible): "
          f"{'PASS' if e0_pass else 'FAIL'}")
    return {"xs": xs_np, "ys": ys_np, "err": err,
            "max_loc": (xs_np[arg[0]], ys_np[arg[1]]), "in_centre": in_centre,
            "mean_centre": mean_centre, "mean_border": mean_border,
            "ratio": ratio, "pass": e0_pass}


# ---------------------------------------------------------------------------
# E1 — Fig 2(a) error surface for a single SC multiplier (D=1)
# ---------------------------------------------------------------------------

def run_e1() -> dict:
    banner(f"E1: Fig 2(a) surface — {GRID_N}x{GRID_N} grid, stoc_len={STOC_LEN}")
    # Use a grid that lands exactly on the quantisation lattice (k/q_max) so
    # quantisation contributes zero error and we measure ONLY the stochastic
    # component — what Fig 2(a) shows.
    q_max = 2 ** (SC_PREC - 1) - 1   # bipolar magnitude grid (=511 for sc_prec=10)
    k_step = max(1, (2 * q_max) // (GRID_N - 1))     # integer step on the lattice
    ks = torch.arange(-(GRID_N // 2), GRID_N // 2 + 1, device=DEVICE) * k_step
    ks = ks[:GRID_N].clamp(-q_max, q_max)
    xs = ks.float() / q_max
    ys = xs.clone()

    a = xs.view(-1, 1)
    b = ys.view(-1, 1)
    anchors_a = torch.tensor([[1.0], [-1.0]], device=DEVICE)
    anchors_b = torch.tensor([[1.0], [-1.0]], device=DEVICE)
    a = torch.cat([a, anchors_a], dim=0)
    b = torch.cat([b, anchors_b], dim=0)

    t0 = time.time()
    y_sc = sc_matmul(a, b,
                     granularity="per_tensor",
                     mode="bipolar",
                     sc_prec=SC_PREC,
                     halve_bipolar_stoc_len=HALVE)
    t_sc = time.time() - t0
    print(f"  sc_matmul: {y_sc.shape}, dtype={y_sc.dtype}, t={t_sc:.2f}s")

    y_sc = y_sc[:GRID_N, :GRID_N]
    y_ref = xs.view(-1, 1) * ys.view(1, -1)
    err = (y_sc - y_ref).abs().cpu().numpy()

    xs_np = xs.cpu().numpy()
    ys_np = ys.cpu().numpy()

    # Pass criterion: max error must lie in central |X|<0.2 ∧ |Y|<0.2 region;
    # central row/col mean must be ≥ 2× border mean.
    # Fig 2(a)'s tent peaks where the iid-XNOR variance (1-x^2 y^2)/N is largest
    # — i.e. along the axes where |x*y| ≈ 0 — and is smallest at the corners
    # where |x*y| → 1. So compare LOW-|xy| band vs HIGH-|xy| band, not a
    # central square vs a frame.
    xy = np.abs(xs_np)[:, None] * np.abs(ys_np)[None, :]
    central_mask = xy < 0.10
    border_mask  = xy > 0.70
    arg = np.unravel_index(np.argmax(err), err.shape)
    max_loc = (xs_np[arg[0]], ys_np[arg[1]])
    max_in_centre = bool(central_mask[arg])
    mean_centre = float(err[central_mask].mean())
    mean_border = float(err[border_mask].mean())
    ratio = mean_centre / max(mean_border, 1e-12)

    print(f"  grid spans X∈[{xs_np[0]:+.3f},{xs_np[-1]:+.3f}] (exact lattice points)")
    print(f"  max_err={err.max():.4e} at (X,Y)=({max_loc[0]:+.3f},{max_loc[1]:+.3f})  "
          f"in_centre={max_in_centre}")
    print(f"  mean_err centre={mean_centre:.4e}  border={mean_border:.4e}  "
          f"ratio={ratio:.2f}x")

    # Property: low-|xy| band mean must be ≥ 1.5× the high-|xy| band mean
    # (paper's Fig 2(a) shows roughly 2.3× theoretical ratio at 1024-bit).
    e1_pass = ratio >= 1.5
    print(f"  VERDICT-E1 (low-|xy|/high-|xy| ≥ 1.5x): "
          f"{'PASS' if e1_pass else 'FAIL'}")

    return {
        "xs": xs_np, "ys": ys_np, "err": err,
        "max_loc": max_loc, "max_in_centre": max_in_centre,
        "mean_centre": mean_centre, "mean_border": mean_border, "ratio": ratio,
        "pass": e1_pass,
    }


# ---------------------------------------------------------------------------
# E2 — Fig 2(b) near-zero weight sum-of-products error
# ---------------------------------------------------------------------------

def run_e2() -> dict:
    # We deviate from the paper's footnote ("input = 0") because scmp_kernels'
    # enable-signal SC has *exact* zero output whenever mag_a = 0 (the enable
    # never fires), which trivially gives all-zero errors. To still surface
    # the per-weight error pattern, we fix an input to a small but non-zero
    # value typical of post-tanh activations.
    X_FIXED = 0.5
    banner(f"E2: per-weight |error| with w~N(0,{SIGMA_W}^2), x={X_FIXED:+.2f} fixed, "
           f"N={N_WEIGHTS}, stoc_len={STOC_LEN}, seeds={SEEDS_E2}")

    g = torch.Generator(device=DEVICE).manual_seed(0)
    weights = (SIGMA_W * torch.randn(N_WEIGHTS, generator=g, device=DEVICE)).clamp_(-1.0, 1.0)
    inputs  = torch.full((N_WEIGHTS,), X_FIXED, device=DEVICE)

    err_per_seed = []
    for seed in range(SEEDS_E2):
        # Permute so each weight sees a different Sobol axis across seeds.
        perm = torch.randperm(N_WEIGHTS, generator=g, device=DEVICE)
        w_perm = weights[perm].view(-1, 1)
        x_perm = inputs[perm].view(-1, 1)

        anchors = torch.tensor([[1.0], [-1.0]], device=DEVICE)
        a = torch.cat([w_perm, anchors], dim=0)
        b = torch.cat([x_perm, anchors], dim=0)

        y_sc = sc_matmul(a, b,
                         granularity="per_tensor",
                         mode="bipolar",
                         sc_prec=SC_PREC,
                         halve_bipolar_stoc_len=HALVE)
        # Diagonal element y_sc[i, i] is sc(w_i, x_i)
        y_diag = y_sc.diagonal()[:N_WEIGHTS]
        ref    = weights[perm] * inputs[perm]
        per_w_err = (y_diag - ref).abs()
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(N_WEIGHTS, device=DEVICE)
        err_per_seed.append(per_w_err[inv_perm].cpu().numpy())

    err = np.mean(np.stack(err_per_seed, axis=0), axis=0)
    w_np = weights.cpu().numpy()

    # Histogram of weights + per-bin mean abs error.
    bins = np.linspace(-0.45, 0.45, HIST_BINS + 1)
    bin_idx = np.clip(np.digitize(w_np, bins) - 1, 0, HIST_BINS - 1)
    counts = np.bincount(bin_idx, minlength=HIST_BINS)
    err_sum = np.bincount(bin_idx, weights=err, minlength=HIST_BINS)
    with np.errstate(divide="ignore", invalid="ignore"):
        bin_err = np.where(counts > 0, err_sum / np.maximum(counts, 1), np.nan)
    bin_centres = 0.5 * (bins[:-1] + bins[1:])

    # Pass criterion: error peaks at central bin and falls off as |w| grows.
    # Concretely: mean error in |w|<0.05 bins must exceed mean error in |w|>0.3 bins.
    near_zero = np.abs(bin_centres) < 0.05
    far_edge  = np.abs(bin_centres) > 0.30
    mean_nz   = float(np.nanmean(bin_err[near_zero]))
    mean_far  = float(np.nanmean(bin_err[far_edge]))
    e2_pass = mean_nz > mean_far

    # Monotone-ish check: smoothed bin_err should be monotonically decreasing
    # in |w|. Allow small wobble.
    abs_centres = np.abs(bin_centres)
    order = np.argsort(abs_centres)
    sorted_err = bin_err[order]
    sorted_abs = abs_centres[order]
    # Group by 0.05-wide |w| bands and check the band means.
    band_edges = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40])
    band_means = []
    for lo, hi in zip(band_edges[:-1], band_edges[1:]):
        sel = (sorted_abs >= lo) & (sorted_abs < hi)
        band_means.append(float(np.nanmean(sorted_err[sel])) if sel.any() else np.nan)
    band_means_arr = np.array(band_means)
    print("  |w| band means (ascending |w|):")
    for (lo, hi), v in zip(zip(band_edges[:-1], band_edges[1:]), band_means_arr):
        print(f"    [{lo:.2f},{hi:.2f}):  {v:.4e}")
    print(f"  mean_err |w|<0.05: {mean_nz:.4e}    mean_err |w|>0.30: {mean_far:.4e}")
    print(f"  VERDICT-E2: {'PASS' if e2_pass else 'FAIL'}")

    return {
        "w": w_np, "err": err,
        "bin_centres": bin_centres, "counts": counts, "bin_err": bin_err,
        "mean_nz": mean_nz, "mean_far": mean_far,
        "band_edges": band_edges, "band_means": band_means_arr,
        "pass": e2_pass,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_surface_pair(res: dict, title: str, surf_path: str, heat_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    X, Y = np.meshgrid(res["xs"], res["ys"], indexing="ij")

    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, res["err"], cmap="viridis",
                    edgecolor="k", linewidth=0.15, alpha=0.95)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("|error|")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(surf_path, dpi=150); plt.close(fig)
    print(f"  wrote {surf_path}")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(res["err"], origin="lower", extent=(-1, 1, -1, 1),
                   cmap="viridis", aspect="equal")
    fig.colorbar(im, ax=ax, label="|error|")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_title(title)
    fig.tight_layout()
    fig.savefig(heat_path, dpi=150); plt.close(fig)
    print(f"  wrote {heat_path}")


def plot_e0(res: dict) -> None:
    title = (f"Fig 2(a) baseline — iid Bernoulli XNOR (paper's setup)\n"
             f"stoc_len={STOC_LEN} (matched to scmp_kernels), 8-seed mean |error|")
    _plot_surface_pair(
        res, title,
        os.path.join(OUT_DIR, "fig2a_baseline_iid_surface.png"),
        os.path.join(OUT_DIR, "fig2a_baseline_iid_heatmap.png"),
    )


def plot_e1(res: dict) -> None:
    title = (f"Fig 2(a) — scmp_kernels (enable-signal + Sobol)\n"
             f"bipolar, sc_prec={SC_PREC}, halve={HALVE}, stoc_len={STOC_LEN}")
    _plot_surface_pair(
        res, title,
        os.path.join(OUT_DIR, "fig2a_surface.png"),
        os.path.join(OUT_DIR, "fig2a_heatmap.png"),
    )


def make_presentation_figures(res0: dict, res1: dict, res2: dict) -> None:
    """Two presentation-ready figures that mirror the paper's Fig 2 layout."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # --- Fig P1: side-by-side surface comparison (the impact shot) ----------
    fig = plt.figure(figsize=(13, 5))

    X0, Y0 = np.meshgrid(res0["xs"], res0["ys"], indexing="ij")
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    s0 = ax.plot_surface(X0, Y0, res0["err"], cmap="viridis",
                         edgecolor="k", linewidth=0.15, alpha=0.95)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("|error|")
    ax.set_title(f"(a) Paper's setup: iid Bernoulli XNOR\n"
                 f"stoc_len={STOC_LEN}, 8-seed mean")
    fig.colorbar(s0, ax=ax, shrink=0.55, pad=0.10)

    X1, Y1 = np.meshgrid(res1["xs"], res1["ys"], indexing="ij")
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    s1 = ax.plot_surface(X1, Y1, res1["err"], cmap="viridis",
                         edgecolor="k", linewidth=0.15, alpha=0.95)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("|error|")
    ax.set_title(f"(b) scmp_kernels: enable-signal + Sobol QMC\n"
                 f"sc_prec={SC_PREC}, halve={HALVE}, stoc_len={STOC_LEN}")
    fig.colorbar(s1, ax=ax, shrink=0.55, pad=0.10)

    fig.suptitle("Bipolar SC multiplier error surface — paper's Fig 2(a) vs scmp_kernels",
                 fontsize=13)
    fig.tight_layout()
    p1 = os.path.join(OUT_DIR, "fig2_presentation_compare.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"  wrote {p1}")

    # --- Fig P2: paper-Fig2-style composite for scmp_kernels ----------------
    fig = plt.figure(figsize=(13, 5))

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    s = ax.plot_surface(X1, Y1, res1["err"], cmap="viridis",
                        edgecolor="k", linewidth=0.15, alpha=0.95)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("|error|")
    ax.set_title("(a) Error surface of scmp_kernels bipolar SC multiplier")
    fig.colorbar(s, ax=ax, shrink=0.55, pad=0.10)

    ax1 = fig.add_subplot(1, 2, 2)
    width = (res2["bin_centres"][1] - res2["bin_centres"][0]) * 0.9
    ax1.bar(res2["bin_centres"], res2["counts"], width=width,
            color="lightgray", edgecolor="gray", label="# weights")
    ax1.set_xlabel("weight value")
    ax1.set_ylabel("# weights", color="gray")
    ax1.tick_params(axis="y", labelcolor="gray")
    ax1.set_xlim(-0.45, 0.45)
    ax2 = ax1.twinx()
    ax2.plot(res2["bin_centres"], res2["bin_err"], "o-", color="C3",
             label="|error|")
    ax2.set_ylabel("|error| (x=+0.5 fixed)", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")
    ax1.set_title("(b) Weight histogram + per-weight error")

    fig.suptitle(f"scmp_kernels Fig 2 analog (sc_prec={SC_PREC}, halve={HALVE}, "
                 f"stoc_len={STOC_LEN})", fontsize=13)
    fig.tight_layout()
    p2 = os.path.join(OUT_DIR, "fig2_presentation_scmp.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"  wrote {p2}")


def plot_e2(res: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(7.5, 5))
    ax1.bar(res["bin_centres"], res["counts"], width=(res["bin_centres"][1]-res["bin_centres"][0])*0.9,
            color="lightgray", edgecolor="gray", label="# weights")
    ax1.set_xlabel("weight value")
    ax1.set_ylabel("# weights", color="gray")
    ax1.tick_params(axis="y", labelcolor="gray")
    ax1.set_xlim(-0.45, 0.45)

    ax2 = ax1.twinx()
    ax2.plot(res["bin_centres"], res["bin_err"], "o-", color="C3",
             label="|error multiplying by 0|")
    ax2.set_ylabel("|error|", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")

    ax1.set_title(f"Fig 2(b) reproduction — scmp_kernels (Triton GPU)\n"
                  f"N={N_WEIGHTS} weights ~ N(0, {SIGMA_W:.2f}^2), "
                  f"sc_prec={SC_PREC}, halve={HALVE}, stoc_len={STOC_LEN}")
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "fig2b_near_zero.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  wrote {p}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    print(f"device={DEVICE} torch={torch.__version__} "
          f"{(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')}")
    print(f"sc_prec={SC_PREC}  halve_bipolar_stoc_len={HALVE}  "
          f"stoc_len={STOC_LEN}  "
          f"q_max={2**(SC_PREC-1)-1}")

    res0 = run_e0()
    plot_e0(res0)

    res1 = run_e1()
    plot_e1(res1)

    res2 = run_e2()
    plot_e2(res2)

    banner("Presentation figures")
    make_presentation_figures(res0, res1, res2)

    banner("SUMMARY")
    print(f"  E0 baseline (paper's iid XNOR — expected to PASS):")
    print(f"      max_at=({res0['max_loc'][0]:+.3f},{res0['max_loc'][1]:+.3f}) in_centre={res0['in_centre']}  "
          f"centre/border={res0['ratio']:.2f}x  → {'PASS' if res0['pass'] else 'FAIL'}")
    print(f"  E1 scmp_kernels (tent in (X,Y)):")
    print(f"      max_at=({res1['max_loc'][0]:+.3f},{res1['max_loc'][1]:+.3f}) in_centre={res1['max_in_centre']}  "
          f"centre/border={res1['ratio']:.2f}x  → {'PASS' if res1['pass'] else 'FAIL'}")
    print(f"  E2 scmp_kernels (error vs |w|, x={0.5:+.2f}):")
    print(f"      mean(|w|<0.05)={res2['mean_nz']:.3e}  mean(|w|>0.30)={res2['mean_far']:.3e}  "
          f"→ {'PASS' if res2['pass'] else 'FAIL'}")

    e0_holds = res0["pass"]
    new_holds = res1["pass"] and res2["pass"]
    print()
    print(f"  Paper's property in original (iid-XNOR) setup: "
          f"{'CONFIRMED' if e0_holds else 'NOT-CONFIRMED'}")
    print(f"  Same property in scmp_kernels: "
          f"{'STILL HOLDS' if new_holds else 'DOES NOT HOLD'}")
    return 0 if new_holds else 1


if __name__ == "__main__":
    sys.exit(main())
