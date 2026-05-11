"""
Visualize Mixed Precision distribution across timesteps, blocks, and operators.

Usage:
    python scripts/plot_mp_distribution.py <csv_path> [--output_dir <dir>]

Generates stacked bar/area charts showing the fraction of rows at each
stoc_len level, grouped by operator, with one subplot per operator.
"""
import argparse
import os
import re

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Plot MP distribution from CSV")
    parser.add_argument("csv_path", type=str, help="Path to debug_mp_distribution.csv")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for plots (default: same as CSV)")
    return parser.parse_args()


def get_stoc_len_columns(df):
    """Extract stoc_len levels from column names like sl_256_frac."""
    frac_cols = [c for c in df.columns if re.match(r"sl_\d+_frac", c)]
    levels = []
    for c in frac_cols:
        sl = int(c.split("_")[1])
        levels.append(sl)
    levels.sort(reverse=True)
    return levels


def get_color_map(levels):
    """Assign distinct colors to each stoc_len level, high=blue, low=red."""
    n = len(levels)
    if n <= 6:
        # Hand-picked distinguishable palette
        palette = ["#2166ac", "#67a9cf", "#d1e5f0", "#fddbc7", "#ef8a62", "#b2182b"]
        colors = palette[:n]
    else:
        cmap = plt.cm.RdYlBu_r
        colors = [mcolors.to_hex(cmap(i / max(n - 1, 1))) for i in range(n)]
    # Reverse so highest stoc_len = coolest color (blue)
    return {sl: colors[i] for i, sl in enumerate(levels)}


def plot_by_operator_vs_timestep(df, levels, color_map, output_dir):
    """For each operator, plot stacked area: x=timestep, y=fraction per level.

    Averages across all blocks.
    """
    operators = sorted(df["operator"].unique())
    n_ops = len(operators)

    fig, axes = plt.subplots(1, n_ops, figsize=(5 * n_ops, 4), sharey=True)
    if n_ops == 1:
        axes = [axes]

    for ax, op in zip(axes, operators):
        sub = df[df["operator"] == op]
        # Average fractions across blocks for each timestep
        # Sort descending so x-axis goes noisy (high t) → clean (low t)
        timesteps = sorted(sub["timestep"].unique(), reverse=True)
        frac_data = {sl: [] for sl in levels}

        for t in timesteps:
            t_sub = sub[sub["timestep"] == t]
            for sl in levels:
                col = f"sl_{sl}_frac"
                if col in t_sub.columns:
                    frac_data[sl].append(t_sub[col].mean())
                else:
                    frac_data[sl].append(0.0)

        # Stacked area
        bottom = np.zeros(len(timesteps))
        for sl in levels:
            vals = np.array(frac_data[sl])
            ax.bar(range(len(timesteps)), vals, bottom=bottom,
                   color=color_map[sl], label=f"sl={sl}", width=1.0, linewidth=0)
            bottom += vals

        ax.set_title(op, fontsize=13, fontweight="bold")
        ax.set_xlabel("Timestep (noisy → clean)")
        ax.set_xticks(range(0, len(timesteps), max(1, len(timesteps) // 5)))
        ax.set_xticklabels([str(timesteps[i]) for i in
                            range(0, len(timesteps), max(1, len(timesteps) // 5))])
        ax.set_ylim(0, 1)

    axes[0].set_ylabel("Fraction of rows")

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(levels),
               bbox_to_anchor=(0.5, 1.08), fontsize=9)

    fig.suptitle("MP Distribution: Fraction per Precision Level vs Timestep\n(averaged across blocks)",
                 fontsize=14, y=1.15)
    fig.tight_layout()
    path = os.path.join(output_dir, "mp_dist_by_timestep.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_by_operator_vs_block(df, levels, color_map, output_dir):
    """For each operator, plot stacked bar: x=block, y=fraction per level.

    Averages across all timesteps.
    """
    operators = sorted(df["operator"].unique())
    n_ops = len(operators)

    fig, axes = plt.subplots(1, n_ops, figsize=(5 * n_ops, 4), sharey=True)
    if n_ops == 1:
        axes = [axes]

    for ax, op in zip(axes, operators):
        sub = df[df["operator"] == op]
        blocks = sorted(sub["block"].unique())
        frac_data = {sl: [] for sl in levels}

        for b in blocks:
            b_sub = sub[sub["block"] == b]
            for sl in levels:
                col = f"sl_{sl}_frac"
                if col in b_sub.columns:
                    frac_data[sl].append(b_sub[col].mean())
                else:
                    frac_data[sl].append(0.0)

        bottom = np.zeros(len(blocks))
        for sl in levels:
            vals = np.array(frac_data[sl])
            ax.bar(range(len(blocks)), vals, bottom=bottom,
                   color=color_map[sl], label=f"sl={sl}", width=0.8, linewidth=0)
            bottom += vals

        ax.set_title(op, fontsize=13, fontweight="bold")
        ax.set_xlabel("Block Index")
        ax.set_xticks(range(0, len(blocks), max(1, len(blocks) // 7)))
        ax.set_xticklabels([str(blocks[i]) for i in
                            range(0, len(blocks), max(1, len(blocks) // 7))])
        ax.set_ylim(0, 1)

    axes[0].set_ylabel("Fraction of rows")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(levels),
               bbox_to_anchor=(0.5, 1.08), fontsize=9)

    fig.suptitle("MP Distribution: Fraction per Precision Level vs Block\n(averaged across timesteps)",
                 fontsize=14, y=1.15)
    fig.tight_layout()
    path = os.path.join(output_dir, "mp_dist_by_block.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_heatmap_per_operator(df, levels, output_dir):
    """For each operator, plot a heatmap: x=block, y=timestep, color=dominant stoc_len.

    Also generates per-level heatmaps showing the fraction at that level.
    """
    operators = sorted(df["operator"].unique())

    for op in operators:
        sub = df[df["operator"] == op]
        timesteps = sorted(sub["timestep"].unique())
        blocks = sorted(sub["block"].unique())

        # Per-level fraction heatmaps
        n_levels = len(levels)
        fig, axes = plt.subplots(1, n_levels, figsize=(4 * n_levels, 6), sharey=True)
        if n_levels == 1:
            axes = [axes]

        for ax, sl in zip(axes, levels):
            col = f"sl_{sl}_frac"
            grid = np.zeros((len(timesteps), len(blocks)))
            for ti, t in enumerate(timesteps):
                for bi, b in enumerate(blocks):
                    row = sub[(sub["timestep"] == t) & (sub["block"] == b)]
                    if len(row) > 0 and col in row.columns:
                        grid[ti, bi] = row[col].values[0]

            im = ax.imshow(grid, aspect="auto", cmap="YlOrRd",
                           vmin=0, vmax=1, origin="upper")
            ax.set_title(f"sl={sl}", fontsize=11)
            ax.set_xlabel("Block")
            ax.set_xticks(range(0, len(blocks), max(1, len(blocks) // 7)))
            ax.set_xticklabels([str(blocks[i]) for i in
                                range(0, len(blocks), max(1, len(blocks) // 7))])
            if ax == axes[0]:
                ax.set_ylabel("Timestep")
                ax.set_yticks(range(0, len(timesteps), max(1, len(timesteps) // 10)))
                ax.set_yticklabels([str(timesteps[i]) for i in
                                    range(0, len(timesteps), max(1, len(timesteps) // 10))])

        fig.colorbar(im, ax=axes, shrink=0.6, label="Fraction")
        fig.suptitle(f"MP Fraction Heatmap — {op}\n(rows=timestep, cols=block)",
                     fontsize=13, y=1.02)
        fig.tight_layout()
        path = os.path.join(output_dir, f"mp_heatmap_{op}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
        plt.close(fig)


def main():
    args = parse_args()
    df = pd.read_csv(args.csv_path)

    output_dir = args.output_dir or os.path.dirname(args.csv_path)
    os.makedirs(output_dir, exist_ok=True)

    levels = get_stoc_len_columns(df)
    color_map = get_color_map(levels)

    print(f"Operators: {sorted(df['operator'].unique())}")
    print(f"Stoc_len levels: {levels}")
    print(f"Timesteps: {sorted(df['timestep'].unique())}")
    print(f"Blocks: {sorted(df['block'].unique())}")

    # Plot: Stacked bars by timestep (averaged across blocks)
    plot_by_operator_vs_timestep(df, levels, color_map, output_dir)

    print(f"\nPlot saved to: {output_dir}")


if __name__ == "__main__":
    main()
