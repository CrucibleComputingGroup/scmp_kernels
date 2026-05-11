#!/usr/bin/env python3
"""
Plot mixed-precision schedules from a calibrated adaptive-MP table.

For each operator, this script creates one figure with one subplot per block.
Each subplot is a stacked area chart over diffusion timestep. Colors indicate
the fraction assigned to each precision level.

If a SC precision config JSON is provided, operators not covered by the
adaptive-MP table (for example fixed qk/av assignments) are also visualized.
Disabled / out-of-timewise regions are shown as FP (gray).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np

_mplconfig_dir = Path("/tmp/matplotlib-codex")
_mplconfig_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mplconfig_dir))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FP_LABEL = "FP"


def parse_args():
    parser = argparse.ArgumentParser(description="Plot calibrated MP schedule by operator/block.")
    parser.add_argument(
        "--adaptive_mp_table",
        type=str,
        required=True,
        help="Path to threshold_mp_calibration*.json.",
    )
    parser.add_argument(
        "--sc_config",
        type=str,
        default=None,
        help="Optional SC precision map JSON. Used to show fixed qk/av allocations.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for output plots. Default: alongside the adaptive MP table.",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=100,
        help="Total diffusion timesteps. Default: 100.",
    )
    parser.add_argument(
        "--total_blocks",
        type=int,
        default=None,
        help="Total transformer blocks. Default: infer from sc_config, else 28.",
    )
    parser.add_argument(
        "--operators",
        type=str,
        default=None,
        help="Comma-separated operator subset. Default: union of table/sc_config operators.",
    )
    parser.add_argument(
        "--ncols",
        type=int,
        default=4,
        help="Number of subplot columns per operator figure. Default: 4.",
    )
    return parser.parse_args()


def bucket_index(value: int, total: int, num_buckets: int) -> int:
    if num_buckets <= 1 or total <= 1:
        return 0
    ratio = value / max(total - 1, 1)
    return min(num_buckets - 1, int(ratio * num_buckets))


def parse_bucket_key(bucket_key: str) -> tuple[str, int, int]:
    operator, t_part, l_part = bucket_key.split(":")
    return operator, int(t_part[1:]), int(l_part[1:])


def parse_operator_set(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def get_enabled_fraction(operator_cfg: dict, timestep: int, total_timesteps: int) -> float:
    if not operator_cfg.get("enabled", False):
        return 0.0
    timewise = float(operator_cfg.get("timewise", 0.0))
    threshold = int(timewise * total_timesteps)
    if threshold <= 0:
        return 0.0
    cutoff = total_timesteps - threshold
    return 1.0 if timestep >= cutoff else 0.0


def fixed_fraction_from_config(operator_cfg: dict) -> dict[int, float]:
    group_stoc_lens = operator_cfg.get("group_stoc_lens")
    if group_stoc_lens:
        counts = Counter(int(x) for x in group_stoc_lens)
        total = float(sum(counts.values()))
        return {sl: count / total for sl, count in counts.items()}
    return {int(operator_cfg["stoc_len"]): 1.0}


def collect_level_labels(table: dict, sc_config: dict | None) -> list[int | str]:
    levels = {int(x) for x in table.get("stoc_len_levels", [])}
    if sc_config is not None:
        for block_cfg in sc_config.get("blocks", []):
            for operator_cfg in block_cfg.values():
                levels.add(int(operator_cfg["stoc_len"]))
                for sl in operator_cfg.get("group_stoc_lens") or []:
                    levels.add(int(sl))
    ordered = sorted(levels, reverse=True)
    return ordered + [FP_LABEL]


def get_color_map(level_labels: list[int | str]) -> dict[int | str, str]:
    base_colors = {
        256: "#0b3c5d",
        224: "#1d5f8c",
        192: "#2e86ab",
        160: "#4fa3c7",
        128: "#67c587",
        96: "#a4d86f",
        64: "#f1c453",
        32: "#f48c42",
        16: "#d94f3d",
        0: "#7f0000",
        FP_LABEL: "#d9d9d9",
    }
    fallback = ["#0b3c5d", "#2e86ab", "#67c587", "#f1c453", "#f48c42", "#d94f3d"]
    color_map: dict[int | str, str] = {}
    fallback_idx = 0
    for label in level_labels:
        if label in base_colors:
            color_map[label] = base_colors[label]
        else:
            color_map[label] = fallback[fallback_idx % len(fallback)]
            fallback_idx += 1
    return color_map


def build_table_lookup(table: dict) -> tuple[dict[str, dict], dict[tuple[str, int, int], dict]]:
    operator_defaults = dict(table.get("operator_defaults", {}))
    buckets: dict[tuple[str, int, int], dict] = {}
    for bucket_key, payload in table.get("buckets", {}).items():
        buckets[parse_bucket_key(bucket_key)] = payload
    return operator_defaults, buckets


def resolve_fraction_schedule(
    operator: str,
    block_idx: int,
    total_blocks: int,
    total_timesteps: int,
    level_labels: list[int | str],
    table: dict,
    operator_defaults: dict[str, dict],
    bucket_payloads: dict[tuple[str, int, int], dict],
    sc_config: dict | None,
) -> dict[int | str, np.ndarray]:
    values = {
        label: np.zeros(total_timesteps, dtype=np.float32)
        for label in level_labels
    }

    table_levels = [int(x) for x in table.get("stoc_len_levels", [])]
    timestep_buckets = int(table.get("timestep_buckets", 1))
    layer_buckets = int(table.get("layer_buckets", 1))

    operator_block_cfg = None
    if sc_config is not None and 0 <= block_idx < len(sc_config.get("blocks", [])):
        operator_block_cfg = sc_config["blocks"][block_idx].get(operator)

    fixed_fraction = (
        fixed_fraction_from_config(operator_block_cfg)
        if operator_block_cfg is not None
        else None
    )

    for timestep in range(total_timesteps):
        enabled_fraction = (
            get_enabled_fraction(operator_block_cfg, timestep, total_timesteps)
            if operator_block_cfg is not None
            else 1.0
        )
        if enabled_fraction <= 0.0:
            values[FP_LABEL][timestep] = 1.0
            continue

        payload = None
        if operator in operator_defaults:
            t_bucket = bucket_index(timestep, total_timesteps, timestep_buckets)
            l_bucket = bucket_index(block_idx, total_blocks, layer_buckets)
            payload = bucket_payloads.get((operator, t_bucket, l_bucket))
            if payload is None:
                payload = operator_defaults[operator]

        if payload is not None:
            fractions = [float(x) for x in payload["fractions"]]
            for sl, frac in zip(table_levels, fractions):
                values[sl][timestep] = enabled_fraction * frac
            values[FP_LABEL][timestep] = max(0.0, 1.0 - enabled_fraction)
        elif fixed_fraction is not None:
            for sl, frac in fixed_fraction.items():
                values[sl][timestep] = enabled_fraction * frac
            values[FP_LABEL][timestep] = max(0.0, 1.0 - enabled_fraction)
        else:
            values[FP_LABEL][timestep] = 1.0

    return values


def plot_operator(
    operator: str,
    total_blocks: int,
    total_timesteps: int,
    ncols: int,
    level_labels: list[int | str],
    color_map: dict[int | str, str],
    schedules: dict[int, dict[int | str, np.ndarray]],
    output_dir: Path,
):
    nrows = math.ceil(total_blocks / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols, 1.9 * nrows),
        sharex=True,
        sharey=True,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([[axes]])
    axes = axes.reshape(nrows, ncols)

    x = np.arange(total_timesteps)
    plot_labels = [
        label
        for label in level_labels
        if any(np.any(schedules[block_idx][label] > 0) for block_idx in range(total_blocks))
    ]
    for block_idx in range(total_blocks):
        ax = axes[block_idx // ncols, block_idx % ncols]
        y_values = [schedules[block_idx][label] for label in plot_labels]
        ax.stackplot(
            x,
            *y_values,
            colors=[color_map[label] for label in plot_labels],
            linewidth=0.0,
        )
        ax.set_title(f"block {block_idx}", fontsize=9)
        ax.set_xlim(total_timesteps - 1, 0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", linewidth=0.3, alpha=0.3)

        row = block_idx // ncols
        col = block_idx % ncols
        if row == nrows - 1:
            xticks = np.linspace(total_timesteps - 1, 0, num=min(6, total_timesteps), dtype=int)
            ax.set_xticks(xticks)
            ax.set_xlabel("timestep", fontsize=8)
        if col == 0:
            ax.set_ylabel("fraction", fontsize=8)

    for extra_idx in range(total_blocks, nrows * ncols):
        axes[extra_idx // ncols, extra_idx % ncols].axis("off")

    handles = [
        plt.Line2D([0], [0], color=color_map[label], lw=6)
        for label in plot_labels
    ]
    labels = [f"sl={label}" if label != FP_LABEL else FP_LABEL for label in plot_labels]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=min(len(labels), 6),
        frameon=False,
    )
    fig.suptitle(f"MP Schedule — {operator}", fontsize=14, y=1.06)
    fig.tight_layout()

    output_path = output_dir / f"mp_schedule_{operator}.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    args = parse_args()
    table_path = Path(args.adaptive_mp_table)
    table = load_json(str(table_path))
    sc_config = load_json(args.sc_config) if args.sc_config else None

    total_blocks = args.total_blocks
    if total_blocks is None:
        if sc_config is not None:
            total_blocks = int(sc_config["total_blocks"])
        else:
            total_blocks = 28

    output_dir = Path(args.output_dir) if args.output_dir else table_path.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    operator_defaults, bucket_payloads = build_table_lookup(table)
    operator_filter = parse_operator_set(args.operators)

    operators = set(operator_defaults.keys())
    operators.update(op for op, _, _ in bucket_payloads.keys())
    if sc_config is not None:
        for block_cfg in sc_config.get("blocks", []):
            operators.update(block_cfg.keys())
    operators = sorted(operators)
    if operator_filter is not None:
        operators = [op for op in operators if op in operator_filter]

    level_labels = collect_level_labels(table, sc_config)
    color_map = get_color_map(level_labels)

    print(f"Operators: {operators}")
    print(f"Total blocks: {total_blocks}, total timesteps: {args.total_timesteps}")
    print(f"Levels: {level_labels}")

    for operator in operators:
        schedules = {}
        for block_idx in range(total_blocks):
            schedules[block_idx] = resolve_fraction_schedule(
                operator=operator,
                block_idx=block_idx,
                total_blocks=total_blocks,
                total_timesteps=args.total_timesteps,
                level_labels=level_labels,
                table=table,
                operator_defaults=operator_defaults,
                bucket_payloads=bucket_payloads,
                sc_config=sc_config,
            )
        plot_operator(
            operator=operator,
            total_blocks=total_blocks,
            total_timesteps=args.total_timesteps,
            ncols=args.ncols,
            level_labels=level_labels,
            color_map=color_map,
            schedules=schedules,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
