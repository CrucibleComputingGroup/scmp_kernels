"""Compose a 6×2 grid (rows = stoc_len, cols = counter | random) and report
per-stoc_len pixel difference between SC_OWEN_MODE=counter vs random.

Saves:
  - counter_vs_random_grid.png : side-by-side composite (one big image)
  - counter_vs_random_diff.png : per-stoc_len heatmap of |counter - random|
  - stats printed to stdout: MAE, RMSE, PSNR per stoc_len.
"""
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

BASE = Path("/scratch/nbleier_owned_root/nbleier_owned1/zhkangqi/scmp_llm/owen_e2e")
LEVELS = [16, 32, 48, 64, 96, 128]


def load(mode: str, sl: int) -> np.ndarray:
    p = BASE / f"{mode}_uniform{sl}" / "sample_sc.png"
    img = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
    return img


def annotate(img: np.ndarray, label: str, color=(255, 255, 255)) -> np.ndarray:
    """Add a small label strip at the top of an image (in-place style)."""
    from PIL import ImageDraw, ImageFont
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 28)
    except Exception:
        font = ImageFont.load_default()
    draw.rectangle([(0, 0), (pil.width, 36)], fill=(0, 0, 0))
    draw.text((8, 4), label, fill=color, font=font)
    return np.asarray(pil)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = ((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean()
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(255.0 ** 2 / mse)


def main():
    rows = []
    diffs = []
    print(f"{'stoc_len':>8}  {'MAE':>7}  {'RMSE':>7}  {'PSNR_dB':>8}  {'maxAE':>6}")
    print("-" * 50)
    for sl in LEVELS:
        a = load("counter", sl)
        b = load("random", sl)
        assert a.shape == b.shape, f"shape mismatch at sl={sl}: {a.shape} vs {b.shape}"
        diff = np.abs(a.astype(np.int32) - b.astype(np.int32))
        mae = diff.mean()
        rmse = np.sqrt((diff ** 2).mean())
        max_ae = diff.max()
        ps = psnr(a, b)
        print(f"{sl:>8}  {mae:>7.3f}  {rmse:>7.3f}  {ps:>8.2f}  {max_ae:>6}")

        a_lab = annotate(a, f"counter sl={sl}")
        b_lab = annotate(b, f"random  sl={sl}")
        # Stack horizontally: [counter | random]
        row = np.concatenate([a_lab, b_lab], axis=1)
        rows.append(row)

        # Diff visualization: scale 8x for visibility, clip
        d_vis = np.clip(diff * 8, 0, 255).astype(np.uint8)
        d_lab = annotate(d_vis, f"|counter - random| ×8  sl={sl}",
                         color=(255, 200, 0))
        diffs.append(d_lab)

    grid = np.concatenate(rows, axis=0)
    diff_grid = np.concatenate(diffs, axis=0)
    out_grid = BASE / "counter_vs_random_grid.png"
    out_diff = BASE / "counter_vs_random_diff.png"
    Image.fromarray(grid).save(out_grid)
    Image.fromarray(diff_grid).save(out_diff)
    print()
    print(f"Saved grid:  {out_grid}")
    print(f"Saved diffs: {out_diff}")


if __name__ == "__main__":
    main()
