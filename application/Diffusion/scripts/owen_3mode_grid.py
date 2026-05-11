"""Compose a 6×3 grid (rows = stoc_len, cols = counter | random | bitrev) and
report per-stoc_len pixel diffs of (counter - bitrev) and (random - bitrev).

Saves:
  - 3mode_grid.png : counter | random | bitrev side-by-side, all 6 stoc_len.
  - 3mode_diff_vs_random.png : |counter-random| and |bitrev-random| per sl.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE = Path("/scratch/nbleier_owned_root/nbleier_owned1/zhkangqi/scmp_llm/owen_e2e")
LEVELS = [16, 32, 48, 64, 96, 128]
MODES = ["counter", "random", "bitrev"]


def load(mode: str, sl: int) -> np.ndarray:
    return np.asarray(Image.open(BASE / f"{mode}_uniform{sl}" / "sample_sc.png").convert("RGB"), dtype=np.uint8)


def annotate(img: np.ndarray, label: str, color=(255, 255, 255)) -> np.ndarray:
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 28)
    except Exception:
        font = ImageFont.load_default()
    draw.rectangle([(0, 0), (pil.width, 36)], fill=(0, 0, 0))
    draw.text((8, 4), label, fill=color, font=font)
    return np.asarray(pil)


def stats(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, int]:
    diff = np.abs(a.astype(np.int32) - b.astype(np.int32))
    mae = float(diff.mean())
    rmse = float(np.sqrt((diff ** 2).mean()))
    mse = (diff ** 2).mean()
    psnr = 10.0 * np.log10(255.0 ** 2 / mse) if mse > 0 else float("inf")
    return mae, rmse, psnr, int(diff.max())


def main():
    rows, diff_rows = [], []
    print(f"{'stoc_len':>8}  {'counter-random':>18}  {'bitrev-random':>17}  {'counter-bitrev':>18}")
    print(f"{'':>8}  {'MAE  PSNR':>18}  {'MAE  PSNR':>17}  {'MAE  PSNR':>18}")
    print("-" * 78)
    for sl in LEVELS:
        c = load("counter", sl)
        r = load("random", sl)
        b = load("bitrev", sl)
        cr = stats(c, r)
        br = stats(b, r)
        cb = stats(c, b)
        print(f"{sl:>8}  {cr[0]:>5.2f}  {cr[2]:>5.2f}dB    "
              f"{br[0]:>5.2f}  {br[2]:>5.2f}dB    "
              f"{cb[0]:>5.2f}  {cb[2]:>5.2f}dB")

        c_lab = annotate(c, f"counter sl={sl}")
        r_lab = annotate(r, f"random  sl={sl}")
        b_lab = annotate(b, f"bitrev  sl={sl}")
        rows.append(np.concatenate([c_lab, r_lab, b_lab], axis=1))

        d_cr = np.clip(np.abs(c.astype(np.int32) - r.astype(np.int32)) * 8, 0, 255).astype(np.uint8)
        d_br = np.clip(np.abs(b.astype(np.int32) - r.astype(np.int32)) * 8, 0, 255).astype(np.uint8)
        d_cr_lab = annotate(d_cr, f"|counter-random| ×8  sl={sl}", color=(255, 200, 0))
        d_br_lab = annotate(d_br, f"|bitrev -random| ×8  sl={sl}", color=(255, 200, 0))
        diff_rows.append(np.concatenate([d_cr_lab, d_br_lab], axis=1))

    grid = np.concatenate(rows, axis=0)
    diff_grid = np.concatenate(diff_rows, axis=0)
    out = BASE / "3mode_grid.png"
    out_diff = BASE / "3mode_diff_vs_random.png"
    Image.fromarray(grid).save(out)
    Image.fromarray(diff_grid).save(out_diff)
    print()
    print(f"Saved: {out}")
    print(f"Saved: {out_diff}")


if __name__ == "__main__":
    main()
