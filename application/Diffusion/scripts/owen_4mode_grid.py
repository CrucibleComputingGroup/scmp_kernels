"""Final 4-mode × 6-stoc_len comparison grid + pixel-distance ranking.

Layout: rows = stoc_len, cols = off | counter | random | bitrev

Saves:
  - 4mode_grid.png
  - Per-stoc_len pairwise PSNR table to stdout.
"""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE = Path("/scratch/nbleier_owned_root/nbleier_owned1/zhkangqi/scmp_llm/owen_e2e")
LEVELS = [16, 32, 48, 64, 96, 128]
MODES = ["off", "counter", "random", "bitrev"]


def load(mode, sl):
    return np.asarray(Image.open(BASE / f"{mode}_uniform{sl}" / "sample_sc.png").convert("RGB"), dtype=np.uint8)


def annotate(img, label, color=(255, 255, 255)):
    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    try:
        f = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 28)
    except Exception:
        f = ImageFont.load_default()
    d.rectangle([(0, 0), (pil.width, 36)], fill=(0, 0, 0))
    d.text((8, 4), label, fill=color, font=f)
    return np.asarray(pil)


def psnr(a, b):
    mse = ((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean()
    return 10.0 * np.log10(255.0 ** 2 / mse) if mse > 0 else float("inf")


def main():
    rows = []
    print(f"\n=== Pairwise PSNR (dB) per stoc_len ===")
    print(f"{'sl':>4}  " + "  ".join(
        f"{a:>10}↔{b:<6}" for a in MODES for b in MODES if a < b))
    for sl in LEVELS:
        imgs = {m: load(m, sl) for m in MODES}
        # pairwise PSNR
        pairs = [(a, b) for a in MODES for b in MODES if a < b]
        ps_str = "  ".join(f"{psnr(imgs[a], imgs[b]):>16.2f}" for a, b in pairs)
        print(f"{sl:>4}  {ps_str}")

        labeled = [annotate(imgs[m], f"{m:>7} sl={sl}") for m in MODES]
        rows.append(np.concatenate(labeled, axis=1))

    grid = np.concatenate(rows, axis=0)
    out = BASE / "4mode_grid.png"
    Image.fromarray(grid).save(out)
    print(f"\nSaved: {out}  ({grid.shape[1]}×{grid.shape[0]} px)")


if __name__ == "__main__":
    main()
