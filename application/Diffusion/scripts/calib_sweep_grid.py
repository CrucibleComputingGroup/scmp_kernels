"""Compose 5-row grid of sample_sc.png at avg_sl ∈ {128,96,64,48,32}.

Saves:
  - calib_sweep_ref256_e2e/avg_sl_grid.png : 5 rows side by side
  - prints assignment counts per op per budget.
"""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

E2E_BASE = Path("/scratch/nbleier_owned_root/nbleier_owned1/zhkangqi/scmp_llm/calib_sweep_ref256_e2e")
CALIB_BASE = Path("/scratch/nbleier_owned_root/nbleier_owned1/zhkangqi/scmp_llm/calib_sweep_ref256")
TARGETS = [128, 96, 64, 48, 32]


def annotate(img, label):
    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    try:
        f = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 32)
    except Exception:
        f = ImageFont.load_default()
    d.rectangle([(0, 0), (pil.width, 42)], fill=(0, 0, 0))
    d.text((10, 5), label, fill=(255, 255, 255), font=f)
    return np.asarray(pil)


def main():
    rows = []
    for t in TARGETS:
        p = E2E_BASE / f"avg{t}" / "sample_sc.png"
        img = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
        rows.append(annotate(img, f"avg_sl = {t}  (b = {t/256:.4f})"))

    grid = np.concatenate(rows, axis=0)
    out = E2E_BASE / "avg_sl_grid.png"
    Image.fromarray(grid).save(out)
    print(f"\nSaved grid: {out}  ({grid.shape[1]}×{grid.shape[0]} px)\n")


if __name__ == "__main__":
    main()
