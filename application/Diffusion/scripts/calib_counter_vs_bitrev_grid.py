"""Compose:
1. bitrev-only 5-row grid (rows = avg_sl)
2. counter vs bitrev side-by-side at each budget
"""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

COUNTER_BASE = Path("/scratch/nbleier_owned_root/nbleier_owned1/zhkangqi/scmp_llm/calib_sweep_ref256_e2e")
BITREV_BASE = Path("/scratch/nbleier_owned_root/nbleier_owned1/zhkangqi/scmp_llm/calib_sweep_ref256_e2e_bitrev")
TARGETS = [128, 96, 64, 48, 32]


def annotate(img, label, color=(255, 255, 255)):
    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    try:
        f = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 28)
    except Exception:
        f = ImageFont.load_default()
    d.rectangle([(0, 0), (pil.width, 38)], fill=(0, 0, 0))
    d.text((10, 5), label, fill=color, font=f)
    return np.asarray(pil)


def psnr(a, b):
    mse = ((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean()
    return 10.0 * np.log10(255.0 ** 2 / mse) if mse > 0 else float("inf")


def main():
    bitrev_rows = []
    sxs_rows = []
    print(f"\n=== counter ↔ bitrev pairwise PSNR (per budget) ===")
    print(f"{'avg_sl':>8}  {'PSNR (dB)':>10}  {'MAE':>7}")
    for t in TARGETS:
        c = np.asarray(Image.open(COUNTER_BASE / f"avg{t}" / "sample_sc.png").convert("RGB"), dtype=np.uint8)
        b = np.asarray(Image.open(BITREV_BASE / f"avg{t}" / "sample_sc.png").convert("RGB"), dtype=np.uint8)
        ps = psnr(c, b)
        mae = float(np.abs(c.astype(np.int32) - b.astype(np.int32)).mean())
        print(f"{t:>8}  {ps:>10.2f}  {mae:>7.3f}")

        bitrev_rows.append(annotate(b, f"bitrev avg_sl = {t}  (b = {t/256:.4f})"))
        c_lab = annotate(c, f"counter avg_sl = {t}")
        b_lab = annotate(b, f"bitrev  avg_sl = {t}")
        sxs_rows.append(np.concatenate([c_lab, b_lab], axis=1))

    bitrev_grid = np.concatenate(bitrev_rows, axis=0)
    sxs_grid = np.concatenate(sxs_rows, axis=0)
    out_b = BITREV_BASE / "avg_sl_grid.png"
    out_s = BITREV_BASE / "counter_vs_bitrev_grid.png"
    Image.fromarray(bitrev_grid).save(out_b)
    Image.fromarray(sxs_grid).save(out_s)
    print(f"\nSaved:")
    print(f"  {out_b}  ({bitrev_grid.shape[1]}×{bitrev_grid.shape[0]} px)")
    print(f"  {out_s}  ({sxs_grid.shape[1]}×{sxs_grid.shape[0]} px)")


if __name__ == "__main__":
    main()
