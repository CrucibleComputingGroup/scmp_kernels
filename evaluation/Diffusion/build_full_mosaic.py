"""Build a thumbnail mosaic of all currently generated mix96 samples + a manifest CSV.

- Mosaic: 64x64 thumbs in a near-square grid, sorted by global index. Single PNG.
- Manifest: filename, class_idx (class = idx // 10) for every generated image.
"""
import os, csv, glob, math
from PIL import Image

SAMPLE_DIR = "/workspace/Projects/scmp_llm/fid_sweep_bitrev/adaptive_avg96/samples"
OUT_DIR    = "/workspace/Projects/scmp_llm/evaluation/sample_grids"
os.makedirs(OUT_DIR, exist_ok=True)

THUMB = 64

def main():
    paths = sorted(glob.glob(f"{SAMPLE_DIR}/[0-9][0-9][0-9][0-9][0-9][0-9].png"))
    n = len(paths)
    print(f"found {n} samples")

    # write manifest
    manifest = os.path.join(OUT_DIR, f"sample_manifest_{n}.csv")
    with open(manifest, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["filename", "class_idx"])
        for p in paths:
            fn = os.path.basename(p)
            cls = int(fn[:6]) // 10
            w.writerow([fn, cls])
    print(f"wrote manifest: {manifest}")

    # mosaic dimensions: pick cols so layout is roughly square
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    print(f"mosaic: {cols} cols x {rows} rows = {cols*rows} cells (n={n})")
    canvas = Image.new("RGB", (cols * THUMB, rows * THUMB), (32, 32, 32))
    for i, p in enumerate(paths):
        r, c = divmod(i, cols)
        img = Image.open(p).convert("RGB").resize((THUMB, THUMB), Image.BICUBIC)
        canvas.paste(img, (c * THUMB, r * THUMB))
        if (i + 1) % 500 == 0:
            print(f"  ..{i+1}/{n}")
    out_png = os.path.join(OUT_DIR, f"mix96_full_mosaic_{n}.png")
    # PNG with optimization — slower save but smaller file for git
    canvas.save(out_png, format="PNG", optimize=True)
    print(f"wrote mosaic: {out_png}  ({canvas.size[0]}x{canvas.size[1]} px)")

if __name__ == "__main__":
    main()
