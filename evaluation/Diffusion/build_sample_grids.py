"""Build side-by-side contact sheets of mix96 generated vs real ImageNet val.

Picks 8 visually diverse ImageNet classes; for each, takes 8 samples from each
source. Output: 2 PNGs (mix96_grid.png, real_grid.png), each 8 cols × 8 rows
@ 256 px = 2048×2048 px.
"""
import os, glob, random
from PIL import Image, ImageDraw, ImageFont

MIX96_DIR = "/workspace/Projects/scmp_llm/fid_sweep_bitrev/adaptive_avg96/samples"
REAL_DIR  = "/workspace/imagenet_val_256"
OUT_DIR   = "/workspace/Projects/scmp_llm/evaluation/sample_grids"
os.makedirs(OUT_DIR, exist_ok=True)

# Class index → name (ImageNet 1k). 8 visually distinct picks.
CLASSES = [
    (  0, "tench"),
    (125, "hermit crab"),
    (250, "Siberian husky"),
    (375, "colobus"),
    (500, "cliff dwelling"),
    (625, "lifeboat"),
    (750, "quilt"),
    (875, "trombone"),
]
N_PER_CLASS = 8
TILE = 256
LABEL_H = 24  # leave room for a class-label strip on top of each row

def pick_paths(folder, class_idx, ratio_per_class, n):
    """Filenames have idx//ratio == class_idx."""
    lo = class_idx * ratio_per_class
    hi = lo + ratio_per_class
    candidates = []
    for i in range(lo, hi):
        p = os.path.join(folder, f"{i:06d}.png")
        if os.path.exists(p):
            candidates.append(p)
    random.Random(class_idx).shuffle(candidates)
    return candidates[:n]

def build_grid(pairs, out_path, title):
    cols, rows = N_PER_CLASS, len(CLASSES)
    grid_w = cols * TILE
    grid_h = rows * (TILE + LABEL_H) + LABEL_H  # extra strip for title
    canvas = Image.new("RGB", (grid_w, grid_h), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except OSError:
        font = small = ImageFont.load_default()
    draw.text((10, 4), title, fill=(0, 0, 0), font=font)
    y0 = LABEL_H
    for r, ((cls_idx, cls_name), paths) in enumerate(pairs):
        # row label strip
        ystrip = y0 + r * (TILE + LABEL_H)
        draw.rectangle([(0, ystrip), (grid_w, ystrip + LABEL_H)], fill=(230, 230, 230))
        draw.text((6, ystrip + 4), f"class {cls_idx}: {cls_name}",
                  fill=(0, 0, 0), font=small)
        for c, p in enumerate(paths):
            img = Image.open(p).convert("RGB")
            if img.size != (TILE, TILE):
                img = img.resize((TILE, TILE), Image.BICUBIC)
            canvas.paste(img, (c * TILE, ystrip + LABEL_H))
        # if we have fewer than N_PER_CLASS images, leave gray cells
        for c in range(len(paths), cols):
            draw.rectangle([(c * TILE, ystrip + LABEL_H),
                            ((c+1) * TILE, ystrip + LABEL_H + TILE)],
                           fill=(200, 200, 200))
    canvas.save(out_path, "PNG")
    print(f"wrote {out_path}  ({canvas.size[0]}×{canvas.size[1]})")

def main():
    mix_pairs, real_pairs = [], []
    for cls_idx, cls_name in CLASSES:
        mix_paths  = pick_paths(MIX96_DIR, cls_idx, 10, N_PER_CLASS)
        real_paths = pick_paths(REAL_DIR,  cls_idx, 50, N_PER_CLASS)
        mix_pairs.append(((cls_idx, cls_name), mix_paths))
        real_pairs.append(((cls_idx, cls_name), real_paths))
        print(f"class {cls_idx} ({cls_name}): mix96={len(mix_paths)}, real={len(real_paths)}")

    build_grid(mix_pairs,  os.path.join(OUT_DIR, "mix96_grid.png"),
               "mix96 (adaptive_mp@avg_sl=96, INT8 + SC, bitrev) — generated samples")
    build_grid(real_pairs, os.path.join(OUT_DIR, "imagenet_real_grid.png"),
               "ImageNet val 256×256 — real reference samples (same classes)")

if __name__ == "__main__":
    main()
