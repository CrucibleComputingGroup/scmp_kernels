"""Combine sample images into an A/B comparison grid for Owen-scramble fix.

Rows: stoc_len levels (32, 48, 64, 96, 128)
Cols: [baseline (no Owen), Owen-fixed]
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torchvision.utils as vutils
from PIL import Image

RESULTS = Path("/gpfs/accounts/nbleier_owned_root/nbleier_owned1/zhkangqi/scmp_llm/results")


# (stoc_len, baseline_dir_prefix, owen_dir_prefix)
# baseline pre-Owen were 078/079/080. 081 killed, 082+ are debug runs (batch 4)
# baseline Owen-disabled re-runs go to 095+/096+ once launched
ROWS = [
    (32,  "078", "090"),
    (48,  "079", "091"),
    (64,  "080", "092"),
    (96,  "095", "093"),
    (128, "096", "094"),
]


def find_sample(prefix: str) -> Path | None:
    matches = sorted(RESULTS.glob(f"{prefix}-*/sample_sc.png"))
    return matches[0] if matches else None


def load_img(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = torch.from_numpy(
        bytes_to_array(img)
    )
    return arr


def bytes_to_array(img):
    import numpy as np
    return np.asarray(img).transpose(2, 0, 1).astype("float32") / 255.0


def main():
    panels = []
    labels = []
    for (sl, base_prefix, owen_prefix) in ROWS:
        base_path = find_sample(base_prefix)
        owen_path = find_sample(owen_prefix)
        if base_path is None or owen_path is None:
            print(f"skip sl={sl}: base={base_path} owen={owen_path}")
            continue
        base_img = torch.as_tensor(bytes_to_array(Image.open(base_path).convert("RGB")))
        owen_img = torch.as_tensor(bytes_to_array(Image.open(owen_path).convert("RGB")))
        # Make both same H/W
        H = min(base_img.shape[1], owen_img.shape[1])
        W = min(base_img.shape[2], owen_img.shape[2])
        panels.append(base_img[:, :H, :W])
        panels.append(owen_img[:, :H, :W])
        labels.append(f"sl={sl} baseline")
        labels.append(f"sl={sl} owen")

    grid = vutils.make_grid(torch.stack(panels), nrow=2, padding=10, pad_value=1.0)
    out = RESULTS / "owen_ab_grid.png"
    vutils.save_image(grid, out)
    print(f"Saved grid -> {out}")
    print("Panel order (top to bottom, left then right):")
    for i, l in enumerate(labels):
        print(f"  {i}: {l}")


if __name__ == "__main__":
    main()
