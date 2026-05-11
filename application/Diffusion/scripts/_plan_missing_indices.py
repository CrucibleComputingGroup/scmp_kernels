"""Plan a work-steal partition of missing global sample indices across GPUs.

Scans `samples_dir` for files matching `{0,1,...,num_fid-1:06d}.png` and writes
one index list per GPU to `{out_dir}/gpu_{i}.txt` (one global index per line).

Each GPU's chunk may straddle non-contiguous holes — that's the whole point of
this layout vs the legacy per-GPU contiguous range. The python worker
(`quant_sc_main.py --target_indices_path`) reads the list and produces exactly
those indices, writing each PNG with its global index in the filename so the
samples_dir is single-source-of-truth across runs and GPU counts.

Usage:
    python scripts/_plan_missing_indices.py SAMPLES_DIR NUM_FID NUM_GPUS OUT_DIR [ALIGN]

ALIGN (optional, default 1): each GPU's chunk size is rounded up to a multiple
of ALIGN, with the last GPU taking whatever remainder is left. Set ALIGN to
NUM_FID/NUM_CLASSES (samples-per-class) so partial-run progress stays
class-balanced — every chunk advances complete groups of per-class samples.
"""
from __future__ import annotations

import glob
import math
import os
import sys


def main() -> int:
    if len(sys.argv) not in (5, 6):
        print(__doc__, file=sys.stderr)
        return 2
    samples_dir = sys.argv[1]
    num_fid = int(sys.argv[2])
    num_gpus = int(sys.argv[3])
    out_dir = sys.argv[4]
    align = int(sys.argv[5]) if len(sys.argv) == 6 else 1
    if align < 1:
        raise ValueError(f"align must be >= 1, got {align}")

    os.makedirs(out_dir, exist_ok=True)

    existing: set[int] = set()
    for p in glob.glob(os.path.join(samples_dir, "[0-9]" * 6 + ".png")):
        try:
            existing.add(int(os.path.basename(p)[:6]))
        except ValueError:
            continue
    # Ignore PNGs whose index is outside [0, NUM_FID); they belong to a prior
    # bigger budget and shouldn't count toward the current target.
    existing &= set(range(num_fid))

    missing = sorted(set(range(num_fid)) - existing)
    n_missing = len(missing)
    print(f"[plan] samples_dir={samples_dir}")
    print(f"[plan] target NUM_FID={num_fid}; existing in range={len(existing)}; missing={n_missing}")

    if num_gpus <= 0:
        raise ValueError(f"num_gpus must be >= 1, got {num_gpus}")

    if n_missing:
        per = math.ceil(n_missing / num_gpus / align) * align
    else:
        per = 0
    for i in range(num_gpus):
        chunk = missing[i * per : (i + 1) * per]
        out_path = os.path.join(out_dir, f"gpu_{i}.txt")
        with open(out_path, "w") as f:
            f.write("\n".join(str(x) for x in chunk))
            if chunk:
                f.write("\n")
        first = chunk[0] if chunk else "-"
        last = chunk[-1] if chunk else "-"
        print(f"[plan]   GPU {i}: {len(chunk):>6} indices  (first={first} last={last}) -> {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
