"""
Pack a folder of 256x256 PNG samples into the npz format consumed by OpenAI's
guided-diffusion evaluator (key 'arr_0', dtype uint8, shape (N, 256, 256, 3)).

Multi-threaded image decode: ~5 min per 10k images on GPFS, vs. ~hours with
the single-threaded reference Q-DiT/models/evaluations/convert_npz.py.

Usage:
  python pngs_to_npz.py <samples_dir> <out_npz>
"""
import argparse, os
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("samples_dir")
    ap.add_argument("out_npz")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--image-size", type=int, default=256)
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.samples_dir) if f.endswith(".png"))
    n = len(files)
    print(f"[load] {n} pngs from {args.samples_dir}  workers={args.workers}", flush=True)

    def load(p: str) -> np.ndarray:
        return np.asarray(Image.open(os.path.join(args.samples_dir, p)).convert("RGB"),
                          dtype=np.uint8)

    arr = np.empty((n, args.image_size, args.image_size, 3), dtype=np.uint8)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, im in enumerate(tqdm(ex.map(load, files), total=n)):
            arr[i] = im
    np.savez(args.out_npz, arr_0=arr)
    print(f"[ok] wrote {args.out_npz}  shape={arr.shape}", flush=True)


if __name__ == "__main__":
    main()
