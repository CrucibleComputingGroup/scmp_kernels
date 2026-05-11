"""
Extract OpenAI's VIRTUAL_imagenet256_labeled.npz reference batch into a
folder of PNGs that clean-fid can consume. The npz holds 10000 256x256 images
under key 'arr_0'.

Download once with:
  wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz

Usage:
  python extract_virtual_ref.py <virtual_npz> <out_dir>
"""
import argparse, os
import numpy as np
from PIL import Image
from tqdm import tqdm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("virtual_npz")
    ap.add_argument("out_dir")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    arr = np.load(args.virtual_npz)["arr_0"]
    print(f"[virtual] shape={arr.shape}  dtype={arr.dtype}", flush=True)
    for i in tqdm(range(arr.shape[0])):
        Image.fromarray(arr[i]).save(os.path.join(args.out_dir, f"{i:06d}.png"))
    print(f"[ok] wrote {len(os.listdir(args.out_dir))} images to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
