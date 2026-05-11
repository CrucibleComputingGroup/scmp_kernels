# evaluation/Diffusion/imagenet_ref

Ad-hoc helpers used while building the ImageNet 256×256 reference set for FID/KID evaluation.

## Contents

- `extract.py` — extract individual PNGs from the `VIRTUAL_imagenet256_labeled.npz` archive.
- `parallel_npz.py` — threaded PNG → npz converter.
- `compute_fid_kid.py` — quick FID + KID using `cleanfid` against a pre-built reference NPZ.
- `compare_grid.py` — side-by-side sample-grid diff for two sweep directories.

## Caveat

These were written in-place during a specific sweep and still contain hardcoded absolute paths under `/scratch/.../scmp_llm/...`. Edit the paths at the top of each file before running.
