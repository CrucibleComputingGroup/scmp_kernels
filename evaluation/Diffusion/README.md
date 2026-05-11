# evaluation/Diffusion

Higher-level evaluation orchestration for the Diffusion application.

## Contents

- `kid.py` — KID (Kernel Inception Distance) scorer.
- `compare_images.py` — pairwise image comparison utility.
- `build_full_mosaic.py` — assemble full sample mosaics for visual inspection.
- `build_sample_grids.py` — build per-class / per-config sample grids.

## See also

- `application/Diffusion/models/evaluations/evaluator.py` — OpenAI-style FID/sFID/IS evaluator (low-level, needs the Inception checkpoint).
- `application/Diffusion/scripts/eval/` — pipeline scripts (`pngs_to_npz.py`, `openai_fid_sfid_is.sh`, `extract_virtual_ref.py`, `clean_fid_kid.py`).
