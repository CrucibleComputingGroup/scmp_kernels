# evaluation/

Top-level evaluation harnesses and orchestration scripts, organized per application.

| Subfolder | Status |
|---|---|
| `Diffusion/` | active — FID/KID, mosaic builders, image comparison |
| `ViT/` | placeholder |
| `WorldModel/` | placeholder |

Note: low-level inception-graph FID evaluator (`evaluator.py`, `convert_npz.py`) lives next to its DiT consumer at `application/Diffusion/models/evaluations/` because the calibration / sampling scripts reference that relative path. The 92MB Inception checkpoint (`classify_image_graph_def.pb`) is not included — download it separately and place alongside `evaluator.py`.
