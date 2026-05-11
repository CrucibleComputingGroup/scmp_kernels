"""
Compute FID and KID via clean-fid against a folder reference.

Usage:
  python clean_fid_kid.py <samples_dir> <out_json> [mode]
  python clean_fid_kid.py --ref <ref_dir> <samples_dir> <out_json> [mode]

mode: clean | legacy_pytorch | legacy_tensorflow  (default: clean)

Reference folder (default $REF env var) should contain individual PNG/JPG
images (e.g. ImageNet 256 val). Use eval/extract_virtual_ref.py to materialize
images from OpenAI's VIRTUAL_imagenet256_labeled.npz.

Requires: cleanfid (pip install clean-fid), torch, numpy.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from cleanfid.features import build_feature_extractor
from cleanfid.fid import get_folder_features, frechet_distance, kernel_distance


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("samples_dir")
    ap.add_argument("out_json")
    ap.add_argument("mode", nargs="?", default="clean",
                    choices=["clean", "legacy_pytorch", "legacy_tensorflow"])
    ap.add_argument("--ref", default=os.environ.get("REF", ""),
                    help="reference image folder; falls back to $REF env var")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    if not args.ref:
        sys.exit("error: pass --ref <dir> or set $REF")
    assert os.path.isdir(args.ref), f"missing ref: {args.ref}"
    assert os.path.isdir(args.samples_dir), f"missing samples: {args.samples_dir}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run] ref={args.ref}  samples={args.samples_dir}  mode={args.mode}  device={device}",
          flush=True)

    t0 = time.time()
    feat_model = build_feature_extractor(args.mode, device, use_dataparallel=False)

    t1 = time.time()
    feats_ref = get_folder_features(args.ref, feat_model,
                                    num_workers=args.num_workers, batch_size=args.batch_size,
                                    device=device, mode=args.mode, description="ref ")
    print(f"[ref ] {feats_ref.shape}  ({time.time()-t1:.1f}s)", flush=True)

    t2 = time.time()
    feats_smp = get_folder_features(args.samples_dir, feat_model,
                                    num_workers=args.num_workers, batch_size=args.batch_size,
                                    device=device, mode=args.mode, description="smp ")
    print(f"[smp ] {feats_smp.shape}  ({time.time()-t2:.1f}s)", flush=True)

    mu_r, sig_r = feats_ref.mean(0), np.cov(feats_ref, rowvar=False)
    mu_s, sig_s = feats_smp.mean(0), np.cov(feats_smp, rowvar=False)

    t3 = time.time()
    fid_score = frechet_distance(mu_s, sig_s, mu_r, sig_r)
    print(f"[fid ] {fid_score:.6f}  ({time.time()-t3:.1f}s)", flush=True)

    t4 = time.time()
    kid_score = kernel_distance(feats_ref, feats_smp)
    print(f"[kid ] {kid_score:.6e}  ({time.time()-t4:.1f}s)", flush=True)

    result = {"samples_dir": args.samples_dir, "ref_dir": args.ref, "mode": args.mode,
              "n_samples": int(feats_smp.shape[0]), "n_ref": int(feats_ref.shape[0]),
              "fid": float(fid_score), "kid": float(kid_score),
              "total_seconds": time.time() - t0}
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    print("[ok] wrote", args.out_json, flush=True)


if __name__ == "__main__":
    main()
