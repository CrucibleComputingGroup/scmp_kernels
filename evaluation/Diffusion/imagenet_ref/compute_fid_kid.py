import sys, time, json, os
import numpy as np
import torch
from cleanfid.features import build_feature_extractor
from cleanfid.fid import get_folder_features, frechet_distance, kernel_distance

REF = "/scratch/nbleier_owned_root/nbleier_owned1/zhkangqi/scmp_llm/imagenet256_ref/images"
SAMPLES = sys.argv[1]
OUT_JSON = sys.argv[2]
MODE = sys.argv[3] if len(sys.argv) > 3 else "clean"

assert os.path.isdir(REF) and os.path.isdir(SAMPLES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[run] ref={REF}  samples={SAMPLES}  mode={MODE}  device={device}", flush=True)

t0 = time.time()
feat_model = build_feature_extractor(MODE, device, use_dataparallel=False)

t1 = time.time()
feats_ref = get_folder_features(REF, feat_model, num_workers=4, batch_size=64,
                                device=device, mode=MODE, description="ref ")
print(f"[ref ] {feats_ref.shape}  ({time.time()-t1:.1f}s)", flush=True)

t2 = time.time()
feats_smp = get_folder_features(SAMPLES, feat_model, num_workers=4, batch_size=64,
                                device=device, mode=MODE, description="smp ")
print(f"[smp ] {feats_smp.shape}  ({time.time()-t2:.1f}s)", flush=True)

mu_r, sig_r = feats_ref.mean(0), np.cov(feats_ref, rowvar=False)
mu_s, sig_s = feats_smp.mean(0), np.cov(feats_smp, rowvar=False)

t3 = time.time()
fid_score = frechet_distance(mu_s, sig_s, mu_r, sig_r)
print(f"[fid ] {fid_score:.6f}  ({time.time()-t3:.1f}s)", flush=True)

t4 = time.time()
kid_score = kernel_distance(feats_ref, feats_smp)
print(f"[kid ] {kid_score:.6e}  ({time.time()-t4:.1f}s)", flush=True)

result = {"samples_dir": SAMPLES, "ref_dir": REF, "mode": MODE,
          "n_samples": int(feats_smp.shape[0]), "n_ref": int(feats_ref.shape[0]),
          "fid": float(fid_score), "kid": float(kid_score),
          "total_seconds": time.time() - t0}
with open(OUT_JSON, "w") as f:
    json.dump(result, f, indent=2)
print("[ok] wrote", OUT_JSON)
