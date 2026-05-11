# Evaluation scripts: FID, KID, sFID, Inception Score

Two evaluators against the same reference (OpenAI's `VIRTUAL_imagenet256_labeled.npz`, 10k images):

| Tool | Metrics | Backend | Inception checkpoint |
|---|---|---|---|
| `clean_fid_kid.py` | FID, KID | PyTorch (clean-fid) | clean-fid's PyTorch port |
| `openai_fid_sfid_is.sh` | FID, sFID, IS, Precision, Recall | TensorFlow 2.15 | OpenAI's TF Inception graph |

Numbers from the two are **not** directly comparable in absolute terms (different Inception checkpoint, different resize). Within each tool, comparisons across runs are valid.

## One-time setup

```bash
# Reference
mkdir -p $REF_BASE && cd $REF_BASE
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz
python extract_virtual_ref.py VIRTUAL_imagenet256_labeled.npz images   # for clean-fid

# OpenAI Inception graph (for sFID/IS)
cd $REPO/Q-DiT/models/evaluations
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/classify_image_graph_def.pb

# Envs
# qdit (existing)        ->  pip install clean-fid
# tfeval (new, isolated) ->  conda create -n tfeval python=3.11 -y && \
#                            conda activate tfeval && \
#                            pip install "tensorflow[and-cuda]==2.15.*" numpy scipy tqdm requests
```

## Per-run usage

```bash
# clean-fid (FID + KID), ~5–10 min on a single GPU
REF=$REF_BASE/images python clean_fid_kid.py \
    /scratch/.../uniform_avg48/samples \
    /scratch/.../uniform_avg48/fid_kid_clean.json

# OpenAI evaluator (FID + sFID + IS + P/R)
python pngs_to_npz.py /scratch/.../uniform_avg48/samples /scratch/.../uniform_avg48/samples.npz
./openai_fid_sfid_is.sh \
    /scratch/.../uniform_avg48/samples.npz \
    /scratch/.../uniform_avg48/openai_eval.txt \
    $REF_BASE/VIRTUAL_imagenet256_labeled.npz
```

Both expect 256x256 RGB samples named `000000.png` through `<N-1>.png`.
