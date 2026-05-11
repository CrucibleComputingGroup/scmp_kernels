import os, numpy as np
from PIL import Image

D48 = '/scratch/nbleier_owned_root/nbleier_owned1/zhkangqi/scmp_llm/fid_sweep_bitrev/uniform_avg48/samples'
D128 = '/scratch/nbleier_owned_root/nbleier_owned1/zhkangqi/scmp_llm/fid_sweep_bitrev/uniform_avg128/samples'
OUT = '/scratch/nbleier_owned_root/nbleier_owned1/zhkangqi/scmp_llm/fid_sweep_bitrev/compare_avg48_vs_avg128.png'

idxs = [0, 7, 73, 250, 731, 1500, 3000, 5000, 6500, 8888, 9999]
N = len(idxs); H = 256; pad = 8
canvas = np.full((2*H + pad, N*H + (N-1)*pad, 3), 255, dtype=np.uint8)
for i, idx in enumerate(idxs):
    a = np.asarray(Image.open(f'{D48}/{idx:06d}.png').convert('RGB'))
    b = np.asarray(Image.open(f'{D128}/{idx:06d}.png').convert('RGB'))
    x = i*(H+pad)
    canvas[0:H, x:x+H] = a
    canvas[H+pad:2*H+pad, x:x+H] = b
Image.fromarray(canvas).save(OUT)
print(f'wrote {OUT}  shape={canvas.shape}')
print(f'top row = uniform_avg48  bottom row = uniform_avg128  indices={idxs}')

# also print L2/perceptual diff per index
for idx in idxs:
    a = np.asarray(Image.open(f'{D48}/{idx:06d}.png').convert('RGB'), dtype=np.float32)
    b = np.asarray(Image.open(f'{D128}/{idx:06d}.png').convert('RGB'), dtype=np.float32)
    rmse = np.sqrt(((a-b)**2).mean())
    mean_a = a.mean(axis=(0,1))
    mean_b = b.mean(axis=(0,1))
    std_a = a.std(axis=(0,1)).mean()
    std_b = b.std(axis=(0,1)).mean()
    print(f'idx={idx:6d}  rmse={rmse:6.1f}  mean(48)={mean_a.round(0)}  mean(128)={mean_b.round(0)}  std(48)={std_a:5.1f}  std(128)={std_b:5.1f}')
