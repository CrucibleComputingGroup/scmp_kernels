import sys, os, numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

src = sys.argv[1]
dst = sys.argv[2]
files = sorted(f for f in os.listdir(src) if f.endswith(".png"))
N = len(files)
print(f"loading {N} from {src}", flush=True)

def load(p):
    return np.asarray(Image.open(os.path.join(src, p)).convert("RGB"), dtype=np.uint8)

arr = np.empty((N, 256, 256, 3), dtype=np.uint8)
with ThreadPoolExecutor(max_workers=16) as ex:
    for i, im in enumerate(tqdm(ex.map(load, files), total=N)):
        arr[i] = im
np.savez(dst, arr_0=arr)
print(f"wrote {dst} shape={arr.shape}", flush=True)
