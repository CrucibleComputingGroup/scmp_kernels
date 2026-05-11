import numpy as np, os
from PIL import Image
from tqdm import tqdm

NPZ = "/scratch/nbleier_owned_root/nbleier_owned1/zhkangqi/scmp_llm/imagenet256_ref/VIRTUAL_imagenet256_labeled.npz"
OUT = "/scratch/nbleier_owned_root/nbleier_owned1/zhkangqi/scmp_llm/imagenet256_ref/images"
os.makedirs(OUT, exist_ok=True)
arr = np.load(NPZ)["arr_0"]
print("shape", arr.shape, "dtype", arr.dtype)
for i in tqdm(range(arr.shape[0])):
    Image.fromarray(arr[i]).save(os.path.join(OUT, f"{i:06d}.png"))
print("done:", len(os.listdir(OUT)))
