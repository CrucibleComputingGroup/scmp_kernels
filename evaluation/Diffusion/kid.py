#!/usr/bin/env python3
"""
KID (Kernel Inception Distance) evaluation tool
Computes KID between a folder of generated images and a folder of reference images.
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image


def load_inception_model(device):
    """Load InceptionV3 model for feature extraction."""
    from torchvision.models import inception_v3, Inception_V3_Weights
    weights = Inception_V3_Weights.IMAGENET1K_V1
    # Newer torchvision rejects `aux_logits=False` when loading IMAGENET1K_V1
    # weights (which were trained with aux head). Load with default and disable
    # aux post-hoc; eval-mode forward already returns only the main logits.
    model = inception_v3(weights=weights)
    model.aux_logits = False
    model.AuxLogits = None
    model.fc = nn.Identity()
    model.eval()
    model.to(device)
    return model


class ImageFolderDataset(Dataset):
    EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}

    def __init__(self, folder, transform=None):
        self.paths = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in self.EXTENSIONS
        ])
        if len(self.paths) == 0:
            raise ValueError(f"No images found in folder: {folder}")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img


def extract_features(folder, model, device, batch_size=64):
    """Extract InceptionV3 pool features for all images in a folder."""
    from torchvision.models import Inception_V3_Weights
    transform = Inception_V3_Weights.IMAGENET1K_V1.transforms()
    dataset = ImageFolderDataset(folder, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    all_features = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            features = model(batch)
            all_features.append(features.cpu().numpy())

    return np.concatenate(all_features, axis=0)  # (N, 2048)


def polynomial_mmd(X, Y, degree=3, gamma=None, coef0=1.0):
    """
    Compute MMD^2 with polynomial kernel between feature sets X and Y.
    Uses unbiased estimator.

    Args:
        X: (m, d) numpy array
        Y: (n, d) numpy array

    Returns:
        mmd2: scalar, unbiased MMD^2 estimate
        mmd2_var: scalar, variance of the estimate (for std computation)
    """
    m = X.shape[0]
    n = Y.shape[0]

    if gamma is None:
        gamma = 1.0 / X.shape[1]

    def poly_kernel(A, B):
        return (gamma * A @ B.T + coef0) ** degree

    Kxx = poly_kernel(X, X)
    Kyy = poly_kernel(Y, Y)
    Kxy = poly_kernel(X, Y)

    # Unbiased MMD^2
    np.fill_diagonal(Kxx, 0)
    np.fill_diagonal(Kyy, 0)
    mmd2 = (Kxx.sum() / (m * (m - 1))
            + Kyy.sum() / (n * (n - 1))
            - 2 * Kxy.mean())

    return float(mmd2)


def compute_kid(real_features, gen_features, n_subsets=100, subset_size=1000, degree=3):
    """
    Compute KID as mean and std of MMD^2 over random subsets.

    Args:
        real_features: (N, d) numpy array of real image features
        gen_features:  (M, d) numpy array of generated image features
        n_subsets:     number of random subsets to average over
        subset_size:   size of each subset (capped by min(N, M))
        degree:        polynomial kernel degree

    Returns:
        kid_mean: mean MMD^2
        kid_std:  std of MMD^2 across subsets
    """
    n = min(len(real_features), len(gen_features), subset_size)
    if n < 2:
        raise ValueError("Not enough images to compute KID.")

    mmd2_values = []
    rng = np.random.default_rng(42)

    for _ in range(n_subsets):
        real_idx = rng.choice(len(real_features), size=n, replace=False)
        gen_idx = rng.choice(len(gen_features), size=n, replace=False)
        mmd2 = polynomial_mmd(real_features[real_idx], gen_features[gen_idx], degree=degree)
        mmd2_values.append(mmd2)

    mmd2_values = np.array(mmd2_values)
    return float(mmd2_values.mean()), float(mmd2_values.std())


def main():
    parser = argparse.ArgumentParser(
        description='Compute KID (Kernel Inception Distance) between two image folders',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python kid.py --real /path/to/real_images --gen /path/to/generated_images
  python kid.py --real /data/coco/val --gen /data/generated --batch-size 32 --subsets 50
        """)

    parser.add_argument('--real', type=str, required=True,
                        help='Path to folder of real/reference images')
    parser.add_argument('--gen', type=str, required=True,
                        help='Path to folder of generated images')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for feature extraction (default: 64)')
    parser.add_argument('--subsets', type=int, default=100,
                        help='Number of random subsets for KID estimation (default: 100)')
    parser.add_argument('--subset-size', type=int, default=1000,
                        help='Size of each subset (default: 1000)')
    parser.add_argument('--degree', type=int, default=3,
                        help='Polynomial kernel degree (default: 3)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use: cuda / cpu (default: auto-detect)')

    args = parser.parse_args()

    if not os.path.isdir(args.real):
        print(f"Error: real image folder not found: {args.real}")
        return
    if not os.path.isdir(args.gen):
        print(f"Error: generated image folder not found: {args.gen}")
        return

    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")
    print(f"Loading InceptionV3 model...")
    model = load_inception_model(device)

    print(f"Extracting features from real images: {args.real}")
    real_features = extract_features(args.real, model, device, batch_size=args.batch_size)
    print(f"  -> {len(real_features)} images, feature shape: {real_features.shape}")

    print(f"Extracting features from generated images: {args.gen}")
    gen_features = extract_features(args.gen, model, device, batch_size=args.batch_size)
    print(f"  -> {len(gen_features)} images, feature shape: {gen_features.shape}")

    print(f"\nComputing KID ({args.subsets} subsets, subset size={args.subset_size}, degree={args.degree})...")
    kid_mean, kid_std = compute_kid(
        real_features, gen_features,
        n_subsets=args.subsets,
        subset_size=args.subset_size,
        degree=args.degree,
    )

    print(f"\nResults:")
    print(f"  KID mean: {kid_mean:.6f}")
    print(f"  KID std:  {kid_std:.6f}")
    print(f"  KID x1000: {kid_mean * 1000:.4f} +/- {kid_std * 1000:.4f}")


if __name__ == '__main__':
    main()
