"""
Convert grid sample images to npz format for FID evaluation.
Supports both single grid images and folders of individual images.
"""
import argparse
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path


def split_grid_image(image_path, grid_size=None, image_size=256):
    """
    Split a grid image into individual images.

    Args:
        image_path: Path to the grid image
        grid_size: Tuple of (rows, cols), if None will be inferred
        image_size: Size of each individual image (assumes square)

    Returns:
        List of numpy arrays [H, W, 3]
    """
    img = Image.open(image_path).convert('RGB')
    width, height = img.size

    if grid_size is None:
        cols = width // image_size
        rows = height // image_size
    else:
        rows, cols = grid_size

    samples = []
    for row in range(rows):
        for col in range(cols):
            left = col * image_size
            upper = row * image_size
            right = left + image_size
            lower = upper + image_size

            crop = img.crop((left, upper, right, lower))
            samples.append(np.asarray(crop).astype(np.uint8))

    return samples


def load_individual_images(sample_dir, num_samples=None):
    """
    Load individual images from a folder.
    Supports both numbered format (000000.png) and arbitrary names.
    """
    sample_dir = Path(sample_dir)

    # Try numbered format first
    if (sample_dir / "000000.png").exists():
        samples = []
        i = 0
        while True:
            img_path = sample_dir / f"{i:06d}.png"
            if not img_path.exists():
                break
            if num_samples and i >= num_samples:
                break
            sample_pil = Image.open(img_path).convert('RGB')
            samples.append(np.asarray(sample_pil).astype(np.uint8))
            i += 1
        return samples

    # Otherwise load all png/jpg files
    image_files = sorted(list(sample_dir.glob("*.png")) + list(sample_dir.glob("*.jpg")))
    if num_samples:
        image_files = image_files[:num_samples]

    samples = []
    for img_path in tqdm(image_files, desc="Loading images"):
        sample_pil = Image.open(img_path).convert('RGB')
        samples.append(np.asarray(sample_pil).astype(np.uint8))

    return samples


def create_npz(samples, output_path):
    """
    Create npz file from list of image arrays.
    """
    samples = np.stack(samples)
    np.savez(output_path, arr_0=samples)
    print(f"Saved .npz file to {output_path} [shape={samples.shape}]")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert images to npz format for evaluation")
    parser.add_argument("input", type=str, help="Input image path (grid) or folder of images")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output npz path")
    parser.add_argument("--image-size", type=int, default=256, help="Size of individual images")
    parser.add_argument("--grid-rows", type=int, default=None, help="Number of rows in grid")
    parser.add_argument("--grid-cols", type=int, default=None, help="Number of cols in grid")
    parser.add_argument("--num-samples", type=int, default=None, help="Max number of samples to load")
    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_file():
        # Single grid image
        grid_size = None
        if args.grid_rows and args.grid_cols:
            grid_size = (args.grid_rows, args.grid_cols)
        samples = split_grid_image(input_path, grid_size, args.image_size)
        print(f"Split grid image into {len(samples)} samples")

        if args.output is None:
            output_path = input_path.with_suffix('.npz')
        else:
            output_path = args.output

    elif input_path.is_dir():
        # Folder of images
        samples = load_individual_images(input_path, args.num_samples)
        print(f"Loaded {len(samples)} images from folder")

        if args.output is None:
            output_path = str(input_path) + '.npz'
        else:
            output_path = args.output

    else:
        raise ValueError(f"Input path does not exist: {input_path}")

    create_npz(samples, output_path)


if __name__ == "__main__":
    main()
