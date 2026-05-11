#!/usr/bin/env python3
"""
Image comparison tool - Compare two images and visualize differences
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import os


def load_and_resize_images(img1_path, img2_path):
    """Load two images and resize to same dimensions"""
    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')
    
    # Resize to same size if dimensions differ
    if img1.size != img2.size:
        print(f"Image sizes differ: {img1.size} vs {img2.size}")
        # Use smaller dimensions
        target_size = (min(img1.size[0], img2.size[0]), 
                      min(img1.size[1], img2.size[1]))
        print(f"Resizing to: {target_size}")
        img1 = img1.resize(target_size, Image.LANCZOS)
        img2 = img2.resize(target_size, Image.LANCZOS)
    
    return np.array(img1), np.array(img2)


def compute_difference(img1, img2, method='mse'):
    """Calculate image difference
    
    Args:
        img1, img2: numpy arrays (H, W, 3)
        method: 'mse' (mean squared error), 'mae' (mean absolute error)
    """
    img1_float = img1.astype(np.float32)
    img2_float = img2.astype(np.float32)
    
    if method == 'mse':
        # Mean squared error
        diff = np.mean((img1_float - img2_float) ** 2, axis=2)
    elif method == 'mae':
        # Mean absolute error
        diff = np.mean(np.abs(img1_float - img2_float), axis=2)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return diff


def visualize_comparison(img1, img2, diff_map, output_path, title1="Image 1", title2="Image 2"):
    """Create comparison visualization with error map and PSNR
    
    Args:
        img1, img2: Original images
        diff_map: Error/difference map
        output_path: Output file path
        title1, title2: Image titles
    """
    psnr = calculate_psnr(img1, img2)
    mse = np.mean((img1.astype(float) - img2.astype(float))**2)
    mean_diff = np.mean(diff_map)
    max_diff = np.max(diff_map)
    
    # Create simple visualization: error map with PSNR info
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    im = ax.imshow(diff_map, cmap='viridis')
    ax.set_title(f'Error Map | PSNR: {psnr:.2f} dB | MSE: {mse:.2f}', fontsize=14)
    ax.axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Pixel-wise MSE', rotation=270, labelpad=15)
    
    # Add text with statistics
    stats_text = f'Mean Error: {mean_diff:.2f}\nMax Error: {max_diff:.2f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Comparison saved to: {output_path}")
    plt.close()


def calculate_psnr(img1, img2):
    """Calculate PSNR (Peak Signal-to-Noise Ratio)"""
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def main():
    parser = argparse.ArgumentParser(
        description='Compare two images and visualize differences',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python compare_images.py img1.png img2.png
  python compare_images.py img1.png img2.png -o result.png
  python compare_images.py results/exp1/sample.png results/exp2/sample.png
        """)
    
    parser.add_argument('image1', type=str, help='First image path')
    parser.add_argument('image2', type=str, help='Second image path')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='Output image path (default: error_map.png)')
    parser.add_argument('-m', '--method', type=str, default='mse',
                       choices=['mse', 'mae'],
                       help='Difference calculation method (default: mse)')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.image1):
        print(f"Error: Cannot find image {args.image1}")
        return
    if not os.path.exists(args.image2):
        print(f"Error: Cannot find image {args.image2}")
        return
    
    # Set output path
    if args.output is None:
        args.output = 'error_map.png'
    
    print(f"Comparing images...")
    print(f"  Image 1: {args.image1}")
    print(f"  Image 2: {args.image2}")
    
    # Load images
    img1, img2 = load_and_resize_images(args.image1, args.image2)
    
    # Calculate difference
    diff_map = compute_difference(img1, img2, method=args.method)
    
    # Generate visualization
    title1 = os.path.basename(args.image1)
    title2 = os.path.basename(args.image2)
    visualize_comparison(img1, img2, diff_map, args.output,
                        title1=title1, title2=title2)
    
    # Print statistics
    psnr = calculate_psnr(img1, img2)
    print(f"\nResults:")
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  Mean Error: {np.mean(diff_map):.2f}")
    print(f"  Max Error: {np.max(diff_map):.2f}")


if __name__ == '__main__':
    main()
