"""
Analyze Q/K distribution in DiT model.

This script runs inference on the DiT model and collects statistics
about the Q and K tensors in attention layers across heads, layers, and timesteps.

Usage:
    python scripts/analyze_qk_distribution.py --num-sampling-steps 50
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import torch
from pytorch_lightning import seed_everything
from tqdm import tqdm

from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from utils.download import find_model
from models.models import DiT_models

from qdit.sc_integration.qk_stats_collector import QKStatsCollector


def run_analysis(args):
    """Run inference and collect Q/K statistics."""
    seed_everything(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    print("\nLoading model...")
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)

    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model: {args.model}, Blocks: {len(model.blocks)}")

    # Create diffusion
    diffusion = create_diffusion(str(args.num_sampling_steps))
    print(f"Diffusion steps: {args.num_sampling_steps}")

    # Create statistics collector
    collector = QKStatsCollector(sparsity_threshold=0.01)
    collector.register_hooks(model)

    # Prepare inputs
    print("\nPreparing inputs...")
    class_labels = [207, 360, 387, 974]  # Example classes
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Classifier-free guidance setup
    using_cfg = args.cfg_scale > 1.0
    if using_cfg:
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([1000] * n, device=device)
        y = torch.cat([y, y_null], 0)
        model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
    else:
        model_kwargs = dict(y=y)

    # Run sampling with statistics collection
    print("\nRunning inference and collecting statistics...")
    print(f"This will collect stats for {args.num_sampling_steps} timesteps × {len(model.blocks)} layers")

    indices = list(range(diffusion.num_timesteps))[::-1]
    img = z

    with torch.no_grad():
        for i in tqdm(indices, desc="Sampling"):
            # Set current timestep for collector
            collector.set_timestep(i)

            t = torch.tensor([i] * z.shape[0], device=device)
            out = diffusion.ddim_sample(
                model, img, t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )
            img = out["sample"]

    # Remove hooks
    collector.remove_hooks()

    # Print summary
    collector.print_summary()

    # Save detailed stats
    if args.save_stats:
        output_path = args.save_stats
        collector.save_stats(output_path)

    return collector


def main():
    parser = argparse.ArgumentParser(description="Analyze Q/K distribution in DiT")
    parser.add_argument("--model", type=str, default="DiT-XL/2",
                        choices=list(DiT_models.keys()))
    parser.add_argument("--image-size", type=int, default=256, choices=[256, 512])
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--save-stats", type=str, default=None,
                        help="Path to save detailed statistics JSON")

    args = parser.parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
