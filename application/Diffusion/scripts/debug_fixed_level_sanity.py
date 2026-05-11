import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SC_ROOT = REPO_ROOT / "SC"
if str(SC_ROOT) not in sys.path:
    sys.path.insert(0, str(SC_ROOT))

from config_helpers import make_sobol_simple_config
from sc_triton import (
    sc_matmul_enable_batched_bipolar,
    sc_matmul_enable_triton,
    sc_matmul_grouped_enable_triton,
)


def rel_err(pred: torch.Tensor, target: torch.Tensor) -> float:
    num = (pred - target).pow(2).mean().sqrt()
    den = target.pow(2).mean().sqrt().clamp_min(1e-8)
    return float((num / den).item())


def print_curve(name: str, levels: list[int], fn):
    print(name)
    for stoc_len in levels:
        err, pred_mean, target_mean = fn(stoc_len)
        print(
            f"  sl={stoc_len:>3}  rel_err={err:.6f}  "
            f"pred_abs_mean={pred_mean:.6f}  target_abs_mean={target_mean:.6f}"
        )
    print()


def main():
    assert torch.cuda.is_available(), "CUDA is required"
    torch.manual_seed(0)
    levels = [32, 48, 64, 96, 128]

    def linear_curve(stoc_len: int):
        n, d, m = 32, 1152, 512
        x = torch.randn(n, d, device="cuda", dtype=torch.float32)
        w = torch.randn(m, d, device="cuda", dtype=torch.float32)
        fp = x @ w.t()
        config = make_sobol_simple_config(d, d, 8)
        sc = sc_matmul_enable_triton(
            x,
            w,
            x.max().item(),
            x.min().item(),
            w.max().item(),
            w.min().item(),
            mode="bipolar",
            sc_prec=8,
            config=config,
            stoc_len=stoc_len,
        )
        return rel_err(sc, fp), float(sc.abs().mean().item()), float(fp.abs().mean().item())

    def av_curve(stoc_len: int):
        n, d = 128, 72
        attn = torch.softmax(torch.randn(n, n, device="cuda"), dim=-1)
        v = torch.randn(n, d, device="cuda")
        fp = attn @ v
        config = make_sobol_simple_config(n, n, 8)
        sc = sc_matmul_grouped_enable_triton(
            attn,
            v.t().contiguous(),
            group_a=n,
            group_b=d,
            mode="bipolar",
            sc_prec=8,
            config=config,
            stoc_len=stoc_len,
        )
        return rel_err(sc, fp), float(sc.abs().mean().item()), float(fp.abs().mean().item())

    def qk_curve(stoc_len: int):
        bh, n, d = 2, 128, 72
        q = torch.randn(bh, n, d, device="cuda")
        k = torch.randn(bh, n, d, device="cuda")
        fp = q @ k.transpose(-1, -2)
        qmax = q.amax(dim=(1, 2))
        qmin = q.amin(dim=(1, 2))
        kmax = k.amax(dim=(1, 2))
        kmin = k.amin(dim=(1, 2))
        config = make_sobol_simple_config(d, d, 8)
        sc = sc_matmul_enable_batched_bipolar(
            q,
            k,
            qmax,
            qmin,
            kmax,
            kmin,
            8,
            config,
            stoc_len=stoc_len,
        )
        return rel_err(sc, fp), float(sc.abs().mean().item()), float(fp.abs().mean().item())

    print_curve("linear", levels, linear_curve)
    print_curve("av", levels, av_curve)
    print_curve("qk", levels, qk_curve)


if __name__ == "__main__":
    main()
