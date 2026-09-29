"""Throughput vs row count for the chunked-MLP SC path.

Measured B1 result: caching the cum_indicator is bit-identical but worth 0.4%.
The two shapes measured there instead showed a UTILIZATION problem --

    30B expert (N=128,  D=2048, M=768 )  2.01e8 MAC in 2.748 ms = 7.3e10 MAC/s
    14B dense  (N=2048, D=5120, M=2048)  2.15e10 MAC in 53.6 ms = 4.0e11 MAC/s

87x the arithmetic for 19.5x the time.  Qwen3-30B-A3B gives each expert only
2048 tokens * top_k 8 / 128 experts = 128 rows per window, so if throughput is
row-starved then batching windows (B*128 rows per expert) is the real lever and
caching tables never was.

This sweeps N at FIXED D, M to get the actual curve, so the batching speedup is
predicted BEFORE any batching code is written.  It also checks bit-identity of
the per-row outputs across N, which is the property window batching must not
break: rows are independent, so row i's result must not depend on how many
other rows were in the call.
"""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scmp_kernels.sc.kernels import (  # noqa: E402
    _sc_matmul_per_row_mlp,
    clear_rng_cache,
)

D, M = 2048, 768          # Qwen3-30B-A3B expert projection
ROW_COUNTS = [128, 256, 512, 1024, 2048, 4096]
STOC_LEN = 64
CHUNK_D = 128
SC_PREC = 8


def call(a, b):
    return _sc_matmul_per_row_mlp(
        a, b, mode="bipolar", sc_prec=SC_PREC, chunk_d=CHUNK_D,
        stoc_len=STOC_LEN)


def main():
    if not torch.cuda.is_available():
        print("FAIL: no CUDA device")
        return 1
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"D={D} M={M} stoc_len={STOC_LEN} chunk_d={CHUNK_D}\n")
    os.environ["SC_CUM_INDICATOR_CACHE"] = "1"
    clear_rng_cache()

    torch.manual_seed(0)
    b = torch.randn(M, D, device="cuda")
    a_full = torch.randn(max(ROW_COUNTS), D, device="cuda")

    base = None
    print(f"{'N':>6} {'ms/call':>9} {'MAC/s':>11} {'vs N=128':>9} "
          f"{'ms per 128 rows':>16}")
    for n in ROW_COUNTS:
        a = a_full[:n].contiguous()
        call(a, b)                       # warm
        torch.cuda.synchronize()
        reps = 20
        t0 = time.perf_counter()
        for _ in range(reps):
            call(a, b)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / reps * 1e3
        macs = n * D * M / (ms * 1e-3)
        base = base or macs
        per128 = ms / (n / 128)
        print(f"{n:6d} {ms:9.3f} {macs:11.3e} {macs/base:8.2f}x "
              f"{per128:16.3f}")

    # Row independence: row i's output must not depend on batch size.  This is
    # exactly what window batching relies on, and what would silently break if
    # tile selection changed the reduction order.
    print("\nrow-independence across N (bit-exact):")
    a_small = a_full[:128].contiguous()
    ref = call(a_small, b)
    for n in (256, 1024, 4096):
        big = call(a_full[:n].contiguous(), b)
        same = torch.equal(ref, big[:128])
        if not same:
            d = (ref - big[:128]).abs().max().item()
            print(f"  N={n:5d}: *** DIFFERS *** max|d|={d:.3e}")
        else:
            print(f"  N={n:5d}: first 128 rows bit-identical to the N=128 call")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
