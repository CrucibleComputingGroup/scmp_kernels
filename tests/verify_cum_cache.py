"""B1 verification: the chunked-MLP cum_indicator cache must be BIT-IDENTICAL.

Runs the same chunked-MLP SC matmul with the cache on and off (same binary,
switched by SC_CUM_INDICATOR_CACHE) and requires torch.equal -- not allclose.
Bit-identity is the property that lets 30B results stay comparable with the
4B/8B/14B results produced before the cache existed.

Also reports the speedup, which is the whole point of the change: the table was
rebuilt on EVERY sc_matmul call (37,621 times per 2048-token window on
Qwen3-30B-A3B, 983 on 14B), each a ~4.3 MB int16 alloc plus a Triton launch.
"""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scmp_kernels.sc.kernels import (  # noqa: E402
    _sc_matmul_per_row_mlp,
    _cum_indicator_cache,
    clear_rng_cache,
)

# Qwen3-30B-A3B expert-ish shape: 128 rows/expert, moe_intermediate 768.
SHAPES = [
    ("30B expert  (N=128,  D=2048, M=768)", 128, 2048, 768),
    ("14B dense   (N=2048, D=5120, M=2048)", 2048, 5120, 2048),
]
STOC_LENS = [96, 64, 48, 32, 24, 16]
CHUNK_D = 128
SC_PREC = 8


def run_once(a, b, stoc_len, cache_on, clear=True):
    os.environ["SC_CUM_INDICATOR_CACHE"] = "1" if cache_on else "0"
    if clear:
        clear_rng_cache()
    return _sc_matmul_per_row_mlp(
        a, b, mode="bipolar", sc_prec=SC_PREC, chunk_d=CHUNK_D,
        stoc_len=stoc_len)


def main():
    if not torch.cuda.is_available():
        print("FAIL: no CUDA device")
        return 1
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"torch:  {torch.__version__}\n")

    ok = True
    for label, N, D, M in SHAPES:
        torch.manual_seed(0)
        a = torch.randn(N, D, device="cuda")
        b = torch.randn(M, D, device="cuda")
        print(f"--- {label}")
        for stoc_len in STOC_LENS:
            ref = run_once(a, b, stoc_len, cache_on=False)
            got = run_once(a, b, stoc_len, cache_on=True)
            # second cached call exercises the HIT path, not just the fill
            hit = run_once(a, b, stoc_len, cache_on=True, clear=False)
            torch.cuda.synchronize()
            exact = torch.equal(ref, got) and torch.equal(ref, hit)
            if not exact:
                d1 = (ref - got).abs().max().item()
                d2 = (ref - hit).abs().max().item()
                print(f"    sl={stoc_len:3d}  *** NOT BIT-IDENTICAL *** "
                      f"max|d| fill={d1:.3e} hit={d2:.3e}")
                ok = False
            else:
                print(f"    sl={stoc_len:3d}  bit-identical (fill and hit)")

        # timing: repeated calls, which is what the model actually does
        reps = 30
        for cache_on in (False, True):
            run_once(a, b, 64, cache_on)          # warm triton + fill
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(reps):
                run_once(a, b, 64, cache_on, clear=False)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / reps * 1e3
            print(f"    cache={'ON ' if cache_on else 'OFF'}  "
                  f"{dt:8.3f} ms/call")
        print(f"    cache entries held: {len(_cum_indicator_cache)}\n")

    print("RESULT:", "PASS - bit-identical everywhere" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
