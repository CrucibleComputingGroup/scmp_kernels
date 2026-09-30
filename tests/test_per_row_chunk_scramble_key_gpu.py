"""The (R, D, V) k_table stack must carry the scramble tag in its cache key
like the k_only / enable-table / cum keys (#28).

Sensitivity note: with M=64 bitrev masks and L=64 the per-dim count table is
invariant to the mask (the first 2^6 Sobol points are one-per-length-4 stratum
and bitrev masks never touch bits 0-1), so the check has to run at L=32 and
first prove the mode switch changes the UNIFORM result at all."""
import os
import pytest
import torch

from scmp_kernels import sc_matmul
from scmp_kernels.sc.config_helpers import make_sobol_simple_config

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

CD, LENS, P = 128, [32, 64, 128], 8
CFG = make_sobol_simple_config(CD, CD, P)


def _uniform(a, b, L):
    return sc_matmul(a, b, granularity="per_row", mode="bipolar", sc_prec=P,
                     chunk_d=CD, stoc_len=L, halve_bipolar_stoc_len=True, config=CFG)


def _prc(a, b, rt):
    return sc_matmul(a, b, granularity="per_row", mode="bipolar", sc_prec=P,
                     chunk_d=CD, stoc_len=128, halve_bipolar_stoc_len=True,
                     config=CFG, rung_table=rt, level_lens=LENS)


@pytest.mark.parametrize("new_mode", ["off", "random"])
def test_k_stack_key_tracks_scramble_mode(new_mode):
    g = torch.Generator().manual_seed(3)
    a = torch.randn(64, 640, generator=g).cuda()
    b = torch.randn(48, 640, generator=g).cuda()
    rt = torch.zeros((64, 5), dtype=torch.int32, device="cuda")      # rung 0 = L 32
    old = os.environ.get("SC_OWEN_MODE")
    try:
        os.environ["SC_OWEN_MODE"] = "bitrev"
        u_bitrev = _uniform(a, b, 32)
        assert torch.equal(_prc(a, b, rt), u_bitrev)
        os.environ["SC_OWEN_MODE"] = new_mode          # no clear_rng_cache() on purpose
        u_new = _uniform(a, b, 32)
        assert not torch.equal(u_new, u_bitrev), "mode switch had no effect; test insensitive"
        p_new = _prc(a, b, rt)
        assert torch.equal(p_new, u_new), \
            f"per-row path served a stale k_table stack after SC_OWEN_MODE -> {new_mode}"
    finally:
        if old is None:
            os.environ.pop("SC_OWEN_MODE", None)
        else:
            os.environ["SC_OWEN_MODE"] = old
