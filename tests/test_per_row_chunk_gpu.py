"""PER_ROW_LEN path (per-(row, chunk) stream lengths): kernel invariants.

Invariants that must hold if the k_table stack + shared cum + per-row scale
are right:
  1. rung_table filled with a single rung r  ==  uniform sc_matmul(stoc_len=L_r)
     bit-for-bit (same config, same chunk_d, same halve/rng grid).
  2. rung constant per ROW but varying across rows  ==  row-wise gather of the
     uniform results (per-row quantization makes rows independent).
  3. rung varying per (row, chunk) is within the accuracy envelope of the
     shortest uniform length and differs from every uniform result.
  4. malformed inputs raise instead of silently running one length.
"""
import pytest
import torch

from scmp_kernels import sc_matmul
from scmp_kernels.sc.config_helpers import make_sobol_simple_config

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

CD = 128
LENS = [32, 64, 128]
SC_PREC = 8
CFG = make_sobol_simple_config(CD, CD, SC_PREC)


def _inputs(N, D, M, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(N, D, generator=g)
    b = torch.randn(M, D, generator=g)
    return a.cuda(), b.cuda()


def _uniform(a, b, L):
    return sc_matmul(a, b, granularity="per_row", mode="bipolar", sc_prec=SC_PREC,
                     chunk_d=CD, stoc_len=L, halve_bipolar_stoc_len=True, config=CFG)


def _prc(a, b, rt, lens=LENS):
    return sc_matmul(a, b, granularity="per_row", mode="bipolar", sc_prec=SC_PREC,
                     chunk_d=CD, stoc_len=max(lens), halve_bipolar_stoc_len=True,
                     config=CFG, rung_table=rt, level_lens=lens)


@pytest.mark.parametrize("D", [640, 700])   # 5 full chunks / 5 full + 60-wide tail
@pytest.mark.parametrize("r", [0, 1, 2])
def test_constant_rung_equals_uniform(D, r):
    a, b = _inputs(96, D, 80)
    nch = (D + CD - 1) // CD
    rt = torch.full((96, nch), r, dtype=torch.int32, device="cuda")
    out = _prc(a, b, rt)
    ref = _uniform(a, b, LENS[r])
    assert torch.equal(out, ref), \
        f"rung {r} (L={LENS[r]}) differs from uniform: max|d|={(out-ref).abs().max().item()}"


@pytest.mark.parametrize("D", [640, 700])
def test_per_row_rung_equals_rowwise_gather(D):
    a, b = _inputs(96, D, 80, seed=1)
    nch = (D + CD - 1) // CD
    g = torch.Generator().manual_seed(7)
    row_rung = torch.randint(0, len(LENS), (96,), generator=g)
    rt = row_rung[:, None].expand(96, nch).contiguous().to(torch.int32).cuda()
    out = _prc(a, b, rt)
    refs = [_uniform(a, b, L) for L in LENS]
    exp = torch.stack([refs[int(row_rung[i])][i] for i in range(96)])
    assert torch.equal(out, exp)


def test_per_row_chunk_mix_is_sane():
    a, b = _inputs(96, 640, 80, seed=2)
    nch = 5
    g = torch.Generator().manual_seed(11)
    rt = torch.randint(0, len(LENS), (96, nch), generator=g).to(torch.int32).cuda()
    out = _prc(a, b, rt)
    fp = a @ b.T
    err = (out - fp).norm() / fp.norm()
    errs = {L: ((_uniform(a, b, L) - fp).norm() / fp.norm()).item() for L in LENS}
    assert err <= max(errs.values()) * 1.05, (err.item(), errs)
    assert err >= min(errs.values()) * 0.95, (err.item(), errs)
    for L in LENS:
        assert not torch.equal(out, _uniform(a, b, L)), f"mixed table equals uniform L={L}"


def test_bad_inputs_raise():
    a, b = _inputs(16, 640, 8)
    good = torch.zeros(16, 5, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError):                       # wrong shape
        _prc(a, b, torch.zeros(16, 4, dtype=torch.int32, device="cuda"))
    with pytest.raises(ValueError):                       # rung out of range
        _prc(a, b, good + 3)
    with pytest.raises(ValueError):                       # level above stoc_len
        sc_matmul(a, b, granularity="per_row", mode="bipolar", sc_prec=SC_PREC,
                  chunk_d=CD, stoc_len=64, halve_bipolar_stoc_len=True, config=CFG,
                  rung_table=good, level_lens=[32, 64, 128])
    with pytest.raises(ValueError):                       # not the chunked bipolar path
        sc_matmul(a, b, granularity="per_row", mode="bipolar", sc_prec=SC_PREC,
                  chunk_d=0, stoc_len=128, config=CFG, rung_table=good, level_lens=LENS)
    with pytest.raises(ValueError):                       # nesting guard: unscrambled L_max
        sc_matmul(a, b, granularity="per_row", mode="bipolar", sc_prec=SC_PREC,
                  chunk_d=CD, stoc_len=256, rng_levels=256, config=CFG,
                  rung_table=good, level_lens=[64, 128, 256])
