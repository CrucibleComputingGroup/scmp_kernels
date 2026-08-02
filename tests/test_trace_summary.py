"""Summary-trace grouping invariants (no GPU, no torch).

The regression these guard: `d_in`/`d_out` used to be excluded from the
summary key so KV growth during decode could not mint a group per step. That
also merged *static* shapes whenever they shared a stoc_len — in particular an
operator's protected-channel slice landing on the same rung as its main slice —
producing groups whose reported dims describe a matmul that was never run.
"""
import importlib.util
import json
import os

# Load trace.py by path: it is pure stdlib, while scmp_kernels/__init__.py
# pulls in torch. Keeps these invariants testable on a login node.
_PATH = os.path.join(os.path.dirname(__file__), "..", "scmp_kernels", "trace.py")
_spec = importlib.util.spec_from_file_location("_scmp_trace_under_test", _PATH)
trace = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(trace)


def _rec(op, block, stoc_len, d_in, d_out, rows=8, unit=None):
    trace.set_context(op, block, unit)
    trace.record_matmul(rows=rows, d_in=d_in, d_out=d_out, batch=1,
                        stoc_len=stoc_len, sc_prec=8, mode="bipolar",
                        granularity="per_row", halve=True, rng_levels=128,
                        chunk_d=128, smoothed=True)


def _flush(tmp_path, name="t.json"):
    out = str(tmp_path / name)
    trace.flush(path=out)
    with open(out) as f:
        return json.load(f)["groups"]


def _fresh(tmp_path):
    trace.reset()
    trace.enable(str(tmp_path / "unused.json"), mode="summary")


def test_pc_slice_sharing_a_rung_stays_a_separate_group(tmp_path):
    """The exact mp_best t96 failure: 9144-wide main + 584-wide protected
    slice, both at stoc_len 128, must NOT merge into one 584-wide group."""
    _fresh(tmp_path)
    for _ in range(3):
        _rec("down_proj", 0, 128, 9144, 2560, rows=100)   # main slice
        _rec("down_proj", 0, 128, 584, 2560, rows=100)    # protected slice
    groups = _flush(tmp_path)

    assert len(groups) == 2, f"expected 2 shape groups, got {len(groups)}"
    assert {g["d_in"] for g in groups} == {9144, 584}
    for g in groups:
        assert g["dims_vary"] is False
        assert g["rows"] * g["d_in"] * g["d_out"] == g["macs"]
    assert sum(g["calls"] for g in groups) == 6


def test_macs_identity_holds_for_every_non_varying_group(tmp_path):
    _fresh(tmp_path)
    for sl in (32, 48, 96, 128):
        _rec("up_proj", 1, sl, 2483, 9728, rows=7)
        _rec("up_proj", 1, sl, 77, 9728, rows=5)
    _rec("qk", 2, 64, 128, 2048, rows=64)
    groups = _flush(tmp_path)

    assert groups, "no groups recorded"
    for g in groups:
        assert not g["dims_vary"]
        assert g["rows"] * g["d_in"] * g["d_out"] == g["macs"], g


def test_kv_growth_is_bounded_by_the_shape_cap(tmp_path):
    """Decode-style growth must not mint an unbounded number of groups."""
    _fresh(tmp_path)
    cap = trace._MAX_SHAPES
    n = cap + 40
    total_macs = 0
    for step in range(n):
        kv = 128 + step
        _rec("av", 3, 64, kv, 128, rows=4)
        total_macs += 4 * kv * 128
    groups = _flush(tmp_path)

    assert len(groups) == cap + 1, (
        f"expected {cap} exact + 1 catch-all, got {len(groups)}")
    varying = [g for g in groups if g["dims_vary"]]
    assert len(varying) == 1
    assert varying[0]["calls"] == n - cap
    # Totals stay exact even where the shape is only representative.
    assert sum(g["macs"] for g in groups) == total_macs
    assert sum(g["calls"] for g in groups) == n
    for g in groups:
        if not g["dims_vary"]:
            assert g["rows"] * g["d_in"] * g["d_out"] == g["macs"]


def test_reset_clears_the_shape_registry(tmp_path):
    _fresh(tmp_path)
    _rec("o_proj", 0, 128, 4055, 2560)
    trace.reset()
    trace.enable(str(tmp_path / "unused.json"), mode="summary")
    _rec("o_proj", 0, 128, 4055, 2560)
    groups = _flush(tmp_path, "t2.json")

    assert len(groups) == 1
    assert groups[0]["calls"] == 1, "reset() leaked state into the next run"
    trace.reset()
    trace.disable()
