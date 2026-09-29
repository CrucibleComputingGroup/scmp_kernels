"""K-band (Phase 3) table schema: per-group stream lengths inside a row.

Row dispatch is unchanged -- one metric, one rung index per row. What changes
is that the residual contraction axis is partitioned into bands of whole
quantization chunks, and band b runs rung k at its own length. Setting every
band's ladder equal to the row ladder reproduces the per-row parent, which is
what makes the refinement unable to lose.

The load-time checks are the safety net for the whole experiment: a malformed
band spec that quietly degraded to per-row would produce a cell that LOOKS like
a Phase-3 result, and realized_flop_avg_sl would not reveal it (the tracker
prices the assignment it is handed rather than measuring the kernel).
"""

import json
import tempfile
import unittest
from pathlib import Path

from scmp_kernels.mp.config import (
    AdaptiveMPConfig,
    residual_chunk_widths,
)

GLOBAL = [96, 64, 48, 32]
# down_proj-shaped: 9728 - 584 protected = 9144 = 71 x 128 + 56 (ragged tail)
R_DOWN = 9144
N_CHUNKS = len(residual_chunk_widths(R_DOWN, 128))          # 72


def _bucket(thresholds):
    return {"thresholds": list(thresholds),
            "metric_mean": 0.5, "metric_std": 0.1}


def _halves():
    """Band map splitting the 72 chunks into two equal-count halves."""
    return [0] * (N_CHUNKS // 2) + [1] * (N_CHUNKS - N_CHUNKS // 2)


def _write(k_bands, buckets=None):
    payload = {
        "stoc_len_levels": list(GLOBAL),
        "buckets": buckets if buckets is not None else {
            f"down_proj:t0:l{i}": _bucket([0.7, 0.5, 0.3]) for i in range(4)},
    }
    if k_bands is not None:
        payload["k_bands"] = k_bands
    path = Path(tempfile.mkdtemp()) / "table.json"
    path.write_text(json.dumps(payload))
    return str(path)


def _cfg(k_bands, buckets=None):
    return AdaptiveMPConfig(
        stoc_len_levels=list(GLOBAL),
        threshold_table_path=_write(k_bands, buckets),
        timestep_buckets=1,
        layer_buckets=4,
    )


def _spec(ladders, bands=None, n_bands=2, n_blocks=4):
    return {
        "n_bands": n_bands,
        "chunk_d": 128,
        "residual_width": {"down_proj": R_DOWN},
        "chunk_bands": {f"down_proj:b{b}": (bands if bands is not None
                                            else _halves())
                        for b in range(n_blocks)},
        "ladders": {f"down_proj:t0:l{i}": ladders for i in range(4)},
    }


class ChunkGeometryTest(unittest.TestCase):

    def test_ragged_tail_is_its_own_chunk_and_comes_last(self):
        w = residual_chunk_widths(R_DOWN, 128)
        self.assertEqual(len(w), 72)
        self.assertEqual(w[:-1], [128] * 71)
        self.assertEqual(w[-1], 56)
        self.assertEqual(sum(w), R_DOWN)

    def test_exact_multiple_has_no_tail(self):
        self.assertEqual(residual_chunk_widths(512, 128), [128] * 4)


class ParentIsInTheSpaceTest(unittest.TestCase):
    """All bands = the row ladder must load, and must be the identity."""

    def test_parent_ladder_in_every_band_loads(self):
        cfg = _cfg(_spec([list(GLOBAL), list(GLOBAL)]))
        self.assertEqual(cfg.k_band_count, 2)
        bands, ladders = cfg.get_k_bands("down_proj", 0, 4)
        self.assertEqual(ladders, [GLOBAL, GLOBAL])
        self.assertEqual(len(bands), N_CHUNKS)

    def test_absent_section_disables_the_path(self):
        cfg = _cfg(None)
        self.assertEqual(cfg.k_band_count, 0)
        self.assertIsNone(cfg.get_k_bands("down_proj", 0, 4))

    def test_operator_without_bands_runs_per_row(self):
        cfg = _cfg(_spec([list(GLOBAL), list(GLOBAL)]))
        self.assertIsNone(cfg.get_k_bands("q_proj", 0, 4))


class IsoCostIdentityTest(unittest.TestCase):
    """Phase 3 REDISTRIBUTES stream length; it must not spend more."""

    def test_overspending_band_is_rejected(self):
        hot = [x + 16 for x in GLOBAL]          # every rung longer, no payback
        with self.assertRaisesRegex(ValueError, "OVERSPENDS"):
            _cfg(_spec([hot, list(GLOBAL)]))

    def test_gross_underspend_is_rejected_as_a_solver_bug(self):
        # Underspending is SAFE for the iso-compute claim, but a solver leaving
        # 8 cycles unspent is broken, not conservative.
        cold = [max(x - 16, 1) for x in GLOBAL]
        with self.assertRaisesRegex(ValueError, "UNDERSPENDS"):
            _cfg(_spec([cold, list(GLOBAL)]))

    def test_small_underspend_is_ALLOWED(self):
        # Discrete water-filling legitimately leaves a fraction of a cycle
        # unspent on a coarse grid. Rejecting that threw away valid
        # allocations (v_proj:t0:l1 realized 47.70 against a parent rung of 48).
        # A cheaper cell that still wins is a STRONGER result, not a failure.
        w0, w1 = 36 * 128, 35 * 128 + 56
        cold, warm = [], []
        for L in GLOBAL:
            # shave ~0.5 cycle off the MAC-weighted mean
            cold.append(L - 1)
            warm.append(L)
        cfg = _cfg(_spec([cold, warm]))
        _, ladders = cfg.get_k_bands("down_proj", 0, 4)
        self.assertEqual(ladders, [cold, warm])
        for k, L in enumerate(GLOBAL):
            realized = (w0 * cold[k] + w1 * warm[k]) / R_DOWN
            self.assertLess(realized, L, "should be under the parent rung")
            self.assertLess(L - realized, cfg.K_BAND_MAX_UNDERSPEND)

    def test_legal_redistribution_within_tolerance_loads(self):
        # widths: band0 = 36*128 = 4608, band1 = 35*128 + 56 = 4536; R = 9144.
        # Move d cycles onto band0 and pay it back from band1 exactly:
        #   4608*(L+d) + 4536*(L-e) = 9144*L   ->  e = d * 4608/4536
        w0, w1 = 36 * 128, 35 * 128 + 56
        self.assertEqual(w0 + w1, R_DOWN)
        hot, cold = [], []
        for L in GLOBAL:
            d = 8
            e = round(d * w0 / w1)
            hot.append(L + d)
            cold.append(L - e)
        cfg = _cfg(_spec([hot, cold]))
        _, ladders = cfg.get_k_bands("down_proj", 0, 4)
        self.assertEqual(ladders, [hot, cold])
        # and the realized MAC-weighted mean really is the parent
        for k, L in enumerate(GLOBAL):
            realized = (w0 * hot[k] + w1 * cold[k]) / R_DOWN
            self.assertLess(abs(realized - L), cfg.K_BAND_ISO_COST_TOL)


class RungIndexResolverTest(unittest.TestCase):
    """Band ladders are indexed by the ROW LADDER's rung index.

    Regression: the runtime originally sized its dispatch loop with
    ``classify_level_values``, which APPENDS the escape length as an extra
    dispatch index when the gate is on and escape_stoc_len is not already a
    rung. Band ladders carry one entry per real rung, so the loop ran one index
    past every band ladder and died with IndexError inside o_proj's forward.
    ``get_levels`` is the correct resolver; the escape slot is handled
    explicitly. These pin the two lists apart so the distinction cannot be
    quietly lost again.
    """

    ESC = 128           # deliberately NOT a rung of GLOBAL

    def _cfg(self, gate=True):
        return AdaptiveMPConfig(
            stoc_len_levels=list(GLOBAL),
            threshold_table_path=_write(_spec([list(GLOBAL), list(GLOBAL)])),
            timestep_buckets=1, layer_buckets=4,
            escape_gate_k=2.0 if gate else None,
            escape_stoc_len=self.ESC,
        )

    def test_classify_level_values_is_longer_than_the_ladder_with_the_gate_on(self):
        cfg = self._cfg(gate=True)
        ladder = cfg.get_levels(operator="down_proj", block_idx=0,
                                total_blocks=36)
        dispatch = cfg.classify_level_values(operator="down_proj", block_idx=0,
                                             total_blocks=36)
        self.assertEqual(len(ladder), len(GLOBAL))
        self.assertEqual(len(dispatch), len(GLOBAL) + 1,
                         "escape entry should be appended for dispatch")
        self.assertEqual(dispatch[-1], self.ESC)

    def test_band_ladders_match_get_levels_not_the_dispatch_map(self):
        cfg = self._cfg(gate=True)
        _, band_ladders = cfg.get_k_bands("down_proj", 0, 36)
        ladder = cfg.get_levels(operator="down_proj", block_idx=0,
                                total_blocks=36)
        for b, lad in enumerate(band_ladders):
            self.assertEqual(
                len(lad), len(ladder),
                f"band {b} must have one entry per REAL rung; sizing the "
                f"dispatch loop off classify_level_values overruns it")

    def test_escape_length_already_a_rung_appends_nothing(self):
        # gate on, escape length IS a rung -> escaped rows fold into it, so
        # dispatch map and ladder have equal length and the band loop's
        # explicit escape slot simply finds no rows
        cfg = AdaptiveMPConfig(
            stoc_len_levels=list(GLOBAL),
            threshold_table_path=_write(_spec([list(GLOBAL), list(GLOBAL)])),
            timestep_buckets=1, layer_buckets=4,
            escape_gate_k=2.0, escape_stoc_len=GLOBAL[0],
        )
        self.assertEqual(
            len(cfg.classify_level_values(operator="down_proj", block_idx=0,
                                          total_blocks=36)),
            len(cfg.get_levels(operator="down_proj", block_idx=0,
                               total_blocks=36)))


class MalformedSpecIsRejectedTest(unittest.TestCase):

    def test_single_band_is_not_a_band_split(self):
        with self.assertRaisesRegex(ValueError, "n_bands must be >= 2"):
            _cfg(_spec([list(GLOBAL)], n_bands=1))

    def test_band_with_one_chunk_falls_off_the_chunked_kernel_path(self):
        lonely = [0] + [1] * (N_CHUNKS - 1)
        with self.assertRaisesRegex(ValueError, r"needs >= 2"):
            _cfg(_spec([list(GLOBAL), list(GLOBAL)], bands=lonely))

    def test_chunk_count_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "entries but the residual has"):
            _cfg(_spec([list(GLOBAL), list(GLOBAL)], bands=[0, 1] * 4))

    def test_rung_count_must_match_the_row_ladder(self):
        short = GLOBAL[:3]
        with self.assertRaisesRegex(ValueError, "rungs but the row ladder has"):
            _cfg(_spec([short, short]))

    def test_band_id_out_of_range_is_rejected(self):
        bad = _halves()
        bad[0] = 5
        with self.assertRaisesRegex(ValueError, "outside"):
            _cfg(_spec([list(GLOBAL), list(GLOBAL)], bands=bad))

    def test_missing_residual_width_is_rejected(self):
        spec = _spec([list(GLOBAL), list(GLOBAL)])
        spec.pop("residual_width")
        with self.assertRaisesRegex(ValueError, "residual_width is required"):
            _cfg(spec)

    def test_ladders_without_chunk_bands_is_rejected(self):
        spec = _spec([list(GLOBAL), list(GLOBAL)])
        spec["chunk_bands"] = {}
        with self.assertRaisesRegex(ValueError, "no matching chunk_bands"):
            _cfg(spec)

    def test_chunk_bands_without_ladders_would_silently_run_per_row(self):
        spec = _spec([list(GLOBAL), list(GLOBAL)])
        spec["ladders"] = {}
        with self.assertRaisesRegex(ValueError, "no ladders"):
            _cfg(spec)

    def test_band_widths_must_agree_across_blocks_of_an_operator(self):
        # ladders are per (op, layer-bucket) but membership is per (op, block);
        # drifting widths mean no single ladder set can hold the identity
        spec = _spec([list(GLOBAL), list(GLOBAL)])
        skewed = [0] * 40 + [1] * (N_CHUNKS - 40)
        spec["chunk_bands"]["down_proj:b2"] = skewed
        with self.assertRaisesRegex(ValueError, "must be constant per operator"):
            _cfg(spec)

    def test_rung_above_the_halved_cap_is_rejected(self):
        # Above 2**(sc_prec-1) the stream WRAPS -- meaningless, not merely
        # worse -- so a band that "won" by overrunning the cap would read as a
        # Phase-3 gain. The allocator's search grid is clamped too; this is the
        # backstop for a hand-written or mis-solved table.
        with self.assertRaisesRegex(ValueError, "exceeds the stream-length cap"):
            _cfg(_spec([[136, 64, 48, 32], [56, 64, 48, 32]]))

    def test_non_positive_rung_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-positive rung"):
            _cfg(_spec([[96, 64, 48, 0], [96, 64, 48, 64]]))


if __name__ == "__main__":
    unittest.main()
