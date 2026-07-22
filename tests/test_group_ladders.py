"""Per-(op x layer-bucket) ladders in the calibrated MP table.

The table historically carried ONE ladder shared by all 36 (op x layer)
buckets, with only thresholds varying per bucket.  Measured rung occupancy
(14B t32, ladder [97,64,49,32,24,20]) shows the MLP buckets put ZERO MAC on
rungs 0-1 while qk puts 94-97% on rung 0 and nothing below rung 2, so a shared
ladder gives each population roughly half its resolution.

These tests pin the two properties that matter:
  1. a table WITHOUT per-group ladders is byte-identical to the old behaviour
     (get_levels returns the very same list object), and
  2. a table WITH them routes each bucket to its own ladder.
"""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from scmp_kernels.mp.config import (
    AdaptiveMPConfig,
    _extract_group_levels,
    adaptive_classify_rows,
)

GLOBAL = [96, 64, 48, 32]
ATTN = [112, 96, 80, 64]
MLP = [48, 32, 24, 16]


def _bucket(thresholds, levels=None):
    payload = {"thresholds": list(thresholds),
               "metric_mean": 0.5, "metric_std": 0.1}
    if levels is not None:
        payload["stoc_len_levels"] = list(levels)
    return payload


def _write(payload):
    payload = {"stoc_len_levels": list(GLOBAL), **payload}
    path = Path(tempfile.mkdtemp()) / "table.json"
    path.write_text(json.dumps(payload))
    return str(path)


def _config(path):
    return AdaptiveMPConfig(
        stoc_len_levels=list(GLOBAL),
        threshold_table_path=path,
        timestep_buckets=1,
        layer_buckets=4,
    )


class ExtractGroupLevelsTest(unittest.TestCase):

    def test_absent_returns_none_not_the_global_list(self):
        self.assertIsNone(_extract_group_levels({"thresholds": []}, "k"))

    def test_strictly_descending_required(self):
        # duplicates would silently merge in level_row_indices, which is keyed
        # by the stoc_len VALUE -- rows of the lower rung would run at the
        # higher rung's length with no error raised anywhere
        with self.assertRaisesRegex(ValueError, "strictly"):
            _extract_group_levels({"stoc_len_levels": [64, 32, 32, 16]}, "k")
        with self.assertRaisesRegex(ValueError, "strictly"):
            _extract_group_levels({"stoc_len_levels": [16, 32, 48, 64]}, "k")

    def test_needs_at_least_two_rungs(self):
        with self.assertRaisesRegex(ValueError, ">= 2 rungs"):
            _extract_group_levels({"stoc_len_levels": [64]}, "k")


class NoGroupLaddersIsUnchangedTest(unittest.TestCase):

    def setUp(self):
        self.cfg = _config(_write({
            "buckets": {f"qk:t0:l{i}": _bucket([0.7, 0.5, 0.3])
                        for i in range(4)},
        }))

    def test_get_levels_returns_the_global_list_itself(self):
        got = self.cfg.get_levels(
            operator="qk", block_idx=0, total_blocks=4)
        self.assertIs(got, self.cfg.stoc_len_levels)

    def test_unknown_operator_also_falls_back(self):
        self.assertIs(
            self.cfg.get_levels(operator="nope", block_idx=0, total_blocks=4),
            self.cfg.stoc_len_levels)

    def test_missing_block_context_falls_back(self):
        self.assertIs(self.cfg.get_levels(operator="qk"),
                      self.cfg.stoc_len_levels)


class GroupLaddersTest(unittest.TestCase):

    def setUp(self):
        buckets = {}
        for i in range(4):
            buckets[f"qk:t0:l{i}"] = _bucket([0.7, 0.5, 0.3], ATTN)
            buckets[f"down_proj:t0:l{i}"] = _bucket([0.7, 0.5, 0.3], MLP)
            buckets[f"o_proj:t0:l{i}"] = _bucket([0.7, 0.5, 0.3])  # no group
        self.cfg = _config(_write({"buckets": buckets}))

    def test_each_bucket_gets_its_own_ladder(self):
        self.assertEqual(
            self.cfg.get_levels(operator="qk", block_idx=0, total_blocks=4),
            ATTN)
        self.assertEqual(
            self.cfg.get_levels(operator="down_proj", block_idx=0,
                                total_blocks=4),
            MLP)

    def test_bucket_without_a_group_ladder_still_falls_back(self):
        self.assertIs(
            self.cfg.get_levels(operator="o_proj", block_idx=0,
                                total_blocks=4),
            self.cfg.stoc_len_levels)

    def test_layer_bucket_is_resolved_like_thresholds(self):
        # same bucket arithmetic as get_thresholds: block 3 of 4 -> l3
        for block in range(4):
            self.assertEqual(
                self.cfg.get_levels(operator="qk", block_idx=block,
                                    total_blocks=4),
                ATTN)

    def test_classification_dispatches_on_the_group_ladder(self):
        metric = torch.linspace(0.0, 1.0, 32)
        attn = adaptive_classify_rows(
            metric, self.cfg, operator="qk", block_idx=0, total_blocks=4)
        mlp = adaptive_classify_rows(
            metric, self.cfg, operator="down_proj", block_idx=0,
            total_blocks=4)
        self.assertEqual(sorted(attn.level_row_indices), sorted(ATTN))
        self.assertEqual(sorted(mlp.level_row_indices), sorted(MLP))
        # every row is assigned exactly once, in both groups
        for assignment, levels in ((attn, ATTN), (mlp, MLP)):
            total = sum(v.numel() for v in assignment.level_row_indices.values())
            self.assertEqual(total, metric.numel())
            self.assertEqual(len(levels), len(assignment.level_row_indices))

    def test_empty_row_batch_uses_the_group_ladder(self):
        # MoE experts routinely receive zero tokens in a forward
        out = adaptive_classify_rows(
            torch.empty(0), self.cfg, operator="down_proj", block_idx=0,
            total_blocks=4)
        self.assertEqual(sorted(out.level_row_indices), sorted(MLP))


class GroupLadderRungCountTest(unittest.TestCase):

    def test_group_may_have_a_different_number_of_rungs(self):
        cfg = _config(_write({
            "buckets": {
                # 3 rungs -> 2 thresholds, against a 4-rung global default
                "qk:t0:l0": _bucket([0.6, 0.3], [112, 96, 64]),
            },
        }))
        self.assertEqual(
            cfg.get_levels(operator="qk", block_idx=0, total_blocks=4),
            [112, 96, 64])

    def test_threshold_count_is_validated_against_the_group_not_the_global(self):
        with self.assertRaisesRegex(ValueError, "length 3, expected 2"):
            _config(_write({
                "buckets": {"qk:t0:l0": _bucket([0.7, 0.5, 0.3],
                                                [112, 96, 64])},
            }))


class ClassifyLevelValuesTest(unittest.TestCase):
    """index -> stream-length map MUST resolve the same bucket as the rows.

    The SC attention path classifies rows with adaptive_classify_rows (which
    uses the bucket ladder) and then maps index -> stoc_len via
    classify_level_values.  When the latter ignored the bucket and returned
    the global list, indices stayed in range -- the ladders have equal length
    -- but every attention row ran at the WRONG stream length, with nothing
    downstream cross-checking the two.  It only surfaced as a realized-cost
    reconciliation failure once per-group ladders diverged from the global
    one, i.e. it would have been silent in every prior wave.
    """

    def _cfg(self, grouped):
        buckets = {}
        for i in range(4):
            buckets[f"qk:t0:l{i}"] = _bucket(
                [0.7, 0.5, 0.3], ATTN if grouped else None)
            buckets[f"down_proj:t0:l{i}"] = _bucket(
                [0.7, 0.5, 0.3], MLP if grouped else None)
        return _config(_write({"buckets": buckets}))

    def test_resolves_the_bucket_ladder(self):
        cfg = self._cfg(grouped=True)
        self.assertEqual(
            cfg.classify_level_values(operator="qk", block_idx=0,
                                      total_blocks=4), ATTN)
        self.assertEqual(
            cfg.classify_level_values(operator="down_proj", block_idx=0,
                                      total_blocks=4), MLP)

    def test_matches_what_classification_actually_used(self):
        # the invariant that was violated: dispatch map == classification map
        cfg = self._cfg(grouped=True)
        for op in ("qk", "down_proj"):
            rows = adaptive_classify_rows(
                torch.linspace(0.0, 1.0, 16), cfg, operator=op,
                block_idx=0, total_blocks=4)
            values = cfg.classify_level_values(
                operator=op, block_idx=0, total_blocks=4)
            self.assertEqual(sorted(rows.level_row_indices), sorted(values))

    def test_no_groups_is_unchanged(self):
        cfg = self._cfg(grouped=False)
        self.assertEqual(cfg.classify_level_values(), cfg.stoc_len_levels)
        self.assertEqual(
            cfg.classify_level_values(operator="qk", block_idx=0,
                                      total_blocks=4),
            cfg.stoc_len_levels)

    def test_no_arg_call_still_works(self):
        # existing callers (test_mp_escape_gate) invoke it with no arguments
        self.assertEqual(self._cfg(grouped=True).classify_level_values(),
                         GLOBAL)


if __name__ == "__main__":
    unittest.main()
