"""Regression tests for features merged from the LLM kernel branch."""

import json
from collections import OrderedDict

import torch

from scmp_kernels import trace
from scmp_kernels import sc_conv2d
from scmp_kernels.mp import (
    AdaptiveMPConfig,
    adaptive_classify_rows,
    compute_row_metric,
)
import scmp_kernels.sc.kernels as kernel_mod
import scmp_kernels.sc.matmul as matmul_mod
import scmp_kernels.sc.conv as conv_mod


def test_row_metrics_and_dispatch_table(tmp_path):
    x = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
    assert torch.equal(compute_row_metric(x, "amax"), torch.tensor([4.0, 2.0]))
    assert torch.equal(compute_row_metric(x, "l2"), torch.tensor([5.0, 2.0]))
    crest = compute_row_metric(x, "crest")
    assert torch.allclose(crest, torch.tensor([0.8, 1.0]))

    table = tmp_path / "mp.json"
    table.write_text(json.dumps({
        "stoc_len_levels": [128, 64],
        "operator_defaults": {"pw": {"thresholds": [0.5]}},
        "dispatch_metrics": {"pw": {"metric": "crest", "sign": -1}},
    }))
    cfg = AdaptiveMPConfig([128, 64], threshold_table_path=str(table))
    assert cfg.get_dispatch_metric("pw") == ("crest", -1.0)
    assert cfg.get_dispatch_metric("missing") == ("amax", 1.0)


def test_lru_helpers_bound_and_refresh_recency(monkeypatch):
    monkeypatch.setattr(kernel_mod, "_ENABLE_TABLE_CACHE_MAX", 2)
    cache = OrderedDict()
    kernel_mod._lru_put(cache, "a", 1)
    kernel_mod._lru_put(cache, "b", 2)
    assert kernel_mod._lru_get(cache, "a") == 1
    kernel_mod._lru_put(cache, "c", 3)
    assert list(cache) == ["a", "c"]
    assert kernel_mod._lru_get(cache, "b") is None


def test_conv_dispatch_consumes_selected_metric(monkeypatch):
    def return_level(a, b, **kwargs):
        return torch.full(
            (a.shape[0], b.shape[0]), float(kwargs["stoc_len"]),
            dtype=torch.float32)

    monkeypatch.setattr(conv_mod, "sc_matmul", return_level)
    # amax ranks rows 0/1 highest; l2 ranks rows 2/3 highest.
    rows = torch.tensor([[5.0, 0.0], [4.9, 0.0],
                         [4.0, 4.0], [3.9, 3.9]])
    x = rows.t().reshape(1, 2, 1, 4)
    weight = torch.ones(1, 2, 1, 1)
    cfg = AdaptiveMPConfig(
        [64, 32], target_fractions=[0.5, 0.5],
        dispatch_metrics={"pw": ("l2", 1.0)},
    )
    out = sc_conv2d(x, weight, mp_config=cfg, mp_operator="pw")
    assert torch.equal(out.flatten(), torch.tensor([32.0, 32.0, 64.0, 64.0]))


def test_group_ladder_and_escape_gate_dispatch(tmp_path, monkeypatch):
    table = tmp_path / "mp_group.json"
    table.write_text(json.dumps({
        "stoc_len_levels": [128, 96, 64],
        "layer_buckets": 2,
        "operator_defaults": {
            "pw": {
                "stoc_len_levels": [120, 80, 48],
                "thresholds": [0.7, 0.3],
                "metric_mean": 0.5,
                "metric_std": 0.1,
            },
        },
        "buckets": {
            "pw:t0:l1": {
                "stoc_len_levels": [112, 72, 48],
                "thresholds": [0.75, 0.25],
                "metric_mean": 0.5,
                "metric_std": 0.1,
            },
        },
    }))
    cfg = AdaptiveMPConfig(
        [128, 96, 64],
        threshold_table_path=str(table),
        escape_gate_k=1.0,
        escape_stoc_len=128,
    )
    assert cfg.get_levels(
        operator="pw", block_idx=1, total_blocks=2) == [112, 72, 48]
    assert cfg.get_levels(operator="pw") == [120, 80, 48]
    assert cfg.classify_level_values(
        operator="pw", block_idx=1, total_blocks=2
    ) == [112, 72, 48, 128]

    metric = torch.tensor([0.0, 0.3, 0.5, 0.7, 1.0])
    assignment = adaptive_classify_rows(
        metric, cfg, operator="pw", block_idx=1, total_blocks=2)
    assert torch.equal(
        assignment.level_row_indices[48], torch.tensor([0]))
    assert torch.equal(
        assignment.level_row_indices[72], torch.tensor([1, 2]))
    assert assignment.level_row_indices[112].numel() == 0
    assert torch.equal(
        assignment.level_row_indices[128], torch.tensor([3, 4]))

    def return_level(a, b, **kwargs):
        return torch.full(
            (a.shape[0], b.shape[0]), float(kwargs["stoc_len"]),
            dtype=torch.float32)

    monkeypatch.setattr(conv_mod, "sc_matmul", return_level)
    x = metric.reshape(1, 1, 1, -1)
    weight = torch.ones(1, 1, 1, 1)
    out = sc_conv2d(
        x,
        weight,
        mp_config=cfg,
        mp_operator="pw",
        mp_block_idx=1,
        mp_total_blocks=2,
    )
    assert torch.equal(
        out.flatten(), torch.tensor([48.0, 72.0, 72.0, 128.0, 128.0]))


def test_feature_aware_dispatch_keeps_legacy_two_arg_logger(monkeypatch):
    def return_level(a, b, **kwargs):
        return torch.zeros((a.shape[0], b.shape[0]), dtype=torch.float32)

    monkeypatch.setattr(conv_mod, "sc_matmul", return_level)
    calls = []
    sc_conv2d(
        torch.randn(1, 2, 1, 4),
        torch.ones(1, 2, 1, 1),
        mp_config=AdaptiveMPConfig(
            [64, 32], target_fractions=[0.5, 0.5]),
        mp_logger=lambda stoc_len, n_rows: calls.append(
            (stoc_len, n_rows)),
    )
    assert {stoc_len for stoc_len, _ in calls} == {64, 32}


def test_sc_matmul_trace_hook_summary(tmp_path, monkeypatch):
    def fake_per_row(a, b, **kwargs):
        return torch.zeros((a.shape[0], b.shape[0]), dtype=torch.float32)

    monkeypatch.setattr(matmul_mod, "_sc_matmul_per_row", fake_per_row)
    out_path = tmp_path / "trace.json"
    trace.reset()
    trace.enable(str(out_path), mode="summary")
    trace.set_context("pw_expand", 3)
    try:
        out = matmul_mod.sc_matmul(
            torch.ones(2, 4), torch.ones(5, 4),
            granularity="per_row", mode="bipolar",
            sc_prec=8, stoc_len=64,
            halve_bipolar_stoc_len=True,
        )
        assert out.shape == (2, 5)
        assert trace.flush() == str(out_path)
    finally:
        trace.disable()
        trace.reset()
        trace.clear_context()

    payload = json.loads(out_path.read_text())
    assert payload["schema"] == "scmp-trace-summary-v1"
    assert len(payload["groups"]) == 1
    group = payload["groups"][0]
    assert (group["op"], group["block"]) == ("pw_expand", 3)
    assert group["stoc_len"] == 64
    assert group["rng_levels"] == 128
    assert group["halve"] is True
    assert group["rows"] == 2
    assert group["macs"] == 40
