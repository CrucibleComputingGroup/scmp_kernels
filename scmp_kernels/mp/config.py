"""
Mixed Precision configuration for per-token-row SC.

Each token row gets assigned a stoc_len level based on its importance metric.
Rows with higher importance use longer stoc_len (higher precision), while
less important rows use shorter stoc_len for faster computation.

Includes:
- MPConfig: Fixed-fraction quantile-based assignment (original).
- AdaptiveMPConfig: Timestep-adaptive thresholds with per-operator and
  per-layer control, inspired by HPCA APT's APDT algorithm.
- FreeBoundaryMPConfig: Zero-hyperparameter per-(block, op) free boundaries
  (k-1 for k levels), filled in by an offline oracle search.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch


# ---------------------------------------------------------------------
# Per-block context: classifiers that index by (block, op) read this
# global; the auto-calibrator and runtime pre-hooks set it per forward.
# ---------------------------------------------------------------------
_CURRENT_BLOCK_IDX: int = 0


def set_current_block_idx(i: int) -> None:
    global _CURRENT_BLOCK_IDX
    _CURRENT_BLOCK_IDX = int(i)


def get_current_block_idx() -> int:
    return _CURRENT_BLOCK_IDX


@dataclass
class MPConfig:
    """Configuration for per-token-row mixed precision SC."""
    stoc_len_levels: list[int]                  # e.g. [256, 128, 64, 32], sorted descending
    level_fractions: Optional[list[float]] = None  # e.g. [0.25, 0.25, 0.25, 0.25]; None = equal
    qk_metric: str = "q_row_max"               # "q_row_max" (||Q_row||_inf)
    av_metric: str = "attn_row_max"             # "attn_row_max" (max of attn row)
    mlp_metric: str = "x_row_max"              # "x_row_max" (||x_row||_inf)

    def __post_init__(self):
        if self.level_fractions is None:
            n = len(self.stoc_len_levels)
            self.level_fractions = [1.0 / n] * n
        if len(self.level_fractions) != len(self.stoc_len_levels):
            raise ValueError(
                f"level_fractions length ({len(self.level_fractions)}) must match "
                f"stoc_len_levels length ({len(self.stoc_len_levels)})"
            )
        if abs(sum(self.level_fractions) - 1.0) >= 1e-6:
            raise ValueError(
                f"level_fractions must sum to 1.0, got {sum(self.level_fractions)}"
            )


@dataclass
class RowAssignment:
    """Per-head row-to-level assignment for one (batch, head) pair."""
    row_levels: torch.Tensor                        # [N] int, index into stoc_len_levels
    level_row_indices: dict[int, torch.Tensor]       # stoc_len -> LongTensor of row indices


def _bucket_index(value: int, total: int, num_buckets: int) -> int:
    """Map an absolute timestep / block index to a calibration bucket."""
    if num_buckets <= 1 or total <= 1:
        return 0
    ratio = value / max(total - 1, 1)
    return min(num_buckets - 1, int(ratio * num_buckets))


def _parse_bucket_key(bucket_key: str) -> tuple[str, int, int]:
    """Parse calibration keys like 'proj:t3:l1'."""
    try:
        operator, t_part, l_part = bucket_key.split(":")
        if not t_part.startswith("t") or not l_part.startswith("l"):
            raise ValueError
        return operator, int(t_part[1:]), int(l_part[1:])
    except Exception as exc:  # pragma: no cover - defensive parsing
        raise ValueError(
            f"Invalid adaptive MP bucket key '{bucket_key}'. "
            "Expected format '<operator>:t<int>:l<int>'."
        ) from exc


def _parse_protected_channel_key(key: str) -> tuple[str, int, Optional[int]]:
    """Parse protected-channel keys like 'q_proj:b12' or 'up_proj:b3:u7'."""
    try:
        parts = key.split(":")
        if len(parts) not in (2, 3):
            raise ValueError
        operator, b_part = parts[0], parts[1]
        if not b_part.startswith("b"):
            raise ValueError
        unit = None
        if len(parts) == 3:
            u_part = parts[2]
            if not u_part.startswith("u"):
                raise ValueError
            unit = int(u_part[1:])
        return operator, int(b_part[1:]), unit
    except Exception as exc:  # pragma: no cover - defensive parsing
        raise ValueError(
            f"Invalid protected-channel key '{key}'. "
            "Expected '<operator>:b<int>' or '<operator>:b<int>:u<int>'."
        ) from exc


def _extract_group_levels(payload, source: str) -> Optional[list[int]]:
    """Extract a per-group ladder from a bucket payload, or None if absent.

    Returning None (not the global list) is what keeps a table without
    per-group ladders on the byte-identical path.

    STRICTLY descending is required, not merely non-increasing: RowAssignment
    keys level_row_indices by the stoc_len VALUE, so two rungs sharing a value
    would silently collapse into one entry and the rows of the lower rung
    would be evaluated at the higher rung's length with no error anywhere.
    """
    if not isinstance(payload, dict):
        return None
    raw = payload.get("stoc_len_levels")
    if raw is None:
        return None
    levels = [int(x) for x in raw]
    if len(levels) < 2:
        raise ValueError(
            f"Per-group stoc_len_levels for {source} must have >= 2 rungs, "
            f"got {levels}.")
    for idx in range(1, len(levels)):
        if levels[idx] >= levels[idx - 1]:
            raise ValueError(
                f"Per-group stoc_len_levels for {source} must be strictly "
                f"descending (duplicate or ascending rungs silently merge in "
                f"level_row_indices), got {levels}.")
    if levels[-1] < 0:
        raise ValueError(
            f"Per-group stoc_len_levels for {source} has a negative rung: "
            f"{levels}.")
    return levels


def residual_chunk_widths(residual_width: int, chunk_d: int) -> list[int]:
    """Widths of the residual's quantization chunks, tail last.

    Mirrors the kernel's chunk loop (`for d_start in range(0, D, chunk_d)`,
    sc/kernels.py) so calibration and runtime agree on what a "group" is. The
    last chunk is short whenever chunk_d does not divide the residual width,
    which is the common case once protected channels are carved out (e.g.
    down_proj 9728 - 584 = 9144 -> 71 x 128 + 56).
    """
    if residual_width <= 0 or chunk_d <= 0:
        return []
    full, tail = divmod(residual_width, chunk_d)
    return [chunk_d] * full + ([tail] if tail else [])


def _band_widths(band_of_chunk: list[int], chunk_widths: list[int],
                 n_bands: int) -> list[int]:
    """Total columns per band. Bands need not be contiguous in chunk index."""
    widths = [0] * n_bands
    for chunk_idx, band in enumerate(band_of_chunk):
        widths[band] += chunk_widths[chunk_idx]
    return widths


def _extract_thresholds(payload, n_levels: int, source: str) -> list[float]:
    """Extract a threshold list of length n_levels-1 from a table payload."""
    raw_thresholds = payload.get("thresholds") if isinstance(payload, dict) else payload
    if raw_thresholds is None:
        raise ValueError(f"Missing 'thresholds' in adaptive MP payload for {source}.")
    thresholds = [float(x) for x in raw_thresholds]
    expected = max(n_levels - 1, 0)
    if len(thresholds) != expected:
        raise ValueError(
            f"Adaptive MP thresholds for {source} have length {len(thresholds)}, "
            f"expected {expected} for {n_levels} stoc_len levels."
        )
    for idx, threshold in enumerate(thresholds):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"Adaptive MP threshold {threshold} for {source} is outside [0, 1]."
            )
        if idx > 0 and threshold > thresholds[idx - 1] + 1e-6:
            raise ValueError(
                f"Adaptive MP thresholds for {source} must be non-increasing, "
                f"got {thresholds}."
            )
    return thresholds


_ROW_METRIC_EPS = 1e-12
# Candidate per-row dispatch metrics (act_global_v2 ρ-selection). All are O(D)
# reductions over the last dim — same runtime cost class as the original amax.
ROW_METRIC_NAMES = ("amax", "l2", "crest")


def compute_row_metric(x: torch.Tensor, name: str) -> torch.Tensor:
    """Per-row dispatch metric over the LAST dim of ``x``.

    ``amax``  = ‖row‖_inf (the original metric),
    ``l2``    = ‖row‖_2,
    ``crest`` = ‖row‖_inf / ‖row‖_2 (scale-free peakedness).

    The caller multiplies by the calibrated sign (−1 inverts the ranking; the
    min–max normalization inside ``adaptive_classify_rows`` maps sign-flipped
    values to exactly ``1 − normalized(raw)``, matching the calibration-side
    transform in calibrate_mp_thresholds.py).
    """
    if name == "amax":
        return x.abs().amax(dim=-1)
    if name == "l2":
        return x.float().norm(dim=-1)
    if name == "crest":
        xf = x.float()
        return xf.abs().amax(dim=-1) / (xf.norm(dim=-1) + _ROW_METRIC_EPS)
    raise ValueError(f"unknown dispatch metric '{name}' "
                     f"(expected one of {ROW_METRIC_NAMES})")


def _classify_rows_by_thresholds(
    metric_norm: torch.Tensor,
    stoc_len_levels: list[int],
    thresholds: list[float],
) -> RowAssignment:
    """Assign levels from explicit non-uniform thresholds."""
    n_levels = len(stoc_len_levels)
    expected = max(n_levels - 1, 0)
    if len(thresholds) != expected:
        raise ValueError(
            f"Expected {expected} thresholds for {n_levels} levels, got {len(thresholds)}."
        )

    row_levels = torch.full(
        (metric_norm.shape[0],),
        n_levels - 1,
        dtype=torch.long,
        device=metric_norm.device,
    )
    for level_idx, threshold in enumerate(thresholds):
        lower = metric_norm.new_tensor(threshold)
        if level_idx == 0:
            mask = metric_norm >= lower
        else:
            upper = metric_norm.new_tensor(thresholds[level_idx - 1])
            mask = (metric_norm >= lower) & (metric_norm < upper)
        row_levels[mask] = level_idx

    level_row_indices: dict[int, torch.Tensor] = {}
    for level_idx, stoc_len in enumerate(stoc_len_levels):
        level_row_indices[stoc_len] = torch.where(row_levels == level_idx)[0]

    return RowAssignment(row_levels=row_levels, level_row_indices=level_row_indices)


def classify_rows_by_metric(
    metric: torch.Tensor,
    stoc_len_levels: list[int],
    level_fractions: list[float],
) -> RowAssignment:
    """
    Rank rows by metric, bucket into levels by quantile fractions.

    Top fraction[0] rows -> levels[0] (highest stoc_len)
    Next fraction[1] rows -> levels[1]
    ...

    Args:
        metric: [N] importance values per row
        stoc_len_levels: sorted descending list of stoc_len values
        level_fractions: fraction of rows per level
    """
    N = metric.shape[0]
    sorted_indices = metric.argsort(descending=True)

    row_levels = torch.empty(N, dtype=torch.long, device=metric.device)
    level_row_indices = {}
    offset = 0
    for i, (sl, frac) in enumerate(zip(stoc_len_levels, level_fractions)):
        if i < len(stoc_len_levels) - 1:
            count = round(frac * N)
        else:
            count = N - offset
        # Clamp to remaining rows: cumulative round() on small N + many
        # levels can otherwise drive the final count negative, or have
        # earlier levels overrun N and leave later levels with no rows.
        count = max(0, min(count, N - offset))
        rows = sorted_indices[offset:offset + count]
        row_levels[rows] = i
        level_row_indices[sl] = rows
        offset += count

    return RowAssignment(row_levels=row_levels, level_row_indices=level_row_indices)


# =====================================================================
# Adaptive Mixed Precision (inspired by HPCA APT APDT)
# =====================================================================

@dataclass
class AdaptiveMPConfig:
    """Mixed precision driven by calibrated per-row thresholds.

    Rows are classified by one of three data-driven paths (checked in this
    order by ``adaptive_classify_rows``):

      1. Free-boundary (``FreeBoundaryMPConfig`` subclass): per-(block, op)
         boundaries populated by the offline auto-MP oracle search.
      2. Quantile (``target_fractions`` set): top frac[0] rows -> levels[0],
         etc. — distribution-independent fixed fractions.
      3. Calibrated table (``threshold_table_path`` set): per-(operator,
         timestep_bucket, layer_bucket) thresholds from
         ``calibrate_mp_thresholds.py``.

    There is no closed-form fallback. The earlier ``alpha * progress + beta``
    dynamic-threshold mode was removed — it was unused by every consumer and
    only added a confusing path alongside the calibrated table. A classify
    call that matches none of the three paths is a configuration bug and
    raises.

    Args:
        stoc_len_levels: Descending list of stoc_len values.
            Use 0 as the last level to enable pruning (skip).
        enable_pruning: Allow stoc_len=0 (skip) level.
    """
    stoc_len_levels: list[int]
    enable_pruning: bool = True
    threshold_table_path: Optional[str] = None
    timestep_buckets: int = 1
    layer_buckets: int = 1
    operator_default_thresholds: dict[str, list[float]] = field(default_factory=dict)
    bucket_thresholds: dict[tuple[str, int, int], list[float]] = field(default_factory=dict)
    # Per-(op, timestep-bucket, layer-bucket) LADDER, parallel to
    # bucket_thresholds.  Empty => every bucket uses stoc_len_levels, which is
    # the historical behaviour.  See get_levels().
    bucket_stoc_len_levels: dict[tuple[str, int, int], list[int]] = field(
        default_factory=dict)
    operator_default_stoc_len_levels: dict[str, list[int]] = field(
        default_factory=dict)
    protected_channel_stoc_len: Optional[int] = None
    protected_channel_indices: dict[tuple[str, int, Optional[int]], list[int]] = field(default_factory=dict)
    # ---- K-bands (Phase 3: per-group stream lengths) ----------------------
    # The residual (non-protected) contraction axis is partitioned into
    # ``k_band_count`` bands of WHOLE quantization chunks. Row dispatch is
    # unchanged -- one metric, one rung index k per row -- but band b executes
    # rung k at its own length ``k_band_ladders[(op,t,l)][b][k]``. Setting
    # every band's ladder equal to the bucket ladder reproduces the per-row
    # parent exactly, which is what makes this refinement unable to lose.
    # k_band_count == 0 (default) disables the whole path.
    k_band_count: int = 0
    k_band_chunk_d: int = 128
    # (operator, block_idx) -> band id per residual chunk, ascending chunk order
    k_band_chunks: dict[tuple[str, int], list[int]] = field(default_factory=dict)
    # (operator, t_bucket, l_bucket) -> [n_bands][n_rungs] stream lengths
    k_band_ladders: dict[tuple[str, int, int], list[list[int]]] = field(
        default_factory=dict)
    # operator -> residual width R (constant per op: |protected| is per-op
    # constant even though WHICH channels are protected varies per block)
    k_band_residual_width: dict[str, int] = field(default_factory=dict)
    # act_global_v2 ρ-selected dispatch metric: operator -> (name, sign).
    # Absent operator => ("amax", +1.0), byte-identical to the original
    # dispatch. Populated from the table's "dispatch_metrics" payload.
    dispatch_metrics: dict[str, tuple[str, float]] = field(default_factory=dict)
    # When set, bypass the linear-threshold classifier and use these fractions
    # as quantile targets per level (top frac[0] rows -> levels[0], etc.).
    # Length must match stoc_len_levels; sums to 1.
    target_fractions: Optional[list[float]] = None
    # ---- Absolute escape gate (R7) ----------------------------------------
    # When ``escape_gate_k`` is set, calibrated-table classification adds one
    # extra compare after the per-call min–max normalization: rows whose
    # NORMALIZED metric exceeds t_esc_b = metric_mean_b + k * metric_std_b
    # escape to ``escape_stoc_len``, regardless of the band thresholds.
    # mu_b / sigma_b are the per-bucket ``metric_mean`` / ``metric_std``
    # already stored in the calibration table — statistics of the pooled
    # per-call-normalized SIGNED dispatch metric, i.e. the SAME post-sign
    # normalized space the band thresholds live in (sign −1 metrics were
    # stored as 1 − normalized(raw)), so the compare direction never flips.
    # ``escape_gate_k=None`` (default) disables the gate and is byte-identical
    # to the pre-gate classifier.
    escape_gate_k: Optional[float] = None
    escape_stoc_len: int = 128
    # Precomputed at table load (one float per bucket / operator default).
    # NOT clamped to <= 1.0 on purpose: the normalized metric lives in [0, 1]
    # (max EXACTLY 1.0) and the compare is strict, so a t_esc >= 1.0 simply
    # never fires for that bucket.
    bucket_escape_thresholds: dict[tuple[str, int, int], float] = field(default_factory=dict)
    operator_default_escape_thresholds: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        assert len(self.stoc_len_levels) >= 2, (
            "Need at least 2 levels (high + low or high + skip)")
        for i in range(len(self.stoc_len_levels) - 1):
            assert self.stoc_len_levels[i] > self.stoc_len_levels[i + 1], (
                f"stoc_len_levels must be sorted descending, "
                f"got {self.stoc_len_levels}")
        if not self.enable_pruning and 0 in self.stoc_len_levels:
            self.stoc_len_levels = [s for s in self.stoc_len_levels if s > 0]
        if self.escape_gate_k is not None:
            self.escape_gate_k = float(self.escape_gate_k)
            self.escape_stoc_len = int(self.escape_stoc_len)
            if self.escape_stoc_len <= 0:
                raise ValueError(
                    f"escape_stoc_len must be a positive cycle count, "
                    f"got {self.escape_stoc_len}")
            if not self.threshold_table_path:
                raise ValueError(
                    "escape_gate_k requires a calibrated threshold table "
                    "(threshold_table_path): the gate constants are the "
                    "table's per-bucket metric_mean/metric_std.")
        if self.threshold_table_path:
            self.load_threshold_table(self.threshold_table_path)
        if self.target_fractions is not None:
            assert len(self.target_fractions) == len(self.stoc_len_levels), (
                f"target_fractions length {len(self.target_fractions)} "
                f"must match stoc_len_levels length {len(self.stoc_len_levels)}")
            s = sum(self.target_fractions)
            assert abs(s - 1.0) < 1e-6, (
                f"target_fractions must sum to 1.0, got {s}")

    def load_threshold_table(self, path: str):
        """Load calibrated thresholds exported by calibrate_mp_thresholds.py."""
        table_path = Path(path)
        with open(table_path) as f:
            payload = json.load(f)

        table_levels = [int(x) for x in payload["stoc_len_levels"]]
        if table_levels != self.stoc_len_levels:
            raise ValueError(
                f"Adaptive MP table levels {table_levels} do not match runtime "
                f"levels {self.stoc_len_levels}."
            )

        self.timestep_buckets = int(payload.get("timestep_buckets", 1))
        self.layer_buckets = int(payload.get("layer_buckets", 1))
        self.operator_default_thresholds = {}
        self.bucket_thresholds = {}
        self.operator_default_escape_thresholds = {}
        self.bucket_escape_thresholds = {}
        self.bucket_stoc_len_levels = {}
        self.operator_default_stoc_len_levels = {}
        self.protected_channel_stoc_len = None
        self.protected_channel_indices = {}
        self.k_band_count = 0
        self.k_band_chunks = {}
        self.k_band_ladders = {}
        self.k_band_residual_width = {}

        for operator, operator_payload in payload.get("operator_defaults", {}).items():
            op_levels = _extract_group_levels(
                operator_payload, f"operator_default:{operator}")
            if op_levels is not None:
                self.operator_default_stoc_len_levels[operator] = op_levels
            self.operator_default_thresholds[operator] = _extract_thresholds(
                operator_payload,
                len(op_levels if op_levels is not None
                    else self.stoc_len_levels),
                f"operator_default:{operator}",
            )
            t_esc = self._escape_threshold_from_payload(operator_payload)
            if t_esc is not None:
                self.operator_default_escape_thresholds[operator] = t_esc

        for bucket_key, bucket_payload in payload.get("buckets", {}).items():
            operator, t_bucket, l_bucket = _parse_bucket_key(bucket_key)
            # Per-group ladder, if this table declares one.  Thresholds are
            # then validated against THIS bucket's rung count, not the global
            # one -- a per-group ladder may legitimately have a different
            # number of rungs from the table-level default.
            grp_levels = _extract_group_levels(bucket_payload, bucket_key)
            if grp_levels is not None:
                self.bucket_stoc_len_levels[
                    (operator, t_bucket, l_bucket)] = grp_levels
            self.bucket_thresholds[(operator, t_bucket, l_bucket)] = _extract_thresholds(
                bucket_payload,
                len(grp_levels if grp_levels is not None
                    else self.stoc_len_levels),
                bucket_key,
            )
            t_esc = self._escape_threshold_from_payload(bucket_payload)
            if t_esc is not None:
                self.bucket_escape_thresholds[(operator, t_bucket, l_bucket)] = t_esc

        if self.escape_gate_k is not None and not (
                self.bucket_escape_thresholds
                or self.operator_default_escape_thresholds):
            raise ValueError(
                f"escape_gate_k={self.escape_gate_k} is set but the threshold "
                f"table {path} carries no metric_mean/metric_std in any bucket "
                "or operator default — the gate constants cannot be derived. "
                "Recalibrate with a calibrator that exports metric stats, or "
                "unset escape_gate_k.")

        self._load_k_bands(
            payload.get("k_bands"),
            sc_prec=int(payload.get("sc_prec", 8)),
            halve=bool(payload.get("halve_bipolar_stoc_len", True)))

        protected = payload.get("protected_channels") or {}
        indices = protected.get("indices") or {}
        if indices:
            self.protected_channel_stoc_len = int(protected.get("stoc_len", 128))
            for key, vals in indices.items():
                self.protected_channel_indices[_parse_protected_channel_key(key)] = [
                    int(v) for v in vals
                ]

        self.dispatch_metrics = {}
        for op, spec in (payload.get("dispatch_metrics") or {}).items():
            name = str(spec.get("metric", "amax"))
            if name not in ROW_METRIC_NAMES:
                raise ValueError(
                    f"Adaptive MP table dispatch_metrics[{op}] names unknown "
                    f"metric '{name}' (expected one of {ROW_METRIC_NAMES}).")
            self.dispatch_metrics[op] = (name, float(spec.get("sign", 1.0)))

    def get_thresholds(
        self,
        timestep: int,
        total_timesteps: int,
        operator: Optional[str] = None,
        block_idx: Optional[int] = None,
        total_blocks: Optional[int] = None,
    ) -> Optional[list[float]]:
        """Get calibrated thresholds for one operator/timestep/block bucket."""
        if self.bucket_thresholds and operator and block_idx is not None and total_blocks is not None:
            t_bucket = _bucket_index(timestep, total_timesteps, self.timestep_buckets)
            l_bucket = _bucket_index(block_idx, total_blocks, self.layer_buckets)
            thresholds = self.bucket_thresholds.get((operator, t_bucket, l_bucket))
            if thresholds is not None:
                return thresholds
        if operator and operator in self.operator_default_thresholds:
            return self.operator_default_thresholds[operator]
        return None

    # ---- K-bands ---------------------------------------------------------
    # Tolerance on the per-rung iso-cost identity, in halved cycles, MAC
    # weighted. Band ladders are integers so exact equality is generally
    # unreachable; 0.25 is well inside the smallest effect worth chasing (a
    # t48 budget moves ~1 cycle for a 2% change) and far tighter than the
    # realized-trace band check used downstream.
    K_BAND_ISO_COST_TOL = 0.25
    # Underspend is SAFE for the iso-compute claim (a win at lower cost is
    # strictly stronger), so it gets a looser bound than overspend -- but it is
    # still bounded, because a solver leaving many cycles unspent is a solver
    # bug, not a conservative choice.
    K_BAND_MAX_UNDERSPEND = 2.0

    def _load_k_bands(self, section, sc_prec: int = 8,
                      halve: bool = True) -> None:
        """Parse and VALIDATE the k_bands section (Phase 3).

        Everything here is a hard failure rather than a fallback: a malformed
        band spec that silently degrades to per-row would produce a cell that
        looks like a Phase-3 result but is not one, and realized_flop_avg_sl
        would not reveal it.
        """
        if not section:
            return
        n_bands = int(section.get("n_bands", 0))
        if n_bands < 2:
            raise ValueError(
                f"k_bands.n_bands must be >= 2 (got {n_bands}); omit the "
                f"section entirely to run the per-row parent.")
        chunk_d = int(section.get("chunk_d", 128))
        cap = 2 ** (sc_prec - 1) if halve else 2 ** sc_prec
        residual_width = {str(k): int(v)
                          for k, v in (section.get("residual_width") or {}).items()}
        if not residual_width:
            raise ValueError(
                "k_bands.residual_width is required: band widths price the "
                "iso-cost identity and cannot be inferred at load time.")

        chunk_bands: dict[tuple[str, int], list[int]] = {}
        # band -> width, per operator; must agree across every block of an op
        op_band_widths: dict[str, list[int]] = {}
        for key_str, band_ids in (section.get("chunk_bands") or {}).items():
            operator, block_idx, _ = _parse_protected_channel_key(key_str)
            if operator not in residual_width:
                raise ValueError(
                    f"k_bands.chunk_bands has '{key_str}' but no "
                    f"residual_width['{operator}'].")
            widths = residual_chunk_widths(residual_width[operator], chunk_d)
            bands = [int(b) for b in band_ids]
            if len(bands) != len(widths):
                raise ValueError(
                    f"k_bands.chunk_bands['{key_str}'] has {len(bands)} "
                    f"entries but the residual has {len(widths)} chunks "
                    f"(residual_width={residual_width[operator]}, "
                    f"chunk_d={chunk_d}).")
            if any(b < 0 or b >= n_bands for b in bands):
                raise ValueError(
                    f"k_bands.chunk_bands['{key_str}'] has a band id outside "
                    f"[0, {n_bands}).")
            n_op = max(bands) + 1
            counts = [bands.count(b) for b in range(n_op)]
            if min(counts) < 2:
                # A band of a single chunk is <= chunk_d wide and would fall
                # off the chunked kernel path onto a different quantization
                # implementation (sc/kernels.py gates on `D > chunk_d`).
                raise ValueError(
                    f"k_bands.chunk_bands['{key_str}'] leaves a band with "
                    f"{min(counts)} chunk(s); every band needs >= 2 so its "
                    f"width exceeds chunk_d={chunk_d}.")
            bw = _band_widths(bands, widths, n_op)
            prior = op_band_widths.setdefault(operator, bw)
            if prior != bw:
                # Ladders are per (op, layer-bucket) but membership is per
                # (op, block); if widths drifted across blocks of a bucket the
                # per-rung identity could not hold for all of them at once.
                raise ValueError(
                    f"k_bands.chunk_bands['{key_str}'] gives band widths {bw}, "
                    f"but another block of '{operator}' gives {prior}. Band "
                    f"widths must be constant per operator so one ladder set "
                    f"can satisfy the iso-cost identity for every block.")
            chunk_bands[(operator, block_idx)] = bands

        ladders: dict[tuple[str, int, int], list[list[int]]] = {}
        for bucket_key, per_band in (section.get("ladders") or {}).items():
            operator, t_bucket, l_bucket = _parse_bucket_key(bucket_key)
            # Band count is PER OPERATOR. `n_bands` is the MAXIMUM: an operator
            # whose residual has too few chunks to give every band >= 2 uses
            # fewer. Requiring exactly n_bands everywhere silently dropped every
            # narrow projection (~19 chunks) out of the K-band path whenever
            # n_bands was raised for down_proj (71 chunks), which is what made
            # `--n-bands 16` cover 4 of 28 buckets.
            n_op = len(per_band)
            if n_op < 2 or n_op > n_bands:
                raise ValueError(
                    f"k_bands.ladders['{bucket_key}'] has {n_op} ladders; "
                    f"expected between 2 and n_bands={n_bands}.")
            op_bands = {b for (op, _), bm in chunk_bands.items() if op == operator
                        for b in bm}
            if op_bands and max(op_bands) + 1 != n_op:
                raise ValueError(
                    f"k_bands.ladders['{bucket_key}'] has {n_op} ladders but "
                    f"'{operator}' chunk_bands use {max(op_bands) + 1} bands.")
            parent = self.get_levels(
                operator=operator,
                block_idx=l_bucket, total_blocks=self.layer_buckets)
            band_ladders = []
            for b, rungs in enumerate(per_band):
                rungs = [int(x) for x in rungs]
                if len(rungs) != len(parent):
                    raise ValueError(
                        f"k_bands.ladders['{bucket_key}'] band {b} has "
                        f"{len(rungs)} rungs but the row ladder has "
                        f"{len(parent)}; the row's rung index indexes EVERY "
                        f"band ladder, so they must agree.")
                if any(r <= 0 for r in rungs):
                    raise ValueError(
                        f"k_bands.ladders['{bucket_key}'] band {b} has a "
                        f"non-positive rung: {rungs}.")
                # Above the halved cap the stream WRAPS -- the result is
                # meaningless rather than merely worse -- and a band that
                # "wins" by overrunning it would look like a Phase-3 gain.
                if any(r > cap for r in rungs):
                    raise ValueError(
                        f"k_bands.ladders['{bucket_key}'] band {b} exceeds the "
                        f"stream-length cap {cap}: {rungs}.")
                band_ladders.append(rungs)
            widths = op_band_widths.get(operator)
            if widths is None:
                raise ValueError(
                    f"k_bands.ladders['{bucket_key}'] has no matching "
                    f"chunk_bands entry for operator '{operator}'.")
            if len(widths) != n_op:
                raise ValueError(
                    f"k_bands.ladders['{bucket_key}'] has {n_op} ladders but "
                    f"'{operator}' has {len(widths)} band widths.")
            total = float(sum(widths))
            for k, parent_len in enumerate(parent):
                # per-operator band count, NOT the global maximum
                realized = sum(widths[b] * band_ladders[b][k]
                               for b in range(n_op)) / total
                # ONE-SIDED. Overspending breaks the iso-compute claim and is a
                # hard error. UNDERspending cannot: it only means the cell used
                # less compute than the parent, so a quality win there is
                # strictly stronger, not weaker. A two-sided check was correct
                # only while the solver was two-sided; discrete water-filling
                # legitimately leaves a fraction of a cycle unspent when the
                # measurement grid is coarse, and rejecting that threw away
                # valid allocations (v_proj:t0:l1 at 47.70 vs parent 48).
                if realized - parent_len > self.K_BAND_ISO_COST_TOL:
                    raise ValueError(
                        f"k_bands.ladders['{bucket_key}'] OVERSPENDS at rung "
                        f"{k}: MAC-weighted band mean {realized:.4f} vs parent "
                        f"rung {parent_len} (tolerance "
                        f"{self.K_BAND_ISO_COST_TOL}). Phase 3 redistributes "
                        f"stream length, it does not spend more.")
                if parent_len - realized > self.K_BAND_MAX_UNDERSPEND:
                    raise ValueError(
                        f"k_bands.ladders['{bucket_key}'] UNDERSPENDS at rung "
                        f"{k} by {parent_len - realized:.4f} cycles (limit "
                        f"{self.K_BAND_MAX_UNDERSPEND}). That is safe for the "
                        f"iso-compute claim but means the solver is leaving "
                        f"budget on the table — investigate the grid, do not "
                        f"just widen this.")
            ladders[(operator, t_bucket, l_bucket)] = band_ladders

        missing = {op for (op, _) in chunk_bands} - {op for (op, _, _) in ladders}
        if missing:
            raise ValueError(
                f"k_bands: operators {sorted(missing)} have chunk_bands but no "
                f"ladders; they would silently run per-row.")

        self.k_band_count = n_bands
        self.k_band_chunk_d = chunk_d
        self.k_band_chunks = chunk_bands
        self.k_band_ladders = ladders
        self.k_band_residual_width = residual_width

    def get_k_bands(
        self,
        operator: Optional[str],
        block_idx: Optional[int],
        total_blocks: Optional[int],
        *,
        timestep: int = 0,
        total_timesteps: int = 1,
    ):
        """(band_of_chunk, band_ladders) for one module, or None to run per-row.

        ``band_of_chunk[c]`` is the band owning residual chunk ``c`` (ascending
        chunk order, tail last); ``band_ladders[b][k]`` is the stream length
        band ``b`` runs when the row landed on rung ``k``.
        """
        if not self.k_band_count or operator is None or block_idx is None:
            return None
        bands = self.k_band_chunks.get((operator, int(block_idx)))
        if bands is None:
            return None
        t_bucket = _bucket_index(timestep, total_timesteps, self.timestep_buckets)
        l_bucket = _bucket_index(block_idx, total_blocks, self.layer_buckets) \
            if total_blocks is not None else 0
        ladders = self.k_band_ladders.get((operator, t_bucket, l_bucket))
        if ladders is None:
            return None
        return bands, ladders

    def get_levels(
        self,
        timestep: int = 0,
        total_timesteps: int = 1,
        operator: Optional[str] = None,
        block_idx: Optional[int] = None,
        total_blocks: Optional[int] = None,
    ) -> list[int]:
        """Per-(operator, layer-bucket) ladder, falling back to the global one.

        The calibrated table historically carried ONE ladder shared by every
        one of its 36 (op x layer-quartile) buckets, with only the thresholds
        varying per bucket.  Measured occupancy shows that wastes roughly half
        the rungs: on 14B t32 the MLP buckets put ZERO MAC on rungs 0-1 (97,
        64) while qk puts 94-97% on rung 0 and nothing below rung 2 -- the two
        populations live at opposite ends of a shared ladder, so each gets
        about half the available resolution.

        When ``bucket_stoc_len_levels`` is empty this returns the global list
        UNCHANGED (identity is preserved, not just equality), so a table
        without per-group ladders behaves exactly as before.

        Bucket resolution deliberately mirrors ``get_thresholds`` so a bucket's
        ladder and its thresholds can never disagree about which bucket it is.
        """
        if (self.bucket_stoc_len_levels and operator
                and block_idx is not None and total_blocks is not None):
            t_bucket = _bucket_index(timestep, total_timesteps,
                                     self.timestep_buckets)
            l_bucket = _bucket_index(block_idx, total_blocks,
                                     self.layer_buckets)
            levels = self.bucket_stoc_len_levels.get(
                (operator, t_bucket, l_bucket))
            if levels is not None:
                return levels
        if operator and operator in self.operator_default_stoc_len_levels:
            return self.operator_default_stoc_len_levels[operator]
        return self.stoc_len_levels

    def _escape_threshold_from_payload(self, bucket_payload) -> Optional[float]:
        """t_esc = metric_mean + k * metric_std for one table payload.

        Returns None when the gate is off or the payload has no stats.
        Deliberately NOT clamped to <= 1.0 — a t_esc >= 1.0 never fires
        because the normalized metric support is [0, 1]."""
        if self.escape_gate_k is None or not isinstance(bucket_payload, dict):
            return None
        mu = bucket_payload.get("metric_mean")
        sigma = bucket_payload.get("metric_std")
        if mu is None or sigma is None:
            return None
        return float(mu) + float(self.escape_gate_k) * float(sigma)

    def get_escape_threshold(
        self,
        timestep: int,
        total_timesteps: int,
        operator: Optional[str] = None,
        block_idx: Optional[int] = None,
        total_blocks: Optional[int] = None,
    ) -> Optional[float]:
        """Escape-gate threshold for one operator/timestep/block bucket.

        Mirrors :meth:`get_thresholds` lookup order (bucket, then operator
        default). None = the gate cannot fire for this call (gate off, or no
        stats for this bucket)."""
        if self.escape_gate_k is None:
            return None
        if (self.bucket_escape_thresholds and operator and block_idx is not None
                and total_blocks is not None):
            t_bucket = _bucket_index(timestep, total_timesteps, self.timestep_buckets)
            l_bucket = _bucket_index(block_idx, total_blocks, self.layer_buckets)
            t_esc = self.bucket_escape_thresholds.get((operator, t_bucket, l_bucket))
            if t_esc is not None:
                return t_esc
        if operator and operator in self.operator_default_escape_thresholds:
            return self.operator_default_escape_thresholds[operator]
        return None

    def classify_level_values(
        self,
        timestep: int = 0,
        total_timesteps: int = 1,
        operator: Optional[str] = None,
        block_idx: Optional[int] = None,
        total_blocks: Optional[int] = None,
    ) -> list[int]:
        """Level-index -> stoc_len map for consumers of ``row_levels``.

        MUST resolve the SAME bucket that produced ``row_levels``.  The SC
        attention path classifies rows with adaptive_classify_rows (which uses
        the bucket's own ladder via get_levels) and then maps index -> stream
        length through here.  When this ignored the bucket and returned the
        global list, indices stayed in range (the ladders have equal length)
        but every row ran at the WRONG stream length -- silently, since
        nothing downstream cross-checks the two.  It surfaced only as a
        realized-cost reconciliation failure ("SC lengths outside the
        adaptive/protected set") once per-group ladders diverged.

        With the escape gate ON and ``escape_stoc_len`` not already a rung,
        escaped rows carry level index ``len(levels)``; this returns the
        ladder plus that appended escape entry so index-driven dispatch loops
        cover it. Gate off -- or escape length already a rung -- returns the
        ladder itself, so the dispatch loop is byte-identical to the pre-gate
        code. The appended entry intentionally breaks the descending-order
        convention: this is an index map for dispatch, not a ladder."""
        levels = self.get_levels(
            timestep=timestep,
            total_timesteps=total_timesteps,
            operator=operator,
            block_idx=block_idx,
            total_blocks=total_blocks,
        )
        if (self.escape_gate_k is None
                or int(self.escape_stoc_len) in levels):
            return levels
        return list(levels) + [int(self.escape_stoc_len)]

    def get_dispatch_metric(self, operator: Optional[str]) -> tuple[str, float]:
        """(metric_name, sign) for one operator's per-row dispatch.

        Default ("amax", +1.0) — the original dispatch — for operators the
        table did not switch (or when the table predates dispatch_metrics)."""
        if operator and self.dispatch_metrics:
            return self.dispatch_metrics.get(operator, ("amax", 1.0))
        return ("amax", 1.0)

    def get_protected_channels(
        self,
        operator: Optional[str] = None,
        block_idx: Optional[int] = None,
        unit_idx: Optional[int] = None,
    ) -> Optional[list[int]]:
        """Return protected input-channel indices for one linear module."""
        if not operator or block_idx is None:
            return None
        key = (operator, int(block_idx), unit_idx)
        vals = self.protected_channel_indices.get(key)
        if vals is not None:
            return vals
        return self.protected_channel_indices.get((operator, int(block_idx), None))


def adaptive_classify_rows(
    metric: torch.Tensor,
    config: AdaptiveMPConfig,
    operator: Optional[str] = None,
    block_idx: Optional[int] = None,
    total_blocks: Optional[int] = None,
    timestep: int = 0,
    total_timesteps: int = 1,
) -> RowAssignment:
    """Classify rows by one of three data-driven paths (no closed-form mode).

    Checked in order: free-boundary (``FreeBoundaryMPConfig``), quantile
    (``target_fractions``), then calibrated table (``threshold_table_path``).
    Matching none of them is a configuration bug and raises.

    Args:
        metric: [N] per-row importance values (e.g. row abs-max).
        config: AdaptiveMPConfig instance.
        operator: Operator name for table / boundary lookup (e.g. "q_proj", "qk").
        block_idx: Layer / block index for table / boundary lookup.
        total_blocks: Total number of layers / blocks (used for bucketing).
        timestep: Diffusion timestep for table bucketing. LLM inference: 0.
        total_timesteps: Total diffusion timesteps for bucketing. LLM: 1.

    Returns:
        RowAssignment compatible with existing dispatch code.
    """
    N = metric.shape[0]
    # Per-group ladder when the table declares one; otherwise this IS
    # config.stoc_len_levels, so the no-groups path is byte-identical.
    levels = config.get_levels(
        timestep=timestep,
        total_timesteps=total_timesteps,
        operator=operator,
        block_idx=block_idx,
        total_blocks=total_blocks,
    )
    n_levels = len(levels)

    # Empty row batch — e.g. a MoE expert that received ZERO tokens this forward
    # (sparse top-k routing). metric is empty, so .min()/.argsort() below would
    # crash on the empty reduction. Return an empty assignment; the caller's
    # per-level dispatch loop then does nothing (empty expert → empty output).
    if N == 0:
        empty = torch.empty(0, dtype=torch.long, device=metric.device)
        return RowAssignment(row_levels=empty,
                             level_row_indices={sl: empty for sl in levels})

    # ---------- Free-boundary path (FreeBoundaryMPConfig) ----------
    # Per-(block, op) learned boundaries; no alpha/beta/progress dependency.
    # Check subclass first so inherited isinstance(cfg, AdaptiveMPConfig)
    # dispatch still works elsewhere while we dispatch correctly here.
    if isinstance(config, FreeBoundaryMPConfig):
        fixed_level = config.get_fixed_level(operator or "")
        if fixed_level is not None:
            return _classify_all_rows_to_level(metric, levels, fixed_level)
        boundaries = config.get_boundaries(operator or "")
        return _classify_with_free_boundaries(metric, boundaries, levels)

    # ---------- Quantile path (target_fractions set) ----------
    # Independent of (t, T) / alpha / beta. Top frac[0] rows -> levels[0], etc.
    if config.target_fractions is not None:
        sorted_idx = metric.argsort(descending=True)
        row_levels_q = torch.empty(N, dtype=torch.long, device=metric.device)
        level_row_indices_q: dict[int, torch.Tensor] = {}
        offset = 0
        for i, (sl, frac) in enumerate(zip(levels, config.target_fractions)):
            if i < n_levels - 1:
                count = round(frac * N)
            else:
                count = N - offset
            rows_q = sorted_idx[offset:offset + count]
            row_levels_q[rows_q] = i
            level_row_indices_q[sl] = rows_q
            offset += count
        return RowAssignment(row_levels=row_levels_q,
                             level_row_indices=level_row_indices_q)

    # ---------- Calibrated-table path (threshold_table_path set) ----------
    # Normalize metric to [0, 1]
    m_min = metric.min()
    m_max = metric.max()
    if (m_max - m_min).item() < 1e-8:
        # All metric values are equal — no meaningful ranking.
        # Default to highest precision (all rows at level 0).
        row_levels = torch.zeros(N, dtype=torch.long, device=metric.device)
        level_row_indices = {}
        for idx, sl in enumerate(levels):
            if idx == 0:
                level_row_indices[sl] = torch.arange(N, device=metric.device)
            else:
                level_row_indices[sl] = torch.empty(0, dtype=torch.long,
                                                     device=metric.device)
        return RowAssignment(row_levels=row_levels,
                             level_row_indices=level_row_indices)
    metric_norm = (metric - m_min) / (m_max - m_min)

    calibrated_thresholds = config.get_thresholds(
        timestep=timestep,
        total_timesteps=total_timesteps,
        operator=operator,
        block_idx=block_idx,
        total_blocks=total_blocks,
    )
    if calibrated_thresholds is not None:
        assignment = _classify_rows_by_thresholds(
            metric_norm, levels, calibrated_thresholds)
        _apply_escape_gate(
            assignment, metric_norm, config,
            operator=operator, block_idx=block_idx,
            total_blocks=total_blocks,
            timestep=timestep, total_timesteps=total_timesteps,
        )
        return assignment

    # No path matched (not free-boundary, no target_fractions, and no
    # calibrated thresholds for this operator/bucket). There is no closed-form
    # fallback — this is a configuration bug.
    raise ValueError(
        f"AdaptiveMPConfig: no classification path for operator={operator!r} "
        f"block_idx={block_idx} (not a FreeBoundaryMPConfig, target_fractions "
        f"unset, and no calibrated thresholds — bucket miss and no "
        f"operator_default). Re-run calibration covering this operator/layer, "
        f"or set target_fractions."
    )


def _apply_escape_gate(
    assignment: RowAssignment,
    metric_norm: torch.Tensor,
    config: AdaptiveMPConfig,
    *,
    operator: Optional[str],
    block_idx: Optional[int],
    total_blocks: Optional[int],
    timestep: int,
    total_timesteps: int,
) -> None:
    """In-place absolute escape gate (R7) on a threshold classification.

    Rows with normalized metric STRICTLY above t_esc_b = mu_b + k * sigma_b
    (per-bucket constants precomputed at table load) are reassigned to
    ``config.escape_stoc_len`` regardless of the band thresholds. The compare
    happens in the same post-sign normalized space the band thresholds live
    in, so no extra sign handling is needed (sign −1 metrics arrive already
    negated; min–max normalization maps them to 1 − normalized(raw), exactly
    as calibration stored them before computing mu/sigma).

    Escaped rows get level index ``len(stoc_len_levels)`` (a NEW index one
    past the ladder) and a ``level_row_indices[escape_stoc_len]`` entry —
    unless the escape length already is a ladder rung, in which case they are
    folded into that rung's existing index. Gate off (``escape_gate_k`` None
    ⇒ ``get_escape_threshold`` returns None) leaves the assignment untouched.
    """
    t_esc = config.get_escape_threshold(
        timestep=timestep,
        total_timesteps=total_timesteps,
        operator=operator,
        block_idx=block_idx,
        total_blocks=total_blocks,
    )
    if t_esc is None or t_esc >= 1.0:
        # t_esc >= 1.0 can never fire: metric_norm lives in [0, 1] with max
        # EXACTLY 1.0 and the compare is strict. Skipping the tensor compare
        # here is a shortcut, not a clamp — behavior is identical.
        return
    esc_mask = metric_norm > metric_norm.new_tensor(t_esc)
    if not bool(esc_mask.any().item()):
        return
    # MUST be the SAME ladder the classification used (adaptive_classify_rows
    # resolves it via get_levels). Reading config.stoc_len_levels here instead
    # was latent-correct only while every bucket shared the global ladder: the
    # rebuild below re-keys level_row_indices by stoc_len VALUE, so with a
    # per-bucket ladder every row would be re-keyed to the GLOBAL rung value at
    # its index and the bucket's ladder would be silently discarded. That is
    # unobservable in realized_flop_avg_sl (the tracker prices the assignment it
    # is handed), so the cell would look in-budget while running wrong lengths.
    levels = config.get_levels(
        timestep=timestep,
        total_timesteps=total_timesteps,
        operator=operator,
        block_idx=block_idx,
        total_blocks=total_blocks,
    )
    esc_sl = int(config.escape_stoc_len)
    esc_idx = levels.index(esc_sl) if esc_sl in levels else len(levels)
    assignment.row_levels[esc_mask] = esc_idx
    for level_idx, sl in enumerate(levels):
        assignment.level_row_indices[sl] = torch.where(
            assignment.row_levels == level_idx)[0]
    if esc_idx == len(levels):
        assignment.level_row_indices[esc_sl] = torch.where(esc_mask)[0]


# =====================================================================
# Free-boundary MP (zero hyperparameter; offline oracle-search populated)
# =====================================================================

@dataclass
class FreeBoundaryMPConfig(AdaptiveMPConfig):
    """Per-(block, op) k-1 free boundaries on normalized metric in [0, 1].

    Subclasses ``AdaptiveMPConfig`` so existing ``isinstance(cfg,
    AdaptiveMPConfig)`` dispatch in the SC attention patch continues to
    fire. The inherited ``alpha`` / ``beta`` / ``target_fractions`` fields
    are ignored when the classifier takes the free-boundary branch.

    Boundaries are keyed by ``(block_idx, op_name)``; block_idx is read
    from the module-level ``_CURRENT_BLOCK_IDX`` at classification time
    (set by forward pre-hooks installed by the auto-calibrator).

    Missing entries fall back to ``default_boundaries`` (equal spacing).
    Callers may also pin an op to a fixed level index via ``fixed_levels``;
    this is useful when some ops should stay coarse/static while others are
    searched by auto-MP.
    """
    # {(block_idx, op_name): tensor of k-1 boundaries, descending in (0, 1)}
    boundaries: dict = field(default_factory=dict)
    # {(block_idx, op_name): level_idx}, where level_idx indexes stoc_len_levels
    fixed_levels: dict = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        # sanity-check any pre-populated entries
        k = len(self.stoc_len_levels)
        for key, b in self.boundaries.items():
            assert isinstance(key, tuple) and len(key) == 2, (
                f"boundaries key must be (block_idx, op_name), got {key!r}")
            bt = b if isinstance(b, torch.Tensor) else torch.as_tensor(b)
            assert bt.numel() == k - 1, (
                f"boundaries[{key!r}] must have length {k-1}, got {bt.numel()}")
        for key, level_idx in self.fixed_levels.items():
            assert isinstance(key, tuple) and len(key) == 2, (
                f"fixed_levels key must be (block_idx, op_name), got {key!r}")
            li = int(level_idx)
            assert 0 <= li < k, (
                f"fixed_levels[{key!r}] must be in [0, {k}), got {li}")

    def default_boundaries(self) -> torch.Tensor:
        """Equal-spacing boundaries in (0, 1) descending, length k-1."""
        k = len(self.stoc_len_levels)
        return torch.tensor(
            [(k - 1 - i) / k for i in range(k - 1)], dtype=torch.float32)

    def get_boundaries(self, operator: str,
                       block_idx: Optional[int] = None) -> torch.Tensor:
        if block_idx is None:
            block_idx = _CURRENT_BLOCK_IDX
        key = (int(block_idx), operator)
        if key in self.boundaries:
            b = self.boundaries[key]
            return b if isinstance(b, torch.Tensor) else torch.as_tensor(b)
        return self.default_boundaries()

    def set_boundaries(self, operator: str, block_idx: int,
                       boundaries: torch.Tensor) -> None:
        bt = (boundaries.detach().cpu().float() if isinstance(boundaries, torch.Tensor)
              else torch.as_tensor(boundaries, dtype=torch.float32))
        k = len(self.stoc_len_levels)
        assert bt.numel() == k - 1, (
            f"expected {k-1} boundaries, got {bt.numel()}")
        self.fixed_levels.pop((int(block_idx), operator), None)
        self.boundaries[(int(block_idx), operator)] = bt

    def get_fixed_level(self, operator: str,
                        block_idx: Optional[int] = None) -> Optional[int]:
        if block_idx is None:
            block_idx = _CURRENT_BLOCK_IDX
        level_idx = self.fixed_levels.get((int(block_idx), operator))
        return None if level_idx is None else int(level_idx)

    def set_fixed_level(self, operator: str, block_idx: int,
                        level_idx: int) -> None:
        li = int(level_idx)
        k = len(self.stoc_len_levels)
        assert 0 <= li < k, f"level_idx must be in [0, {k}), got {li}"
        key = (int(block_idx), operator)
        self.boundaries.pop(key, None)
        self.fixed_levels[key] = li

    def clear_fixed_level(self, operator: str, block_idx: int) -> None:
        self.fixed_levels.pop((int(block_idx), operator), None)


def _classify_all_rows_to_level(
    metric: torch.Tensor,
    stoc_len_levels: list[int],
    level_idx: int,
) -> "RowAssignment":
    """Assign every row/head to one fixed level index."""
    N = metric.shape[0]
    row_levels = torch.full(
        (N,), int(level_idx), dtype=torch.long, device=metric.device)
    level_row_indices: dict[int, torch.Tensor] = {}
    for idx, sl in enumerate(stoc_len_levels):
        if idx == int(level_idx):
            level_row_indices[sl] = torch.arange(N, device=metric.device)
        else:
            level_row_indices[sl] = torch.empty(
                0, dtype=torch.long, device=metric.device)
    return RowAssignment(row_levels=row_levels,
                         level_row_indices=level_row_indices)


def _classify_with_free_boundaries(
    metric: torch.Tensor,
    boundaries: torch.Tensor,
    stoc_len_levels: list[int],
) -> "RowAssignment":
    """Bucket rows by normalized metric against free, non-equal-spaced
    descending boundaries. See ``adaptive_classify_rows`` for the semantics
    (level 0 = highest stoc_len, assigned to rows above the first boundary).
    """
    N = metric.shape[0]
    n_levels = len(stoc_len_levels)

    m_min = metric.min()
    m_max = metric.max()
    if (m_max - m_min).item() < 1e-8:
        # Degenerate distribution: default to level 0.
        row_levels = torch.zeros(N, dtype=torch.long, device=metric.device)
        level_row_indices: dict[int, torch.Tensor] = {}
        for idx, sl in enumerate(stoc_len_levels):
            if idx == 0:
                level_row_indices[sl] = torch.arange(N, device=metric.device)
            else:
                level_row_indices[sl] = torch.empty(
                    0, dtype=torch.long, device=metric.device)
        return RowAssignment(row_levels=row_levels,
                             level_row_indices=level_row_indices)

    metric_norm = (metric - m_min) / (m_max - m_min)

    # Ensure descending order for safety — boundaries may come from a
    # coord-descent step that hasn't yet re-sorted.
    b_sorted, _ = torch.sort(boundaries.to(metric.device).float(),
                             descending=True)
    row_levels = torch.zeros(N, dtype=torch.long, device=metric.device)
    for k in range(n_levels - 1):
        row_levels[metric_norm < b_sorted[k]] = k + 1

    level_row_indices = {}
    for i, sl in enumerate(stoc_len_levels):
        level_row_indices[sl] = torch.where(row_levels == i)[0]
    return RowAssignment(row_levels=row_levels,
                         level_row_indices=level_row_indices)


# =====================================================================
# Auto-MP budget logger (compute savings tracking during oracle search)
# =====================================================================

class AutoMPBudgetLogger:
    """Lightweight per-forward compute logger for budget-aware auto-MP.

    SC operators record a baseline cost (all rows/heads at max stoc_len) and
    the actual weighted stoc_len cost induced by the current assignment. The
    auto-MP calibrator enables this logger only while scoring candidate
    boundaries, so it sees the true block-local compute for that candidate.
    """

    _enabled: bool = False
    _log: list[dict] = []

    @classmethod
    def enable(cls):
        cls._enabled = True

    @classmethod
    def disable(cls):
        cls._enabled = False

    @classmethod
    def clear(cls):
        cls._log.clear()

    @classmethod
    def record(cls, block_idx: int, operator: str,
               baseline: float, actual: float):
        if not cls._enabled:
            return
        cls._log.append({
            "block": int(block_idx),
            "operator": operator,
            "baseline": float(baseline),
            "actual": float(actual),
        })

    @classmethod
    def snapshot(cls, clear: bool = False) -> list[dict]:
        out = list(cls._log)
        if clear:
            cls.clear()
        return out

    @classmethod
    def totals(cls, clear: bool = False) -> dict[str, float]:
        total_baseline = 0.0
        total_actual = 0.0
        for entry in cls._log:
            total_baseline += entry["baseline"]
            total_actual += entry["actual"]
        out = {"baseline": total_baseline, "actual": total_actual}
        if clear:
            cls.clear()
        return out


# =====================================================================
# MP Distribution Logger
# =====================================================================

class MPDistributionLogger:
    """Logs the fraction of rows/heads assigned to each precision level.

    Collects per-(timestep, block, operator) distribution and dumps to CSV.
    Also tracks actual compute cost for accurate savings when range-based MP
    is used (where per-row stoc_len varies across weight groups).
    """

    _log: list[dict] = []
    _compute_log: list[dict] = []  # {timestep, block, operator, baseline, actual}

    @classmethod
    def log(cls, timestep: int, block_idx: int, operator: str,
            assignment: RowAssignment, total_rows: int):
        """Record one distribution entry.

        Args:
            timestep: Current diffusion timestep.
            block_idx: Block index.
            operator: Operator name (qk, av, mlp_fc1, mlp_fc2).
            assignment: RowAssignment from classify_rows_by_metric.
            total_rows: Total number of rows/heads being classified.
        """
        entry = {
            "timestep": timestep,
            "block": block_idx,
            "operator": operator,
            "total_rows": total_rows,
        }
        for sl, rows in sorted(assignment.level_row_indices.items(), reverse=True):
            count = len(rows)
            entry[f"sl_{sl}_count"] = count
            entry[f"sl_{sl}_frac"] = round(count / max(total_rows, 1), 4)
        cls._log.append(entry)

    @classmethod
    def log_compute(cls, timestep: int, block_idx: int, operator: str,
                    baseline: int, actual: float):
        """Record actual compute cost (stoc_len * elements) for accurate savings.

        Use this instead of / in addition to log() when range-based MP is active,
        since per-row stoc_len varies across weight groups.

        Args:
            baseline: Total cost if all at max_stoc_len (M * out_features * max_sl).
            actual: Sum of effective_stoc_len * num_rows * num_out_channels per group.
        """
        cls._compute_log.append({
            "timestep": timestep,
            "block": block_idx,
            "operator": operator,
            "baseline": baseline,
            "actual": actual,
        })

    @classmethod
    def dump_csv(cls, path: str = "debug_mp_distribution.csv"):
        """Write collected distribution stats to CSV and clear."""
        if not cls._log:
            return
        import csv
        # Gather all column names (stoc_len columns vary)
        all_keys = {}
        for entry in cls._log:
            for k in entry:
                all_keys[k] = True
        # Sort: fixed columns first, then sl_* columns sorted descending
        fixed = ["timestep", "block", "operator", "total_rows"]
        sl_keys = sorted(
            [k for k in all_keys if k.startswith("sl_")],
            key=lambda k: (-int(k.split("_")[1]), k.split("_")[2]))
        fieldnames = fixed + sl_keys

        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for entry in cls._log:
                w.writerow(entry)
        print(f"[MPDistributionLogger] Wrote {len(cls._log)} rows to {path}")
        cls._log.clear()

    @classmethod
    def summary(cls, max_stoc_len: int = 256, save_path: str | None = None):
        """Print compute savings summary from collected logs.

        When _compute_log has data (range-based or combined MP), uses those
        exact baseline/actual values for accurate savings.  Otherwise falls
        back to per-row _log entries (dynamic MP only).

        Args:
            max_stoc_len: The baseline stoc_len if no MP were used.
            save_path: If provided, also save the summary to this file.
        """
        if not cls._log and not cls._compute_log:
            print("[MPDistributionLogger] No data for summary.")
            return

        total_baseline = 0
        total_actual = 0.0
        per_op_baseline: dict[str, int] = {}
        per_op_actual: dict[str, float] = {}

        # Use compute log (accurate for range-based / combined MP)
        if cls._compute_log:
            for entry in cls._compute_log:
                op = entry["operator"]
                b = entry["baseline"]
                a = entry["actual"]
                total_baseline += b
                total_actual += a
                per_op_baseline[op] = per_op_baseline.get(op, 0) + b
                per_op_actual[op] = per_op_actual.get(op, 0.0) + a

            # Also include operators that only appear in _log (e.g. qk/av
            # which may still use dynamic-only MP)
            compute_ops = {e["operator"] for e in cls._compute_log}
            for entry in cls._log:
                op = entry["operator"]
                if op in compute_ops:
                    continue  # already counted via compute_log
                n = entry["total_rows"]
                baseline = n * max_stoc_len
                total_baseline += baseline
                per_op_baseline[op] = per_op_baseline.get(op, 0) + baseline

                actual = 0.0
                for k, v in entry.items():
                    if k.startswith("sl_") and k.endswith("_count"):
                        sl = int(k.split("_")[1])
                        actual += sl * v
                total_actual += actual
                per_op_actual[op] = per_op_actual.get(op, 0.0) + actual
        else:
            # Fallback: dynamic MP only (old behaviour)
            for entry in cls._log:
                n = entry["total_rows"]
                op = entry["operator"]
                baseline = n * max_stoc_len
                total_baseline += baseline
                per_op_baseline[op] = per_op_baseline.get(op, 0) + baseline

                actual = 0.0
                for k, v in entry.items():
                    if k.startswith("sl_") and k.endswith("_count"):
                        sl = int(k.split("_")[1])
                        actual += sl * v
                total_actual += actual
                per_op_actual[op] = per_op_actual.get(op, 0.0) + actual

        savings = 1.0 - total_actual / max(total_baseline, 1)
        lines = []
        lines.append(f"{'=' * 70}")
        lines.append(f"{'MP Compute Savings Summary':^70}")
        lines.append(f"{'=' * 70}")
        lines.append(f"  Baseline (all sl={max_stoc_len}): {total_baseline:>14,}")
        lines.append(f"  Actual weighted stoc_len:         {total_actual:>14,.0f}")
        lines.append(f"  Total savings:                    {savings:>14.1%}")
        lines.append(f"  {'-' * 66}")
        lines.append(f"  {'Operator':<15s}  {'Baseline':>12s}  {'Actual':>12s}  {'Savings':>8s}")
        lines.append(f"  {'-' * 66}")
        for op in sorted(per_op_baseline.keys()):
            b = per_op_baseline[op]
            a = per_op_actual[op]
            s = 1.0 - a / max(b, 1)
            lines.append(f"  {op:<15s}  {b:>12,}  {a:>12,.0f}  {s:>8.1%}")
        lines.append(f"{'=' * 70}")

        text = "\n".join(lines)
        print(f"\n{text}\n")

        if save_path:
            with open(save_path, "w") as f:
                f.write(text + "\n")
            print(f"[MPDistributionLogger] Summary saved to {save_path}")

    @classmethod
    def clear(cls):
        cls._log.clear()
        cls._compute_log.clear()


# =====================================================================
# Metric Profiler — collects μ/σ of importance metrics per (t, block, op)
# =====================================================================

class MetricProfiler:
    """Lightweight profiler: records per-(timestep, block, operator) metric stats.

    Call MetricProfiler.record(metric, timestep, block, operator) from
    the MP classification functions.  At the end of inference, call
    MetricProfiler.dump_csv() to write the collected statistics.

    The CSV contains: timestep, block, operator, N, mean, std, min, max,
    q25, q75, q95, q99.
    """

    _log: list[dict] = []
    _enabled: bool = False

    @classmethod
    def enable(cls):
        cls._enabled = True

    @classmethod
    def disable(cls):
        cls._enabled = False

    @classmethod
    def record(cls, metric: torch.Tensor, timestep: int, block_idx: int,
               operator: str):
        """Record statistics for a single metric vector."""
        if not cls._enabled:
            return

        m = metric.float()
        cls._log.append({
            "timestep": timestep,
            "block": block_idx,
            "operator": operator,
            "N": m.numel(),
            "mean": m.mean().item(),
            "std": m.std().item(),
            "min": m.min().item(),
            "max": m.max().item(),
            "q25": m.quantile(0.25).item(),
            "q75": m.quantile(0.75).item(),
            "q95": m.quantile(0.95).item(),
            "q99": m.quantile(0.99).item(),
        })

    @classmethod
    def dump_csv(cls, path: str = "profile_metric_sigma.csv"):
        """Write collected metric statistics to CSV and clear."""
        if not cls._log:
            print("[MetricProfiler] No data to dump.")
            return
        import csv
        fieldnames = ["timestep", "block", "operator", "N",
                      "mean", "std", "min", "max",
                      "q25", "q75", "q95", "q99"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(cls._log)
        print(f"[MetricProfiler] Wrote {len(cls._log)} rows to {path}")
        cls._log.clear()

    @classmethod
    def clear(cls):
        cls._log.clear()


# =====================================================================
# Range-based Mixed Precision (weight min/max range)
# =====================================================================

@dataclass
class RangeMPConfig:
    """Range-based mixed precision: assigns stoc_len levels based on
    per-group weight (max-min) range.

    Groups with small range -> low stoc_len (tight values, low precision ok).
    Groups with large range -> high stoc_len (spread values, need precision).

    Uses threshold-based mapping similar to AdaptiveMPConfig:
    - Normalize ranges to [0, 1]
    - base_threshold controls the cutoff between highest and lower levels
    - Ranges with normalized value >= base_threshold -> highest stoc_len
    - Ranges below -> split among lower levels via evenly-spaced boundaries

    Args:
        stoc_len_levels: Descending list of stoc_len values.
        base_threshold: Normalized range threshold (0-1). Higher = more
            groups get lower precision (more aggressive).
        operator_thresholds: Per-operator threshold overrides.
            Keys: "qk", "av", "mlp_fc1", "mlp_fc2", "input_proj", "proj".
    """
    stoc_len_levels: list[int]
    base_threshold: float = 0.3
    operator_thresholds: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        assert len(self.stoc_len_levels) >= 2, (
            "Need at least 2 levels (high + low)")
        for i in range(len(self.stoc_len_levels) - 1):
            assert self.stoc_len_levels[i] > self.stoc_len_levels[i + 1], (
                f"stoc_len_levels must be sorted descending, "
                f"got {self.stoc_len_levels}")

    def get_threshold(self, operator: Optional[str] = None) -> float:
        """Get threshold for an operator, falling back to global."""
        if operator and operator in self.operator_thresholds:
            return self.operator_thresholds[operator]
        return self.base_threshold


def classify_groups_by_range(
    weight: torch.Tensor,
    group_size: int,
    config: RangeMPConfig,
    operator: Optional[str] = None,
) -> list[int]:
    """Compute per-group (max-min) range and assign stoc_len levels.

    Groups with large range need more SC precision (high stoc_len),
    groups with small range can use lower precision (low stoc_len).

    The mapping uses threshold-based classification analogous to
    adaptive_classify_rows:
    - Normalize per-group ranges to [0, 1]
    - range_norm >= base_threshold -> level 0 (highest stoc_len)
    - Below base_threshold -> split evenly among lower levels

    Args:
        weight: [out_features, in_features] weight tensor (already quantized).
        group_size: Number of output rows per group.

            * ``1``  selects true per-row grouping (``num_groups == out_features``).
            * Values ``<= 0`` or ``>= out_features`` collapse to a single
              per-tensor group (``num_groups == 1``) — one ``stoc_len`` for
              the whole weight matrix.
            * Any other value ``g`` produces ``out_features // g`` groups and
              currently requires ``out_features % g == 0`` (the reshape below
              will raise otherwise).
        config: RangeMPConfig instance.
        operator: Operator name for per-op threshold lookup.

    Returns:
        List of stoc_len values, one per group (length ``num_groups``).
    """
    out_features, in_features = weight.shape
    if group_size <= 0 or group_size >= out_features:
        group_size = out_features

    num_groups = out_features // group_size
    levels = config.stoc_len_levels
    n_levels = len(levels)
    threshold = config.get_threshold(operator)
    threshold = min(threshold, 0.95)

    # Reshape to [num_groups, group_size * in_features]
    w = weight.reshape(num_groups, -1).float()
    group_max = w.amax(dim=-1)   # [num_groups]
    group_min = w.amin(dim=-1)   # [num_groups]
    group_range = group_max - group_min  # [num_groups]

    # Normalize to [0, 1]
    r_min = group_range.min()
    r_max = group_range.max()
    range_norm = (group_range - r_min) / (r_max - r_min + 1e-8)

    # Threshold-based classification (same logic as adaptive_classify_rows)
    # range_norm >= threshold -> level 0 (highest stoc_len, needs high precision)
    # Below threshold -> split evenly among lower levels
    group_levels = torch.zeros(num_groups, dtype=torch.long, device=weight.device)

    boundaries = []
    for k in range(n_levels - 1):
        b = threshold * (n_levels - 1 - k) / (n_levels - 1)
        boundaries.append(b)

    for k in range(n_levels - 1):
        group_levels[range_norm < boundaries[k]] = k + 1

    # Convert level indices to stoc_len values
    result = [levels[group_levels[g].item()] for g in range(num_groups)]

    # Log distribution
    dist = {}
    for sl in levels:
        count = result.count(sl)
        dist[sl] = count
    print(f"  [RangeMP] {operator or 'unknown'}: "
          f"groups={num_groups}, threshold={threshold:.2f}, "
          f"distribution={dist}, "
          f"range_stats: min={group_range.min().item():.4f}, "
          f"max={group_range.max().item():.4f}, "
          f"mean={group_range.mean().item():.4f}")

    return result
