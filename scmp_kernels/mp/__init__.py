"""Mixed-precision dispatch + config.

Re-exports the public API from `scmp_kernels.mp.config`.
"""

from .config import (
    MPConfig,
    AdaptiveMPConfig,
    RangeMPConfig,
    RowAssignment,
    classify_rows_by_metric,
    adaptive_classify_rows,
    classify_groups_by_range,
    MPDistributionLogger,
    MetricProfiler,
)

__all__ = [
    "MPConfig",
    "AdaptiveMPConfig",
    "RangeMPConfig",
    "RowAssignment",
    "classify_rows_by_metric",
    "adaptive_classify_rows",
    "classify_groups_by_range",
    "MPDistributionLogger",
    "MetricProfiler",
]
