"""
Stochastic Computing Integration for Q-DiT

This module provides SC-based matrix multiplication as an alternative to
standard floating-point operations in the diffusion transformer.

Key components:
- SCController: Manages timewise and layerwise SC control
- SCPrecisionMap: Per-block, per-operator, per-group precision config
- SCAttention: Attention module with SC q@k^T support
- SCDiTBlock: DiT block with SC support
- sc_matmul_qk: SC matrix multiplication wrapper
- add_sc_wrapper: Utility to convert model to SC-enabled version
"""

from .sc_controller import SCController
from .sc_precision_map import SCPrecisionMap, OperatorConfig, OPERATORS
from .mp_config import (MPConfig, AdaptiveMPConfig, RangeMPConfig,
                        RowAssignment,
                        classify_rows_by_metric, adaptive_classify_rows,
                        classify_groups_by_range,
                        MPDistributionLogger)
from .sc_attention import SCAttention
from .sc_mlp import SCMlp
from .sc_block import SCDiTBlock
from .sc_matmul import sc_matmul_qk
from .sc_modelutils import (
    add_sc_wrapper,
    quantize_sc_model,
    create_sc_controller_from_args,
)

__all__ = [
    'SCController',
    'SCPrecisionMap',
    'OperatorConfig',
    'OPERATORS',
    'MPConfig',
    'AdaptiveMPConfig',
    'RangeMPConfig',
    'RowAssignment',
    'classify_rows_by_metric',
    'adaptive_classify_rows',
    'classify_groups_by_range',
    'MPDistributionLogger',
    'SCAttention',
    'SCMlp',
    'SCDiTBlock',
    'sc_matmul_qk',
    'add_sc_wrapper',
    'quantize_sc_model',
    'create_sc_controller_from_args',
]
