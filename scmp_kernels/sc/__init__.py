"""Stochastic-computing kernels.

Public surface: a single ``sc_matmul`` function with a ``granularity``
parameter ("per_tensor" / "per_row" / "per_head"). All inputs/outputs
are float32; quantization happens inside the Triton kernels.
"""

from .sc_triton import sc_matmul

__all__ = [
    "sc_matmul",
]
