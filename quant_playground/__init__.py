"""quant-playground: an educational quantization playground for low-precision formats.

Public surface: format converters, quantizers, metrics, and the toy MLP demo.
"""

from __future__ import annotations

from . import demo, formats, metrics, quantizers
from .metrics import (
    compression_ratio,
    compressed_bytes,
    cosine_similarity,
    mae,
    mse,
    original_bytes,
    round_trip_report,
    snr,
)
from .quantizers import (
    FP4_E2M1,
    FP8_E4M3,
    FP8_E5M2,
    INT4,
    INT8,
    FloatFormatQuantizer,
    IntQuantizer,
    all_quantizers,
    get_quantizer,
)

__version__ = "0.1.0"

__all__ = [
    "demo",
    "formats",
    "metrics",
    "quantizers",
    "mse",
    "mae",
    "snr",
    "cosine_similarity",
    "original_bytes",
    "compressed_bytes",
    "compression_ratio",
    "round_trip_report",
    "FP8_E4M3",
    "FP8_E5M2",
    "FP4_E2M1",
    "INT8",
    "INT4",
    "FloatFormatQuantizer",
    "IntQuantizer",
    "all_quantizers",
    "get_quantizer",
    "__version__",
]
