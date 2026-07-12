"""Quantizer API and per-format quantize / dequantize implementations.

A :class:`Quantizer` knows how to compress a tensor into a low-precision
representation (returned as an integer bit-field array) and reconstruct it.
The "stored" integer array plus the format's ``bytes_per_element`` drive the
size / compression-ratio metrics.
"""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

from . import formats

logger = logging.getLogger(__name__)


class Quantizer(Protocol):
    """Interface every quantizer implements."""

    name: str
    bits: int
    bytes_per_element: float
    is_float: bool

    def quantize(self, x: np.ndarray) -> np.ndarray:
        """Compress ``x`` into the low-precision integer bit-field array."""
        ...

    def dequantize(self, q: np.ndarray) -> np.ndarray:
        """Reconstruct an ``float32`` approximation from ``q``."""
        ...


class FloatFormatQuantizer:
    """Quantizer backed by a manual float-format encoder/decoder."""

    def __init__(
        self,
        name: str,
        bits: int,
        encode,
        decode,
    ) -> None:
        self.name = name
        self.bits = bits
        self.bytes_per_element = float(bits) / 8.0
        self.is_float = True
        self._encode = encode
        self._decode = decode

    def quantize(self, x: np.ndarray) -> np.ndarray:
        return self._encode(np.asarray(x, dtype=np.float64))

    def dequantize(self, q: np.ndarray) -> np.ndarray:
        return self._decode(np.asarray(q, dtype=np.int64))


class IntQuantizer:
    """Symmetric integer quantizer (INT8 / INT4) with a per-tensor scale."""

    def __init__(self, bits: int) -> None:
        self.bits = bits
        self.is_float = False
        self.bytes_per_element = float(bits) / 8.0
        self.name = f"int{bits}"
        self.max_int = (1 << (bits - 1)) - 1  # INT8 -> 127, INT4 -> 7
        self._scale: float = 1.0

    @staticmethod
    def _compute_scale(x: np.ndarray) -> float:
        max_abs = float(np.max(np.abs(x))) if x.size else 0.0
        return max_abs / 127.0 if max_abs > 0.0 else 1.0

    def quantize(self, x: np.ndarray, scale: float | None = None) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        self._scale = scale if scale is not None else self._compute_scale(x)
        q = np.round(x / self._scale)
        return np.clip(q, -self.max_int, self.max_int).astype(np.int64)

    def dequantize(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.int64)
        return (q.astype(np.float64) * self._scale).astype(np.float32)


# ---------------------------------------------------------------------------
# Concrete quantizer instances
# ---------------------------------------------------------------------------
FP8_E4M3 = FloatFormatQuantizer(
    name="fp8_e4m3", bits=8, encode=formats.fp32_to_fp8_e4m3, decode=formats.fp8_e4m3_to_fp32
)
FP8_E5M2 = FloatFormatQuantizer(
    name="fp8_e5m2", bits=8, encode=formats.fp32_to_fp8_e5m2, decode=formats.fp8_e5m2_to_fp32
)
FP4_E2M1 = FloatFormatQuantizer(
    name="fp4_e2m1", bits=4, encode=formats.fp32_to_fp4_e2m1, decode=formats.fp4_e2m1_to_fp32
)
INT8 = IntQuantizer(bits=8)
INT4 = IntQuantizer(bits=4)


def get_quantizer(format: str, bits: int = 8) -> Quantizer:
    """Resolve a :class:`Quantizer` from a ``--format``/``--bits`` pair.

    Accepted ``format`` values:

    * ``"fp"`` / ``"float"`` with ``bits`` in ``{8, 4}``
    * ``"int"`` / ``"integer"`` with ``bits`` in ``{8, 4}``
    * explicit names: ``fp8_e4m3``, ``fp8_e5m2``, ``fp4_e2m1``, ``int8``, ``int4``
    """
    fmt = (format or "").lower()
    if fmt in ("fp", "float"):
        if bits == 8:
            return FP8_E4M3
        if bits == 4:
            return FP4_E2M1
        raise ValueError(f"float format requires bits 4 or 8, got {bits}")
    if fmt in ("int", "integer"):
        if bits == 8:
            return INT8
        if bits == 4:
            return INT4
        raise ValueError(f"int format requires bits 4 or 8, got {bits}")
    explicit = {
        "fp8_e4m3": FP8_E4M3,
        "fp8_e5m2": FP8_E5M2,
        "fp4_e2m1": FP4_E2M1,
        "int8": INT8,
        "int4": INT4,
    }
    if fmt in explicit:
        return explicit[fmt]
    raise ValueError(f"unknown format '{format}'")


def all_quantizers() -> list[Quantizer]:
    """Return every built-in quantizer (float formats first, then int)."""
    return [FP8_E4M3, FP8_E5M2, FP4_E2M1, INT8, INT4]
