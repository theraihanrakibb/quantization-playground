"""Error and size metrics for quantization round-trips.

All functions accept array-like inputs and return plain Python scalars so the
results are easy to log, print, and assert on in tests.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def mse(a: Any, b: Any) -> float:
    """Mean squared error between tensors ``a`` and ``b``."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean((a - b) ** 2))


def mae(a: Any, b: Any) -> float:
    """Mean absolute error between tensors ``a`` and ``b``."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean(np.abs(a - b)))


def snr(a: Any, b: Any) -> float:
    """Signal-to-noise ratio (dB) of ``a`` versus reconstruction error ``a - b``.

    Returns ``inf`` when the reconstruction is exact (zero noise power).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    noise = a - b
    signal_power = float(np.sum(a * a))
    noise_power = float(np.sum(noise * noise))
    if noise_power == 0.0:
        return float("inf")
    return 10.0 * math.log10(signal_power / noise_power)


def cosine_similarity(a: Any, b: Any) -> float:
    """Cosine similarity between flat vectors ``a`` and ``b`` (range [-1, 1])."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def original_bytes(n_elements: int, element_bytes: int = 4) -> int:
    """Number of bytes the tensor occupies in the reference ``float32`` format."""
    return int(n_elements) * int(element_bytes)


def compressed_bytes(n_elements: int, bytes_per_element: float) -> float:
    """Number of bytes the tensor occupies in the compressed format."""
    return int(n_elements) * float(bytes_per_element)


def compression_ratio(original: float, compressed: float) -> float:
    """Ratio of original to compressed size (>= 1 means a reduction)."""
    if compressed == 0:
        return float("inf")
    return float(original) / float(compressed)


def round_trip_report(x: Any, quantizer: Any) -> dict[str, float]:
    """Run a full quantize -> dequantize round-trip and summarize the results.

    Returns a dict with MSE, MAE, SNR (dB), cosine similarity, and the size /
    compression metrics for ``x``.
    """
    x = np.asarray(x, dtype=np.float64)
    q = quantizer.quantize(x)
    x_hat = np.asarray(quantizer.dequantize(q), dtype=np.float64)
    n = int(x.size)
    orig = original_bytes(n)
    comp = compressed_bytes(n, quantizer.bytes_per_element)
    return {
        "format": quantizer.name,
        "mse": mse(x, x_hat),
        "mae": mae(x, x_hat),
        "snr_db": snr(x, x_hat),
        "cosine": cosine_similarity(x, x_hat),
        "original_bytes": float(orig),
        "compressed_bytes": comp,
        "compression_ratio": compression_ratio(orig, comp),
    }
