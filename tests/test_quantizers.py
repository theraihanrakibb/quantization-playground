"""Tests for the quantizer API (quantizers.py)."""

from __future__ import annotations

import numpy as np
import pytest

from quant_playground import quantizers


def test_get_quantizer_float_and_int() -> None:
    assert quantizers.get_quantizer("fp", 8).name == "fp8_e4m3"
    assert quantizers.get_quantizer("fp", 4).name == "fp4_e2m1"
    assert quantizers.get_quantizer("int", 8).name == "int8"
    assert quantizers.get_quantizer("int", 4).name == "int4"
    assert quantizers.get_quantizer("fp8_e5m2").name == "fp8_e5m2"


def test_get_quantizer_rejects_bad_bits() -> None:
    with pytest.raises(ValueError):
        quantizers.get_quantizer("fp", 16)
    with pytest.raises(ValueError):
        quantizers.get_quantizer("int", 2)


def test_float_quantizer_returns_int_bits_and_float_back() -> None:
    qz = quantizers.get_quantizer("fp8_e4m3")
    x = np.array([1.0, 2.0, -3.0], dtype=np.float32)
    q = qz.quantize(x)
    assert np.issubdtype(q.dtype, np.integer)
    back = qz.dequantize(q)
    assert np.allclose(x, back, atol=1e-6)


def test_int8_quantizer_explicit_scale() -> None:
    qz = quantizers.INT8
    x = np.array([-2.0, 0.0, 2.0])
    q = qz.quantize(x, scale=2.0)
    assert q.tolist() == [-1, 0, 1]
    assert np.allclose(qz.dequantize(q), x, atol=1e-9)


def test_int4_quantizer_explicit_scale() -> None:
    qz = quantizers.INT4
    x = np.array([-2.0, 0.0, 2.0])
    q = qz.quantize(x, scale=2.0)
    assert q.tolist() == [-1, 0, 1]
    assert np.allclose(qz.dequantize(q), x, atol=1e-9)


def test_int8_round_trip_error_within_half_step() -> None:
    rng = np.random.default_rng(3)
    x = (rng.random(1000) * 10) - 5
    qz = quantizers.INT8
    qz.quantize(x)  # sets self._scale
    back = qz.dequantize(qz.quantize(x))
    expected_scale = (np.max(np.abs(x)) / 127.0) if np.max(np.abs(x)) > 0 else 1.0
    assert np.max(np.abs(x - back)) <= 0.5 * expected_scale + 1e-9


def test_bytes_per_element() -> None:
    assert quantizers.FP8_E4M3.bytes_per_element == 1.0
    assert quantizers.FP4_E2M1.bytes_per_element == 0.5
    assert quantizers.INT8.bytes_per_element == 1.0
    assert quantizers.INT4.bytes_per_element == 0.5


def test_all_quantizers_count() -> None:
    assert len(quantizers.all_quantizers()) == 5
