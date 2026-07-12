"""Tests for the metrics module (metrics.py)."""

from __future__ import annotations

import numpy as np
import pytest

from quant_playground import metrics
from quant_playground.quantizers import FP8_E4M3


def test_mse_mae_known() -> None:
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 0.0, 3.0])
    assert metrics.mse(a, b) == pytest.approx(4.0 / 3.0)
    assert metrics.mae(a, b) == pytest.approx(2.0 / 3.0)


def test_snr_identical_is_inf() -> None:
    a = np.array([1.0, 2.0, 3.0])
    assert metrics.snr(a, a) == float("inf")


def test_snr_decreases_with_noise() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0])
    quiet = a + 1e-6
    loud = a + 1.0
    assert metrics.snr(a, quiet) > metrics.snr(a, loud)


def test_cosine_identical_is_one() -> None:
    a = np.array([1.0, 2.0, 3.0])
    assert metrics.cosine_similarity(a, a) == pytest.approx(1.0)


def test_size_math_fp32_to_fp8_is_4x() -> None:
    n = 100
    orig = metrics.original_bytes(n)
    comp = metrics.compressed_bytes(n, FP8_E4M3.bytes_per_element)
    assert orig == 400
    assert comp == 100.0
    assert metrics.compression_ratio(orig, comp) == pytest.approx(4.0)


def test_round_trip_report_structure() -> None:
    rng = np.random.default_rng(0)
    x = (rng.random(64) * 4) - 2
    report = metrics.round_trip_report(x, FP8_E4M3)
    assert report["format"] == "fp8_e4m3"
    assert "mse" in report and "mae" in report and "snr_db" in report
    assert "cosine" in report and "compression_ratio" in report
    assert report["original_bytes"] == 64 * 4
    assert report["compressed_bytes"] == 64.0
    assert report["compression_ratio"] == pytest.approx(4.0)
