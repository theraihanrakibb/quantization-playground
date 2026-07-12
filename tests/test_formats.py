"""Tests for the manual low-precision format emulation (formats.py)."""

from __future__ import annotations

import numpy as np
import pytest

from quant_playground import formats


def test_fp8_e4m3_round_trip_exact_on_representable_values() -> None:
    values = np.array(
        [0.0, 1.0, 2.0, 1.5, 0.5, -3.0, 240.0, 0.125, 0.00390625, -0.5], dtype=np.float64
    )
    q = formats.fp32_to_fp8_e4m3(values)
    back = formats.fp8_e4m3_to_fp32(q)
    assert np.allclose(values, back, atol=1e-6)


def test_fp8_e5m2_round_trip_exact_on_representable_values() -> None:
    values = np.array([0.0, 1.0, 2.0, 1.5, 0.5, -3.0, 57344.0, 0.25, -0.5], dtype=np.float64)
    q = formats.fp32_to_fp8_e5m2(values)
    back = formats.fp8_e5m2_to_fp32(q)
    assert np.allclose(values, back, atol=1e-4)


def test_fp4_e2m1_round_trip_exact_on_representable_values() -> None:
    values = np.array([0.0, 1.0, 2.0, 1.5, 0.5, -3.0, 6.0, 4.0, -6.0], dtype=np.float64)
    q = formats.fp32_to_fp4_e2m1(values)
    back = formats.fp4_e2m1_to_fp32(q)
    assert np.allclose(values, back, atol=1e-6)


@pytest.mark.parametrize(
    "value,expected",
    [
        (1000.0, 240.0),  # FP8 E4M3 saturates at 240
        (-1000.0, -240.0),
        (1e6, 57344.0),   # FP8 E5M2 saturates at 57344
        (100.0, 6.0),     # FP4 E2M1 saturates at 6
        (-100.0, -6.0),
    ],
)
def test_saturation(value: float, expected: float) -> None:
    if expected == 240.0 or expected == -240.0:
        back = float(formats.fp8_e4m3_to_fp32(formats.fp32_to_fp8_e4m3(np.array([value])))[0])
    elif expected == 57344.0 or expected == -57344.0:
        back = float(formats.fp8_e5m2_to_fp32(formats.fp32_to_fp8_e5m2(np.array([value])))[0])
    else:
        back = float(formats.fp4_e2m1_to_fp32(formats.fp32_to_fp4_e2m1(np.array([value])))[0])
    assert back == pytest.approx(expected, rel=0.0, abs=1e-6)


def test_sign_preserved_for_floats() -> None:
    pos = np.array([1.25, 3.0, 0.75], dtype=np.float64)
    neg = -pos
    q_pos = formats.fp32_to_fp8_e4m3(pos)
    q_neg = formats.fp32_to_fp8_e4m3(neg)
    # sign bit is the most significant bit (bit 7 for fp8)
    assert np.all((q_pos >> 7) == 0)
    assert np.all((q_neg >> 7) == 1)
    assert np.allclose(formats.fp8_e4m3_to_fp32(q_neg), neg, atol=1e-6)


def test_round_trip_error_bounded_for_fp8() -> None:
    rng = np.random.default_rng(0)
    x = (rng.random(2000) * 16) - 8  # magnitude < 8
    for encode, decode, man_bits, bound in [
        (formats.fp32_to_fp8_e4m3, formats.fp8_e4m3_to_fp32, 3, 0.5),
        (formats.fp32_to_fp8_e5m2, formats.fp8_e5m2_to_fp32, 2, 1.0),
    ]:
        q = encode(x)
        back = decode(q)
        assert np.max(np.abs(x - back)) <= bound
        assert np.all(np.isfinite(back))


def test_round_trip_error_bounded_for_fp4() -> None:
    rng = np.random.default_rng(1)
    x = (rng.random(2000) * 8) - 4  # magnitude < 4 (within FP4 range 6)
    q = formats.fp32_to_fp4_e2m1(x)
    back = formats.fp4_e2m1_to_fp32(q)
    assert np.max(np.abs(x - back)) <= 1.0
    assert np.all(np.isfinite(back))


def test_int8_symmetric_round_trip() -> None:
    values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    scale = 0.5
    q = formats.fp32_to_int8(values, scale)
    assert q.tolist() == [-2, -1, 0, 1, 2]
    back = formats.int8_to_fp32(q, scale)
    assert np.allclose(values, back, atol=1e-9)


def test_int4_symmetric_round_trip() -> None:
    values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    scale = 0.5
    q = formats.fp32_to_int4(values, scale)
    assert q.tolist() == [-2, -1, 0, 1, 2]
    back = formats.int4_to_fp32(q, scale)
    assert np.allclose(values, back, atol=1e-9)


def test_int8_clamps_to_range() -> None:
    values = np.array([1e6, -1e6, 0.0])
    scale = 1.0
    q = formats.fp32_to_int8(values, scale)
    assert int(q[0]) == 127
    assert int(q[1]) == -127
    assert int(q[2]) == 0


def test_int4_clamps_to_range() -> None:
    values = np.array([1e6, -1e6, 0.0])
    scale = 1.0
    q = formats.fp32_to_int4(values, scale)
    assert int(q[0]) == 7
    assert int(q[1]) == -7
    assert int(q[2]) == 0
