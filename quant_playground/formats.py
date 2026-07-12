"""Low-precision floating-point and integer format emulation.

This module implements pure-Python (numpy-vectorized) encoders/decoders for the
low-precision number formats used in modern inference hardware:

* FP8 E4M3  - 1 sign / 4 exponent / 3 mantissa  (bias 7,  max 240)
* FP8 E5M2  - 1 sign / 5 exponent / 2 mantissa  (bias 15, max 57344)
* FP4 E2M1  - 1 sign / 2 exponent / 1 mantissa  (bias 1,  max 6)
* INT8      - symmetric 8-bit integer with a scalar scale
* INT4      - symmetric 4-bit integer with a scalar scale

The float formats are implemented by manually packing/unpacking the
sign / exponent / mantissa bit fields exactly as the IEEE-ish specs describe.
Values beyond the representable range saturate to the format maximum
(round-to-nearest, saturate on overflow). No torch or GPU is required.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "fp32_to_fp8_e4m3",
    "fp8_e4m3_to_fp32",
    "fp32_to_fp8_e5m2",
    "fp8_e5m2_to_fp32",
    "fp32_to_fp4_e2m1",
    "fp4_e2m1_to_fp32",
    "fp32_to_int8",
    "int8_to_fp32",
    "fp32_to_int4",
    "int4_to_fp32",
    "FP8_E4M3",
    "FP8_E5M2",
    "FP4_E2M1",
]


# ---------------------------------------------------------------------------
# Format specifications.
# exp_bits / man_bits / bias define the bit layout. ``reserved`` is True when
# the maximum exponent field is reserved for inf/NaN (FP8). FP4 E2M1 uses every
# exponent field as a normal/subnormal, so its top exponent (=> max 6) is valid.
# ---------------------------------------------------------------------------
FP8_E4M3 = {"exp_bits": 4, "man_bits": 3, "bias": 7, "max": 240.0, "reserved": True}
FP8_E5M2 = {"exp_bits": 5, "man_bits": 2, "bias": 15, "max": 57344.0, "reserved": True}
FP4_E2M1 = {"exp_bits": 2, "man_bits": 1, "bias": 1, "max": 6.0, "reserved": False}


# ---------------------------------------------------------------------------
# Vectorized float encode / decode
# ---------------------------------------------------------------------------
def _encode_array(
    x: np.ndarray,
    exp_bits: int,
    man_bits: int,
    bias: int,
    reserved: bool,
    out_dtype: np.dtype = np.int64,
) -> np.ndarray:
    """Encode an array of ``float64`` values into packed integer bit fields.

    Strategy: decompose each value with ``np.frexp`` (true exponent + mantissa
    fraction in ``[0, 1)``), then re-quantize into the target exponent/mantissa
    widths. Subnormals, normals, and saturation are handled per element.

    When ``reserved`` is True (FP8) the maximum exponent field is reserved for
    inf/NaN and the largest normal exponent is ``emax - 1``. When False (FP4)
    every exponent field is a valid normal, so the top field yields the max.
    """
    x = np.asarray(x, dtype=np.float64)
    emax = (1 << exp_bits) - 1  # maximum exponent field
    man_mask = (1 << man_bits) - 1
    normal_max_e = (emax - 1) if reserved else emax

    out = np.zeros(x.shape, dtype=np.int64)
    neg = x < 0
    xa = np.abs(x)

    # Zero shortcut -> all zero bits.
    zero = xa == 0.0

    # Decompose magnitude into (mantissa fraction, true exponent).
    mant, exp = np.frexp(xa)  # xa == mant * 2**exp, mant in [0.5, 1)
    true_exp = exp - 1  # exponent of the normalized 1.xxx form
    frac = mant * 2.0 - 1.0  # mantissa fraction in [0, 1)
    etarget = true_exp + bias

    normal = (etarget >= 1) & (etarget <= normal_max_e)
    subnormal = (~normal) & (etarget <= 0)

    # --- normal path -----------------------------------------------------
    e_field_n = etarget.astype(np.int64)
    m_full_n = frac * (1 << man_bits)
    m_field_n = np.round(m_full_n).astype(np.int64)
    round_overflow = m_field_n >= (1 << man_bits)
    m_field_n = np.where(round_overflow, 0, m_field_n)
    e_field_n = np.where(round_overflow, e_field_n + 1, e_field_n)
    # if rounding still overflows the exponent, saturate to max normal
    saturate_flag = e_field_n > normal_max_e
    e_field_n = np.where(saturate_flag, normal_max_e, e_field_n)
    m_field_n = np.where(saturate_flag, man_mask, m_field_n)

    # --- subnormal path --------------------------------------------------
    sub_exp = 1 - bias
    scale = 2.0 ** (sub_exp - man_bits)
    m_full_s = xa / scale
    m_field_s = np.round(m_full_s).astype(np.int64)
    m_field_s = np.clip(m_field_s, 0, man_mask)
    e_field_s = np.zeros_like(m_field_s)

    # --- saturation path (value beyond max normal) -----------------------
    e_field_sat = normal_max_e
    m_field_sat = man_mask

    e_field = np.where(normal, e_field_n, np.where(subnormal, e_field_s, e_field_sat))
    m_field = np.where(normal, m_field_n, np.where(subnormal, m_field_s, m_field_sat))

    # zero overrides everything
    e_field = np.where(zero, 0, e_field)
    m_field = np.where(zero, 0, m_field)

    sign = neg.astype(np.int64)
    out = (sign << (exp_bits + man_bits)) | (e_field << man_bits) | m_field
    return out.astype(out_dtype)


def _decode_array(
    bits: np.ndarray,
    exp_bits: int,
    man_bits: int,
    bias: int,
    reserved: bool,
    out_dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Decode packed integer bit fields back into ``float64`` values."""
    bits = np.asarray(bits, dtype=np.int64)
    emax = (1 << exp_bits) - 1
    man_mask = (1 << man_bits) - 1

    sign = ((bits >> (exp_bits + man_bits)) & 1).astype(np.float64)
    e_field = (bits >> man_bits) & emax
    m_field = bits & man_mask

    inf_nan = reserved and (e_field == emax)
    subnormal = e_field == 0

    sub_val = (m_field.astype(np.float64) * 2.0 ** (-man_bits)) * 2.0 ** (1 - bias)
    true_exp = e_field - bias
    norm_val = (
        1.0 + m_field.astype(np.float64) * 2.0 ** (-man_bits)
    ) * 2.0 ** true_exp

    val = np.where(subnormal, sub_val, norm_val)
    # Reserved exponent field => treat as +/- inf (we never emit true NaN).
    val = np.where(inf_nan, np.inf, val)
    val = np.where(sign == 1, -val, val)
    return val.astype(out_dtype)


# ---------------------------------------------------------------------------
# Public float format converters (scalar-friendly via np.vectorize)
# ---------------------------------------------------------------------------
def _spec_kwargs(spec: dict) -> dict:
    """Extract only the bit-layout keys a converter needs from a spec dict."""
    return {
        "exp_bits": spec["exp_bits"],
        "man_bits": spec["man_bits"],
        "bias": spec["bias"],
        "reserved": spec["reserved"],
    }


def fp32_to_fp8_e4m3(x: np.ndarray | float | list[float]) -> np.ndarray:
    """Quantize ``float32`` values to FP8 E4M3 (1s/4e/3m), returning bit fields."""
    return _encode_array(np.asarray(x, dtype=np.float64), **_spec_kwargs(FP8_E4M3))


def fp8_e4m3_to_fp32(bits: np.ndarray | int | list[int]) -> np.ndarray:
    """Decode FP8 E4M3 bit fields back to ``float32`` values."""
    return _decode_array(np.asarray(bits, dtype=np.int64), **_spec_kwargs(FP8_E4M3))


def fp32_to_fp8_e5m2(x: np.ndarray | float | list[float]) -> np.ndarray:
    """Quantize ``float32`` values to FP8 E5M2 (1s/5e/2m), returning bit fields."""
    return _encode_array(np.asarray(x, dtype=np.float64), **_spec_kwargs(FP8_E5M2))


def fp8_e5m2_to_fp32(bits: np.ndarray | int | list[int]) -> np.ndarray:
    """Decode FP8 E5M2 bit fields back to ``float32`` values."""
    return _decode_array(np.asarray(bits, dtype=np.int64), **_spec_kwargs(FP8_E5M2))


def fp32_to_fp4_e2m1(x: np.ndarray | float | list[float]) -> np.ndarray:
    """Quantize ``float32`` values to FP4 E2M1 (1s/2e/1m), returning bit fields."""
    return _encode_array(np.asarray(x, dtype=np.float64), **_spec_kwargs(FP4_E2M1))


def fp4_e2m1_to_fp32(bits: np.ndarray | int | list[int]) -> np.ndarray:
    """Decode FP4 E2M1 bit fields back to ``float32`` values."""
    return _decode_array(np.asarray(bits, dtype=np.int64), **_spec_kwargs(FP4_E2M1))


# ---------------------------------------------------------------------------
# Symmetric integer converters
# ---------------------------------------------------------------------------
def fp32_to_int8(x: np.ndarray | float | list[float], scale: float) -> np.ndarray:
    """Symmetrically quantize to signed INT8 using ``scale`` (clamped to [-127, 127])."""
    x = np.asarray(x, dtype=np.float64)
    q = np.round(x / scale)
    return np.clip(q, -127, 127).astype(np.int64)


def int8_to_fp32(q: np.ndarray | int | list[int], scale: float) -> np.ndarray:
    """Dequantize signed INT8 values back to ``float32`` using ``scale``."""
    q = np.asarray(q, dtype=np.int64)
    return (q.astype(np.float64) * scale).astype(np.float32)


def fp32_to_int4(x: np.ndarray | float | list[float], scale: float) -> np.ndarray:
    """Symmetrically quantize to signed INT4 using ``scale`` (clamped to [-7, 7])."""
    x = np.asarray(x, dtype=np.float64)
    q = np.round(x / scale)
    return np.clip(q, -7, 7).astype(np.int64)


def int4_to_fp32(q: np.ndarray | int | list[int], scale: float) -> np.ndarray:
    """Dequantize signed INT4 values back to ``float32`` using ``scale``."""
    q = np.asarray(q, dtype=np.int64)
    return (q.astype(np.float64) * scale).astype(np.float32)
