"""Tests for the toy MLP demo (demo.py)."""

from __future__ import annotations

import numpy as np

from quant_playground import demo, metrics
from quant_playground.quantizers import FP8_E4M3


def test_toy_mlp_forward_shape() -> None:
    model = demo.ToyMLP(seed=0)
    x = demo.build_demo_inputs(batch=16, in_dim=4, seed=1)
    out = model.forward(x)
    assert out.shape == (16, 2)


def test_quantized_weights_stay_close_for_higher_precision() -> None:
    model = demo.ToyMLP(seed=0)
    # FP8 keeps weights very close; outputs should barely move.
    qmodel = model.with_quantized_weights(FP8_E4M3)
    x = demo.build_demo_inputs(batch=16, in_dim=4, seed=1)
    ref = model.forward(x)
    qout = qmodel.forward(x)
    # cosine similarity must be extremely high for fp8-quantized weights
    assert metrics.cosine_similarity(ref, qout) > 0.99


def test_run_model_comparison_returns_all_formats() -> None:
    model = demo.ToyMLP(seed=0)
    x = demo.build_demo_inputs(seed=1)
    result = demo.run_model_comparison(model, x)
    assert len(result["rows"]) == 5
    names = {r["format"] for r in result["rows"]}
    assert names == {"fp8_e4m3", "fp8_e5m2", "fp4_e2m1", "int8", "int4"}
    assert result["best_format"] in names


def test_demo_outputs_remain_comparable() -> None:
    model = demo.ToyMLP(seed=2)
    x = demo.build_demo_inputs(seed=3)
    result = demo.run_model_comparison(model, x)
    # Every format keeps the network's outputs in the same direction (cosine
    # well above zero); the coarsest formats (FP4/INT4) naturally drift most.
    for row in result["rows"]:
        assert row["output_cosine"] > 0.5
    # The accuracy-preserving winner should be a high-precision format.
    assert result["best_format"] in {"fp8_e4m3", "fp8_e5m2", "int8"}


def test_demo_is_deterministic() -> None:
    model = demo.ToyMLP(seed=5)
    x = demo.build_demo_inputs(seed=6)
    r1 = demo.run_model_comparison(model, x)
    r2 = demo.run_model_comparison(model, x)
    assert r1["best_format"] == r2["best_format"]
    assert [round(r["output_mse"], 10) for r in r1["rows"]] == [
        round(r["output_mse"], 10) for r in r2["rows"]
    ]
