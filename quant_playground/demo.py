"""A tiny numpy MLP used to demonstrate quantization's impact on accuracy.

The network is intentionally small and fully deterministic (seeded) so the
demo is reproducible on CPU without any model downloads.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class ToyMLP:
    """A 2-layer MLP: ``tanh(hidden)`` then a linear output.

    Weights are stored as ``float32`` and can be swapped for quantized copies
    to measure how much a low-precision format shifts the network outputs.
    """

    def __init__(
        self,
        in_dim: int = 4,
        hidden: int = 8,
        out_dim: int = 2,
        seed: int = 0,
    ) -> None:
        self.in_dim = in_dim
        self.hidden = hidden
        self.out_dim = out_dim
        rng = np.random.default_rng(seed)
        # Scale down so activations stay in a comfortable, representable range.
        self.w1 = (rng.standard_normal((hidden, in_dim)) * 0.3).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.w2 = (rng.standard_normal((out_dim, hidden)) * 0.3).astype(np.float32)
        self.b2 = np.zeros(out_dim, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for a batch ``x`` of shape ``(batch, in_dim)``."""
        x = np.asarray(x, dtype=np.float32)
        h = np.tanh(x @ self.w1.T + self.b1)
        return h @ self.w2.T + self.b2

    def with_quantized_weights(self, quantizer: Any) -> "ToyMLP":
        """Return a clone whose weights have been quantized then dequantized."""
        clone = ToyMLP(self.in_dim, self.hidden, self.out_dim)
        clone.w1 = quantizer.dequantize(quantizer.quantize(self.w1))
        clone.b1 = self.b1
        clone.w2 = quantizer.dequantize(quantizer.quantize(self.w2))
        clone.b2 = self.b2
        return clone


def build_demo_inputs(batch: int = 16, in_dim: int = 4, seed: int = 1) -> np.ndarray:
    """Deterministic random inputs the demo runs the MLP on."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((batch, in_dim)) * 0.5).astype(np.float32)


def run_model_comparison(model: ToyMLP, inputs: np.ndarray) -> dict[str, Any]:
    """Quantize the model's weights per format and report output delta.

    Returns a dict with the reference outputs and a per-format list of metrics,
    including which format preserves accuracy best (lowest output MSE).
    """
    from . import metrics as _metrics
    from .quantizers import all_quantizers

    reference = model.forward(inputs)
    rows: list[dict[str, float]] = []
    for qz in all_quantizers():
        qmodel = model.with_quantized_weights(qz)
        outputs = qmodel.forward(inputs)
        rows.append(
            {
                "format": qz.name,
                "output_mse": _metrics.mse(reference, outputs),
                "output_cosine": _metrics.cosine_similarity(reference, outputs),
                "weight_mse": _metrics.mse(
                    np.concatenate([model.w1.ravel(), model.w2.ravel()]),
                    np.concatenate([qmodel.w1.ravel(), qmodel.w2.ravel()]),
                ),
                "compression_ratio": qz.bytes_per_element and (4.0 / qz.bytes_per_element),
            }
        )

    best = min(rows, key=lambda r: r["output_mse"])
    return {
        "reference_outputs": reference,
        "rows": rows,
        "best_format": best["format"],
    }
