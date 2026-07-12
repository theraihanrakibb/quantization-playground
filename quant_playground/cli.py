"""Command-line interface for quant-playground.

Subcommands:

* ``quantize`` - quantize a synthetic tensor and print a round-trip report.
* ``demo``    - run the toy MLP comparison and print an accuracy-impact table.

Run as a module: ``python -m quant_playground.cli <subcommand> ...``.
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np

from . import metrics as _metrics
from . import demo as _demo
from .quantizers import get_quantizer

logger = logging.getLogger(__name__)


def _build_tensor(args: argparse.Namespace) -> np.ndarray:
    shape = tuple(int(s) for s in args.shape.split(","))
    rng = np.random.default_rng(args.seed)
    if args.tensor == "random":
        return (rng.standard_normal(shape) * args.scale).astype(np.float32)
    if args.tensor == "linspace":
        return np.linspace(-args.scale, args.scale, int(np.prod(shape))).reshape(shape).astype(
            np.float32
        )
    if args.tensor == "ones":
        return np.ones(shape, dtype=np.float32)
    raise ValueError(f"unknown tensor kind '{args.tensor}'")


def _cmd_quantize(args: argparse.Namespace) -> int:
    quantizer = get_quantizer(args.format, args.bits)
    x = _build_tensor(args)
    report = _metrics.round_trip_report(x, quantizer)
    print(f"Format : {report['format']} ({quantizer.bits}-bit)")
    print(f"Shape  : {tuple(x.shape)}  elements={int(x.size)}")
    print(f"MSE    : {report['mse']:.6e}")
    print(f"MAE    : {report['mae']:.6e}")
    print(f"SNR    : {report['snr_db']:.3f} dB")
    print(f"Cosine : {report['cosine']:.6f}")
    print(f"Bytes  : {report['original_bytes']} -> {report['compressed_bytes']:.1f}")
    print(f"Ratio  : {report['compression_ratio']:.3f}x smaller")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    model = _demo.ToyMLP(seed=args.seed)
    inputs = _demo.build_demo_inputs(seed=args.seed + 1)
    result = _demo.run_model_comparison(model, inputs)

    if args.format:
        rows = [r for r in result["rows"] if r["format"] == args.format]
        if not rows:
            print(f"Format '{args.format}' not found.", file=sys.stderr)
            return 2
    else:
        rows = result["rows"]

    header = f"{'format':<11} {'out_mse':>12} {'out_cos':>9} {'w_mse':>12} {'ratio':>7}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['format']:<11} {r['output_mse']:>12.4e} "
            f"{r['output_cosine']:>9.5f} {r['weight_mse']:>12.4e} {r['compression_ratio']:>6.2f}x"
        )
    print()
    print(f"Best accuracy-preserving format: {result['best_format']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quant-playground",
        description="Educational quantization playground for low-precision formats.",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: WARNING).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    q = sub.add_parser("quantize", help="Quantize a synthetic tensor and report error/size.")
    q.add_argument("--bits", type=int, default=8, help="Bit width (4 or 8).")
    q.add_argument(
        "--format",
        default="fp",
        help="Format family: 'fp'/'float' or 'int'/'integer', or an explicit name.",
    )
    q.add_argument(
        "--tensor",
        default="random",
        choices=["random", "linspace", "ones"],
        help="How to synthesize the input tensor.",
    )
    q.add_argument("--shape", default="1024", help="Comma-separated tensor shape.")
    q.add_argument("--scale", type=float, default=3.0, help="Magnitude scale of the tensor.")
    q.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility.")
    q.set_defaults(func=_cmd_quantize)

    d = sub.add_parser("demo", help="Run the toy MLP comparison and print a table.")
    d.add_argument(
        "--format",
        default=None,
        help="Restrict the table to a single format (default: all).",
    )
    d.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility.")
    d.set_defaults(func=_cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level), format="%(levelname)s %(name)s: %(message)s"
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
