#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from particles2snr.yeast_simulation import (
    build_simulation_dataset,
    build_support_calibrated_simulation_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build paired-view identifiable yeast passage simulations.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-latents", type=int, default=5000)
    parser.add_argument("--validation-latents", type=int, default=1000)
    parser.add_argument("--test-latents", type=int, default=1000)
    parser.add_argument("--views-per-latent", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--envelope-model",
        choices=("gaussian", "finite_support_tukey"),
        default="gaussian",
    )
    parser.add_argument("--real-calibration-root", type=Path)
    parser.add_argument("--tukey-alpha", type=float, default=0.50)
    parser.add_argument("--quantile-knots", type=int, default=101)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    common = {
        "output_dir": args.output_dir,
        "n_train_latents": args.train_latents,
        "n_validation_latents": args.validation_latents,
        "n_test_latents": args.test_latents,
        "views_per_latent": args.views_per_latent,
        "seed": args.seed,
    }
    if args.envelope_model == "finite_support_tukey":
        if args.real_calibration_root is None:
            raise SystemExit("--real-calibration-root is required for finite-support generation")
        summary = build_support_calibrated_simulation_dataset(
            real_root=args.real_calibration_root,
            tukey_alpha=args.tukey_alpha,
            quantile_knots=args.quantile_knots,
            **common,
        )
    else:
        if args.real_calibration_root is not None:
            raise SystemExit("--real-calibration-root is only valid for finite-support generation")
        summary = build_simulation_dataset(**common)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
