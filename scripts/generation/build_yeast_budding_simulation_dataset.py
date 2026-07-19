#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from particles2snr.yeast_budding_dataset import (
    build_budding_calibration,
    build_budding_simulation_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build paired-view budding simulations from the approved M1/M2 "
            "development-train calibration."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fit-summaries-csv", type=Path, required=True)
    parser.add_argument("--real-dataset-root", type=Path, required=True)
    parser.add_argument(
        "--source-dataset-id",
        default="yeast-events-representation@v3",
    )
    parser.add_argument(
        "--generator",
        choices=("data", "biophysics"),
        required=True,
    )
    parser.add_argument("--train-latents", type=int, default=5000)
    parser.add_argument("--validation-latents", type=int, default=1000)
    parser.add_argument("--test-latents", type=int, default=1000)
    parser.add_argument("--views-per-latent", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7301)
    parser.add_argument("--amplitude-size-exponent", type=float, default=2.0)
    parser.add_argument("--beam-radius-relative", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    calibration = build_budding_calibration(
        fit_summaries_csv=args.fit_summaries_csv,
        real_dataset_root=args.real_dataset_root,
        source_dataset_id=args.source_dataset_id,
    )
    summary = build_budding_simulation_dataset(
        output_dir=args.output_dir,
        calibration=calibration,
        generator=args.generator,
        n_train_latents=args.train_latents,
        n_validation_latents=args.validation_latents,
        n_test_latents=args.test_latents,
        views_per_latent=args.views_per_latent,
        seed=args.seed,
        amplitude_size_exponent=args.amplitude_size_exponent,
        beam_radius_relative=args.beam_radius_relative,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
