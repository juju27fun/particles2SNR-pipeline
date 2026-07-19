#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from particles2snr.yeast_budding_dataset import (
    build_budding_calibration,
    build_budding_simulation_dataset,
)


CONFIGURATIONS = (
    (2.0, 0.5),
    (2.0, 1.0),
    (2.0, 2.0),
    (3.0, 0.5),
    (3.0, 1.0),
    (3.0, 2.0),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the frozen budding biophysics validation grid."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fit-summaries-csv", type=Path, required=True)
    parser.add_argument("--real-dataset-root", type=Path, required=True)
    parser.add_argument(
        "--source-dataset-id",
        default="yeast-events-representation@v3",
    )
    parser.add_argument("--validation-latents", type=int, default=128)
    parser.add_argument("--views-per-latent", type=int, default=2)
    parser.add_argument("--seed", type=int, default=18301)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    calibration = build_budding_calibration(
        fit_summaries_csv=args.fit_summaries_csv,
        real_dataset_root=args.real_dataset_root,
        source_dataset_id=args.source_dataset_id,
    )
    configurations = []
    for exponent, beam in CONFIGURATIONS:
        name = f"alpha-{exponent:g}_beam-{beam:g}"
        summary = build_budding_simulation_dataset(
            output_dir=args.output_dir / name,
            calibration=calibration,
            generator="biophysics",
            n_train_latents=1,
            n_validation_latents=args.validation_latents,
            n_test_latents=1,
            views_per_latent=args.views_per_latent,
            seed=args.seed,
            amplitude_size_exponent=exponent,
            beam_radius_relative=beam,
        )
        configurations.append(
            {
                "name": name,
                "amplitude_size_exponent": exponent,
                "beam_radius_relative": beam,
                "validation_signals": (
                    args.validation_latents * args.views_per_latent
                ),
                "generator_id": summary["generator_id"],
            }
        )
    summary = {
        "schema_version": 1,
        "dataset_id": "yeast-budding-biophysics-grid@v1",
        "purpose": (
            "validation-only selection of the relative contacting-double-sphere "
            "configuration before final test generation"
        ),
        "source_dataset": args.source_dataset_id,
        "calibration_id": calibration["calibration_id"],
        "selection_split": "development_validation",
        "test_split_used_for_selection": False,
        "configurations": configurations,
        "seed": args.seed,
    }
    (args.output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
