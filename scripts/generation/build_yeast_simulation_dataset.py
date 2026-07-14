#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from particles2snr.yeast_simulation import build_simulation_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build paired-view identifiable yeast passage simulations.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-latents", type=int, default=5000)
    parser.add_argument("--validation-latents", type=int, default=1000)
    parser.add_argument("--test-latents", type=int, default=1000)
    parser.add_argument("--views-per-latent", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = build_simulation_dataset(
        output_dir=args.output_dir,
        n_train_latents=args.train_latents,
        n_validation_latents=args.validation_latents,
        n_test_latents=args.test_latents,
        views_per_latent=args.views_per_latent,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
