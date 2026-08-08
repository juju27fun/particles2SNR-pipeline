#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from particles2snr.yeast_4class_dataset import build_dataset
from particles2snr.yeast_raw_data import read_raw_dataset_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the immutable four-class yeast classification dataset.")
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--exclude-csv", type=Path, action="append", default=[])
    raw = parser.add_mutually_exclusive_group(required=True)
    raw.add_argument("--raw-dataset-root", type=Path)
    raw.add_argument("--raw-dataset-map", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--dataset-id", default="yeast-budding-mix-shmoo-background-classification@v1")
    parser.add_argument("--run-id", default="yeast-4class-classification-dataset-build-r1")
    parser.add_argument("--method-evidence-id", default="yeast-4class-conv1dgap-latent-method-r1")
    parser.add_argument("--include-shmoo1", action="store_true", help="Map both historical shmoo and shmoo2 to the final shmoo class.")
    parser.add_argument(
        "--background-source",
        action="append",
        choices=("budding", "mix", "shmoo", "shmoo2"),
        help="Source used for clean backgrounds; repeat to override the default of all mapped sources.",
    )
    args = parser.parse_args()
    summary = build_dataset(
        candidate_csv=args.candidate_csv,
        exclusion_csvs=args.exclude_csv,
        raw_dataset_root=args.raw_dataset_root,
        raw_dataset_roots=read_raw_dataset_map(args.raw_dataset_map) if args.raw_dataset_map else None,
        output_dir=args.output_dir,
        seed=args.seed,
        dataset_id=args.dataset_id,
        run_id=args.run_id,
        method_evidence_id=args.method_evidence_id,
        source_to_class={"budding": "budding", "mix": "mix", "shmoo": "shmoo", "shmoo2": "shmoo"} if args.include_shmoo1 else {"budding": "budding", "mix": "mix", "shmoo2": "shmoo"},
        background_sources=tuple(args.background_source) if args.background_source else None,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
