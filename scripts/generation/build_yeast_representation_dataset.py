#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from particles2snr.yeast_representation_dataset import build_representation_dataset
from particles2snr.yeast_raw_data import read_raw_dataset_map


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the frozen yeast event representation input.")
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--candidate-dataset-id", required=True)
    raw = parser.add_mutually_exclusive_group(required=True)
    raw.add_argument("--raw-dataset-root", type=Path)
    raw.add_argument(
        "--raw-dataset-map",
        type=Path,
        help="JSON mapping registered raw dataset IDs to their resolved roots.",
    )
    parser.add_argument("--raw-dataset-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = build_representation_dataset(
        candidate_csv=args.candidate_csv,
        raw_dataset_root=args.raw_dataset_root,
        raw_dataset_roots=(
            read_raw_dataset_map(args.raw_dataset_map) if args.raw_dataset_map else None
        ),
        output_dir=args.output_dir,
        raw_dataset_id=args.raw_dataset_id,
        candidate_dataset_id=args.candidate_dataset_id,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
