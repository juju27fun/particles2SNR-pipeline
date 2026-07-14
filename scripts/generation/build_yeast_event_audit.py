#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from particles2snr.yeast_event_audit import build_candidate_audit
from particles2snr.yeast_events import YeastDetectionConfig
from particles2snr.yeast_raw_data import read_raw_dataset_map


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build yeast event candidates and a stratified manual review queue.")
    parser.add_argument("--source-index", type=Path, required=True)
    raw = parser.add_mutually_exclusive_group(required=True)
    raw.add_argument("--raw-dataset-root", type=Path)
    raw.add_argument(
        "--raw-dataset-map",
        type=Path,
        help="JSON mapping registered raw dataset IDs to their resolved roots.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--review-crop-length", type=int, default=8192)
    parser.add_argument("--review-per-stratum", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-files", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = build_candidate_audit(
        source_index_csv=args.source_index,
        raw_dataset_root=args.raw_dataset_root,
        raw_dataset_roots=(
            read_raw_dataset_map(args.raw_dataset_map) if args.raw_dataset_map else None
        ),
        output_dir=args.output_dir,
        config=YeastDetectionConfig(),
        review_crop_length=args.review_crop_length,
        review_per_stratum=args.review_per_stratum,
        seed=args.seed,
        max_files=args.max_files,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
