#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from particles2snr.yeast_event_audit import build_candidate_audit
from particles2snr.yeast_events import (
    YeastDetectionConfig,
    review_calibrated_detection_config_v1,
)
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
    parser.add_argument(
        "--file-review-per-stratum",
        type=int,
        help="Full-trace rows per stratum; defaults to --review-per-stratum.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument(
        "--config-preset",
        choices=("default", "review-calibrated-v1"),
        default="default",
        help="Named detector configuration; calibrated presets are development results.",
    )
    parser.add_argument(
        "--review-exclusion-csv",
        action="append",
        default=[],
        type=Path,
        help="CSV containing record_id; may be repeated to prevent review-set reuse.",
    )
    return parser


def _read_excluded_record_ids(paths: list[Path]) -> set[str]:
    record_ids: set[str] = set()
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "record_id" not in reader.fieldnames:
                raise ValueError(f"Review exclusion CSV has no record_id column: {path}")
            record_ids.update(row["record_id"] for row in reader if row["record_id"])
    return record_ids


def main() -> None:
    args = build_parser().parse_args()
    config = (
        YeastDetectionConfig()
        if args.config_preset == "default"
        else review_calibrated_detection_config_v1()
    )
    summary = build_candidate_audit(
        source_index_csv=args.source_index,
        raw_dataset_root=args.raw_dataset_root,
        raw_dataset_roots=(
            read_raw_dataset_map(args.raw_dataset_map) if args.raw_dataset_map else None
        ),
        output_dir=args.output_dir,
        config=config,
        review_crop_length=args.review_crop_length,
        review_per_stratum=args.review_per_stratum,
        file_review_per_stratum=args.file_review_per_stratum,
        seed=args.seed,
        max_files=args.max_files,
        review_excluded_record_ids=_read_excluded_record_ids(args.review_exclusion_csv),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
