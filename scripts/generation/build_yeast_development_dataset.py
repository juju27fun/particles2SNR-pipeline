#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from particles2snr.yeast_development_dataset import build_development_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a physical development-only yeast event representation dataset."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source-dataset-id",
        default="yeast-events-representation@v3",
    )
    parser.add_argument(
        "--output-dataset-id",
        default="yeast-events-development@v1",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = build_development_dataset(
        input_root=args.input_root,
        output_dir=args.output_dir,
        source_dataset_id=args.source_dataset_id,
        output_dataset_id=args.output_dataset_id,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
