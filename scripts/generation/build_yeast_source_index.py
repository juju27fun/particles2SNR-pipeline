#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from particles2snr.yeast_source_index import (
    build_source_index,
    read_source_inventory,
    summarize_source_index,
    write_source_index,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a deduplicated yeast source index with honest split scope.")
    parser.add_argument("--source-inventory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raw-dataset", required=True)
    parser.add_argument("--acquisition-id", required=True)
    parser.add_argument("--capture-block-size", type=int, default=64)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = build_source_index(
        read_source_inventory(args.source_inventory),
        raw_dataset=args.raw_dataset,
        acquisition_id=args.acquisition_id,
        capture_block_size=args.capture_block_size,
    )
    summary = summarize_source_index(rows)
    write_source_index(args.output_dir, rows, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
