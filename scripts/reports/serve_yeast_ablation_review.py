#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from particles2snr.yeast_ablation_review_app import serve_ablation_review


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve the editable yeast detector ablation disagreement review."
    )
    parser.add_argument("--raw-dataset-root", type=Path, required=True)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    serve_ablation_review(
        raw_dataset_root=args.raw_dataset_root,
        review_dir=args.review_dir,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
