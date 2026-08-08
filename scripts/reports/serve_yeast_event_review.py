#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from particles2snr.yeast_review_app import serve_review


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve the local yeast candidate and full-trace annotation interface."
    )
    parser.add_argument("--candidate-dataset", type=Path, required=True)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="Disable all annotation writes and hide the editing form.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    serve_review(
        args.candidate_dataset,
        args.review_dir,
        host=args.host,
        port=args.port,
        read_only=args.read_only,
    )


if __name__ == "__main__":
    main()
