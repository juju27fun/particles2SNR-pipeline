#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from particles2snr.yeast_review_analysis import ReviewGateThresholds, analyze_review


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate yeast event-review annotations and evaluate Gate 1.")
    parser.add_argument("--candidate-dataset", type=Path, required=True)
    parser.add_argument(
        "--review-dir",
        type=Path,
        help="Directory containing completed queue CSV copies; never edit a registered candidate dataset",
    )
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--require-complete", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = analyze_review(args.candidate_dataset, ReviewGateThresholds(), args.review_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "review_analysis.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    repo_root = Path(__file__).resolve().parents[2]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True
    ).stdout.strip()
    run = {
        "schema_version": 1,
        "project": "particles2SNR-pipeline",
        "run_id": args.run_id,
        "dataset": args.dataset_id,
        "repositories": {"particles2SNR-pipeline": revision},
        "command": " ".join(sys.argv),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "outputs": ["review_analysis.json"],
    }
    (args.output_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.require_complete and summary["event_review_status"] == "pending":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
