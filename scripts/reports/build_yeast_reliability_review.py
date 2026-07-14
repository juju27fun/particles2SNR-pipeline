#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from particles2snr.yeast_review_reliability import build_reliability_review


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a stratified yeast review reliability subset.")
    parser.add_argument("--candidate-dataset", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = build_reliability_review(
        args.candidate_dataset,
        args.output_dir,
        fraction=args.fraction,
        seed=args.seed,
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
        "status": summary["status"],
        "outputs": [
            "manual_review_queue.csv",
            "manual_file_review_queue.csv",
            "reliability_review_summary.json",
        ],
    }
    (args.output_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
