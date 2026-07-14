#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from particles2snr.yeast_review_figures import render_review_figures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render candidate-precision and full-trace-recall yeast review PDFs.")
    parser.add_argument("--candidate-dataset", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = render_review_figures(args.candidate_dataset, args.output_dir)
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
        "status": "awaiting_manual_annotation",
        "outputs": [
            summary["candidate_pdf"],
            summary["full_trace_pdf"],
            summary["candidate_annotation_csv"],
            summary["full_trace_annotation_csv"],
            "review_figure_summary.json",
        ],
    }
    (args.output_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
