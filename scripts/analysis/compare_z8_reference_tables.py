#!/usr/bin/env python3
"""Create the manifested, diagnostic-only Z8 v1→v2 event comparison."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from particles2snr.z8_comparison import write_comparison
from particles2snr.z8_reference_dataset import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-events", type=Path, required=True)
    parser.add_argument("--new-events", type=Path, required=True)
    parser.add_argument("--old-dataset-id", required=True)
    parser.add_argument("--new-dataset-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = write_comparison(
        old_events_path=args.old_events,
        new_events_path=args.new_events,
        output_dir=args.output_dir,
        old_dataset_id=args.old_dataset_id,
        new_dataset_id=args.new_dataset_id,
    )
    run = {
        "schema_version": 1,
        "project": "particles2SNR-pipeline",
        "run_id": args.output_dir.name,
        "kind": "analysis",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "inputs": {
            str(args.old_events): sha256_file(args.old_events),
            str(args.new_events): sha256_file(args.new_events),
        },
        "summary": summary,
        "outputs": ["event_delta.csv", "event_delta_summary.json"],
    }
    (args.output_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
