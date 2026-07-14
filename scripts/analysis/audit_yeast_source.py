#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from particles2snr.yeast_source_audit import inventory_source, summarize_inventory, write_inventory


def _revision(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inventory raw yeast acquisition files and exact duplicates.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--documented-acquisition-group",
        action="append",
        default=[],
        help="Independent acquisition ID supported by source metadata; repeat for each group.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty audit directory: {output_dir}")

    records = inventory_source(args.source_root)
    summary = summarize_inventory(
        records,
        source_root=args.source_root,
        documented_acquisition_groups=args.documented_acquisition_group,
    )
    write_inventory(output_dir, records, summary)

    repo_root = Path(__file__).resolve().parents[2]
    command = " ".join(sys.argv)
    run = {
        "schema_version": 1,
        "project": "particles2SNR-pipeline",
        "run_id": args.run_id,
        "dataset": "unregistered-yeast-source",
        "repositories": {"particles2SNR-pipeline": _revision(repo_root)},
        "command": command,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "outputs": ["source_inventory.csv", "exact_duplicates.csv", "source_inventory_summary.json"],
        "gate": {"gate_0_provenance": "fail", "gate_1_split": summary["split_readiness"]["status"]},
    }
    (output_dir / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
