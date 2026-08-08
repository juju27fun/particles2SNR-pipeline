#!/usr/bin/env python3
"""Build the immutable budding detector-audit gold inventory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATASET_ID = "yeast-budding-reviewed-event-inventory"
VERSION = "v1"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-run", type=Path, required=True)
    parser.add_argument("--blind-review", type=Path, required=True)
    parser.add_argument("--precision-review", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    workspace = Path(__file__).resolve().parents[3]
    analysis = (workspace / arguments.analysis_run).resolve() if not arguments.analysis_run.is_absolute() else arguments.analysis_run.resolve()
    blind = (workspace / arguments.blind_review).resolve() if not arguments.blind_review.is_absolute() else arguments.blind_review.resolve()
    precision = (workspace / arguments.precision_review).resolve() if not arguments.precision_review.is_absolute() else arguments.precision_review.resolve()
    output = (workspace / arguments.output_dir).resolve() if not arguments.output_dir.is_absolute() else arguments.output_dir.resolve()
    output.relative_to((workspace / "datasets/processed").resolve())
    if output.exists():
        raise FileExistsError(f"immutable dataset already exists: {output}")
    output.mkdir(parents=True)

    human = read_csv(analysis / "human_events.csv")
    candidates = read_csv(analysis / "detected_candidate_review.csv")
    blind_private = json.loads((blind / "review_private.json").read_text(encoding="utf-8"))
    source_by_case = {row["case_id"]: row for row in blind_private["cases"]}
    blind_receipt_sha = sha256(blind / "review/receipt.json")
    precision_receipt_sha = sha256(precision / "review/receipt.json")
    events = []
    for row in human:
        source = source_by_case[row["case_id"]]
        events.append(
            {
                **row,
                "dataset_partition": "detector_audit",
                "training_allowed": False,
                "raw_dataset_id": "yeast-hf-10-5-20260610@v1",
                "raw_signal_path": source["signal_path"],
                "raw_signal_sha256": source["signal_sha256"],
                "annotation_run_id": "yeast-budding-detector-blind-review-r2",
                "annotation_receipt_sha256": blind_receipt_sha,
            }
        )
    candidate_review = [
        {
            **row,
            "dataset_partition": "detector_audit_candidate_precision",
            "training_allowed": False,
            "raw_dataset_id": "yeast-hf-10-5-20260610@v1",
            "annotation_run_id": "yeast-budding-detector-precision-review-r1",
            "annotation_receipt_sha256": precision_receipt_sha,
        }
        for row in candidates
    ]
    if len(events) != 146 or len({row["human_event_id"] for row in events}) != 146:
        raise ValueError("gold event inventory must contain 146 unique human events")
    if len(candidate_review) != 20 or len({row["event_id"] for row in candidate_review}) != 20:
        raise ValueError("candidate precision table must contain 20 unique proposals")
    if any(str(row["training_allowed"]).lower() != "false" for row in events + candidate_review):
        raise ValueError("detector audit rows must be training-forbidden")

    write_csv(output / "events.csv", events)
    write_csv(output / "candidate_review.csv", candidate_review)
    contract = {
        "schema_version": 1,
        "dataset_id": f"{DATASET_ID}@{VERSION}",
        "grain": {"events.csv": "one certain human budding event", "candidate_review.csv": "one independently reviewed strict detector proposal"},
        "primary_keys": {"events.csv": ["human_event_id"], "candidate_review.csv": ["event_id"]},
        "partitions": ["detector_audit", "detector_audit_candidate_precision"],
        "training_allowed": False,
        "units": {"start_sample": "sample", "end_sample": "sample", "duration_ms": "ms", "snr_proxy": "detector robust z proxy"},
        "compatibility": {"n_minus_1": "initial version; additive columns only for future compatible revisions"},
        "sealed_holdout_accessed": False,
    }
    (output / "dataset-contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    provenance = {
        "dataset_id": f"{DATASET_ID}@{VERSION}",
        "source_dataset_ids": ["yeast-hf-10-5-20260610@v1", "yeast-event-candidates@v7"],
        "source_run_ids": ["yeast-budding-detector-audit-method-r2", "yeast-budding-detector-blind-review-r2", "yeast-budding-detector-precision-review-r1", "yeast-budding-detector-audit-analysis-r1"],
        "annotation_receipts": {"blind_120": blind_receipt_sha, "precision_20": precision_receipt_sha},
        "detector_preset": "review-calibrated-v1",
        "matching_rule": "strict candidate center inclusively inside human particle interval",
        "thresholds_tuned_after_review": False,
        "sealed_holdout_accessed": False,
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload_files = ["events.csv", "candidate_review.csv", "dataset-contract.json", "provenance.json"]
    manifest = {
        "schema_version": 1,
        "dataset_id": f"{DATASET_ID}@{VERSION}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "producer": "particles2SNR-pipeline/scripts/generation/build_yeast_budding_reviewed_event_inventory.py",
        "row_counts": {"events.csv": len(events), "candidate_review.csv": len(candidate_review)},
        "files": [{"path": name, "size": (output / name).stat().st_size, "sha256": sha256(output / name)} for name in payload_files],
    }
    (output / "dataset-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
