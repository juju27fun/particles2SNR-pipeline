#!/usr/bin/env python3
"""Apply saturation GT arbitration and detector-consensus box expansions."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from internship_workspace.config import Workspace
from internship_workspace.datasets import (
    register_record,
    resolve_path,
    select_record,
)
from internship_workspace.saturation_gt_review import current_decisions
from particles2snr.saturation_gt_dataset import (
    apply_reviewed_labels,
    write_application_metadata,
)


PARENT_DATASET = (
    "particles2snr-f-dual-clean-c1-yolo-4class-saturation-candidate@v1"
)
OUTPUT_DATASET_ID = (
    "particles2snr-f-dual-clean-c1-yolo-4class-saturation-reviewed"
)
OUTPUT_VERSION = "v1"
OUTPUT_RELATIVE = (
    "interim/particles2snr-f-dual-clean-c1-yolo-4class-"
    "saturation-reviewed/v1"
)
SESSION_RELATIVE = (
    "artifacts/particles2SNR-pipeline/audits/"
    "dual-clean-saturation-gt-review-v1-jlb"
)
PROPOSAL_RELATIVE = (
    "artifacts/SMI_Detection_CNN_transformers/research/"
    "saturation-disputed-box-inference-r1"
)
RUN_RELATIVE = (
    "artifacts/particles2SNR-pipeline/runs/"
    "dual_clean_saturation_reviewed_20260718"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _revision(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _read_proposals(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {row["candidate_id"]: row for row in rows}


def main() -> None:
    workspace = Workspace.load()
    parent_id, parent_version = PARENT_DATASET.rsplit("@", 1)
    parent_record = select_record(workspace, parent_id, parent_version)
    parent_root = resolve_path(workspace, parent_record)
    output_root = workspace.datasets_root / OUTPUT_RELATIVE
    session_dir = workspace.root / SESSION_RELATIVE
    proposal_dir = workspace.root / PROPOSAL_RELATIVE
    run_dir = workspace.root / RUN_RELATIVE
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run: {run_dir}")
    proposal_run = json.loads(
        (proposal_dir / "run.json").read_text(encoding="utf-8")
    )
    if proposal_run["status"] != "awaiting_visual_review":
        raise RuntimeError("Unexpected detector proposal status")
    queue = json.loads(
        (session_dir / "queue_snapshot.json").read_text(encoding="utf-8")
    )
    decisions = current_decisions(session_dir)
    proposals = _read_proposals(proposal_dir / "box_proposals.csv")
    disputed = {
        candidate_id
        for candidate_id, row in decisions.items()
        if row["decision"] == "needs_review"
    }
    if disputed != set(proposals):
        raise RuntimeError(
            "Detector proposals do not exactly cover the disputed decisions"
        )
    if any(
        row.get("geometry_only") != "True"
        or row.get("all_seeds_detected") != "True"
        for row in proposals.values()
    ):
        raise RuntimeError("Detector proposal contract is incomplete")

    application_rows = apply_reviewed_labels(
        source_root=parent_root,
        output_root=output_root,
        queue=queue,
        decisions=decisions,
        proposals=proposals,
    )
    summary = write_application_metadata(
        output_root=output_root,
        application_rows=application_rows,
        parent_dataset=PARENT_DATASET,
        parent_manifest_sha256=parent_record.payload["manifest_sha256"],
        review_session=SESSION_RELATIVE,
        review_queue_sha256=queue["queue_sha256"],
        proposal_artifact=PROPOSAL_RELATIVE,
        proposal_run_sha256=_sha256(proposal_dir / "run.json"),
    )
    expected = {
        "delete": 172,
        "keep": 17,
        "expand_detector_consensus": 4,
    }
    if summary["actions"] != expected:
        raise RuntimeError(
            f"Unexpected application counts: {summary['actions']} != {expected}"
        )
    record = register_record(
        workspace,
        dataset_id=OUTPUT_DATASET_ID,
        version=OUTPUT_VERSION,
        relative_path=OUTPUT_RELATIVE,
        status="reference",
        producer="particles2SNR-pipeline",
        data_format="yolo-1d-saturation-reviewed-reference",
        command=(
            "build_saturation_reviewed_dataset.py; apply complete saturation "
            "GT review and four detector-consensus wider boxes"
        ),
    )
    run_dir.mkdir(parents=True)
    run: dict[str, Any] = {
        "schema_version": 1,
        "project": "particles2SNR-pipeline",
        "run_id": run_dir.name,
        "dataset": f"{OUTPUT_DATASET_ID}@{OUTPUT_VERSION}",
        "command": (
            "particles2SNR-pipeline/scripts/generation/"
            "build_saturation_reviewed_dataset.py"
        ),
        "parent_dataset": PARENT_DATASET,
        "repositories": {
            "workspace": _revision(workspace.root),
            "particles2SNR-pipeline": _revision(
                workspace.root / "particles2SNR-pipeline"
            ),
            "SMI_Detection_CNN_transformers": _revision(
                workspace.root / "SMI_Detection_CNN_transformers"
            ),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_reference_candidate",
        "scientific_scope": (
            "Review decisions applied; disputed box geometry comes from "
            "three-seed detector consensus, while human 10um classes are preserved."
        ),
        "inputs": {
            "parent_manifest_sha256": parent_record.payload["manifest_sha256"],
            "decision_journal_sha256": _sha256(
                session_dir / "decisions.jsonl"
            ),
            "proposal_run_sha256": _sha256(proposal_dir / "run.json"),
            "proposal_csv_sha256": _sha256(
                proposal_dir / "box_proposals.csv"
            ),
        },
        "application_summary": summary,
        "output_manifest_sha256": record.payload["manifest_sha256"],
        "outputs": [
            "datasets/registry/"
            f"{OUTPUT_DATASET_ID}-{OUTPUT_VERSION}.jsonl",
            f"datasets/{OUTPUT_RELATIVE}/review_application.csv",
            f"datasets/{OUTPUT_RELATIVE}/review_application_summary.json",
        ],
    }
    (run_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(run, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
