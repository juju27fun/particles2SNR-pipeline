#!/usr/bin/env python3
"""Promote the visually approved saturation-GT reference to an active version."""

from __future__ import annotations

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
    validate_record,
)
from particles2snr.saturation_gt_dataset import promote_reviewed_dataset


SOURCE_DATASET = (
    "particles2snr-f-dual-clean-c1-yolo-4class-saturation-reviewed@v1"
)
OUTPUT_DATASET_ID = (
    "particles2snr-f-dual-clean-c1-yolo-4class-saturation-reviewed"
)
OUTPUT_VERSION = "v2"
OUTPUT_RELATIVE = (
    "processed/particles2snr-f-dual-clean-c1-yolo-4class-"
    "saturation-reviewed/v2"
)
PROPOSAL_RELATIVE = (
    "artifacts/SMI_Detection_CNN_transformers/research/"
    "saturation-disputed-box-inference-r1"
)
RUN_RELATIVE = (
    "artifacts/particles2SNR-pipeline/runs/"
    "dual_clean_saturation_reviewed_active_20260718"
)
EVIDENCE_FILENAMES = (
    "HFocusing_5_10_10um_0_1627.png",
    "HFocusing_5_10_10um_0_1641.png",
    "HFocusing_5_10_10um_0_90.png",
    "HFocusing_5_10_10um_0_1769.png",
)
BOUNDARY_TRUNCATED = (
    "train:HFocusing_5_10_10um_0_1627:0",
)


def _revision(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def main() -> None:
    workspace = Workspace.load()
    source_id, source_version = SOURCE_DATASET.rsplit("@", 1)
    source_record = select_record(workspace, source_id, source_version)
    errors = validate_record(workspace, source_record, full=True)
    if errors:
        raise RuntimeError("Invalid source dataset:\n" + "\n".join(errors))
    if source_record.payload["status"] != "reference":
        raise RuntimeError("Source dataset must remain a reference version")

    source_root = resolve_path(workspace, source_record)
    output_root = workspace.datasets_root / OUTPUT_RELATIVE
    proposal_root = workspace.root / PROPOSAL_RELATIVE
    run_dir = workspace.root / RUN_RELATIVE
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run: {run_dir}")
    evidence = [
        (
            Path(PROPOSAL_RELATIVE)
            / "plots"
            / filename
        ).as_posix()
        for filename in EVIDENCE_FILENAMES
    ]
    missing = [
        relative
        for relative in evidence
        if not (workspace.root / relative).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing validation evidence: {missing}")
    proposal_run = json.loads(
        (proposal_root / "run.json").read_text(encoding="utf-8")
    )
    if proposal_run["status"] != "awaiting_visual_review":
        raise RuntimeError("Unexpected proposal artifact status")

    created_at = datetime.now(timezone.utc).isoformat()
    promotion = promote_reviewed_dataset(
        source_root=source_root,
        output_root=output_root,
        source_dataset=SOURCE_DATASET,
        source_manifest_sha256=source_record.payload["manifest_sha256"],
        reviewer="jlb",
        validated_at=created_at,
        evidence_plots=evidence,
        boundary_truncated_candidates=list(BOUNDARY_TRUNCATED),
    )
    record = register_record(
        workspace,
        dataset_id=OUTPUT_DATASET_ID,
        version=OUTPUT_VERSION,
        relative_path=OUTPUT_RELATIVE,
        status="active",
        producer="particles2SNR-pipeline",
        data_format="yolo-1d-saturation-reviewed-active",
        command=(
            "promote_saturation_reviewed_dataset.py; promote the visually "
            "approved four-box review without mutating the reference version"
        ),
    )
    run_dir.mkdir(parents=True)
    run: dict[str, Any] = {
        "schema_version": 1,
        "project": "particles2SNR-pipeline",
        "run_id": run_dir.name,
        "created_at": created_at,
        "status": "complete_active_dataset",
        "dataset": f"{OUTPUT_DATASET_ID}@{OUTPUT_VERSION}",
        "command": (
            "particles2SNR-pipeline/scripts/generation/"
            "promote_saturation_reviewed_dataset.py"
        ),
        "source_dataset": SOURCE_DATASET,
        "repositories": {
            "workspace": _revision(workspace.root),
            "particles2SNR-pipeline": _revision(
                workspace.root / "particles2SNR-pipeline"
            ),
        },
        "validation": promotion,
        "output_manifest_sha256": record.payload["manifest_sha256"],
        "outputs": [
            "datasets/registry/"
            f"{OUTPUT_DATASET_ID}-{OUTPUT_VERSION}.jsonl",
            f"datasets/{OUTPUT_RELATIVE}/dataset.yaml",
            f"datasets/{OUTPUT_RELATIVE}/visual_validation.json",
        ],
    }
    (run_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(run, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
