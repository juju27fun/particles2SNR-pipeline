#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from internship_workspace.config import Workspace
from internship_workspace.datasets import load_records
from particles2snr.equation_roundtrip import sha256_file
from particles2snr.p0_2500_roundtrip import (
    CHECKPOINT_SHA256,
    DATASET_ID,
    METHOD_EVIDENCE_ID,
    P0_DATASET_ID,
    RAW_DATASET_ID,
    build_candidate,
    validate_candidate,
)


DEFAULT_CHECKPOINT = Path(
    "artifacts/SMI_CNN_limitations/training/output/"
    "Conv1D-dataset_3c-rerun_report/best_model.pth"
)
DEFAULT_DETECTOR_PARTICLES = Path(
    "artifacts/particles2SNR-pipeline/runs/"
    "p0_c1_Particles2SNR_F/train/snr_particles.csv"
)


def _record(workspace: Workspace, key: str) -> dict[str, Any]:
    match = next(
        (
            record.payload
            for record in load_records(workspace)
            if f"{record.payload['id']}@{record.payload['version']}" == key
        ),
        None,
    )
    if match is None or match["status"] not in {"active", "reference"}:
        raise ValueError(f"eligible registered dataset not found: {key}")
    return match


def _git_state(path: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the approved p0 2500-sample roundtrip candidate."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--detector-particles",
        type=Path,
        default=DEFAULT_DETECTOR_PARTICLES,
    )
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--maximum-parents", type=int)
    args = parser.parse_args()

    workspace = Workspace.load()
    p0_record = _record(workspace, P0_DATASET_ID)
    raw_record = _record(workspace, RAW_DATASET_ID)
    checkpoint = (workspace.root / args.checkpoint).resolve()
    detector_particles = (workspace.root / args.detector_particles).resolve()
    if sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("frozen legacy classifier checkpoint hash mismatch")
    if args.artifact_dir.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {args.artifact_dir}")
    evidence = build_candidate(
        p0_dataset_root=workspace.datasets_root / p0_record["path"],
        raw_dataset_root=workspace.datasets_root / raw_record["path"],
        detector_particles_csv=detector_particles,
        output_dir=args.output_dir,
        p0_manifest_sha256=p0_record["manifest_sha256"],
        raw_manifest_sha256=raw_record["manifest_sha256"],
        checkpoint_sha256=CHECKPOINT_SHA256,
        seed=args.seed,
        maximum_parents=args.maximum_parents,
    )
    validation = validate_candidate(args.output_dir)
    args.artifact_dir.mkdir(parents=True)
    payload = {"build": evidence, "validation": validation}
    (args.artifact_dir / "build_evidence.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    run = {
        "schema_version": 1,
        "project": "particles2SNR-pipeline",
        "run_id": args.artifact_dir.name,
        "dataset": DATASET_ID,
        "datasets": {
            P0_DATASET_ID: {
                "id": P0_DATASET_ID,
                "manifest_sha256": p0_record["manifest_sha256"],
            },
            RAW_DATASET_ID: {
                "id": RAW_DATASET_ID,
                "manifest_sha256": raw_record["manifest_sha256"],
            },
        },
        "command": (
            "particles2SNR-pipeline/scripts/generation/"
            "build_p0_2500_roundtrip_candidate.py"
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "repositories": {
            "workspace": _git_state(workspace.root),
            "particles2SNR-pipeline": _git_state(
                workspace.root / "particles2SNR-pipeline"
            ),
        },
        "outputs": ["build_evidence.json"],
        "method_evidence_id": METHOD_EVIDENCE_ID,
        "detector_particles_csv_sha256": sha256_file(detector_particles),
        "classifier_checkpoint_sha256": CHECKPOINT_SHA256,
        "sealed_test_accessed": False,
        "claim_boundary": evidence["claim_boundary"],
    }
    (args.artifact_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
