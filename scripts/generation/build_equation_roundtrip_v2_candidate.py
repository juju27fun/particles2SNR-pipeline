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
from particles2snr.equation_roundtrip import CHECKPOINT_SHA256, sha256_file
from particles2snr.equation_roundtrip_v2 import (
    DATASET_ID,
    METHOD_EVIDENCE_ID,
    SIGNAL_DATASET_ID,
    SOURCE_DATASET_ID,
    build_detector_faithful_candidate,
    validate_detector_faithful_candidate,
)


DEFAULT_CHECKPOINT = Path(
    "artifacts/unsupervised-learning-flow-cytometry/"
    "pretrained_backbones-10dB/"
    "particles2snr_f_3class_native_params_moment_patchtst_conv1dgap/"
    "conv1dgap_same_input_3class/best_model.pt"
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
        description="Build the approved detector-faithful equation candidate."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--source-dataset", default=SOURCE_DATASET_ID)
    parser.add_argument("--signal-dataset", default=SIGNAL_DATASET_ID)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--detector-particles",
        type=Path,
        action="append",
    )
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--maximum-events", type=int)
    args = parser.parse_args()

    workspace = Workspace.load()
    source = _record(workspace, args.source_dataset)
    signal_dataset = _record(workspace, args.signal_dataset)
    checkpoint = (workspace.root / args.checkpoint).resolve()
    detector_particle_args = args.detector_particles or [DEFAULT_DETECTOR_PARTICLES]
    detector_particles = tuple(
        (workspace.root / path).resolve() for path in detector_particle_args
    )
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != CHECKPOINT_SHA256:
        raise ValueError("frozen classifier checkpoint hash mismatch")
    if args.artifact_dir.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {args.artifact_dir}")
    evidence = build_detector_faithful_candidate(
        event_table_root=workspace.datasets_root / source["path"],
        signal_dataset_root=workspace.datasets_root / signal_dataset["path"],
        detector_particles_csv=detector_particles,
        output_dir=args.output_dir,
        source_manifest_sha256=source["manifest_sha256"],
        signal_manifest_sha256=signal_dataset["manifest_sha256"],
        checkpoint_sha256=checkpoint_sha256,
        dataset_id=args.dataset_id,
        source_dataset_id=args.source_dataset,
        signal_dataset_id=args.signal_dataset,
        method_evidence_id=METHOD_EVIDENCE_ID,
        seed=args.seed,
        maximum_events=args.maximum_events,
    )
    validation = validate_detector_faithful_candidate(
        args.output_dir, expected_dataset_id=args.dataset_id
    )
    args.artifact_dir.mkdir(parents=True)
    payload = {"build": evidence, "validation": validation}
    (args.artifact_dir / "build_evidence.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    run = {
        "schema_version": 1,
        "project": "particles2SNR-pipeline",
        "run_id": args.artifact_dir.name,
        "dataset": args.dataset_id,
        "datasets": {
            args.source_dataset: {
                "id": args.source_dataset,
                "manifest_sha256": source["manifest_sha256"],
            },
            args.signal_dataset: {
                "id": args.signal_dataset,
                "manifest_sha256": signal_dataset["manifest_sha256"],
            },
        },
        "command": (
            "particles2SNR-pipeline/scripts/generation/"
            "build_equation_roundtrip_v2_candidate.py"
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
        "detector_particles_csv_sha256s": {
            path.relative_to(workspace.root).as_posix(): sha256_file(path)
            for path in detector_particles
        },
        "claim_boundary": evidence["claim_boundary"],
    }
    (args.artifact_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
