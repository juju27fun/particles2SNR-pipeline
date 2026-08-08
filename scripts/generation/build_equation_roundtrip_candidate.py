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
from particles2snr.equation_roundtrip import (
    CHECKPOINT_SHA256,
    DATASET_ID,
    SIGNAL_DATASET_ID,
    SOURCE_DATASET_ID,
    build_equation_roundtrip_candidate,
    sha256_file,
    validate_equation_roundtrip_candidate,
)


DEFAULT_CHECKPOINT = Path(
    "artifacts/unsupervised-learning-flow-cytometry/"
    "pretrained_backbones-10dB/"
    "particles2snr_f_3class_native_params_moment_patchtst_conv1dgap/"
    "conv1dgap_same_input_3class/best_model.pt"
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
        description="Build the gated particles2SNR equation-roundtrip candidate."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--maximum-events", type=int)
    args = parser.parse_args()

    workspace = Workspace.load()
    source = _record(workspace, SOURCE_DATASET_ID)
    signals = _record(workspace, SIGNAL_DATASET_ID)
    checkpoint = (workspace.root / args.checkpoint).resolve()
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != CHECKPOINT_SHA256:
        raise ValueError("frozen classifier checkpoint hash mismatch")
    if args.artifact_dir.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {args.artifact_dir}")
    evidence = build_equation_roundtrip_candidate(
        event_table_root=workspace.datasets_root / source["path"],
        signal_dataset_root=workspace.datasets_root / signals["path"],
        output_dir=args.output_dir,
        source_manifest_sha256=source["manifest_sha256"],
        signal_manifest_sha256=signals["manifest_sha256"],
        checkpoint_sha256=checkpoint_sha256,
        seed=args.seed,
        maximum_events=args.maximum_events,
    )
    validation = validate_equation_roundtrip_candidate(args.output_dir)
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
            SOURCE_DATASET_ID: {
                "id": SOURCE_DATASET_ID,
                "manifest_sha256": source["manifest_sha256"],
            },
            SIGNAL_DATASET_ID: {
                "id": SIGNAL_DATASET_ID,
                "manifest_sha256": signals["manifest_sha256"],
            },
        },
        "command": (
            "particles2SNR-pipeline/scripts/generation/"
            "build_equation_roundtrip_candidate.py"
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
        "method_evidence_id": "particles2snr-equation-latent-method-v1",
        "claim_boundary": evidence["claim_boundary"],
    }
    (args.artifact_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
