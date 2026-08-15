#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from particles2snr.particle_events import ParticleDetectionConfig, config_fingerprint
from particles2snr.particle_mad_teacher_dataset import (
    build_mad_teacher_dataset,
    sha256_file,
)


EXPECTED_PROFILES = {
    "mad-v2.1": {
        "traces_total": 2_888,
        "events_total": 3_618,
        "mad_empty_traces": 783,
        "events_10um": 706,
        "events_2um": 1_145,
        "events_4um": 1_767,
        "same_yolo_cell_collisions": 0,
        "saturation_center_vetoed": 79,
        "repaired_traces": 255,
        "additions": 85,
        "losses": 216,
        "audit_cases": 60,
    }
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Build an immutable MAD-teacher YOLO-1D dataset.")
    result.add_argument("--workspace-root", type=Path, required=True)
    result.add_argument("--source-dataset-root", type=Path, required=True)
    result.add_argument("--source-manifest", type=Path)
    result.add_argument("--repair-manifest", type=Path)
    result.add_argument("--predecessor-root", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--dataset-id", required=True)
    result.add_argument("--source-dataset-id", required=True)
    result.add_argument("--source-manifest-sha256", required=True)
    result.add_argument("--config-json", type=Path, required=True)
    result.add_argument("--expected-json", type=Path)
    result.add_argument("--expected-profile", choices=tuple(EXPECTED_PROFILES))
    result.add_argument("--saturation-center-veto", action="store_true")
    result.add_argument(
        "--audit-mode",
        choices=("v2_admissions", "source_correction_changes"),
        default="v2_admissions",
    )
    result.add_argument("--run-output-dir", type=Path, required=True)
    result.add_argument("--run-id", required=True)
    result.add_argument("--repository-revision", required=True)
    result.add_argument("--repository-dirty", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.run_output_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable run: {args.run_output_dir}")
    config_payload = json.loads(args.config_json.read_text(encoding="utf-8"))
    config = ParticleDetectionConfig(**config_payload.get("config", config_payload))
    expected_hash = config_payload.get("config_sha256")
    if expected_hash and expected_hash != config_fingerprint(config):
        raise ValueError("configuration fingerprint mismatch")
    if args.expected_json and args.expected_profile:
        raise ValueError("choose either --expected-json or --expected-profile")
    expected = (
        json.loads(args.expected_json.read_text(encoding="utf-8"))
        if args.expected_json
        else EXPECTED_PROFILES.get(args.expected_profile)
    )
    manifest = build_mad_teacher_dataset(
        workspace_root=args.workspace_root,
        source_dataset_root=args.source_dataset_root,
        predecessor_root=args.predecessor_root,
        output_dir=args.output_dir,
        dataset_id=args.dataset_id,
        source_dataset_id=args.source_dataset_id,
        source_manifest_sha256=args.source_manifest_sha256,
        config=config,
        expected=expected,
        source_manifest_path=args.source_manifest,
        repair_manifest=args.repair_manifest,
        saturation_center_veto=args.saturation_center_veto,
        audit_mode=args.audit_mode,
    )
    args.run_output_dir.mkdir(parents=True)
    run = {
        "schema_version": 1,
        "run_id": args.run_id,
        "project": "particles2SNR-pipeline",
        "kind": "mad-teacher-dataset-build",
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset_id,
        "command": "scripts/generation/build_mad_teacher_detection_dataset.py",
        "repositories": {
            "particles2SNR-pipeline": {
                "commit": args.repository_revision,
                "dirty": bool(args.repository_dirty),
            }
        },
        "config_sha256": manifest["config_sha256"],
        "source_manifest": (
            {
                "path": args.source_manifest.resolve().relative_to(args.workspace_root.resolve()).as_posix(),
                "sha256": sha256_file(args.source_manifest),
            }
            if args.source_manifest
            else None
        ),
        "repair_manifest": (
            {
                "path": args.repair_manifest.resolve().relative_to(args.workspace_root.resolve()).as_posix(),
                "sha256": sha256_file(args.repair_manifest),
            }
            if args.repair_manifest
            else None
        ),
        "saturation_center_veto": args.saturation_center_veto,
        "audit_mode": args.audit_mode,
        "dataset_manifest": {
            "path": (args.output_dir / "dataset-manifest.json").resolve().relative_to(args.workspace_root.resolve()).as_posix(),
            "sha256": sha256_file(args.output_dir / "dataset-manifest.json"),
        },
        "payload_digest_sha256": manifest["payload_digest_sha256"],
        "outputs": [
            args.output_dir.resolve().relative_to(args.workspace_root.resolve()).as_posix()
        ],
    }
    (args.run_output_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "dataset_id": manifest["dataset_id"],
                "config_sha256": manifest["config_sha256"],
                "counts": manifest["counts"],
                "payload_digest_sha256": manifest["payload_digest_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
