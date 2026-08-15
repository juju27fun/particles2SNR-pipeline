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


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Build an immutable MAD-teacher YOLO-1D dataset.")
    result.add_argument("--workspace-root", type=Path, required=True)
    result.add_argument("--source-dataset-root", type=Path, required=True)
    result.add_argument("--predecessor-root", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--dataset-id", required=True)
    result.add_argument("--source-dataset-id", required=True)
    result.add_argument("--source-manifest-sha256", required=True)
    result.add_argument("--config-json", type=Path, required=True)
    result.add_argument("--expected-json", type=Path, required=True)
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
    expected = json.loads(args.expected_json.read_text(encoding="utf-8"))
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
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
