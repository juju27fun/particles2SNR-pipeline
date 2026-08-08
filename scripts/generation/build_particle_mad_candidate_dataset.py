#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from particles2snr.particle_candidate_dataset import build_candidate_dataset
from particles2snr.particle_events import ParticleDetectionConfig, config_fingerprint


def _load_config(path: Path | None) -> ParticleDetectionConfig:
    if path is None:
        return ParticleDetectionConfig()
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("config", payload)
    config = ParticleDetectionConfig(**values)
    expected = payload.get("config_sha256")
    if expected is not None and expected != config_fingerprint(config):
        raise ValueError("frozen detector configuration fingerprint mismatch")
    return config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build immutable class-agnostic particle MAD proposals."
    )
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-inventory", type=Path, required=True)
    parser.add_argument("--repair-manifest", type=Path)
    parser.add_argument("--signal-dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--run-output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repository-revision", required=True)
    parser.add_argument("--repository-dirty", action="store_true")
    parser.add_argument("--input-dataset-id", required=True)
    parser.add_argument("--input-manifest-sha256", required=True)
    parser.add_argument(
        "--role",
        action="append",
        choices=("mad_calibration", "mad_holdout", "legacy_exploration"),
        required=True,
    )
    parser.add_argument("--config-json", type=Path)
    parser.add_argument("--split-seed", type=int, default=20_260_731)
    parser.add_argument("--holdout-authorization", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_output = args.run_output_dir.resolve()
    if run_output.exists():
        raise FileExistsError(f"refusing to overwrite immutable run: {run_output}")
    config = _load_config(args.config_json)
    manifest = build_candidate_dataset(
        workspace_root=args.workspace_root,
        source_inventory=args.source_inventory,
        signal_dataset_root=args.signal_dataset_root,
        output_dir=args.output_dir,
        dataset_id=args.dataset_id,
        input_dataset_id=args.input_dataset_id,
        input_manifest_sha256=args.input_manifest_sha256,
        config=config,
        roles=args.role,
        repair_manifest=args.repair_manifest,
        seed=args.split_seed,
        holdout_authorization=args.holdout_authorization,
    )
    run_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{run_output.name}.", dir=run_output.parent))
    workspace = args.workspace_root.resolve()
    try:
        dataset_path = args.output_dir.resolve().relative_to(workspace).as_posix()
        manifest_path = (args.output_dir.resolve() / "dataset_manifest.json").relative_to(workspace).as_posix()
        run = {
            "schema_version": 1,
            "run_id": args.run_id,
            "project": "particles2SNR-pipeline",
            "command": "scripts/generation/build_particle_mad_candidate_dataset.py",
            "status": "complete",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": args.dataset_id,
            "datasets": {
                args.input_dataset_id: {
                    "id": args.input_dataset_id,
                    "manifest_sha256": args.input_manifest_sha256,
                }
            },
            "repositories": {
                "particles2SNR-pipeline": {
                    "commit": args.repository_revision,
                    "dirty": args.repository_dirty,
                }
            },
            "code": {
                "entrypoint": {
                    "path": "particles2SNR-pipeline/scripts/generation/build_particle_mad_candidate_dataset.py",
                    "sha256": _sha256(Path(__file__)),
                },
                "dataset_builder": {
                    "path": "particles2SNR-pipeline/particles2snr/particle_candidate_dataset.py",
                    "sha256": _sha256(Path(__file__).resolve().parents[2] / "particles2snr/particle_candidate_dataset.py"),
                },
                "detector": {
                    "path": "particles2SNR-pipeline/particles2snr/particle_events.py",
                    "sha256": _sha256(Path(__file__).resolve().parents[2] / "particles2snr/particle_events.py"),
                },
            },
            "config_sha256": manifest["config_sha256"],
            "roles_executed": manifest["roles_executed"],
            "holdout_authorization": (
                {
                    "path": args.holdout_authorization.resolve().relative_to(workspace).as_posix(),
                    "sha256": _sha256(args.holdout_authorization),
                }
                if args.holdout_authorization
                else None
            ),
            "sealed_holdout_accessed": "mad_holdout" in manifest["roles_executed"],
            "new_acquisition_in_scope": False,
            "dataset_path": dataset_path,
            "dataset_manifest": manifest_path,
            "outputs": [dataset_path, manifest_path],
        }
        (temporary / "run.json").write_text(
            json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, run_output)
    except Exception:
        import shutil

        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
