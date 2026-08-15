#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from internship_workspace.config import Workspace
from internship_workspace.datasets import resolve_path, select_record
from particles2snr.saturation_first_source_dataset import (
    build_saturation_first_source_dataset,
    sha256_file,
)


RAW_DATASETS = {
    "2um": "c1-hf-5-10-2um-doublet@v1",
    "4um": "c1-hf-5-10-4um-doublet@v1",
    "10um": "c1-hf-5-10-10um-doublet@v1",
}
EXPECTED_COUNTS = {
    "traces_total": 2_888,
    "traces_by_source_split": {"train": 2_310, "test": 578},
    "traces_by_output_split": {"train": 1_964, "val": 346, "test": 578},
    "traces_by_class": {"10um": 943, "2um": 1_202, "4um": 743},
    "repaired_traces": 255,
    "repair_regions": 263,
    "validated_development_numerically_equivalent": 2_310,
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Build the full immutable saturation-first class-folder source."
    )
    result.add_argument("--workspace-root", type=Path, required=True)
    result.add_argument("--predecessor-root", type=Path, required=True)
    result.add_argument("--predecessor-dataset-id", required=True)
    result.add_argument("--predecessor-manifest-sha256", required=True)
    result.add_argument("--repair-reference", required=True)
    result.add_argument("--canonical-development", required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--dataset-id", required=True)
    result.add_argument("--run-output-dir", type=Path, required=True)
    result.add_argument("--run-id", required=True)
    result.add_argument("--expected-traces", type=int, default=2_888)
    return result


def _dataset(workspace: Workspace, key: str):
    dataset_id, version = key.rsplit("@", 1)
    record = select_record(workspace, dataset_id, version)
    return resolve_path(workspace, record), record.payload


def _git_state(path: Path) -> dict[str, object]:
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
    args = parser().parse_args()
    workspace = Workspace.load(args.workspace_root)
    if args.run_output_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable run: {args.run_output_dir}")
    repair_root, repair_record = _dataset(workspace, args.repair_reference)
    canonical_root, canonical_record = _dataset(workspace, args.canonical_development)
    raw_roots = {}
    raw_records = {}
    for class_name, key in RAW_DATASETS.items():
        raw_roots[class_name], raw_records[class_name] = _dataset(workspace, key)
    manifest = build_saturation_first_source_dataset(
        workspace_root=workspace.root,
        predecessor_root=args.predecessor_root,
        frozen_repair_manifest=repair_root / "saturation_repair_manifest.csv",
        raw_dataset_roots=raw_roots,
        raw_dataset_ids=RAW_DATASETS,
        raw_manifest_sha256s={
            class_name: raw_records[class_name]["manifest_sha256"]
            for class_name in RAW_DATASETS
        },
        output_dir=args.output_dir,
        dataset_id=args.dataset_id,
        predecessor_dataset_id=args.predecessor_dataset_id,
        predecessor_manifest_sha256=args.predecessor_manifest_sha256,
        repair_reference_dataset_id=args.repair_reference,
        repair_reference_manifest_sha256=repair_record["manifest_sha256"],
        canonical_development_root=canonical_root,
        canonical_development_dataset_id=args.canonical_development,
        canonical_development_manifest_sha256=canonical_record["manifest_sha256"],
        expected_traces=args.expected_traces,
        expected_counts=EXPECTED_COUNTS,
    )
    args.run_output_dir.mkdir(parents=True)
    run = {
        "schema_version": 1,
        "run_id": args.run_id,
        "project": "particles2SNR-pipeline",
        "kind": "dataset-generation",
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset_id,
        "command": [".venv/bin/python", *sys.argv],
        "sealed_test_accessed": True,
        "sealed_test_role": "deterministic source correction with frozen pre-test method and parameters",
        "repositories": {
            "particles2SNR-pipeline": _git_state(Path(__file__).resolve().parents[2])
        },
        "dataset_manifest": {
            "path": args.output_dir.resolve().relative_to(workspace.root.resolve()).as_posix()
            + "/dataset-manifest.json",
            "sha256": sha256_file(args.output_dir / "dataset-manifest.json"),
        },
        "payload_digest_sha256": manifest["payload_digest_sha256"],
        "summary": manifest["counts"],
        "expected_invariants": EXPECTED_COUNTS,
        "outputs": [args.output_dir.resolve().relative_to(workspace.root.resolve()).as_posix()],
    }
    (args.run_output_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "dataset_id": manifest["dataset_id"],
                "counts": manifest["counts"],
                "payload_digest_sha256": manifest["payload_digest_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
