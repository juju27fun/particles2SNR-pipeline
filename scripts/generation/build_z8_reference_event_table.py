#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from internship_workspace.config import Workspace
from internship_workspace.datasets import load_records, validate_record
from internship_workspace.visual_review_store import ReviewStore
from particles2snr.z8_reference_dataset import (
    _workspace_relative,
    build_z8_reference_event_table,
    sha256_file,
    validate_fresh_parent_contract,
    validate_z8_reference_event_table,
)


EXPECTED_PARENT_DATASET = "particles2snr-f-dual-clean-c1-yolo-4class@v2"
HARD_VETO_METHOD = (
    "particle-z8-saturation-hard-veto-method",
    "particle-z8-saturation-hard-veto-method-r1",
    "approved",
)
P2_REPAIR_RESULT = (
    "particle-p2snrf-v2-repair-result",
    "particle-p2snrf-v2-repair-result-r6",
    "supported",
)
FROZEN_STRICT_RUN_ID = "particles2snr-f-dual-clean-prefilter-v2-candidate"
FROZEN_STRICT_RUN_STATUS = "complete_pending_scientific_result_validation"
P2_QUALIFICATION_RUN_ID = "particle-p2snrf-v2-repair-qualification-v4"


def _record(workspace: Workspace, key: str) -> dict[str, Any]:
    match = next((record for record in load_records(workspace) if record.key == key), None)
    if match is None or match.payload["status"] not in {"active", "reference"}:
        raise ValueError(f"Eligible registered dataset not found: {key}")
    errors = validate_record(workspace, match, full=True)
    if errors:
        raise ValueError("Registered dataset validation failed: " + "; ".join(errors))
    return match.payload


def _review_gate(
    directory: Path,
    *,
    expected_evidence_id: str,
    expected_run_id: str,
    expected_decision: str,
) -> dict[str, str]:
    """Verify the exact completed review receipt that authorizes this build."""
    if directory.name != expected_run_id:
        raise ValueError(f"Unexpected evidence run directory: {directory.name}")
    run_path = directory / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if (
        run.get("run_id") != expected_run_id
        or run.get("evidence_id") != expected_evidence_id
        or run.get("status") != "visual_review_complete"
        or run.get("visual_checkpoint", {}).get("approved") is not True
        or run.get("visual_checkpoint", {}).get("next_stage_blocked") is not False
    ):
        raise PermissionError("Evidence run is not an approved completed gate")
    store = ReviewStore(directory)
    contract = store.contract()
    if contract.get("evidence_id") != expected_evidence_id:
        raise ValueError("Review contract evidence ID differs from requested gate")
    receipt = store.verify_receipt()
    decision = store.current().get("decisions", {}).get(expected_evidence_id, {})
    if decision.get("decision") != expected_decision:
        raise PermissionError("Evidence review decision does not authorize Z8 build")
    return {
        "evidence_id": expected_evidence_id,
        "run_id": expected_run_id,
        "run_sha256": sha256_file(run_path),
        "receipt_sha256": sha256_file(store.receipt_path),
        "decisions_sha256": sha256_file(store.decisions_path),
        "receipt_reviewer": str(receipt["reviewer"]),
    }


def _require_frozen_strict_run(
    workspace: Workspace,
    *,
    result_evidence_dir: Path,
    strict_run: Path,
    strict_dataset_id: str,
) -> dict[str, Any]:
    """Bind detector inputs to the exact P2 qualification evidence.

    A matching dataset ID is insufficient: this rejects a replacement run or a
    post-qualification edit even when it advertises the same v2 dataset.
    """
    if strict_run.name != FROZEN_STRICT_RUN_ID:
        raise ValueError("Strict run is not the frozen prefilter v2 candidate")
    run_path = strict_run / "run.json"
    strict_metadata = json.loads(run_path.read_text(encoding="utf-8"))
    if (
        strict_metadata.get("run_id") != FROZEN_STRICT_RUN_ID
        or strict_metadata.get("status") != FROZEN_STRICT_RUN_STATUS
        or strict_metadata.get("dataset") != strict_dataset_id
    ):
        raise ValueError("Strict detector run does not match its frozen P2 contract")

    result_run = json.loads((result_evidence_dir / "run.json").read_text(encoding="utf-8"))
    reference = result_run.get("analysis_reference")
    if not isinstance(reference, dict):
        raise ValueError("P2 result evidence lacks an analysis reference")
    raw_analysis_path = Path(str(reference.get("run_path", "")))
    analysis_dir = _resolve_output(workspace, raw_analysis_path, (workspace.artifacts_root,))
    if analysis_dir.name != P2_QUALIFICATION_RUN_ID:
        raise ValueError("P2 result does not reference the frozen qualification run")
    analysis_run = json.loads((analysis_dir / "run.json").read_text(encoding="utf-8"))
    metrics_manifest = json.loads(
        (analysis_dir / "metrics_manifest.json").read_text(encoding="utf-8")
    )
    fingerprint = reference.get("computation_fingerprint")
    if (
        reference.get("run_id") != P2_QUALIFICATION_RUN_ID
        or analysis_run.get("run_id") != P2_QUALIFICATION_RUN_ID
        or analysis_run.get("status") != "complete"
        or metrics_manifest.get("analysis_run_id") != P2_QUALIFICATION_RUN_ID
        or metrics_manifest.get("computation_fingerprint") != fingerprint
        or analysis_run.get("computation_fingerprint") != fingerprint
    ):
        raise ValueError("P2 qualification run/metrics fingerprint mismatch")
    referenced_metrics = {
        str(item["path"]): str(item["sha256"])
        for item in reference.get("metrics", [])
    }
    manifest_metrics = {
        str(item["path"]): str(item["sha256"])
        for item in metrics_manifest.get("metrics", [])
    }
    if referenced_metrics != manifest_metrics:
        raise ValueError("P2 result metrics do not match qualification manifest")
    for relative, expected_hash in manifest_metrics.items():
        if sha256_file(analysis_dir / relative) != expected_hash:
            raise ValueError(f"P2 qualification metric hash mismatch: {relative}")

    provenance = metrics_manifest.get("computation_provenance", {})
    inputs = provenance.get("inputs", {})
    expected_run_key = _workspace_relative(run_path)
    if inputs.get(expected_run_key) != sha256_file(run_path):
        raise ValueError("Strict run.json differs from the P2 qualification input")
    tree_path = analysis_dir / "dataset_tree_manifest.csv"
    frozen_hashes: dict[str, str] = {}
    with tree_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("kind") in {"train_data_json", "val_data_json"}:
                frozen_hashes[str(row["kind"])] = str(row["sha256"])
    if set(frozen_hashes) != {"train_data_json", "val_data_json"}:
        raise ValueError("P2 qualification lacks frozen strict train/val hashes")
    actual_hashes = {
        "train_data_json": sha256_file(strict_run / "train" / "data.json"),
        "val_data_json": sha256_file(strict_run / "val" / "data.json"),
    }
    if actual_hashes != frozen_hashes:
        raise ValueError("Strict train/val data differs from P2 qualification")
    return {
        "path": _workspace_relative(strict_run),
        "run_sha256": sha256_file(run_path),
        "dataset": strict_metadata["dataset"],
        "status": strict_metadata["status"],
        "qualification_run_id": P2_QUALIFICATION_RUN_ID,
        "qualification_tree_manifest_sha256": sha256_file(tree_path),
        "split_data_sha256": actual_hashes,
    }


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


def _resolve_output(
    workspace: Workspace, path: Path, allowed_roots: tuple[Path, ...]
) -> Path:
    resolved = path.resolve() if path.is_absolute() else (
        workspace.root / path
    ).resolve()
    if not any(
        resolved == root.resolve() or root.resolve() in resolved.parents
        for root in allowed_roots
    ):
        raise ValueError(f"output is outside allowed workspace roots: {path}")
    return resolved


def _comparison_inputs(
    workspace: Workspace, values: list[str]
) -> dict[str, dict[str, str]]:
    """Resolve and hash named diagnostic comparison inputs.

    Values use ``name=path`` so a Z8 v1/v2 comparison is explicit provenance,
    rather than an invisible dependency baked into this builder.
    """
    result: dict[str, dict[str, str]] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError("comparison inputs must use name=path")
        if name in result:
            raise ValueError(f"duplicate comparison input name: {name}")
        path = _resolve_output(
            workspace,
            Path(raw_path),
            (workspace.datasets_root, workspace.artifacts_root),
        )
        if not path.is_file():
            raise FileNotFoundError(f"comparison input is not a file: {path}")
        result[name] = {
            "path": _workspace_relative(path),
            "sha256": sha256_file(path),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the F-base dual-clean z8 reference event table.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--strict-dataset", required=True)
    parser.add_argument("--output-dataset", required=True)
    parser.add_argument("--strict-run", type=Path, required=True)
    parser.add_argument(
        "--strict-run-splits",
        default="train,val",
        help="Comma-separated detector run splits.",
    )
    parser.add_argument(
        "--saturation-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--method-evidence-id",
        required=True,
    )
    parser.add_argument("--method-evidence-dir", type=Path, required=True)
    parser.add_argument("--result-evidence-id", required=True)
    parser.add_argument(
        "--result-evidence-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--comparison-input",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Optional hash-bound diagnostic comparison file.",
    )
    parser.add_argument(
        "--expected-development-signal-count",
        type=int,
        default=None,
    )
    args = parser.parse_args()

    workspace = Workspace.load()
    args.output_dir = _resolve_output(
        workspace,
        args.output_dir,
        (workspace.datasets_root / "interim",),
    )
    args.artifact_dir = _resolve_output(
        workspace,
        args.artifact_dir,
        (workspace.artifacts_root / "particles2SNR-pipeline",),
    )
    strict_run_splits = tuple(
        value.strip()
        for value in args.strict_run_splits.split(",")
        if value.strip()
    )
    if set(strict_run_splits) - {"train", "val"}:
        raise ValueError("Z8 accepts only development detector splits: train,val")
    if set(strict_run_splits) != {"train", "val"}:
        raise ValueError("fresh Z8 requires both train and val detector splits")
    source = _record(workspace, args.source_dataset)
    strict = _record(workspace, args.strict_dataset)
    if (
        args.source_dataset != EXPECTED_PARENT_DATASET
        or args.strict_dataset != EXPECTED_PARENT_DATASET
    ):
        raise ValueError(
            "Z8 v2 requires source and strict datasets to be "
            f"{EXPECTED_PARENT_DATASET}"
        )
    strict_run = _resolve_output(
        workspace,
        args.strict_run,
        (workspace.artifacts_root / "particles2SNR-pipeline",),
    )
    saturation_manifest = workspace.root / args.saturation_manifest
    source_root = workspace.datasets_root / source["path"]
    method_evidence_dir = _resolve_output(
        workspace, args.method_evidence_dir, (workspace.artifacts_root,)
    )
    result_evidence_dir = _resolve_output(
        workspace, args.result_evidence_dir, (workspace.artifacts_root,)
    )
    if not method_evidence_dir.is_dir() or not result_evidence_dir.is_dir():
        raise FileNotFoundError("Z8 evidence directories must exist before build")
    if args.method_evidence_id != HARD_VETO_METHOD[0]:
        raise ValueError("Unexpected hard-veto method evidence ID")
    if args.result_evidence_id != P2_REPAIR_RESULT[0]:
        raise ValueError("Unexpected P2SNR_F repair result evidence ID")
    method_gate = _review_gate(
        method_evidence_dir,
        expected_evidence_id=HARD_VETO_METHOD[0],
        expected_run_id=HARD_VETO_METHOD[1],
        expected_decision=HARD_VETO_METHOD[2],
    )
    result_gate = _review_gate(
        result_evidence_dir,
        expected_evidence_id=P2_REPAIR_RESULT[0],
        expected_run_id=P2_REPAIR_RESULT[1],
        expected_decision=P2_REPAIR_RESULT[2],
    )
    frozen_strict_run = _require_frozen_strict_run(
        workspace,
        result_evidence_dir=result_evidence_dir,
        strict_run=strict_run,
        strict_dataset_id=args.strict_dataset,
    )
    validate_fresh_parent_contract(
        source_dataset_id=args.source_dataset,
        strict_dataset_id=args.strict_dataset,
        source_root=source_root,
        strict_run=strict_run,
        saturation_manifest=saturation_manifest,
    )
    if args.artifact_dir.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {args.artifact_dir}")
    summary = build_z8_reference_event_table(
        source_root=source_root,
        historical_run=None,
        strict_run=strict_run,
        saturation_manifest=saturation_manifest,
        output_dir=args.output_dir,
        source_dataset_id=args.source_dataset,
        source_manifest_sha256=source["manifest_sha256"],
        strict_dataset_id=args.strict_dataset,
        strict_manifest_sha256=strict["manifest_sha256"],
        output_dataset_id=args.output_dataset,
        strict_run_splits=strict_run_splits,
        fresh_detector_mode=True,
        saturation_center_veto=True,
        expected_development_signal_count=(
            args.expected_development_signal_count
        ),
    )
    validation = validate_z8_reference_event_table(
        args.output_dir, saturation_manifest=saturation_manifest
    )
    args.artifact_dir.mkdir(parents=True)
    evidence = {
        "summary": summary,
        "validation": validation,
        "output_files": {
            path.name: sha256_file(path)
            for path in sorted(args.output_dir.iterdir())
            if path.is_file()
        },
        "input_files": {
            **{
                args.strict_run.as_posix() + f"/{split}/data.json": (
                    sha256_file(strict_run / split / "data.json")
                )
                for split in strict_run_splits
            },
            args.saturation_manifest.as_posix(): sha256_file(saturation_manifest),
        },
        "comparison_inputs": _comparison_inputs(workspace, args.comparison_input),
    }
    (args.artifact_dir / "build_evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    run = {
        "schema_version": 1,
        "project": "particles2SNR-pipeline",
        "run_id": args.artifact_dir.name,
        "dataset": args.output_dataset,
        "datasets": {
            args.source_dataset: {"id": args.source_dataset, "manifest_sha256": source["manifest_sha256"]},
            args.strict_dataset: {"id": args.strict_dataset, "manifest_sha256": strict["manifest_sha256"]},
        },
        "command": "particles2SNR-pipeline/scripts/generation/build_z8_reference_event_table.py",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "repositories": {
            "workspace": _git_state(workspace.root),
            "particles2SNR-pipeline": _git_state(workspace.root / "particles2SNR-pipeline"),
        },
        "outputs": ["build_evidence.json"],
        "method_evidence": {
            **method_gate,
            "directory": _workspace_relative(method_evidence_dir),
        },
        "result_evidence": {
            **result_gate,
            "directory": _workspace_relative(result_evidence_dir),
        },
        "strict_run": {
            **frozen_strict_run,
        },
        "claim_boundary": summary["known_limitations"],
    }
    (args.artifact_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
