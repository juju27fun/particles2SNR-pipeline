#!/usr/bin/env python3
"""Crash-safe immutable promotion for the approved P2SNR_F v2 candidate."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from internship_workspace.config import Workspace
from internship_workspace.datasets import (
    DatasetRecord,
    build_manifest,
    index_path,
    load_records,
    register_record,
    registry_lock,
    validate_record,
)
from internship_workspace.visual_review_store import ReviewStore, ReviewStoreError


DATASET_ID = "particles2snr-f-dual-clean-c1-yolo-4class"
VERSION = "v2"
EXPECTED_TREE_HASH = "56325de0d726e281a9763fc5a1c9e06750c032af5cb03085795a54a522f0002d"
EXPECTED_SPLITS = {"train": 1848, "val": 462, "test": 0}
EXPECTED_REPAIR_REGIONS = 213
EXPECTED_REPAIRED_SIGNALS = 206
CHECKPOINT = Path("artifacts/cross-project/reviews/particle-p2snrf-v2-repair-result-r6")
QUALIFICATION = Path("artifacts/cross-project/analyses/particle-p2snrf-v2-repair-qualification-v4")
CANDIDATE = Path("datasets/interim/particles2snr-f-dual-clean-c1-yolo-4class/v2")
DESTINATION_RELATIVE = Path("processed/particles2snr-f-dual-clean-c1-yolo-4class/v2")
RUN_RELATIVE = Path("artifacts/particles2SNR-pipeline/runs/particles2snr-f-dual-clean-prefilter-v2-promotion")
STATE_FILE = "promotion_state.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        yaml.safe_dump(payload, handle, sort_keys=False)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def canonical_tree_hash(rows: list[dict[str, str]]) -> str:
    """Exact path-tab-hash implementation from the frozen qualification audit."""
    canonical = "\n".join(
        f"{row['path']}\t{row['sha256']}" for row in sorted(rows, key=lambda item: item["path"])
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _relative(workspace: Workspace, path: Path) -> str:
    return path.resolve().relative_to(workspace.root.resolve()).as_posix()


def _canonical_paths(workspace: Workspace) -> tuple[Path, Path, Path]:
    return (
        workspace.root / CANDIDATE,
        workspace.datasets_root / DESTINATION_RELATIVE,
        workspace.root / RUN_RELATIVE,
    )


def _assert_canonical_paths(workspace: Workspace, candidate: Path, destination: Path, run_dir: Path) -> None:
    expected = tuple(path.resolve() for path in _canonical_paths(workspace))
    actual = (candidate.resolve(), destination.resolve(), run_dir.resolve())
    if actual != expected:
        raise ValueError("promotion paths are fixed; refusing a non-canonical candidate, destination, or run directory")


def _candidate_tree_rows(workspace: Workspace, candidate: Path) -> list[dict[str, str]]:
    candidate_run = workspace.root / "artifacts/particles2SNR-pipeline/runs/particles2snr-f-dual-clean-prefilter-v2-candidate"
    rows: list[dict[str, str]] = []
    for row in sorted(_read_csv(candidate / "source_inventory.csv"), key=lambda item: (item["split"], item["filename"])):
        split, filename = row["split"], row["filename"]
        for kind, path in (
            ("signal", candidate / split / "signals" / filename),
            ("label", candidate / split / "labels" / filename.replace(".npy", ".txt")),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"missing qualified {kind}: {path}")
            array = np.asarray(np.load(path), dtype=np.float64) if kind == "signal" else None
            if array is not None and (array.ndim != 1 or not np.isfinite(array).all()):
                raise ValueError(f"non-finite/non-vector candidate signal: {path}")
            rows.append({"kind": kind, "path": _relative(workspace, path), "sha256": sha256_file(path)})
    for kind, path in (
        ("dataset_yaml", candidate / "dataset.yaml"),
        ("source_inventory", candidate / "source_inventory.csv"),
        ("saturation_repair_manifest", candidate / "saturation_repair_manifest.csv"),
        ("train_data_json", candidate_run / "train/data.json"),
        ("val_data_json", candidate_run / "val/data.json"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"missing qualified support file: {path}")
        rows.append({"kind": kind, "path": _relative(workspace, path), "sha256": sha256_file(path)})
    return rows


def qualified_payload_inventory(candidate: Path) -> list[dict[str, str]]:
    """Close the candidate inventory: no payload file may escape qualification."""
    inventory = _read_csv(candidate / "source_inventory.csv")
    expected = {Path("dataset.yaml"), Path("source_inventory.csv"), Path("saturation_repair_manifest.csv")}
    for row in inventory:
        expected.add(Path(row["split"]) / "signals" / row["filename"])
        expected.add(Path(row["split"]) / "labels" / row["filename"].replace(".npy", ".txt"))
    actual = {path.relative_to(candidate) for path in candidate.rglob("*") if path.is_file()}
    if actual != expected:
        unknown = sorted(path.as_posix() for path in actual - expected)
        missing = sorted(path.as_posix() for path in expected - actual)
        raise ValueError(f"candidate payload inventory differs from qualification; unknown={unknown}, missing={missing}")
    return [{"path": path.as_posix(), "sha256": sha256_file(candidate / path)} for path in sorted(expected)]


def _inventory_hash(rows: list[dict[str, str]]) -> str:
    return canonical_tree_hash(rows)


def _verify_copy(source_rows: list[dict[str, str]], copied: Path) -> None:
    expected_paths = {Path(row["path"]) for row in source_rows}
    actual_paths = {path.relative_to(copied) for path in copied.rglob("*") if path.is_file()}
    if actual_paths != expected_paths:
        raise ValueError("staged payload inventory differs from the qualified candidate")
    actual = [{"path": row["path"], "sha256": sha256_file(copied / row["path"])} for row in source_rows]
    if actual != source_rows:
        raise ValueError("staged payload is not byte-for-byte identical to the qualified candidate")


def _destination_snapshot(destination: Path, state: dict[str, Any]) -> list[dict[str, str]]:
    """Snapshot all destination bytes, separating frozen payload from metadata."""
    frozen = [row for row in state["qualification"]["payload_inventory"] if row["path"] != "dataset.yaml"]
    metadata = state["release_metadata"]
    return sorted(
        [*frozen, {"path": "dataset.yaml", "sha256": metadata["dataset_yaml_sha256"]},
         {"path": "release_manifest.json", "sha256": metadata["release_manifest_sha256"]}],
        key=lambda row: row["path"],
    )


def _assert_destination_snapshot(destination: Path, state: dict[str, Any]) -> None:
    expected = _destination_snapshot(destination, state)
    actual_paths = {path.relative_to(destination) for path in destination.rglob("*") if path.is_file()}
    expected_paths = {Path(row["path"]) for row in expected}
    if actual_paths != expected_paths:
        raise RuntimeError("moved destination file inventory differs from the persisted promotion snapshot")
    actual = [{"path": row["path"], "sha256": sha256_file(destination / row["path"])} for row in expected]
    if actual != expected:
        raise RuntimeError("moved destination bytes differ from the persisted promotion snapshot")


def assert_supported_checkpoint(workspace: Workspace, checkpoint: Path) -> dict[str, Any]:
    store = ReviewStore(workspace.root / checkpoint)
    try:
        receipt = store.verify_receipt()
    except ReviewStoreError as exc:
        raise ValueError(f"approved review receipt is invalid: {exc}") from exc
    decisions = store.current()
    run = json.loads((workspace.root / checkpoint / "run.json").read_text(encoding="utf-8"))
    if not decisions.get("complete") or not decisions.get("decisions") or any(
        item.get("decision") != "supported" for item in decisions["decisions"].values()
    ):
        raise ValueError("result checkpoint is not fully supported")
    visual = run.get("visual_checkpoint", {})
    if run.get("status") != "visual_review_complete" or not visual.get("approved") or visual.get("next_stage_blocked"):
        raise ValueError("result checkpoint does not authorize the next stage")
    return {
        "checkpoint": checkpoint.as_posix(), "receipt_sha256": sha256_file(store.receipt_path),
        "decisions_sha256": receipt["decisions_sha256"], "contract_sha256": receipt["contract_sha256"],
        "asset_hashes": receipt["primary_assets"], "reviewer": receipt["reviewer"], "completed_at": receipt["completed_at"],
    }


def verify_candidate(workspace: Workspace, candidate: Path) -> dict[str, Any]:
    metadata = yaml.safe_load((candidate / "dataset.yaml").read_text(encoding="utf-8"))
    if metadata.get("dataset_id") != f"{DATASET_ID}@{VERSION}" or metadata.get("status") != "candidate_pending_result_validation":
        raise ValueError("candidate release identity/status does not match the qualified candidate")
    splits = {name: int(metadata["splits"][name]["total"]) for name in EXPECTED_SPLITS}
    if splits != EXPECTED_SPLITS:
        raise ValueError(f"candidate split contract mismatch: {splits}")
    repairs = _read_csv(candidate / "saturation_repair_manifest.csv")
    if len(repairs) != EXPECTED_REPAIR_REGIONS or len({(row["split"], row["filename"]) for row in repairs}) != EXPECTED_REPAIRED_SIGNALS:
        raise ValueError("candidate repair-region contract mismatch")
    payload_rows = qualified_payload_inventory(candidate)
    tree_rows = _candidate_tree_rows(workspace, candidate)
    actual_tree_hash = canonical_tree_hash(tree_rows)
    if actual_tree_hash != EXPECTED_TREE_HASH:
        raise ValueError(f"candidate tree hash mismatch: expected {EXPECTED_TREE_HASH}, got {actual_tree_hash}")
    qualification = json.loads((workspace.root / QUALIFICATION / "metrics_manifest.json").read_text(encoding="utf-8"))
    if qualification["computation_provenance"]["dataset_tree_hash"] != actual_tree_hash:
        raise ValueError("candidate tree hash differs from the scientific qualification")
    return {"candidate_tree_hash": actual_tree_hash, "tree_entries": len(tree_rows), "payload_inventory": payload_rows, "payload_inventory_hash": _inventory_hash(payload_rows)}


def _git_revision(path: Path) -> str:
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _state_path(run_dir: Path) -> Path:
    return run_dir / STATE_FILE


def _load_state(run_dir: Path) -> dict[str, Any] | None:
    path = _state_path(run_dir)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _save_state(run_dir: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(_state_path(run_dir), state)


def _write_release_metadata(destination: Path, state: dict[str, Any]) -> None:
    yaml_path = destination / "dataset.yaml"
    metadata = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    metadata.update({"path": ".", "dataset_id": f"{DATASET_ID}@{VERSION}", "status": "active"})
    metadata.setdefault("provenance", {}).update({
        "promotion_checkpoint": state["approval"]["checkpoint"],
        "promotion_receipt_sha256": state["approval"]["receipt_sha256"],
        "qualified_candidate_tree_hash": state["qualification"]["candidate_tree_hash"],
        "qualified_payload_inventory_hash": state["qualification"]["payload_inventory_hash"],
        "release_manifest": "release_manifest.json",
    })
    yaml_path.write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
    release = {
        "schema_version": 1, "dataset": f"{DATASET_ID}@{VERSION}", "status": "active_immutable_release",
        "created_at": state["created_at"], "qualification": state["qualification"], "promotion_gate": state["approval"],
        "payload_boundary": "All files listed in qualification.payload_inventory are byte-for-byte copied before this release metadata is written.",
        "rollback": "Do not delete this immutable release. Consumers may select particles2snr-f-dual-clean-c1-yolo-4class@v1; preserve v2 and its manifest for audit.",
    }
    _atomic_json(destination / "release_manifest.json", release)
    state["release_metadata"] = {
        "dataset_yaml_sha256": sha256_file(yaml_path),
        "release_manifest_sha256": sha256_file(destination / "release_manifest.json"),
    }


def _release_payload_validation(workspace: Workspace, destination: Path, run_dir: Path) -> dict[str, Any]:
    staged_manifest = run_dir / "release_payload_manifest.jsonl"
    count, digest = build_manifest(destination, staged_manifest)
    if not (destination / "release_manifest.json").is_file():
        raise ValueError("release metadata is missing before registry creation")
    return {"file_count": count, "manifest_sha256": digest, "validation_manifest": _relative(workspace, staged_manifest)}


def _assert_destination_or_recoverable(workspace: Workspace, destination: Path) -> None:
    key = f"{DATASET_ID}@{VERSION}"
    records = [record for record in load_records(workspace) if record.key == key]
    if len(records) > 1:
        raise RuntimeError("registry has duplicate v2 records")
    if records:
        errors = validate_record(workspace, records[0], full=True)
        if errors:
            raise RuntimeError("existing v2 registry record is invalid:\n" + "\n".join(errors))


def _backup_file(path: Path, workspace: Workspace) -> dict[str, Any]:
    exists = path.is_file()
    raw = path.read_bytes() if exists else b""
    return {"path": _relative(workspace, path), "exists": exists,
            "sha256": hashlib.sha256(raw).hexdigest(), "bytes_b64": base64.b64encode(raw).decode("ascii")}


def _restore_backup(workspace: Workspace, backup: dict[str, Any]) -> None:
    path = workspace.root / backup["path"]
    raw = base64.b64decode(backup["bytes_b64"])
    if hashlib.sha256(raw).hexdigest() != backup["sha256"]:
        raise RuntimeError("registry backup checksum is invalid")
    if not backup["exists"]:
        if path.exists():
            path.unlink()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.restore")
    temporary.write_bytes(raw)
    os.replace(temporary, path)


def _registry_backups(workspace: Workspace) -> dict[str, Any]:
    registry = index_path(workspace).parent
    return {
        "index": _backup_file(index_path(workspace), workspace),
        "manifest": _backup_file(registry / f"{DATASET_ID}-{VERSION}.jsonl", workspace),
    }


def _recover_registry_if_needed(workspace: Workspace, state: dict[str, Any]) -> None:
    backups = state.get("registry_backups")
    if not backups:
        return
    key = f"{DATASET_ID}@{VERSION}"
    try:
        records = [record for record in load_records(workspace) if record.key == key]
    except Exception:
        _restore_backup(workspace, backups["index"])
        _restore_backup(workspace, backups["manifest"])
        return
    if records:
        errors = validate_record(workspace, records[0], full=True)
        if not errors:
            return
    _restore_backup(workspace, backups["index"])
    _restore_backup(workspace, backups["manifest"])


def _rollback_own_new_record(workspace: Workspace, state: dict[str, Any]) -> None:
    """Remove only this transaction's new key; never replace an old index."""
    payload = state.get("created_record_payload")
    if not payload or state.get("registry_created_in_current_attempt") is not True:
        return
    with registry_lock(workspace):
        records = load_records(workspace)
        current = [record for record in records if record.key == f"{DATASET_ID}@{VERSION}"]
        if len(current) != 1 or current[0].payload != payload:
            return
        retained = [record.payload for record in records if record.key != f"{DATASET_ID}@{VERSION}"]
        _atomic_yaml(index_path(workspace), {"version": 1, "datasets": retained})
        manifest = index_path(workspace).parent / str(payload["manifest"])
        if manifest.is_file() and sha256_file(manifest) == payload["manifest_sha256"]:
            manifest.unlink()


def promote(workspace: Workspace, *, candidate: Path, destination: Path, run_dir: Path, failure_hook: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Resume safely after any interruption; only the registry pointer is final."""
    _assert_canonical_paths(workspace, candidate, destination, run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    state = _load_state(run_dir)
    if state is None:
        if destination.exists():
            raise FileExistsError("destination exists without a promotion transaction")
        _assert_destination_or_recoverable(workspace, destination)
        state = {
            "schema_version": 1, "dataset": f"{DATASET_ID}@{VERSION}", "created_at": datetime.now(timezone.utc).isoformat(),
            "phase": "initialized", "approval": assert_supported_checkpoint(workspace, CHECKPOINT),
            "qualification": verify_candidate(workspace, candidate),
            "candidate": _relative(workspace, candidate), "destination": _relative(workspace, destination),
            "rollback": "Never delete v2. Recover consumers by selecting @v1; retain payload, release manifest, and registry manifest.",
        }
        _save_state(run_dir, state)
    if state.get("dataset") != f"{DATASET_ID}@{VERSION}" or state.get("candidate") != _relative(workspace, candidate):
        raise RuntimeError("promotion state does not match the canonical release")
    # A stale marker can only describe a prior interrupted attempt.  It must
    # never authorize deletion during a retry of an already-active record.
    state.pop("created_record_payload", None)
    state.pop("registry_created_in_current_attempt", None)
    _save_state(run_dir, state)

    staging = destination.with_name(f".{destination.name}.promotion-staging")
    if not destination.exists():
        if not staging.exists():
            if failure_hook:
                failure_hook("before_copy")
            shutil.copytree(candidate, staging, copy_function=shutil.copy2)
        _verify_copy(state["qualification"]["payload_inventory"], staging)
        state["staging_payload_snapshot"] = {
            "rows": state["qualification"]["payload_inventory"],
            "digest": state["qualification"]["payload_inventory_hash"],
        }
        state["phase"] = "payload_staged"
        _save_state(run_dir, state)
        if failure_hook:
            failure_hook("after_payload_staged")
        _write_release_metadata(staging, state)
        _save_state(run_dir, state)
        state["phase"] = "metadata_written"
        _save_state(run_dir, state)
        os.replace(staging, destination)
        state["phase"] = "payload_moved"
        _save_state(run_dir, state)
        if failure_hook:
            failure_hook("after_payload_moved")
    elif staging.exists():
        raise RuntimeError("both destination and staging exist; inspect transaction before retrying")
    else:
        _assert_destination_snapshot(destination, state)
        state["phase"] = "payload_moved"
        _save_state(run_dir, state)

    _assert_destination_snapshot(destination, state)
    pre_registry = _release_payload_validation(workspace, destination, run_dir)
    state["release_payload_validation"] = pre_registry
    _save_state(run_dir, state)
    key = f"{DATASET_ID}@{VERSION}"
    try:
        if failure_hook:
            failure_hook("before_register")
        _assert_destination_snapshot(destination, state)
        records = [record for record in load_records(workspace) if record.key == key]
        if not records:
            record = register_record(
                workspace, dataset_id=DATASET_ID, version=VERSION, relative_path=DESTINATION_RELATIVE.as_posix(), status="active",
                producer="particles2SNR-pipeline", data_format="yolo-1d-dual-clean-cosine-prefilter",
                command="promote_dual_clean_prefilter_v2.py; approved P2SNR_F cosine-repair release",
            )
            state["created_record_payload"] = record.payload
            state["registry_created_in_current_attempt"] = True
            _save_state(run_dir, state)
        else:
            record = records[0]
        _assert_destination_snapshot(destination, state)
        if record.payload["file_count"] != pre_registry["file_count"]:
            raise RuntimeError("registered file_count differs from the pre-registry payload validation")
        if record.payload["manifest_sha256"] != pre_registry["manifest_sha256"]:
            raise RuntimeError("registered manifest hash differs from the pre-registry payload validation")
        errors = validate_record(workspace, record, full=True)
        if errors:
            raise RuntimeError("registered release validation failed:\n" + "\n".join(errors))
    except Exception:
        _rollback_own_new_record(workspace, state)
        raise
    state["phase"] = "registry_active"
    state["registry"] = {"manifest": record.payload["manifest"], "manifest_sha256": record.payload["manifest_sha256"], "file_count": record.payload["file_count"]}
    state.pop("created_record_payload", None)
    state.pop("registry_created_in_current_attempt", None)
    _save_state(run_dir, state)
    if failure_hook:
        failure_hook("after_record")
    run = {"schema_version": 1, "project": "particles2SNR-pipeline", "run_id": run_dir.name, "kind": "immutable_dataset_promotion", "status": "complete_active_dataset", **state,
           "repositories": {"workspace": _git_revision(workspace.root), "particles2SNR-pipeline": _git_revision(workspace.root / "particles2SNR-pipeline")}}
    _atomic_json(run_dir / "run.json", run)
    return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, default=CANDIDATE)
    parser.add_argument("--destination", type=Path, default=Path("datasets") / DESTINATION_RELATIVE)
    parser.add_argument("--run-dir", type=Path, default=RUN_RELATIVE)
    args = parser.parse_args()
    workspace = Workspace.load()
    candidate = args.candidate if args.candidate.is_absolute() else workspace.root / args.candidate
    destination = args.destination if args.destination.is_absolute() else workspace.root / args.destination
    run_dir = args.run_dir if args.run_dir.is_absolute() else workspace.root / args.run_dir
    print(json.dumps(promote(workspace, candidate=candidate, destination=destination, run_dir=run_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
