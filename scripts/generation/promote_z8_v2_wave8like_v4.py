#!/usr/bin/env python3
"""Promote the approved Z8-v2 Wave8-like v4 candidate as an immutable reference."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from internship_workspace.config import Workspace
from internship_workspace.datasets import (
    DatasetRecord,
    build_manifest,
    index_path,
    load_records,
    register_record,
    validate_record,
)
from internship_workspace.visual_review_store import ReviewStore, ReviewStoreError


DATASET_ID = "particles2snr-z8-v2-wave8like-known3-background-development"
VERSION = "v4"
DATASET_KEY = f"{DATASET_ID}@{VERSION}"
CANDIDATE = Path("datasets/interim/particles2SNR-pipeline") / DATASET_ID / VERSION
DESTINATION_RELATIVE = Path("processed/particles2SNR-pipeline") / DATASET_ID / VERSION
RUN_RELATIVE = Path("artifacts/particles2SNR-pipeline/runs/particle-z8-v2-wave8like-promotion-v4")
GENERATION_RUN = Path("artifacts/particles2SNR-pipeline/runs/particle-z8-v2-wave8like-generation-v4")
QUALIFICATION = Path("artifacts/particles2SNR-pipeline/audits/particle-z8-v2-wave8like-join-audit-v4")
CHECKPOINT = Path("artifacts/cross-project/reviews/particle-z8-v2-wave8like-join-audit-result-r4")
EXPECTED_DATASET_MANIFEST_SHA256 = "913adb8f3ca6c84f3eee010cbcbaeceec18f04047211d594d1bd7369b569d154"
EXPECTED_LOGICAL_MANIFEST_SHA256 = "0fb1b0ebaed71745ef48b38eac2ffada0d050b920c0525ead4d3c1cac0f3246b"
EXPECTED_ROWS = {"train": 4800, "val": 960}
DATA_FORMAT = "yolo-1d-long-sequence-z8-wave8like-development"
COMMAND = "promote_z8_v2_wave8like_v4.py; approved r4 development reference"
STATE_FILE = "promotion_state.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_tree_hash(rows: list[dict[str, str]]) -> str:
    canonical = "\n".join(
        f"{row['path']}\t{row['sha256']}" for row in sorted(rows, key=lambda item: item["path"])
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _paths(workspace: Workspace) -> tuple[Path, Path, Path]:
    return (
        workspace.root / CANDIDATE,
        workspace.datasets_root / DESTINATION_RELATIVE,
        workspace.root / RUN_RELATIVE,
    )


def _assert_paths(workspace: Workspace, candidate: Path, destination: Path, run_dir: Path) -> None:
    expected = tuple(path.resolve() for path in _paths(workspace))
    actual = tuple(path.resolve() for path in (candidate, destination, run_dir))
    if actual != expected:
        raise ValueError("promotion paths are fixed; refusing a non-canonical release")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _payload_inventory(candidate: Path) -> list[dict[str, str]]:
    manifest_path = candidate / "dataset-manifest.json"
    if sha256_file(manifest_path) != EXPECTED_DATASET_MANIFEST_SHA256:
        raise ValueError("candidate dataset manifest differs from the approved v4 build")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected: dict[str, str] = {
        "dataset-manifest.json": EXPECTED_DATASET_MANIFEST_SHA256,
        "dataset-contract.json": manifest["payload"]["dataset_contract_sha256"],
        "dataset.yaml": manifest["payload"]["dataset_yaml_sha256"],
        "manifest.csv": manifest["payload"]["manifest_csv_sha256"],
    }
    counts = {"train": 0, "val": 0}
    for row in _read_csv(candidate / "manifest.csv"):
        split = row["split"]
        if split not in counts:
            raise ValueError(f"unexpected candidate split: {split}")
        counts[split] += 1
        long_id = row["long_id"]
        expected[f"{split}/signals/{long_id}.npy"] = row["signal_sha256"]
        expected[f"{split}/labels/{long_id}.txt"] = row["label_sha256"]
    if counts != EXPECTED_ROWS:
        raise ValueError(f"candidate row-count contract mismatch: {counts}")
    actual_paths = {
        path.relative_to(candidate).as_posix()
        for path in candidate.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if actual_paths != set(expected):
        raise ValueError(
            "candidate payload inventory differs from the approved manifest; "
            f"unknown={sorted(actual_paths - set(expected))[:5]}, "
            f"missing={sorted(set(expected) - actual_paths)[:5]}"
        )
    rows = [{"path": relative, "sha256": sha256_file(candidate / relative)} for relative in sorted(expected)]
    if any(row["sha256"] != expected[row["path"]] for row in rows):
        raise ValueError("candidate payload hash differs from the approved manifest")
    return rows


def assert_supported_checkpoint(workspace: Workspace) -> dict[str, Any]:
    root = workspace.root / CHECKPOINT
    run = json.loads((root / "run.json").read_text(encoding="utf-8"))
    if (
        run.get("run_id") != CHECKPOINT.name
        or run.get("status") != "visual_review_complete"
        or run.get("visual_approval", {}).get("decision") != "supported"
        or not run.get("visual_checkpoint", {}).get("approved")
        or run.get("visual_checkpoint", {}).get("next_stage_blocked")
    ):
        raise ValueError("r4 visual review does not authorize promotion")
    store = ReviewStore(root)
    try:
        receipt = store.verify_receipt()
    except ReviewStoreError as exc:
        raise ValueError(f"approved r4 review receipt is invalid: {exc}") from exc
    contract = store.contract()
    decisions = store.current()
    evidence_id = "particle-z8-v2-wave8like-join-audit-result"
    if (
        contract.get("evidence_id") != evidence_id
        or decisions.get("decisions", {}).get(evidence_id, {}).get("decision") != "supported"
    ):
        raise ValueError("r4 receipt does not support the exact join-audit result")
    reference = contract.get("analysis_reference", {})
    qualification = workspace.root / QUALIFICATION
    metrics_manifest = json.loads((qualification / "metrics_manifest.json").read_text(encoding="utf-8"))
    receipt_metrics = {item["path"]: item["sha256"] for item in reference.get("metrics", [])}
    qualified_metrics = {item["path"]: item["sha256"] for item in metrics_manifest.get("metrics", [])}
    if (
        reference.get("run_id") != QUALIFICATION.name
        or reference.get("run_path") != QUALIFICATION.as_posix()
        or reference.get("computation_fingerprint") != metrics_manifest.get("computation_fingerprint")
        or receipt_metrics != qualified_metrics
    ):
        raise ValueError("r4 receipt does not bind the exact join-audit metrics")
    for relative, expected_hash in qualified_metrics.items():
        if sha256_file(qualification / relative) != expected_hash:
            raise ValueError(f"r4 qualification metric hash mismatch: {relative}")
    retention = contract.get("retention_manifest", {})
    if sha256_file(root / retention["path"]) != retention["sha256"]:
        raise ValueError("r4 receipt-bound retention manifest hash mismatch")
    return {
        "checkpoint": CHECKPOINT.as_posix(),
        "receipt_sha256": sha256_file(store.receipt_path),
        "contract_sha256": receipt["contract_sha256"],
        "decisions_sha256": receipt["decisions_sha256"],
        "reviewer": receipt["reviewer"],
        "completed_at": receipt["completed_at"],
    }


def verify_candidate(workspace: Workspace, candidate: Path) -> dict[str, Any]:
    inventory = _payload_inventory(candidate)
    manifest = json.loads((candidate / "dataset-manifest.json").read_text(encoding="utf-8"))
    contract = json.loads((candidate / "dataset-contract.json").read_text(encoding="utf-8"))
    generation = json.loads((workspace.root / GENERATION_RUN / "run.json").read_text(encoding="utf-8"))
    if (
        manifest.get("dataset_id") != DATASET_KEY
        or manifest.get("status") != "immutable_interim_candidate_awaiting_visual_join_audit"
        or manifest.get("audit", {}).get("status") != "pass"
        or manifest.get("audit", {}).get("sealed_test_accessed") is not False
        or manifest.get("audit", {}).get("events_intersecting_join_guards") != 0
        or manifest.get("audit", {}).get("logical_manifest_sha256") != EXPECTED_LOGICAL_MANIFEST_SHA256
    ):
        raise ValueError("candidate manifest violates the approved development contract")
    if (
        contract.get("dataset_id") != DATASET_KEY
        or contract.get("development_only") is not True
        or contract.get("sealed_test_accessed") is not False
        or contract.get("splits") != ["train", "val"]
    ):
        raise ValueError("candidate dataset contract is not development-only train/val")
    if (
        generation.get("run_id") != GENERATION_RUN.name
        or generation.get("status") != "complete_candidate_awaiting_visual_join_audit"
        or generation.get("dataset") != DATASET_KEY
        or generation.get("dataset_manifest_sha256") != EXPECTED_DATASET_MANIFEST_SHA256
    ):
        raise ValueError("generation run does not bind the exact v4 candidate")
    return {
        "candidate_manifest_sha256": EXPECTED_DATASET_MANIFEST_SHA256,
        "logical_manifest_sha256": EXPECTED_LOGICAL_MANIFEST_SHA256,
        "payload_inventory": inventory,
        "payload_inventory_hash": canonical_tree_hash(inventory),
        "file_count": len(inventory),
        "development_only": True,
        "sealed_test_accessed": False,
    }


def _release_payload(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "dataset": DATASET_KEY,
        "status": "immutable_development_reference",
        "created_at": state["created_at"],
        "candidate": state["qualification"],
        "promotion_gate": state["approval"],
        "scope": "Development-only train/val reference; no sealed-test access or performance claim.",
        "training_guard_policy": (
            "The 300-sample guards on both sides of each join are ignored in objectness loss "
            "and reported separately as a false-positive diagnostic."
        ),
        "rollback": (
            "Do not delete this immutable reference. Consumers can stop selecting @v4 while "
            "retaining its registry record, payload, review receipt, and release manifest."
        ),
    }


def _release_hash(state: dict[str, Any]) -> str:
    payload = json.dumps(_release_payload(state), indent=2, sort_keys=True) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def _write_release(destination: Path, state: dict[str, Any]) -> None:
    _atomic_json(destination / "release_manifest.json", _release_payload(state))
    state["release_manifest_sha256"] = _release_hash(state)


def _snapshot(destination: Path, state: dict[str, Any]) -> list[dict[str, str]]:
    rows = [
        *state["qualification"]["payload_inventory"],
        {"path": "release_manifest.json", "sha256": state["release_manifest_sha256"]},
    ]
    actual = {path.relative_to(destination).as_posix() for path in destination.rglob("*") if path.is_file()}
    if actual != {row["path"] for row in rows}:
        raise RuntimeError("release inventory differs from the persisted promotion snapshot")
    if any(sha256_file(destination / row["path"]) != row["sha256"] for row in rows):
        raise RuntimeError("release bytes differ from the persisted promotion snapshot")
    return rows


def _load_state(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / STATE_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _save_state(run_dir: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(run_dir / STATE_FILE, state)


@contextmanager
def _promotion_lock(run_dir: Path) -> Iterator[None]:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / ".promotion.lock").open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _assert_record(
    workspace: Workspace, record: DatasetRecord, *, count: int, manifest_hash: str
) -> None:
    expected = {
        "id": DATASET_ID,
        "version": VERSION,
        "status": "reference",
        "path": DESTINATION_RELATIVE.as_posix(),
        "producer": {"repository": "particles2SNR-pipeline", "command": COMMAND},
        "format": DATA_FORMAT,
        "file_count": count,
        "manifest": f"{DATASET_ID}-{VERSION}.jsonl",
        "manifest_sha256": manifest_hash,
    }
    if record.payload != expected:
        raise RuntimeError("existing registry record differs from the immutable v4 contract")
    registry_manifest = index_path(workspace).parent / expected["manifest"]
    if not registry_manifest.is_file() or sha256_file(registry_manifest) != manifest_hash:
        raise RuntimeError("registry manifest differs from the immutable v4 contract")
    errors = validate_record(workspace, record, full=True)
    if errors:
        raise RuntimeError("registered reference validation failed:\n" + "\n".join(errors))


def _register_checked(
    workspace: Workspace,
    *,
    count: int,
    manifest_hash: str,
    failure_hook: Callable[[str], None] | None,
) -> DatasetRecord:
    records = [record for record in load_records(workspace) if record.key == DATASET_KEY]
    if len(records) > 1:
        raise RuntimeError("registry contains duplicate v4 records")
    if records:
        _assert_record(workspace, records[0], count=count, manifest_hash=manifest_hash)
        return records[0]
    if failure_hook:
        failure_hook("before_register")
    record = register_record(
        workspace,
        dataset_id=DATASET_ID,
        version=VERSION,
        relative_path=DESTINATION_RELATIVE.as_posix(),
        status="reference",
        producer="particles2SNR-pipeline",
        data_format=DATA_FORMAT,
        command=COMMAND,
    )
    _assert_record(workspace, record, count=count, manifest_hash=manifest_hash)
    return record


def promote(
    workspace: Workspace,
    *,
    candidate: Path,
    destination: Path,
    run_dir: Path,
    failure_hook: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    _assert_paths(workspace, candidate, destination, run_dir)
    with _promotion_lock(run_dir):
        state = _load_state(run_dir)
        if state is None:
            if destination.exists():
                raise FileExistsError("destination exists without a promotion transaction")
            state = {
                "schema_version": 1,
                "dataset": DATASET_KEY,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "phase": "initialized",
                "approval": assert_supported_checkpoint(workspace),
                "qualification": verify_candidate(workspace, candidate),
                "candidate": candidate.resolve().relative_to(workspace.root.resolve()).as_posix(),
                "destination": destination.resolve().relative_to(workspace.root.resolve()).as_posix(),
            }
            _save_state(run_dir, state)
        staging = destination.with_name(f".{destination.name}.promotion-staging")
        if not destination.exists():
            if staging.exists():
                quarantine = run_dir / "quarantine-invalid-staging"
                if quarantine.exists():
                    raise RuntimeError("invalid staging is already quarantined; inspect before retrying")
                os.replace(staging, quarantine)
                state["quarantined_staging"] = quarantine.relative_to(workspace.root).as_posix()
                _save_state(run_dir, state)
            if failure_hook:
                failure_hook("before_copy")
            shutil.copytree(candidate, staging, copy_function=shutil.copy2)
            expected = state["qualification"]["payload_inventory"]
            if any(sha256_file(staging / row["path"]) != row["sha256"] for row in expected):
                raise ValueError("staged payload is not byte-for-byte identical to the candidate")
            _write_release(staging, state)
            _snapshot(staging, state)
            state["phase"] = "payload_staged"
            _save_state(run_dir, state)
            if failure_hook:
                failure_hook("after_payload_staged")
            _snapshot(staging, state)
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging, destination)
            state["phase"] = "payload_moved"
            _save_state(run_dir, state)
            if failure_hook:
                failure_hook("after_payload_moved")
        elif staging.exists():
            raise RuntimeError("both destination and staging exist; inspect before retrying")
        _snapshot(destination, state)
        count, manifest_hash = build_manifest(destination, run_dir / "release_payload_manifest.jsonl")
        _snapshot(destination, state)
        record = _register_checked(
            workspace, count=count, manifest_hash=manifest_hash, failure_hook=failure_hook
        )
        _assert_record(workspace, record, count=count, manifest_hash=manifest_hash)
        state["phase"] = "registry_reference"
        state["registry"] = {
            "status": "reference",
            "manifest": record.payload["manifest"],
            "manifest_sha256": record.payload["manifest_sha256"],
            "file_count": record.payload["file_count"],
        }
        _save_state(run_dir, state)
        run = {
            "schema_version": 1,
            "project": "particles2SNR-pipeline",
            "run_id": run_dir.name,
            "kind": "immutable_dataset_promotion",
            "status": "complete_reference_dataset",
            **state,
        }
        _atomic_json(run_dir / "run.json", run)
        return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, default=CANDIDATE)
    parser.add_argument("--destination", type=Path, default=Path("datasets") / DESTINATION_RELATIVE)
    parser.add_argument("--run-dir", type=Path, default=RUN_RELATIVE)
    args = parser.parse_args()
    workspace = Workspace.load()
    resolve = lambda path: path if path.is_absolute() else workspace.root / path
    result = promote(
        workspace,
        candidate=resolve(args.candidate),
        destination=resolve(args.destination),
        run_dir=resolve(args.run_dir),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
