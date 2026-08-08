#!/usr/bin/env python3
"""Crash-safe immutable promotion for the approved development-only Z8 v2 table."""

from __future__ import annotations

import argparse
import base64
import csv
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from internship_workspace.config import Workspace
from internship_workspace.datasets import DatasetRecord, build_manifest, index_path, load_records, register_record, registry_lock, validate_record
from internship_workspace.visual_review_store import ReviewStore, ReviewStoreError


DATASET_ID = "particles2snr-fbase-dual-clean-z8-events-3class-plus-unclear-development"
VERSION = "v2"
CANDIDATE = Path("datasets/interim/particles2SNR-pipeline") / DATASET_ID / VERSION
DESTINATION_RELATIVE = Path("processed") / DATASET_ID / VERSION
RUN_RELATIVE = Path("artifacts/particles2SNR-pipeline/runs/particle-z8-hard-veto-v2-promotion")
CHECKPOINT = Path("artifacts/cross-project/reviews/particle-z8-hard-veto-v2-result-r3")
QUALIFICATION = Path("artifacts/cross-project/analyses/particle-z8-hard-veto-v2-qualification-v2")
CANDIDATE_RUN = Path("artifacts/particles2SNR-pipeline/runs/particle-z8-hard-veto-v2-candidate")
EXPECTED_PAYLOAD = {
    "dataset_summary.json": "b873debd69aaba8f3b3fabd4f2fac42b8531991d01a75d943c0d9ef9a332814d",
    "events.csv": "8f3ac226c61d0de766eea4e58edc3cdde44d469af960933ebbe962a06618deeb",
    "exclusions.csv": "a612a52bfbb0caaa7e7291a56035629489d933ee1fa57b9091c3d72e8a892562",
    "input_contract.json": "5c235e92e865cafc36e878f32b80a11c981b7a24773962e1a4936ce49458aff9",
}
EXPECTED_TREE_HASH = "dfd508c1bcc3fedcb4360775fe75b396e752ccaebaf0798e7a29148ee148395f"
STATE_FILE = "promotion_state.json"
DATA_FORMAT = "event-reference-table-z8-hard-veto-development-only"
COMMAND = "promote_z8_hard_veto_v2.py; approved r3 development-only immutable release"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_tree_hash(rows: list[dict[str, str]]) -> str:
    return hashlib.sha256("\n".join(f"{row['path']}\t{row['sha256']}" for row in sorted(rows, key=lambda row: row["path"])).encode()).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _relative(workspace: Workspace, path: Path) -> str:
    return path.resolve().relative_to(workspace.root.resolve()).as_posix()


def _paths(workspace: Workspace) -> tuple[Path, Path, Path]:
    return workspace.root / CANDIDATE, workspace.datasets_root / DESTINATION_RELATIVE, workspace.root / RUN_RELATIVE


def _assert_paths(workspace: Workspace, candidate: Path, destination: Path, run_dir: Path) -> None:
    if tuple(path.resolve() for path in (candidate, destination, run_dir)) != tuple(path.resolve() for path in _paths(workspace)):
        raise ValueError("promotion paths are fixed; refusing a non-canonical candidate, destination, or run directory")


def _payload_inventory(candidate: Path) -> list[dict[str, str]]:
    actual = {path.name for path in candidate.iterdir() if path.is_file()}
    if actual != set(EXPECTED_PAYLOAD):
        raise ValueError(f"candidate payload inventory differs from qualification: {sorted(actual)}")
    rows = [{"path": name, "sha256": sha256_file(candidate / name)} for name in sorted(actual)]
    if {row["path"]: row["sha256"] for row in rows} != EXPECTED_PAYLOAD:
        raise ValueError("candidate payload hash differs from the frozen build evidence")
    if canonical_tree_hash(rows) != EXPECTED_TREE_HASH:
        raise ValueError("candidate payload tree hash differs from the approved candidate")
    return rows


def assert_supported_checkpoint(workspace: Workspace) -> dict[str, Any]:
    root = workspace.root / CHECKPOINT
    run = json.loads((root / "run.json").read_text(encoding="utf-8"))
    if (run.get("run_id") != CHECKPOINT.name or run.get("status") != "visual_review_complete" or run.get("visual_approval", {}).get("decision") != "supported" or not run.get("visual_checkpoint", {}).get("approved") or run["visual_checkpoint"].get("next_stage_blocked")):
        raise ValueError("r3 visual review does not authorize Z8 v2 promotion")
    store = ReviewStore(root)
    try:
        receipt = store.verify_receipt()
    except ReviewStoreError as exc:
        raise ValueError(f"approved r3 review receipt is invalid: {exc}") from exc
    contract = store.contract()
    decisions = store.current()
    evidence_id = "particle-z8-hard-veto-v2-result"
    if contract.get("evidence_id") != evidence_id or decisions.get("decisions", {}).get(evidence_id, {}).get("decision") != "supported":
        raise ValueError("r3 receipt/contract/decision does not support the exact Z8 result")
    # The receipt authenticates this contract.  Never use mutable run.json as
    # the authority for analysis provenance or retention material.
    reference = contract.get("analysis_reference", {})
    qualification = workspace.root / QUALIFICATION
    manifest = json.loads((qualification / "metrics_manifest.json").read_text())
    qualification_run = json.loads((qualification / "run.json").read_text())
    receipt_metrics = {item["path"]: item["sha256"] for item in reference.get("metrics", [])}
    manifest_metrics = {item["path"]: item["sha256"] for item in manifest.get("metrics", [])}
    if (reference.get("run_id") != QUALIFICATION.name or reference.get("run_path") != QUALIFICATION.as_posix() or reference.get("computation_fingerprint") != manifest.get("computation_fingerprint") or qualification_run.get("computation_fingerprint") != manifest.get("computation_fingerprint") or receipt_metrics != manifest_metrics):
        raise ValueError("r3 review does not bind the exact qualification fingerprint and metrics")
    for relative, expected_hash in manifest_metrics.items():
        if sha256_file(qualification / relative) != expected_hash:
            raise ValueError(f"r3 qualification metric hash mismatch: {relative}")
    retention = contract.get("retention_manifest", {})
    relative, expected_hash = retention.get("path"), retention.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected_hash, str) or sha256_file(root / relative) != expected_hash:
        raise ValueError("r3 receipt-bound retention manifest hash mismatch")
    return {"checkpoint": CHECKPOINT.as_posix(), "run_sha256": sha256_file(root / "run.json"), "receipt_sha256": sha256_file(store.receipt_path), "contract_sha256": receipt["contract_sha256"], "decisions_sha256": receipt["decisions_sha256"], "asset_hashes": receipt["primary_assets"], "reviewer": receipt["reviewer"]}


def verify_candidate(workspace: Workspace, candidate: Path) -> dict[str, Any]:
    inventory = _payload_inventory(candidate)
    summary = json.loads((candidate / "dataset_summary.json").read_text())
    contract = json.loads((candidate / "input_contract.json").read_text())
    events, exclusions = _read_csv(candidate / "events.csv"), _read_csv(candidate / "exclusions.csv")
    run = json.loads((workspace.root / CANDIDATE_RUN / "run.json").read_text())
    evidence = json.loads((workspace.root / CANDIDATE_RUN / "build_evidence.json").read_text())
    key = f"{DATASET_ID}@{VERSION}"
    if run.get("run_id") != CANDIDATE_RUN.name or run.get("status") != "complete" or run.get("dataset") != key:
        raise ValueError("candidate build run is not the exact completed hard-veto v2 run")
    if evidence.get("output_files") != EXPECTED_PAYLOAD or evidence.get("summary") != summary:
        raise ValueError("candidate build evidence does not bind the exact payload")
    if summary.get("dataset_id") != key or summary.get("sealed_test_accessed") is not False or summary.get("event_count") != 2194:
        raise ValueError("candidate summary identity, event count, or sealed-test contract is invalid")
    if contract.get("splits") != ["train", "val"] or contract.get("sealed_splits") != ["test"] or contract.get("format") != "event-reference-table":
        raise ValueError("candidate input contract must be train/val-only with a sealed test")
    if len(events) != 2194 or len(exclusions) != 135 or {row.get("split") for row in events} - {"train", "val"}:
        raise ValueError("candidate events violate the development-only split contract")
    if any(row.get("reason") != "z8_center_inside_saturation_repair" for row in exclusions) or any(row.get("center_inside_saturation_repair") != "False" for row in events):
        raise ValueError("candidate exclusions or hard-veto flags violate the frozen contract")
    qualification = json.loads((workspace.root / QUALIFICATION / "metrics_manifest.json").read_text())
    provenance = qualification.get("computation_provenance", {})
    qualified_inputs = provenance.get("inputs", {})
    if qualification.get("analysis_run_id") != QUALIFICATION.name or any(
        qualified_inputs.get(_relative(workspace, candidate / name)) != digest
        for name, digest in EXPECTED_PAYLOAD.items()
    ):
        raise ValueError("qualification does not bind the exact candidate payload")
    summary_metrics = json.loads((workspace.root / QUALIFICATION / "summary_metrics.json").read_text())
    if not summary_metrics.get("publication_gate_passes") or summary_metrics.get("sealed_test_accessed") is not False or summary_metrics.get("decision_scope") != "Approval authorizes development-only Z8 v2 publication; it does not establish generalization, open the sealed test, or authorize P1/P2.":
        raise ValueError("qualification does not authorize the limited Z8 v2 release")
    for item in qualification.get("metrics", []):
        if sha256_file(workspace.root / QUALIFICATION / item["path"]) != item["sha256"]:
            raise ValueError(f"qualification metric hash mismatch: {item['path']}")
    return {"candidate_tree_hash": EXPECTED_TREE_HASH, "payload_inventory": inventory, "payload_inventory_hash": canonical_tree_hash(inventory), "candidate_run": _relative(workspace, workspace.root / CANDIDATE_RUN), "candidate_run_sha256": sha256_file(workspace.root / CANDIDATE_RUN / "run.json"), "qualification": _relative(workspace, workspace.root / QUALIFICATION), "qualification_manifest_sha256": sha256_file(workspace.root / QUALIFICATION / "metrics_manifest.json"), "development_only": True, "no_transfer_from_v1": True}


def _load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.is_file() else None


def _save(run_dir: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(run_dir / STATE_FILE, state)


def _backup(path: Path, workspace: Workspace) -> dict[str, Any]:
    raw = path.read_bytes() if path.is_file() else b""
    return {"path": _relative(workspace, path), "exists": path.is_file(), "sha256": hashlib.sha256(raw).hexdigest(), "bytes_b64": base64.b64encode(raw).decode()}


def _restore(workspace: Workspace, backup: dict[str, Any]) -> None:
    path, raw = workspace.root / backup["path"], base64.b64decode(backup["bytes_b64"])
    if hashlib.sha256(raw).hexdigest() != backup["sha256"]:
        raise RuntimeError("registry backup checksum is invalid")
    if not backup["exists"]:
        if path.exists(): path.unlink()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.restore")
    temporary.write_bytes(raw)
    os.replace(temporary, path)


def _assert_record(workspace: Workspace, record: DatasetRecord, *, count: int, manifest_hash: str) -> None:
    expected = {"id": DATASET_ID, "version": VERSION, "status": "active", "path": DESTINATION_RELATIVE.as_posix(), "producer": {"repository": "particles2SNR-pipeline", "command": COMMAND}, "format": DATA_FORMAT, "file_count": count, "manifest": f"{DATASET_ID}-{VERSION}.jsonl", "manifest_sha256": manifest_hash}
    if record.payload != expected:
        raise RuntimeError("existing active record differs from the immutable Z8 v2 release contract")
    manifest = index_path(workspace).parent / expected["manifest"]
    if not manifest.is_file() or sha256_file(manifest) != manifest_hash:
        raise RuntimeError("registry manifest bytes/hash differ from the active Z8 v2 record")
    errors = validate_record(workspace, record, full=True)
    if errors: raise RuntimeError("registered release validation failed:\n" + "\n".join(errors))


def _register_checked(workspace: Workspace, *, destination: Path, count: int, manifest_hash: str, failure_hook: Callable[[str], None] | None) -> DatasetRecord:
    """Use the shared datasets API lock; only verify existing immutable records."""
    records = [record for record in load_records(workspace) if record.key == f"{DATASET_ID}@{VERSION}"]
    if len(records) > 1: raise RuntimeError("registry has duplicate v2 records")
    if records:
        _assert_record(workspace, records[0], count=count, manifest_hash=manifest_hash)
        return records[0]
    if failure_hook: failure_hook("before_register")
    record = register_record(workspace, dataset_id=DATASET_ID, version=VERSION, relative_path=DESTINATION_RELATIVE.as_posix(), status="active", producer="particles2SNR-pipeline", data_format=DATA_FORMAT, command=COMMAND)
    _assert_record(workspace, record, count=count, manifest_hash=manifest_hash)
    return record


def _snapshot(destination: Path, state: dict[str, Any]) -> list[dict[str, str]]:
    rows = [*state["qualification"]["payload_inventory"], {"path": "release_manifest.json", "sha256": state["release_manifest_sha256"]}]
    actual = {path.relative_to(destination) for path in destination.rglob("*") if path.is_file()}
    if actual != {Path(row["path"]) for row in rows} or any(sha256_file(destination / row["path"]) != row["sha256"] for row in rows):
        raise RuntimeError("destination bytes differ from the persisted promotion snapshot")
    return rows


def _release_payload(state: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": 1, "dataset": f"{DATASET_ID}@{VERSION}", "status": "active_immutable_release", "created_at": state["created_at"], "candidate": state["qualification"], "promotion_gate": state["approval"], "scope": "development-only train/val event-reference table; test remains sealed", "no_transfer_from_v1": True, "rollback": "Do not delete this immutable v2 release; consumers may explicitly select @v1 while retaining this release and manifest for audit."}


def _release_hash(state: dict[str, Any]) -> str:
    return hashlib.sha256((json.dumps(_release_payload(state), indent=2, sort_keys=True) + "\n").encode()).hexdigest()


def _write_release(destination: Path, state: dict[str, Any]) -> None:
    _atomic_json(destination / "release_manifest.json", _release_payload(state))
    state["release_manifest_sha256"] = _release_hash(state)


def _staged_release_is_exact(staging: Path, state: dict[str, Any]) -> bool:
    """Verify the complete staged release before deciding it is recoverable."""
    expected = state["qualification"]["payload_inventory"]
    allowed = {Path(row["path"]) for row in expected} | {Path("release_manifest.json")}
    actual: set[Path] = set()
    for path in staging.rglob("*"):
        # The release format is deliberately flat; any nested directory, link,
        # special file, or extra payload invalidates the staging transaction.
        if path.is_symlink() or path.is_dir() or not path.is_file():
            return False
        actual.add(path.relative_to(staging))
    if actual != allowed:
        return False
    if any(sha256_file(staging / row["path"]) != row["sha256"] for row in expected):
        return False
    return sha256_file(staging / "release_manifest.json") == _release_hash(state)


@contextmanager
def _promotion_lock(run_dir: Path) -> Iterator[None]:
    """Serialize retries for this release without shadowing the registry lock."""
    lock = run_dir / ".promotion.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def promote(workspace: Workspace, *, candidate: Path, destination: Path, run_dir: Path, failure_hook: Callable[[str], None] | None = None) -> dict[str, Any]:
    _assert_paths(workspace, candidate, destination, run_dir)
    with _promotion_lock(run_dir):
        run_dir.mkdir(parents=True, exist_ok=True)
        state = _load(run_dir / STATE_FILE)
        if state is None:
            if destination.exists(): raise FileExistsError("destination exists without a promotion transaction")
            state = {"schema_version": 1, "dataset": f"{DATASET_ID}@{VERSION}", "created_at": datetime.now(timezone.utc).isoformat(), "phase": "initialized", "approval": assert_supported_checkpoint(workspace), "qualification": verify_candidate(workspace, candidate), "candidate": _relative(workspace, candidate), "destination": _relative(workspace, destination)}
            _save(run_dir, state)
        if state.get("dataset") != f"{DATASET_ID}@{VERSION}" or state.get("candidate") != _relative(workspace, candidate): raise RuntimeError("promotion state does not match the canonical release")
        staging = destination.with_name(f".{destination.name}.promotion-staging")
        if not destination.exists():
            if staging.exists() and (staging / "release_manifest.json").is_file() and not _staged_release_is_exact(staging, state):
                quarantine = run_dir / "quarantine-invalid-staging"
                if quarantine.exists():
                    raise RuntimeError("invalid staging is already quarantined; inspect before retrying")
                os.replace(staging, quarantine)
                state.pop("release_manifest_sha256", None)
                state["quarantined_staging"] = _relative(workspace, quarantine)
                _save(run_dir, state)
            if not staging.exists():
                if failure_hook: failure_hook("before_copy")
                shutil.copytree(candidate, staging, copy_function=shutil.copy2)
            expected = state["qualification"]["payload_inventory"]
            actual = [{"path": row["path"], "sha256": sha256_file(staging / row["path"])} for row in expected]
            staged_files = {path.name for path in staging.iterdir() if path.is_file()}
            has_release = (staging / "release_manifest.json").is_file()
            allowed = set(EXPECTED_PAYLOAD) | ({"release_manifest.json"} if has_release else set())
            if actual != expected or staged_files != allowed: raise ValueError("staged payload is not byte-for-byte identical to the qualified candidate")
            if has_release:
                expected_hash = _release_hash(state)
                if sha256_file(staging / "release_manifest.json") != expected_hash:
                    raise RuntimeError("staged release metadata is not the deterministic promotion manifest")
                state["release_manifest_sha256"] = expected_hash
                _save(run_dir, state)
            state["phase"] = "payload_staged"; _save(run_dir, state)
            if failure_hook: failure_hook("after_payload_staged")
            # Hooks model an interrupted/concurrent writer: never move bytes
            # that were only checked before that boundary.
            post_hook = [{"path": row["path"], "sha256": sha256_file(staging / row["path"])} for row in expected]
            post_allowed = set(EXPECTED_PAYLOAD) | ({"release_manifest.json"} if has_release else set())
            if post_hook != expected or {path.name for path in staging.iterdir() if path.is_file()} != post_allowed:
                raise ValueError("staged payload changed after byte verification; refusing move")
            _write_release(staging, state)
            if failure_hook: failure_hook("after_release_metadata_written")
            _snapshot(staging, state)
            _save(run_dir, state)
            os.replace(staging, destination); state["phase"] = "payload_moved"; _save(run_dir, state)
            if failure_hook: failure_hook("after_payload_moved")
        elif staging.exists(): raise RuntimeError("both destination and staging exist; inspect transaction before retrying")
        _snapshot(destination, state)
        pre_count, pre_hash = build_manifest(destination, run_dir / "release_payload_manifest.jsonl")
        _snapshot(destination, state)
        record = _register_checked(workspace, destination=destination, count=pre_count, manifest_hash=pre_hash, failure_hook=failure_hook)
        _snapshot(destination, state)
        _assert_record(workspace, record, count=pre_count, manifest_hash=pre_hash)
        state["phase"] = "registry_active"; state["registry"] = {"manifest": record.payload["manifest"], "manifest_sha256": record.payload["manifest_sha256"], "file_count": record.payload["file_count"]}; _save(run_dir, state)
        if failure_hook: failure_hook("after_record")
        run = {"schema_version": 1, "project": "particles2SNR-pipeline", "run_id": run_dir.name, "kind": "immutable_dataset_promotion", "status": "complete_active_dataset", **state}
        _atomic_json(run_dir / "run.json", run)
        return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, default=CANDIDATE); parser.add_argument("--destination", type=Path, default=Path("datasets") / DESTINATION_RELATIVE); parser.add_argument("--run-dir", type=Path, default=RUN_RELATIVE)
    args = parser.parse_args(); workspace = Workspace.load()
    resolve = lambda path: path if path.is_absolute() else workspace.root / path
    print(json.dumps(promote(workspace, candidate=resolve(args.candidate), destination=resolve(args.destination), run_dir=resolve(args.run_dir)), indent=2, sort_keys=True))


if __name__ == "__main__": main()
