#!/usr/bin/env python3
"""Promote the approved deterministic Z8 Cholesky synthetic v2 candidate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from internship_workspace.config import Workspace
from internship_workspace.datasets import (
    build_manifest,
    index_path,
    load_records,
    register_record,
    validate_record,
)
from internship_workspace.visual_review_store import ReviewStore, ReviewStoreError


DATASET_ID = (
    "particles2snr-fbase-z8-cholesky-physicalcorr-effective-snr-"
    "synthetic-events"
)
VERSION = "v2"
DATASET_KEY = f"{DATASET_ID}@{VERSION}"
CANDIDATE_RELATIVE = (
    Path("datasets/interim/particles2SNR-pipeline") / DATASET_ID / "v2-r3"
)
DESTINATION_RELATIVE = Path("processed") / DATASET_ID / VERSION
GENERATION_RUN_RELATIVE = Path(
    "artifacts/particles2SNR-pipeline/runs/"
    "particle-z8-cholesky-synthetic-generation-v2-r3"
)
CHECKPOINT_RELATIVE = Path(
    "artifacts/cross-project/reviews/"
    "particle-z8-v2-synthetic-generation-result-r1"
)
PROMOTION_RUN_RELATIVE = Path(
    "artifacts/particles2SNR-pipeline/runs/"
    "particle-z8-cholesky-synthetic-v2-promotion-r1"
)
EXPECTED_PAYLOAD = {
    "dataset_summary.json": (
        "0730e8240d255c7f9854e9eeb9ea06abd7a36900de19948b9db54ff9613dcfe8"
    ),
    "events.csv": (
        "972734d4f59b99a12246f145f845ce4c15e78f04fbe493b03254ba39fc451e4d"
    ),
    "input_contract.json": (
        "64b2a42a339991d5e6e3dc686c30c9aefd1833008b916c69e74f2e44742a6242"
    ),
    "signals_conv1dgap_512.npy": (
        "f775e9d1e1d28cc03b69334204194f13347e54caa05438093fef22bdae853bf2"
    ),
    "signals_raw_4096.npy": (
        "131a138f34b1e3608753e7caf910613d67deba218fe51e8534fbdcabbe8be2c1"
    ),
}
EXPECTED_FINGERPRINT = (
    "f86edd3231f38f38b3d0d89ff6150bd26beb4b0cd03c836ed208c5df34442594"
)
DATA_FORMAT = "synthetic-event-arrays-z8-cholesky-v2"
COMMAND = (
    "promote_z8_cholesky_synthetic_v2.py; approved deterministic v2 release"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _relative(workspace: Workspace, path: Path) -> str:
    return path.resolve().relative_to(workspace.root.resolve()).as_posix()


def _payload_inventory(root: Path) -> list[dict[str, Any]]:
    actual = {path.name for path in root.iterdir() if path.is_file()}
    if actual != set(EXPECTED_PAYLOAD):
        raise ValueError(f"Unexpected candidate payload: {sorted(actual)}")
    inventory = []
    for name in sorted(EXPECTED_PAYLOAD):
        path = root / name
        digest = sha256_file(path)
        if digest != EXPECTED_PAYLOAD[name]:
            raise ValueError(f"Approved payload hash mismatch: {name}")
        inventory.append(
            {"path": name, "sha256": digest, "size": path.stat().st_size}
        )
    return inventory


def assert_supported_checkpoint(workspace: Workspace) -> dict[str, Any]:
    checkpoint = workspace.root / CHECKPOINT_RELATIVE
    run = json.loads((checkpoint / "run.json").read_text(encoding="utf-8"))
    if (
        run.get("run_id") != CHECKPOINT_RELATIVE.name
        or run.get("status") != "visual_review_complete"
        or run.get("visual_approval", {}).get("decision") != "supported"
        or not run.get("visual_checkpoint", {}).get("approved")
        or run.get("visual_checkpoint", {}).get("next_stage_blocked")
    ):
        raise ValueError("Synthetic v2 result checkpoint does not authorize promotion")
    store = ReviewStore(checkpoint)
    try:
        receipt = store.verify_receipt()
    except ReviewStoreError as exc:
        raise ValueError(f"Invalid visual review receipt: {exc}") from exc
    contract = store.contract()
    current = store.current()
    evidence_id = "particle-z8-v2-synthetic-generation-result"
    if (
        contract.get("evidence_id") != evidence_id
        or current.get("decisions", {}).get(evidence_id, {}).get("decision")
        != "supported"
    ):
        raise ValueError("Receipt does not support the synthetic v2 result")
    reference = contract.get("analysis_reference", {})
    if (
        reference.get("run_id") != GENERATION_RUN_RELATIVE.name
        or reference.get("computation_fingerprint") != EXPECTED_FINGERPRINT
    ):
        raise ValueError("Checkpoint does not bind the approved generation run")
    generation_run = workspace.root / GENERATION_RUN_RELATIVE
    metrics_manifest = json.loads(
        (generation_run / "metrics_manifest.json").read_text(encoding="utf-8")
    )
    if metrics_manifest.get("computation_fingerprint") != EXPECTED_FINGERPRINT:
        raise ValueError("Generation metrics fingerprint changed after review")
    reference_metrics = {
        row["path"]: row["sha256"] for row in reference.get("metrics", [])
    }
    manifest_metrics = {
        row["path"]: row["sha256"] for row in metrics_manifest.get("metrics", [])
    }
    if reference_metrics != manifest_metrics:
        raise ValueError("Reviewed metric hashes do not match the generation run")
    for name, digest in manifest_metrics.items():
        if sha256_file(generation_run / name) != digest:
            raise ValueError(f"Reviewed generation metric changed: {name}")
    return {
        "checkpoint": CHECKPOINT_RELATIVE.as_posix(),
        "run_sha256": sha256_file(checkpoint / "run.json"),
        "receipt_sha256": sha256_file(store.receipt_path),
        "contract_sha256": receipt["contract_sha256"],
        "reviewer": receipt["reviewer"],
    }


def verify_candidate(workspace: Workspace) -> dict[str, Any]:
    candidate = workspace.root / CANDIDATE_RELATIVE
    inventory = _payload_inventory(candidate)
    summary = json.loads(
        (candidate / "dataset_summary.json").read_text(encoding="utf-8")
    )
    contract = json.loads(
        (candidate / "input_contract.json").read_text(encoding="utf-8")
    )
    generation_run = json.loads(
        (workspace.root / GENERATION_RUN_RELATIVE / "run.json").read_text(
            encoding="utf-8"
        )
    )
    candidate_manifest = json.loads(
        (
            workspace.root
            / GENERATION_RUN_RELATIVE
            / "candidate_dataset_manifest.json"
        ).read_text(encoding="utf-8")
    )
    if (
        summary.get("dataset_id") != DATASET_KEY
        or summary.get("event_count") != 4798
        or summary.get("class_counts")
        != {"2um": 1151, "4um": 3281, "10um": 366}
        or summary.get("seed") != 20260723
        or summary.get("sealed_test_accessed") is not False
    ):
        raise ValueError("Candidate summary violates the approved v2 contract")
    if (
        contract.get("raw_signals", {}).get("shape") != [4798, 4096]
        or contract.get("conv1dgap_signals", {}).get("shape") != [4798, 512]
    ):
        raise ValueError("Candidate signal contract has unexpected shapes")
    with (candidate / "events.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if (
        len(rows) != 4798
        or len({row["sample_id"] for row in rows}) != 4798
        or {row["class_name"] for row in rows} != {"2um", "4um", "10um"}
    ):
        raise ValueError("Candidate event table violates identity/count contracts")
    raw = np.load(candidate / "signals_raw_4096.npy", mmap_mode="r")
    encoded = np.load(candidate / "signals_conv1dgap_512.npy", mmap_mode="r")
    if (
        raw.shape != (4798, 4096)
        or encoded.shape != (4798, 512)
        or raw.dtype != np.float32
        or encoded.dtype != np.float32
        or not np.all(np.isfinite(raw))
        or not np.all(np.isfinite(encoded))
    ):
        raise ValueError("Candidate arrays violate shape, dtype, or finiteness")
    manifest_hashes = {
        row["path"]: row["sha256"] for row in candidate_manifest.get("files", [])
    }
    if manifest_hashes != EXPECTED_PAYLOAD:
        raise ValueError("Generation candidate manifest differs from frozen hashes")
    if (
        generation_run.get("run_id") != GENERATION_RUN_RELATIVE.name
        or generation_run.get("computation_fingerprint") != EXPECTED_FINGERPRINT
        or generation_run.get("candidate_dataset", {}).get("id") != DATASET_KEY
    ):
        raise ValueError("Generation run does not identify the approved candidate")
    return {
        "candidate": CANDIDATE_RELATIVE.as_posix(),
        "payload_inventory": inventory,
        "generation_run": GENERATION_RUN_RELATIVE.as_posix(),
        "generation_run_sha256": sha256_file(
            workspace.root / GENERATION_RUN_RELATIVE / "run.json"
        ),
        "deterministic_seed": 20260723,
        "event_count": 4798,
        "sealed_test_accessed": False,
    }


def _release_manifest(
    *,
    approval: dict[str, Any],
    qualification: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "dataset": DATASET_KEY,
        "status": "active_immutable_release",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": (
            "particles2snr-fbase-dual-clean-z8-events-3class-plus-unclear-"
            "development@v2"
        ),
        "approval": approval,
        "qualification": qualification,
        "scope": "Development-derived synthetic event arrays; sealed test excluded.",
        "v1_immutable": True,
        "known_limit": (
            "Maximum realized off-diagonal correlation delta is 0.146629; "
            "accepted by the scientific result checkpoint as diagnostic."
        ),
    }


def _assert_registered(workspace: Workspace, manifest_hash: str, count: int) -> None:
    matches = [record for record in load_records(workspace) if record.key == DATASET_KEY]
    if len(matches) != 1:
        raise RuntimeError("Expected exactly one registered synthetic v2 record")
    record = matches[0]
    expected_path = DESTINATION_RELATIVE.as_posix()
    if (
        record.payload.get("status") != "active"
        or record.payload.get("path") != expected_path
        or record.payload.get("manifest_sha256") != manifest_hash
        or record.payload.get("file_count") != count
    ):
        raise RuntimeError("Registered synthetic v2 record differs from release")
    errors = validate_record(workspace, record, full=True)
    if errors:
        raise RuntimeError("Registered release validation failed:\n" + "\n".join(errors))


def promote(workspace: Workspace, *, verify_only: bool = False) -> dict[str, Any]:
    candidate = workspace.root / CANDIDATE_RELATIVE
    destination = workspace.datasets_root / DESTINATION_RELATIVE
    run_dir = workspace.root / PROMOTION_RUN_RELATIVE
    approval = assert_supported_checkpoint(workspace)
    qualification = verify_candidate(workspace)
    if verify_only:
        return {
            "status": "verified_approved_candidate",
            "dataset": DATASET_KEY,
            "approval": approval,
            "qualification": qualification,
        }

    existing = [record for record in load_records(workspace) if record.key == DATASET_KEY]
    if destination.exists() or existing:
        if not destination.is_dir() or len(existing) != 1:
            raise FileExistsError("Partial or conflicting synthetic v2 release exists")
        count, manifest_hash = build_manifest(
            destination,
            run_dir / "release_payload_manifest.jsonl",
        )
        _assert_registered(workspace, manifest_hash, count)
        return {
            "status": "already_active",
            "dataset": DATASET_KEY,
            "path": _relative(workspace, destination),
            "manifest_sha256": manifest_hash,
        }

    run_dir.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.promotion-staging")
    if staging.exists():
        raise FileExistsError(f"Promotion staging already exists: {staging}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(candidate, staging, copy_function=shutil.copy2)
    _payload_inventory(staging)
    release = _release_manifest(approval=approval, qualification=qualification)
    _atomic_json(staging / "release_manifest.json", release)
    os.replace(staging, destination)

    count, manifest_hash = build_manifest(
        destination,
        run_dir / "release_payload_manifest.jsonl",
    )
    record = register_record(
        workspace,
        dataset_id=DATASET_ID,
        version=VERSION,
        relative_path=DESTINATION_RELATIVE.as_posix(),
        status="active",
        producer="particles2SNR-pipeline",
        data_format=DATA_FORMAT,
        command=COMMAND,
    )
    if record.payload.get("manifest_sha256") != manifest_hash:
        raise RuntimeError("Registry manifest differs from pre-registration manifest")
    _assert_registered(workspace, manifest_hash, count)
    run = {
        "schema_version": 1,
        "project": "particles2SNR-pipeline",
        "run_id": PROMOTION_RUN_RELATIVE.name,
        "kind": "immutable_dataset_promotion",
        "status": "complete_active_dataset",
        "dataset": DATASET_KEY,
        "destination": DESTINATION_RELATIVE.as_posix(),
        "approval": approval,
        "qualification": qualification,
        "registry": {
            "manifest": index_path(workspace).parent
            .joinpath(f"{DATASET_ID}-{VERSION}.jsonl")
            .relative_to(workspace.root)
            .as_posix(),
            "manifest_sha256": manifest_hash,
            "file_count": count,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(run_dir / "run.json", run)
    return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            promote(Workspace.load(), verify_only=args.verify_only),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
