from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .particle_events import (
    ParticleDetectionConfig,
    config_fingerprint,
    detect_particle_events,
)


REQUIRED_SOURCE_FIELDS = {
    "split",
    "class",
    "filename",
    "raw_path",
    "raw_sha256",
    "staging_action",
    "repair_region_count",
}
POPULATION_ROLES = {"mad_calibration", "mad_holdout", "legacy_exploration"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable(path: Path, workspace_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(workspace_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"path is outside workspace: {path}") from exc


def read_source_inventory(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = REQUIRED_SOURCE_FIELDS - fields
        if missing:
            raise ValueError(f"source inventory missing fields: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("source inventory is empty")
    filenames = [row["filename"] for row in rows]
    if len(filenames) != len(set(filenames)):
        raise ValueError("source inventory filenames must be unique")
    return rows


def assign_population_roles(
    rows: list[dict[str, str]],
    *,
    seed: int = 20_260_731,
    holdout_fraction: float = 0.20,
) -> list[dict[str, str]]:
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be in (0, 1)")
    train_strata: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    output: list[dict[str, str]] = []
    for row in rows:
        if row["split"] == "val":
            output.append({**row, "population_role": "legacy_exploration"})
        elif row["split"] == "train":
            repair = "repaired" if int(row["repair_region_count"]) > 0 else "clean"
            train_strata[(row["class"], repair)].append(row)
        else:
            raise ValueError(f"unexpected historical split: {row['split']}")
    for stratum, members in sorted(train_strata.items()):
        ordered = sorted(
            members,
            key=lambda row: hashlib.sha256(
                f"{seed}:{row['raw_sha256']}".encode("utf-8")
            ).hexdigest(),
        )
        holdout_count = int(math.ceil(len(ordered) * holdout_fraction))
        for index, row in enumerate(ordered):
            role = "mad_holdout" if index < holdout_count else "mad_calibration"
            output.append(
                {
                    **row,
                    "population_role": role,
                    "population_stratum": f"{stratum[0]}:{stratum[1]}",
                }
            )
    return sorted(output, key=lambda row: row["filename"])


def validate_holdout_authorization(
    authorization_path: Path, *, config_sha256: str
) -> None:
    if authorization_path.name != "run.json":
        raise PermissionError("holdout authorization must be a reviewed run.json")
    payload = json.loads(authorization_path.read_text(encoding="utf-8"))
    if payload.get("status") != "visual_review_complete":
        raise PermissionError("holdout authorization is not a completed visual review")
    checkpoint = payload.get("visual_checkpoint", {})
    if checkpoint.get("approved") is not True or checkpoint.get("next_stage_blocked") is not False:
        raise PermissionError("holdout authorization gate is not open")
    spec_path = authorization_path.parent / "checkpoint_spec.json"
    spec = (
        json.loads(spec_path.read_text(encoding="utf-8"))
        if spec_path.is_file()
        else {}
    )
    frozen_config = payload.get("frozen_config_sha256") or spec.get(
        "frozen_config_sha256"
    )
    if frozen_config != config_sha256:
        raise PermissionError("holdout authorization does not match detector config")
    run_dir = authorization_path.parent
    receipt_path = run_dir / "review" / "receipt.json"
    if not receipt_path.is_file():
        raise PermissionError("holdout authorization has no verified review receipt")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    contract_path = run_dir / "review_contract.json"
    decisions_path = run_dir / "review" / "decisions.json"
    if not contract_path.is_file() or not decisions_path.is_file():
        raise PermissionError("holdout authorization review contract is incomplete")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    if (
        receipt.get("schema_version") != 1
        or receipt.get("run_id") != payload.get("run_id")
        or contract.get("run_id") != payload.get("run_id")
        or decisions.get("run_id") != payload.get("run_id")
    ):
        raise PermissionError("holdout authorization receipt belongs to another run")
    checks = {
        "review_complete": decisions.get("complete") is True,
        "reviewer": receipt.get("reviewer") == decisions.get("reviewer"),
        "decision_count": receipt.get("decision_count")
        == len(decisions.get("decisions", {})),
        "decisions_file": receipt.get("decisions_file") == "review/decisions.json",
        "contract_file": receipt.get("contract_file") == "review_contract.json",
        "decisions_sha256": receipt.get("decisions_sha256")
        == _sha256(decisions_path),
        "contract_sha256": receipt.get("contract_sha256") == _sha256(contract_path),
    }
    primary_assets = receipt.get("primary_assets", [])
    if primary_assets != contract.get("primary_assets", []):
        checks["primary_assets"] = False
    else:
        checks["primary_assets"] = all(
            isinstance(item, dict)
            and not Path(str(item.get("path", ""))).is_absolute()
            and ".." not in Path(str(item.get("path", ""))).parts
            and (run_dir / str(item.get("path", ""))).is_file()
            and _sha256(run_dir / str(item["path"])) == item.get("sha256")
            for item in primary_assets
        )
    failed = sorted(name for name, valid in checks.items() if not valid)
    if failed:
        raise PermissionError(
            f"holdout authorization review receipt mismatch: {failed}"
        )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path.name}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_repair_regions(path: Path | None) -> dict[str, list[tuple[int, int]]]:
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"filename", "expanded_start_sample", "expanded_end_sample"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"repair manifest missing fields: {sorted(missing)}")
        regions: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for row in reader:
            left = int(row["expanded_start_sample"])
            right = int(row["expanded_end_sample"])
            if right <= left:
                raise ValueError("repair manifest contains an invalid interval")
            regions[row["filename"]].append((left, right))
    return dict(regions)


def build_candidate_dataset(
    *,
    workspace_root: Path,
    source_inventory: Path,
    signal_dataset_root: Path,
    output_dir: Path,
    dataset_id: str,
    input_dataset_id: str,
    input_manifest_sha256: str,
    config: ParticleDetectionConfig,
    roles: Iterable[str],
    repair_manifest: Path | None = None,
    seed: int = 20_260_731,
    holdout_authorization: Path | None = None,
) -> dict[str, Any]:
    workspace_root = workspace_root.resolve()
    output_dir = output_dir.resolve()
    for path in (source_inventory, signal_dataset_root, output_dir.parent):
        _portable(path, workspace_root)
    if len(input_manifest_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in input_manifest_sha256.lower()
    ):
        raise ValueError("input_manifest_sha256 must be a SHA-256 hex digest")
    requested_roles = set(roles)
    if not requested_roles or not requested_roles <= POPULATION_ROLES:
        raise ValueError("roles must be a non-empty subset of known population roles")
    fingerprint = config_fingerprint(config)
    if "mad_holdout" in requested_roles:
        if holdout_authorization is None:
            raise PermissionError("mad_holdout requires a completed authorization")
        validate_holdout_authorization(
            holdout_authorization, config_sha256=fingerprint
        )
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable output: {output_dir}")
    rows = assign_population_roles(read_source_inventory(source_inventory), seed=seed)
    repair_regions = read_repair_regions(repair_manifest)
    selected = [row for row in rows if row["population_role"] in requested_roles]
    if not selected:
        raise ValueError("selected population is empty")
    repaired_selected = [
        row for row in selected if int(row["repair_region_count"]) > 0
    ]
    if repaired_selected and repair_manifest is None:
        raise ValueError("selected repaired sources require a repair manifest")
    for row in repaired_selected:
        expected = int(row["repair_region_count"])
        observed = len(repair_regions.get(row["filename"], ()))
        if observed != expected:
            raise ValueError(
                f"repair interval count mismatch for {row['filename']}: "
                f"expected {expected}, observed {observed}"
            )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        candidate_rows: list[dict[str, Any]] = []
        source_rows: list[dict[str, Any]] = []
        population_rows: list[dict[str, Any]] = []
        for row in rows:
            population_rows.append(
                {
                    "filename": row["filename"],
                    "historical_split": row["split"],
                    "source_class": row["class"],
                    "repair_region_count": int(row["repair_region_count"]),
                    "population_role": row["population_role"],
                    "population_stratum": row.get("population_stratum", "legacy"),
                    "raw_sha256": row["raw_sha256"],
                }
            )
        for row in selected:
            signal_path = (
                signal_dataset_root
                / row["split"]
                / "signals"
                / row["filename"]
            )
            signal = np.load(signal_path, allow_pickle=False)
            candidates, _diagnostics = detect_particle_events(
                signal,
                config,
                repair_regions=repair_regions.get(row["filename"], ()),
            )
            retained = sum(candidate.quality == "retained" for candidate in candidates)
            source_rows.append(
                {
                    "filename": row["filename"],
                    "source_class": row["class"],
                    "population_role": row["population_role"],
                    "signal_length": int(np.asarray(signal).size),
                    "candidate_count": len(candidates),
                    "retained_count": retained,
                    "rejected_count": len(candidates) - retained,
                    "empty_retained": retained == 0,
                    "repair_region_count": int(row["repair_region_count"]),
                    "signal_path": _portable(signal_path, workspace_root),
                    "signal_sha256": _sha256(signal_path),
                }
            )
            for candidate in candidates:
                candidate_rows.append(
                    {
                        "event_id": f"{Path(row['filename']).stem}:mad:{candidate.candidate_index:02d}",
                        "filename": row["filename"],
                        "source_class": row["class"],
                        "population_role": row["population_role"],
                        **asdict(candidate),
                    }
                )
        _write_csv(temporary / "population_assignments.csv", population_rows)
        _write_csv(temporary / "source_detection_report.csv", source_rows)
        if candidate_rows:
            _write_csv(temporary / "candidate_events.csv", candidate_rows)
        else:
            (temporary / "candidate_events.csv").write_text(
                "event_id,filename,source_class,population_role\n", encoding="utf-8"
            )
        contract = {
            "schema_version": 1,
            "dataset_id": dataset_id,
            "grain": {
                "candidate_events.csv": "one MAD proposal",
                "source_detection_report.csv": "one executed physical source trace",
                "population_assignments.csv": "one source trace",
            },
            "keys": {
                "candidate_events.csv": ["event_id"],
                "source_detection_report.csv": ["filename"],
                "population_assignments.csv": ["filename"],
            },
            "units": {
                "center_index": "sample",
                "event_start": "sample",
                "event_end": "sample",
                "width_ms": "ms",
                "dominant_frequency_hz": "Hz",
                "spectral_bandwidth_hz": "Hz",
            },
            "class_policy": "source_class is context metadata and is never an input to event detection",
            "label_policy": "Z8 labels and historical event boxes are forbidden inputs",
        }
        _write_json(temporary / "dataset_contract.json", contract)
        payload_files = [
            "candidate_events.csv",
            "population_assignments.csv",
            "source_detection_report.csv",
            "dataset_contract.json",
        ]
        manifest = {
            "schema_version": 1,
            "dataset_id": dataset_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input_datasets": {
                input_dataset_id: input_manifest_sha256,
            },
            "source_inventory": _portable(source_inventory, workspace_root),
            "repair_manifest": (
                _portable(repair_manifest, workspace_root)
                if repair_manifest is not None
                else None
            ),
            "config": asdict(config),
            "config_sha256": fingerprint,
            "split_seed": seed,
            "roles_executed": sorted(requested_roles),
            "counts": {
                "population": len(population_rows),
                "sources_executed": len(source_rows),
                "candidates": len(candidate_rows),
                "retained": sum(row["quality"] == "retained" for row in candidate_rows),
            },
            "files": {
                name: {"sha256": _sha256(temporary / name), "size": (temporary / name).stat().st_size}
                for name in payload_files
            },
        }
        _write_json(temporary / "dataset_manifest.json", manifest)
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest
