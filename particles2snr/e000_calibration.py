from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


PROTOCOL_ID = "E000-bead-instrument-calibration"
PROTOCOL_REVISION = 2
PROTOCOL_DIGEST = "7e6b896291716adcc48bdb28d1249f26670b858f51f3665b913fdc510752201b"
STUDY_SCOPE = "shmoo_autonomous_study"
ACQUISITION_FIELDS = (
    "acquisition_id",
    "source_acquisition_id",
    "measurement_run_id",
)


@dataclass(frozen=True)
class DatasetBinding:
    dataset_id: str
    manifest_relative_path: str
    data_relative_path: str
    manifest_sha256: str
    manifest_file_count: int
    split: str


DATASETS = (
    DatasetBinding(
        dataset_id="c1-hf-5-10-2um-doublet@v1",
        manifest_relative_path=(
            "datasets/registry/c1-hf-5-10-2um-doublet-v1.jsonl"
        ),
        data_relative_path="datasets/raw/c1-hf-5-10-2um-doublet/v1",
        manifest_sha256=(
            "4c4c44a129a7e677cc9b3f5132bc828bdb0f848b02ddf2b7189b0600989861d8"
        ),
        manifest_file_count=1202,
        split="calibration_development",
    ),
    DatasetBinding(
        dataset_id="c1-hf-5-10-4um-doublet@v1",
        manifest_relative_path=(
            "datasets/registry/c1-hf-5-10-4um-doublet-v1.jsonl"
        ),
        data_relative_path="datasets/raw/c1-hf-5-10-4um-doublet/v1",
        manifest_sha256=(
            "110f3d0d221b2fa5f721350d9c64f6a3f7730fb0279c7582d6018131b1be4876"
        ),
        manifest_file_count=743,
        split="calibration_development",
    ),
    DatasetBinding(
        dataset_id="c1-hf-5-10-10um-doublet@v1",
        manifest_relative_path=(
            "datasets/registry/c1-hf-5-10-10um-doublet-v1.jsonl"
        ),
        data_relative_path="datasets/raw/c1-hf-5-10-10um-doublet/v1",
        manifest_sha256=(
            "ffb4efdfeaaaa6c38d4e481a9944532554e34a8bb09840a29762e658f3d3e8fa"
        ),
        manifest_file_count=943,
        split="calibration_confirmation",
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}: invalid JSON on line {line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}: line {line_number} is not an object")
            rows.append(row)
    return rows


def _dataset_audit(
    workspace_root: Path,
    binding: DatasetBinding,
    *,
    verify_content: bool,
) -> tuple[dict[str, object], list[dict[str, str]], list[str]]:
    manifest_path = workspace_root / binding.manifest_relative_path
    data_root = workspace_root / binding.data_relative_path
    errors: list[str] = []
    if not manifest_path.is_file():
        return (
            {"dataset_id": binding.dataset_id, "manifest_exists": False},
            [],
            [f"{binding.dataset_id}: registered manifest is missing"],
        )

    actual_manifest_digest = sha256_file(manifest_path)
    if actual_manifest_digest != binding.manifest_sha256:
        errors.append(
            f"{binding.dataset_id}: manifest digest differs from E000 revision 2"
        )
    rows = _read_manifest(manifest_path)
    if len(rows) != binding.manifest_file_count:
        errors.append(
            f"{binding.dataset_id}: manifest count {len(rows)} differs from "
            f"{binding.manifest_file_count}"
        )

    acquisition_fields = sorted(
        {
            field
            for row in rows
            for field in ACQUISITION_FIELDS
            if row.get(field) not in (None, "")
        }
    )
    content_rows: list[dict[str, str]] = []
    unreadable = 0
    nonfinite = 0
    unexpected_shape = 0
    size_mismatch = 0
    content_mismatch = 0
    shapes: set[tuple[int, ...]] = set()
    dtypes: set[str] = set()

    for row_number, row in enumerate(rows, start=1):
        relative = row.get("path")
        expected_sha = row.get("sha256")
        expected_size = row.get("size")
        if (
            not isinstance(relative, str)
            or not isinstance(expected_sha, str)
            or not isinstance(expected_size, int)
        ):
            errors.append(
                f"{binding.dataset_id}: row {row_number} lacks path/sha256/size"
            )
            continue
        content_rows.append(
            {
                "dataset_id": binding.dataset_id,
                "path": relative,
                "sha256": expected_sha,
                "split": binding.split,
            }
        )
        source = data_root / relative
        if not source.is_file():
            unreadable += 1
            continue
        if source.stat().st_size != expected_size:
            size_mismatch += 1
        if verify_content and sha256_file(source) != expected_sha:
            content_mismatch += 1
        try:
            values = np.load(source, allow_pickle=False)
        except (OSError, ValueError):
            unreadable += 1
            continue
        shapes.add(tuple(values.shape))
        dtypes.add(str(values.dtype))
        if values.shape != (16384,):
            unexpected_shape += 1
        if not np.isfinite(values).all():
            nonfinite += 1

    counters = {
        "unreadable_or_missing": unreadable,
        "nonfinite": nonfinite,
        "unexpected_shape": unexpected_shape,
        "size_mismatch": size_mismatch,
        "content_hash_mismatch": content_mismatch,
    }
    for name, count in counters.items():
        if count:
            errors.append(f"{binding.dataset_id}: {count} file(s) with {name}")

    return (
        {
            **asdict(binding),
            "manifest_exists": True,
            "actual_manifest_sha256": actual_manifest_digest,
            "actual_manifest_file_count": len(rows),
            "acquisition_fields_present": acquisition_fields,
            "array_shapes": [list(shape) for shape in sorted(shapes)],
            "array_dtypes": sorted(dtypes),
            "file_checks": counters,
        },
        content_rows,
        errors,
    )


def audit_preflight(
    workspace_root: Path,
    *,
    bindings: Iterable[DatasetBinding] = DATASETS,
    verify_content: bool = True,
) -> dict[str, object]:
    workspace_root = workspace_root.resolve()
    datasets: list[dict[str, object]] = []
    content_rows: list[dict[str, str]] = []
    blockers: list[dict[str, str]] = []
    integrity_errors: list[str] = []

    for binding in bindings:
        result, rows, errors = _dataset_audit(
            workspace_root, binding, verify_content=verify_content
        )
        datasets.append(result)
        content_rows.extend(rows)
        integrity_errors.extend(errors)

    if integrity_errors:
        blockers.append(
            {
                "code": "dataset_integrity_mismatch",
                "contract": "stopping_rules[0]",
                "detail": "; ".join(integrity_errors),
            }
        )

    by_digest: dict[str, list[dict[str, str]]] = {}
    for row in content_rows:
        by_digest.setdefault(row["sha256"], []).append(row)
    cross_split_duplicates = [
        {
            "sha256": digest,
            "members": members,
        }
        for digest, members in by_digest.items()
        if len({member["split"] for member in members}) > 1
    ]
    if cross_split_duplicates:
        blockers.append(
            {
                "code": "cross_split_duplicate_content",
                "contract": "grouping_contract.duplicate_rule",
                "detail": (
                    f"{len(cross_split_duplicates)} content digest(s) occur in "
                    "both development and confirmation"
                ),
            }
        )

    missing_acquisition_metadata = [
        str(dataset["dataset_id"])
        for dataset in datasets
        if not dataset.get("acquisition_fields_present")
    ]
    if missing_acquisition_metadata:
        blockers.append(
            {
                "code": "unresolved_acquisition_confounding",
                "contract": "grouping_contract.hard_stop",
                "detail": (
                    "The registered manifests expose only per-file identity and "
                    "do not bind an acquisition-level group for: "
                    + ", ".join(missing_acquisition_metadata)
                ),
            }
        )

    decision = "insufficient" if blockers else "preflight_passed"
    return {
        "schema_version": 1,
        "experiment_id": PROTOCOL_ID,
        "experiment_revision": PROTOCOL_REVISION,
        "experiment_digest": PROTOCOL_DIGEST,
        "study_scope": STUDY_SCOPE,
        "phase": "preflight",
        "verification_mode": "full_content" if verify_content else "manifest_only",
        "datasets": datasets,
        "cross_split_duplicate_count": len(cross_split_duplicates),
        "blockers": blockers,
        "decision": decision,
        "confirmation_opened": False,
        "development_fit_started": False,
    }


def _git_revision(path: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _computation_fingerprint(provenance: dict[str, object]) -> str:
    required = {
        "datasets",
        "inputs",
        "parameters",
        "metric_definitions",
        "code",
        "git_revision",
    }
    canonical = json.dumps(
        {key: provenance[key] for key in sorted(required)},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def write_preflight_run(
    workspace_root: Path,
    output_dir: Path,
    *,
    verify_content: bool = True,
    command: str,
) -> dict[str, object]:
    workspace_root = workspace_root.resolve()
    output_dir = output_dir.resolve()
    artifact_root = (
        workspace_root / "artifacts" / "particles2SNR-pipeline"
    ).resolve()
    if artifact_root not in output_dir.parents:
        raise ValueError(
            "E000 output must be below artifacts/particles2SNR-pipeline"
        )
    if (output_dir / "run.json").exists():
        existing = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
        if (
            existing.get("experiment", {}).get("digest") != PROTOCOL_DIGEST
            or existing.get("phase") != "preflight"
        ):
            raise ValueError("existing output is not the same E000 preflight")
        return existing

    output_dir.mkdir(parents=True, exist_ok=False)
    result = audit_preflight(
        workspace_root,
        verify_content=verify_content,
    )
    preflight_path = output_dir / "preflight.json"
    preflight_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    source_path = Path(__file__).resolve()
    source_revision = _git_revision(source_path.parents[1])
    provenance: dict[str, object] = {
        "datasets": [binding.dataset_id for binding in DATASETS],
        "inputs": {
            "experiment_digest": PROTOCOL_DIGEST,
            "population": "all rows in the three frozen E000 revision 2 manifests",
            "verification_mode": result["verification_mode"],
        },
        "parameters": {
            "acquisition_fields": list(ACQUISITION_FIELDS),
            "expected_array_shape": [16384],
            "allow_pickle": False,
        },
        "metric_definitions": {
            "preflight": (
                "Exact manifest and file identity, readable finite NPY shape, "
                "cross-split content uniqueness, and acquisition-group presence"
            )
        },
        "code": {
            "entrypoint": "scripts/analysis/run_e000_bead_calibration.py",
            "source_sha256": sha256_file(source_path),
        },
        "git_revision": source_revision,
    }
    metrics_manifest = {
        "schema_version": 1,
        "analysis_run_id": output_dir.name,
        "computation_provenance": provenance,
        "computation_fingerprint": _computation_fingerprint(provenance),
        "metrics": [
            {
                "path": "preflight.json",
                "sha256": sha256_file(preflight_path),
            }
        ],
    }
    (output_dir / "metrics_manifest.json").write_text(
        json.dumps(metrics_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    run = {
        "schema_version": 1,
        "project": "particles2SNR-pipeline",
        "run_id": output_dir.name,
        "dataset": " + ".join(binding.dataset_id for binding in DATASETS),
        "datasets": {
            binding.dataset_id: {
                "manifest": binding.manifest_relative_path,
                "manifest_sha256": binding.manifest_sha256,
                "file_count": binding.manifest_file_count,
                "split": binding.split,
            }
            for binding in DATASETS
        },
        "repositories": {
            "workspace": _git_revision(workspace_root),
            "particles2SNR-pipeline": source_revision,
        },
        "command": command,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "insufficient_preflight"
        if result["decision"] == "insufficient"
        else "preflight_complete",
        "phase": "preflight",
        "experiment": {
            "id": PROTOCOL_ID,
            "revision": PROTOCOL_REVISION,
            "digest": PROTOCOL_DIGEST,
            "study_scope": STUDY_SCOPE,
        },
        "implementation_binding": {
            "entrypoint": "scripts/analysis/run_e000_bead_calibration.py",
            "preflight_source_sha256": sha256_file(source_path),
            "dependency_lock": "requirements/workspace.lock.txt",
            "dependency_lock_sha256": sha256_file(
                workspace_root / "requirements/workspace.lock.txt"
            ),
        },
        "outputs": ["metrics_manifest.json", "preflight.json", "run.json"],
        "normalized_outcome": {
            "code": result["decision"],
            "confirmation_opened": False,
            "development_fit_started": False,
            "blocker_codes": [
                blocker["code"] for blocker in result["blockers"]
            ],
        },
        "claim_boundary": (
            "This run verifies E000 revision 2 input identity and grouping "
            "readiness only. It makes no calibration, mechanism, morphology, "
            "or confirmation claim."
        ),
    }
    (output_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return run
