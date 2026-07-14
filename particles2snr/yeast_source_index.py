from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CONDITION_POLICY: dict[str, dict[str, Any]] = {
    "budding": {
        "condition_id": "exponential-budding",
        "condition_description": "exponential-phase budding yeast",
        "concentration_regime": "documented-normal",
        "concentration_million_per_ml": 1.53,
        "label_scope": "acquisition-condition-proxy",
    },
    "mix": {
        "condition_id": "stationary-mixed",
        "condition_description": "stationary-phase mixture of budding and single-cell yeast",
        "concentration_regime": "documented-high",
        "concentration_million_per_ml": 30.87,
        "label_scope": "mixed-acquisition-condition-not-event-label",
    },
    "shmoo": {
        "condition_id": "alpha-factor-shmoo-low-concentration",
        "condition_description": "alpha-factor shmoo yeast, very low concentration subset",
        "concentration_regime": "readme-very-low",
        "concentration_million_per_ml": None,
        "label_scope": "acquisition-condition-proxy",
    },
    "shmoo2": {
        "condition_id": "alpha-factor-shmoo-normal-concentration",
        "condition_description": "alpha-factor shmoo yeast, normal concentration subset",
        "concentration_regime": "readme-normal",
        "concentration_million_per_ml": 2.484,
        "label_scope": "acquisition-condition-proxy",
    },
}


def read_source_inventory(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("suffix") == ".npy"]
    if not rows:
        raise ValueError(f"No .npy rows in source inventory: {path}")
    return rows


def _manifest_relative_path(path: str, manifest: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else manifest.parent / candidate


def _assign_group_stratified_block_splits(
    block_groups: dict[str, str],
) -> dict[str, str]:
    """Assign complete capture blocks while retaining each sufficiently large group."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for block_id, group in block_groups.items():
        grouped[group].append(block_id)

    assignments: dict[str, str] = {}
    for group, block_ids in sorted(grouped.items()):
        ordered = sorted(
            block_ids,
            key=lambda block_id: hashlib.sha256(
                f"yeast-development-v2:{block_id}".encode()
            ).hexdigest(),
        )
        n_blocks = len(ordered)
        if n_blocks < 3:
            assignments.update({block_id: "development_train" for block_id in ordered})
            continue
        n_validation = max(1, int(round(0.10 * n_blocks)))
        n_test = max(1, int(round(0.10 * n_blocks)))
        while n_validation + n_test >= n_blocks:
            if n_validation >= n_test and n_validation > 1:
                n_validation -= 1
            elif n_test > 1:
                n_test -= 1
            else:
                break
        for index, block_id in enumerate(ordered):
            if index < n_validation:
                split = "development_validation"
            elif index < n_validation + n_test:
                split = "in_session_test"
            else:
                split = "development_train"
            assignments[block_id] = split
    return assignments


def build_source_index(
    rows: list[dict[str, str]],
    *,
    raw_dataset: str,
    acquisition_id: str,
    capture_block_size: int = 64,
) -> list[dict[str, Any]]:
    if capture_block_size <= 0:
        raise ValueError("capture_block_size must be positive")
    by_digest: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        group = row["source_group"]
        if group not in CONDITION_POLICY:
            raise ValueError(f"No condition policy for source group: {group}")
        by_digest[row["sha256"]].append(row)

    output: list[dict[str, Any]] = []
    for digest, family in sorted(by_digest.items()):
        family = sorted(family, key=lambda item: item["relative_path"])
        canonical_path = family[0]["relative_path"]
        groups = {item["source_group"] for item in family}
        if len(groups) != 1:
            raise ValueError(f"Duplicate family crosses source groups: {sorted(groups)}")
        canonical_row = family[0]
        canonical_index = (
            int(canonical_row["filename_index"]) if canonical_row.get("filename_index") else -1
        )
        canonical_block_number = (
            canonical_index // capture_block_size if canonical_index >= 0 else -1
        )
        split_block_id = (
            f"{acquisition_id}:{canonical_row['source_group']}:"
            f"{canonical_row['filename_series']}:"
            f"block-{canonical_block_number:04d}"
        )
        for row in family:
            group = row["source_group"]
            condition = CONDITION_POLICY[group]
            capture_index = int(row["filename_index"]) if row.get("filename_index") else -1
            nominal_block_number = capture_index // capture_block_size if capture_index >= 0 else -1
            nominal_block_id = (
                f"{acquisition_id}:{group}:{row['filename_series']}:"
                f"block-{nominal_block_number:04d}"
            )
            output.append(
                {
                    "record_id": hashlib.sha256(
                        f"{raw_dataset}:{row['relative_path']}".encode()
                    ).hexdigest()[:20],
                    "raw_dataset": raw_dataset,
                    "relative_path": row["relative_path"],
                    "source_group": group,
                    **condition,
                    "acquisition_id": acquisition_id,
                    "capture_series": row["filename_series"],
                    "capture_index": capture_index,
                    "nominal_capture_block_id": nominal_block_id,
                    "capture_block_id": split_block_id,
                    "duplicate_family_id": digest,
                    "duplicate_family_size": len(family),
                    "is_canonical_duplicate_member": row["relative_path"] == canonical_path,
                    "development_split": "pending-group-stratified-assignment",
                    "split_unit": "capture-block-proxy",
                    "acquisition_role": "development",
                    "evidence_scope": "single-acquisition-development-only",
                    "sha256": digest,
                    "size_bytes": int(row["size_bytes"]),
                    "n_values": int(row["n_values"]),
                    "dtype": row["npy_dtype"],
                }
            )
    block_groups = {row["capture_block_id"]: row["source_group"] for row in output}
    assignments = _assign_group_stratified_block_splits(block_groups)
    for row in output:
        row["development_split"] = assignments[row["capture_block_id"]]
    return sorted(output, key=lambda item: item["relative_path"])


def combine_acquisition_indexes(
    indexes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Combine independently indexed acquisitions and seal declared OOD rows."""
    if len(indexes) < 2:
        raise ValueError("A multi-acquisition index requires at least two acquisitions")

    acquisition_ids = [str(item["acquisition_id"]) for item in indexes]
    raw_datasets = [str(item["raw_dataset"]) for item in indexes]
    if len(acquisition_ids) != len(set(acquisition_ids)):
        raise ValueError("Acquisition IDs must be unique")
    if len(raw_datasets) != len(set(raw_datasets)):
        raise ValueError("Raw dataset IDs must be unique per acquisition")

    allowed_roles = {"development", "sealed_ood_test"}
    roles = {str(item["role"]) for item in indexes}
    invalid_roles = roles - allowed_roles
    if invalid_roles:
        raise ValueError(f"Unsupported acquisition roles: {sorted(invalid_roles)}")
    if "development" not in roles or "sealed_ood_test" not in roles:
        raise ValueError("At least one development and one sealed_ood_test acquisition are required")

    combined: list[dict[str, Any]] = []
    digest_acquisitions: dict[str, set[str]] = defaultdict(set)
    for item in indexes:
        acquisition_id = str(item["acquisition_id"])
        raw_dataset = str(item["raw_dataset"])
        role = str(item["role"])
        rows = list(item["rows"])
        if not rows:
            raise ValueError(f"Acquisition {acquisition_id} has no indexed rows")
        for row in rows:
            if row["acquisition_id"] != acquisition_id or row["raw_dataset"] != raw_dataset:
                raise ValueError(f"Index metadata mismatch for acquisition {acquisition_id}")
            digest_acquisitions[row["sha256"]].add(acquisition_id)
            updated = dict(row)
            updated["acquisition_role"] = role
            if role == "sealed_ood_test":
                updated["development_split"] = "sealed_acquisition_test"
                updated["split_unit"] = "acquisition"
                updated["evidence_scope"] = "sealed-acquisition-ood-only"
            else:
                updated["evidence_scope"] = "multi-acquisition-development"
            combined.append(updated)

    crossing = {
        digest: sorted(acquisitions)
        for digest, acquisitions in digest_acquisitions.items()
        if len(acquisitions) > 1
    }
    if crossing:
        examples = list(sorted(crossing.items()))[:3]
        raise ValueError(
            "Exact signal duplicates cross declared independent acquisitions; "
            f"audit provenance before proceeding: {examples}"
        )

    record_ids = [row["record_id"] for row in combined]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("Record IDs collide across acquisitions")
    return sorted(combined, key=lambda row: (row["acquisition_id"], row["relative_path"]))


def build_source_index_from_manifest(
    path: Path,
    *,
    capture_block_size: int = 64,
) -> list[dict[str, Any]]:
    manifest = path.resolve()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Acquisition manifest schema_version must be 1")
    acquisitions = payload.get("acquisitions")
    if not isinstance(acquisitions, list):
        raise ValueError("Acquisition manifest must contain an acquisitions list")

    indexes: list[dict[str, Any]] = []
    for acquisition in acquisitions:
        if not isinstance(acquisition, dict):
            raise ValueError("Each acquisition manifest entry must be an object")
        required = {"acquisition_id", "raw_dataset", "source_inventory", "role"}
        missing = required - set(acquisition)
        if missing:
            raise ValueError(f"Acquisition manifest entry is missing {sorted(missing)}")
        acquisition_id = str(acquisition["acquisition_id"])
        raw_dataset = str(acquisition["raw_dataset"])
        rows = build_source_index(
            read_source_inventory(
                _manifest_relative_path(str(acquisition["source_inventory"]), manifest)
            ),
            raw_dataset=raw_dataset,
            acquisition_id=acquisition_id,
            capture_block_size=capture_block_size,
        )
        indexes.append(
            {
                "acquisition_id": acquisition_id,
                "raw_dataset": raw_dataset,
                "role": str(acquisition["role"]),
                "rows": rows,
            }
        )
    return combine_acquisition_indexes(indexes)


def summarize_source_index(rows: list[dict[str, Any]]) -> dict[str, Any]:
    canonical = [row for row in rows if row["is_canonical_duplicate_member"]]
    family_splits: dict[str, set[str]] = defaultdict(set)
    block_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        family_splits[row["duplicate_family_id"]].add(row["development_split"])
        block_splits[row["capture_block_id"]].add(row["development_split"])
    acquisitions = sorted({row["acquisition_id"] for row in rows})
    acquisition_roles = {
        acquisition: sorted(
            {
                row.get("acquisition_role", "development")
                for row in rows
                if row["acquisition_id"] == acquisition
            }
        )
        for acquisition in acquisitions
    }
    group_split_counts = {
        group: dict(
            sorted(
                Counter(
                    row["development_split"] for row in canonical if row["source_group"] == group
                ).items()
            )
        )
        for group in sorted({row["source_group"] for row in canonical})
    }
    groups_with_all_splits = [
        group
        for group, counts in group_split_counts.items()
        if set(counts) == {"development_train", "development_validation", "in_session_test"}
    ]
    return {
        "schema_version": 1,
        "n_raw_rows": len(rows),
        "n_canonical_rows": len(canonical),
        "n_duplicate_excess_rows": len(rows) - len(canonical),
        "source_group_counts_raw": dict(sorted(Counter(row["source_group"] for row in rows).items())),
        "source_group_counts_canonical": dict(
            sorted(Counter(row["source_group"] for row in canonical).items())
        ),
        "condition_counts_canonical": dict(
            sorted(Counter(row["condition_id"] for row in canonical).items())
        ),
        "development_split_counts_canonical": dict(
            sorted(Counter(row["development_split"] for row in canonical).items())
        ),
        "development_split_policy": "deterministic-group-stratified-capture-block-v2",
        "source_group_split_counts_canonical": group_split_counts,
        "source_groups_with_all_development_splits": groups_with_all_splits,
        "n_capture_blocks": len(block_splits),
        "n_duplicate_families_crossing_splits": sum(len(splits) > 1 for splits in family_splits.values()),
        "n_capture_blocks_crossing_splits": sum(len(splits) > 1 for splits in block_splits.values()),
        "documented_acquisition_ids": acquisitions,
        "acquisition_roles": acquisition_roles,
        "acquisition_split_counts_canonical": {
            acquisition: dict(
                sorted(
                    Counter(
                        row["development_split"]
                        for row in canonical
                        if row["acquisition_id"] == acquisition
                    ).items()
                )
            )
            for acquisition in acquisitions
        },
        "acquisition_ood_ready": (
            len(acquisitions) >= 2
            and any("development" in roles for roles in acquisition_roles.values())
            and any("sealed_ood_test" in roles for roles in acquisition_roles.values())
        ),
        "scientific_limit": (
            "Acquisition-condition folders are not event-level biological labels. "
            + (
                "The sealed acquisition may be used only after protocol freeze."
                if len(acquisitions) >= 2
                else "The single documented acquisition cannot support acquisition-level OOD evaluation."
            )
        ),
    }


def write_source_index(output_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "source_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "condition_policy.json").write_text(
        json.dumps(CONDITION_POLICY, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
