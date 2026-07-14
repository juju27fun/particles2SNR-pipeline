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
            f"{canonical_row['source_group']}:{canonical_row['filename_series']}:"
            f"block-{canonical_block_number:04d}"
        )
        for row in family:
            group = row["source_group"]
            condition = CONDITION_POLICY[group]
            capture_index = int(row["filename_index"]) if row.get("filename_index") else -1
            nominal_block_number = capture_index // capture_block_size if capture_index >= 0 else -1
            nominal_block_id = f"{group}:{row['filename_series']}:block-{nominal_block_number:04d}"
            output.append(
                {
                    "record_id": hashlib.sha256(row["relative_path"].encode()).hexdigest()[:20],
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


def summarize_source_index(rows: list[dict[str, Any]]) -> dict[str, Any]:
    canonical = [row for row in rows if row["is_canonical_duplicate_member"]]
    family_splits: dict[str, set[str]] = defaultdict(set)
    block_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        family_splits[row["duplicate_family_id"]].add(row["development_split"])
        block_splits[row["capture_block_id"]].add(row["development_split"])
    acquisitions = sorted({row["acquisition_id"] for row in rows})
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
        "acquisition_ood_ready": len(acquisitions) >= 2,
        "scientific_limit": (
            "Acquisition-condition folders are not event-level biological labels, and the single documented acquisition cannot support acquisition-level OOD evaluation."
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
