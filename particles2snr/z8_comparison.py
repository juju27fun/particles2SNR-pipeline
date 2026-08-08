"""Deterministic, audit-only comparison of two Z8 event tables."""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


CATEGORIES = {
    "retained",
    "moved",
    "reclassified",
    "removed",
    "new",
    "ambiguous",
}


def read_events(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _interval(row: dict[str, Any]) -> tuple[float, float]:
    return float(row["start_norm"]), float(row["end_norm"])


def _iou(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_start, left_end = _interval(left)
    right_start, right_end = _interval(right)
    intersection = max(
        0.0, min(left_end, right_end) - max(left_start, right_start)
    )
    union = max(left_end, right_end) - min(left_start, right_start)
    return intersection / union if union > 0.0 else 0.0


def _center_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    return abs(float(left["center_norm"]) - float(right["center_norm"]))


def _physical_class(row: dict[str, Any]) -> str:
    return str(row.get("physical_source_class") or row["class_name"])


def _eligible(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if _physical_class(left) != _physical_class(right):
        return False
    iou = _iou(left, right)
    left_width = float(left["end_norm"]) - float(left["start_norm"])
    right_width = float(right["end_norm"]) - float(right["start_norm"])
    return iou > 0.0 or _center_distance(left, right) <= 0.5 * max(
        left_width, right_width
    )


def compare_events(
    old_rows: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return a one-to-one geometric audit without transferring identities."""
    old_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    new_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in old_rows:
        old_groups[(str(row["split"]), str(row["source_filename"]))].append(row)
    for row in new_rows:
        new_groups[(str(row["split"]), str(row["source_filename"]))].append(row)

    result = []
    for group_key in sorted(set(old_groups) | set(new_groups)):
        old_group = sorted(old_groups[group_key], key=lambda row: row["event_id"])
        new_group = sorted(new_groups[group_key], key=lambda row: row["event_id"])
        ambiguous_old_ids: set[str] = set()
        for old in old_group:
            candidates = [
                (
                    -_iou(old, new),
                    _center_distance(old, new),
                    str(new["event_id"]),
                    new,
                )
                for new in new_group
                if _eligible(old, new)
            ]
            candidates.sort(key=lambda item: item[:3])
            if not candidates:
                continue
            best = candidates[0]
            score = best[:2]
            ties = [
                item
                for item in candidates
                if abs(item[0] - score[0]) <= 1e-12
                and abs(item[1] - score[1]) <= 1e-12
            ]
            if len(ties) > 1:
                ambiguous_old_ids.add(str(old["event_id"]))
                result.append(
                    _delta_row(
                        "ambiguous",
                        old,
                        None,
                        candidate_new_event_ids=";".join(
                            item[2] for item in ties
                        ),
                    )
                )
        assignable_old = [
            row
            for row in old_group
            if str(row["event_id"]) not in ambiguous_old_ids
        ]
        matched_old: set[str] = set()
        matched_new: set[str] = set()
        if assignable_old:
            old_count = len(assignable_old)
            new_count = len(new_group)
            costs = np.full(
                (old_count, new_count + old_count),
                1_000_000.0,
                dtype=np.float64,
            )
            for old_index, old in enumerate(assignable_old):
                for new_index, new in enumerate(new_group):
                    if not _eligible(old, new):
                        costs[old_index, new_index] = 1_000_000_000.0
                        continue
                    costs[old_index, new_index] = (
                        (1.0 - _iou(old, new)) * 1_000.0
                        + _center_distance(old, new)
                        + new_index * 1e-12
                    )
            row_indices, column_indices = linear_sum_assignment(costs)
            for old_index, column_index in zip(
                row_indices.tolist(), column_indices.tolist(), strict=True
            ):
                old = assignable_old[old_index]
                if column_index >= new_count:
                    continue
                new = new_group[column_index]
                if not _eligible(old, new):
                    continue
                matched_old.add(str(old["event_id"]))
                matched_new.add(str(new["event_id"]))
                iou = _iou(old, new)
                if str(old["class_name"]) != str(new["class_name"]):
                    category = "reclassified"
                elif iou >= 0.5:
                    category = "retained"
                else:
                    category = "moved"
                result.append(_delta_row(category, old, new))
        for old in assignable_old:
            if str(old["event_id"]) not in matched_old:
                result.append(_delta_row("removed", old, None))
        for new in new_group:
            if str(new["event_id"]) not in matched_new:
                result.append(_delta_row("new", None, new))
    return result


def _delta_row(
    category: str,
    old: dict[str, Any] | None,
    new: dict[str, Any] | None,
    *,
    candidate_new_event_ids: str = "",
) -> dict[str, Any]:
    if category not in CATEGORIES:
        raise ValueError(f"unknown comparison category: {category}")
    anchor = old or new
    assert anchor is not None
    return {
        "category": category,
        "split": anchor["split"],
        "source_filename": anchor["source_filename"],
        "old_event_id": old["event_id"] if old else "",
        "new_event_id": new["event_id"] if new else "",
        "old_class_name": old["class_name"] if old else "",
        "new_class_name": new["class_name"] if new else "",
        "old_origin": old["annotation_origin"] if old else "",
        "new_origin": new["annotation_origin"] if new else "",
        "old_start_norm": old["start_norm"] if old else "",
        "old_end_norm": old["end_norm"] if old else "",
        "new_start_norm": new["start_norm"] if new else "",
        "new_end_norm": new["end_norm"] if new else "",
        "iou": _iou(old, new) if old and new else "",
        "center_shift_norm": (
            _center_distance(old, new) if old and new else ""
        ),
        "candidate_new_event_ids": candidate_new_event_ids,
        "mapping_role": "diagnostic_only_no_identity_or_label_transfer",
    }


def write_comparison(
    *,
    old_events_path: Path,
    new_events_path: Path,
    output_dir: Path,
    old_dataset_id: str,
    new_dataset_id: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite comparison: {output_dir}")
    rows = compare_events(
        read_events(old_events_path), read_events(new_events_path)
    )
    counts = Counter(row["category"] for row in rows)
    summary = {
        "schema_version": 1,
        "old_dataset_id": old_dataset_id,
        "new_dataset_id": new_dataset_id,
        "mapping_role": "diagnostic_only_no_identity_or_label_transfer",
        "row_count": len(rows),
        "category_counts": dict(sorted(counts.items())),
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        with (temporary / "event_delta.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (temporary / "event_delta_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return summary
