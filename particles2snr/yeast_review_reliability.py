from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import cohen_kappa_score

from .yeast_review_analysis import parse_bool, parse_count, wilson_interval


CANDIDATE_FIELDS = (
    "review_event_present",
    "review_center_acceptable",
    "review_full_event_visible",
    "review_artifact",
)
FILE_FIELDS = (
    "review_true_event_count",
    "review_false_retained_candidate_count",
    "review_true_rejected_candidate_count",
    "review_missed_event_count",
)
QUEUE_CONFIG = {
    "candidate": ("manual_review_queue.csv", "event_id", CANDIDATE_FIELDS),
    "file": ("manual_file_review_queue.csv", "record_id", FILE_FIELDS),
}


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if not fields or not rows:
        raise ValueError(f"Review CSV is empty or malformed: {path}")
    return fields, rows


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _select_stratified(
    rows: list[dict[str, str]],
    *,
    id_field: str,
    fraction: float,
    seed: int,
) -> list[dict[str, str]]:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    target = int(math.ceil(fraction * len(rows)))
    strata: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        strata[row["review_stratum"]].append(row)
    selected_ids: set[str] = set()
    selected: list[dict[str, str]] = []
    for stratum, stratum_rows in sorted(strata.items()):
        first = min(
            stratum_rows,
            key=lambda row: _stable_key(seed, f"{stratum}:{row[id_field]}"),
        )
        selected.append(first)
        selected_ids.add(first[id_field])
    if len(selected) < target:
        remaining = sorted(
            (row for row in rows if row[id_field] not in selected_ids),
            key=lambda row: _stable_key(seed + 1, row[id_field]),
        )
        selected.extend(remaining[: target - len(selected)])
    return sorted(selected, key=lambda row: (row["review_stratum"], row[id_field]))


def build_reliability_review(
    template_dir: Path,
    output_dir: Path,
    *,
    fraction: float = 0.20,
    seed: int = 20260714,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    queue_summaries: dict[str, Any] = {}
    for queue, (filename, id_field, review_fields) in QUEUE_CONFIG.items():
        fields, rows = _read_csv(template_dir / filename)
        selected = _select_stratified(
            rows,
            id_field=id_field,
            fraction=fraction,
            seed=seed,
        )
        cleared = []
        for row in selected:
            copy = dict(row)
            for field in (*review_fields, "reviewer", "review_notes"):
                copy[field] = ""
            cleared.append(copy)
        _write_csv(output_dir / filename, fields, cleared)
        queue_summaries[queue] = {
            "source_rows": len(rows),
            "selected_rows": len(cleared),
            "selected_fraction": len(cleared) / len(rows),
            "review_strata": sorted({row["review_stratum"] for row in cleared}),
        }
    summary = {
        "schema_version": 1,
        "template_dir": str(template_dir),
        "fraction_minimum": fraction,
        "seed": seed,
        "queues": queue_summaries,
        "status": "awaiting_independent_review",
    }
    (output_dir / "reliability_review_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _candidate_complete(row: dict[str, str]) -> bool:
    required = ["review_event_present", "review_artifact"]
    if row["quality"] in {"strict", "medium"}:
        required.extend(["review_center_acceptable", "review_full_event_visible"])
    return bool(row.get("reviewer", "").strip()) and all(row.get(field, "").strip() for field in required)


def _file_complete(row: dict[str, str]) -> bool:
    return bool(row.get("reviewer", "").strip()) and all(
        row.get(field, "").strip() for field in FILE_FIELDS
    )


def _binary_agreement(pairs: list[tuple[bool, bool]]) -> dict[str, Any]:
    if not pairs:
        return {"n": 0, "n_agree": 0, "agreement": None, "wilson_95": None, "cohen_kappa": None}
    first = np.asarray([int(left) for left, _right in pairs], dtype=np.int64)
    second = np.asarray([int(right) for _left, right in pairs], dtype=np.int64)
    n_agree = int(np.count_nonzero(first == second))
    with np.errstate(invalid="ignore"):
        kappa = float(cohen_kappa_score(first, second, labels=[0, 1]))
    return {
        "n": len(pairs),
        "n_agree": n_agree,
        "agreement": n_agree / len(pairs),
        "wilson_95": wilson_interval(n_agree, len(pairs)),
        "cohen_kappa": kappa if math.isfinite(kappa) else None,
    }


def _count_agreement(pairs: list[tuple[int, int]]) -> dict[str, Any]:
    if not pairs:
        return {"n": 0, "exact_agreement": None, "wilson_95": None, "mean_absolute_difference": None}
    differences = np.asarray([abs(left - right) for left, right in pairs], dtype=np.float64)
    n_exact = int(np.count_nonzero(differences == 0))
    return {
        "n": len(pairs),
        "n_exact": n_exact,
        "exact_agreement": n_exact / len(pairs),
        "wilson_95": wilson_interval(n_exact, len(pairs)),
        "mean_absolute_difference": float(np.mean(differences)),
        "maximum_absolute_difference": int(np.max(differences)),
    }


def compare_review_reliability(
    primary_dir: Path,
    repeat_dir: Path,
) -> dict[str, Any]:
    candidate_fields, primary_candidates = _read_csv(primary_dir / "manual_review_queue.csv")
    _repeat_fields, repeat_candidates = _read_csv(repeat_dir / "manual_review_queue.csv")
    _file_fields, primary_files = _read_csv(primary_dir / "manual_file_review_queue.csv")
    _repeat_file_fields, repeat_files = _read_csv(repeat_dir / "manual_file_review_queue.csv")
    if "event_id" not in candidate_fields:
        raise ValueError("Primary candidate review has no event_id")

    candidate_primary = {row["event_id"]: row for row in primary_candidates}
    candidate_repeat = {row["event_id"]: row for row in repeat_candidates}
    file_primary = {row["record_id"]: row for row in primary_files}
    file_repeat = {row["record_id"]: row for row in repeat_files}
    candidate_ids = sorted(set(candidate_primary) & set(candidate_repeat))
    file_ids = sorted(set(file_primary) & set(file_repeat))
    complete_candidate_ids = [
        row_id
        for row_id in candidate_ids
        if _candidate_complete(candidate_primary[row_id]) and _candidate_complete(candidate_repeat[row_id])
    ]
    complete_file_ids = [
        row_id
        for row_id in file_ids
        if _file_complete(file_primary[row_id]) and _file_complete(file_repeat[row_id])
    ]

    binary_metrics: dict[str, Any] = {}
    for field in CANDIDATE_FIELDS:
        pairs = []
        for row_id in complete_candidate_ids:
            left = parse_bool(candidate_primary[row_id].get(field, ""), field=field, row_id=row_id)
            right = parse_bool(candidate_repeat[row_id].get(field, ""), field=field, row_id=row_id)
            if left is not None and right is not None:
                pairs.append((left, right))
        binary_metrics[field] = _binary_agreement(pairs)

    count_metrics: dict[str, Any] = {}
    for field in FILE_FIELDS:
        pairs = []
        for row_id in complete_file_ids:
            left = parse_count(file_primary[row_id].get(field, ""), field=field, row_id=row_id)
            right = parse_count(file_repeat[row_id].get(field, ""), field=field, row_id=row_id)
            if left is not None and right is not None:
                pairs.append((left, right))
        count_metrics[field] = _count_agreement(pairs)

    primary_reviewers = sorted(
        {row["reviewer"].strip() for row in primary_candidates + primary_files if row.get("reviewer", "").strip()}
    )
    repeat_reviewers = sorted(
        {row["reviewer"].strip() for row in repeat_candidates + repeat_files if row.get("reviewer", "").strip()}
    )
    independent = bool(primary_reviewers and repeat_reviewers) and not (
        set(primary_reviewers) & set(repeat_reviewers)
    )
    all_complete = (
        len(complete_candidate_ids) == len(candidate_ids)
        and len(complete_file_ids) == len(file_ids)
    )
    disagreements = sum(
        metric["n"] - metric["n_agree"] for metric in binary_metrics.values()
    ) + sum(
        metric["n"] - metric["n_exact"] for metric in count_metrics.values()
    )
    status = "pending" if not all_complete else (
        "needs_adjudication" if disagreements else "agreement_complete"
    )
    return {
        "schema_version": 1,
        "primary_dir": str(primary_dir),
        "repeat_dir": str(repeat_dir),
        "candidate_overlap": len(candidate_ids),
        "candidate_complete_pairs": len(complete_candidate_ids),
        "file_overlap": len(file_ids),
        "file_complete_pairs": len(complete_file_ids),
        "binary_agreement": binary_metrics,
        "count_agreement": count_metrics,
        "primary_reviewers": primary_reviewers,
        "repeat_reviewers": repeat_reviewers,
        "review_mode": "independent_inter_rater" if independent else "intra_rater_or_unverified",
        "n_field_level_disagreements": disagreements,
        "status": status,
        "interpretation": "Preserve both raw reviews and adjudicate disagreements separately.",
    }
