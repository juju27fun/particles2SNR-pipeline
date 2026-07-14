from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ReviewGateThresholds:
    retained_precision_min: float = 0.90
    retained_precision_lower_95_min: float = 0.80
    full_trace_recall_min: float = 0.85
    full_trace_recall_lower_95_min: float = 0.75
    per_group_precision_min: float = 0.75
    per_group_recall_min: float = 0.70
    rejected_event_presence_max: float = 0.25
    minimum_independent_acquisitions: int = 2


TRUE_VALUES = {"1", "true", "yes", "y"}
FALSE_VALUES = {"0", "false", "no", "n"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_bool(value: str, *, field: str, row_id: str) -> bool | None:
    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ValueError(f"{row_id}: {field} must be one of yes/no, true/false, or 1/0")


def parse_count(value: str, *, field: str, row_id: str) -> int | None:
    if not value.strip():
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{row_id}: {field} must be a non-negative integer") from exc
    if parsed < 0:
        raise ValueError(f"{row_id}: {field} must be a non-negative integer")
    return parsed


def wilson_interval(successes: float, total: float, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    lower = 0.0 if successes <= 0 else max(0.0, center - radius)
    upper = 1.0 if successes >= total else min(1.0, center + radius)
    return [lower, upper]


def _binary_metric(successes: float, total: float) -> dict[str, Any]:
    return {
        "successes": successes,
        "total": total,
        "value": successes / total if total else None,
        "wilson_95": wilson_interval(successes, total),
    }


def _weighted_binary_metric(rows: Iterable[tuple[float, float]], weights: Iterable[float]) -> dict[str, Any]:
    weighted_successes = 0.0
    weighted_total = 0.0
    for (successes, total), weight in zip(rows, weights):
        weighted_successes += successes * weight
        weighted_total += total * weight
    return {
        "estimated_successes": weighted_successes,
        "estimated_total": weighted_total,
        "value": weighted_successes / weighted_total if weighted_total else None,
        "uncertainty": "not estimated; stratified expansion-weight point estimate only",
    }


def _stratum_weights(
    population_rows: list[dict[str, str]],
    review_rows: list[dict[str, str]],
    key,
) -> dict[tuple[str, ...], float]:
    population = Counter(key(row) for row in population_rows)
    reviewed = Counter(key(row) for row in review_rows)
    return {stratum: population[stratum] / count for stratum, count in reviewed.items() if count}


def _candidate_analysis(
    candidate_rows: list[dict[str, str]], review_rows: list[dict[str, str]]
) -> dict[str, Any]:
    retained_population = [row for row in candidate_rows if row["quality"] in {"strict", "medium"}]
    proposed_review = [row for row in review_rows if int(row["candidate_index"]) >= 0]
    retained_review = [
        row
        for row in proposed_review
        if row["quality"] in {"strict", "medium"}
    ]
    required_fields = (
        "review_event_present",
        "review_center_acceptable",
        "review_full_event_visible",
        "review_artifact",
        "reviewer",
    )
    parsed_all: list[dict[str, Any]] = []
    pending = 0
    for row in review_rows:
        row_id = row["event_id"]
        values = {
            field: parse_bool(row[field], field=field, row_id=row_id)
            for field in required_fields[:-1]
        }
        reviewer = row["reviewer"].strip()
        if any(value is None for value in values.values()) or not reviewer:
            pending += 1
            continue
        parsed_all.append({**row, **values})
    parsed = [row for row in parsed_all if row["quality"] in {"strict", "medium"}]
    rejected = [row for row in parsed_all if row["quality"] == "reject"]
    background = [row for row in parsed_all if int(row["candidate_index"]) < 0]

    by_group: dict[str, dict[str, Any]] = {}
    for group in sorted({row["source_group"] for row in retained_review}):
        selected = [row for row in parsed if row["source_group"] == group]
        by_group[group] = _binary_metric(
            sum(bool(row["review_event_present"]) for row in selected), len(selected)
        )
    by_acquisition: dict[str, dict[str, Any]] = {}
    for acquisition in sorted({row["acquisition_id"] for row in retained_review}):
        selected = [row for row in parsed if row["acquisition_id"] == acquisition]
        by_acquisition[acquisition] = _binary_metric(
            sum(bool(row["review_event_present"]) for row in selected), len(selected)
        )

    weights = _stratum_weights(
        retained_population,
        retained_review,
        lambda row: (row["acquisition_id"], row["source_group"], row["quality"]),
    )
    weighted_rows = [
        (float(bool(row["review_event_present"])), 1.0)
        for row in parsed
    ]
    weighted_values = [
        weights[(row["acquisition_id"], row["source_group"], row["quality"])]
        for row in parsed
    ]
    event_present = _binary_metric(
        sum(bool(row["review_event_present"]) for row in parsed), len(parsed)
    )
    return {
        "n_expected": len(review_rows),
        "n_complete": len(parsed_all),
        "n_pending": pending,
        "retained_candidate_precision_balanced": event_present,
        "retained_candidate_precision_population_weighted": _weighted_binary_metric(
            weighted_rows, weighted_values
        ),
        "center_acceptable": _binary_metric(
            sum(bool(row["review_center_acceptable"]) for row in parsed), len(parsed)
        ),
        "full_event_visible": _binary_metric(
            sum(bool(row["review_full_event_visible"]) for row in parsed), len(parsed)
        ),
        "artifact_free": _binary_metric(
            sum(not bool(row["review_artifact"]) for row in parsed), len(parsed)
        ),
        "rejected_candidate_event_presence": _binary_metric(
            sum(bool(row["review_event_present"]) for row in rejected), len(rejected)
        ),
        "background_window_event_presence": _binary_metric(
            sum(bool(row["review_event_present"]) for row in background), len(background)
        ),
        "precision_by_source_group": by_group,
        "precision_by_acquisition": by_acquisition,
        "sampling": "balanced acquisition x source-group x retained-quality review; population point estimate uses expansion weights",
    }


def _file_analysis(
    file_population: list[dict[str, str]], review_rows: list[dict[str, str]]
) -> dict[str, Any]:
    parsed: list[dict[str, Any]] = []
    pending = 0
    for row in review_rows:
        row_id = row["record_id"]
        required_fields = (
            "review_true_event_count",
            "review_false_retained_candidate_count",
            "review_true_rejected_candidate_count",
            "review_missed_event_count",
        )
        missing_fields = [field for field in required_fields if field not in row]
        if missing_fields:
            raise ValueError(
                "Full-trace review schema is obsolete; rebuild the candidate audit with fields: "
                + ", ".join(missing_fields)
            )
        counts = {
            field: parse_count(row[field], field=field, row_id=row_id)
            for field in required_fields
        }
        if any(value is None for value in counts.values()) or not row["reviewer"].strip():
            pending += 1
            continue
        n_retained = int(row["n_retained_candidates"])
        n_rejected = int(row["n_rejected_candidates"])
        false_positive = int(counts["review_false_retained_candidate_count"])
        true_rejected = int(counts["review_true_rejected_candidate_count"])
        missed = int(counts["review_missed_event_count"])
        true_events = int(counts["review_true_event_count"])
        if false_positive > n_retained:
            raise ValueError(f"{row_id}: false retained count exceeds retained candidate count")
        if true_rejected > n_rejected:
            raise ValueError(f"{row_id}: true rejected count exceeds rejected candidate count")
        true_positive = n_retained - false_positive
        false_negative = true_rejected + missed
        if true_events != true_positive + false_negative:
            raise ValueError(
                f"{row_id}: true event count must equal retained true, rejected true, plus missed events"
            )
        parsed.append(
            {
                **row,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "true_rejected": true_rejected,
                "missed": missed,
            }
        )

    totals = {
        key: sum(int(row[key]) for row in parsed)
        for key in ("true_positive", "false_positive", "false_negative")
    }
    by_group: dict[str, dict[str, Any]] = {}
    for group in sorted({row["source_group"] for row in review_rows}):
        selected = [row for row in parsed if row["source_group"] == group]
        tp = sum(int(row["true_positive"]) for row in selected)
        fn = sum(int(row["false_negative"]) for row in selected)
        by_group[group] = _binary_metric(tp, tp + fn)
    by_acquisition: dict[str, dict[str, Any]] = {}
    for acquisition in sorted({row["acquisition_id"] for row in review_rows}):
        selected = [row for row in parsed if row["acquisition_id"] == acquisition]
        tp = sum(int(row["true_positive"]) for row in selected)
        fn = sum(int(row["false_negative"]) for row in selected)
        by_acquisition[acquisition] = _binary_metric(tp, tp + fn)

    def stratum(row: dict[str, str]) -> tuple[str, str, str]:
        return (
            row["acquisition_id"],
            row["source_group"],
            str(min(int(row["n_candidates"]), 3)),
        )

    weights = _stratum_weights(file_population, review_rows, stratum)
    recall_rows = [
        (float(row["true_positive"]), float(row["true_positive"] + row["false_negative"]))
        for row in parsed
    ]
    weighted_values = [weights[stratum(row)] for row in parsed]
    return {
        "n_expected": len(review_rows),
        "n_complete": len(parsed),
        "n_pending": pending,
        "event_recall_balanced": _binary_metric(
            totals["true_positive"], totals["true_positive"] + totals["false_negative"]
        ),
        "event_recall_population_weighted": _weighted_binary_metric(recall_rows, weighted_values),
        "event_precision_cross_check": _binary_metric(
            totals["true_positive"], totals["true_positive"] + totals["false_positive"]
        ),
        "recall_by_source_group": by_group,
        "recall_by_acquisition": by_acquisition,
        "event_counts": totals,
        "sampling": "balanced acquisition x source-group x detected-count stratum; population point estimate uses expansion weights",
    }


def analyze_review(
    candidate_dataset: Path,
    thresholds: ReviewGateThresholds = ReviewGateThresholds(),
    review_dir: Path | None = None,
) -> dict[str, Any]:
    review_root = review_dir or candidate_dataset
    candidate_rows = read_csv(candidate_dataset / "candidate_events.csv")
    file_rows = read_csv(candidate_dataset / "file_detection_report.csv")
    candidate_review = read_csv(review_root / "manual_review_queue.csv")
    file_review = read_csv(review_root / "manual_file_review_queue.csv")
    candidate = _candidate_analysis(candidate_rows, candidate_review)
    traces = _file_analysis(file_rows, file_review)
    acquisitions = sorted({row["acquisition_id"] for row in file_rows})
    acquisition_roles: dict[str, list[str]] = {
        acquisition: sorted(
            {
                row.get("acquisition_role", "").strip()
                for row in file_rows
                if row["acquisition_id"] == acquisition and row.get("acquisition_role", "").strip()
            }
        )
        for acquisition in acquisitions
    }
    modern_roles_present = any(acquisition_roles.values())
    roles_are_valid = (
        (
            all(len(roles) == 1 for roles in acquisition_roles.values())
            and {roles[0] for roles in acquisition_roles.values()}
            >= {"development", "sealed_ood_test"}
        )
        if modern_roles_present
        else True
    )
    complete = candidate["n_pending"] == 0 and traces["n_pending"] == 0

    checks: dict[str, bool | None] = {key: None for key in (
        "retained_precision_point",
        "retained_precision_lower_95",
        "full_trace_recall_point",
        "full_trace_recall_lower_95",
        "per_group_precision",
        "per_group_recall",
        "per_acquisition_precision",
        "per_acquisition_recall",
        "rejected_event_presence",
    )}
    if complete:
        precision = candidate["retained_candidate_precision_balanced"]
        recall = traces["event_recall_balanced"]
        checks = {
            "retained_precision_point": precision["value"] >= thresholds.retained_precision_min,
            "retained_precision_lower_95": precision["wilson_95"][0]
            >= thresholds.retained_precision_lower_95_min,
            "full_trace_recall_point": recall["value"] >= thresholds.full_trace_recall_min,
            "full_trace_recall_lower_95": recall["wilson_95"][0]
            >= thresholds.full_trace_recall_lower_95_min,
            "per_group_precision": all(
                metric["value"] is not None and metric["value"] >= thresholds.per_group_precision_min
                for metric in candidate["precision_by_source_group"].values()
            ),
            "per_group_recall": all(
                metric["value"] is not None and metric["value"] >= thresholds.per_group_recall_min
                for metric in traces["recall_by_source_group"].values()
            ),
            "per_acquisition_precision": all(
                metric["value"] is not None and metric["value"] >= thresholds.per_group_precision_min
                for metric in candidate["precision_by_acquisition"].values()
            ),
            "per_acquisition_recall": all(
                metric["value"] is not None and metric["value"] >= thresholds.per_group_recall_min
                for metric in traces["recall_by_acquisition"].values()
            ),
            "rejected_event_presence": (
                candidate["rejected_candidate_event_presence"]["value"] is not None
                and candidate["rejected_candidate_event_presence"]["value"]
                <= thresholds.rejected_event_presence_max
            ),
        }

    event_review_status = "pending" if not complete else ("pass" if all(checks.values()) else "fail")
    acquisition_ood_ready = (
        len(acquisitions) >= thresholds.minimum_independent_acquisitions and roles_are_valid
    )
    if event_review_status != "pass":
        gate_1_status = event_review_status
    elif not acquisition_ood_ready:
        gate_1_status = "blocked_independent_acquisition"
    else:
        gate_1_status = "pass"
    return {
        "schema_version": 1,
        "candidate_dataset": str(candidate_dataset),
        "review_dir": str(review_root),
        "thresholds": asdict(thresholds),
        "candidate_review": candidate,
        "full_trace_review": traces,
        "event_review_checks": checks,
        "event_review_status": event_review_status,
        "acquisition_ids": acquisitions,
        "acquisition_roles": acquisition_roles,
        "acquisition_ood_ready": acquisition_ood_ready,
        "gate_1_status": gate_1_status,
        "interpretation": (
            "Gate 1 requires both an acceptable reviewed detector and at least two independent "
            "acquisitions; source-condition folders are not biological event labels."
        ),
    }
