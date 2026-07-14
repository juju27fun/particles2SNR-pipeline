from __future__ import annotations

import csv
from pathlib import Path

import pytest

from particles2snr.yeast_review_analysis import ReviewGateThresholds, analyze_review, wilson_interval


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _dataset(tmp_path: Path, *, pending: bool = False) -> Path:
    candidate = {
        "event_id": "event-1",
        "record_id": "record-1",
        "source_group": "budding",
        "acquisition_id": "session-1",
        "candidate_index": 0,
        "quality": "strict",
    }
    reviewed = {
        **candidate,
        "review_event_present": "" if pending else "yes",
        "review_center_acceptable": "" if pending else "yes",
        "review_full_event_visible": "" if pending else "yes",
        "review_artifact": "" if pending else "no",
        "reviewer": "" if pending else "reviewer-a",
    }
    rejected = {
        **candidate,
        "event_id": "event-2",
        "candidate_index": 1,
        "quality": "reject",
    }
    rejected_review = {
        **rejected,
        "review_event_present": "" if pending else "no",
        "review_center_acceptable": "" if pending else "no",
        "review_full_event_visible": "" if pending else "yes",
        "review_artifact": "" if pending else "no",
        "reviewer": "" if pending else "reviewer-a",
    }
    file_row = {
        "record_id": "record-1",
        "source_group": "budding",
        "acquisition_id": "session-1",
        "n_candidates": 2,
        "n_retained_candidates": 1,
        "n_rejected_candidates": 1,
    }
    file_review = {
        **file_row,
        "review_true_event_count": "" if pending else 1,
        "review_false_retained_candidate_count": "" if pending else 0,
        "review_true_rejected_candidate_count": "" if pending else 0,
        "review_missed_event_count": "" if pending else 0,
        "reviewer": "" if pending else "reviewer-a",
    }
    _write(tmp_path / "candidate_events.csv", [candidate, rejected])
    _write(tmp_path / "manual_review_queue.csv", [reviewed, rejected_review])
    _write(tmp_path / "file_detection_report.csv", [file_row])
    _write(tmp_path / "manual_file_review_queue.csv", [file_review])
    return tmp_path


def test_pending_annotations_do_not_pass(tmp_path: Path) -> None:
    result = analyze_review(_dataset(tmp_path, pending=True))
    assert result["event_review_status"] == "pending"
    assert result["gate_1_status"] == "pending"


def test_complete_review_separates_event_gate_from_acquisition_gate(tmp_path: Path) -> None:
    permissive = ReviewGateThresholds(
        retained_precision_min=0.0,
        retained_precision_lower_95_min=0.0,
        full_trace_recall_min=0.0,
        full_trace_recall_lower_95_min=0.0,
        per_group_precision_min=0.0,
        per_group_recall_min=0.0,
    )
    result = analyze_review(_dataset(tmp_path), permissive)
    assert result["event_review_status"] == "pass"
    assert result["gate_1_status"] == "blocked_independent_acquisition"
    assert result["candidate_review"]["retained_candidate_precision_balanced"]["value"] == 1.0


def test_inconsistent_full_trace_counts_are_rejected(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    rows = list(csv.DictReader((dataset / "manual_file_review_queue.csv").open()))
    rows[0]["review_missed_event_count"] = "1"
    _write(dataset / "manual_file_review_queue.csv", rows)
    with pytest.raises(ValueError, match="true event count"):
        analyze_review(dataset)


def test_complete_review_requires_and_reports_each_acquisition(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    for filename, id_field in (
        ("candidate_events.csv", "event_id"),
        ("manual_review_queue.csv", "event_id"),
        ("file_detection_report.csv", "record_id"),
        ("manual_file_review_queue.csv", "record_id"),
    ):
        path = dataset / filename
        rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
        copies = []
        for row in rows:
            copied = dict(row)
            copied["acquisition_id"] = "session-2"
            copied[id_field] = f"{copied[id_field]}-session-2"
            copied["record_id"] = f"{copied['record_id']}-session-2"
            copies.append(copied)
        _write(path, rows + copies)

    permissive = ReviewGateThresholds(
        retained_precision_min=0.0,
        retained_precision_lower_95_min=0.0,
        full_trace_recall_min=0.0,
        full_trace_recall_lower_95_min=0.0,
        per_group_precision_min=0.0,
        per_group_recall_min=0.0,
    )
    result = analyze_review(dataset, permissive)
    assert result["gate_1_status"] == "pass"
    assert set(result["candidate_review"]["precision_by_acquisition"]) == {
        "session-1",
        "session-2",
    }
    assert set(result["full_trace_review"]["recall_by_acquisition"]) == {
        "session-1",
        "session-2",
    }


def test_modern_review_rejects_two_development_acquisitions_as_ood(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    for filename in ("candidate_events.csv", "manual_review_queue.csv"):
        path = dataset / filename
        rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
        for row in rows:
            row["acquisition_role"] = "development"
        _write(path, rows)
    for filename in ("file_detection_report.csv", "manual_file_review_queue.csv"):
        path = dataset / filename
        rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
        for row in rows:
            row["acquisition_role"] = "development"
        copies = []
        for row in rows:
            copied = dict(row)
            copied["record_id"] = f"{row['record_id']}-session-2"
            copied["acquisition_id"] = "session-2"
            copies.append(copied)
        _write(path, rows + copies)

    permissive = ReviewGateThresholds(
        retained_precision_min=0.0,
        retained_precision_lower_95_min=0.0,
        full_trace_recall_min=0.0,
        full_trace_recall_lower_95_min=0.0,
        per_group_precision_min=0.0,
        per_group_recall_min=0.0,
    )
    result = analyze_review(dataset, permissive)
    assert result["acquisition_ood_ready"] is False
    assert result["acquisition_roles"] == {
        "session-1": ["development"],
        "session-2": ["development"],
    }


def test_wilson_interval_is_bounded() -> None:
    assert wilson_interval(0, 10)[0] == 0.0
    assert wilson_interval(10, 10)[1] == 1.0
