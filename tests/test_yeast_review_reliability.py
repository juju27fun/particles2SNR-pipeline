from __future__ import annotations

import csv
from pathlib import Path

from particles2snr.yeast_review_reliability import (
    build_reliability_review,
    compare_review_reliability,
)


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _template(tmp_path: Path) -> Path:
    root = tmp_path / "template"
    root.mkdir()
    candidate_rows = []
    file_rows = []
    for index in range(10):
        stratum = "budding:strict" if index < 5 else "mix:reject"
        quality = "strict" if index < 5 else "reject"
        candidate_rows.append(
            {
                "event_id": f"event-{index}",
                "review_stratum": stratum,
                "quality": quality,
                "review_event_present": "yes",
                "review_center_acceptable": "yes" if quality == "strict" else "",
                "review_full_event_visible": "yes" if quality == "strict" else "",
                "review_artifact": "no",
                "reviewer": "template-reviewer",
                "review_notes": "template",
            }
        )
        file_rows.append(
            {
                "record_id": f"record-{index}",
                "review_stratum": "budding:n_1" if index < 5 else "mix:n_2",
                "review_true_event_count": "1",
                "review_false_retained_candidate_count": "0",
                "review_true_rejected_candidate_count": "0",
                "review_missed_event_count": "0",
                "reviewer": "template-reviewer",
                "review_notes": "template",
            }
        )
    _write(root / "manual_review_queue.csv", candidate_rows)
    _write(root / "manual_file_review_queue.csv", file_rows)
    return root


def test_reliability_subset_is_stratified_deterministic_and_blank(tmp_path: Path) -> None:
    template = _template(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    summary = build_reliability_review(template, first, fraction=0.20, seed=7)
    build_reliability_review(template, second, fraction=0.20, seed=7)

    with (first / "manual_review_queue.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert summary["queues"]["candidate"]["selected_rows"] == 2
    assert {row["review_stratum"] for row in rows} == {"budding:strict", "mix:reject"}
    assert all(row["reviewer"] == "" and row["review_event_present"] == "" for row in rows)
    assert (first / "manual_review_queue.csv").read_bytes() == (
        second / "manual_review_queue.csv"
    ).read_bytes()


def test_reliability_comparison_reports_disagreement_and_reviewer_mode(tmp_path: Path) -> None:
    template = _template(tmp_path)
    repeat = tmp_path / "repeat"
    build_reliability_review(template, repeat, fraction=0.20, seed=7)

    for directory, reviewer in ((template, "reviewer-a"), (repeat, "reviewer-b")):
        for filename, decision_fields in (
            (
                "manual_review_queue.csv",
                {
                    "review_event_present": "yes",
                    "review_center_acceptable": "yes",
                    "review_full_event_visible": "yes",
                    "review_artifact": "no",
                },
            ),
            (
                "manual_file_review_queue.csv",
                {
                    "review_true_event_count": "1",
                    "review_false_retained_candidate_count": "0",
                    "review_true_rejected_candidate_count": "0",
                    "review_missed_event_count": "0",
                },
            ),
        ):
            path = directory / filename
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                row.update(decision_fields)
                if row.get("quality") == "reject":
                    row["review_center_acceptable"] = ""
                    row["review_full_event_visible"] = ""
                row["reviewer"] = reviewer
            _write(path, rows)

    repeat_path = repeat / "manual_review_queue.csv"
    with repeat_path.open(newline="", encoding="utf-8") as handle:
        repeat_rows = list(csv.DictReader(handle))
    repeat_rows[0]["review_event_present"] = "no"
    _write(repeat_path, repeat_rows)

    result = compare_review_reliability(template, repeat)
    assert result["candidate_complete_pairs"] == 2
    assert result["binary_agreement"]["review_event_present"]["agreement"] == 0.5
    assert result["review_mode"] == "independent_inter_rater"
    assert result["status"] == "needs_adjudication"
