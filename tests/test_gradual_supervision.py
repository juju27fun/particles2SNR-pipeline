from __future__ import annotations

from collections import Counter

from particles2snr.gradual_supervision import (
    WEIGHTS,
    assign_control_folds,
    build_audit_folds,
    is_verified_empty_audit_row,
)


def test_folds_have_four_sources_per_class_and_distribute_common_misses() -> None:
    traces = []
    events = {}
    misses = {}
    for class_name in ("2um", "4um", "10um"):
        for index in range(20):
            source_id = f"{class_name}-{index:02d}"
            traces.append({"source_id": source_id, "source_class": class_name})
            events[source_id] = index % 4
    special = ["2um-00", "2um-01", "4um-00", "10um-00", "10um-01"]
    for source_id in special:
        misses[source_id] = 1
    misses["10um-00"] = 2
    assignment = build_audit_folds(traces, events, misses)
    assert len(assignment) == 60
    assert len({assignment[source] for source in special}) == 5
    for fold in range(5):
        rows = [row for row in traces if assignment[row["source_id"]] == fold]
        assert len(rows) == 12
        assert Counter(row["source_class"] for row in rows) == {"2um": 4, "4um": 4, "10um": 4}


def test_control_folds_are_deterministic_and_balanced() -> None:
    first = assign_control_folds([f"control-{i}" for i in range(23)])
    second = assign_control_folds(reversed([f"control-{i}" for i in range(23)]))
    assert first == second
    loads = Counter(first.values())
    assert max(loads.values()) - min(loads.values()) <= 1


def test_frozen_confidence_weights() -> None:
    assert WEIGHTS["human_confirmed"]["box"] == 1.0
    assert WEIGHTS["detector_seed_unreviewed"]["box"] == 0.25
    assert WEIGHTS["human_uncertain"]["box"] == 0.0
    assert WEIGHTS["mad_weak"] == {"presence": 0.25, "class": 0.25, "center": 0.25, "box": 0.10}


def test_verified_empty_requires_explicit_reviewed_status() -> None:
    base = {
        "trace_status": "reviewed",
        "cardinality_evaluable": "True",
        "confirmed_event_count": "0",
        "uncertain_event_count": "0",
    }
    assert is_verified_empty_audit_row(base)
    assert not is_verified_empty_audit_row({**base, "trace_status": "uncertain"})
    assert not is_verified_empty_audit_row({**base, "cardinality_evaluable": "False"})
    assert not is_verified_empty_audit_row({**base, "confirmed_event_count": "1"})
    assert not is_verified_empty_audit_row({**base, "uncertain_event_count": "1"})
