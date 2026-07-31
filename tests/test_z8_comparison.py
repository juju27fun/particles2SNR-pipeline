from __future__ import annotations

from particles2snr.z8_comparison import compare_events


def _event(
    event_id: str,
    *,
    start: float,
    end: float,
    class_name: str = "10um",
) -> dict:
    return {
        "event_id": event_id,
        "split": "val",
        "source_filename": "sample.npy",
        "class_name": class_name,
        "physical_source_class": "10um",
        "annotation_origin": "dual_clean_strict",
        "start_norm": start,
        "end_norm": end,
        "center_norm": (start + end) / 2.0,
    }


def test_comparison_covers_retained_reclassified_removed_and_new() -> None:
    old = [
        _event("old-retained", start=0.10, end=0.20),
        _event("old-reclassified", start=0.30, end=0.40),
        _event("old-removed", start=0.70, end=0.75),
    ]
    new = [
        _event("new-retained", start=0.11, end=0.21),
        _event(
            "new-reclassified",
            start=0.30,
            end=0.40,
            class_name="unclear",
        ),
        _event("new-only", start=0.90, end=0.95),
    ]
    rows = compare_events(old, new)
    assert {row["category"] for row in rows} == {
        "retained",
        "reclassified",
        "removed",
        "new",
    }
    assert all(
        row["mapping_role"]
        == "diagnostic_only_no_identity_or_label_transfer"
        for row in rows
    )


def test_comparison_marks_exact_geometric_tie_ambiguous() -> None:
    old = [_event("old", start=0.40, end=0.60)]
    new = [
        _event("new-a", start=0.35, end=0.55),
        _event("new-b", start=0.45, end=0.65),
    ]
    rows = compare_events(old, new)
    ambiguous = [row for row in rows if row["category"] == "ambiguous"]
    assert len(ambiguous) == 1
    assert ambiguous[0]["candidate_new_event_ids"] == "new-a;new-b"
    assert {row["category"] for row in rows} == {"ambiguous", "new"}


def test_comparison_uses_global_matching_for_conflicting_candidates() -> None:
    old = [
        _event("old-a", start=0.00, end=0.40),
        _event("old-b", start=0.25, end=0.45),
    ]
    new = [
        _event("new-x", start=0.20, end=0.40),
        _event("new-y", start=0.00, end=0.10),
    ]
    rows = compare_events(old, new)
    pairs = {
        (row["old_event_id"], row["new_event_id"])
        for row in rows
        if row["old_event_id"] and row["new_event_id"]
    }
    assert pairs == {("old-a", "new-y"), ("old-b", "new-x")}
    assert not {"removed", "new"} & {row["category"] for row in rows}
