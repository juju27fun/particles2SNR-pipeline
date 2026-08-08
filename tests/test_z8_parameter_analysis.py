from __future__ import annotations

from collections import Counter

import numpy as np

from particles2snr.z8_parameter_analysis import (
    build_analysis,
    boundary_eligible_rows,
    grouped_values,
    select_extremes,
    validate_events,
    resolve_registered_z8_dataset,
)
from internship_workspace.config import Workspace


UNCLEAR_BY_SOURCE = {"2um": 48, "4um": 29, "10um": 17}
EXPECTED_CLASS_COUNTS = {"2um": 534, "4um": 1288, "10um": 336, "unclear": 94}


def _row(
    *, event_id: str, class_name: str, physical_source_class: str, index: int
) -> dict[str, str]:
    class_offset = {"2um": 0.0, "4um": 1.0, "10um": 2.0}[physical_source_class]
    unclear = class_name == "unclear"
    return {
        "event_id": event_id,
        "split": "train" if index % 2 == 0 else "val",
        "source_filename": f"signal-{event_id}.npy",
        "source_signal_relative_path": f"train/signals/signal-{event_id}.npy",
        "physical_source_class": physical_source_class,
        "class_name": class_name,
        "annotation_origin": "dual_clean_strict" if index % 3 else "z8_rescue",
        "start_sample": "6000",
        "end_sample": "7000",
        "particles2snr_amplitude": str(
            0.02 + class_offset * 0.1 + index * (0.004 if index == 0 else 0.0001)
        ),
        "frequency_hz": str(6835.9375 + class_offset * 2000.0 + index * 10.0),
        "tau_ms": str(0.07 + class_offset * 0.01 + index * 0.0001),
        "snr_db": str(-12.0 - index * 0.01 if unclear else -9.0 + class_offset + index * 0.01),
    }


def _events() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for class_name in ("2um", "4um", "10um"):
        for index in range(EXPECTED_CLASS_COUNTS[class_name]):
            rows.append(
                _row(
                    event_id=f"{class_name}-physical-{index}",
                    class_name=class_name,
                    physical_source_class=class_name,
                    index=index,
                )
            )
    for class_name in ("2um", "4um", "10um"):
        for index in range(UNCLEAR_BY_SOURCE[class_name]):
            rows.append(
                _row(
                    event_id=f"{class_name}-unclear-{index}",
                    class_name="unclear",
                    physical_source_class=class_name,
                    index=index,
                )
            )
    return rows


def test_unclear_only_enriches_snr_population() -> None:
    rows = _events()
    validate_events(rows)
    amplitude = grouped_values(rows, "amplitude_p0")
    snr = grouped_values(rows, "snr_effective_fbase_db")

    assert len(amplitude["2um"]) == 534
    assert len(amplitude["4um"]) == 1288
    assert len(amplitude["10um"]) == 336
    assert len(snr["2um"]) == 582
    assert len(snr["4um"]) == 1317
    assert len(snr["10um"]) == 353
    assert np.min(snr["2um"]) < -10.0


def test_analysis_builds_empirical_statistics_and_three_supports() -> None:
    analysis = build_analysis(_events())

    assert analysis["event_count"] == 2252
    assert analysis["sealed_test_accessed"] is False
    assert len(analysis["statistics_rows"]) == 12
    assert len(analysis["support_candidates"]) == 36
    assert {
        row["margin_id"] for row in analysis["support_candidates"]
    } == {"M0", "M10", "M20"}
    assert all(
        row["distribution_policy"] == "empirical_observed"
        for row in analysis["statistics_rows"]
    )

    frequency_rows = [
        row
        for row in analysis["support_candidates"]
        if row["class_name"] == "2um" and row["parameter"] == "frequency_khz"
    ]
    observed_minimum = frequency_rows[0]["observed_minimum"]
    assert observed_minimum < 7.0
    assert all(row["lower_bound"] == observed_minimum for row in frequency_rows)
    assert all(row["upper_bound"] <= 80.0 for row in frequency_rows)


def test_extreme_gallery_selection_is_deduplicated() -> None:
    extremes = select_extremes(_events())
    role_count = sum(len(row["extreme_roles"]) for row in extremes)

    assert role_count == 24
    assert len({row["event_id"] for row in extremes}) == len(extremes)
    assert len(extremes) < 24
    assert any(row["class_name"] == "unclear" for row in extremes)
    for row in extremes:
        if row["class_name"] == "unclear":
            assert all(
                role["parameter"] == "snr_effective_fbase_db"
                for role in row["extreme_roles"]
            )


def test_boundary_censoring_changes_statistics_not_dataset_membership() -> None:
    rows = _events()
    rows[0]["start_sample"] = "0"
    rows[1]["end_sample"] = "16384"
    lengths = {
        row["source_signal_relative_path"]: 16384
        for row in rows
    }

    analysis = build_analysis(
        rows,
        dataset_id="z8-development@v2",
        source_signal_lengths=lengths,
    )

    assert analysis["event_count"] == len(rows)
    assert analysis["eligible_event_count"] == len(rows) - 2
    assert analysis["boundary_censored_event_count"] == 2
    assert {row["event_id"] for row in analysis["boundary_censored_events"]} == {
        rows[0]["event_id"],
        rows[1]["event_id"],
    }
    assert all(
        row["n"] < EXPECTED_CLASS_COUNTS[row["class_name"]]
        if row["class_name"] == "2um" and row["parameter"] != "snr_effective_fbase_db"
        else True
        for row in analysis["statistics_rows"]
    )


def test_boundary_censoring_uses_inclusive_signal_boundaries() -> None:
    rows = _events()[:3]
    rows[0]["start_sample"] = "-0.1"
    rows[0]["end_sample"] = "2"
    rows[1]["start_sample"] = "8"
    rows[1]["end_sample"] = "10"
    rows[2]["start_sample"] = "0.1"
    rows[2]["end_sample"] = "9.9"
    lengths = {row["source_signal_relative_path"]: 10 for row in rows}

    eligible, censored = boundary_eligible_rows(
        rows, source_signal_lengths=lengths
    )

    assert [row["event_id"] for row in eligible] == [rows[2]["event_id"]]
    assert [row["event_id"] for row in censored] == sorted(
        [rows[0]["event_id"], rows[1]["event_id"]]
    )


def test_extreme_ties_use_stable_event_id() -> None:
    rows = _events()
    rows[0]["frequency_hz"] = rows[1]["frequency_hz"]
    rows[0]["event_id"] = "z-tied"
    rows[1]["event_id"] = "a-tied"

    selected = select_extremes(rows)
    roles = [
        (row["event_id"], role)
        for row in selected
        for role in row["extreme_roles"]
        if role["class_name"] == "2um"
        and role["parameter"] == "frequency_khz"
        and role["direction"] == "minimum"
    ]

    assert roles[0][0] == "a-tied"
    assert roles[0][1]["tie_count"] == 2
    assert roles[0][1]["tied_event_ids"] == ["a-tied", "z-tied"]


def test_validation_rejects_sealed_test_rows() -> None:
    rows = _events()
    rows[0]["split"] = "test"
    try:
        validate_events(rows)
    except ValueError as exc:
        assert "Sealed test" in str(exc)
    else:
        raise AssertionError("test split should be rejected")


def test_physical_class_counts_are_unchanged() -> None:
    rows = _events()
    assert Counter(row["class_name"] for row in rows) == EXPECTED_CLASS_COUNTS


def test_validation_accepts_selected_v2_like_summary_not_v1_constants() -> None:
    rows = _events()[:6]
    for index, row in enumerate(rows):
        row["class_name"] = ("2um", "4um", "10um")[index % 3]
        row["physical_source_class"] = row["class_name"]
        row["split"] = "train" if index % 2 else "val"
    summary = {
        "event_count": len(rows),
        "class_counts": {"2um": 2, "4um": 2, "10um": 2},
        "snr_population_counts": {"2um": 2, "4um": 2, "10um": 2},
        "split_counts": {"train": 3, "val": 3},
        "origin_counts": {"dual_clean_strict": 4, "z8_rescue": 2},
    }
    validate_events(rows, dataset_summary=summary)


def test_validation_rejects_counts_not_owned_by_selected_summary() -> None:
    rows = _events()[:6]
    summary = {"event_count": len(rows), "class_counts": {"2um": 5}}
    try:
        validate_events(rows, dataset_summary=summary)
    except ValueError as exc:
        assert "selected dataset summary" in str(exc)
    else:
        raise AssertionError("selected dataset summary must control accepted counts")


def test_analysis_is_deterministic_for_an_explicit_dataset_id() -> None:
    first = build_analysis(_events(), dataset_id="z8-development@v2")
    second = build_analysis(_events(), dataset_id="z8-development@v2")

    assert first == second
    assert first["dataset_id"] == "z8-development@v2"


def test_real_registered_v1_and_v2_z8_summaries_resolve() -> None:
    workspace = Workspace.load()
    base = "particles2snr-fbase-dual-clean-z8-events-3class-plus-unclear-development"
    for version, expected_count in (("v1", 2252), ("v2", 2194)):
        _, _, summary = resolve_registered_z8_dataset(workspace, f"{base}@{version}")
        assert summary is not None
        assert summary["event_count"] == expected_count


def test_real_z8_source_bindings_match_registered_manifest_hashes() -> None:
    workspace = Workspace.load()
    base = "particles2snr-fbase-dual-clean-z8-events-3class-plus-unclear-development"
    expected = {
        "v1": "particles2snr-f-c1-yolo-4class@v1",
        "v2": "particles2snr-f-dual-clean-c1-yolo-4class@v2",
    }
    for version, source_id in expected.items():
        _, _, summary = resolve_registered_z8_dataset(workspace, f"{base}@{version}")
        source_record, _, _ = resolve_registered_z8_dataset(
            workspace, source_id, require_z8_summary=False
        )
        source_binding = summary["source_datasets"][source_id]
        source_hash = source_binding.get("manifest_sha256") if isinstance(source_binding, dict) else source_binding
        assert source_hash == source_record["manifest_sha256"]
