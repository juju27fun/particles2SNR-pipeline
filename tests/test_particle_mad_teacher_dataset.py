from __future__ import annotations

from dataclasses import replace

from particles2snr.particle_events import ParticleDetectionConfig
from particles2snr.particle_mad_teacher_dataset import (
    AUDIT_SEED,
    interval_iou,
    match_events,
    select_audit_cases,
)


def _addition(index: int, class_name: str, split: str, empty: bool = False) -> dict:
    return {
        "new_event_id": f"mad-event-{index:020d}",
        "source_id": f"campaign_{class_name}_{index}",
        "output_stem": f"{class_name}_{index}",
        "source_path": f"{split}/{class_name}/{index}.npy",
        "source_sha256": f"{index:064x}",
        "source_class": class_name,
        "class_id": {"2um": 0, "4um": 1, "10um": 2}[class_name],
        "output_split": split,
        "campaign": f"campaign_{class_name}",
        "v1_empty_trace": empty,
        "new_event_start": 100 + index,
        "new_event_end": 500 + index,
        "new_width_samples": 400,
        "new_robust_energy_z": 12.0,
        "new_energy_concentration": 0.9,
        "new_dominant_frequency_hz": 20_000.0,
        "roi_start_unclipped": -2_000,
        "roi_end_unclipped": 4_144,
        "status": "added",
        "old_event_id": "",
        "old_event_start": "",
        "old_event_end": "",
        "old_width_samples": "",
        "iou": "",
        "center_delta_samples": "",
        "width_delta_samples": "",
        "admission_mechanism": "previously_rejected",
        "previous_rejection_reason": "too_wide",
        "overlapping_old_candidates": 1,
    }


def test_interval_matching_preserves_identity_after_width_shrink() -> None:
    old = [{"event_start": 100, "event_end": 1_000}, {"event_start": 2_000, "event_end": 3_000}]
    new = [{"event_start": 180, "event_end": 920}, {"event_start": 2_200, "event_end": 2_800}]
    matches = match_events(old, new)
    assert [(old_index, new_index) for old_index, new_index, _ in matches] == [(0, 0), (1, 1)]
    assert interval_iou((100, 1_000), (180, 920)) > 0.8


def test_audit_sample_is_exact_deterministic_and_contains_mandatory_strata() -> None:
    rows = []
    index = 0
    for class_name, total in (("10um", 103), ("2um", 32), ("4um", 11)):
        for offset in range(total):
            split = "test" if (
                (class_name == "10um" and offset < 20)
                or (class_name == "2um" and offset < 7)
                or (class_name == "4um" and offset < 2)
            ) else ("val" if offset % 6 == 0 else "train")
            rows.append(_addition(index, class_name, split, empty=offset % 2 == 0))
            index += 1
    first = select_audit_cases(rows)
    second = select_audit_cases(list(reversed(rows)))
    selected_ids = {row["new_event_id"] for row in first}
    mandatory_ids = {
        row["new_event_id"]
        for row in rows
        if row["output_split"] == "test" or row["source_class"] == "4um"
    }
    assert AUDIT_SEED
    assert len(first) == len({row["case_id"] for row in first}) == 60
    assert [row["case_id"] for row in first] == [row["case_id"] for row in second]
    assert mandatory_ids <= selected_ids
    assert sum(row["output_split"] == "test" for row in first) == 29
    assert sum(row["source_class"] == "4um" for row in first) == 11
    assert sum(row["source_class"] == "2um" for row in first) == 18
    assert sum(row["source_class"] == "10um" for row in first) == 31


def test_active_frames_only_configuration_is_explicit() -> None:
    config = replace(
        ParticleDetectionConfig(),
        boundary_expansion_enabled=False,
        boundary_pad_ms=0.0,
        active_min_concentration=0.0,
    )
    assert config.boundary_expansion_enabled is False
    assert config.boundary_pad_ms == 0.0
    assert config.active_min_concentration == 0.0
