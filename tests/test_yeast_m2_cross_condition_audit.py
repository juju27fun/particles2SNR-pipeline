from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from particles2snr.yeast_m2_cross_condition_audit import (
    fit_event,
    input_event_bounds,
    load_split_roles,
    select_strict_event_population,
    stable_shard,
    supervision_status,
    support_is_observed,
    validate_method_approval,
    validate_merged_population,
)


def _row(event_id: str, *, class_name: str = "budding", split: str = "development_train") -> dict[str, str]:
    source = "shmoo2" if class_name == "shmoo" else class_name
    return {
        "sample_id": f"event:{event_id}",
        "sample_kind": "event",
        "event_id": event_id,
        "record_id": f"record:{event_id}",
        "capture_block_id": f"{source}:block-001",
        "class_name": class_name,
        "source_group_original": source,
        "development_split": split,
        "quality": "strict",
        "signal_row": "0",
        "crop_start": "4096",
        "event_start": "7296",
        "event_end": "9088",
        "crop_8192_pad_left": "0",
        "crop_8192_pad_right": "0",
    }


def _passage(center_ms: float, sigma_ms: float, frequency_khz: float, amplitude: float = 1.0) -> np.ndarray:
    time_ms = np.arange(4096, dtype=np.float64) / 1000.0
    envelope = np.exp(-0.5 * np.square((time_ms - center_ms) / sigma_ms))
    return amplitude * envelope * np.cos(2.0 * np.pi * frequency_khz * (time_ms - center_ms))


def test_split_roles_refuse_an_open_external_validation(tmp_path: Path) -> None:
    path = tmp_path / "split.json"
    path.write_text(
        json.dumps(
            {
                "external_holdout_status": "open",
                "assignments": {"train_core": [], "model_selection": []},
            }
        )
    )
    with pytest.raises(ValueError, match="development_validation closed"):
        load_split_roles(path)


def test_method_approval_requires_a_complete_hashed_decision(tmp_path: Path) -> None:
    review = tmp_path / "review"
    review.mkdir()
    decisions = {
        "complete": True,
        "decisions": {
            "yeast-physics-grounded-classifier-method": {"decision": "approved"}
        },
    }
    decisions_path = review / "decisions.json"
    decisions_path.write_text(json.dumps(decisions), encoding="utf-8")
    import hashlib

    digest = hashlib.sha256(decisions_path.read_bytes()).hexdigest()
    (review / "receipt.json").write_text(
        json.dumps(
            {
                "run_id": "yeast-physics-grounded-classifier-method-r1",
                "decisions_sha256": digest,
            }
        ),
        encoding="utf-8",
    )
    assert validate_method_approval(tmp_path)["receipt_sha256"]


def test_population_uses_only_strict_development_train_events() -> None:
    rows = [
        _row("b:00", class_name="budding"),
        _row("m:00", class_name="mix"),
        _row("s:00", class_name="shmoo"),
        {**_row("held:00", class_name="budding", split="development_validation")},
        {**_row("background:00"), "sample_kind": "background", "class_name": "background"},
    ]
    roles = {row["sample_id"]: "train_core" for row in rows[:3]}
    selected = select_strict_event_population(rows, roles)
    assert [row["event_id"] for row in selected] == ["b:00", "m:00", "s:00"]
    assert all(row["development_split"] == "development_train" for row in selected)


def test_input_bounds_and_guard_use_the_processed_4096_domain() -> None:
    start, end = input_event_bounds(_row("b:00"))
    assert start == 1600.0
    assert end == 2496.0
    assert support_is_observed(start, end)
    assert not support_is_observed(200.0, 1200.0)


def test_fit_event_emits_canonical_time_order_and_frozen_weight() -> None:
    signal = _passage(1.75, 0.12, 14.0) + _passage(2.30, 0.10, 22.0, 0.70)
    row = {**_row("double:00"), "role": "train_core"}
    result = fit_event(row, signal.astype(np.float32))
    assert result["fit_success"] is True
    assert result["fit_finite"] is True
    assert result["fit_eligible"] is True
    assert result["delta_t0_ms"] > 0.0
    assert np.isfinite([result["log_A_A"], result["delta_phi_rad"], result["fit_weight"]]).all()
    expected = np.sqrt(
        np.clip(result["delta_bic_m1_minus_m2"] / 50.0, 0.0, 1.0)
        * np.clip(result["resolvability_score"], 0.0, 1.0)
    )
    assert result["fit_weight"] == pytest.approx(expected)


def test_shards_are_stable_and_merge_rejects_duplicates() -> None:
    first = [stable_shard(f"event-{index}", 16) for index in range(20)]
    second = [stable_shard(f"event-{index}", 16) for index in range(20)]
    assert first == second
    population = [{"event_id": "a"}, {"event_id": "b"}]
    with pytest.raises(ValueError, match="duplicates"):
        validate_merged_population(population, [{"event_id": "a"}, {"event_id": "a"}])


def test_supervision_status_thresholds_are_exact() -> None:
    assert supervision_status(300, 600) == "ready"
    assert supervision_status(299, 500) == "partial"
    assert supervision_status(100, 1000) == "partial"
    assert supervision_status(99, 100) == "synthetic_only"
