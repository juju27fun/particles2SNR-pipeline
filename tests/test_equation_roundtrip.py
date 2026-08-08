from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from particles2snr.equation_roundtrip import (
    CHECKPOINT_SHA256,
    MODEL_INPUT_LENGTH,
    annotation_width_ms,
    build_equation_roundtrip_candidate,
    centered_crop,
    classifier_preprocess,
    deterministic_seed,
    sigma_ms,
    synthesize_equation_view,
    validate_equation_roundtrip_candidate,
)


def _event(event_id: str, center: float, tau: float = 0.2) -> dict[str, str]:
    return {
        "event_id": event_id,
        "split": "val",
        "source_filename": f"{event_id}.npy",
        "source_signal_relative_path": f"val/signals/{event_id}.npy",
        "class_name": "2um",
        "annotation_origin": "dual_clean_strict",
        "center_norm": str(center),
        "start_sample": str(center * 16384 - 500),
        "end_sample": str(center * 16384 + 500),
        "particles2snr_amplitude": "0.4",
        "frequency_hz": "22000",
        "tau_ms": str(tau),
        "snr_db": "10",
    }


def test_width_contract_distinguishes_tau_and_annotation_passage() -> None:
    event = _event("a", 0.5, tau=0.2)
    assert annotation_width_ms(event) == pytest.approx(0.5)
    assert sigma_ms(event, "fitted_tau_as_sigma") == pytest.approx(0.2)
    assert sigma_ms(event, "fitted_tau_as_fwhm") == pytest.approx(0.2 / 2.355)
    assert sigma_ms(event, "annotation_width_as_fwhm") == pytest.approx(0.5 / 2.355)
    assert sigma_ms(event, "annotation_width_as_sigma") == pytest.approx(0.5)


def test_centered_crop_pads_edges_and_preprocesses_by_mean_decimation() -> None:
    signal = np.arange(4096, dtype=np.float32)
    crop = centered_crop(signal, 0.0)
    assert crop.shape == (4096,)
    assert np.all(crop[:2048] == 0.0)
    native = np.repeat(np.arange(MODEL_INPUT_LENGTH, dtype=np.float32), 8)
    result = classifier_preprocess(native)
    expected = np.arange(MODEL_INPUT_LENGTH, dtype=np.float32)
    expected = (expected - expected.mean()) / expected.std()
    np.testing.assert_allclose(result, expected, atol=1.0e-6)


def test_equation_views_are_deterministic_and_finite() -> None:
    kwargs = {
        "amplitude": 0.3,
        "frequency_hz": 25000.0,
        "sigma_ms_value": 0.15,
        "snr_db": 8.0,
        "phase_rad": 0.3,
        "seed": deterministic_seed(7, "event", "fitted_tau_as_sigma", 2),
    }
    first = synthesize_equation_view(**kwargs)
    second = synthesize_equation_view(**kwargs)
    np.testing.assert_array_equal(first, second)
    assert first.dtype == np.float32
    assert np.isfinite(first).all()
    assert float(np.std(first)) > 0.0


def test_candidate_pairing_and_negative_control_are_valid(tmp_path: Path) -> None:
    table = tmp_path / "table"
    source = tmp_path / "source"
    output = tmp_path / "candidate"
    table.mkdir()
    (source / "val/signals").mkdir(parents=True)
    events = [_event("a", 0.45), _event("b", 0.55, tau=0.25)]
    with (table / "events.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(events[0]))
        writer.writeheader()
        writer.writerows(events)
    for event in events:
        rng = np.random.default_rng(int(event["event_id"] == "b"))
        np.save(
            source / event["source_signal_relative_path"],
            rng.normal(size=16384).astype(np.float32),
        )
    summary = build_equation_roundtrip_candidate(
        event_table_root=table,
        signal_dataset_root=source,
        output_dir=output,
        source_manifest_sha256="a" * 64,
        signal_manifest_sha256="b" * 64,
        checkpoint_sha256=CHECKPOINT_SHA256,
        maximum_events=2,
    )
    validation = validate_equation_roundtrip_candidate(output)
    assert validation["valid"] is True
    assert summary["event_count"] == 2
    assert summary["record_count"] == 2 * (1 + 8 * 8)
    with (output / "records.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    shuffled = [row for row in rows if row["variant"] == "shuffle_tau"]
    assert shuffled
    assert all(
        row["source_event_id"] != row["shuffled_parameter_source_event_id"]
        for row in shuffled
    )
    dataset_summary = json.loads(
        (output / "dataset_summary.json").read_text(encoding="utf-8")
    )
    assert dataset_summary["sealed_test_accessed"] is False
