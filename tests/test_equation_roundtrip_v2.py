from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from particles2snr.equation_roundtrip import CHECKPOINT_SHA256
from particles2snr.equation_roundtrip_v2 import (
    CLAIM_VARIANTS,
    VARIANTS,
    VIEWS_PER_EVENT,
    build_detector_faithful_candidate,
    detector_spectral_calibration,
    join_detector_provenance,
    lowest_energy_windows,
    validate_detector_faithful_candidate,
)


def _event(event_id: str, filename: str, center: float) -> dict[str, str]:
    snr_db = 10.0 * np.log10(400.0 / 100.0)
    return {
        "event_id": event_id,
        "split": "val",
        "source_filename": filename,
        "source_signal_relative_path": f"val/signals/{filename}",
        "class_name": "2um",
        "annotation_origin": "dual_clean_strict",
        "center_norm": str(center),
        "start_sample": str(center * 16384 - 500),
        "end_sample": str(center * 16384 + 500),
        "particles2snr_amplitude": "0.4",
        "frequency_hz": "22000",
        "tau_ms": "0.2",
        "snr_db": str(snr_db),
    }


def _particle(event: dict[str, str], index: int) -> dict[str, str]:
    return {
        "filename": event["source_filename"],
        "particle_idx": str(index),
        "frequency": event["frequency_hz"],
        "P0": event["particles2snr_amplitude"],
        "t0": str(float(event["center_norm"]) * 16384 / 2_000_000),
        "tau": str(float(event["tau_ms"]) / 1000.0),
        "phi": str(0.3 + index),
        "energy": "400",
        "snr_db": event["snr_db"],
        "noise_floor": "100",
        "noise_floor_N": "3",
        "snr_method": "peak_bin_energy_over_lowest_window_energy",
        "source_window_idx": "2",
        "source_window_center": "4096",
        "source_window_energy": "900",
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_detector_provenance_join_is_exact_and_rejects_ambiguity() -> None:
    event = _event("a", "a.npy", 0.5)
    particle = _particle(event, 0)
    joined = join_detector_provenance([event], [particle])
    assert joined["a"]["phi"] == particle["phi"]
    with pytest.raises(ValueError, match="2 matches"):
        join_detector_provenance([event], [particle, dict(particle)])


def test_detector_spectral_calibration_matches_both_energy_targets() -> None:
    time = (np.arange(4096) - 2047.5) / 2_000_000
    clean = np.cos(2 * np.pi * 22000 * time) * np.exp(
        -0.5 * np.square(time / 0.0002)
    )
    noise = np.random.default_rng(3).normal(size=4096)
    calibrated, evidence = detector_spectral_calibration(
        clean,
        noise,
        frequency_hz=22000,
        target_peak_energy=400.0,
        target_noise_floor=100.0,
    )
    assert calibrated.shape == (4096,)
    assert calibrated.dtype == np.float32
    assert evidence["achieved_peak_energy"] == pytest.approx(400.0, rel=2.0e-5)
    assert evidence["achieved_noise_floor"] == pytest.approx(100.0)
    assert evidence["achieved_detector_snr_db"] == pytest.approx(
        10.0 * np.log10(4.0), abs=1.0e-4
    )


def test_low_energy_windows_are_deterministic() -> None:
    signal = np.random.default_rng(2).normal(size=16384).astype(np.float32)
    first = lowest_energy_windows(signal, stride=512)
    second = lowest_energy_windows(signal, stride=512)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (VIEWS_PER_EVENT, 4096)


def test_v2_candidate_is_deterministic_source_disjoint_and_valid(
    tmp_path: Path,
) -> None:
    table = tmp_path / "table"
    source = tmp_path / "source"
    detector = tmp_path / "detector_train_particles.csv"
    output = tmp_path / "candidate"
    table.mkdir()
    (source / "val/signals").mkdir(parents=True)
    events = [
        _event("a", "a.npy", 0.45),
        _event("b", "b.npy", 0.55),
    ]
    _write_csv(table / "events.csv", events)
    _write_csv(detector, [_particle(event, index) for index, event in enumerate(events)])
    for index, event in enumerate(events):
        rng = np.random.default_rng(index)
        signal = rng.normal(scale=0.02, size=16384).astype(np.float32)
        np.save(source / event["source_signal_relative_path"], signal)

    summary = build_detector_faithful_candidate(
        event_table_root=table,
        signal_dataset_root=source,
        detector_particles_csv=detector,
        output_dir=output,
        source_manifest_sha256="a" * 64,
        signal_manifest_sha256="b" * 64,
        checkpoint_sha256=CHECKPOINT_SHA256,
    )
    validation = validate_detector_faithful_candidate(output)

    assert validation["valid"] is True
    assert summary["record_count"] == 2 * (
        1 + len(VARIANTS) * VIEWS_PER_EVENT
    )
    with (output / "records.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    claim_rows = [row for row in rows if row["variant"] in CLAIM_VARIANTS]
    assert claim_rows
    assert all(
        row["noise_source_filename"] != row["source_group"]
        for row in claim_rows
        if row["noise_source_filename"]
    )
    ceiling = [
        row
        for row in rows
        if row["variant"] == "detector_empirical_same_source_fitted_phase"
    ]
    assert all(row["claim_role"] == "leakage_ceiling" for row in ceiling)
    contract = json.loads((output / "input_contract.json").read_text())
    assert (
        contract["parameter_units"]["deprecated_passage_time_ms"]
        == "ms; legacy alias, do not filter"
    )


def test_candidate_accepts_split_detector_tables_and_explicit_dataset_ids(
    tmp_path: Path,
) -> None:
    table = tmp_path / "table"
    source = tmp_path / "source"
    detector_train = tmp_path / "detector_train_particles.csv"
    detector_val = tmp_path / "detector_val_particles.csv"
    output = tmp_path / "candidate"
    table.mkdir()
    (source / "val/signals").mkdir(parents=True)
    events = [
        _event("a", "a.npy", 0.45),
        _event("b", "b.npy", 0.55),
    ]
    _write_csv(table / "events.csv", events)
    _write_csv(detector_train, [_particle(events[0], 0)])
    _write_csv(detector_val, [_particle(events[1], 1)])
    for index, event in enumerate(events):
        signal = np.random.default_rng(index).normal(
            scale=0.02, size=16384
        ).astype(np.float32)
        np.save(source / event["source_signal_relative_path"], signal)

    dataset_id = "particles2snr-z8-equation-roundtrip@v3"
    summary = build_detector_faithful_candidate(
        event_table_root=table,
        signal_dataset_root=source,
        detector_particles_csv=(detector_train, detector_val),
        output_dir=output,
        source_manifest_sha256="a" * 64,
        signal_manifest_sha256="b" * 64,
        checkpoint_sha256=CHECKPOINT_SHA256,
        dataset_id=dataset_id,
        source_dataset_id="z8@v2",
        signal_dataset_id="p2snr-f@v2",
    )
    validation = validate_detector_faithful_candidate(
        output, expected_dataset_id=dataset_id
    )

    assert validation["valid"] is True
    assert summary["dataset_id"] == dataset_id
    assert set(summary["source_datasets"]) == {"z8@v2", "p2snr-f@v2"}
    assert len(summary["detector_particles_csv_sha256s"]) == 2
