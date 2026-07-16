from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from particles2snr.yeast_simulation import FACTOR_POLICY, build_simulation_dataset, simulate_view
from particles2snr.yeast_simulation import (
    _finite_support_tukey_envelope,
    build_support_calibrated_simulation_dataset,
    fit_support_calibration,
)


def _write_real_calibration_fixture(root: Path, *, forbidden_split: bool = False) -> None:
    root.mkdir()
    time = np.arange(4096, dtype=np.float64) / 1_000_000.0
    signals = []
    for width_ms, frequency_hz in ((0.20, 12_000.0), (0.45, 18_000.0), (0.70, 22_000.0)):
        sigma = width_ms / 1000.0 / 2.355
        envelope = np.exp(-0.5 * np.square((time - 0.002) / sigma))
        signals.append((envelope * np.cos(2.0 * np.pi * frequency_hz * time)).astype(np.float32))
    np.save(root / "signals.npy", np.asarray(signals))
    rows = [
        {"signal_row": 0, "development_split": "followup_train"},
        {"signal_row": 1, "development_split": "followup_train"},
        {
            "signal_row": 2,
            "development_split": "followup_test" if forbidden_split else "followup_validation",
        },
    ]
    with (root / "development_events.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "sealed_followup_test_events.csv").write_text(
        "this file must never be parsed by calibration\n", encoding="utf-8"
    )
    (root / "dataset_summary.json").write_text(
        json.dumps({"dataset_id": "yeast-events-followup@fixture"}), encoding="utf-8"
    )


def test_paired_views_preserve_factors_and_change_nuisances(tmp_path: Path) -> None:
    output = tmp_path / "simulation"
    summary = build_simulation_dataset(
        output_dir=output,
        n_train_latents=2,
        n_validation_latents=1,
        n_test_latents=1,
        views_per_latent=2,
        seed=7,
    )
    with (output / "simulation_metadata.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    signals = np.load(output / "signals.npy", mmap_mode="r")
    assert signals.shape == (8, 4096)
    assert rows[0]["latent_id"] == rows[1]["latent_id"]
    assert rows[0]["duration_ms"] == rows[1]["duration_ms"]
    assert rows[0]["phase_rad"] != rows[1]["phase_rad"]
    assert "envelope_model" not in rows[0]
    assert not np.allclose(signals[0], signals[1])
    assert summary["split_signal_counts"] == {"test": 2, "train": 4, "validation": 2}
    assert "envelope_model" not in summary
    assert json.loads((output / "factor_policy.json").read_text())["phase_rad"]["role"] == (
        "randomize_invariant"
    )


def test_simulated_view_matches_frozen_shape_and_finite_range() -> None:
    factors = {
        "duration_ms": 0.8,
        "doppler_khz": 18.0,
        "component_count": 2,
        "component_separation_ms": 0.2,
        "relative_component_amplitude": 0.7,
        "frequency_separation_khz": 3.0,
    }
    signal, nuisance = simulate_view(np.random.default_rng(3), factors)
    assert signal.shape == (4096,)
    assert np.isfinite(signal).all()
    assert 0.39 <= float(np.sqrt(np.mean(np.square(signal)))) <= 1.71
    assert nuisance["snr_db"] >= 0.0
    assert FACTOR_POLICY["yeast_morphology"]["role"] == "unresolved_excluded"


def test_support_calibration_uses_train_development_signals_only(tmp_path: Path) -> None:
    root = tmp_path / "real"
    _write_real_calibration_fixture(root)
    calibration = fit_support_calibration(root, quantile_knots=5)
    assert calibration["source_split"] == "followup_train"
    assert calibration["n_train_signals"] == 2
    assert len(calibration["support_duration_ms_quantiles"]) == 5
    assert calibration["sealed_splits_used"] == []
    assert calibration["source_checksums"]["development_events.csv"]


def test_support_calibration_rejects_final_split_in_development_metadata(tmp_path: Path) -> None:
    root = tmp_path / "real"
    _write_real_calibration_fixture(root, forbidden_split=True)
    try:
        fit_support_calibration(root)
    except PermissionError as error:
        assert "forbidden splits" in str(error)
    else:
        raise AssertionError("Final split metadata must be rejected")


def test_finite_support_envelope_matches_declared_threshold_duration() -> None:
    time = np.arange(8192, dtype=np.float64) / 2_000_000.0
    envelope = _finite_support_tukey_envelope(
        time,
        center=0.002,
        target_support_ms=0.60,
        alpha=0.50,
    )
    observed_ms = np.sum(envelope >= 0.25 * envelope.max()) / 2_000_000.0 * 1000.0
    assert abs(observed_ms - 0.60) <= 0.002
    assert np.count_nonzero(envelope == 0.0) > 0


def test_support_calibrated_dataset_is_deterministic_and_manifested(tmp_path: Path) -> None:
    root = tmp_path / "real"
    _write_real_calibration_fixture(root)
    first = tmp_path / "first"
    second = tmp_path / "second"
    kwargs = {
        "real_root": root,
        "n_train_latents": 2,
        "n_validation_latents": 1,
        "n_test_latents": 1,
        "views_per_latent": 2,
        "seed": 7,
        "quantile_knots": 5,
    }
    first_summary = build_support_calibrated_simulation_dataset(output_dir=first, **kwargs)
    second_summary = build_support_calibrated_simulation_dataset(output_dir=second, **kwargs)
    first_signals = np.load(first / "signals.npy")
    second_signals = np.load(second / "signals.npy")
    assert np.array_equal(first_signals, second_signals)
    assert np.isfinite(first_signals).all()
    assert first_summary == second_summary
    assert first_summary["generator_id"] == "yeast-passage-finite-support-v2"
    calibration = json.loads((first / "support_calibration.json").read_text())
    assert calibration["sealed_splits_used"] == []
