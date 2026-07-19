from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from particles2snr.yeast_budding_dataset import (
    COPULA_FEATURES,
    build_budding_calibration,
    build_budding_simulation_dataset,
    sample_biophysics_oriented_latent,
    sample_data_oriented_latent,
)


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    real_root = tmp_path / "real"
    real_root.mkdir()
    signals = np.random.default_rng(2).normal(size=(16, 4096)).astype(np.float32)
    np.save(real_root / "signals.npy", signals)
    event_fields = (
        "event_id",
        "signal_row",
        "source_group",
        "development_split",
        "quality",
        "event_start_input_index",
        "event_end_input_index",
    )
    with (real_root / "events.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=event_fields)
        writer.writeheader()
        for index in range(16):
            writer.writerow(
                {
                    "event_id": f"event-{index}",
                    "signal_row": index,
                    "source_group": "budding",
                    "development_split": "development_train",
                    "quality": "strict",
                    "event_start_input_index": 1400,
                    "event_end_input_index": 2600,
                }
            )
    fit_path = tmp_path / "fits.csv"
    fields = [
        "event_id",
        "delta_bic_m1_minus_m2",
        "resolvability_score",
        *[
            f"m2_c{component}_{name}"
            for component in (1, 2)
            for name in (
                "amplitude",
                "center_ms",
                "sigma_left_ms",
                "sigma_right_ms",
                "shape",
                "frequency_khz",
                "chirp_khz_per_ms",
            )
        ],
    ]
    with fit_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(16):
            resolved = index < 8
            writer.writerow(
                {
                    "event_id": f"event-{index}",
                    "delta_bic_m1_minus_m2": 20 if resolved else 2,
                    "resolvability_score": 0.5 if resolved else 0.02,
                    "m2_c1_amplitude": 1.0 + index * 0.01,
                    "m2_c1_center_ms": 1.8 + index * 0.01,
                    "m2_c1_sigma_left_ms": 0.14 + index * 0.001,
                    "m2_c1_sigma_right_ms": 0.16 + index * 0.001,
                    "m2_c1_shape": 1.8 + index * 0.01,
                    "m2_c1_frequency_khz": 18.0 + index * 0.1,
                    "m2_c1_chirp_khz_per_ms": -1.0 + index * 0.1,
                    "m2_c2_amplitude": 0.55 + index * 0.005,
                    "m2_c2_center_ms": 2.1 + index * 0.01,
                    "m2_c2_sigma_left_ms": 0.10 + index * 0.001,
                    "m2_c2_sigma_right_ms": 0.12 + index * 0.001,
                    "m2_c2_shape": 2.0 + index * 0.01,
                    "m2_c2_frequency_khz": 19.0 + index * 0.1,
                    "m2_c2_chirp_khz_per_ms": -0.8 + index * 0.1,
                }
            )
    return fit_path, real_root


def test_calibration_and_samplers_are_deterministic(tmp_path: Path) -> None:
    fit_path, real_root = _fixture(tmp_path)
    calibration = build_budding_calibration(
        fit_summaries_csv=fit_path,
        real_dataset_root=real_root,
        source_dataset_id="fixture@v1",
    )
    assert calibration["sealed_splits_used"] == []
    assert calibration["m2_identifiability_rule"]["resolved_fraction"] == 0.5
    left = sample_data_oriented_latent(np.random.default_rng(4), calibration)
    right = sample_data_oriented_latent(np.random.default_rng(4), calibration)
    assert left == right
    assert set(COPULA_FEATURES) <= set(left.to_dict())

    physical = sample_biophysics_oriented_latent(
        np.random.default_rng(7),
        calibration,
        amplitude_size_exponent=2.0,
        beam_radius_relative=1.0,
    )
    assert 0.25 <= physical.bud_radius_ratio <= 0.95
    assert physical.relative_amplitude == physical.bud_radius_ratio**2
    assert physical.mother_radius_relative is not None


def test_small_dataset_has_paired_views_and_manifest(tmp_path: Path) -> None:
    fit_path, real_root = _fixture(tmp_path)
    calibration = build_budding_calibration(
        fit_summaries_csv=fit_path,
        real_dataset_root=real_root,
        source_dataset_id="fixture@v1",
    )
    output = tmp_path / "generated"
    summary = build_budding_simulation_dataset(
        output_dir=output,
        calibration=calibration,
        generator="biophysics",
        n_train_latents=2,
        n_validation_latents=1,
        n_test_latents=1,
        views_per_latent=2,
        seed=11,
    )
    signals = np.load(output / "signals.npy")
    assert signals.shape == (8, 4096)
    assert np.isfinite(signals).all()
    assert summary["split_signal_counts"] == {
        "test": 2,
        "train": 4,
        "validation": 2,
    }
    with (output / "simulation_metadata.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert [row["view_index"] for row in rows[:2]] == ["0", "1"]
    assert rows[0]["latent_id"] == rows[1]["latent_id"]
