from __future__ import annotations

import json

import numpy as np

from particles2snr.yeast_schmoo_simulation import (
    FAMILIES,
    OUTPUT_LENGTH,
    build_schmoo_physical_sweep,
    sample_schmoo_latent,
    shape_cloud,
    simulate_schmoo_view,
)


def _calibration() -> dict:
    probabilities = np.linspace(0.01, 0.99, 99).tolist()
    contract = {
        "n_events": 20,
        "frequency_khz_quantiles": np.linspace(8.0, 27.0, 99).tolist(),
        "width_ms_quantiles": np.linspace(0.65, 1.55, 99).tolist(),
        "target_rms_quantiles": np.linspace(0.08, 0.20, 99).tolist(),
        "snr_db_quantiles": np.linspace(6.0, 18.0, 99).tolist(),
    }
    return {
        "schema_version": 1,
        "calibration_id": "test-calibration",
        "source_split": "development_train",
        "strata": {"shmoo": contract, "shmoo2": contract},
        "quantile_probabilities": probabilities,
        "sealed_splits_used": [],
        "m2_identifiability_rule": {"ambiguous_policy": "not applicable"},
    }


def test_shape_families_are_three_dimensional_and_distinct() -> None:
    calibration = _calibration()
    volumes = {}
    for index, family in enumerate(FAMILIES):
        latent = sample_schmoo_latent(
            np.random.default_rng(100 + index),
            calibration,
            family=family,
        )
        points, weights = shape_cloud(latent)
        assert points.ndim == 2 and points.shape[1] == 3
        assert np.all(np.isfinite(points))
        assert np.all(weights > 0.0)
        volumes[family] = float(np.sum(weights))
    assert volumes["T0"] != volumes["S0"]
    assert volumes["M1"] != volumes["S0"]


def test_schmoo_view_is_deterministic_finite_and_has_expected_shape() -> None:
    calibration = _calibration()
    latent = sample_schmoo_latent(
        np.random.default_rng(7),
        calibration,
        family="M1",
    )
    first, metadata = simulate_schmoo_view(
        np.random.default_rng(99),
        latent,
        calibration,
        variant="base",
    )
    second, _ = simulate_schmoo_view(
        np.random.default_rng(99),
        latent,
        calibration,
        variant="base",
    )
    assert first.shape == (OUTPUT_LENGTH,)
    assert first.dtype == np.float32
    assert np.all(np.isfinite(first))
    assert np.array_equal(first, second)
    assert metadata["n_shape_points"] > 96
    assert float(np.std(first)) > 0.0


def test_small_sweep_writes_balanced_manifestable_dataset(tmp_path) -> None:
    output = tmp_path / "candidate"
    summary = build_schmoo_physical_sweep(
        output_dir=output,
        calibration=_calibration(),
        n_train_per_family=1,
        n_validation_per_family=1,
        n_test_per_family=1,
        seed=123,
    )
    signals = np.load(output / "signals.npy", allow_pickle=False)
    assert signals.shape == (9, OUTPUT_LENGTH)
    assert summary["family_counts"] == {"M1": 3, "S0": 3, "T0": 3}
    assert summary["sealed_splits_used_for_generation"] == []
    contract = json.loads((output / "parameter_contract.json").read_text())
    assert contract["families"]["M1"].startswith("sphere plus")
