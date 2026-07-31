from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from particles2snr.z8_cholesky_generation import (
    MODEL_LENGTH,
    RAW_LENGTH,
    generate_parameters,
    load_gaussian_targets,
    load_physical_cholesky,
    load_recommended_cholesky,
    preprocess_conv1dgap_512,
    synthesize_signals,
    validate_candidate,
)


def _targets() -> dict[str, dict]:
    return {
        class_name: {
            "mean": np.asarray([-1.0, 20.0, -2.0, 3.0]),
            "sigma": np.asarray([0.2, 2.0, 0.1, 1.5]),
            "populations": ["physical", "physical", "physical", "inclusive"],
        }
        for class_name in ("2um", "4um", "10um")
    }


def _factors() -> dict[str, np.ndarray]:
    correlation = np.asarray(
        [
            [1.0, 0.2, 0.0, 0.5],
            [0.2, 1.0, -0.3, 0.0],
            [0.0, -0.3, 1.0, 0.1],
            [0.5, 0.0, 0.1, 1.0],
        ]
    )
    factor = np.linalg.cholesky(correlation)
    return {class_name: factor for class_name in ("2um", "4um", "10um")}


def test_generate_parameters_is_deterministic_and_correlated() -> None:
    budgets = {"2um": 2_000, "4um": 2_000, "10um": 2_000}
    first, first_rejections = generate_parameters(
        _targets(),
        _factors(),
        seed=17,
        budgets=budgets,
        dataset_id="synthetic-test@v2",
    )
    second, second_rejections = generate_parameters(
        _targets(),
        _factors(),
        seed=17,
        budgets=budgets,
        dataset_id="synthetic-test@v2",
    )
    assert first == second
    assert first_rejections == second_rejections
    matrix = np.asarray(
        [
            [
                row["log_amplitude_p0"],
                row["frequency_khz"],
                row["log_tau_ms"],
                row["snr_db"],
            ]
            for row in first
            if row["class_name"] == "2um"
        ]
    )
    target = _factors()["2um"] @ _factors()["2um"].T
    assert np.max(np.abs(np.corrcoef(matrix, rowvar=False) - target)) < 0.06
    assert all(row["amplitude_p0"] > 0.0 and row["tau_ms"] > 0.0 for row in first)
    assert all(7.0 <= row["frequency_khz"] <= 80.0 for row in first)
    assert all(row["t0_fraction"] == 0.5 for row in first)


def test_generate_parameters_supports_single_class_density_ablation() -> None:
    records, rejections = generate_parameters(
        _targets(),
        _factors(),
        seed=41,
        budgets={"10um": 37},
        dataset_id="particles2snr-test-10um-density10x@v1",
    )
    repeated, repeated_rejections = generate_parameters(
        _targets(),
        _factors(),
        seed=41,
        budgets={"10um": 37},
        dataset_id="particles2snr-test-10um-density10x@v1",
    )

    assert records == repeated
    assert rejections == repeated_rejections
    assert len(records) == 37
    assert {row["class_name"] for row in records} == {"10um"}
    assert len({row["sample_id"] for row in records}) == 37


def test_synthesis_achieves_snr_and_conv1dgap_contract() -> None:
    records, _ = generate_parameters(
        _targets(),
        _factors(),
        seed=23,
        budgets={"2um": 2, "4um": 2, "10um": 2},
        dataset_id="synthetic-test@v2",
    )
    raw, encoded = synthesize_signals(records, seed=24, batch_size=3)
    assert raw.shape == (6, RAW_LENGTH)
    assert encoded.shape == (6, MODEL_LENGTH)
    assert raw.dtype == np.float32
    assert encoded.dtype == np.float32
    assert np.all(np.isfinite(raw))
    assert np.allclose(encoded.mean(axis=1), 0.0, atol=2e-6)
    assert np.allclose(encoded.std(axis=1), 1.0, atol=2e-6)
    assert max(abs(row["snr_db"] - row["achieved_snr_db"]) for row in records) < 1e-5


def test_preprocess_rejects_wrong_length() -> None:
    wrong = np.ones((2, RAW_LENGTH - 1), dtype=np.float32)
    try:
        preprocess_conv1dgap_512(wrong)
    except ValueError as error:
        assert "Expected signals" in str(error)
    else:
        raise AssertionError("Wrong-length signal should be rejected")


def test_loaders_enforce_physical_matrix_and_inclusive_snr(tmp_path: Path) -> None:
    gaussian_path = tmp_path / "gaussian.csv"
    with gaussian_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "class_name",
            "marginal",
            "population",
            "gaussian_mean_transformed",
            "gaussian_sigma_transformed",
            "class_synthetic_event_count",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        budgets = {"2um": 1151, "4um": 3281, "10um": 366}
        for class_name in budgets:
            for marginal in ("amplitude_p0", "frequency_khz", "tau_ms", "snr_db"):
                writer.writerow(
                    {
                        "class_name": class_name,
                        "marginal": marginal,
                        "population": "inclusive" if marginal == "snr_db" else "physical",
                        "gaussian_mean_transformed": 1.0,
                        "gaussian_sigma_transformed": 2.0,
                        "class_synthetic_event_count": budgets[class_name],
                    }
                )
    targets = load_gaussian_targets(gaussian_path)
    assert targets["10um"]["populations"][-1] == "inclusive"
    loaded_targets, loaded_budgets = load_gaussian_targets(
        gaussian_path,
        include_budgets=True,
    )
    assert loaded_targets["4um"]["budget"] == 3281
    assert loaded_budgets == budgets

    factor_path = tmp_path / "factors.csv"
    with factor_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "class_name",
            "population",
            "row_parameter",
            "column_parameter",
            "cholesky_value",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        parameters = (
            "log_amplitude_p0",
            "frequency_khz",
            "log_tau_ms",
            "snr_db",
        )
        for class_name in ("2um", "4um", "10um"):
            for population in ("physical", "inclusive"):
                for row in range(4):
                    for column in range(row + 1):
                        writer.writerow(
                            {
                                "class_name": class_name,
                                "population": population,
                                "row_parameter": parameters[row],
                                "column_parameter": parameters[column],
                                "cholesky_value": 1.0 if row == column else 0.0,
                            }
                        )
    factors = load_physical_cholesky(factor_path)
    assert all(np.array_equal(value, np.eye(4)) for value in factors.values())

    recommendations_path = tmp_path / "recommendations.csv"
    with recommendations_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["class_name", "recommended_dependency_matrix"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "class_name": "2um",
                    "recommended_dependency_matrix": "physical",
                },
                {
                    "class_name": "4um",
                    "recommended_dependency_matrix": "inclusive",
                },
                {
                    "class_name": "10um",
                    "recommended_dependency_matrix": "physical",
                },
            ]
        )
    recommended, populations = load_recommended_cholesky(
        factor_path,
        recommendations_path,
    )
    assert populations == {
        "2um": "physical",
        "4um": "inclusive",
        "10um": "physical",
    }
    assert all(np.array_equal(value, np.eye(4)) for value in recommended.values())


def test_full_validation_flags_large_correlation_delta() -> None:
    budgets = {"2um": 1299, "4um": 3538, "10um": 754}
    records, _ = generate_parameters(
        _targets(),
        _factors(),
        seed=31,
        budgets=budgets,
        dataset_id="synthetic-test@v2",
    )
    raw, encoded = synthesize_signals(records, seed=32, batch_size=256)
    validation = validate_candidate(
        dataset_id="synthetic-test@v2",
        budgets=budgets,
        records=records,
        raw_signals=raw,
        model_signals=encoded,
        factors=_factors(),
    )
    assert validation["event_count"] == sum(budgets.values())
    assert validation["checks"]["no_sealed_test_access"] is True


def test_validation_accepts_numeric_metadata_reloaded_from_csv() -> None:
    budgets = {"2um": 2, "4um": 2, "10um": 2}
    records, _ = generate_parameters(
        _targets(),
        _factors(),
        seed=51,
        budgets=budgets,
        dataset_id="synthetic-csv-roundtrip@v3",
    )
    raw, encoded = synthesize_signals(records, seed=52, batch_size=3)
    csv_style = [
        {
            key: str(value) if isinstance(value, (float, int)) else value
            for key, value in row.items()
        }
        for row in records
    ]

    validation = validate_candidate(
        dataset_id="synthetic-csv-roundtrip@v3",
        budgets=budgets,
        records=csv_style,
        raw_signals=raw,
        model_signals=encoded,
        factors=_factors(),
    )

    assert all(validation["checks"].values())
