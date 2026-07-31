from __future__ import annotations

import numpy as np

from particles2snr.z8_cholesky_analysis import (
    analyze_cholesky_correlations,
    pearson_correlation_matrix,
    regularize_for_cholesky,
    rows_for_population,
    transformed_parameter_matrix,
    validate_inclusive_difference_threshold,
)


def test_inclusive_difference_threshold_must_be_finite_positive() -> None:
    assert validate_inclusive_difference_threshold(0.1) == 0.1
    for value in (0.0, -1.0, float("inf"), float("nan")):
        try:
            validate_inclusive_difference_threshold(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"threshold {value!r} should be rejected")


def _row(
    *,
    event_id: str,
    class_name: str,
    physical_source_class: str,
    source_filename: str,
    index: int,
    unclear_shift: float = 0.0,
) -> dict[str, str]:
    class_offset = {"2um": 0.0, "4um": 0.5, "10um": 1.0}[physical_source_class]
    return {
        "event_id": event_id,
        "class_name": class_name,
        "physical_source_class": physical_source_class,
        "source_filename": source_filename,
        "particles2snr_amplitude": str(
            np.exp(-2.2 + class_offset + 0.035 * index + unclear_shift)
        ),
        "frequency_hz": str(
            9_000.0 + 800.0 * class_offset + 170.0 * index - 11.0 * (index % 3)
        ),
        "tau_ms": str(
            np.exp(-1.4 - 0.018 * index + 0.008 * (index % 4) - unclear_shift)
        ),
        "snr_db": str(-9.0 + 0.65 * index + 0.2 * (index % 5) + 4.0 * unclear_shift),
    }


def _small_population(*, shifted_unclear: bool = False) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for class_name in ("2um", "4um", "10um"):
        for index in range(24):
            rows.append(
                _row(
                    event_id=f"{class_name}-{index}",
                    class_name=class_name,
                    physical_source_class=class_name,
                    source_filename=f"{class_name}-source-{index // 2}",
                    index=index,
                )
            )
        for index in range(3):
            rows.append(
                _row(
                    event_id=f"{class_name}-unclear-{index}",
                    class_name="unclear",
                    physical_source_class=class_name,
                    source_filename=f"{class_name}-unclear-source-{index}",
                    index=24 + index,
                    unclear_shift=2.0 if shifted_unclear and class_name == "10um" else 0.0,
                )
            )
    return rows


def test_transform_uses_natural_logs_and_khz() -> None:
    rows = [
        {
            "particles2snr_amplitude": str(np.e**2),
            "frequency_hz": "12500",
            "tau_ms": str(np.e**-1),
            "snr_db": "-3.5",
        }
        for _ in range(3)
    ]

    matrix = transformed_parameter_matrix(rows)

    np.testing.assert_allclose(matrix[0], [2.0, 12.5, -1.0, -3.5])


def test_population_mapping_keeps_unclear_out_of_physical() -> None:
    rows = _small_population()

    physical = rows_for_population(rows, "4um", "physical")
    inclusive = rows_for_population(rows, "4um", "inclusive")

    assert len(physical) == 24
    assert len(inclusive) == 27
    assert all(row["class_name"] == "4um" for row in physical)
    assert sum(row["class_name"] == "unclear" for row in inclusive) == 3


def test_pearson_matrix_is_symmetric_with_unit_diagonal() -> None:
    rng = np.random.default_rng(20260723)
    matrix = rng.normal(size=(100, 4))

    correlation = pearson_correlation_matrix(matrix)

    np.testing.assert_allclose(correlation, correlation.T)
    np.testing.assert_allclose(np.diag(correlation), np.ones(4))


def test_positive_definite_matrix_is_not_regularized() -> None:
    matrix = np.asarray(
        [
            [1.0, 0.2, 0.1, 0.0],
            [0.2, 1.0, -0.3, 0.1],
            [0.1, -0.3, 1.0, 0.2],
            [0.0, 0.1, 0.2, 1.0],
        ]
    )

    regularized, shrinkage = regularize_for_cholesky(matrix)

    assert shrinkage == 0.0
    np.testing.assert_allclose(regularized, matrix)
    np.linalg.cholesky(regularized)


def test_singular_matrix_gets_minimal_identity_shrinkage() -> None:
    matrix = np.ones((4, 4), dtype=np.float64)

    regularized, shrinkage = regularize_for_cholesky(matrix)

    assert 0.0 < shrinkage < 1.0
    np.testing.assert_allclose(np.diag(regularized), np.ones(4))
    assert np.min(np.linalg.eigvalsh(regularized)) > 0.0
    np.linalg.cholesky(regularized)


def test_analysis_emits_two_matrices_and_one_recommendation_per_class() -> None:
    analysis = analyze_cholesky_correlations(_small_population())

    assert set(analysis["matrices"]) == {"2um", "4um", "10um"}
    assert all(
        set(populations) == {"physical", "inclusive"}
        for populations in analysis["matrices"].values()
    )
    assert len(analysis["correlation_coefficients"]) == 36
    assert len(analysis["coefficient_differences"]) == 18
    assert len(analysis["matrix_diagnostics"]) == 6
    assert len(analysis["cholesky_factors"]) == 60
    assert len(analysis["recommendations"]) == 3


def test_large_unclear_effect_selects_physical_matrix() -> None:
    analysis = analyze_cholesky_correlations(
        _small_population(shifted_unclear=True),
        inclusive_difference_threshold=0.02,
    )
    recommendation = next(
        row for row in analysis["recommendations"] if row["class_name"] == "10um"
    )

    assert recommendation["recommended_dependency_matrix"] == "physical"
    assert "SNR marginal only" in recommendation["unclear_policy"]
