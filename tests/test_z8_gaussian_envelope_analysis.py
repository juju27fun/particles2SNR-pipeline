from __future__ import annotations

import numpy as np

from particles2snr.z8_gaussian_envelope_analysis import (
    analyze_gaussian_intensity_envelopes,
    optimize_gaussian_intensity_envelope,
)


def _row(
    *,
    class_name: str,
    physical_source_class: str,
    index: int,
) -> dict[str, str]:
    offset = {"2um": 0.0, "4um": 0.25, "10um": 0.5}[physical_source_class]
    return {
        "class_name": class_name,
        "physical_source_class": physical_source_class,
        "particles2snr_amplitude": str(
            np.exp(-2.2 + offset + 0.035 * index + 0.01 * (index % 3))
        ),
        "frequency_hz": str(
            8_000.0 + 250.0 * index + 400.0 * offset - 25.0 * (index % 4)
        ),
        "tau_ms": str(
            np.exp(-2.0 + 0.02 * index + 0.08 * offset + 0.006 * (index % 5))
        ),
        "snr_db": str(-8.0 + 0.55 * index + offset + 0.1 * (index % 3)),
    }


def _population() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for class_name in ("2um", "4um", "10um"):
        for index in range(30):
            rows.append(
                _row(
                    class_name=class_name,
                    physical_source_class=class_name,
                    index=index,
                )
            )
        for index in range(3):
            row = _row(
                class_name="unclear",
                physical_source_class=class_name,
                index=30 + index,
            )
            row["snr_db"] = str(-16.0 - index)
            rows.append(row)
    return rows


def test_iterative_envelope_has_zero_dense_grid_violations() -> None:
    values = np.concatenate(
        [
            np.linspace(-2.0, -0.8, 80),
            np.linspace(0.1, 0.8, 20),
        ]
    )

    fit = optimize_gaussian_intensity_envelope(
        values,
        transform=lambda array: array,
    )

    assert fit["optimization_success"]
    assert fit["optimization_iterations"] > 0
    assert fit["kde_envelope_violation_count"] == 0
    assert fit["max_real_to_candidate_envelope_ratio"] <= 1.0
    assert fit["required_synthetic_count"] > values.size


def test_complete_analysis_uses_one_budget_per_class_and_covers_all_marginals() -> None:
    analysis = analyze_gaussian_intensity_envelopes(_population())

    assert len(analysis["fits"]) == 12
    assert len(analysis["class_budgets"]) == 3
    assert all(row["class_envelope_valid"] for row in analysis["fits"])
    assert all(row["class_envelope_violation_count"] == 0 for row in analysis["fits"])
    for class_name in ("2um", "4um", "10um"):
        class_rows = [
            row for row in analysis["fits"] if row["class_name"] == class_name
        ]
        budgets = {row["class_synthetic_event_count"] for row in class_rows}
        assert len(budgets) == 1
        assert next(iter(budgets)) == max(
            row["required_synthetic_count"] for row in class_rows
        )


def test_analysis_is_deterministic() -> None:
    rows = _population()

    first = analyze_gaussian_intensity_envelopes(rows)
    second = analyze_gaussian_intensity_envelopes(list(reversed(rows)))

    for first_row, second_row in zip(first["fits"], second["fits"], strict=True):
        assert first_row["class_name"] == second_row["class_name"]
        assert first_row["marginal"] == second_row["marginal"]
        assert np.isclose(
            first_row["gaussian_sigma_transformed"],
            second_row["gaussian_sigma_transformed"],
        )
        assert (
            first_row["class_synthetic_event_count"]
            == second_row["class_synthetic_event_count"]
        )
