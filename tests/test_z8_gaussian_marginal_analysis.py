from __future__ import annotations

import numpy as np

from particles2snr.z8_gaussian_marginal_analysis import (
    analyze_gaussian_marginals,
    fit_gaussian_marginal,
    rows_for_marginal,
)


def _row(
    *,
    class_name: str,
    physical_source_class: str,
    index: int,
) -> dict[str, str]:
    offset = {"2um": 0.0, "4um": 0.3, "10um": 0.6}[physical_source_class]
    return {
        "class_name": class_name,
        "physical_source_class": physical_source_class,
        "particles2snr_amplitude": str(np.exp(-2.0 + offset + 0.04 * index)),
        "frequency_hz": str(8_000.0 + 300.0 * index + 500.0 * offset),
        "tau_ms": str(np.exp(-1.8 + 0.015 * index + 0.1 * offset)),
        "snr_db": str(-7.0 + 0.5 * index + offset),
    }


def _population() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for class_name in ("2um", "4um", "10um"):
        for index in range(20):
            rows.append(
                _row(
                    class_name=class_name,
                    physical_source_class=class_name,
                    index=index,
                )
            )
        for index in range(2):
            row = _row(
                class_name="unclear",
                physical_source_class=class_name,
                index=20 + index,
            )
            row["snr_db"] = str(-15.0 - index)
            rows.append(row)
    return rows


def test_fit_places_observed_minmax_inside_two_sigma() -> None:
    values = np.asarray([1.0, 1.1, 1.2, 1.4, 2.0, 3.5])

    fit = fit_gaussian_marginal(
        values,
        transform=np.log,
        inverse=np.exp,
    )

    assert fit["observed_range_covered_by_2sigma"]
    assert fit["gaussian_lower_2sigma_transformed"] <= np.log(values.min())
    assert fit["gaussian_upper_2sigma_transformed"] >= np.log(values.max())
    assert fit["gaussian_sigma_transformed"] > 0.0


def test_snr_includes_unclear_but_other_marginals_do_not() -> None:
    rows = _population()

    snr = rows_for_marginal(rows, "4um", "snr_db")
    amplitude = rows_for_marginal(rows, "4um", "amplitude_p0")

    assert len(snr) == 22
    assert len(amplitude) == 20
    assert sum(row["class_name"] == "unclear" for row in snr) == 2
    assert all(row["class_name"] == "4um" for row in amplitude)


def test_complete_analysis_emits_twelve_fits() -> None:
    analysis = analyze_gaussian_marginals(_population())

    assert len(analysis["fits"]) == 12
    assert all(row["observed_range_covered_by_2sigma"] for row in analysis["fits"])
    assert {
        row["population"] for row in analysis["fits"] if row["marginal"] == "snr_db"
    } == {"inclusive"}
    assert all(
        row["gaussian_mass_outside_observed_range"] > 0.0
        for row in analysis["fits"]
    )


def test_frequency_diagnostics_detect_unphysical_mass() -> None:
    analysis = analyze_gaussian_marginals(_population())
    frequency_rows = [
        row for row in analysis["fits"] if row["marginal"] == "frequency_khz"
    ]

    assert all(row["gaussian_mass_below_zero_raw"] > 0.0 for row in frequency_rows)
    assert all(
        row["gaussian_mass_outside_fbase_band"] > 0.0
        for row in frequency_rows
    )
