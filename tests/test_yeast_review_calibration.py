from __future__ import annotations

from types import SimpleNamespace

import pytest

from particles2snr.yeast_review_calibration import (
    CalibrationSpec,
    accepted_count,
    calibration_spec_from_row,
    count_proxy_metrics,
    select_development_variant,
)


def candidate(snr: float, width: float = 0.8, concentration: float = 0.5):
    return SimpleNamespace(
        snr_proxy=snr,
        width_ms=width,
        energy_concentration=concentration,
    )


def test_accepted_count_caps_before_applying_acceptance() -> None:
    spec = CalibrationSpec(1.5, 0.25, 5.0, 1.6, 2)
    candidates = [candidate(10.0, width=2.0), candidate(8.0), candidate(7.0)]
    assert accepted_count(candidates, spec) == 1


def test_count_proxy_metrics() -> None:
    metrics = count_proxy_metrics([2, 0, 3], [1, 1, 3])
    assert metrics["true_positive_count_proxy"] == 4
    assert metrics["false_positive_count_proxy"] == 1
    assert metrics["false_negative_count_proxy"] == 1
    assert metrics["precision_count_proxy"] == pytest.approx(0.8)
    assert metrics["recall_count_proxy"] == pytest.approx(0.8)
    assert metrics["exact_count_fraction"] == pytest.approx(1 / 3)


def test_select_development_variant_respects_both_gates() -> None:
    failing = {
        "precision_count_proxy": 0.95,
        "recall_count_proxy": 0.70,
        "f1_count_proxy": 0.81,
        "exact_count_fraction": 0.5,
        "mean_absolute_count_error": 0.5,
        "acceptance_snr_z": 8.0,
    }
    passing = {
        "precision_count_proxy": 0.91,
        "recall_count_proxy": 0.86,
        "f1_count_proxy": 0.884,
        "exact_count_fraction": 0.7,
        "mean_absolute_count_error": 0.3,
        "acceptance_snr_z": 5.0,
    }
    assert select_development_variant([failing, passing]) == passing


def test_calibration_spec_from_row() -> None:
    row = {
        "boundary_snr_z": 2.0,
        "cluster_gap_ms": 0.128,
        "acceptance_snr_z": 12.0,
        "maximum_width_ms": 2.0,
        "maximum_events": 5,
        "minimum_concentration": 0.08,
    }
    assert calibration_spec_from_row(row) == CalibrationSpec(2.0, 0.128, 12.0, 2.0, 5)
