from __future__ import annotations

import numpy as np

from particles2snr.particle_class_coverage import (
    CLASS_ORDER,
    FEATURE_NAMES,
    clustered_bootstrap,
    nearest_distances,
    relative_descriptors,
    robust_feature_contract,
)
from particles2snr.ssl_realism_audit import SignalRecord


def _record(identifier: str, offset: float) -> SignalRecord:
    descriptors = {name: 1.0 + offset for name in FEATURE_NAMES}
    descriptors["temporal_peak_count"] = 1.0
    descriptors["spectral_peak_count"] = 1.0
    return SignalRecord(identifier, np.ones(4096), {"source_group": identifier}, descriptors)


def test_relative_descriptors_are_dimensionless_for_amplitude_and_energy() -> None:
    time = np.arange(4096) / 1_000_000.0
    signal = 3.0 * np.sin(2 * np.pi * 20_000.0 * time)
    descriptors = relative_descriptors(signal)
    scaled = relative_descriptors(signal * 7.0)
    assert np.isclose(
        descriptors["envelope_peak_over_rms"],
        scaled["envelope_peak_over_rms"],
    )
    assert np.isclose(
        descriptors["event_energy_fraction"],
        scaled["event_energy_fraction"],
    )


def test_leave_one_out_does_not_select_self() -> None:
    rows = [_record("a", 0.0), _record("b", 1.0), _record("c", 2.0)]
    _, scales = robust_feature_contract(rows)
    result = nearest_distances(rows, rows, scales=scales, exclude_self=True)
    assert np.all(result.nearest_indices != np.arange(3))
    assert result.contributions.shape == (3, len(FEATURE_NAMES))


def test_clustered_bootstrap_returns_all_rate_intervals() -> None:
    distances = np.asarray(
        [[0.1, 0.5, 0.9], [0.2, 0.7, 0.3], [0.8, 0.4, 0.2], [0.6, 0.2, 0.8]]
    )
    loo = {name: np.asarray([0.3, 0.4, 0.5, 0.6]) for name in CLASS_ORDER}
    groups = {name: ["a", "a", "b", "b"] for name in CLASS_ORDER}
    result = clustered_bootstrap(
        simulation_distances=distances,
        real_loo_distances=loo,
        real_groups=groups,
        simulation_groups=["x", "x", "y", "y"],
        repeats=20,
        seed=7,
    )
    assert set(CLASS_ORDER) <= set(result)
    assert 0.0 <= result["any_class"]["low"] <= result["any_class"]["high"] <= 1.0
