from __future__ import annotations

import numpy as np

from particles2snr.bead_metric_priority import (
    CORE_FEATURE_NAMES,
    CORE_WEIGHTS,
    amplitude_alignment_factor,
    core_feature_matrix,
)
from particles2snr.ssl_realism_audit import SignalRecord


def _record(identifier: str, peak: float) -> SignalRecord:
    return SignalRecord(
        identifier=identifier,
        signal=np.ones(512, dtype=np.float32),
        metadata={"source_group": identifier},
        descriptors={
            "envelope_peak": peak,
            "dominant_frequency_khz": 20.0,
            "duration_25_ms": 0.5,
        },
    )


def test_core_contract_has_three_equal_weights() -> None:
    assert CORE_FEATURE_NAMES == (
        "log_aligned_envelope_peak",
        "dominant_frequency_khz",
        "duration_25_ms",
    )
    assert np.isclose(sum(CORE_WEIGHTS.values()), 1.0)
    assert len(set(CORE_WEIGHTS.values())) == 1
    assert all("snr" not in name.lower() for name in CORE_FEATURE_NAMES)


def test_alignment_is_invariant_to_global_simulation_scale() -> None:
    real = [_record("r1", 2.0), _record("r2", 4.0)]
    simulations = [_record("s1", 10.0), _record("s2", 20.0)]
    scale = amplitude_alignment_factor(real, simulations)
    baseline = core_feature_matrix(simulations, amplitude_scale=scale)

    scaled = [_record("s1", 100.0), _record("s2", 200.0)]
    scaled_factor = amplitude_alignment_factor(real, scaled)
    transformed = core_feature_matrix(scaled, amplitude_scale=scaled_factor)
    np.testing.assert_allclose(baseline, transformed)
