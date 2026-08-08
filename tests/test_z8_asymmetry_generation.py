from __future__ import annotations

import numpy as np
import pytest

from particles2snr.z8_asymmetry_generation import (
    conditional_standard_normal,
    paired_asymmetric_signals,
    sample_paired_asymmetry,
)


def test_conditional_standard_normal_matches_independent_case() -> None:
    coordinates = np.zeros((3, 4))
    residual = np.asarray([-1.0, 0.0, 1.0])

    result = conditional_standard_normal(coordinates, np.eye(5), residual)

    assert np.array_equal(result, residual)


def test_sample_paired_asymmetry_is_deterministic_and_bounded() -> None:
    rows = [
        {"class_name": class_name, **{f"u{i}": 0.0 for i in range(1, 5)}}
        for class_name in ("2um", "4um", "10um")
        for _ in range(4)
    ]
    correlations = {class_name: np.eye(5) for class_name in ("2um", "4um", "10um")}
    targets = {
        class_name: {"mean": 0.0, "sigma": 0.2, "minimum": -0.5, "maximum": 0.5}
        for class_name in correlations
    }

    first = sample_paired_asymmetry(rows, correlations=correlations, targets=targets, seed=42)
    second = sample_paired_asymmetry(rows, correlations=correlations, targets=targets, seed=42)

    assert first == second
    assert all(-0.8 < float(row["waveform_asymmetry"]) < 0.8 for row in first)


def test_paired_signal_keeps_requested_snr() -> None:
    rows = [{
        "amplitude_p0": 0.4,
        "frequency_khz": 20.0,
        "tau_ms": 0.18,
        "snr_db": 3.0,
        "phi_rad": 0.4,
    }]
    generator = np.random.default_rng(19)
    baseline = generator.normal(0.0, 0.02, size=(1, 4096)).astype(np.float32)

    raw, model, achieved, clean_rms, noise_rms = paired_asymmetric_signals(
        rows, baseline, np.asarray([0.3])
    )

    assert raw.shape == (1, 4096)
    assert model.shape == (1, 512)
    assert achieved[0] == pytest.approx(3.0, abs=1.0e-10)
    assert clean_rms[0] > 0.0 and noise_rms[0] > 0.0
    assert np.isfinite(raw).all() and np.isfinite(model).all()
