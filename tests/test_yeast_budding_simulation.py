from __future__ import annotations

import numpy as np

from particles2snr.yeast_budding_simulation import (
    compare_budding_models,
    fit_budding_model,
    reconstruct_budding_fit,
)


def _passage(
    *,
    center_ms: float,
    sigma_ms: float,
    frequency_khz: float,
    amplitude: float = 1.0,
) -> np.ndarray:
    time_ms = np.arange(4096, dtype=np.float64) / 1000.0
    envelope = np.exp(-0.5 * np.square((time_ms - center_ms) / sigma_ms))
    return amplitude * envelope * np.cos(
        2.0 * np.pi * frequency_khz * (time_ms - center_ms)
    )


def test_single_component_fit_is_finite_and_reconstructable() -> None:
    signal = _passage(center_ms=2.0, sigma_ms=0.18, frequency_khz=18.0)
    fit = fit_budding_model(
        signal,
        event_start_index=1600,
        event_end_index=2400,
        component_count=1,
    )
    reconstruction, components = reconstruct_budding_fit(signal, fit)

    assert fit.component_count == 1
    assert abs(fit.components[0].center_ms - 2.0) < 0.08
    assert abs(fit.components[0].frequency_khz - 18.0) < 1.0
    assert np.isfinite(reconstruction).all()
    assert len(components) == 1


def test_two_component_fit_prefers_separated_passages_deterministically() -> None:
    signal = _passage(
        center_ms=1.75,
        sigma_ms=0.12,
        frequency_khz=14.0,
    ) + _passage(
        center_ms=2.30,
        sigma_ms=0.10,
        frequency_khz=22.0,
        amplitude=0.70,
    )
    first = compare_budding_models(
        "synthetic-double",
        signal,
        event_start_index=1450,
        event_end_index=2550,
    )
    second = compare_budding_models(
        "synthetic-double",
        signal,
        event_start_index=1450,
        event_end_index=2550,
    )

    centers = sorted(component.center_ms for component in first.m2.components)
    assert centers[0] < 1.95
    assert centers[1] > 2.10
    assert first.delta_bic_m1_minus_m2 > 10.0
    assert first.resolvability_score > 0.0
    assert first.to_dict() == second.to_dict()


def test_component_count_contract_is_enforced() -> None:
    signal = np.zeros(4096, dtype=np.float32)
    try:
        fit_budding_model(
            signal,
            event_start_index=1600,
            event_end_index=2400,
            component_count=3,
        )
    except ValueError as error:
        assert "component_count" in str(error)
    else:
        raise AssertionError("component_count=3 must be rejected")
