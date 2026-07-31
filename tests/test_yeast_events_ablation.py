from __future__ import annotations

import numpy as np

from particles2snr.yeast_events_ablation import (
    TemporalEnergyCandidate,
    TemporalEnergyConfig,
    _deduplicate_candidates,
    detect_temporal_energy_events,
    match_candidate_centers,
    retained_temporal_candidates,
)


def test_temporal_energy_detector_finds_localized_band_limited_burst() -> None:
    rng = np.random.default_rng(7)
    signal = rng.normal(0.0, 0.01, 16384).astype(np.float32)
    start, end = 7200, 9000
    time = np.arange(end - start) / 2_000_000.0
    signal[start:end] += (
        np.sin(2.0 * np.pi * 22_000.0 * time)
        * np.hanning(end - start)
    ).astype(np.float32)

    candidates = detect_temporal_energy_events(signal)

    assert candidates
    strongest = max(candidates, key=lambda item: item.energy_z_max)
    assert start <= strongest.center_index <= end
    assert strongest.energy_z_max > 12.0


def test_temporal_quality_gate_and_center_matching() -> None:
    rng = np.random.default_rng(11)
    signal = rng.normal(0.0, 0.01, 16384).astype(np.float32)
    time = np.arange(1400) / 2_000_000.0
    signal[6000:7400] += (
        np.sin(2.0 * np.pi * 18_000.0 * time) * np.hanning(1400)
    ).astype(np.float32)
    candidates = detect_temporal_energy_events(signal)
    retained = retained_temporal_candidates(candidates, quality_z=12.0)
    assert retained

    matches, current_only, simple_only = match_candidate_centers(
        [6500, 12000],
        [retained[0].center_index, 15000],
        tolerance_samples=1000,
    )
    assert matches == [(0, 0)]
    assert current_only == [1]
    assert simple_only == [1]


def test_width_gate_rejects_overlong_candidate() -> None:
    config = TemporalEnergyConfig(max_width_ms=0.1)
    signal = np.zeros(16384, dtype=np.float32)
    time = np.arange(2000) / 2_000_000.0
    signal[6000:8000] = (
        np.sin(2.0 * np.pi * 20_000.0 * time) * np.hanning(2000)
    ).astype(np.float32)
    candidates = detect_temporal_energy_events(signal, config)
    assert candidates
    assert retained_temporal_candidates(
        candidates,
        quality_z=3.5,
        config=config,
    ) == []


def test_exactly_identical_expanded_candidates_are_deduplicated() -> None:
    first = TemporalEnergyCandidate(
        candidate_index=0,
        center_index=8000,
        event_start=7000,
        event_end=9000,
        width_samples=2000,
        width_ms=1.0,
        energy_z_max=20.0,
    )
    duplicate = TemporalEnergyCandidate(
        candidate_index=1,
        center_index=8000,
        event_start=7000,
        event_end=9000,
        width_samples=2000,
        width_ms=1.0,
        energy_z_max=19.0,
    )

    assert _deduplicate_candidates([first, duplicate]) == [first]
