from __future__ import annotations

from dataclasses import replace

import numpy as np

from particles2snr.particle_events import (
    ParticleDetectionConfig,
    config_fingerprint,
    detect_particle_events,
)


FS = 2_000_000.0


def _signal(*centers: int, amplitude: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(42)
    values = rng.normal(0.0, 0.002, 16_384)
    samples = np.arange(values.size)
    for center in centers:
        envelope = np.exp(-0.5 * np.square((samples - center) / 420.0))
        values += amplitude * envelope * np.sin(2 * np.pi * 22_000 * samples / FS)
    return values.astype(np.float32)


def _permissive() -> ParticleDetectionConfig:
    return ParticleDetectionConfig(
        active_z=2.5,
        acceptance_z=3.0,
        active_min_concentration=0.0,
        acceptance_min_concentration=0.0,
    )


def test_detects_and_localizes_two_physical_events() -> None:
    candidates, diagnostics = detect_particle_events(_signal(4_000, 11_500), _permissive())
    retained = [candidate for candidate in candidates if candidate.quality == "retained"]
    assert len(retained) == 2
    assert abs(retained[0].center_index - 4_000) < 500
    assert abs(retained[1].center_index - 11_500) < 500
    assert all(7_000 <= item.dominant_frequency_hz <= 80_000 for item in retained)
    assert diagnostics.energy_z.shape == diagnostics.frame_centers.shape


def test_mad_detection_is_invariant_to_affine_signal_scale() -> None:
    config = _permissive()
    first, _ = detect_particle_events(_signal(8_000), config)
    second, _ = detect_particle_events(7.5 * _signal(8_000) + 12.0, config)
    assert [(item.event_start, item.event_end, item.quality) for item in first] == [
        (item.event_start, item.event_end, item.quality) for item in second
    ]


def test_repair_overlap_is_reported_but_not_vetoed() -> None:
    candidates, _ = detect_particle_events(
        _signal(8_000), _permissive(), repair_regions=[(7_700, 8_300)]
    )
    retained = [candidate for candidate in candidates if candidate.quality == "retained"]
    assert len(retained) == 1
    assert retained[0].repair_overlap is True
    assert retained[0].repair_overlap_fraction > 0


def test_config_fingerprint_is_stable_and_sensitive() -> None:
    config = ParticleDetectionConfig()
    assert config_fingerprint(config) == config_fingerprint(config)
    assert config_fingerprint(config) != config_fingerprint(replace(config, active_z=4.0))
