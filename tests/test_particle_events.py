from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from particles2snr.particle_events import (
    ParticleDetectionConfig,
    _unified_rescue_limits,
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
    assert config_fingerprint(config) == (
        "4633f685aab7e1394831653de4ddbc7be7e5ee155a75edb84e170779838be947"
    )
    assert config_fingerprint(config) != config_fingerprint(replace(config, active_z=4.0))
    assert config_fingerprint(config) != config_fingerprint(
        replace(config, deblend_enabled=True)
    )
    assert config_fingerprint(config) != config_fingerprint(
        replace(config, independent_weak_enabled=True)
    )
    assert config_fingerprint(config) != config_fingerprint(
        replace(config, unified_rescue_enabled=True)
    )
    assert config_fingerprint(config) != config_fingerprint(
        replace(config, boundary_expansion_enabled=False)
    )


def test_boundary_expansion_can_be_disabled_without_changing_event_identity() -> None:
    signal = _signal(8_000)
    expanded, _ = detect_particle_events(signal, _permissive())
    activation_only, _ = detect_particle_events(
        signal,
        replace(_permissive(), boundary_expansion_enabled=False),
    )
    assert len(expanded) == len(activation_only) == 1
    assert expanded[0].quality == activation_only[0].quality == "retained"
    assert expanded[0].event_start <= activation_only[0].event_start
    assert expanded[0].event_end >= activation_only[0].event_end
    assert expanded[0].width_samples > activation_only[0].width_samples
    assert abs(expanded[0].center_index - activation_only[0].center_index) < 256


def test_unified_rescue_interpolates_the_predeclared_physical_limits() -> None:
    config = ParticleDetectionConfig(unified_rescue_enabled=True)
    assert _unified_rescue_limits(7.0, config) == pytest.approx((0.90, 8_000.0))
    assert _unified_rescue_limits(9.5, config) == pytest.approx((0.85, 10_000.0))
    assert _unified_rescue_limits(12.0, config) == pytest.approx((0.80, 12_000.0))


def test_unified_rescue_cannot_compose_with_legacy_rescue_branches() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        detect_particle_events(
            _signal(8_000),
            replace(
                ParticleDetectionConfig(),
                unified_rescue_enabled=True,
                weak_rescue_enabled=True,
            ),
        )


def test_unified_rescue_retains_an_interior_rejected_packet_only_once() -> None:
    config = replace(
        ParticleDetectionConfig(),
        unified_rescue_enabled=True,
        unified_rescue_z=3.5,
        acceptance_z=1e12,
        unified_rescue_low_z_min_concentration=0.0,
        unified_rescue_strong_z_min_concentration=0.0,
        unified_rescue_low_z_max_bandwidth_hz=80_000.0,
        unified_rescue_strong_z_max_bandwidth_hz=80_000.0,
    )
    candidates, _ = detect_particle_events(_signal(8_000), config)
    retained = [candidate for candidate in candidates if candidate.quality == "retained"]
    assert len(retained) == 1
    assert abs(retained[0].center_index - 8_000) < 500


def test_deblending_splits_two_close_energy_peaks_without_changing_default() -> None:
    signal = _signal(4_000, 5_100)
    baseline, _ = detect_particle_events(signal, _permissive())
    deblended, _ = detect_particle_events(
        signal,
        replace(
            _permissive(),
            deblend_enabled=True,
            deblend_min_peak_z=3.0,
            deblend_min_prominence_z=0.5,
            deblend_min_separation_ms=0.25,
        ),
    )
    deblended_retained = [item for item in deblended if item.quality == "retained"]
    assert len(baseline) == 1
    assert baseline[0].rejection_reason == "width_above_max"
    assert len(deblended_retained) == 2
    assert deblended_retained[0].event_end <= deblended_retained[1].event_start
    assert abs(deblended_retained[0].center_index - 4_000) < 500
    assert abs(deblended_retained[1].center_index - 5_100) < 500


def test_weak_rescue_rejects_boundary_artifact() -> None:
    config = replace(
        ParticleDetectionConfig(),
        weak_rescue_enabled=True,
        weak_rescue_z=3.5,
        acceptance_z=1e12,
    )
    interior, _ = detect_particle_events(_signal(8_000), config)
    boundary, _ = detect_particle_events(_signal(100), config)
    assert any(item.quality == "retained" for item in interior)
    assert not any(item.quality == "retained" for item in boundary)


def test_independent_weak_pass_finds_event_below_strong_active_population() -> None:
    rng = np.random.default_rng(1)
    samples = np.arange(16_384)
    signal = rng.normal(0.0, 0.01, samples.size)
    for center, amplitude in ((4_000, 1.0), (11_500, 0.005)):
        envelope = np.exp(-0.5 * np.square((samples - center) / 420.0))
        signal += amplitude * envelope * np.sin(2 * np.pi * 22_000 * samples / FS)
    baseline = ParticleDetectionConfig()
    strong, _ = detect_particle_events(signal.astype(np.float32), baseline)
    weak, _ = detect_particle_events(
        signal.astype(np.float32),
        replace(
            baseline,
            independent_weak_enabled=True,
            independent_weak_active_z=2.0,
            independent_weak_acceptance_z=2.5,
            independent_weak_min_concentration=0.4,
            independent_weak_max_bandwidth_hz=20_000.0,
        ),
    )
    strong_centers = [item.center_index for item in strong if item.quality == "retained"]
    weak_centers = [item.center_index for item in weak if item.quality == "retained"]
    assert len(strong_centers) == 1
    assert len(weak_centers) == 2
    assert any(abs(center - 11_500) < 500 for center in weak_centers)


def test_local_boundary_module_only_extends_within_declared_limit() -> None:
    baseline = replace(_permissive(), max_width_ms=3.0)
    original, _ = detect_particle_events(_signal(8_000), baseline)
    refined, _ = detect_particle_events(
        _signal(8_000),
        replace(
            baseline,
            local_boundary_enabled=True,
            local_boundary_envelope_z=0.5,
            local_boundary_max_extension_ms=0.4,
        ),
    )
    first = next(item for item in original if item.quality == "retained")
    second = next(item for item in refined if item.quality == "retained")
    maximum_extension = int(0.4 / 1000.0 * FS)
    assert second.event_start <= first.event_start
    assert second.event_end >= first.event_end
    assert first.event_start - second.event_start <= maximum_extension
    assert second.event_end - first.event_end <= maximum_extension


def test_adaptive_deblend_requires_the_deblend_stage() -> None:
    with pytest.raises(ValueError, match="requires deblend_enabled"):
        detect_particle_events(
            _signal(8_000),
            replace(_permissive(), adaptive_deblend_enabled=True),
        )


def test_adaptive_deblend_executes_on_paired_peaks() -> None:
    candidates, _ = detect_particle_events(
        _signal(4_000, 5_100),
        replace(
            _permissive(),
            deblend_enabled=True,
            adaptive_deblend_enabled=True,
            adaptive_deblend_min_peak_z=3.0,
            adaptive_deblend_min_prominence_z=0.5,
            adaptive_deblend_min_segment_z=3.0,
            adaptive_deblend_max_valley_ratio=0.95,
        ),
    )
    retained = [item for item in candidates if item.quality == "retained"]
    assert len(retained) == 2


def test_adaptive_deblend_composes_with_legacy_weak_rescue() -> None:
    candidates, _ = detect_particle_events(
        _signal(8_000),
        replace(
            ParticleDetectionConfig(),
            deblend_enabled=True,
            adaptive_deblend_enabled=True,
            weak_rescue_enabled=True,
            acceptance_z=1e12,
        ),
    )
    assert any(item.quality == "retained" for item in candidates)
