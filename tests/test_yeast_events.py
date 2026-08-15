from __future__ import annotations

import numpy as np

from particles2snr.yeast_events import (
    YeastDetectionConfig,
    crop_around_index,
    detect_yeast_events,
    review_calibrated_detection_config_v1,
)


def test_review_calibrated_config_v1_matches_frozen_selection() -> None:
    config = review_calibrated_detection_config_v1()
    assert config.medium_min_snr == config.strict_min_snr == 12.0
    assert config.boundary_expansion_enabled is False
    assert config.boundary_snr_z == 1.5  # retained so the review comparison stays runnable
    assert config.cluster_gap_ms == 0.128
    assert config.max_width_ms == 2.0
    assert config.max_events_per_signal == 5


def _synthetic_event(length: int = 16384) -> np.ndarray:
    rng = np.random.default_rng(7)
    index = np.arange(length, dtype=np.float32)
    time = index / 2_000_000.0
    envelope = np.exp(-0.5 * np.square((index - length // 2) / 420.0))
    signal = envelope * (
        np.sin(2.0 * np.pi * 22_000.0 * time)
        + 0.75 * np.sin(2.0 * np.pi * 34_000.0 * time + 0.45)
    )
    return (signal + 0.015 * rng.normal(size=length)).astype(np.float32)


def test_detector_finds_centered_multi_frequency_event() -> None:
    config = YeastDetectionConfig(
        active_snr_z=2.5,
        strict_min_snr=3.0,
        medium_min_snr=2.0,
        strict_min_concentration=0.08,
        medium_min_concentration=0.04,
        min_width_ms=0.05,
        max_width_ms=2.0,
    )
    signal = _synthetic_event()
    events, reason = detect_yeast_events(signal, config)
    usable = [event for event in events if event.quality in {"strict", "medium"}]
    assert reason == ""
    assert len(usable) == 1
    assert abs(usable[0].center_index - signal.size // 2) < 512
    assert usable[0].n_doppler_peaks >= 2


def test_crop_zero_pads_edges() -> None:
    signal = np.arange(5, dtype=np.float32)
    assert crop_around_index(signal, 0, 7).tolist() == [0, 0, 0, 0, 1, 2, 3]


def test_event_quality_does_not_depend_on_review_crop_length() -> None:
    signal = np.roll(_synthetic_event(), -5000)
    config = YeastDetectionConfig(
        active_snr_z=2.5,
        strict_min_snr=3.0,
        medium_min_snr=2.0,
        strict_min_concentration=0.08,
        medium_min_concentration=0.04,
        min_width_ms=0.05,
        max_width_ms=2.0,
    )
    short, _ = detect_yeast_events(signal, config, review_crop_length=4096)
    long, _ = detect_yeast_events(signal, config, review_crop_length=8192)
    assert [(row.center_index, row.quality) for row in short] == [
        (row.center_index, row.quality) for row in long
    ]
    assert any(row.quality in {"strict", "medium"} for row in long)
