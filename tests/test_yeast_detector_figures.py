from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from particles2snr.yeast_detector_figures import (
    count_active_runs,
    plot_activation,
    plot_bandpass,
    plot_detected_events,
    plot_event_support,
    plot_frame_energy,
    plot_frequency_baseline,
    plot_legacy_vs_energy,
    plot_robust_band,
    plot_spectrogram,
    plot_stft_windows,
    plot_trace_overview,
)
from particles2snr.yeast_events import (
    detector_trace,
    review_calibrated_detection_config_v1,
)


@pytest.fixture(scope="module")
def example():
    rng = np.random.default_rng(7)
    length = 16384
    index = np.arange(length, dtype=np.float32)
    time = index / 2_000_000.0
    envelope = np.exp(-0.5 * np.square((index - length // 2) / 420.0))
    signal = (
        envelope * np.sin(2.0 * np.pi * 22_000.0 * time)
        + 0.015 * rng.normal(size=length)
    ).astype(np.float32)
    config = review_calibrated_detection_config_v1()
    return signal, config, detector_trace(signal, config)


def test_every_helper_renders_and_returns_axes(example) -> None:
    signal, config, trace = example
    center = signal.size // 2
    fake_replay = {
        "raw_all": [
            {"start_ms": 3.6, "end_ms": 4.6, "t0_ms": 4.1, "frequency_khz": 22.0,
             "dropped_at": None},
            {"start_ms": 1.0, "end_ms": 2.9, "t0_ms": 1.9, "frequency_khz": 18.0,
             "dropped_at": "width"},
        ]
    }
    produced = [
        plot_trace_overview(signal, config, zoom_center=center),
        plot_event_support(signal, config, event_start=7800, event_end=8600),
        plot_bandpass(signal, trace),
        plot_stft_windows(trace, center=center),
        plot_spectrogram(trace),
        plot_frequency_baseline(trace),
        plot_frame_energy(trace),
        plot_robust_band(trace),
        plot_robust_band(trace, scaled=True),
        plot_legacy_vs_energy(trace, replay=fake_replay, truth_spans_ms=[(3.9, 4.3)]),
        plot_activation(trace),
        plot_activation(trace, z_threshold=2.0),
        plot_robust_band(trace, both=True),
        plot_detected_events(signal, [], config),
    ]
    for axes in produced:
        first = axes.flat[0] if isinstance(axes, np.ndarray) else axes
        assert first.figure is not None
    plt.close("all")


def test_helpers_draw_into_provided_axes_without_new_figure(example) -> None:
    signal, config, trace = example
    figure, axis = plt.subplots()
    open_before = len(plt.get_fignums())
    returned = plot_robust_band(trace, ax=axis)
    assert returned is axis
    assert len(plt.get_fignums()) == open_before
    plt.close("all")


def test_count_active_runs_counts_contiguous_blocks() -> None:
    assert count_active_runs(np.array([False, False, False])) == 0
    assert count_active_runs(np.array([True, True, False, True])) == 2
    assert count_active_runs(np.array([True, False, True, False, True])) == 3
    assert count_active_runs(np.array([True, True, True])) == 1


def test_activation_thresholds_reproduce_the_detector_mask(example) -> None:
    _signal, config, trace = example
    axis = plot_activation(trace)
    expected = int(trace.active.sum())
    # A single z panel: no concentration panel and no separate a[m] strip.
    assert f"{expected} frames of" in axis.get_title()
    plt.close("all")


def test_detected_events_marks_accepted_and_rejected(example) -> None:
    from particles2snr.yeast_events import detect_yeast_events

    signal, config, _trace = example
    events, _reason = detect_yeast_events(signal, config)
    axis = plot_detected_events(signal, events, config, truth_spans_ms=[(3.9, 4.3)])
    kept = sum(1 for event in events if event.quality in {"strict", "medium"})
    assert f"{len(events)} interval(s), {kept} accepted" in axis.get_title()
    plt.close("all")
