"""Non-regression contract for detector_trace and the refactored detector.

The golden candidate tuples below were captured by running
detect_yeast_events with review_calibrated_detection_config_v1 on the four
manifested review records BEFORE the detector_trace refactor. They pin the
observable behaviour of the detector across the refactor.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.ndimage import uniform_filter1d
from scipy.signal import spectrogram

from particles2snr.yeast_events import (
    YeastDetectionConfig,
    bandpass_yeast_signal,
    detect_yeast_events,
    detector_trace,
    review_calibrated_detection_config_v1,
)


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


def test_trace_matches_legacy_inline_computation_bit_for_bit() -> None:
    config = review_calibrated_detection_config_v1()
    signal = _synthetic_event()
    trace = detector_trace(signal, config)

    filtered = bandpass_yeast_signal(signal, config)
    frequencies, times, complex_values = spectrogram(
        filtered - float(np.mean(filtered)),
        fs=config.sampling_frequency_hz,
        nperseg=config.stft_nperseg,
        noverlap=config.stft_noverlap,
        window="hann",
        mode="complex",
    )
    mask = (frequencies >= config.low_freq_hz) & (frequencies <= config.high_freq_hz)
    band_complex = complex_values[mask]
    power = np.square(np.abs(band_complex)).astype(np.float64)
    baseline = np.percentile(power, 25, axis=1, keepdims=True)
    excess = np.clip(power - baseline, 0.0, None)
    frame_energy = uniform_filter1d(
        excess.sum(axis=0), size=config.smooth_frames, mode="nearest"
    )
    energy_median = float(np.median(frame_energy))
    raw_mad = float(np.median(np.abs(frame_energy - energy_median)))
    energy_scale = max(1.4826 * raw_mad, 1.0e-12)
    energy_z = (frame_energy - energy_median) / energy_scale

    np.testing.assert_array_equal(trace.filtered, filtered)
    np.testing.assert_array_equal(trace.frequencies, frequencies[mask].astype(np.float32))
    np.testing.assert_array_equal(trace.times, times)
    np.testing.assert_array_equal(trace.complex_stft, band_complex)
    np.testing.assert_array_equal(trace.power, power)
    np.testing.assert_array_equal(trace.baseline, baseline)
    np.testing.assert_array_equal(trace.excess, excess)
    np.testing.assert_array_equal(trace.frame_energy, frame_energy)
    np.testing.assert_array_equal(trace.energy_z, energy_z)
    assert trace.energy_median == energy_median
    assert trace.raw_mad == raw_mad
    assert trace.energy_scale == energy_scale


def test_trace_shape_contract_and_mad_scaling() -> None:
    config = review_calibrated_detection_config_v1()
    signal = _synthetic_event()
    trace = detector_trace(signal, config)

    n_bins, n_frames = trace.complex_stft.shape
    assert trace.filtered.shape == (signal.size,)
    assert trace.frequencies.shape == (n_bins,)
    assert trace.times.shape == (n_frames,)
    assert trace.power.shape == (n_bins, n_frames)
    assert trace.baseline.shape == (n_bins, 1)
    assert trace.excess.shape == (n_bins, n_frames)
    assert trace.frame_energy.shape == (n_frames,)
    assert trace.energy_z.shape == (n_frames,)
    assert trace.concentration.shape == (n_frames,)
    assert trace.active.shape == (n_frames,)
    assert trace.active.dtype == np.bool_
    assert trace.hop == config.stft_nperseg - config.stft_noverlap
    assert trace.energy_scale == max(1.4826 * trace.raw_mad, 1.0e-12)
    np.testing.assert_array_equal(
        trace.active,
        (trace.energy_z >= config.active_snr_z)
        & (trace.concentration >= config.medium_min_concentration),
    )


def test_trace_rejects_signal_shorter_than_one_window() -> None:
    config = review_calibrated_detection_config_v1()
    with pytest.raises(ValueError):
        detector_trace(np.zeros(config.stft_nperseg - 1, dtype=np.float32), config)


def test_detector_reason_codes_are_preserved() -> None:
    config = review_calibrated_detection_config_v1()
    _, reason = detect_yeast_events(np.zeros(16, dtype=np.float32), config)
    assert reason == "signal_too_short"
    # 2 MHz / 512 puts STFT bins every 3906.25 Hz; this band contains none.
    narrow = YeastDetectionConfig(low_freq_hz=100_100.0, high_freq_hz=101_000.0)
    _, reason = detect_yeast_events(_synthetic_event(), narrow)
    assert reason == "detection_band_empty"


GOLDEN_CANDIDATES = {
    "9459e76ce29342debc90:00": [
        (0, 8236, 6704, 9296, 2592, 1.296, 75.53513989790409, 0.9654530873826674,
         0.3420297261327505, 2, 11718.75, 19531.25, 11718.75, "strict", ""),
    ],
    "214f4ce4967af98a954c:00": [
        (0, 5005, 4144, 5968, 1824, 0.912, 137.88375771241945, 0.9779900552479162,
         0.9680505990982056, 1, 7812.5, 7812.5, 7812.5, "strict", ""),
    ],
    "e1b4603f8b9de6204003:02": [
        (0, 1926, 1328, 2512, 1184, 0.592, 5.066511954329469, 0.8635289797443678,
         0.46220457553863525, 2, 11718.75, 19531.25, 11718.75, "reject",
         "quality_below_threshold"),
        (1, 4682, 4144, 5200, 1056, 0.528, 4.6372648854562595, 0.8059275862999495,
         0.8556122779846191, 1, 15625.0, 15625.0, 15625.0, "reject",
         "quality_below_threshold"),
        (2, 11783, 10416, 13136, 2720, 1.36, 429.0063412603064, 0.9821539563930546,
         0.749121367931366, 1, 11718.75, 11718.75, 11718.75, "strict", ""),
        (3, 14932, 14384, 15568, 1184, 0.592, 19.570400710359653, 0.9423069632398359,
         0.9991697072982788, 1, 19531.25, 19531.25, 19531.25, "strict", ""),
    ],
    "09f788a7473797b794f6:01": [
        (0, 11189, 10288, 12112, 1824, 0.912, 87.9829139571677, 0.9723372135933122,
         0.7926072478294373, 1, 15625.0, 15625.0, 15625.0, "strict", ""),
        (1, 12486, 11952, 13008, 1056, 0.528, 3.899382774331652, 0.8522155977636485,
         0.9930859804153442, 1, 11718.75, 11718.75, 11718.75, "reject",
         "quality_below_threshold"),
    ],
}


def _load_workspace_or_skip():
    try:
        from internship_workspace.config import Workspace

        workspace = Workspace.load()
    except Exception:  # pragma: no cover - depends on the host checkout
        pytest.skip("internship workspace unavailable")
    from particles2snr.yeast_review_records import review_queue_paths

    if not all(path.is_file() for path in review_queue_paths(workspace)):
        pytest.skip("manifested review queues unavailable")
    return workspace


def _assert_close(actual: float, expected: float) -> None:
    if math.isnan(expected):
        assert math.isnan(actual)
    else:
        assert actual == pytest.approx(expected, rel=1.0e-9)


def test_detector_output_matches_pre_refactor_golden_records() -> None:
    workspace = _load_workspace_or_skip()
    from particles2snr.yeast_review_records import load_reviewed_event

    config = review_calibrated_detection_config_v1()
    for event_id, expected_rows in GOLDEN_CANDIDATES.items():
        record = load_reviewed_event(workspace, event_id)
        assert record.event_id == event_id
        candidates, reason = detect_yeast_events(record.signal, config)
        assert reason == ""
        assert len(candidates) == len(expected_rows)
        for candidate, expected in zip(candidates, expected_rows):
            assert candidate.candidate_index == expected[0]
            assert candidate.center_index == expected[1]
            assert candidate.event_start == expected[2]
            assert candidate.event_end == expected[3]
            assert candidate.width_samples == expected[4]
            _assert_close(candidate.width_ms, expected[5])
            _assert_close(candidate.snr_proxy, expected[6])
            _assert_close(candidate.energy_concentration, expected[7])
            _assert_close(candidate.phase_coherence, expected[8])
            assert candidate.n_doppler_peaks == expected[9]
            _assert_close(candidate.doppler_low_hz, expected[10])
            _assert_close(candidate.doppler_high_hz, expected[11])
            _assert_close(candidate.doppler_peak_hz, expected[12])
            assert candidate.quality == expected[13]
            assert candidate.rejection_reason == expected[14]
