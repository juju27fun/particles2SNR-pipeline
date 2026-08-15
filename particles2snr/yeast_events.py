from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt, find_peaks, spectrogram


@dataclass(frozen=True)
class YeastDetectionConfig:
    sampling_frequency_hz: float = 2_000_000.0
    low_freq_hz: float = 7_000.0
    high_freq_hz: float = 80_000.0
    filter_order: int = 4
    stft_nperseg: int = 512
    stft_noverlap: int = 384
    smooth_frames: int = 3
    active_snr_z: float = 3.5
    boundary_snr_z: float = 1.5
    medium_min_snr: float = 3.0
    strict_min_snr: float = 5.0
    # Frame-level spectral-concentration floor. Set to 0.0 to deactivate the
    # test: it then admits every frame and activation is decided by z alone.
    # Independent of the event-level concentration used at qualification.
    active_min_concentration: float = 0.08
    medium_min_concentration: float = 0.08
    strict_min_concentration: float = 0.12
    strict_min_phase_coherence: float = 0.0
    frequency_peak_height_frac: float = 0.20
    frequency_peak_prominence_frac: float = 0.08
    cluster_gap_ms: float = 0.25
    boundary_pad_ms: float = 0.04
    min_width_ms: float = 0.06
    max_width_ms: float = 1.60
    max_events_per_signal: int = 3


def review_calibrated_detection_config_v1() -> YeastDetectionConfig:
    """Return the detector configuration selected on the completed v5 review.

    This preset is a development result. It must be evaluated on records that
    were not present in the v5 candidate-window or full-trace review queues.

    ``active_min_concentration`` is 0.0: the frame-level concentration test is
    off. An ablation over 1316 traces of yeast-hf-10-5-20260610@v1 and 600
    traces of the bead development corpus produced byte-identical detections
    with and without it, so activation is stated in terms of z alone rather
    than carrying a term that never changes an outcome.
    """
    return YeastDetectionConfig(
        boundary_snr_z=1.5,
        medium_min_snr=12.0,
        strict_min_snr=12.0,
        active_min_concentration=0.0,
        cluster_gap_ms=0.128,
        max_width_ms=2.0,
        max_events_per_signal=5,
    )


@dataclass(frozen=True)
class YeastEventCandidate:
    candidate_index: int
    center_index: int
    event_start: int
    event_end: int
    width_samples: int
    width_ms: float
    snr_proxy: float
    energy_concentration: float
    phase_coherence: float
    n_doppler_peaks: int
    doppler_low_hz: float
    doppler_high_hz: float
    doppler_peak_hz: float
    quality: str
    rejection_reason: str


def ensure_1d_signal(values: np.ndarray) -> np.ndarray:
    signal = np.asarray(values)
    signal = np.squeeze(signal)
    if signal.ndim != 1:
        raise ValueError(f"Expected one-dimensional signal, got {signal.shape}")
    return signal.astype(np.float32, copy=False)


def bandpass_yeast_signal(
    signal: np.ndarray, config: YeastDetectionConfig
) -> np.ndarray:
    """Apply the detector's time-domain preprocessing to a yeast signal."""
    raw = ensure_1d_signal(signal)
    nyquist = config.sampling_frequency_hz / 2.0
    low = config.low_freq_hz / nyquist
    high = config.high_freq_hz / nyquist
    coefficients = butter(config.filter_order, [low, high], btype="band")
    return filtfilt(*coefficients, raw - float(np.mean(raw))).astype(np.float32)


def crop_around_index(signal: np.ndarray, center_index: int, length: int) -> np.ndarray:
    values = ensure_1d_signal(signal)
    start = int(center_index) - int(length) // 2
    end = start + int(length)
    output = np.zeros(int(length), dtype=np.float32)
    source_start = max(0, start)
    source_end = min(values.size, end)
    if source_end > source_start:
        destination_start = source_start - start
        output[destination_start : destination_start + source_end - source_start] = values[
            source_start:source_end
        ]
    return output


class DetectionBandError(ValueError):
    """Raised when the configured detection band contains no STFT bins."""


@dataclass(frozen=True)
class DetectorTrace:
    """Every intermediate stage of detect_yeast_events, in pipeline order.

    ``raw_mad`` is the unscaled median absolute deviation; ``energy_scale`` is
    the Gaussian-consistent scale ``max(1.4826 * raw_mad, 1e-12)`` actually
    used to normalise ``energy_z``. They are exposed separately because both
    have been called "MAD" elsewhere and differ by the 1.4826 factor.
    """

    config: YeastDetectionConfig
    # ISOLATE
    filtered: np.ndarray            # (n_samples,)
    # LOCALISE
    frequencies: np.ndarray         # (n_bins,) restricted to the detection band
    times: np.ndarray               # (n_frames,) seconds
    complex_stft: np.ndarray        # (n_bins, n_frames) complex
    power: np.ndarray               # (n_bins, n_frames) |X|^2
    hop: int                        # stft_nperseg - stft_noverlap
    # REFERENCE
    baseline: np.ndarray            # (n_bins, 1) per-bin Q25
    excess: np.ndarray              # (n_bins, n_frames) positive excess
    # AGGREGATE
    frame_energy: np.ndarray        # (n_frames,) smoothed
    # NORMALISE
    energy_median: float
    raw_mad: float
    energy_scale: float
    energy_z: np.ndarray            # (n_frames,)
    # STRUCTURE
    concentration: np.ndarray       # (n_frames,) top-5 excess / total band power
    # ACTIVATE
    active: np.ndarray              # (n_frames,) bool


def detector_trace(signal: np.ndarray, config: YeastDetectionConfig) -> DetectorTrace:
    """Run the detector front-end and return every intermediate stage."""
    raw = ensure_1d_signal(signal)
    if raw.size < config.stft_nperseg:
        raise ValueError(
            f"signal shorter than one STFT window: {raw.size} < {config.stft_nperseg}"
        )
    filtered = bandpass_yeast_signal(raw, config)
    frequencies, times, complex_values = spectrogram(
        filtered - float(np.mean(filtered)),
        fs=config.sampling_frequency_hz,
        nperseg=config.stft_nperseg,
        noverlap=config.stft_noverlap,
        window="hann",
        mode="complex",
    )
    frequency_mask = (frequencies >= config.low_freq_hz) & (frequencies <= config.high_freq_hz)
    if not np.any(frequency_mask):
        raise DetectionBandError("no STFT bins inside the detection band")
    band_frequencies = frequencies[frequency_mask].astype(np.float32)
    band_complex = complex_values[frequency_mask]
    power = np.square(np.abs(band_complex)).astype(np.float64)
    baseline = np.percentile(power, 25, axis=1, keepdims=True)
    excess = np.clip(power - baseline, 0.0, None)
    frame_energy = excess.sum(axis=0)
    if config.smooth_frames > 1:
        frame_energy = uniform_filter1d(frame_energy, size=config.smooth_frames, mode="nearest")
    energy_median = float(np.median(frame_energy))
    raw_mad = float(np.median(np.abs(frame_energy - energy_median)))
    energy_scale = max(1.4826 * raw_mad, 1.0e-12)
    energy_z = (frame_energy - energy_median) / energy_scale
    top_count = min(5, excess.shape[0])
    top_power = np.partition(excess, excess.shape[0] - top_count, axis=0)[-top_count:].sum(axis=0)
    concentration = top_power / (power.sum(axis=0) + 1.0e-12)
    active = energy_z >= config.active_snr_z
    if config.active_min_concentration > 0.0:
        active = active & (concentration >= config.active_min_concentration)
    return DetectorTrace(
        config=config,
        filtered=filtered,
        frequencies=band_frequencies,
        times=times,
        complex_stft=band_complex,
        power=power,
        hop=config.stft_nperseg - config.stft_noverlap,
        baseline=baseline,
        excess=excess,
        frame_energy=frame_energy,
        energy_median=energy_median,
        raw_mad=raw_mad,
        energy_scale=energy_scale,
        energy_z=energy_z,
        concentration=concentration,
        active=active,
    )


def _group_active_frames(active: np.ndarray, max_gap_frames: int) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    start: int | None = None
    last = -1
    gap = 0
    for index, is_active in enumerate(active.astype(bool).tolist()):
        if is_active:
            if start is None:
                start = index
            elif gap > max_gap_frames:
                groups.append((start, last))
                start = index
            last = index
            gap = 0
        elif start is not None:
            gap += 1
    if start is not None:
        groups.append((start, last))
    return groups


def _expand_bounds(left: int, right: int, energy_z: np.ndarray, threshold: float) -> tuple[int, int]:
    while left > 0 and float(energy_z[left - 1]) >= threshold:
        left -= 1
    while right < energy_z.size - 1 and float(energy_z[right + 1]) >= threshold:
        right += 1
    return left, right


@dataclass(frozen=True)
class EventBounds:
    """The grouping stage, one entry per candidate, in pipeline order."""

    group: tuple[int, int]      # frames selected by activation, gaps bridged
    expanded: tuple[int, int]   # after growing outwards while z >= boundary_snr_z
    start: int                  # sample bound, padding applied
    end: int


def event_bounds(trace: DetectorTrace, n_samples: int) -> list[EventBounds]:
    """Group active frames into candidates and resolve their sample bounds."""
    config = trace.config
    max_gap = max(
        0,
        int(round(config.cluster_gap_ms / 1000.0 * config.sampling_frequency_hz / trace.hop)),
    )
    pad = int(round(config.boundary_pad_ms / 1000.0 * config.sampling_frequency_hz))
    bounds: list[EventBounds] = []
    for group_left, group_right in _group_active_frames(trace.active, max_gap):
        left, right = _expand_bounds(
            group_left, group_right, trace.energy_z, config.boundary_snr_z
        )
        bounds.append(
            EventBounds(
                group=(group_left, group_right),
                expanded=(left, right),
                start=max(0, left * trace.hop - pad),
                end=min(int(n_samples), right * trace.hop + config.stft_nperseg + pad),
            )
        )
    return bounds


def _frequency_peaks(
    power: np.ndarray, frequencies: np.ndarray, config: YeastDetectionConfig
) -> tuple[np.ndarray, np.ndarray]:
    if power.size == 0 or float(np.max(power)) <= 0.0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
    maximum = float(np.max(power))
    peaks, _ = find_peaks(
        power,
        height=config.frequency_peak_height_frac * maximum,
        prominence=config.frequency_peak_prominence_frac * maximum,
    )
    if peaks.size == 0:
        peaks = np.asarray([int(np.argmax(power))], dtype=np.int64)
    peaks = peaks[np.argsort(power[peaks])[::-1]].astype(np.int64, copy=False)
    return peaks, frequencies[peaks].astype(np.float32, copy=False)


def _phase_coherence(complex_stft: np.ndarray, frames: np.ndarray, peaks: np.ndarray) -> float:
    if frames.size < 2 or peaks.size == 0:
        return float("nan")
    selected = peaks[:3]
    values = complex_stft[selected[:, None], frames[None, :]]
    phases = np.unwrap(np.angle(values), axis=1)
    temporal = np.abs(np.mean(np.exp(1j * np.diff(phases, axis=1)), axis=1))
    scores = [float(np.mean(temporal))]
    if selected.size >= 2:
        relative = np.angle(values[0] * np.conj(values[1]))
        scores.append(float(np.abs(np.mean(np.exp(1j * relative)))))
    return float(np.mean(scores))


def _quality(
    *,
    snr_proxy: float,
    concentration: float,
    phase_coherence: float,
    width_ms: float,
    config: YeastDetectionConfig,
) -> tuple[str, str]:
    if width_ms < config.min_width_ms:
        return "reject", "width_below_min"
    if width_ms > config.max_width_ms:
        return "reject", "width_above_max"
    phase_ok = not math.isfinite(phase_coherence) or phase_coherence >= config.strict_min_phase_coherence
    if (
        snr_proxy >= config.strict_min_snr
        and concentration >= config.strict_min_concentration
        and phase_ok
    ):
        return "strict", ""
    if snr_proxy >= config.medium_min_snr and concentration >= config.medium_min_concentration:
        return "medium", ""
    return "reject", "quality_below_threshold"


def detect_yeast_events(
    signal: np.ndarray,
    config: YeastDetectionConfig,
    *,
    review_crop_length: int = 8192,
) -> tuple[list[YeastEventCandidate], str]:
    del review_crop_length  # Crop feasibility is audited separately from event quality.
    raw = ensure_1d_signal(signal)
    if raw.size < config.stft_nperseg:
        return [], "signal_too_short"
    try:
        trace = detector_trace(raw, config)
    except DetectionBandError:
        return [], "detection_band_empty"
    except Exception as exc:
        return [], f"time_frequency_failed:{type(exc).__name__}"

    band_frequencies = trace.frequencies
    band_complex = trace.complex_stft
    excess = trace.excess
    frame_energy = trace.frame_energy
    energy_median = trace.energy_median
    energy_z = trace.energy_z
    hop = trace.hop
    bounds = event_bounds(trace, raw.size)
    if not bounds:
        return [], "no_active_time_frequency_group"

    half_window = config.stft_nperseg // 2
    candidates: list[YeastEventCandidate] = []
    for candidate_bounds in bounds:
        left, right = candidate_bounds.expanded
        frames = np.arange(left, right + 1, dtype=np.int64)
        centers = frames.astype(np.float64) * hop + half_window
        weights = np.maximum(frame_energy[frames] - energy_median, 0.0)
        center = int(round(np.average(centers, weights=weights))) if np.sum(weights) > 0 else int(round(np.mean(centers)))
        event_start = candidate_bounds.start
        event_end = candidate_bounds.end
        width = int(event_end - event_start)
        if width <= 0:
            continue
        width_ms = width / config.sampling_frequency_hz * 1000.0
        event_power = excess[:, frames].sum(axis=1)
        total_power = float(np.sum(event_power))
        if total_power <= 0.0:
            continue
        peaks, peak_frequencies = _frequency_peaks(event_power, band_frequencies, config)
        concentration_bins = min(5, event_power.size)
        concentration = float(np.sort(event_power)[-concentration_bins:].sum() / total_power)
        coherence = _phase_coherence(band_complex, frames, peaks)
        if peaks.size:
            dominant = int(peaks[int(np.argmax(event_power[peaks]))])
            peak = float(band_frequencies[dominant])
            low_peak = float(np.min(peak_frequencies))
            high_peak = float(np.max(peak_frequencies))
        else:
            peak = low_peak = high_peak = float("nan")
        quality, reason = _quality(
            snr_proxy=float(np.max(energy_z[frames])),
            concentration=concentration,
            phase_coherence=coherence,
            width_ms=width_ms,
            config=config,
        )
        candidates.append(
            YeastEventCandidate(
                candidate_index=len(candidates),
                center_index=center,
                event_start=event_start,
                event_end=event_end,
                width_samples=width,
                width_ms=width_ms,
                snr_proxy=float(np.max(energy_z[frames])),
                energy_concentration=concentration,
                phase_coherence=coherence,
                n_doppler_peaks=int(peaks.size),
                doppler_low_hz=low_peak,
                doppler_high_hz=high_peak,
                doppler_peak_hz=peak,
                quality=quality,
                rejection_reason=reason,
            )
        )

    candidates.sort(key=lambda item: item.snr_proxy, reverse=True)
    if config.max_events_per_signal > 0:
        candidates = candidates[: config.max_events_per_signal]
    candidates.sort(key=lambda item: item.center_index)
    return [
        YeastEventCandidate(**{**candidate.__dict__, "candidate_index": index})
        for index, candidate in enumerate(candidates)
    ], ""
