from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt, find_peaks, spectrogram


@dataclass(frozen=True)
class ParticleDetectionConfig:
    """Configuration for class-agnostic, trace-first particle proposals."""

    sampling_frequency_hz: float = 2_000_000.0
    low_freq_hz: float = 7_000.0
    high_freq_hz: float = 80_000.0
    filter_order: int = 4
    stft_nperseg: int = 512
    stft_noverlap: int = 384
    smooth_frames: int = 3
    active_z: float = 3.5
    boundary_z: float = 1.5
    acceptance_z: float = 12.0
    active_min_concentration: float = 0.0
    acceptance_min_concentration: float = 0.08
    cluster_gap_ms: float = 0.128
    boundary_pad_ms: float = 0.04
    min_width_ms: float = 0.06
    max_width_ms: float = 2.0
    frequency_peak_height_frac: float = 0.20
    frequency_peak_prominence_frac: float = 0.08


@dataclass(frozen=True)
class ParticleEventCandidate:
    candidate_index: int
    center_index: int
    event_start: int
    event_end: int
    width_samples: int
    width_ms: float
    robust_energy_z: float
    energy_concentration: float
    dominant_frequency_hz: float
    spectral_bandwidth_hz: float
    spectral_peak_count: int
    repair_overlap: bool
    repair_overlap_fraction: float
    quality: str
    rejection_reason: str


@dataclass(frozen=True)
class ParticleDetectionDiagnostics:
    filtered_signal: np.ndarray
    frequencies_hz: np.ndarray
    frame_centers: np.ndarray
    power_excess: np.ndarray
    frame_energy: np.ndarray
    energy_z: np.ndarray
    concentration: np.ndarray


def config_fingerprint(config: ParticleDetectionConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def ensure_1d_signal(values: np.ndarray) -> np.ndarray:
    signal = np.squeeze(np.asarray(values))
    if signal.ndim != 1:
        raise ValueError(f"Expected one-dimensional signal, got {signal.shape}")
    if signal.size == 0 or not np.all(np.isfinite(signal)):
        raise ValueError("Signal must be non-empty and finite")
    return signal.astype(np.float32, copy=False)


def validate_config(config: ParticleDetectionConfig) -> None:
    if config.sampling_frequency_hz <= 0:
        raise ValueError("sampling_frequency_hz must be positive")
    nyquist = config.sampling_frequency_hz / 2.0
    if not 0 < config.low_freq_hz < config.high_freq_hz < nyquist:
        raise ValueError("detection band must be ordered and below Nyquist")
    if not 0 <= config.stft_noverlap < config.stft_nperseg:
        raise ValueError("stft_noverlap must be smaller than stft_nperseg")
    if config.smooth_frames < 1:
        raise ValueError("smooth_frames must be at least one")
    if config.boundary_z > config.active_z:
        raise ValueError("boundary_z must not exceed active_z")
    if config.min_width_ms < 0 or config.max_width_ms <= config.min_width_ms:
        raise ValueError("event width bounds are invalid")
    if config.cluster_gap_ms < 0 or config.boundary_pad_ms < 0:
        raise ValueError("temporal margins must be non-negative")
    for value in (
        config.active_min_concentration,
        config.acceptance_min_concentration,
        config.frequency_peak_height_frac,
        config.frequency_peak_prominence_frac,
    ):
        if not 0 <= value <= 1:
            raise ValueError("concentration and peak fractions must be in [0, 1]")


def bandpass_particle_signal(
    signal: np.ndarray, config: ParticleDetectionConfig
) -> np.ndarray:
    raw = ensure_1d_signal(signal)
    nyquist = config.sampling_frequency_hz / 2.0
    coefficients = butter(
        config.filter_order,
        [config.low_freq_hz / nyquist, config.high_freq_hz / nyquist],
        btype="band",
    )
    return filtfilt(*coefficients, raw - float(np.median(raw))).astype(np.float32)


def _robust_z(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    median = float(np.median(values))
    mad = 1.4826 * float(np.median(np.abs(values - median)))
    scale = max(mad, np.finfo(np.float64).eps)
    return (values - median) / scale, median, scale


def _group_active(active: np.ndarray, max_gap_frames: int) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    start: int | None = None
    last_active = -1
    gap = 0
    for index, enabled in enumerate(active.astype(bool).tolist()):
        if enabled:
            if start is None:
                start = index
            elif gap > max_gap_frames:
                groups.append((start, last_active))
                start = index
            last_active = index
            gap = 0
        elif start is not None:
            gap += 1
    if start is not None:
        groups.append((start, last_active))
    return groups


def _repair_overlap(
    left: int, right: int, repair_regions: Iterable[tuple[int, int]]
) -> tuple[bool, float]:
    width = max(1, right - left)
    overlap = 0
    for region_left, region_right in repair_regions:
        if region_right <= region_left:
            raise ValueError("repair intervals must have positive width")
        overlap += max(0, min(right, int(region_right)) - max(left, int(region_left)))
    fraction = min(1.0, overlap / width)
    return overlap > 0, float(fraction)


def _spectral_descriptors(
    event_power: np.ndarray,
    frequencies: np.ndarray,
    config: ParticleDetectionConfig,
) -> tuple[float, float, int]:
    total = float(event_power.sum())
    if total <= 0:
        return float("nan"), float("nan"), 0
    normalized = event_power / total
    dominant = float(frequencies[int(np.argmax(event_power))])
    mean = float(np.sum(frequencies * normalized))
    bandwidth = math.sqrt(
        max(0.0, float(np.sum(np.square(frequencies - mean) * normalized)))
    )
    maximum = float(np.max(event_power))
    peaks, _ = find_peaks(
        event_power,
        height=config.frequency_peak_height_frac * maximum,
        prominence=config.frequency_peak_prominence_frac * maximum,
    )
    if peaks.size == 0 and maximum > 0:
        peaks = np.asarray([int(np.argmax(event_power))])
    return dominant, bandwidth, int(peaks.size)


def particle_event_diagnostics(
    signal: np.ndarray, config: ParticleDetectionConfig
) -> ParticleDetectionDiagnostics:
    validate_config(config)
    raw = ensure_1d_signal(signal)
    if raw.size < config.stft_nperseg:
        raise ValueError("signal is shorter than the STFT window")
    filtered = bandpass_particle_signal(raw, config)
    frequencies, times, complex_values = spectrogram(
        filtered - float(np.mean(filtered)),
        fs=config.sampling_frequency_hz,
        nperseg=config.stft_nperseg,
        noverlap=config.stft_noverlap,
        window="hann",
        mode="complex",
    )
    mask = (frequencies >= config.low_freq_hz) & (
        frequencies <= config.high_freq_hz
    )
    if not np.any(mask):
        raise ValueError("detection band contains no STFT bin")
    band_frequencies = frequencies[mask].astype(np.float64)
    power = np.square(np.abs(complex_values[mask])).astype(np.float64)
    baseline = np.percentile(power, 25, axis=1, keepdims=True)
    excess = np.clip(power - baseline, 0.0, None)
    frame_energy = excess.sum(axis=0)
    if config.smooth_frames > 1:
        frame_energy = uniform_filter1d(
            frame_energy, size=config.smooth_frames, mode="nearest"
        )
    energy_z, _median, _scale = _robust_z(frame_energy)
    top_count = min(5, excess.shape[0])
    top_power = np.partition(
        excess, excess.shape[0] - top_count, axis=0
    )[-top_count:].sum(axis=0)
    concentration = top_power / (power.sum(axis=0) + np.finfo(float).eps)
    frame_centers = np.rint(times * config.sampling_frequency_hz).astype(np.int64)
    return ParticleDetectionDiagnostics(
        filtered_signal=filtered,
        frequencies_hz=band_frequencies,
        frame_centers=frame_centers,
        power_excess=excess,
        frame_energy=frame_energy,
        energy_z=energy_z,
        concentration=concentration,
    )


def detect_particle_events(
    signal: np.ndarray,
    config: ParticleDetectionConfig = ParticleDetectionConfig(),
    *,
    repair_regions: Iterable[tuple[int, int]] = (),
) -> tuple[list[ParticleEventCandidate], ParticleDetectionDiagnostics]:
    raw = ensure_1d_signal(signal)
    diagnostics = particle_event_diagnostics(raw, config)
    active = (diagnostics.energy_z >= config.active_z) & (
        diagnostics.concentration >= config.active_min_concentration
    )
    hop = config.stft_nperseg - config.stft_noverlap
    max_gap = max(
        0,
        int(
            round(
                config.cluster_gap_ms
                / 1000.0
                * config.sampling_frequency_hz
                / hop
            )
        ),
    )
    groups = _group_active(active, max_gap)
    pad = int(
        round(config.boundary_pad_ms / 1000.0 * config.sampling_frequency_hz)
    )
    regions = tuple((int(left), int(right)) for left, right in repair_regions)
    candidates: list[ParticleEventCandidate] = []
    for left, right in groups:
        while left > 0 and diagnostics.energy_z[left - 1] >= config.boundary_z:
            left -= 1
        while (
            right < diagnostics.energy_z.size - 1
            and diagnostics.energy_z[right + 1] >= config.boundary_z
        ):
            right += 1
        frames = np.arange(left, right + 1, dtype=np.int64)
        weights = np.maximum(
            diagnostics.frame_energy[frames]
            - float(np.median(diagnostics.frame_energy)),
            0.0,
        )
        centers = diagnostics.frame_centers[frames]
        center = (
            int(round(np.average(centers, weights=weights)))
            if float(weights.sum()) > 0
            else int(round(float(np.mean(centers))))
        )
        event_start = max(0, int(centers[0]) - config.stft_nperseg // 2 - pad)
        event_end = min(
            raw.size,
            int(centers[-1]) + config.stft_nperseg // 2 + pad,
        )
        width = event_end - event_start
        if width <= 0:
            continue
        width_ms = width / config.sampling_frequency_hz * 1000.0
        event_power = diagnostics.power_excess[:, frames].sum(axis=1)
        total_power = float(event_power.sum())
        concentration_bins = min(5, event_power.size)
        concentration = (
            float(np.sort(event_power)[-concentration_bins:].sum() / total_power)
            if total_power > 0
            else 0.0
        )
        robust_z = float(np.max(diagnostics.energy_z[frames]))
        dominant, bandwidth, peak_count = _spectral_descriptors(
            event_power, diagnostics.frequencies_hz, config
        )
        repair_overlap, repair_fraction = _repair_overlap(
            event_start, event_end, regions
        )
        reasons = []
        if robust_z < config.acceptance_z:
            reasons.append("energy_below_acceptance")
        if concentration < config.acceptance_min_concentration:
            reasons.append("concentration_below_acceptance")
        if width_ms < config.min_width_ms:
            reasons.append("width_below_min")
        if width_ms > config.max_width_ms:
            reasons.append("width_above_max")
        candidates.append(
            ParticleEventCandidate(
                candidate_index=len(candidates),
                center_index=center,
                event_start=event_start,
                event_end=event_end,
                width_samples=width,
                width_ms=float(width_ms),
                robust_energy_z=robust_z,
                energy_concentration=concentration,
                dominant_frequency_hz=dominant,
                spectral_bandwidth_hz=bandwidth,
                spectral_peak_count=peak_count,
                repair_overlap=repair_overlap,
                repair_overlap_fraction=repair_fraction,
                quality="retained" if not reasons else "rejected",
                rejection_reason=";".join(reasons),
            )
        )
    return candidates, diagnostics
