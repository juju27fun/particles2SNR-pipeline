from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
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
    # Legacy particle datasets expanded each active run while z stayed above
    # boundary_z. Keep that behaviour as the compatibility default so frozen
    # dataset fingerprints remain reproducible; new datasets can disable it
    # explicitly and retain only the z >= active_z frames plus the fixed pad.
    boundary_expansion_enabled: bool = True
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
    deblend_enabled: bool = False
    deblend_min_separation_ms: float = 0.25
    deblend_min_peak_z: float = 12.0
    deblend_min_prominence_z: float = 3.0
    deblend_min_segment_concentration: float = 0.80
    deblend_max_segment_bandwidth_hz: float = 12_000.0
    wide_rescue_enabled: bool = False
    wide_rescue_boundary_z: float = 2.0
    wide_rescue_min_concentration: float = 0.80
    wide_rescue_max_bandwidth_hz: float = 12_000.0
    weak_rescue_enabled: bool = False
    weak_rescue_z: float = 7.0
    weak_rescue_min_concentration: float = 0.90
    weak_rescue_max_bandwidth_hz: float = 8_000.0
    unified_rescue_enabled: bool = False
    unified_rescue_z: float = 7.0
    unified_rescue_boundary_z: float = 2.0
    unified_rescue_low_z_min_concentration: float = 0.90
    unified_rescue_strong_z_min_concentration: float = 0.80
    unified_rescue_low_z_max_bandwidth_hz: float = 8_000.0
    unified_rescue_strong_z_max_bandwidth_hz: float = 12_000.0
    independent_weak_enabled: bool = False
    independent_weak_active_z: float = 2.5
    independent_weak_acceptance_z: float = 7.0
    independent_weak_min_concentration: float = 0.80
    independent_weak_max_bandwidth_hz: float = 12_000.0
    adaptive_deblend_enabled: bool = False
    adaptive_deblend_min_peak_z: float = 8.0
    adaptive_deblend_min_prominence_z: float = 2.0
    adaptive_deblend_min_segment_z: float = 7.0
    adaptive_deblend_max_valley_ratio: float = 0.80
    local_boundary_enabled: bool = False
    local_boundary_envelope_z: float = 2.0
    local_boundary_smooth_ms: float = 0.02
    local_boundary_max_extension_ms: float = 0.40


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
    values = asdict(config)
    if config.boundary_expansion_enabled:
        # The field did not exist when the reviewed particle configurations
        # were frozen. Omitting the legacy-true value preserves their exact
        # fingerprints while false creates a distinct configuration identity.
        values.pop("boundary_expansion_enabled")
    if not config.deblend_enabled:
        # Preserve the exact fingerprint of the already-reviewed legacy detector.
        for field in (
            "deblend_enabled",
            "deblend_min_separation_ms",
            "deblend_min_peak_z",
            "deblend_min_prominence_z",
            "deblend_min_segment_concentration",
            "deblend_max_segment_bandwidth_hz",
        ):
            values.pop(field)
    if not config.wide_rescue_enabled:
        for field in (
            "wide_rescue_enabled",
            "wide_rescue_boundary_z",
            "wide_rescue_min_concentration",
            "wide_rescue_max_bandwidth_hz",
        ):
            values.pop(field)
    if not config.weak_rescue_enabled:
        for field in (
            "weak_rescue_enabled",
            "weak_rescue_z",
            "weak_rescue_min_concentration",
            "weak_rescue_max_bandwidth_hz",
        ):
            values.pop(field)
    if not config.unified_rescue_enabled:
        for field in (
            "unified_rescue_enabled",
            "unified_rescue_z",
            "unified_rescue_boundary_z",
            "unified_rescue_low_z_min_concentration",
            "unified_rescue_strong_z_min_concentration",
            "unified_rescue_low_z_max_bandwidth_hz",
            "unified_rescue_strong_z_max_bandwidth_hz",
        ):
            values.pop(field)
    if not config.independent_weak_enabled:
        for field in (
            "independent_weak_enabled",
            "independent_weak_active_z",
            "independent_weak_acceptance_z",
            "independent_weak_min_concentration",
            "independent_weak_max_bandwidth_hz",
        ):
            values.pop(field)
    if not config.adaptive_deblend_enabled:
        for field in (
            "adaptive_deblend_enabled",
            "adaptive_deblend_min_peak_z",
            "adaptive_deblend_min_prominence_z",
            "adaptive_deblend_min_segment_z",
            "adaptive_deblend_max_valley_ratio",
        ):
            values.pop(field)
    if not config.local_boundary_enabled:
        for field in (
            "local_boundary_enabled",
            "local_boundary_envelope_z",
            "local_boundary_smooth_ms",
            "local_boundary_max_extension_ms",
        ):
            values.pop(field)
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
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
    if config.deblend_min_separation_ms <= 0:
        raise ValueError("deblend_min_separation_ms must be positive")
    if config.deblend_min_peak_z < config.active_z:
        raise ValueError("deblend_min_peak_z must not be below active_z")
    if config.deblend_min_prominence_z < 0:
        raise ValueError("deblend_min_prominence_z must be non-negative")
    if not 0 <= config.deblend_min_segment_concentration <= 1:
        raise ValueError("deblend_min_segment_concentration must be in [0, 1]")
    if config.deblend_max_segment_bandwidth_hz <= 0:
        raise ValueError("deblend_max_segment_bandwidth_hz must be positive")
    if not config.boundary_z <= config.wide_rescue_boundary_z <= config.active_z:
        raise ValueError("wide_rescue_boundary_z must lie between boundary_z and active_z")
    if not 0 <= config.wide_rescue_min_concentration <= 1:
        raise ValueError("wide_rescue_min_concentration must be in [0, 1]")
    if config.wide_rescue_max_bandwidth_hz <= 0:
        raise ValueError("wide_rescue_max_bandwidth_hz must be positive")
    if config.weak_rescue_enabled and not (
        config.active_z <= config.weak_rescue_z <= config.acceptance_z
    ):
        raise ValueError("weak_rescue_z must lie between active_z and acceptance_z")
    if not 0 <= config.weak_rescue_min_concentration <= 1:
        raise ValueError("weak_rescue_min_concentration must be in [0, 1]")
    if config.weak_rescue_max_bandwidth_hz <= 0:
        raise ValueError("weak_rescue_max_bandwidth_hz must be positive")
    if config.unified_rescue_enabled and any(
        (
            config.wide_rescue_enabled,
            config.weak_rescue_enabled,
            config.independent_weak_enabled,
        )
    ):
        raise ValueError("unified rescue is mutually exclusive with legacy rescue branches")
    if config.unified_rescue_enabled and not (
        config.active_z <= config.unified_rescue_z <= config.acceptance_z
    ):
        raise ValueError("unified_rescue_z must lie between active_z and acceptance_z")
    if not config.boundary_z <= config.unified_rescue_boundary_z <= config.active_z:
        raise ValueError("unified_rescue_boundary_z must lie between boundary_z and active_z")
    if not (
        0
        <= config.unified_rescue_strong_z_min_concentration
        <= config.unified_rescue_low_z_min_concentration
        <= 1
    ):
        raise ValueError("unified rescue concentration limits must be ordered in [0, 1]")
    if not (
        0
        < config.unified_rescue_low_z_max_bandwidth_hz
        <= config.unified_rescue_strong_z_max_bandwidth_hz
    ):
        raise ValueError("unified rescue bandwidth limits must be positive and ordered")
    if config.independent_weak_enabled and not (
        0 < config.independent_weak_active_z
        <= config.independent_weak_acceptance_z
        <= config.acceptance_z
    ):
        raise ValueError(
            "independent weak thresholds must be positive, ordered and no higher than acceptance_z"
        )
    if not 0 <= config.independent_weak_min_concentration <= 1:
        raise ValueError("independent_weak_min_concentration must be in [0, 1]")
    if config.independent_weak_max_bandwidth_hz <= 0:
        raise ValueError("independent_weak_max_bandwidth_hz must be positive")
    if config.adaptive_deblend_enabled and not config.deblend_enabled:
        raise ValueError("adaptive_deblend_enabled requires deblend_enabled")
    if config.adaptive_deblend_min_peak_z < config.active_z:
        raise ValueError("adaptive_deblend_min_peak_z must not be below active_z")
    if config.adaptive_deblend_min_prominence_z < 0:
        raise ValueError("adaptive_deblend_min_prominence_z must be non-negative")
    if config.adaptive_deblend_min_segment_z < config.active_z:
        raise ValueError("adaptive_deblend_min_segment_z must not be below active_z")
    if not 0 <= config.adaptive_deblend_max_valley_ratio <= 1:
        raise ValueError("adaptive_deblend_max_valley_ratio must be in [0, 1]")
    if config.local_boundary_envelope_z < 0:
        raise ValueError("local_boundary_envelope_z must be non-negative")
    if config.local_boundary_smooth_ms <= 0:
        raise ValueError("local_boundary_smooth_ms must be positive")
    if config.local_boundary_max_extension_ms < 0:
        raise ValueError("local_boundary_max_extension_ms must be non-negative")
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


def _expand_group(
    left: int, right: int, energy_z: np.ndarray, boundary_z: float
) -> tuple[int, int]:
    while left > 0 and energy_z[left - 1] >= boundary_z:
        left -= 1
    while right < energy_z.size - 1 and energy_z[right + 1] >= boundary_z:
        right += 1
    return left, right


def _merge_overlapping_groups(
    groups: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for left, right in groups:
        if merged and left <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
        else:
            merged.append((left, right))
    return merged


def _deblend_peaks(
    profile: np.ndarray, config: ParticleDetectionConfig, hop: int
) -> list[int]:
    minimum_distance = max(
        1,
        int(
            round(
                config.deblend_min_separation_ms
                / 1000.0
                * config.sampling_frequency_hz
                / hop
            )
        ),
    )
    peak_z = (
        config.adaptive_deblend_min_peak_z
        if config.adaptive_deblend_enabled
        else config.deblend_min_peak_z
    )
    prominence_z = (
        config.adaptive_deblend_min_prominence_z
        if config.adaptive_deblend_enabled
        else config.deblend_min_prominence_z
    )
    peaks, _properties = find_peaks(
        profile,
        height=peak_z,
        prominence=prominence_z,
        distance=minimum_distance,
    )
    candidates = peaks.astype(int).tolist()
    edge_span = min(profile.size, minimum_distance + 1)
    if (
        profile.size > 1
        and profile[0] >= peak_z
        and profile[0] >= profile[1]
        and profile[0] - float(np.min(profile[:edge_span]))
        >= prominence_z
    ):
        candidates.append(0)
    if (
        profile.size > 1
        and profile[-1] >= peak_z
        and profile[-1] >= profile[-2]
        and profile[-1] - float(np.min(profile[-edge_span:]))
        >= prominence_z
    ):
        candidates.append(profile.size - 1)
    selected: list[int] = []
    for index in sorted(set(candidates), key=lambda value: float(profile[value]), reverse=True):
        if all(abs(index - other) >= minimum_distance for other in selected):
            selected.append(index)
    return sorted(selected)


def _deblended_regions(
    groups: list[tuple[int, int]],
    diagnostics: ParticleDetectionDiagnostics,
    config: ParticleDetectionConfig,
    hop: int,
) -> list[tuple[int, int, int | None, int | None]]:
    expanded = (
        [
            _expand_group(left, right, diagnostics.energy_z, config.boundary_z)
            for left, right in groups
        ]
        if config.boundary_expansion_enabled
        else list(groups)
    )
    merged = _merge_overlapping_groups(expanded)
    regions: list[tuple[int, int, int | None, int | None]] = []
    for left, right in merged:
        profile = diagnostics.energy_z[left : right + 1]
        peaks = [left + value for value in _deblend_peaks(profile, config, hop)]
        if len(peaks) < 2:
            regions.append((left, right, None, None))
            continue
        valleys = [
            int(first + np.argmin(diagnostics.energy_z[first : second + 1]))
            for first, second in zip(peaks, peaks[1:])
        ]
        if config.adaptive_deblend_enabled and any(
            diagnostics.energy_z[valley]
            > config.adaptive_deblend_max_valley_ratio
            * min(diagnostics.energy_z[first], diagnostics.energy_z[second])
            for first, second, valley in zip(
                peaks[:-1], peaks[1:], valleys, strict=True
            )
        ):
            regions.append((left, right, None, None))
            continue
        frame_bounds = [left, *(valley + 1 for valley in valleys), right + 1]
        sample_bounds = [
            int(
                round(
                    (
                        diagnostics.frame_centers[valley]
                        + diagnostics.frame_centers[min(valley + 1, right)]
                    )
                    / 2.0
                )
            )
            for valley in valleys
        ]
        proposed: list[tuple[int, int, int | None, int | None]] = []
        for index, (frame_left, frame_stop) in enumerate(
            zip(frame_bounds, frame_bounds[1:])
        ):
            proposed.append(
                (
                    frame_left,
                    frame_stop - 1,
                    sample_bounds[index - 1] if index > 0 else None,
                    sample_bounds[index] if index < len(sample_bounds) else None,
                )
            )
        if all(
            _deblend_segment_is_particle_like(
                diagnostics, frame_left, frame_right, config
            )
            for frame_left, frame_right, _hard_left, _hard_right in proposed
        ):
            regions.extend(proposed)
        else:
            regions.append((left, right, None, None))
    return regions


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


def _deblend_segment_is_particle_like(
    diagnostics: ParticleDetectionDiagnostics,
    left: int,
    right: int,
    config: ParticleDetectionConfig,
) -> bool:
    frames = np.arange(left, right + 1, dtype=np.int64)
    event_power = diagnostics.power_excess[:, frames].sum(axis=1)
    total_power = float(event_power.sum())
    top_count = min(5, event_power.size)
    concentration = (
        float(np.sort(event_power)[-top_count:].sum() / total_power)
        if total_power > 0
        else 0.0
    )
    _dominant, bandwidth, _peak_count = _spectral_descriptors(
        event_power, diagnostics.frequencies_hz, config
    )
    return (
        float(np.max(diagnostics.energy_z[frames]))
        >= (
            config.adaptive_deblend_min_segment_z
            if config.adaptive_deblend_enabled
            else config.acceptance_z
        )
        and concentration >= config.deblend_min_segment_concentration
        and np.isfinite(bandwidth)
        and bandwidth <= config.deblend_max_segment_bandwidth_hz
    )


def _candidate_matches_region(
    candidate: ParticleEventCandidate, region: ParticleEventCandidate
) -> bool:
    intersection = max(
        0,
        min(candidate.event_end, region.event_end)
        - max(candidate.event_start, region.event_start),
    )
    return intersection > 0 and region.event_start <= candidate.center_index <= region.event_end


def _unified_rescue_limits(
    robust_z: float, config: ParticleDetectionConfig
) -> tuple[float, float]:
    denominator = max(config.acceptance_z - config.unified_rescue_z, np.finfo(float).eps)
    interpolation = float(
        np.clip((robust_z - config.unified_rescue_z) / denominator, 0.0, 1.0)
    )
    concentration = config.unified_rescue_low_z_min_concentration + interpolation * (
        config.unified_rescue_strong_z_min_concentration
        - config.unified_rescue_low_z_min_concentration
    )
    bandwidth = config.unified_rescue_low_z_max_bandwidth_hz + interpolation * (
        config.unified_rescue_strong_z_max_bandwidth_hz
        - config.unified_rescue_low_z_max_bandwidth_hz
    )
    return float(concentration), float(bandwidth)


def _is_duplicate_candidate(
    candidate: ParticleEventCandidate,
    retained: list[ParticleEventCandidate],
    *,
    center_tolerance: int,
) -> bool:
    for existing in retained:
        intersection = max(
            0,
            min(candidate.event_end, existing.event_end)
            - max(candidate.event_start, existing.event_start),
        )
        union = candidate.width_samples + existing.width_samples - intersection
        iou = intersection / union if union > 0 else 0.0
        if iou >= 0.80 and abs(candidate.center_index - existing.center_index) <= center_tolerance:
            return True
    return False


def _local_boundary_extension(
    filtered_signal: np.ndarray,
    event_start: int,
    event_end: int,
    config: ParticleDetectionConfig,
) -> tuple[int, int]:
    smooth_samples = max(
        1,
        int(
            round(
                config.local_boundary_smooth_ms
                / 1000.0
                * config.sampling_frequency_hz
            )
        ),
    )
    envelope = uniform_filter1d(
        np.abs(filtered_signal).astype(np.float64),
        size=smooth_samples,
        mode="nearest",
    )
    envelope_z, _median, _scale = _robust_z(envelope)
    maximum_extension = int(
        round(
            config.local_boundary_max_extension_ms
            / 1000.0
            * config.sampling_frequency_hz
        )
    )
    left_limit = max(0, event_start - maximum_extension)
    right_limit = min(filtered_signal.size, event_end + maximum_extension)
    left = event_start
    right = event_end
    while left > left_limit and envelope_z[left - 1] >= config.local_boundary_envelope_z:
        left -= 1
    while right < right_limit and envelope_z[right] >= config.local_boundary_envelope_z:
        right += 1
    return left, right


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
    if config.deblend_enabled:
        candidate_regions = _deblended_regions(groups, diagnostics, config, hop)
    else:
        candidate_regions = (
            [
                (
                    *_expand_group(
                        left, right, diagnostics.energy_z, config.boundary_z
                    ),
                    None,
                    None,
                )
                for left, right in groups
            ]
            if config.boundary_expansion_enabled
            else [(left, right, None, None) for left, right in groups]
        )
    for left, right, hard_left, hard_right in candidate_regions:
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
        if hard_left is not None:
            event_start = max(event_start, hard_left)
        if hard_right is not None:
            event_end = min(event_end, hard_right)
        if config.local_boundary_enabled:
            event_start, event_end = _local_boundary_extension(
                diagnostics.filtered_signal,
                event_start,
                event_end,
                config,
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
    retained = [candidate for candidate in candidates if candidate.quality == "retained"]
    if config.unified_rescue_enabled:
        trigger_regions = [
            candidate
            for candidate in candidates
            if "energy_below_acceptance" in candidate.rejection_reason
            or "width_above_max" in candidate.rejection_reason
        ]
        if trigger_regions:
            rescue_config = replace(
                config,
                deblend_enabled=False,
                adaptive_deblend_enabled=False,
                local_boundary_enabled=False,
                unified_rescue_enabled=False,
                wide_rescue_enabled=False,
                weak_rescue_enabled=False,
                independent_weak_enabled=False,
                cluster_gap_ms=0.0,
                boundary_z=config.unified_rescue_boundary_z,
                boundary_pad_ms=min(config.boundary_pad_ms, 0.02),
                acceptance_z=config.unified_rescue_z,
                acceptance_min_concentration=min(
                    config.unified_rescue_low_z_min_concentration,
                    config.unified_rescue_strong_z_min_concentration,
                ),
            )
            rescue_candidates, _ = detect_particle_events(
                raw, rescue_config, repair_regions=regions
            )
            for candidate in rescue_candidates:
                minimum_concentration, maximum_bandwidth = _unified_rescue_limits(
                    candidate.robust_energy_z, config
                )
                if (
                    candidate.quality != "retained"
                    or candidate.event_start == 0
                    or candidate.event_end == raw.size
                    or candidate.energy_concentration < minimum_concentration
                    or not np.isfinite(candidate.spectral_bandwidth_hz)
                    or candidate.spectral_bandwidth_hz > maximum_bandwidth
                    or not any(
                        _candidate_matches_region(candidate, region)
                        for region in trigger_regions
                    )
                    or _is_duplicate_candidate(
                        candidate, retained, center_tolerance=hop
                    )
                ):
                    continue
                appended = replace(candidate, candidate_index=len(candidates))
                candidates.append(appended)
                retained.append(appended)
    if config.wide_rescue_enabled:
        wide_regions = [
            candidate
            for candidate in candidates
            if "width_above_max" in candidate.rejection_reason
        ]
        if wide_regions:
            rescue_config = replace(
                config,
                wide_rescue_enabled=False,
                weak_rescue_enabled=False,
                cluster_gap_ms=0.0,
                boundary_z=config.wide_rescue_boundary_z,
                boundary_pad_ms=min(config.boundary_pad_ms, 0.02),
                acceptance_min_concentration=max(
                    config.acceptance_min_concentration,
                    config.wide_rescue_min_concentration,
                ),
            )
            rescue_candidates, _ = detect_particle_events(
                raw, rescue_config, repair_regions=regions
            )
            for candidate in rescue_candidates:
                if (
                    candidate.quality != "retained"
                    or candidate.energy_concentration
                    < config.wide_rescue_min_concentration
                    or not np.isfinite(candidate.spectral_bandwidth_hz)
                    or candidate.spectral_bandwidth_hz
                    > config.wide_rescue_max_bandwidth_hz
                    or not any(
                        _candidate_matches_region(candidate, region)
                        for region in wide_regions
                    )
                    or _is_duplicate_candidate(
                        candidate, retained, center_tolerance=hop
                    )
                ):
                    continue
                appended = replace(candidate, candidate_index=len(candidates))
                candidates.append(appended)
                retained.append(appended)
    if config.weak_rescue_enabled:
        weak_regions = [
            candidate
            for candidate in candidates
            if candidate.rejection_reason == "energy_below_acceptance"
        ]
        if weak_regions:
            rescue_config = replace(
                config,
                deblend_enabled=False,
                adaptive_deblend_enabled=False,
                wide_rescue_enabled=False,
                weak_rescue_enabled=False,
                acceptance_z=config.weak_rescue_z,
            )
            rescue_candidates, _ = detect_particle_events(
                raw, rescue_config, repair_regions=regions
            )
            for candidate in rescue_candidates:
                if (
                    candidate.quality != "retained"
                    or candidate.event_start == 0
                    or candidate.event_end == raw.size
                    or candidate.energy_concentration
                    < config.weak_rescue_min_concentration
                    or not np.isfinite(candidate.spectral_bandwidth_hz)
                    or candidate.spectral_bandwidth_hz
                    > config.weak_rescue_max_bandwidth_hz
                    or not any(
                        _candidate_matches_region(candidate, region)
                        for region in weak_regions
                    )
                    or _is_duplicate_candidate(
                        candidate, retained, center_tolerance=hop
                    )
                ):
                    continue
                appended = replace(candidate, candidate_index=len(candidates))
                candidates.append(appended)
                retained.append(appended)
    if config.independent_weak_enabled:
        weak_config = replace(
            config,
            independent_weak_enabled=False,
            wide_rescue_enabled=False,
            weak_rescue_enabled=False,
            active_z=config.independent_weak_active_z,
            acceptance_z=config.independent_weak_acceptance_z,
            acceptance_min_concentration=max(
                config.acceptance_min_concentration,
                config.independent_weak_min_concentration,
            ),
            cluster_gap_ms=0.0,
        )
        weak_candidates, _ = detect_particle_events(
            raw, weak_config, repair_regions=regions
        )
        for candidate in weak_candidates:
            if (
                candidate.quality != "retained"
                or candidate.event_start == 0
                or candidate.event_end == raw.size
                or candidate.energy_concentration
                < config.independent_weak_min_concentration
                or not np.isfinite(candidate.spectral_bandwidth_hz)
                or candidate.spectral_bandwidth_hz
                > config.independent_weak_max_bandwidth_hz
                or _is_duplicate_candidate(
                    candidate, retained, center_tolerance=hop
                )
            ):
                continue
            appended = replace(candidate, candidate_index=len(candidates))
            candidates.append(appended)
            retained.append(appended)
    return candidates, diagnostics
