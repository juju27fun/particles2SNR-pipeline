from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.optimize import least_squares
from scipy.signal import find_peaks, hilbert, spectrogram


FS = 1_000_000.0
FIT_POINTS = 512


@dataclass(frozen=True)
class BuddingComponentFit:
    amplitude: float
    center_ms: float
    sigma_left_ms: float
    sigma_right_ms: float
    shape: float
    frequency_khz: float
    chirp_khz_per_ms: float
    phase_rad: float
    integrated_envelope: float


@dataclass(frozen=True)
class BuddingModelFit:
    component_count: int
    bic: float
    envelope_residual_fraction: float
    waveform_residual_fraction: float
    components: tuple[BuddingComponentFit, ...]
    fit_start_ms: float
    fit_end_ms: float
    n_fit_points: int

    def to_dict(self) -> dict[str, Any]:
        return {
            **{
                key: value
                for key, value in asdict(self).items()
                if key != "components"
            },
            "components": [asdict(component) for component in self.components],
        }


@dataclass(frozen=True)
class BuddingFitComparison:
    event_id: str
    m1: BuddingModelFit
    m2: BuddingModelFit
    delta_bic_m1_minus_m2: float
    resolvability_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "m1": self.m1.to_dict(),
            "m2": self.m2.to_dict(),
            "delta_bic_m1_minus_m2": self.delta_bic_m1_minus_m2,
            "resolvability_score": self.resolvability_score,
        }


def _component_envelope(time_ms: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    log_amplitude, center, log_left, log_right, shape = parameters
    left = np.exp(log_left)
    right = np.exp(log_right)
    scale = np.where(time_ms < center, left, right)
    distance = np.abs((time_ms - center) / np.maximum(scale, 1.0e-9))
    return np.exp(log_amplitude) * np.exp(-0.5 * np.power(distance, shape))


def _mixture_envelope(
    time_ms: np.ndarray,
    parameters: np.ndarray,
    component_count: int,
) -> np.ndarray:
    values = np.zeros_like(time_ms, dtype=np.float64)
    for index in range(component_count):
        values += _component_envelope(
            time_ms,
            parameters[index * 5 : (index + 1) * 5],
        )
    return values


def _fit_bounds(
    *,
    component_count: int,
    peak: float,
    start_ms: float,
    end_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    lower_component = np.asarray(
        [
            np.log(max(peak * 0.01, 1.0e-8)),
            start_ms,
            np.log(0.015),
            np.log(0.015),
            1.0,
        ],
        dtype=np.float64,
    )
    upper_component = np.asarray(
        [
            np.log(max(peak * 2.5, 1.0e-7)),
            end_ms,
            np.log(0.8),
            np.log(0.8),
            4.0,
        ],
        dtype=np.float64,
    )
    return (
        np.tile(lower_component, component_count),
        np.tile(upper_component, component_count),
    )


def _initial_components(
    time_ms: np.ndarray,
    envelope: np.ndarray,
    component_count: int,
) -> list[np.ndarray]:
    peak = float(np.max(envelope))
    peak_indices, properties = find_peaks(
        envelope,
        prominence=max(peak * 0.04, 1.0e-9),
        distance=max(1, int(round(0.06 / np.median(np.diff(time_ms))))),
    )
    ranked = sorted(
        peak_indices,
        key=lambda index: (
            -float(properties["prominences"][
                int(np.where(peak_indices == index)[0][0])
            ]),
            int(index),
        ),
    )
    dominant_index = int(np.argmax(envelope))
    dominant_center = float(time_ms[dominant_index])
    starts: list[np.ndarray] = []
    if component_count == 1:
        starts.append(
            np.asarray(
                [
                    np.log(peak),
                    dominant_center,
                    np.log(0.18),
                    np.log(0.18),
                    2.0,
                ],
                dtype=np.float64,
            )
        )
        return starts

    if len(ranked) >= 2:
        centers = sorted(float(time_ms[index]) for index in ranked[:2])
        amplitudes = [
            float(envelope[int(np.argmin(np.abs(time_ms - center)))])
            for center in centers
        ]
        starts.append(
            np.concatenate(
                [
                    np.asarray(
                        [np.log(max(amplitude, 1.0e-8)), center, np.log(0.14), np.log(0.14), 2.0]
                    )
                    for amplitude, center in zip(amplitudes, centers, strict=True)
                ]
            )
        )
    for separation_ms in (0.08, 0.18, 0.36):
        for ratio in (0.45, 0.80):
            centers = (
                dominant_center - separation_ms / 2.0,
                dominant_center + separation_ms / 2.0,
            )
            starts.append(
                np.concatenate(
                    [
                        np.asarray(
                            [
                                np.log(max(peak * amplitude_ratio, 1.0e-8)),
                                center,
                                np.log(0.16),
                                np.log(0.16),
                                2.0,
                            ],
                            dtype=np.float64,
                        )
                        for amplitude_ratio, center in zip(
                            (1.0, ratio),
                            centers,
                            strict=True,
                        )
                    ]
                )
            )
    return starts


def _ridge_parameters(
    signal: np.ndarray,
    time_ms: np.ndarray,
    envelope: np.ndarray,
    center_ms: float,
) -> tuple[float, float]:
    frequencies, times, density = spectrogram(
        signal,
        fs=FS,
        nperseg=256,
        noverlap=224,
        nfft=2048,
        mode="psd",
    )
    mask = (frequencies >= 5_000.0) & (frequencies <= 80_000.0)
    band = density[mask]
    band_frequencies = frequencies[mask] / 1000.0
    ridge = band_frequencies[np.argmax(band, axis=0)]
    frame_ms = times * 1000.0
    weights = np.interp(frame_ms, time_ms, envelope, left=0.0, right=0.0)
    weights *= np.max(band, axis=0)
    usable = weights > max(float(np.max(weights)) * 0.08, 1.0e-15)
    if np.count_nonzero(usable) < 2:
        return float(np.median(ridge)), 0.0
    x = frame_ms[usable] - center_ms
    design = np.column_stack((np.ones_like(x), x))
    root_weights = np.sqrt(weights[usable] / max(float(np.max(weights[usable])), 1.0e-15))
    coefficients, *_ = np.linalg.lstsq(
        design * root_weights[:, None],
        ridge[usable] * root_weights,
        rcond=None,
    )
    return float(coefficients[0]), float(np.clip(coefficients[1], -40.0, 40.0))


def reconstruct_budding_fit(
    signal: np.ndarray,
    fit: BuddingModelFit,
    *,
    sampling_frequency_hz: float = FS,
) -> tuple[np.ndarray, list[np.ndarray]]:
    values = np.asarray(signal, dtype=np.float64).squeeze()
    time_ms = np.arange(values.size, dtype=np.float64) / sampling_frequency_hz * 1000.0
    bases = []
    envelopes = []
    for component in fit.components:
        parameters = np.asarray(
            [
                0.0,
                component.center_ms,
                np.log(component.sigma_left_ms),
                np.log(component.sigma_right_ms),
                component.shape,
            ],
            dtype=np.float64,
        )
        envelope = _component_envelope(time_ms, parameters)
        relative = time_ms - component.center_ms
        phase = 2.0 * np.pi * (
            component.frequency_khz * relative
            + 0.5 * component.chirp_khz_per_ms * np.square(relative)
        )
        bases.extend((envelope * np.cos(phase), envelope * np.sin(phase)))
        envelopes.append(envelope)
    design = np.column_stack(bases)
    mask = (time_ms >= fit.fit_start_ms) & (time_ms <= fit.fit_end_ms)
    coefficients, *_ = np.linalg.lstsq(design[mask], values[mask], rcond=None)
    reconstruction = design @ coefficients
    component_signals = [
        design[:, 2 * index : 2 * index + 2]
        @ coefficients[2 * index : 2 * index + 2]
        for index in range(len(fit.components))
    ]
    return reconstruction, component_signals


def fit_budding_model(
    signal: np.ndarray,
    *,
    event_start_index: float,
    event_end_index: float,
    component_count: int,
    sampling_frequency_hz: float = FS,
) -> BuddingModelFit:
    if component_count not in {1, 2}:
        raise ValueError("component_count must be 1 or 2")
    values = np.asarray(signal, dtype=np.float64).squeeze()
    if values.ndim != 1 or values.size != 4096:
        raise ValueError("Expected a 4096-sample signal")
    centered = values - float(np.mean(values))
    time_ms = np.arange(values.size, dtype=np.float64) / sampling_frequency_hz * 1000.0
    pad_samples = int(round(0.25e-3 * sampling_frequency_hz))
    start_index = max(0, int(np.floor(event_start_index)) - pad_samples)
    end_index = min(values.size, int(np.ceil(event_end_index)) + pad_samples)
    if end_index - start_index < 128:
        raise ValueError("Event support is too short for budding fitting")
    fit_indices = np.unique(
        np.linspace(
            start_index,
            end_index - 1,
            min(FIT_POINTS, end_index - start_index),
        ).round().astype(int)
    )
    fit_time = time_ms[fit_indices]
    full_envelope = uniform_filter1d(
        np.abs(hilbert(centered)),
        size=32,
        mode="nearest",
    )
    event_left = max(0, int(np.floor(event_start_index)))
    event_right = min(values.size, int(np.ceil(event_end_index)))
    outside = np.ones(values.size, dtype=bool)
    outside[event_left:event_right] = False
    baseline = float(
        np.median(full_envelope[outside])
        if np.any(outside)
        else np.quantile(full_envelope, 0.25)
    )
    excess_envelope = np.clip(full_envelope - baseline, 0.0, None)
    fit_envelope = excess_envelope[fit_indices]
    peak = float(np.max(fit_envelope))
    lower, upper = _fit_bounds(
        component_count=component_count,
        peak=peak,
        start_ms=float(fit_time[0]),
        end_ms=float(fit_time[-1]),
    )

    best = None
    for initial in _initial_components(fit_time, fit_envelope, component_count):
        candidate = np.clip(initial, lower + 1.0e-8, upper - 1.0e-8)
        result = least_squares(
            lambda parameters: (
                _mixture_envelope(fit_time, parameters, component_count)
                - fit_envelope
            )
            / max(peak, 1.0e-9),
            candidate,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=0.05,
            max_nfev=700,
        )
        residual = _mixture_envelope(
            fit_time,
            result.x,
            component_count,
        ) - fit_envelope
        rss = float(np.sum(np.square(residual)))
        key = (rss, tuple(float(value) for value in result.x))
        if best is None or key < best[0]:
            best = (key, result.x, residual)
    if best is None:
        raise RuntimeError("No budding fit initialization succeeded")
    parameters = best[1]
    residual = best[2]
    rss = max(float(np.sum(np.square(residual))), 1.0e-18)
    bic = float(len(fit_indices) * np.log(rss / len(fit_indices)) + len(parameters) * np.log(len(fit_indices)))
    total_envelope = float(np.sum(np.square(fit_envelope)))
    envelope_residual = float(np.sqrt(rss / max(total_envelope, 1.0e-18)))

    provisional = []
    for index in range(component_count):
        raw = parameters[index * 5 : (index + 1) * 5]
        component_envelope = _component_envelope(fit_time, raw)
        frequency, chirp = _ridge_parameters(
            centered,
            fit_time,
            component_envelope,
            float(raw[1]),
        )
        provisional.append(
            {
                "center_ms": float(raw[1]),
                "sigma_left_ms": float(np.exp(raw[2])),
                "sigma_right_ms": float(np.exp(raw[3])),
                "shape": float(raw[4]),
                "frequency_khz": frequency,
                "chirp_khz_per_ms": chirp,
                "integrated_envelope": float(np.trapezoid(component_envelope, fit_time)),
            }
        )
    provisional.sort(
        key=lambda row: (-row["integrated_envelope"], row["center_ms"])
    )
    placeholder_components = tuple(
        BuddingComponentFit(
            amplitude=1.0,
            phase_rad=0.0,
            **row,
        )
        for row in provisional
    )
    placeholder = BuddingModelFit(
        component_count=component_count,
        bic=bic,
        envelope_residual_fraction=envelope_residual,
        waveform_residual_fraction=0.0,
        components=placeholder_components,
        fit_start_ms=float(fit_time[0]),
        fit_end_ms=float(fit_time[-1]),
        n_fit_points=len(fit_indices),
    )
    reconstruction, component_signals = reconstruct_budding_fit(
        centered,
        placeholder,
        sampling_frequency_hz=sampling_frequency_hz,
    )
    fit_mask = (time_ms >= placeholder.fit_start_ms) & (time_ms <= placeholder.fit_end_ms)
    waveform_rms = float(np.sqrt(np.mean(np.square(centered[fit_mask]))))
    waveform_residual = float(
        np.sqrt(np.mean(np.square(centered[fit_mask] - reconstruction[fit_mask])))
        / max(waveform_rms, 1.0e-12)
    )
    components = []
    for component, component_signal in zip(
        placeholder_components,
        component_signals,
        strict=True,
    ):
        component_rms = float(np.sqrt(np.mean(np.square(component_signal[fit_mask]))))
        basis_phase = np.angle(
            hilbert(component_signal.astype(np.float64))
        )
        center_index = int(
            np.clip(
                round(component.center_ms / 1000.0 * sampling_frequency_hz),
                0,
                values.size - 1,
            )
        )
        components.append(
            BuddingComponentFit(
                **{
                    **asdict(component),
                    "amplitude": component_rms * np.sqrt(2.0),
                    "phase_rad": float(basis_phase[center_index]),
                }
            )
        )
    components.sort(
        key=lambda component: (
            -component.amplitude
            * component.integrated_envelope,
            component.center_ms,
        )
    )
    return BuddingModelFit(
        component_count=component_count,
        bic=bic,
        envelope_residual_fraction=envelope_residual,
        waveform_residual_fraction=waveform_residual,
        components=tuple(components),
        fit_start_ms=placeholder.fit_start_ms,
        fit_end_ms=placeholder.fit_end_ms,
        n_fit_points=placeholder.n_fit_points,
    )


def compare_budding_models(
    event_id: str,
    signal: np.ndarray,
    *,
    event_start_index: float,
    event_end_index: float,
    sampling_frequency_hz: float = FS,
) -> BuddingFitComparison:
    m1 = fit_budding_model(
        signal,
        event_start_index=event_start_index,
        event_end_index=event_end_index,
        component_count=1,
        sampling_frequency_hz=sampling_frequency_hz,
    )
    m2 = fit_budding_model(
        signal,
        event_start_index=event_start_index,
        event_end_index=event_end_index,
        component_count=2,
        sampling_frequency_hz=sampling_frequency_hz,
    )
    first, second = m2.components
    separation = abs(first.center_ms - second.center_ms)
    mean_width = np.mean(
        [
            first.sigma_left_ms,
            first.sigma_right_ms,
            second.sigma_left_ms,
            second.sigma_right_ms,
        ]
    )
    amplitude_ratio = min(first.amplitude, second.amplitude) / max(
        first.amplitude,
        second.amplitude,
        1.0e-12,
    )
    resolvability = float(
        separation / max(float(mean_width), 1.0e-6)
        * amplitude_ratio
        * max(0.0, np.tanh((m1.bic - m2.bic) / 25.0))
    )
    return BuddingFitComparison(
        event_id=event_id,
        m1=m1,
        m2=m2,
        delta_bic_m1_minus_m2=float(m1.bic - m2.bic),
        resolvability_score=resolvability,
    )
