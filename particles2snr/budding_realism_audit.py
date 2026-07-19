from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance

from .ssl_realism_audit import signal_descriptors
from .yeast_events import (
    detect_yeast_events,
    review_calibrated_detection_config_v1,
)


FEATURES = (
    "duration_25_ms",
    "envelope_concentration",
    "dominant_frequency_khz",
    "spectral_bandwidth_khz",
    "temporal_peak_count",
    "spectral_peak_count",
    "rms",
)
FEATURE_LABELS = {
    "duration_25_ms": "Durée à 25 %",
    "envelope_concentration": "Concentration 50/25 %",
    "dominant_frequency_khz": "Fréquence dominante",
    "spectral_bandwidth_khz": "Largeur spectrale",
    "temporal_peak_count": "Pics temporels",
    "spectral_peak_count": "Pics spectraux",
    "rms": "RMS",
}
FEATURE_UNITS = {
    "duration_25_ms": "ms",
    "envelope_concentration": "",
    "dominant_frequency_khz": "kHz",
    "spectral_bandwidth_khz": "kHz",
    "temporal_peak_count": "",
    "spectral_peak_count": "",
    "rms": "",
}
FEATURE_WEIGHTS = np.asarray(
    [0.20, 0.10, 0.20, 0.15, 0.10, 0.10, 0.15],
    dtype=np.float64,
)


def descriptor_matrix(signals: np.ndarray) -> np.ndarray:
    values = np.asarray(signals)
    if values.ndim != 2:
        raise ValueError("signals must have shape (n_signals, n_samples)")
    return np.asarray(
        [
            [signal_descriptors(signal)[name] for name in FEATURES]
            for signal in values
        ],
        dtype=np.float64,
    )


def robust_feature_contract(
    train_descriptors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(train_descriptors, dtype=np.float64)
    center = np.median(values, axis=0)
    scale = np.quantile(values, 0.75, axis=0) - np.quantile(
        values,
        0.25,
        axis=0,
    )
    floors = np.asarray(
        [
            1.0 if name in {"temporal_peak_count", "spectral_peak_count"} else 1.0e-6
            for name in FEATURES
        ]
    )
    return center, np.maximum(scale, floors)


def scaled_descriptors(
    descriptors: np.ndarray,
    *,
    center: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    return (np.asarray(descriptors, dtype=np.float64) - center) / scale


def stable_equalized_rows(
    rows: list[dict[str, str]],
    *,
    split: str,
    count: int,
    component_count: int | None = None,
) -> list[dict[str, str]]:
    eligible = [
        row
        for row in rows
        if row["split"] == split
        and (
            component_count is None
            or int(row["component_count"]) == component_count
        )
    ]
    eligible.sort(
        key=lambda row: (
            hashlib.sha256(
                f"{row['latent_id']}:{row['view_index']}".encode("utf-8")
            ).hexdigest(),
            int(row["signal_row"]),
        )
    )
    if len(eligible) < count:
        raise ValueError(
            f"Only {len(eligible)} eligible rows for requested count {count}"
        )
    return eligible[:count]


def aggregate_metrics(
    real: np.ndarray,
    simulated: np.ndarray,
    *,
    center: np.ndarray,
    scale: np.ndarray,
) -> dict[str, float]:
    x = scaled_descriptors(real, center=center, scale=scale)
    y = scaled_descriptors(simulated, center=center, scale=scale)
    weighted_x = x * np.sqrt(FEATURE_WEIGHTS)
    weighted_y = y * np.sqrt(FEATURE_WEIGHTS)
    cross = cdist(weighted_x, weighted_y)

    def safe_correlation(values: np.ndarray) -> np.ndarray:
        correlation = np.eye(values.shape[1], dtype=np.float64)
        variable = np.std(values, axis=0) > 1.0e-12
        if np.count_nonzero(variable) > 1:
            correlation[np.ix_(variable, variable)] = np.corrcoef(
                values[:, variable],
                rowvar=False,
            )
        return correlation

    return {
        "joint_energy_distance": (
            2.0 * float(np.mean(cross))
            - float(np.mean(cdist(weighted_x, weighted_x)))
            - float(np.mean(cdist(weighted_y, weighted_y)))
        ),
        "median_real_to_sim_nearest_distance": float(
            np.median(np.min(cross, axis=1))
        ),
        "mean_normalized_marginal_wasserstein": float(
            np.sum(
                FEATURE_WEIGHTS
                * np.asarray(
                    [
                        wasserstein_distance(x[:, index], y[:, index])
                        for index in range(x.shape[1])
                    ]
                )
            )
        ),
        "correlation_frobenius": float(
            np.linalg.norm(
                safe_correlation(x) - safe_correlation(y),
                ord="fro",
            )
            / x.shape[1]
        ),
    }


def exact_nearest_neighbor(
    real_descriptor: np.ndarray,
    simulated_descriptors: np.ndarray,
    *,
    center: np.ndarray,
    scale: np.ndarray,
) -> tuple[int, float]:
    real = scaled_descriptors(
        np.asarray(real_descriptor)[None, :],
        center=center,
        scale=scale,
    )
    simulated = scaled_descriptors(
        simulated_descriptors,
        center=center,
        scale=scale,
    )
    distances = np.sqrt(
        np.sum(
            FEATURE_WEIGHTS
            * np.square(simulated - real),
            axis=1,
        )
    )
    index = int(np.argmin(distances))
    return index, float(distances[index])


def deterministic_blind_order(
    case_id: str,
    source_names: list[str],
) -> list[str]:
    return sorted(
        source_names,
        key=lambda source: hashlib.sha256(
            f"budding-blind-v1:{case_id}:{source}".encode("utf-8")
        ).hexdigest(),
    )


def representation_detector_acceptance(signals: np.ndarray) -> float:
    """Apply the frozen v8 timing/frequency contract in 1 MHz representation space.

    The original detector consumes 2 MHz raw traces. All comparison datasets
    here are already band-passed and downsampled to 1 MHz, so STFT sizes and
    overlap are halved while all physical thresholds remain unchanged.
    """
    config = review_calibrated_detection_config_v1()
    adapted = replace(
        config,
        sampling_frequency_hz=1_000_000.0,
        stft_nperseg=config.stft_nperseg // 2,
        stft_noverlap=config.stft_noverlap // 2,
    )
    accepted = 0
    for signal in np.asarray(signals):
        candidates, _reason = detect_yeast_events(signal, adapted)
        if any(candidate.quality == "strict" for candidate in candidates):
            accepted += 1
    return accepted / max(len(signals), 1)


def feature_comparison(
    real: np.ndarray,
    simulated: np.ndarray,
) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "label": FEATURE_LABELS[name],
            "unit": FEATURE_UNITS[name],
            "real": float(real[index]),
            "simulated": float(simulated[index]),
            "delta": float(simulated[index] - real[index]),
            "weight": float(FEATURE_WEIGHTS[index]),
        }
        for index, name in enumerate(FEATURES)
    ]
