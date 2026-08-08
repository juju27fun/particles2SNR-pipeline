from __future__ import annotations

import csv
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from particles2snr.z8_cholesky_generation import (
    RAW_LENGTH,
    SAMPLING_FREQUENCY_HZ,
    preprocess_conv1dgap_512,
)
from particles2snr.z8_parameter_analysis import CLASS_ORDER


ASYMMETRY_BOUND = 0.8
PARAMETER_ORDER_5D = (
    "log_amplitude_p0",
    "frequency_khz",
    "log_tau_ms",
    "snr_db",
    "transformed_asymmetry",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_asymmetry_targets(path: Path) -> dict[str, dict[str, float]]:
    rows = read_csv(path)
    targets: dict[str, dict[str, float]] = {}
    for class_name in CLASS_ORDER:
        selected = [row for row in rows if row["class_name"] == class_name]
        if len(selected) != 1:
            raise ValueError(f"expected one asymmetry target for {class_name}")
        row = selected[0]
        target = {
            "mean": float(row["gaussian_mean_transformed"]),
            "sigma": float(row["gaussian_sigma_transformed"]),
            "minimum": float(row["observed_minimum_transformed"]),
            "maximum": float(row["observed_maximum_transformed"]),
        }
        if not all(math.isfinite(value) for value in target.values()):
            raise ValueError(f"non-finite asymmetry target for {class_name}")
        if target["sigma"] <= 0.0 or target["minimum"] >= target["maximum"]:
            raise ValueError(f"invalid asymmetry target for {class_name}")
        targets[class_name] = target
    return targets


def load_regularized_5d_correlations(path: Path) -> dict[str, np.ndarray]:
    rows = read_csv(path)
    output: dict[str, np.ndarray] = {}
    for class_name in CLASS_ORDER:
        selected = [
            row for row in rows
            if row["class_name"] == class_name
            and row["matrix_kind"] == "regularized_pearson"
        ]
        if len(selected) != 25:
            raise ValueError(f"expected 25 matrix cells for {class_name}")
        matrix = np.zeros((5, 5), dtype=np.float64)
        for row in selected:
            i = PARAMETER_ORDER_5D.index(row["row_parameter"])
            j = PARAMETER_ORDER_5D.index(row["column_parameter"])
            matrix[i, j] = float(row["value"])
        if not np.allclose(matrix, matrix.T, atol=1.0e-12):
            raise ValueError(f"asymmetric correlation matrix for {class_name}")
        if not np.allclose(np.diag(matrix), 1.0, atol=1.0e-12):
            raise ValueError(f"invalid correlation diagonal for {class_name}")
        np.linalg.cholesky(matrix)
        output[class_name] = matrix
    return output


def conditional_standard_normal(
    u4: np.ndarray,
    correlation: np.ndarray,
    residual: np.ndarray,
) -> np.ndarray:
    values = np.asarray(u4, dtype=np.float64)
    matrix = np.asarray(correlation, dtype=np.float64)
    epsilon = np.asarray(residual, dtype=np.float64).reshape(-1)
    if values.ndim != 2 or values.shape[1] != 4 or matrix.shape != (5, 5):
        raise ValueError("expected n×4 coordinates and a 5×5 correlation")
    if epsilon.size != values.shape[0]:
        raise ValueError("residual count does not match coordinates")
    xx = matrix[:4, :4]
    ax = matrix[4, :4]
    weights = np.linalg.solve(xx, ax)
    conditional_mean = values @ weights
    variance = float(matrix[4, 4] - ax @ weights)
    if variance <= 0.0 or not math.isfinite(variance):
        raise ValueError("conditional asymmetry variance is not positive")
    return conditional_mean + np.sqrt(variance) * epsilon


def sample_paired_asymmetry(
    rows: Sequence[Mapping[str, Any]],
    *,
    correlations: Mapping[str, np.ndarray],
    targets: Mapping[str, Mapping[str, float]],
    seed: int,
) -> list[dict[str, float | int]]:
    generator = np.random.default_rng(seed)
    output: list[dict[str, float | int] | None] = [None] * len(rows)
    for class_name in CLASS_ORDER:
        indices = [index for index, row in enumerate(rows) if str(row["class_name"]) == class_name]
        u4 = np.asarray(
            [[float(rows[index][f"u{column}"]) for column in range(1, 5)] for index in indices],
            dtype=np.float64,
        )
        target = targets[class_name]
        accepted = np.zeros(len(indices), dtype=bool)
        transformed = np.empty(len(indices), dtype=np.float64)
        residuals = np.empty(len(indices), dtype=np.float64)
        attempts = np.zeros(len(indices), dtype=np.int64)
        while not np.all(accepted):
            pending = np.flatnonzero(~accepted)
            epsilon = generator.normal(size=pending.size)
            proposal_u = conditional_standard_normal(
                u4[pending], correlations[class_name], epsilon
            )
            proposal = float(target["mean"]) + float(target["sigma"]) * proposal_u
            valid = (
                np.isfinite(proposal)
                & (proposal >= float(target["minimum"]))
                & (proposal <= float(target["maximum"]))
            )
            attempts[pending] += 1
            if np.max(attempts) > 100_000:
                raise RuntimeError(f"conditional asymmetry rejection stalled for {class_name}")
            chosen = pending[valid]
            transformed[chosen] = proposal[valid]
            residuals[chosen] = epsilon[valid]
            accepted[chosen] = True
        physical = ASYMMETRY_BOUND * np.tanh(transformed)
        for local_index, event_index in enumerate(indices):
            output[event_index] = {
                "z5": float(residuals[local_index]),
                "u5": float((transformed[local_index] - float(target["mean"])) / float(target["sigma"])),
                "transformed_asymmetry": float(transformed[local_index]),
                "waveform_asymmetry": float(physical[local_index]),
                "asymmetry_rejection_attempts": int(attempts[local_index]),
            }
    if any(value is None for value in output):
        raise AssertionError("missing paired asymmetry draw")
    return [value for value in output if value is not None]


def clean_waveforms(
    rows: Sequence[Mapping[str, Any]],
    asymmetry: np.ndarray,
) -> np.ndarray:
    if len(rows) != len(asymmetry):
        raise ValueError("row and asymmetry counts differ")
    time_s = (
        np.arange(RAW_LENGTH, dtype=np.float64) - (RAW_LENGTH - 1) / 2.0
    ) / SAMPLING_FREQUENCY_HZ
    output = np.empty((len(rows), RAW_LENGTH), dtype=np.float64)
    for index, row in enumerate(rows):
        tau_s = float(row["tau_ms"]) / 1000.0
        side_tau = np.where(
            time_s < 0.0,
            tau_s * np.exp(-float(asymmetry[index])),
            tau_s * np.exp(float(asymmetry[index])),
        )
        envelope = float(row["amplitude_p0"]) * np.exp(
            -0.5 * np.square(time_s / side_tau)
        )
        output[index] = envelope * np.cos(
            2.0 * np.pi * float(row["frequency_khz"]) * 1000.0 * time_s
            + float(row["phi_rad"])
        )
    return output


def paired_asymmetric_signals(
    rows: Sequence[Mapping[str, Any]],
    baseline_raw: np.ndarray,
    asymmetry: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    baseline = np.asarray(baseline_raw, dtype=np.float64)
    if baseline.shape != (len(rows), RAW_LENGTH):
        raise ValueError("baseline signals are not aligned")
    symmetric = clean_waveforms(rows, np.zeros(len(rows), dtype=np.float64))
    noise = baseline - symmetric
    noise -= np.mean(noise, axis=1, keepdims=True)
    noise_rms = np.sqrt(np.mean(np.square(noise), axis=1))
    if np.any(noise_rms <= 1.0e-12):
        raise ValueError("recovered carrier has zero RMS")
    asymmetric = clean_waveforms(rows, asymmetry)
    clean_rms = np.sqrt(np.mean(np.square(asymmetric), axis=1))
    requested_snr = np.asarray([float(row["snr_db"]) for row in rows])
    target_noise_rms = clean_rms / np.power(10.0, requested_snr / 20.0)
    scaled_noise = noise * (target_noise_rms / noise_rms)[:, None]
    noisy = (asymmetric + scaled_noise).astype(np.float32)
    achieved = 20.0 * np.log10(
        clean_rms / np.sqrt(np.mean(np.square(scaled_noise), axis=1))
    )
    return noisy, preprocess_conv1dgap_512(noisy), achieved, clean_rms, target_noise_rms
