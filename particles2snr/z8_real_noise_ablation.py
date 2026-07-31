from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from particles2snr.z8_cholesky_generation import (
    DECIMATION_FACTOR,
    MODEL_LENGTH,
    RAW_LENGTH,
    SAMPLING_FREQUENCY_HZ,
    preprocess_conv1dgap_512,
)
from particles2snr.z8_parameter_analysis import CLASS_ORDER


@dataclass(frozen=True)
class Carrier:
    class_name: str
    split: str
    source_relative_path: str
    start_sample: int
    end_sample: int
    source_round: int
    rms: float
    sha256: str
    values: np.ndarray


@dataclass(frozen=True)
class CarrierRef:
    class_name: str
    split: str
    source_relative_path: str
    start_sample: int
    end_sample: int
    source_round: int
    rms: float


def yolo_blocked_intervals(
    label_path: Path,
    *,
    signal_length: int,
    guard_samples: int,
) -> list[tuple[int, int]]:
    intervals = []
    if not label_path.is_file():
        return intervals
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f"invalid 1-D YOLO label: {label_path}")
        center = float(fields[1]) * signal_length
        width = float(fields[2]) * signal_length
        start = max(0, int(np.floor(center - width / 2.0)) - guard_samples)
        stop = min(
            signal_length,
            int(np.ceil(center + width / 2.0)) + guard_samples,
        )
        intervals.append((start, stop))
    return intervals


def repair_blocked_intervals(
    rows: Sequence[Mapping[str, Any]],
    *,
    signal_length: int,
) -> dict[tuple[str, str], list[tuple[int, int]]]:
    output: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for row in rows:
        radius = int(float(row["filter_response_radius_samples"]))
        start = max(0, int(float(row["expanded_start_sample"])) - radius)
        stop = min(
            signal_length,
            int(float(row["expanded_end_sample"])) + radius,
        )
        output[(str(row["split"]), str(row["filename"]))].append((start, stop))
    return dict(output)


def eligible_window_starts(
    *,
    signal_length: int,
    window_length: int,
    stride: int,
    blocked_intervals: Sequence[tuple[int, int]],
) -> list[int]:
    if signal_length < window_length or window_length <= 0 or stride <= 0:
        return []
    starts = []
    for start in range(0, signal_length - window_length + 1, stride):
        stop = start + window_length
        if all(stop <= left or start >= right for left, right in blocked_intervals):
            starts.append(start)
    return starts


def select_source_windows(
    signal: np.ndarray,
    starts: Sequence[int],
    *,
    window_length: int = RAW_LENGTH,
    maximum_per_source: int = 8,
    minimum_start_separation: int = 1024,
) -> list[tuple[int, np.ndarray, float]]:
    values = np.asarray(signal, dtype=np.float32).reshape(-1)
    candidates = []
    for start in starts:
        crop = values[start : start + window_length].copy()
        centered = crop - float(np.mean(crop))
        rms = float(np.sqrt(np.mean(np.square(centered, dtype=np.float64))))
        if np.isfinite(rms) and rms > 1.0e-12:
            candidates.append((rms, int(start), crop))
    candidates.sort(key=lambda item: (item[0], item[1]))
    selected = []
    for rms, start, crop in candidates:
        if any(
            abs(start - existing_start) < minimum_start_separation
            for existing_start, _, _ in selected
        ):
            continue
        selected.append((start, crop, rms))
        if len(selected) == maximum_per_source:
            break
    return selected


def select_source_window_refs(
    signal: np.ndarray,
    starts: Sequence[int],
    *,
    window_length: int = RAW_LENGTH,
    maximum_per_source: int = 256,
    minimum_start_separation: int = 64,
) -> list[tuple[int, float]]:
    """Select low-RMS windows without retaining their sample arrays."""
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    if values.size < window_length:
        return []
    prefix = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    prefix_square = np.concatenate(
        ([0.0], np.cumsum(np.square(values), dtype=np.float64))
    )
    candidates = []
    for start in starts:
        stop = int(start) + window_length
        total = prefix[stop] - prefix[int(start)]
        total_square = prefix_square[stop] - prefix_square[int(start)]
        mean = total / window_length
        variance = max(total_square / window_length - mean * mean, 0.0)
        rms = float(np.sqrt(variance))
        if np.isfinite(rms) and rms > 1.0e-12:
            candidates.append((rms, int(start)))
    candidates.sort(key=lambda item: (item[0], item[1]))
    selected: list[tuple[int, float]] = []
    for rms, start in candidates:
        if any(
            abs(start - existing_start) < minimum_start_separation
            for existing_start, _ in selected
        ):
            continue
        selected.append((start, rms))
        if len(selected) == maximum_per_source:
            break
    return selected


def carrier_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(values, dtype=np.float32).tobytes()
    ).hexdigest()


def round_robin_carriers(
    carriers: Sequence[Carrier],
    *,
    class_name: str,
    required: int,
    seed: int,
    allow_reuse_after_exhaustion: bool = False,
) -> list[Carrier]:
    by_source: dict[str, list[Carrier]] = defaultdict(list)
    for carrier in carriers:
        if carrier.class_name == class_name:
            by_source[carrier.source_relative_path].append(carrier)
    if not by_source:
        raise ValueError(f"no eligible real-noise carriers for {class_name}")
    for values in by_source.values():
        values.sort(key=lambda item: (item.source_round, item.rms, item.start_sample))
    generator = np.random.default_rng(seed)
    sources = np.asarray(sorted(by_source), dtype=object)
    generator.shuffle(sources)
    selected = []
    round_index = 0
    while len(selected) < required:
        added = 0
        for source in sources:
            values = by_source[str(source)]
            if round_index < len(values):
                selected.append(values[round_index])
                added += 1
                if len(selected) == required:
                    break
        if added == 0:
            if not allow_reuse_after_exhaustion:
                raise ValueError(
                    f"insufficient {class_name} carriers: {len(selected)} < {required}"
                )
            unique = list(selected)
            if not unique:
                raise ValueError(f"no reusable {class_name} carriers")
            while len(selected) < required:
                selected.append(unique[(len(selected) - len(unique)) % len(unique)])
            break
        round_index += 1
    return selected


def reconstruct_clean(records: Sequence[Mapping[str, Any]]) -> np.ndarray:
    time_s = (
        np.arange(RAW_LENGTH, dtype=np.float64) - (RAW_LENGTH - 1) / 2.0
    ) / SAMPLING_FREQUENCY_HZ
    output = np.empty((len(records), RAW_LENGTH), dtype=np.float32)
    for index, row in enumerate(records):
        amplitude = float(row["amplitude_p0"])
        frequency_hz = float(row["frequency_khz"]) * 1000.0
        tau_s = float(row["tau_ms"]) / 1000.0
        phase = float(row["phi_rad"])
        envelope = amplitude * np.exp(-0.5 * np.square(time_s / tau_s))
        output[index] = (
            envelope * np.cos(2.0 * np.pi * frequency_hz * time_s + phase)
        ).astype(np.float32)
    return output


def inject_real_noise(
    records: Sequence[Mapping[str, Any]],
    clean: np.ndarray,
    assigned_carriers: Sequence[Carrier],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    clean_values = np.asarray(clean, dtype=np.float64)
    if clean_values.shape != (len(records), RAW_LENGTH):
        raise ValueError("clean waveform array is not aligned")
    if len(assigned_carriers) != len(records):
        raise ValueError("carrier assignment is not aligned")
    noisy = np.empty_like(clean_values, dtype=np.float32)
    metadata = []
    for index, (row, carrier) in enumerate(
        zip(records, assigned_carriers, strict=True)
    ):
        noise = np.asarray(carrier.values, dtype=np.float64)
        noise -= np.mean(noise)
        carrier_rms = float(np.sqrt(np.mean(np.square(noise))))
        clean_rms = float(np.sqrt(np.mean(np.square(clean_values[index]))))
        requested_snr = float(row["snr_db"])
        target_noise_rms = clean_rms / np.power(10.0, requested_snr / 20.0)
        scaled_noise = noise * (target_noise_rms / carrier_rms)
        noisy[index] = (clean_values[index] + scaled_noise).astype(np.float32)
        achieved = 20.0 * np.log10(
            clean_rms / np.sqrt(np.mean(np.square(scaled_noise)))
        )
        metadata.append({
            "noise_source_split": carrier.split,
            "noise_source_relative_path": carrier.source_relative_path,
            "noise_start_sample": carrier.start_sample,
            "noise_end_sample": carrier.end_sample,
            "noise_source_round": carrier.source_round,
            "noise_carrier_sha256": carrier.sha256,
            "noise_carrier_rms": carrier.rms,
            "scaled_noise_rms": target_noise_rms,
            "achieved_snr_db": float(achieved),
        })
    return noisy, preprocess_conv1dgap_512(noisy), metadata


def validate_paired_parameters(
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> None:
    frozen_fields = (
        "sample_id",
        "class_name",
        "amplitude_p0",
        "frequency_khz",
        "tau_ms",
        "snr_db",
        "phi_rad",
        "t0_fraction",
    )
    if len(baseline_rows) != len(candidate_rows):
        raise ValueError("paired row counts differ")
    for baseline, candidate in zip(baseline_rows, candidate_rows, strict=True):
        for field in frozen_fields:
            if str(candidate[field]) != str(baseline[field]):
                raise ValueError(f"paired field changed: {field}")


def class_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        class_name: sum(str(row["class_name"]) == class_name for row in rows)
        for class_name in CLASS_ORDER
    }
