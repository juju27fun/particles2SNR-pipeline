from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

from .yeast_raw_data import normalize_raw_dataset_roots, resolve_raw_signal


def read_usable_candidates(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["quality"] in {"strict", "medium"}]
    if not rows:
        raise ValueError(f"No strict or medium candidates found in {path}")
    event_ids = [row["event_id"] for row in rows]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("Candidate event IDs are not unique")
    return rows


def clamped_crop(signal: np.ndarray, center_index: int, length: int) -> tuple[np.ndarray, int]:
    values = np.asarray(signal).squeeze()
    if values.ndim != 1:
        raise ValueError(f"Expected one-dimensional signal, got {values.shape}")
    if values.size < length:
        raise ValueError(f"Signal length {values.size} is shorter than crop length {length}")
    start = min(max(int(center_index) - length // 2, 0), values.size - length)
    return values[start : start + length].astype(np.float32, copy=False), int(start)


def preprocess_crop(
    crop: np.ndarray,
    *,
    sampling_frequency_hz: float = 2_000_000.0,
    low_hz: float = 5_000.0,
    high_hz: float = 100_000.0,
    filter_order: int = 4,
    downsample_factor: int = 2,
) -> np.ndarray:
    values = np.asarray(crop, dtype=np.float32)
    sos = butter(
        filter_order,
        [low_hz, high_hz],
        btype="bandpass",
        fs=sampling_frequency_hz,
        output="sos",
    )
    filtered = sosfiltfilt(sos, values - float(np.mean(values))).astype(np.float32)
    output = resample_poly(filtered, up=1, down=downsample_factor).astype(np.float32)
    if not np.all(np.isfinite(output)):
        raise ValueError("Preprocessing produced non-finite values")
    return output


def build_representation_dataset(
    *,
    candidate_csv: Path,
    raw_dataset_root: Path | None,
    raw_dataset_roots: Mapping[str, Path] | None = None,
    output_dir: Path,
    raw_dataset_id: str,
    candidate_dataset_id: str,
    crop_length: int = 8192,
    downsample_factor: int = 2,
) -> dict[str, Any]:
    rows = read_usable_candidates(candidate_csv)
    output_length = crop_length // downsample_factor
    if crop_length % downsample_factor:
        raise ValueError("crop_length must be divisible by downsample_factor")
    output_dir.mkdir(parents=True, exist_ok=False)
    signals = np.lib.format.open_memmap(
        output_dir / "signals.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(rows), output_length),
    )
    metadata: list[dict[str, Any]] = []
    train_sum = 0.0
    train_square_sum = 0.0
    train_count = 0
    raw_root, raw_roots = normalize_raw_dataset_roots(
        raw_dataset_root=raw_dataset_root,
        raw_dataset_roots=raw_dataset_roots,
    )

    for index, row in enumerate(rows):
        raw = np.load(
            resolve_raw_signal(row, single_root=raw_root, roots_by_dataset=raw_roots),
            allow_pickle=False,
        )
        crop, crop_start = clamped_crop(raw, int(row["center_index"]), crop_length)
        processed = preprocess_crop(crop, downsample_factor=downsample_factor)
        if processed.size != output_length:
            raise ValueError(f"Unexpected output length for {row['event_id']}: {processed.size}")
        signals[index] = processed
        if row["development_split"] == "development_train":
            values = processed.astype(np.float64, copy=False)
            train_sum += float(np.sum(values))
            train_square_sum += float(np.sum(np.square(values)))
            train_count += int(values.size)
        metadata.append(
            {
                **row,
                "signal_row": index,
                "crop_start": crop_start,
                "crop_end": crop_start + crop_length,
                "event_center_input_index": (int(row["center_index"]) - crop_start) / downsample_factor,
                "event_start_input_index": (int(row["event_start"]) - crop_start) / downsample_factor,
                "event_end_input_index": (int(row["event_end"]) - crop_start) / downsample_factor,
            }
        )
    if train_count == 0:
        raise ValueError("Development train split is empty")
    train_mean = train_sum / train_count
    train_variance = max(train_square_sum / train_count - train_mean**2, 0.0)
    train_std = float(np.sqrt(train_variance))
    if train_std <= 1.0e-12:
        raise ValueError("Training global standard deviation is zero")
    for start in range(0, len(rows), 256):
        end = min(start + 256, len(rows))
        signals[start:end] = (signals[start:end] - train_mean) / train_std
    signals.flush()
    del signals

    with (output_dir / "events.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata[0]))
        writer.writeheader()
        writer.writerows(metadata)

    contract = {
        "schema_version": 1,
        "contract_id": "yeast-event-8192to4096-bandpass-global-v1",
        "raw_dataset": raw_dataset_id,
        "raw_datasets": sorted(raw_roots) if raw_roots else [raw_dataset_id],
        "candidate_dataset": candidate_dataset_id,
        "input_channels": 1,
        "raw_sampling_frequency_hz": 2_000_000.0,
        "raw_crop_length": crop_length,
        "raw_crop_duration_ms": crop_length / 2_000_000.0 * 1000.0,
        "crop_policy": "clamp centered start to source bounds; never pad",
        "bandpass_hz": [5_000.0, 100_000.0],
        "bandpass_order": 4,
        "bandpass_phase": "zero-phase SOS forward-backward",
        "downsampling": "scipy.signal.resample_poly up=1 down=2",
        "output_sampling_frequency_hz": 1_000_000.0,
        "output_length": output_length,
        "output_duration_ms": output_length / 1_000_000.0 * 1000.0,
        "normalization": {
            "policy": "global development-train mean and standard deviation",
            "mean": train_mean,
            "std": train_std,
        },
        "information_policy": {
            "in_band_amplitude": "unresolved-preserve-no-augmentation-no-supervision",
            "dc_offset_and_out_of_band_drift": "nuisance-fixed-filter",
            "event_position_in_crop": "nuisance-retained-as-metadata",
            "duration_envelope_and_doppler": "preserve",
            "padding": "forbidden",
        },
        "split_scope": (
            "sealed acquisition OOD available; normalization fitted on development_train only"
            if any(row["development_split"] == "sealed_acquisition_test" for row in rows)
            else "single-acquisition development only; no acquisition-level OOD claim"
        ),
    }
    (output_dir / "input_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "schema_version": 1,
        "n_events": len(rows),
        "split_counts": dict(sorted(Counter(row["development_split"] for row in rows).items())),
        "quality_counts": dict(sorted(Counter(row["quality"] for row in rows).items())),
        "source_group_counts": dict(sorted(Counter(row["source_group"] for row in rows).items())),
        "condition_counts": dict(sorted(Counter(row["condition_id"] for row in rows).items())),
        "acquisition_counts": dict(
            sorted(Counter(row.get("acquisition_id", "unknown") for row in rows).items())
        ),
        "n_source_records": len({row["record_id"] for row in rows}),
        "input_contract": contract["contract_id"],
        "signals_shape": [len(rows), output_length],
        "signals_dtype": "float32",
        "event_center_input_index_quantiles": {
            name: float(value)
            for name, value in zip(
                ("p05", "p25", "p50", "p75", "p95"),
                np.quantile(
                    [float(row["event_center_input_index"]) for row in metadata],
                    [0.05, 0.25, 0.5, 0.75, 0.95],
                ),
            )
        },
        "scientific_scope": "unlabeled representation input; condition fields remain acquisition-level proxies",
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
