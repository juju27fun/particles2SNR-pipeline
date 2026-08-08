from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import uniform_filter1d

from .yeast_events import (
    YeastDetectionConfig,
    bandpass_yeast_signal,
    review_calibrated_detection_config_v1,
)


@dataclass(frozen=True)
class TemporalEnergyConfig:
    sampling_frequency_hz: float = 2_000_000.0
    energy_window_samples: int = 512
    hop_samples: int = 128
    smooth_frames: int = 3
    active_z: float = 3.5
    boundary_z: float = 1.5
    cluster_gap_ms: float = 0.128
    boundary_pad_ms: float = 0.04
    min_width_ms: float = 0.06
    max_width_ms: float = 2.0
    max_events_per_signal: int = 5


@dataclass(frozen=True)
class TemporalEnergyCandidate:
    candidate_index: int
    center_index: int
    event_start: int
    event_end: int
    width_samples: int
    width_ms: float
    energy_z_max: float


def _deduplicate_candidates(
    candidates: list[TemporalEnergyCandidate],
) -> list[TemporalEnergyCandidate]:
    """Collapse groups that expand to the exact same temporal event."""
    unique: dict[tuple[int, int, int], TemporalEnergyCandidate] = {}
    for candidate in candidates:
        key = (
            candidate.center_index,
            candidate.event_start,
            candidate.event_end,
        )
        previous = unique.get(key)
        if previous is None or candidate.energy_z_max > previous.energy_z_max:
            unique[key] = candidate
    return list(unique.values())


def _robust_z(values: np.ndarray) -> np.ndarray:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median))) * 1.4826
    return (values - median) / max(mad, 1.0e-12)


def _group_active(active: np.ndarray, max_gap_frames: int) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    start: int | None = None
    last_active = -1
    gap = 0
    for index, is_active in enumerate(active.astype(bool).tolist()):
        if is_active:
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


def temporal_energy_diagnostics(
    signal: np.ndarray,
    config: TemporalEnergyConfig = TemporalEnergyConfig(),
    filter_config: YeastDetectionConfig | None = None,
) -> dict[str, np.ndarray]:
    detector_config = filter_config or review_calibrated_detection_config_v1()
    filtered = bandpass_yeast_signal(signal, detector_config)
    instantaneous_power = np.square(filtered.astype(np.float64))
    local_energy = uniform_filter1d(
        instantaneous_power,
        size=config.energy_window_samples,
        mode="nearest",
    )
    half = config.energy_window_samples // 2
    centers = np.arange(
        half,
        max(half + 1, filtered.size - half + 1),
        config.hop_samples,
        dtype=np.int64,
    )
    centers = centers[centers < filtered.size]
    frame_energy = local_energy[centers]
    if config.smooth_frames > 1:
        frame_energy = uniform_filter1d(
            frame_energy,
            size=config.smooth_frames,
            mode="nearest",
        )
    return {
        "filtered": filtered,
        "centers": centers,
        "frame_energy": frame_energy,
        "energy_z": _robust_z(frame_energy),
    }


def detect_temporal_energy_events(
    signal: np.ndarray,
    config: TemporalEnergyConfig = TemporalEnergyConfig(),
) -> list[TemporalEnergyCandidate]:
    diagnostics = temporal_energy_diagnostics(signal, config)
    centers = diagnostics["centers"]
    frame_energy = diagnostics["frame_energy"]
    energy_z = diagnostics["energy_z"]
    if not centers.size:
        return []

    max_gap = max(
        0,
        int(
            round(
                config.cluster_gap_ms
                / 1000.0
                * config.sampling_frequency_hz
                / config.hop_samples
            )
        ),
    )
    groups = _group_active(energy_z >= config.active_z, max_gap)
    pad = int(
        round(
            config.boundary_pad_ms
            / 1000.0
            * config.sampling_frequency_hz
        )
    )
    half = config.energy_window_samples // 2
    candidates: list[TemporalEnergyCandidate] = []
    for left, right in groups:
        while left > 0 and float(energy_z[left - 1]) >= config.boundary_z:
            left -= 1
        while right < energy_z.size - 1 and float(energy_z[right + 1]) >= config.boundary_z:
            right += 1
        frames = np.arange(left, right + 1, dtype=np.int64)
        weights = np.maximum(frame_energy[frames] - np.median(frame_energy), 0.0)
        if float(np.sum(weights)) > 0:
            center = int(round(np.average(centers[frames], weights=weights)))
        else:
            center = int(round(float(np.mean(centers[frames]))))
        event_start = max(0, int(centers[left]) - half - pad)
        event_end = min(
            int(np.asarray(signal).size),
            int(centers[right]) + half + pad,
        )
        width_samples = event_end - event_start
        if width_samples <= 0:
            continue
        candidates.append(
            TemporalEnergyCandidate(
                candidate_index=len(candidates),
                center_index=center,
                event_start=event_start,
                event_end=event_end,
                width_samples=width_samples,
                width_ms=(
                    width_samples
                    / config.sampling_frequency_hz
                    * 1000.0
                ),
                energy_z_max=float(np.max(energy_z[frames])),
            )
        )

    candidates = _deduplicate_candidates(candidates)
    candidates.sort(key=lambda item: item.energy_z_max, reverse=True)
    if config.max_events_per_signal > 0:
        candidates = candidates[: config.max_events_per_signal]
    candidates.sort(key=lambda item: item.center_index)
    return [
        TemporalEnergyCandidate(
            **{**asdict(candidate), "candidate_index": index}
        )
        for index, candidate in enumerate(candidates)
    ]


def retained_temporal_candidates(
    candidates: list[TemporalEnergyCandidate],
    *,
    quality_z: float,
    config: TemporalEnergyConfig = TemporalEnergyConfig(),
) -> list[TemporalEnergyCandidate]:
    return [
        candidate
        for candidate in candidates
        if candidate.energy_z_max >= quality_z
        and config.min_width_ms <= candidate.width_ms <= config.max_width_ms
    ]


def match_candidate_centers(
    reference_centers: list[int],
    predicted_centers: list[int],
    *,
    tolerance_samples: int,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    available_reference = set(range(len(reference_centers)))
    available_predicted = set(range(len(predicted_centers)))
    pairs = sorted(
        (
            (abs(reference_centers[reference] - predicted_centers[predicted]), reference, predicted)
            for reference in available_reference
            for predicted in available_predicted
            if abs(reference_centers[reference] - predicted_centers[predicted])
            <= tolerance_samples
        ),
        key=lambda item: (item[0], item[1], item[2]),
    )
    matches: list[tuple[int, int]] = []
    for _distance, reference, predicted in pairs:
        if reference not in available_reference or predicted not in available_predicted:
            continue
        matches.append((reference, predicted))
        available_reference.remove(reference)
        available_predicted.remove(predicted)
    return (
        matches,
        sorted(available_reference),
        sorted(available_predicted),
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty comparison CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _reference_by_record(
    candidate_rows: list[dict[str, str]],
) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidate_rows:
        if row["quality"] in {"strict", "medium"}:
            grouped[row["record_id"]].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["center_index"]))
    return grouped


def _score_threshold(
    *,
    record_ids: list[str],
    reference_by_record: dict[str, list[dict[str, str]]],
    simple_by_record: dict[str, list[TemporalEnergyCandidate]],
    threshold: float,
    tolerance_samples: int,
    config: TemporalEnergyConfig,
) -> dict[str, float | int]:
    true_positive = false_positive = false_negative = 0
    for record_id in record_ids:
        reference = reference_by_record.get(record_id, [])
        predicted = retained_temporal_candidates(
            simple_by_record.get(record_id, []),
            quality_z=threshold,
            config=config,
        )
        matches, unmatched_reference, unmatched_predicted = match_candidate_centers(
            [int(row["center_index"]) for row in reference],
            [row.center_index for row in predicted],
            tolerance_samples=tolerance_samples,
        )
        true_positive += len(matches)
        false_negative += len(unmatched_reference)
        false_positive += len(unmatched_predicted)
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "threshold": float(threshold),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision_vs_current": precision,
        "recall_vs_current": recall,
        "f1_vs_current": f1,
    }


def build_temporal_ablation_comparison(
    *,
    source_index_csv: Path,
    raw_dataset_root: Path,
    current_candidate_csv: Path,
    output_dir: Path,
    config: TemporalEnergyConfig = TemporalEnergyConfig(),
    calibration_split: str = "development_train",
    validation_split: str = "development_validation",
    match_tolerance_ms: float = 0.5,
) -> dict[str, Any]:
    source_rows = [
        row
        for row in _read_csv(source_index_csv)
        if row["is_canonical_duplicate_member"].lower() == "true"
        and row["development_split"] in {calibration_split, validation_split}
    ]
    current_rows = _read_csv(current_candidate_csv)
    reference_by_record = _reference_by_record(current_rows)
    tolerance_samples = int(
        round(match_tolerance_ms / 1000.0 * config.sampling_frequency_hz)
    )

    simple_by_record: dict[str, list[TemporalEnergyCandidate]] = {}
    candidate_output: list[dict[str, Any]] = []
    for source in source_rows:
        signal = np.load(
            raw_dataset_root / source["relative_path"],
            allow_pickle=False,
        )
        candidates = detect_temporal_energy_events(signal, config)
        simple_by_record[source["record_id"]] = candidates
        for candidate in candidates:
            candidate_output.append(
                {
                    "record_id": source["record_id"],
                    "relative_path": source["relative_path"],
                    "source_group": source["source_group"],
                    "development_split": source["development_split"],
                    **asdict(candidate),
                }
            )

    calibration_ids = [
        row["record_id"]
        for row in source_rows
        if row["development_split"] == calibration_split
    ]
    validation_ids = [
        row["record_id"]
        for row in source_rows
        if row["development_split"] == validation_split
    ]
    eligible_scores = sorted(
        {
            candidate.energy_z_max
            for record_id in calibration_ids
            for candidate in simple_by_record.get(record_id, [])
            if config.min_width_ms
            <= candidate.width_ms
            <= config.max_width_ms
        }
    )
    if not eligible_scores:
        raise ValueError("Temporal ablation produced no calibration candidates")
    score_indices = np.unique(
        np.linspace(0, len(eligible_scores) - 1, min(401, len(eligible_scores))).astype(int)
    )
    thresholds = [eligible_scores[int(index)] for index in score_indices]
    calibration_scores = [
        _score_threshold(
            record_ids=calibration_ids,
            reference_by_record=reference_by_record,
            simple_by_record=simple_by_record,
            threshold=threshold,
            tolerance_samples=tolerance_samples,
            config=config,
        )
        for threshold in thresholds
    ]
    selected = max(
        calibration_scores,
        key=lambda row: (
            float(row["f1_vs_current"]),
            float(row["precision_vs_current"]),
            float(row["recall_vs_current"]),
            float(row["threshold"]),
        ),
    )
    selected_threshold = float(selected["threshold"])

    split_scores = {}
    for split, record_ids in (
        (calibration_split, calibration_ids),
        (validation_split, validation_ids),
    ):
        split_scores[split] = _score_threshold(
            record_ids=record_ids,
            reference_by_record=reference_by_record,
            simple_by_record=simple_by_record,
            threshold=selected_threshold,
            tolerance_samples=tolerance_samples,
            config=config,
        )

    comparison_rows: list[dict[str, Any]] = []
    center_errors_ms: dict[str, list[float]] = defaultdict(list)
    category_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for source in source_rows:
        record_id = source["record_id"]
        reference = reference_by_record.get(record_id, [])
        predicted = retained_temporal_candidates(
            simple_by_record.get(record_id, []),
            quality_z=selected_threshold,
            config=config,
        )
        matches, unmatched_reference, unmatched_predicted = match_candidate_centers(
            [int(row["center_index"]) for row in reference],
            [row.center_index for row in predicted],
            tolerance_samples=tolerance_samples,
        )
        split = source["development_split"]
        for reference_index, predicted_index in matches:
            current = reference[reference_index]
            simple = predicted[predicted_index]
            error_ms = (
                simple.center_index - int(current["center_index"])
            ) / config.sampling_frequency_hz * 1000.0
            center_errors_ms[split].append(abs(error_ms))
            category_counts[split]["matched"] += 1
            comparison_rows.append(
                {
                    "record_id": record_id,
                    "relative_path": source["relative_path"],
                    "source_group": source["source_group"],
                    "development_split": split,
                    "category": "matched",
                    "current_event_id": current["event_id"],
                    "current_center_index": current["center_index"],
                    "current_start": current["event_start"],
                    "current_end": current["event_end"],
                    "current_snr": current["snr_proxy"],
                    "simple_candidate_index": simple.candidate_index,
                    "simple_center_index": simple.center_index,
                    "simple_start": simple.event_start,
                    "simple_end": simple.event_end,
                    "simple_energy_z": simple.energy_z_max,
                    "center_error_ms": error_ms,
                }
            )
        for reference_index in unmatched_reference:
            current = reference[reference_index]
            category_counts[split]["current_only"] += 1
            comparison_rows.append(
                {
                    "record_id": record_id,
                    "relative_path": source["relative_path"],
                    "source_group": source["source_group"],
                    "development_split": split,
                    "category": "current_only",
                    "current_event_id": current["event_id"],
                    "current_center_index": current["center_index"],
                    "current_start": current["event_start"],
                    "current_end": current["event_end"],
                    "current_snr": current["snr_proxy"],
                    "simple_candidate_index": "",
                    "simple_center_index": "",
                    "simple_start": "",
                    "simple_end": "",
                    "simple_energy_z": "",
                    "center_error_ms": "",
                }
            )
        for predicted_index in unmatched_predicted:
            simple = predicted[predicted_index]
            category_counts[split]["simple_only"] += 1
            comparison_rows.append(
                {
                    "record_id": record_id,
                    "relative_path": source["relative_path"],
                    "source_group": source["source_group"],
                    "development_split": split,
                    "category": "simple_only",
                    "current_event_id": "",
                    "current_center_index": "",
                    "current_start": "",
                    "current_end": "",
                    "current_snr": "",
                    "simple_candidate_index": simple.candidate_index,
                    "simple_center_index": simple.center_index,
                    "simple_start": simple.event_start,
                    "simple_end": simple.event_end,
                    "simple_energy_z": simple.energy_z_max,
                    "center_error_ms": "",
                }
            )

    for split, values in center_errors_ms.items():
        split_scores[split]["center_abs_error_ms"] = {
            "p50": float(np.quantile(values, 0.50)) if values else None,
            "p95": float(np.quantile(values, 0.95)) if values else None,
            "max": float(np.max(values)) if values else None,
        }
        split_scores[split]["categories"] = dict(category_counts[split])

    output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(output_dir / "temporal_candidates.csv", candidate_output)
    _write_csv(output_dir / "event_comparison.csv", comparison_rows)
    _write_csv(output_dir / "calibration_curve.csv", calibration_scores)
    summary = {
        "schema_version": 1,
        "method": "bandpass_temporal_energy_mad",
        "scientific_scope": (
            "Ablation compared with the current detector, not an independent biological ground truth."
        ),
        "source_index": str(source_index_csv),
        "current_candidate_csv": str(current_candidate_csv),
        "calibration_split": calibration_split,
        "validation_split": validation_split,
        "match_tolerance_ms": match_tolerance_ms,
        "config": asdict(config),
        "selected_quality_z": selected_threshold,
        "n_source_records": len(source_rows),
        "n_temporal_proposals": len(candidate_output),
        "split_scores": split_scores,
    }
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
