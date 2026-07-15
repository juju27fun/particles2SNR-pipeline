from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Iterable, Sequence

import numpy as np

from .yeast_events import YeastDetectionConfig, YeastEventCandidate, detect_yeast_events


@dataclass(frozen=True)
class CalibrationSpec:
    boundary_snr_z: float
    cluster_gap_ms: float
    acceptance_snr_z: float
    maximum_width_ms: float
    maximum_events: int
    minimum_concentration: float = 0.08


def accepted_count(
    candidates: Sequence[YeastEventCandidate], spec: CalibrationSpec
) -> int:
    ranked = sorted(candidates, key=lambda candidate: candidate.snr_proxy, reverse=True)
    if spec.maximum_events > 0:
        ranked = ranked[: spec.maximum_events]
    return sum(
        candidate.snr_proxy >= spec.acceptance_snr_z
        and candidate.energy_concentration >= spec.minimum_concentration
        and candidate.width_ms <= spec.maximum_width_ms
        for candidate in ranked
    )


def count_proxy_metrics(
    predicted_counts: Sequence[int], true_counts: Sequence[int]
) -> dict[str, float | int]:
    if len(predicted_counts) != len(true_counts):
        raise ValueError("Predicted and true count arrays must have the same length")
    if not predicted_counts:
        raise ValueError("At least one reviewed trace is required")
    predicted = np.asarray(predicted_counts, dtype=np.int64)
    truth = np.asarray(true_counts, dtype=np.int64)
    if np.any(predicted < 0) or np.any(truth < 0):
        raise ValueError("Event counts must be non-negative")
    true_positive = int(np.minimum(predicted, truth).sum())
    false_positive = int(np.maximum(predicted - truth, 0).sum())
    false_negative = int(np.maximum(truth - predicted, 0).sum())
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 1.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 1.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n_traces": int(truth.size),
        "predicted_events": int(predicted.sum()),
        "true_events": int(truth.sum()),
        "true_positive_count_proxy": true_positive,
        "false_positive_count_proxy": false_positive,
        "false_negative_count_proxy": false_negative,
        "precision_count_proxy": precision,
        "recall_count_proxy": recall,
        "f1_count_proxy": f1,
        "exact_count_fraction": float(np.mean(predicted == truth)),
        "mean_absolute_count_error": float(np.mean(np.abs(predicted - truth))),
    }


def detect_segmentation_variants(
    signals_by_record: dict[str, np.ndarray],
    base_config: YeastDetectionConfig,
    *,
    boundary_values: Iterable[float],
    cluster_gap_values: Iterable[float],
) -> dict[tuple[float, float], dict[str, list[YeastEventCandidate]]]:
    output: dict[tuple[float, float], dict[str, list[YeastEventCandidate]]] = {}
    for boundary in boundary_values:
        for cluster_gap in cluster_gap_values:
            config = replace(
                base_config,
                boundary_snr_z=float(boundary),
                cluster_gap_ms=float(cluster_gap),
                max_events_per_signal=0,
            )
            by_record = {}
            for record_id, signal in signals_by_record.items():
                candidates, _reason = detect_yeast_events(signal, config)
                by_record[record_id] = candidates
            output[(float(boundary), float(cluster_gap))] = by_record
    return output


def sweep_count_calibration(
    detections: dict[tuple[float, float], dict[str, list[YeastEventCandidate]]],
    true_counts_by_record: dict[str, int],
    *,
    acceptance_snr_values: Iterable[float],
    maximum_width_values: Iterable[float],
    maximum_event_values: Iterable[int],
    minimum_concentration: float = 0.08,
) -> list[dict[str, float | int]]:
    record_ids = list(true_counts_by_record)
    truth = [true_counts_by_record[record_id] for record_id in record_ids]
    rows: list[dict[str, float | int]] = []
    for (boundary, cluster_gap), by_record in detections.items():
        for acceptance_snr in acceptance_snr_values:
            for maximum_width in maximum_width_values:
                for maximum_events in maximum_event_values:
                    spec = CalibrationSpec(
                        boundary_snr_z=boundary,
                        cluster_gap_ms=cluster_gap,
                        acceptance_snr_z=float(acceptance_snr),
                        maximum_width_ms=float(maximum_width),
                        maximum_events=int(maximum_events),
                        minimum_concentration=float(minimum_concentration),
                    )
                    predicted = [
                        accepted_count(by_record[record_id], spec)
                        for record_id in record_ids
                    ]
                    rows.append({**asdict(spec), **count_proxy_metrics(predicted, truth)})
    return rows


def calibration_spec_from_row(row: dict[str, float | int]) -> CalibrationSpec:
    return CalibrationSpec(
        boundary_snr_z=float(row["boundary_snr_z"]),
        cluster_gap_ms=float(row["cluster_gap_ms"]),
        acceptance_snr_z=float(row["acceptance_snr_z"]),
        maximum_width_ms=float(row["maximum_width_ms"]),
        maximum_events=int(row["maximum_events"]),
        minimum_concentration=float(row["minimum_concentration"]),
    )


def evaluate_calibration_spec(
    detections: dict[tuple[float, float], dict[str, list[YeastEventCandidate]]],
    true_counts_by_record: dict[str, int],
    spec: CalibrationSpec,
    *,
    record_ids: Iterable[str] | None = None,
) -> dict[str, float | int]:
    selected_ids = list(record_ids) if record_ids is not None else list(true_counts_by_record)
    by_record = detections[(spec.boundary_snr_z, spec.cluster_gap_ms)]
    predicted = [accepted_count(by_record[record_id], spec) for record_id in selected_ids]
    truth = [true_counts_by_record[record_id] for record_id in selected_ids]
    return count_proxy_metrics(predicted, truth)


def select_development_variant(
    rows: Sequence[dict[str, float | int]],
    *,
    minimum_precision: float = 0.90,
    minimum_recall: float = 0.85,
) -> dict[str, float | int] | None:
    eligible = [
        row
        for row in rows
        if float(row["precision_count_proxy"]) >= minimum_precision
        and float(row["recall_count_proxy"]) >= minimum_recall
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda row: (
            float(row["f1_count_proxy"]),
            float(row["exact_count_fraction"]),
            -float(row["mean_absolute_count_error"]),
            -float(row["acceptance_snr_z"]),
        ),
    )
