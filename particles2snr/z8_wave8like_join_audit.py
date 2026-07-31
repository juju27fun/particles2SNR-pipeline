"""Quantitative join audit for a Z8 v2 Wave8-like candidate."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from particles2snr.z8_wave8like_dataset import CLASS_NAMES, sha256_file


@dataclass(frozen=True)
class AuditConfig:
    segment_length: int = 16_384
    guard_samples: int = 300
    sampling_frequency_hz: int = 2_000_000

    @property
    def long_length(self) -> int:
        return 4 * self.segment_length

    @property
    def boundaries(self) -> tuple[int, int, int]:
        return tuple(
            index * self.segment_length for index in range(1, 4)
        )


JOIN_METRIC_FIELDS = (
    "long_id",
    "stratum",
    "boundary_index",
    "boundary_sample",
    "boundary_jump",
    "boundary_jump_robust_z",
    "join_rms",
    "left_control_rms",
    "right_control_rms",
    "join_to_control_rms_ratio",
    "join_peak_robust_z",
)


def _robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median, max(1.4826 * mad, 1e-12)


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


def load_manifest_rows(candidate_root: Path) -> list[dict[str, str]]:
    with (Path(candidate_root) / "manifest.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    validation = [row for row in rows if row["split"] == "val"]
    if not validation:
        raise ValueError("candidate has no validation rows")
    if any(row["split"] not in {"train", "val"} for row in rows):
        raise ValueError("candidate manifest contains a sealed/unknown split")
    return validation


def analyze_join_metrics(
    candidate_root: Path,
    config: AuditConfig,
) -> list[dict[str, object]]:
    candidate_root = Path(candidate_root)
    rows = load_manifest_rows(candidate_root)
    metrics: list[dict[str, object]] = []
    for row in rows:
        signal = np.load(
            candidate_root
            / "val"
            / "signals"
            / f"{row['long_id']}.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        if signal.shape != (config.long_length,) or signal.dtype != np.float64:
            raise ValueError(f"{row['long_id']}: invalid validation signal")
        differences = np.abs(np.diff(signal))
        difference_mask = np.ones(differences.size, dtype=bool)
        amplitude_mask = np.ones(signal.size, dtype=bool)
        for boundary in config.boundaries:
            difference_mask[
                max(0, boundary - config.guard_samples - 1) :
                min(differences.size, boundary + config.guard_samples)
            ] = False
            amplitude_mask[
                boundary - config.guard_samples :
                boundary + config.guard_samples
            ] = False
        diff_location, diff_scale = _robust_location_scale(
            differences[difference_mask]
        )
        amp_location, amp_scale = _robust_location_scale(
            np.asarray(signal[amplitude_mask])
        )
        for boundary_index, boundary in enumerate(config.boundaries, start=1):
            guard = config.guard_samples
            join = np.asarray(signal[boundary - guard : boundary + guard])
            left_control = np.asarray(
                signal[boundary - 3 * guard : boundary - guard]
            )
            right_control = np.asarray(
                signal[boundary + guard : boundary + 3 * guard]
            )
            boundary_jump = abs(float(signal[boundary] - signal[boundary - 1]))
            join_rms = _rms(join)
            left_rms = _rms(left_control)
            right_rms = _rms(right_control)
            control_rms = max((left_rms + right_rms) / 2.0, 1e-12)
            metrics.append(
                {
                    "long_id": row["long_id"],
                    "stratum": row["stratum"],
                    "boundary_index": boundary_index,
                    "boundary_sample": boundary,
                    "boundary_jump": boundary_jump,
                    "boundary_jump_robust_z": (
                        boundary_jump - diff_location
                    )
                    / diff_scale,
                    "join_rms": join_rms,
                    "left_control_rms": left_rms,
                    "right_control_rms": right_rms,
                    "join_to_control_rms_ratio": join_rms / control_rms,
                    "join_peak_robust_z": float(
                        np.max(np.abs(join - amp_location)) / amp_scale
                    ),
                }
            )
    return metrics


def _parse_label_rows(path: Path, long_length: int) -> list[dict[str, float | int]]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f"{path}:{line_number}: invalid label")
        class_id = int(fields[0])
        center = float(fields[1]) * long_length
        width = float(fields[2]) * long_length
        rows.append(
            {
                "class_id": class_id,
                "left": center - width / 2.0,
                "right": center + width / 2.0,
            }
        )
    return rows


def select_predeclared_cases(
    candidate_root: Path,
    config: AuditConfig,
) -> dict[str, object]:
    candidate_root = Path(candidate_root)
    rows = load_manifest_rows(candidate_root)
    positive_rows = [row for row in rows if row["stratum"] == "positive"]
    background_rows = [row for row in rows if row["stratum"] == "background"]
    if not positive_rows or not background_rows:
        raise ValueError("audit requires positive and background validation rows")
    full_trace = positive_rows[0]["long_id"]
    nearest: dict[int, dict[str, object]] = {}
    for row in positive_rows:
        labels = _parse_label_rows(
            candidate_root
            / "val"
            / "labels"
            / f"{row['long_id']}.txt",
            config.long_length,
        )
        event_ids = row["event_ids"].split(";") if row["event_ids"] else []
        if len(labels) != len(event_ids):
            raise ValueError(f"{row['long_id']}: event ID/label mismatch")
        for event_index, (label, event_id) in enumerate(zip(labels, event_ids)):
            class_id = int(label["class_id"])
            segment_index = int(float(label["left"])) // config.segment_length
            segment_start = segment_index * config.segment_length
            local_left = float(label["left"]) - segment_start
            local_right = float(label["right"]) - segment_start
            candidates: list[tuple[float, int, str]] = []
            if segment_index > 0:
                candidates.append(
                    (
                        local_left - config.guard_samples,
                        segment_start,
                        "left",
                    )
                )
            if segment_index < 3:
                candidates.append(
                    (
                        config.segment_length
                        - config.guard_samples
                        - local_right,
                        segment_start + config.segment_length,
                        "right",
                    )
                )
            if not candidates:
                continue
            distance, boundary_sample, side = min(candidates)
            record = {
                "long_id": row["long_id"],
                "event_id": event_id,
                "event_index": event_index,
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "left_sample": float(label["left"]),
                "right_sample": float(label["right"]),
                "nearest_boundary_sample": boundary_sample,
                "nearest_boundary_side": side,
                "distance_to_guard_samples": float(distance),
            }
            previous = nearest.get(class_id)
            ordering = (
                float(distance),
                row["long_id"],
                event_id,
                event_index,
            )
            if previous is None or ordering < (
                float(previous["distance_to_guard_samples"]),
                str(previous["long_id"]),
                str(previous["event_id"]),
                int(previous["event_index"]),
            ):
                nearest[class_id] = record
    if set(nearest) != {0, 1, 2}:
        raise ValueError("could not select one nearest-safe event per class")
    return {
        "selection_rule": (
            "manifest-first validation positive trace and background trace; "
            "lexicographically deterministic nearest surviving event to an "
            "actual internal join guard for each physical class"
        ),
        "selected_before_rendering": True,
        "full_trace_long_id": full_trace,
        "join_zoom_long_id": full_trace,
        "nearest_safe_events": [
            nearest[class_id] for class_id in range(3)
        ],
        "background_counterexample_long_id": background_rows[0]["long_id"],
    }


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "p999": float(np.quantile(array, 0.999)),
        "max": float(np.max(array)),
    }


def summarize_metrics(
    metrics: Sequence[Mapping[str, object]],
    *,
    candidate_manifest_sha256: str,
    computation_fingerprint: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "validation_rows": len({str(row["long_id"]) for row in metrics}),
        "validation_joins": len(metrics),
        "stratum_join_counts": {
            stratum: sum(row["stratum"] == stratum for row in metrics)
            for stratum in ("positive", "background")
        },
        "boundary_jump": _quantiles(
            [float(row["boundary_jump"]) for row in metrics]
        ),
        "boundary_jump_robust_z": _quantiles(
            [float(row["boundary_jump_robust_z"]) for row in metrics]
        ),
        "join_to_control_rms_ratio": _quantiles(
            [float(row["join_to_control_rms_ratio"]) for row in metrics]
        ),
        "join_peak_robust_z": _quantiles(
            [float(row["join_peak_robust_z"]) for row in metrics]
        ),
        "candidate_dataset_manifest_sha256": candidate_manifest_sha256,
        "computation_fingerprint": computation_fingerprint,
        "structural_generation_audit": "pass",
        "sealed_test_accessed": False,
        "visual_decision_required": True,
        "claim_boundary": (
            "Numerical discontinuity and amplitude summaries do not determine "
            "whether a join is morphologically particle-like."
        ),
    }


def write_analysis(
    *,
    candidate_root: Path,
    output_root: Path,
    config: AuditConfig,
    computation_provenance: Mapping[str, object],
    computation_fingerprint: str,
    run_payload: Mapping[str, object],
) -> dict[str, object]:
    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(f"refusing to mutate existing analysis: {output_root}")
    output_root.mkdir(parents=True)
    metrics = analyze_join_metrics(candidate_root, config)
    with (output_root / "join_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=JOIN_METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(metrics)
    selection = select_predeclared_cases(candidate_root, config)
    (output_root / "selected_cases.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    candidate_manifest_sha256 = sha256_file(
        Path(candidate_root) / "dataset-manifest.json"
    )
    summary = summarize_metrics(
        metrics,
        candidate_manifest_sha256=candidate_manifest_sha256,
        computation_fingerprint=computation_fingerprint,
    )
    (output_root / "summary_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metric_names = (
        "join_metrics.csv",
        "selected_cases.json",
        "summary_metrics.json",
    )
    metrics_manifest = {
        "schema_version": 1,
        "analysis_run_id": str(run_payload["run_id"]),
        "computation_provenance": dict(computation_provenance),
        "computation_fingerprint": computation_fingerprint,
        "metrics": [
            {"path": name, "sha256": sha256_file(output_root / name)}
            for name in metric_names
        ],
    }
    (output_root / "metrics_manifest.json").write_text(
        json.dumps(metrics_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload = dict(run_payload)
    payload.update(
        {
            "analysis_files": {
                name: sha256_file(output_root / name)
                for name in (*metric_names, "metrics_manifest.json")
            },
            "computation_fingerprint": computation_fingerprint,
            "outputs": [*metric_names, "metrics_manifest.json"],
            "summary": summary,
            "status": "complete_awaiting_visual_review",
        }
    )
    (output_root / "run.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload
