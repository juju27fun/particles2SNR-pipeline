from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .yeast_events import YeastDetectionConfig, crop_around_index, detect_yeast_events
from .yeast_raw_data import normalize_raw_dataset_roots, resolve_raw_signal


def read_source_index(path: Path, canonical_only: bool = True) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if canonical_only:
        rows = [row for row in rows if row["is_canonical_duplicate_member"].lower() == "true"]
    if not rows:
        raise ValueError(f"No source rows found in {path}")
    return rows


def _stable_order(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _all_fields(rows: list[dict[str, Any]]) -> list[str]:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    return fields


def _quantiles(values: list[float]) -> dict[str, float]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if not finite.size:
        return {}
    result = np.quantile(finite, [0.05, 0.25, 0.5, 0.75, 0.95])
    return dict(zip(("p05", "p25", "p50", "p75", "p95"), result.astype(float).tolist()))


def _coverage(candidates: list[dict[str, Any]], sampling_frequency_hz: float) -> dict[str, Any]:
    usable = [row for row in candidates if row["quality"] in {"strict", "medium"}]
    required_context = int(round(0.5e-3 * sampling_frequency_hz))
    output: dict[str, Any] = {}
    for length in (4096, 8192, 16384):
        half = length // 2
        full_event = 0
        measured_context = 0
        edge_safe = 0
        padding_fractions: list[float] = []
        for row in usable:
            center = int(row["center_index"])
            signal_length = int(row["signal_length"])
            crop_start = center - half
            crop_end = crop_start + length
            full_event += int(
                int(row["event_start"]) >= crop_start and int(row["event_end"]) <= crop_end
            )
            measured_context += int(
                int(row["event_start"]) - required_context >= max(crop_start, 0)
                and int(row["event_end"]) + required_context <= min(crop_end, signal_length)
            )
            edge_safe += int(crop_start >= 0 and crop_end <= signal_length)
            padding = max(0, -crop_start) + max(0, crop_end - signal_length)
            padding_fractions.append(padding / length)
        denominator = max(len(usable), 1)
        output[str(length)] = {
            "n_usable": len(usable),
            "event_fits_centered_crop_fraction": full_event / denominator,
            "measured_context_0p5ms_each_side_fraction": measured_context / denominator,
            "unpadded_crop_fraction": edge_safe / denominator,
            "padding_fraction_quantiles": _quantiles(padding_fractions),
            "physical_duration_ms": length / sampling_frequency_hz * 1000.0,
        }
    return output


def build_candidate_audit(
    *,
    source_index_csv: Path,
    raw_dataset_root: Path | None,
    raw_dataset_roots: Mapping[str, Path] | None = None,
    output_dir: Path,
    config: YeastDetectionConfig,
    review_crop_length: int = 8192,
    review_per_stratum: int = 6,
    file_review_per_stratum: int | None = None,
    seed: int = 42,
    max_files: int = 0,
    review_excluded_record_ids: set[str] | None = None,
) -> dict[str, Any]:
    source_rows = read_source_index(source_index_csv)
    if max_files > 0:
        source_rows = source_rows[:max_files]
    candidate_rows: list[dict[str, Any]] = []
    file_rows: list[dict[str, Any]] = []
    raw_root, raw_roots = normalize_raw_dataset_roots(
        raw_dataset_root=raw_dataset_root,
        raw_dataset_roots=raw_dataset_roots,
    )
    excluded_record_ids = review_excluded_record_ids or set()
    full_trace_review_per_stratum = (
        review_per_stratum
        if file_review_per_stratum is None
        else file_review_per_stratum
    )

    for source in source_rows:
        signal_path = resolve_raw_signal(
            source,
            single_root=raw_root,
            roots_by_dataset=raw_roots,
        )
        signal = np.load(signal_path, allow_pickle=False)
        candidates, no_candidate_reason = detect_yeast_events(
            signal,
            config,
            review_crop_length=review_crop_length,
        )
        file_rows.append(
            {
                "record_id": source["record_id"],
                "raw_dataset": source.get("raw_dataset", ""),
                "relative_path": source["relative_path"],
                "source_group": source["source_group"],
                "condition_id": source["condition_id"],
                "acquisition_id": source["acquisition_id"],
                "acquisition_role": source.get("acquisition_role", ""),
                "capture_block_id": source["capture_block_id"],
                "development_split": source["development_split"],
                "n_candidates": len(candidates),
                "n_retained_candidates": sum(
                    candidate.quality in {"strict", "medium"} for candidate in candidates
                ),
                "n_rejected_candidates": sum(candidate.quality == "reject" for candidate in candidates),
                "no_candidate_reason": no_candidate_reason,
            }
        )
        for candidate in candidates:
            crop_geometry: dict[str, Any] = {}
            for crop_length in (4096, 8192, 16384):
                crop_start = candidate.center_index - crop_length // 2
                crop_end = crop_start + crop_length
                crop_geometry[f"crop_{crop_length}_pad_left"] = max(0, -crop_start)
                crop_geometry[f"crop_{crop_length}_pad_right"] = max(0, crop_end - int(np.asarray(signal).size))
            row = {
                "event_id": f"{source['record_id']}:{candidate.candidate_index:02d}",
                "record_id": source["record_id"],
                "raw_dataset": source.get("raw_dataset", ""),
                "relative_path": source["relative_path"],
                "source_group": source["source_group"],
                "condition_id": source["condition_id"],
                "label_scope": source["label_scope"],
                "acquisition_id": source["acquisition_id"],
                "acquisition_role": source.get("acquisition_role", ""),
                "capture_block_id": source["capture_block_id"],
                "development_split": source["development_split"],
                "signal_length": int(np.asarray(signal).size),
                **asdict(candidate),
                **crop_geometry,
            }
            candidate_rows.append(row)

    strata: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        if row["record_id"] in excluded_record_ids:
            continue
        strata[(row["acquisition_id"], row["source_group"], row["quality"])].append(row)
    review_rows: list[dict[str, Any]] = []
    for stratum, rows in sorted(strata.items()):
        selected = sorted(rows, key=lambda row: _stable_order(row["event_id"], seed))[:review_per_stratum]
        for row in selected:
            review_rows.append(
                {**row, "review_stratum": f"{stratum[0]}:{stratum[1]}:{stratum[2]}"}
            )

    no_candidate_by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in file_rows:
        if row["record_id"] in excluded_record_ids:
            continue
        if row["n_candidates"] == 0:
            no_candidate_by_group[(row["acquisition_id"], row["source_group"])].append(row)
    for (acquisition, group), rows in sorted(no_candidate_by_group.items()):
        selected = sorted(rows, key=lambda row: _stable_order(row["record_id"], seed))[:review_per_stratum]
        for row in selected:
            review_rows.append(
                {
                    "event_id": f"{row['record_id']}:background",
                    **row,
                    "label_scope": "none",
                    "signal_length": 16384,
                    "candidate_index": -1,
                    "center_index": 8192,
                    "event_start": -1,
                    "event_end": -1,
                    "width_samples": 0,
                    "width_ms": 0.0,
                    "snr_proxy": float("nan"),
                    "energy_concentration": float("nan"),
                    "phase_coherence": float("nan"),
                    "n_doppler_peaks": 0,
                    "doppler_low_hz": float("nan"),
                    "doppler_high_hz": float("nan"),
                    "doppler_peak_hz": float("nan"),
                    "quality": "no_candidate",
                    "rejection_reason": row["no_candidate_reason"],
                    "review_stratum": f"{acquisition}:{group}:no_candidate",
                }
            )

    review_rows.sort(key=lambda row: (row["review_stratum"], row["event_id"]))
    review_signals = []
    for row in review_rows:
        signal = np.load(
            resolve_raw_signal(row, single_root=raw_root, roots_by_dataset=raw_roots),
            allow_pickle=False,
        )
        review_signals.append(crop_around_index(signal, int(row["center_index"]), review_crop_length))
        row.update(
            {
                "review_event_present": "",
                "review_center_acceptable": "",
                "review_full_event_visible": "",
                "review_artifact": "",
                "reviewer": "",
                "review_notes": "",
            }
        )

    candidates_by_record: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        candidates_by_record[row["record_id"]].append(row)
    file_review_strata: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in file_rows:
        if row["record_id"] in excluded_record_ids:
            continue
        count_bucket = str(min(int(row["n_candidates"]), 3))
        file_review_strata[
            (row["acquisition_id"], row["source_group"], count_bucket)
        ].append(row)
    file_review_rows: list[dict[str, Any]] = []
    for stratum, rows in sorted(file_review_strata.items()):
        selected = sorted(rows, key=lambda row: _stable_order(row["record_id"], seed + 1))[
            :full_trace_review_per_stratum
        ]
        for row in selected:
            detected = candidates_by_record[row["record_id"]]
            file_review_rows.append(
                {
                    **row,
                    "review_stratum": (
                        f"{stratum[0]}:{stratum[1]}:n_candidates_{stratum[2]}"
                    ),
                    "detected_event_ids": json.dumps([item["event_id"] for item in detected]),
                    "detected_centers": json.dumps([item["center_index"] for item in detected]),
                    "review_true_event_count": "",
                    "review_false_retained_candidate_count": "",
                    "review_true_rejected_candidate_count": "",
                    "review_missed_event_count": "",
                    "reviewer": "",
                    "review_notes": "",
                }
            )
    file_review_rows.sort(key=lambda row: (row["review_stratum"], row["record_id"]))
    file_review_signals = [
        np.asarray(
            np.load(
                resolve_raw_signal(row, single_root=raw_root, roots_by_dataset=raw_roots),
                allow_pickle=False,
            ),
            dtype=np.float32,
        )
        for row in file_review_rows
    ]

    output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(output_dir / "candidate_events.csv", candidate_rows, _all_fields(candidate_rows))
    _write_csv(output_dir / "file_detection_report.csv", file_rows, _all_fields(file_rows))
    _write_csv(output_dir / "manual_review_queue.csv", review_rows, _all_fields(review_rows))
    _write_csv(
        output_dir / "manual_file_review_queue.csv",
        file_review_rows,
        _all_fields(file_review_rows),
    )
    np.savez_compressed(
        output_dir / "manual_review_signals.npz",
        event_id=np.asarray([row["event_id"] for row in review_rows]),
        signals=(
            np.stack(review_signals).astype(np.float32)
            if review_signals
            else np.empty((0, review_crop_length), dtype=np.float32)
        ),
    )
    np.savez_compressed(
        output_dir / "manual_file_review_signals.npz",
        record_id=np.asarray([row["record_id"] for row in file_review_rows]),
        signals=(
            np.stack(file_review_signals).astype(np.float32)
            if file_review_signals
            else np.empty((0, 0), dtype=np.float32)
        ),
    )
    summary = {
        "schema_version": 1,
        "source_index": str(source_index_csv),
        "raw_dataset_root": str(raw_dataset_root) if raw_dataset_root is not None else None,
        "raw_datasets": sorted(raw_roots) if raw_roots else sorted(
            {row.get("raw_dataset", "") for row in source_rows if row.get("raw_dataset")}
        ),
        "n_files": len(file_rows),
        "n_files_with_candidates": sum(row["n_candidates"] > 0 for row in file_rows),
        "n_candidates": len(candidate_rows),
        "candidate_quality_counts": dict(sorted(Counter(row["quality"] for row in candidate_rows).items())),
        "candidate_source_group_counts": dict(
            sorted(Counter(row["source_group"] for row in candidate_rows).items())
        ),
        "no_candidate_reason_counts": dict(
            sorted(Counter(row["no_candidate_reason"] for row in file_rows if row["no_candidate_reason"]).items())
        ),
        "n_manual_review_rows": len(review_rows),
        "n_manual_file_review_rows": len(file_review_rows),
        "candidate_review_per_stratum": review_per_stratum,
        "file_review_per_stratum": full_trace_review_per_stratum,
        "review_sampling_excluded_record_count": len(
            excluded_record_ids.intersection(row["record_id"] for row in source_rows)
        ),
        "review_sampling_eligible_file_count": sum(
            row["record_id"] not in excluded_record_ids for row in source_rows
        ),
        "manual_review_status": "pending",
        "width_ms_quantiles": _quantiles([float(row["width_ms"]) for row in candidate_rows]),
        "snr_proxy_quantiles": _quantiles([float(row["snr_proxy"]) for row in candidate_rows]),
        "input_window_coverage": _coverage(candidate_rows, config.sampling_frequency_hz),
        "detection_config": asdict(config),
        "scientific_scope": "candidate extraction only; condition folders are not event-level labels",
    }
    (output_dir / "candidate_audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
