from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, replace
from typing import Any, Iterable

import numpy as np

from .yeast_events import (
    detect_yeast_events,
    detector_trace,
    event_bounds,
    review_calibrated_detection_config_v1,
)


SEED = 20260815
ANCHOR_RECORD_ID = "9459e76ce29342debc90"
GROUP_QUOTAS = {"mix": 40, "shmoo2": 25, "budding": 25, "shmoo": 10}
VERDICTS = ("extension_signal", "extension_margin", "uncertain")
REVIEWED_EXPANSION_SNR_Z = 1.5
REVIEWED_BOUNDARY_PAD_MS = 0.04


def _candidate_payload(candidate: Any) -> dict[str, Any]:
    return {
        key: (None if isinstance(value, float) and not math.isfinite(value) else value)
        for key, value in asdict(candidate).items()
    }


def stable_key(record_id: str, *, seed: int = SEED) -> str:
    return hashlib.sha256(f"{seed}:{record_id}".encode()).hexdigest()


def select_traces(
    eligible: Iterable[dict[str, str]],
    *,
    quotas: dict[str, int] = GROUP_QUOTAS,
    anchor_record_id: str = ANCHOR_RECORD_ID,
    seed: int = SEED,
) -> list[dict[str, str]]:
    rows = list(eligible)
    by_group = {
        group: [row for row in rows if row["source_group"] == group]
        for group in quotas
    }
    selected: list[dict[str, str]] = []
    for group, quota in quotas.items():
        candidates = by_group[group]
        if len(candidates) < quota:
            raise ValueError(f"not enough eligible {group} traces: {len(candidates)} < {quota}")
        group_selected: list[dict[str, str]] = []
        anchor = next((row for row in candidates if row["record_id"] == anchor_record_id), None)
        if anchor is not None:
            group_selected.append(anchor)
        excluded = {row["record_id"] for row in group_selected}
        ranked = sorted(
            (row for row in candidates if row["record_id"] not in excluded),
            key=lambda row: stable_key(row["record_id"], seed=seed),
        )
        group_selected.extend(ranked[: quota - len(group_selected)])
        selected.extend(group_selected)
    if sum(quotas.values()) != len(selected):
        raise RuntimeError("trace selection cardinality drifted")
    return sorted(selected, key=lambda row: (row["source_group"], stable_key(row["record_id"], seed=seed)))


def detector_positive(signal: np.ndarray) -> bool:
    config = review_calibrated_detection_config_v1()
    trace = detector_trace(signal, config)
    return bool(event_bounds(trace, int(np.asarray(signal).size)))


def compare_trace(signal: np.ndarray, *, record_id: str, source_group: str) -> list[dict[str, Any]]:
    # Pinned to the parameters the review was run under, so the comparison
    # stays reproducible now that the preset neither expands boundaries nor
    # pads them. boundary_pad_ms is pinned to the reviewed 0.04 for the same
    # reason as boundary_snr_z: the preset has since moved to 0.0, and letting
    # that through would shift both arms of this comparison by 80 samples per
    # side and stop it reproducing the r2 artifact.
    base = replace(
        review_calibrated_detection_config_v1(), boundary_pad_ms=REVIEWED_BOUNDARY_PAD_MS
    )
    with_expansion = replace(
        base, boundary_expansion_enabled=True, boundary_snr_z=REVIEWED_EXPANSION_SNR_Z
    )
    without_expansion = replace(base, boundary_expansion_enabled=False)
    current, current_error = detect_yeast_events(signal, with_expansion)
    plain, plain_error = detect_yeast_events(signal, without_expansion)
    if current_error or plain_error:
        raise RuntimeError(f"detector failed for {record_id}: current={current_error!r}, plain={plain_error!r}")
    if len(current) != len(plain):
        raise RuntimeError(f"candidate count changed for {record_id}: {len(plain)} -> {len(current)}")
    rows: list[dict[str, Any]] = []
    for index, (before, after) in enumerate(zip(plain, current, strict=True)):
        if max(before.event_start, after.event_start) >= min(before.event_end, after.event_end):
            raise RuntimeError(f"candidate pairing lost overlap for {record_id}:{index:02d}")
        rows.append(
            {
                "case_id": f"{record_id}:{index:02d}",
                "record_id": record_id,
                "source_group": source_group,
                "event_index": index,
                "without_expansion": _candidate_payload(before),
                "current": _candidate_payload(after),
                "width_delta_samples": after.width_samples - before.width_samples,
                "width_delta_ms": after.width_ms - before.width_ms,
                "left_extension_samples": max(0, before.event_start - after.event_start),
                "right_extension_samples": max(0, after.event_end - before.event_end),
                "center_delta_samples": after.center_index - before.center_index,
                "snr_proxy_delta": after.snr_proxy - before.snr_proxy,
                "energy_concentration_delta": after.energy_concentration - before.energy_concentration,
                "quality_changed": after.quality != before.quality,
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]], *, trace_count: int) -> dict[str, Any]:
    changed = [row for row in rows if row["width_delta_samples"] != 0]
    deltas = np.asarray([row["width_delta_ms"] for row in changed], dtype=np.float64)
    by_group: dict[str, dict[str, int]] = {}
    for group in GROUP_QUOTAS:
        group_rows = [row for row in rows if row["source_group"] == group]
        by_group[group] = {
            "events": len(group_rows),
            "width_changed": sum(row["width_delta_samples"] != 0 for row in group_rows),
            "quality_changed": sum(bool(row["quality_changed"]) for row in group_rows),
        }
    return {
        "trace_count": trace_count,
        "event_count": len(rows),
        "width_changed_count": len(changed),
        "width_changed_fraction": len(changed) / len(rows) if rows else 0.0,
        "width_delta_ms_mean_changed": float(np.mean(deltas)) if deltas.size else 0.0,
        "width_delta_ms_median_changed": float(np.median(deltas)) if deltas.size else 0.0,
        "width_delta_ms_max": float(np.max(deltas)) if deltas.size else 0.0,
        "quality_changed_count": sum(bool(row["quality_changed"]) for row in rows),
        "by_group": by_group,
        "claim_boundary": "Quota-stratified 100-trace development cohort; unweighted totals are not corpus prevalence.",
    }
