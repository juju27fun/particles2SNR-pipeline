from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from .particle_events import (
    ParticleDetectionConfig,
    ParticleEventCandidate,
    config_fingerprint,
    detect_particle_events,
)
from .particle_candidate_dataset import read_repair_regions
from .saturation_cleaning import proposal_center_inside_intervals


CLASS_IDS = {"2um": 0, "4um": 1, "10um": 2}
SIGNAL_LENGTH = 16_384
ROI_LENGTH = 6_144
AUDIT_SEED = "particle-mad-v2-admissions-audit60-v1"
CHANGE_AUDIT_SEED = "particle-mad-v2.1-source-correction-audit60-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable(path: Path, workspace_root: Path) -> str:
    try:
        return path.resolve().relative_to(workspace_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"path is outside workspace: {path}") from exc


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str] | None = None) -> None:
    fieldnames = list(fields or ())
    if not fieldnames:
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    if not fieldnames:
        raise ValueError(f"cannot infer fields for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def interval_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    intersection = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return intersection / union if union > 0 else 0.0


def _center(bounds: tuple[int, int]) -> float:
    return (bounds[0] + bounds[1]) / 2.0


def match_events(
    old_events: list[dict[str, Any]],
    new_events: list[dict[str, Any]],
) -> list[tuple[int, int, float]]:
    """Match event identities, prioritising cardinality then overlap quality."""
    if not old_events or not new_events:
        return []
    utility = np.zeros((len(old_events), len(new_events)), dtype=np.float64)
    ious = np.zeros_like(utility)
    for old_index, old in enumerate(old_events):
        old_bounds = (int(old["event_start"]), int(old["event_end"]))
        for new_index, new in enumerate(new_events):
            new_bounds = (int(new["event_start"]), int(new["event_end"]))
            iou = interval_iou(old_bounds, new_bounds)
            center_delta = abs(_center(old_bounds) - _center(new_bounds))
            ious[old_index, new_index] = iou
            if iou > 0.0 or center_delta <= 512:
                utility[old_index, new_index] = 1_000.0 + iou - center_delta * 1e-7
    old_indices, new_indices = linear_sum_assignment(utility, maximize=True)
    return [
        (int(old_index), int(new_index), float(ious[old_index, new_index]))
        for old_index, new_index in zip(old_indices, new_indices, strict=True)
        if utility[old_index, new_index] > 0.0
    ]


def _candidate_row(
    candidate: ParticleEventCandidate,
    *,
    source: dict[str, str],
    dataset_id: str,
) -> dict[str, Any]:
    payload = asdict(candidate)
    token = ":".join(
        (
            dataset_id,
            source["source_sha256"],
            str(candidate.candidate_index),
            str(candidate.event_start),
            str(candidate.event_end),
        )
    )
    return {
        "event_id": f"mad-event-{hashlib.sha256(token.encode()).hexdigest()[:20]}",
        "source_id": source["source_id"],
        "output_stem": source["output_stem"],
        "source_sha256": source["source_sha256"],
        "source_class": source["source_class"],
        "class_id": int(source["class_id"]),
        "output_split": source["output_split"],
        **payload,
    }


def _campaign(source_id: str) -> str:
    prefix, separator, suffix = source_id.rpartition("_")
    return prefix if separator and suffix.isdigit() else source_id


def _admission_mechanism(
    event: dict[str, Any], old_candidates: list[ParticleEventCandidate]
) -> tuple[str, str, int]:
    bounds = (int(event["event_start"]), int(event["event_end"]))
    overlaps = [
        candidate
        for candidate in old_candidates
        if interval_iou(bounds, (candidate.event_start, candidate.event_end)) > 0.0
    ]
    rejected = [candidate for candidate in overlaps if candidate.quality != "retained"]
    if len(rejected) == 1:
        reason = rejected[0].rejection_reason or "unspecified"
        return "previously_rejected", reason, 1
    if len(rejected) > 1:
        reasons = "+".join(sorted({candidate.rejection_reason or "unspecified" for candidate in rejected}))
        return "split_from_rejected_support", reasons, len(rejected)
    if overlaps:
        return "changed_cardinality_near_retained", "", len(overlaps)
    return "new_support_after_geometry_change", "", 0


def _stratified_hash_sample(
    rows: list[dict[str, Any]], *, quota: int, seed: str
) -> list[dict[str, Any]]:
    strata: dict[tuple[str, str, bool], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[(row["output_split"], row["campaign"], bool(row["v1_empty_trace"]))].append(row)
    for key, members in strata.items():
        members.sort(
            key=lambda row: hashlib.sha256(
                f"{seed}:{key}:{row['new_event_id']}".encode()
            ).hexdigest()
        )
    selected: list[dict[str, Any]] = []
    keys = sorted(strata)
    while len(selected) < quota:
        progressed = False
        for key in keys:
            if strata[key] and len(selected) < quota:
                selected.append(strata[key].pop(0))
                progressed = True
        if not progressed:
            raise ValueError(f"audit quota {quota} exceeds eligible population")
    return selected


def select_audit_cases(additions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mandatory = {
        row["new_event_id"]: row
        for row in additions
        if row["output_split"] == "test" or row["source_class"] == "4um"
    }
    development = [
        row for row in additions if row["output_split"] in {"train", "val"}
    ]
    selected = list(mandatory.values())
    for class_name in ("2um", "10um"):
        eligible = [
            row
            for row in development
            if row["source_class"] == class_name and row["new_event_id"] not in mandatory
        ]
        selected.extend(
            _stratified_hash_sample(
                eligible, quota=11, seed=f"{AUDIT_SEED}:{class_name}"
            )
        )
    unique = {row["new_event_id"]: row for row in selected}
    if len(unique) != 60:
        raise RuntimeError(f"audit sample must contain 60 unique events, got {len(unique)}")
    ordered = sorted(
        unique.values(),
        key=lambda row: (
            row["output_split"] != "test",
            row["source_class"] != "4um",
            row["source_class"],
            row["source_id"],
            int(row["new_event_start"]),
        ),
    )
    return [
        {
            **row,
            "audit_order": index + 1,
            "case_id": f"mad-v2-admission-{row['new_event_id'].removeprefix('mad-event-')}",
            "selection_seed": AUDIT_SEED,
        }
        for index, row in enumerate(ordered)
    ]


def select_change_audit_cases(
    diff_rows: list[dict[str, Any]],
    veto_rows: list[dict[str, Any]],
    *,
    quota: int = 60,
) -> list[dict[str, Any]]:
    """Select a deterministic, stratified audit of source-driven changes."""
    changes: list[dict[str, Any]] = [
        {**row, "change_kind": row["status"]}
        for row in diff_rows
        if row["status"] in {"added", "lost"}
    ]
    changes.extend(
        {
            **row,
            "status": "saturation_center_vetoed",
            "change_kind": "saturation_center_vetoed",
        }
        for row in veto_rows
    )
    if len(changes) < quota:
        raise ValueError(
            f"change audit quota {quota} exceeds population {len(changes)}"
        )
    strata: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in changes:
        strata[
            (
                str(row["change_kind"]),
                str(row["source_class"]),
                str(row["output_split"]),
            )
        ].append(row)
    for key, members in strata.items():
        members.sort(
            key=lambda row: hashlib.sha256(
                (
                    f"{CHANGE_AUDIT_SEED}:{key}:"
                    f"{row.get('new_event_id') or row.get('old_event_id') or row.get('veto_id')}"
                ).encode()
            ).hexdigest()
        )
    selected: list[dict[str, Any]] = []
    keys = sorted(strata)
    while len(selected) < quota:
        progressed = False
        for key in keys:
            if strata[key] and len(selected) < quota:
                selected.append(strata[key].pop(0))
                progressed = True
        if not progressed:
            raise RuntimeError("change audit sampler exhausted unexpectedly")
    ordered = sorted(
        selected,
        key=lambda row: (
            row["output_split"] != "test",
            row["change_kind"],
            row["source_class"],
            row["source_id"],
            int(
                row.get("new_event_start")
                or row.get("old_event_start")
                or row.get("event_start")
                or 0
            ),
        ),
    )
    result = []
    for index, row in enumerate(ordered):
        identity = (
            row.get("new_event_id")
            or row.get("old_event_id")
            or row.get("veto_id")
        )
        result.append(
            {
                **row,
                "audit_order": index + 1,
                "case_id": f"mad-v2.1-change-{identity}",
                "selection_seed": CHANGE_AUDIT_SEED,
            }
        )
    return result


def _tree_files(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*") if path.is_file() and path.name != "dataset-manifest.json"
    )


def payload_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in _tree_files(root):
        relative = path.relative_to(root).as_posix()
        digest.update(f"{relative}\0{sha256_file(path)}\n".encode())
    return digest.hexdigest()


def build_mad_teacher_dataset(
    *,
    workspace_root: Path,
    source_dataset_root: Path,
    predecessor_root: Path,
    output_dir: Path,
    dataset_id: str,
    source_dataset_id: str,
    source_manifest_sha256: str,
    config: ParticleDetectionConfig,
    expected: dict[str, Any] | None = None,
    source_manifest_path: Path | None = None,
    repair_manifest: Path | None = None,
    saturation_center_veto: bool = False,
    audit_mode: str = "v2_admissions",
) -> dict[str, Any]:
    workspace_root = workspace_root.resolve()
    source_dataset_root = source_dataset_root.resolve()
    predecessor_root = predecessor_root.resolve()
    output_dir = output_dir.resolve()
    optional_inputs = tuple(
        path.resolve()
        for path in (source_manifest_path, repair_manifest)
        if path is not None
    )
    for path in (
        source_dataset_root,
        predecessor_root,
        output_dir.parent,
        *optional_inputs,
    ):
        _portable(path, workspace_root)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable dataset: {output_dir}")
    predecessor_manifest_path = predecessor_root / "dataset-manifest.json"
    predecessor_manifest = json.loads(predecessor_manifest_path.read_text(encoding="utf-8"))
    if source_manifest_path is not None:
        if source_manifest_path.resolve() != (
            source_dataset_root / "source_manifest.csv"
        ).resolve():
            raise ValueError("corrected source manifest must belong to its source dataset")
        source_dataset_manifest = source_dataset_root / "dataset-manifest.json"
        if not source_dataset_manifest.is_file():
            raise FileNotFoundError(source_dataset_manifest)
        if sha256_file(source_dataset_manifest) != source_manifest_sha256:
            raise ValueError("corrected source dataset manifest hash mismatch")
    if saturation_center_veto and repair_manifest is not None:
        if repair_manifest.resolve() != (
            source_dataset_root / "saturation_repair_manifest.csv"
        ).resolve():
            raise ValueError("saturation veto manifest must belong to its source dataset")
    old_config = ParticleDetectionConfig(**predecessor_manifest["config"])
    predecessor_source_rows = _read_csv(predecessor_root / "source_manifest.csv")
    source_rows = _read_csv(source_manifest_path) if source_manifest_path else predecessor_source_rows
    predecessor_by_id = {row["source_id"]: row for row in predecessor_source_rows}
    if set(predecessor_by_id) != {row["source_id"] for row in source_rows}:
        raise ValueError("corrected source population differs from predecessor")
    for source in source_rows:
        predecessor_source = predecessor_by_id[source["source_id"]]
        for field in (
            "source_id",
            "output_stem",
            "source_path",
            "source_class",
            "class_id",
            "source_split",
            "output_split",
        ):
            if source[field] != predecessor_source[field]:
                raise ValueError(
                    f"corrected source identity drift for {source['source_id']}: {field}"
                )
    repair_regions = read_repair_regions(repair_manifest)
    if saturation_center_veto and repair_manifest is None:
        raise ValueError("saturation center veto requires a repair manifest")
    for source in source_rows:
        expected_repairs = int(source.get("repair_region_count", 0))
        observed_repairs = len(repair_regions.get(f"{source['source_id']}.npy", ()))
        if expected_repairs != observed_repairs:
            raise ValueError(
                f"repair interval count mismatch for {source['source_id']}: "
                f"{expected_repairs} != {observed_repairs}"
            )
    old_events_by_stem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_csv(predecessor_root / "events.csv"):
        old_events_by_stem[row["output_stem"]].append(row)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        event_rows: list[dict[str, Any]] = []
        output_sources: list[dict[str, Any]] = []
        diff_rows: list[dict[str, Any]] = []
        additions: list[dict[str, Any]] = []
        veto_rows: list[dict[str, Any]] = []
        class_widths: dict[str, list[int]] = defaultdict(list)
        for source in source_rows:
            source_path = source_dataset_root / source["source_path"]
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            if sha256_file(source_path) != source["source_sha256"]:
                raise ValueError(f"source hash drift: {source['source_id']}")
            signal = np.asarray(np.load(source_path, allow_pickle=False)).squeeze()
            if signal.size != SIGNAL_LENGTH:
                raise ValueError(f"unexpected signal length for {source['source_id']}: {signal.size}")
            filename = f"{source['source_id']}.npy"
            regions = repair_regions.get(filename, ())
            new_candidates, _ = detect_particle_events(
                signal, config, repair_regions=regions
            )
            predecessor_source = predecessor_by_id[source["source_id"]]
            old_signal_path = (
                predecessor_root
                / predecessor_source["output_split"]
                / "signals"
                / f"{predecessor_source['output_stem']}.npy"
            )
            if sha256_file(old_signal_path) != predecessor_source["output_signal_sha256"]:
                raise ValueError(f"predecessor signal hash drift: {source['source_id']}")
            old_signal = np.asarray(
                np.load(old_signal_path, allow_pickle=False)
            ).squeeze()
            old_candidates, _ = detect_particle_events(old_signal, old_config)
            retained = []
            source_vetoes = 0
            for candidate in new_candidates:
                if candidate.quality != "retained":
                    continue
                center_sample, repair_index, repair_interval = (
                    proposal_center_inside_intervals(
                        candidate.event_start,
                        candidate.event_end,
                        regions,
                    )
                )
                if saturation_center_veto and repair_interval is not None:
                    veto_id = (
                        f"mad-veto-{hashlib.sha256((dataset_id + ':' + source['source_id'] + ':' + str(candidate.candidate_index) + ':' + str(candidate.event_start) + ':' + str(candidate.event_end)).encode()).hexdigest()[:20]}"
                    )
                    veto_rows.append(
                        {
                            "veto_id": veto_id,
                            "source_id": source["source_id"],
                            "output_stem": source["output_stem"],
                            "source_path": source["source_path"],
                            "source_sha256": source["source_sha256"],
                            "source_class": source["source_class"],
                            "class_id": int(source["class_id"]),
                            "output_split": source["output_split"],
                            "candidate_index": candidate.candidate_index,
                            "event_start": candidate.event_start,
                            "event_end": candidate.event_end,
                            "center_sample": center_sample,
                            "repair_index": repair_index,
                            "expanded_start_sample": repair_interval[0],
                            "expanded_end_sample": repair_interval[1],
                            "robust_energy_z": candidate.robust_energy_z,
                            "repair_overlap": candidate.repair_overlap,
                            "repair_overlap_fraction": candidate.repair_overlap_fraction,
                            "reason": "z8_center_inside_saturation_repair",
                        }
                    )
                    source_vetoes += 1
                    continue
                retained.append(
                    _candidate_row(candidate, source=source, dataset_id=dataset_id)
                )
            old_retained = old_events_by_stem.get(source["output_stem"], [])
            matches = match_events(old_retained, retained)
            matched_old = {old_index for old_index, _, _ in matches}
            matched_new = {new_index for _, new_index, _ in matches}
            match_by_new = {new_index: (old_index, iou) for old_index, new_index, iou in matches}
            for new_index, event in enumerate(retained):
                new_bounds = (int(event["event_start"]), int(event["event_end"]))
                common = {
                    "source_id": source["source_id"],
                    "output_stem": source["output_stem"],
                    "source_path": source["source_path"],
                    "source_sha256": source["source_sha256"],
                    "source_class": source["source_class"],
                    "class_id": int(source["class_id"]),
                    "output_split": source["output_split"],
                    "campaign": _campaign(source["source_id"]),
                    "v1_empty_trace": len(old_retained) == 0,
                    "new_event_id": event["event_id"],
                    "new_event_start": new_bounds[0],
                    "new_event_end": new_bounds[1],
                    "new_width_samples": new_bounds[1] - new_bounds[0],
                    "new_robust_energy_z": float(event["robust_energy_z"]),
                    "new_energy_concentration": float(event["energy_concentration"]),
                    "new_dominant_frequency_hz": float(event["dominant_frequency_hz"]),
                    "roi_start_unclipped": int(round(_center(new_bounds) - ROI_LENGTH / 2)),
                    "roi_end_unclipped": int(round(_center(new_bounds) + ROI_LENGTH / 2)),
                }
                if new_index in match_by_new:
                    old_index, iou = match_by_new[new_index]
                    old = old_retained[old_index]
                    old_bounds = (int(old["event_start"]), int(old["event_end"]))
                    diff_rows.append(
                        {
                            **common,
                            "status": "matched",
                            "old_event_id": old["event_id"],
                            "old_event_start": old_bounds[0],
                            "old_event_end": old_bounds[1],
                            "old_width_samples": old_bounds[1] - old_bounds[0],
                            "iou": iou,
                            "center_delta_samples": abs(_center(old_bounds) - _center(new_bounds)),
                            "width_delta_samples": (new_bounds[1] - new_bounds[0]) - (old_bounds[1] - old_bounds[0]),
                            "admission_mechanism": "",
                            "previous_rejection_reason": "",
                            "overlapping_old_candidates": "",
                        }
                    )
                else:
                    mechanism, reason, overlap_count = _admission_mechanism(event, old_candidates)
                    addition = {
                        **common,
                        "status": "added",
                        "old_event_id": "",
                        "old_event_start": "",
                        "old_event_end": "",
                        "old_width_samples": "",
                        "iou": "",
                        "center_delta_samples": "",
                        "width_delta_samples": "",
                        "admission_mechanism": mechanism,
                        "previous_rejection_reason": reason,
                        "overlapping_old_candidates": overlap_count,
                    }
                    diff_rows.append(addition)
                    additions.append(addition)
            for old_index, old in enumerate(old_retained):
                if old_index not in matched_old:
                    diff_rows.append(
                        {
                            "source_id": source["source_id"],
                            "output_stem": source["output_stem"],
                            "source_path": source["source_path"],
                            "source_sha256": source["source_sha256"],
                            "source_class": source["source_class"],
                            "class_id": int(source["class_id"]),
                            "output_split": source["output_split"],
                            "campaign": _campaign(source["source_id"]),
                            "v1_empty_trace": False,
                            "status": "lost",
                            "old_event_id": old["event_id"],
                            "old_event_start": int(old["event_start"]),
                            "old_event_end": int(old["event_end"]),
                            "old_width_samples": int(old["event_end"]) - int(old["event_start"]),
                            "new_event_id": "",
                            "new_event_start": "",
                            "new_event_end": "",
                            "new_width_samples": "",
                            "new_robust_energy_z": "",
                            "new_energy_concentration": "",
                            "new_dominant_frequency_hz": "",
                            "roi_start_unclipped": "",
                            "roi_end_unclipped": "",
                            "iou": "",
                            "center_delta_samples": "",
                            "width_delta_samples": "",
                            "admission_mechanism": "",
                            "previous_rejection_reason": "",
                            "overlapping_old_candidates": "",
                        }
                    )

            split = source["output_split"]
            signal_dir = temporary / split / "signals"
            label_dir = temporary / split / "labels"
            signal_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            output_signal = signal_dir / f"{source['output_stem']}.npy"
            shutil.copy2(source_path, output_signal)
            label_lines = []
            for event in sorted(retained, key=lambda row: (int(row["event_start"]), int(row["candidate_index"]))):
                width = int(event["event_end"]) - int(event["event_start"])
                midpoint = (int(event["event_start"]) + int(event["event_end"])) / 2.0
                label_lines.append(
                    f"{int(event['class_id'])} {midpoint / SIGNAL_LENGTH:.12f} {width / SIGNAL_LENGTH:.12f}"
                )
                event_rows.append(event)
                class_widths[source["source_class"]].append(width)
            output_label = label_dir / f"{source['output_stem']}.txt"
            output_label.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
            output_sources.append(
                {
                    **source,
                    "retained_events": len(retained),
                    "rejected_candidates": sum(candidate.quality != "retained" for candidate in new_candidates),
                    "empty_mad_label": len(retained) == 0,
                    "repair_region_count": len(regions),
                    "saturation_center_vetoed": source_vetoes,
                    "output_signal_sha256": sha256_file(output_signal),
                    "output_label_sha256": sha256_file(output_label),
                }
            )

        if audit_mode == "v2_admissions":
            audit_cases = select_audit_cases(additions)
        elif audit_mode == "source_correction_changes":
            audit_cases = select_change_audit_cases(diff_rows, veto_rows)
        else:
            raise ValueError(f"unknown audit mode: {audit_mode}")
        _write_csv(temporary / "source_manifest.csv", output_sources)
        _write_csv(temporary / "events.csv", event_rows)
        _write_csv(temporary / "event_diff_v1_v2.csv", diff_rows)
        _write_csv(temporary / "audit_sample_60.csv", audit_cases)
        _write_csv(
            temporary / "saturation_veto_exclusions.csv",
            veto_rows,
            fields=(
                "veto_id",
                "source_id",
                "output_stem",
                "source_path",
                "source_sha256",
                "source_class",
                "class_id",
                "output_split",
                "candidate_index",
                "event_start",
                "event_end",
                "center_sample",
                "repair_index",
                "expanded_start_sample",
                "expanded_end_sample",
                "robust_energy_z",
                "repair_overlap",
                "repair_overlap_fraction",
                "reason",
            ),
        )

        counts = {
            "traces_total": len(output_sources),
            "traces_by_split": dict(Counter(row["output_split"] for row in output_sources)),
            "events_total": len(event_rows),
            "events_by_class": dict(Counter(row["source_class"] for row in event_rows)),
            "events_by_split": dict(Counter(row["output_split"] for row in event_rows)),
            "mad_empty_traces": sum(str(row["empty_mad_label"]).lower() == "true" for row in output_sources),
            "additions": len(additions),
            "additions_by_class": dict(Counter(row["source_class"] for row in additions)),
            "additions_by_split": dict(Counter(row["output_split"] for row in additions)),
            "losses": sum(row["status"] == "lost" for row in diff_rows),
            "audit_cases": len(audit_cases),
            "saturation_center_vetoed": len(veto_rows),
            "repaired_traces": sum(int(row.get("repair_region_count", 0)) > 0 for row in output_sources),
        }
        for class_name in CLASS_IDS:
            counts[f"events_{class_name}"] = counts["events_by_class"].get(class_name, 0)
            counts[f"additions_{class_name}"] = counts["additions_by_class"].get(class_name, 0)
        same_cell_collisions = 0
        for stem, events in _group_by(event_rows, "output_stem").items():
            cells = [int(((int(row["event_start"]) + int(row["event_end"])) / 2.0) // 32) for row in events]
            same_cell_collisions += len(cells) - len(set(cells))
        counts["same_yolo_cell_collisions"] = same_cell_collisions
        if expected is not None:
            mismatches = {
                key: {"expected": value, "observed": counts.get(key)}
                for key, value in expected.items()
                if counts.get(key) != value
            }
            if mismatches:
                raise RuntimeError(f"v2 invariant mismatch: {mismatches}")

        contract = {
            "schema_version": 1,
            "dataset_id": dataset_id,
            "format": "yolo-1d-mad-teacher-development",
            "grain": {
                "source_manifest.csv": "one source trace",
                "events.csv": "one retained MAD event",
                "event_diff_v1_v2.csv": "one matched, added, or lost event identity",
                "audit_sample_60.csv": "one descriptively reviewed new admission",
                "saturation_veto_exclusions.csv": "one retained MAD proposal excluded by the frozen z8v2 centre rule",
            },
            "keys": {
                "source_manifest.csv": ["source_id"],
                "events.csv": ["event_id"],
                "audit_sample_60.csv": ["case_id"],
                "saturation_veto_exclusions.csv": ["veto_id"],
            },
            "units": {
                "event_start": "sample, inclusive",
                "event_end": "sample, exclusive",
                "width_samples": "sample",
                "width_ms": "ms",
                "dominant_frequency_hz": "Hz",
            },
            "classes": CLASS_IDS,
            "nullability": {
                "event_diff_v1_v2.old_event_id": "null-equivalent empty string only for additions",
                "event_diff_v1_v2.new_event_id": "null-equivalent empty string only for losses",
            },
            "duplicate_policy": "source_id, event_id, and case_id duplicates are forbidden",
            "label_policy": "MAD retained supports become YOLO boxes; acquisition folder supplies class only after detection",
            "audit_policy": "descriptive only; audit decisions never mutate dataset labels",
            "saturation_policy": (
                "signals are repaired before annotation; retained proposals whose bounds midpoint "
                "falls inclusively inside an expanded repair interval are excluded"
                if saturation_center_veto
                else "no saturation center veto"
            ),
        }
        # Keep the workspace-standard hyphenated contract and the explicit
        # experiment-interface spelling. They are byte-identical by design.
        _write_json(temporary / "dataset-contract.json", contract)
        _write_json(temporary / "dataset_contract.json", contract)
        dataset_yaml = {
            "path": ".",
            "dataset_id": dataset_id,
            "status": "immutable_candidate",
            "format": "yolo-1d-mad-teacher-development",
            "names": ["2um", "4um", "10um"],
            "signal_lengths": {"input": SIGNAL_LENGTH, "source_segment": SIGNAL_LENGTH},
            "splits": {
                split: {"total": count}
                for split, count in sorted(counts["traces_by_split"].items())
            },
            "generation_params": {
                "annotation_source": "MAD active frames only",
                "mad_config_sha256": config_fingerprint(config),
                "class_source": "registered acquisition class folder",
                "negative_policy": "zero retained MAD proposals yields an empty YOLO label",
                "saturation_center_veto": saturation_center_veto,
                "saturation_center_veto_rule": (
                    "inclusive midpoint of proposal bounds within expanded repair interval"
                    if saturation_center_veto
                    else None
                ),
            },
            "provenance": {
                "parent_dataset_id": source_dataset_id,
                "parent_manifest_sha256": source_manifest_sha256,
                "repair_manifest_sha256": (
                    sha256_file(repair_manifest) if repair_manifest else None
                ),
                "predecessor_dataset_id": predecessor_manifest["dataset_id"],
                "predecessor_manifest_sha256": sha256_file(predecessor_manifest_path),
            },
        }
        _write_json(temporary / "dataset.yaml", dataset_yaml)
        digest = payload_digest(temporary)
        payload_files = _tree_files(temporary)
        manifest = {
            "schema_version": 1,
            "dataset_id": dataset_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_dataset_id": source_dataset_id,
            "source_manifest_sha256": source_manifest_sha256,
            "predecessor_dataset_id": predecessor_manifest["dataset_id"],
            "predecessor_manifest_sha256": sha256_file(predecessor_manifest_path),
            "config": asdict(config),
            "config_sha256": config_fingerprint(config),
            "audit_selection_seed": (
                AUDIT_SEED
                if audit_mode == "v2_admissions"
                else CHANGE_AUDIT_SEED
            ),
            "audit_mode": audit_mode,
            "repair_manifest": (
                _portable(repair_manifest, workspace_root) if repair_manifest else None
            ),
            "repair_manifest_sha256": (
                sha256_file(repair_manifest) if repair_manifest else None
            ),
            "saturation_center_veto": saturation_center_veto,
            "counts": counts,
            "expected_invariants": expected,
            "payload_digest_sha256": digest,
            "files": {
                path.relative_to(temporary).as_posix(): {
                    "sha256": sha256_file(path),
                    "size": path.stat().st_size,
                }
                for path in payload_files
            },
            "claim_boundary": "MAD pseudo-label reference and within-corpus diagnostic only; no physical ground-truth or morphology-generalization claim.",
        }
        _write_json(temporary / "dataset-manifest.json", manifest)
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def _group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return dict(grouped)
