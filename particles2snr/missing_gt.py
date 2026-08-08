"""Versioned missing-ground-truth adjudication helpers.

The review journal is candidate-oriented because several model predictions can
point at the same historical event.  This module converts it into an
event-oriented, immutable overlay and applies only explicit source-GT actions.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import yaml


EXPECTED_CANDIDATE_COUNTS = {
    "not_particle": 3,
    "real_particle": 14,
    "uncertain": 1,
}
EXPECTED_UNIQUE_COUNTS = {
    "not_particle": 3,
    "real_particle": 13,
    "uncertain": 1,
}
SOURCE_RESTORE_CLASSES = {"2um": 0, "4um": 1, "10um": 2}


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw.strip():
            continue
        try:
            rows.append(json.loads(raw))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            )


def latest_decisions(session_dir: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(session_dir / "decisions.jsonl"):
        candidate_id = str(row["candidate_id"])
        if (
            candidate_id not in latest
            or int(row["revision"]) > int(latest[candidate_id]["revision"])
        ):
            latest[candidate_id] = row
    return latest


def load_locked_review(
    session_dir: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest_path = session_dir / "review_manifest.json"
    run_path = session_dir / "run.json"
    summary_path = session_dir / "session_summary.json"
    for path in (manifest_path, run_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run = json.loads(run_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    decisions = latest_decisions(session_dir)
    if not summary.get("complete"):
        raise ValueError("review session is incomplete")
    if run.get("status") != "analysis_complete":
        raise ValueError("review analysis is not complete")
    if run.get("review_manifest_sha256") != manifest.get("manifest_sha256"):
        raise ValueError("run and review manifest hashes differ")
    candidate_ids = {
        str(candidate["candidate_id"]) for candidate in manifest["candidates"]
    }
    if set(decisions) != candidate_ids:
        raise ValueError("latest decisions do not cover the review manifest")
    counts = Counter(row["existence"] for row in decisions.values())
    if counts != EXPECTED_CANDIDATE_COUNTS:
        raise ValueError(f"unexpected review decisions: {dict(counts)}")
    return manifest, decisions


def _event_identity(
    candidate: dict[str, Any],
    *,
    historical_manifest_sha256: str,
) -> dict[str, Any]:
    fs = float(candidate["sampling_frequency_hz"])
    start_ms, end_ms = map(float, candidate["historical_interval_ms"])
    start_sample = int(round(start_ms / 1000.0 * fs))
    end_sample = int(round(end_ms / 1000.0 * fs))
    identity = {
        "historical_manifest_sha256": historical_manifest_sha256,
        "source_id": candidate["source_id"],
        "historical_annotation_id": int(
            candidate["historical_annotation_id"]
        ),
        "historical_class_id": int(candidate["historical_class_id"]),
        "interval_samples": [start_sample, end_sample],
    }
    return {
        **identity,
        "event_id": stable_hash(identity)[:16],
        "interval_ms": [start_ms, end_ms],
    }


def build_adjudication_rows(
    manifest: dict[str, Any],
    decisions: dict[str, dict[str, Any]],
    *,
    historical_dataset_id: str,
    historical_manifest_sha256: str,
    edge_pad_samples: int = 300,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    identities: dict[str, dict[str, Any]] = {}
    for candidate in manifest["candidates"]:
        identity = _event_identity(
            candidate,
            historical_manifest_sha256=historical_manifest_sha256,
        )
        event_id = identity["event_id"]
        grouped.setdefault(event_id, []).append(
            (candidate, decisions[candidate["candidate_id"]])
        )
        identities[event_id] = identity

    rows: list[dict[str, Any]] = []
    for event_id, members in grouped.items():
        candidate = members[0][0]
        existence = {decision["existence"] for _, decision in members}
        if len(existence) != 1:
            raise ValueError(f"conflicting decisions for event {event_id}")
        estimated_classes = {
            decision.get("estimated_class")
            for _, decision in members
            if decision.get("estimated_class") not in (None, "unknown")
        }
        if len(estimated_classes) > 1:
            raise ValueError(f"conflicting class decisions for event {event_id}")
        decision_name = next(iter(existence))
        identity = identities[event_id]
        start_sample, end_sample = identity["interval_samples"]
        source_length = int(candidate["source_length"])
        touches_edge_guard = (
            start_sample < edge_pad_samples
            or end_sample > source_length - edge_pad_samples
        )
        mechanism = str(candidate["mechanism"])
        historical_class_name = str(candidate["historical_class_name"])
        current_present = candidate.get("current_dual_clean_interval_ms") is not None

        if mechanism == "dual_clean_removed_annotation":
            source_action = (
                "add_source_positive"
                if decision_name == "real_particle"
                else "keep_absent"
            )
        elif decision_name == "not_particle":
            source_action = "keep_absent"
        else:
            source_action = "preserve_existing_label"

        class_status = "confirmed_historical"
        disputed_reason = None
        if decision_name == "uncertain":
            class_status = "disputed_existence_overlay_only"
            disputed_reason = "human_uncertain"
        elif historical_class_name == "unclear":
            class_status = "disputed_for_known3_overlay_only"
            disputed_reason = "historical_class_unclear"

        if (
            source_action == "add_source_positive"
            and historical_class_name not in SOURCE_RESTORE_CLASSES
        ):
            raise ValueError(
                f"cannot restore non-known class for event {event_id}"
            )

        rows.append(
            {
                "schema_version": 1,
                "event_id": event_id,
                "candidate_ids": sorted(
                    member["candidate_id"] for member, _ in members
                ),
                "review_orders": sorted(
                    int(member["order"]) for member, _ in members
                ),
                "mechanism": mechanism,
                "source_id": candidate["source_id"],
                "source_split": "test",
                "raw_dataset_id": candidate["raw_dataset_id"],
                "raw_signal_sha256": candidate["raw_signal_sha256"],
                "filtered_dataset_id": candidate["filtered_dataset_id"],
                "filtered_signal_sha256": candidate[
                    "filtered_signal_sha256"
                ],
                "historical_dataset_id": historical_dataset_id,
                "historical_manifest_sha256": historical_manifest_sha256,
                "historical_annotation_id": identity[
                    "historical_annotation_id"
                ],
                "historical_class_id": identity["historical_class_id"],
                "historical_class_name": historical_class_name,
                "historical_interval_samples": identity["interval_samples"],
                "historical_interval_ms": identity["interval_ms"],
                "source_length": source_length,
                "sampling_frequency_hz": float(
                    candidate["sampling_frequency_hz"]
                ),
                "human_existence": decision_name,
                "human_estimated_class": (
                    next(iter(estimated_classes))
                    if estimated_classes
                    else None
                ),
                "reviewer": members[0][1]["reviewer"],
                "reviewed_at": max(
                    decision["reviewed_at"] for _, decision in members
                ),
                "source_action": source_action,
                "class_status": class_status,
                "disputed_reason": disputed_reason,
                "current_source_label_present": current_present,
                "touches_wave8_edge_guard": touches_edge_guard,
                "wave8_policy": _wave8_policy(
                    decision_name=decision_name,
                    mechanism=mechanism,
                    historical_class_name=historical_class_name,
                    current_present=current_present,
                    touches_edge_guard=touches_edge_guard,
                ),
                "evidence": {
                    "filtered_peak_z": candidate.get("filtered_peak_z"),
                    "clean_peak_z": candidate.get("clean_peak_z"),
                    "historical_snr_db": candidate.get("historical_snr_db"),
                    "dual_clean_drop_reason": candidate.get(
                        "dual_clean_drop_reason"
                    ),
                    "edge_distance_ms": candidate.get("edge_distance_ms"),
                },
            }
        )
    rows.sort(
        key=lambda row: (
            min(row["review_orders"]),
            row["source_id"],
            row["historical_annotation_id"],
        )
    )
    counts = Counter(row["human_existence"] for row in rows)
    if counts != EXPECTED_UNIQUE_COUNTS:
        raise ValueError(f"unexpected unique decisions: {dict(counts)}")
    if sum(row["source_action"] == "add_source_positive" for row in rows) != 9:
        raise ValueError("expected exactly nine source restorations")
    return rows


def _wave8_policy(
    *,
    decision_name: str,
    mechanism: str,
    historical_class_name: str,
    current_present: bool,
    touches_edge_guard: bool,
) -> str:
    if decision_name == "not_particle":
        return "no_change"
    if decision_name == "uncertain" or historical_class_name == "unclear":
        return "disputed_overlay_only"
    if touches_edge_guard:
        return "ignore_modified_edge"
    if mechanism == "dual_clean_removed_annotation":
        return "add_positive"
    if current_present:
        return "already_scored_positive"
    return "no_change"


def yolo_line(
    class_id: int,
    start_sample: int,
    end_sample: int,
    signal_length: int,
) -> str:
    if not 0 <= start_sample < end_sample <= signal_length:
        raise ValueError("invalid source interval")
    center = (start_sample + end_sample) / (2.0 * signal_length)
    width = (end_sample - start_sample) / signal_length
    return f"{class_id} {center:.10f} {width:.10f}"


def _line_start(line: str, signal_length: int) -> float:
    _class_id, center, width = line.split()
    return (float(center) - float(width) / 2.0) * signal_length


def apply_source_restorations(
    *,
    parent_root: Path,
    output_root: Path,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    shutil.copytree(parent_root, output_root, copy_function=shutil.copy2)
    changes: list[dict[str, Any]] = []
    for row in rows:
        if row["source_action"] != "add_source_positive":
            continue
        label_path = (
            output_root
            / row["source_split"]
            / "labels"
            / f"{row['source_id']}.txt"
        )
        original_path = (
            parent_root
            / row["source_split"]
            / "labels"
            / f"{row['source_id']}.txt"
        )
        original_lines = [
            line
            for line in original_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        new_line = yolo_line(
            int(row["historical_class_id"]),
            int(row["historical_interval_samples"][0]),
            int(row["historical_interval_samples"][1]),
            int(row["source_length"]),
        )
        if new_line in original_lines:
            raise ValueError(f"restoration already exists: {row['event_id']}")
        output_lines = original_lines + [new_line]
        output_lines.sort(
            key=lambda line: _line_start(line, int(row["source_length"]))
        )
        label_path.write_text(
            "\n".join(output_lines) + "\n", encoding="utf-8"
        )
        changes.append(
            {
                "event_id": row["event_id"],
                "source_id": row["source_id"],
                "split": row["source_split"],
                "historical_annotation_id": row[
                    "historical_annotation_id"
                ],
                "class_id": row["historical_class_id"],
                "class_name": row["historical_class_name"],
                "interval_samples": row["historical_interval_samples"],
                "old_label_count": len(original_lines),
                "new_label_count": len(output_lines),
                "new_label_line": new_line,
            }
        )
    if len(changes) != 9:
        raise ValueError(f"expected nine changes, got {len(changes)}")
    return changes


def label_counts(dataset_root: Path) -> dict[str, Any]:
    class_names = ("2um", "4um", "10um", "unclear")
    result: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        paths = sorted((dataset_root / split / "labels").glob("*.txt"))
        classes: Counter[str] = Counter()
        event_count = 0
        empty = 0
        for path in paths:
            lines = [line for line in path.read_text().splitlines() if line]
            empty += not lines
            for line in lines:
                classes[class_names[int(line.split()[0])]] += 1
                event_count += 1
        result[split] = {
            "signals": len(paths),
            "background_signals": empty,
            "event_bearing_signals": len(paths) - empty,
            "events": event_count,
            "events_by_class": {
                name: classes[name] for name in class_names
            },
        }
    return result


def project_wave8_overlay(
    rows: list[dict[str, Any]],
    *,
    wave8_manifest_path: Path,
    source_length: int = 16_384,
) -> list[dict[str, Any]]:
    with wave8_manifest_path.open(newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))
    occurrences: dict[str, list[tuple[dict[str, str], int]]] = {}
    for manifest_row in manifest_rows:
        if manifest_row["split"] != "test":
            continue
        for segment_index, source_id in enumerate(
            manifest_row["ordered_source_ids"].split(";")
        ):
            occurrences.setdefault(source_id, []).append(
                (manifest_row, segment_index)
            )

    projected: list[dict[str, Any]] = []
    for row in rows:
        policy = row["wave8_policy"]
        if policy not in {"add_positive", "ignore_modified_edge"}:
            continue
        for manifest_row, segment_index in occurrences.get(
            row["source_id"], []
        ):
            start, end = map(int, row["historical_interval_samples"])
            offset = segment_index * source_length
            projected.append(
                {
                    "schema_version": 1,
                    "long_id": manifest_row["long_id"],
                    "split": manifest_row["split"],
                    "group_id": int(manifest_row["group_id"]),
                    "permutation_id": int(
                        manifest_row["permutation_id"]
                    ),
                    "segment_index": segment_index,
                    "source_id": row["source_id"],
                    "event_id": row["event_id"],
                    "class_id": row["historical_class_id"],
                    "class_name": row["historical_class_name"],
                    "source_interval_samples": [start, end],
                    "long_interval_samples": [
                        start + offset,
                        end + offset,
                    ],
                    "action": (
                        "add_positive"
                        if policy == "add_positive"
                        else "ignore_modified_edge"
                    ),
                }
            )
    projected.sort(
        key=lambda row: (
            row["long_id"],
            row["long_interval_samples"][0],
            row["event_id"],
        )
    )
    counts = Counter(row["action"] for row in projected)
    if counts != {"add_positive": 360, "ignore_modified_edge": 216}:
        raise ValueError(f"unexpected Wave8 projection counts: {dict(counts)}")
    return projected


def update_source_metadata(
    output_root: Path,
    *,
    output_dataset_id: str,
    parent_dataset_id: str,
    parent_manifest_sha256: str,
    overlay_dataset_id: str,
    overlay_sha256: str,
    changes: list[dict[str, Any]],
) -> dict[str, Any]:
    yaml_path = output_root / "dataset.yaml"
    metadata = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    counts = label_counts(output_root)
    metadata["dataset_id"] = output_dataset_id
    metadata["splits"] = {
        split: {
            **counts[split]["events_by_class"],
            "background": counts[split]["background_signals"],
            "total": counts[split]["signals"],
        }
        for split in ("train", "val", "test")
    }
    metadata["annotation_counts"] = counts
    metadata.setdefault("provenance", {})
    metadata["provenance"].update(
        {
            "missing_gt_parent_dataset": parent_dataset_id,
            "missing_gt_parent_manifest_sha256": parent_manifest_sha256,
            "missing_gt_overlay": overlay_dataset_id,
            "missing_gt_overlay_sha256": overlay_sha256,
            "missing_gt_application": {
                "restored_source_positives": len(changes),
                "disputed_policy": "overlay_only_no_label_change",
            },
            "candidate_status": (
                "reference only; pending independent Particle2SNR tuning"
            ),
        }
    )
    metadata["missing_gt_review"] = {
        "restoration_manifest": "missing_gt_restorations.jsonl",
        "restored_source_positives": len(changes),
    }
    yaml_path.write_text(
        yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8"
    )
    return counts
