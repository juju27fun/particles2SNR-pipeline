#!/usr/bin/env python3
"""Build the versioned dual-clean saturation-repair candidate and review queue."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from internship_workspace.config import Workspace
from internship_workspace.datasets import resolve_path, select_record
from particles2snr.detect_saturation import detect_saturation
from particles2snr.saturation_cleaning import (
    merge_intervals,
    repair_saturation_intervals_filtered_domain,
)


PARENT_DATASET = "particles2snr-f-dual-clean-c1-yolo-4class@v1"
CANDIDATE_DATASET = (
    "particles2snr-f-dual-clean-c1-yolo-4class-saturation-candidate@v1"
)
RAW_DATASETS = {
    "2um": "c1-hf-5-10-2um-doublet@v1",
    "4um": "c1-hf-5-10-4um-doublet@v1",
    "10um": "c1-hf-5-10-10um-doublet@v1",
}
DEFAULT_OUTPUT = Path(
    "datasets/interim/"
    "particles2snr-f-dual-clean-c1-yolo-4class-saturation-candidate/v1"
)
LEGACY_RUN = Path(
    "artifacts/particles2SNR-pipeline/runs/"
    "p0_c1_Particles2SNR_F_dual_clean_candidate"
)
METHOD_GATE = Path(
    "artifacts/particles2SNR-pipeline/audits/"
    "saturation-repair-review-v1-jlb/current_validation.json"
)
RUN_ID = "dual_clean_saturation_candidate_20260718"
DEFAULT_RUN_DIR = Path(f"artifacts/particles2SNR-pipeline/runs/{RUN_ID}")
SPLITS = ("train", "val", "test")
FS = 2_000_000.0
FMIN = 7_000.0
FMAX = 80_000.0
FILTER_ORDER = 4
GUARD_SAMPLES = 300
MIN_SAFE_WIDTH_SAMPLES = 160


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--legacy-run-dir", type=Path, default=LEGACY_RUN)
    parser.add_argument("--method-gate", type=Path, default=METHOD_GATE)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()


def relative(workspace: Workspace, path: Path) -> str:
    return path.resolve().relative_to(workspace.root.resolve()).as_posix()


def resolve_workspace_path(workspace: Workspace, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (workspace.root / path).resolve()


def resolve_dataset(
    workspace: Workspace, dataset_key: str
) -> tuple[Path, dict[str, Any]]:
    dataset_id, version = dataset_key.rsplit("@", 1)
    record = select_record(workspace, dataset_id, version)
    return resolve_path(workspace, record), record.payload


def git_state(path: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty}


def load_actions(legacy_run: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_split in ("train", "test"):
        manifest = legacy_run / source_split / "saturation_cleaning_manifest.csv"
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        with manifest.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["interval_idx"] == "":
                    continue
                rows.append(
                    {
                        **row,
                        "source_split": source_split,
                        "start_sample": int(row["start_sample"]),
                        "end_sample": int(row["end_sample"]),
                        "interval_idx": int(row["interval_idx"]),
                    }
                )
    return rows


def locate_parent_signals(parent_root: Path) -> dict[str, tuple[str, Path]]:
    located: dict[str, tuple[str, Path]] = {}
    for split in SPLITS:
        for path in sorted((parent_root / split / "signals").glob("*.npy")):
            if path.name in located:
                raise RuntimeError(f"duplicate parent signal filename: {path.name}")
            located[path.name] = (split, path)
    return located


def detected_regions(
    raw: np.ndarray,
    recorded_intervals: list[tuple[int, int]],
) -> list[dict[str, Any]]:
    info = detect_saturation(
        raw,
        fs=FS,
        fmin=FMIN,
        fmax=FMAX,
        min_flat=500,
        zero_threshold=1e-4,
    )
    cores = [
        (int(row["start_sample"]), int(row["end_sample"]))
        for row in info["intervals"]
    ]
    expanded_components = [
        (
            max(0, core_start - GUARD_SAMPLES),
            min(len(raw), core_end + GUARD_SAMPLES),
        )
        for core_start, core_end in cores
    ]
    merged = merge_intervals(expanded_components, len(raw))
    if merged != recorded_intervals:
        raise RuntimeError(
            "reconstructed saturation intervals differ from the historical "
            f"manifest: detected={merged}, recorded={recorded_intervals}"
        )
    regions: list[dict[str, Any]] = []
    for expanded_start, expanded_end in merged:
        member_cores = [
            core
            for core, expanded in zip(cores, expanded_components)
            if expanded[0] < expanded_end and expanded[1] > expanded_start
        ]
        if not member_cores:
            raise RuntimeError("expanded interval has no saturation core")
        regions.append(
            {
                "core_interval": [
                    min(row[0] for row in member_cores),
                    max(row[1] for row in member_cores),
                ],
                "expanded_interval": [expanded_start, expanded_end],
                "detected_core_count": len(member_cores),
            }
        )
    return regions


def parse_yolo_labels(path: Path, signal_length: int) -> list[dict[str, Any]]:
    rows = []
    for annotation_id, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines()
    ):
        if not raw.strip():
            continue
        fields = raw.split()
        if len(fields) != 3:
            raise RuntimeError(f"invalid 1D YOLO label in {path}: {raw}")
        class_id, center, width = int(fields[0]), float(fields[1]), float(fields[2])
        start = max(0.0, (center - width / 2.0) * signal_length)
        end = min(float(signal_length), (center + width / 2.0) * signal_length)
        rows.append(
            {
                "annotation_id": annotation_id,
                "class_id": class_id,
                "center_normalized": center,
                "width_normalized": width,
                "start_sample": start,
                "end_sample": end,
                "label_line": raw,
            }
        )
    return rows


def intervals_overlap(
    left: tuple[float, float], right: tuple[float, float]
) -> bool:
    return left[0] < right[1] and left[1] > right[0]


def subtract_intervals(
    interval: tuple[float, float],
    unsafe: list[tuple[int, int]],
) -> list[tuple[float, float]]:
    fragments = [interval]
    for unsafe_start, unsafe_end in sorted(unsafe):
        next_fragments = []
        for start, end in fragments:
            if end <= unsafe_start or start >= unsafe_end:
                next_fragments.append((start, end))
                continue
            if start < unsafe_start:
                next_fragments.append((start, min(end, float(unsafe_start))))
            if end > unsafe_end:
                next_fragments.append((max(start, float(unsafe_end)), end))
        fragments = next_fragments
    return [(start, end) for start, end in fragments if end > start]


def interval_max_robust_z(signal: np.ndarray, start: float, end: float) -> float:
    values = np.asarray(signal, dtype=np.float64)
    median = float(np.median(values))
    sigma = max(float(np.median(np.abs(values - median)) * 1.4826), 1e-12)
    left = max(0, int(np.floor(start)))
    right = min(len(values), int(np.ceil(end)))
    return float(np.max(np.abs(values[left:right] - median)) / sigma)


def copy_parent(parent_root: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(
            f"candidate output already exists; refusing to mutate it: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(parent_root, output, copy_function=shutil.copy2)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    workspace = Workspace.load()
    output = resolve_workspace_path(workspace, args.output)
    run_dir = resolve_workspace_path(workspace, args.run_dir)
    legacy_run = resolve_workspace_path(workspace, args.legacy_run_dir)
    method_gate = resolve_workspace_path(workspace, args.method_gate)
    output.relative_to(workspace.datasets_root / "interim")
    run_dir.relative_to(workspace.artifacts_root / "particles2SNR-pipeline")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"run artifact already exists; refusing to mutate it: {run_dir}"
        )
    if not method_gate.is_file():
        raise FileNotFoundError(method_gate)
    gate = json.loads(method_gate.read_text(encoding="utf-8"))
    current_gate = gate.get("current")
    if not current_gate or current_gate.get("decision") != "approve_B":
        raise RuntimeError("method B has not been approved by the human gate")

    parent_root, parent_record = resolve_dataset(workspace, PARENT_DATASET)
    raw_roots: dict[str, Path] = {}
    raw_records: dict[str, dict[str, Any]] = {}
    for class_name, dataset_key in RAW_DATASETS.items():
        raw_roots[class_name], raw_records[class_name] = resolve_dataset(
            workspace, dataset_key
        )

    actions = load_actions(legacy_run)
    grouped_actions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in actions:
        grouped_actions[row["filename"]].append(row)
    parent_signals = locate_parent_signals(parent_root)
    missing = sorted(set(grouped_actions) - set(parent_signals))
    if missing:
        raise RuntimeError(f"affected files absent from parent dataset: {missing}")

    copy_parent(parent_root, output)
    run_dir.mkdir(parents=True, exist_ok=False)
    created_at = datetime.now(timezone.utc).isoformat()
    repair_rows: list[dict[str, Any]] = []
    review_candidates: list[dict[str, Any]] = []
    affected_final_splits: dict[str, int] = defaultdict(int)
    affected_annotation_files: set[tuple[str, str]] = set()
    class_names = yaml.safe_load(
        (parent_root / "dataset.yaml").read_text(encoding="utf-8")
    )["names"]

    for filename in sorted(grouped_actions):
        rows = sorted(
            grouped_actions[filename],
            key=lambda row: (row["start_sample"], row["end_sample"]),
        )
        classes = {row["class"] for row in rows}
        source_splits = {row["source_split"] for row in rows}
        if len(classes) != 1 or len(source_splits) != 1:
            raise RuntimeError(f"inconsistent manifest rows for {filename}")
        class_name = next(iter(classes))
        source_split = next(iter(source_splits))
        final_split, parent_signal_path = parent_signals[filename]
        affected_final_splits[final_split] += 1
        raw_path = raw_roots[class_name] / filename
        historical_clean_path = (
            legacy_run
            / source_split
            / "peak_evidence_clean_signals"
            / class_name
            / filename
        )
        for path in (raw_path, historical_clean_path, parent_signal_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        raw = np.asarray(np.load(raw_path))
        historical_clean = np.asarray(np.load(historical_clean_path))
        current = np.asarray(np.load(parent_signal_path))
        if not (len(raw) == len(historical_clean) == len(current)):
            raise RuntimeError(f"signal length mismatch for {filename}")
        recorded = [
            (row["start_sample"], row["end_sample"]) for row in rows
        ]
        regions = detected_regions(raw, recorded)
        replacements = []
        for region in regions:
            expanded_start, expanded_end = region["expanded_interval"]
            replacements.append(
                {
                    **region,
                    "replacement": historical_clean[
                        expanded_start:expanded_end
                    ],
                }
            )
        repaired = repair_saturation_intervals_filtered_domain(
            raw,
            replacements,
            fs=FS,
            fmin=FMIN,
            fmax=FMAX,
            order=FILTER_ORDER,
        )
        candidate_signal = repaired["filtered_signal"]
        candidate_path = output / final_split / "signals" / filename
        np.save(candidate_path, candidate_signal)
        current_sha = sha256_file(parent_signal_path)
        candidate_sha = sha256_file(candidate_path)
        for interval_idx, region in enumerate(regions):
            core_start, core_end = region["core_interval"]
            expanded_start, expanded_end = region["expanded_interval"]
            carrier = historical_clean[expanded_start:expanded_end]
            repair_rows.append(
                {
                    "final_split": final_split,
                    "source_split": source_split,
                    "class": class_name,
                    "filename": filename,
                    "interval_idx": interval_idx,
                    "core_start_sample": core_start,
                    "core_end_sample": core_end,
                    "expanded_start_sample": expanded_start,
                    "expanded_end_sample": expanded_end,
                    "detected_core_count": region["detected_core_count"],
                    "method": "cosine-filtered-domain",
                    "guard_samples": GUARD_SAMPLES,
                    "raw_dataset": RAW_DATASETS[class_name],
                    "raw_path": relative(workspace, raw_path),
                    "raw_sha256": sha256_file(raw_path),
                    "historical_carrier_path": relative(
                        workspace, historical_clean_path
                    ),
                    "historical_carrier_slice_sha256": hashlib.sha256(
                        np.asarray(carrier).tobytes()
                    ).hexdigest(),
                    "parent_signal_sha256": current_sha,
                    "candidate_signal_sha256": candidate_sha,
                }
            )

        label_path = parent_root / final_split / "labels" / (
            f"{Path(filename).stem}.txt"
        )
        expanded_intervals = [
            tuple(region["expanded_interval"]) for region in regions
        ]
        for annotation in parse_yolo_labels(label_path, len(raw)):
            annotation_interval = (
                annotation["start_sample"],
                annotation["end_sample"],
            )
            overlapping = [
                region
                for region in regions
                if intervals_overlap(
                    annotation_interval,
                    tuple(region["expanded_interval"]),
                )
            ]
            if not overlapping:
                continue
            safe_fragments = subtract_intervals(
                annotation_interval, expanded_intervals
            )
            eligible_fragments = [
                fragment
                for fragment in safe_fragments
                if fragment[1] - fragment[0] >= MIN_SAFE_WIDTH_SAMPLES
            ]
            safe_proposal = (
                max(
                    eligible_fragments,
                    key=lambda fragment: fragment[1] - fragment[0],
                )
                if eligible_fragments
                else None
            )
            affected_annotation_files.add((final_split, filename))
            source_id = Path(filename).stem
            review_candidates.append(
                {
                    "candidate_id": (
                        f"{final_split}:{source_id}:"
                        f"{annotation['annotation_id']}"
                    ),
                    "source_id": source_id,
                    "filename": filename,
                    "split": final_split,
                    "source_class": class_name,
                    "annotation_id": annotation["annotation_id"],
                    "class_id": annotation["class_id"],
                    "class_name": class_names[annotation["class_id"]],
                    "source_length": int(len(raw)),
                    "source_duration_ms": len(raw) / FS * 1000.0,
                    "sampling_frequency_hz": FS,
                    "raw_dataset": RAW_DATASETS[class_name],
                    "raw_signal_path": relative(workspace, raw_path),
                    "parent_dataset": PARENT_DATASET,
                    "parent_signal_path": (
                        f"{final_split}/signals/{filename}"
                    ),
                    "candidate_dataset": CANDIDATE_DATASET,
                    "candidate_signal_path": (
                        f"{final_split}/signals/{filename}"
                    ),
                    "label_path": f"{final_split}/labels/{label_path.name}",
                    "label_line": annotation["label_line"],
                    "original_interval_samples": [
                        annotation["start_sample"],
                        annotation["end_sample"],
                    ],
                    "original_interval_ms": [
                        annotation["start_sample"] / FS * 1000.0,
                        annotation["end_sample"] / FS * 1000.0,
                    ],
                    "repair_regions": [
                        {
                            "core_interval_samples": region["core_interval"],
                            "core_interval_ms": [
                                value / FS * 1000.0
                                for value in region["core_interval"]
                            ],
                            "expanded_interval_samples": region[
                                "expanded_interval"
                            ],
                            "expanded_interval_ms": [
                                value / FS * 1000.0
                                for value in region["expanded_interval"]
                            ],
                        }
                        for region in overlapping
                    ],
                    "safe_fragments_samples": [
                        list(fragment) for fragment in safe_fragments
                    ],
                    "safe_fragments_ms": [
                        [value / FS * 1000.0 for value in fragment]
                        for fragment in safe_fragments
                    ],
                    "safe_interval_proposal_samples": (
                        list(safe_proposal) if safe_proposal else None
                    ),
                    "safe_interval_proposal_ms": (
                        [value / FS * 1000.0 for value in safe_proposal]
                        if safe_proposal
                        else None
                    ),
                    "safe_interval_eligible": safe_proposal is not None,
                    "parent_interval_max_abs_z": interval_max_robust_z(
                        current, *annotation_interval
                    ),
                    "candidate_interval_max_abs_z": interval_max_robust_z(
                        candidate_signal, *annotation_interval
                    ),
                }
            )

    split_order = {name: index for index, name in enumerate(SPLITS)}
    review_candidates.sort(
        key=lambda row: (
            split_order[row["split"]],
            row["source_id"],
            row["annotation_id"],
        )
    )
    for order, row in enumerate(review_candidates):
        row["order"] = order

    if len(actions) != 263 or len(grouped_actions) != 255:
        raise RuntimeError("unexpected saturation repair population")
    if len(review_candidates) != 193 or len(affected_annotation_files) != 124:
        raise RuntimeError("unexpected annotation arbitration population")

    queue = {
        "schema_version": 1,
        "dataset_id": CANDIDATE_DATASET,
        "created_at": created_at,
        "repair_method": "cosine-filtered-domain",
        "method_gate": {
            "path": relative(workspace, method_gate),
            "sha256": sha256_file(method_gate),
            "decision": current_gate,
        },
        "parent": {
            "dataset_id": PARENT_DATASET,
            "manifest_sha256": parent_record["manifest_sha256"],
        },
        "candidate_count": len(review_candidates),
        "unique_source_count": len(affected_annotation_files),
        "minimum_safe_width_samples": MIN_SAFE_WIDTH_SAMPLES,
        "minimum_safe_width_ms": MIN_SAFE_WIDTH_SAMPLES / FS * 1000.0,
        "allowed_decisions": ["keep", "delete", "clip", "needs_review"],
        "candidates": review_candidates,
    }
    queue["queue_sha256"] = stable_hash(
        {key: value for key, value in queue.items() if key != "created_at"}
    )
    (output / "arbitration_queue.json").write_text(
        json.dumps(queue, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(output / "saturation_repair_manifest.csv", repair_rows)

    summary = {
        "schema_version": 1,
        "dataset_id": CANDIDATE_DATASET,
        "created_at": created_at,
        "status": "candidate_pending_human_annotation_review",
        "parent_dataset": PARENT_DATASET,
        "repair_method": "cosine-filtered-domain",
        "saturation_actions": len(actions),
        "affected_signals": len(grouped_actions),
        "affected_signals_by_final_split": dict(affected_final_splits),
        "annotations_pending_review": len(review_candidates),
        "annotation_bearing_signals_pending_review": len(
            affected_annotation_files
        ),
        "safe_clip_proposals": sum(
            bool(row["safe_interval_eligible"]) for row in review_candidates
        ),
        "labels_changed": 0,
        "parent_v1_mutated": False,
        "queue_sha256": queue["queue_sha256"],
    }
    (output / "candidate_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    dataset_yaml = yaml.safe_load(
        (output / "dataset.yaml").read_text(encoding="utf-8")
    )
    dataset_yaml["path"] = "."
    dataset_yaml["provenance"] = {
        "parent_dataset": PARENT_DATASET,
        "parent_manifest_sha256": parent_record["manifest_sha256"],
        "raw_datasets": {
            dataset_key: raw_records[class_name]["manifest_sha256"]
            for class_name, dataset_key in RAW_DATASETS.items()
        },
        "repair_method": "cosine-filtered-domain",
        "method_gate": relative(workspace, method_gate),
        "legacy_replacement_carrier_run": relative(workspace, legacy_run),
        "annotation_policy": (
            "labels copied byte-for-byte from v1 pending human arbitration"
        ),
        "candidate_status": "not_for_training_or_metrics",
    }
    dataset_yaml["review"] = {
        "queue": "arbitration_queue.json",
        "candidate_count": len(review_candidates),
        "queue_sha256": queue["queue_sha256"],
    }
    (output / "dataset.yaml").write_text(
        yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8"
    )

    run = {
        "schema_version": 1,
        "project": "particles2SNR-pipeline",
        "run_id": RUN_ID,
        "kind": "dataset-generation",
        "dataset": CANDIDATE_DATASET,
        "command": (
            "particles2SNR-pipeline/scripts/generation/"
            "build_dual_clean_saturation_candidate.py"
        ),
        "created_at": created_at,
        "status": "complete_pending_human_annotation_review",
        "repositories": {
            "workspace": git_state(workspace.root),
            "particles2SNR-pipeline": git_state(
                workspace.root / "particles2SNR-pipeline"
            ),
        },
        "parents": {
            PARENT_DATASET: parent_record["manifest_sha256"],
            **{
                dataset_key: raw_records[class_name]["manifest_sha256"]
                for class_name, dataset_key in RAW_DATASETS.items()
            },
        },
        "method_gate_sha256": sha256_file(method_gate),
        "outputs": [
            relative(workspace, output / "dataset.yaml"),
            relative(workspace, output / "candidate_summary.json"),
            relative(workspace, output / "arbitration_queue.json"),
            relative(workspace, output / "saturation_repair_manifest.csv"),
        ],
        "summary": summary,
    }
    (run_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = [
        "# Dual-clean saturation candidate",
        "",
        f"- Parent: `{PARENT_DATASET}`",
        f"- Candidate: `{CANDIDATE_DATASET}`",
        "- Approved repair: `cosine-filtered-domain` (method B)",
        f"- Repaired signals: {len(grouped_actions)}",
        f"- Repair intervals: {len(actions)}",
        f"- Annotations pending human review: {len(review_candidates)}",
        f"- Affected annotation-bearing signals: {len(affected_annotation_files)}",
        "",
        "The parent v1 dataset was not modified. Candidate labels remain "
        "byte-identical to v1 and this candidate must not be used for training "
        "or metrics before the arbitration journal is complete.",
        "",
    ]
    (run_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
