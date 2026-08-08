#!/usr/bin/env python3
"""Build the gated saturation-first Particle2SNR_F v2 development candidate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from internship_workspace.config import Workspace
from internship_workspace.datasets import resolve_path, select_record
from internship_workspace.visual_review_store import ReviewStore
from particles2snr.detect_saturation import detect_saturation
from particles2snr.saturation_cleaning import (
    forward_backward_filter_response_radius,
    merge_intervals,
    repair_saturation_intervals_pre_filter,
)


PARENT_DATASET = "particles2snr-f-dual-clean-c1-yolo-4class@v1"
OUTPUT_DATASET = "particles2snr-f-dual-clean-c1-yolo-4class@v2"
RAW_DATASETS = {
    "2um": "c1-hf-5-10-2um-doublet@v1",
    "4um": "c1-hf-5-10-4um-doublet@v1",
    "10um": "c1-hf-5-10-10um-doublet@v1",
}
LEGACY_RUN = Path(
    "artifacts/particles2SNR-pipeline/runs/"
    "p0_c1_Particles2SNR_F_dual_clean_candidate"
)
METHOD_GATE = Path(
    "artifacts/cross-project/reviews/"
    "particle-saturation-first-v2-method-r1"
)
DEFAULT_OUTPUT = Path(
    "datasets/interim/"
    "particles2snr-f-dual-clean-c1-yolo-4class/v2"
)
DEFAULT_RUN_DIR = Path(
    "artifacts/particles2SNR-pipeline/runs/"
    "particles2snr-f-dual-clean-prefilter-v2-candidate"
)
DEFAULT_STAGING = Path(
    ".cache/particles2snr-f-dual-clean-prefilter-v2"
)
FS = 2_000_000.0
FMIN = 7_000.0
FMAX = 80_000.0
FILTER_ORDER = 4
GUARD_SAMPLES = 300
FILTER_MASS_FRACTION = 0.999
SPLITS = ("train", "val")
CLASSES = ("2um", "4um", "10um", "unclear")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--staging-root", type=Path, default=DEFAULT_STAGING)
    parser.add_argument("--legacy-run-dir", type=Path, default=LEGACY_RUN)
    parser.add_argument("--method-gate", type=Path, default=METHOD_GATE)
    parser.add_argument(
        "--only-filename",
        action="append",
        default=[],
        help="Bounded smoke selection; never publish an output built with it.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare repaired raw staging and manifest without detection.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(workspace: Workspace, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (workspace.root / path).resolve()


def _relative(workspace: Workspace, path: Path) -> str:
    return path.resolve().relative_to(workspace.root.resolve()).as_posix()


def _normalize_text_provenance_paths(
    workspace: Workspace, root: Path
) -> None:
    absolute_prefix = workspace.root.resolve().as_posix() + "/"
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {
            ".csv",
            ".json",
            ".txt",
            ".yaml",
            ".yml",
        }:
            continue
        content = path.read_text(encoding="utf-8")
        normalized = content.replace(absolute_prefix, "")
        if normalized != content:
            path.write_text(normalized, encoding="utf-8")


def _dataset(
    workspace: Workspace, key: str
) -> tuple[Path, dict[str, Any]]:
    dataset_id, version = key.rsplit("@", 1)
    record = select_record(workspace, dataset_id, version)
    return resolve_path(workspace, record), record.payload


def _assert_approved_gate(path: Path) -> dict[str, Any]:
    run = json.loads((path / "run.json").read_text(encoding="utf-8"))
    review = json.loads(
        (path / "review/decisions.json").read_text(encoding="utf-8")
    )
    decisions = review.get("decisions", {})
    receipt = ReviewStore(path).verify_receipt()
    if run.get("status") != "visual_review_complete":
        raise RuntimeError("saturation-first method gate is not complete")
    if not decisions or {
        row.get("decision") for row in decisions.values()
    } != {"approved"}:
        raise RuntimeError("saturation-first method gate is not approved")
    return {
        "run_id": run["run_id"],
        "evidence_id": run["evidence_id"],
        "run_sha256": sha256_file(path / "run.json"),
        "decisions_sha256": sha256_file(path / "review/decisions.json"),
        "receipt_sha256": sha256_file(path / "review/receipt.json"),
        "receipt_reviewer": receipt["reviewer"],
    }


def _load_legacy_actions(legacy_run: Path) -> list[dict[str, Any]]:
    manifest = legacy_run / "train/saturation_cleaning_manifest.csv"
    rows = []
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["interval_idx"] == "":
                continue
            rows.append(
                {
                    **row,
                    "start_sample": int(row["start_sample"]),
                    "end_sample": int(row["end_sample"]),
                }
            )
    return rows


def _development_signals(parent_root: Path) -> dict[str, tuple[str, Path]]:
    result = {}
    for split in SPLITS:
        for path in sorted((parent_root / split / "signals").glob("*.npy")):
            if path.name in result:
                raise RuntimeError(f"duplicate development filename: {path.name}")
            result[path.name] = (split, path)
    if not result:
        raise RuntimeError("parent dataset contains no development signals")
    return result


def _raw_index(
    raw_roots: dict[str, Path],
) -> dict[str, tuple[str, Path]]:
    result = {}
    for class_name, root in raw_roots.items():
        for path in sorted(root.rglob("*.npy")):
            if path.name in result:
                raise RuntimeError(f"duplicate raw filename: {path.name}")
            result[path.name] = (class_name, path)
    return result


def _regions(
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
    expanded = [
        (
            max(0, start - GUARD_SAMPLES),
            min(len(raw), end + GUARD_SAMPLES),
        )
        for start, end in cores
    ]
    merged = merge_intervals(expanded, len(raw))
    if merged != recorded_intervals:
        raise RuntimeError(
            "reconstructed intervals differ from frozen historical intervals: "
            f"detected={merged}, recorded={recorded_intervals}"
        )
    result = []
    for expanded_start, expanded_end in merged:
        member_cores = [
            core
            for core, component in zip(cores, expanded)
            if component[0] < expanded_end and component[1] > expanded_start
        ]
        result.append(
            {
                "core_interval": [
                    min(row[0] for row in member_cores),
                    max(row[1] for row in member_cores),
                ],
                "expanded_interval": [expanded_start, expanded_end],
                "detected_core_count": len(member_cores),
            }
        )
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _git_state(path: Path) -> dict[str, Any]:
    head = subprocess.run(
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
    return {"commit": head, "dirty": dirty}


def _run_detection(
    *,
    workspace: Workspace,
    repository: Path,
    staging_input: Path,
    filtered_working: Path,
    output: Path,
    run_dir: Path,
    boundary_manifest: Path,
) -> None:
    command = [
        str(workspace.root / ".venv/bin/python"),
        str(repository / "scripts/generation/generate_particles2SNR_dataset.py"),
        "--input-root",
        str(staging_input),
        "--output-root",
        str(filtered_working),
        "--particles2SNR-output",
        str(run_dir),
        "--detseg-output",
        str(output),
        "--splits",
        "train,val",
        "--classes",
        "2um,4um,10um,unclear",
        "--val-fraction",
        "0",
        "--saturation-policy",
        "keep",
        "--apply-bandpass-output",
        "--peak-evidence-signal-mode",
        "dual_clean",
        "--saturation-boundary-manifest",
        str(boundary_manifest),
        "--saturation-boundary-clean-local-min-z",
        "1.5",
        "--unclear-snr-threshold-db",
        "-10",
        "--device",
        "cpu",
    ]
    subprocess.run(command, cwd=workspace.root, check=True)


def main() -> None:
    args = parse_args()
    workspace = Workspace.load()
    repository = workspace.root / "particles2SNR-pipeline"
    output = _resolve(workspace, args.output)
    run_dir = _resolve(workspace, args.run_dir)
    staging = _resolve(workspace, args.staging_root)
    legacy_run = _resolve(workspace, args.legacy_run_dir)
    gate_path = _resolve(workspace, args.method_gate)
    output.relative_to(workspace.datasets_root / "interim")
    run_dir.relative_to(workspace.artifacts_root / "particles2SNR-pipeline")
    staging.relative_to(workspace.root / ".cache")
    for path in (output, run_dir, staging):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing path: {path}")

    gate = _assert_approved_gate(gate_path)
    parent_root, parent_record = _dataset(workspace, PARENT_DATASET)
    raw_roots = {}
    raw_records = {}
    for class_name, key in RAW_DATASETS.items():
        raw_roots[class_name], raw_records[class_name] = _dataset(workspace, key)
    parent_signals = _development_signals(parent_root)
    selected = set(args.only_filename)
    if selected:
        missing = selected - set(parent_signals)
        if missing:
            raise FileNotFoundError(f"unknown selected filenames: {sorted(missing)}")
        parent_signals = {
            name: value
            for name, value in parent_signals.items()
            if name in selected
        }
    raw_index = _raw_index(raw_roots)
    actions = _load_legacy_actions(legacy_run)
    grouped_actions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in actions:
        if row["filename"] in parent_signals:
            grouped_actions[row["filename"]].append(row)

    response_radius = forward_backward_filter_response_radius(
        signal_length=16_384,
        fs=FS,
        fmin=FMIN,
        fmax=FMAX,
        order=FILTER_ORDER,
        mass_fraction=FILTER_MASS_FRACTION,
    )
    staging_input = staging / "raw-repaired"
    filtered_working = staging / "filtered"
    manifest_path = staging / "saturation_repair_manifest.csv"
    repair_rows = []
    inventory_rows = []
    for split in SPLITS:
        for class_name in CLASSES:
            (staging_input / split / class_name).mkdir(
                parents=True, exist_ok=True
            )

    for filename, (split, _parent_path) in sorted(parent_signals.items()):
        if filename not in raw_index:
            raise FileNotFoundError(f"raw signal not found: {filename}")
        class_name, raw_path = raw_index[filename]
        target = staging_input / split / class_name / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = np.asarray(np.load(raw_path))
        rows = sorted(
            grouped_actions.get(filename, []),
            key=lambda row: (row["start_sample"], row["end_sample"]),
        )
        if not rows:
            target.symlink_to(raw_path.resolve())
            inventory_rows.append(
                {
                    "split": split,
                    "class": class_name,
                    "filename": filename,
                    "raw_path": _relative(workspace, raw_path),
                    "raw_sha256": sha256_file(raw_path),
                    "staging_action": "symlinked_unaffected_raw",
                    "repair_region_count": 0,
                }
            )
            continue

        recorded = [
            (row["start_sample"], row["end_sample"]) for row in rows
        ]
        regions = _regions(raw, recorded)
        historical_clean_path = (
            legacy_run
            / "train/peak_evidence_clean_signals"
            / class_name
            / filename
        )
        historical_clean = np.asarray(np.load(historical_clean_path))
        replacements = []
        for region in regions:
            start, end = region["expanded_interval"]
            replacements.append(
                {
                    **region,
                    "replacement": historical_clean[start:end],
                }
            )
        repaired = repair_saturation_intervals_pre_filter(
            raw,
            replacements,
            fs=FS,
            fmin=FMIN,
            fmax=FMAX,
            order=FILTER_ORDER,
        )
        np.save(target, repaired["clean_signal"])
        inventory_rows.append(
            {
                "split": split,
                "class": class_name,
                "filename": filename,
                "raw_path": _relative(workspace, raw_path),
                "raw_sha256": sha256_file(raw_path),
                "staging_action": "cosine_prefilter_repaired_raw",
                "repair_region_count": len(regions),
            }
        )
        for interval_index, region in enumerate(regions):
            start, end = region["expanded_interval"]
            carrier = historical_clean[start:end]
            repair_rows.append(
                {
                    "split": split,
                    "class": class_name,
                    "filename": filename,
                    "interval_idx": interval_index,
                    "core_start_sample": region["core_interval"][0],
                    "core_end_sample": region["core_interval"][1],
                    "expanded_start_sample": start,
                    "expanded_end_sample": end,
                    "detected_core_count": region["detected_core_count"],
                    "method": "cosine-pre-filter",
                    "guard_samples": GUARD_SAMPLES,
                    "fs_hz": FS,
                    "bandpass_low_hz": FMIN,
                    "bandpass_high_hz": FMAX,
                    "bandpass_order": FILTER_ORDER,
                    "filter_response_mass_fraction": FILTER_MASS_FRACTION,
                    "filter_response_radius_samples": response_radius,
                    "clean_local_min_z": 1.5,
                    "distance_16_samples_role": "diagnostic_only",
                    "raw_dataset": RAW_DATASETS[class_name],
                    "raw_path": _relative(workspace, raw_path),
                    "raw_sha256": sha256_file(raw_path),
                    "historical_carrier_path": _relative(
                        workspace, historical_clean_path
                    ),
                    "historical_carrier_slice_sha256": hashlib.sha256(
                        np.asarray(carrier).tobytes()
                    ).hexdigest(),
                    "repaired_raw_sha256": sha256_file(target),
                    "candidate_signal_sha256": "",
                }
            )
    if not repair_rows:
        raise RuntimeError("selection contains no frozen saturation repairs")
    _write_csv(manifest_path, repair_rows)
    _write_csv(staging / "source_inventory.csv", inventory_rows)

    if args.prepare_only:
        print(
            json.dumps(
                {
                    "status": "prepared_only",
                    "signals": len(parent_signals),
                    "repair_regions": len(repair_rows),
                    "filter_response_radius_samples": response_radius,
                    "staging": str(staging),
                },
                indent=2,
            )
        )
        return

    _run_detection(
        workspace=workspace,
        repository=repository,
        staging_input=staging_input,
        filtered_working=filtered_working,
        output=output,
        run_dir=run_dir,
        boundary_manifest=manifest_path,
    )
    _normalize_text_provenance_paths(workspace, run_dir)
    for row in repair_rows:
        candidate = output / row["split"] / "signals" / row["filename"]
        row["candidate_signal_sha256"] = sha256_file(candidate)
    _write_csv(output / "saturation_repair_manifest.csv", repair_rows)
    shutil.copy2(staging / "source_inventory.csv", output / "source_inventory.csv")

    dataset_yaml = yaml.safe_load(
        (output / "dataset.yaml").read_text(encoding="utf-8")
    )
    dataset_yaml["path"] = "."
    dataset_yaml["dataset_id"] = OUTPUT_DATASET
    dataset_yaml["status"] = "candidate_pending_result_validation"
    dataset_yaml["splits"]["test"]["total"] = 0
    dataset_yaml["provenance"] = {
        "parent_split_reference_only": PARENT_DATASET,
        "parent_manifest_sha256": parent_record["manifest_sha256"],
        "raw_datasets": {
            key: raw_records[class_name]["manifest_sha256"]
            for class_name, key in RAW_DATASETS.items()
        },
        "repair_method": "cosine-pre-filter",
        "historical_carrier_run": _relative(workspace, legacy_run),
        "method_evidence_id": gate["evidence_id"],
        "method_evidence_run_id": gate["run_id"],
        "annotation_policy": "fresh particles2SNR plus dual-clean; no v1 label or event-id input",
        "sealed_test_accessed": False,
    }
    dataset_yaml["generation_params"]["saturation_policy"] = (
        "historical-carrier cosine pre-filter"
    )
    dataset_yaml["generation_params"]["filter_response_radius_samples"] = (
        response_radius
    )
    dataset_yaml["generation_params"]["filter_response_mass_fraction"] = (
        FILTER_MASS_FRACTION
    )
    dataset_yaml["generation_params"]["distance_16_samples_role"] = (
        "diagnostic_only"
    )
    (output / "dataset.yaml").write_text(
        yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8"
    )

    created_at = datetime.now(timezone.utc).isoformat()
    run = {
        "schema_version": 1,
        "project": "particles2SNR-pipeline",
        "run_id": run_dir.name,
        "kind": "dataset-generation",
        "command": [
            ".venv/bin/python",
            *sys.argv,
        ],
        "dataset": OUTPUT_DATASET,
        "created_at": created_at,
        "status": "complete_pending_scientific_result_validation",
        "sealed_test_accessed": False,
        "bounded_smoke_selection": sorted(selected),
        "publishable_candidate": not bool(selected),
        "method_evidence": gate,
        "repositories": {
            "workspace": _git_state(workspace.root),
            "particles2SNR-pipeline": _git_state(repository),
        },
        "code_inputs": {
            "builder": sha256_file(Path(__file__)),
            "generator": sha256_file(
                repository
                / "scripts/generation/generate_particles2SNR_dataset.py"
            ),
            "saturation_cleaning": sha256_file(
                repository / "particles2snr/saturation_cleaning.py"
            ),
            "run_dataset": sha256_file(
                repository / "particles2snr/run_dataset.py"
            ),
            "pipeline": sha256_file(
                repository
                / "particles2snr/fft_analysis_pipeline_particles2SNR.py"
            ),
        },
        "parents": {
            PARENT_DATASET: parent_record["manifest_sha256"],
            **{
                key: raw_records[class_name]["manifest_sha256"]
                for class_name, key in RAW_DATASETS.items()
            },
        },
        "summary": {
            "development_signals": len(parent_signals),
            "repaired_signals": len(grouped_actions),
            "repair_regions": len(repair_rows),
            "filter_response_radius_samples": response_radius,
            "repair_method": "cosine-pre-filter",
            "bandpass_passes": 1,
            "labels_inherited": 0,
        },
        "outputs": [
            _relative(workspace, output / "dataset.yaml"),
            _relative(workspace, output / "saturation_repair_manifest.csv"),
            _relative(workspace, output / "source_inventory.csv"),
        ],
    }
    (run_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(run["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
