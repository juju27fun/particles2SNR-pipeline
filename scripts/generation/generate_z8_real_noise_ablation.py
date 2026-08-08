#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from internship_workspace.config import Workspace
from internship_workspace.datasets import load_records
from internship_workspace.visual_review_store import ReviewStore
from particles2snr.z8_density_gallery import BASE_COUNTS
from particles2snr.z8_real_noise_ablation import (
    Carrier,
    carrier_sha256,
    class_counts,
    eligible_window_starts,
    inject_real_noise,
    reconstruct_clean,
    repair_blocked_intervals,
    round_robin_carriers,
    select_source_windows,
    validate_paired_parameters,
    yolo_blocked_intervals,
)
from particles2snr.z8_reference_dataset import sha256_file


METHOD_EVIDENCE_ID = "particle-z8-v2-real-noise-ablation-method"
METHOD_RUN_ID = "particle-z8-v2-real-noise-ablation-method-r1"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def record(workspace: Workspace, key: str) -> dict[str, Any]:
    selected = next(
        (
            item.payload
            for item in load_records(workspace)
            if f"{item.payload['id']}@{item.payload['version']}" == key
        ),
        None,
    )
    if selected is None or selected["status"] not in {"active", "reference"}:
        raise ValueError(f"eligible dataset not found: {key}")
    return selected


def approved_method(workspace: Workspace) -> dict[str, str]:
    run_dir = workspace.root / "artifacts/cross-project/reviews" / METHOD_RUN_ID
    store = ReviewStore(run_dir)
    receipt = store.verify_receipt()
    decision = store.current()["decisions"][METHOD_EVIDENCE_ID]["decision"]
    if decision != "approved":
        raise PermissionError("real-noise ablation method is not approved")
    return {
        "evidence_id": METHOD_EVIDENCE_ID,
        "run_id": METHOD_RUN_ID,
        "receipt_sha256": sha256_file(store.receipt_path),
        "contract_sha256": str(receipt["contract_sha256"]),
    }


def extract_carriers(
    source_root: Path,
    *,
    window_stride: int,
    maximum_per_source: int,
    minimum_start_separation: int,
) -> tuple[list[Carrier], dict[str, Any]]:
    inventory = read_csv(source_root / "source_inventory.csv")
    repair_rows = read_csv(source_root / "saturation_repair_manifest.csv")
    repairs = repair_blocked_intervals(repair_rows, signal_length=16_384)
    carriers = []
    source_candidate_counts = {}
    rejection_counts = Counter()
    for row in inventory:
        split = row["split"]
        if split not in {"train", "val"}:
            continue
        filename = row["filename"]
        class_name = row["class"]
        relative = f"{split}/signals/{filename}"
        signal = np.load(source_root / relative, allow_pickle=False)
        if signal.shape != (16_384,) or not np.isfinite(signal).all():
            raise ValueError(f"invalid P2SNR_F signal: {relative}")
        label_path = source_root / split / "labels" / f"{Path(filename).stem}.txt"
        blocked = yolo_blocked_intervals(
            label_path, signal_length=len(signal), guard_samples=804
        )
        blocked.extend(repairs.get((split, filename), []))
        starts = eligible_window_starts(
            signal_length=len(signal),
            window_length=4096,
            stride=window_stride,
            blocked_intervals=blocked,
        )
        selected = select_source_windows(
            signal,
            starts,
            maximum_per_source=maximum_per_source,
            minimum_start_separation=minimum_start_separation,
        )
        source_candidate_counts[relative] = len(selected)
        if not selected:
            rejection_counts[f"{class_name}:no_eligible_window"] += 1
        for source_round, (start, values, rms) in enumerate(selected):
            carriers.append(
                Carrier(
                    class_name=class_name,
                    split=split,
                    source_relative_path=relative,
                    start_sample=start,
                    end_sample=start + len(values),
                    source_round=source_round,
                    rms=rms,
                    sha256=carrier_sha256(values),
                    values=values,
                )
            )
    summary = {
        "source_count": sum(
            row["split"] in {"train", "val"} for row in inventory
        ),
        "carrier_count": len(carriers),
        "class_counts": Counter(item.class_name for item in carriers),
        "sources_with_carriers": sum(value > 0 for value in source_candidate_counts.values()),
        "sources_without_carriers": sum(
            value == 0 for value in source_candidate_counts.values()
        ),
        "rejection_counts": dict(rejection_counts),
    }
    return carriers, summary


def git_revision(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--baseline-dataset", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset-output-dir", type=Path, required=True)
    parser.add_argument("--run-output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--window-stride", type=int, default=512)
    parser.add_argument("--maximum-per-source", type=int, default=8)
    parser.add_argument("--minimum-start-separation", type=int, default=1024)
    args = parser.parse_args()

    workspace = Workspace.load()
    dataset_output = args.dataset_output_dir.resolve()
    run_output = args.run_output_dir.resolve()
    if not dataset_output.is_relative_to(workspace.datasets_root / "interim"):
        raise ValueError("candidate output must be below datasets/interim")
    if not run_output.is_relative_to(workspace.artifacts_root):
        raise ValueError("run output must be below artifacts")
    if dataset_output.exists() or run_output.exists():
        raise FileExistsError("refusing to overwrite candidate or run")

    source_record = record(workspace, args.source_dataset)
    baseline_record = record(workspace, args.baseline_dataset)
    source_root = workspace.datasets_root / source_record["path"]
    baseline_root = workspace.datasets_root / baseline_record["path"]
    method = approved_method(workspace)

    baseline_rows = read_csv(baseline_root / "events.csv")
    baseline_raw = np.load(
        baseline_root / "signals_raw_4096.npy", mmap_mode="r", allow_pickle=False
    )
    if class_counts(baseline_rows) != BASE_COUNTS:
        raise ValueError("baseline v2 class counts changed")
    if baseline_raw.shape != (len(baseline_rows), 4096):
        raise ValueError("baseline v2 signal shape changed")

    carriers, carrier_summary = extract_carriers(
        source_root,
        window_stride=args.window_stride,
        maximum_per_source=args.maximum_per_source,
        minimum_start_separation=args.minimum_start_separation,
    )
    assigned_by_class = {
        class_name: round_robin_carriers(
            carriers,
            class_name=class_name,
            required=BASE_COUNTS[class_name],
            seed=args.seed + index,
            allow_reuse_after_exhaustion=True,
        )
        for index, class_name in enumerate(BASE_COUNTS)
    }
    class_offsets = defaultdict(int)
    assigned = []
    for row in baseline_rows:
        class_name = row["class_name"]
        assigned.append(assigned_by_class[class_name][class_offsets[class_name]])
        class_offsets[class_name] += 1

    clean = reconstruct_clean(baseline_rows)
    candidate_raw, candidate_model, noise_metadata = inject_real_noise(
        baseline_rows, clean, assigned
    )
    carrier_use_counts = defaultdict(int)
    for metadata in noise_metadata:
        carrier_hash = metadata["noise_carrier_sha256"]
        metadata["noise_carrier_use_index"] = carrier_use_counts[carrier_hash]
        carrier_use_counts[carrier_hash] += 1
    candidate_rows = [
        {
            **row,
            **metadata,
            "paired_baseline_dataset": args.baseline_dataset,
            "noise_model": "p2snrf_v2_event_free_real_carrier",
        }
        for row, metadata in zip(baseline_rows, noise_metadata, strict=True)
    ]
    validate_paired_parameters(baseline_rows, candidate_rows)
    achieved_error = max(
        abs(float(row["achieved_snr_db"]) - float(row["snr_db"]))
        for row in candidate_rows
    )
    if achieved_error > 1.0e-9:
        raise ValueError("candidate SNR mismatch")
    if not np.isfinite(candidate_raw).all() or not np.isfinite(candidate_model).all():
        raise ValueError("candidate contains non-finite signals")

    dataset_output.mkdir(parents=True)
    write_csv(dataset_output / "events.csv", candidate_rows)
    write_csv(
        dataset_output / "carrier_manifest.csv",
        [
            {
                "sample_id": row["sample_id"],
                "class_name": row["class_name"],
                **metadata,
            }
            for row, metadata in zip(baseline_rows, noise_metadata, strict=True)
        ],
    )
    np.save(dataset_output / "signals_raw_4096.npy", candidate_raw, allow_pickle=False)
    np.save(
        dataset_output / "signals_conv1dgap_512.npy",
        candidate_model,
        allow_pickle=False,
    )
    summary = {
        "schema_version": 1,
        "dataset_id": args.dataset_id,
        "status": "interim_candidate_awaiting_ablation_result",
        "event_count": len(candidate_rows),
        "class_counts": class_counts(candidate_rows),
        "sealed_test_accessed": False,
        "source_dataset": {
            "id": args.source_dataset,
            "manifest_sha256": source_record["manifest_sha256"],
        },
        "paired_baseline": {
            "id": args.baseline_dataset,
            "manifest_sha256": baseline_record["manifest_sha256"],
            "events_sha256": sha256_file(baseline_root / "events.csv"),
            "raw_sha256": sha256_file(baseline_root / "signals_raw_4096.npy"),
        },
        "method_evidence": method,
        "carrier_policy": {
            "splits": ["train", "val"],
            "class_matched": True,
            "window_length": 4096,
            "window_stride": args.window_stride,
            "annotation_guard_samples": 804,
            "repair_guard": "expanded repair interval plus documented filter radius",
            "maximum_per_source": args.maximum_per_source,
            "minimum_start_separation": args.minimum_start_separation,
            "selection": "per-source low-RMS then deterministic source round-robin",
        },
        "carrier_summary": carrier_summary,
        "paired_contract": {
            "frozen": [
                "sample_id",
                "class_name",
                "amplitude_p0",
                "frequency_khz",
                "tau_ms",
                "snr_db",
                "phi_rad",
                "t0_fraction",
                "single centered symmetric Gaussian-cosine equation",
            ],
            "changed_only": "noise carrier",
            "maximum_snr_error_db": achieved_error,
        },
    }
    (dataset_output / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (dataset_output / "input_contract.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "paired real-noise synthetic ablation",
                "events": "events.csv",
                "carrier_manifest": "carrier_manifest.csv",
                "raw_signals": "signals_raw_4096.npy",
                "model_signals": "signals_conv1dgap_512.npy",
                "review_gate": "diagnostic only; do not register or promote",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    run_output.mkdir(parents=True)
    validation = {
        "event_count": len(candidate_rows),
        "class_counts": class_counts(candidate_rows),
        "carrier_summary": carrier_summary,
        "maximum_snr_error_db": achieved_error,
        "unique_carrier_hash_count": len(
            {row["noise_carrier_sha256"] for row in candidate_rows}
        ),
        "unique_carrier_source_count": len(
            {row["noise_source_relative_path"] for row in candidate_rows}
        ),
        "maximum_carrier_use_count": max(carrier_use_counts.values()),
        "sealed_test_accessed": False,
    }
    (run_output / "generation_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pipeline_revision = git_revision(workspace.root / "particles2SNR-pipeline")
    workspace_revision = git_revision(workspace.root)
    run = {
        "schema_version": 1,
        "project": "particles2SNR-pipeline",
        "run_id": args.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_awaiting_ablation_analysis",
        "dataset": args.dataset_id,
        "command": shlex.join([sys.executable, *sys.argv]),
        "repositories": {
            "workspace": workspace_revision,
            "particles2SNR-pipeline": pipeline_revision,
        },
        "sealed_test_accessed": False,
        "outputs": ["generation_validation.json"],
    }
    (run_output / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"dataset": str(dataset_output), "validation": validation}, indent=2))


if __name__ == "__main__":
    main()
