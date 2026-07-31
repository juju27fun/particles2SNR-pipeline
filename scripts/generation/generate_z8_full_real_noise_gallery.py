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
from numpy.lib.format import open_memmap

from internship_workspace.config import Workspace
from internship_workspace.datasets import load_records
from internship_workspace.visual_review_store import ReviewStore
from particles2snr.z8_density_gallery import TARGET_COUNTS
from particles2snr.z8_real_noise_ablation import (
    CarrierRef,
    carrier_sha256,
    class_counts,
    eligible_window_starts,
    preprocess_conv1dgap_512,
    repair_blocked_intervals,
    round_robin_carriers,
    select_source_window_refs,
    yolo_blocked_intervals,
)
from particles2snr.z8_reference_dataset import sha256_file


METHOD_EVIDENCE_ID = "particle-z8-v2-full-real-noise-method"
METHOD_RUN_ID = "particle-z8-v2-full-real-noise-method-r1"
RAW_LENGTH = 4096
SIGNAL_LENGTH = 16_384
SAMPLING_FREQUENCY_HZ = 2_000_000.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
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
        raise PermissionError("full real-noise method is not approved")
    return {
        "evidence_id": METHOD_EVIDENCE_ID,
        "run_id": METHOD_RUN_ID,
        "receipt_sha256": sha256_file(store.receipt_path),
        "contract_sha256": str(receipt["contract_sha256"]),
    }


def anchor_sources(path: Path) -> tuple[set[str], str]:
    rows = read_csv(path)
    sources = {row["source_signal_relative_path"] for row in rows}
    if len(rows) != 30 or len(sources) != 30:
        raise ValueError("expected exactly 30 distinct frozen anchor sources")
    return sources, sha256_file(path)


def extract_carrier_refs(
    source_root: Path,
    *,
    excluded_sources: set[str],
    window_stride: int,
    maximum_per_source: int,
    minimum_start_separation: int,
) -> tuple[list[CarrierRef], dict[str, Any]]:
    inventory = read_csv(source_root / "source_inventory.csv")
    repair_rows = read_csv(source_root / "saturation_repair_manifest.csv")
    repairs = repair_blocked_intervals(repair_rows, signal_length=SIGNAL_LENGTH)
    carriers: list[CarrierRef] = []
    source_candidate_counts: dict[str, int] = {}
    rejection_counts = Counter()
    excluded_counts = Counter()
    for row in inventory:
        split = row["split"]
        if split not in {"train", "val"}:
            continue
        filename = row["filename"]
        class_name = row["class"]
        relative = f"{split}/signals/{filename}"
        if relative in excluded_sources:
            excluded_counts[class_name] += 1
            continue
        signal = np.load(source_root / relative, allow_pickle=False)
        if signal.shape != (SIGNAL_LENGTH,) or not np.isfinite(signal).all():
            raise ValueError(f"invalid P2SNR_F signal: {relative}")
        label_path = source_root / split / "labels" / f"{Path(filename).stem}.txt"
        blocked = yolo_blocked_intervals(
            label_path,
            signal_length=len(signal),
            guard_samples=804,
        )
        blocked.extend(repairs.get((split, filename), []))
        starts = eligible_window_starts(
            signal_length=len(signal),
            window_length=RAW_LENGTH,
            stride=window_stride,
            blocked_intervals=blocked,
        )
        selected = select_source_window_refs(
            signal,
            starts,
            maximum_per_source=maximum_per_source,
            minimum_start_separation=minimum_start_separation,
        )
        source_candidate_counts[relative] = len(selected)
        if not selected:
            rejection_counts[f"{class_name}:no_eligible_window"] += 1
        for source_round, (start, rms) in enumerate(selected):
            carriers.append(
                CarrierRef(
                    class_name=class_name,
                    split=split,
                    source_relative_path=relative,
                    start_sample=start,
                    end_sample=start + RAW_LENGTH,
                    source_round=source_round,
                    rms=rms,
                )
            )
    summary = {
        "source_count_after_anchor_exclusion": len(source_candidate_counts),
        "carrier_count": len(carriers),
        "class_counts": dict(Counter(item.class_name for item in carriers)),
        "sources_with_carriers": sum(value > 0 for value in source_candidate_counts.values()),
        "sources_without_carriers": sum(value == 0 for value in source_candidate_counts.values()),
        "excluded_anchor_source_counts": dict(excluded_counts),
        "rejection_counts": dict(rejection_counts),
    }
    return carriers, summary


def clean_waveforms(rows: list[dict[str, str]]) -> np.ndarray:
    time_s = (
        np.arange(RAW_LENGTH, dtype=np.float64) - (RAW_LENGTH - 1) / 2.0
    ) / SAMPLING_FREQUENCY_HZ
    output = np.empty((len(rows), RAW_LENGTH), dtype=np.float64)
    for index, row in enumerate(rows):
        amplitude = float(row["amplitude_p0"])
        frequency_hz = float(row["frequency_khz"]) * 1000.0
        tau_s = float(row["tau_ms"]) / 1000.0
        phase = float(row["phi_rad"])
        envelope = amplitude * np.exp(-0.5 * np.square(time_s / tau_s))
        output[index] = envelope * np.cos(
            2.0 * np.pi * frequency_hz * time_s + phase
        )
    return output


def validate_frozen_rows(
    baseline_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, Any]],
) -> None:
    mutable = {
        "achieved_snr_db",
        "noise_rms",
        "noise_model",
        "noise_source_split",
        "noise_source_relative_path",
        "noise_start_sample",
        "noise_end_sample",
        "noise_source_round",
        "noise_carrier_sha256",
        "noise_carrier_rms",
        "noise_carrier_use_index",
    }
    if len(baseline_rows) != len(candidate_rows):
        raise ValueError("candidate event count differs from v3")
    for baseline, candidate in zip(baseline_rows, candidate_rows, strict=True):
        for field, value in baseline.items():
            if field not in mutable and str(candidate[field]) != str(value):
                raise ValueError(f"frozen v3 field changed: {field}")


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
    parser.add_argument("--anchors-csv", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--window-stride", type=int, default=32)
    parser.add_argument("--maximum-per-source", type=int, default=256)
    parser.add_argument("--minimum-start-separation", type=int, default=64)
    args = parser.parse_args()

    workspace = Workspace.load()
    dataset_output = args.dataset_output_dir.resolve()
    run_output = args.run_output_dir.resolve()
    anchors_path = args.anchors_csv.resolve()
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
    excluded_sources, anchors_sha256 = anchor_sources(anchors_path)

    baseline_rows = read_csv(baseline_root / "events.csv")
    if class_counts(baseline_rows) != TARGET_COUNTS:
        raise ValueError("baseline v3 class counts changed")
    carriers, carrier_summary = extract_carrier_refs(
        source_root,
        excluded_sources=excluded_sources,
        window_stride=args.window_stride,
        maximum_per_source=args.maximum_per_source,
        minimum_start_separation=args.minimum_start_separation,
    )
    assigned_by_class = {
        class_name: round_robin_carriers(
            carriers,
            class_name=class_name,
            required=TARGET_COUNTS[class_name],
            seed=args.seed + index,
            allow_reuse_after_exhaustion=False,
        )
        for index, class_name in enumerate(TARGET_COUNTS)
    }
    offsets = defaultdict(int)
    assigned: list[CarrierRef] = []
    for row in baseline_rows:
        class_name = row["class_name"]
        assigned.append(assigned_by_class[class_name][offsets[class_name]])
        offsets[class_name] += 1
    assigned_keys = {
        (item.source_relative_path, item.start_sample, item.end_sample)
        for item in assigned
    }
    if len(assigned_keys) != len(assigned):
        raise ValueError("a real carrier interval was assigned more than once")

    dataset_output.mkdir(parents=True)
    raw_path = dataset_output / "signals_raw_4096.npy"
    model_path = dataset_output / "signals_conv1dgap_512.npy"
    raw_output = open_memmap(
        raw_path, mode="w+", dtype=np.float32, shape=(len(baseline_rows), RAW_LENGTH)
    )
    model_output = open_memmap(
        model_path, mode="w+", dtype=np.float32, shape=(len(baseline_rows), 512)
    )
    metadata: list[dict[str, Any] | None] = [None] * len(baseline_rows)
    groups: dict[str, list[tuple[int, CarrierRef]]] = defaultdict(list)
    for index, carrier in enumerate(assigned):
        groups[carrier.source_relative_path].append((index, carrier))
    seen_hashes: set[str] = set()
    maximum_snr_error = 0.0
    for relative, group in groups.items():
        source = np.load(source_root / relative, allow_pickle=False)
        for chunk_start in range(0, len(group), 256):
            chunk = group[chunk_start : chunk_start + 256]
            indices = np.asarray([item[0] for item in chunk], dtype=np.int64)
            rows = [baseline_rows[index] for index in indices]
            clean = clean_waveforms(rows)
            noise = np.stack(
                [
                    np.asarray(
                        source[carrier.start_sample : carrier.end_sample],
                        dtype=np.float64,
                    )
                    for _, carrier in chunk
                ]
            )
            noise -= np.mean(noise, axis=1, keepdims=True)
            carrier_rms = np.sqrt(np.mean(np.square(noise), axis=1))
            clean_rms = np.sqrt(np.mean(np.square(clean), axis=1))
            requested_snr = np.asarray(
                [float(row["snr_db"]) for row in rows], dtype=np.float64
            )
            target_noise_rms = clean_rms / np.power(10.0, requested_snr / 20.0)
            scaled_noise = noise * (target_noise_rms / carrier_rms)[:, None]
            noisy = (clean + scaled_noise).astype(np.float32)
            achieved = 20.0 * np.log10(
                clean_rms / np.sqrt(np.mean(np.square(scaled_noise), axis=1))
            )
            maximum_snr_error = max(
                maximum_snr_error,
                float(np.max(np.abs(achieved - requested_snr))),
            )
            raw_output[indices] = noisy
            model_output[indices] = preprocess_conv1dgap_512(noisy)
            for local_index, (event_index, carrier) in enumerate(chunk):
                digest = carrier_sha256(noise[local_index].astype(np.float32))
                if digest in seen_hashes:
                    raise ValueError("a real carrier hash was assigned more than once")
                seen_hashes.add(digest)
                metadata[event_index] = {
                    "noise_source_split": carrier.split,
                    "noise_source_relative_path": carrier.source_relative_path,
                    "noise_start_sample": carrier.start_sample,
                    "noise_end_sample": carrier.end_sample,
                    "noise_source_round": carrier.source_round,
                    "noise_carrier_sha256": digest,
                    "noise_carrier_rms": carrier.rms,
                    "noise_carrier_use_index": 0,
                    "noise_rms": float(target_noise_rms[local_index]),
                    "achieved_snr_db": float(achieved[local_index]),
                }
    raw_output.flush()
    model_output.flush()
    if any(item is None for item in metadata):
        raise AssertionError("missing generated-event metadata")
    if maximum_snr_error > 1.0e-9:
        raise ValueError("candidate SNR mismatch")
    candidate_rows = [
        {
            **row,
            **item,
            "noise_model": "p2snrf_v2_event_free_real_carrier_unique",
        }
        for row, item in zip(baseline_rows, metadata, strict=True)
    ]
    validate_frozen_rows(baseline_rows, candidate_rows)
    write_csv(dataset_output / "events.csv", candidate_rows)
    write_csv(
        dataset_output / "carrier_manifest.csv",
        [
            {
                "sample_id": row["sample_id"],
                "class_name": row["class_name"],
                **item,
            }
            for row, item in zip(baseline_rows, metadata, strict=True)
        ],
    )

    summary = {
        "schema_version": 1,
        "dataset_id": args.dataset_id,
        "status": "interim_reference_candidate_awaiting_scientific_result",
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
        },
        "method_evidence": method,
        "frozen_anchor_sources": {
            "count": len(excluded_sources),
            "anchors_csv_sha256": anchors_sha256,
        },
        "carrier_policy": {
            "splits": ["train", "val"],
            "class_matched": True,
            "window_length": RAW_LENGTH,
            "window_stride": args.window_stride,
            "annotation_guard_samples": 804,
            "repair_guard": "expanded repair interval plus documented filter radius",
            "maximum_per_source": args.maximum_per_source,
            "minimum_start_separation": args.minimum_start_separation,
            "selection": "per-source low-RMS then deterministic source round-robin",
            "exact_interval_reuse": False,
            "exact_hash_reuse": False,
        },
        "carrier_summary": carrier_summary,
        "paired_contract": {
            "frozen": "all v3 event fields except achieved_snr_db and noise_rms",
            "changed_only": "noise carrier and noise-derived fields",
            "maximum_snr_error_db": maximum_snr_error,
        },
    }
    (dataset_output / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (dataset_output / "input_contract.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "full paired real-noise synthetic gallery",
                "events": "events.csv",
                "carrier_manifest": "carrier_manifest.csv",
                "raw_signals": "signals_raw_4096.npy",
                "model_signals": "signals_conv1dgap_512.npy",
                "review_gate": "reference candidate only; active promotion requires result",
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
        "assigned_unique_interval_count": len(assigned_keys),
        "assigned_unique_hash_count": len(seen_hashes),
        "excluded_anchor_source_count": len(excluded_sources),
        "maximum_snr_error_db": maximum_snr_error,
        "raw_shape": list(np.load(raw_path, mmap_mode="r", allow_pickle=False).shape),
        "model_shape": list(np.load(model_path, mmap_mode="r", allow_pickle=False).shape),
        "sealed_test_accessed": False,
    }
    (run_output / "generation_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    run = {
        "schema_version": 1,
        "project": "particles2SNR-pipeline",
        "run_id": args.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_awaiting_scientific_analysis",
        "dataset": args.dataset_id,
        "command": shlex.join([sys.executable, *sys.argv]),
        "repositories": {
            "workspace": git_revision(workspace.root),
            "particles2SNR-pipeline": git_revision(
                workspace.root / "particles2SNR-pipeline"
            ),
        },
        "sealed_test_accessed": False,
        "outputs": ["generation_validation.json"],
    }
    (run_output / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"dataset": str(dataset_output), "validation": validation}, indent=2))


if __name__ == "__main__":
    main()
