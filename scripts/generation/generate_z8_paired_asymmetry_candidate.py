#!/usr/bin/env python3
"""Generate a one-to-one asymmetry-enriched candidate from immutable synthetic v4."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.format import open_memmap

from internship_workspace.config import Workspace
from internship_workspace.datasets import resolve_path, select_record
from internship_workspace.scientific_visual import computation_fingerprint
from internship_workspace.visual_review_store import ReviewStore
from internship_workspace.z8_twin_analysis import sha256_file
from particles2snr.z8_asymmetry_generation import (
    PARAMETER_ORDER_5D,
    load_asymmetry_targets,
    load_regularized_5d_correlations,
    paired_asymmetric_signals,
    read_csv,
    sample_paired_asymmetry,
)
from particles2snr.z8_cholesky_generation import MODEL_LENGTH, RAW_LENGTH
from particles2snr.z8_parameter_analysis import CLASS_ORDER


METHOD_RUN_ID = "particle-z8-v2-asymmetry-generation-pca-method-r2"
METHOD_EVIDENCE_ID = "particle-z8-v2-asymmetry-generation-pca-method"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def git_revision(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def verify_method(workspace: Workspace) -> dict[str, str]:
    store = ReviewStore(workspace.artifacts_root / "cross-project/reviews" / METHOD_RUN_ID)
    receipt = store.verify_receipt()
    decision = store.current()["decisions"][METHOD_EVIDENCE_ID]["decision"]
    if decision != "approved":
        raise PermissionError("paired asymmetry generation method is not approved")
    return {
        "run_id": METHOD_RUN_ID,
        "evidence_id": METHOD_EVIDENCE_ID,
        "decision": decision,
        "receipt_sha256": sha256_file(store.receipt_path),
        "contract_sha256": str(receipt["contract_sha256"]),
    }


def class_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {name: sum(str(row["class_name"]) == name for row in rows) for name in CLASS_ORDER}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dataset", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--targets-run", type=Path, required=True)
    parser.add_argument("--dataset-output-dir", type=Path, required=True)
    parser.add_argument("--run-output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    workspace = Workspace.load()
    dataset_output = args.dataset_output_dir.resolve()
    run_output = args.run_output_dir.resolve()
    if not dataset_output.is_relative_to(workspace.datasets_root / "interim"):
        raise ValueError("candidate dataset must be written below datasets/interim")
    if not run_output.is_relative_to(workspace.artifacts_root):
        raise ValueError("analysis run must be written below artifacts")
    if dataset_output.exists() or run_output.exists():
        raise FileExistsError("refusing to overwrite candidate or run")
    method = verify_method(workspace)
    baseline_id, _, baseline_version = args.baseline_dataset.rpartition("@")
    baseline_record = select_record(workspace, baseline_id, baseline_version)
    baseline_root = resolve_path(workspace, baseline_record)
    baseline_rows = read_csv(baseline_root / "events.csv")
    if len(baseline_rows) != 47_980:
        raise ValueError(f"expected 47,980 v4 events, got {len(baseline_rows)}")
    if any(str(row.get("noise_model")) != "p2snrf_v2_event_free_real_carrier_unique" for row in baseline_rows):
        raise ValueError("baseline is not the full-real-noise v4 gallery")

    targets_root = (args.targets_run if args.targets_run.is_absolute() else workspace.root / args.targets_run).resolve()
    targets = load_asymmetry_targets(targets_root / "gaussian_asymmetry_targets.csv")
    correlations = load_regularized_5d_correlations(targets_root / "dependence_matrices.csv")
    draws = sample_paired_asymmetry(
        baseline_rows, correlations=correlations, targets=targets, seed=args.seed
    )
    dataset_output.mkdir(parents=True)
    raw_path = dataset_output / "signals_raw_4096.npy"
    model_path = dataset_output / "signals_conv1dgap_512.npy"
    raw_output = open_memmap(raw_path, mode="w+", dtype=np.float32, shape=(len(baseline_rows), RAW_LENGTH))
    model_output = open_memmap(model_path, mode="w+", dtype=np.float32, shape=(len(baseline_rows), MODEL_LENGTH))
    baseline_raw = np.load(baseline_root / "signals_raw_4096.npy", mmap_mode="r", allow_pickle=False)
    candidate_rows: list[dict[str, Any]] = []
    maximum_snr_error = 0.0
    for start in range(0, len(baseline_rows), args.batch_size):
        stop = min(len(baseline_rows), start + args.batch_size)
        rows = baseline_rows[start:stop]
        chunk_draws = draws[start:stop]
        asymmetry = np.asarray([float(row["waveform_asymmetry"]) for row in chunk_draws])
        raw, model, achieved, clean_rms, noise_rms = paired_asymmetric_signals(
            rows, np.asarray(baseline_raw[start:stop]), asymmetry
        )
        raw_output[start:stop] = raw
        model_output[start:stop] = model
        requested = np.asarray([float(row["snr_db"]) for row in rows])
        maximum_snr_error = max(maximum_snr_error, float(np.max(np.abs(achieved - requested))))
        for local, (baseline, draw) in enumerate(zip(rows, chunk_draws, strict=True)):
            candidate_rows.append({
                **baseline,
                "paired_baseline_sample_id": baseline["sample_id"],
                "paired_baseline_index": start + local,
                "z5": draw["z5"],
                "u5": draw["u5"],
                "transformed_asymmetry": draw["transformed_asymmetry"],
                "waveform_asymmetry": draw["waveform_asymmetry"],
                "asymmetry_rejection_attempts": draw["asymmetry_rejection_attempts"],
                "clean_rms": float(clean_rms[local]),
                "noise_rms": float(noise_rms[local]),
                "scaled_noise_rms": float(noise_rms[local]),
                "achieved_snr_db": float(achieved[local]),
                "generation_batch": "v5_paired_asymmetry",
                "baseline_member": False,
            })
    raw_output.flush()
    model_output.flush()
    if maximum_snr_error > 1.0e-9:
        raise ValueError(f"SNR realization mismatch: {maximum_snr_error}")

    frozen = ("sample_id", "class_name", "amplitude_p0", "frequency_khz", "tau_ms", "snr_db", "phi_rad", "t0_fraction", "noise_source_relative_path", "noise_start_sample", "noise_end_sample", "noise_carrier_sha256")
    for baseline, candidate in zip(baseline_rows, candidate_rows, strict=True):
        for field in frozen:
            if str(baseline[field]) != str(candidate[field]):
                raise ValueError(f"paired baseline field changed: {field}")
    write_csv(dataset_output / "events.csv", candidate_rows)

    correlation_rows = []
    max_correlation_delta = 0.0
    for class_name in CLASS_ORDER:
        selected = [row for row in candidate_rows if row["class_name"] == class_name]
        realized = np.corrcoef(
            np.asarray([[float(row[f"u{i}"]) for i in range(1, 6)] for row in selected]),
            rowvar=False,
        )
        target = correlations[class_name]
        for i, row_parameter in enumerate(PARAMETER_ORDER_5D):
            for j, column_parameter in enumerate(PARAMETER_ORDER_5D):
                delta = float(realized[i, j] - target[i, j])
                if i != j:
                    max_correlation_delta = max(max_correlation_delta, abs(delta))
                correlation_rows.append({"class_name": class_name, "row_parameter": row_parameter, "column_parameter": column_parameter, "target_correlation": float(target[i, j]), "realized_correlation": float(realized[i, j]), "delta": delta})

    summary = {
        "schema_version": 1,
        "dataset_id": args.dataset_id,
        "status": "interim_paired_candidate_awaiting_scientific_result",
        "event_count": len(candidate_rows),
        "class_counts": class_counts(candidate_rows),
        "paired_baseline": {"id": args.baseline_dataset, "manifest_sha256": baseline_record.payload["manifest_sha256"], "events_sha256": sha256_file(baseline_root / "events.csv")},
        "paired_contract": {"changed_only": "waveform asymmetry plus clean/noise scale required to preserve target SNR", "frozen_fields": list(frozen), "maximum_snr_error_db": maximum_snr_error},
        "asymmetry_policy": {"transform": "atanh(a/0.8)", "conditional_coordinates": ["u1", "u2", "u3", "u4"], "support": "class-specific retained observed transformed min-max", "seed": args.seed},
        "method_evidence": method,
        "targets_run": {"run_id": targets_root.name, "metrics_manifest_sha256": sha256_file(targets_root / "metrics_manifest.json")},
        "sealed_test_accessed": False,
    }
    (dataset_output / "dataset_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (dataset_output / "input_contract.json").write_text(json.dumps({"schema_version": 1, "format": "paired asymmetry synthetic candidate", "events": "events.csv", "raw_signals": "signals_raw_4096.npy", "model_signals": "signals_conv1dgap_512.npy", "baseline_identity": "paired_baseline_sample_id and paired_baseline_index", "review_gate": "no registration or promotion before paired PCA result"}, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    run_output.mkdir(parents=True)
    write_csv(run_output / "correlation_validation.csv", correlation_rows)
    validation = {"schema_version": 1, "event_count": len(candidate_rows), "class_counts": class_counts(candidate_rows), "maximum_snr_error_db": maximum_snr_error, "maximum_absolute_5d_correlation_delta": max_correlation_delta, "raw_shape": list(np.load(raw_path, mmap_mode="r", allow_pickle=False).shape), "model_shape": list(np.load(model_path, mmap_mode="r", allow_pickle=False).shape), "finite_signals": bool(np.isfinite(np.load(model_path, mmap_mode="r", allow_pickle=False)).all()), "sealed_test_accessed": False}
    (run_output / "generation_validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    provenance = {
        "datasets": {args.baseline_dataset: baseline_record.payload["manifest_sha256"]},
        "inputs": {"baseline_events_sha256": sha256_file(baseline_root / "events.csv"), "baseline_raw_sha256": sha256_file(baseline_root / "signals_raw_4096.npy"), "targets_metrics_manifest_sha256": sha256_file(targets_root / "metrics_manifest.json"), "method": method},
        "parameters": {"seed": args.seed, "event_count": len(candidate_rows), "batch_size": args.batch_size, "paired_frozen_fields": list(frozen), "conditional_formula": "u5|u1:4 from partitioned 5D correlation", "support_policy": "rejection within observed transformed retained support"},
        "metric_definitions": {"maximum_snr_error_db": "max absolute requested-minus-realized RMS SNR", "maximum_absolute_5d_correlation_delta": "max off-diagonal absolute realized-minus-target correlation among u1-u5"},
        "code": {"entrypoint": Path(__file__).relative_to(workspace.root).as_posix(), "entrypoint_sha256": sha256_file(Path(__file__).resolve()), "module_sha256": sha256_file(workspace.root / "particles2SNR-pipeline/particles2snr/z8_asymmetry_generation.py")},
        "git_revision": git_revision(workspace.root),
    }
    fingerprint = computation_fingerprint(provenance)
    (run_output / "metrics_manifest.json").write_text(json.dumps({"schema_version": 1, "analysis_run_id": args.run_id, "computation_provenance": provenance, "computation_fingerprint": fingerprint, "metrics": [{"path": "generation_validation.json", "sha256": sha256_file(run_output / "generation_validation.json"), "computation_fingerprint": fingerprint}, {"path": "correlation_validation.csv", "sha256": sha256_file(run_output / "correlation_validation.csv"), "computation_fingerprint": fingerprint}]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    candidate_files = ["events.csv", "signals_raw_4096.npy", "signals_conv1dgap_512.npy", "dataset_summary.json", "input_contract.json"]
    (run_output / "candidate_dataset_manifest.json").write_text(json.dumps({"schema_version": 1, "dataset_id": args.dataset_id, "status": "interim_candidate", "files": [{"path": name, "size": (dataset_output / name).stat().st_size, "sha256": sha256_file(dataset_output / name)} for name in candidate_files]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    run = {"schema_version": 1, "project": "particles2SNR-pipeline", "run_id": args.run_id, "created_at": datetime.now(timezone.utc).isoformat(), "status": "complete_awaiting_paired_pca", "dataset": args.dataset_id, "command": shlex.join([sys.executable, *sys.argv]), "repositories": {"workspace": git_revision(workspace.root), "particles2SNR-pipeline": git_revision(workspace.root / "particles2SNR-pipeline")}, "computation_fingerprint": fingerprint, "sealed_test_accessed": False, "outputs": ["generation_validation.json", "correlation_validation.csv", "metrics_manifest.json", "candidate_dataset_manifest.json"]}
    (run_output / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": str(dataset_output), "validation": validation}, indent=2))


if __name__ == "__main__":
    main()
