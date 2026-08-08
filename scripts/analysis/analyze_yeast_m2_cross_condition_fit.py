#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from particles2snr.yeast_m2_cross_condition_audit import (
    fit_shard,
    load_split_roles,
    parameter_statistics,
    read_csv,
    select_gallery,
    select_strict_event_population,
    sha256_file,
    summarize_groups,
    validate_method_approval,
    validate_merged_population,
    write_csv,
)


DATASET_ID = "yeast-budding-mix-shmoo-background-classification@v2"
METHOD_EVIDENCE_ID = "yeast-physics-grounded-classifier-method-r1"


def _git_revision(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def _base_inputs(dataset_root: Path, split_manifest: Path) -> tuple[list[dict[str, Any]], np.ndarray]:
    contract = json.loads((dataset_root / "dataset-contract.json").read_text(encoding="utf-8"))
    if contract.get("dataset_id") != DATASET_ID:
        raise ValueError(f"Unexpected dataset: {contract.get('dataset_id')}")
    if contract.get("sealed_holdout_accessed") is not False:
        raise ValueError("Dataset contract must state that the sealed holdout was not accessed")
    rows = read_csv(dataset_root / "samples.csv")
    roles = load_split_roles(split_manifest)
    population = select_strict_event_population(rows, roles)
    signals = np.load(dataset_root / "signals.npy", mmap_mode="r")
    if signals.shape[1:] != (4096,) or signals.dtype != np.float32:
        raise ValueError(f"Unexpected signal tensor: {signals.shape} {signals.dtype}")
    return population, signals


def _run_shard(args: argparse.Namespace) -> None:
    validate_method_approval(args.method_review_dir)
    if args.shard_index is None:
        raise ValueError("--shard-index is required in shard mode")
    population, signals = _base_inputs(args.dataset_root, args.split_manifest)
    rows = fit_shard(
        population,
        signals,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        max_events=args.max_events,
        max_events_per_class=args.max_events_per_class,
    )
    if not rows:
        raise ValueError(f"Shard {args.shard_index} selected no events")
    shard_dir = args.output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    table = shard_dir / f"shard-{args.shard_index:02d}.csv"
    receipt = shard_dir / f"shard-{args.shard_index:02d}.json"
    if table.exists() or receipt.exists():
        if args.resume and table.exists() and receipt.exists():
            previous = json.loads(receipt.read_text(encoding="utf-8"))
            if (
                previous.get("shard_index") == args.shard_index
                and previous.get("shard_count") == args.shard_count
                and previous.get("table_sha256") == sha256_file(table)
            ):
                print(json.dumps({"shard": args.shard_index, "status": "already_complete"}))
                return
        raise FileExistsError(f"Refusing to overwrite shard {args.shard_index}")
    write_csv(table, rows)
    receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
                "row_count": len(rows),
                "table": table.name,
                "table_sha256": sha256_file(table),
                "numeric_threads": {
                    key: os.environ.get(key, "")
                    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"shard": args.shard_index, "rows": len(rows), "output": str(table)}))


def _run_merge(args: argparse.Namespace) -> None:
    approval = validate_method_approval(args.method_review_dir)
    population, _signals = _base_inputs(args.dataset_root, args.split_manifest)
    shard_dir = args.output_dir / "shards"
    expected_tables = [shard_dir / f"shard-{index:02d}.csv" for index in range(args.shard_count)]
    expected_receipts = [shard_dir / f"shard-{index:02d}.json" for index in range(args.shard_count)]
    missing = [str(path) for path in [*expected_tables, *expected_receipts] if not path.exists()]
    if missing:
        raise ValueError(f"Missing shard artifacts: {missing[:3]}")
    fit_rows: list[dict[str, Any]] = []
    for index, (table, receipt_path) in enumerate(zip(expected_tables, expected_receipts, strict=True)):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt["shard_index"] != index or receipt["shard_count"] != args.shard_count:
            raise ValueError(f"Shard receipt mismatch: {receipt_path}")
        if receipt["table_sha256"] != sha256_file(table):
            raise ValueError(f"Shard hash mismatch: {table}")
        fit_rows.extend(read_csv(table))
    fit_rows.sort(key=lambda row: row["event_id"])
    validate_merged_population(population, fit_rows)
    targets_path = args.output_dir / "m2_targets.csv"
    if targets_path.exists():
        raise FileExistsError(f"Refusing to overwrite completed target table: {targets_path}")
    write_csv(targets_path, fit_rows)
    summary = summarize_groups(fit_rows)
    parameters = parameter_statistics(fit_rows)
    gallery = select_gallery(fit_rows)
    write_csv(args.output_dir / "per_group_summary.csv", summary)
    write_csv(args.output_dir / "parameter_statistics.csv", parameters)
    (args.output_dir / "gallery_selection.json").write_text(
        json.dumps(gallery, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    final_status = {
        row["group"]: row["threshold_status"]
        for row in summary
        if str(row["is_final_class"]).lower() == "true" or row["is_final_class"] is True
    }
    metrics = {
        "schema_version": 1,
        "method_evidence_id": METHOD_EVIDENCE_ID,
        "dataset_id": DATASET_ID,
        "population_events": len(population),
        "fit_rows": len(fit_rows),
        "fit_failures": sum(str(row["fit_success"]).lower() != "true" for row in fit_rows),
        "group_summary": summary,
        "final_class_threshold_status": final_status,
        "thresholds": {
            "delta_bic_m1_minus_m2": 10.0,
            "resolvability": 0.1,
            "fit_guard_samples": 250,
            "ready": "eligible >= 300 and eligible/observable >= 0.5",
            "partial": "eligible >= 100",
            "synthetic_only": "eligible < 100",
        },
        "sealed_holdout_accessed": False,
        "development_validation_accessed": False,
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    workspace_root = Path(__file__).resolve().parents[3]
    provenance = {
        "code": {
            "runner": "particles2SNR-pipeline/scripts/analysis/analyze_yeast_m2_cross_condition_fit.py",
            "runner_sha256": sha256_file(Path(__file__)),
            "library": "particles2SNR-pipeline/particles2snr/yeast_m2_cross_condition_audit.py",
            "library_sha256": sha256_file(Path(__file__).resolve().parents[2] / "particles2snr" / "yeast_m2_cross_condition_audit.py"),
            "canonical_fitter": "particles2SNR-pipeline/particles2snr/yeast_budding_simulation.py",
            "canonical_fitter_sha256": sha256_file(Path(__file__).resolve().parents[2] / "particles2snr" / "yeast_budding_simulation.py"),
        },
        "datasets": [
            {
                "id": DATASET_ID,
                "manifest_sha256": sha256_file(args.dataset_root / "dataset-manifest.json"),
            }
        ],
        "git_revision": {
            "workspace": _git_revision(workspace_root),
            "particles2SNR-pipeline": _git_revision(workspace_root / "particles2SNR-pipeline"),
        },
        "inputs": {
            "dataset_manifest_sha256": sha256_file(args.dataset_root / "dataset-manifest.json"),
            "split_manifest_sha256": sha256_file(args.split_manifest),
            "method_approval_receipt_sha256": approval["receipt_sha256"],
            "shards": [
                {"path": str(path.relative_to(args.output_dir)), "sha256": sha256_file(path)}
                for path in expected_tables
            ],
        },
        "parameters": {
            "shard_count": args.shard_count,
            "component_order": "A is earlier than B",
            "fit_weight": "sqrt(clip(delta_bic/50,0,1)*clip(resolvability,0,1)); zero unless eligible",
            "source_partition": "development_train only",
        },
        "metric_definitions": {
            "fit_eligible": "finite M2 fit with delta-BIC >= 10, resolvability >= 0.1, and fully observed event support plus a 250-sample guard",
            "eligible_fraction_of_observable": "eligible M2 fits divided by strict events whose full support and guard are observed",
            "effective_sample_size": "Kish effective sample size computed from the preregistered fit weights",
            "threshold_status": "ready when eligible >= 300 and eligible/observable >= 0.5; partial when eligible >= 100; synthetic_only otherwise",
            "waveform_residual_fraction": "root-mean-square waveform residual divided by root-mean-square observed waveform on the fit support",
        },
        "repositories": {
            "workspace": _git_revision(workspace_root),
            "particles2SNR-pipeline": _git_revision(workspace_root / "particles2SNR-pipeline"),
        },
    }
    fingerprint_fields = (
        "code",
        "datasets",
        "git_revision",
        "inputs",
        "metric_definitions",
        "parameters",
    )
    fingerprint = hashlib.sha256(
        json.dumps(
            {key: provenance[key] for key in fingerprint_fields},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    metric_files = (
        "metrics.json",
        "m2_targets.csv",
        "per_group_summary.csv",
        "parameter_statistics.csv",
        "gallery_selection.json",
    )
    manifest = {
        "schema_version": 1,
        "analysis_run_id": args.run_id,
        "computation_fingerprint": fingerprint,
        "computation_provenance": provenance,
        "metrics": [
            {"path": name, "sha256": sha256_file(args.output_dir / name), "computation_fingerprint": fingerprint}
            for name in metric_files
        ],
    }
    (args.output_dir / "metrics_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    run = {
        "schema_version": 1,
        "project": "particles2SNR-pipeline",
        "run_id": args.run_id,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": f"analyze_yeast_m2_cross_condition_fit.py --mode merge --shard-count {args.shard_count}",
        "dataset": DATASET_ID,
        "datasets": {"classification": {"id": DATASET_ID, "manifest_sha256": provenance["inputs"]["dataset_manifest_sha256"]}},
        "method_evidence_id": METHOD_EVIDENCE_ID,
        "repositories": provenance["repositories"],
        "sealed_holdout_accessed": False,
        "development_validation_accessed": False,
        "outputs": list(metric_files) + ["metrics_manifest.json", "shards/"],
    }
    (args.output_dir / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("shard", "merge"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--method-review-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", default="yeast-m2-cross-condition-fit-audit-r1")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int, default=16)
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--max-events-per-class", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_events < 0 or args.max_events_per_class < 0:
        raise ValueError("smoke limits must be non-negative")
    if args.mode == "merge" and (args.max_events or args.max_events_per_class):
        raise ValueError("Merge mode cannot be used with a truncated smoke population")
    if args.mode == "shard":
        _run_shard(args)
    else:
        _run_merge(args)


if __name__ == "__main__":
    main()
