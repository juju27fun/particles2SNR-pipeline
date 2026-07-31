#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from internship_workspace.config import Workspace
from internship_workspace.datasets import load_records
from internship_workspace.scientific_visual import computation_fingerprint
from internship_workspace.visual_review_store import ReviewStore
from particles2snr.z8_cholesky_generation import (
    correlation_validation,
    generate_parameters,
    load_gaussian_targets,
    load_recommended_cholesky,
    synthesize_signals,
    validate_candidate,
    write_analysis_outputs,
    write_candidate_dataset,
)
from particles2snr.z8_density_gallery import (
    BASE_COUNTS,
    EXTENSION_COUNTS,
    TARGET_COUNTS,
    build_nested_density10x_gallery,
)
from particles2snr.z8_reference_dataset import sha256_file


DEFAULT_EXTENSION_SEED = 20_260_728


def _record(workspace: Workspace, key: str) -> dict[str, Any]:
    match = next(
        (
            record.payload
            for record in load_records(workspace)
            if f"{record.payload['id']}@{record.payload['version']}" == key
        ),
        None,
    )
    if match is None or match["status"] not in {"active", "reference"}:
        raise ValueError(f"eligible registered dataset not found: {key}")
    return match


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _git_state(path: Path) -> dict[str, Any]:
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


def _approved_method(
    workspace: Workspace, evidence_id: str, evidence_run_id: str
) -> dict[str, str]:
    root = workspace.root / "artifacts/cross-project/reviews" / evidence_run_id
    run = json.loads((root / "run.json").read_text(encoding="utf-8"))
    store = ReviewStore(root)
    receipt = store.verify_receipt()
    decision = (
        store.current().get("decisions", {}).get(evidence_id, {}).get("decision")
    )
    if (
        run.get("evidence_id") != evidence_id
        or run.get("visual_checkpoint", {}).get("approved") is not True
        or run.get("visual_checkpoint", {}).get("next_stage_blocked") is True
        or decision not in {"approved", "supported"}
    ):
        raise PermissionError(f"method evidence does not authorize generation: {root}")
    return {
        "evidence_id": evidence_id,
        "evidence_run_id": evidence_run_id,
        "decision": str(decision),
        "receipt_sha256": sha256_file(store.receipt_path),
        "contract_sha256": str(receipt["contract_sha256"]),
    }


def _load_baseline(
    root: Path,
) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray, dict[str, Any]]:
    rows = _read_csv(root / "events.csv")
    raw = np.load(root / "signals_raw_4096.npy", mmap_mode="r", allow_pickle=False)
    model = np.load(
        root / "signals_conv1dgap_512.npy", mmap_mode="r", allow_pickle=False
    )
    summary = json.loads((root / "dataset_summary.json").read_text(encoding="utf-8"))
    if (
        summary.get("event_count") != sum(BASE_COUNTS.values())
        or summary.get("class_counts") != BASE_COUNTS
        or summary.get("sealed_test_accessed") is not False
    ):
        raise ValueError("baseline synthetic v2 contract changed")
    return rows, raw, model, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the nested Z8-v2 synthetic gallery at 10× density."
    )
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--baseline-dataset", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gaussian-run-id", required=True)
    parser.add_argument("--cholesky-run-id", required=True)
    parser.add_argument("--method-evidence-id", required=True)
    parser.add_argument("--method-evidence-run-id", required=True)
    parser.add_argument("--dataset-output-dir", type=Path, required=True)
    parser.add_argument("--run-output-dir", type=Path, required=True)
    parser.add_argument("--extension-seed", type=int, default=DEFAULT_EXTENSION_SEED)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    workspace = Workspace.load()
    dataset_output = args.dataset_output_dir.resolve()
    run_output = args.run_output_dir.resolve()
    if not dataset_output.is_relative_to(workspace.datasets_root / "interim"):
        raise ValueError("--dataset-output-dir must be below datasets/interim")
    if not run_output.is_relative_to(workspace.artifacts_root):
        raise ValueError("--run-output-dir must be below artifacts")
    if dataset_output.exists() or run_output.exists():
        raise FileExistsError("refusing to overwrite an existing candidate or run")

    source_record = _record(workspace, args.source_dataset)
    baseline_record = _record(workspace, args.baseline_dataset)
    if baseline_record["status"] != "active":
        raise ValueError("the immutable v2 baseline must be the active dataset")
    method_evidence = _approved_method(
        workspace, args.method_evidence_id, args.method_evidence_run_id
    )
    baseline_root = workspace.datasets_root / baseline_record["path"]
    baseline_rows, baseline_raw, baseline_model, baseline_summary = _load_baseline(
        baseline_root
    )

    gaussian_run = (
        workspace.root
        / "artifacts/particles2SNR-pipeline/analysis"
        / args.gaussian_run_id
    )
    cholesky_run = (
        workspace.root
        / "artifacts/particles2SNR-pipeline/analysis"
        / args.cholesky_run_id
    )
    gaussian_parameters = gaussian_run / "gaussian_envelope_parameters.csv"
    cholesky_factors = cholesky_run / "cholesky_factors.csv"
    cholesky_recommendations = cholesky_run / "recommendations.csv"
    targets, approved_budgets = load_gaussian_targets(
        gaussian_parameters, include_budgets=True
    )
    if approved_budgets != BASE_COUNTS:
        raise ValueError("approved Gaussian budgets no longer match synthetic v2")
    factors, dependency_populations = load_recommended_cholesky(
        cholesky_factors, cholesky_recommendations
    )

    extension_rows, rejection_counts = generate_parameters(
        targets,
        factors,
        seed=args.extension_seed,
        budgets=EXTENSION_COUNTS,
        dataset_id=args.dataset_id,
    )
    extension_raw, extension_model = synthesize_signals(
        extension_rows,
        seed=args.extension_seed + 1,
        batch_size=args.batch_size,
    )
    rows, raw, model = build_nested_density10x_gallery(
        baseline_rows,
        baseline_raw,
        baseline_model,
        extension_rows,
        extension_raw,
        extension_model,
    )

    gaussian_fingerprint = json.loads(
        (gaussian_run / "run.json").read_text(encoding="utf-8")
    )["computation_fingerprint"]
    cholesky_fingerprint = json.loads(
        (cholesky_run / "run.json").read_text(encoding="utf-8")
    )["computation_fingerprint"]
    outputs = write_candidate_dataset(
        dataset_id=args.dataset_id,
        source_dataset_id=args.source_dataset,
        gaussian_run_id=args.gaussian_run_id,
        cholesky_run_id=args.cholesky_run_id,
        dependency_populations=dependency_populations,
        seed=args.extension_seed,
        output_dir=dataset_output,
        records=rows,
        raw_signals=raw,
        model_signals=model,
        rejection_counts=rejection_counts,
        source_manifest_sha256=source_record["manifest_sha256"],
        gaussian_run_fingerprint=gaussian_fingerprint,
        cholesky_run_fingerprint=cholesky_fingerprint,
    )
    summary_path = dataset_output / "dataset_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["status"] = "reference_candidate_awaiting_twin_result"
    summary["nested_gallery"] = {
        "baseline_dataset_id": args.baseline_dataset,
        "baseline_manifest_sha256": baseline_record["manifest_sha256"],
        "baseline_event_count": len(baseline_rows),
        "baseline_class_counts": BASE_COUNTS,
        "baseline_events_sha256": sha256_file(baseline_root / "events.csv"),
        "baseline_raw_sha256": sha256_file(
            baseline_root / "signals_raw_4096.npy"
        ),
        "baseline_model_sha256": sha256_file(
            baseline_root / "signals_conv1dgap_512.npy"
        ),
        "baseline_member_prefix": True,
        "extension_event_count": len(extension_rows),
        "extension_class_counts": EXTENSION_COUNTS,
        "target_class_counts": TARGET_COUNTS,
        "extension_parameter_seed": args.extension_seed,
        "extension_signal_noise_seed": args.extension_seed + 1,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    validation = validate_candidate(
        dataset_id=args.dataset_id,
        budgets=TARGET_COUNTS,
        records=rows,
        raw_signals=raw,
        model_signals=model,
        factors=factors,
    )
    validation["checks"]["v2_baseline_raw_bit_identical"] = bool(
        np.array_equal(raw[: len(baseline_rows)], baseline_raw)
    )
    validation["checks"]["v2_baseline_model_bit_identical"] = bool(
        np.array_equal(model[: len(baseline_rows)], baseline_model)
    )
    validation["checks"]["v2_baseline_ids_identical"] = [
        str(row["sample_id"]) for row in rows[: len(baseline_rows)]
    ] == [str(row["sample_id"]) for row in baseline_rows]
    validation["checks"]["nested_counts_exact"] = (
        validation["class_counts"] == TARGET_COUNTS
    )
    if not all(validation["checks"].values()):
        raise ValueError(f"density10x validation failed: {validation['checks']}")

    correlation_rows = correlation_validation(rows, factors)
    analysis_outputs = write_analysis_outputs(
        output_dir=run_output,
        validation=validation,
        correlation_rows=correlation_rows,
    )
    candidate_manifest = {
        "schema_version": 1,
        "dataset_id": args.dataset_id,
        "status": "reference_candidate_awaiting_twin_result",
        "files": [
            {
                "path": name,
                "size": (dataset_output / name).stat().st_size,
                "sha256": sha256_file(dataset_output / name),
            }
            for name in outputs
        ],
    }
    (run_output / "candidate_dataset_manifest.json").write_text(
        json.dumps(candidate_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    analysis_outputs.append("candidate_dataset_manifest.json")

    repository = workspace.root / "particles2SNR-pipeline"
    script_path = (
        repository
        / "scripts/generation/generate_z8_density10x_nested_gallery.py"
    )
    module_path = repository / "particles2snr/z8_density_gallery.py"
    provenance = {
        "datasets": {
            args.source_dataset: source_record["manifest_sha256"],
            args.baseline_dataset: baseline_record["manifest_sha256"],
        },
        "method_evidence": method_evidence,
        "inputs": {
            "baseline_summary_sha256": sha256_file(
                baseline_root / "dataset_summary.json"
            ),
            "gaussian_parameters_sha256": sha256_file(gaussian_parameters),
            "cholesky_factors_sha256": sha256_file(cholesky_factors),
            "cholesky_recommendations_sha256": sha256_file(
                cholesky_recommendations
            ),
        },
        "parameters": {
            "baseline_counts": BASE_COUNTS,
            "extension_counts": EXTENSION_COUNTS,
            "target_counts": TARGET_COUNTS,
            "extension_parameter_seed": args.extension_seed,
            "extension_signal_noise_seed": args.extension_seed + 1,
            "frequency_acceptance_khz": [7.0, 80.0],
            "raw_length": 4096,
            "conv1dgap_length": 512,
            "sealed_test_accessed": False,
        },
        "metric_definitions": {
            "nested_baseline_identity": (
                "exact equality of the first 4,798 raw/model rows and sample IDs "
                "against the immutable synthetic v2 dataset"
            ),
            "realized_correlation": (
                "Pearson correlation of generated [log(P0), frequency_khz, "
                "log(tau_ms), SNR_dB] coordinates"
            ),
            "correlation_delta": (
                "realized correlation minus the approved class-specific Z8 v2 target"
            ),
            "snr_error_db": (
                "achieved signal SNR minus requested generation-parameter SNR"
            ),
        },
        "code": {
            script_path.relative_to(repository).as_posix(): sha256_file(script_path),
            module_path.relative_to(repository).as_posix(): sha256_file(module_path),
        },
        "git_revision": {
            "workspace": _git_state(workspace.root),
            "particles2SNR-pipeline": _git_state(repository),
        },
    }
    fingerprint = computation_fingerprint(provenance)
    metrics_manifest = {
        "schema_version": 1,
        "analysis_run_id": args.run_id,
        "computation_provenance": provenance,
        "computation_fingerprint": fingerprint,
        "metrics": [
            {
                "path": name,
                "sha256": sha256_file(run_output / name),
                "computation_fingerprint": fingerprint,
            }
            for name in ("summary_metrics.json", "correlation_validation.csv")
        ],
    }
    (run_output / "metrics_manifest.json").write_text(
        json.dumps(metrics_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    dataset_relative = dataset_output.relative_to(workspace.root).as_posix()
    run = {
        "schema_version": 1,
        "project": "particles2SNR-pipeline",
        "run_id": args.run_id,
        "dataset": args.source_dataset,
        "datasets": {
            args.source_dataset: {
                "id": args.source_dataset,
                "manifest_sha256": source_record["manifest_sha256"],
            },
            args.baseline_dataset: {
                "id": args.baseline_dataset,
                "manifest_sha256": baseline_record["manifest_sha256"],
            },
        },
        "candidate_dataset": {
            "id": args.dataset_id,
            "path": dataset_relative,
            "status": "reference_candidate_awaiting_twin_result",
            "registered": False,
        },
        "command": " ".join(
            [
                "particles2SNR-pipeline/scripts/generation/"
                "generate_z8_density10x_nested_gallery.py",
                "--source-dataset",
                args.source_dataset,
                "--baseline-dataset",
                args.baseline_dataset,
                "--dataset-id",
                args.dataset_id,
                "--run-id",
                args.run_id,
                "--gaussian-run-id",
                args.gaussian_run_id,
                "--cholesky-run-id",
                args.cholesky_run_id,
                "--method-evidence-id",
                args.method_evidence_id,
                "--method-evidence-run-id",
                args.method_evidence_run_id,
                "--dataset-output-dir",
                dataset_relative,
                "--run-output-dir",
                run_output.relative_to(workspace.root).as_posix(),
                "--extension-seed",
                str(args.extension_seed),
                "--batch-size",
                str(args.batch_size),
            ]
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repositories": provenance["git_revision"],
        "status": "complete_candidate_awaiting_reference_registration",
        "method_evidence_id": args.method_evidence_id,
        "method_evidence_run_id": args.method_evidence_run_id,
        "computation_fingerprint": fingerprint,
        "sealed_test_accessed": False,
        "outputs": analysis_outputs + ["metrics_manifest.json"],
        "candidate_dataset_outputs": outputs,
        "claim_boundary": (
            "The candidate proves deterministic nested 10× generation and "
            "statistical validity. It is not eligible for active publication "
            "until all three human twin classes reach 9/10."
        ),
    }
    (run_output / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "candidate_dataset": args.dataset_id,
                "event_count": validation["event_count"],
                "class_counts": validation["class_counts"],
                "dataset_output_dir": dataset_relative,
                "computation_fingerprint": fingerprint,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
