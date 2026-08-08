#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from internship_workspace.config import Workspace
from internship_workspace.datasets import load_records
from internship_workspace.scientific_visual import computation_fingerprint
from particles2snr.z8_cholesky_generation import (
    SEED,
    correlation_validation,
    generate_parameters,
    load_gaussian_targets,
    load_recommended_cholesky,
    read_csv,
    render_cholesky_pipeline,
    render_signal_gallery,
    render_statistical_validation,
    synthesize_signals,
    validate_candidate,
    write_analysis_outputs,
    write_candidate_dataset,
)
from particles2snr.z8_parameter_analysis import load_approved_estimation_population
from particles2snr.z8_reference_dataset import sha256_file


def _record(workspace: Workspace, key: str) -> dict[str, Any]:
    records = [record.payload for record in load_records(workspace)]
    match = next((row for row in records if f"{row['id']}@{row['version']}" == key), None)
    if match is None or match["status"] not in {"active", "reference"}:
        raise ValueError(f"Eligible registered dataset not found: {key}")
    return match


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


def _load_fingerprint(run_dir: Path) -> str:
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    fingerprint = run.get("computation_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError(f"Missing computation fingerprint: {run_dir}")
    return fingerprint


def _validate_result_evidence(
    workspace: Workspace,
    *,
    evidence_id: str,
    evidence_run_id: str,
    analysis_run_id: str,
    analysis_fingerprint: str,
) -> dict[str, Any]:
    review_dir = (
        workspace.root
        / "artifacts/cross-project/reviews"
        / evidence_run_id
    )
    review_run = json.loads((review_dir / "run.json").read_text(encoding="utf-8"))
    if review_run.get("evidence_id") != evidence_id:
        raise ValueError(f"Evidence ID mismatch in {review_dir}")
    if review_run.get("run_id") != evidence_run_id:
        raise ValueError(f"Evidence run ID mismatch in {review_dir}")
    approval = review_run.get("visual_approval", {})
    checkpoint = review_run.get("visual_checkpoint", {})
    if approval.get("decision") != "supported" or not checkpoint.get("approved"):
        raise ValueError(f"Scientific result evidence is not approved: {evidence_run_id}")
    source_run_ids = review_run.get("visual_protocol", {}).get("source_run_ids", [])
    if analysis_run_id not in source_run_ids:
        raise ValueError(
            f"Evidence {evidence_run_id} does not support run {analysis_run_id}"
        )
    reference = review_run.get("analysis_reference", {})
    if reference.get("run_id") != analysis_run_id:
        raise ValueError(
            f"Evidence analysis reference mismatch for {analysis_run_id}"
        )
    if reference.get("computation_fingerprint") != analysis_fingerprint:
        raise ValueError(
            f"Evidence fingerprint mismatch for {analysis_run_id}"
        )
    return {
        "evidence_id": evidence_id,
        "evidence_run_id": evidence_run_id,
        "analysis_run_id": analysis_run_id,
        "computation_fingerprint": analysis_fingerprint,
        "decision": approval["decision"],
        "reviewed_at": approval.get("reviewed_at"),
        "review_run_sha256": sha256_file(review_dir / "run.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an evidence-bound SSL v3 z8 Cholesky candidate."
    )
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gaussian-run-id", required=True)
    parser.add_argument("--cholesky-run-id", required=True)
    parser.add_argument("--gaussian-result-evidence-id", required=True)
    parser.add_argument("--gaussian-result-evidence-run-id", required=True)
    parser.add_argument("--cholesky-result-evidence-id", required=True)
    parser.add_argument("--cholesky-result-evidence-run-id", required=True)
    parser.add_argument("--population-run-id", required=True)
    parser.add_argument("--population-evidence-id", required=True)
    parser.add_argument("--population-evidence-run-id", required=True)
    parser.add_argument("--dataset-output-dir", type=Path, required=True)
    parser.add_argument("--run-output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    workspace = Workspace.load()
    args.dataset_output_dir = args.dataset_output_dir.resolve()
    args.run_output_dir = args.run_output_dir.resolve()
    if not args.dataset_output_dir.is_relative_to(workspace.datasets_root / "interim"):
        raise ValueError("--dataset-output-dir must be below datasets/interim")
    if not args.run_output_dir.is_relative_to(workspace.artifacts_root):
        raise ValueError("--run-output-dir must be below artifacts")
    source_record = _record(workspace, args.source_dataset)
    source_root = workspace.datasets_root / source_record["path"]
    events_path = source_root / "events.csv"
    real_rows = read_csv(events_path)
    real_rows, population = load_approved_estimation_population(
        workspace,
        real_rows,
        dataset_id=args.source_dataset,
        analysis_run_id=args.population_run_id,
        evidence_id=args.population_evidence_id,
        evidence_run_id=args.population_evidence_run_id,
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
    targets, budgets = load_gaussian_targets(
        gaussian_parameters,
        include_budgets=True,
    )
    factors, dependency_populations = load_recommended_cholesky(
        cholesky_factors,
        cholesky_recommendations,
    )

    gaussian_fingerprint = _load_fingerprint(gaussian_run)
    cholesky_fingerprint = _load_fingerprint(cholesky_run)
    gaussian_evidence = _validate_result_evidence(
        workspace,
        evidence_id=args.gaussian_result_evidence_id,
        evidence_run_id=args.gaussian_result_evidence_run_id,
        analysis_run_id=args.gaussian_run_id,
        analysis_fingerprint=gaussian_fingerprint,
    )
    cholesky_evidence = _validate_result_evidence(
        workspace,
        evidence_id=args.cholesky_result_evidence_id,
        evidence_run_id=args.cholesky_result_evidence_run_id,
        analysis_run_id=args.cholesky_run_id,
        analysis_fingerprint=cholesky_fingerprint,
    )

    records, rejection_counts = generate_parameters(
        targets,
        factors,
        seed=args.seed,
        budgets=budgets,
        dataset_id=args.dataset_id,
    )
    raw_signals, model_signals = synthesize_signals(
        records, seed=args.seed + 1, batch_size=args.batch_size
    )
    dataset_outputs = write_candidate_dataset(
        dataset_id=args.dataset_id,
        source_dataset_id=args.source_dataset,
        gaussian_run_id=args.gaussian_run_id,
        cholesky_run_id=args.cholesky_run_id,
        dependency_populations=dependency_populations,
        seed=args.seed,
        output_dir=args.dataset_output_dir,
        records=records,
        raw_signals=raw_signals,
        model_signals=model_signals,
        rejection_counts=rejection_counts,
        source_manifest_sha256=source_record["manifest_sha256"],
        gaussian_run_fingerprint=gaussian_fingerprint,
        cholesky_run_fingerprint=cholesky_fingerprint,
    )

    validation = validate_candidate(
        dataset_id=args.dataset_id,
        budgets=budgets,
        records=records,
        raw_signals=raw_signals,
        model_signals=model_signals,
        factors=factors,
    )
    correlation_rows = correlation_validation(records, factors)
    analysis_outputs = write_analysis_outputs(
        output_dir=args.run_output_dir,
        validation=validation,
        correlation_rows=correlation_rows,
    )
    candidate_manifest = {
        "schema_version": 1,
        "dataset_id": args.dataset_id,
        "status": "interim_candidate_awaiting_visual_review",
        "files": [
            {
                "path": name,
                "size": (args.dataset_output_dir / name).stat().st_size,
                "sha256": sha256_file(args.dataset_output_dir / name),
            }
            for name in dataset_outputs
        ],
    }
    (args.run_output_dir / "candidate_dataset_manifest.json").write_text(
        json.dumps(candidate_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    analysis_outputs.append("candidate_dataset_manifest.json")
    render_cholesky_pipeline(
        records,
        targets,
        dependency_populations,
        args.run_output_dir / "actual_cholesky_generation_path.png",
    )
    render_statistical_validation(
        real_rows=real_rows,
        generated_records=records,
        targets=targets,
        factors=factors,
        dependency_populations=dependency_populations,
        destination=args.run_output_dir / "realized_marginals_and_correlations.png",
    )
    render_signal_gallery(
        records=records,
        raw_signals=raw_signals,
        destination=args.run_output_dir / "representative_signal_gallery.png",
    )
    figure_outputs = [
        "actual_cholesky_generation_path.png",
        "realized_marginals_and_correlations.png",
        "representative_signal_gallery.png",
    ]

    repository = workspace.root / "particles2SNR-pipeline"
    module_path = repository / "particles2snr/z8_cholesky_generation.py"
    script_path = repository / "scripts/generation/generate_z8_cholesky_synthetic_events.py"
    git_states = {
        "workspace": _git_state(workspace.root),
        "particles2SNR-pipeline": _git_state(repository),
    }
    provenance = {
        "datasets": {args.source_dataset: source_record["manifest_sha256"]},
        "inputs": {
            "z8_events_csv_sha256": sha256_file(events_path),
            "approved_estimation_population": population,
            "gaussian_parameters_sha256": sha256_file(gaussian_parameters),
            "cholesky_factors_sha256": sha256_file(cholesky_factors),
            "cholesky_recommendations_sha256": sha256_file(
                cholesky_recommendations
            ),
        },
        "source_runs": {
            args.gaussian_run_id: gaussian_fingerprint,
            args.cholesky_run_id: cholesky_fingerprint,
        },
        "source_result_evidence": [gaussian_evidence, cholesky_evidence],
        "parameters": {
            "seed": args.seed,
            "parameter_seed": args.seed,
            "signal_noise_seed": args.seed + 1,
            "class_budgets": budgets,
            "dependency_population_by_class": dependency_populations,
            "snr_marginal_population": "inclusive",
            "frequency_acceptance_khz": [7.0, 80.0],
            "raw_length": 4096,
            "conv1dgap_length": 512,
            "sampling_frequency_hz": 2_000_000.0,
            "phi_policy": "Uniform(0, 2*pi)",
            "t0_fraction": 0.5,
        },
        "metric_definitions": {
            "realized_marginal": (
                "empirical generated density in the same transformed coordinate as "
                "the validated Gaussian target"
            ),
            "realized_correlation": (
                "Pearson correlation of generated [log(P0), frequency_khz, "
                "log(tau_ms), SNR_dB]"
            ),
            "correlation_delta": (
                "realized correlation minus the approved class-specific z8 target"
            ),
            "snr_error_db": "achieved signal SNR minus requested parameter SNR",
        },
        "code": {
            "particles2snr/z8_cholesky_generation.py": sha256_file(module_path),
            "scripts/generation/generate_z8_cholesky_synthetic_events.py": sha256_file(
                script_path
            ),
        },
        "git_revision": git_states,
    }
    fingerprint = computation_fingerprint(provenance)
    metric_names = ["summary_metrics.json", "correlation_validation.csv"]
    metrics_manifest = {
        "schema_version": 1,
        "analysis_run_id": args.run_id,
        "computation_provenance": provenance,
        "computation_fingerprint": fingerprint,
        "metrics": [
            {"path": name, "sha256": sha256_file(args.run_output_dir / name)}
            for name in metric_names
        ],
    }
    (args.run_output_dir / "metrics_manifest.json").write_text(
        json.dumps(metrics_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    dataset_relative = args.dataset_output_dir.resolve().relative_to(
        workspace.root.resolve()
    ).as_posix()
    run = {
        "schema_version": 1,
        "project": "particles2SNR-pipeline",
        "run_id": args.run_id,
        "dataset": args.source_dataset,
        "datasets": {
            args.source_dataset: {
                "id": args.source_dataset,
                "manifest_sha256": source_record["manifest_sha256"],
            }
        },
        "candidate_dataset": {
            "id": args.dataset_id,
            "path": dataset_relative,
            "status": "interim_candidate_awaiting_visual_review",
            "registered": False,
        },
        "command": " ".join(
            [
                "particles2SNR-pipeline/scripts/generation/"
                "generate_z8_cholesky_synthetic_events.py",
                "--source-dataset",
                args.source_dataset,
                "--dataset-id",
                args.dataset_id,
                "--run-id",
                args.run_id,
                "--gaussian-run-id",
                args.gaussian_run_id,
                "--cholesky-run-id",
                args.cholesky_run_id,
                "--gaussian-result-evidence-id",
                args.gaussian_result_evidence_id,
                "--gaussian-result-evidence-run-id",
                args.gaussian_result_evidence_run_id,
                "--cholesky-result-evidence-id",
                args.cholesky_result_evidence_id,
                "--cholesky-result-evidence-run-id",
                args.cholesky_result_evidence_run_id,
                "--population-run-id",
                args.population_run_id,
                "--population-evidence-id",
                args.population_evidence_id,
                "--population-evidence-run-id",
                args.population_evidence_run_id,
                "--dataset-output-dir",
                dataset_relative,
                "--run-output-dir",
                args.run_output_dir.relative_to(workspace.root).as_posix(),
                "--seed",
                str(args.seed),
                "--batch-size",
                str(args.batch_size),
            ]
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_awaiting_visual_review",
        "repositories": git_states,
        "outputs": analysis_outputs + figure_outputs + ["metrics_manifest.json"],
        "candidate_dataset_outputs": dataset_outputs,
        "method_evidence_ids": [
            "particle-z8-cholesky-method",
            "particle-z8-gaussian-intensity-envelope-method",
        ],
        "source_result_evidence_ids": [
            args.cholesky_result_evidence_id,
            args.gaussian_result_evidence_id,
        ],
        "computation_fingerprint": fingerprint,
        "claim_boundary": validation["claim_boundary"],
    }
    (args.run_output_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "candidate_dataset": args.dataset_id,
                "event_count": validation["event_count"],
                "class_counts": validation["class_counts"],
                "correlation_warning": validation["correlation_warning"],
                "maximum_absolute_off_diagonal_correlation_delta": validation[
                    "maximum_absolute_off_diagonal_correlation_delta"
                ],
                "dataset_output_dir": str(args.dataset_output_dir),
                "run_output_dir": str(args.run_output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
