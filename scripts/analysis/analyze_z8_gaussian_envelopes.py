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
from particles2snr.z8_gaussian_envelope_analysis import (
    COUNT_SAFETY_FACTOR,
    FIT_GRID_SIZE,
    VALIDATION_GRID_SIZE,
    analyze_gaussian_intensity_envelopes,
    write_gaussian_envelope_outputs,
)
from particles2snr.z8_parameter_analysis import (
    load_approved_estimation_population,
    read_events,
    resolve_registered_z8_dataset,
    validate_method_evidence,
)
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize Gaussian event-intensity envelopes for SSL v3."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method-evidence-id", required=True)
    parser.add_argument("--method-evidence-run-id", required=True)
    parser.add_argument("--evidence-contract", type=Path, required=True)
    parser.add_argument("--population-run-id", required=True)
    parser.add_argument("--population-evidence-id", required=True)
    parser.add_argument("--population-evidence-run-id", required=True)
    args = parser.parse_args()

    workspace = Workspace.load()
    args.output_dir = args.output_dir.resolve()
    if not args.output_dir.is_relative_to(workspace.artifacts_root):
        raise ValueError("--output-dir must be below the workspace artifacts root")
    dataset_record, dataset_root, summary = resolve_registered_z8_dataset(workspace, args.dataset)
    events_path = dataset_root / "events.csv"
    rows = read_events(events_path, dataset_summary=summary)
    evidence = validate_method_evidence(workspace, evidence_id=args.method_evidence_id, evidence_run_id=args.method_evidence_run_id, contract_path=args.evidence_contract, dataset_id=args.dataset, method="gaussian_envelopes")
    rows, population = load_approved_estimation_population(
        workspace,
        rows,
        dataset_id=args.dataset,
        analysis_run_id=args.population_run_id,
        evidence_id=args.population_evidence_id,
        evidence_run_id=args.population_evidence_run_id,
    )
    analysis = analyze_gaussian_intensity_envelopes(rows)
    outputs = write_gaussian_envelope_outputs(
        rows=rows, analysis=analysis, output_dir=args.output_dir
    )

    repository = workspace.root / "particles2SNR-pipeline"
    module_path = repository / "particles2snr/z8_gaussian_envelope_analysis.py"
    script_path = repository / "scripts/analysis/analyze_z8_gaussian_envelopes.py"
    git_states = {
        "workspace": _git_state(workspace.root),
        "particles2SNR-pipeline": _git_state(repository),
    }
    provenance = {
        "datasets": {args.dataset: dataset_record["manifest_sha256"]},
        "inputs": {
            "z8_events_csv_sha256": sha256_file(events_path),
            "z8_events_csv_path": events_path.relative_to(workspace.root).as_posix(),
            "approved_estimation_population": population,
        },
        "evidence": evidence,
        "parameters": {
            "cholesky_space": ["log(P0)", "frequency_khz", "log(tau_ms)", "SNR_dB"],
            "mean_policy": "fixed at deterministic KDE mode",
            "sigma_optimization": (
                "bounded scalar minimization of the maximum log KDE-to-Gaussian ratio"
            ),
            "fit_grid_size": FIT_GRID_SIZE,
            "validation_grid_size": VALIDATION_GRID_SIZE,
            "count_safety_factor": COUNT_SAFETY_FACTOR,
            "class_budget": "maximum required event count across four marginals",
        },
        "metric_definitions": {
            "envelope": "N_synth * Gaussian PDF >= N_real * real KDE",
            "violation_count": (
                "number of validation-grid points where real KDE exceeds synthetic "
                "event-intensity envelope"
            ),
        },
        "code": {
            "particles2snr/z8_gaussian_marginal_analysis.py": sha256_file(repository / "particles2snr/z8_gaussian_marginal_analysis.py"),
            "evidence_contract": evidence["contract_sha256"],
            "evidence_receipt": evidence["receipt_sha256"],
            "particles2snr/z8_parameter_analysis.py": sha256_file(repository / "particles2snr/z8_parameter_analysis.py"),
            "particles2snr/z8_gaussian_envelope_analysis.py": sha256_file(module_path),
            "scripts/analysis/analyze_z8_gaussian_envelopes.py": sha256_file(
                script_path
            ),
        },
        "git_revision": git_states,
    }
    fingerprint = computation_fingerprint(provenance)
    metric_names = [
        "summary_metrics.json",
        "gaussian_envelope_parameters.csv",
        "class_synthetic_budgets.csv",
    ]
    metrics_manifest = {
        "schema_version": 1,
        "analysis_run_id": args.run_id,
        "computation_provenance": provenance,
        "computation_fingerprint": fingerprint,
        "metrics": [
            {"path": name, "sha256": sha256_file(args.output_dir / name)}
            for name in metric_names
        ],
    }
    (args.output_dir / "metrics_manifest.json").write_text(
        json.dumps(metrics_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    run = {
        "schema_version": 1,
        "project": "particles2SNR-pipeline",
        "run_id": args.run_id,
        "dataset": args.dataset,
        "datasets": {
            args.dataset: {
                "id": args.dataset,
                "manifest_sha256": dataset_record["manifest_sha256"],
                "path": dataset_root.relative_to(workspace.root).as_posix(),
            }
        },
        "command": " ".join(str(part) for part in [script_path.relative_to(workspace.root), "--dataset", args.dataset, "--run-id", args.run_id, "--output-dir", args.output_dir.relative_to(workspace.root), "--method-evidence-id", args.method_evidence_id, "--method-evidence-run-id", args.method_evidence_run_id, "--evidence-contract", evidence["contract_path"], "--population-run-id", args.population_run_id, "--population-evidence-id", args.population_evidence_id, "--population-evidence-run-id", args.population_evidence_run_id]),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "repositories": git_states,
        "outputs": outputs + ["metrics_manifest.json"],
        "method_evidence_id": args.method_evidence_id,
        "method_evidence_run_id": args.method_evidence_run_id,
        "population_evidence_id": args.population_evidence_id,
        "population_evidence_run_id": args.population_evidence_run_id,
        "population_run_id": args.population_run_id,
        "computation_fingerprint": fingerprint,
        "claim_boundary": analysis["claim_boundary"],
    }
    (args.output_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "output_dir": str(args.output_dir),
                "class_budgets": analysis["class_budgets"],
                "total_violations": sum(
                    row["class_envelope_violation_count"] for row in analysis["fits"]
                ),
                "outputs": outputs,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
