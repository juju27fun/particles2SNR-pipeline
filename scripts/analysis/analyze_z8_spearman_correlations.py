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
from particles2snr.z8_parameter_analysis import (
    load_approved_estimation_population,
    read_events,
    resolve_registered_z8_dataset,
    validate_method_evidence,
)
from particles2snr.z8_reference_dataset import sha256_file
from particles2snr.z8_spearman_analysis import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CI_PERCENTILES,
    CLASS_ORDER,
    PAIR_ORDER,
    PARAMETER_ORDER,
    analyze_spearman,
    write_spearman_outputs,
)




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
        description=(
            "Run the complete within-class Spearman analysis for the post-processed "
            "particles2SNR z8 development dataset."
        )
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
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    args = parser.parse_args()

    workspace = Workspace.load()
    args.output_dir = args.output_dir.resolve()
    if not args.output_dir.is_relative_to(workspace.artifacts_root):
        raise ValueError("--output-dir must be below the workspace artifacts root")
    dataset_record, dataset_root, summary = resolve_registered_z8_dataset(workspace, args.dataset)
    events_path = dataset_root / "events.csv"
    rows = read_events(events_path, dataset_summary=summary)
    evidence = validate_method_evidence(workspace, evidence_id=args.method_evidence_id, evidence_run_id=args.method_evidence_run_id, contract_path=args.evidence_contract, dataset_id=args.dataset, method="spearman_correlations")
    rows, population = load_approved_estimation_population(
        workspace,
        rows,
        dataset_id=args.dataset,
        analysis_run_id=args.population_run_id,
        evidence_id=args.population_evidence_id,
        evidence_run_id=args.population_evidence_run_id,
    )
    analysis = analyze_spearman(
        rows,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=BOOTSTRAP_SEED,
    )
    outputs = write_spearman_outputs(
        analysis=analysis,
        rows=rows,
        output_dir=args.output_dir,
    )

    repository = workspace.root / "particles2SNR-pipeline"
    module_path = repository / "particles2snr/z8_spearman_analysis.py"
    script_path = repository / "scripts/analysis/analyze_z8_spearman_correlations.py"
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
            "physical_classes": list(CLASS_ORDER),
            "parameters": list(PARAMETER_ORDER),
            "pairs": [list(pair) for pair in PAIR_ORDER],
            "primary_population": "physical class rows from train and validation",
            "unclear_policy": (
                "sensitivity only for SNR-related pairs after physical_source_class mapping"
            ),
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_unit": "source_filename",
            "ci_percentiles": list(CI_PERCENTILES),
            "selection_rule": (
                "rank by median absolute primary Spearman rho across classes and retain top 3"
            ),
        },
        "metric_definitions": {
            "marginal_spearman": "Pearson correlation between average-rank vectors",
            "partial_spearman": (
                "partial correlation from the inverse rank-correlation matrix, controlling "
                "the remaining two parameters"
            ),
            "confidence_interval": (
                "2.5th and 97.5th percentiles of source_filename grouped bootstrap replicates"
            ),
            "overall_score": "median absolute marginal Spearman rho across 2um, 4um, and 10um",
        },
        "code": {
            "evidence_contract": evidence["contract_sha256"],
            "evidence_receipt": evidence["receipt_sha256"],
            "particles2snr/z8_parameter_analysis.py": sha256_file(repository / "particles2snr/z8_parameter_analysis.py"),
            "particles2snr/z8_spearman_analysis.py": sha256_file(module_path),
            "scripts/analysis/analyze_z8_spearman_correlations.py": sha256_file(script_path),
        },
        "git_revision": git_states,
    }
    fingerprint = computation_fingerprint(provenance)
    metric_names = [
        "summary_metrics.json",
        "spearman_correlations.csv",
        "partial_spearman_correlations.csv",
        "unclear_sensitivity.csv",
        "correlation_ranking.csv",
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
        "command": " ".join(str(part) for part in [script_path.relative_to(workspace.root), "--dataset", args.dataset, "--run-id", args.run_id, "--output-dir", args.output_dir.relative_to(workspace.root), "--method-evidence-id", args.method_evidence_id, "--method-evidence-run-id", args.method_evidence_run_id, "--evidence-contract", evidence["contract_path"], "--population-run-id", args.population_run_id, "--population-evidence-id", args.population_evidence_id, "--population-evidence-run-id", args.population_evidence_run_id, "--bootstrap-replicates", args.bootstrap_replicates]),
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
                "selected_pairs": [row["pair_label"] for row in analysis["selected_pairs"]],
                "outputs": outputs,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
