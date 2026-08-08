#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from internship_workspace.config import Workspace
from internship_workspace.datasets import load_records
from internship_workspace.scientific_visual import computation_fingerprint
from particles2snr.z8_parameter_analysis import (
    CLASS_ORDER,
    FBASE_NOMINAL_BAND_KHZ,
    MARGIN_FRACTIONS,
    PARAMETERS,
    build_analysis,
    resolve_registered_z8_dataset,
    read_events,
    validate_method_evidence,
    write_analysis_outputs,
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
        description=(
            "Analyze empirical parameter distributions in the post-processed "
            "particles2SNR z8 development dataset."
        )
    )
    parser.add_argument("--dataset", required=True, help="Registered z8 development dataset ID@version")
    parser.add_argument("--source-dataset", required=True, help="Registered source-signal dataset ID@version")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method-evidence-id", required=True)
    parser.add_argument("--method-evidence-run-id", required=True)
    parser.add_argument("--evidence-contract", type=Path, required=True)
    args = parser.parse_args()

    workspace = Workspace.load()
    args.output_dir = args.output_dir.resolve()
    if not args.output_dir.is_relative_to(workspace.artifacts_root):
        raise ValueError("--output-dir must be below the workspace artifacts root")
    z8_record, z8_root, summary = resolve_registered_z8_dataset(workspace, args.dataset)
    source_record, source_root, _ = resolve_registered_z8_dataset(workspace, args.source_dataset, require_z8_summary=False)
    events_path = z8_root / "events.csv"
    rows = read_events(events_path, dataset_summary=summary)
    sources = summary.get("source_datasets", {})
    source_binding = sources.get(args.source_dataset)
    source_hash = (
        source_binding.get("manifest_sha256")
        if isinstance(source_binding, dict)
        else source_binding
    )
    if source_hash != source_record["manifest_sha256"]:
        raise ValueError("Selected source dataset does not match the z8 summary source_datasets")
    for row in rows:
        relative = Path(row["source_signal_relative_path"])
        target = (source_root / relative).resolve()
        if relative.is_absolute() or "." in relative.parts or ".." in relative.parts or not target.is_relative_to(source_root.resolve()) or not target.is_file():
            raise ValueError("source_signal_relative_path escapes selected source dataset")
    source_signal_lengths: dict[str, int] = {}
    for relative_path in sorted({row["source_signal_relative_path"] for row in rows}):
        signal = np.load(source_root / relative_path, mmap_mode="r", allow_pickle=False)
        if signal.ndim != 1 or signal.shape[0] <= 0:
            raise ValueError(f"Invalid source waveform: {relative_path}")
        source_signal_lengths[relative_path] = int(signal.shape[0])
    evidence = validate_method_evidence(workspace, evidence_id=args.method_evidence_id, evidence_run_id=args.method_evidence_run_id, contract_path=args.evidence_contract, dataset_id=args.dataset, method="parameter_distributions")
    analysis = build_analysis(
        rows,
        dataset_id=args.dataset,
        source_signal_lengths=source_signal_lengths,
    )
    outputs = write_analysis_outputs(
        analysis=analysis,
        rows=rows,
        source_root=source_root,
        output_dir=args.output_dir,
    )

    repository = workspace.root / "particles2SNR-pipeline"
    module_path = repository / "particles2snr/z8_parameter_analysis.py"
    script_path = repository / "scripts/analysis/analyze_z8_parameter_distributions.py"
    git_states = {
        "workspace": _git_state(workspace.root),
        "particles2SNR-pipeline": _git_state(repository),
    }
    provenance = {
        "datasets": {
            args.dataset: z8_record["manifest_sha256"],
            args.source_dataset: source_record["manifest_sha256"],
        },
        "inputs": {
            "z8_events_csv_sha256": sha256_file(events_path),
            "z8_events_csv_path": events_path.relative_to(workspace.root).as_posix(),
            "source_signal_dataset_manifest_sha256": source_record["manifest_sha256"],
            "source_dataset_path": source_root.relative_to(workspace.root).as_posix(),
        },
        "evidence": evidence,
        "parameters": {
            "classes": list(CLASS_ORDER),
            "parameter_contract": PARAMETERS,
            "quantiles": [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99],
            "margins": MARGIN_FRACTIONS,
            "nominal_fbase_band_khz": list(FBASE_NOMINAL_BAND_KHZ),
            "unclear_policy": "SNR only, mapped to physical_source_class",
            "boundary_censoring": (
                "statistical_only: start_sample<=0 or "
                "end_sample>=source_signal_length"
            ),
            "dataset_membership_changed": False,
        },
        "metric_definitions": {
            "descriptive_statistics": "sample statistics over observed events",
            "distribution": "empirical observed distribution with histogram and KDE display",
            "correlation": "Pearson correlation on physical-class rows only",
            "extremes": (
                "deduplicated min/max over boundary-eligible events; tied roles "
                "use lexicographically smallest event_id for the gallery"
            ),
            "supports": "M0/M10/M20 based on observed min-max width",
        },
        "code": {
            "evidence_contract": evidence["contract_sha256"],
            "evidence_receipt": evidence["receipt_sha256"],
            "particles2snr/z8_parameter_analysis.py": sha256_file(module_path),
            "scripts/analysis/analyze_z8_parameter_distributions.py": sha256_file(script_path),
        },
        "git_revision": git_states,
    }
    fingerprint = computation_fingerprint(provenance)
    metric_names = [
        "summary_metrics.json",
        "parameter_statistics.csv",
        "support_candidates.csv",
        "extremes.csv",
        "boundary_censored_events.csv",
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
                "manifest_sha256": z8_record["manifest_sha256"],
                "path": z8_root.relative_to(workspace.root).as_posix(),
            },
            args.source_dataset: {
                "id": args.source_dataset,
                "manifest_sha256": source_record["manifest_sha256"],
                "path": source_root.relative_to(workspace.root).as_posix(),
            },
        },
        "command": " ".join(str(part) for part in [script_path.relative_to(workspace.root), "--dataset", args.dataset, "--source-dataset", args.source_dataset, "--run-id", args.run_id, "--output-dir", args.output_dir.relative_to(workspace.root), "--method-evidence-id", args.method_evidence_id, "--method-evidence-run-id", args.method_evidence_run_id, "--evidence-contract", evidence["contract_path"]]),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "repositories": git_states,
        "outputs": outputs + ["metrics_manifest.json"],
        "method_evidence_id": args.method_evidence_id,
        "method_evidence_run_id": args.method_evidence_run_id,
        "computation_fingerprint": fingerprint,
        "claim_boundary": analysis["claim_boundary"],
    }
    (args.output_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"run_id": args.run_id, "output_dir": str(args.output_dir), "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
