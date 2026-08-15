#!/usr/bin/env python3
"""Audit every validation join in the Z8 v2 Wave8-like candidate."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from internship_workspace.config import Workspace
from internship_workspace.scientific_visual import computation_fingerprint
from particles2snr.z8_wave8like_dataset import sha256_file
from particles2snr.z8_wave8like_join_audit import (
    AuditConfig,
    write_analysis,
)


OUTPUT_DATASET = "particles2snr-z8-v2-wave8like-known3-background-development"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repair-missing-contract",
        action="store_true",
        help=(
            "Repair an analysis produced without metrics_manifest.json after "
            "proving its metric payloads match a fresh recomputation."
        ),
    )
    parser.add_argument(
        "--candidate-version",
        choices=("v1", "v2", "v3", "v4"),
        default="v1",
    )
    args = parser.parse_args()
    workspace = Workspace.load()
    candidate_id = f"{OUTPUT_DATASET}@{args.candidate_version}"
    candidate_path = (
        Path("datasets/interim/particles2SNR-pipeline")
        / OUTPUT_DATASET
        / args.candidate_version
    )
    generation_run_id = (
        f"particle-z8-v2-wave8like-generation-{args.candidate_version}"
    )
    generation_run_path = (
        workspace.root
        / f"artifacts/particles2SNR-pipeline/runs/{generation_run_id}/run.json"
    )
    run_id = f"particle-z8-v2-wave8like-join-audit-{args.candidate_version}"
    output_path = Path(f"artifacts/particles2SNR-pipeline/audits/{run_id}")
    candidate_root = workspace.root / candidate_path
    output_root = workspace.root / output_path
    generation = json.loads(generation_run_path.read_text(encoding="utf-8"))
    if generation.get("status") != "complete_candidate_awaiting_visual_join_audit":
        raise RuntimeError("completed immutable candidate generation is required")
    module_path = (
        workspace.root
        / "particles2SNR-pipeline/particles2snr/z8_wave8like_join_audit.py"
    )
    script_path = Path(__file__).resolve()
    git_states = {
        "workspace": subprocess.run(
            ["git", "-C", str(workspace.root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "particles2SNR-pipeline": subprocess.run(
            [
                "git",
                "-C",
                str(workspace.root / "particles2SNR-pipeline"),
                "rev-parse",
                "HEAD",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
    }
    provenance = {
        "datasets": {
            candidate_id: sha256_file(candidate_root / "dataset-manifest.json"),
        },
        "inputs": {
            "candidate_manifest_csv_sha256": sha256_file(
                candidate_root / "manifest.csv"
            ),
            "generation_run_sha256": sha256_file(generation_run_path),
        },
        "parameters": {
            "segment_length": 16_384,
            "guard_samples": 300,
            "sampling_frequency_hz": 2_000_000,
            "split": "val",
            "sealed_test_accessed": False,
        },
        "metric_definitions": {
            "boundary_jump": (
                "absolute sample-to-sample jump at each source boundary"
            ),
            "boundary_jump_robust_z": (
                "boundary jump centered and scaled by non-guard absolute "
                "differences using median and 1.4826 times MAD"
            ),
            "join_to_control_rms_ratio": (
                "RMS in the 600-sample join divided by mean RMS of its two "
                "adjacent 600-sample control windows"
            ),
            "join_peak_robust_z": (
                "largest absolute join amplitude relative to the non-guard "
                "median and 1.4826 times MAD"
            ),
        },
        "code": {
            "particles2snr/z8_wave8like_join_audit.py": sha256_file(module_path),
            (
                "scripts/analysis/audit_z8_v2_wave8like_joins.py"
            ): sha256_file(script_path),
        },
        "git_revision": git_states,
    }
    fingerprint = computation_fingerprint(provenance)
    command = Path(__file__).resolve().relative_to(workspace.root).as_posix()
    run_payload = {
        "schema_version": 1,
        "run_id": run_id,
        "project": "particles2SNR-pipeline",
        "kind": "dataset-join-audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": candidate_id,
        "command": f"{command} --candidate-version {args.candidate_version}",
        "source_run_ids": [generation_run_id],
        "candidate_path": candidate_path.as_posix(),
        "candidate_registered": False,
        "sealed_test_accessed": False,
        "repositories": git_states,
    }
    target_root = output_root
    repairing = args.repair_missing_contract
    if repairing:
        if not output_root.is_dir():
            raise FileNotFoundError(f"analysis to repair is missing: {output_root}")
        if (output_root / "metrics_manifest.json").exists():
            raise FileExistsError("analysis already has a metrics manifest")
        target_root = output_root.parent / f".{output_root.name}.repairing"
        if target_root.exists():
            stale_run = json.loads(
                (target_root / "run.json").read_text(encoding="utf-8")
            )
            if stale_run.get("run_id") != run_id:
                raise RuntimeError(
                    f"refusing to remove unrelated repair staging: {target_root}"
                )
            shutil.rmtree(target_root)
    result = write_analysis(
        candidate_root=candidate_root,
        output_root=target_root,
        config=AuditConfig(),
        computation_provenance=provenance,
        computation_fingerprint=fingerprint,
        run_payload=run_payload,
    )
    if repairing:
        metric_names = ("join_metrics.csv", "selected_cases.json")
        mismatches = [
            name
            for name in metric_names
            if sha256_file(output_root / name) != sha256_file(target_root / name)
        ]
        old_summary = json.loads(
            (output_root / "summary_metrics.json").read_text(encoding="utf-8")
        )
        new_summary = json.loads(
            (target_root / "summary_metrics.json").read_text(encoding="utf-8")
        )
        old_summary.pop("computation_fingerprint", None)
        new_summary.pop("computation_fingerprint", None)
        if old_summary != new_summary:
            mismatches.append("summary_metrics.json")
        if mismatches:
            raise RuntimeError(
                "refusing contract repair after metric mismatch: "
                + ", ".join(mismatches)
            )
        os.replace(
            target_root / "summary_metrics.json",
            output_root / "summary_metrics.json",
        )
        os.replace(
            target_root / "metrics_manifest.json",
            output_root / "metrics_manifest.json",
        )
        os.replace(target_root / "run.json", output_root / "run.json")
        shutil.rmtree(target_root)
    print(
        json.dumps(
            {
                "run_id": run_id,
                "output": output_path.as_posix(),
                "summary": result["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
