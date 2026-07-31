from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from particles2snr.particle_candidate_dataset import (
    assign_population_roles,
    build_candidate_dataset,
    validate_holdout_authorization,
)
from particles2snr.particle_events import ParticleDetectionConfig, config_fingerprint


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts/generation/build_particle_mad_candidate_dataset.py"


def _rows() -> list[dict[str, str]]:
    rows = []
    for class_name in ("2um", "4um", "10um"):
        for index in range(10):
            rows.append(
                {
                    "split": "train",
                    "class": class_name,
                    "filename": f"{class_name}_{index}.npy",
                    "raw_path": f"datasets/raw/{class_name}_{index}.npy",
                    "raw_sha256": f"{class_name}-{index:02d}".ljust(64, "0"),
                    "staging_action": "copied",
                    "repair_region_count": "1" if class_name == "10um" and index < 5 else "0",
                }
            )
    for index in range(3):
        rows.append(
            {
                "split": "val",
                "class": "2um",
                "filename": f"val_{index}.npy",
                "raw_path": f"datasets/raw/val_{index}.npy",
                "raw_sha256": f"val-{index:02d}".ljust(64, "0"),
                "staging_action": "copied",
                "repair_region_count": "0",
            }
        )
    return rows


def _write_inventory(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _repair_rows() -> list[dict[str, str]]:
    return [
        {
            "filename": f"10um_{index}.npy",
            "expanded_start_sample": "1500",
            "expanded_end_sample": "2500",
        }
        for index in range(5)
    ]


def test_population_assignment_is_deterministic_and_stratified() -> None:
    first = assign_population_roles(_rows())
    second = assign_population_roles(_rows())
    assert first == second
    counts = Counter(row["population_role"] for row in first)
    assert counts == {
        "mad_calibration": 24,
        "mad_holdout": 6,
        "legacy_exploration": 3,
    }
    assert all(row["population_role"] != "mad_holdout" for row in first if row["split"] == "val")


def test_holdout_requires_matching_completed_authorization(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.csv"
    _write_inventory(inventory, _rows())
    with pytest.raises(PermissionError, match="requires"):
        build_candidate_dataset(
            workspace_root=tmp_path,
            source_inventory=inventory,
            signal_dataset_root=tmp_path,
            output_dir=tmp_path / "output",
            dataset_id="candidate@v1",
            input_dataset_id="source@v1",
            input_manifest_sha256="a" * 64,
            config=ParticleDetectionConfig(),
            roles=["mad_holdout"],
        )
    authorization_dir = tmp_path / "authorization"
    authorization_dir.mkdir()
    authorization = authorization_dir / "run.json"
    authorization.write_text(
        json.dumps(
            {
                "run_id": "calibration-result-r1",
                "status": "visual_review_complete",
                "visual_checkpoint": {
                    "approved": True,
                    "next_stage_blocked": False,
                },
                "frozen_config_sha256": "wrong",
            }
        )
    )
    with pytest.raises(PermissionError, match="does not match"):
        build_candidate_dataset(
            workspace_root=tmp_path,
            source_inventory=inventory,
            signal_dataset_root=tmp_path,
            output_dir=tmp_path / "output",
            dataset_id="candidate@v1",
            input_dataset_id="source@v1",
            input_manifest_sha256="a" * 64,
            config=ParticleDetectionConfig(),
            roles=["mad_holdout"],
            holdout_authorization=authorization,
        )


def test_holdout_authorization_binds_run_receipt_and_config(tmp_path: Path) -> None:
    run_dir = tmp_path / "reviewed"
    (run_dir / "review").mkdir(parents=True)
    config = ParticleDetectionConfig()
    fingerprint = config_fingerprint(config)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "calibration-result-r1",
                "status": "visual_review_complete",
                "visual_checkpoint": {
                    "approved": True,
                    "next_stage_blocked": False,
                },
                "frozen_config_sha256": fingerprint,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "review" / "receipt.json").write_text(
        json.dumps({"run_id": "calibration-result-r1"}), encoding="utf-8"
    )
    validate_holdout_authorization(run_dir / "run.json", config_sha256=fingerprint)


def test_calibration_build_is_immutable_and_portable(tmp_path: Path) -> None:
    rows = _rows()
    inventory = tmp_path / "inventory.csv"
    _write_inventory(inventory, rows)
    repair_manifest = tmp_path / "repair.csv"
    _write_inventory(repair_manifest, _repair_rows())
    signal_root = tmp_path / "signals"
    for row in rows:
        path = signal_root / row["split"] / "signals" / row["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        time = np.arange(4096) / 2_000_000.0
        signal = 0.1 * np.sin(2 * np.pi * 20_000 * time)
        np.save(path, signal.astype(np.float32))
    config = ParticleDetectionConfig(acceptance_z=1_000.0)
    output = tmp_path / "candidate"
    manifest = build_candidate_dataset(
        workspace_root=tmp_path,
        source_inventory=inventory,
        signal_dataset_root=signal_root,
        output_dir=output,
        dataset_id="particle-mad@v1",
        input_dataset_id="source@v1",
        input_manifest_sha256="a" * 64,
        config=config,
        roles=["mad_calibration"],
        repair_manifest=repair_manifest,
    )
    assert manifest["config_sha256"] == config_fingerprint(config)
    assert manifest["counts"]["sources_executed"] == 24
    assert "/private/" not in json.dumps(manifest)
    assert json.loads((output / "dataset_contract.json").read_text())["label_policy"].startswith("Z8")
    with pytest.raises(FileExistsError):
        build_candidate_dataset(
            workspace_root=tmp_path,
            source_inventory=inventory,
            signal_dataset_root=signal_root,
            output_dir=output,
            dataset_id="particle-mad@v1",
            input_dataset_id="source@v1",
            input_manifest_sha256="a" * 64,
            config=config,
            roles=["mad_calibration"],
            repair_manifest=repair_manifest,
        )


def test_repaired_calibration_sources_require_exact_repair_manifest(tmp_path: Path) -> None:
    rows = _rows()
    inventory = tmp_path / "inventory.csv"
    _write_inventory(inventory, rows)
    with pytest.raises(ValueError, match="require a repair manifest"):
        build_candidate_dataset(
            workspace_root=tmp_path,
            source_inventory=inventory,
            signal_dataset_root=tmp_path,
            output_dir=tmp_path / "candidate",
            dataset_id="particle-mad@v1",
            input_dataset_id="source@v1",
            input_manifest_sha256="a" * 64,
            config=ParticleDetectionConfig(),
            roles=["mad_calibration"],
        )


def test_cli_writes_portable_manifested_run(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    rows = _rows()
    inventory = workspace / "inventory.csv"
    _write_inventory(inventory, rows)
    repair_manifest = workspace / "repair.csv"
    _write_inventory(repair_manifest, _repair_rows())
    signal_root = workspace / "signals"
    for row in rows:
        path = signal_root / row["split"] / "signals" / row["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.zeros(4096, dtype=np.float32))
    output = workspace / "datasets/interim/particle-mad/v1"
    run_output = workspace / "artifacts/particles2SNR-pipeline/runs/particle-mad-r1"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT)
    subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--workspace-root",
            str(workspace),
            "--source-inventory",
            str(inventory),
            "--repair-manifest",
            str(repair_manifest),
            "--signal-dataset-root",
            str(signal_root),
            "--output-dir",
            str(output),
            "--dataset-id",
            "particle-mad@v1",
            "--input-dataset-id",
            "source@v1",
            "--input-manifest-sha256",
            "a" * 64,
            "--role",
            "mad_calibration",
            "--run-output-dir",
            str(run_output),
            "--run-id",
            "particle-mad-r1",
            "--repository-revision",
            "b" * 40,
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    run = json.loads((run_output / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "complete"
    assert run["dataset_path"] == "datasets/interim/particle-mad/v1"
    assert str(tmp_path) not in json.dumps(run)
