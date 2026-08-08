#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from internship_workspace.config import Workspace
from internship_workspace.datasets import load_records, resolve_path
from internship_workspace.scientific_visual import computation_fingerprint
from particles2snr.z8_cholesky_generation import (
    CORRELATION_WARNING_THRESHOLD,
    FREQUENCY_BAND_KHZ,
    MODEL_LENGTH,
    PARAMETER_ORDER,
    RAW_LENGTH,
    SAMPLING_FREQUENCY_HZ,
    generate_parameters,
    load_gaussian_targets,
    load_physical_cholesky,
    synthesize_signals,
)


DATASET_ID = (
    "particles2snr-fbase-z8-cholesky-10um-density10x-physicalcorr-"
    "effective-snr-synthetic-events@v1"
)
BASELINE_DATASET_ID = (
    "particles2snr-fbase-z8-cholesky-physicalcorr-effective-snr-"
    "synthetic-events@v1"
)
SOURCE_DATASET_ID = (
    "particles2snr-fbase-dual-clean-z8-events-3class-plus-unclear-"
    "development@v1"
)
RUN_ID = "particle-z8-cholesky-10um-density10x-generation-v1"
GAUSSIAN_RUN_ID = "particle-z8-gaussian-intensity-envelope-v1"
CHOLESKY_RUN_ID = "particle-z8-cholesky-correlation-analysis-v1"
CLASS_NAME = "10um"
BASELINE_COUNT = 754
TARGET_COUNT = 7_540
SEED = 20_260_724


def _record(workspace: Workspace, key: str) -> dict[str, Any]:
    match = next(
        (
            record
            for record in load_records(workspace)
            if record.key == key
        ),
        None,
    )
    if match is None or match.payload["status"] not in {"active", "reference"}:
        raise ValueError(f"eligible registered dataset not found: {key}")
    return match.payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _run_fingerprint(run_dir: Path) -> str:
    payload = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    value = payload.get("computation_fingerprint")
    if not isinstance(value, str) or not value:
        raise ValueError(f"missing computation fingerprint: {run_dir}")
    return value


def _parameter_matrix(records: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [
            [
                row["log_amplitude_p0"],
                row["frequency_khz"],
                row["log_tau_ms"],
                row["snr_db"],
            ]
            for row in records
        ],
        dtype=np.float64,
    )


def _validate(
    records: list[dict[str, Any]],
    raw: np.ndarray,
    model: np.ndarray,
    factor: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    matrix = _parameter_matrix(records)
    realized = np.corrcoef(matrix, rowvar=False)
    target = factor @ factor.T
    delta = realized - target
    mask = ~np.eye(4, dtype=bool)
    maximum_delta = float(np.max(np.abs(delta[mask])))
    requested = np.asarray([row["snr_db"] for row in records])
    achieved = np.asarray([row["achieved_snr_db"] for row in records])
    checks = {
        "event_count": len(records) == TARGET_COUNT,
        "single_class_10um": {row["class_name"] for row in records}
        == {CLASS_NAME},
        "unique_ids": len({row["sample_id"] for row in records})
        == TARGET_COUNT,
        "raw_shape": raw.shape == (TARGET_COUNT, RAW_LENGTH),
        "model_shape": model.shape == (TARGET_COUNT, MODEL_LENGTH),
        "float32": raw.dtype == np.float32 and model.dtype == np.float32,
        "finite": bool(np.isfinite(raw).all() and np.isfinite(model).all()),
        "positive_amplitude_tau": all(
            row["amplitude_p0"] > 0.0 and row["tau_ms"] > 0.0
            for row in records
        ),
        "frequency_in_band": all(
            FREQUENCY_BAND_KHZ[0]
            <= row["frequency_khz"]
            <= FREQUENCY_BAND_KHZ[1]
            for row in records
        ),
        "snr_realization": bool(np.max(np.abs(requested - achieved)) < 1e-5),
        "no_sealed_test_access": True,
    }
    if not all(checks.values()):
        failed = [name for name, value in checks.items() if not value]
        raise ValueError(f"density10x candidate validation failed: {failed}")
    correlation_rows = []
    for row_index, row_name in enumerate(PARAMETER_ORDER):
        for column_index, column_name in enumerate(PARAMETER_ORDER):
            correlation_rows.append(
                {
                    "class_name": CLASS_NAME,
                    "row_parameter": row_name,
                    "column_parameter": column_name,
                    "target_correlation": float(target[row_index, column_index]),
                    "realized_correlation": float(
                        realized[row_index, column_index]
                    ),
                    "delta": float(delta[row_index, column_index]),
                }
            )
    summary = {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "status": (
            "warning_correlation_delta_above_threshold"
            if maximum_delta > CORRELATION_WARNING_THRESHOLD
            else "ready_for_scientific_review"
        ),
        "checks": checks,
        "event_count": TARGET_COUNT,
        "class_counts": {CLASS_NAME: TARGET_COUNT},
        "baseline_count": BASELINE_COUNT,
        "density_multiplier": TARGET_COUNT / BASELINE_COUNT,
        "maximum_absolute_off_diagonal_correlation_delta": maximum_delta,
        "correlation_warning_threshold": CORRELATION_WARNING_THRESHOLD,
        "correlation_warning": maximum_delta
        > CORRELATION_WARNING_THRESHOLD,
        "maximum_absolute_snr_error_db": float(
            np.max(np.abs(requested - achieved))
        ),
        "claim_boundary": (
            "This validates a 10x 10 µm sampling-density ablation with unchanged "
            "Gaussian marginals, physical-only Cholesky dependency matrix and "
            "signal synthesis. It does not yet establish improved latent twin "
            "coverage or visual realism."
        ),
    }
    return summary, correlation_rows


def _render_validation(
    *,
    generated: list[dict[str, Any]],
    baseline: list[dict[str, str]],
    target_mean: np.ndarray,
    target_sigma: np.ndarray,
    factor: np.ndarray,
    destination: Path,
) -> None:
    generated_matrix = _parameter_matrix(generated)
    baseline_matrix = np.asarray(
        [
            [
                float(row["log_amplitude_p0"]),
                float(row["frequency_khz"]),
                float(row["log_tau_ms"]),
                float(row["snr_db"]),
            ]
            for row in baseline
            if row["class_name"] == CLASS_NAME
        ]
    )
    if baseline_matrix.shape != (BASELINE_COUNT, 4):
        raise ValueError("baseline 10um count changed")
    figure = plt.figure(figsize=(18, 11), constrained_layout=True)
    grid = figure.add_gridspec(2, 4, height_ratios=(1.15, 1.0))
    labels = ("log(P0)", "Frequency (kHz)", "log(tau / ms)", "SNR (dB)")
    for index, label in enumerate(labels):
        axis = figure.add_subplot(grid[0, index])
        values = np.concatenate(
            (generated_matrix[:, index], baseline_matrix[:, index])
        )
        x = np.linspace(float(np.min(values)), float(np.max(values)), 700)
        axis.hist(
            generated_matrix[:, index],
            bins=70,
            density=True,
            alpha=0.35,
            color="#2563eb",
            label="7,540 density10x",
        )
        axis.hist(
            baseline_matrix[:, index],
            bins=42,
            density=True,
            histtype="step",
            linewidth=1.5,
            color="#64748b",
            label="754 baseline",
        )
        axis.plot(
            x,
            norm.pdf(
                (x - target_mean[index]) / target_sigma[index]
            )
            / target_sigma[index],
            color="#e11d48",
            linewidth=2.0,
            label="Validated target",
        )
        axis.set_title(label, fontweight="bold")
        axis.grid(alpha=0.15)
        if index == 0:
            axis.set_ylabel("Normalized density")
            axis.legend(fontsize=8)

    target = factor @ factor.T
    realized = np.corrcoef(generated_matrix, rowvar=False)
    matrices = (
        ("Target physical R", target, 1.0),
        ("Density10x realized R", realized, 1.0),
        ("Realized − target", realized - target, 0.10),
    )
    short = ("log(P0)", "fD", "log(tau)", "SNR")
    for index, (title, matrix, limit) in enumerate(matrices):
        axis = figure.add_subplot(grid[1, index])
        image = axis.imshow(
            matrix,
            vmin=-limit,
            vmax=limit,
            cmap="coolwarm",
        )
        axis.set_xticks(range(4), short, rotation=35, ha="right")
        axis.set_yticks(range(4), short)
        axis.set_title(title, fontweight="bold")
        for row in range(4):
            for column in range(4):
                value = matrix[row, column]
                axis.text(
                    column,
                    row,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white"
                    if abs(value) > 0.55 * limit
                    else "#0f172a",
                )
        figure.colorbar(image, ax=axis, shrink=0.7)
    note = figure.add_subplot(grid[1, 3])
    note.axis("off")
    note.text(
        0.05,
        0.82,
        "Controlled density ablation",
        fontsize=18,
        fontweight="bold",
        color="#0f172a",
    )
    note.text(
        0.05,
        0.62,
        (
            "Class: 10 µm\n"
            "Baseline: 754 events\n"
            "Candidate: 7,540 events\n"
            "Only changed factor: draw count\n\n"
            "Unchanged:\n"
            "• validated Gaussian marginals\n"
            "• physical-only Cholesky matrix\n"
            "• effective-SNR marginal\n"
            "• 7–80 kHz noise and signal contract\n"
            "• 4096 → 512 preprocessing"
        ),
        fontsize=12,
        va="top",
        linespacing=1.45,
        color="#334155",
    )
    figure.suptitle(
        "SSL v3 · 10 µm synthetic-density ablation (10×)",
        fontsize=22,
        fontweight="bold",
    )
    figure.savefig(destination, dpi=175, facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the controlled 10 µm Cholesky density10x ablation."
    )
    parser.add_argument("--dataset-output-dir", type=Path, required=True)
    parser.add_argument("--run-output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--resume-finalize", action="store_true")
    args = parser.parse_args()
    if (
        args.dataset_output_dir.exists() or args.run_output_dir.exists()
    ) and not args.resume_finalize:
        raise FileExistsError("refusing to overwrite an existing dataset or run")
    if args.resume_finalize and (
        not args.dataset_output_dir.is_dir()
        or not args.run_output_dir.is_dir()
    ):
        raise FileNotFoundError("resume requires the failed dataset and run directories")

    workspace = Workspace.load()
    source_record = _record(workspace, SOURCE_DATASET_ID)
    baseline_record = _record(workspace, BASELINE_DATASET_ID)
    gaussian_run = (
        workspace.root
        / "artifacts/particles2SNR-pipeline/analysis"
        / GAUSSIAN_RUN_ID
    )
    cholesky_run = (
        workspace.root
        / "artifacts/particles2SNR-pipeline/analysis"
        / CHOLESKY_RUN_ID
    )
    targets = load_gaussian_targets(
        gaussian_run / "gaussian_envelope_parameters.csv"
    )
    factors = load_physical_cholesky(cholesky_run / "cholesky_factors.csv")

    records, rejection_counts = generate_parameters(
        targets,
        factors,
        seed=SEED,
        budgets={CLASS_NAME: TARGET_COUNT},
        dataset_id=DATASET_ID,
    )
    raw, model = synthesize_signals(
        records,
        seed=SEED + 1,
        batch_size=args.batch_size,
    )
    summary, correlation_rows = _validate(
        records,
        raw,
        model,
        factors[CLASS_NAME],
    )

    if args.resume_finalize:
        existing_raw = np.load(
            args.dataset_output_dir / "signals_raw_4096.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        existing_model = np.load(
            args.dataset_output_dir / "signals_conv1dgap_512.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        existing_rows = _read_csv(args.dataset_output_dir / "events.csv")
        if (
            not np.array_equal(existing_raw, raw)
            or not np.array_equal(existing_model, model)
            or [row["sample_id"] for row in existing_rows]
            != [row["sample_id"] for row in records]
        ):
            raise ValueError("failed-run payload differs from deterministic replay")
    else:
        args.dataset_output_dir.mkdir(parents=True)
        _write_csv(args.dataset_output_dir / "events.csv", records)
        np.save(
            args.dataset_output_dir / "signals_raw_4096.npy",
            raw,
            allow_pickle=False,
        )
        np.save(
            args.dataset_output_dir / "signals_conv1dgap_512.npy",
            model,
            allow_pickle=False,
        )
    dataset_summary = {
        **summary,
        "status": "interim_candidate_awaiting_density_ablation_review",
        "seed": SEED,
        "rejection_counts": rejection_counts,
        "sealed_test_accessed": False,
        "source_dataset": {
            "id": SOURCE_DATASET_ID,
            "manifest_sha256": source_record["manifest_sha256"],
        },
        "baseline_comparison_dataset": {
            "id": BASELINE_DATASET_ID,
            "manifest_sha256": baseline_record["manifest_sha256"],
        },
        "source_analyses": {
            GAUSSIAN_RUN_ID: _run_fingerprint(gaussian_run),
            CHOLESKY_RUN_ID: _run_fingerprint(cholesky_run),
        },
        "signal_contract": {
            "sampling_frequency_hz": SAMPLING_FREQUENCY_HZ,
            "raw_length": RAW_LENGTH,
            "conv1dgap_length": MODEL_LENGTH,
            "conv1dgap_preprocessing": (
                "mean over contiguous blocks of 8, then per-window z-score"
            ),
        },
        "ablation_policy": {
            "class": CLASS_NAME,
            "baseline_count": BASELINE_COUNT,
            "candidate_count": TARGET_COUNT,
            "changed_factor": "synthetic draw count only",
        },
    }
    if not args.resume_finalize:
        (args.dataset_output_dir / "dataset_summary.json").write_text(
            json.dumps(dataset_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    input_contract = {
        "schema_version": 1,
        "format": "aligned synthetic event arrays plus metadata table",
        "events": "events.csv",
        "raw_signals": {
            "path": "signals_raw_4096.npy",
            "shape": [TARGET_COUNT, RAW_LENGTH],
            "dtype": "float32",
        },
        "conv1dgap_signals": {
            "path": "signals_conv1dgap_512.npy",
            "shape": [TARGET_COUNT, MODEL_LENGTH],
            "dtype": "float32",
        },
        "class_mapping": {"0": CLASS_NAME},
        "review_gate": (
            "Reference-only density ablation; never promote active before "
            "latent twin evaluation and explicit human review."
        ),
    }
    if not args.resume_finalize:
        (args.dataset_output_dir / "input_contract.json").write_text(
            json.dumps(input_contract, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if not args.resume_finalize:
        args.run_output_dir.mkdir(parents=True)
        (args.run_output_dir / "summary_metrics.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_csv(
            args.run_output_dir / "correlation_validation.csv",
            correlation_rows,
        )
    baseline_root = resolve_path(
        workspace,
        next(
            record
            for record in load_records(workspace)
            if record.key == BASELINE_DATASET_ID
        ),
    )
    figure_path = args.run_output_dir / "density10x_statistical_validation.png"
    if not args.resume_finalize:
        _render_validation(
            generated=records,
            baseline=_read_csv(baseline_root / "events.csv"),
            target_mean=np.asarray(targets[CLASS_NAME]["mean"]),
            target_sigma=np.asarray(targets[CLASS_NAME]["sigma"]),
            factor=factors[CLASS_NAME],
            destination=figure_path,
        )

    dataset_files = [
        "events.csv",
        "signals_raw_4096.npy",
        "signals_conv1dgap_512.npy",
        "dataset_summary.json",
        "input_contract.json",
    ]
    candidate_manifest = {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "status": "interim_candidate_awaiting_density_ablation_review",
        "files": [
            {
                "path": name,
                "size": (args.dataset_output_dir / name).stat().st_size,
                "sha256": _sha256(args.dataset_output_dir / name),
            }
            for name in dataset_files
        ],
    }
    if not args.resume_finalize:
        (args.run_output_dir / "candidate_dataset_manifest.json").write_text(
            json.dumps(candidate_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    repository = workspace.root / "particles2SNR-pipeline"
    script_path = (
        repository
        / "scripts/generation/generate_z8_cholesky_10um_density10x.py"
    )
    module_path = repository / "particles2snr/z8_cholesky_generation.py"
    git_states = {
        "workspace": _git_state(workspace.root),
        "particles2SNR-pipeline": _git_state(repository),
    }
    provenance = {
        "datasets": {
            SOURCE_DATASET_ID: source_record["manifest_sha256"],
            BASELINE_DATASET_ID: baseline_record["manifest_sha256"],
        },
        "source_runs": {
            GAUSSIAN_RUN_ID: _run_fingerprint(gaussian_run),
            CHOLESKY_RUN_ID: _run_fingerprint(cholesky_run),
        },
        "parameters": {
            "class": CLASS_NAME,
            "baseline_count": BASELINE_COUNT,
            "candidate_count": TARGET_COUNT,
            "density_multiplier": 10,
            "parameter_seed": SEED,
            "signal_noise_seed": SEED + 1,
            "changed_factor": "synthetic draw count only",
            "sealed_test_accessed": False,
        },
        "inputs": {
            "gaussian_parameters_sha256": _sha256(
                gaussian_run / "gaussian_envelope_parameters.csv"
            ),
            "cholesky_factors_sha256": _sha256(
                cholesky_run / "cholesky_factors.csv"
            ),
            "baseline_events_sha256": _sha256(
                baseline_root / "events.csv"
            ),
        },
        "metric_definitions": {
            "realized_correlation": (
                "Pearson correlation of generated [log(P0), frequency_khz, "
                "log(tau_ms), SNR_dB]"
            ),
            "correlation_delta": (
                "density10x realized correlation minus the validated physical "
                "10 µm target correlation"
            ),
            "snr_error_db": (
                "achieved signal SNR minus requested synthetic SNR"
            ),
        },
        "code": {
            "particles2snr/z8_cholesky_generation.py": _sha256(module_path),
            (
                "scripts/generation/"
                "generate_z8_cholesky_10um_density10x.py"
            ): _sha256(script_path),
        },
        "git_revision": git_states,
    }
    fingerprint = computation_fingerprint(provenance)
    metrics_manifest = {
        "schema_version": 1,
        "analysis_run_id": RUN_ID,
        "computation_provenance": provenance,
        "computation_fingerprint": fingerprint,
        "metrics": [
            {
                "path": name,
                "sha256": _sha256(args.run_output_dir / name),
            }
            for name in (
                "summary_metrics.json",
                "correlation_validation.csv",
            )
        ],
    }
    (args.run_output_dir / "metrics_manifest.json").write_text(
        json.dumps(metrics_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    run = {
        "schema_version": 1,
        "project": "particles2SNR-pipeline",
        "run_id": RUN_ID,
        "dataset": SOURCE_DATASET_ID,
        "datasets": {
            SOURCE_DATASET_ID: {
                "id": SOURCE_DATASET_ID,
                "manifest_sha256": source_record["manifest_sha256"],
            },
            BASELINE_DATASET_ID: {
                "id": BASELINE_DATASET_ID,
                "manifest_sha256": baseline_record["manifest_sha256"],
            },
        },
        "candidate_dataset": {
            "id": DATASET_ID,
            "path": args.dataset_output_dir.resolve().relative_to(
                workspace.root
            ).as_posix(),
            "status": "interim_candidate_awaiting_density_ablation_review",
            "registered": False,
        },
        "command": (
            "particles2SNR-pipeline/scripts/generation/"
            "generate_z8_cholesky_10um_density10x.py"
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_awaiting_scientific_review",
        "repositories": git_states,
        "outputs": [
            "summary_metrics.json",
            "correlation_validation.csv",
            "candidate_dataset_manifest.json",
            "density10x_statistical_validation.png",
            "metrics_manifest.json",
        ],
        "candidate_dataset_outputs": dataset_files,
        "method_evidence_ids": [
            "particle-z8-cholesky-method",
            "particle-z8-gaussian-intensity-envelope-method",
        ],
        "source_result_evidence_ids": [
            "particle-z8-cholesky-correlation-result",
            "particle-z8-gaussian-intensity-envelope-result",
        ],
        "computation_fingerprint": fingerprint,
        "claim_boundary": summary["claim_boundary"],
    }
    (args.run_output_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "run_id": RUN_ID,
                "candidate_dataset": DATASET_ID,
                "event_count": TARGET_COUNT,
                "maximum_correlation_delta": summary[
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
