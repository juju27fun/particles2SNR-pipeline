"""Ablate particles2SNR post-processing on the Noise negative control.

This script assumes the raw and filtered Noise detector runs already exist. It
does not rerun particles2SNR; it reuses ``dataset_results.json`` and exports
several post-processing variants to locate where false positives start to
survive.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from generate_particles2SNR_dataset import export_yolo_json
from repo_paths import RESULTS_REPORTS, RESULTS_RUNS


DEFAULT_CLASSES = ("2um", "4um", "10um")
DEFAULT_FS = 2_000_000.0
DEFAULT_MIN_PASSAGE_TIME_MS = 0.07
DEFAULT_MAX_PASSAGE_TIME_MS = 0.65
HIGH_SNR_THRESHOLD_DB = -10.0


VARIANTS = (
    {
        "stage": "detector",
        "label": "Detector candidates",
        "kind": "detector",
    },
    {
        "stage": "tau_only",
        "label": "Tau filter",
        "kind": "post",
        "kwargs": {
            "yolo_width_filter": False,
            "peak_evidence_filter": False,
            "merge_overlaps": False,
            "resolve_boundary_crossings": False,
        },
    },
    {
        "stage": "tau_width",
        "label": "Tau + width",
        "kind": "post",
        "kwargs": {
            "yolo_width_filter": True,
            "peak_evidence_filter": False,
            "merge_overlaps": False,
            "resolve_boundary_crossings": False,
        },
    },
    {
        "stage": "tau_width_nms_no_peak",
        "label": "Tau + width + NMS",
        "kind": "post",
        "kwargs": {
            "yolo_width_filter": True,
            "peak_evidence_filter": False,
            "merge_overlaps": True,
            "resolve_boundary_crossings": True,
        },
    },
    {
        "stage": "tau_width_peak_no_nms",
        "label": "Tau + width + peak",
        "kind": "post",
        "kwargs": {
            "yolo_width_filter": True,
            "peak_evidence_filter": True,
            "merge_overlaps": False,
            "resolve_boundary_crossings": False,
        },
    },
    {
        "stage": "default_postprocessed",
        "label": "Default post-processing",
        "kind": "post",
        "kwargs": {
            "yolo_width_filter": True,
            "peak_evidence_filter": True,
            "merge_overlaps": True,
            "resolve_boundary_crossings": True,
        },
    },
)


def parse_csv_arg(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def as_float(value) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def json_safe(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, Path):
        return str(value)
    return value


def read_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def detector_rows(run_dir: Path, pipeline: str, classes: tuple[str, ...]) -> list[dict]:
    file_rows = read_csv_rows(run_dir / "noise_by_file.csv")
    particle_rows = read_csv_rows(run_dir / "snr_particles.csv")
    return summarize_flat_rows(file_rows, particle_rows, pipeline, "detector", classes)


def export_variant(run_dir: Path, output_path: Path, classes: tuple[str, ...], kwargs: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_yolo_json(
        run_dir / "dataset_results.json",
        output_path,
        class_names=classes,
        fs=DEFAULT_FS,
        min_passage_time_ms=DEFAULT_MIN_PASSAGE_TIME_MS,
        max_passage_time_ms=DEFAULT_MAX_PASSAGE_TIME_MS,
        **kwargs,
    )


def summarize_flat_rows(
    file_rows: list[dict],
    particle_rows: list[dict],
    pipeline: str,
    stage: str,
    classes: tuple[str, ...],
) -> list[dict]:
    rows = []
    for class_name in classes:
        files = [row for row in file_rows if row.get("class") == class_name]
        particles = [
            row for row in particle_rows
            if row.get("class") == class_name and as_float(row.get("snr_db")) is not None
        ]
        counts = [int(float(row.get("num_particles", 0) or 0)) for row in files]
        snrs = [float(row["snr_db"]) for row in particles if as_float(row.get("snr_db")) is not None]
        rows.append(summarize_counts(pipeline, stage, class_name, len(files), counts, snrs))
    return rows


def summarize_data_json(path: Path, pipeline: str, stage: str, classes: tuple[str, ...]) -> list[dict]:
    with path.open() as f:
        data = json.load(f)
    rows = []
    for class_name in classes:
        samples = [row for row in data.get("data", []) if row.get("class_name") == class_name]
        counts = [len(row.get("annotations", [])) for row in samples]
        snrs = []
        for row in samples:
            for ann in row.get("annotations", []):
                snr = as_float(ann.get("snr_db"))
                if snr is not None:
                    snrs.append(snr)
        rows.append(summarize_counts(pipeline, stage, class_name, len(samples), counts, snrs))
    return rows


def summarize_counts(
    pipeline: str,
    stage: str,
    class_name: str,
    n_files: int,
    counts: list[int],
    snrs: list[float],
) -> dict:
    arr = np.asarray(counts, dtype=float)
    total = int(np.sum(arr)) if len(arr) else 0
    high_snr = int(sum(1 for value in snrs if value >= HIGH_SNR_THRESHOLD_DB))
    return {
        "pipeline": pipeline,
        "stage": stage,
        "class": class_name,
        "n_files": int(n_files),
        "total_false_particles": total,
        "mean_false_particles_per_file": float(np.mean(arr)) if len(arr) else math.nan,
        "median_false_particles_per_file": float(np.median(arr)) if len(arr) else math.nan,
        "p90_false_particles_per_file": float(np.percentile(arr, 90)) if len(arr) else math.nan,
        "max_false_particles_per_file": float(np.max(arr)) if len(arr) else math.nan,
        "false_particles_snr_ge_neg10db": high_snr,
        "false_particles_snr_ge_neg10db_per_file": high_snr / n_files if n_files else math.nan,
        "median_snr_db": float(np.median(snrs)) if snrs else math.nan,
        "p90_snr_db": float(np.percentile(snrs, 90)) if snrs else math.nan,
        "max_snr_db": float(np.max(snrs)) if snrs else math.nan,
    }


def build_delta_rows(summary_rows: list[dict]) -> list[dict]:
    by_key = {(row["stage"], row["class"], row["pipeline"]): row for row in summary_rows}
    rows = []
    stages = [variant["stage"] for variant in VARIANTS]
    classes = sorted({row["class"] for row in summary_rows})
    for stage in stages:
        for class_name in classes:
            raw = by_key.get((stage, class_name, "raw"))
            filtered = by_key.get((stage, class_name, "filtered"))
            if raw is None or filtered is None:
                continue
            raw_mean = raw["mean_false_particles_per_file"]
            filt_mean = filtered["mean_false_particles_per_file"]
            rows.append({
                "stage": stage,
                "class": class_name,
                "raw_mean_false_particles_per_file": raw_mean,
                "filtered_mean_false_particles_per_file": filt_mean,
                "filtered_minus_raw_fp_per_file": filt_mean - raw_mean,
                "filtered_effect_percent": (1.0 - filt_mean / raw_mean) * 100.0 if raw_mean > 0 else math.nan,
                "raw_high_snr_fp_per_file": raw["false_particles_snr_ge_neg10db_per_file"],
                "filtered_high_snr_fp_per_file": filtered["false_particles_snr_ge_neg10db_per_file"],
                "filtered_minus_raw_high_snr_fp_per_file": (
                    filtered["false_particles_snr_ge_neg10db_per_file"]
                    - raw["false_particles_snr_ge_neg10db_per_file"]
                ),
            })
    return rows


def plot_ablation(summary_rows: list[dict], delta_rows: list[dict], output_base: Path) -> None:
    stage_labels = {variant["stage"]: variant["label"] for variant in VARIANTS}
    stages = [variant["stage"] for variant in VARIANTS]
    classes = ("2um", "4um", "10um")
    colors = {"raw": "#4c72b0", "filtered": "#dd8452"}
    lookup = {(row["pipeline"], row["stage"], row["class"]): row for row in summary_rows}

    fig, axes = plt.subplots(1, len(classes), figsize=(15, 5.0), sharey=True)
    x = np.arange(len(stages))
    width = 0.38
    for ax, class_name in zip(axes, classes):
        for idx, pipeline in enumerate(("raw", "filtered")):
            values = [
                lookup[(pipeline, stage, class_name)]["mean_false_particles_per_file"]
                for stage in stages
            ]
            ax.bar(x + (idx - 0.5) * width, values, width, color=colors[pipeline], label=pipeline)
        ax.set_title(class_name)
        ax.set_xticks(x)
        ax.set_xticklabels([stage_labels[stage] for stage in stages], rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
    axes[0].set_ylabel("False positives / Noise file")
    fig.suptitle("Noise negative-control ablation", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_base.with_suffix(".png"), dpi=220)
    plt.close(fig)

    with PdfPages(output_base.with_suffix(".pdf")) as pdf:
        pdf.savefig(fig_from_png_summary(summary_rows, stages, classes, stage_labels, colors))
        fig, axes = plt.subplots(1, len(classes), figsize=(15, 5.0), sharey=True)
        delta_lookup = {(row["stage"], row["class"]): row for row in delta_rows}
        for ax, class_name in zip(axes, classes):
            values = [delta_lookup[(stage, class_name)]["filtered_minus_raw_fp_per_file"] for stage in stages]
            bar_colors = ["#c44e52" if value > 0 else "#55a868" for value in values]
            ax.bar(x, values, color=bar_colors)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_title(class_name)
            ax.set_xticks(x)
            ax.set_xticklabels([stage_labels[stage] for stage in stages], rotation=35, ha="right", fontsize=8)
            ax.grid(axis="y", alpha=0.25)
        axes[0].set_ylabel("Filtered - raw FP/file")
        fig.suptitle("Where the filtered pass diverges", fontsize=15, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        pdf.savefig(fig)
        plt.close(fig)


def fig_from_png_summary(summary_rows, stages, classes, stage_labels, colors):
    lookup = {(row["pipeline"], row["stage"], row["class"]): row for row in summary_rows}
    fig, axes = plt.subplots(1, len(classes), figsize=(15, 5.0), sharey=True)
    x = np.arange(len(stages))
    width = 0.38
    for ax, class_name in zip(axes, classes):
        for idx, pipeline in enumerate(("raw", "filtered")):
            values = [
                lookup[(pipeline, stage, class_name)]["mean_false_particles_per_file"]
                for stage in stages
            ]
            ax.bar(x + (idx - 0.5) * width, values, width, color=colors[pipeline], label=pipeline)
        ax.set_title(class_name)
        ax.set_xticks(x)
        ax.set_xticklabels([stage_labels[stage] for stage in stages], rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
    axes[0].set_ylabel("False positives / Noise file")
    fig.suptitle("Noise negative-control ablation", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ablate Noise negative-control post-processing stages.")
    parser.add_argument("--raw-output", type=Path, default=RESULTS_RUNS / "noise_negative_control_raw")
    parser.add_argument("--filtered-output", type=Path, default=RESULTS_RUNS / "noise_negative_control_F")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_REPORTS / "noise_negative_control_ablation")
    parser.add_argument("--classes", type=parse_csv_arg, default=DEFAULT_CLASSES)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    classes = tuple(args.classes)
    run_dirs = {"raw": args.raw_output, "filtered": args.filtered_output}
    variant_dir = args.output_dir / "variant_data"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for variant in VARIANTS:
        stage = variant["stage"]
        for pipeline, run_dir in run_dirs.items():
            if variant["kind"] == "detector":
                summary_rows.extend(detector_rows(run_dir, pipeline, classes))
                continue
            output_path = variant_dir / pipeline / f"{stage}.json"
            export_variant(run_dir, output_path, classes, variant["kwargs"])
            summary_rows.extend(summarize_data_json(output_path, pipeline, stage, classes))

    delta_rows = build_delta_rows(summary_rows)
    write_csv(
        args.output_dir / "noise_ablation_summary.csv",
        summary_rows,
        [
            "pipeline", "stage", "class", "n_files", "total_false_particles",
            "mean_false_particles_per_file", "median_false_particles_per_file",
            "p90_false_particles_per_file", "max_false_particles_per_file",
            "false_particles_snr_ge_neg10db", "false_particles_snr_ge_neg10db_per_file",
            "median_snr_db", "p90_snr_db", "max_snr_db",
        ],
    )
    write_csv(
        args.output_dir / "noise_ablation_filtered_vs_raw.csv",
        delta_rows,
        [
            "stage", "class", "raw_mean_false_particles_per_file",
            "filtered_mean_false_particles_per_file", "filtered_minus_raw_fp_per_file",
            "filtered_effect_percent", "raw_high_snr_fp_per_file",
            "filtered_high_snr_fp_per_file", "filtered_minus_raw_high_snr_fp_per_file",
        ],
    )
    with (args.output_dir / "noise_ablation_summary.json").open("w") as f:
        json.dump(
            {
                "description": "Ablation of particles2SNR post-processing stages on the Noise negative control.",
                "variants": VARIANTS,
                "summary": summary_rows,
                "filtered_vs_raw": delta_rows,
            },
            f,
            indent=2,
            default=json_safe,
            allow_nan=False,
        )
    plot_ablation(summary_rows, delta_rows, args.output_dir / "noise_ablation")
    print(f"Wrote ablation report to {args.output_dir}")
    print(f"- {args.output_dir / 'noise_ablation_summary.csv'}")
    print(f"- {args.output_dir / 'noise_ablation_filtered_vs_raw.csv'}")
    print(f"- {args.output_dir / 'noise_ablation.pdf'}")


if __name__ == "__main__":
    main()
