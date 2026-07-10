"""Negative-control audit for particles2SNR false positives on Noise files.

Every particle detected in the standalone Noise folder is treated as a false
positive. The script runs the detector on the same Noise files under the class
specific FFT settings used by the project, then compares a raw Noise pass with
the clean-filtered (_F) pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

from generate_particles2SNR_dataset import butter_bandpass_filter, export_yolo_json
from particles2snr.repo_paths import MONOREPO_ROOT, RESULTS_REPORTS, RESULTS_RUNS
from particles2snr.run_dataset import export_results, get_config_for_folder, process_signal


DEFAULT_CLASSES = ("2um", "4um", "10um")
DEFAULT_THRESHOLDS_DB = (-10.0, 0.0, 5.0, 10.0)
DEFAULT_FS = 2_000_000.0
DEFAULT_MIN_PASSAGE_TIME_MS = 0.07
DEFAULT_MAX_PASSAGE_TIME_MS = 0.65
BANDS_HZ = (
    ("0-1 kHz", 0.0, 1_000.0),
    ("1-7 kHz", 1_000.0, 7_000.0),
    ("7-10 kHz", 7_000.0, 10_000.0),
    ("10-40 kHz", 10_000.0, 40_000.0),
    ("40-80 kHz", 40_000.0, 80_000.0),
    ("80 kHz-Nyq", 80_000.0, math.inf),
)
PLOT_BAND_LABELS = ("0-1", "1-7", "7-10", "10-40", "40-80", "80-Nyq")


def parse_csv_arg(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_float_csv_arg(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def json_safe(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def as_float(value) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def percentile(values: list[float], q: float) -> float | None:
    clean = np.asarray(values, dtype=float)
    if len(clean) == 0:
        return None
    return float(np.percentile(clean, q))


def summarize_values(values: list[float]) -> dict:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "median": None,
            "p90": None,
            "p99": None,
            "max": None,
        }
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def frequency_band(freq_hz: float) -> str:
    for label, low, high in BANDS_HZ:
        if math.isinf(high):
            if freq_hz >= low:
                return label
        elif low <= freq_hz < high:
            return label
    return BANDS_HZ[-1][0]


def reduction_fraction(raw_value: float | None, filtered_value: float | None) -> float | None:
    if raw_value is None or filtered_value is None or raw_value <= 0:
        return None
    return float(1.0 - filtered_value / raw_value)


def require_clean_output(path: Path, force: bool) -> None:
    if not path.exists():
        return
    if not force:
        raise FileExistsError(f"{path} already exists. Use --force to overwrite generated audit outputs.")
    shutil.rmtree(path)


def prepare_noise_dataset(
    noise_dir: Path,
    output_dir: Path,
    classes: tuple[str, ...],
    mode: str,
    max_files: int | None,
    force: bool,
) -> list[dict]:
    require_clean_output(output_dir, force)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(noise_dir.glob("*.npy"))
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No .npy files found in Noise directory: {noise_dir}")

    manifest = []
    for class_name in classes:
        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        config = get_config_for_folder(class_name)
        for source_path in files:
            target_path = class_dir / source_path.name
            if mode == "raw":
                try:
                    target_path.symlink_to(source_path.resolve())
                    action = "symlinked"
                except OSError:
                    shutil.copy2(source_path, target_path)
                    action = "copied"
            elif mode == "filtered":
                signal = np.load(source_path)
                filtered = butter_bandpass_filter(
                    signal,
                    fs=float(config.sampling_rate),
                    fmin=float(config.bandpass_lowcut),
                    fmax=float(config.bandpass_highcut),
                )
                np.save(target_path, filtered)
                action = "bandpass_filtered"
            else:
                raise ValueError(f"Unsupported mode: {mode}")

            manifest.append({
                "mode": mode,
                "class": class_name,
                "filename": source_path.name,
                "source_path": str(source_path),
                "prepared_path": str(target_path),
                "action": action,
            })
    return manifest


def iter_dataset_files(dataset_dir: Path, classes: tuple[str, ...]) -> list[tuple[str, str]]:
    files = []
    for class_name in classes:
        class_dir = dataset_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing prepared class directory: {class_dir}")
        for path in sorted(class_dir.glob("*.npy")):
            files.append((str(path), class_name))
    return files


def run_particles2snr_on_dataset(
    dataset_dir: Path,
    output_dir: Path,
    classes: tuple[str, ...],
    device: str,
    force: bool,
) -> None:
    require_clean_output(output_dir, force)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_files = iter_dataset_files(dataset_dir, classes)
    if not data_files:
        raise RuntimeError(f"No prepared files found in {dataset_dir}")

    args = SimpleNamespace(device=device, verbose=False)
    results = []
    start = time.perf_counter()
    for signal_idx, (file_path, class_name) in enumerate(tqdm(data_files, desc=f"Processing {output_dir.name}")):
        config = get_config_for_folder(class_name)
        result = process_signal(file_path, class_name, config, args, signal_idx)
        if result is not None:
            results.append(result)
    export_results(results, str(output_dir), (time.perf_counter() - start) * 1000.0)
    export_yolo_json(
        output_dir / "dataset_results.json",
        output_dir / "data.json",
        class_names=classes,
        fs=DEFAULT_FS,
        min_passage_time_ms=DEFAULT_MIN_PASSAGE_TIME_MS,
        max_passage_time_ms=DEFAULT_MAX_PASSAGE_TIME_MS,
    )


def read_csv_rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_pipeline_rows(run_dirs: dict[str, Path]) -> tuple[list[dict], list[dict]]:
    file_rows = []
    particle_rows = []
    for pipeline, run_dir in run_dirs.items():
        for row in read_csv_rows(run_dir / "noise_by_file.csv"):
            out = dict(row)
            out["pipeline"] = pipeline
            for key in ("signal_idx", "signal_length", "num_particles", "num_windows", "num_valid_windows"):
                out[key] = int(float(out[key])) if out.get(key) not in (None, "") else 0
            out["particles_per_file"] = out["num_particles"]
            out["particles_per_valid_window"] = (
                out["num_particles"] / out["num_valid_windows"]
                if out["num_valid_windows"] > 0 else math.nan
            )
            file_rows.append(out)

        for row in read_csv_rows(run_dir / "snr_particles.csv"):
            out = dict(row)
            out["pipeline"] = pipeline
            out["snr_db"] = as_float(out.get("snr_db"))
            out["frequency"] = as_float(out.get("frequency"))
            out["tau"] = as_float(out.get("tau"))
            out["energy"] = as_float(out.get("energy"))
            out["frequency_band"] = frequency_band(out["frequency"]) if out["frequency"] is not None else ""
            particle_rows.append(out)
    return file_rows, particle_rows


def load_postprocessed_rows(run_dirs: dict[str, Path]) -> tuple[list[dict], list[dict]]:
    file_rows = []
    particle_rows = []
    for pipeline, run_dir in run_dirs.items():
        path = run_dir / "data.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing post-processed data.json: {path}")
        with path.open() as f:
            data = json.load(f)
        for signal_idx, row in enumerate(data.get("data", [])):
            annotations = list(row.get("annotations", []))
            file_row = {
                "pipeline": pipeline,
                "class": row.get("class_name"),
                "filename": row.get("filename"),
                "path": row.get("path"),
                "signal_idx": signal_idx,
                "signal_length": int(row.get("length") or 0),
                "num_particles": len(annotations),
                "num_windows": 0,
                "num_valid_windows": 0,
                "particles_per_file": len(annotations),
                "particles_per_valid_window": math.nan,
                "noise_floor": "",
                "noise_floor_N": "",
                "raw_std": "",
                "filtered_std": "",
                "inband_energy_ratio": "",
                "spectral_flatness": "",
            }
            file_rows.append(file_row)
            for ann_idx, ann in enumerate(annotations):
                snr = as_float(ann.get("snr_db"))
                frequency = as_float(ann.get("frequency"))
                particle_rows.append({
                    "pipeline": pipeline,
                    "class": row.get("class_name"),
                    "filename": row.get("filename"),
                    "path": row.get("path"),
                    "signal_idx": signal_idx,
                    "signal_length": int(row.get("length") or 0),
                    "particle_idx": ann_idx,
                    "frequency": frequency,
                    "frequency_band": frequency_band(frequency) if frequency is not None else "",
                    "P0": ann.get("amplitude"),
                    "t0": ann.get("center"),
                    "tau": ann.get("passage_time_ms"),
                    "phi": "",
                    "energy": "",
                    "snr_db": snr,
                    "noise_floor": "",
                    "noise_floor_N": "",
                    "source_window_idx": "",
                    "source_window_center": "",
                    "source_window_energy": "",
                    "peak_support": ann.get("peak_support"),
                    "peak_z": ann.get("peak_z"),
                    "local_peak_z": ann.get("local_peak_z"),
                })
    return file_rows, particle_rows


def build_summary_rows(
    file_rows: list[dict],
    particle_rows: list[dict],
    classes: tuple[str, ...],
    thresholds_db: tuple[float, ...],
) -> list[dict]:
    rows = []
    for pipeline in sorted({row["pipeline"] for row in file_rows}):
        for class_name in classes:
            files = [row for row in file_rows if row["pipeline"] == pipeline and row["class"] == class_name]
            particles = [
                row for row in particle_rows
                if row["pipeline"] == pipeline and row["class"] == class_name and row["snr_db"] is not None
            ]
            counts = [int(row["num_particles"]) for row in files]
            rates = [
                float(row["particles_per_valid_window"]) for row in files
                if math.isfinite(float(row["particles_per_valid_window"]))
            ]
            snrs = [float(row["snr_db"]) for row in particles]
            summary_counts = summarize_values(counts)
            summary_rates = summarize_values(rates)
            summary_snrs = summarize_values(snrs)
            row = {
                "pipeline": pipeline,
                "class": class_name,
                "n_files": len(files),
                "total_false_particles": int(sum(counts)),
                "mean_false_particles_per_file": summary_counts["mean"],
                "median_false_particles_per_file": summary_counts["median"],
                "p90_false_particles_per_file": summary_counts["p90"],
                "max_false_particles_per_file": summary_counts["max"],
                "mean_false_particles_per_valid_window": summary_rates["mean"],
                "median_snr_db": summary_snrs["median"],
                "p90_snr_db": summary_snrs["p90"],
                "p99_snr_db": summary_snrs["p99"],
                "max_snr_db": summary_snrs["max"],
            }
            for threshold in thresholds_db:
                key = f"false_particles_snr_ge_{threshold:g}db".replace("-", "neg").replace(".", "p")
                count = sum(1 for value in snrs if value >= threshold)
                row[key] = int(count)
                row[f"{key}_per_file"] = count / len(files) if files else math.nan
            rows.append(row)
    return rows


def build_reduction_rows(summary_rows: list[dict], thresholds_db: tuple[float, ...]) -> list[dict]:
    by_class = defaultdict(dict)
    for row in summary_rows:
        by_class[row["class"]][row["pipeline"]] = row
    rows = []
    for class_name, grouped in sorted(by_class.items()):
        raw = grouped.get("raw")
        filt = grouped.get("filtered")
        if not raw or not filt:
            continue
        out = {
            "class": class_name,
            "raw_total_false_particles": raw["total_false_particles"],
            "filtered_total_false_particles": filt["total_false_particles"],
            "total_false_particle_reduction_frac": reduction_fraction(
                raw["total_false_particles"], filt["total_false_particles"]
            ),
            "raw_mean_false_particles_per_file": raw["mean_false_particles_per_file"],
            "filtered_mean_false_particles_per_file": filt["mean_false_particles_per_file"],
            "mean_false_particles_per_file_reduction_frac": reduction_fraction(
                raw["mean_false_particles_per_file"], filt["mean_false_particles_per_file"]
            ),
            "raw_p90_false_particles_per_file": raw["p90_false_particles_per_file"],
            "filtered_p90_false_particles_per_file": filt["p90_false_particles_per_file"],
            "p90_false_particles_per_file_reduction_frac": reduction_fraction(
                raw["p90_false_particles_per_file"], filt["p90_false_particles_per_file"]
            ),
        }
        for threshold in thresholds_db:
            key = f"false_particles_snr_ge_{threshold:g}db".replace("-", "neg").replace(".", "p")
            out[f"raw_{key}"] = raw[key]
            out[f"filtered_{key}"] = filt[key]
            out[f"{key}_reduction_frac"] = reduction_fraction(raw[key], filt[key])
        rows.append(out)
    return rows


def build_threshold_rows(
    particle_rows: list[dict],
    file_rows: list[dict],
    classes: tuple[str, ...],
    thresholds_db: tuple[float, ...],
) -> list[dict]:
    rows = []
    n_files = {
        (row["pipeline"], row["class"]): 0 for row in file_rows
    }
    for row in file_rows:
        n_files[(row["pipeline"], row["class"])] += 1
    for pipeline in sorted({row["pipeline"] for row in file_rows}):
        for class_name in classes:
            snrs = [
                float(row["snr_db"]) for row in particle_rows
                if row["pipeline"] == pipeline and row["class"] == class_name and row["snr_db"] is not None
            ]
            files = n_files.get((pipeline, class_name), 0)
            for threshold in thresholds_db:
                count = sum(1 for value in snrs if value >= threshold)
                rows.append({
                    "pipeline": pipeline,
                    "class": class_name,
                    "snr_threshold_db": threshold,
                    "false_particles_ge_threshold": int(count),
                    "false_particles_ge_threshold_per_file": count / files if files else math.nan,
                })
    return rows


def build_band_rows(particle_rows: list[dict], classes: tuple[str, ...]) -> list[dict]:
    rows = []
    for pipeline in sorted({row["pipeline"] for row in particle_rows}):
        for class_name in classes:
            particles = [
                row for row in particle_rows
                if row["pipeline"] == pipeline and row["class"] == class_name and row.get("frequency_band")
            ]
            total = len(particles)
            for label, low, high in BANDS_HZ:
                count = sum(1 for row in particles if row["frequency_band"] == label)
                rows.append({
                    "pipeline": pipeline,
                    "class": class_name,
                    "band": label,
                    "band_low_hz": low,
                    "band_high_hz": "" if math.isinf(high) else high,
                    "false_particles": int(count),
                    "false_particle_pct": count / total * 100.0 if total else math.nan,
                })
    return rows


def build_worst_file_rows(file_rows: list[dict], particle_rows: list[dict], limit: int) -> list[dict]:
    max_snr_by_key = defaultdict(lambda: None)
    high_snr_by_key = defaultdict(int)
    for row in particle_rows:
        key = (row["pipeline"], row["class"], row["filename"])
        snr = row.get("snr_db")
        if snr is None:
            continue
        max_snr_by_key[key] = snr if max_snr_by_key[key] is None else max(max_snr_by_key[key], snr)
        if snr >= -10.0:
            high_snr_by_key[key] += 1

    rows = []
    for row in file_rows:
        key = (row["pipeline"], row["class"], row["filename"])
        out = {
            "pipeline": row["pipeline"],
            "class": row["class"],
            "filename": row["filename"],
            "path": row["path"],
            "num_particles": row["num_particles"],
            "num_valid_windows": row["num_valid_windows"],
            "particles_per_valid_window": row["particles_per_valid_window"],
            "false_particles_snr_ge_neg10db": high_snr_by_key[key],
            "max_snr_db": max_snr_by_key[key],
        }
        rows.append(out)
    rows.sort(key=lambda item: (item["num_particles"], item["max_snr_db"] or -math.inf), reverse=True)
    return rows[:limit]


def write_markdown(
    path: Path,
    summary_rows: list[dict],
    reduction_rows: list[dict],
    thresholds_db: tuple[float, ...],
) -> None:
    lines = [
        "# Noise Negative-Control Audit",
        "",
        "Every detected particle in the standalone Noise folder is counted as a false positive.",
        "The raw pass uses the Noise files directly. The filtered pass applies the same final",
        "7-80 kHz Butterworth bandpass used by the accepted `_F` pipeline before detection.",
        "",
        "## Main Reduction",
        "",
        "| class | raw FP/file | filtered FP/file | filtered-pass effect | raw total FP | filtered total FP |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in reduction_rows:
        red = row["mean_false_particles_per_file_reduction_frac"]
        lines.append(
            f"| {row['class']} | {row['raw_mean_false_particles_per_file']:.3g} | "
            f"{row['filtered_mean_false_particles_per_file']:.3g} | "
            f"{'n/a' if red is None else f'{red * 100.0:.1f}%'} | {row['raw_total_false_particles']} | "
            f"{row['filtered_total_false_particles']} |"
        )
    lines.extend(["", "## High-SNR False Positives", ""])
    for threshold in thresholds_db:
        key = f"false_particles_snr_ge_{threshold:g}db".replace("-", "neg").replace(".", "p")
        lines.extend([
            f"### SNR >= {threshold:g} dB",
            "",
            "| class | raw | filtered | reduction |",
            "|---|---:|---:|---:|",
        ])
        for row in reduction_rows:
            red = row.get(f"{key}_reduction_frac")
            red_text = "n/a" if red is None else f"{red * 100.0:.1f}%"
            lines.append(f"| {row['class']} | {row[f'raw_{key}']} | {row[f'filtered_{key}']} | {red_text} |")
        lines.append("")
    lines.extend([
        "## Per-Pipeline Summary",
        "",
        "| pipeline | class | files | total FP | mean FP/file | p90 FP/file | median SNR | p99 SNR |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary_rows:
        lines.append(
            f"| {row['pipeline']} | {row['class']} | {row['n_files']} | "
            f"{row['total_false_particles']} | {row['mean_false_particles_per_file']:.3g} | "
            f"{row['p90_false_particles_per_file']:.3g} | "
            f"{row['median_snr_db'] if row['median_snr_db'] is not None else 'n/a'} | "
            f"{row['p99_snr_db'] if row['p99_snr_db'] is not None else 'n/a'} |"
        )
    path.write_text("\n".join(lines) + "\n")


def grouped_counts(file_rows: list[dict], pipeline: str, class_name: str) -> list[float]:
    return [
        float(row["num_particles"]) for row in file_rows
        if row["pipeline"] == pipeline and row["class"] == class_name
    ]


def grouped_snrs(particle_rows: list[dict], pipeline: str, class_name: str) -> list[float]:
    return [
        float(row["snr_db"]) for row in particle_rows
        if row["pipeline"] == pipeline and row["class"] == class_name and row["snr_db"] is not None
    ]


def plot_summary_png(output_path: Path, summary_rows: list[dict], reduction_rows: list[dict], classes: tuple[str, ...]) -> None:
    colors = {"raw": "#4c72b0", "filtered": "#dd8452"}
    x = np.arange(len(classes))
    width = 0.36
    summary_by_key = {(row["pipeline"], row["class"]): row for row in summary_rows}
    reduction_by_class = {row["class"]: row for row in reduction_rows}

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for idx, pipeline in enumerate(("raw", "filtered")):
        means = [summary_by_key[(pipeline, cls)]["mean_false_particles_per_file"] for cls in classes]
        axes[0].bar(x + (idx - 0.5) * width, means, width, label=pipeline, color=colors[pipeline])
    axes[0].set_title("False positives per Noise file")
    axes[0].set_ylabel("Detected particles / file")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(classes)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    reductions = []
    for class_name in classes:
        row = reduction_by_class.get(class_name, {})
        value = row.get("mean_false_particles_per_file_reduction_frac")
        reductions.append(value * 100.0 if value is not None else math.nan)
    bar_colors = ["#55a868" if value >= 0 else "#c44e52" for value in reductions]
    axes[1].bar(classes, reductions, color=bar_colors)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_title("Filtered-pass effect")
    axes[1].set_ylabel("Change in mean FP/file (%)")
    finite_reductions = [value for value in reductions if math.isfinite(value)]
    if finite_reductions:
        lower = min(0.0, min(finite_reductions) - max(1.0, abs(min(finite_reductions)) * 0.15))
        upper = max(0.0, max(finite_reductions) + max(1.0, abs(max(finite_reductions)) * 0.15))
        if upper == 0.0:
            upper = 1.0
        axes[1].set_ylim(lower, upper)
    axes[1].grid(axis="y", alpha=0.25)
    for idx, value in enumerate(reductions):
        if math.isfinite(value):
            axes[1].text(idx, value, f"{value:.1f}%", ha="center", va="bottom" if value >= 0 else "top", fontsize=9)
    fig.suptitle("Noise negative-control false-positive audit", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_pdf(
    output_path: Path,
    file_rows: list[dict],
    particle_rows: list[dict],
    summary_rows: list[dict],
    reduction_rows: list[dict],
    threshold_rows: list[dict],
    band_rows: list[dict],
    worst_rows: list[dict],
    classes: tuple[str, ...],
) -> None:
    colors = {"raw": "#4c72b0", "filtered": "#dd8452"}
    summary_by_key = {(row["pipeline"], row["class"]): row for row in summary_rows}

    with PdfPages(output_path) as pdf:
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.set_title("Noise Negative-Control Audit", fontsize=16, fontweight="bold", pad=18)
        lines = [
            "Every detected particle in Noise is a false positive.",
            "Raw = direct Noise files. Filtered = same 7-80 kHz bandpass as the _F pipeline.",
            "",
            "Main result by class:",
        ]
        for row in reduction_rows:
            red = row["mean_false_particles_per_file_reduction_frac"]
            red_text = "n/a" if red is None else f"{red * 100.0:.1f}%"
            lines.append(
                f"{row['class']}: raw {row['raw_mean_false_particles_per_file']:.2f} FP/file, "
                f"filtered {row['filtered_mean_false_particles_per_file']:.2f}, "
                f"filtered-pass effect {red_text}"
            )
        lines.extend(["", "Worst files are listed at the end of this report for visual follow-up."])
        ax.text(0.05, 0.88, "\n".join(lines), va="top", fontsize=11, family="monospace")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        x = np.arange(len(classes))
        width = 0.36
        fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
        for idx, pipeline in enumerate(("raw", "filtered")):
            means = [summary_by_key[(pipeline, cls)]["mean_false_particles_per_file"] for cls in classes]
            p90s = [summary_by_key[(pipeline, cls)]["p90_false_particles_per_file"] for cls in classes]
            axes[0].bar(x + (idx - 0.5) * width, means, width, label=pipeline, color=colors[pipeline])
            axes[1].bar(x + (idx - 0.5) * width, p90s, width, label=pipeline, color=colors[pipeline])
        axes[0].set_title("Mean false positives per file")
        axes[1].set_title("P90 false positives per file")
        for ax in axes:
            ax.set_ylabel("Detected particles / file")
            ax.set_xticks(x)
            ax.set_xticklabels(classes)
            ax.grid(axis="y", alpha=0.25)
            ax.legend()
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(1, len(classes), figsize=(4.2 * len(classes), 5.2), sharey=True)
        if len(classes) == 1:
            axes = [axes]
        for ax, class_name in zip(axes, classes):
            data = [grouped_counts(file_rows, pipeline, class_name) for pipeline in ("raw", "filtered")]
            bp = ax.boxplot(data, tick_labels=("raw", "filtered"), showmeans=True, patch_artist=True)
            for patch, color in zip(bp["boxes"], (colors["raw"], colors["filtered"])):
                patch.set_facecolor(color)
                patch.set_alpha(0.55)
            ax.set_title(class_name)
            ax.grid(axis="y", alpha=0.25)
        axes[0].set_ylabel("Detected particles / file")
        fig.suptitle("File-level false-positive distribution", fontsize=14, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(1, len(classes), figsize=(4.2 * len(classes), 5.2), sharey=True)
        if len(classes) == 1:
            axes = [axes]
        thresholds = sorted({float(row["snr_threshold_db"]) for row in threshold_rows})
        for ax, class_name in zip(axes, classes):
            for pipeline in ("raw", "filtered"):
                values = [
                    next(
                        row["false_particles_ge_threshold_per_file"]
                        for row in threshold_rows
                        if row["pipeline"] == pipeline and row["class"] == class_name and float(row["snr_threshold_db"]) == threshold
                    )
                    for threshold in thresholds
                ]
                ax.plot(thresholds, values, marker="o", label=pipeline, color=colors[pipeline])
            ax.axvline(-10.0, color="#c44e52", linestyle="--", linewidth=1.0, alpha=0.75)
            ax.set_title(class_name)
            ax.set_xlabel("SNR threshold (dB)")
            ax.grid(alpha=0.25)
            ax.legend()
        axes[0].set_ylabel("False positives >= threshold / file")
        fig.suptitle("High-SNR false-positive survival curve", fontsize=14, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(1, len(classes), figsize=(4.2 * len(classes), 5.2), sharey=True)
        if len(classes) == 1:
            axes = [axes]
        for ax, class_name in zip(axes, classes):
            for pipeline in ("raw", "filtered"):
                snrs = grouped_snrs(particle_rows, pipeline, class_name)
                if snrs:
                    ax.hist(snrs, bins=40, density=True, histtype="step", linewidth=1.8, label=pipeline, color=colors[pipeline])
            ax.axvline(-10.0, color="#c44e52", linestyle="--", linewidth=1.0, alpha=0.75)
            ax.set_title(class_name)
            ax.set_xlabel("False-positive SNR (dB)")
            ax.grid(alpha=0.25)
            ax.legend()
        axes[0].set_ylabel("Density")
        fig.suptitle("False-positive SNR distributions", fontsize=14, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(len(classes), 1, figsize=(11, 3.1 * len(classes)), sharex=True)
        if len(classes) == 1:
            axes = [axes]
        band_lookup = {(row["pipeline"], row["class"], row["band"]): row for row in band_rows}
        x = np.arange(len(BANDS_HZ))
        for ax, class_name in zip(axes, classes):
            for idx, pipeline in enumerate(("raw", "filtered")):
                values = [
                    band_lookup.get((pipeline, class_name, label), {}).get("false_particle_pct", math.nan)
                    for label, _, _ in BANDS_HZ
                ]
                ax.bar(x + (idx - 0.5) * width, values, width, label=pipeline, color=colors[pipeline])
            ax.set_title(f"{class_name} false-positive frequency bands")
            ax.set_ylabel("% false positives")
            ax.grid(axis="y", alpha=0.25)
            ax.legend()
        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(PLOT_BAND_LABELS, rotation=25, ha="right")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(1, len(classes), figsize=(4.2 * len(classes), 5.2), sharey=True)
        if len(classes) == 1:
            axes = [axes]
        for ax, class_name in zip(axes, classes):
            for pipeline in ("raw", "filtered"):
                xs = [
                    row["frequency"] / 1000.0 for row in particle_rows
                    if row["pipeline"] == pipeline and row["class"] == class_name and row["frequency"] is not None and row["snr_db"] is not None
                ]
                ys = [
                    row["snr_db"] for row in particle_rows
                    if row["pipeline"] == pipeline and row["class"] == class_name and row["frequency"] is not None and row["snr_db"] is not None
                ]
                ax.scatter(xs, ys, s=8, alpha=0.35, label=pipeline, color=colors[pipeline], edgecolors="none")
            ax.axhline(-10.0, color="#c44e52", linestyle="--", linewidth=1.0, alpha=0.75)
            ax.set_title(class_name)
            ax.set_xlabel("Frequency (kHz)")
            ax.grid(alpha=0.25)
            ax.legend(markerscale=2)
        axes[0].set_ylabel("False-positive SNR (dB)")
        fig.suptitle("False-positive frequency vs SNR", fontsize=14, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.set_title("Worst Noise Files", fontsize=14, fontweight="bold")
        lines = ["pipeline class num_fp fp>=-10dB max_snr filename"]
        for row in worst_rows[:30]:
            max_snr = "n/a" if row["max_snr_db"] is None else f"{row['max_snr_db']:.2f}"
            lines.append(
                f"{row['pipeline']:<8} {row['class']:<5} {row['num_particles']:>6} "
                f"{row['false_particles_snr_ge_neg10db']:>9} {max_snr:>7} {row['filename']}"
            )
        ax.text(0.03, 0.94, "\n".join(lines), va="top", fontsize=9, family="monospace")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run and report a Noise negative-control audit for particles2SNR.")
    parser.add_argument("--noise-dir", type=Path, default=MONOREPO_ROOT / "P0" / "data" / "processed" / "Noise")
    parser.add_argument("--work-dir", type=Path, default=RESULTS_RUNS / "noise_negative_control_datasets")
    parser.add_argument("--raw-output", type=Path, default=RESULTS_RUNS / "noise_negative_control_raw")
    parser.add_argument("--filtered-output", type=Path, default=RESULTS_RUNS / "noise_negative_control_F")
    parser.add_argument("--report-dir", type=Path, default=RESULTS_REPORTS / "noise_negative_control_audit")
    parser.add_argument("--classes", type=parse_csv_arg, default=DEFAULT_CLASSES)
    parser.add_argument("--thresholds-db", type=parse_float_csv_arg, default=DEFAULT_THRESHOLDS_DB)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-files", type=int, default=None, help="Optional smoke-test limit; omitted uses all Noise files.")
    parser.add_argument("--skip-runs", action="store_true", help="Only build reports from existing raw/filtered outputs.")
    parser.add_argument("--force", action="store_true", help="Overwrite generated audit datasets, runs, and report directory.")
    parser.add_argument("--worst-files", type=int, default=40)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    classes = tuple(args.classes)
    thresholds_db = tuple(args.thresholds_db)

    if not args.skip_runs:
        raw_dataset = args.work_dir / "raw"
        filtered_dataset = args.work_dir / "filtered"
        manifest = []
        manifest.extend(prepare_noise_dataset(args.noise_dir, raw_dataset, classes, "raw", args.max_files, args.force))
        manifest.extend(prepare_noise_dataset(args.noise_dir, filtered_dataset, classes, "filtered", args.max_files, args.force))
        args.work_dir.mkdir(parents=True, exist_ok=True)
        write_csv(
            args.work_dir / "noise_negative_control_manifest.csv",
            manifest,
            ["mode", "class", "filename", "source_path", "prepared_path", "action"],
        )
        run_particles2snr_on_dataset(raw_dataset, args.raw_output, classes, args.device, args.force)
        run_particles2snr_on_dataset(filtered_dataset, args.filtered_output, classes, args.device, args.force)

    require_clean_output(args.report_dir, args.force)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = {"raw": args.raw_output, "filtered": args.filtered_output}
    file_rows, particle_rows = load_pipeline_rows(run_dirs)
    post_file_rows, post_particle_rows = load_postprocessed_rows(run_dirs)
    summary_rows = build_summary_rows(file_rows, particle_rows, classes, thresholds_db)
    reduction_rows = build_reduction_rows(summary_rows, thresholds_db)
    threshold_rows = build_threshold_rows(particle_rows, file_rows, classes, thresholds_db)
    band_rows = build_band_rows(particle_rows, classes)
    worst_rows = build_worst_file_rows(file_rows, particle_rows, args.worst_files)

    post_summary_rows = build_summary_rows(post_file_rows, post_particle_rows, classes, thresholds_db)
    post_reduction_rows = build_reduction_rows(post_summary_rows, thresholds_db)
    post_threshold_rows = build_threshold_rows(post_particle_rows, post_file_rows, classes, thresholds_db)
    post_band_rows = build_band_rows(post_particle_rows, classes) if post_particle_rows else []
    post_worst_rows = build_worst_file_rows(post_file_rows, post_particle_rows, args.worst_files)

    write_csv(
        args.report_dir / "noise_fp_by_file.csv",
        file_rows,
        [
            "pipeline", "class", "filename", "path", "signal_idx", "signal_length",
            "num_particles", "num_windows", "num_valid_windows", "particles_per_file",
            "particles_per_valid_window", "noise_floor", "noise_floor_N",
            "raw_std", "filtered_std", "inband_energy_ratio", "spectral_flatness",
        ],
    )
    write_csv(
        args.report_dir / "noise_fp_particles.csv",
        particle_rows,
        [
            "pipeline", "class", "filename", "path", "signal_idx", "signal_length",
            "particle_idx", "frequency", "frequency_band", "P0", "t0", "tau",
            "phi", "energy", "snr_db", "noise_floor", "noise_floor_N",
            "source_window_idx", "source_window_center", "source_window_energy",
        ],
    )
    write_csv(
        args.report_dir / "noise_fp_summary.csv",
        summary_rows,
        list(summary_rows[0].keys()) if summary_rows else [],
    )
    write_csv(
        args.report_dir / "noise_fp_reduction.csv",
        reduction_rows,
        list(reduction_rows[0].keys()) if reduction_rows else [],
    )
    write_csv(
        args.report_dir / "noise_fp_by_threshold.csv",
        threshold_rows,
        ["pipeline", "class", "snr_threshold_db", "false_particles_ge_threshold", "false_particles_ge_threshold_per_file"],
    )
    write_csv(
        args.report_dir / "noise_fp_frequency_bands.csv",
        band_rows,
        ["pipeline", "class", "band", "band_low_hz", "band_high_hz", "false_particles", "false_particle_pct"],
    )
    write_csv(
        args.report_dir / "noise_fp_worst_files.csv",
        worst_rows,
        [
            "pipeline", "class", "filename", "path", "num_particles",
            "num_valid_windows", "particles_per_valid_window",
            "false_particles_snr_ge_neg10db", "max_snr_db",
        ],
    )
    write_csv(
        args.report_dir / "noise_fp_postprocessed_by_file.csv",
        post_file_rows,
        [
            "pipeline", "class", "filename", "path", "signal_idx", "signal_length",
            "num_particles", "particles_per_file",
        ],
    )
    write_csv(
        args.report_dir / "noise_fp_postprocessed_annotations.csv",
        post_particle_rows,
        [
            "pipeline", "class", "filename", "path", "signal_idx", "signal_length",
            "particle_idx", "frequency", "frequency_band", "P0", "t0", "tau",
            "snr_db", "peak_support", "peak_z", "local_peak_z",
        ],
    )
    write_csv(
        args.report_dir / "noise_fp_postprocessed_summary.csv",
        post_summary_rows,
        list(post_summary_rows[0].keys()) if post_summary_rows else [],
    )
    write_csv(
        args.report_dir / "noise_fp_postprocessed_reduction.csv",
        post_reduction_rows,
        list(post_reduction_rows[0].keys()) if post_reduction_rows else [],
    )
    write_csv(
        args.report_dir / "noise_fp_postprocessed_by_threshold.csv",
        post_threshold_rows,
        ["pipeline", "class", "snr_threshold_db", "false_particles_ge_threshold", "false_particles_ge_threshold_per_file"],
    )
    write_csv(
        args.report_dir / "noise_fp_postprocessed_frequency_bands.csv",
        post_band_rows,
        ["pipeline", "class", "band", "band_low_hz", "band_high_hz", "false_particles", "false_particle_pct"],
    )

    report = {
        "description": "Negative-control audit: every particles2SNR detection in Noise is a false positive.",
        "noise_dir": args.noise_dir,
        "raw_output": args.raw_output,
        "filtered_output": args.filtered_output,
        "classes": classes,
        "thresholds_db": thresholds_db,
        "summary": summary_rows,
        "reduction": reduction_rows,
        "postprocessed_summary": post_summary_rows,
        "postprocessed_reduction": post_reduction_rows,
        "worst_files": worst_rows,
        "procedure_notes": [
            "Raw pass uses standalone Noise files directly.",
            "Filtered pass applies the final 7-80 kHz Butterworth bandpass used by the _F pipeline.",
            "The same Noise files are evaluated independently under each class-specific FFT configuration.",
            "Postprocessed survivor counts reuse generate_particles2SNR_dataset.export_yolo_json defaults from the accepted _F pipeline.",
        ],
    }
    with (args.report_dir / "noise_fp_audit.json").open("w") as f:
        json.dump(report, f, indent=2, default=json_safe, allow_nan=False)
    write_markdown(args.report_dir / "noise_fp_audit.md", summary_rows, reduction_rows, thresholds_db)
    write_markdown(args.report_dir / "noise_fp_postprocessed_audit.md", post_summary_rows, post_reduction_rows, thresholds_db)
    plot_summary_png(args.report_dir / "noise_fp_audit.png", summary_rows, reduction_rows, classes)
    plot_summary_png(args.report_dir / "noise_fp_postprocessed_audit.png", post_summary_rows, post_reduction_rows, classes)
    plot_pdf(
        args.report_dir / "noise_fp_audit.pdf",
        file_rows,
        particle_rows,
        summary_rows,
        reduction_rows,
        threshold_rows,
        band_rows,
        worst_rows,
        classes,
    )
    plot_pdf(
        args.report_dir / "noise_fp_postprocessed_audit.pdf",
        post_file_rows,
        post_particle_rows,
        post_summary_rows,
        post_reduction_rows,
        post_threshold_rows,
        post_band_rows,
        post_worst_rows,
        classes,
    )

    print(f"Wrote Noise negative-control audit to {args.report_dir}")
    print(f"- {args.report_dir / 'noise_fp_audit.pdf'}")
    print(f"- {args.report_dir / 'noise_fp_summary.csv'}")
    print(f"- {args.report_dir / 'noise_fp_reduction.csv'}")


if __name__ == "__main__":
    main()
