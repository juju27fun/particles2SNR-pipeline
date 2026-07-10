"""Bench candidate fixes for the filtered (_F) Noise false positives.

The bench reuses the existing Noise negative-control raw and filtered detector
runs. It does not change the production pipeline; it exports experimental
post-processing variants so the false-positive effect of each hypothesis can be
compared on the same Noise files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import find_peaks, peak_widths

from generate_particles2SNR_dataset import (
    DEFAULT_BANDPASS_FMIN,
    DEFAULT_BANDPASS_FMAX,
    annotate_particle_peak_evidence,
    butter_bandpass_filter,
    detect_peak_groups,
    export_yolo_json,
    filter_annotations_by_yolo_width,
    filter_particles_by_yolo_width,
    keep_particle_by_passage_time,
    merge_overlapping_particles,
    moving_average_abs_envelope,
    particle_interval_samples,
    particle_to_annotation,
    refine_particles_with_peak_evidence,
    resolve_annotation_boundary_crossings,
    robust_z_values,
)
from repo_paths import RESULTS_REPORTS, RESULTS_RUNS
from run_dataset import get_config_for_folder


DEFAULT_CLASSES = ("2um", "4um", "10um")
DEFAULT_FS = 2_000_000.0
DEFAULT_MIN_PASSAGE_TIME_MS = 0.07
DEFAULT_MAX_PASSAGE_TIME_MS = 0.65
HIGH_SNR_THRESHOLD_DB = -10.0


DEFAULT_EXPORT_KWARGS = {
    "min_passage_time_ms": DEFAULT_MIN_PASSAGE_TIME_MS,
    "max_passage_time_ms": DEFAULT_MAX_PASSAGE_TIME_MS,
    "yolo_width_filter": True,
    "peak_evidence_filter": True,
    "merge_overlaps": True,
    "resolve_boundary_crossings": True,
}


BENCH_VARIANTS = (
    {
        "stage": "F_current",
        "option": "baseline",
        "label": "Current _F",
        "kind": "export",
        "source": "filtered",
        "kwargs": {},
    },
    {
        "stage": "F_peak_on_raw_clean",
        "option": "1_peak_signal",
        "label": "F detect, raw-clean peak",
        "kind": "path_swap_export",
        "source": "filtered",
        "peak_source": "raw",
        "kwargs": {},
    },
    {
        "stage": "F_dual_raw_and_filtered_peak",
        "option": "1_peak_signal",
        "label": "F detect, dual peak",
        "kind": "dual_peak",
        "source": "filtered",
        "kwargs": {},
    },
    {
        "stage": "F_strict_z5_prom3",
        "option": "2_strict_peak",
        "label": "F strict z5/prom3",
        "kind": "export",
        "source": "filtered",
        "kwargs": {"peak_min_z": 5.0, "peak_prominence_z": 3.0, "peak_group_valley_ratio": 0.65},
    },
    {
        "stage": "F_strict_z6_prom4",
        "option": "2_strict_peak",
        "label": "F strict z6/prom4",
        "kind": "export",
        "source": "filtered",
        "kwargs": {"peak_min_z": 6.0, "peak_prominence_z": 4.0, "peak_group_valley_ratio": 0.70},
    },
    {
        "stage": "F_strict_shape_spectral",
        "option": "2_strict_peak",
        "label": "F strict + shape/spectrum",
        "kind": "strict_shape",
        "source": "filtered",
        "kwargs": {"peak_min_z": 5.0, "peak_prominence_z": 3.0, "peak_group_valley_ratio": 0.65},
    },
    {
        "stage": "F_noise_like_guard",
        "option": "3_noise_guard",
        "label": "F file guard",
        "kind": "guard",
        "source": "filtered",
        "kwargs": {},
    },
    {
        "stage": "raw_detector_filtered_peak",
        "option": "4_bandpass_timing",
        "label": "Raw detect, filtered peak",
        "kind": "path_swap_export",
        "source": "raw",
        "peak_source": "filtered",
        "kwargs": {},
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


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_results(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def load_data_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def save_data_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, default=json_safe, allow_nan=False)


def signal_lookup(results: dict) -> dict[tuple[str, str], str]:
    lookup = {}
    for signal in results.get("signals", []):
        lookup[(signal.get("class"), signal.get("filename"))] = signal.get("path")
    return lookup


def write_path_swapped_results(source_results_path: Path, peak_lookup: dict[tuple[str, str], str], output_path: Path) -> None:
    results = load_results(source_results_path)
    for signal in results.get("signals", []):
        key = (signal.get("class"), signal.get("filename"))
        replacement = peak_lookup.get(key)
        if replacement:
            signal["path"] = replacement
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2, allow_nan=False)


def export_standard_variant(source_results_path: Path, output_path: Path, classes: tuple[str, ...], kwargs: dict) -> None:
    params = dict(DEFAULT_EXPORT_KWARGS)
    params.update(kwargs)
    export_yolo_json(
        source_results_path,
        output_path,
        class_names=classes,
        fs=DEFAULT_FS,
        **params,
    )


def filtered_temp_signal(raw_path: Path, class_name: str, temp_dir: Path) -> str:
    config = get_config_for_folder(class_name)
    target = temp_dir / class_name / raw_path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        signal = np.load(raw_path)
        filtered = butter_bandpass_filter(
            signal,
            fs=float(config.sampling_rate),
            fmin=float(config.bandpass_lowcut or DEFAULT_BANDPASS_FMIN),
            fmax=float(config.bandpass_highcut or DEFAULT_BANDPASS_FMAX),
        )
        np.save(target, filtered)
    return str(target)


def build_filtered_peak_lookup(raw_results: dict, temp_dir: Path) -> dict[tuple[str, str], str]:
    lookup = {}
    for signal in raw_results.get("signals", []):
        class_name = signal.get("class")
        filename = signal.get("filename")
        raw_path = signal.get("path")
        if not class_name or not filename or not raw_path:
            continue
        lookup[(class_name, filename)] = filtered_temp_signal(Path(raw_path), class_name, temp_dir)
    return lookup


def prefilter_particles(signal: dict, fs: float) -> tuple[list[dict], list[dict]]:
    length = int(signal.get("signal_length") or 0)
    kept = []
    dropped = []
    for particle_idx, particle in enumerate(signal.get("particles", [])):
        keep, reason, tau_ms = keep_particle_by_passage_time(
            particle,
            DEFAULT_MIN_PASSAGE_TIME_MS,
            DEFAULT_MAX_PASSAGE_TIME_MS,
        )
        if keep:
            out = dict(particle)
            out["_bench_particle_idx"] = int(particle_idx)
            kept.append(out)
        else:
            dropped.append({
                "reason": reason,
                "stage": "passage_time",
                "passage_time_ms": tau_ms,
                "snr_db": particle.get("snr_db"),
                "frequency": particle.get("frequency"),
            })
    kept, width_drops = filter_particles_by_yolo_width(
        kept,
        length,
        fs,
        min_width_ms=0.08,
        max_width_ms=1.5,
    )
    dropped.extend(width_drops)
    return kept, dropped


def group_shape_metrics(signal_values: np.ndarray, fs: float, params: dict) -> tuple[list[dict], np.ndarray]:
    envelope = moving_average_abs_envelope(signal_values, fs, 0.08)
    z_values, baseline, scale = robust_z_values(envelope)
    distance = max(1, int(round(0.18 / 1000.0 * fs)))
    peaks, properties = find_peaks(
        envelope,
        height=baseline + float(params.get("peak_min_z", 4.0)) * scale,
        prominence=float(params.get("peak_prominence_z", 2.0)) * scale,
        distance=distance,
    )
    groups = detect_peak_groups(
        signal_values,
        fs,
        envelope_window_ms=0.08,
        min_z=float(params.get("peak_min_z", 4.0)),
        prominence_z=float(params.get("peak_prominence_z", 2.0)),
        min_separation_ms=0.18,
        valley_ratio=float(params.get("peak_group_valley_ratio", 0.55)),
    )[0]
    if len(peaks):
        widths = peak_widths(envelope, peaks, rel_height=0.5)[0] / fs * 1000.0
        width_by_peak = {int(peak): float(width) for peak, width in zip(peaks, widths)}
        prominence_by_peak = {
            int(peak): float(prom / scale) if scale > 0 else 0.0
            for peak, prom in zip(peaks, properties.get("prominences", []))
        }
    else:
        width_by_peak = {}
        prominence_by_peak = {}
    for group in groups:
        peak = int(group["peak_sample"])
        group["peak_width_ms"] = width_by_peak.get(peak, 0.0)
        group["peak_prominence_z"] = prominence_by_peak.get(peak, 0.0)
    return groups, z_values


def spectral_isolation_ratio(signal_values: np.ndarray, left: float, right: float) -> float:
    lo = max(0, int(math.floor(left)))
    hi = min(len(signal_values), int(math.ceil(right)))
    if hi - lo < 16:
        return 1.0
    segment = np.asarray(signal_values[lo:hi], dtype=np.float64)
    segment = segment - float(np.mean(segment))
    window = np.hanning(len(segment))
    power = np.abs(np.fft.rfft(segment * window)) ** 2
    total = float(np.sum(power))
    if total <= 0.0:
        return 1.0
    return float(np.max(power) / total)


def periodicity_ratio(signal_values: np.ndarray, left: float, right: float) -> float:
    lo = max(0, int(math.floor(left)))
    hi = min(len(signal_values), int(math.ceil(right)))
    if hi - lo < 24:
        return 1.0
    segment = np.asarray(signal_values[lo:hi], dtype=np.float64)
    segment = segment - float(np.mean(segment))
    denom = float(np.dot(segment, segment))
    if denom <= 0.0:
        return 1.0
    corr = np.correlate(segment, segment, mode="full")[len(segment) - 1:]
    corr = corr / denom
    start = max(2, len(segment) // 20)
    if start >= len(corr):
        return 0.0
    return float(np.max(np.abs(corr[start:])))


def annotate_from_groups(particles: list[dict], signal_length: int, fs: float, groups: list[dict], z_values: np.ndarray) -> list[dict]:
    return [
        annotate_particle_peak_evidence(particle, signal_length, fs, groups, z_values)
        for particle in particles
    ]


def select_supported_particles(
    particles: list[dict],
    signal_values: np.ndarray,
    signal_length: int,
    fs: float,
    params: dict,
    shape_spectral: bool,
) -> tuple[list[dict], list[dict], list[dict]]:
    groups, z_values = group_shape_metrics(signal_values, fs, params)
    group_by_id = {int(group["id"]): group for group in groups}
    if shape_spectral:
        annotated, dropped, peak_summary = refine_particles_with_peak_evidence(
            particles,
            signal_values,
            signal_length,
            fs,
            envelope_window_ms=0.08,
            min_z=float(params.get("peak_min_z", 4.0)),
            prominence_z=float(params.get("peak_prominence_z", 2.0)),
            min_separation_ms=0.18,
            valley_ratio=float(params.get("peak_group_valley_ratio", 0.55)),
            cluster_gap_ms=0.25,
            keep_high_snr_db=4.0,
        )
        for item in peak_summary:
            group = group_by_id.get(int(item["id"]))
            if group:
                item["peak_width_ms"] = float(group.get("peak_width_ms", 0.0))
                item["peak_prominence_z"] = float(group.get("peak_prominence_z", 0.0))
    else:
        annotated = annotate_from_groups(particles, signal_length, fs, groups, z_values)
        dropped = []
        peak_summary = [{
            "id": int(group["id"]),
            "peak_center_ms": float(group["peak_sample"] / fs * 1000.0),
            "peak_z": float(group["peak_z"]),
            "peak_width_ms": float(group.get("peak_width_ms", 0.0)),
            "peak_prominence_z": float(group.get("peak_prominence_z", 0.0)),
            "num_peaks": int(len(group.get("peaks", []))),
        } for group in groups]
    kept = []
    for particle in annotated:
        left, right, _ = particle_interval_samples(particle, signal_length, fs)
        reason = None
        if not particle.get("peak_support"):
            reason = "no_peak_evidence"
        elif shape_spectral:
            group = group_by_id.get(int(particle.get("peak_group_id")))
            width_ms = float(group.get("peak_width_ms", 0.0)) if group else 0.0
            spectral_ratio = spectral_isolation_ratio(signal_values, left, right)
            periodic_ratio = periodicity_ratio(signal_values, left, right)
            particle["peak_width_ms"] = width_ms
            particle["spectral_isolation_ratio"] = spectral_ratio
            particle["periodicity_ratio"] = periodic_ratio
            if width_ms < 0.025:
                reason = "peak_envelope_too_narrow"
            elif width_ms > 0.55:
                reason = "peak_envelope_too_wide"
            elif spectral_ratio > 0.42:
                reason = "spectrally_isolated_peak"
            elif periodic_ratio > 0.78:
                reason = "periodic_peak"
        if reason is None:
            kept.append(particle)
        else:
            dropped.append({
                "reason": reason,
                "stage": "peak_evidence",
                "snr_db": particle.get("snr_db"),
                "frequency": particle.get("frequency"),
                "passage_time_ms": particle.get("tau"),
                "peak_z": particle.get("peak_z"),
                "local_peak_z": particle.get("local_peak_z"),
            })
    return kept, dropped, peak_summary


def export_custom_variant(
    source_results_path: Path,
    output_path: Path,
    classes: tuple[str, ...],
    raw_lookup: dict[tuple[str, str], str],
    filtered_lookup: dict[tuple[str, str], str],
    mode: str,
    kwargs: dict,
) -> None:
    results = load_results(source_results_path)
    class_to_id = {class_name: idx for idx, class_name in enumerate(classes)}
    data_rows = []
    for sample_id, signal in enumerate(results.get("signals", [])):
        class_name = signal.get("class")
        class_id = class_to_id.get(class_name)
        if class_id is None:
            continue
        key = (class_name, signal.get("filename"))
        length = int(signal.get("signal_length") or 0)
        particles, drops = prefilter_particles(signal, DEFAULT_FS)
        peak_summary = []
        if mode == "dual_peak":
            filtered_values = np.load(filtered_lookup[key])
            raw_values = np.load(raw_lookup[key])
            filtered_particles, filtered_drops, filtered_summary = select_supported_particles(
                particles,
                filtered_values,
                length,
                DEFAULT_FS,
                kwargs,
                shape_spectral=False,
            )
            raw_particles, raw_drops, raw_summary = select_supported_particles(
                particles,
                raw_values,
                length,
                DEFAULT_FS,
                kwargs,
                shape_spectral=False,
            )
            raw_supported = {int(particle["_bench_particle_idx"]) for particle in raw_particles}
            kept = []
            for particle in filtered_particles:
                if int(particle["_bench_particle_idx"]) in raw_supported:
                    particle["dual_peak_support"] = True
                    kept.append(particle)
                else:
                    drops.append({
                        "reason": "missing_raw_clean_peak_support",
                        "stage": "dual_peak_evidence",
                        "snr_db": particle.get("snr_db"),
                        "frequency": particle.get("frequency"),
                    })
            drops.extend(filtered_drops)
            drops.extend([{**drop, "stage": "raw_peak_evidence"} for drop in raw_drops])
            particles = kept
            peak_summary = filtered_summary
            for item in peak_summary:
                item["dual_mode"] = "filtered_summary_raw_support_required"
                item["raw_peak_group_count"] = len(raw_summary)
        elif mode == "strict_shape":
            signal_values = np.load(filtered_lookup[key])
            particles, peak_drops, peak_summary = select_supported_particles(
                particles,
                signal_values,
                length,
                DEFAULT_FS,
                kwargs,
                shape_spectral=True,
            )
            drops.extend(peak_drops)
        else:
            raise ValueError(mode)

        particles, nms_drops = merge_overlapping_particles(particles, length, DEFAULT_FS)
        drops.extend(nms_drops)
        annotations = [
            particle_to_annotation(particle, length, DEFAULT_FS, class_id, ann_id)
            for ann_id, particle in enumerate(particles)
        ]
        annotations, annotation_width_drops = filter_annotations_by_yolo_width(
            annotations,
            length,
            DEFAULT_FS,
            min_width_ms=0.08,
            max_width_ms=1.5,
        )
        drops.extend([{**drop, "stage": "post_annotation"} for drop in annotation_width_drops])
        annotations, boundary_drops, boundary_adjustments = resolve_annotation_boundary_crossings(
            annotations,
            length,
            DEFAULT_FS,
            min_width_ms=0.08,
        )
        drops.extend(boundary_drops)
        data_rows.append({
            "filename": signal.get("filename"),
            "path": signal.get("path"),
            "id": int(sample_id),
            "class_id": int(class_id),
            "class_name": class_name,
            "length": length,
            "annotations": annotations,
            "dropped_annotations": drops,
            "peak_groups": peak_summary,
            "boundary_adjustments": boundary_adjustments,
        })
    save_data_json(
        output_path,
        {
            "info": {
                "description": f"Experimental Noise negative-control _F bench variant: {mode}",
                "source_results": str(source_results_path),
                "mode": mode,
                "parameters": kwargs,
            },
            "data": data_rows,
        },
    )


def apply_noise_like_guard(input_path: Path, output_path: Path, raw_reference_path: Path) -> dict:
    data = load_data_json(input_path)
    raw_reference = load_data_json(raw_reference_path)
    raw_counts = {}
    for class_name in DEFAULT_CLASSES:
        counts = [
            len(row.get("annotations", []))
            for row in raw_reference.get("data", [])
            if row.get("class_name") == class_name
        ]
        raw_counts[class_name] = {
            "p99": float(np.percentile(counts, 99)) if counts else 0.0,
            "max": int(max(counts)) if counts else 0,
        }
    suspect_rows = 0
    suspect_annotations = 0
    for row in data.get("data", []):
        class_name = row.get("class_name")
        threshold = max(2, int(math.ceil(raw_counts.get(class_name, {}).get("p99", 0.0))))
        annotations = row.get("annotations", [])
        row["noise_like_guard"] = {
            "threshold_annotations_per_file": threshold,
            "suspect": len(annotations) > threshold,
            "raw_reference_p99": raw_counts.get(class_name, {}).get("p99", 0.0),
        }
        if len(annotations) > threshold:
            suspect_rows += 1
            suspect_annotations += len(annotations)
            row.setdefault("dropped_annotations", []).extend([
                {
                    "reason": "noise_like_file_guard",
                    "stage": "file_guard",
                    "threshold_annotations_per_file": threshold,
                    "annotation_id": ann.get("id"),
                    "snr_db": ann.get("snr_db"),
                    "frequency": ann.get("frequency"),
                }
                for ann in annotations
            ])
            row["annotations"] = []
    data.setdefault("info", {})["noise_like_guard"] = {
        "raw_reference": str(raw_reference_path),
        "rule": "clear annotations for files above max(2, ceil(raw default p99)) per class",
        "suspect_files": suspect_rows,
        "cleared_annotations": suspect_annotations,
    }
    save_data_json(output_path, data)
    return {"suspect_files": suspect_rows, "cleared_annotations": suspect_annotations}


def summarize_data_json(path: Path, stage: str, option: str, label: str, guard_stats: dict | None = None) -> list[dict]:
    data = load_data_json(path)
    rows = []
    for class_name in DEFAULT_CLASSES:
        samples = [row for row in data.get("data", []) if row.get("class_name") == class_name]
        counts = [len(row.get("annotations", [])) for row in samples]
        snrs = []
        for row in samples:
            for ann in row.get("annotations", []):
                snr = as_float(ann.get("snr_db"))
                if snr is not None:
                    snrs.append(snr)
        arr = np.asarray(counts, dtype=float)
        n_files = len(samples)
        suspicious = sum(1 for row in samples if row.get("noise_like_guard", {}).get("suspect"))
        rows.append({
            "stage": stage,
            "option": option,
            "label": label,
            "class": class_name,
            "n_files": n_files,
            "total_false_particles": int(np.sum(arr)) if len(arr) else 0,
            "mean_false_particles_per_file": float(np.mean(arr)) if len(arr) else math.nan,
            "median_false_particles_per_file": float(np.median(arr)) if len(arr) else math.nan,
            "p90_false_particles_per_file": float(np.percentile(arr, 90)) if len(arr) else math.nan,
            "max_false_particles_per_file": float(np.max(arr)) if len(arr) else math.nan,
            "false_particles_snr_ge_neg10db": int(sum(1 for value in snrs if value >= HIGH_SNR_THRESHOLD_DB)),
            "false_particles_snr_ge_neg10db_per_file": (
                sum(1 for value in snrs if value >= HIGH_SNR_THRESHOLD_DB) / n_files if n_files else math.nan
            ),
            "median_snr_db": float(np.median(snrs)) if snrs else math.nan,
            "p90_snr_db": float(np.percentile(snrs, 90)) if snrs else math.nan,
            "max_snr_db": float(np.max(snrs)) if snrs else math.nan,
            "suspect_files": suspicious if suspicious else (guard_stats or {}).get("suspect_files", 0),
            "suspect_file_pct": suspicious / n_files * 100.0 if n_files else 0.0,
        })
    return rows


def build_comparison_rows(summary_rows: list[dict]) -> list[dict]:
    baseline = {
        row["class"]: row
        for row in summary_rows
        if row["stage"] == "F_current"
    }
    out = []
    for row in summary_rows:
        base = baseline.get(row["class"])
        if not base:
            continue
        mean = row["mean_false_particles_per_file"]
        base_mean = base["mean_false_particles_per_file"]
        out.append({
            "stage": row["stage"],
            "option": row["option"],
            "label": row["label"],
            "class": row["class"],
            "baseline_F_mean_fp_per_file": base_mean,
            "variant_mean_fp_per_file": mean,
            "delta_vs_F_fp_per_file": mean - base_mean,
            "reduction_vs_F_percent": (1.0 - mean / base_mean) * 100.0 if base_mean > 0 else math.nan,
            "variant_high_snr_fp_per_file": row["false_particles_snr_ge_neg10db_per_file"],
            "suspect_file_pct": row["suspect_file_pct"],
        })
    return out


def plot_bench(summary_rows: list[dict], output_base: Path) -> None:
    stages = [variant["stage"] for variant in BENCH_VARIANTS]
    labels = {variant["stage"]: variant["label"] for variant in BENCH_VARIANTS}
    options = {variant["stage"]: variant["option"] for variant in BENCH_VARIANTS}
    classes = DEFAULT_CLASSES
    colors = {
        "baseline": "#4c72b0",
        "1_peak_signal": "#55a868",
        "2_strict_peak": "#8172b2",
        "3_noise_guard": "#c44e52",
        "4_bandpass_timing": "#ccb974",
    }
    lookup = {(row["stage"], row["class"]): row for row in summary_rows}
    fig, axes = plt.subplots(1, len(classes), figsize=(17, 5.5), sharey=True)
    x = np.arange(len(stages))
    for ax, class_name in zip(axes, classes):
        values = [lookup[(stage, class_name)]["mean_false_particles_per_file"] for stage in stages]
        bar_colors = [colors[options[stage]] for stage in stages]
        ax.bar(x, values, color=bar_colors)
        ax.axhline(0.5, color="black", linestyle="--", linewidth=0.9, alpha=0.6)
        ax.set_title(class_name)
        ax.set_xticks(x)
        ax.set_xticklabels([labels[stage] for stage in stages], rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("False positives / Noise file")
    fig.suptitle("Noise negative-control _F fix bench", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_base.with_suffix(".png"), dpi=220)
    with PdfPages(output_base.with_suffix(".pdf")) as pdf:
        pdf.savefig(fig)
    plt.close(fig)


def write_markdown(path: Path, summary_rows: list[dict], comparison_rows: list[dict]) -> None:
    means = {}
    for row in summary_rows:
        means.setdefault(row["stage"], []).append(row["mean_false_particles_per_file"])
    ranked = sorted((float(np.mean(values)), stage) for stage, values in means.items())
    label_by_stage = {variant["stage"]: variant["label"] for variant in BENCH_VARIANTS}
    option_by_stage = {variant["stage"]: variant["option"] for variant in BENCH_VARIANTS}
    lines = [
        "# Noise Negative-Control _F Bench",
        "",
        "All annotations on Noise are false positives. Lower FP/file is better.",
        "The dashed reference line in the figure is ~0.5 FP/file, matching the raw post-processed baseline observed in the audit.",
        "",
        "## Ranked variants",
        "",
        "| rank | option | variant | mean FP/file across classes |",
        "|---:|---|---|---:|",
    ]
    for rank, (mean_value, stage) in enumerate(ranked, start=1):
        lines.append(f"| {rank} | {option_by_stage[stage]} | {label_by_stage[stage]} | {mean_value:.3f} |")
    lines.extend([
        "",
        "## Interpretation notes",
        "",
        "- `F_peak_on_raw_clean` keeps filtered detections but validates peak evidence on the non-bandpassed Noise signal.",
        "- `F_dual_raw_and_filtered_peak` requires support in both filtered and non-bandpassed signals.",
        "- strict variants keep the current filtered-signal evidence but tighten peak z/prominence/valley; the shape/spectrum variant also rejects narrow/wide, periodic, or spectrally isolated envelope peaks.",
        "- `F_noise_like_guard` is a procedural guardrail, not a clean physical criterion: files above a raw-reference p99 count threshold are marked suspect and cleared.",
        "- `raw_detector_filtered_peak` tests the bandpass timing hypothesis by detecting before bandpass and validating peak evidence after bandpass.",
        "",
        "## Key CSVs",
        "",
        f"- `{path.parent / 'noise_F_bench_summary.csv'}`",
        f"- `{path.parent / 'noise_F_bench_vs_current.csv'}`",
    ])
    path.write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bench candidate _F fixes on the Noise negative control.")
    parser.add_argument("--raw-output", type=Path, default=RESULTS_RUNS / "noise_negative_control_raw")
    parser.add_argument("--filtered-output", type=Path, default=RESULTS_RUNS / "noise_negative_control_F")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_REPORTS / "noise_negative_control_F_bench")
    parser.add_argument("--classes", type=parse_csv_arg, default=DEFAULT_CLASSES)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    classes = tuple(args.classes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    variant_dir = args.output_dir / "variant_data"
    raw_results_path = args.raw_output / "dataset_results.json"
    filtered_results_path = args.filtered_output / "dataset_results.json"
    raw_results = load_results(raw_results_path)
    filtered_results = load_results(filtered_results_path)
    raw_lookup = signal_lookup(raw_results)
    filtered_lookup = signal_lookup(filtered_results)

    summary_rows = []
    guard_stats = {}
    with tempfile.TemporaryDirectory(prefix="noise_f_bench_") as temp_name:
        temp_dir = Path(temp_name)
        generated_filtered_lookup = build_filtered_peak_lookup(raw_results, temp_dir)
        for variant in BENCH_VARIANTS:
            stage = variant["stage"]
            output_path = variant_dir / f"{stage}.json"
            kind = variant["kind"]
            source_results_path = filtered_results_path if variant["source"] == "filtered" else raw_results_path
            kwargs = dict(variant.get("kwargs", {}))
            if kind == "export":
                export_standard_variant(source_results_path, output_path, classes, kwargs)
            elif kind == "path_swap_export":
                if variant["peak_source"] == "raw":
                    peak_lookup = raw_lookup
                elif variant["peak_source"] == "filtered":
                    peak_lookup = filtered_lookup if variant["source"] == "filtered" else generated_filtered_lookup
                else:
                    raise ValueError(variant["peak_source"])
                swapped_path = variant_dir / "swapped_results" / f"{stage}_dataset_results.json"
                write_path_swapped_results(source_results_path, peak_lookup, swapped_path)
                export_standard_variant(swapped_path, output_path, classes, kwargs)
            elif kind in {"dual_peak", "strict_shape"}:
                export_custom_variant(
                    source_results_path,
                    output_path,
                    classes,
                    raw_lookup=raw_lookup,
                    filtered_lookup=filtered_lookup,
                    mode=kind,
                    kwargs=kwargs,
                )
            elif kind == "guard":
                current_path = variant_dir / "F_current.json"
                if not current_path.exists():
                    export_standard_variant(filtered_results_path, current_path, classes, {})
                raw_default_path = variant_dir / "raw_default_reference.json"
                export_standard_variant(raw_results_path, raw_default_path, classes, {})
                guard_stats = apply_noise_like_guard(current_path, output_path, raw_default_path)
            else:
                raise ValueError(kind)
            summary_rows.extend(
                summarize_data_json(
                    output_path,
                    stage,
                    variant["option"],
                    variant["label"],
                    guard_stats=guard_stats if kind == "guard" else None,
                )
            )

    comparison_rows = build_comparison_rows(summary_rows)
    write_csv(
        args.output_dir / "noise_F_bench_summary.csv",
        summary_rows,
        [
            "stage", "option", "label", "class", "n_files",
            "total_false_particles", "mean_false_particles_per_file",
            "median_false_particles_per_file", "p90_false_particles_per_file",
            "max_false_particles_per_file", "false_particles_snr_ge_neg10db",
            "false_particles_snr_ge_neg10db_per_file", "median_snr_db",
            "p90_snr_db", "max_snr_db", "suspect_files", "suspect_file_pct",
        ],
    )
    write_csv(
        args.output_dir / "noise_F_bench_vs_current.csv",
        comparison_rows,
        [
            "stage", "option", "label", "class", "baseline_F_mean_fp_per_file",
            "variant_mean_fp_per_file", "delta_vs_F_fp_per_file",
            "reduction_vs_F_percent", "variant_high_snr_fp_per_file",
            "suspect_file_pct",
        ],
    )
    with (args.output_dir / "noise_F_bench_summary.json").open("w") as f:
        json.dump(
            {
                "description": "Candidate _F fixes benchmarked on the Noise negative control.",
                "variants": BENCH_VARIANTS,
                "summary": summary_rows,
                "vs_current": comparison_rows,
            },
            f,
            indent=2,
            default=json_safe,
            allow_nan=False,
        )
    plot_bench(summary_rows, args.output_dir / "noise_F_bench")
    write_markdown(args.output_dir / "noise_F_bench.md", summary_rows, comparison_rows)
    print(f"Wrote _F bench report to {args.output_dir}")
    print(f"- {args.output_dir / 'noise_F_bench_summary.csv'}")
    print(f"- {args.output_dir / 'noise_F_bench_vs_current.csv'}")
    print(f"- {args.output_dir / 'noise_F_bench.pdf'}")


if __name__ == "__main__":
    main()
