"""Spectral comparison for YOLO detection datasets with explicit labels.

The report separates:
- particle-labelled intervals from YOLO labels,
- dataset noise windows outside every label,
- standalone Noise-folder windows.

It is designed for comparing detection datasets without using particles2SNR
detections to define the noise masks.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy import signal as scipy_signal


BANDS_HZ = (
    ("0-1 kHz", 0.0, 1_000.0),
    ("1-7 kHz", 1_000.0, 7_000.0),
    ("7-10 kHz", 7_000.0, 10_000.0),
    ("10-40 kHz", 10_000.0, 40_000.0),
    ("40-80 kHz", 40_000.0, 80_000.0),
    ("80 kHz-Nyq", 80_000.0, math.inf),
)
PLOT_BAND_LABELS = ("0-1", "1-7", "7-10", "10-40", "40-80", "80-Nyq")

DEFAULT_CLASSES = ("2um", "4um", "10um")
SAMPLING_RATE_HZ = 2_000_000.0
REFERENCE_KEY = "noise_reference"
DEFAULT_OLD_LABEL = "YOLO v3 (old pipeline)"
DEFAULT_NEW_LABEL = "C1 particles2SNR (refined labels)"
REFERENCE_LABEL = "Standalone Noise reference"
CLASS_FFT_CONFIGS = {
    "2um": {"fft_window_length": 4096, "fft_stride": 512},
    "4um": {"fft_window_length": 4096, "fft_stride": 1024},
    "10um": {"fft_window_length": 4096, "fft_stride": 1024},
}
DEFAULT_FFT_CONFIG = {"fft_window_length": 4096, "fft_stride": 1024}


def parse_csv_arg(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_khz_range(value: str) -> tuple[float, float]:
    parts = [item.strip() for item in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected LOW,HIGH in kHz")
    low, high = float(parts[0]) * 1000.0, float(parts[1]) * 1000.0
    if low >= high:
        raise argparse.ArgumentTypeError("LOW must be below HIGH")
    return low, high


def format_khz_band(band_hz: tuple[float, float] | None) -> str:
    if band_hz is None:
        return "none"
    return f"{band_hz[0] / 1000:g}-{band_hz[1] / 1000:g} kHz"


def bandpass_filter_signal(signal: np.ndarray, fs: float, band_hz: tuple[float, float], order: int = 4) -> np.ndarray:
    arr = np.asarray(signal, dtype=float).squeeze()
    low, high = band_hz
    sos = scipy_signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    try:
        return scipy_signal.sosfiltfilt(sos, arr)
    except ValueError:
        return scipy_signal.sosfilt(sos, arr)


def apply_analysis_preprocess(signal: np.ndarray, analysis_bandpass_hz: tuple[float, float] | None) -> np.ndarray:
    arr = np.asarray(signal, dtype=float).squeeze()
    if analysis_bandpass_hz is None:
        return arr
    return bandpass_filter_signal(arr, SAMPLING_RATE_HZ, analysis_bandpass_hz)


def parse_dataset_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected LABEL=PATH")
    label, path = value.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise argparse.ArgumentTypeError("Expected non-empty LABEL=PATH")
    return label, Path(path)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fft_config(class_name: str) -> dict:
    return dict(CLASS_FFT_CONFIGS.get(class_name, DEFAULT_FFT_CONFIG))


def iter_split_signal_paths(dataset_root: Path, splits: tuple[str, ...]) -> list[Path]:
    paths = []
    for split in splits:
        signals_dir = dataset_root / split / "signals"
        if not signals_dir.is_dir():
            raise FileNotFoundError(f"Missing signals directory: {signals_dir}")
        paths.extend(sorted(signals_dir.glob("*.npy")))
    return paths


def label_path_for_signal(signal_path: Path) -> Path:
    return signal_path.parent.parent / "labels" / f"{signal_path.stem}.txt"


def read_yolo_labels(label_path: Path, signal_length: int, guard_samples: int) -> list[dict]:
    if not label_path.is_file():
        return []
    labels = []
    with label_path.open() as f:
        for line_number, line in enumerate(f, start=1):
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                class_id = int(float(parts[0]))
                center = float(parts[1])
                width = float(parts[2])
            except ValueError as exc:
                raise ValueError(f"Bad label in {label_path}:{line_number}: {line.strip()}") from exc
            start = int(math.floor((center - width / 2.0) * signal_length)) - guard_samples
            end = int(math.ceil((center + width / 2.0) * signal_length)) + guard_samples
            start = max(0, start)
            end = min(signal_length, end)
            if end <= start:
                continue
            labels.append({
                "class_id": class_id,
                "center": center,
                "width": width,
                "start": start,
                "end": end,
            })
    return labels


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def coverage_fraction(labels: list[dict], signal_length: int) -> float:
    intervals = merge_intervals([(label["start"], label["end"]) for label in labels])
    covered = sum(end - start for start, end in intervals)
    return covered / signal_length if signal_length > 0 else math.nan


def band_metrics_from_segment(segment: np.ndarray, fs: float) -> tuple[dict[str, float], dict[str, float]] | None:
    arr = np.asarray(segment, dtype=float).squeeze()
    if arr.ndim != 1 or len(arr) < 8:
        return None
    arr = arr - float(np.mean(arr))
    tapered = arr * np.hamming(len(arr))
    freqs = np.fft.rfftfreq(len(tapered), d=1.0 / fs)
    energy = np.abs(np.fft.rfft(tapered)) ** 2
    band_totals = {}
    for label, low, high in BANDS_HZ:
        if math.isinf(high):
            mask = freqs >= low
        else:
            mask = (freqs >= low) & (freqs < high)
        band_totals[label] = float(np.sum(energy[mask]))
    total = sum(band_totals.values())
    if total <= 0:
        return None

    # Normalize by segment length so variable-width particle labels are comparable
    # to fixed-width noise windows in absolute-energy summaries.
    denom = float(len(tapered) ** 2)
    pct = {label: value / total * 100.0 for label, value in band_totals.items()}
    energy_db = {label: 10.0 * math.log10(max(value / denom, 1e-30)) for label, value in band_totals.items()}
    return pct, energy_db


def band_distribution_from_segment(segment: np.ndarray, fs: float) -> dict[str, float] | None:
    metrics = band_metrics_from_segment(segment, fs)
    return metrics[0] if metrics is not None else None


def band_metrics_from_windows(
    signal: np.ndarray,
    fs: float,
    fft_len: int,
    stride: int,
    blocked_intervals: list[tuple[int, int]],
) -> tuple[list[dict[str, float]], list[dict[str, float]], int]:
    arr = np.asarray(signal, dtype=float).squeeze()
    if arr.ndim != 1 or len(arr) < fft_len:
        return [], [], 0
    blocked = np.zeros(len(arr), dtype=bool)
    for start, end in blocked_intervals:
        blocked[start:end] = True
    dists = []
    energy_dbs = []
    n_windows = 0
    for start in range(0, len(arr) - fft_len + 1, stride):
        end = start + fft_len
        if blocked[start:end].any():
            continue
        metrics = band_metrics_from_segment(arr[start:end], fs)
        if metrics is not None:
            dist, energy_db = metrics
            dists.append(dist)
            energy_dbs.append(energy_db)
        n_windows += 1
    return dists, energy_dbs, n_windows


def band_distribution_from_windows(
    signal: np.ndarray,
    fs: float,
    fft_len: int,
    stride: int,
    blocked_intervals: list[tuple[int, int]],
) -> tuple[list[dict[str, float]], int]:
    dists, _energy_dbs, n_windows = band_metrics_from_windows(signal, fs, fft_len, stride, blocked_intervals)
    return dists, n_windows

def summarize_distributions(distributions: list[dict[str, float]]) -> dict[str, dict]:
    out = {}
    for label, _, _ in BANDS_HZ:
        values = np.asarray(
            [dist[label] for dist in distributions if math.isfinite(dist.get(label, math.nan))],
            dtype=float,
        )
        out[label] = {
            "mean_pct": float(np.mean(values)) if len(values) else math.nan,
            "std_pct": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if len(values) == 1 else math.nan,
            "n": int(len(values)),
        }
    return out


def summarize_energy_db(energy_rows: list[dict[str, float]]) -> dict[str, dict]:
    out = {}
    for label, _, _ in BANDS_HZ:
        values = np.asarray(
            [row[label] for row in energy_rows if math.isfinite(row.get(label, math.nan))],
            dtype=float,
        )
        out[label] = {
            "mean_energy_db": float(np.mean(values)) if len(values) else math.nan,
            "median_energy_db": float(np.median(values)) if len(values) else math.nan,
            "std_energy_db": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if len(values) == 1 else math.nan,
            "n": int(len(values)),
        }
    return out


def analyze_dataset(
    dataset_root: Path,
    pipeline_name: str,
    classes: tuple[str, ...],
    splits: tuple[str, ...],
    guard_samples: int,
    analysis_bandpass_hz: tuple[float, float] | None,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    class_by_id = {idx: class_name for idx, class_name in enumerate(classes)}
    particle_dists = {class_name: [] for class_name in classes}
    particle_energy = {class_name: [] for class_name in classes}
    noise_dists = {class_name: [] for class_name in classes}
    noise_energy = {class_name: [] for class_name in classes}
    noise_window_counts = defaultdict(int)
    coverage_rows = []

    for signal_path in iter_split_signal_paths(dataset_root, splits):
        signal = np.load(signal_path)
        signal = apply_analysis_preprocess(signal, analysis_bandpass_hz)
        signal = np.asarray(signal).squeeze()
        if signal.ndim != 1:
            raise ValueError(f"Expected 1D signal in {signal_path}, got {signal.shape}")
        labels = read_yolo_labels(label_path_for_signal(signal_path), len(signal), guard_samples)
        all_intervals = [(label["start"], label["end"]) for label in labels]
        merged = merge_intervals(all_intervals)
        coverage_rows.append({
            "pipeline": pipeline_name,
            "split": signal_path.parent.parent.name,
            "filename": signal_path.name,
            "signal_length": len(signal),
            "n_labels": len(labels),
            "coverage_pct": coverage_fraction(labels, len(signal)) * 100.0,
        })

        for label in labels:
            class_name = class_by_id.get(label["class_id"])
            if class_name is None:
                continue
            metrics = band_metrics_from_segment(signal[label["start"]:label["end"]], SAMPLING_RATE_HZ)
            if metrics is not None:
                dist, energy_db = metrics
                particle_dists[class_name].append(dist)
                particle_energy[class_name].append(energy_db)

        for class_name in classes:
            cfg = fft_config(class_name)
            dists, energy_dbs, n_windows = band_metrics_from_windows(
                signal,
                SAMPLING_RATE_HZ,
                int(cfg["fft_window_length"]),
                int(cfg["fft_stride"]),
                merged,
            )
            noise_dists[class_name].extend(dists)
            noise_energy[class_name].extend(energy_dbs)
            noise_window_counts[class_name] += n_windows

    band_rows = []
    energy_rows = []
    plot_data = {}
    for class_name in classes:
        particle_summary = summarize_distributions(particle_dists[class_name])
        noise_summary = summarize_distributions(noise_dists[class_name])
        particle_energy_summary = summarize_energy_db(particle_energy[class_name])
        noise_energy_summary = summarize_energy_db(noise_energy[class_name])
        plot_data[class_name] = {
            "particle_labels": {label: particle_summary[label]["mean_pct"] for label, _, _ in BANDS_HZ},
            "dataset_noise_windows": {label: noise_summary[label]["mean_pct"] for label, _, _ in BANDS_HZ},
            "particle_labels_energy_db": {label: particle_energy_summary[label]["mean_energy_db"] for label, _, _ in BANDS_HZ},
            "dataset_noise_windows_energy_db": {label: noise_energy_summary[label]["mean_energy_db"] for label, _, _ in BANDS_HZ},
        }
        for source, summary, energy_summary, n_windows in (
            ("particle_labels", particle_summary, particle_energy_summary, len(particle_dists[class_name])),
            ("dataset_noise_windows", noise_summary, noise_energy_summary, noise_window_counts[class_name]),
        ):
            for label, low, high in BANDS_HZ:
                band_rows.append({
                    "pipeline": pipeline_name,
                    "class": class_name,
                    "source": source,
                    "band": label,
                    "band_low_hz": low,
                    "band_high_hz": "" if math.isinf(high) else high,
                    "mean_pct": summary[label]["mean_pct"],
                    "std_pct": summary[label]["std_pct"],
                    "n_segments_or_windows": summary[label]["n"],
                    "n_candidate_windows": n_windows if source == "dataset_noise_windows" else "",
                })
                energy_rows.append({
                    "pipeline": pipeline_name,
                    "class": class_name,
                    "source": source,
                    "band": label,
                    "band_low_hz": low,
                    "band_high_hz": "" if math.isinf(high) else high,
                    "mean_energy_db": energy_summary[label]["mean_energy_db"],
                    "median_energy_db": energy_summary[label]["median_energy_db"],
                    "std_energy_db": energy_summary[label]["std_energy_db"],
                    "n_segments_or_windows": energy_summary[label]["n"],
                })
    return band_rows, coverage_rows, energy_rows, plot_data

def analyze_standalone_noise(
    noise_dir: Path,
    classes: tuple[str, ...],
    analysis_bandpass_hz: tuple[float, float] | None,
) -> tuple[list[dict], list[dict], dict]:
    if not noise_dir.is_dir():
        raise FileNotFoundError(f"Missing Noise directory: {noise_dir}")
    band_rows = []
    energy_rows = []
    plot_data = {}
    files = sorted(noise_dir.glob("*.npy"))
    for class_name in classes:
        cfg = fft_config(class_name)
        dists = []
        energy_dbs = []
        n_windows_total = 0
        for path in files:
            signal = np.load(path)
            signal = apply_analysis_preprocess(signal, analysis_bandpass_hz)
            file_dists, file_energy_dbs, n_windows = band_metrics_from_windows(
                signal,
                SAMPLING_RATE_HZ,
                int(cfg["fft_window_length"]),
                int(cfg["fft_stride"]),
                [],
            )
            dists.extend(file_dists)
            energy_dbs.extend(file_energy_dbs)
            n_windows_total += n_windows
        summary = summarize_distributions(dists)
        energy_summary = summarize_energy_db(energy_dbs)
        plot_data[class_name] = {
            "standalone_noise_folder": {label: summary[label]["mean_pct"] for label, _, _ in BANDS_HZ},
            "standalone_noise_folder_energy_db": {label: energy_summary[label]["mean_energy_db"] for label, _, _ in BANDS_HZ},
        }
        for label, low, high in BANDS_HZ:
            band_rows.append({
                "pipeline": REFERENCE_KEY,
                "class": class_name,
                "source": "standalone_noise_folder",
                "band": label,
                "band_low_hz": low,
                "band_high_hz": "" if math.isinf(high) else high,
                "mean_pct": summary[label]["mean_pct"],
                "std_pct": summary[label]["std_pct"],
                "n_segments_or_windows": summary[label]["n"],
                "n_candidate_windows": n_windows_total,
            })
            energy_rows.append({
                "pipeline": REFERENCE_KEY,
                "class": class_name,
                "source": "standalone_noise_folder",
                "band": label,
                "band_low_hz": low,
                "band_high_hz": "" if math.isinf(high) else high,
                "mean_energy_db": energy_summary[label]["mean_energy_db"],
                "median_energy_db": energy_summary[label]["median_energy_db"],
                "std_energy_db": energy_summary[label]["std_energy_db"],
                "n_segments_or_windows": energy_summary[label]["n"],
            })
    return band_rows, energy_rows, plot_data

def dominant_band(distribution: dict[str, float]) -> str:
    clean = {key: value for key, value in distribution.items() if math.isfinite(value)}
    if not clean:
        return ""
    return max(clean, key=clean.get)


def sum_range(distribution: dict[str, float], low_hz: float, high_hz: float) -> float:
    total = 0.0
    seen = False
    for label, low, high in BANDS_HZ:
        band_high = high if math.isfinite(high) else math.inf
        if low < high_hz and band_high > low_hz:
            value = distribution.get(label, math.nan)
            if math.isfinite(value):
                total += value
                seen = True
    return total if seen else math.nan


def overlap_pct(left: dict[str, float], right: dict[str, float]) -> float:
    values = []
    for label, _, _ in BANDS_HZ:
        a = left.get(label, math.nan)
        b = right.get(label, math.nan)
        if math.isfinite(a) and math.isfinite(b):
            values.append(min(a, b))
    return float(sum(values)) if values else math.nan


def build_overlap_rows(plot_data: dict, classes: tuple[str, ...], doppler_band_hz: tuple[float, float]) -> list[dict]:
    rows = []
    doppler_label = f"{doppler_band_hz[0] / 1000:g}-{doppler_band_hz[1] / 1000:g} kHz"
    for pipeline_name, by_class in plot_data.items():
        if pipeline_name == REFERENCE_KEY:
            continue
        for class_name in classes:
            particle = by_class[class_name]["particle_labels"]
            dataset_noise = by_class[class_name]["dataset_noise_windows"]
            standalone = by_class[class_name]["standalone_noise_folder"]
            particle_dom = dominant_band(particle)
            noise_dom = dominant_band(dataset_noise)
            rows.append({
                "pipeline": pipeline_name,
                "class": class_name,
                "doppler_band": doppler_label,
                "particle_label_doppler_band_energy_pct": sum_range(particle, *doppler_band_hz),
                "dataset_noise_doppler_band_energy_pct": sum_range(dataset_noise, *doppler_band_hz),
                "standalone_noise_doppler_band_energy_pct": sum_range(standalone, *doppler_band_hz),
                "delta_dataset_noise_minus_standalone_pct": (
                    sum_range(dataset_noise, *doppler_band_hz) - sum_range(standalone, *doppler_band_hz)
                    if math.isfinite(sum_range(dataset_noise, *doppler_band_hz))
                    and math.isfinite(sum_range(standalone, *doppler_band_hz))
                    else math.nan
                ),
                "particle_dominant_band": particle_dom,
                "dataset_noise_dominant_band": noise_dom,
                "dominant_band_overlap": bool(particle_dom and particle_dom == noise_dom),
                "particle_dataset_noise_distribution_overlap_pct": overlap_pct(particle, dataset_noise),
            })
    return rows


def aggregate_coverage(coverage_rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    labels = defaultdict(int)
    for row in coverage_rows:
        key = (row["pipeline"], row["split"])
        grouped[key].append(float(row["coverage_pct"]))
        labels[key] += int(row["n_labels"])
    out = []
    for (pipeline, split), values in sorted(grouped.items()):
        arr = np.asarray(values, dtype=float)
        out.append({
            "pipeline": pipeline,
            "split": split,
            "n_signals": len(values),
            "n_labels": labels[(pipeline, split)],
            "coverage_pct_mean": float(np.mean(arr)),
            "coverage_pct_median": float(np.median(arr)),
            "coverage_pct_p95": float(np.percentile(arr, 95)),
            "coverage_pct_max": float(np.max(arr)),
        })
    return out


def values_for_plot(distribution: dict[str, float]) -> list[float]:
    return [distribution.get(label, math.nan) for label, _, _ in BANDS_HZ]


def energy_db_for_indices(distribution: dict[str, float], indices: list[int]) -> float:
    linear = 0.0
    seen = False
    for idx in indices:
        label = BANDS_HZ[idx][0]
        value = distribution.get(label, math.nan)
        if math.isfinite(value):
            linear += 10.0 ** (value / 10.0)
            seen = True
    return 10.0 * math.log10(max(linear, 1e-30)) if seen else math.nan


def find_target_band_indices(target_band_hz: tuple[float, float]) -> list[int]:
    indices = []
    target_low, target_high = target_band_hz
    for idx, (_, low, high) in enumerate(BANDS_HZ):
        band_high = high if math.isfinite(high) else math.inf
        if low < target_high and band_high > target_low:
            indices.append(idx)
    return indices


def format_pct(value: float) -> str:
    return f"{value:.1f}%" if math.isfinite(float(value)) else "NaN"


def coverage_by_pipeline(coverage_rows: list[dict]) -> dict[str, dict]:
    return {row["pipeline"]: row for row in coverage_rows}


def overlap_by_pipeline_class(overlap_rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {(row["pipeline"], row["class"]): row for row in overlap_rows}


def highlight_target_band(ax, target_indices: list[int]) -> None:
    for idx in target_indices:
        ax.axvspan(idx - 0.5, idx + 0.5, color="#f2d16b", alpha=0.22, zorder=0)


def write_pdf(
    path: Path,
    classes: tuple[str, ...],
    plot_data: dict,
    overlap_rows: list[dict],
    coverage_rows: list[dict],
    target_band_hz: tuple[float, float],
    display_names: dict[str, str],
    dataset_paths: dict[str, str],
    dataset_keys: tuple[str, ...],
    analysis_bandpass_hz: tuple[float, float] | None,
) -> None:
    labels = list(PLOT_BAND_LABELS)
    x = np.arange(len(labels))
    target_indices = find_target_band_indices(target_band_hz)
    target_label = f"{target_band_hz[0] / 1000:g}-{target_band_hz[1] / 1000:g} kHz"
    coverage_lookup = coverage_by_pipeline(coverage_rows)
    colors = ["#4c72b0", "#dd8452", "#55a868", "#8172b2", "#c44e52", "#64b5cd"]
    bar_width = min(0.24, 0.78 / max(1, len(dataset_keys)))
    offsets = (np.arange(len(dataset_keys)) - (len(dataset_keys) - 1) / 2.0) * bar_width

    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        title_suffix = ""
        if analysis_bandpass_hz is not None:
            title_suffix = f" - common {format_khz_band(analysis_bandpass_hz)} analysis bandpass"
        ax.set_title(f"YOLO Detection Dataset Spectral Comparison{title_suffix}", fontsize=15, fontweight="bold")
        lines = [
            "Particles are exact YOLO label intervals.",
            "Dataset noise is made of FFT windows that do not overlap any label.",
            "Standalone Noise is a single external reference, shown once in neutral color.",
        ]
        if analysis_bandpass_hz is not None:
            lines.extend([
                f"All datasets and the standalone Noise reference are analyzed after the same {format_khz_band(analysis_bandpass_hz)} bandpass filter.",
                "Relative percentages are comparable within the retained passband; absolute target-band energy is reported separately in dB.",
            ])
        else:
            lines.append("Relative percentages are computed from each dataset's own spectral energy.")
        lines.extend(["", "Datasets:"])
        for key in dataset_keys:
            lines.append(f"- {display_names[key]}: {dataset_paths[key]}")
        lines.extend([
            f"- {REFERENCE_LABEL}: {dataset_paths[REFERENCE_KEY]}",
            "",
            "Coverage summary:",
        ])
        for row in coverage_rows:
            lines.append(
                f"{display_names[row['pipeline']]} {row['split']}: signals={row['n_signals']}, "
                f"labels={row['n_labels']}, mean coverage={row['coverage_pct_mean']:.1f}%, "
                f"p95={row['coverage_pct_p95']:.1f}%"
            )
        lines.append("")
        lines.append(f"{target_label} and overlap summary:")
        for row in overlap_rows:
            lines.append(
                f"{display_names[row['pipeline']]} {row['class']}: "
                f"particle={float(row['particle_label_doppler_band_energy_pct']):.1f}%, "
                f"dataset-noise={float(row['dataset_noise_doppler_band_energy_pct']):.1f}%, "
                f"Noise reference={float(row['standalone_noise_doppler_band_energy_pct']):.1f}%, "
                f"dominant overlap={row['dominant_band_overlap']}"
            )
        ax.text(0.03, 0.92, "\n".join(lines), va="top", fontsize=8.2, family="monospace")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        for class_name in classes:
            fig = plt.figure(figsize=(11, 8.5))
            gs = fig.add_gridspec(5, 1, height_ratios=[0.85, 1.05, 1.05, 1.05, 0.75], hspace=0.48)
            axes = [fig.add_subplot(gs[i]) for i in range(4)]
            table_ax = fig.add_subplot(gs[4])
            table_ax.axis("off")
            compared = " vs ".join(display_names[key] for key in dataset_keys)
            fig.suptitle(f"{class_name} - {compared}", fontsize=13, fontweight="bold")

            reference_values = values_for_plot(plot_data[REFERENCE_KEY][class_name]["standalone_noise_folder"])
            highlight_target_band(axes[0], target_indices)
            axes[0].bar(x, reference_values, width=0.58, label=REFERENCE_LABEL, color="#6f777d")
            axes[0].set_title("Standalone Noise reference - relative distribution", fontsize=9.5)
            axes[0].set_ylabel("% total energy")
            axes[0].grid(axis="y", alpha=0.25)
            axes[0].legend(loc="upper right", fontsize=8)

            for ax, source, title in (
                (axes[1], "particle_labels", "Particle-labelled intervals - relative distribution"),
                (axes[2], "dataset_noise_windows", "Dataset noise windows outside labels - relative distribution"),
            ):
                highlight_target_band(ax, target_indices)
                for idx, key in enumerate(dataset_keys):
                    values = values_for_plot(plot_data[key][class_name][source])
                    ax.bar(
                        x + offsets[idx],
                        values,
                        bar_width,
                        label=display_names[key],
                        color=colors[idx % len(colors)],
                    )
                ax.set_title(title, fontsize=9.5)
                ax.set_ylabel("% total energy")
                ax.grid(axis="y", alpha=0.25)
                ax.legend(loc="upper right", fontsize=8)

            energy_ax = axes[3]
            positions = np.arange(len(dataset_keys))
            particle_vals = [
                energy_db_for_indices(plot_data[key][class_name]["particle_labels_energy_db"], target_indices)
                for key in dataset_keys
            ]
            noise_vals = [
                energy_db_for_indices(plot_data[key][class_name]["dataset_noise_windows_energy_db"], target_indices)
                for key in dataset_keys
            ]
            ref_val = energy_db_for_indices(
                plot_data[REFERENCE_KEY][class_name]["standalone_noise_folder_energy_db"], target_indices
            )
            energy_ax.bar(positions - 0.18, particle_vals, 0.36, label="Particle labels", color="#7aa6dc")
            energy_ax.bar(positions + 0.18, noise_vals, 0.36, label="Dataset noise", color="#f0a35e")
            if math.isfinite(ref_val):
                energy_ax.axhline(ref_val, color="#6f777d", linestyle="--", linewidth=1.2, label=REFERENCE_LABEL)
            energy_ax.set_title(f"Absolute normalized spectral energy in {target_label}", fontsize=9.5)
            energy_ax.set_ylabel("energy (dB)")
            energy_ax.set_xticks(positions)
            energy_ax.set_xticklabels([display_names[key] for key in dataset_keys], rotation=14, ha="right", fontsize=8)
            energy_ax.grid(axis="y", alpha=0.25)
            energy_ax.legend(loc="upper right", fontsize=8)

            for ax in axes[:3]:
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=8)
                ax.set_xlabel("Frequency band (kHz)")

            overlap_lookup = overlap_by_pipeline_class(overlap_rows)
            table_rows = []
            for key in dataset_keys:
                overlap = overlap_lookup[(key, class_name)]
                cov = coverage_lookup[key]
                table_rows.append([
                    display_names[key],
                    format_pct(overlap["particle_label_doppler_band_energy_pct"]),
                    format_pct(overlap["dataset_noise_doppler_band_energy_pct"]),
                    format_pct(overlap["standalone_noise_doppler_band_energy_pct"]),
                    format_pct(cov["coverage_pct_mean"]),
                ])
            table = table_ax.table(
                cellText=table_rows,
                colLabels=[
                    "Dataset",
                    f"Particle {target_label}",
                    f"Dataset-noise {target_label}",
                    f"Noise ref. {target_label}",
                    "Mean label coverage",
                ],
                loc="center",
                cellLoc="center",
                colLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.25)
            fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.08, hspace=0.58)
            pdf.savefig(fig)
            plt.close(fig)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare YOLO detection datasets spectrally.")
    parser.add_argument(
        "--dataset",
        action="append",
        type=parse_dataset_spec,
        help="Dataset spec as LABEL=PATH. Can be repeated. Overrides --old-dataset/--new-dataset when provided.",
    )
    parser.add_argument("--old-dataset")
    parser.add_argument("--new-dataset")
    parser.add_argument("--noise-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits", type=parse_csv_arg, default=("test",))
    parser.add_argument("--classes", type=parse_csv_arg, default=DEFAULT_CLASSES)
    parser.add_argument("--guard-samples", type=int, default=0)
    parser.add_argument("--doppler-band-khz", type=parse_khz_range, default=(10_000.0, 40_000.0))
    parser.add_argument("--analysis-bandpass-khz", type=parse_khz_range, default=None)
    parser.add_argument("--old-label", default=DEFAULT_OLD_LABEL)
    parser.add_argument("--new-label", default=DEFAULT_NEW_LABEL)
    parser.add_argument("--output-name", default="yolo_detection_spectral_comparison")
    return parser


def dataset_specs_from_args(args: argparse.Namespace) -> list[tuple[str, Path]]:
    if args.dataset:
        return list(args.dataset)
    if not args.old_dataset or not args.new_dataset:
        raise SystemExit("Either provide repeated --dataset LABEL=PATH or both --old-dataset and --new-dataset.")
    return [
        (args.old_label, Path(args.old_dataset)),
        (args.new_label, Path(args.new_dataset)),
    ]


def safe_key(label: str, used: set[str]) -> str:
    key = "".join(ch.lower() if ch.isalnum() else "_" for ch in label).strip("_") or "dataset"
    while "__" in key:
        key = key.replace("__", "_")
    base = key
    idx = 2
    while key in used or key == REFERENCE_KEY:
        key = f"{base}_{idx}"
        idx += 1
    used.add(key)
    return key


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.classes = tuple(args.classes)
    args.splits = tuple(args.splits)

    output_dir = Path(args.output_dir)
    dataset_specs = dataset_specs_from_args(args)
    used_keys = set()
    datasets = []
    for label, dataset_path in dataset_specs:
        key = safe_key(label, used_keys)
        datasets.append((key, label, dataset_path))

    all_band_rows = []
    all_energy_rows = []
    all_coverage_rows = []
    plot_data = {REFERENCE_KEY: {}}
    display_names = {key: label for key, label, _ in datasets}
    dataset_paths = {key: str(path) for key, _, path in datasets}
    dataset_paths[REFERENCE_KEY] = args.noise_dir

    for key, _label, dataset_path in datasets:
        rows, coverage, energy_rows, plot = analyze_dataset(
            dataset_path, key, args.classes, args.splits, args.guard_samples, args.analysis_bandpass_khz
        )
        all_band_rows.extend(rows)
        all_energy_rows.extend(energy_rows)
        all_coverage_rows.extend(coverage)
        plot_data[key] = plot

    reference_noise_rows, reference_noise_energy_rows, reference_noise_plot = analyze_standalone_noise(
        Path(args.noise_dir), args.classes, args.analysis_bandpass_khz
    )
    for class_name in args.classes:
        plot_data[REFERENCE_KEY][class_name] = reference_noise_plot[class_name]
        for key, _label, _dataset_path in datasets:
            plot_data[key][class_name]["standalone_noise_folder"] = reference_noise_plot[class_name]["standalone_noise_folder"]
            plot_data[key][class_name]["standalone_noise_folder_energy_db"] = reference_noise_plot[class_name]["standalone_noise_folder_energy_db"]

    band_rows = all_band_rows + reference_noise_rows
    energy_rows = all_energy_rows + reference_noise_energy_rows
    coverage_summary = aggregate_coverage(all_coverage_rows)
    overlap_rows = build_overlap_rows(plot_data, args.classes, args.doppler_band_khz)

    prefix = args.output_name
    legacy_names = not args.dataset and prefix == "yolo_detection_spectral_comparison"
    band_summary_name = "yolo_spectral_band_summary.csv" if legacy_names else f"{prefix}_band_summary.csv"
    coverage_summary_name = "yolo_label_coverage_summary.csv" if legacy_names else f"{prefix}_label_coverage_summary.csv"
    overlap_summary_name = "yolo_overlap_summary.csv" if legacy_names else f"{prefix}_overlap_summary.csv"
    energy_summary_name = f"{prefix}_absolute_energy_summary.csv"

    write_csv(
        output_dir / band_summary_name,
        band_rows,
        [
            "pipeline", "class", "source", "band", "band_low_hz", "band_high_hz",
            "mean_pct", "std_pct", "n_segments_or_windows", "n_candidate_windows",
        ],
    )
    write_csv(
        output_dir / energy_summary_name,
        energy_rows,
        [
            "pipeline", "class", "source", "band", "band_low_hz", "band_high_hz",
            "mean_energy_db", "median_energy_db", "std_energy_db", "n_segments_or_windows",
        ],
    )
    write_csv(
        output_dir / coverage_summary_name,
        coverage_summary,
        [
            "pipeline", "split", "n_signals", "n_labels", "coverage_pct_mean",
            "coverage_pct_median", "coverage_pct_p95", "coverage_pct_max",
        ],
    )
    write_csv(
        output_dir / overlap_summary_name,
        overlap_rows,
        [
            "pipeline", "class", "doppler_band", "particle_label_doppler_band_energy_pct",
            "dataset_noise_doppler_band_energy_pct", "standalone_noise_doppler_band_energy_pct",
            "delta_dataset_noise_minus_standalone_pct", "particle_dominant_band",
            "dataset_noise_dominant_band", "dominant_band_overlap",
            "particle_dataset_noise_distribution_overlap_pct",
        ],
    )
    write_pdf(
        output_dir / f"{prefix}.pdf",
        args.classes,
        plot_data,
        overlap_rows,
        coverage_summary,
        args.doppler_band_khz,
        display_names,
        dataset_paths,
        tuple(key for key, _label, _dataset_path in datasets),
        args.analysis_bandpass_khz,
    )

    print(f"Wrote YOLO spectral report to {output_dir}")
    print(f"- {output_dir / f'{prefix}.pdf'}")
    print(f"- {output_dir / band_summary_name}")
    print(f"- {output_dir / energy_summary_name}")
    print(f"- {output_dir / coverage_summary_name}")
    print(f"- {output_dir / overlap_summary_name}")


if __name__ == "__main__":
    main()
