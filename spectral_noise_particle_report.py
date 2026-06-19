"""Compare candidate-noise spectra, standalone noise, and Doppler picks.

This report is meant to support the signal-processing refinement argument:
candidate noise extracted from particle signals can still share spectral bands
with Doppler detections, which is consistent with missed or weak events.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


BANDS_HZ = (
    ("0-1 kHz", 0.0, 1_000.0),
    ("1-7 kHz", 1_000.0, 7_000.0),
    ("7-10 kHz", 7_000.0, 10_000.0),
    ("10-40 kHz", 10_000.0, 40_000.0),
    ("40-80 kHz", 40_000.0, 80_000.0),
    ("80 kHz-Nyq", 80_000.0, math.inf),
)

SAMPLING_RATE_HZ = 2_000_000.0
CLASS_FFT_CONFIGS = {
    "2um": {"fft_window_length": 4096, "fft_stride": 512},
    "4um": {"fft_window_length": 4096, "fft_stride": 1024},
    "10um": {"fft_window_length": 4096, "fft_stride": 1024},
    "yeast": {"fft_window_length": 2048, "fft_stride": 512},
}
DEFAULT_FFT_CONFIG = {"fft_window_length": 4096, "fft_stride": 1024}


def get_fft_config(class_name: str) -> dict:
    return dict(CLASS_FFT_CONFIGS.get(class_name, DEFAULT_FFT_CONFIG))


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


def read_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_dataset_results(output_dir: Path) -> list[dict]:
    path = output_dir / "dataset_results.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset results: {path}")
    with path.open() as f:
        data = json.load(f)
    return list(data.get("signals", []))


def resolve_signal_path(path_value: str, base_dir: Path) -> Path:
    path = Path(path_value)
    candidates = [path]
    if not path.is_absolute():
        candidates.append(base_dir / path)
        candidates.append(Path.cwd() / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Signal file not found: {path_value}")


def iter_windows(length: int, fft_len: int, stride: int) -> list[tuple[int, int]]:
    if length < fft_len:
        return []
    return [(start, start + fft_len) for start in range(0, length - fft_len + 1, stride)]


def particle_mask(signal_length: int, particles: list[dict], fs: float, sigma: float) -> np.ndarray:
    mask = np.zeros(signal_length, dtype=bool)
    for particle in particles:
        t0 = particle.get("t0")
        tau = particle.get("tau")
        if t0 is None or tau is None:
            continue
        center = float(t0) * fs
        half_width = max(1.0, sigma * float(tau) * fs)
        start = max(0, int(math.floor(center - half_width)))
        end = min(signal_length, int(math.ceil(center + half_width)))
        if end > start:
            mask[start:end] = True
    return mask


def band_energy_percentages(
    signal: np.ndarray,
    fs: float,
    fft_len: int,
    stride: int,
    valid_window_mask: np.ndarray | None = None,
) -> tuple[dict[str, float], int, float]:
    arr = np.asarray(signal, dtype=float).squeeze()
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1D signal, got shape {arr.shape}")

    freqs = np.fft.rfftfreq(fft_len, d=1.0 / fs)
    taper = np.hamming(fft_len)
    band_totals = {label: 0.0 for label, _, _ in BANDS_HZ}
    n_windows = 0

    for start, end in iter_windows(len(arr), fft_len, stride):
        if valid_window_mask is not None and valid_window_mask[start:end].any():
            continue
        segment = arr[start:end] * taper
        energy = np.abs(np.fft.rfft(segment)) ** 2
        for label, low, high in BANDS_HZ:
            if math.isinf(high):
                band_mask = freqs >= low
            else:
                band_mask = (freqs >= low) & (freqs < high)
            band_totals[label] += float(np.sum(energy[band_mask]))
        n_windows += 1

    total = sum(band_totals.values())
    if n_windows == 0 or total <= 0:
        return {label: math.nan for label, _, _ in BANDS_HZ}, n_windows, total
    return {label: value / total * 100.0 for label, value in band_totals.items()}, n_windows, total


def summarize_file_distributions(file_distributions: list[dict[str, float]]) -> dict[str, dict]:
    summary = {}
    for label, _, _ in BANDS_HZ:
        values = np.asarray(
            [row[label] for row in file_distributions if math.isfinite(row.get(label, math.nan))],
            dtype=float,
        )
        summary[label] = {
            "mean_pct": float(np.mean(values)) if len(values) else math.nan,
            "std_pct": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if len(values) == 1 else math.nan,
            "n_files": int(len(values)),
        }
    return summary


def analyze_candidate_noise(output_dir: Path, class_name: str, sigma: float) -> tuple[dict, int]:
    signals = [row for row in load_dataset_results(output_dir) if row.get("class") == class_name]
    config = get_fft_config(class_name)
    fs = SAMPLING_RATE_HZ
    file_distributions = []
    total_windows = 0

    for row in signals:
        signal_path = resolve_signal_path(str(row.get("path", "")), output_dir)
        signal = np.load(signal_path)
        mask = particle_mask(len(signal), row.get("particles", []), fs, sigma)
        dist, n_windows, _ = band_energy_percentages(
            signal,
            fs,
            int(config["fft_window_length"]),
            int(config["fft_stride"]),
            mask,
        )
        if n_windows > 0:
            file_distributions.append(dist)
        total_windows += n_windows

    return summarize_file_distributions(file_distributions), total_windows


def analyze_standalone_noise(noise_dir: Path, class_name: str) -> tuple[dict, int]:
    if not noise_dir.is_dir():
        raise FileNotFoundError(f"Missing standalone Noise directory: {noise_dir}")
    config = get_fft_config(class_name)
    fs = SAMPLING_RATE_HZ
    file_distributions = []
    total_windows = 0

    for path in sorted(noise_dir.glob("*.npy")):
        signal = np.load(path)
        dist, n_windows, _ = band_energy_percentages(
            signal,
            fs,
            int(config["fft_window_length"]),
            int(config["fft_stride"]),
            None,
        )
        if n_windows > 0:
            file_distributions.append(dist)
        total_windows += n_windows

    return summarize_file_distributions(file_distributions), total_windows


def band_for_frequency(freq_hz: float) -> str:
    for label, low, high in BANDS_HZ:
        if math.isinf(high):
            if freq_hz >= low:
                return label
        elif low <= freq_hz < high:
            return label
    return BANDS_HZ[-1][0]


def analyze_doppler_picks(output_dir: Path, class_name: str) -> tuple[dict[str, float], int]:
    path = output_dir / "snr_particles.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing Doppler picks CSV: {path}")
    counts = {label: 0 for label, _, _ in BANDS_HZ}
    total = 0
    for row in read_csv_rows(path):
        if row.get("class") != class_name:
            continue
        try:
            freq = float(row.get("frequency", "nan"))
        except ValueError:
            continue
        if not math.isfinite(freq):
            continue
        counts[band_for_frequency(freq)] += 1
        total += 1
    if total == 0:
        return {label: math.nan for label, _, _ in BANDS_HZ}, 0
    return {label: count / total * 100.0 for label, count in counts.items()}, total


def dominant_band(distribution: dict[str, float]) -> str | None:
    clean = {key: value for key, value in distribution.items() if math.isfinite(value)}
    if not clean:
        return None
    return max(clean, key=clean.get)


def distribution_overlap(left: dict[str, float], right: dict[str, float]) -> float:
    values = []
    for label, _, _ in BANDS_HZ:
        a = left.get(label, math.nan)
        b = right.get(label, math.nan)
        if math.isfinite(a) and math.isfinite(b):
            values.append(min(a, b))
    return float(sum(values)) if values else math.nan


def doppler_band_label(doppler_band_hz: tuple[float, float]) -> str:
    low_khz = doppler_band_hz[0] / 1000.0
    high_khz = doppler_band_hz[1] / 1000.0
    return f"{low_khz:g}-{high_khz:g} kHz"


def sum_range(distribution: dict[str, float], low_hz: float, high_hz: float) -> float:
    total = 0.0
    seen = False
    for label, low, high in BANDS_HZ:
        band_high = high if math.isfinite(high) else math.inf
        overlaps = low < high_hz and band_high > low_hz
        value = distribution.get(label, math.nan)
        if overlaps and math.isfinite(value):
            total += value
            seen = True
    return total if seen else math.nan


def collect_metrics(args: argparse.Namespace) -> tuple[list[dict], list[dict], list[dict], dict]:
    pipelines = {
        "old": Path(args.old_output),
        "new": Path(args.new_output),
    }
    classes = tuple(args.classes)
    noise_dir = Path(args.noise_dir)
    doppler_low, doppler_high = args.doppler_band_hz

    band_rows = []
    doppler_rows = []
    overlap_rows = []
    plot_data = {}

    standalone_cache = {}
    for class_name in classes:
        standalone_cache[class_name] = analyze_standalone_noise(noise_dir, class_name)

    for pipeline_name, output_dir in pipelines.items():
        for class_name in classes:
            candidate_summary, candidate_windows = analyze_candidate_noise(
                output_dir, class_name, args.particle_sigma
            )
            standalone_summary, standalone_windows = standalone_cache[class_name]
            doppler_dist, n_picks = analyze_doppler_picks(output_dir, class_name)

            candidate_dist = {label: candidate_summary[label]["mean_pct"] for label, _, _ in BANDS_HZ}
            standalone_dist = {label: standalone_summary[label]["mean_pct"] for label, _, _ in BANDS_HZ}

            plot_data[(pipeline_name, class_name)] = {
                "candidate": candidate_dist,
                "standalone": standalone_dist,
                "doppler": doppler_dist,
            }

            for source, summary, n_windows in (
                ("candidate_noise_from_particle_signals", candidate_summary, candidate_windows),
                ("standalone_noise_folder", standalone_summary, standalone_windows),
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
                        "n_files": summary[label]["n_files"],
                        "n_windows": n_windows,
                    })

            for label, low, high in BANDS_HZ:
                doppler_rows.append({
                    "pipeline": pipeline_name,
                    "class": class_name,
                    "band": label,
                    "band_low_hz": low,
                    "band_high_hz": "" if math.isinf(high) else high,
                    "pick_pct": doppler_dist[label],
                    "n_picks": n_picks,
                })

            cand_dom = dominant_band(candidate_dist)
            dop_dom = dominant_band(doppler_dist)
            cand_doppler = sum_range(candidate_dist, doppler_low, doppler_high)
            standalone_doppler = sum_range(standalone_dist, doppler_low, doppler_high)
            picks_doppler = sum_range(doppler_dist, doppler_low, doppler_high)
            overlap_rows.append({
                "pipeline": pipeline_name,
                "class": class_name,
                "doppler_band": doppler_band_label(args.doppler_band_hz),
                "candidate_noise_doppler_band_energy_pct": cand_doppler,
                "standalone_noise_doppler_band_energy_pct": standalone_doppler,
                "delta_candidate_minus_standalone_pct": (
                    cand_doppler - standalone_doppler
                    if math.isfinite(cand_doppler) and math.isfinite(standalone_doppler)
                    else math.nan
                ),
                "doppler_picks_in_band_pct": picks_doppler,
                "candidate_noise_dominant_band": cand_dom or "",
                "doppler_picks_dominant_band": dop_dom or "",
                "dominant_band_overlap": bool(cand_dom and dop_dom and cand_dom == dop_dom),
                "distribution_overlap_pct": distribution_overlap(candidate_dist, doppler_dist),
                "candidate_noise_windows": candidate_windows,
                "standalone_noise_windows": standalone_windows,
                "doppler_picks": n_picks,
            })

    return band_rows, doppler_rows, overlap_rows, plot_data


def values_for_plot(distribution: dict[str, float]) -> list[float]:
    return [distribution.get(label, math.nan) for label, _, _ in BANDS_HZ]


def write_pdf(path: Path, classes: tuple[str, ...], overlap_rows: list[dict], plot_data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [label for label, _, _ in BANDS_HZ]
    x = np.arange(len(labels))
    width = 0.36

    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.set_title("Spectral Noise vs Doppler Picks - Old/New Pipeline", fontsize=15, fontweight="bold")
        summary_text = [
            "Candidate noise = windows from particle signals that do not overlap detected particles (t0 +/- 3 tau).",
            "Standalone noise = files from the Noise folder, analyzed with the same class FFT settings.",
            "Interpretation: shared dominant bands are consistent with contamination by weak or missed events.",
            "",
        ]
        for row in overlap_rows:
            summary_text.append(
                f"{row['pipeline']} {row['class']}: candidate {row['doppler_band']}="
                f"{float(row['candidate_noise_doppler_band_energy_pct']):.1f}%, "
                f"Noise={float(row['standalone_noise_doppler_band_energy_pct']):.1f}%, "
                f"Doppler picks={float(row['doppler_picks_in_band_pct']):.1f}%, "
                f"dominant overlap={row['dominant_band_overlap']}"
            )
        ax.text(0.03, 0.92, "\n".join(summary_text), va="top", fontsize=10, family="monospace")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        for class_name in classes:
            fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
            fig.suptitle(f"{class_name} - Old vs New Spectral Comparison", fontsize=14, fontweight="bold")
            panels = (
                ("candidate", "Candidate noise from particle signals", "% total energy"),
                ("standalone", "Standalone Noise folder", "% total energy"),
                ("doppler", "Doppler picks", "% picks"),
            )
            for ax, (source, title, ylabel) in zip(axes, panels):
                old_vals = values_for_plot(plot_data.get(("old", class_name), {}).get(source, {}))
                new_vals = values_for_plot(plot_data.get(("new", class_name), {}).get(source, {}))
                ax.bar(x - width / 2, old_vals, width, label="old", color="#4c72b0")
                ax.bar(x + width / 2, new_vals, width, label="new", color="#dd8452")
                ax.set_title(title, fontsize=10)
                ax.set_ylabel(ylabel)
                ax.grid(axis="y", alpha=0.25)
                ax.legend(loc="upper right")
            axes[-1].set_xticks(x)
            axes[-1].set_xticklabels(labels, rotation=25, ha="right")
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            pdf.savefig(fig)
            plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build old/new spectral comparison for candidate noise, Noise folder, and Doppler picks."
    )
    parser.add_argument("--old-output", required=True, help="Old particles2SNR output split directory")
    parser.add_argument("--new-output", required=True, help="New particles2SNR output split directory")
    parser.add_argument("--noise-dir", required=True, help="Standalone Noise .npy directory")
    parser.add_argument("--output-dir", required=True, help="Directory for PDF and CSV outputs")
    parser.add_argument("--classes", type=parse_csv_arg, default=("2um", "4um", "10um"))
    parser.add_argument("--doppler-band-khz", type=parse_khz_range, default=(10_000.0, 40_000.0))
    parser.add_argument("--particle-sigma", type=float, default=3.0)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.classes = tuple(args.classes)
    args.doppler_band_hz = args.doppler_band_khz

    output_dir = Path(args.output_dir)
    band_rows, doppler_rows, overlap_rows, plot_data = collect_metrics(args)

    write_csv(
        output_dir / "spectral_band_summary.csv",
        band_rows,
        [
            "pipeline", "class", "source", "band", "band_low_hz", "band_high_hz",
            "mean_pct", "std_pct", "n_files", "n_windows",
        ],
    )
    write_csv(
        output_dir / "doppler_peak_summary.csv",
        doppler_rows,
        ["pipeline", "class", "band", "band_low_hz", "band_high_hz", "pick_pct", "n_picks"],
    )
    write_csv(
        output_dir / "overlap_summary.csv",
        overlap_rows,
        [
            "pipeline", "class", "doppler_band",
            "candidate_noise_doppler_band_energy_pct",
            "standalone_noise_doppler_band_energy_pct",
            "delta_candidate_minus_standalone_pct",
            "doppler_picks_in_band_pct",
            "candidate_noise_dominant_band",
            "doppler_picks_dominant_band",
            "dominant_band_overlap",
            "distribution_overlap_pct",
            "candidate_noise_windows",
            "standalone_noise_windows",
            "doppler_picks",
        ],
    )
    write_pdf(output_dir / "spectral_comparison.pdf", args.classes, overlap_rows, plot_data)

    print(f"Wrote report to {output_dir}")
    print(f"- {output_dir / 'spectral_comparison.pdf'}")
    print(f"- {output_dir / 'spectral_band_summary.csv'}")
    print(f"- {output_dir / 'doppler_peak_summary.csv'}")
    print(f"- {output_dir / 'overlap_summary.csv'}")


if __name__ == "__main__":
    main()
