#!/usr/bin/env python3
"""Generate presentation PNGs highlighting SNR behavior on two 4um signals.

The script creates three variants per source signal:
- particles2SNR
- particles2SNR_F
- particles2SNR_F with annotations filtered to snr_db >= -10 dB
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from particles2snr.repo_paths import RESULTS_FIGURES, RESULTS_RUNS


DATASET_PARTICLES2SNR = "p0_c1_particles2SNR"
DATASET_PARTICLES2SNR_F = "p0_c1_Particles2SNR_F"
SIGNALS = (
    "HFocusing_5_10_4um_0_1515.npy",
    "HFocusing_5_10_4um_0_335.npy",
)

LABEL_COLORS = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
    "#bcbd22",
    "#7f7f7f",
    "#ff7f0e",
]

OVERLAP_STYLES = {
    1: ("#f6b26b", 0.14),
    2: ("#f28e2b", 0.26),
    3: ("#d94801", 0.38),
}


def load_particles2SNR_metadata(jsons: dict[str, Path]) -> dict[tuple[str, str], dict]:
    metadata = {}
    for split, path in jsons.items():
        if not path.exists():
            continue
        with path.open() as handle:
            data = json.load(handle)
        for row in data.get("data", []):
            filename = row.get("filename")
            if filename:
                metadata[(split, filename)] = row
                metadata.setdefault(("*", filename), row)
    return metadata


def annotation_segments(annotations: list[dict], threshold_db: float | None = None) -> list[dict]:
    labels = []
    for ann in annotations:
        snr = as_float(ann.get("snr_db"))
        if threshold_db is not None and (snr is None or snr < threshold_db):
            continue
        start = as_float(ann.get("start"))
        end = as_float(ann.get("end"))
        if start is None or end is None or end <= start:
            continue
        labels.append(
            {
                "start": start,
                "end": end,
                "snr_db": snr,
                "peak_group_id": ann.get("peak_group_id"),
            }
        )
    return labels


def overlap_segments(labels: list[dict]) -> list[dict]:
    events = []
    for label in labels:
        start = float(label.get("start", 0.0))
        end = float(label.get("end", 0.0))
        if end <= start:
            continue
        events.append((start, 1))
        events.append((end, -1))
    if not events:
        return []
    events.sort(key=lambda item: (item[0], -item[1]))
    segments = []
    active = 0
    prev = None
    idx = 0
    while idx < len(events):
        pos = events[idx][0]
        if prev is not None and pos > prev and active > 0:
            segments.append({"start": float(prev), "end": float(pos), "overlap": int(active)})
        while idx < len(events) and events[idx][0] == pos:
            active += events[idx][1]
            idx += 1
        prev = pos
    return segments


def max_overlap(labels: list[dict]) -> int:
    segments = overlap_segments(labels)
    return max((segment["overlap"] for segment in segments), default=0)


def draw_overlap_density(ax, labels: list[dict], signal_len: int, fs: float) -> None:
    for segment in overlap_segments(labels):
        overlap = min(3, int(segment["overlap"]))
        color, alpha = OVERLAP_STYLES[overlap]
        start_ms = segment["start"] * signal_len / fs * 1000.0
        end_ms = segment["end"] * signal_len / fs * 1000.0
        ax.axvspan(start_ms, end_ms, color=color, alpha=alpha, linewidth=0)


def draw_label_outlines(ax, labels: list[dict], signal_len: int, fs: float) -> None:
    ymin, ymax = ax.get_ylim()
    y_text = ymax - 0.08 * (ymax - ymin)
    for idx, label in enumerate(labels):
        color = LABEL_COLORS[idx % len(LABEL_COLORS)]
        start_ms = label["start"] * signal_len / fs * 1000.0
        end_ms = label["end"] * signal_len / fs * 1000.0
        center_ms = (start_ms + end_ms) / 2.0
        ax.axvline(start_ms, color=color, linewidth=1.1, alpha=0.95)
        ax.axvline(end_ms, color=color, linewidth=1.1, alpha=0.95)
        ax.hlines(y_text, start_ms, end_ms, color=color, linewidth=1.4, alpha=0.95)
        peak = label.get("peak_group_id")
        suffix = f"->p{peak}" if peak is not None else ""
        ax.text(center_ms, y_text, f"a{idx}{suffix}", color=color, fontsize=7, ha="center", va="bottom")


def draw_peak_groups(ax, peak_groups: list[dict]) -> None:
    ymin, ymax = ax.get_ylim()
    y_text = ymin + 0.08 * (ymax - ymin)
    for group in peak_groups:
        peak_ms = group.get("peak_center_ms")
        if peak_ms is None:
            continue
        gid = group.get("id")
        ax.axvline(float(peak_ms), color="#0057b8", linewidth=0.9, linestyle="--", alpha=0.75)
        ax.text(float(peak_ms), y_text, f"p{gid}", color="#0057b8", fontsize=7, ha="center", va="top")


def add_legend(ax, include_peaks: bool) -> None:
    handles = [
        plt.Line2D([0], [0], color=OVERLAP_STYLES[1][0], linewidth=5, alpha=OVERLAP_STYLES[1][1], label="overlap=1"),
        plt.Line2D([0], [0], color=OVERLAP_STYLES[2][0], linewidth=5, alpha=OVERLAP_STYLES[2][1], label="overlap=2"),
        plt.Line2D([0], [0], color=OVERLAP_STYLES[3][0], linewidth=5, alpha=OVERLAP_STYLES[3][1], label="overlap>=3"),
    ]
    if include_peaks:
        handles.append(plt.Line2D([0], [0], color="#0057b8", linestyle="--", linewidth=1, label="peak group"))
    ax.legend(handles=handles, loc="upper right", fontsize=7, frameon=False, ncol=min(4, len(handles)))


def summarize_snr(labels: list[dict]) -> dict[str, float | int | None]:
    snrs = []
    for label in labels:
        snr = label.get("snr_db")
        try:
            snr_value = float(snr)
        except (TypeError, ValueError):
            continue
        if np.isfinite(snr_value):
            snrs.append(snr_value)
    if not snrs:
        return {
            "count": 0,
            "min_snr_db": None,
            "median_snr_db": None,
            "max_snr_db": None,
        }
    arr = np.asarray(snrs, dtype=float)
    return {
        "count": int(arr.size),
        "min_snr_db": float(np.min(arr)),
        "median_snr_db": float(np.median(arr)),
        "max_snr_db": float(np.max(arr)),
    }


def as_float(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def load_sample(
    root: Path,
    split: str,
    filename: str,
    *,
    class_name: str | None = None,
    direct: bool = False,
) -> dict | None:
    if direct:
        signal_path = root / filename
    elif class_name is not None:
        signal_path = root / split / class_name / filename
    else:
        signal_path = root / split / "signals" / filename
    if not signal_path.exists():
        return None
    return {"split": split, "signal_path": signal_path}


def load_annotation_row(metadata: dict[tuple[str, str], dict], split: str, filename: str) -> dict | None:
    return metadata.get((split, filename)) or metadata.get(("*", filename))


def signal_ylim(signal: np.ndarray) -> tuple[float, float]:
    ymin, ymax = float(np.min(signal)), float(np.max(signal))
    pad = max(1e-6, 0.08 * (ymax - ymin))
    return ymin - pad, ymax + pad


def plot_signal_variant(
    ax,
    sample: dict,
    fs: float,
    display_label: str,
    variant_label: str,
    annotations: list[dict],
    peak_groups: list[dict],
    threshold_db: float | None = None,
    pair_ylim: tuple[float, float] | None = None,
) -> dict:
    signal = np.asarray(np.load(sample["signal_path"])).squeeze()
    time_ms = np.arange(len(signal)) / fs * 1000.0
    total_annotations = len(annotations)
    labels = annotation_segments(annotations, threshold_db=threshold_db)
    draw_overlap_density(ax, labels, len(signal), fs)
    ax.plot(time_ms, signal, color="#222222", linewidth=0.7, zorder=3)
    ax.set_ylim(*(pair_ylim or signal_ylim(signal)))
    draw_label_outlines(ax, labels, len(signal), fs)
    draw_peak_groups(ax, peak_groups)
    add_legend(ax, bool(peak_groups))
    ax.set_ylabel(display_label, fontsize=8)
    ax.grid(True, alpha=0.18)
    snr_info = summarize_snr(labels)
    if threshold_db is None:
        title = f"{sample['signal_path'].name} | {variant_label} | annotations={snr_info['count']}"
    else:
        title = (
            f"{sample['signal_path'].name} | particles2SNR_F | SNR >= {threshold_db:.0f} dB "
            f"| kept {snr_info['count']}"
        )
    ax.set_title(title, loc="left", fontsize=9)
    return {
        "signal_name": sample["signal_path"].name,
        "signal_path": str(sample["signal_path"]),
        "variant": variant_label,
        "split": sample["split"],
        "dataset": sample.get("dataset", ""),
        "labels_total": total_annotations,
        "labels_kept": snr_info["count"],
        "peak_groups": len(peak_groups),
        "min_snr_db": snr_info["min_snr_db"],
        "median_snr_db": snr_info["median_snr_db"],
        "max_snr_db": snr_info["max_snr_db"],
    }


def write_manifest(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "signal_name",
        "signal_path",
        "variant",
        "split",
        "dataset",
        "output_path",
        "labels_total",
        "labels_kept",
        "peak_groups",
        "threshold_db",
        "min_snr_db",
        "median_snr_db",
        "max_snr_db",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate presentation SNR assets.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_FIGURES / "presentation_snr_assets",
    )
    parser.add_argument("--fs", type=float, default=2_000_000.0)
    parser.add_argument("--threshold-db", type=float, default=-10.0)
    args = parser.parse_args()

    particles2SNR_jsons = {
        "train": RESULTS_RUNS / DATASET_PARTICLES2SNR / "train" / "data.json",
        "test": RESULTS_RUNS / DATASET_PARTICLES2SNR / "test" / "data.json",
    }
    particles2SNR_F_jsons = {
        "train": RESULTS_RUNS / DATASET_PARTICLES2SNR_F / "train" / "data.json",
        "test": RESULTS_RUNS / DATASET_PARTICLES2SNR_F / "test" / "data.json",
    }
    metadata_particles2SNR = load_particles2SNR_metadata(particles2SNR_jsons)
    metadata_particles2SNR_F = load_particles2SNR_metadata(particles2SNR_F_jsons)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    roots = {
        DATASET_PARTICLES2SNR: Path("datasets/raw/c1-hf-5-10-4um-doublet/v1"),
        DATASET_PARTICLES2SNR_F: Path("datasets/processed/particles2snr-f-c1-yolo-3class/v1"),
    }
    rows = []
    variants = [
        (DATASET_PARTICLES2SNR, "particles2SNR", metadata_particles2SNR, None),
        (DATASET_PARTICLES2SNR_F, "particles2SNR_F", metadata_particles2SNR_F, None),
        (
            DATASET_PARTICLES2SNR_F,
            "particles2SNR_F_snr_ge_m10db",
            metadata_particles2SNR_F,
            args.threshold_db,
        ),
    ]

    for filename in SIGNALS:
        c1_sample = load_sample(roots[DATASET_PARTICLES2SNR], "test", filename, direct=True)
        clean_sample = load_sample(roots[DATASET_PARTICLES2SNR_F], "test", filename)
        if c1_sample is None or clean_sample is None:
            raise FileNotFoundError(f"Missing source signal for {filename}")
        signal = np.asarray(np.load(c1_sample["signal_path"])).squeeze()
        ylim = signal_ylim(signal)
        for dataset_name, variant_label, metadata, threshold_db in variants:
            sample = c1_sample if dataset_name == DATASET_PARTICLES2SNR else clean_sample
            sample = dict(sample)
            sample["dataset"] = dataset_name
            row = load_annotation_row(metadata, "test", filename)
            if row is None:
                raise KeyError(f"Missing particles2SNR metadata for {filename} in {dataset_name}")
            output_path = args.output_dir / f"{Path(filename).stem}_{variant_label}.png"
            fig, ax = plt.subplots(1, 1, figsize=(14, 3.1))
            row = plot_signal_variant(
                ax,
                sample,
                args.fs,
                display_label=dataset_name.replace("p0_c1_", ""),
                variant_label=variant_label,
                annotations=row.get("annotations", []),
                peak_groups=row.get("peak_groups", []) if dataset_name == DATASET_PARTICLES2SNR_F else [],
                threshold_db=threshold_db,
                pair_ylim=ylim,
            )
            if threshold_db is not None:
                row["threshold_db"] = threshold_db
            else:
                row["threshold_db"] = ""
            fig.tight_layout()
            fig.savefig(output_path, dpi=180)
            plt.close(fig)
            row["output_path"] = str(output_path)
            rows.append(row)

    write_manifest(args.output_dir / "presentation_snr_assets_manifest.csv", rows)
    print(f"Wrote {len(rows)} PNGs to {args.output_dir}")


if __name__ == "__main__":
    main()
