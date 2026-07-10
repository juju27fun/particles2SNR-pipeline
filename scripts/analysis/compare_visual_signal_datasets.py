#!/usr/bin/env python3
"""Generate visual comparisons between YOLO detection datasets."""

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


DATASET_C1 = "p0_c1_particles2SNR"
DATASET_CLEAN = "p0_c1_clean_7_80khz"
DATASET_YOLO_V3 = "yolo_v3_source_named"

DATASETS = {
    DATASET_C1: {
        "root": Path("datasets/interim/particles2SNR-pipeline/particles2snr-c1-yolo"),
        "classes": ("2um", "4um", "10um"),
        "particles2SNR_jsons": {
            "train": RESULTS_RUNS / "p0_c1_particles2SNR" / "train" / "data.json",
            "test": RESULTS_RUNS / "p0_c1_particles2SNR" / "test" / "data.json",
        },
        "display": "P0 C1 particles2SNR",
    },
    DATASET_CLEAN: {
        "root": Path("datasets/processed/particles2snr-f-c1-yolo-3class/v1"),
        "classes": ("2um", "4um", "10um"),
        "particles2SNR_jsons": {
            "train": RESULTS_RUNS / "p0_c1_Particles2SNR_F" / "train" / "data.json",
            "test": RESULTS_RUNS / "p0_c1_Particles2SNR_F" / "test" / "data.json",
        },
        "display": "P0 C1 clean + 7-80 kHz",
    },
    DATASET_YOLO_V3: {
        "root": Path("datasets/processed/yolo-v3-source-named/v1"),
        "classes": ("2um", "4um", "10um"),
        "particles2SNR_jsons": {},
        "display": "YOLO v3 source-named",
    },
}

REFERENCE_SELECTION = {
    ("test", "4um"): (
        "HFocusing_5_10_4um_0_1515.npy",
        "HFocusing_5_10_4um_0_335.npy",
        "HFocusing_5_10_4um_0_986.npy",
        "HFocusing_5_10_4um_0_1004.npy",
    ),
}

FOCUSED_4UM_FILES = (
    "HFocusing_5_10_4um_0_1515.npy",
    "HFocusing_5_10_4um_0_986.npy",
)

LABEL_COLORS = [
    "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b",
    "#e377c2", "#17becf", "#bcbd22", "#7f7f7f", "#ff7f0e",
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


def read_yolo_labels(label_path: Path, classes: tuple[str, ...]) -> list[dict]:
    if not label_path.exists():
        return []
    labels = []
    with label_path.open() as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            class_id = int(float(parts[0]))
            center = float(parts[1])
            width = float(parts[2])
            labels.append({
                "class": classes[class_id] if class_id < len(classes) else str(class_id),
                "start": max(0.0, center - width / 2.0),
                "end": min(1.0, center + width / 2.0),
            })
    return labels


def load_sample(dataset_name: str, split: str, filename: str) -> dict | None:
    config = DATASETS[dataset_name]
    signal_path = config["root"] / split / "signals" / filename
    if not signal_path.exists():
        return None
    labels = read_yolo_labels(signal_path.parent.parent / "labels" / f"{signal_path.stem}.txt", config["classes"])
    return {"dataset": dataset_name, "split": split, "signal_path": signal_path, "labels": labels}


def load_sample_any_split(dataset_name: str, filename: str) -> dict | None:
    for split in ("train", "val", "test"):
        sample = load_sample(dataset_name, split, filename)
        if sample is not None:
            return sample
    return None


def class_for_yolo(labels: list[dict]) -> str:
    if not labels:
        return "no_label"
    counts = defaultdict(int)
    for label in labels:
        counts[label["class"]] += 1
    return max(sorted(counts), key=counts.get)


def collect_by_class(dataset_name: str, split: str) -> dict[str, list[dict]]:
    config = DATASETS[dataset_name]
    signals_dir = config["root"] / split / "signals"
    by_class = defaultdict(list)
    if not signals_dir.exists():
        return by_class
    for signal_path in sorted(signals_dir.glob("*.npy")):
        labels = read_yolo_labels(signal_path.parent.parent / "labels" / f"{signal_path.stem}.txt", config["classes"])
        sample = {"dataset": dataset_name, "split": split, "signal_path": signal_path, "labels": labels}
        by_class[class_for_yolo(labels)].append(sample)
    return by_class


def load_previous_selection(path: Path) -> dict[tuple[str, str], list[str]]:
    selection = defaultdict(list)
    if path.exists():
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("dataset") != "p0_yolo_clean_filt_trainval":
                    continue
                split = row.get("split")
                class_name = row.get("class")
                signal_path = row.get("signal_path")
                if split and class_name and signal_path:
                    selection[(split, class_name)].append(Path(signal_path).name)
    for key, names in REFERENCE_SELECTION.items():
        current = selection[key]
        selection[key] = list(names) + [name for name in current if name not in names]
    return selection


def select_common_samples(split: str, class_name: str, previous_selection: dict[tuple[str, str], list[str]], max_samples: int) -> list[str]:
    c1_dir = DATASETS[DATASET_C1]["root"] / split / "signals"
    clean_dir = DATASETS[DATASET_CLEAN]["root"] / split / "signals"
    common = {p.name for p in c1_dir.glob("*.npy")} & {p.name for p in clean_dir.glob("*.npy")}
    selected = [name for name in previous_selection.get((split, class_name), []) if name in common]
    if len(selected) < max_samples:
        clean_by_class = collect_by_class(DATASET_CLEAN, split)
        for sample in clean_by_class.get(class_name, []):
            name = sample["signal_path"].name
            if name in common and name not in selected:
                selected.append(name)
            if len(selected) >= max_samples:
                break
    return selected[:max_samples]


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


def enrich_labels_with_particles2SNR(labels: list[dict], split: str, filename: str, metadata: dict[tuple[str, str], dict]) -> tuple[list[dict], list[dict]]:
    row = metadata.get((split, filename)) or metadata.get(("*", filename))
    if row is None:
        return labels, []
    annotations = row.get("annotations", [])
    enriched = []
    used = set()
    for label in labels:
        best_idx = None
        best_score = float("inf")
        for idx, ann in enumerate(annotations):
            if idx in used:
                continue
            score = abs(float(label.get("start", 0.0)) - float(ann.get("start", 0.0))) + abs(float(label.get("end", 0.0)) - float(ann.get("end", 0.0)))
            if score < best_score:
                best_idx = idx
                best_score = score
        out = dict(label)
        if best_idx is not None and best_score < 1e-3:
            used.add(best_idx)
            ann = annotations[best_idx]
            out.update({
                "ann_id": int(best_idx),
                "peak_group_id": ann.get("peak_group_id"),
                "peak_center_ms": ann.get("peak_center_ms"),
                "peak_z": ann.get("peak_z"),
            })
        enriched.append(out)
    return enriched, row.get("peak_groups", [])


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


def label_summary(labels: list[dict]) -> str:
    if not labels:
        return ""
    counts = defaultdict(int)
    for label in labels:
        counts[label["class"]] += 1
    return ", ".join(f"{name}={counts[name]}" for name in sorted(counts))


def plot_sample_axis(ax, sample: dict, fs: float, metadata: dict[tuple[str, str], dict], row_label: str, pair_ylim: tuple[float, float] | None = None) -> dict:
    signal = np.load(sample["signal_path"])
    signal = np.asarray(signal).squeeze()
    time_ms = np.arange(len(signal)) / fs * 1000.0
    labels, peak_groups = enrich_labels_with_particles2SNR(sample["labels"], sample["split"], sample["signal_path"].name, metadata)
    draw_overlap_density(ax, labels, len(signal), fs)
    ax.plot(time_ms, signal, color="#222222", linewidth=0.7, zorder=3)
    if pair_ylim is None:
        ymin, ymax = float(np.min(signal)), float(np.max(signal))
        pad = max(1e-6, 0.08 * (ymax - ymin))
        ax.set_ylim(ymin - pad, ymax + pad)
    else:
        ax.set_ylim(*pair_ylim)
    draw_label_outlines(ax, labels, len(signal), fs)
    draw_peak_groups(ax, peak_groups)
    add_legend(ax, bool(peak_groups))
    ax.set_ylabel(row_label, fontsize=8)
    ax.grid(True, alpha=0.18)
    ax.set_title(
        f"{sample['signal_path'].name} | labels: {label_summary(labels)}",
        loc="left",
        fontsize=9,
    )
    return {
        "dataset": sample["dataset"],
        "signal_name": sample["signal_path"].name,
        "signal_path": str(sample["signal_path"]),
        "label_count": len(labels),
        "overlap_max": max_overlap(labels),
        "peak_group_count": len(peak_groups),
        "mean_abs": f"{float(np.mean(np.abs(signal))):.9g}",
        "min": f"{float(np.min(signal)):.9g}",
        "max": f"{float(np.max(signal)):.9g}",
    }


def pair_ylim(samples: list[dict]) -> tuple[float, float]:
    mins = []
    maxs = []
    for sample in samples:
        signal = np.asarray(np.load(sample["signal_path"])).squeeze()
        mins.append(float(np.min(signal)))
        maxs.append(float(np.max(signal)))
    ymin, ymax = min(mins), max(maxs)
    pad = max(1e-6, 0.08 * (ymax - ymin))
    return ymin - pad, ymax + pad


def plot_aligned_comparison(samples_by_name: list[tuple[str, dict, dict]], title: str, output_path: Path, fs: float, metadata_by_dataset: dict[str, dict]) -> list[dict]:
    if not samples_by_name:
        return []
    fig, axes = plt.subplots(len(samples_by_name) * 2, 1, figsize=(15, 2.0 * len(samples_by_name) * 2), squeeze=False)
    rows = []
    axis_idx = 0
    for filename, c1_sample, clean_sample in samples_by_name:
        ylim = pair_ylim([c1_sample, clean_sample])
        for sample in (c1_sample, clean_sample):
            ax = axes[axis_idx, 0]
            dataset_name = sample["dataset"]
            row = plot_sample_axis(ax, sample, fs, metadata_by_dataset.get(dataset_name, {}), DATASETS[dataset_name]["display"], ylim)
            row.update({"comparison_type": "aligned_p0_same_filename", "filename": filename})
            rows.append(row)
            axis_idx += 1
    axes[-1, 0].set_xlabel("time (ms)")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return rows


def plot_reference(samples: list[dict], title: str, output_path: Path, fs: float) -> list[dict]:
    if not samples:
        return []
    fig, axes = plt.subplots(len(samples), 1, figsize=(15, 2.2 * len(samples)), squeeze=False)
    rows = []
    for ax, sample in zip(axes[:, 0], samples):
        row = plot_sample_axis(ax, sample, fs, {}, DATASETS[sample["dataset"]]["display"])
        row.update({"comparison_type": "unaligned_yolo_v3_reference", "filename": sample["signal_path"].name})
        rows.append(row)
    axes[-1, 0].set_xlabel("time (ms)")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return rows


def plot_focused_4um_impact(output_dir: Path, fs: float, metadata_by_dataset: dict[str, dict]) -> list[dict]:
    split = "test"
    groups = []
    for filename in FOCUSED_4UM_FILES:
        c1_sample = load_sample(DATASET_C1, split, filename)
        clean_sample = load_sample(DATASET_CLEAN, split, filename)
        if c1_sample is None or clean_sample is None:
            continue
        yolo_sample = load_sample_any_split(DATASET_YOLO_V3, filename)
        groups.append((filename, c1_sample, clean_sample, yolo_sample))

    if not groups:
        return []

    fig, axes = plt.subplots(len(groups) * 3, 1, figsize=(15, 2.0 * len(groups) * 3), squeeze=False)
    rows = []
    axis_idx = 0
    for filename, c1_sample, clean_sample, yolo_sample in groups:
        ylim = pair_ylim([c1_sample, clean_sample])
        for sample in (c1_sample, clean_sample):
            ax = axes[axis_idx, 0]
            dataset_name = sample["dataset"]
            row = plot_sample_axis(
                ax,
                sample,
                fs,
                metadata_by_dataset.get(dataset_name, {}),
                DATASETS[dataset_name]["display"],
                ylim,
            )
            row.update({"comparison_type": "focused_4um_aligned_p0_same_filename", "filename": filename})
            rows.append(row)
            axis_idx += 1

        ax = axes[axis_idx, 0]
        if yolo_sample is not None:
            row = plot_sample_axis(ax, yolo_sample, fs, {}, DATASETS[DATASET_YOLO_V3]['display'])
            row.update({
                "comparison_type": "focused_4um_aligned_yolo_v3_source_filename",
                "filename": filename,
            })
            rows.append(row)
            ax.text(
                0.01,
                0.92,
                f"Same source filename, YOLO split={yolo_sample['split']}",
                transform=ax.transAxes,
                fontsize=8,
                color="#555555",
                ha="left",
                va="top",
                bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 2},
            )
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, "Missing source-named YOLO v3 sample", ha="center", va="center")
        axis_idx += 1

    axes[-1, 0].set_xlabel("time (ms)")
    fig.suptitle(
        "Focused 4um comparison - same source files across particles2SNR, clean, and YOLO v3",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.965,
        "YOLO v3 rows use source-preserved filenames; split may differ from the particles2SNR test split.",
        ha="center",
        va="top",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path = output_dir / "focused_4um_impact_3row.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return rows


def build_manifest_rows(rows: list[dict]) -> list[dict]:
    manifest_rows = []
    for row in rows:
        signal_path = Path(row["signal_path"])
        split = signal_path.parent.parent.name if signal_path.parent.name == "signals" else ""
        labels = read_yolo_labels(signal_path.parent.parent / "labels" / f"{signal_path.stem}.txt", DATASETS[row["dataset"]]["classes"])
        manifest_rows.append({
            "comparison_type": row.get("comparison_type", ""),
            "split": split,
            "class": class_for_yolo(labels),
            "filename": row.get("filename", row.get("signal_name", "")),
            "dataset": row["dataset"],
            "signal_name": row["signal_name"],
            "signal_path": row["signal_path"],
            "label_count": row["label_count"],
            "overlap_max": row["overlap_max"],
            "peak_group_count": row["peak_group_count"],
            "mean_abs": row["mean_abs"],
            "min": row["min"],
            "max": row["max"],
        })
    return manifest_rows


def write_manifest(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "comparison_type", "split", "class", "filename", "dataset", "signal_name", "signal_path",
        "label_count", "overlap_max", "peak_group_count", "mean_abs", "min", "max",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(build_manifest_rows(rows))


def apply_dataset_overrides(args: argparse.Namespace) -> None:
    if args.left_root is not None:
        DATASETS[DATASET_C1]["root"] = args.left_root
    if args.right_root is not None:
        DATASETS[DATASET_CLEAN]["root"] = args.right_root
    if args.left_label:
        DATASETS[DATASET_C1]["display"] = args.left_label
    if args.right_label:
        DATASETS[DATASET_CLEAN]["display"] = args.right_label
    if args.left_train_json is not None:
        DATASETS[DATASET_C1]["particles2SNR_jsons"]["train"] = args.left_train_json
    if args.left_test_json is not None:
        DATASETS[DATASET_C1]["particles2SNR_jsons"]["test"] = args.left_test_json
    if args.right_train_json is not None:
        DATASETS[DATASET_CLEAN]["particles2SNR_jsons"]["train"] = args.right_train_json
    if args.right_test_json is not None:
        DATASETS[DATASET_CLEAN]["particles2SNR_jsons"]["test"] = args.right_test_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=RESULTS_FIGURES / "visual_signal_checks" / "dataset_comparison")
    parser.add_argument("--source-manifest", type=Path, default=RESULTS_FIGURES / "visual_signal_checks" / "visual_signal_checks_manifest.csv")
    parser.add_argument("--fs", type=float, default=2_000_000.0)
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--splits", default="test")
    parser.add_argument("--classes", default="2um,4um,10um")
    parser.add_argument("--focused-4um-impact", action="store_true")
    parser.add_argument("--left-root", type=Path)
    parser.add_argument("--right-root", type=Path)
    parser.add_argument("--left-label")
    parser.add_argument("--right-label")
    parser.add_argument("--left-train-json", type=Path)
    parser.add_argument("--left-test-json", type=Path)
    parser.add_argument("--right-train-json", type=Path)
    parser.add_argument("--right-test-json", type=Path)
    parser.add_argument("--skip-yolo-reference", action="store_true")
    args = parser.parse_args()

    apply_dataset_overrides(args)
    splits = tuple(item.strip() for item in args.splits.split(",") if item.strip())
    classes = tuple(item.strip() for item in args.classes.split(",") if item.strip())
    previous_selection = load_previous_selection(args.source_manifest)
    metadata_by_dataset = {
        name: load_particles2SNR_metadata(config["particles2SNR_jsons"])
        for name, config in DATASETS.items()
    }
    rows = []
    overview_pairs = []

    if args.focused_4um_impact:
        focused_rows = plot_focused_4um_impact(args.output_dir, args.fs, metadata_by_dataset)
        manifest_path = args.output_dir / "focused_4um_impact_manifest.csv"
        write_manifest(manifest_path, focused_rows)
        print(f"Wrote {len(focused_rows)} focused manifest rows to {manifest_path}")
        return

    for split in splits:
        for class_name in classes:
            names = select_common_samples(split, class_name, previous_selection, args.max_samples)
            paired = []
            for name in names:
                c1_sample = load_sample(DATASET_C1, split, name)
                clean_sample = load_sample(DATASET_CLEAN, split, name)
                if c1_sample and clean_sample:
                    paired.append((name, c1_sample, clean_sample))
            if paired:
                rows.extend(plot_aligned_comparison(
                    paired,
                    f"{DATASETS[DATASET_C1]['display']} vs {DATASETS[DATASET_CLEAN]['display']} | same source filename | {split} | {class_name}",
                    args.output_dir / f"{split}_{class_name}_comparison.png",
                    args.fs,
                    metadata_by_dataset,
                ))
                overview_pairs.extend(paired[:1])

            if not args.skip_yolo_reference:
                reference_samples = collect_by_class(DATASET_YOLO_V3, split).get(class_name, [])[: args.max_samples]
                rows.extend(plot_reference(
                    reference_samples,
                    f"YOLO v3 source-named | source-preserved filenames | {split} | {class_name}",
                    args.output_dir / f"yolo_v3_reference_{split}_{class_name}.png",
                    args.fs,
                ))

    if overview_pairs:
        rows.extend(plot_aligned_comparison(
            overview_pairs[: args.max_samples],
            f"{DATASETS[DATASET_C1]['display']} vs {DATASETS[DATASET_CLEAN]['display']} | overview | same source filenames",
            args.output_dir / "overview_comparison.png",
            args.fs,
            metadata_by_dataset,
        ))

    manifest_path = args.output_dir / "dataset_comparison_manifest.csv"
    write_manifest(manifest_path, rows)
    print(f"Wrote {len(rows)} manifest rows to {manifest_path}")


if __name__ == "__main__":
    main()
