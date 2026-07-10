#!/usr/bin/env python3
"""Generate compact PNG signal checks for particles2SNR C1 datasets."""

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

from repo_paths import RESULTS_FIGURES, RESULTS_RUNS


REFERENCE_SELECTION = {
    ("p0_yolo_clean_filt_trainval", "train", "4um"): (
        "HFocusing_5_10_4um_0_1.npy",
        "HFocusing_5_10_4um_0_166.npy",
        "HFocusing_5_10_4um_0_381.npy",
        "HFocusing_5_10_4um_0_995.npy",
    ),
}


DEFAULT_DATASETS = {
    "p0_class_folder_clean_filt": {
        "root": Path("P0/data/processed/dataset_Particles2SNR_F_c1"),
        "kind": "class_folder",
        "classes": ("2um", "4um", "10um"),
    },
    "p0_yolo_clean_filt_trainval": {
        "root": Path("P0/data/processed/dataset_Particles2SNR_F_c1_yolo_trainval"),
        "kind": "yolo",
        "classes": ("2um", "4um", "10um"),
    },
    "p1_yolo_clean_filt_4class_lim10": {
        "root": Path("P1/data/yolo/canonical/particles2snr_f_c1_4class_lim10_trainval"),
        "kind": "yolo",
        "classes": ("2um", "4um", "10um", "unclear"),
    },
}


DEFAULT_PARTICLES2SNR_JSONS = {
    "train": RESULTS_RUNS / "p0_c1_Particles2SNR_F" / "train" / "data.json",
    "test": RESULTS_RUNS / "p0_c1_Particles2SNR_F" / "test" / "data.json",
}
LABEL_COLORS = [
    "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b",
    "#e377c2", "#17becf", "#bcbd22", "#7f7f7f", "#ff7f0e",
]
OVERLAP_STYLES = {
    1: ("#f6b26b", 0.18),
    2: ("#f28e2b", 0.30),
    3: ("#d94801", 0.42),
}


def load_particles2SNR_metadata(particles2SNR_jsons: dict[str, Path]) -> dict[str, dict]:
    metadata = {}
    for split, path in particles2SNR_jsons.items():
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


def enrich_labels_with_particles2SNR(labels: list[dict], split: str, filename: str, metadata: dict[str, dict]) -> tuple[list[dict], list[dict]]:
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


def draw_label_outlines(ax, labels: list[dict], signal_len: int, fs: float, show_ids: bool) -> None:
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
        if show_ids:
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


def add_overlap_legend(ax, labels: list[dict], peak_groups: list[dict]) -> None:
    if not labels:
        return
    handles = [
        plt.Line2D([0], [0], color=OVERLAP_STYLES[1][0], linewidth=5, alpha=OVERLAP_STYLES[1][1], label="overlap=1"),
        plt.Line2D([0], [0], color=OVERLAP_STYLES[2][0], linewidth=5, alpha=OVERLAP_STYLES[2][1], label="overlap=2"),
        plt.Line2D([0], [0], color=OVERLAP_STYLES[3][0], linewidth=5, alpha=OVERLAP_STYLES[3][1], label="overlap>=3"),
    ]
    if peak_groups:
        handles.append(plt.Line2D([0], [0], color="#0057b8", linestyle="--", linewidth=1, label="peak group"))
    ax.legend(handles=handles, loc="upper right", fontsize=7, frameon=False, ncol=min(4, len(handles)))


def load_previous_selection(manifest_path: Path) -> dict[tuple[str, str, str], list[str]]:
    selection: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    if not manifest_path.exists():
        return selection
    with manifest_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            group = row.get("class") or row.get("group")
            signal_path = row.get("signal_path") or row.get("path") or row.get("file")
            if not group or not signal_path:
                continue
            key = (row["dataset"], row["split"], group)
            selection[key].append(Path(signal_path).name)
    for key, names in REFERENCE_SELECTION.items():
        current = selection[key]
        selection[key] = list(names) + [name for name in current if name not in names]
    return selection


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
            labels.append(
                {
                    "class": classes[class_id] if class_id < len(classes) else str(class_id),
                    "start": max(0.0, center - width / 2.0),
                    "end": min(1.0, center + width / 2.0),
                }
            )
    return labels


def class_for_yolo(labels: list[dict]) -> str:
    if not labels:
        return "no_label"
    counts: dict[str, int] = defaultdict(int)
    for label in labels:
        counts[label["class"]] += 1
    return max(sorted(counts), key=counts.get)


def collect_samples(dataset_name: str, config: dict, split: str) -> dict[str, list[dict]]:
    root = config["root"]
    classes = config["classes"]
    by_class: dict[str, list[dict]] = defaultdict(list)
    if config["kind"] == "class_folder":
        for class_name in classes:
            for signal_path in sorted((root / split / class_name).glob("*.npy")):
                by_class[class_name].append({"signal_path": signal_path, "labels": [], "split": split})
    else:
        signals_dir = root / split / "signals"
        labels_dir = root / split / "labels"
        for signal_path in sorted(signals_dir.glob("*.npy")):
            labels = read_yolo_labels(labels_dir / f"{signal_path.stem}.txt", classes)
            by_class[class_for_yolo(labels)].append({"signal_path": signal_path, "labels": labels, "split": split})
    return by_class


def select_samples(
    dataset_name: str,
    split: str,
    class_name: str,
    samples: list[dict],
    previous_selection: dict[tuple[str, str, str], list[str]],
    max_samples: int,
) -> list[dict]:
    by_name = {sample["signal_path"].name: sample for sample in samples}
    selected = [
        by_name[name]
        for name in previous_selection.get((dataset_name, split, class_name), [])
        if name in by_name
    ]
    for sample in samples:
        if len(selected) >= max_samples:
            break
        if sample not in selected:
            selected.append(sample)
    return selected[:max_samples]


def label_summary(labels: list[dict]) -> str:
    if not labels:
        return ""
    counts: dict[str, int] = defaultdict(int)
    for label in labels:
        counts[label["class"]] += 1
    return ", ".join(f"{name}={counts[name]}" for name in sorted(counts))


def plot_group(
    samples: list[dict],
    title: str,
    output_path: Path,
    fs: float,
    particles2SNR_metadata: dict,
    show_overlap_density: bool = True,
    show_peak_groups: bool = True,
    show_label_ids: bool = True,
) -> list[dict]:
    if not samples:
        return []
    fig, axes = plt.subplots(len(samples), 1, figsize=(14, 2.2 * len(samples)), squeeze=False)
    manifest_rows = []
    for ax, sample in zip(axes[:, 0], samples):
        signal = np.load(sample["signal_path"])
        time_ms = np.arange(len(signal)) / fs * 1000.0
        labels, peak_groups = enrich_labels_with_particles2SNR(
            sample["labels"], sample.get("split", ""), sample["signal_path"].name, particles2SNR_metadata
        )
        if show_overlap_density:
            draw_overlap_density(ax, labels, len(signal), fs)
        ax.plot(time_ms, signal, color="#222222", linewidth=0.7, zorder=3)
        # Set limits before drawing labels so text positions are stable.
        if len(signal):
            ymin, ymax = float(np.min(signal)), float(np.max(signal))
            pad = max(1e-6, 0.08 * (ymax - ymin))
            ax.set_ylim(ymin - pad, ymax + pad)
        draw_label_outlines(ax, labels, len(signal), fs, show_label_ids)
        if show_peak_groups:
            draw_peak_groups(ax, peak_groups)
        add_overlap_legend(ax, labels, peak_groups if show_peak_groups else [])
        ax.set_ylabel("amplitude")
        ax.grid(True, alpha=0.18)
        ax.set_title(
            f"{sample['signal_path'].name} | labels: {label_summary(labels)}",
            loc="left",
            fontsize=9,
        )
        manifest_rows.append(
            {
                "signal_name": sample["signal_path"].name,
                "signal_path": str(sample["signal_path"]),
                "mean_abs": f"{float(np.mean(np.abs(signal))):.9g}",
                "min": f"{float(np.min(signal)):.9g}",
                "max": f"{float(np.max(signal)):.9g}",
                "labels": label_summary(labels),
                "peak_groups": str(len(peak_groups)),
            }
        )
    axes[-1, 0].set_xlabel("time (ms)")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return manifest_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=RESULTS_FIGURES / "visual_signal_checks")
    parser.add_argument("--p0-class-folder-root", type=Path, default=DEFAULT_DATASETS["p0_class_folder_clean_filt"]["root"])
    parser.add_argument("--p0-yolo-root", type=Path, default=DEFAULT_DATASETS["p0_yolo_clean_filt_trainval"]["root"])
    parser.add_argument("--p1-yolo-root", type=Path, default=DEFAULT_DATASETS["p1_yolo_clean_filt_4class_lim10"]["root"])
    parser.add_argument("--particles2snr-run-root", type=Path, default=RESULTS_RUNS / "p0_c1_Particles2SNR_F")
    parser.add_argument("--selection-manifest", type=Path, default=None,
                        help="Existing visual_signal_checks_manifest.csv used to reuse the same sampled files.")
    parser.add_argument("--fs", type=float, default=2_000_000.0)
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--show-overlap-density", dest="show_overlap_density", action="store_true", default=True)
    parser.add_argument("--no-show-overlap-density", dest="show_overlap_density", action="store_false")
    parser.add_argument("--show-peak-groups", dest="show_peak_groups", action="store_true", default=True)
    parser.add_argument("--no-show-peak-groups", dest="show_peak_groups", action="store_false")
    parser.add_argument("--show-label-ids", dest="show_label_ids", action="store_true", default=True)
    parser.add_argument("--no-show-label-ids", dest="show_label_ids", action="store_false")
    args = parser.parse_args()

    datasets = {
        "p0_class_folder_clean_filt": {
            **DEFAULT_DATASETS["p0_class_folder_clean_filt"],
            "root": args.p0_class_folder_root,
        },
        "p0_yolo_clean_filt_trainval": {
            **DEFAULT_DATASETS["p0_yolo_clean_filt_trainval"],
            "root": args.p0_yolo_root,
        },
        "p1_yolo_clean_filt_4class_lim10": {
            **DEFAULT_DATASETS["p1_yolo_clean_filt_4class_lim10"],
            "root": args.p1_yolo_root,
        },
    }
    particles2SNR_jsons = {
        "train": args.particles2snr_run_root / "train" / "data.json",
        "test": args.particles2snr_run_root / "test" / "data.json",
    }
    particles2SNR_metadata = load_particles2SNR_metadata(particles2SNR_jsons)
    manifest_path = args.output_dir / "visual_signal_checks_manifest.csv"
    previous_selection = load_previous_selection(args.selection_manifest or manifest_path)
    rows = []

    for dataset_name, config in datasets.items():
        dataset_dir = args.output_dir / dataset_name
        overview_samples = []
        for split in ("train", "val", "test"):
            if not (config["root"] / split).exists():
                continue
            by_class = collect_samples(dataset_name, config, split)
            if config["kind"] == "yolo":
                all_samples = {
                    sample["signal_path"].name: sample
                    for group_samples in by_class.values()
                    for sample in group_samples
                }
                previous_groups = {
                    group
                    for prev_dataset, prev_split, group in previous_selection
                    if prev_dataset == dataset_name and prev_split == split
                }
                for group in previous_groups:
                    existing = {sample["signal_path"].name for sample in by_class[group]}
                    for name in previous_selection.get((dataset_name, split, group), []):
                        if name in all_samples and name not in existing:
                            by_class[group].append(all_samples[name])
                            existing.add(name)
            for class_name in sorted(by_class):
                selected = select_samples(
                    dataset_name, split, class_name, by_class[class_name], previous_selection, args.max_samples
                )
                if not selected:
                    continue
                output_path = dataset_dir / f"{split}_{class_name}.png"
                group_rows = plot_group(
                    selected, f"{dataset_name} | {split} | {class_name}", output_path, args.fs,
                    particles2SNR_metadata, args.show_overlap_density, args.show_peak_groups, args.show_label_ids,
                )
                for row in group_rows:
                    row.update({"dataset": dataset_name, "split": split, "class": class_name})
                    rows.append(row)
                overview_samples.extend(selected[:1])
        plot_group(
            overview_samples[: args.max_samples * 3], f"{dataset_name} | overview", dataset_dir / "overview.png", args.fs,
            particles2SNR_metadata, args.show_overlap_density, args.show_peak_groups, args.show_label_ids,
        )

    fieldnames = ["dataset", "split", "class", "signal_name", "signal_path", "mean_abs", "min", "max", "labels", "peak_groups"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} manifest rows to {manifest_path}")


if __name__ == "__main__":
    main()
