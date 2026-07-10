#!/usr/bin/env python3
"""Compare event-level classifier accuracy as a function of SNR across pipelines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_CLASSES = ("2um", "4um", "10um")
DEFAULT_TARGETS = (0.85, 0.90, 0.95, 0.97)


def parse_csv_arg(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_run_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must use LABEL=PATH")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("--run label cannot be empty")
    return label, Path(raw_path.strip())


def as_float(value) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def read_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def normalize_prediction_rows(path: Path, label: str) -> list[dict]:
    rows = []
    for row in read_csv_rows(path):
        snr = as_float(row.get("snr_db"))
        true_class = row.get("true_class") or row.get("y_true") or row.get("label")
        pred_class = row.get("pred_class") or row.get("y_pred") or row.get("prediction")
        if snr is None or not true_class or not pred_class:
            continue
        correct_value = row.get("correct")
        correct = as_bool(correct_value) if correct_value not in (None, "") else str(true_class) == str(pred_class)
        rows.append({
            "dataset_label": label,
            "source_predictions_csv": str(path),
            "event_key": row.get("event_key") or f"{row.get('split', '')}:{row.get('filename', '')}:{row.get('annotation_id', len(rows))}",
            "split": row.get("split"),
            "filename": row.get("filename"),
            "annotation_id": row.get("annotation_id"),
            "true_class": str(true_class),
            "pred_class": str(pred_class),
            "snr_db": float(snr),
            "correct": bool(correct),
        })
    return rows


def make_common_bins(rows: list[dict], n_bins: int, bin_width: float | None = None) -> list[tuple[float, float]]:
    values = np.asarray([float(row["snr_db"]) for row in rows], dtype=float)
    if len(values) == 0:
        return []
    if bin_width is not None and bin_width > 0:
        lo = math.floor(float(np.min(values)) / bin_width) * bin_width
        hi = math.ceil(float(np.max(values)) / bin_width) * bin_width
        edges = np.arange(lo, hi + bin_width, bin_width)
    else:
        edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) < 2:
        edges = np.asarray([float(values[0]) - 0.5, float(values[0]) + 0.5])
    return [(float(edges[idx]), float(edges[idx + 1])) for idx in range(len(edges) - 1)]


def rows_in_bin(rows: list[dict], left: float, right: float, is_last: bool) -> list[dict]:
    if is_last:
        return [row for row in rows if left <= float(row["snr_db"]) <= right]
    return [row for row in rows if left <= float(row["snr_db"]) < right]


def stable_seed(seed: int, *parts: object) -> int:
    text = "|".join(str(part) for part in (seed, *parts))
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def sample_balanced_subset(subset: list[dict], classes: tuple[str, ...], seed: int, label: str, bin_idx: int) -> tuple[list[dict], dict, str | None]:
    by_class = {cls: [row for row in subset if row["true_class"] == cls] for cls in classes}
    counts = {cls: len(rows) for cls, rows in by_class.items()}
    min_count = min(counts.values()) if counts else 0
    if min_count <= 0:
        missing = [cls for cls, count in counts.items() if count == 0]
        return [], counts, "missing_class:" + ",".join(missing)
    rng = np.random.default_rng(stable_seed(seed, label, bin_idx))
    sampled = []
    for cls in classes:
        cls_rows = by_class[cls]
        if len(cls_rows) == min_count:
            sampled.extend(cls_rows)
        else:
            indices = rng.choice(len(cls_rows), size=min_count, replace=False)
            sampled.extend(cls_rows[int(idx)] for idx in sorted(indices.tolist()))
    return sampled, counts, None


def macro_f1(rows: list[dict]) -> float:
    labels = sorted({row["true_class"] for row in rows} | {row["pred_class"] for row in rows})
    scores = []
    for label in labels:
        tp = sum(row["true_class"] == label and row["pred_class"] == label for row in rows)
        fp = sum(row["true_class"] != label and row["pred_class"] == label for row in rows)
        fn = sum(row["true_class"] == label and row["pred_class"] != label for row in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def stat_row(rows: list[dict], classes: tuple[str, ...], label: str, bin_idx: int, left: float, right: float, mode: str, source_counts: dict | None = None, skip_reason: str | None = None) -> dict:
    n_by_class = {cls: sum(row["true_class"] == cls for row in rows) for cls in classes}
    if source_counts is None:
        source_counts = n_by_class
    out = {
        "dataset_label": label,
        "mode": mode,
        "bin_idx": bin_idx,
        "snr_left": left,
        "snr_right": right,
        "snr_center": float(np.mean([left, right])),
        "n": len(rows),
        "n_by_class": json.dumps(source_counts, sort_keys=True),
        "sampled_n_by_class": json.dumps(n_by_class, sort_keys=True),
        "skip_reason": skip_reason,
        "accuracy": None,
        "macro_f1": None,
    }
    if rows:
        out["accuracy"] = float(np.mean([bool(row["correct"]) for row in rows]))
        out["macro_f1"] = macro_f1(rows)
    for cls in classes:
        cls_rows = [row for row in rows if row["true_class"] == cls]
        out[f"recall_{cls}"] = float(np.mean([bool(row["correct"]) for row in cls_rows])) if cls_rows else None
    return out


def bin_stats_by_dataset(rows: list[dict], bins: list[tuple[float, float]], classes: tuple[str, ...], seed: int, mode: str) -> list[dict]:
    by_label = defaultdict(list)
    for row in rows:
        by_label[row["dataset_label"]].append(row)
    stats = []
    for label in sorted(by_label):
        label_rows = by_label[label]
        for idx, (left, right) in enumerate(bins):
            subset = rows_in_bin(label_rows, left, right, idx == len(bins) - 1)
            if mode == "available":
                if subset:
                    stats.append(stat_row(subset, classes, label, idx, left, right, mode))
                continue
            sampled, counts, skip_reason = sample_balanced_subset(subset, classes, seed, label, idx)
            if skip_reason:
                stats.append(stat_row([], classes, label, idx, left, right, mode, counts, skip_reason))
            else:
                stats.append(stat_row(sampled, classes, label, idx, left, right, mode, counts))
    return stats


def threshold_at_target_accuracy(bin_stats: list[dict], target: float) -> float | None:
    """Return the first SNR where the target is reached and sustained.

    Accuracy-vs-SNR curves can be noisy. A single early bin above target is not a
    meaningful operating threshold if later, higher-SNR bins fall below target.
    "Sustained" means all usable bins at and above the returned threshold are
    at least the requested target.
    """
    usable = [row for row in bin_stats if row.get("accuracy") is not None]
    if len(usable) < 2:
        return None
    x = np.asarray([row["snr_center"] for row in usable], dtype=float)
    y = np.asarray([row["accuracy"] for row in usable], dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    suffix_ok = np.asarray([np.all(y[idx:] >= target) for idx in range(len(y))], dtype=bool)
    if not np.any(suffix_ok):
        return None
    first_ok = int(np.argmax(suffix_ok))
    if first_ok == 0:
        return float(x[0])
    prev_idx = first_ok - 1
    if y[prev_idx] >= target:
        return float(x[first_ok])
    if y[first_ok] == y[prev_idx]:
        return float(x[first_ok])
    frac = (target - y[prev_idx]) / (y[first_ok] - y[prev_idx])
    return float(x[prev_idx] + frac * (x[first_ok] - x[prev_idx]))


def thresholds_by_dataset(stats: list[dict], targets: tuple[float, ...]) -> dict:
    by_label = defaultdict(list)
    for row in stats:
        by_label[row["dataset_label"]].append(row)
    return {
        label: {
            f"{target:.2f}": {
                "target_accuracy": target,
                "threshold_db": threshold_at_target_accuracy(rows, target),
            }
            for target in targets
        }
        for label, rows in sorted(by_label.items())
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def plot_comparison(stats: list[dict], thresholds: dict, target: float, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    labels = sorted({row["dataset_label"] for row in stats})
    for label in labels:
        rows = sorted(
            [row for row in stats if row["dataset_label"] == label and row.get("accuracy") is not None],
            key=lambda row: row["snr_center"],
        )
        if not rows:
            continue
        x = [row["snr_center"] for row in rows]
        y = [row["accuracy"] for row in rows]
        ax.plot(x, y, marker="o", label=label)
        for xi, yi, row in zip(x, y, rows):
            ax.text(xi, yi, str(row["n"]), fontsize=7, ha="center", va="bottom")
        target_info = thresholds.get(label, {}).get(f"{target:.2f}", {})
        threshold_db = target_info.get("threshold_db")
        if threshold_db is not None:
            ax.axvline(float(threshold_db), linestyle="--", alpha=0.25)
    ax.axhline(target, linestyle=":", color="tab:red", label=f"target {target:.2f}")
    ax.set_xlabel("Event SNR (dB)")
    ax.set_ylabel("Classification accuracy")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare event-level accuracy-vs-SNR curves across prediction CSVs.")
    parser.add_argument("--run", action="append", type=parse_run_arg, required=True,
                        help="Pipeline label and event_predictions.csv path as LABEL=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--classes", type=parse_csv_arg, default=DEFAULT_CLASSES)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--bin-width", type=float, default=None)
    parser.add_argument("--balance", choices=("class-snr",), default="class-snr")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--targets", type=parse_csv_arg, default=tuple(str(v) for v in DEFAULT_TARGETS))
    parser.add_argument("--plot-target", type=float, default=0.97)
    args = parser.parse_args()

    rows = []
    sources = []
    for label, path in args.run:
        source_rows = normalize_prediction_rows(path, label)
        if not source_rows:
            raise RuntimeError(f"No usable prediction rows found in {path}")
        rows.extend(source_rows)
        sources.append({"label": label, "path": str(path), "rows": len(source_rows)})
    bins = make_common_bins(rows, args.bins, args.bin_width)
    if not bins:
        raise RuntimeError("No SNR bins could be built")

    classes = tuple(args.classes)
    targets = tuple(float(value) for value in args.targets)
    balanced = bin_stats_by_dataset(rows, bins, classes, args.seed, "balanced_class_snr")
    available = bin_stats_by_dataset(rows, bins, classes, args.seed, "available")
    balanced_thresholds = thresholds_by_dataset(balanced, targets)
    available_thresholds = thresholds_by_dataset(available, targets)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    balanced_csv = args.output_dir / "event_accuracy_comparison_balanced.csv"
    available_csv = args.output_dir / "event_accuracy_comparison_available.csv"
    balanced_json = args.output_dir / "event_accuracy_comparison_balanced.json"
    available_json = args.output_dir / "event_accuracy_comparison_available.json"
    balanced_pdf = args.output_dir / "event_accuracy_comparison_balanced.pdf"
    available_pdf = args.output_dir / "event_accuracy_comparison_available.pdf"

    write_csv(balanced_csv, balanced)
    write_csv(available_csv, available)
    plot_comparison(balanced, balanced_thresholds, args.plot_target, balanced_pdf, "Balanced event accuracy by SNR")
    plot_comparison(available, available_thresholds, args.plot_target, available_pdf, "Available event accuracy by SNR")

    common = {
        "description": "Event-level accuracy-vs-SNR comparison across particles2SNR pipelines",
        "sources": sources,
        "classes": list(classes),
        "bins": [
            {"bin_idx": idx, "snr_left": left, "snr_right": right, "snr_center": float(np.mean([left, right]))}
            for idx, (left, right) in enumerate(bins)
        ],
        "seed": args.seed,
        "targets": list(targets),
    }
    with balanced_json.open("w") as f:
        json.dump({**common, "mode": "balanced_class_snr", "thresholds": balanced_thresholds, "rows": balanced}, f, indent=2, default=json_safe, allow_nan=False)
    with available_json.open("w") as f:
        json.dump({**common, "mode": "available", "thresholds": available_thresholds, "rows": available}, f, indent=2, default=json_safe, allow_nan=False)

    print(f"Balanced CSV: {balanced_csv}")
    print(f"Balanced PDF: {balanced_pdf}")
    print(f"Available CSV: {available_csv}")
    print(f"Available PDF: {available_pdf}")


if __name__ == "__main__":
    main()
