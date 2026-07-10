#!/usr/bin/env python3
"""Evaluate classifier accuracy as a function of SNR for particles2SNR YOLO events."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from particles2snr.repo_paths import RESULTS_RUNS


from p0.data import AdaptiveBandpassDecimate
from p0.models import create_model
from p0.snr_utils import load_model_weights, macro_f1


DEFAULT_CLASSES = ("2um", "4um", "10um")
DEFAULT_TARGETS = (0.85, 0.90, 0.95, 0.97)


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


def read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def infer_dataset_label(data_json_paths: list[Path], output_dir: Path) -> str:
    if data_json_paths:
        path = data_json_paths[0]
        if len(path.parents) >= 2:
            return path.parents[1].name
    return output_dir.name


def crop_centered(signal: np.ndarray, center_sample: int, length: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("Crop length must be positive")
    arr = np.asarray(signal, dtype=np.float32).reshape(-1)
    start = int(center_sample) - length // 2
    end = start + length
    out = np.zeros(length, dtype=np.float32)
    src_start = max(0, start)
    src_end = min(len(arr), end)
    if src_end > src_start:
        dst_start = src_start - start
        out[dst_start:dst_start + (src_end - src_start)] = arr[src_start:src_end]
    return out


def event_rows_from_data_json(path: Path, class_names: tuple[str, ...]) -> list[dict]:
    data = read_json(path)
    split = path.parent.name
    rows = []
    id_to_name = {idx: name for idx, name in enumerate(class_names)}
    for row in data.get("data", []):
        signal_path = row.get("path")
        length = int(row.get("length") or 0)
        if not signal_path or length <= 0:
            continue
        for ann in row.get("annotations", []):
            class_id = int(ann.get("class_id", -1))
            true_class = id_to_name.get(class_id)
            snr_db = as_float(ann.get("snr_db"))
            if true_class is None or snr_db is None:
                continue
            start = float(ann.get("start", 0.0))
            end = float(ann.get("end", start))
            center = float(ann.get("center", ann.get("mean", (start + end) / 2.0)))
            rows.append({
                "split": split,
                "filename": row.get("filename"),
                "path": signal_path,
                "row_id": int(row.get("id", -1)),
                "annotation_id": int(ann.get("id", len(rows))),
                "true_class": true_class,
                "true_class_id": class_id,
                "snr_db": float(snr_db),
                "center": center,
                "start": start,
                "end": end,
                "signal_length": length,
                "frequency": ann.get("frequency"),
                "passage_time_ms": ann.get("passage_time_ms"),
                "peak_group_id": ann.get("peak_group_id"),
                "boundary_adjusted": bool(ann.get("boundary_adjusted", False)),
            })
    return rows


def preprocess_batch(crops: np.ndarray, args: argparse.Namespace) -> torch.Tensor:
    tensor = torch.from_numpy(crops.astype(np.float32))
    if args.preprocess == "adaptive-bandpass":
        transform = AdaptiveBandpassDecimate(
            target_length=args.input_length,
            native_length=args.crop_native_length,
            native_fs_hz=args.sample_rate_mhz * 1_000_000.0,
            low_khz=args.bandpass_low_khz,
            high_khz_max=args.bandpass_high_khz,
        )
        tensor = transform(tensor)
    elif args.preprocess == "none":
        if args.crop_native_length != args.input_length:
            raise ValueError("--preprocess none requires --crop-native-length == --input-length")
    else:
        raise ValueError(f"Unsupported preprocess mode: {args.preprocess}")
    return tensor.unsqueeze(1)


def predict_events(events: list[dict], args: argparse.Namespace) -> list[dict]:
    class_names = tuple(args.class_names)
    model = create_model(args.model_name, input_length=args.input_length, num_classes=len(class_names))
    load_info = load_model_weights(model, args.checkpoint, strict=args.strict_checkpoint)
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    rows = []
    signal_cache: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for offset in range(0, len(events), args.batch_size):
            batch_events = events[offset:offset + args.batch_size]
            crops = []
            for event in batch_events:
                path = event["path"]
                if path not in signal_cache:
                    signal_cache[path] = np.load(path).astype(np.float32)
                center_sample = int(round(float(event["center"]) * int(event["signal_length"])))
                crops.append(crop_centered(signal_cache[path], center_sample, args.crop_native_length))
            inputs = preprocess_batch(np.stack(crops, axis=0), args).to(device)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            pred_ids = np.argmax(probs, axis=1)
            for event, pred_id, prob in zip(batch_events, pred_ids, probs):
                pred_class = class_names[int(pred_id)]
                center_ms = float(event["center"]) * int(event["signal_length"]) / args.fs * 1000.0
                start_ms = float(event["start"]) * int(event["signal_length"]) / args.fs * 1000.0
                end_ms = float(event["end"]) * int(event["signal_length"]) / args.fs * 1000.0
                out = {
                    **event,
                    "dataset_label": args.dataset_label,
                    "event_key": f"{event['split']}:{event['filename']}:{event['annotation_id']}",
                    "center_ms": center_ms,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "width_ms": max(0.0, end_ms - start_ms),
                    "pred_class": pred_class,
                    "pred_class_id": int(pred_id),
                    "correct": pred_class == event["true_class"],
                    "checkpoint": os.path.abspath(args.checkpoint),
                    "model_name": args.model_name,
                    "preprocess": args.preprocess,
                    "checkpoint_loaded_keys": int(load_info["loaded_keys"]),
                    "checkpoint_skipped_keys": int(len(load_info["skipped_keys"])),
                }
                for idx, name in enumerate(class_names):
                    out[f"prob_{name}"] = float(prob[idx])
                rows.append(out)
    return rows


def make_bins(values: list[float], n_bins: int, bin_width: float | None = None) -> list[tuple[float, float]]:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return []
    if bin_width is not None and bin_width > 0:
        lo = math.floor(float(np.min(arr)) / bin_width) * bin_width
        hi = math.ceil(float(np.max(arr)) / bin_width) * bin_width
        edges = np.arange(lo, hi + bin_width, bin_width)
    else:
        edges = np.unique(np.quantile(arr, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) < 2:
        edges = np.asarray([float(arr[0]) - 0.5, float(arr[0]) + 0.5])
    return [(float(edges[i]), float(edges[i + 1])) for i in range(len(edges) - 1)]


def bin_predictions(rows: list[dict], bins: list[tuple[float, float]]) -> list[dict]:
    out = []
    labels = sorted({row["true_class"] for row in rows})
    for idx, (left, right) in enumerate(bins):
        if idx == len(bins) - 1:
            subset = [row for row in rows if left <= float(row["snr_db"]) <= right]
        else:
            subset = [row for row in rows if left <= float(row["snr_db"]) < right]
        if not subset:
            continue
        stat = {
            "bin_idx": idx,
            "snr_left": left,
            "snr_right": right,
            "snr_center": float(np.mean([left, right])),
            "n": len(subset),
            "accuracy": float(np.mean([bool(row["correct"]) for row in subset])),
            "macro_f1": macro_f1([
                {
                    "y_true": row["true_class"],
                    "y_pred": row["pred_class"],
                    "correct": bool(row["correct"]),
                }
                for row in subset
            ]),
        }
        for label in labels:
            cls = [row for row in subset if row["true_class"] == label]
            stat[f"recall_{label}"] = float(np.mean([bool(row["correct"]) for row in cls])) if cls else None
        out.append(stat)
    return out


def threshold_at_target_accuracy(bin_stats: list[dict], target: float) -> float | None:
    if len(bin_stats) < 2:
        return None
    x = np.asarray([row["snr_center"] for row in bin_stats], dtype=float)
    y = np.asarray([row["accuracy"] for row in bin_stats], dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if np.all(y >= target):
        return float(x[0])
    if np.all(y < target):
        return None
    for idx in range(len(y) - 1):
        if y[idx] >= target:
            return float(x[idx])
        if y[idx] < target <= y[idx + 1]:
            if y[idx + 1] == y[idx]:
                return float(x[idx + 1])
            frac = (target - y[idx]) / (y[idx + 1] - y[idx])
            return float(x[idx] + frac * (x[idx + 1] - x[idx]))
    return float(x[-1])


def derivative_threshold(bin_stats: list[dict], derivative_frac: float = 0.2) -> dict:
    if len(bin_stats) < 3:
        return {"threshold_db": None, "method": "not_enough_bins", "derivatives": []}
    x = np.asarray([row["snr_center"] for row in bin_stats], dtype=float)
    y = np.asarray([row["accuracy"] for row in bin_stats], dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    y_smooth = np.convolve(np.pad(y, (1, 1), mode="edge"), np.ones(3) / 3.0, mode="valid") if len(y) >= 5 else y
    deriv = np.diff(y_smooth) / np.maximum(np.diff(x), 1e-9)
    positive = deriv[deriv > 0]
    if len(positive) == 0:
        return {"threshold_db": float(x[0]), "method": "flat_or_decreasing_curve", "derivatives": deriv.tolist()}
    cutoff = float(np.max(positive)) * derivative_frac
    peak_idx = int(np.argmax(deriv))
    threshold = None
    for idx in range(peak_idx + 1, len(deriv)):
        if deriv[idx] <= cutoff:
            threshold = float(x[idx + 1])
            break
    if threshold is None:
        best = float(np.max(y_smooth))
        candidates = x[y_smooth >= 0.95 * best]
        threshold = float(candidates[0]) if len(candidates) else float(x[-1])
    return {
        "threshold_db": threshold,
        "method": "post_peak_derivative_slowdown",
        "derivative_fraction": derivative_frac,
        "derivatives": deriv.tolist(),
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


def plot_curve(bin_stats: list[dict], thresholds: dict, target_to_plot: float, output_path: Path) -> None:
    x = [row["snr_center"] for row in bin_stats]
    y = [row["accuracy"] for row in bin_stats]
    n = [row["n"] for row in bin_stats]
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(x, y, marker="o", label="event accuracy")
    for xi, yi, ni in zip(x, y, n):
        ax.text(xi, yi, str(ni), fontsize=8, ha="center", va="bottom")
    target_key = f"{target_to_plot:.2f}"
    target_info = thresholds.get("target_accuracy", {}).get(target_key, {})
    target_db = target_info.get("threshold_db")
    if target_db is not None:
        ax.axvline(float(target_db), linestyle="--", color="tab:red", label=f"target {target_to_plot:.2f}: {target_db:.2f} dB")
    else:
        ax.text(0.02, 0.05, f"target {target_to_plot:.2f} not reached", transform=ax.transAxes, color="tab:red")
    ax.axhline(target_to_plot, linestyle=":", color="tab:red", alpha=0.8)
    deriv_db = thresholds.get("derivative", {}).get("threshold_db")
    if deriv_db is not None:
        ax.axvline(float(deriv_db), linestyle="--", color="tab:gray", alpha=0.7, label=f"derivative: {deriv_db:.2f} dB")
    ax.set_xlabel("Event SNR (dB)")
    ax.set_ylabel("Classification accuracy")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Event-level accuracy-vs-SNR for particles2SNR YOLO annotations.")
    parser.add_argument("--data-json", action="append", type=Path,
                        default=[RESULTS_RUNS / "p0_c1_Particles2SNR_F" / "test" / "data.json"])
    parser.add_argument("--checkpoint", default="artifacts/SMI_CNN_limitations/particles2SNR_c1_conv1dgap_retrained/checkpoints/Conv1DGAP-L-L4096-decim-dataset_particles2SNR_c1-tier1-seed42/best_model.pth")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_RUNS / "p0_c1_Particles2SNR_F" / "test" / "event_conv1dgap_snr")
    parser.add_argument("--dataset-label", default=None,
                        help="Label written to event_predictions.csv for downstream comparisons")
    parser.add_argument("--model-name", default="Conv1DGAP-L")
    parser.add_argument("--class-names", type=parse_csv_arg, default=DEFAULT_CLASSES)
    parser.add_argument("--input-length", type=int, default=4096)
    parser.add_argument("--crop-native-length", type=int, default=16384)
    parser.add_argument("--preprocess", choices=("adaptive-bandpass", "none"), default="adaptive-bandpass")
    parser.add_argument("--bandpass-low-khz", type=float, default=5.0)
    parser.add_argument("--bandpass-high-khz", type=float, default=100.0)
    parser.add_argument("--sample-rate-mhz", type=float, default=2.0)
    parser.add_argument("--fs", type=float, default=2_000_000.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--bin-width", type=float, default=None)
    parser.add_argument("--targets", type=parse_csv_arg, default=tuple(str(v) for v in DEFAULT_TARGETS))
    parser.add_argument("--plot-target", type=float, default=0.97)
    parser.add_argument("--derivative-frac", type=float, default=0.2)
    parser.add_argument("--strict-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.crop_native_length % args.input_length != 0 and args.preprocess == "adaptive-bandpass":
        raise ValueError("--crop-native-length must be divisible by --input-length for adaptive-bandpass")
    if args.dataset_label is None:
        args.dataset_label = infer_dataset_label(args.data_json, args.output_dir)

    events = []
    for path in args.data_json:
        events.extend(event_rows_from_data_json(path, tuple(args.class_names)))
    if not events:
        raise RuntimeError("No usable event annotations found")

    predictions = predict_events(events, args)
    bins = make_bins([float(row["snr_db"]) for row in predictions], args.bins, args.bin_width)
    bin_stats = bin_predictions(predictions, bins)
    targets = [float(value) for value in args.targets]
    thresholds = {
        "target_accuracy": {
            f"{target:.2f}": {
                "target_accuracy": target,
                "threshold_db": threshold_at_target_accuracy(bin_stats, target),
            }
            for target in targets
        },
        "derivative": derivative_threshold(bin_stats, args.derivative_frac),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = args.output_dir / "event_predictions.csv"
    csv_path = args.output_dir / "event_accuracy_by_snr.csv"
    json_path = args.output_dir / "event_accuracy_by_snr.json"
    pdf_path = args.output_dir / "event_accuracy_by_snr.pdf"
    write_csv(pred_path, predictions)
    write_csv(csv_path, bin_stats)
    plot_curve(bin_stats, thresholds, args.plot_target, pdf_path)

    overall = {
        "n": len(predictions),
        "accuracy": float(np.mean([bool(row["correct"]) for row in predictions])),
        "macro_f1": macro_f1([
            {"y_true": row["true_class"], "y_pred": row["pred_class"], "correct": bool(row["correct"])}
            for row in predictions
        ]),
        "class_counts": dict(Counter(row["true_class"] for row in predictions)),
    }
    report = {
        "description": "Event-level classifier accuracy as a function of particles2SNR annotation SNR",
        "dataset_label": args.dataset_label,
        "data_json": [str(path) for path in args.data_json],
        "checkpoint": os.path.abspath(args.checkpoint),
        "model_name": args.model_name,
        "class_names": list(args.class_names),
        "overall": overall,
        "thresholds": thresholds,
        "bins": bin_stats,
        "outputs": {
            "predictions_csv": str(pred_path),
            "accuracy_csv": str(csv_path),
            "accuracy_pdf": str(pdf_path),
        },
        "preprocessing": {
            "crop_native_length": args.crop_native_length,
            "input_length": args.input_length,
            "preprocess": args.preprocess,
            "bandpass_low_khz": args.bandpass_low_khz,
            "bandpass_high_khz": args.bandpass_high_khz,
            "sample_rate_mhz": args.sample_rate_mhz,
        },
    }
    with json_path.open("w") as f:
        json.dump(report, f, indent=2, default=json_safe, allow_nan=False)

    print(f"Events used: {overall['n']}")
    print(f"Overall accuracy: {overall['accuracy']:.4f}")
    print(f"Target 0.97 threshold: {thresholds['target_accuracy'].get('0.97', {}).get('threshold_db')}")
    print(f"CSV: {csv_path}")
    print(f"PDF: {pdf_path}")


if __name__ == "__main__":
    main()
