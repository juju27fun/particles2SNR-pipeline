"""Compare particles2SNR particle SNR distributions across several datasets."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_CLASSES = ("2um", "4um", "10um")


def parse_classes(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_dataset_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected LABEL=PATH")
    label, path = value.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise argparse.ArgumentTypeError("Expected non-empty LABEL=PATH")
    return label, Path(path)


def as_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def read_snr_by_class(path: Path, classes: tuple[str, ...]) -> dict[str, list[float]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    class_filter = set(classes)
    grouped = {class_name: [] for class_name in classes}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            class_name = row.get("class")
            if class_name not in class_filter:
                continue
            snr = as_float(row.get("snr_db"))
            if snr is not None:
                grouped[class_name].append(snr)
    return grouped


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def summarize(label: str, grouped: dict[str, list[float]], classes: tuple[str, ...]) -> list[dict]:
    rows = []
    for class_name in classes:
        values = grouped.get(class_name, [])
        if not values:
            rows.append({
                "dataset": label,
                "class": class_name,
                "n_particles": 0,
                "mean_snr_db": "",
                "median_snr_db": "",
                "p10_snr_db": "",
                "p90_snr_db": "",
            })
            continue
        arr = np.asarray(values, dtype=float)
        rows.append({
            "dataset": label,
            "class": class_name,
            "n_particles": int(len(arr)),
            "mean_snr_db": float(np.mean(arr)),
            "median_snr_db": float(np.median(arr)),
            "p10_snr_db": percentile(values, 10),
            "p90_snr_db": percentile(values, 90),
        })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "class",
        "n_particles",
        "mean_snr_db",
        "median_snr_db",
        "p10_snr_db",
        "p90_snr_db",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_snr(
    output_base: Path,
    dataset_values: dict[str, dict[str, list[float]]],
    summary_rows: list[dict],
    classes: tuple[str, ...],
    threshold_db: float | None,
) -> None:
    labels = list(dataset_values)
    colors = ["#4c72b0", "#dd8452", "#55a868", "#8172b2", "#c44e52"]
    fig, axes = plt.subplots(1, len(classes), figsize=(4.2 * len(classes), 5.0), sharey=True)
    if len(classes) == 1:
        axes = [axes]

    for ax, class_name in zip(axes, classes):
        data = [dataset_values[label][class_name] for label in labels]
        positions = np.arange(1, len(labels) + 1)
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=0.55,
            patch_artist=True,
            showmeans=True,
            tick_labels=labels,
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        if threshold_db is not None:
            ax.axhline(threshold_db, color="#c44e52", linestyle="--", linewidth=1.2)
            ax.text(
                0.02,
                threshold_db,
                f" {threshold_db:g} dB",
                transform=ax.get_yaxis_transform(),
                va="bottom",
                ha="left",
                fontsize=8,
                color="#9c2f32",
            )
        ax.set_title(class_name)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        for idx, values in enumerate(data, start=1):
            ax.text(idx, 0.98, f"n={len(values)}", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8)

    axes[0].set_ylabel("SNR (dB)")
    fig.suptitle("particles2SNR particle SNR by class and dataset", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    fig.savefig(output_base.with_suffix(".pdf"))
    fig.savefig(output_base.with_suffix(".png"), dpi=180)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare class-wise SNR distributions from particles2SNR snr_particles.csv files.")
    parser.add_argument("--dataset", action="append", type=parse_dataset_spec, required=True, help="Dataset spec as LABEL=PATH_TO_snr_particles.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", default="snr_by_class_3way")
    parser.add_argument("--classes", type=parse_classes, default=DEFAULT_CLASSES)
    parser.add_argument("--threshold-db", type=float, default=-10.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    classes = tuple(args.classes)
    dataset_values = {}
    summary_rows = []
    for label, path in args.dataset:
        grouped = read_snr_by_class(path, classes)
        dataset_values[label] = grouped
        summary_rows.extend(summarize(label, grouped, classes))

    output_dir = Path(args.output_dir)
    output_base = output_dir / args.output_name
    write_csv(output_base.with_name(f"{args.output_name}_summary.csv"), summary_rows)
    plot_snr(output_base, dataset_values, summary_rows, classes, args.threshold_db)
    print(f"Wrote SNR comparison to {output_base.with_suffix('.pdf')}")
    print(f"Wrote SNR summary to {output_base.with_name(f'{args.output_name}_summary.csv')}")


if __name__ == "__main__":
    main()
