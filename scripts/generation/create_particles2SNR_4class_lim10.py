"""Create a 4-class particles2SNR dataset with low-SNR samples marked unclear.

The source dataset is not modified. Files are copied or symlinked into a new
class-folder tree with classes: 2um, 4um, 10um, unclear.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from particles2snr.repo_paths import DATA_DERIVED, RESULTS_FIGURES, RESULTS_RUNS


DEFAULT_CLASSES = ("2um", "4um", "10um")
UNCLEAR_CLASS = "unclear"
ASSIGNED_CLASS_COLORS = {
    "2um": "#1f77b4",
    "4um": "#d62728",
    "10um": "#2ca02c",
    UNCLEAR_CLASS: "#9467bd",
}


def parse_csv_arg(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def as_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def read_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_snr_lookup(snr_csv: Path) -> dict[str, dict]:
    if not snr_csv.is_file():
        raise FileNotFoundError(f"Missing SNR CSV: {snr_csv}")

    grouped = defaultdict(list)
    class_by_file = {}
    for row in read_csv_rows(snr_csv):
        filename = row.get("filename")
        snr = as_float(row.get("snr_db"))
        if not filename or snr is None:
            continue
        grouped[filename].append(snr)
        if row.get("class"):
            class_by_file[filename] = row["class"]

    lookup = {}
    for filename, values in grouped.items():
        lookup[filename] = {
            "median_snr_db": float(np.median(np.asarray(values, dtype=float))),
            "mean_snr_db": float(np.mean(np.asarray(values, dtype=float))),
            "min_snr_db": float(np.min(np.asarray(values, dtype=float))),
            "max_snr_db": float(np.max(np.asarray(values, dtype=float))),
            "n_particles_with_snr": int(len(values)),
            "snr_class": class_by_file.get(filename, ""),
        }
    return lookup


def iter_source_files(input_root: Path, split: str, classes: tuple[str, ...]):
    split_dir = input_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")
    for class_name in classes:
        class_dir = split_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        for path in sorted(class_dir.glob("*.npy")):
            yield class_name, path


def choose_assignment(
    original_class: str,
    filename: str,
    snr_lookup: dict[str, dict],
    threshold_db: float,
) -> tuple[str, str, dict]:
    info = snr_lookup.get(filename)
    if info is None:
        return UNCLEAR_CLASS, "missing_snr", {
            "median_snr_db": None,
            "mean_snr_db": None,
            "min_snr_db": None,
            "max_snr_db": None,
            "n_particles_with_snr": 0,
            "snr_class": "",
        }
    if info["median_snr_db"] < threshold_db:
        return UNCLEAR_CLASS, "median_snr_below_threshold", info
    return original_class, "median_snr_at_or_above_threshold", info


def materialize_file(source_path: Path, output_path: Path, link_mode: str, dry_run: bool) -> str:
    if dry_run:
        return "dry_run"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() or output_path.is_symlink():
        output_path.unlink()

    if link_mode == "symlink":
        try:
            output_path.symlink_to(source_path.resolve())
            return "symlinked"
        except OSError:
            shutil.copy2(source_path, output_path)
            return "copied_symlink_failed"
    if link_mode == "copy":
        shutil.copy2(source_path, output_path)
        return "copied"
    raise ValueError(f"Unsupported link mode: {link_mode}")


def build_split(
    input_root: Path,
    output_root: Path,
    artifact_root: Path,
    split: str,
    classes: tuple[str, ...],
    threshold_db: float,
    link_mode: str,
    dry_run: bool,
) -> list[dict]:
    snr_lookup = build_snr_lookup(artifact_root.parent / split / "snr_particles.csv")
    rows = []

    for original_class, source_path in iter_source_files(input_root, split, classes):
        assigned_class, reason, info = choose_assignment(
            original_class,
            source_path.name,
            snr_lookup,
            threshold_db,
        )
        output_path = output_root / split / assigned_class / source_path.name
        action = materialize_file(source_path, output_path, link_mode, dry_run)
        rows.append({
            "split": split,
            "filename": source_path.name,
            "source_path": str(source_path),
            "output_path": str(output_path),
            "original_class": original_class,
            "assigned_class": assigned_class,
            "median_snr_db": info["median_snr_db"],
            "mean_snr_db": info["mean_snr_db"],
            "min_snr_db": info["min_snr_db"],
            "max_snr_db": info["max_snr_db"],
            "n_particles_with_snr": info["n_particles_with_snr"],
            "snr_threshold_db": threshold_db,
            "assignment_reason": reason,
            "file_action": action,
        })
    return rows


def percentile(values: list[float], q: float) -> float | None:
    clean = [value for value in values if value is not None and math.isfinite(value)]
    if not clean:
        return None
    return float(np.percentile(np.asarray(clean, dtype=float), q))


def summarize(rows: list[dict], args: argparse.Namespace) -> dict:
    by_split_assigned = Counter((row["split"], row["assigned_class"]) for row in rows)
    by_split_original = Counter((row["split"], row["original_class"]) for row in rows)
    unclear_by_origin = Counter(
        (row["split"], row["original_class"])
        for row in rows
        if row["assigned_class"] == UNCLEAR_CLASS
    )
    reason_counts = Counter((row["split"], row["assignment_reason"]) for row in rows)

    split_summary = {}
    for split in args.splits:
        split_rows = [row for row in rows if row["split"] == split]
        unclear = sum(row["assigned_class"] == UNCLEAR_CLASS for row in split_rows)
        split_summary[split] = {
            "total_files": len(split_rows),
            "unclear_files": unclear,
            "unclear_fraction": unclear / len(split_rows) if split_rows else None,
            "assigned_class_counts": {
                class_name: by_split_assigned[(split, class_name)]
                for class_name in (*args.classes, UNCLEAR_CLASS)
            },
            "original_class_counts": {
                class_name: by_split_original[(split, class_name)]
                for class_name in args.classes
            },
            "unclear_by_original_class": {
                class_name: unclear_by_origin[(split, class_name)]
                for class_name in args.classes
            },
            "assignment_reason_counts": {
                reason: count
                for (reason_split, reason), count in reason_counts.items()
                if reason_split == split
            },
        }

    snr_summary = {}
    for group_name in ("original_class", "assigned_class"):
        grouped = defaultdict(list)
        for row in rows:
            value = as_float(row.get("median_snr_db"))
            if value is not None:
                grouped[row[group_name]].append(value)
        snr_summary[group_name] = {
            key: {
                "n": len(values),
                "median": percentile(values, 50),
                "p10": percentile(values, 10),
                "p90": percentile(values, 90),
                "min": percentile(values, 0),
                "max": percentile(values, 100),
            }
            for key, values in sorted(grouped.items())
        }

    return {
        "dataset_name": args.dataset_name,
        "input_root": str(Path(args.input_root).resolve()),
        "output_root": str(Path(args.output_root).resolve()),
        "artifact_root": str(Path(args.artifact_root).resolve()),
        "source_particles2SNR_output_root": str(Path(args.source_particles2SNR_output_root).resolve()),
        "classes": list(args.classes),
        "assigned_classes": [*args.classes, UNCLEAR_CLASS],
        "splits": list(args.splits),
        "snr_threshold_db": args.snr_threshold_db,
        "snr_aggregation": "median_snr_db_per_file",
        "link_mode": args.link_mode,
        "dry_run": args.dry_run,
        "split_summary": split_summary,
        "snr_summary": snr_summary,
    }


def write_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(summary, f, indent=2)


def snr_values(rows: list[dict], group_field: str, group_name: str) -> list[float]:
    values = []
    for row in rows:
        if row.get(group_field) != group_name:
            continue
        value = as_float(row.get("median_snr_db"))
        if value is not None:
            values.append(value)
    return values


def save_assigned_class_counts(summary: dict, figure_root: Path) -> Path:
    splits = list(summary["splits"])
    classes = list(summary["assigned_classes"])
    x = np.arange(len(splits))
    bottom = np.zeros(len(splits))

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for class_name in classes:
        counts = [
            summary["split_summary"][split]["assigned_class_counts"].get(class_name, 0)
            for split in splits
        ]
        ax.bar(
            x,
            counts,
            bottom=bottom,
            label=class_name,
            color=ASSIGNED_CLASS_COLORS.get(class_name),
        )
        bottom += np.asarray(counts, dtype=float)

    ax.set_title("Assigned class counts by split")
    ax.set_ylabel("Files")
    ax.set_xticks(x, splits)
    ax.legend(frameon=False, ncols=min(4, len(classes)))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    output_path = figure_root / "assigned_class_counts_by_split.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_unclear_fraction(summary: dict, figure_root: Path) -> Path:
    splits = list(summary["splits"])
    classes = list(summary["classes"])
    x = np.arange(len(classes))
    width = 0.8 / max(1, len(splits))

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for idx, split in enumerate(splits):
        values = []
        split_info = summary["split_summary"][split]
        for class_name in classes:
            original = split_info["original_class_counts"].get(class_name, 0)
            unclear = split_info["unclear_by_original_class"].get(class_name, 0)
            values.append((unclear / original * 100.0) if original else 0.0)
        offset = (idx - (len(splits) - 1) / 2) * width
        ax.bar(x + offset, values, width=width, label=split)

    ax.set_title("Unclear fraction by original class")
    ax.set_ylabel("Files assigned unclear (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(x, classes)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    output_path = figure_root / "unclear_fraction_by_original_class.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_snr_boxplot(
    rows: list[dict],
    figure_root: Path,
    group_field: str,
    classes: list[str],
    threshold_db: float,
    output_name: str,
    title: str,
) -> Path:
    values = [snr_values(rows, group_field, class_name) for class_name in classes]
    labels = [
        f"{class_name}\n(n={len(class_values)})"
        for class_name, class_values in zip(classes, values)
    ]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    nonempty_positions = [idx + 1 for idx, class_values in enumerate(values) if class_values]
    nonempty_values = [class_values for class_values in values if class_values]
    if nonempty_values:
        ax.boxplot(
            nonempty_values,
            positions=nonempty_positions,
            widths=0.55,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "#d9e8f5", "edgecolor": "#4c6272"},
            medianprops={"color": "#111111", "linewidth": 1.5},
            whiskerprops={"color": "#4c6272"},
            capprops={"color": "#4c6272"},
        )
    ax.axhline(threshold_db, color="#b03a2e", linestyle="--", linewidth=1.2, label="threshold")
    ax.set_title(title)
    ax.set_ylabel("Median SNR per file (dB)")
    ax.set_xticks(np.arange(1, len(classes) + 1), labels)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    output_path = figure_root / output_name
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def generate_plots(rows: list[dict], summary: dict, figure_root: Path) -> list[Path]:
    figure_root.mkdir(parents=True, exist_ok=True)
    classes = list(summary["classes"])
    assigned_classes = list(summary["assigned_classes"])
    threshold_db = float(summary["snr_threshold_db"])

    return [
        save_assigned_class_counts(summary, figure_root),
        save_unclear_fraction(summary, figure_root),
        save_snr_boxplot(
            rows,
            figure_root,
            "original_class",
            classes,
            threshold_db,
            "median_snr_by_original_class.png",
            "Median SNR by original class",
        ),
        save_snr_boxplot(
            rows,
            figure_root,
            "assigned_class",
            assigned_classes,
            threshold_db,
            "median_snr_by_assigned_class.png",
            "Median SNR by assigned class",
        ),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create particles2SNR_4_class_lim10 by assigning low-SNR files to unclear."
    )
    parser.add_argument("--dataset-name", default="particles2SNR_4_class_lim10")
    parser.add_argument("--input-root", default="datasets/interim/particles2SNR-pipeline/particles2snr-c1")
    parser.add_argument("--output-root", default="datasets/interim/particles2SNR-pipeline/particles2snr-4class-lim10-candidate")
    parser.add_argument(
        "--source-particles2SNR-output-root",
        default=str(RESULTS_RUNS / "p0_c1_particles2SNR"),
    )
    parser.add_argument(
        "--artifact-root",
        default=str(DATA_DERIVED / "particles2SNR_4_class_lim10"),
    )
    parser.add_argument(
        "--figure-root",
        default=str(RESULTS_FIGURES / "particles2SNR_4_class_lim10"),
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--splits", type=parse_csv_arg, default=("train", "test"))
    parser.add_argument("--classes", type=parse_csv_arg, default=DEFAULT_CLASSES)
    parser.add_argument("--snr-threshold-db", type=float, default=-10.0)
    parser.add_argument("--link-mode", choices=("symlink", "copy"), default="symlink")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.splits = tuple(args.splits)
    args.classes = tuple(args.classes)

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    artifact_root = Path(args.artifact_root)
    source_particles2SNR_output_root = Path(args.source_particles2SNR_output_root)
    figure_root = Path(args.figure_root)

    all_rows = []
    for split in args.splits:
        split_rows = build_split(
            input_root,
            output_root,
            source_particles2SNR_output_root / split,
            split,
            args.classes,
            args.snr_threshold_db,
            args.link_mode,
            args.dry_run,
        )
        all_rows.extend(split_rows)
        write_csv(
            artifact_root / split / "class_assignment_manifest.csv",
            split_rows,
            [
                "split", "filename", "source_path", "output_path",
                "original_class", "assigned_class", "median_snr_db",
                "mean_snr_db", "min_snr_db", "max_snr_db",
                "n_particles_with_snr", "snr_threshold_db",
                "assignment_reason", "file_action",
            ],
        )

    summary = summarize(all_rows, args)
    write_summary(artifact_root / "class_balance_summary.json", summary)
    plot_paths = []
    if not args.no_plots:
        plot_paths = generate_plots(all_rows, summary, figure_root)

    print(f"Rows: {len(all_rows)}")
    print(f"Dry run: {args.dry_run}")
    for split, info in summary["split_summary"].items():
        unclear_pct = info["unclear_fraction"] * 100 if info["unclear_fraction"] is not None else 0.0
        print(
            f"{split}: total={info['total_files']} unclear={info['unclear_files']} "
            f"({unclear_pct:.1f}%) assigned={info['assigned_class_counts']}"
        )
    print(f"Manifest root: {artifact_root}")
    if plot_paths:
        print(f"Figure root: {figure_root}")
    if not args.dry_run:
        print(f"Dataset root: {output_root}")


if __name__ == "__main__":
    main()
