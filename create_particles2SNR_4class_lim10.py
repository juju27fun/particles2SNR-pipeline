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

import numpy as np


DEFAULT_CLASSES = ("2um", "4um", "10um")
UNCLEAR_CLASS = "unclear"


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create particles2SNR_4_class_lim10 by assigning low-SNR files to unclear."
    )
    parser.add_argument("--dataset-name", default="particles2SNR_4_class_lim10")
    parser.add_argument("--input-root", default="P0/data/dataset_particles2SNR_c1")
    parser.add_argument("--output-root", default="P0/data/particles2SNR_4_class_lim10")
    parser.add_argument("--source-particles2SNR-output-root", default="particles2SNR_pipeline/output/p0_c1_particles2SNR")
    parser.add_argument("--artifact-root", default="particles2SNR_pipeline/output/particles2SNR_4_class_lim10")
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

    print(f"Rows: {len(all_rows)}")
    print(f"Dry run: {args.dry_run}")
    for split, info in summary["split_summary"].items():
        unclear_pct = info["unclear_fraction"] * 100 if info["unclear_fraction"] is not None else 0.0
        print(
            f"{split}: total={info['total_files']} unclear={info['unclear_files']} "
            f"({unclear_pct:.1f}%) assigned={info['assigned_class_counts']}"
        )
    print(f"Manifest root: {artifact_root}")
    if not args.dry_run:
        print(f"Dataset root: {output_root}")


if __name__ == "__main__":
    main()
