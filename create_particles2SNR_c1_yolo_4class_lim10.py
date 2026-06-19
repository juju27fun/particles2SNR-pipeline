#!/usr/bin/env python3
"""Create a 4-class YOLO detection dataset from particles2SNR C1 annotations.

The source dataset keeps particle classes only.  This converter preserves the
same train/val/test signal files, but rewrites each event label so annotations
with snr_db < -10 dB become class 3, "unclear".
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from collections import Counter
from pathlib import Path

import yaml


CLASS_NAMES = ("2um", "4um", "10um", "unclear")
UNCLEAR_CLASS_ID = 3
SPLITS = ("train", "val", "test")


def load_particles2SNR_jsons(paths: list[Path]) -> tuple[dict[str, dict], dict]:
    rows: dict[str, dict] = {}
    info: dict = {}
    for path in paths:
        with path.open() as f:
            data = json.load(f)
        if not info:
            info = dict(data.get("info", {}))
        for row in data.get("data", []):
            filename = row.get("filename")
            if filename:
                rows[filename] = row
    return rows, info


def load_particles2SNR_annotations(paths: list[Path]) -> dict[str, dict]:
    rows, _ = load_particles2SNR_jsons(paths)
    return rows


def label_for_annotation(annotation: dict, threshold_db: float) -> tuple[int, str]:
    snr = annotation.get("snr_db")
    try:
        snr_value = float(snr)
    except (TypeError, ValueError):
        return UNCLEAR_CLASS_ID, "missing_or_invalid_snr"
    if not math.isfinite(snr_value):
        return UNCLEAR_CLASS_ID, "missing_or_invalid_snr"
    if snr_value < threshold_db:
        return UNCLEAR_CLASS_ID, "below_threshold"
    return int(annotation["class_id"]), "kept_particle"


def safe_link_or_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        target.symlink_to(source.resolve())
        return "symlinked"
    except OSError:
        shutil.copy2(source, target)
        return "copied"


def normalized_interval(annotation: dict) -> tuple[float, float]:
    if "start" in annotation and "end" in annotation:
        left = float(annotation["start"])
        right = float(annotation["end"])
    else:
        center = float(annotation.get("center", annotation.get("mean", 0.0)))
        half_width = float(annotation.get("half_width", 0.0))
        left = center - half_width
        right = center + half_width
    left = min(1.0, max(0.0, left))
    right = min(1.0, max(left, right))
    return left, right


def convert_split(
    source_root: Path,
    output_root: Path,
    split: str,
    annotations_by_file: dict[str, dict],
    threshold_db: float,
) -> tuple[Counter, list[dict]]:
    split_counts: Counter = Counter()
    manifest_rows = []

    signals_in = source_root / split / "signals"
    labels_out = output_root / split / "labels"
    signals_out = output_root / split / "signals"
    labels_out.mkdir(parents=True, exist_ok=True)
    signals_out.mkdir(parents=True, exist_ok=True)

    for signal_path in sorted(signals_in.glob("*.npy")):
        action = safe_link_or_copy(signal_path, signals_out / signal_path.name)
        row = annotations_by_file.get(signal_path.name)
        if row is None:
            raise KeyError(f"No particles2SNR data.json row found for {signal_path.name}")

        lines = []
        for ann in row.get("annotations", []):
            new_class, reason = label_for_annotation(ann, threshold_db)
            left, right = normalized_interval(ann)
            center = (left + right) / 2.0
            width = right - left
            lines.append(f"{new_class} {center:.10f} {width:.10f}")
            split_counts[CLASS_NAMES[new_class]] += 1
            manifest_rows.append({
                "split": split,
                "filename": signal_path.name,
                "annotation_id": int(ann.get("id", len(manifest_rows))),
                "old_class_id": int(ann.get("class_id", -1)),
                "old_class_name": CLASS_NAMES[int(ann.get("class_id", -1))]
                if 0 <= int(ann.get("class_id", -1)) < UNCLEAR_CLASS_ID else "",
                "new_class_id": int(new_class),
                "new_class_name": CLASS_NAMES[new_class],
                "snr_db": ann.get("snr_db"),
                "threshold_db": threshold_db,
                "reason": reason,
                "signal_action": action,
            })

        (labels_out / f"{signal_path.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )

    return split_counts, manifest_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split", "filename", "annotation_id", "old_class_id", "old_class_name",
        "new_class_id", "new_class_name", "snr_db", "threshold_db", "reason",
        "signal_action",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def count_files(root: Path, split: str) -> int:
    return len(list((root / split / "signals").glob("*.npy")))


def write_dataset_yaml(
    output_root: Path,
    source_root: Path,
    split_counts: dict[str, Counter],
    threshold_db: float,
    particles2SNR_jsons: list[Path],
    particles2SNR_info: dict | None = None,
) -> None:
    splits = {}
    for split in SPLITS:
        split_data = {name: int(split_counts[split].get(name, 0)) for name in CLASS_NAMES}
        split_data["background"] = 0
        split_data["total"] = count_files(output_root, split)
        splits[split] = split_data

    yaml_data = {
        "path": str(output_root.resolve()),
        "train": "train/signals",
        "val": "val/signals",
        "test": "test/signals",
        "nc": len(CLASS_NAMES),
        "names": list(CLASS_NAMES),
        "sampling_frequency_hz": 2_000_000,
        "signal_lengths": {"particles": 16384},
        "preprocessing": {
            "bandpass": {"enabled": True, "low_hz": 7000, "high_hz": 80000, "order": 4},
            "saturation_policy": "replace",
        },
        "splits": splits,
        "provenance": {
            "source_dataset": str(source_root),
            "source_particles2SNR_jsons": [str(path) for path in particles2SNR_jsons],
            "split_policy": "same files as source train/val/test",
            "label_policy": "snr_db below threshold becomes unclear",
        },
        "generation_params": {
            "source": "particles2SNR_pipeline/create_particles2SNR_c1_yolo_4class_lim10.py",
            "snr_threshold_db": float(threshold_db),
            "unclear_rule": "snr_db < threshold_db",
            "class_names": list(CLASS_NAMES),
            "inherited_particles2SNR_annotation_params": {
                key: value
                for key, value in (particles2SNR_info or {}).items()
                if key in {"passage_time_filter", "overlap_merge", "peak_evidence_filter", "yolo_width_filter", "boundary_resolution"}
            },
        },
        "audit_results": {
            "long_sequence": {
                "status": "not_applicable_single_segment_particles2SNR_c1_4class_lim10",
                "note": "Structural label audit is performed by P1/detseg/audit_dataset.py.",
            }
        },
    }
    with (output_root / "dataset.yaml").open("w") as f:
        yaml.safe_dump(yaml_data, f, sort_keys=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create particles2SNR C1 YOLO 4-class lim10 detection dataset."
    )
    parser.add_argument(
        "--source-root",
        default="P0/data/dataset_Particles2SNR_F_c1_yolo_trainval",
        help="Existing 3-class particles2SNR C1 YOLO train/val/test dataset.",
    )
    parser.add_argument(
        "--output-root",
        default="P1/yolo_dataset_Particles2SNR_F_c1_4class_lim10_trainval",
        help="Output 4-class YOLO dataset.",
    )
    parser.add_argument(
        "--particles2SNR-train-json",
        default="particles2SNR_pipeline/output/p0_c1_Particles2SNR_F/train/data.json",
    )
    parser.add_argument(
        "--particles2SNR-test-json",
        default="particles2SNR_pipeline/output/p0_c1_Particles2SNR_F/test/data.json",
    )
    parser.add_argument("--threshold-db", type=float, default=-10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    particles2SNR_jsons = [Path(args.particles2SNR_train_json), Path(args.particles2SNR_test_json)]

    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    annotations_by_file, particles2SNR_info = load_particles2SNR_jsons(particles2SNR_jsons)

    split_counts: dict[str, Counter] = {}
    all_manifest_rows: list[dict] = []
    for split in SPLITS:
        counts, rows = convert_split(
            source_root, output_root, split, annotations_by_file, args.threshold_db
        )
        split_counts[split] = counts
        all_manifest_rows.extend(rows)

    write_csv(output_root / "class_assignment_manifest.csv", all_manifest_rows)
    summary = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "threshold_db": float(args.threshold_db),
        "class_names": list(CLASS_NAMES),
        "splits": {
            split: {
                "files": count_files(output_root, split),
                "events": int(sum(split_counts[split].values())),
                "events_by_class": {
                    name: int(split_counts[split].get(name, 0))
                    for name in CLASS_NAMES
                },
            }
            for split in SPLITS
        },
    }
    (output_root / "class_balance_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_dataset_yaml(output_root, source_root, split_counts, args.threshold_db, particles2SNR_jsons, particles2SNR_info)

    print(f"Dataset written to: {output_root}")
    for split, data in summary["splits"].items():
        print(f"  {split}: files={data['files']} events={data['events']} {data['events_by_class']}")


if __name__ == "__main__":
    main()
