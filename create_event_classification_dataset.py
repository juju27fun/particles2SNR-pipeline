#!/usr/bin/env python3
"""Create a 3-class event-crop classification dataset from particles2SNR YOLO data."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from repo_paths import RESULTS_RUNS


DEFAULT_CLASSES = ("2um", "4um", "10um")


def parse_csv_arg(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def load_rows(paths: list[Path]) -> dict[str, dict]:
    rows = {}
    for path in paths:
        data = read_json(path)
        for row in data.get("data", []):
            filename = row.get("filename")
            if filename:
                rows[filename] = row
    return rows


def crop_centered(signal: np.ndarray, center_sample: int, length: int) -> np.ndarray:
    arr = np.asarray(signal, dtype=np.float32).reshape(-1)
    out = np.zeros(int(length), dtype=np.float32)
    start = int(center_sample) - int(length) // 2
    end = start + int(length)
    src_start = max(0, start)
    src_end = min(len(arr), end)
    if src_end > src_start:
        dst_start = src_start - start
        out[dst_start:dst_start + (src_end - src_start)] = arr[src_start:src_end]
    return out


def split_filenames(yolo_root: Path, split: str) -> list[str]:
    signal_dir = yolo_root / split / "signals"
    if not signal_dir.is_dir():
        raise FileNotFoundError(f"Missing YOLO signal split dir: {signal_dir}")
    return sorted(path.name for path in signal_dir.glob("*.npy"))


def safe_stem(value: str) -> str:
    return Path(value).stem.replace(" ", "_")


def materialize_split(
    split: str,
    filenames: list[str],
    rows_by_file: dict[str, dict],
    output_root: Path,
    class_names: tuple[str, ...],
    crop_length: int,
    fs: float,
) -> list[dict]:
    manifest = []
    class_by_id = {idx: name for idx, name in enumerate(class_names)}
    signal_cache: dict[str, np.ndarray] = {}
    for filename in filenames:
        row = rows_by_file.get(filename)
        if row is None:
            raise KeyError(f"No particles2SNR data.json row found for {split}/{filename}")
        source_path = str(row["path"])
        if source_path not in signal_cache:
            signal_cache[source_path] = np.load(source_path).astype(np.float32)
        signal = signal_cache[source_path]
        signal_length = int(row.get("length") or len(signal))
        for ann in row.get("annotations", []):
            class_id = int(ann.get("class_id", -1))
            class_name = class_by_id.get(class_id)
            if class_name is None:
                continue
            center = float(ann.get("center", ann.get("mean", 0.0)))
            start = float(ann.get("start", center))
            end = float(ann.get("end", center))
            center_sample = int(round(center * signal_length))
            crop = crop_centered(signal, center_sample, crop_length)
            out_name = f"{safe_stem(filename)}__ann{int(ann.get('id', len(manifest))):03d}.npy"
            out_path = output_root / split / class_name / out_name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, crop.astype(np.float32))
            start_ms = start * signal_length / fs * 1000.0
            end_ms = end * signal_length / fs * 1000.0
            manifest.append({
                "split": split,
                "output_path": str(out_path),
                "output_filename": out_name,
                "source_filename": filename,
                "source_path": source_path,
                "annotation_id": int(ann.get("id", -1)),
                "class_id": class_id,
                "class_name": class_name,
                "snr_db": ann.get("snr_db"),
                "frequency": ann.get("frequency"),
                "passage_time_ms": ann.get("passage_time_ms"),
                "center": center,
                "start": start,
                "end": end,
                "center_ms": center * signal_length / fs * 1000.0,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "width_ms": max(0.0, end_ms - start_ms),
                "crop_length": crop_length,
                "peak_group_id": ann.get("peak_group_id"),
                "boundary_adjusted": bool(ann.get("boundary_adjusted", False)),
            })
    return manifest


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create event-crop classification dataset from clean particles2SNR YOLO annotations.")
    parser.add_argument("--yolo-root", type=Path, default=Path("P0/data/processed/dataset_Particles2SNR_F_c1_yolo_trainval"))
    parser.add_argument("--particles2SNR-json", type=Path, action="append", default=None)
    parser.add_argument("--output-root", type=Path, default=Path("P0/data/processed/dataset_Particles2SNR_F_c1_events"))
    parser.add_argument("--artifact-root", type=Path, default=RESULTS_RUNS / "p0_c1_Particles2SNR_F" / "event_classification_dataset")
    parser.add_argument("--splits", type=parse_csv_arg, default=("train", "val", "test"))
    parser.add_argument("--classes", type=parse_csv_arg, default=DEFAULT_CLASSES)
    parser.add_argument("--crop-length", type=int, default=16384)
    parser.add_argument("--fs", type=float, default=2_000_000.0)
    parser.add_argument("--keep-existing", action="store_true")
    args = parser.parse_args()
    if args.particles2SNR_json is None:
        args.particles2SNR_json = [
            RESULTS_RUNS / "p0_c1_Particles2SNR_F" / "train" / "data.json",
            RESULTS_RUNS / "p0_c1_Particles2SNR_F" / "test" / "data.json",
        ]

    if args.output_root.exists() and not args.keep_existing:
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.artifact_root.mkdir(parents=True, exist_ok=True)

    rows_by_file = load_rows(args.particles2SNR_json)
    all_rows = []
    summary = {
        "source_yolo_root": str(args.yolo_root),
        "source_particles2SNR_json": [str(path) for path in args.particles2SNR_json],
        "output_root": str(args.output_root),
        "artifact_root": str(args.artifact_root),
        "classes": list(args.classes),
        "splits": {},
        "crop_length": args.crop_length,
        "fs": args.fs,
        "unit": "particles2SNR_yolo_annotation_event",
    }
    for split in args.splits:
        filenames = split_filenames(args.yolo_root, split)
        rows = materialize_split(
            split,
            filenames,
            rows_by_file,
            args.output_root,
            tuple(args.classes),
            args.crop_length,
            args.fs,
        )
        write_csv(args.artifact_root / f"{split}_event_manifest.csv", rows)
        all_rows.extend(rows)
        counts = Counter(row["class_name"] for row in rows)
        summary["splits"][split] = {
            "source_files": len(filenames),
            "events": len(rows),
            "events_by_class": {name: int(counts.get(name, 0)) for name in args.classes},
        }
        print(f"{split}: files={len(filenames)} events={len(rows)} {summary['splits'][split]['events_by_class']}")
    write_csv(args.artifact_root / "event_manifest.csv", all_rows)
    with (args.artifact_root / "dataset_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Dataset: {args.output_root}")
    print(f"Manifest: {args.artifact_root / 'event_manifest.csv'}")


if __name__ == "__main__":
    main()
