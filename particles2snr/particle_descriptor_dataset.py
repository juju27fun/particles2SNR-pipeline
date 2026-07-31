from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .yeast_representation_dataset import clamped_crop, preprocess_crop


CLASS_NAMES = {"0": "2um", "1": "4um", "2": "10um"}
RAW_CROP_LENGTH = 8192
OUTPUT_LENGTH = 4096


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _annotations(path: Path) -> list[tuple[int, str, float, float]]:
    rows: list[tuple[int, str, float, float]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 3:
            raise ValueError(f"Malformed annotation at {path}:{index + 1}")
        rows.append((index, fields[0], float(fields[1]), float(fields[2])))
    return rows


def build_particle_descriptor_dataset(
    *,
    source_root: Path,
    output_dir: Path,
    source_dataset_id: str,
    source_manifest_sha256: str,
    population_id: str,
    splits: tuple[str, ...] = ("train", "val"),
    crop_policy: str = "exact-centered",
) -> dict[str, Any]:
    if set(splits) - {"train", "val"}:
        raise ValueError("Only train and val are permitted; test is sealed")
    if crop_policy not in {"exact-centered", "shift-window"}:
        raise ValueError(f"Unsupported crop policy: {crop_policy}")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite dataset: {output_dir}")

    eligible: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    source_hashes: dict[Path, str] = {}
    source_annotation_counts: Counter[str] = Counter()

    for split in splits:
        label_root = source_root / split / "labels"
        signal_root = source_root / split / "signals"
        for label_path in sorted(label_root.glob("*.txt")):
            signal_path = signal_root / f"{label_path.stem}.npy"
            if not signal_path.is_file():
                raise FileNotFoundError(f"Missing signal for {label_path}")
            source = np.load(signal_path, mmap_mode="r", allow_pickle=False)
            if source.ndim != 1:
                raise ValueError(f"Expected 1-D signal at {signal_path}, got {source.shape}")
            signal_length = int(source.size)
            for annotation_index, class_id, center, width in _annotations(label_path):
                source_annotation_counts[f"{split}:{class_id}"] += 1
                event_id = f"{split}:{label_path.stem}:{annotation_index}"
                common = {
                    "event_id": event_id,
                    "split": split,
                    "source_filename": signal_path.name,
                    "source_group": f"{split}:{label_path.stem}",
                    "annotation_index": annotation_index,
                    "source_class_id": class_id,
                    "center_norm": center,
                    "width_norm": width,
                }
                if class_id not in CLASS_NAMES:
                    exclusions.append({**common, "reason": "class_unclear_or_unknown"})
                    continue
                center_index = int(round(center * signal_length))
                centered_start = center_index - RAW_CROP_LENGTH // 2
                centered_end = centered_start + RAW_CROP_LENGTH
                if (
                    crop_policy == "exact-centered"
                    and (centered_start < 0 or centered_end > signal_length)
                ):
                    exclusions.append({**common, "reason": "incomplete_centered_crop"})
                    continue
                if crop_policy == "shift-window":
                    crop_start = min(
                        max(centered_start, 0),
                        signal_length - RAW_CROP_LENGTH,
                    )
                else:
                    crop_start = centered_start
                crop_end = crop_start + RAW_CROP_LENGTH
                event_start_raw = center * signal_length - width * signal_length / 2.0
                event_end_raw = center * signal_length + width * signal_length / 2.0
                clipped_event_start_raw = max(0.0, event_start_raw)
                clipped_event_end_raw = min(float(signal_length), event_end_raw)
                if event_start_raw < crop_start or event_end_raw > crop_end:
                    if (
                        crop_policy == "exact-centered"
                        or clipped_event_start_raw < crop_start
                        or clipped_event_end_raw > crop_end
                    ):
                        exclusions.append({**common, "reason": "event_exceeds_crop"})
                        continue
                eligible.append(
                    {
                        **common,
                        "class_id": int(class_id),
                        "class_name": CLASS_NAMES[class_id],
                        "signal_path": signal_path,
                        "label_path": label_path,
                        "signal_length": signal_length,
                        "center_index": center_index,
                        "crop_start_raw": crop_start,
                        "crop_end_raw": crop_end,
                        "event_start_raw_unclipped": event_start_raw,
                        "event_end_raw_unclipped": event_end_raw,
                        "event_start_raw": clipped_event_start_raw,
                        "event_end_raw": clipped_event_end_raw,
                        "annotation_clipped_to_source": bool(
                            clipped_event_start_raw != event_start_raw
                            or clipped_event_end_raw != event_end_raw
                        ),
                    }
                )

    if not eligible:
        raise ValueError("No eligible particle events")
    event_ids = [row["event_id"] for row in eligible]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("Event IDs are not unique")

    output_dir.mkdir(parents=True)
    signals = np.lib.format.open_memmap(
        output_dir / "signals.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(eligible), OUTPUT_LENGTH),
    )
    public_rows: list[dict[str, Any]] = []
    for signal_row, row in enumerate(eligible):
        signal_path = row["signal_path"]
        label_path = row["label_path"]
        if signal_path not in source_hashes:
            source_hashes[signal_path] = sha256_file(signal_path)
        if label_path not in source_hashes:
            source_hashes[label_path] = sha256_file(label_path)
        source = np.load(signal_path, mmap_mode="r", allow_pickle=False)
        if crop_policy == "shift-window":
            crop, actual_start = clamped_crop(
                source,
                row["center_index"],
                RAW_CROP_LENGTH,
            )
            if actual_start != row["crop_start_raw"]:
                raise ValueError(f"Crop policy mismatch for {row['event_id']}")
        else:
            crop = np.asarray(
                source[row["crop_start_raw"] : row["crop_end_raw"]],
                dtype=np.float32,
            )
        processed = preprocess_crop(crop)
        if processed.shape != (OUTPUT_LENGTH,) or not np.all(np.isfinite(processed)):
            raise ValueError(f"Invalid processed signal for {row['event_id']}")
        signals[signal_row] = processed
        public_rows.append(
            {
                "signal_row": signal_row,
                "event_id": row["event_id"],
                "split": row["split"],
                "class_id": row["class_id"],
                "class_name": row["class_name"],
                "source_filename": row["source_filename"],
                "source_group": row["source_group"],
                "annotation_index": row["annotation_index"],
                "center_norm": row["center_norm"],
                "width_norm": row["width_norm"],
                "source_signal_sha256": source_hashes[signal_path],
                "source_label_sha256": source_hashes[label_path],
                "source_signal_length": row["signal_length"],
                "crop_start_raw": row["crop_start_raw"],
                "crop_end_raw": row["crop_end_raw"],
                "event_start_raw_unclipped": row["event_start_raw_unclipped"],
                "event_end_raw_unclipped": row["event_end_raw_unclipped"],
                "annotation_clipped_to_source": row[
                    "annotation_clipped_to_source"
                ],
                "event_start_index": (
                    row["event_start_raw"] - row["crop_start_raw"]
                )
                / 2.0,
                "event_end_index": (
                    row["event_end_raw"] - row["crop_start_raw"]
                )
                / 2.0,
            }
        )
    signals.flush()
    del signals

    with (output_dir / "events.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(public_rows[0]))
        writer.writeheader()
        writer.writerows(public_rows)
    exclusion_fields = [
        "event_id",
        "split",
        "source_filename",
        "source_group",
        "annotation_index",
        "source_class_id",
        "center_norm",
        "width_norm",
        "reason",
    ]
    with (output_dir / "exclusions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=exclusion_fields)
        writer.writeheader()
        writer.writerows(exclusions)

    included_counts = Counter(
        f"{row['split']}:{row['class_name']}" for row in public_rows
    )
    exclusion_counts = Counter(row["reason"] for row in exclusions)
    summary = {
        "schema_version": 1,
        "population_id": population_id,
        "source_dataset": source_dataset_id,
        "source_manifest_sha256": source_manifest_sha256,
        "included_events": len(public_rows),
        "included_counts": dict(sorted(included_counts.items())),
        "source_annotation_counts": dict(sorted(source_annotation_counts.items())),
        "excluded_events": len(exclusions),
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "annotations_clipped_to_source": sum(
            bool(row["annotation_clipped_to_source"]) for row in eligible
        ),
        "test_split_accessed": False,
        "crop_policy": crop_policy,
        "dtype": "float32",
        "shape": [len(public_rows), OUTPUT_LENGTH],
        "finite": True,
    }
    contract = {
        "schema_version": 1,
        "contract_id": "particle-f-event-8192to4096-descriptor-v1",
        "source_dataset": source_dataset_id,
        "source_manifest_sha256": source_manifest_sha256,
        "eligible_classes": CLASS_NAMES,
        "splits": list(splits),
        "sealed_splits": ["test"],
        "raw_sampling_frequency_hz": 2_000_000.0,
        "raw_crop_length": RAW_CROP_LENGTH,
        "crop_policy": (
            "exact centered crop; reject incomplete windows; never clamp or pad"
            if crop_policy == "exact-centered"
            else "fixed-length crop shifted at signal boundaries; never pad"
        ),
        "bandpass_hz": [5_000.0, 100_000.0],
        "bandpass_order": 4,
        "downsampling": "scipy.signal.resample_poly up=1 down=2",
        "output_sampling_frequency_hz": 1_000_000.0,
        "output_length": OUTPUT_LENGTH,
        "normalization": "none; C2 metrics use dimensionless within-signal amplitude and energy ratios",
    }
    for name, payload in (
        ("dataset_summary.json", summary),
        ("input_contract.json", contract),
    ):
        (output_dir / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return summary


def validate_particle_descriptor_dataset(root: Path) -> dict[str, Any]:
    summary = json.loads((root / "dataset_summary.json").read_text(encoding="utf-8"))
    contract = json.loads((root / "input_contract.json").read_text(encoding="utf-8"))
    signals = np.load(root / "signals.npy", mmap_mode="r", allow_pickle=False)
    with (root / "events.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if signals.shape != tuple(summary["shape"]):
        raise ValueError("Signal shape does not match dataset summary")
    if signals.dtype != np.float32:
        raise ValueError("Signals must be float32")
    if len(rows) != signals.shape[0]:
        raise ValueError("events.csv row count does not match signals.npy")
    if any(row["split"] == "test" for row in rows):
        raise ValueError("Sealed test split was included")
    if len({row["event_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate event IDs")
    if any(not row["source_group"] for row in rows):
        raise ValueError("Every event must retain a source group")
    for row in rows:
        start = float(row["event_start_index"])
        end = float(row["event_end_index"])
        if not 0.0 <= start < end <= signals.shape[1]:
            raise ValueError(
                f"Event bounds exceed processed signal for {row['event_id']}: "
                f"[{start}, {end})"
            )
    for start in range(0, len(rows), 256):
        if not np.all(np.isfinite(signals[start : start + 256])):
            raise ValueError("Non-finite values in signals.npy")
    if contract["source_manifest_sha256"] != summary["source_manifest_sha256"]:
        raise ValueError("Source manifest mismatch")
    return {
        "valid": True,
        "events": len(rows),
        "shape": list(signals.shape),
        "test_split_accessed": False,
    }
