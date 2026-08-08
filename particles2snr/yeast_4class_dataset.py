from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .yeast_raw_data import normalize_raw_dataset_roots, resolve_raw_signal
from .yeast_representation_dataset import clamped_crop, preprocess_crop


DATASET_ID = "yeast-budding-mix-shmoo-background-classification@v1"
CLASS_NAMES = ("background", "budding", "mix", "shmoo")
SOURCE_TO_CLASS = {"budding": "budding", "mix": "mix", "shmoo2": "shmoo"}
ALLOWED_SPLITS = ("development_train", "development_validation")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exclusion_record_ids(paths: Iterable[Path]) -> set[str]:
    records: set[str] = set()
    for path in paths:
        rows = read_csv(path)
        if rows and "record_id" not in rows[0]:
            raise ValueError(f"Exclusion table has no record_id column: {path}")
        records.update(row["record_id"] for row in rows if row.get("record_id"))
    return records


def select_event_rows(
    candidate_rows: list[dict[str, str]],
    excluded: set[str],
    *,
    source_to_class: Mapping[str, str] = SOURCE_TO_CLASS,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in candidate_rows:
        source = row.get("source_group", "")
        split = row.get("development_split", "")
        if row.get("quality") != "strict" or source not in source_to_class or split not in ALLOWED_SPLITS:
            continue
        if row["record_id"] in excluded:
            continue
        class_name = source_to_class[source]
        selected.append(
            {
                **row,
                "sample_id": f"event:{row['event_id']}",
                "sample_kind": "event",
                "class_name": class_name,
                "class_id": CLASS_NAMES.index(class_name),
                "source_group_original": source,
                "background_selection": "",
            }
        )
    if not selected:
        raise ValueError("No eligible strict event rows")
    ids = [row["sample_id"] for row in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("Event sample IDs are not unique")
    if "shmoo" not in source_to_class and any(row["source_group_original"] == "shmoo" for row in selected):
        raise AssertionError("Historical low-concentration shmoo must be absent")
    return selected


def _intersects(start: int, end: int, intervals: list[tuple[int, int]]) -> bool:
    return any(start < right and end > left for left, right in intervals)


def _stable_key(seed: int, *parts: object) -> str:
    return hashlib.sha256((str(seed) + "|" + "|".join(map(str, parts))).encode()).hexdigest()


def enumerate_background_candidates(
    *,
    candidate_rows: list[dict[str, str]],
    excluded: set[str],
    raw_dataset_root: Path | None,
    raw_dataset_roots: Mapping[str, Path] | None,
    crop_length: int,
    guard_samples: int,
    stride_samples: int,
    source_to_class: Mapping[str, str] = SOURCE_TO_CLASS,
) -> list[dict[str, Any]]:
    by_record: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidate_rows:
        if (
            row.get("source_group") in source_to_class
            and row.get("development_split") in ALLOWED_SPLITS
            and row.get("record_id") not in excluded
        ):
            by_record[row["record_id"]].append(row)
    raw_root, raw_roots = normalize_raw_dataset_roots(
        raw_dataset_root=raw_dataset_root, raw_dataset_roots=raw_dataset_roots
    )
    candidates: list[dict[str, Any]] = []
    for record_id, rows in sorted(by_record.items()):
        exemplar = rows[0]
        raw = np.asarray(
            np.load(resolve_raw_signal(exemplar, single_root=raw_root, roots_by_dataset=raw_roots), allow_pickle=False)
        ).squeeze()
        if raw.ndim != 1 or raw.size < crop_length:
            raise ValueError(f"Invalid raw signal for {record_id}: {raw.shape}")
        intervals = sorted(
            (
                max(0, int(row["event_start"]) - guard_samples),
                min(raw.size, int(row["event_end"]) + guard_samples),
            )
            for row in rows
        )
        for start in range(0, raw.size - crop_length + 1, stride_samples):
            end = start + crop_length
            if _intersects(start, end, intervals):
                continue
            values = raw[start:end].astype(np.float64, copy=False)
            energy = float(np.sqrt(np.mean(np.square(values - np.median(values)))))
            candidates.append(
                {
                    **exemplar,
                    "sample_id": f"background:{record_id}:{start:05d}",
                    "sample_kind": "background",
                    "class_name": "background",
                    "class_id": 0,
                    "source_group_original": exemplar["source_group"],
                    "background_start": start,
                    "background_end": end,
                    "background_energy": energy,
                }
            )
    return candidates


def select_background_rows(
    candidates: list[dict[str, Any]],
    events: list[dict[str, Any]],
    *,
    seed: int,
    background_sources: tuple[str, ...] = tuple(SOURCE_TO_CLASS),
) -> list[dict[str, Any]]:
    event_counts = Counter((row["development_split"], row["class_name"]) for row in events)
    selected: list[dict[str, Any]] = []
    for split in ALLOWED_SPLITS:
        target = max(event_counts[(split, name)] for name in CLASS_NAMES[1:])
        base, remainder = divmod(target, len(background_sources))
        for source_index, source in enumerate(background_sources):
            quota = base + int(source_index < remainder)
            pool = [
                row for row in candidates
                if row["development_split"] == split and row["source_group_original"] == source
            ]
            if len(pool) < quota:
                raise ValueError(f"Insufficient background candidates for {split}/{source}: {len(pool)} < {quota}")
            uniform_count = quota // 2
            uniform = sorted(pool, key=lambda row: _stable_key(seed, row["sample_id"]))[:uniform_count]
            used = {row["sample_id"] for row in uniform}
            high = sorted(
                (row for row in pool if row["sample_id"] not in used),
                key=lambda row: (-float(row["background_energy"]), row["sample_id"]),
            )[: quota - uniform_count]
            for row in uniform:
                selected.append({**row, "background_selection": "uniform"})
            for row in high:
                selected.append({**row, "background_selection": "high_energy_clean"})
    ids = [row["sample_id"] for row in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("Background sample IDs are not unique")
    return selected


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_dataset(
    *,
    candidate_csv: Path,
    exclusion_csvs: list[Path],
    raw_dataset_root: Path | None,
    raw_dataset_roots: Mapping[str, Path] | None,
    output_dir: Path,
    raw_dataset_id: str = "yeast-hf-10-5-20260610@v1",
    candidate_dataset_id: str = "yeast-event-candidates@v7",
    seed: int = 20260805,
    crop_length: int = 8192,
    downsample_factor: int = 2,
    guard_samples: int = 1000,
    background_stride_samples: int = 256,
    dataset_id: str = DATASET_ID,
    run_id: str = "yeast-4class-classification-dataset-build-r1",
    method_evidence_id: str = "yeast-4class-conv1dgap-latent-method-r1",
    source_to_class: Mapping[str, str] = SOURCE_TO_CLASS,
    background_sources: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Immutable dataset already exists: {output_dir}")
    candidate_rows = read_csv(candidate_csv)
    excluded = exclusion_record_ids(exclusion_csvs)
    source_to_class = dict(source_to_class)
    background_sources = tuple(source_to_class) if background_sources is None else tuple(background_sources)
    if not background_sources or any(source not in source_to_class for source in background_sources):
        raise ValueError("Background sources must be a non-empty subset of source_to_class")
    events = select_event_rows(candidate_rows, excluded, source_to_class=source_to_class)
    background_pool = enumerate_background_candidates(
        candidate_rows=candidate_rows,
        excluded=excluded,
        raw_dataset_root=raw_dataset_root,
        raw_dataset_roots=raw_dataset_roots,
        crop_length=crop_length,
        guard_samples=guard_samples,
        stride_samples=background_stride_samples,
        source_to_class=source_to_class,
    )
    backgrounds = select_background_rows(background_pool, events, seed=seed, background_sources=background_sources)
    samples = sorted(events + backgrounds, key=lambda row: (row["development_split"], int(row["class_id"]), row["sample_id"]))
    output_length = crop_length // downsample_factor
    output_dir.mkdir(parents=True)
    signals = np.lib.format.open_memmap(
        output_dir / "signals.npy", mode="w+", dtype=np.float32, shape=(len(samples), output_length)
    )
    for index, row in enumerate(samples):
        row["signal_row"] = index
    indices_by_record: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(samples):
        indices_by_record[row["record_id"]].append(index)
    raw_root, raw_roots = normalize_raw_dataset_roots(
        raw_dataset_root=raw_dataset_root, raw_dataset_roots=raw_dataset_roots
    )
    train_sum = train_square_sum = 0.0
    train_count = 0
    for record_id, indices in sorted(indices_by_record.items()):
        exemplar = samples[indices[0]]
        raw = np.load(resolve_raw_signal(exemplar, single_root=raw_root, roots_by_dataset=raw_roots), allow_pickle=False)
        for index in indices:
            row = samples[index]
            if row["sample_kind"] == "event":
                crop, start = clamped_crop(raw, int(row["center_index"]), crop_length)
            else:
                start = int(row["background_start"])
                crop = np.asarray(raw).squeeze()[start : start + crop_length].astype(np.float32, copy=False)
            processed = preprocess_crop(crop, downsample_factor=downsample_factor)
            signals[index] = processed
            row["crop_start"] = start
            row["crop_end"] = start + crop_length
            if row["development_split"] == "development_train":
                values = processed.astype(np.float64, copy=False)
                train_sum += float(values.sum())
                train_square_sum += float(np.square(values).sum())
                train_count += int(values.size)
    train_mean = train_sum / train_count
    train_std = float(np.sqrt(max(train_square_sum / train_count - train_mean**2, 0.0)))
    if train_std <= 1e-12:
        raise ValueError("Training standard deviation is zero")
    for start in range(0, len(samples), 256):
        stop = min(start + 256, len(samples))
        signals[start:stop] = (signals[start:stop] - train_mean) / train_std
    signals.flush()
    del signals

    _write_csv(output_dir / "samples.csv", samples)
    contract = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "grain": "one 4096-sample classified crop",
        "primary_key": ["sample_id"],
        "classes": list(CLASS_NAMES),
        "class_mapping": source_to_class,
        "excluded_source_groups": sorted({"budding", "mix", "shmoo", "shmoo2"} - set(source_to_class)),
        "partitions": list(ALLOWED_SPLITS),
        "input_contract": "yeast-event-8192to4096-bandpass-global-v1",
        "raw_crop_length": crop_length,
        "output_length": output_length,
        "bandpass_hz": [5000.0, 100000.0],
        "bandpass_phase": "zero-phase SOS forward-backward",
        "downsampling": "scipy.signal.resample_poly up=1 down=2",
        "normalization": {"policy": "combined development_train global mean/std", "mean": train_mean, "std": train_std},
        "background_policy": {
            "candidate_guard_samples": guard_samples,
            "stride_samples": background_stride_samples,
            "provenance_quota": f"equal {'/'.join(background_sources)} within split",
            "selection": "half deterministic uniform, half highest-energy clean",
        },
        "label_scope": "single-acquisition source-condition proxies plus clean background",
        "sealed_holdout_accessed": False,
        "compatibility": {"n_minus_1": "initial version; future changes require a new immutable version"},
    }
    (output_dir / "dataset-contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "dataset_id": dataset_id,
        "n_samples": len(samples),
        "class_counts": dict(sorted(Counter(row["class_name"] for row in samples).items())),
        "split_class_counts": {
            split: dict(sorted(Counter(row["class_name"] for row in samples if row["development_split"] == split).items()))
            for split in ALLOWED_SPLITS
        },
        "background_source_counts": dict(sorted(Counter(row["source_group_original"] for row in backgrounds).items())),
        "excluded_record_count": len(excluded),
        "signals_shape": [len(samples), output_length],
        "signals_dtype": "float32",
        "sealed_holdout_accessed": False,
    }
    (output_dir / "dataset-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    source_files = [candidate_csv, *exclusion_csvs]
    provenance = {
        "dataset_id": dataset_id,
        "source_dataset_ids": [raw_dataset_id, candidate_dataset_id, "yeast-event-review-annotations@v1", "yeast-budding-reviewed-event-inventory@v1"],
        "method_evidence_id": method_evidence_id,
        "source_to_class": source_to_class,
        "background_sources": list(background_sources),
        "seed": seed,
        "source_files": [{"path": str(path), "sha256": sha256_file(path)} for path in source_files],
        "excluded_record_count": len(excluded),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload = ["signals.npy", "samples.csv", "dataset-contract.json", "dataset-summary.json", "provenance.json"]
    manifest = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "producer": "particles2SNR-pipeline/scripts/generation/build_yeast_4class_classification_dataset.py",
        "row_count": len(samples),
        "files": [{"path": name, "size": (output_dir / name).stat().st_size, "sha256": sha256_file(output_dir / name)} for name in payload],
    }
    (output_dir / "dataset-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    run = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "complete",
        "dataset": dataset_id,
        "command": "build_yeast_4class_classification_dataset.py",
        "method_evidence_id": method_evidence_id,
        "dataset_manifest_sha256": sha256_file(output_dir / "dataset-manifest.json"),
        "sealed_holdout_accessed": False,
    }
    (output_dir / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
