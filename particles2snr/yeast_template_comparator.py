from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _record_partition(record_id: str, seed: int) -> str:
    value = int(hashlib.sha256(f"template-comparator:{seed}:{record_id}".encode()).hexdigest()[:8], 16)
    return "train" if value % 4 else "validation"


def _augment_template(
    values: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, float | int]]:
    shift = int(rng.integers(-128, 129))
    scale = float(rng.uniform(0.80, 1.20))
    noise_scale = float(rng.uniform(0.01, 0.12))
    white = rng.normal(size=len(values)).astype(np.float32)
    colored = gaussian_filter1d(white, sigma=float(rng.uniform(1.0, 8.0)))
    colored /= max(float(np.std(colored)), 1.0e-6)
    output = scale * np.roll(values, shift) + noise_scale * colored
    return output.astype(np.float32), {
        "time_shift_samples": shift,
        "amplitude_scale": scale,
        "noise_scale": noise_scale,
    }


def build_template_comparator(
    *,
    followup_root: Path,
    output_dir: Path,
    n_train: int = 2000,
    n_validation: int = 1000,
    seed: int = 20260716,
) -> dict[str, Any]:
    if min(n_train, n_validation) <= 0:
        raise ValueError("Comparator split sizes must be positive")
    development_index = followup_root / "development_events.csv"
    if not development_index.is_file():
        raise FileNotFoundError("Template generation requires development_events.csv")
    rows = [
        row
        for row in _read_rows(development_index)
        if row["development_split"] == "followup_train"
    ]
    if not rows:
        raise ValueError("No followup_train templates")
    if any(row["development_split"] != "followup_train" for row in rows):
        raise AssertionError("Template bank includes a forbidden split")
    signals = np.load(followup_root / "signals.npy", mmap_mode="r")

    pools: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        pools[_record_partition(row["record_id"], seed)].append(row)
    if not pools["train"] or not pools["validation"]:
        raise ValueError("Record-group comparator partition is empty")
    train_records = {row["record_id"] for row in pools["train"]}
    validation_records = {row["record_id"] for row in pools["validation"]}
    if train_records & validation_records:
        raise AssertionError("Template source record crosses comparator splits")

    output_dir.mkdir(parents=True, exist_ok=False)
    output = np.lib.format.open_memmap(
        output_dir / "signals.npy",
        mode="w+",
        dtype=np.float32,
        shape=(n_train + n_validation, signals.shape[1]),
    )
    metadata: list[dict[str, Any]] = []
    offset = 0
    for split, count, split_seed in (
        ("train", n_train, seed),
        ("validation", n_validation, seed + 100_000),
    ):
        rng = np.random.default_rng(split_seed)
        pool = pools[split]
        for index in range(count):
            template = pool[int(rng.integers(0, len(pool)))]
            values, nuisance = _augment_template(
                np.asarray(signals[int(template["signal_row"])], dtype=np.float32), rng
            )
            output[offset] = values
            metadata.append(
                {
                    "signal_row": offset,
                    "latent_id": f"{split}-{index:07d}",
                    "view_index": 0,
                    "split": split,
                    "generator_variant": "train-only-real-template-diagnostic",
                    "template_source_record_id": template["record_id"],
                    "template_source_event_id": template["event_id"],
                    "template_source_group": template["source_group"],
                    "template_source_split": template["development_split"],
                    **nuisance,
                }
            )
            offset += 1
    output.flush()
    del output
    with (output_dir / "simulation_metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata[0]))
        writer.writeheader()
        writer.writerows(metadata)
    summary = {
        "schema_version": 1,
        "generator_id": "yeast-train-template-diagnostic-v1",
        "input_contract": "yeast-event-4096-followup-train-normalized-v2",
        "n_signals": len(metadata),
        "split_signal_counts": dict(sorted(Counter(row["split"] for row in metadata).items())),
        "n_template_source_records": {
            split: len({row["record_id"] for row in pool}) for split, pool in sorted(pools.items())
        },
        "template_record_crossings": 0,
        "template_source_split": "followup_train only",
        "retained_physical_factors": False,
        "scientific_scope": (
            "Diagnostic visual-realism upper bound only; not a physical simulator and not eligible for SSL supervision."
        ),
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
