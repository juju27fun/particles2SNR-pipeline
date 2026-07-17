from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


OUTPUT_LENGTH = 4096
DEVELOPMENT_SPLITS = ("development_train", "development_validation")
KNOWN_SOURCE_SPLITS = frozenset(
    (*DEVELOPMENT_SPLITS, "in_session_test", "sealed_acquisition_test")
)
SOURCE_FILES = ("signals.npy", "events.csv", "input_contract.json", "dataset_summary.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_events(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError(f"Missing CSV header in {path}")
        rows = list(reader)
    required = {"development_split", "signal_row"}
    missing = sorted(required - set(fieldnames))
    if missing:
        raise ValueError(f"events.csv is missing required columns: {missing}")
    if "parent_signal_row" in fieldnames:
        raise ValueError("events.csv already contains parent_signal_row")
    return fieldnames, rows


def _validate_source(
    *,
    rows: list[dict[str, str]],
    signals: np.ndarray,
    contract: dict[str, Any],
    summary: dict[str, Any],
) -> Counter[str]:
    if signals.ndim != 2 or signals.shape[1] != OUTPUT_LENGTH:
        raise ValueError(
            f"Expected signals shape (N, {OUTPUT_LENGTH}), got {tuple(signals.shape)}"
        )
    if signals.shape[0] != len(rows):
        raise ValueError(
            f"signals/events row count mismatch: {signals.shape[0]} signals, {len(rows)} events"
        )
    if contract.get("output_length") != OUTPUT_LENGTH:
        raise ValueError(f"Parent input contract must declare output_length={OUTPUT_LENGTH}")

    expected_shape = [len(rows), OUTPUT_LENGTH]
    if summary.get("n_events") != len(rows) or summary.get("signals_shape") != expected_shape:
        raise ValueError("Parent dataset summary does not match events/signals shape")
    if summary.get("signals_dtype") != str(signals.dtype):
        raise ValueError("Parent dataset summary does not match signals dtype")

    split_counts = Counter(row["development_split"] for row in rows)
    invalid_splits = sorted(set(split_counts) - KNOWN_SOURCE_SPLITS)
    if invalid_splits:
        raise ValueError(f"Unexpected or empty development splits: {invalid_splits}")
    empty_splits = [split for split in DEVELOPMENT_SPLITS if split_counts[split] == 0]
    if empty_splits:
        raise ValueError(f"Required development splits are empty: {empty_splits}")
    if summary.get("split_counts") != dict(sorted(split_counts.items())):
        raise ValueError("Parent dataset summary split counts do not match events.csv")

    signal_rows: list[int] = []
    for event_index, row in enumerate(rows):
        try:
            signal_row = int(row["signal_row"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid signal_row at event row {event_index}") from exc
        if signal_row < 0 or signal_row >= signals.shape[0]:
            raise ValueError(f"signal_row out of range at event row {event_index}: {signal_row}")
        signal_rows.append(signal_row)
    if len(set(signal_rows)) != len(signal_rows):
        raise ValueError("Parent signal_row values are not unique")
    if set(signal_rows) != set(range(signals.shape[0])):
        raise ValueError("Parent signal_row values do not cover signals.npy contiguously")
    return split_counts


def build_development_dataset(
    *,
    input_root: Path,
    output_dir: Path,
    source_dataset_id: str = "yeast-events-representation@v3",
    output_dataset_id: str = "yeast-events-development@v1",
) -> dict[str, Any]:
    """Create a physical derivative containing development rows and signals only."""
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    source_paths = {name: input_root / name for name in SOURCE_FILES}
    missing = [name for name, path in source_paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Input dataset is missing required files: {missing}")

    fieldnames, rows = _read_events(source_paths["events.csv"])
    parent_contract = _read_json(source_paths["input_contract.json"])
    parent_summary = _read_json(source_paths["dataset_summary.json"])
    source_signals = np.load(source_paths["signals.npy"], mmap_mode="r", allow_pickle=False)
    source_split_counts = _validate_source(
        rows=rows,
        signals=source_signals,
        contract=parent_contract,
        summary=parent_summary,
    )
    parent_contract_id = parent_contract.get("contract_id")
    if not isinstance(parent_contract_id, str) or not parent_contract_id:
        raise ValueError("Parent input contract has no contract_id")

    selected = [row for row in rows if row["development_split"] in DEVELOPMENT_SPLITS]
    source_checksums = {name: _sha256(path) for name, path in sorted(source_paths.items())}

    output_dir.mkdir(parents=True, exist_ok=False)
    output_signals = np.lib.format.open_memmap(
        output_dir / "signals.npy",
        mode="w+",
        dtype=source_signals.dtype,
        shape=(len(selected), OUTPUT_LENGTH),
    )
    output_rows: list[dict[str, str]] = []
    for output_index, source_row in enumerate(selected):
        parent_signal_row = int(source_row["signal_row"])
        output_signals[output_index] = source_signals[parent_signal_row]
        output_row = dict(source_row)
        output_row["signal_row"] = str(output_index)
        output_row["parent_signal_row"] = str(parent_signal_row)
        output_rows.append(output_row)
    output_signals.flush()
    del output_signals
    del source_signals

    with (output_dir / "events.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*fieldnames, "parent_signal_row"])
        writer.writeheader()
        writer.writerows(output_rows)

    contract = dict(parent_contract)
    contract.update(
        {
            "parent_contract": parent_contract_id,
            "parent_dataset": source_dataset_id,
            "split_scope": (
                "physical development_train and development_validation derivative; "
                "sealed and test signals absent"
            ),
            "provenance": {
                "derivation": "physical-development-only-row-selection",
                "selected_splits": list(DEVELOPMENT_SPLITS),
                "sealed_splits_used": [],
                "source_checksums": source_checksums,
                "source_dataset_id": source_dataset_id,
            },
        }
    )
    _write_json(output_dir / "input_contract.json", contract)

    output_checksums = {
        name: _sha256(output_dir / name)
        for name in ("events.csv", "input_contract.json", "signals.npy")
    }
    selected_split_counts = Counter(row["development_split"] for row in output_rows)
    excluded_split_counts = {
        split: count
        for split, count in sorted(source_split_counts.items())
        if split not in DEVELOPMENT_SPLITS
    }
    summary = {
        "schema_version": 1,
        "dataset_id": output_dataset_id,
        "source_dataset_id": source_dataset_id,
        "n_events": len(output_rows),
        "n_source_events": len(rows),
        "split_counts": dict(sorted(selected_split_counts.items())),
        "excluded_split_counts": excluded_split_counts,
        "sealed_splits_used": [],
        "input_contract": parent_contract_id,
        "signals_shape": [len(output_rows), OUTPUT_LENGTH],
        "signals_dtype": str(np.dtype(parent_summary["signals_dtype"])),
        "source_checksums": source_checksums,
        "output_checksums": output_checksums,
    }
    _write_json(output_dir / "dataset_summary.json", summary)
    return summary
