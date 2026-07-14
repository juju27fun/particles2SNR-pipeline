from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def read_raw_dataset_map(path: Path) -> dict[str, Path]:
    manifest = path.resolve()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Raw dataset map schema_version must be 1")
    datasets = payload.get("raw_datasets")
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError("Raw dataset map must contain a non-empty raw_datasets object")
    output: dict[str, Path] = {}
    for dataset_id, value in datasets.items():
        candidate = Path(str(value))
        output[str(dataset_id)] = (
            candidate.resolve() if candidate.is_absolute() else (manifest.parent / candidate).resolve()
        )
    return output


def normalize_raw_dataset_roots(
    *,
    raw_dataset_root: Path | None,
    raw_dataset_roots: Mapping[str, Path] | None,
) -> tuple[Path | None, dict[str, Path]]:
    if raw_dataset_root is not None and raw_dataset_roots:
        raise ValueError("Provide one raw dataset root or a dataset-ID-to-root map, not both")
    if raw_dataset_root is None and not raw_dataset_roots:
        raise ValueError("A raw dataset root or raw dataset map is required")
    single = raw_dataset_root.resolve() if raw_dataset_root is not None else None
    mapped = {str(key): Path(value).resolve() for key, value in (raw_dataset_roots or {}).items()}
    missing = [dataset_id for dataset_id, root in mapped.items() if not root.is_dir()]
    if missing:
        raise ValueError(f"Raw dataset roots do not exist for IDs: {sorted(missing)}")
    if single is not None and not single.is_dir():
        raise ValueError(f"Raw dataset root does not exist: {single}")
    return single, mapped


def resolve_raw_signal(
    row: Mapping[str, Any],
    *,
    single_root: Path | None,
    roots_by_dataset: Mapping[str, Path],
) -> Path:
    if single_root is not None:
        root = single_root
    else:
        dataset_id = str(row.get("raw_dataset", ""))
        if not dataset_id:
            raise ValueError("Multi-acquisition rows must declare raw_dataset")
        try:
            root = roots_by_dataset[dataset_id]
        except KeyError as exc:
            raise ValueError(f"No raw root registered in the map for dataset {dataset_id}") from exc
    signal = (root / str(row["relative_path"])).resolve()
    if not signal.is_relative_to(root):
        raise ValueError(f"Raw signal path escapes its registered dataset root: {signal}")
    return signal
