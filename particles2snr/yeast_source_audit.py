from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


_TRAILING_INDEX = re.compile(r"^(?P<series>.+)_(?P<index>[0-9]+)$")


@dataclass(frozen=True)
class YeastSourceRecord:
    relative_path: str
    source_group: str
    suffix: str
    size_bytes: int
    sha256: str
    filename_series: str
    filename_index: int | None
    npy_shape: str
    npy_dtype: str
    n_values: int | None
    finite_fraction: float | None
    minimum: float | None
    maximum: float | None
    mean: float | None
    std: float | None
    load_error: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_float(value: float) -> float | None:
    result = float(value)
    return result if math.isfinite(result) else None


def _filename_identity(path: Path) -> tuple[str, int | None]:
    match = _TRAILING_INDEX.fullmatch(path.stem)
    if match is None:
        return path.stem, None
    return match.group("series"), int(match.group("index"))


def inspect_source_file(path: Path, source_root: Path) -> YeastSourceRecord:
    relative = path.relative_to(source_root)
    group = relative.parts[0] if len(relative.parts) > 1 else "root"
    series, index = _filename_identity(path)
    shape = ""
    dtype = ""
    n_values: int | None = None
    finite_fraction: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    std: float | None = None
    load_error = ""

    if path.suffix.lower() == ".npy":
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            shape = "x".join(str(value) for value in array.shape)
            dtype = str(array.dtype)
            n_values = int(array.size)
            if np.issubdtype(array.dtype, np.number) and array.size:
                values = np.asarray(array).reshape(-1)
                finite = np.isfinite(values)
                finite_count = int(np.count_nonzero(finite))
                finite_fraction = float(finite_count / values.size)
                if finite_count:
                    clean = values[finite].astype(np.float64, copy=False)
                    minimum = _safe_float(np.min(clean))
                    maximum = _safe_float(np.max(clean))
                    mean = _safe_float(np.mean(clean))
                    std = _safe_float(np.std(clean))
        except Exception as exc:  # Audit malformed inputs instead of aborting the inventory.
            load_error = f"{type(exc).__name__}: {exc}"

    return YeastSourceRecord(
        relative_path=relative.as_posix(),
        source_group=group,
        suffix=path.suffix.lower() or "<none>",
        size_bytes=int(path.stat().st_size),
        sha256=sha256_file(path),
        filename_series=series,
        filename_index=index,
        npy_shape=shape,
        npy_dtype=dtype,
        n_values=n_values,
        finite_fraction=finite_fraction,
        minimum=minimum,
        maximum=maximum,
        mean=mean,
        std=std,
        load_error=load_error,
    )


def inventory_source(source_root: Path) -> list[YeastSourceRecord]:
    root = source_root.resolve()
    if not root.is_dir():
        raise ValueError(f"Yeast source directory does not exist: {root}")
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    if not paths:
        raise ValueError(f"No source files found under {root}")
    return [inspect_source_file(path, root) for path in paths]


def duplicate_groups(records: Iterable[YeastSourceRecord]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[YeastSourceRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.size_bytes, record.sha256)].append(record)

    output: list[dict[str, Any]] = []
    for duplicate_id, ((size_bytes, digest), items) in enumerate(
        sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0][1]))
    ):
        if len(items) < 2:
            continue
        output.append(
            {
                "duplicate_group_id": duplicate_id,
                "sha256": digest,
                "size_bytes": size_bytes,
                "n_files": len(items),
                "source_groups": sorted({item.source_group for item in items}),
                "relative_paths": sorted(item.relative_path for item in items),
            }
        )
    return output


def summarize_inventory(
    records: list[YeastSourceRecord],
    *,
    source_root: Path,
    documented_acquisition_groups: list[str] | None = None,
) -> dict[str, Any]:
    duplicates = duplicate_groups(records)
    npy_records = [record for record in records if record.suffix == ".npy"]
    load_errors = [record for record in npy_records if record.load_error]
    nonfinite = [
        record
        for record in npy_records
        if record.finite_fraction is not None and record.finite_fraction < 1.0
    ]
    acquisition_groups = sorted(set(documented_acquisition_groups or []))
    duplicate_excess_by_group: Counter[str] = Counter()
    cross_group_duplicates = 0
    adjacent_index_duplicates = 0
    records_by_path = {record.relative_path: record for record in records}
    for group in duplicates:
        paths = group["relative_paths"]
        source_groups = group["source_groups"]
        if len(source_groups) > 1:
            cross_group_duplicates += 1
        else:
            duplicate_excess_by_group[source_groups[0]] += int(group["n_files"] - 1)
        items = [records_by_path[path] for path in paths]
        if (
            len(items) == 2
            and items[0].filename_series == items[1].filename_series
            and items[0].filename_index is not None
            and items[1].filename_index is not None
            and abs(items[0].filename_index - items[1].filename_index) == 1
        ):
            adjacent_index_duplicates += 1

    return {
        "schema_version": 1,
        "source_root": str(source_root.resolve()),
        "n_files": len(records),
        "n_npy_files": len(npy_records),
        "total_size_bytes": int(sum(record.size_bytes for record in records)),
        "source_group_counts": dict(sorted(Counter(record.source_group for record in records).items())),
        "suffix_counts": dict(sorted(Counter(record.suffix for record in records).items())),
        "npy_shape_counts": dict(sorted(Counter(record.npy_shape for record in npy_records).items())),
        "npy_dtype_counts": dict(sorted(Counter(record.npy_dtype for record in npy_records).items())),
        "filename_series_counts": dict(sorted(Counter(record.filename_series for record in npy_records).items())),
        "n_load_errors": len(load_errors),
        "load_error_paths": [record.relative_path for record in load_errors],
        "n_files_with_nonfinite_values": len(nonfinite),
        "nonfinite_paths": [record.relative_path for record in nonfinite],
        "n_exact_duplicate_groups": len(duplicates),
        "n_files_in_exact_duplicate_groups": int(sum(group["n_files"] for group in duplicates)),
        "n_exact_duplicate_excess_files": int(sum(group["n_files"] - 1 for group in duplicates)),
        "exact_duplicate_excess_by_source_group": dict(sorted(duplicate_excess_by_group.items())),
        "n_cross_source_group_duplicate_groups": cross_group_duplicates,
        "n_adjacent_index_duplicate_groups": adjacent_index_duplicates,
        "documented_acquisition_groups": acquisition_groups,
        "split_readiness": {
            "status": "pass" if len(acquisition_groups) >= 2 else "fail",
            "reason": (
                "at least two independent acquisition groups were supplied"
                if len(acquisition_groups) >= 2
                else "fewer than two independently documented acquisition groups; folders and files must not be treated as independent acquisitions"
            ),
        },
    }


def write_inventory(
    output_dir: Path,
    records: list[YeastSourceRecord],
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(records[0]).keys())
    with (output_dir / "source_inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))

    duplicate_fields = (
        "duplicate_group_id",
        "sha256",
        "size_bytes",
        "n_files",
        "source_groups",
        "relative_paths",
    )
    with (output_dir / "exact_duplicates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=duplicate_fields)
        writer.writeheader()
        for group in duplicate_groups(records):
            row = dict(group)
            row["source_groups"] = json.dumps(row["source_groups"], separators=(",", ":"))
            row["relative_paths"] = json.dumps(row["relative_paths"], separators=(",", ":"))
            writer.writerow(row)

    (output_dir / "source_inventory_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
