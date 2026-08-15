from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .saturation_cleaning import (
    butter_bandpass_filter,
    repair_saturation_intervals_pre_filter,
)


SIGNAL_LENGTH = 16_384
FS = 2_000_000.0
FMIN = 7_000.0
FMAX = 80_000.0
FILTER_ORDER = 4
REPAIR_METHOD = "cosine-pre-filter"
CANONICAL_MAX_ABSOLUTE_DELTA = 3e-6
CANONICAL_MAX_RELATIVE_L2_DELTA = 1e-6


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable(path: Path, workspace_root: Path) -> str:
    try:
        return path.resolve().relative_to(workspace_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"path is outside workspace: {path}") from exc


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _tree_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != "dataset-manifest.json"
    )


def _payload_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in _tree_files(root):
        relative = path.relative_to(root).as_posix()
        digest.update(f"{relative}\0{sha256_file(path)}\n".encode())
    return digest.hexdigest()


def _raw_index(raw_dataset_roots: dict[str, Path]) -> dict[str, tuple[str, Path]]:
    index: dict[str, tuple[str, Path]] = {}
    for class_name, root in sorted(raw_dataset_roots.items()):
        for path in sorted(root.rglob("*.npy")):
            if path.name in index:
                raise ValueError(f"duplicate raw filename: {path.name}")
            index[path.name] = (class_name, path)
    return index


def _canonical_index(root: Path | None) -> dict[str, Path]:
    if root is None:
        return {}
    index: dict[str, Path] = {}
    for split in ("train", "val"):
        for path in sorted((root / split / "signals").glob("*.npy")):
            if path.name in index:
                raise ValueError(f"duplicate canonical filename: {path.name}")
            index[path.name] = path
    return index


def _repair_rows(path: Path) -> dict[str, list[dict[str, str]]]:
    rows = _read_csv(path)
    required = {
        "filename",
        "class",
        "core_start_sample",
        "core_end_sample",
        "expanded_start_sample",
        "expanded_end_sample",
        "raw_sha256",
        "historical_carrier_path",
        "historical_carrier_slice_sha256",
    }
    fields = set(rows[0]) if rows else set()
    missing = required - fields
    if missing:
        raise ValueError(f"repair manifest missing fields: {sorted(missing)}")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["filename"]].append(row)
    for members in grouped.values():
        members.sort(key=lambda row: int(row["expanded_start_sample"]))
    return dict(grouped)


def _replacement(
    *, workspace_root: Path, row: dict[str, str]
) -> dict[str, Any]:
    start = int(row["expanded_start_sample"])
    end = int(row["expanded_end_sample"])
    carrier_path = workspace_root / row["historical_carrier_path"]
    _portable(carrier_path, workspace_root)
    if not carrier_path.is_file():
        raise FileNotFoundError(carrier_path)
    carrier = np.asarray(np.load(carrier_path, allow_pickle=False)).squeeze()
    replacement = carrier[start:end]
    replacement_sha256 = hashlib.sha256(np.asarray(replacement).tobytes()).hexdigest()
    if replacement_sha256 != row["historical_carrier_slice_sha256"]:
        raise ValueError(f"historical carrier slice drift: {row['filename']}")
    return {
        "core_interval": [
            int(row["core_start_sample"]),
            int(row["core_end_sample"]),
        ],
        "expanded_interval": [start, end],
        "replacement": replacement,
    }


def build_saturation_first_source_dataset(
    *,
    workspace_root: Path,
    predecessor_root: Path,
    frozen_repair_manifest: Path,
    raw_dataset_roots: dict[str, Path],
    raw_dataset_ids: dict[str, str],
    raw_manifest_sha256s: dict[str, str],
    output_dir: Path,
    dataset_id: str,
    predecessor_dataset_id: str,
    predecessor_manifest_sha256: str,
    repair_reference_dataset_id: str,
    repair_reference_manifest_sha256: str,
    canonical_development_root: Path | None = None,
    canonical_development_dataset_id: str | None = None,
    canonical_development_manifest_sha256: str | None = None,
    expected_traces: int | None = None,
    expected_counts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Materialize full split-stable signals with the validated repair chain.

    The repair geometry and replacement carriers come from the immutable full
    saturation-reviewed reference.  The signal operation itself is the same
    saturation-first helper used by the validated development source: repair
    raw, then apply one 7--80 kHz zero-phase bandpass.
    """
    workspace_root = workspace_root.resolve()
    predecessor_root = predecessor_root.resolve()
    frozen_repair_manifest = frozen_repair_manifest.resolve()
    output_dir = output_dir.resolve()
    for path in (
        predecessor_root,
        frozen_repair_manifest,
        output_dir.parent,
        *raw_dataset_roots.values(),
    ):
        _portable(path, workspace_root)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable dataset: {output_dir}")
    source_rows = _read_csv(predecessor_root / "source_manifest.csv")
    if expected_traces is not None and len(source_rows) != expected_traces:
        raise ValueError(
            f"unexpected predecessor population: {len(source_rows)} != {expected_traces}"
        )
    identities = [row["source_id"] for row in source_rows]
    if len(identities) != len(set(identities)):
        raise ValueError("predecessor source identities are not unique")
    raw_index = _raw_index(raw_dataset_roots)
    repairs = _repair_rows(frozen_repair_manifest)
    canonical = _canonical_index(canonical_development_root)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        output_sources: list[dict[str, Any]] = []
        repair_output_rows: list[dict[str, Any]] = []
        canonical_matches = 0
        canonical_max_absolute_delta = 0.0
        canonical_max_relative_l2_delta = 0.0
        for source in source_rows:
            filename = f"{source['source_id']}.npy"
            if Path(source["source_path"]).name != filename:
                raise ValueError(f"source path identity mismatch: {source['source_id']}")
            if filename not in raw_index:
                raise FileNotFoundError(f"raw source missing: {filename}")
            raw_class, raw_path = raw_index[filename]
            if raw_class != source["source_class"]:
                raise ValueError(f"raw class mismatch: {filename}")
            raw = np.asarray(np.load(raw_path, allow_pickle=False)).squeeze()
            if raw.size != SIGNAL_LENGTH:
                raise ValueError(f"unexpected signal length for {filename}: {raw.size}")
            raw_sha256 = sha256_file(raw_path)
            members = repairs.get(filename, [])
            for row in members:
                if row["class"] != source["source_class"]:
                    raise ValueError(f"repair class mismatch: {filename}")
                if row["raw_sha256"] != raw_sha256:
                    raise ValueError(f"repair raw hash mismatch: {filename}")
            if members:
                result = repair_saturation_intervals_pre_filter(
                    raw,
                    [
                        _replacement(workspace_root=workspace_root, row=row)
                        for row in members
                    ],
                    fs=FS,
                    fmin=FMIN,
                    fmax=FMAX,
                    order=FILTER_ORDER,
                )
                filtered = np.asarray(result["filtered_signal"])
                action = REPAIR_METHOD
            else:
                filtered = butter_bandpass_filter(
                    raw, fs=FS, fmin=FMIN, fmax=FMAX, order=FILTER_ORDER
                )
                action = "single-bandpass-unaffected"
            canonical_path = canonical.get(filename)
            canonical_equivalent = False
            canonical_delta = None
            canonical_relative_l2_delta = None
            if canonical_path is not None:
                canonical_signal = np.asarray(
                    np.load(canonical_path, allow_pickle=False)
                ).squeeze()
                canonical_delta = float(
                    np.max(
                        np.abs(
                            np.asarray(filtered, dtype=np.float64)
                            - np.asarray(canonical_signal, dtype=np.float64)
                        )
                    )
                )
                difference = (
                    np.asarray(filtered, dtype=np.float64)
                    - np.asarray(canonical_signal, dtype=np.float64)
                )
                canonical_relative_l2_delta = float(
                    np.linalg.norm(difference)
                    / max(np.linalg.norm(canonical_signal), np.finfo(float).tiny)
                )
                canonical_equivalent = bool(
                    canonical_delta <= CANONICAL_MAX_ABSOLUTE_DELTA
                    and canonical_relative_l2_delta
                    <= CANONICAL_MAX_RELATIVE_L2_DELTA
                )
                if not canonical_equivalent:
                    raise ValueError(
                        "development output differs numerically from validated source: "
                        f"{filename} (max_abs_delta={canonical_delta})"
                    )
                canonical_max_absolute_delta = max(
                    canonical_max_absolute_delta, canonical_delta
                )
                canonical_max_relative_l2_delta = max(
                    canonical_max_relative_l2_delta,
                    canonical_relative_l2_delta,
                )
            canonical_matches += int(canonical_equivalent)
            target = temporary / source["source_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            if canonical_path is not None:
                shutil.copy2(canonical_path, target)
                materialization_action = "copied_validated_saturation_first_development"
            else:
                np.save(target, filtered)
                materialization_action = "computed_frozen_saturation_first_test"
            target_sha256 = sha256_file(target)
            output_sources.append(
                {
                    **source,
                    "pre_correction_source_sha256": source["source_sha256"],
                    "source_sha256": target_sha256,
                    "raw_path": _portable(raw_path, workspace_root),
                    "raw_sha256": raw_sha256,
                    "source_correction": action,
                    "materialization_action": materialization_action,
                    "repair_region_count": len(members),
                    "validated_development_numerically_equivalent": canonical_equivalent,
                    "validated_development_max_abs_delta": canonical_delta,
                    "validated_development_relative_l2_delta": canonical_relative_l2_delta,
                }
            )
            for row in members:
                repair_output_rows.append(
                    {
                        **row,
                        "method": REPAIR_METHOD,
                        "corrected_source_path": source["source_path"],
                        "corrected_source_sha256": target_sha256,
                    }
                )

        unknown_repairs = sorted(set(repairs) - {f"{row['source_id']}.npy" for row in source_rows})
        if unknown_repairs:
            raise ValueError(f"repair manifest contains unknown sources: {unknown_repairs[:3]}")
        _write_csv(temporary / "source_manifest.csv", output_sources)
        _write_csv(temporary / "saturation_repair_manifest.csv", repair_output_rows)
        counts = {
            "traces_total": len(output_sources),
            "traces_by_source_split": dict(Counter(row["source_split"] for row in output_sources)),
            "traces_by_output_split": dict(Counter(row["output_split"] for row in output_sources)),
            "traces_by_class": dict(Counter(row["source_class"] for row in output_sources)),
            "repaired_traces": sum(int(row["repair_region_count"]) > 0 for row in output_sources),
            "repair_regions": len(repair_output_rows),
            "validated_development_numerically_equivalent": canonical_matches,
            "validated_development_max_absolute_delta": canonical_max_absolute_delta,
            "validated_development_max_relative_l2_delta": canonical_max_relative_l2_delta,
        }
        if expected_counts is not None:
            mismatches = {
                key: {"expected": value, "observed": counts.get(key)}
                for key, value in expected_counts.items()
                if counts.get(key) != value
            }
            if mismatches:
                raise RuntimeError(f"source invariant mismatch: {mismatches}")
        contract = {
            "schema_version": 1,
            "dataset_id": dataset_id,
            "format": "npy-class-folders-saturation-first",
            "grain": {
                "source_manifest.csv": "one physical source trace",
                "saturation_repair_manifest.csv": "one expanded saturation repair interval",
            },
            "keys": {
                "source_manifest.csv": ["source_id"],
                "saturation_repair_manifest.csv": ["filename", "interval_idx"],
            },
            "signal": {
                "length_samples": SIGNAL_LENGTH,
                "sampling_frequency_hz": FS,
                "bandpass_hz": [FMIN, FMAX],
                "bandpass_order": FILTER_ORDER,
                "bandpass_passes": 1,
            },
            "repair_policy": (
                "frozen historical carrier blended into raw with raised-cosine guards, "
                "then one canonical bandpass"
            ),
            "split_policy": "source and output split assignments copied byte-for-byte by identity from MAD v1",
        }
        _write_json(temporary / "dataset-contract.json", contract)
        _write_json(temporary / "dataset_contract.json", contract)
        _write_json(
            temporary / "dataset.yaml",
            {
                "path": ".",
                "dataset_id": dataset_id,
                "status": "immutable_candidate",
                "format": contract["format"],
                "splits": counts["traces_by_output_split"],
                "provenance": {
                    "predecessor_dataset_id": predecessor_dataset_id,
                    "repair_reference_dataset_id": repair_reference_dataset_id,
                    "canonical_development_dataset_id": canonical_development_dataset_id,
                },
            },
        )
        digest = _payload_digest(temporary)
        manifest = {
            "schema_version": 1,
            "dataset_id": dataset_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "counts": counts,
            "expected_invariants": expected_counts,
            "method": {
                "repair": REPAIR_METHOD,
                "repair_before_bandpass": True,
                "bandpass_passes": 1,
                "fs_hz": FS,
                "fmin_hz": FMIN,
                "fmax_hz": FMAX,
                "filter_order": FILTER_ORDER,
            },
            "parents": {
                predecessor_dataset_id: predecessor_manifest_sha256,
                repair_reference_dataset_id: repair_reference_manifest_sha256,
                **{
                    raw_dataset_ids[class_name]: raw_manifest_sha256s[class_name]
                    for class_name in sorted(raw_dataset_ids)
                },
                **(
                    {canonical_development_dataset_id: canonical_development_manifest_sha256}
                    if canonical_development_dataset_id and canonical_development_manifest_sha256
                    else {}
                ),
            },
            "frozen_repair_manifest": {
                "path": _portable(frozen_repair_manifest, workspace_root),
                "sha256": sha256_file(frozen_repair_manifest),
            },
            "payload_digest_sha256": digest,
            "files": {
                path.relative_to(temporary).as_posix(): {
                    "sha256": sha256_file(path),
                    "size": path.stat().st_size,
                }
                for path in _tree_files(temporary)
            },
            "claim_boundary": (
                "Signal-lineage correction only; no MAD annotation, physical event adjudication, "
                "model evaluation, or test-set tuning is performed here."
            ),
        }
        _write_json(temporary / "dataset-manifest.json", manifest)
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest
