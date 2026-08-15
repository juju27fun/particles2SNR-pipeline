from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .saturation_cleaning import proposal_center_inside_intervals


CLASS_NAMES = {0: "2um", 1: "4um", 2: "10um"}
SAMPLING_FREQUENCY_HZ = 2_000_000.0
SOURCE_LENGTH = 16_384
UNCLEAR_SNR_DB = -10.0
EVENT_FIELDS = (
    "event_id", "split", "source_filename", "source_signal_relative_path",
    "source_class_name", "physical_source_class", "class_id", "class_name",
    "annotation_origin", "detector_annotation_id", "start_norm", "end_norm",
    "center_norm", "proposal_center_norm", "width_norm", "start_sample", "end_sample", "center_sample",
    "start_ms", "end_ms", "particles2snr_amplitude", "frequency_hz", "tau_ms",
    "snr_db", "filtered_peak_z", "clean_local_peak_z",
    "overlaps_saturation_repair", "center_inside_saturation_repair",
)
EXCLUSION_FIELDS = (
    "source_filename", "annotation_origin", "detector_annotation_id",
    "dropped_annotation_index", "start_sample", "end_sample", "center_sample",
    "repair_index", "expanded_start_sample", "expanded_end_sample", "reason",
    "match_count", "snr_db", "frequency_hz",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _workspace_relative(path: Path) -> str:
    resolved = path.resolve()
    for anchor in ("artifacts", "datasets"):
        if anchor in resolved.parts:
            index = resolved.parts.index(anchor)
            return Path(*resolved.parts[index:]).as_posix()
    raise ValueError(f"Path is outside portable workspace roots: {path}")


def _load_run_rows(
    run_dir: Path, splits: tuple[str, ...] = ("train",)
) -> list[dict[str, Any]]:
    rows = []
    for split in splits:
        path = run_dir / split / "data.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        split_rows = payload.get("data")
        if not isinstance(split_rows, list):
            raise ValueError(f"Invalid particles2SNR data payload: {path}")
        rows.extend({**row, "_run_split": split} for row in split_rows)
    filenames = [row["filename"] for row in rows]
    if len(filenames) != len(set(filenames)):
        raise ValueError("Duplicate source filename across detector run splits")
    return rows


def _signal_index(source_root: Path) -> dict[str, tuple[str, str]]:
    result: dict[str, tuple[str, str]] = {}
    for split in ("train", "val"):
        root = source_root / split / "signals"
        for path in sorted(root.glob("*.npy")):
            if path.name in result:
                raise ValueError(f"Signal present in several development splits: {path.name}")
            result[path.name] = (split, f"{split}/signals/{path.name}")
    if not result:
        raise ValueError(f"No development signals found below {source_root}")
    return result


def _saturation_intervals(path: Path) -> dict[str, list[tuple[int, int]]]:
    grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            split = row.get("split", row.get("source_split"))
            if split not in {"train", "val"}:
                continue
            grouped[row["filename"]].append(
                (int(row["expanded_start_sample"]), int(row["expanded_end_sample"]))
            )
    return grouped


def validate_fresh_parent_contract(
    *,
    source_dataset_id: str,
    strict_dataset_id: str,
    source_root: Path,
    strict_run: Path,
    saturation_manifest: Path,
) -> None:
    if source_dataset_id != strict_dataset_id:
        raise ValueError(
            "fresh mode requires the corrected detector dataset as signal parent"
        )
    if (
        saturation_manifest.resolve()
        != (source_root / "saturation_repair_manifest.csv").resolve()
    ):
        raise ValueError(
            "fresh mode requires the saturation manifest inside its signal parent"
        )
    run = json.loads((strict_run / "run.json").read_text(encoding="utf-8"))
    if run.get("dataset") != source_dataset_id:
        raise ValueError("fresh detector run is not bound to its signal dataset")


def _overlaps(
    annotation: dict[str, Any],
    intervals: list[tuple[int, int]],
    *,
    length: int,
) -> bool:
    start = float(annotation["start"]) * length
    end = float(annotation["end"]) * length
    return any(max(start, left) < min(end, right) for left, right in intervals)


def _center_inside_repair(
    annotation: dict[str, Any],
    intervals: list[tuple[int, int]],
    *,
    length: int,
) -> tuple[float, int | None, tuple[int, int] | None]:
    """Return the proposal centre and its first inclusive repair match.

    The Z8 policy is deliberately a centre rule, rather than an overlap rule:
    a real event can overlap the edge of a repaired interval while its centre is
    outside it.  Compute the centre from the proposal bounds instead of trusting
    a detector-provided centre field so that strict and rescue proposals use the
    exact same decision rule.
    """
    return proposal_center_inside_intervals(
        float(annotation["start"]) * length,
        float(annotation["end"]) * length,
        intervals,
    )


def _saturation_veto_exclusion(
    *,
    filename: str,
    annotation: dict[str, Any],
    annotation_origin: str,
    center_sample: float,
    repair_index: int,
    repair_interval: tuple[int, int],
) -> dict[str, Any]:
    """Build one fully attributable record for a hard-vetoed proposal."""
    detector_id = annotation.get("detector_annotation_id", annotation.get("id"))
    return {
        "source_filename": filename,
        "annotation_origin": annotation_origin,
        "detector_annotation_id": int(detector_id),
        "start_sample": float(annotation["start"]) * SOURCE_LENGTH,
        "end_sample": float(annotation["end"]) * SOURCE_LENGTH,
        "center_sample": center_sample,
        "repair_index": repair_index,
        "expanded_start_sample": repair_interval[0],
        "expanded_end_sample": repair_interval[1],
        "reason": "z8_center_inside_saturation_repair",
    }


def _label(
    annotation: dict[str, Any], source_class_name: str
) -> tuple[int, str, str]:
    annotation_class_id = int(annotation["class_id"])
    physical_name = str(source_class_name)
    if physical_name not in CLASS_NAMES.values():
        physical_name = CLASS_NAMES.get(annotation_class_id, physical_name)
    if (
        annotation_class_id == 3
        or float(annotation["snr_db"]) < UNCLEAR_SNR_DB
    ):
        return 3, "unclear", physical_name
    if annotation_class_id not in CLASS_NAMES:
        raise ValueError(
            f"unexpected physical annotation class: {annotation_class_id}"
        )
    return annotation_class_id, CLASS_NAMES[annotation_class_id], physical_name


def _event_row(
    *,
    annotation: dict[str, Any],
    annotation_origin: str,
    development_split: str,
    source_relative_path: str,
    filename: str,
    source_class_name: str,
    saturation_overlap: bool,
    center_inside_saturation_repair: bool,
    clean_local_peak_z: float | None,
    event_namespace: str,
) -> dict[str, Any]:
    class_id, class_name, physical_source_class = _label(
        annotation, source_class_name
    )
    length = SOURCE_LENGTH
    start = float(annotation["start"])
    end = float(annotation["end"])
    event_key = (
        f"{event_namespace}:{development_split}:{Path(filename).stem}:"
        f"{annotation_origin}:"
        f"{int(annotation.get('detector_annotation_id', annotation['id']))}:"
        f"{start:.12f}:{end:.12f}"
    )
    return {
        "event_id": hashlib.sha256(event_key.encode()).hexdigest()[:20],
        "split": development_split,
        "source_filename": filename,
        "source_signal_relative_path": source_relative_path,
        "source_class_name": source_class_name,
        "physical_source_class": physical_source_class,
        "class_id": class_id,
        "class_name": class_name,
        "annotation_origin": annotation_origin,
        "detector_annotation_id": int(
            annotation.get("detector_annotation_id", annotation["id"])
        ),
        "start_norm": start,
        "end_norm": end,
        "center_norm": float(annotation["center"]),
        "proposal_center_norm": (start + end) / 2.0,
        "width_norm": end - start,
        "start_sample": start * length,
        "end_sample": end * length,
        "center_sample": (start + end) / 2.0 * length,
        "start_ms": start * length / SAMPLING_FREQUENCY_HZ * 1000.0,
        "end_ms": end * length / SAMPLING_FREQUENCY_HZ * 1000.0,
        "particles2snr_amplitude": float(annotation["amplitude"]),
        "frequency_hz": float(annotation["frequency"]),
        "tau_ms": float(annotation["passage_time_ms"]),
        "snr_db": float(annotation["snr_db"]),
        "filtered_peak_z": float(annotation["peak_z"]),
        "clean_local_peak_z": clean_local_peak_z,
        "overlaps_saturation_repair": saturation_overlap,
        "center_inside_saturation_repair": center_inside_saturation_repair,
    }


def build_z8_reference_event_table(
    *,
    source_root: Path,
    historical_run: Path | None,
    strict_run: Path,
    saturation_manifest: Path,
    output_dir: Path,
    source_dataset_id: str,
    source_manifest_sha256: str,
    strict_dataset_id: str,
    strict_manifest_sha256: str,
    output_dataset_id: str = (
        "particles2snr-fbase-dual-clean-z8-events-3class-plus-"
        "unclear-development@v1"
    ),
    strict_run_splits: tuple[str, ...] = ("train",),
    fresh_detector_mode: bool = False,
    expected_class_counts: dict[str, int] | None = None,
    expected_development_signal_count: int | None = 2310,
    filtered_min_z: float = 8.0,
    clean_local_min_z: float = 1.5,
    saturation_center_veto: bool = False,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite dataset: {output_dir}")
    signal_index = _signal_index(source_root)
    strict_rows = _load_run_rows(strict_run, strict_run_splits)
    strict_index = {row["filename"]: row for row in strict_rows}
    source_filenames = set(signal_index)
    strict_filenames = set(strict_index)
    if strict_filenames != source_filenames:
        missing = sorted(source_filenames - strict_filenames)
        extra = sorted(strict_filenames - source_filenames)
        raise ValueError(
            "Detector run/source index mismatch: "
            f"missing={missing[:5]} ({len(missing)} total), "
            f"extra={extra[:5]} ({len(extra)} total)"
        )
    if fresh_detector_mode:
        split_mismatches = sorted(
            filename
            for filename, row in strict_index.items()
            if row["_run_split"] != signal_index[filename][0]
        )
        if split_mismatches:
            raise ValueError(
                "Detector run/source split mismatch: "
                f"{split_mismatches[:5]} ({len(split_mismatches)} total)"
            )
    historical_lookup = {}
    if fresh_detector_mode:
        if historical_run is not None:
            raise ValueError(
                "fresh detector mode does not accept a historical annotation run"
            )
    else:
        if historical_run is None:
            raise ValueError("legacy mode requires historical_run")
        historical_rows = _load_run_rows(historical_run)
        historical_lookup = {row["filename"]: row for row in historical_rows}
        if len(historical_lookup) != len(historical_rows):
            raise ValueError("Duplicate historical source filename")
    saturation = _saturation_intervals(saturation_manifest)

    events: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for strict in strict_rows:
        filename = strict["filename"]
        if filename not in signal_index:
            raise FileNotFoundError(f"Development F-base signal missing: {filename}")
        split, source_relative = signal_index[filename]
        historical = historical_lookup.get(filename)
        if not fresh_detector_mode and historical is None:
            raise ValueError(f"Historical source row missing: {filename}")
        length = int(strict["length"] if fresh_detector_mode else historical["length"])
        if length != SOURCE_LENGTH:
            raise ValueError(f"Unexpected source length for {filename}: {length}")
        intervals = saturation.get(filename, [])
        if fresh_detector_mode:
            detector_ids = [
                annotation.get("detector_annotation_id")
                for annotation in strict["annotations"]
            ]
            detector_ids.extend(
                dropped.get("detector_annotation_id")
                for dropped in strict.get("dropped_annotations", [])
                if dropped.get("reason") == "missing_clean_peak_support"
            )
            if any(value is None for value in detector_ids):
                raise ValueError(
                    f"fresh detector annotations lack exact IDs: {filename}"
                )
            if len(detector_ids) != len(set(detector_ids)):
                raise ValueError(
                    f"duplicate fresh detector annotation IDs: {filename}"
                )
        for annotation in strict["annotations"]:
            center_sample, repair_index, repair_interval = _center_inside_repair(
                annotation, intervals, length=length
            )
            if saturation_center_veto and repair_interval is not None:
                exclusions.append(
                    _saturation_veto_exclusion(
                        filename=filename,
                        annotation=annotation,
                        annotation_origin="dual_clean_strict",
                        center_sample=center_sample,
                        repair_index=repair_index,
                        repair_interval=repair_interval,
                    )
                )
                continue
            events.append(
                _event_row(
                    annotation=annotation,
                    annotation_origin="dual_clean_strict",
                    development_split=split,
                    source_relative_path=source_relative,
                    filename=filename,
                    source_class_name=strict["class_name"],
                    saturation_overlap=_overlaps(annotation, intervals, length=length),
                    center_inside_saturation_repair=False,
                    clean_local_peak_z=float(annotation["clean_local_peak_z"]),
                    event_namespace=output_dataset_id,
                )
            )
        for dropped_index, dropped in enumerate(strict.get("dropped_annotations", [])):
            if dropped.get("reason") != "missing_clean_peak_support":
                continue
            if fresh_detector_mode:
                annotation = dropped.get("candidate_annotation")
                if not isinstance(annotation, dict):
                    raise ValueError(
                        "fresh detector drop lacks candidate_annotation: "
                        f"{filename}:{dropped_index}"
                    )
                if annotation.get("detector_annotation_id") is None:
                    raise ValueError(
                        "fresh detector drop lacks detector_annotation_id: "
                        f"{filename}:{dropped_index}"
                    )
                annotation = {**annotation, "class_id": int(strict["class_id"])}
            else:
                matches = [
                    annotation
                    for annotation in historical["annotations"]
                    if abs(float(annotation["snr_db"]) - float(dropped["snr_db"])) < 1e-7
                    and abs(float(annotation["frequency"]) - float(dropped["frequency"])) < 1e-7
                ]
                if len(matches) != 1:
                    exclusions.append(
                        {
                            "source_filename": filename,
                            "dropped_annotation_index": dropped_index,
                            "reason": "ambiguous_historical_join",
                            "match_count": len(matches),
                            "snr_db": dropped["snr_db"],
                            "frequency_hz": dropped["frequency"],
                        }
                    )
                    continue
                annotation = matches[0]
            filtered_z = annotation.get("peak_z")
            clean_z = dropped.get("local_peak_z")
            if filtered_z is None or clean_z is None:
                exclusions.append(
                    {
                        "source_filename": filename,
                        "dropped_annotation_index": dropped_index,
                        "reason": "missing_z_evidence",
                        "match_count": 1,
                        "snr_db": dropped["snr_db"],
                        "frequency_hz": dropped["frequency"],
                    }
                )
                continue
            if float(filtered_z) < filtered_min_z or float(clean_z) < clean_local_min_z:
                continue
            center_sample, repair_index, repair_interval = _center_inside_repair(
                annotation, intervals, length=length
            )
            if saturation_center_veto and repair_interval is not None:
                exclusions.append(
                    _saturation_veto_exclusion(
                        filename=filename,
                        annotation=annotation,
                        annotation_origin="z8_rescue",
                        center_sample=center_sample,
                        repair_index=repair_index,
                        repair_interval=repair_interval,
                    )
                )
                continue
            events.append(
                _event_row(
                    annotation=annotation,
                    annotation_origin="z8_rescue",
                    development_split=split,
                    source_relative_path=source_relative,
                    filename=filename,
                    source_class_name=strict["class_name"],
                    saturation_overlap=_overlaps(annotation, intervals, length=length),
                    center_inside_saturation_repair=False,
                    clean_local_peak_z=float(clean_z),
                    event_namespace=output_dataset_id,
                )
            )

    event_ids = [row["event_id"] for row in events]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("Duplicate event IDs")
    counts = Counter(row["class_name"] for row in events)
    origins = Counter(row["annotation_origin"] for row in events)
    exclusion_counts = Counter(row["reason"] for row in exclusions)
    if expected_class_counts is not None and dict(counts) != expected_class_counts:
        raise ValueError(
            f"Projected class counts changed: {dict(counts)} != "
            f"{expected_class_counts}"
        )
    if (
        expected_development_signal_count is not None
        and len(strict_rows) != expected_development_signal_count
    ):
        raise ValueError(
            f"Expected {expected_development_signal_count} development signals, "
            f"got {len(strict_rows)}"
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        with (temporary / "events.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=EVENT_FIELDS)
            writer.writeheader()
            writer.writerows(events)
        with (temporary / "exclusions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=EXCLUSION_FIELDS)
            writer.writeheader()
            writer.writerows(exclusions)
        summary = {
            "schema_version": 1,
            "dataset_id": output_dataset_id,
            "source_datasets": {
                source_dataset_id: source_manifest_sha256,
                strict_dataset_id: strict_manifest_sha256,
            },
            "source_runs": {
                "strict_dual_clean": _workspace_relative(strict_run),
                **(
                    {"historical_fbase": _workspace_relative(historical_run)}
                    if historical_run is not None
                    else {}
                ),
            },
            "event_count": len(events),
            "class_counts": dict(sorted(counts.items())),
            "origin_counts": dict(sorted(origins.items())),
            "development_signal_count": len(signal_index),
            "exclusion_counts": dict(sorted(exclusion_counts.items())),
            "saturation_manifest_sha256": sha256_file(saturation_manifest),
            "policy": {
                "filtered_min_z": filtered_min_z,
                "clean_local_min_z": clean_local_min_z,
                "saturation_center_veto": saturation_center_veto,
                "saturation_center_veto_rule": (
                    "inclusive center within expanded repair interval"
                    if saturation_center_veto
                    else "disabled"
                ),
                "fresh_detector_mode": fresh_detector_mode,
                "event_namespace": output_dataset_id,
                "unclear_snr_db_below": UNCLEAR_SNR_DB,
            },
            "sealed_test_accessed": False,
            "known_limitations": [
                *(
                    []
                    if fresh_detector_mode
                    else [
                        "legacy rescue matching uses SNR/frequency joins",
                    ]
                ),
                "F-base signals are saturation-repaired and 7-80 kHz filtered",
            ],
        }
        contract = {
            "schema_version": 1,
            "format": "event-reference-table",
            "signal_storage": "resolved from registered F-base parent; no copied signal view",
            "sampling_frequency_hz": SAMPLING_FREQUENCY_HZ,
            "source_length": SOURCE_LENGTH,
            "splits": ["train", "val"],
            "sealed_splits": ["test"],
            "class_mapping": {"0": "2um", "1": "4um", "2": "10um", "3": "unclear"},
            "unclear_role": "SNR/noise coverage only; never a physical synthesis class",
        }
        for filename, payload in (
            ("dataset_summary.json", summary),
            ("input_contract.json", contract),
        ):
            (temporary / filename).write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return summary


def validate_z8_reference_event_table(
    root: Path, *, saturation_manifest: Path | None = None
) -> dict[str, Any]:
    summary = json.loads((root / "dataset_summary.json").read_text(encoding="utf-8"))
    with (root / "events.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if tuple(reader.fieldnames or ()) != EVENT_FIELDS:
            raise ValueError("events.csv schema differs from the Z8 contract")
    exclusions_path = root / "exclusions.csv"
    if not exclusions_path.is_file():
        raise ValueError("exclusions.csv is missing")
    with exclusions_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        exclusions = list(reader)
        if tuple(reader.fieldnames or ()) != EXCLUSION_FIELDS:
            raise ValueError("exclusions.csv schema differs from the Z8 contract")
    if len(rows) != int(summary["event_count"]):
        raise ValueError("events.csv count differs from dataset summary")
    if any(row["split"] == "test" for row in rows):
        raise ValueError("Sealed test split included")
    if len({row["event_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate event IDs")
    counts = Counter(row["class_name"] for row in rows)
    if dict(sorted(counts.items())) != summary["class_counts"]:
        raise ValueError("Class counts differ from dataset summary")
    for row in rows:
        if not 0.0 <= float(row["start_norm"]) < float(row["end_norm"]) <= 1.0:
            raise ValueError(f"Invalid event bounds: {row['event_id']}")
        if (
            row.get("center_inside_saturation_repair") is not None
            and row["center_inside_saturation_repair"] != "False"
        ):
            raise ValueError("A Z8 event centre is inside a saturation repair")
        midpoint = (float(row["start_norm"]) + float(row["end_norm"])) / 2.0
        if abs(float(row["proposal_center_norm"]) - midpoint) > 1e-12:
            raise ValueError("Event proposal_center_norm differs from its bounds midpoint")
        if abs(float(row["center_sample"]) - midpoint * SOURCE_LENGTH) > 1e-7:
            raise ValueError("Event center_sample differs from its bounds midpoint")
    exclusion_counts = Counter(row["reason"] for row in exclusions)
    if dict(sorted(exclusion_counts.items())) != summary["exclusion_counts"]:
        raise ValueError("Exclusion counts differ from dataset summary")
    allowed_reasons = {
        "z8_center_inside_saturation_repair",
        "ambiguous_historical_join",
        "missing_z_evidence",
    }
    if any(row["reason"] not in allowed_reasons for row in exclusions):
        raise ValueError("Unknown Z8 exclusion reason")
    vetoes = [
        row
        for row in exclusions
        if row["reason"] == "z8_center_inside_saturation_repair"
    ]
    identities = [
        (row["source_filename"], row["annotation_origin"], row["detector_annotation_id"])
        for row in vetoes
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate hard-veto detector identity")
    for row in vetoes:
        if row["annotation_origin"] not in {"dual_clean_strict", "z8_rescue"}:
            raise ValueError("Hard veto has an invalid annotation origin")
        left = float(row["expanded_start_sample"])
        right = float(row["expanded_end_sample"])
        start = float(row["start_sample"])
        end = float(row["end_sample"])
        center = float(row["center_sample"])
        if not 0.0 <= left <= right <= SOURCE_LENGTH:
            raise ValueError("Hard-veto repair bounds are invalid")
        if not 0.0 <= start < end <= SOURCE_LENGTH:
            raise ValueError("Hard-veto proposal bounds are invalid")
        if abs(center - (start + end) / 2.0) > 1e-7:
            raise ValueError("Hard-veto centre differs from proposal midpoint")
        if not left <= center <= right:
            raise ValueError("Hard-veto centre is outside the repair interval")
    if saturation_manifest is not None:
        if sha256_file(saturation_manifest) != summary.get("saturation_manifest_sha256"):
            raise ValueError("Saturation manifest hash differs from dataset summary")
        intervals = _saturation_intervals(saturation_manifest)
        for row in vetoes:
            interval = (
                int(row["expanded_start_sample"]),
                int(row["expanded_end_sample"]),
            )
            if interval not in intervals.get(row["source_filename"], []):
                raise ValueError("Hard-veto interval is absent from saturation manifest")
        for row in rows:
            center = float(row["center_sample"])
            if any(
                left <= center <= right
                for left, right in intervals.get(row["source_filename"], [])
            ):
                raise ValueError("A Z8 event centre is inside a saturation manifest interval")
    return {
        "valid": True,
        "events": len(rows),
        "class_counts": dict(sorted(counts.items())),
        "sealed_test_accessed": False,
    }
