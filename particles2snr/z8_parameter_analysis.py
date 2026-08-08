from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from internship_workspace.config import Workspace
from internship_workspace.datasets import index_path, load_records, validate_record
from internship_workspace.scientific_visual import resolve_visual_evidence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


CLASS_ORDER = ("2um", "4um", "10um")
CLASS_LABELS = {"2um": "2 µm", "4um": "4 µm", "10um": "10 µm"}
CLASS_COLORS = {"2um": "#2563eb", "4um": "#16a34a", "10um": "#ea580c"}
FBASE_NOMINAL_BAND_KHZ = (7.0, 80.0)
MARGIN_FRACTIONS = {"M0": 0.0, "M10": 0.10, "M20": 0.20}
QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)

PARAMETERS: dict[str, dict[str, Any]] = {
    "amplitude_p0": {
        "source_column": "particles2snr_amplitude",
        "scale": 1.0,
        "units": "acquisition_units",
        "label": "Amplitude P0",
        "population": "physical_class_only",
        "positive": True,
    },
    "frequency_khz": {
        "source_column": "frequency_hz",
        "scale": 1.0 / 1000.0,
        "units": "kHz",
        "label": "Frequency",
        "population": "physical_class_only",
        "positive": True,
    },
    "tau_ms": {
        "source_column": "tau_ms",
        "scale": 1.0,
        "units": "ms",
        "label": "Tau",
        "population": "physical_class_only",
        "positive": True,
    },
    "snr_effective_fbase_db": {
        "source_column": "snr_db",
        "scale": 1.0,
        "units": "dB",
        "label": "Effective F-base SNR",
        "population": "physical_source_class_including_unclear",
        "positive": False,
    },
}


REQUIRED_EVENT_COLUMNS = {
    "event_id", "split", "source_filename", "source_signal_relative_path",
    "physical_source_class", "class_name", "annotation_origin", "start_sample",
    "end_sample", "particles2snr_amplitude", "frequency_hz", "tau_ms", "snr_db",
}


def observed_population_counts(rows: Iterable[dict[str, str]]) -> dict[str, dict[str, int]]:
    rows = list(rows)
    return {
        "class_counts": dict(Counter(row["class_name"] for row in rows)),
        "snr_population_counts": dict(
            Counter(row["physical_source_class"] for row in rows)
        ),
        "split_counts": dict(Counter(row["split"] for row in rows)),
    }


def load_dataset_summary(dataset_root: Path) -> dict[str, Any]:
    path = dataset_root / "dataset_summary.json"
    if not path.is_file():
        raise ValueError(f"Registered z8 dataset is missing dataset_summary.json: {path}")
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid dataset summary: {path}") from exc
    if not isinstance(summary, dict) or summary.get("schema_version") != 1:
        raise ValueError("Dataset summary must declare schema_version 1")
    if not isinstance(summary.get("dataset_id"), str):
        raise ValueError("Dataset summary must contain dataset_id")
    if summary.get("sealed_test_accessed") is not False:
        raise ValueError("Dataset summary must prove sealed_test_accessed is false")
    if not isinstance(summary.get("event_count"), int):
        raise ValueError("Dataset summary must contain integer event_count")
    if not isinstance(summary.get("class_counts"), dict):
        raise ValueError("Dataset summary must contain class_counts")
    if not isinstance(summary.get("origin_counts"), dict):
        raise ValueError("Dataset summary must contain origin_counts")
    if not isinstance(summary.get("source_datasets"), dict):
        raise ValueError("Dataset summary must contain source_datasets")
    return summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_registered_z8_dataset(
    workspace: Workspace, key: str, *, require_z8_summary: bool = True
) -> tuple[dict[str, Any], Path, dict[str, Any] | None]:
    records = [record for record in load_records(workspace) if record.key == key]
    if len(records) != 1:
        raise ValueError(f"Expected exactly one registered dataset: {key}")
    record = records[0]
    payload = record.payload
    if payload.get("status") not in {"active", "reference"}:
        raise ValueError(f"Dataset is not eligible: {key}")
    required = {"id", "version", "status", "path", "manifest", "manifest_sha256", "format", "producer"}
    if required - set(payload):
        raise ValueError(f"Dataset registry record is incomplete: {key}")
    errors = validate_record(workspace, record, full=True)
    if errors:
        raise ValueError(f"Dataset payload validation failed: {'; '.join(errors)}")
    manifest = index_path(workspace).parent / str(payload["manifest"])
    if _sha256(manifest) != payload["manifest_sha256"]:
        raise ValueError("Registry manifest hash does not match its bytes")
    root = (workspace.datasets_root / str(payload["path"])).resolve()
    entries = {json.loads(line)["path"]: json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line}
    if require_z8_summary:
        events = root / "events.csv"
        event_entry = entries.get("events.csv")
        if event_entry is None or _sha256(events) != event_entry.get("sha256"):
            raise ValueError("events.csv is not the hash-validated registered payload")
    summary = load_dataset_summary(root) if require_z8_summary else None
    if summary is not None and summary["dataset_id"] != key:
        raise ValueError("Dataset summary ID does not match selected registry dataset")
    return payload, root, summary


def validate_method_evidence(
    workspace: Workspace, *, evidence_id: str, evidence_run_id: str,
    contract_path: Path, dataset_id: str, method: str,
) -> dict[str, Any]:
    """Require a receipt-bound, dataset-specific method approval contract."""
    contract_path = contract_path.resolve()
    if not contract_path.is_file() or not contract_path.is_relative_to(workspace.root):
        raise ValueError("Evidence applicability contract must be a workspace-relative file")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict) or contract.get("schema_version") != 1:
        raise ValueError("Invalid evidence applicability contract")
    if contract.get("evidence_id") != evidence_id or contract.get("evidence_run_id") != evidence_run_id:
        raise ValueError("Evidence contract ID/run does not match requested evidence")
    if contract.get("dataset_id") != dataset_id or contract.get("method") != method:
        raise ValueError("Evidence contract is not applicable to this dataset/method")
    resolved = resolve_visual_evidence(workspace, evidence_id)
    if resolved.get("source") != "unified":
        raise ValueError("Method evidence requires unified receipt-bound evidence")
    entry = resolved["entry"]
    if entry.get("artifact", {}).get("run_id") != evidence_run_id:
        raise ValueError("Evidence run does not match the evidence index")
    if entry.get("outcome", {}).get("code") != "approved":
        raise ValueError("Method evidence is not approved")
    datasets = entry.get("datasets")
    if not isinstance(datasets, list):
        raise ValueError("Method evidence lacks an explicit dataset scope")
    contract_datasets = contract.get("evidence_dataset_ids")
    if contract_datasets is None:
        contract_datasets = [dataset_id]
    if (
        not isinstance(contract_datasets, list)
        or not all(isinstance(value, str) for value in contract_datasets)
        or datasets != contract_datasets
        or dataset_id not in contract_datasets
    ):
        raise ValueError("Evidence entry dataset scope does not match applicability contract")
    receipt_rel = entry.get("review", {}).get("receipt")
    if not isinstance(receipt_rel, str):
        raise ValueError("Method evidence lacks a verified receipt")
    receipt = workspace.root / receipt_rel
    if not receipt.is_file() or _sha256(receipt) != contract.get("receipt_sha256"):
        raise ValueError("Evidence receipt does not match applicability contract")
    return {"contract_path": contract_path.relative_to(workspace.root).as_posix(), "contract_sha256": _sha256(contract_path), "receipt_path": receipt_rel, "receipt_sha256": _sha256(receipt), "entry": entry}


def load_approved_estimation_population(
    workspace: Workspace,
    rows: list[dict[str, str]],
    *,
    dataset_id: str,
    analysis_run_id: str,
    evidence_id: str,
    evidence_run_id: str,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Load the exact estimation population authorized by a supported result."""
    analysis_root = (
        workspace.artifacts_root
        / "particles2SNR-pipeline"
        / "analysis"
        / analysis_run_id
    ).resolve()
    expected_root = (
        workspace.artifacts_root / "particles2SNR-pipeline" / "analysis"
    ).resolve()
    if (
        not analysis_root.is_relative_to(expected_root)
        or analysis_root.name != analysis_run_id
        or not analysis_root.is_dir()
    ):
        raise ValueError("Approved population analysis run is missing or unsafe")

    run_path = analysis_root / "run.json"
    manifest_path = analysis_root / "metrics_manifest.json"
    summary_path = analysis_root / "summary_metrics.json"
    for path in (run_path, manifest_path, summary_path):
        if not path.is_file():
            raise ValueError(f"Approved population run is missing {path.name}")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        run.get("run_id") != analysis_run_id
        or run.get("status") != "complete"
        or run.get("dataset") != dataset_id
    ):
        raise ValueError("Approved population run metadata does not match")
    metric_rows = manifest.get("metrics")
    if not isinstance(metric_rows, list):
        raise ValueError("Approved population metrics manifest is invalid")
    summary_metric = next(
        (
            item
            for item in metric_rows
            if isinstance(item, dict) and item.get("path") == "summary_metrics.json"
        ),
        None,
    )
    if summary_metric is None or summary_metric.get("sha256") != _sha256(summary_path):
        raise ValueError("Approved population summary hash does not match its manifest")

    resolved = resolve_visual_evidence(workspace, evidence_id)
    if resolved.get("source") != "unified":
        raise ValueError("Approved population requires unified visual evidence")
    entry = resolved["entry"]
    if (
        entry.get("artifact", {}).get("run_id") != evidence_run_id
        or entry.get("outcome", {}).get("code") not in {"supported", "approved"}
    ):
        raise ValueError("Population result evidence is not supported")
    if dataset_id not in entry.get("datasets", []):
        raise ValueError("Population result evidence does not cover this dataset")
    analysis_reference = entry.get("analysis_reference", {})
    if analysis_reference.get("run_id") != analysis_run_id:
        raise ValueError("Population result evidence does not reference the analysis run")
    receipt_rel = entry.get("review", {}).get("receipt")
    if not isinstance(receipt_rel, str):
        raise ValueError("Population result evidence lacks a receipt")
    receipt_path = (workspace.root / receipt_rel).resolve()
    if (
        not receipt_path.is_relative_to(workspace.root)
        or not receipt_path.is_file()
    ):
        raise ValueError("Population result receipt is missing or unsafe")

    by_id = {row["event_id"]: row for row in rows}
    eligible_ids = summary.get("eligible_event_ids")
    if eligible_ids is None:
        censored_rows = summary.get("boundary_censored_events")
        if not isinstance(censored_rows, list) or not all(
            isinstance(item, dict) and isinstance(item.get("event_id"), str)
            for item in censored_rows
        ):
            raise ValueError("Approved population lacks event-level censoring records")
        censored_ids = {item["event_id"] for item in censored_rows}
        eligible_ids = [
            row["event_id"] for row in rows if row["event_id"] not in censored_ids
        ]
    if (
        not isinstance(eligible_ids, list)
        or not all(isinstance(value, str) for value in eligible_ids)
        or len(set(eligible_ids)) != len(eligible_ids)
    ):
        raise ValueError("Approved population has invalid eligible event IDs")
    if set(eligible_ids) - set(by_id):
        raise ValueError("Approved population references unknown event IDs")
    eligible_count = summary.get("eligible_event_count")
    censored_count = summary.get("boundary_censored_event_count")
    if (
        eligible_count != len(eligible_ids)
        or not isinstance(censored_count, int)
        or eligible_count + censored_count != len(rows)
    ):
        raise ValueError("Approved population counts do not reconcile")
    selected = [by_id[event_id] for event_id in eligible_ids]
    return selected, {
        "analysis_run_id": analysis_run_id,
        "analysis_run_path": analysis_root.relative_to(workspace.root).as_posix(),
        "analysis_run_sha256": _sha256(run_path),
        "metrics_manifest_sha256": _sha256(manifest_path),
        "summary_metrics_sha256": _sha256(summary_path),
        "computation_fingerprint": manifest.get("computation_fingerprint"),
        "eligible_event_count": eligible_count,
        "boundary_censored_event_count": censored_count,
        "evidence_id": evidence_id,
        "evidence_run_id": evidence_run_id,
        "receipt_path": receipt_rel,
        "receipt_sha256": _sha256(receipt_path),
    }


def read_events(path: Path, *, dataset_summary: dict[str, Any] | None = None) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    validate_events(rows, dataset_summary=dataset_summary)
    return rows


def validate_events(
    rows: list[dict[str, str]], *, dataset_summary: dict[str, Any] | None = None
) -> None:
    if not rows:
        raise ValueError("Registered z8 dataset contains no events")
    missing = REQUIRED_EVENT_COLUMNS - set(rows[0])
    if missing:
        raise ValueError(f"Event table missing required columns: {sorted(missing)}")
    if len({row["event_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate z8 event IDs")
    unsupported_splits = {row["split"] for row in rows} - {"train", "val"}
    if unsupported_splits:
        if "test" in unsupported_splits:
            raise ValueError("Sealed test split was included")
        raise ValueError(
            f"Only train/val development rows are permitted; got splits {sorted(unsupported_splits)}"
        )
    counts = observed_population_counts(rows)
    if dataset_summary is not None:
        if dataset_summary.get("event_count") != len(rows):
            raise ValueError("Event count does not match selected dataset summary")
        expected_counts = dataset_summary.get("class_counts")
        if expected_counts != counts["class_counts"]:
            raise ValueError("Class counts do not match selected dataset summary")
        for key in ("snr_population_counts", "split_counts"):
            if key in dataset_summary and dataset_summary[key] != counts[key]:
                raise ValueError(f"{key} do not match selected dataset summary")
        origins = dict(Counter(row["annotation_origin"] for row in rows))
        if dataset_summary["origin_counts"] != origins:
            raise ValueError("annotation origins do not match selected dataset summary")
    allowed_classes = set(CLASS_ORDER) | {"unclear"}
    if {row["class_name"] for row in rows} - allowed_classes:
        raise ValueError("Unexpected z8 class_name")
    if {row["physical_source_class"] for row in rows} - set(CLASS_ORDER):
        raise ValueError("Unexpected z8 physical_source_class")
    required = {definition["source_column"] for definition in PARAMETERS.values()}
    for row in rows:
        for column in required:
            try:
                value = float(row[column])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid {column} for {row.get('event_id')}") from exc
            if not math.isfinite(value):
                raise ValueError(f"Non-finite {column} for {row['event_id']}")


def _class_for_parameter(row: dict[str, str], parameter: str) -> str | None:
    if parameter == "snr_effective_fbase_db":
        class_name = row["physical_source_class"]
    else:
        class_name = row["class_name"]
    return class_name if class_name in CLASS_ORDER else None


def parameter_value(row: dict[str, str], parameter: str) -> float:
    definition = PARAMETERS[parameter]
    return float(row[definition["source_column"]]) * float(definition["scale"])


def grouped_values(
    rows: Iterable[dict[str, str]], parameter: str
) -> dict[str, np.ndarray]:
    grouped: dict[str, list[float]] = {name: [] for name in CLASS_ORDER}
    for row in rows:
        class_name = _class_for_parameter(row, parameter)
        if class_name is not None:
            grouped[class_name].append(parameter_value(row, parameter))
    return {
        class_name: np.asarray(values, dtype=np.float64)
        for class_name, values in grouped.items()
    }


def boundary_eligible_rows(
    rows: list[dict[str, str]],
    *,
    source_signal_lengths: dict[str, int],
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """Censor boundary-touching annotations from empirical estimation only."""
    eligible: list[dict[str, str]] = []
    censored: list[dict[str, Any]] = []
    for row in rows:
        relative_path = row["source_signal_relative_path"]
        try:
            signal_length = int(source_signal_lengths[relative_path])
            start = float(row["start_sample"])
            end = float(row["end_sample"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Missing boundary metadata for {row.get('event_id')}"
            ) from exc
        if signal_length <= 0 or not math.isfinite(start) or not math.isfinite(end):
            raise ValueError(f"Invalid boundary metadata for {row['event_id']}")
        if end < start:
            raise ValueError(f"Invalid annotation interval for {row['event_id']}")
        reasons: list[str] = []
        if start <= 0.0:
            reasons.append("start_sample_at_or_before_signal_start")
        if end >= signal_length:
            reasons.append("end_sample_at_or_after_signal_end")
        if reasons:
            censored.append(
                {
                    "event_id": row["event_id"],
                    "class_name": row["class_name"],
                    "physical_source_class": row["physical_source_class"],
                    "source_signal_relative_path": relative_path,
                    "signal_length_samples": signal_length,
                    "start_sample": start,
                    "end_sample": end,
                    "reasons": reasons,
                }
            )
        else:
            eligible.append(row)
    return eligible, sorted(censored, key=lambda item: item["event_id"])


def summarize_array(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.all(np.isfinite(array)):
        raise ValueError("A finite one-dimensional population of at least two values is required")
    quantiles = np.quantile(array, QUANTILES)
    return {
        "n": int(array.size),
        "minimum": float(np.min(array)),
        "q01": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "q25": float(quantiles[2]),
        "median": float(quantiles[3]),
        "q75": float(quantiles[4]),
        "q95": float(quantiles[5]),
        "q99": float(quantiles[6]),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
        "standard_deviation": float(np.std(array, ddof=1)),
    }


def support_candidates(
    *, class_name: str, parameter: str, summary: dict[str, float | int]
) -> list[dict[str, Any]]:
    observed_min = float(summary["minimum"])
    observed_max = float(summary["maximum"])
    width = observed_max - observed_min
    if width <= 0.0:
        raise ValueError(f"Degenerate support for {class_name}/{parameter}")
    result = []
    for margin_id, fraction in MARGIN_FRACTIONS.items():
        lower = observed_min - fraction * width
        upper = observed_max + fraction * width
        lower_inclusive = True
        constraints: list[str] = []
        if PARAMETERS[parameter]["positive"] and lower <= 0.0:
            lower = 0.0
            lower_inclusive = False
            constraints.append("strictly_positive_lower_bound")
        if parameter == "frequency_khz":
            nominal_low, nominal_high = FBASE_NOMINAL_BAND_KHZ
            effective_low = min(nominal_low, observed_min)
            if lower < effective_low:
                lower = effective_low
                constraints.append("effective_fbase_lower_bound")
            if upper > nominal_high:
                upper = nominal_high
                constraints.append("fbase_upper_bound")
        result.append(
            {
                "class_name": class_name,
                "parameter": parameter,
                "units": PARAMETERS[parameter]["units"],
                "margin_id": margin_id,
                "margin_fraction": fraction,
                "observed_minimum": observed_min,
                "observed_maximum": observed_max,
                "observed_width": width,
                "lower_bound": float(lower),
                "lower_inclusive": lower_inclusive,
                "upper_bound": float(upper),
                "upper_inclusive": True,
                "constraints": constraints,
            }
        )
    return result


def select_extremes(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for parameter in PARAMETERS:
        for class_name in CLASS_ORDER:
            candidates = [
                row
                for row in rows
                if _class_for_parameter(row, parameter) == class_name
            ]
            if not candidates:
                raise ValueError(f"Empty eligible population for {class_name}/{parameter}")
            for direction, selector in (("minimum", min), ("maximum", max)):
                target = selector(parameter_value(item, parameter) for item in candidates)
                tied = sorted(
                    (
                        item
                        for item in candidates
                        if parameter_value(item, parameter) == target
                    ),
                    key=lambda item: item["event_id"],
                )
                row = tied[0]
                record = selected.setdefault(
                    row["event_id"],
                    {
                        "event_id": row["event_id"],
                        "class_name": row["class_name"],
                        "physical_source_class": row["physical_source_class"],
                        "annotation_origin": row["annotation_origin"],
                        "source_filename": row["source_filename"],
                        "source_signal_relative_path": row["source_signal_relative_path"],
                        "split": row["split"],
                        "start_sample": float(row["start_sample"]),
                        "end_sample": float(row["end_sample"]),
                        "amplitude_p0": parameter_value(row, "amplitude_p0"),
                        "frequency_khz": parameter_value(row, "frequency_khz"),
                        "tau_ms": parameter_value(row, "tau_ms"),
                        "snr_effective_fbase_db": parameter_value(
                            row, "snr_effective_fbase_db"
                        ),
                        "extreme_roles": [],
                    },
                )
                record["extreme_roles"].append(
                    {
                        "parameter": parameter,
                        "class_name": class_name,
                        "direction": direction,
                        "value": parameter_value(row, parameter),
                        "tie_count": len(tied),
                        "tied_event_ids": [item["event_id"] for item in tied],
                        "selection_policy": "lexicographically_smallest_event_id",
                    }
                )
    return sorted(selected.values(), key=lambda row: row["event_id"])


def correlation_matrices(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    names = list(PARAMETERS)
    for class_name in CLASS_ORDER:
        physical = [row for row in rows if row["class_name"] == class_name]
        matrix = np.asarray(
            [[parameter_value(row, name) for name in names] for row in physical],
            dtype=np.float64,
        )
        correlation = np.corrcoef(matrix, rowvar=False)
        result[class_name] = {
            "n": len(physical),
            "parameters": names,
            "pearson": correlation.tolist(),
            "population": "physical_class_only",
        }
    return result


def build_analysis(
    rows: list[dict[str, str]],
    *,
    dataset_id: str | None = None,
    source_signal_lengths: dict[str, int] | None = None,
) -> dict[str, Any]:
    if source_signal_lengths is None:
        eligible_rows = list(rows)
        censored_rows: list[dict[str, Any]] = []
        boundary_policy = "not_applied"
    else:
        eligible_rows, censored_rows = boundary_eligible_rows(
            rows, source_signal_lengths=source_signal_lengths
        )
        boundary_policy = (
            "exclude_from_empirical_estimation_when_start_sample<=0_or_"
            "end_sample>=source_signal_length"
        )
    distributions: dict[str, dict[str, Any]] = {name: {} for name in CLASS_ORDER}
    statistics_rows: list[dict[str, Any]] = []
    supports: list[dict[str, Any]] = []
    for parameter, definition in PARAMETERS.items():
        grouped = grouped_values(eligible_rows, parameter)
        for class_name in CLASS_ORDER:
            summary = summarize_array(grouped[class_name])
            distributions[class_name][parameter] = {
                **summary,
                "units": definition["units"],
                "population": definition["population"],
                "distribution_policy": "empirical_observed",
            }
            statistics_rows.append(
                {
                    "class_name": class_name,
                    "parameter": parameter,
                    "units": definition["units"],
                    "population": definition["population"],
                    "distribution_policy": "empirical_observed",
                    **summary,
                }
            )
            supports.extend(
                support_candidates(
                    class_name=class_name,
                    parameter=parameter,
                    summary=summary,
                )
            )
    return {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "dataset_role": "post-processed particles2SNR development dataset",
        "event_count": len(rows),
        "eligible_event_count": len(eligible_rows),
        "boundary_censored_event_count": len(censored_rows),
        "boundary_censored_events": censored_rows,
        "boundary_censoring_policy": boundary_policy,
        "eligible_event_ids": sorted(row["event_id"] for row in eligible_rows),
        "class_counts": observed_population_counts(rows)["class_counts"],
        "eligible_class_counts": observed_population_counts(eligible_rows)["class_counts"],
        "snr_population_counts": observed_population_counts(rows)["snr_population_counts"],
        "eligible_snr_population_counts": observed_population_counts(eligible_rows)[
            "snr_population_counts"
        ],
        "sealed_test_accessed": False,
        "parameter_definitions": PARAMETERS,
        "distributions": distributions,
        "statistics_rows": statistics_rows,
        "correlations": correlation_matrices(eligible_rows),
        "support_candidates": supports,
        "extremes": select_extremes(eligible_rows),
        "claim_boundary": {
            "M0": "covers 100% of observed z8 values only",
            "M10_M20": "engineering hypotheses for later twin validation",
            "future_real_population": "not guaranteed by this descriptive analysis",
        },
    }


def _freedman_diaconis_bins(values: np.ndarray) -> int:
    q25, q75 = np.quantile(values, (0.25, 0.75))
    width = 2.0 * (q75 - q25) / np.cbrt(values.size)
    if width <= 0.0:
        return max(8, int(round(np.sqrt(values.size))))
    return int(np.clip(math.ceil((float(np.max(values)) - float(np.min(values))) / width), 8, 80))


def render_distribution_grid(rows: list[dict[str, str]], destination: Path) -> None:
    figure, axes = plt.subplots(4, 3, figsize=(15.5, 14.0), constrained_layout=True)
    for row_index, (parameter, definition) in enumerate(PARAMETERS.items()):
        grouped = grouped_values(rows, parameter)
        for column_index, class_name in enumerate(CLASS_ORDER):
            axis = axes[row_index, column_index]
            values = grouped[class_name]
            color = CLASS_COLORS[class_name]
            axis.hist(
                values,
                bins=_freedman_diaconis_bins(values),
                density=True,
                color=color,
                alpha=0.34,
                edgecolor="white",
                linewidth=0.4,
            )
            if np.unique(values).size > 1:
                density = gaussian_kde(values)
                x = np.linspace(float(np.min(values)), float(np.max(values)), 300)
                axis.plot(x, density(x), color=color, linewidth=2.0)
            axis.axvline(float(np.median(values)), color="#0f172a", linestyle="--", linewidth=1.2)
            axis.set_title(f"{CLASS_LABELS[class_name]} · n={len(values):,}", fontweight="bold")
            axis.set_xlabel(f"{definition['label']} ({definition['units']})")
            axis.set_ylabel("Density")
            axis.grid(alpha=0.16)
    figure.suptitle(
        "Empirical parameter distributions in the post-processed particles2SNR z8 dataset",
        fontsize=18,
        fontweight="bold",
    )
    figure.savefig(destination, dpi=180, facecolor="white")
    plt.close(figure)


def render_correlation_matrices(analysis: dict[str, Any], destination: Path) -> None:
    labels = ["P0", "frequency", "tau", "SNR"]
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), constrained_layout=True)
    image = None
    for axis, class_name in zip(axes, CLASS_ORDER, strict=True):
        matrix = np.asarray(analysis["correlations"][class_name]["pearson"], dtype=float)
        image = axis.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
        axis.set_xticks(range(4), labels, rotation=30, ha="right")
        axis.set_yticks(range(4), labels)
        axis.set_title(
            f"{CLASS_LABELS[class_name]} · physical events n={analysis['correlations'][class_name]['n']:,}",
            fontweight="bold",
        )
        for row_index in range(4):
            for column_index in range(4):
                value = matrix[row_index, column_index]
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if abs(value) > 0.55 else "#0f172a",
                    fontsize=9,
                    fontweight="bold",
                )
    assert image is not None
    figure.colorbar(image, ax=axes, shrink=0.82, label="Pearson correlation")
    figure.suptitle(
        "Parameter correlations · unclear events excluded from physical relationships",
        fontsize=17,
        fontweight="bold",
    )
    figure.savefig(destination, dpi=180, facecolor="white")
    plt.close(figure)


def render_extrema_gallery(
    extremes: list[dict[str, Any]],
    source_root: Path,
    destination: Path,
    *,
    boundary_censored_event_count: int,
) -> None:
    columns = 4
    rows_count = math.ceil(len(extremes) / columns)
    figure, axes = plt.subplots(
        rows_count,
        columns,
        figsize=(16.5, max(4.0, rows_count * 2.8)),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, record in zip(axes.flat, extremes):
        signal_path = source_root / record["source_signal_relative_path"]
        signal = np.load(signal_path, allow_pickle=False)
        if signal.ndim != 1 or not np.all(np.isfinite(signal)):
            raise ValueError(f"Invalid source signal for {record['event_id']}: {signal_path}")
        start = int(round(record["start_sample"]))
        end = int(round(record["end_sample"]))
        center = (start + end) // 2
        half_window = max(1024, 2 * max(1, end - start))
        left = max(0, center - half_window)
        right = min(signal.size, center + half_window)
        x = (np.arange(left, right) - center) / 2_000_000.0 * 1000.0
        axis.plot(x, signal[left:right], color="#0f172a", linewidth=0.75)
        axis.axvspan(
            (start - center) / 2_000_000.0 * 1000.0,
            (end - center) / 2_000_000.0 * 1000.0,
            color="#f59e0b",
            alpha=0.18,
        )
        roles = ", ".join(
            f"{role['parameter'].replace('_effective_fbase_db', '').replace('_khz', '').replace('_ms', '').replace('_p0', '')} {role['direction'][:3]}"
            for role in record["extreme_roles"]
        )
        display_class = CLASS_LABELS[record["physical_source_class"]]
        if record["class_name"] == "unclear":
            display_class += " / unclear"
        axis.set_title(
            f"{display_class} · {roles}\n{record['annotation_origin']}",
            fontsize=9,
            fontweight="bold",
        )
        axis.text(
            0.01,
            0.02,
            (
                f"P0={record['amplitude_p0']:.3g} · f={record['frequency_khz']:.2f} kHz\n"
                f"tau={record['tau_ms']:.3f} ms · SNR={record['snr_effective_fbase_db']:.2f} dB"
            ),
            transform=axis.transAxes,
            fontsize=7.5,
            va="bottom",
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 2},
        )
        axis.set_xlabel("Time from event center (ms)", fontsize=8)
        axis.set_ylabel("Acquisition units", fontsize=8)
        axis.tick_params(labelsize=7)
        axis.grid(alpha=0.12)
    for axis in list(axes.flat)[len(extremes) :]:
        axis.axis("off")
    figure.suptitle(
        (
            f"Deduplicated z8 parameter extremes · {len(extremes)} events · "
            f"{boundary_censored_event_count} boundary-touching events censored from estimation"
        ),
        fontsize=17,
        fontweight="bold",
    )
    figure.savefig(destination, dpi=180, facecolor="white")
    plt.close(figure)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serialized = {
                key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            }
            writer.writerow(serialized)


def write_analysis_outputs(
    *, analysis: dict[str, Any], rows: list[dict[str, str]], source_root: Path, output_dir: Path
) -> list[str]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite analysis run: {output_dir}")
    output_dir.mkdir(parents=True)
    eligible_ids = set(analysis["eligible_event_ids"])
    eligible_rows = [row for row in rows if row["event_id"] in eligible_ids]
    metrics = {
        key: value
        for key, value in analysis.items()
        if key not in {"statistics_rows", "eligible_event_ids"}
    }
    (output_dir / "summary_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    statistic_fields = [
        "class_name",
        "parameter",
        "units",
        "population",
        "distribution_policy",
        "n",
        "minimum",
        "q01",
        "q05",
        "q25",
        "median",
        "q75",
        "q95",
        "q99",
        "maximum",
        "mean",
        "standard_deviation",
    ]
    _write_csv(
        output_dir / "parameter_statistics.csv",
        analysis["statistics_rows"],
        statistic_fields,
    )
    support_fields = [
        "class_name",
        "parameter",
        "units",
        "margin_id",
        "margin_fraction",
        "observed_minimum",
        "observed_maximum",
        "observed_width",
        "lower_bound",
        "lower_inclusive",
        "upper_bound",
        "upper_inclusive",
        "constraints",
    ]
    _write_csv(
        output_dir / "support_candidates.csv",
        analysis["support_candidates"],
        support_fields,
    )
    extreme_fields = [
        "event_id",
        "class_name",
        "physical_source_class",
        "annotation_origin",
        "source_filename",
        "source_signal_relative_path",
        "split",
        "start_sample",
        "end_sample",
        "amplitude_p0",
        "frequency_khz",
        "tau_ms",
        "snr_effective_fbase_db",
        "extreme_roles",
    ]
    _write_csv(output_dir / "extremes.csv", analysis["extremes"], extreme_fields)
    censor_fields = [
        "event_id",
        "class_name",
        "physical_source_class",
        "source_signal_relative_path",
        "signal_length_samples",
        "start_sample",
        "end_sample",
        "reasons",
    ]
    _write_csv(
        output_dir / "boundary_censored_events.csv",
        analysis["boundary_censored_events"],
        censor_fields,
    )
    render_distribution_grid(eligible_rows, output_dir / "parameter_distributions.png")
    render_correlation_matrices(analysis, output_dir / "parameter_correlations.png")
    render_extrema_gallery(
        analysis["extremes"],
        source_root,
        output_dir / "parameter_extremes.png",
        boundary_censored_event_count=analysis["boundary_censored_event_count"],
    )
    return [
        "summary_metrics.json",
        "parameter_statistics.csv",
        "support_candidates.csv",
        "extremes.csv",
        "boundary_censored_events.csv",
        "parameter_distributions.png",
        "parameter_correlations.png",
        "parameter_extremes.png",
    ]
