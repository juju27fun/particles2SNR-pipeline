from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from particles2snr.z8_cholesky_generation import MODEL_LENGTH, RAW_LENGTH
from particles2snr.z8_parameter_analysis import CLASS_ORDER


BASE_COUNTS = {"2um": 1_151, "4um": 3_281, "10um": 366}
TARGET_COUNTS = {"2um": 11_510, "4um": 32_810, "10um": 3_660}
EXTENSION_COUNTS = {
    class_name: TARGET_COUNTS[class_name] - BASE_COUNTS[class_name]
    for class_name in CLASS_ORDER
}
BASE_GENERATION_BATCH = "v2_baseline"
EXTENSION_GENERATION_BATCH = "density10x_extension"


def class_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row["class_name"]) for row in rows)
    unknown = set(counts).difference(CLASS_ORDER)
    if unknown:
        raise ValueError(f"unknown particle classes: {sorted(unknown)}")
    return {class_name: counts[class_name] for class_name in CLASS_ORDER}


def _validate_arrays(
    rows: Sequence[Mapping[str, Any]],
    raw: np.ndarray,
    model: np.ndarray,
) -> None:
    if raw.shape != (len(rows), RAW_LENGTH) or raw.dtype != np.float32:
        raise ValueError("raw gallery must be aligned float32 (n, 4096)")
    if model.shape != (len(rows), MODEL_LENGTH) or model.dtype != np.float32:
        raise ValueError("model gallery must be aligned float32 (n, 512)")
    if not np.isfinite(raw).all() or not np.isfinite(model).all():
        raise ValueError("gallery signals contain non-finite values")


def build_nested_density10x_gallery(
    baseline_rows: Sequence[Mapping[str, Any]],
    baseline_raw: np.ndarray,
    baseline_model: np.ndarray,
    extension_rows: Sequence[Mapping[str, Any]],
    extension_raw: np.ndarray,
    extension_model: np.ndarray,
    *,
    expected_base_counts: Mapping[str, int] = BASE_COUNTS,
    expected_extension_counts: Mapping[str, int] = EXTENSION_COUNTS,
    expected_target_counts: Mapping[str, int] = TARGET_COUNTS,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    """Build a 10× gallery while preserving the v2 prefix exactly."""
    _validate_arrays(baseline_rows, baseline_raw, baseline_model)
    _validate_arrays(extension_rows, extension_raw, extension_model)
    if class_counts(baseline_rows) != dict(expected_base_counts):
        raise ValueError("baseline class counts differ from the immutable v2 contract")
    if class_counts(extension_rows) != dict(expected_extension_counts):
        raise ValueError("extension class counts differ from the 9× contract")

    baseline_ids = [str(row["sample_id"]) for row in baseline_rows]
    extension_ids = [str(row["sample_id"]) for row in extension_rows]
    all_ids = baseline_ids + extension_ids
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("baseline and extension sample IDs are not globally unique")

    rows = [
        {
            **dict(row),
            "baseline_member": True,
            "generation_batch": BASE_GENERATION_BATCH,
        }
        for row in baseline_rows
    ]
    rows.extend(
        {
            **dict(row),
            "baseline_member": False,
            "generation_batch": EXTENSION_GENERATION_BATCH,
        }
        for row in extension_rows
    )
    raw = np.concatenate(
        (
            np.asarray(baseline_raw, dtype=np.float32),
            np.asarray(extension_raw, dtype=np.float32),
        ),
        axis=0,
    )
    model = np.concatenate(
        (
            np.asarray(baseline_model, dtype=np.float32),
            np.asarray(extension_model, dtype=np.float32),
        ),
        axis=0,
    )
    _validate_arrays(rows, raw, model)
    if class_counts(rows) != dict(expected_target_counts):
        raise ValueError("combined class counts differ from the approved 10× contract")

    baseline_count = len(baseline_rows)
    if not np.array_equal(raw[:baseline_count], baseline_raw):
        raise AssertionError("raw v2 baseline changed during gallery assembly")
    if not np.array_equal(model[:baseline_count], baseline_model):
        raise AssertionError("model-input v2 baseline changed during gallery assembly")
    for source, combined in zip(baseline_rows, rows[:baseline_count], strict=True):
        for field, value in source.items():
            if combined[field] != value:
                raise AssertionError(f"v2 metadata changed for field {field}")
    return rows, raw, model
