from __future__ import annotations

import numpy as np
import pytest

from particles2snr.z8_density_gallery import (
    BASE_GENERATION_BATCH,
    EXTENSION_GENERATION_BATCH,
    build_nested_density10x_gallery,
    class_counts,
)

SMALL_BASE = {"2um": 2, "4um": 3, "10um": 1}
SMALL_EXTENSION = {"2um": 4, "4um": 6, "10um": 2}
SMALL_TARGET = {"2um": 6, "4um": 9, "10um": 3}


def _rows(counts: dict[str, int], prefix: str) -> list[dict[str, object]]:
    return [
        {
            "sample_id": f"{prefix}-{class_name}-{index}",
            "class_name": class_name,
            "value": index,
        }
        for class_name, count in counts.items()
        for index in range(count)
    ]


def _signals(count: int, length: int, offset: float) -> np.ndarray:
    row = np.arange(length, dtype=np.float32) + np.float32(offset)
    return np.broadcast_to(row, (count, length)).copy()


def test_nested_gallery_preserves_v2_prefix_and_exact_counts() -> None:
    baseline = _rows(SMALL_BASE, "base")
    extension = _rows(SMALL_EXTENSION, "new")
    baseline_raw = _signals(len(baseline), 4096, 1.0)
    baseline_model = _signals(len(baseline), 512, 2.0)
    extension_raw = _signals(len(extension), 4096, 3.0)
    extension_model = _signals(len(extension), 512, 4.0)

    rows, raw, model = build_nested_density10x_gallery(
        baseline,
        baseline_raw,
        baseline_model,
        extension,
        extension_raw,
        extension_model,
        expected_base_counts=SMALL_BASE,
        expected_extension_counts=SMALL_EXTENSION,
        expected_target_counts=SMALL_TARGET,
    )

    assert class_counts(rows) == SMALL_TARGET
    assert np.array_equal(raw[: len(baseline)], baseline_raw)
    assert np.array_equal(model[: len(baseline)], baseline_model)
    assert [row["sample_id"] for row in rows[: len(baseline)]] == [
        row["sample_id"] for row in baseline
    ]
    assert all(
        row["baseline_member"] is True
        and row["generation_batch"] == BASE_GENERATION_BATCH
        for row in rows[: len(baseline)]
    )
    assert all(
        row["baseline_member"] is False
        and row["generation_batch"] == EXTENSION_GENERATION_BATCH
        for row in rows[len(baseline) :]
    )


def test_nested_gallery_rejects_duplicate_ids() -> None:
    baseline = _rows(SMALL_BASE, "base")
    extension = _rows(SMALL_EXTENSION, "new")
    extension[0]["sample_id"] = baseline[0]["sample_id"]
    with pytest.raises(ValueError, match="globally unique"):
        build_nested_density10x_gallery(
            baseline,
            _signals(len(baseline), 4096, 1.0),
            _signals(len(baseline), 512, 2.0),
            extension,
            _signals(len(extension), 4096, 3.0),
            _signals(len(extension), 512, 4.0),
            expected_base_counts=SMALL_BASE,
            expected_extension_counts=SMALL_EXTENSION,
            expected_target_counts=SMALL_TARGET,
        )


def test_nested_gallery_rejects_wrong_class_budget() -> None:
    baseline = _rows(SMALL_BASE, "base")
    extension = _rows(SMALL_EXTENSION, "new")[:-1]
    with pytest.raises(ValueError, match="extension class counts"):
        build_nested_density10x_gallery(
            baseline,
            _signals(len(baseline), 4096, 1.0),
            _signals(len(baseline), 512, 2.0),
            extension,
            _signals(len(extension), 4096, 3.0),
            _signals(len(extension), 512, 4.0),
            expected_base_counts=SMALL_BASE,
            expected_extension_counts=SMALL_EXTENSION,
            expected_target_counts=SMALL_TARGET,
        )
