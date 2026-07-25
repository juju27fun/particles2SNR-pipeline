from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from particles2snr.e000_calibration import DatasetBinding, audit_preflight


def _fixture_binding(
    root: Path,
    *,
    dataset_id: str,
    split: str,
    value: float,
    acquisition_id: str | None,
) -> DatasetBinding:
    slug = dataset_id.replace("@", "-")
    data_dir = root / "datasets" / "raw" / slug
    data_dir.mkdir(parents=True, exist_ok=True)
    values = np.full(16384, value, dtype=np.float64)
    source = data_dir / "event.npy"
    np.save(source, values)
    row: dict[str, object] = {
        "path": source.name,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "size": source.stat().st_size,
    }
    if acquisition_id is not None:
        row["acquisition_id"] = acquisition_id
    manifest = root / "datasets" / "registry" / f"{slug}.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(row, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    return DatasetBinding(
        dataset_id=dataset_id,
        manifest_relative_path=str(manifest.relative_to(root)),
        data_relative_path=str(data_dir.relative_to(root)),
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        manifest_file_count=1,
        split=split,
    )


def test_preflight_passes_complete_identity_and_grouping_fixture(
    tmp_path: Path,
) -> None:
    development = _fixture_binding(
        tmp_path,
        dataset_id="development@v1",
        split="calibration_development",
        value=1.0,
        acquisition_id="dev-acquisition",
    )
    confirmation = _fixture_binding(
        tmp_path,
        dataset_id="confirmation@v1",
        split="calibration_confirmation",
        value=2.0,
        acquisition_id="confirmation-acquisition",
    )

    result = audit_preflight(
        tmp_path, bindings=(development, confirmation), verify_content=True
    )

    assert result["decision"] == "preflight_passed"
    assert result["blockers"] == []
    assert result["confirmation_opened"] is False


def test_preflight_stops_when_acquisition_groups_are_unresolved(
    tmp_path: Path,
) -> None:
    binding = _fixture_binding(
        tmp_path,
        dataset_id="development@v1",
        split="calibration_development",
        value=1.0,
        acquisition_id=None,
    )

    result = audit_preflight(tmp_path, bindings=(binding,), verify_content=True)

    assert result["decision"] == "insufficient"
    assert [item["code"] for item in result["blockers"]] == [
        "unresolved_acquisition_confounding"
    ]


def test_preflight_stops_on_cross_split_duplicate_content(
    tmp_path: Path,
) -> None:
    development = _fixture_binding(
        tmp_path,
        dataset_id="development@v1",
        split="calibration_development",
        value=1.0,
        acquisition_id="dev-acquisition",
    )
    confirmation = _fixture_binding(
        tmp_path,
        dataset_id="confirmation@v1",
        split="calibration_confirmation",
        value=1.0,
        acquisition_id="confirmation-acquisition",
    )

    result = audit_preflight(
        tmp_path, bindings=(development, confirmation), verify_content=True
    )

    assert result["decision"] == "insufficient"
    assert [item["code"] for item in result["blockers"]] == [
        "cross_split_duplicate_content"
    ]
