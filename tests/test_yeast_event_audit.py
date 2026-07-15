from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from particles2snr.yeast_event_audit import _all_fields, build_candidate_audit
from particles2snr.yeast_events import YeastDetectionConfig


def _synthetic_event(length: int = 16384) -> np.ndarray:
    rng = np.random.default_rng(7)
    index = np.arange(length, dtype=np.float32)
    time = index / 2_000_000.0
    envelope = np.exp(-0.5 * np.square((index - length // 2) / 420.0))
    return (
        envelope
        * (
            np.sin(2.0 * np.pi * 22_000.0 * time)
            + 0.75 * np.sin(2.0 * np.pi * 34_000.0 * time + 0.45)
        )
        + 0.015 * rng.normal(size=length)
    ).astype(np.float32)


def test_candidate_audit_writes_review_contract(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    (raw / "budding").mkdir(parents=True)
    np.save(raw / "budding" / "event.npy", _synthetic_event())
    index = tmp_path / "source_index.csv"
    fields = [
        "record_id",
        "relative_path",
        "source_group",
        "condition_id",
        "label_scope",
        "acquisition_id",
        "capture_block_id",
        "development_split",
        "is_canonical_duplicate_member",
    ]
    with index.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "record_id": "record-1",
                "relative_path": "budding/event.npy",
                "source_group": "budding",
                "condition_id": "exponential-budding",
                "label_scope": "acquisition-condition-proxy",
                "acquisition_id": "session-1",
                "capture_block_id": "block-1",
                "development_split": "development_train",
                "is_canonical_duplicate_member": "True",
            }
        )
    output = tmp_path / "output"
    config = YeastDetectionConfig(
        active_snr_z=2.5,
        strict_min_snr=3.0,
        medium_min_snr=2.0,
        strict_min_concentration=0.08,
        medium_min_concentration=0.04,
        min_width_ms=0.05,
        max_width_ms=2.0,
    )
    summary = build_candidate_audit(
        source_index_csv=index,
        raw_dataset_root=raw,
        output_dir=output,
        config=config,
        review_per_stratum=1,
    )
    assert summary["n_candidates"] == 1
    assert summary["manual_review_status"] == "pending"
    with np.load(output / "manual_review_signals.npz") as data:
        assert data["signals"].shape == (1, 8192)
    with np.load(output / "manual_file_review_signals.npz") as data:
        assert data["signals"].shape == (1, 16384)
    with (output / "manual_review_queue.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["review_event_present"] == ""
    with (output / "manual_file_review_queue.csv").open(newline="", encoding="utf-8") as handle:
        file_rows = list(csv.DictReader(handle))
    assert file_rows[0]["review_missed_event_count"] == ""
    assert file_rows[0]["review_false_retained_candidate_count"] == ""
    assert file_rows[0]["review_true_rejected_candidate_count"] == ""
    assert json.loads((output / "candidate_audit_summary.json").read_text())["n_files"] == 1


def test_review_schema_unions_candidate_and_background_fields() -> None:
    fields = _all_fields(
        [
            {"event_id": "event", "snr_proxy": 4.0},
            {"event_id": "background", "no_candidate_reason": "none"},
        ]
    )
    assert fields == ["event_id", "snr_proxy", "no_candidate_reason"]


def test_candidate_audit_resolves_each_registered_raw_dataset(tmp_path: Path) -> None:
    roots = {"raw-a@v1": tmp_path / "raw-a", "raw-b@v1": tmp_path / "raw-b"}
    for root in roots.values():
        (root / "budding").mkdir(parents=True)
        np.save(root / "budding" / "event.npy", _synthetic_event())
    index = tmp_path / "source_index.csv"
    rows = []
    for suffix, (dataset_id, _root) in enumerate(roots.items()):
        rows.append(
            {
                "record_id": f"record-{suffix}",
                "raw_dataset": dataset_id,
                "relative_path": "budding/event.npy",
                "source_group": "budding",
                "condition_id": "exponential-budding",
                "label_scope": "acquisition-condition-proxy",
                "acquisition_id": f"session-{suffix}",
                "capture_block_id": f"session-{suffix}:block-1",
                "development_split": (
                    "development_train" if suffix == 0 else "sealed_acquisition_test"
                ),
                "is_canonical_duplicate_member": "True",
            }
        )
    with index.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    output = tmp_path / "output"
    summary = build_candidate_audit(
        source_index_csv=index,
        raw_dataset_root=None,
        raw_dataset_roots=roots,
        output_dir=output,
        config=YeastDetectionConfig(
            active_snr_z=2.5,
            strict_min_snr=3.0,
            medium_min_snr=2.0,
            strict_min_concentration=0.08,
            medium_min_concentration=0.04,
            min_width_ms=0.05,
            max_width_ms=2.0,
        ),
        review_per_stratum=1,
    )
    assert summary["n_files"] == 2
    assert summary["raw_datasets"] == ["raw-a@v1", "raw-b@v1"]
    with (output / "candidate_events.csv").open(newline="", encoding="utf-8") as handle:
        candidates = list(csv.DictReader(handle))
    assert {row["raw_dataset"] for row in candidates} == set(roots)


def test_review_sampling_excludes_calibration_records(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    (raw / "budding").mkdir(parents=True)
    index = tmp_path / "source_index.csv"
    rows = []
    for suffix in range(2):
        np.save(raw / "budding" / f"event-{suffix}.npy", _synthetic_event())
        rows.append(
            {
                "record_id": f"record-{suffix}",
                "relative_path": f"budding/event-{suffix}.npy",
                "source_group": "budding",
                "condition_id": "exponential-budding",
                "label_scope": "acquisition-condition-proxy",
                "acquisition_id": "session-1",
                "capture_block_id": "block-1",
                "development_split": "development_train",
                "is_canonical_duplicate_member": "True",
            }
        )
    with index.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    output = tmp_path / "output"
    summary = build_candidate_audit(
        source_index_csv=index,
        raw_dataset_root=raw,
        output_dir=output,
        config=YeastDetectionConfig(
            active_snr_z=2.5,
            strict_min_snr=3.0,
            medium_min_snr=2.0,
            strict_min_concentration=0.08,
            medium_min_concentration=0.04,
            min_width_ms=0.05,
            max_width_ms=2.0,
        ),
        review_per_stratum=2,
        review_excluded_record_ids={"record-0"},
    )

    assert summary["review_sampling_excluded_record_count"] == 1
    assert summary["review_sampling_eligible_file_count"] == 1
    for filename in ("manual_review_queue.csv", "manual_file_review_queue.csv"):
        with (output / filename).open(newline="", encoding="utf-8") as handle:
            review_rows = list(csv.DictReader(handle))
        assert review_rows
        assert {row["record_id"] for row in review_rows} == {"record-1"}


def test_candidate_and_full_trace_sample_sizes_are_independent(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    (raw / "budding").mkdir(parents=True)
    rows = []
    for suffix in range(3):
        np.save(raw / "budding" / f"event-{suffix}.npy", _synthetic_event())
        rows.append(
            {
                "record_id": f"record-{suffix}",
                "relative_path": f"budding/event-{suffix}.npy",
                "source_group": "budding",
                "condition_id": "exponential-budding",
                "label_scope": "acquisition-condition-proxy",
                "acquisition_id": "session-1",
                "capture_block_id": "block-1",
                "development_split": "development_train",
                "is_canonical_duplicate_member": "True",
            }
        )
    index = tmp_path / "source_index.csv"
    with index.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    output = tmp_path / "output"
    summary = build_candidate_audit(
        source_index_csv=index,
        raw_dataset_root=raw,
        output_dir=output,
        config=YeastDetectionConfig(
            active_snr_z=2.5,
            strict_min_snr=3.0,
            medium_min_snr=2.0,
            strict_min_concentration=0.08,
            medium_min_concentration=0.04,
            min_width_ms=0.05,
            max_width_ms=2.0,
        ),
        review_per_stratum=3,
        file_review_per_stratum=1,
    )

    assert summary["n_manual_review_rows"] == 3
    assert summary["n_manual_file_review_rows"] == 1
    assert summary["candidate_review_per_stratum"] == 3
    assert summary["file_review_per_stratum"] == 1
