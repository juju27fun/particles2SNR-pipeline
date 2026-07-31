from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import numpy as np

from particles2snr.z8_reference_dataset import (
    EXCLUSION_FIELDS,
    EVENT_FIELDS,
    build_z8_reference_event_table,
    validate_z8_reference_event_table,
    validate_fresh_parent_contract,
)


def _contract_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {field: "" for field in EVENT_FIELDS}
    row.update(
        {
            "event_id": "event-1",
            "split": "train",
            "class_name": "10um",
            "start_norm": 0.1,
            "end_norm": 0.2,
            "center_norm": 0.15,
            "proposal_center_norm": 0.15,
            "center_sample": 2457.6,
            "annotation_origin": "z8_rescue",
            "overlaps_saturation_repair": False,
            "center_inside_saturation_repair": False,
        }
    )
    row.update(overrides)
    return row


def _write_empty_exclusions(root: Path) -> None:
    with (root / "exclusions.csv").open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=EXCLUSION_FIELDS).writeheader()


def test_validation_rejects_event_center_inside_saturation_repair(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    rows = [_contract_row(center_inside_saturation_repair=True)]
    with (root / "events.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (root / "dataset_summary.json").write_text(
        json.dumps({"event_count": 1, "class_counts": {"10um": 1}}),
        encoding="utf-8",
    )
    _write_empty_exclusions(root)

    with pytest.raises(ValueError, match="centre is inside a saturation repair"):
        validate_z8_reference_event_table(root)


def test_validation_rejects_test_rows(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    rows = [_contract_row(split="test", class_name="2um")]
    with (root / "events.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (root / "dataset_summary.json").write_text(
        json.dumps({"event_count": 1, "class_counts": {"2um": 1}}),
        encoding="utf-8",
    )
    _write_empty_exclusions(root)

    with pytest.raises(ValueError, match="Sealed test split included"):
        validate_z8_reference_event_table(root)


def _fresh_detector_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "datasets/source"
    signal_path = source / "train/signals/sample.npy"
    signal_path.parent.mkdir(parents=True)
    np.save(signal_path, np.zeros(16_384, dtype=np.float32))
    run = tmp_path / "artifacts/run"
    (run / "train").mkdir(parents=True)
    annotation = {
        "id": 0,
        "detector_annotation_id": 10,
        "class_id": 2,
        "start": 0.1,
        "end": 0.2,
        "center": 0.15,
        "amplitude": 1.0,
        "frequency": 12_000.0,
        "passage_time_ms": 0.2,
        "snr_db": 2.0,
        "peak_z": 9.0,
        "clean_local_peak_z": 3.0,
    }
    rescued = {
        **annotation,
        "id": 11,
        "detector_annotation_id": 11,
        "start": 0.4,
        "end": 0.5,
        "center": 0.45,
    }
    payload = {
        "data": [
            {
                "filename": "sample.npy",
                "class_id": 2,
                "class_name": "10um",
                "length": 16_384,
                "annotations": [annotation],
                "dropped_annotations": [
                    {
                        "reason": "missing_clean_peak_support",
                        "detector_annotation_id": 11,
                        "local_peak_z": 2.0,
                        "snr_db": 2.0,
                        "frequency": 12_000.0,
                        "candidate_annotation": rescued,
                    }
                ],
            }
        ]
    }
    (run / "train/data.json").write_text(json.dumps(payload), encoding="utf-8")
    manifest = tmp_path / "repairs.csv"
    manifest.write_text(
        "split,filename,expanded_start_sample,expanded_end_sample\n",
        encoding="utf-8",
    )
    return source, run, manifest


def test_fresh_v2_hard_veto_is_inclusive_for_strict_rescue_edges_and_multiple_repairs(
    tmp_path: Path,
) -> None:
    source, run, manifest = _fresh_detector_fixture(tmp_path)
    payload = json.loads((run / "train/data.json").read_text(encoding="utf-8"))
    strict_at_left_edge = {
        **payload["data"][0]["annotations"][0],
        "id": 12,
        "detector_annotation_id": 12,
        "start": 0.0,
        "end": 100.0 / 16_384.0,
        "center": 0.0,  # Deliberately inconsistent: bounds are authoritative.
    }
    strict_outside_repairs = {
        **payload["data"][0]["annotations"][0],
        "id": 13,
        "detector_annotation_id": 13,
        "start": 0.48,
        "end": 0.50,
        "center": 0.99,
    }
    strict_at_right_endpoint = {
        **payload["data"][0]["annotations"][0],
        "start": 0.125,
        "end": 0.25,
        "center": 0.01,
    }
    rescue_at_left_endpoint = {
        **payload["data"][0]["dropped_annotations"][0]["candidate_annotation"],
        "start": 0.3125,
        "end": 0.4375,
        "center": 0.99,
    }
    payload["data"][0]["annotations"] = [
        strict_at_left_edge,
        strict_at_right_endpoint,
        strict_outside_repairs,
    ]
    payload["data"][0]["dropped_annotations"][0]["candidate_annotation"] = (
        rescue_at_left_endpoint
    )
    (run / "train/data.json").write_text(json.dumps(payload), encoding="utf-8")
    manifest.write_text(
        "split,filename,expanded_start_sample,expanded_end_sample\n"
        "train,sample.npy,0,50\n"
        "train,sample.npy,3072,3072\n"
        "train,sample.npy,6144,6144\n",
        encoding="utf-8",
    )

    output = tmp_path / "z8"
    summary = build_z8_reference_event_table(
        source_root=source,
        historical_run=None,
        strict_run=run,
        saturation_manifest=manifest,
        output_dir=output,
        source_dataset_id="f@v2",
        source_manifest_sha256="source-hash",
        strict_dataset_id="f@v2",
        strict_manifest_sha256="strict-hash",
        output_dataset_id="z8@v2",
        strict_run_splits=("train",),
        fresh_detector_mode=True,
        saturation_center_veto=True,
        expected_development_signal_count=1,
    )
    assert summary["event_count"] == 1
    assert summary["exclusion_counts"] == {
        "z8_center_inside_saturation_repair": 3
    }
    with (output / "events.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["detector_annotation_id"] == "13"
    assert rows[0]["center_inside_saturation_repair"] == "False"
    assert float(rows[0]["center_norm"]) == pytest.approx(0.99)
    assert float(rows[0]["proposal_center_norm"]) == pytest.approx(0.49)
    with (output / "exclusions.csv").open(newline="", encoding="utf-8") as handle:
        exclusions = list(csv.DictReader(handle))
    assert {row["reason"] for row in exclusions} == {
        "z8_center_inside_saturation_repair"
    }
    assert {row["annotation_origin"] for row in exclusions} == {
        "dual_clean_strict",
        "z8_rescue",
    }
    assert {
        (row["center_sample"], row["expanded_start_sample"], row["expanded_end_sample"])
        for row in exclusions
    } == {
        ("50.0", "0", "50"),
        ("3072.0", "3072", "3072"),
        ("6144.0", "6144", "6144"),
    }
    assert validate_z8_reference_event_table(
        output, saturation_manifest=manifest
    )["valid"] is True


def test_validation_rejects_tampered_hard_veto_log(tmp_path: Path) -> None:
    source, run, manifest = _fresh_detector_fixture(tmp_path)
    payload = json.loads((run / "train/data.json").read_text(encoding="utf-8"))
    payload["data"][0]["annotations"][0].update({
        "start": 0.125, "end": 0.25, "center": 0.01,
    })
    (run / "train/data.json").write_text(json.dumps(payload), encoding="utf-8")
    manifest.write_text(
        "split,filename,expanded_start_sample,expanded_end_sample\n"
        "train,sample.npy,3072,3072\n",
        encoding="utf-8",
    )
    output = tmp_path / "z8"
    build_z8_reference_event_table(
        source_root=source,
        historical_run=None,
        strict_run=run,
        saturation_manifest=manifest,
        output_dir=output,
        source_dataset_id="f@v2",
        source_manifest_sha256="source-hash",
        strict_dataset_id="f@v2",
        strict_manifest_sha256="strict-hash",
        output_dataset_id="z8@v2",
        strict_run_splits=("train",),
        fresh_detector_mode=True,
        saturation_center_veto=True,
        expected_development_signal_count=1,
    )
    rows = list(csv.DictReader((output / "exclusions.csv").open(encoding="utf-8")))
    rows[0]["center_sample"] = "3073"
    with (output / "exclusions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="Hard-veto centre differs"):
        validate_z8_reference_event_table(output, saturation_manifest=manifest)


def test_fresh_v2_veto_event_ids_are_deterministic_and_test_signal_is_ignored(
    tmp_path: Path,
) -> None:
    source, run, manifest = _fresh_detector_fixture(tmp_path)
    test_signal = source / "test/signals/sealed.npy"
    test_signal.parent.mkdir(parents=True)
    np.save(test_signal, np.zeros(16_384, dtype=np.float32))
    first = tmp_path / "z8-a"
    second = tmp_path / "z8-b"
    common = dict(
        source_root=source,
        historical_run=None,
        strict_run=run,
        saturation_manifest=manifest,
        source_dataset_id="f@v2",
        source_manifest_sha256="source-hash",
        strict_dataset_id="f@v2",
        strict_manifest_sha256="strict-hash",
        output_dataset_id="z8@v2",
        strict_run_splits=("train",),
        fresh_detector_mode=True,
        saturation_center_veto=True,
        expected_development_signal_count=1,
    )
    build_z8_reference_event_table(output_dir=first, **common)
    build_z8_reference_event_table(output_dir=second, **common)
    with (first / "events.csv").open(newline="", encoding="utf-8") as handle:
        first_rows = list(csv.DictReader(handle))
    with (second / "events.csv").open(newline="", encoding="utf-8") as handle:
        second_rows = list(csv.DictReader(handle))
    assert first_rows == second_rows
    assert {row["split"] for row in first_rows} == {"train"}


def test_fresh_v2_build_uses_new_namespace_and_exact_detector_ids(
    tmp_path: Path,
) -> None:
    source, run, manifest = _fresh_detector_fixture(tmp_path)
    output_v2 = tmp_path / "z8-v2"
    summary = build_z8_reference_event_table(
        source_root=source,
        historical_run=None,
        strict_run=run,
        saturation_manifest=manifest,
        output_dir=output_v2,
        source_dataset_id="f@v2",
        source_manifest_sha256="source-hash",
        strict_dataset_id="f@v2",
        strict_manifest_sha256="strict-hash",
        output_dataset_id="z8@v2",
        strict_run_splits=("train",),
        fresh_detector_mode=True,
        expected_development_signal_count=1,
    )
    assert summary["dataset_id"] == "z8@v2"
    assert summary["event_count"] == 2
    assert summary["policy"]["fresh_detector_mode"] is True
    with (output_v2 / "events.csv").open(newline="", encoding="utf-8") as handle:
        rows_v2 = list(csv.DictReader(handle))
    assert {int(row["detector_annotation_id"]) for row in rows_v2} == {10, 11}
    assert {row["annotation_origin"] for row in rows_v2} == {
        "dual_clean_strict",
        "z8_rescue",
    }

    output_v3 = tmp_path / "z8-v3"
    build_z8_reference_event_table(
        source_root=source,
        historical_run=None,
        strict_run=run,
        saturation_manifest=manifest,
        output_dir=output_v3,
        source_dataset_id="f@v2",
        source_manifest_sha256="source-hash",
        strict_dataset_id="f@v2",
        strict_manifest_sha256="strict-hash",
        output_dataset_id="z8@v3",
        strict_run_splits=("train",),
        fresh_detector_mode=True,
        expected_development_signal_count=1,
    )
    with (output_v3 / "events.csv").open(newline="", encoding="utf-8") as handle:
        rows_v3 = list(csv.DictReader(handle))
    assert {row["event_id"] for row in rows_v2}.isdisjoint(
        {row["event_id"] for row in rows_v3}
    )


def test_fresh_v2_build_rejects_duplicate_detector_ids(tmp_path: Path) -> None:
    source, run, manifest = _fresh_detector_fixture(tmp_path)
    payload = json.loads((run / "train/data.json").read_text(encoding="utf-8"))
    payload["data"][0]["dropped_annotations"][0]["detector_annotation_id"] = 10
    (run / "train/data.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate fresh detector annotation IDs"):
        build_z8_reference_event_table(
            source_root=source,
            historical_run=None,
            strict_run=run,
            saturation_manifest=manifest,
            output_dir=tmp_path / "out",
            source_dataset_id="f@v2",
            source_manifest_sha256="source-hash",
            strict_dataset_id="f@v2",
            strict_manifest_sha256="strict-hash",
            output_dataset_id="z8@v2",
            fresh_detector_mode=True,
            expected_development_signal_count=1,
        )


def test_fresh_v2_build_accepts_explicit_unclear_annotation(
    tmp_path: Path,
) -> None:
    source, run, manifest = _fresh_detector_fixture(tmp_path)
    payload = json.loads((run / "train/data.json").read_text(encoding="utf-8"))
    payload["data"][0]["annotations"][0]["class_id"] = 3
    payload["data"][0]["annotations"][0]["snr_db"] = -12.0
    payload["data"][0]["dropped_annotations"] = []
    (run / "train/data.json").write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "z8"
    build_z8_reference_event_table(
        source_root=source,
        historical_run=None,
        strict_run=run,
        saturation_manifest=manifest,
        output_dir=output,
        source_dataset_id="f@v2",
        source_manifest_sha256="source-hash",
        strict_dataset_id="f@v2",
        strict_manifest_sha256="strict-hash",
        output_dataset_id="z8@v2",
        fresh_detector_mode=True,
        expected_development_signal_count=1,
    )
    with (output / "events.csv").open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["class_name"] == "unclear"
    assert row["physical_source_class"] == "10um"


def test_fresh_parent_contract_rejects_v1_signal_parent(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "dataset"
    source_root.mkdir()
    manifest = source_root / "saturation_repair_manifest.csv"
    manifest.write_text("filename\n", encoding="utf-8")
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.json").write_text(
        json.dumps({"dataset": "corrected@v2"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="corrected detector dataset"):
        validate_fresh_parent_contract(
            source_dataset_id="signals@v1",
            strict_dataset_id="corrected@v2",
            source_root=source_root,
            strict_run=run,
            saturation_manifest=manifest,
        )
