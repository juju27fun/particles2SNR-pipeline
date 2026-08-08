from __future__ import annotations

import csv
from pathlib import Path

from particles2snr.saturation_gt_dataset import (
    apply_reviewed_labels,
    promote_reviewed_dataset,
    yolo_line,
)


def test_yolo_line_expands_interval_without_changing_class() -> None:
    assert yolo_line(
        class_id=2,
        start_ms=2.0,
        end_ms=4.0,
        duration_ms=8.0,
    ) == "2 0.3750000000 0.2500000000"


def test_apply_reviewed_labels_keeps_deletes_and_expands(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    (source / "train/labels").mkdir(parents=True)
    (source / "train/labels/a.txt").write_text(
        "2 0.25 0.10\n2 0.75 0.10\n",
        encoding="utf-8",
    )
    queue = {
        "candidates": [
            {
                "candidate_id": "a:0",
                "label_path": "train/labels/a.txt",
                "annotation_id": 0,
                "class_id": 2,
                "class_name": "10um",
                "source_duration_ms": 8.0,
            },
            {
                "candidate_id": "a:1",
                "label_path": "train/labels/a.txt",
                "annotation_id": 1,
                "class_id": 2,
                "class_name": "10um",
                "source_duration_ms": 8.0,
            },
        ]
    }
    decisions = {
        "a:0": {
            "decision": "needs_review",
            "revision": 2,
            "decision_source": "human_correction",
        },
        "a:1": {
            "decision": "delete",
            "revision": 1,
            "decision_source": "human",
        },
    }
    proposals = {
        "a:0": {
            "class_name": "10um",
            "proposed_start_ms": "1.0",
            "proposed_end_ms": "3.0",
        }
    }
    rows = apply_reviewed_labels(
        source_root=source,
        output_root=output,
        queue=queue,
        decisions=decisions,
        proposals=proposals,
    )

    assert (output / "train/labels/a.txt").read_text() == (
        "2 0.2500000000 0.2500000000\n"
    )
    assert [row["action"] for row in rows] == [
        "expand_detector_consensus",
        "delete",
    ]


def test_promote_reviewed_dataset_records_visual_gate(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    (source / "dataset.yaml").write_text(
        "provenance:\n"
        "  candidate_status: reviewed_reference_not_promoted\n"
        "review:\n"
        "  actions:\n"
        "    delete: 172\n"
        "    expand_detector_consensus: 4\n"
        "    keep: 17\n",
        encoding="utf-8",
    )
    plots = [f"artifact/plot-{index}.png" for index in range(4)]

    result = promote_reviewed_dataset(
        source_root=source,
        output_root=output,
        source_dataset="reviewed@v1",
        source_manifest_sha256="abc",
        reviewer="jlb",
        validated_at="2026-07-18T00:00:00+00:00",
        evidence_plots=plots,
        boundary_truncated_candidates=["trace:1627:0"],
    )

    assert result["status"] == "approved_for_active_use"
    assert result["boundary_truncated_candidates"] == ["trace:1627:0"]
    metadata = (output / "dataset.yaml").read_text(encoding="utf-8")
    assert "candidate_status: active_visual_validation_approved" in metadata
    assert "visual_validation: visual_validation.json" in metadata
