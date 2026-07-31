from __future__ import annotations

import json
from pathlib import Path

from particles2snr.missing_gt import (
    apply_source_restorations,
    build_adjudication_rows,
    project_wave8_overlay,
)


WORKSPACE = Path(__file__).resolve().parents[2]
SESSION = (
    WORKSPACE
    / "artifacts/SMI_Detection_CNN_transformers/research/"
    "wave8like-gt-review-v2-jlb"
)


def _review() -> tuple[dict, dict]:
    manifest = json.loads(
        (SESSION / "review_manifest.json").read_text(encoding="utf-8")
    )
    decisions = {
        row["candidate_id"]: row
        for row in (
            json.loads(line)
            for line in (SESSION / "decisions.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    }
    return manifest, decisions


def test_review_deduplicates_to_locked_event_population() -> None:
    manifest, decisions = _review()
    rows = build_adjudication_rows(
        manifest,
        decisions,
        historical_dataset_id="historical@v1",
        historical_manifest_sha256="a" * 64,
    )
    assert len(rows) == 17
    assert sum(row["source_action"] == "add_source_positive" for row in rows) == 9
    event_701 = [
        row for row in rows if row["source_id"].endswith("_701")
    ]
    assert len(event_701) == 1
    assert event_701[0]["candidate_ids"] == [
        "40cc6352ccfd",
        "55e05d1d5b6b",
    ]
    disputed = [
        row for row in rows if row["class_status"].startswith("disputed")
    ]
    assert {row["source_id"] for row in disputed} == {
        "HFocusing_5_10_4um_0_2015",
        "HFocusing_5_10_4um_0_356",
    }
    assert all(row["source_action"] == "preserve_existing_label" for row in disputed)


def test_source_application_adds_only_nine_labels(tmp_path: Path) -> None:
    manifest, decisions = _review()
    rows = build_adjudication_rows(
        manifest,
        decisions,
        historical_dataset_id="historical@v1",
        historical_manifest_sha256="a" * 64,
    )
    parent = (
        WORKSPACE
        / "datasets/interim/"
        "particles2snr-f-dual-clean-c1-yolo-4class-adjudicated-candidate/v1"
    )
    output = tmp_path / "candidate"
    changes = apply_source_restorations(
        parent_root=parent, output_root=output, rows=rows
    )
    assert len(changes) == 9
    assert {
        row["class_name"] for row in changes
    } == {"2um", "4um", "10um"}
    assert (
        output
        / "test/labels/HFocusing_5_10_4um_0_356.txt"
    ).read_bytes() == (
        parent / "test/labels/HFocusing_5_10_4um_0_356.txt"
    ).read_bytes()


def test_wave8_projection_has_locked_counts() -> None:
    manifest, decisions = _review()
    rows = build_adjudication_rows(
        manifest,
        decisions,
        historical_dataset_id="historical@v1",
        historical_manifest_sha256="a" * 64,
    )
    projection = project_wave8_overlay(
        rows,
        wave8_manifest_path=(
            WORKSPACE
            / "datasets/processed/"
            "particles2snr-wave8like-known3-positive/v1/manifest.csv"
        ),
    )
    positives = [row for row in projection if row["action"] == "add_positive"]
    ignores = [
        row for row in projection if row["action"] == "ignore_modified_edge"
    ]
    assert len(positives) == 360
    assert len(ignores) == 216
    assert "HFocusing_5_10_10um_0_1126" not in {
        row["source_id"] for row in projection
    }
    assert {
        row["source_id"] for row in ignores
    } == {
        "HFocusing_5_10_10um_0_325",
        "HFocusing_5_10_2um2_0_701",
        "HFocusing_5_10_4um_0_1381",
    }
