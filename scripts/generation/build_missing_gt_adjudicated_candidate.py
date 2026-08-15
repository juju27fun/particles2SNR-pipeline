#!/usr/bin/env python3
"""Freeze the missing-GT review and build a versioned source candidate."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from internship_workspace.config import Workspace
from internship_workspace.datasets import (
    register_record,
    resolve_path,
    select_record,
)
from particles2snr.missing_gt import (
    apply_source_restorations,
    build_adjudication_rows,
    load_locked_review,
    project_wave8_overlay,
    sha256_file,
    stable_hash,
    update_source_metadata,
    write_jsonl,
)


REVIEW_SESSION = Path(
    "artifacts/SMI_Detection_CNN_transformers/research/"
    "wave8like-gt-review-v2-jlb"
)
OVERLAY_ID = "particles2snr-dual-clean-wave8like-missing-gt-arbitration"
OVERLAY_KEY = f"{OVERLAY_ID}@v1"
OVERLAY_OUTPUT = Path(
    "datasets/interim/"
    "particles2snr-dual-clean-wave8like-missing-gt-arbitration/v1"
)
PARENT_KEY = (
    "particles2snr-f-dual-clean-c1-yolo-4class-adjudicated-candidate@v1"
)
OUTPUT_ID = (
    "particles2snr-f-dual-clean-c1-yolo-4class-missing-gt-candidate"
)
OUTPUT_KEY = f"{OUTPUT_ID}@v1"
OUTPUT_ROOT = Path(
    "datasets/interim/"
    "particles2snr-f-dual-clean-c1-yolo-4class-missing-gt-candidate/v1"
)
HISTORICAL_KEY = "particles2snr-f-c1-yolo-4class@v1"
WAVE8_KEY = "particles2snr-wave8like-known3-positive@v1"
RUN_ID = "missing_gt_adjudicated_candidate_20260718"
RUN_DIR = Path(f"artifacts/particles2SNR-pipeline/runs/{RUN_ID}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path, default=REVIEW_SESSION)
    parser.add_argument("--overlay-output", type=Path, default=OVERLAY_OUTPUT)
    parser.add_argument("--dataset-output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    return parser.parse_args()


def resolve(workspace: Workspace, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (workspace.root / path).resolve()


def relative(workspace: Workspace, path: Path) -> str:
    return path.resolve().relative_to(workspace.root.resolve()).as_posix()


def git_state(path: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    workspace = Workspace.load()
    session_dir = resolve(workspace, args.session_dir)
    overlay_output = resolve(workspace, args.overlay_output)
    dataset_output = resolve(workspace, args.dataset_output)
    run_dir = resolve(workspace, args.run_dir)
    overlay_output.relative_to(workspace.datasets_root / "interim")
    dataset_output.relative_to(workspace.datasets_root / "interim")
    run_dir.relative_to(
        workspace.artifacts_root / "particles2SNR-pipeline"
    )
    for path in (overlay_output, dataset_output, run_dir):
        if path.exists():
            raise FileExistsError(f"refusing to mutate existing output: {path}")

    parent_id, parent_version = PARENT_KEY.rsplit("@", 1)
    historical_id, historical_version = HISTORICAL_KEY.rsplit("@", 1)
    wave8_id, wave8_version = WAVE8_KEY.rsplit("@", 1)
    parent_record = select_record(workspace, parent_id, parent_version)
    historical_record = select_record(
        workspace, historical_id, historical_version
    )
    wave8_record = select_record(workspace, wave8_id, wave8_version)
    parent_root = resolve_path(workspace, parent_record)
    wave8_root = resolve_path(workspace, wave8_record)

    manifest, decisions = load_locked_review(session_dir)
    rows = build_adjudication_rows(
        manifest,
        decisions,
        historical_dataset_id=HISTORICAL_KEY,
        historical_manifest_sha256=historical_record.payload[
            "manifest_sha256"
        ],
    )
    overlay_sha256 = stable_hash(rows)
    projection = project_wave8_overlay(
        rows, wave8_manifest_path=wave8_root / "manifest.csv"
    )
    created_at = datetime.now(timezone.utc).isoformat()

    overlay_output.mkdir(parents=True)
    write_jsonl(overlay_output / "adjudicated_events.jsonl", rows)
    write_jsonl(
        overlay_output / "wave8_counterfactual_projection.jsonl",
        projection,
    )
    disputed = [
        row
        for row in rows
        if row["class_status"].startswith("disputed")
    ]
    write_jsonl(overlay_output / "disputed_events.jsonl", disputed)
    overlay_summary = {
        "schema_version": 1,
        "dataset_id": OVERLAY_KEY,
        "created_at": created_at,
        "status": "complete_frozen_human_adjudication",
        "review_session": relative(workspace, session_dir),
        "review_manifest_sha256": manifest["manifest_sha256"],
        "historical_dataset": HISTORICAL_KEY,
        "historical_manifest_sha256": historical_record.payload[
            "manifest_sha256"
        ],
        "source_candidate": PARENT_KEY,
        "source_candidate_manifest_sha256": parent_record.payload[
            "manifest_sha256"
        ],
        "wave8_parent": WAVE8_KEY,
        "wave8_parent_manifest_sha256": wave8_record.payload[
            "manifest_sha256"
        ],
        "overlay_sha256": overlay_sha256,
        "candidate_decisions": 18,
        "unique_events": 17,
        "source_positive_restorations": 9,
        "disputed_policy": "overlay_only_no_label_change",
        "wave8_projection_counts": {
            "add_positive": sum(
                row["action"] == "add_positive" for row in projection
            ),
            "ignore_modified_edge": sum(
                row["action"] == "ignore_modified_edge"
                for row in projection
            ),
        },
    }
    (overlay_output / "overlay_summary.json").write_text(
        json.dumps(overlay_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    changes = apply_source_restorations(
        parent_root=parent_root,
        output_root=dataset_output,
        rows=rows,
    )
    write_jsonl(
        dataset_output / "missing_gt_restorations.jsonl", changes
    )
    counts = update_source_metadata(
        dataset_output,
        output_dataset_id=OUTPUT_KEY,
        parent_dataset_id=PARENT_KEY,
        parent_manifest_sha256=parent_record.payload["manifest_sha256"],
        overlay_dataset_id=OVERLAY_KEY,
        overlay_sha256=overlay_sha256,
        changes=changes,
    )

    overlay_record = register_record(
        workspace,
        OVERLAY_ID,
        "v1",
        overlay_output.relative_to(workspace.datasets_root).as_posix(),
        "reference",
        "particles2SNR-pipeline",
        "ground-truth-adjudication-overlay",
        Path(__file__).resolve().relative_to(workspace.root).as_posix(),
    )
    output_record = register_record(
        workspace,
        OUTPUT_ID,
        "v1",
        dataset_output.relative_to(workspace.datasets_root).as_posix(),
        "reference",
        "particles2SNR-pipeline",
        "yolo-1d-missing-gt-candidate",
        Path(__file__).resolve().relative_to(workspace.root).as_posix(),
    )

    run_dir.mkdir(parents=True)
    run = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "project": "particles2SNR-pipeline",
        "kind": "dataset-generation",
        "created_at": created_at,
        "status": "complete_reference_candidates",
        "dataset": f"{OVERLAY_KEY} + {OUTPUT_KEY}",
        "command": Path(__file__).resolve().relative_to(workspace.root).as_posix(),
        "repositories": {
            "workspace": git_state(workspace.root),
            "particles2SNR-pipeline": git_state(
                workspace.root / "particles2SNR-pipeline"
            ),
        },
        "parents": {
            PARENT_KEY: parent_record.payload["manifest_sha256"],
            HISTORICAL_KEY: historical_record.payload["manifest_sha256"],
            WAVE8_KEY: wave8_record.payload["manifest_sha256"],
            "review_manifest": manifest["manifest_sha256"],
        },
        "outputs": [
            relative(workspace, overlay_output),
            relative(workspace, dataset_output),
        ],
        "summary": {
            "overlay_manifest_sha256": overlay_record.payload[
                "manifest_sha256"
            ],
            "output_manifest_sha256": output_record.payload[
                "manifest_sha256"
            ],
            "overlay_sha256": overlay_sha256,
            "unique_events": len(rows),
            "source_restorations": len(changes),
            "wave8_projection_rows": len(projection),
            "annotation_counts": counts,
        },
    }
    (run_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "REPORT.md").write_text(
        "\n".join(
            [
                "# Missing-GT adjudicated candidate",
                "",
                f"- Overlay: `{OVERLAY_KEY}`",
                f"- Source candidate: `{OUTPUT_KEY}`",
                f"- Unique human events: {len(rows)}",
                f"- Restored source positives: {len(changes)}",
                f"- Wave8 projected positives: {overlay_summary['wave8_projection_counts']['add_positive']}",
                f"- Wave8 edge ignores: {overlay_summary['wave8_projection_counts']['ignore_modified_edge']}",
                "",
                "No registered parent dataset was modified.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(run["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
