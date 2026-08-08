#!/usr/bin/env python3
"""Build the strict known3 Wave8 candidate from the missing-GT source."""

from __future__ import annotations

import hashlib
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
from particles2snr.wave8like_dataset import GenerationConfig, generate_dataset


SOURCE_KEY = (
    "particles2snr-f-dual-clean-c1-yolo-4class-missing-gt-candidate@v1"
)
NOISE_KEY = "noise@v1"
OUTPUT_ID = "particles2snr-wave8like-known3-positive-adjudicated-candidate"
OUTPUT_KEY = f"{OUTPUT_ID}@v1"
OUTPUT_PATH = Path(
    "datasets/interim/"
    "particles2snr-wave8like-known3-positive-adjudicated-candidate/v1"
)
RUN_ID = "wave8like_known3_strict_candidate_20260718"
RUN_PATH = Path(f"artifacts/particles2SNR-pipeline/runs/{RUN_ID}")


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


def main() -> None:
    workspace = Workspace.load()
    output_root = (workspace.root / OUTPUT_PATH).resolve()
    run_root = (workspace.root / RUN_PATH).resolve()
    for path in (output_root, run_root):
        if path.exists():
            raise FileExistsError(f"refusing to mutate existing output: {path}")
    source_id, source_version = SOURCE_KEY.rsplit("@", 1)
    noise_id, noise_version = NOISE_KEY.rsplit("@", 1)
    source_record = select_record(workspace, source_id, source_version)
    noise_record = select_record(workspace, noise_id, noise_version)
    source_root = resolve_path(workspace, source_record)
    noise_root = resolve_path(workspace, noise_record)
    module_path = (
        workspace.root
        / "particles2SNR-pipeline/particles2snr/wave8like_dataset.py"
    )
    revision = hashlib.sha256(module_path.read_bytes()).hexdigest()
    config = GenerationConfig(
        mode="known3-positive",
        source_dataset_id=SOURCE_KEY,
        noise_dataset_id=NOISE_KEY,
        output_dataset_id=OUTPUT_KEY,
        seed=42,
        segment_length=16_384,
        segments_per_sequence=4,
        noise_pad=300,
        join_crossfade=300,
        sampling_frequency_hz=2_000_000,
        bandpass_low_hz=8_000.0,
        bandpass_high_hz=500_000.0,
        bandpass_order=4,
        train_groups=100,
        val_groups=30,
        test_groups=30,
        positive_permutations=24,
        background_share=0.0,
        background_permutations=4,
        disjoint_background_groups=True,
        source_eligibility_policy="fully_labeled_for_view",
        generator_revision=f"sha256:{revision}",
    )
    metadata = generate_dataset(
        source_root=source_root,
        noise_root=noise_root,
        output_root=output_root,
        config=config,
    )
    dropped = {
        split: int(summary["dropped_edge_events_at_base_group_level"])
        for split, summary in metadata["splits"].items()
    }
    if any(dropped.values()):
        raise RuntimeError(f"strict candidate dropped events: {dropped}")
    record = register_record(
        workspace,
        OUTPUT_ID,
        "v1",
        output_root.relative_to(workspace.datasets_root).as_posix(),
        "reference",
        "particles2SNR-pipeline",
        "yolo-1d-long-sequence-strict-candidate",
        Path(__file__).relative_to(workspace.root).as_posix(),
    )
    created_at = datetime.now(timezone.utc).isoformat()
    run_root.mkdir(parents=True)
    run = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "project": "particles2SNR-pipeline",
        "kind": "dataset-generation",
        "created_at": created_at,
        "status": "complete_reference_candidate_pending_model_evaluation",
        "dataset": OUTPUT_KEY,
        "command": Path(__file__).relative_to(workspace.root).as_posix(),
        "repositories": {
            "workspace": git_state(workspace.root),
            "particles2SNR-pipeline": git_state(
                workspace.root / "particles2SNR-pipeline"
            ),
        },
        "parents": {
            SOURCE_KEY: source_record.payload["manifest_sha256"],
            NOISE_KEY: noise_record.payload["manifest_sha256"],
        },
        "outputs": [
            output_root.relative_to(workspace.root).as_posix(),
            "REPORT.md",
        ],
        "summary": {
            "manifest_sha256": record.payload["manifest_sha256"],
            "source_eligibility_policy": (
                config.source_eligibility_policy
            ),
            "dropped_edge_events": dropped,
            "splits": metadata["splits"],
        },
    }
    (run_root / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_root / "REPORT.md").write_text(
        "\n".join(
            [
                "# Strict Wave8 known3 candidate",
                "",
                f"- Dataset: `{OUTPUT_KEY}`",
                f"- Source: `{SOURCE_KEY}`",
                "- Eligibility: `fully_labeled_for_view`",
                "- Dropped edge events: 0 in every split",
                "- Status: reference candidate pending fresh model inference",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "dataset": OUTPUT_KEY,
                "manifest_sha256": record.payload["manifest_sha256"],
                "dropped_edge_events": dropped,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
