from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from particles2snr.yeast_ablation_review_app import (
    YeastAblationReviewWorkspace,
)


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    raw = tmp_path / "raw"
    review = tmp_path / "review"
    (raw / "mix").mkdir(parents=True)
    review.mkdir()
    signal = np.zeros(16384, dtype=np.float32)
    time = np.arange(1400) / 2_000_000.0
    signal[7000:8400] = (
        np.sin(2 * np.pi * 20_000 * time) * np.hanning(1400)
    ).astype(np.float32)
    np.save(raw / "mix/example.npy", signal)
    current = [
        {
            "candidate_index": 0,
            "center_index": 7700,
            "event_start": 7000,
            "event_end": 8400,
            "score": 20.0,
            "status": "current_only",
            "match_key": "current:0",
        }
    ]
    simple = [
        {
            "candidate_index": 0,
            "center_index": 10000,
            "event_start": 9400,
            "event_end": 10600,
            "score": 15.0,
            "status": "simple_only",
            "match_key": "simple:0",
        }
    ]
    row = {
        "record_id": "record-1",
        "relative_path": "mix/example.npy",
        "source_group": "mix",
        "development_split": "development_validation",
        "n_current_only": "1",
        "n_simple_only": "1",
        "current_candidates_json": json.dumps(current),
        "simple_candidates_json": json.dumps(simple),
        "review_preference": "",
        "reviewer": "",
        "review_notes": "",
    }
    with (review / "review_queue.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return raw, review


def test_ablation_review_workspace_saves_preference_and_renders_plot(
    tmp_path: Path,
) -> None:
    raw, review = _fixture(tmp_path)
    workspace = YeastAblationReviewWorkspace(raw, review)
    assert workspace.list_items()["complete"] == 0
    assert workspace.plot_png(0).startswith(b"\x89PNG")

    saved = workspace.save_item(
        0,
        {
            "reviewer": "Julien",
            "review_preference": "simple",
            "review_notes": "clearer boundary",
        },
    )

    assert saved["complete"] is True
    assert saved["row"]["review_preference"] == "simple"
    assert (review / "annotation_audit.jsonl").is_file()
