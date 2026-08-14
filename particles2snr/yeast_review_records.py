"""Load manifested yeast review records by event id.

The review queues are frozen audit artifacts; the raw signals are resolved
through the dataset registry, never by hard-coded workspace paths.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from internship_workspace.config import Workspace
from internship_workspace.datasets import resolve_path, select_record

RAW_DATASET_ID = "yeast-hf-10-5-20260610"
RAW_DATASET_VERSION = "v1"
REVIEW_QUEUE_RELATIVE_PATHS = (
    "particles2SNR-pipeline/audits/yeast-event-review-v8-additional-human-audit/manual_review_queue.csv",
    "particles2SNR-pipeline/audits/yeast-event-review-v7-analysis-20260715-v2/manual_review_queue.csv",
)


@dataclass(frozen=True)
class ReviewedEventRecord:
    event_id: str
    row: dict[str, str]
    signal: np.ndarray


def review_queue_paths(workspace: Workspace) -> tuple[Path, ...]:
    return tuple(
        workspace.artifacts_root / relative for relative in REVIEW_QUEUE_RELATIVE_PATHS
    )


def load_reviewed_event(workspace: Workspace, event_id: str) -> ReviewedEventRecord:
    record = select_record(workspace, RAW_DATASET_ID, RAW_DATASET_VERSION)
    raw_root = resolve_path(workspace, record)
    for path in review_queue_paths(workspace):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("event_id") != event_id:
                    continue
                signal = np.load(raw_root / row["relative_path"], allow_pickle=False)
                return ReviewedEventRecord(
                    event_id=event_id,
                    row=dict(row),
                    signal=np.squeeze(np.asarray(signal, dtype=np.float32)),
                )
    raise ValueError(f"Unknown manifested event: {event_id}")
