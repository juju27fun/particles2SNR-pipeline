from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from particles2snr.yeast_review_figures import render_review_figures


def _write_csv(path: Path, row: dict[str, object]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def test_render_review_figures_writes_two_pdfs(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _write_csv(
        dataset / "manual_review_queue.csv",
        {
            "review_stratum": "budding:strict",
            "event_id": "event-1",
            "snr_proxy": 10.0,
            "width_ms": 0.5,
            "candidate_index": 0,
            "center_index": 8192,
            "event_start": 7800,
            "event_end": 8600,
        },
    )
    _write_csv(
        dataset / "manual_file_review_queue.csv",
        {
            "review_stratum": "budding:n_candidates_1",
            "relative_path": "budding/a.npy",
            "n_candidates": 1,
            "detected_event_ids": "[\"event-1\"]",
            "detected_centers": "[8192]",
        },
    )
    _write_csv(dataset / "candidate_events.csv", {"event_id": "event-1", "quality": "strict"})
    np.savez_compressed(dataset / "manual_review_signals.npz", signals=np.ones((1, 8192)))
    np.savez_compressed(dataset / "manual_file_review_signals.npz", signals=np.ones((1, 16384)))
    output = tmp_path / "output"
    summary = render_review_figures(dataset, output)
    assert summary["n_candidate_review_rows"] == 1
    assert (output / "candidate_precision_review.pdf").stat().st_size > 0
    assert (output / "full_trace_recall_review.pdf").stat().st_size > 0
    assert (output / "manual_review_queue.csv").read_text() == (
        dataset / "manual_review_queue.csv"
    ).read_text()
    assert (output / "manual_file_review_queue.csv").read_text() == (
        dataset / "manual_file_review_queue.csv"
    ).read_text()
