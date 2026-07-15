from __future__ import annotations

import csv
import json
import threading
from pathlib import Path
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen

import numpy as np
import pytest

from particles2snr.yeast_review_app import (
    YeastReviewWorkspace,
    build_handler,
    serve_review,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _review_fixture(tmp_path: Path) -> tuple[Path, Path]:
    dataset = tmp_path / "dataset"
    review = tmp_path / "review"
    dataset.mkdir()
    review.mkdir()
    candidate_rows = [
        {
            "event_id": "record-1:00",
            "record_id": "record-1",
            "relative_path": "budding/event.npy",
            "source_group": "budding",
            "condition_id": "condition",
            "acquisition_id": "session-1",
            "acquisition_role": "development",
            "candidate_index": 0,
            "center_index": 4096,
            "event_start": 3600,
            "event_end": 4600,
            "width_ms": 0.5,
            "snr_proxy": 4.0,
            "quality": "strict",
        },
        {
            "event_id": "record-1:01",
            "record_id": "record-1",
            "relative_path": "budding/event.npy",
            "source_group": "budding",
            "condition_id": "condition",
            "acquisition_id": "session-1",
            "acquisition_role": "development",
            "candidate_index": 1,
            "center_index": 9000,
            "event_start": 8600,
            "event_end": 9400,
            "width_ms": 0.4,
            "snr_proxy": 2.0,
            "quality": "reject",
        },
    ]
    file_row = {
        "record_id": "record-1",
        "relative_path": "budding/event.npy",
        "source_group": "budding",
        "condition_id": "condition",
        "acquisition_id": "session-1",
        "acquisition_role": "development",
        "n_candidates": 2,
        "n_retained_candidates": 1,
        "n_rejected_candidates": 1,
    }
    candidate_review = {
        **candidate_rows[0],
        "review_stratum": "session-1:budding:strict",
        "review_event_present": "",
        "review_center_acceptable": "",
        "review_full_event_visible": "",
        "review_artifact": "",
        "reviewer": "",
        "review_notes": "",
    }
    file_review = {
        **file_row,
        "review_stratum": "session-1:budding:n_candidates_2",
        "detected_event_ids": json.dumps(["record-1:00", "record-1:01"]),
        "detected_centers": json.dumps([4096, 9000]),
        "review_true_event_count": "",
        "review_false_retained_candidate_count": "",
        "review_true_rejected_candidate_count": "",
        "review_missed_event_count": "",
        "reviewer": "",
        "review_notes": "",
    }
    _write_csv(dataset / "candidate_events.csv", candidate_rows)
    _write_csv(dataset / "file_detection_report.csv", [file_row])
    _write_csv(review / "manual_review_queue.csv", [candidate_review])
    _write_csv(review / "manual_file_review_queue.csv", [file_review])
    time = np.arange(16384, dtype=np.float32) / 2_000_000.0
    signal = np.sin(2.0 * np.pi * 20_000.0 * time).astype(np.float32)
    np.savez_compressed(
        dataset / "manual_review_signals.npz",
        event_id=np.asarray(["record-1:00"]),
        signals=signal[0:8192][None, :],
    )
    np.savez_compressed(
        dataset / "manual_file_review_signals.npz",
        record_id=np.asarray(["record-1"]),
        signals=signal[None, :],
    )
    return dataset, review


def test_review_workspace_refuses_registered_dataset_edits(tmp_path: Path) -> None:
    dataset, _review = _review_fixture(tmp_path)
    with pytest.raises(ValueError, match="separate working copy"):
        YeastReviewWorkspace(dataset, dataset)


def test_review_workspace_saves_candidate_and_audit_record(tmp_path: Path) -> None:
    dataset, review = _review_fixture(tmp_path)
    workspace = YeastReviewWorkspace(dataset, review)
    result = workspace.save_item(
        "candidate",
        0,
        {
            "review_event_present": "yes",
            "review_center_acceptable": "yes",
            "review_full_event_visible": "yes",
            "review_artifact": "no",
            "reviewer": "reviewer-a",
            "review_notes": "clear passage",
        },
    )
    assert result["complete"] is True
    with (review / "manual_review_queue.csv").open(newline="", encoding="utf-8") as handle:
        persisted = list(csv.DictReader(handle))[0]
    assert persisted["review_event_present"] == "yes"
    assert persisted["reviewer"] == "reviewer-a"
    audit = json.loads((review / "annotation_audit.jsonl").read_text(encoding="utf-8"))
    assert audit["row_id"] == "record-1:00"
    assert len(audit["queue_sha256"]) == 64


def test_review_workspace_validates_full_trace_counts(tmp_path: Path) -> None:
    dataset, review = _review_fixture(tmp_path)
    workspace = YeastReviewWorkspace(dataset, review)
    values = {
        "review_true_event_count": 2,
        "review_false_retained_candidate_count": 0,
        "review_true_rejected_candidate_count": 1,
        "review_missed_event_count": 1,
        "reviewer": "reviewer-a",
        "review_notes": "",
    }
    with pytest.raises(ValueError, match="must equal"):
        workspace.save_item("file", 0, values)
    values["review_true_event_count"] = 3
    assert workspace.save_item("file", 0, values)["complete"] is True


def test_pending_full_trace_defaults_to_algorithm_counts(tmp_path: Path) -> None:
    dataset, review = _review_fixture(tmp_path)
    workspace = YeastReviewWorkspace(dataset, review)
    row = workspace.get_item("file", 0)["row"]
    assert row["review_true_event_count"] == "1"
    assert row["review_false_retained_candidate_count"] == "0"
    assert row["review_true_rejected_candidate_count"] == "0"
    assert row["review_missed_event_count"] == "0"

    workspace.save_item(
        "file",
        0,
        {
            "review_true_event_count": 3,
            "review_false_retained_candidate_count": 0,
            "review_true_rejected_candidate_count": 1,
            "review_missed_event_count": 1,
            "reviewer": "reviewer-a",
            "review_notes": "corrected",
        },
    )
    saved = workspace.get_item("file", 0)["row"]
    assert saved["review_true_event_count"] == "3"
    assert saved["review_true_rejected_candidate_count"] == "1"


def test_nonretained_candidate_does_not_require_center_or_visibility(tmp_path: Path) -> None:
    dataset, review = _review_fixture(tmp_path)
    queue_path = review / "manual_review_queue.csv"
    with queue_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["quality"] = "reject"
    _write_csv(queue_path, rows)
    workspace = YeastReviewWorkspace(dataset, review)
    result = workspace.save_item(
        "candidate",
        0,
        {
            "review_event_present": "no",
            "review_center_acceptable": "",
            "review_full_event_visible": "",
            "review_artifact": "no",
            "reviewer": "reviewer-a",
            "review_notes": "",
        },
    )
    assert result["complete"] is True


def test_review_workspace_renders_candidate_and_full_trace_png(tmp_path: Path) -> None:
    dataset, review = _review_fixture(tmp_path)
    workspace = YeastReviewWorkspace(dataset, review)
    filtered = workspace.plot_png("candidate", 0)
    raw = workspace.plot_png("candidate", 0, "raw")
    assert filtered.startswith(b"\x89PNG\r\n\x1a\n")
    assert raw.startswith(b"\x89PNG\r\n\x1a\n")
    assert filtered != raw
    assert workspace.plot_png("file", 0).startswith(b"\x89PNG\r\n\x1a\n")
    with pytest.raises(ValueError, match="signal_view"):
        workspace.plot_png("candidate", 0, "unknown")


def test_review_workspace_renders_full_trace_without_candidates(tmp_path: Path) -> None:
    dataset, review = _review_fixture(tmp_path)
    queue_path = review / "manual_file_review_queue.csv"
    with queue_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["n_candidates"] = "0"
    rows[0]["n_retained_candidates"] = "0"
    rows[0]["n_rejected_candidates"] = "0"
    rows[0]["detected_event_ids"] = "[]"
    rows[0]["detected_centers"] = "[]"
    _write_csv(queue_path, rows)
    workspace = YeastReviewWorkspace(dataset, review)
    assert workspace.plot_png("file", 0).startswith(b"\x89PNG\r\n\x1a\n")


def test_review_server_refuses_non_loopback_bind(tmp_path: Path) -> None:
    dataset, review = _review_fixture(tmp_path)
    with pytest.raises(ValueError, match="loopback"):
        serve_review(dataset, review, host="0.0.0.0", port=0)


def test_review_http_handler_serves_state_and_plot(tmp_path: Path) -> None:
    dataset, review = _review_fixture(tmp_path)
    workspace = YeastReviewWorkspace(dataset, review)
    server = ThreadingHTTPServer(("127.0.0.1", 0), build_handler(workspace))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urlopen(f"{base}/api/items?queue=candidate") as response:
            state = json.load(response)
        with urlopen(f"{base}/plot/candidate/0.png") as response:
            image = response.read()
        with urlopen(f"{base}/plot/candidate/0.png?signal_view=raw") as response:
            raw_image = response.read()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
    assert state["total"] == 1
    assert state["complete"] == 0
    assert image.startswith(b"\x89PNG\r\n\x1a\n")
    assert raw_image.startswith(b"\x89PNG\r\n\x1a\n")
    assert image != raw_image


def test_review_http_handler_saves_reviewer(tmp_path: Path) -> None:
    dataset, review = _review_fixture(tmp_path)
    workspace = YeastReviewWorkspace(dataset, review)
    server = ThreadingHTTPServer(("127.0.0.1", 0), build_handler(workspace))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    payload = {
        "queue": "candidate",
        "index": 0,
        "values": {
            "review_event_present": "yes",
            "review_center_acceptable": "yes",
            "review_full_event_visible": "yes",
            "review_artifact": "no",
            "reviewer": "reviewer-http",
            "review_notes": "reviewed",
        },
    }
    request = Request(
        f"{base}/api/item",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request) as response:
            saved = json.load(response)
        with urlopen(f"{base}/api/item?queue=candidate&index=0") as response:
            reloaded = json.load(response)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
    assert saved["complete"] is True
    assert reloaded["row"]["reviewer"] == "reviewer-http"
