from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scipy.signal import hilbert, spectrogram

from .yeast_events import YeastDetectionConfig, bandpass_yeast_signal
from .yeast_review_analysis import analyze_review


@dataclass(frozen=True)
class QueueSpec:
    csv_name: str
    signal_name: str
    id_field: str
    signal_id_field: str
    required_fields: tuple[str, ...]


QUEUE_SPECS = {
    "candidate": QueueSpec(
        csv_name="manual_review_queue.csv",
        signal_name="manual_review_signals.npz",
        id_field="event_id",
        signal_id_field="event_id",
        required_fields=(
            "review_event_present",
            "review_center_acceptable",
            "review_full_event_visible",
            "review_artifact",
        ),
    ),
    "file": QueueSpec(
        csv_name="manual_file_review_queue.csv",
        signal_name="manual_file_review_signals.npz",
        id_field="record_id",
        signal_id_field="record_id",
        required_fields=(
            "review_true_event_count",
            "review_false_retained_candidate_count",
            "review_true_rejected_candidate_count",
            "review_missed_event_count",
        ),
    ),
}

def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if not fields or not rows:
        raise ValueError(f"Review queue is empty or malformed: {path}")
    return fields, rows


def _read_candidate_lookup(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["event_id"]: row for row in csv.DictReader(handle)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class YeastReviewWorkspace:
    def __init__(self, candidate_dataset: Path, review_dir: Path) -> None:
        self.candidate_dataset = candidate_dataset.resolve()
        self.review_dir = review_dir.resolve()
        if self.review_dir == self.candidate_dataset or self.review_dir.is_relative_to(
            self.candidate_dataset
        ):
            raise ValueError("Review edits must target a separate working copy, not a dataset")
        self._lock = threading.Lock()
        self._fields: dict[str, list[str]] = {}
        self._rows: dict[str, list[dict[str, str]]] = {}
        self._signals: dict[str, np.ndarray] = {}
        self._signal_indices: dict[str, dict[str, int]] = {}
        self._candidate_by_id = _read_candidate_lookup(
            self.candidate_dataset / "candidate_events.csv"
        )
        for queue, spec in QUEUE_SPECS.items():
            fields, rows = _read_csv(self.review_dir / spec.csv_name)
            with np.load(self.candidate_dataset / spec.signal_name) as payload:
                signal_ids = [str(value) for value in payload[spec.signal_id_field].tolist()]
                signals = np.asarray(payload["signals"], dtype=np.float32)
            if len(signal_ids) != signals.shape[0]:
                raise ValueError(f"Signal IDs and arrays differ for {queue}")
            index_by_id = {value: index for index, value in enumerate(signal_ids)}
            missing = [row[spec.id_field] for row in rows if row[spec.id_field] not in index_by_id]
            if missing:
                raise ValueError(f"Review rows have no signal payload for {queue}: {missing[:3]}")
            self._fields[queue] = fields
            self._rows[queue] = rows
            self._signals[queue] = signals
            self._signal_indices[queue] = index_by_id

    def _spec(self, queue: str) -> QueueSpec:
        try:
            return QUEUE_SPECS[queue]
        except KeyError as exc:
            raise ValueError(f"Unknown review queue: {queue}") from exc

    def _row(self, queue: str, index: int) -> dict[str, str]:
        self._spec(queue)
        rows = self._rows[queue]
        if index < 0 or index >= len(rows):
            raise IndexError(f"Review index out of range: {index}")
        return rows[index]

    def is_complete(self, queue: str, row: dict[str, str]) -> bool:
        required_fields = self._required_fields(queue, row)
        return bool(row.get("reviewer", "").strip()) and all(
            row.get(field, "").strip() for field in required_fields
        )

    def _required_fields(self, queue: str, row: dict[str, str]) -> tuple[str, ...]:
        spec = self._spec(queue)
        if queue == "candidate" and row.get("quality") not in {"strict", "medium"}:
            return ("review_event_present", "review_artifact")
        return spec.required_fields

    def list_items(self, queue: str) -> dict[str, Any]:
        spec = self._spec(queue)
        items = []
        for index, row in enumerate(self._rows[queue]):
            items.append(
                {
                    "index": index,
                    "id": row[spec.id_field],
                    "review_stratum": row.get("review_stratum", ""),
                    "source_group": row.get("source_group", ""),
                    "quality": row.get("quality", ""),
                    "relative_path": row.get("relative_path", ""),
                    "complete": self.is_complete(queue, row),
                }
            )
        return {
            "queue": queue,
            "total": len(items),
            "complete": sum(bool(item["complete"]) for item in items),
            "items": items,
        }

    def get_item(self, queue: str, index: int) -> dict[str, Any]:
        spec = self._spec(queue)
        row = self._row(queue, index)
        display_row = dict(row)
        if queue == "file":
            event_ids = json.loads(row["detected_event_ids"])
            display_row["detected_qualities"] = ", ".join(
                self._candidate_by_id[event_id]["quality"] for event_id in event_ids
            ) or "none"
            review_fields = QUEUE_SPECS[queue].required_fields
            if not any(row.get(field, "").strip() for field in review_fields):
                display_row.update(
                    {
                        "review_true_event_count": row["n_retained_candidates"],
                        "review_false_retained_candidate_count": "0",
                        "review_true_rejected_candidate_count": "0",
                        "review_missed_event_count": "0",
                    }
                )
        return {
            "queue": queue,
            "index": index,
            "total": len(self._rows[queue]),
            "id": row[spec.id_field],
            "complete": self.is_complete(queue, row),
            "row": display_row,
            "plot_url": f"/plot/{queue}/{index}.png",
        }

    @staticmethod
    def _yes_no(value: Any, field: str) -> str:
        normalized = str(value).strip().lower()
        if normalized not in {"yes", "no"}:
            raise ValueError(f"{field} must be yes or no")
        return normalized

    @staticmethod
    def _count(value: Any, field: str) -> int:
        try:
            parsed = int(str(value).strip())
        except ValueError as exc:
            raise ValueError(f"{field} must be a non-negative integer") from exc
        if parsed < 0:
            raise ValueError(f"{field} must be a non-negative integer")
        return parsed

    def _validated_update(self, queue: str, row: dict[str, str], values: dict[str, Any]) -> dict[str, str]:
        reviewer = str(values.get("reviewer", "")).strip()
        if not reviewer:
            raise ValueError("reviewer is required")
        update = {"reviewer": reviewer, "review_notes": str(values.get("review_notes", "")).strip()}
        if queue == "candidate":
            required = set(self._required_fields(queue, row))
            for field in QUEUE_SPECS[queue].required_fields:
                value = str(values.get(field, "")).strip()
                update[field] = self._yes_no(value, field) if field in required or value else ""
            return update

        counts = {
            field: self._count(values.get(field, ""), field)
            for field in QUEUE_SPECS[queue].required_fields
        }
        n_retained = int(row["n_retained_candidates"])
        n_rejected = int(row["n_rejected_candidates"])
        false_retained = counts["review_false_retained_candidate_count"]
        true_rejected = counts["review_true_rejected_candidate_count"]
        missed = counts["review_missed_event_count"]
        true_events = counts["review_true_event_count"]
        if false_retained > n_retained:
            raise ValueError("False retained count exceeds retained candidates")
        if true_rejected > n_rejected:
            raise ValueError("True rejected count exceeds rejected candidates")
        true_positive = n_retained - false_retained
        if true_events != true_positive + true_rejected + missed:
            raise ValueError(
                "True event count must equal retained true, rejected true, plus missed events"
            )
        update.update({field: str(value) for field, value in counts.items()})
        return update

    def _write_queue(self, queue: str) -> None:
        spec = self._spec(queue)
        destination = self.review_dir / spec.csv_name
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=self.review_dir,
            prefix=f".{spec.csv_name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=self._fields[queue])
            writer.writeheader()
            writer.writerows(self._rows[queue])
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)

    def save_item(self, queue: str, index: int, values: dict[str, Any]) -> dict[str, Any]:
        spec = self._spec(queue)
        with self._lock:
            row = self._row(queue, index)
            update = self._validated_update(queue, row, values)
            old = {field: row.get(field, "") for field in update}
            row.update(update)
            self._write_queue(queue)
            audit = {
                "schema_version": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "queue": queue,
                "index": index,
                "row_id": row[spec.id_field],
                "old": old,
                "new": update,
                "queue_sha256": _sha256(self.review_dir / spec.csv_name),
            }
            with (self.review_dir / "annotation_audit.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(json.dumps(audit, sort_keys=True) + "\n")
        return self.get_item(queue, index)

    def analysis(self) -> dict[str, Any]:
        return analyze_review(self.candidate_dataset, review_dir=self.review_dir)

    def _signal_for(self, queue: str, row: dict[str, str]) -> np.ndarray:
        spec = self._spec(queue)
        signal_index = self._signal_indices[queue][row[spec.id_field]]
        return self._signals[queue][signal_index]

    @lru_cache(maxsize=128)
    def plot_png(self, queue: str, index: int, signal_view: str = "filtered") -> bytes:
        if signal_view not in {"filtered", "raw"}:
            raise ValueError("signal_view must be filtered or raw")
        row = self._row(queue, index)
        raw_signal = self._signal_for(queue, row)
        config = YeastDetectionConfig()
        signal = (
            bandpass_yeast_signal(raw_signal, config)
            if signal_view == "filtered"
            else raw_signal
        )
        sampling_frequency_hz = config.sampling_frequency_hz
        frequencies, times, power = spectrogram(
            signal,
            fs=sampling_frequency_hz,
            nperseg=512,
            noverlap=384,
            window="hann",
            mode="magnitude",
        )
        band = (frequencies >= config.low_freq_hz) & (frequencies <= config.high_freq_hz)

        if queue == "candidate":
            figure = Figure(figsize=(12.0, 6.2), constrained_layout=True)
            time_axis, spectrum_axis = figure.subplots(2, 1)
            time_ms = (
                np.arange(signal.size) - signal.size // 2
            ) / sampling_frequency_hz * 1000.0
            spectrum_time_ms = (times - signal.size / 2 / sampling_frequency_hz) * 1000.0
            time_axis.plot(time_ms, signal, color="#202428", linewidth=0.65)
            spectrum_axis.axvline(0.0, color="white", linewidth=0.9)
            if int(row["candidate_index"]) >= 0:
                crop_start = int(row["center_index"]) - signal.size // 2
                left = (
                    int(row["event_start"]) - crop_start - signal.size // 2
                ) / sampling_frequency_hz * 1000.0
                right = (
                    int(row["event_end"]) - crop_start - signal.size // 2
                ) / sampling_frequency_hz * 1000.0
                time_axis.axvspan(left, right, color="#16876b", alpha=0.18)
                time_axis.axvline(0.0, color="#c94b32", linewidth=0.9)
            time_axis.set_xlabel("Time from proposed center (ms)")
        else:
            event_ids = json.loads(row["detected_event_ids"])
            centers = json.loads(row["detected_centers"])
            if event_ids:
                figure = Figure(figsize=(12.0, 8.6), constrained_layout=True)
                grid = figure.add_gridspec(
                    3,
                    len(event_ids),
                    height_ratios=(1.45, 1.0, 1.25),
                )
                time_axis = figure.add_subplot(grid[0, :])
                zoom_axes = [
                    figure.add_subplot(grid[1, item]) for item in range(len(event_ids))
                ]
                spectrum_axis = figure.add_subplot(grid[2, :])
            else:
                figure = Figure(figsize=(12.0, 6.2), constrained_layout=True)
                time_axis, spectrum_axis = figure.subplots(2, 1)
                zoom_axes = []
            time_ms = np.arange(signal.size) / sampling_frequency_hz * 1000.0
            spectrum_time_ms = times * 1000.0
            time_axis.plot(time_ms, signal, color="#202428", linewidth=0.6)
            if signal_view == "raw":
                envelope = np.abs(hilbert(signal))
                time_axis.plot(
                    time_ms,
                    envelope,
                    color="#1769aa",
                    linewidth=0.8,
                    alpha=0.7,
                    label="amplitude envelope",
                )
                time_axis.plot(
                    time_ms,
                    -envelope,
                    color="#1769aa",
                    linewidth=0.8,
                    alpha=0.7,
                )
            quality_colors = {"strict": "#16876b", "medium": "#b97700", "reject": "#777777"}
            seen_qualities: set[str] = set()
            zoom_half_width = 2048
            for ordinal, (event_id, center) in enumerate(zip(event_ids, centers), start=1):
                candidate = self._candidate_by_id[event_id]
                quality = candidate["quality"]
                location = center / sampling_frequency_hz * 1000.0
                event_start_ms = int(candidate["event_start"]) / sampling_frequency_hz * 1000.0
                event_end_ms = int(candidate["event_end"]) / sampling_frequency_hz * 1000.0
                color = quality_colors[quality]
                for axis in (time_axis, spectrum_axis):
                    axis.axvspan(
                        event_start_ms,
                        event_end_ms,
                        color=color,
                        alpha=0.15,
                        zorder=2,
                    )
                    axis.axvline(location, color=color, linewidth=1.0, zorder=3)
                time_axis.text(
                    location,
                    0.96,
                    str(ordinal),
                    transform=time_axis.get_xaxis_transform(),
                    color="white",
                    fontsize=8,
                    fontweight="bold",
                    ha="center",
                    va="top",
                    bbox={"boxstyle": "square,pad=0.22", "facecolor": color, "edgecolor": "none"},
                    zorder=4,
                )

                zoom_axis = zoom_axes[ordinal - 1]
                zoom_start = max(0, int(center) - zoom_half_width)
                zoom_end = min(signal.size, int(center) + zoom_half_width)
                zoom_signal = signal[zoom_start:zoom_end]
                zoom_time_ms = (
                    np.arange(zoom_start, zoom_end) - int(center)
                ) / sampling_frequency_hz * 1000.0
                zoom_axis.plot(zoom_time_ms, zoom_signal, color="#202428", linewidth=0.65)
                if signal_view == "raw":
                    zoom_envelope = np.abs(hilbert(zoom_signal))
                    zoom_axis.plot(
                        zoom_time_ms,
                        zoom_envelope,
                        color="#1769aa",
                        linewidth=0.75,
                        alpha=0.65,
                    )
                    zoom_axis.plot(
                        zoom_time_ms,
                        -zoom_envelope,
                        color="#1769aa",
                        linewidth=0.75,
                        alpha=0.65,
                    )
                zoom_axis.axvspan(
                    (int(candidate["event_start"]) - int(center))
                    / sampling_frequency_hz
                    * 1000.0,
                    (int(candidate["event_end"]) - int(center))
                    / sampling_frequency_hz
                    * 1000.0,
                    color=color,
                    alpha=0.15,
                )
                zoom_axis.axvline(0.0, color=color, linewidth=0.9)
                zoom_axis.set_title(
                    f"Candidate {ordinal}: {quality} | {float(candidate['width_ms']):.3g} ms",
                    fontsize=9,
                    color=color,
                )
                zoom_axis.set_xlabel("Time from candidate center (ms)")
                zoom_axis.grid(color="#d5d9dc", linewidth=0.35, alpha=0.65)
                if ordinal == 1:
                    zoom_axis.set_ylabel("Local signal")
                seen_qualities.add(quality)
            for quality in sorted(seen_qualities):
                time_axis.plot([], [], color=quality_colors[quality], label=quality)
            if signal_view == "raw" or seen_qualities:
                time_axis.legend(
                    loc="upper right",
                    frameon=False,
                    ncol=(1 if signal_view == "raw" else 0) + len(seen_qualities),
                )
            time_axis.set_xlabel("Time from trace start (ms)")

        signal_label = "Filtered signal (7-80 kHz)" if signal_view == "filtered" else "Raw signal"
        time_axis.set_ylabel(signal_label)
        time_axis.grid(color="#d5d9dc", linewidth=0.4, alpha=0.7)
        spectrum_axis.pcolormesh(
            spectrum_time_ms,
            frequencies[band] / 1000.0,
            20.0 * np.log10(power[band] + 1.0e-8),
            shading="auto",
            cmap="magma",
        )
        spectrum_axis.set_xlabel(time_axis.get_xlabel())
        spectrum_axis.set_ylabel("Frequency (kHz)")
        figure.suptitle(f"{row.get('review_stratum', '')} | {row[self._spec(queue).id_field]}")
        output = io.BytesIO()
        FigureCanvasAgg(figure).print_png(output)
        return output.getvalue()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True) + "\n").encode("utf-8")


def build_handler(
    workspace: YeastReviewWorkspace,
    *,
    read_only: bool = False,
) -> type[BaseHTTPRequestHandler]:
    class ReviewHandler(BaseHTTPRequestHandler):
        server_version = "YeastReview/1"

        def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status.value)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                return

        def _error(self, status: HTTPStatus, exc: Exception) -> None:
            self._send(status, _json_bytes({"error": str(exc)}), "application/json")

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    page = REVIEW_HTML.replace(
                        "__READ_ONLY__", "true" if read_only else "false"
                    )
                    self._send(
                        HTTPStatus.OK,
                        page.encode("utf-8"),
                        "text/html; charset=utf-8",
                    )
                    return
                if parsed.path == "/api/items":
                    queue = parse_qs(parsed.query).get("queue", ["candidate"])[0]
                    self._send(HTTPStatus.OK, _json_bytes(workspace.list_items(queue)), "application/json")
                    return
                if parsed.path == "/api/item":
                    query = parse_qs(parsed.query)
                    queue = query.get("queue", ["candidate"])[0]
                    index = int(query.get("index", ["0"])[0])
                    self._send(HTTPStatus.OK, _json_bytes(workspace.get_item(queue, index)), "application/json")
                    return
                if parsed.path == "/api/analysis":
                    self._send(HTTPStatus.OK, _json_bytes(workspace.analysis()), "application/json")
                    return
                parts = parsed.path.strip("/").split("/")
                if len(parts) == 3 and parts[0] == "plot" and parts[2].endswith(".png"):
                    queue = parts[1]
                    index = int(parts[2][:-4])
                    signal_view = parse_qs(parsed.query).get("signal_view", ["filtered"])[0]
                    self._send(
                        HTTPStatus.OK,
                        workspace.plot_png(queue, index, signal_view),
                        "image/png",
                    )
                    return
                self._error(HTTPStatus.NOT_FOUND, ValueError("Not found"))
            except (ValueError, IndexError) as exc:
                self._error(HTTPStatus.BAD_REQUEST, exc)
            except Exception as exc:  # Keep the local reviewer responsive and report the error.
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, exc)

        def do_POST(self) -> None:  # noqa: N802
            if read_only:
                self._error(
                    HTTPStatus.METHOD_NOT_ALLOWED,
                    ValueError("This review is served in read-only mode"),
                )
                return
            if urlparse(self.path).path != "/api/item":
                self._error(HTTPStatus.NOT_FOUND, ValueError("Not found"))
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 1_000_000:
                    raise ValueError("Invalid request size")
                payload = json.loads(self.rfile.read(length))
                result = workspace.save_item(
                    str(payload["queue"]),
                    int(payload["index"]),
                    dict(payload["values"]),
                )
                self._send(HTTPStatus.OK, _json_bytes(result), "application/json")
            except (KeyError, TypeError, ValueError, IndexError, json.JSONDecodeError) as exc:
                self._error(HTTPStatus.BAD_REQUEST, exc)
            except Exception as exc:
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, exc)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return ReviewHandler


def serve_review(
    candidate_dataset: Path,
    review_dir: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    read_only: bool = False,
) -> None:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("The annotation server may bind only to a loopback address")
    workspace = YeastReviewWorkspace(candidate_dataset, review_dir)
    server = ThreadingHTTPServer(
        (host, port),
        build_handler(workspace, read_only=read_only),
    )
    print(f"Yeast review server: http://{host}:{port}", flush=True)
    print(f"Candidate dataset: {workspace.candidate_dataset}", flush=True)
    print(
        f"{'Read-only annotations' if read_only else 'Editable review directory'}: "
        f"{workspace.review_dir}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


REVIEW_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Yeast event review</title>
  <style>
    :root { color-scheme: light; font-family: Inter, Arial, sans-serif; color: #202428; background: #eef1f2; }
    * { box-sizing: border-box; letter-spacing: 0; }
    [hidden] { display: none !important; }
    body { margin: 0; min-width: 320px; }
    header { background: #202428; color: white; padding: 14px 24px; display: flex; gap: 20px; align-items: center; justify-content: space-between; }
    h1 { font-size: 20px; margin: 0; font-weight: 650; }
    #gate { font-size: 13px; color: #d9e2e4; }
    nav { background: white; border-bottom: 1px solid #cbd1d4; padding: 0 24px; display: flex; gap: 4px; }
    nav button { border: 0; border-bottom: 3px solid transparent; background: transparent; padding: 12px 16px 10px; font-weight: 650; color: #485156; cursor: pointer; }
    nav button.active { color: #116b57; border-color: #16876b; }
    main { max-width: 1440px; margin: 0 auto; padding: 18px 24px 32px; }
    .toolbar { display: grid; grid-template-columns: auto auto minmax(160px, 1fr) auto; gap: 8px; align-items: center; margin-bottom: 12px; }
    button, input, textarea { font: inherit; }
    button.command { border: 1px solid #aeb7bb; border-radius: 4px; background: white; color: #202428; min-height: 36px; padding: 7px 12px; cursor: pointer; }
    button.command:disabled { opacity: .45; cursor: default; }
    button.primary { background: #116b57; color: white; border-color: #116b57; font-weight: 650; }
    #progress { font-size: 14px; color: #485156; text-align: center; }
    .pending-toggle { justify-self: end; display: flex; align-items: center; gap: 7px; font-size: 14px; }
    .readonly-banner { margin-bottom: 12px; border: 1px solid #8fb9ad; border-radius: 4px; background: #e6f4ef; color: #0d5b49; padding: 10px 12px; font-size: 13px; font-weight: 650; }
    .meta { background: white; border: 1px solid #cbd1d4; padding: 10px 12px; display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 8px 14px; font-size: 12px; }
    .meta div { min-width: 0; overflow-wrap: anywhere; }
    .meta span { color: #69757a; display: block; margin-bottom: 2px; }
    .plot-toolbar { background: white; border: 1px solid #cbd1d4; border-top: 0; padding: 8px 12px; display: flex; align-items: center; justify-content: flex-end; gap: 10px; }
    .plot-toolbar > span { color: #69757a; font-size: 13px; font-weight: 650; }
    .view-segment { display: grid; grid-template-columns: 1fr 1fr; }
    .view-segment button { border: 1px solid #aeb7bb; background: white; color: #485156; min-height: 32px; padding: 5px 12px; cursor: pointer; }
    .view-segment button:first-child { border-radius: 4px 0 0 4px; }
    .view-segment button:last-child { border-left: 0; border-radius: 0 4px 4px 0; }
    .view-segment button.active { background: #d9eee8; border-color: #16876b; color: #0d5b49; font-weight: 650; }
    .plot { width: 100%; aspect-ratio: 12 / 6.2; object-fit: contain; display: block; background: white; border: 1px solid #cbd1d4; border-top: 0; }
    .plot.full-trace { aspect-ratio: 12 / 8.6; }
    form { background: white; border: 1px solid #cbd1d4; border-top: 0; padding: 14px; }
    .decisions { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
    fieldset { margin: 0; border: 0; padding: 0; min-width: 0; }
    legend, label.title { font-size: 13px; font-weight: 650; margin-bottom: 6px; display: block; }
    .segment { display: grid; grid-template-columns: 1fr 1fr; }
    .segment label { border: 1px solid #aeb7bb; padding: 7px; text-align: center; cursor: pointer; }
    .segment label:first-child { border-radius: 4px 0 0 4px; }
    .segment label:last-child { border-radius: 0 4px 4px 0; border-left: 0; }
    .segment input { position: absolute; opacity: 0; }
    .segment label:has(input:checked) { background: #d9eee8; border-color: #16876b; color: #0d5b49; font-weight: 650; }
    .counts { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
    input[type="number"], input[type="text"], textarea { width: 100%; border: 1px solid #aeb7bb; border-radius: 4px; padding: 8px; background: #fff; }
    .reviewer { display: grid; grid-template-columns: minmax(180px, 1fr) minmax(260px, 3fr) auto; gap: 12px; align-items: end; margin-top: 14px; }
    textarea { min-height: 58px; resize: vertical; }
    #message { min-height: 20px; font-size: 13px; color: #a13929; margin-top: 8px; }
    @media (max-width: 800px) {
      header { align-items: flex-start; flex-direction: column; gap: 6px; }
      main { padding: 12px; }
      .toolbar { grid-template-columns: auto auto 1fr; }
      .pending-toggle { grid-column: 1 / -1; justify-self: start; }
      .meta { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .decisions, .counts { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .reviewer { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header><h1>Yeast event review</h1><div id="gate">Gate 1: loading</div></header>
  <nav><button id="candidate-tab" class="active">Candidate windows</button><button id="file-tab">Full traces</button></nav>
  <main>
    <div id="readonly-banner" class="readonly-banner" hidden>Inspection en lecture seule des annotations finales. Aucune décision ou note ne peut être modifiée.</div>
    <div class="toolbar">
      <button id="prev" class="command">Previous</button>
      <button id="next" class="command">Next</button>
      <div id="progress"></div>
      <label class="pending-toggle"><input id="pending-only" type="checkbox" checked> Pending only</label>
    </div>
    <section id="meta" class="meta"></section>
    <div class="plot-toolbar">
      <span>Signal view</span>
      <div class="view-segment" role="group" aria-label="Signal view">
        <button id="filtered-view" class="active" type="button" aria-pressed="true">Filtered 7-80 kHz</button>
        <button id="raw-view" type="button" aria-pressed="false">Raw</button>
      </div>
    </div>
    <img id="plot" class="plot" alt="Signal and spectrogram review plot">
    <form id="form">
      <div id="candidate-fields" class="decisions"></div>
      <div id="file-fields" class="counts" hidden></div>
      <div class="reviewer">
        <label><span class="title">Reviewer</span><input id="reviewer" type="text" required></label>
        <label><span class="title">Notes</span><textarea id="notes"></textarea></label>
        <button class="command primary" type="submit">Save and next</button>
      </div>
      <div id="message"></div>
    </form>
  </main>
<script>
const readOnly = __READ_ONLY__;
const candidateFields = [
  ["review_event_present", "Box matches an event"],
  ["review_center_acceptable", "Center acceptable"],
  ["review_full_event_visible", "Full event visible"],
  ["review_artifact", "Artifact present"]
];
const fileFields = [
  ["review_true_event_count", "True events"],
  ["review_false_retained_candidate_count", "False retained"],
  ["review_true_rejected_candidate_count", "True rejected"],
  ["review_missed_event_count", "Missed events"]
];
let queue = "candidate", items = [], visible = [], cursor = 0, current = null;
let signalView = localStorage.getItem("yeast-signal-view") || "filtered";
if (!['filtered', 'raw'].includes(signalView)) signalView = "filtered";
const $ = id => document.getElementById(id);

function updateSignalView(nextView, reloadPlot = true) {
  signalView = nextView;
  localStorage.setItem("yeast-signal-view", signalView);
  for (const view of ["filtered", "raw"]) {
    const button = $(`${view}-view`);
    const selected = view === signalView;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-pressed", String(selected));
  }
  if (reloadPlot) updatePlot();
}

function updatePlot() {
  if (!current) return;
  const params = new URLSearchParams({signal_view: signalView, v: Date.now()});
  $("plot").src = `${current.plot_url}?${params}`;
}

function buildFields() {
  $("candidate-fields").innerHTML = candidateFields.map(([field, label]) => `
    <fieldset data-field="${field}"><legend>${label}</legend><div class="segment">
      <label><input type="radio" name="${field}" value="yes">Yes</label>
      <label><input type="radio" name="${field}" value="no">No</label>
    </div></fieldset>`).join("");
  $("file-fields").innerHTML = fileFields.map(([field, label]) => `
    <label><span class="title">${label}</span><input type="number" min="0" step="1" name="${field}" required></label>`).join("");
}

function enforceReadOnly() {
  if (!readOnly) return;
  document.querySelectorAll("#form input, #form textarea").forEach(
    input => input.disabled = true
  );
  document.querySelector('#form button[type="submit"]').hidden = true;
}

async function jsonFetch(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Request failed");
  return data;
}

async function loadItems(preferredIndex = null) {
  const state = await jsonFetch(`/api/items?queue=${queue}`);
  items = state.items;
  await applyFilter(preferredIndex);
  await refreshGate();
}

async function applyFilter(preferredIndex = null) {
  visible = $("pending-only").checked ? items.filter(item => !item.complete) : items.slice();
  if (!visible.length) visible = items.slice();
  if (preferredIndex !== null) {
    const found = visible.findIndex(item => item.index === preferredIndex);
    cursor = found >= 0 ? found : Math.min(cursor, visible.length - 1);
  } else cursor = Math.min(cursor, Math.max(visible.length - 1, 0));
  await loadCurrent();
}

async function loadCurrent() {
  if (!visible.length) return;
  const selected = visible[cursor];
  current = await jsonFetch(`/api/item?queue=${queue}&index=${selected.index}`);
  const row = current.row;
  $("progress").textContent = `${cursor + 1} / ${visible.length} visible | ${items.filter(x => x.complete).length} / ${items.length} complete`;
  $("prev").disabled = cursor <= 0;
  $("next").disabled = cursor >= visible.length - 1;
  const metadata = [
    ["ID", current.id], ["Stratum", row.review_stratum], ["Source", row.source_group],
    ["Quality", row.quality || "full trace"], ["Path", row.relative_path],
    ["Candidates", row.n_candidates || "-"], ["Detected", row.detected_qualities || "-"],
    ["Retained", row.n_retained_candidates || "-"],
    ["Rejected", row.n_rejected_candidates || "-"], ["SNR proxy", row.snr_proxy || "-"],
    ["Width (ms)", row.width_ms || "-"]
  ];
  $("meta").innerHTML = metadata.map(([key, value]) => `<div><span>${escapeHtml(key)}</span>${escapeHtml(value || "-")}</div>`).join("");
  $("candidate-fields").hidden = queue !== "candidate";
  $("file-fields").hidden = queue !== "file";
  $("plot").classList.toggle(
    "full-trace",
    queue === "file" && Number(row.n_candidates) > 0
  );
  document.querySelectorAll("#candidate-fields input").forEach(
    input => input.disabled = queue !== "candidate"
  );
  document.querySelectorAll("#file-fields input").forEach(
    input => input.disabled = queue !== "file"
  );
  const fields = queue === "candidate" ? candidateFields : fileFields;
  for (const [field] of fields) {
    const value = row[field] || "";
    if (queue === "candidate") {
      document.querySelectorAll(`[name="${field}"]`).forEach(input => input.checked = input.value === value.toLowerCase());
    } else document.querySelector(`[name="${field}"]`).value = value;
  }
  if (queue === "candidate") {
    const retained = row.quality === "strict" || row.quality === "medium";
    for (const field of ["review_center_acceptable", "review_full_event_visible"]) {
      document.querySelector(`fieldset[data-field="${field}"]`).hidden = !retained;
    }
  }
  $("reviewer").value = row.reviewer || localStorage.getItem("yeast-reviewer") || "";
  $("notes").value = row.review_notes || "";
  $("message").textContent = "";
  enforceReadOnly();
  updatePlot();
}

async function refreshGate() {
  const result = await jsonFetch("/api/analysis");
  $("gate").textContent = `Gate 1: ${result.gate_1_status} | candidates ${result.candidate_review.n_complete}/${result.candidate_review.n_expected} | traces ${result.full_trace_review.n_complete}/${result.full_trace_review.n_expected}`;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, character => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"})[character]);
}

async function changeQueue(nextQueue) {
  queue = nextQueue; cursor = 0;
  $("candidate-tab").classList.toggle("active", queue === "candidate");
  $("file-tab").classList.toggle("active", queue === "file");
  await loadItems();
}

async function advanceAfterSave(savedIndex) {
  const state = await jsonFetch(`/api/items?queue=${queue}`);
  items = state.items;
  const nextItems = $("pending-only").checked ? items.filter(item => !item.complete) : items;
  const nextItem = nextItems.find(item => item.index > savedIndex) || nextItems[0] || null;
  await applyFilter(nextItem ? nextItem.index : savedIndex);
  await refreshGate();
}

$("form").addEventListener("submit", async event => {
  event.preventDefault();
  try {
    const values = { reviewer: $("reviewer").value, review_notes: $("notes").value };
    const fields = queue === "candidate" ? candidateFields : fileFields;
    for (const [field] of fields) {
      const input = queue === "candidate" ? document.querySelector(`[name="${field}"]:checked`) : document.querySelector(`[name="${field}"]`);
      values[field] = input ? input.value : "";
    }
    localStorage.setItem("yeast-reviewer", values.reviewer);
    await jsonFetch("/api/item", { method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({queue, index: current.index, values}) });
    await advanceAfterSave(current.index);
  } catch (error) { $("message").textContent = error.message; }
});

$("prev").onclick = () => { if (cursor > 0) { cursor -= 1; loadCurrent(); } };
$("next").onclick = () => { if (cursor < visible.length - 1) { cursor += 1; loadCurrent(); } };
$("pending-only").onchange = () => applyFilter(current ? current.index : null);
$("candidate-tab").onclick = () => changeQueue("candidate");
$("file-tab").onclick = () => changeQueue("file");
buildFields();
if (readOnly) {
  $("readonly-banner").hidden = false;
  $("pending-only").checked = false;
  $("pending-only").closest("label").hidden = true;
  enforceReadOnly();
}
updateSignalView(signalView, false);
$("filtered-view").onclick = () => updateSignalView("filtered");
$("raw-view").onclick = () => updateSignalView("raw");
$("reviewer").oninput = () => localStorage.setItem("yeast-reviewer", $("reviewer").value);
if (new URLSearchParams(window.location.search).get("queue") === "file") {
  changeQueue("file").catch(error => $("message").textContent = error.message);
} else {
  loadItems().catch(error => $("message").textContent = error.message);
}
</script>
</body>
</html>
"""
