from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import spectrogram


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _candidate_pdf(rows: list[dict[str, str]], signals: np.ndarray, path: Path) -> None:
    with PdfPages(path) as pdf:
        for start in range(0, len(rows), 2):
            subset = rows[start : start + 2]
            fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.5), constrained_layout=True)
            for local_index, (time_axis, spectrum_axis) in enumerate(axes):
                if local_index >= len(subset):
                    time_axis.axis("off")
                    spectrum_axis.axis("off")
                    continue
                row = subset[local_index]
                signal = signals[start + local_index]
                time_ms = (np.arange(signal.size) - signal.size // 2) / 2_000_000.0 * 1000.0
                time_axis.plot(time_ms, signal, color="black", linewidth=0.55)
                if int(row["candidate_index"]) >= 0:
                    crop_start = int(row["center_index"]) - signal.size // 2
                    left = (int(row["event_start"]) - crop_start - signal.size // 2) / 2_000_000.0 * 1000.0
                    right = (int(row["event_end"]) - crop_start - signal.size // 2) / 2_000_000.0 * 1000.0
                    time_axis.axvspan(left, right, color="#009E73", alpha=0.18)
                    time_axis.axvline(0.0, color="#D55E00", linewidth=0.8)
                time_axis.set_title(
                    f"{row['review_stratum']} | {row['event_id']} | "
                    f"snr={row['snr_proxy']} width={row['width_ms']} ms",
                    fontsize=8,
                )
                time_axis.set_xlabel("time from proposed center (ms)")
                time_axis.set_ylabel("raw")
                frequencies, times, power = spectrogram(
                    signal,
                    fs=2_000_000.0,
                    nperseg=512,
                    noverlap=384,
                    window="hann",
                    mode="magnitude",
                )
                band = (frequencies >= 7_000.0) & (frequencies <= 80_000.0)
                spectrum_axis.pcolormesh(
                    (times - signal.size / 2 / 2_000_000.0) * 1000.0,
                    frequencies[band] / 1000.0,
                    20.0 * np.log10(power[band] + 1.0e-8),
                    shading="auto",
                    cmap="magma",
                )
                spectrum_axis.axvline(0.0, color="white", linewidth=0.8)
                spectrum_axis.set_xlabel("time from proposed center (ms)")
                spectrum_axis.set_ylabel("frequency (kHz)")
            pdf.savefig(fig)
            plt.close(fig)


def _file_pdf(
    rows: list[dict[str, str]],
    signals: np.ndarray,
    candidate_by_id: dict[str, dict[str, str]],
    path: Path,
) -> None:
    quality_colors = {"strict": "#009E73", "medium": "#E69F00", "reject": "#999999"}
    with PdfPages(path) as pdf:
        for start in range(0, len(rows), 2):
            subset = rows[start : start + 2]
            fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.5), constrained_layout=True)
            for local_index, (time_axis, spectrum_axis) in enumerate(axes):
                if local_index >= len(subset):
                    time_axis.axis("off")
                    spectrum_axis.axis("off")
                    continue
                row = subset[local_index]
                signal = signals[start + local_index]
                time_ms = np.arange(signal.size) / 2_000_000.0 * 1000.0
                time_axis.plot(time_ms, signal, color="black", linewidth=0.5)
                event_ids = json.loads(row["detected_event_ids"])
                centers = json.loads(row["detected_centers"])
                qualities = [candidate_by_id[event_id]["quality"] for event_id in event_ids]
                for center, quality in zip(centers, qualities):
                    time_axis.axvline(
                        center / 2_000_000.0 * 1000.0,
                        color=quality_colors[quality],
                        linewidth=1.0,
                    )
                time_axis.set_title(
                    f"{row['review_stratum']} | {row['relative_path']} | qualities={','.join(qualities)}",
                    fontsize=8,
                )
                time_axis.set_xlabel("time from trace start (ms)")
                time_axis.set_ylabel("raw")
                frequencies, times, power = spectrogram(
                    signal,
                    fs=2_000_000.0,
                    nperseg=512,
                    noverlap=384,
                    window="hann",
                    mode="magnitude",
                )
                band = (frequencies >= 7_000.0) & (frequencies <= 80_000.0)
                spectrum_axis.pcolormesh(
                    times * 1000.0,
                    frequencies[band] / 1000.0,
                    20.0 * np.log10(power[band] + 1.0e-8),
                    shading="auto",
                    cmap="magma",
                )
                for center, quality in zip(centers, qualities):
                    spectrum_axis.axvline(
                        center / 2_000_000.0 * 1000.0,
                        color=quality_colors[quality],
                        linewidth=1.0,
                    )
                spectrum_axis.set_xlabel("time from trace start (ms)")
                spectrum_axis.set_ylabel("frequency (kHz)")
            pdf.savefig(fig)
            plt.close(fig)


def render_review_figures(candidate_dataset: Path, output_dir: Path) -> dict[str, Any]:
    candidate_rows = _read_csv(candidate_dataset / "manual_review_queue.csv")
    file_rows = _read_csv(candidate_dataset / "manual_file_review_queue.csv")
    all_candidates = _read_csv(candidate_dataset / "candidate_events.csv")
    candidate_by_id = {row["event_id"]: row for row in all_candidates}
    with np.load(candidate_dataset / "manual_review_signals.npz") as payload:
        candidate_signals = payload["signals"]
    with np.load(candidate_dataset / "manual_file_review_signals.npz") as payload:
        file_signals = payload["signals"]
    if len(candidate_rows) != candidate_signals.shape[0]:
        raise ValueError("Candidate review rows and signals have different lengths")
    if len(file_rows) != file_signals.shape[0]:
        raise ValueError("File review rows and signals have different lengths")

    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(candidate_dataset / "manual_review_queue.csv", output_dir / "manual_review_queue.csv")
    shutil.copy2(
        candidate_dataset / "manual_file_review_queue.csv",
        output_dir / "manual_file_review_queue.csv",
    )
    _candidate_pdf(candidate_rows, candidate_signals, output_dir / "candidate_precision_review.pdf")
    _file_pdf(file_rows, file_signals, candidate_by_id, output_dir / "full_trace_recall_review.pdf")
    summary = {
        "schema_version": 1,
        "n_candidate_review_rows": len(candidate_rows),
        "n_full_trace_review_rows": len(file_rows),
        "candidate_pdf": "candidate_precision_review.pdf",
        "full_trace_pdf": "full_trace_recall_review.pdf",
        "candidate_annotation_csv": "manual_review_queue.csv",
        "full_trace_annotation_csv": "manual_file_review_queue.csv",
        "status": "awaiting_manual_annotation",
    }
    (output_dir / "review_figure_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
