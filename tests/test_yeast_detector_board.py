from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from particles2snr.yeast_detector_board import (
    CASE_SPECS,
    build_pipeline_board_model,
    render_pipeline_board_html,
)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_pipeline_board_uses_reviewed_cases_and_renders_tabs(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    review = tmp_path / "review"
    raw = tmp_path / "raw"
    assets = tmp_path / "report" / "plots"
    candidate.mkdir()
    review.mkdir()
    raw.mkdir()

    candidate_rows = []
    review_rows = []
    for index, spec in enumerate(CASE_SPECS):
        record_id = f"record-{index}"
        relative_path = f"group/signal-{index}.npy"
        (raw / "group").mkdir(exist_ok=True)
        signal = np.zeros(16384, dtype=np.float32)
        signal[7800:8500] = np.sin(np.linspace(0, 40, 700)).astype(np.float32)
        np.save(raw / relative_path, signal)
        candidate_row = {
            "event_id": spec["event_id"],
            "record_id": record_id,
            "relative_path": relative_path,
            "source_group": "group",
            "center_index": "8192",
            "event_start": "7700",
            "event_end": "8600",
            "quality": "strict" if index != 1 else "reject",
            "snr_proxy": "20",
            "energy_concentration": "0.5",
            "phase_coherence": "0.8",
            "n_doppler_peaks": (
                "2" if spec["slug"] == "one-event-multi-doppler" else "1"
            ),
            "doppler_low_hz": "12000",
            "doppler_high_hz": (
                "20000"
                if spec["slug"] == "one-event-multi-doppler"
                else "12000"
            ),
            "doppler_peak_hz": "25000",
            "width_ms": "0.45",
        }
        candidate_rows.append(candidate_row)
        review_rows.append(
            {
                **candidate_row,
                "review_event_present": "yes" if index != 1 else "no",
                "review_center_acceptable": "yes" if index != 1 else "",
                "review_full_event_visible": "yes" if index != 1 else "",
                "review_artifact": "no",
                "reviewer": "Julien",
                "review_notes": "checked",
            }
        )
    _write_csv(candidate / "candidate_events.csv", candidate_rows)
    _write_csv(review / "manual_review_queue.csv", review_rows)

    model = build_pipeline_board_model(
        candidate_dataset=candidate,
        review_dir=review,
        raw_dataset_root=raw,
        assets_dir=assets,
    )
    page = render_pipeline_board_html(model)

    assert [case["slug"] for case in model["cases"]] == [
        "strict-clean",
        "reject-neighbour",
        "multi-event",
        "one-event-multi-doppler",
    ]
    assert all((tmp_path / "report" / case["plot"]).is_file() for case in model["cases"])
    assert "Énergie locale" in page
    assert "z[m]" in page
    assert "pas un SNR classique" in page
    assert "candidat potentiel" in page
    assert "La méthode ne compte pas les pics Doppler" in page
    assert "Le nombre de pics Doppler est mesuré seulement après" in page
    assert "Mauvais modèle mental" in page
    assert "c’est ici que le nombre de candidats est fixé" in page
    assert "L’unité comptée est le groupe temporel" in page
    assert "1 seule boîte candidate" in page
    assert 'data-case="reject-neighbour"' in page
