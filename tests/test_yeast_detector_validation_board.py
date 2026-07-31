from __future__ import annotations

import csv
import json
from pathlib import Path

from particles2snr.yeast_detector_validation_board import (
    CASE_SPECS,
    build_validation_board_model,
    render_validation_board_html,
)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_validation_board_joins_audits_ablation_and_human_cases(
    tmp_path: Path,
) -> None:
    review = tmp_path / "review"
    evidence = tmp_path / "evidence"
    assets = tmp_path / "report" / "plots"
    review.mkdir()
    (evidence / "images").mkdir(parents=True)

    _write_csv(
        review / "manual_review_queue.csv",
        [
            {
                "quality": "strict",
                "review_event_present": "yes",
            },
            {
                "quality": "reject",
                "review_event_present": "no",
            },
        ],
    )
    _write_csv(
        review / "manual_file_review_queue.csv",
        [
            {
                "review_true_event_count": "2",
                "review_false_retained_candidate_count": "0",
                "review_true_rejected_candidate_count": "1",
                "review_missed_event_count": "0",
            }
        ],
    )
    ablation_path = tmp_path / "comparison_summary.json"
    ablation_path.write_text(
        json.dumps(
            {
                "method": "bandpass_temporal_energy_mad",
                "selected_quality_z": 9.0,
                "split_scores": {
                    "development_validation": {
                        "categories": {
                            "matched": 10,
                            "current_only": 2,
                            "simple_only": 1,
                        },
                        "precision_vs_current": 0.9,
                        "recall_vs_current": 0.8,
                        "f1_vs_current": 0.85,
                        "center_abs_error_ms": {
                            "p50": 0.002,
                            "p95": 0.02,
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    disagreement_rows = []
    for index, spec in enumerate(CASE_SPECS):
        preference = spec["winner"]
        disagreement_rows.append(
            {
                "record_id": spec["record_id"],
                "relative_path": f"group/{index}.npy",
                "source_group": "group",
                "review_preference": preference,
                "reviewer": "Julien",
                "review_notes": "checked",
            }
        )
        (evidence / "images" / spec["image"]).write_bytes(b"png")
    disagreement_rows.append(
        {
            "record_id": "uncertain",
            "relative_path": "group/u.npy",
            "source_group": "group",
            "review_preference": "uncertain",
            "reviewer": "Julien",
            "review_notes": "",
        }
    )
    disagreement_path = tmp_path / "review_queue.csv"
    _write_csv(disagreement_path, disagreement_rows)

    model = build_validation_board_model(
        candidate_review_dirs=[review],
        ablation_summary_path=ablation_path,
        disagreement_review_path=disagreement_path,
        evidence_dir=evidence,
        assets_dir=assets,
    )
    page = render_validation_board_html(model)

    assert model["audit"]["retained_confirmed"] == 1
    assert model["audit"]["true_events"] == 2
    assert model["ablation"]["human_preferences"] == {
        "current": 3,
        "simple": 1,
        "uncertain": 1,
    }
    assert len(list(assets.glob("*.png"))) == 4
    assert "Et si un simple seuil d'énergie suffisait ?" in page
    assert "Garder le simple comme sentinelle" in page
    assert "Commentaire humain enregistré" in page
    assert "capture-ablation" in page
    assert "fetch(" not in page
    assert "<form" not in page
