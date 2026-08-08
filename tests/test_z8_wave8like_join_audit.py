from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from internship_workspace.scientific_visual import computation_fingerprint
from particles2snr.z8_wave8like_join_audit import (
    AuditConfig,
    analyze_join_metrics,
    select_predeclared_cases,
    write_analysis,
)


def _write_candidate(root: Path) -> None:
    (root / "val/signals").mkdir(parents=True)
    (root / "val/labels").mkdir(parents=True)
    config = AuditConfig(segment_length=64, guard_samples=4, sampling_frequency_hz=2_000)
    positive_id = "z8w8_val_positive_0000_p00"
    background_id = "z8w8_val_background_0000_p00"
    positive = np.random.default_rng(1).normal(size=config.long_length).astype(np.float64)
    background = np.random.default_rng(2).normal(size=config.long_length).astype(np.float64)
    np.save(root / "val/signals" / f"{positive_id}.npy", positive)
    np.save(root / "val/signals" / f"{background_id}.npy", background)
    labels = [
        (0, 52.0, 60.0),
        (1, 68.0, 76.0),
        (2, 180.0, 188.0),
    ]
    label_lines = []
    for class_id, left, right in labels:
        center = (left + right) / 2 / config.long_length
        width = (right - left) / config.long_length
        label_lines.append(f"{class_id} {center:.12f} {width:.12f}\n")
    (root / "val/labels" / f"{positive_id}.txt").write_text("".join(label_lines))
    (root / "val/labels" / f"{background_id}.txt").write_text("")
    rows = [
        {
            "long_id": positive_id,
            "split": "val",
            "stratum": "positive",
            "event_ids": "event-0;event-1;event-2",
        },
        {
            "long_id": background_id,
            "split": "val",
            "stratum": "background",
            "event_ids": "",
        },
    ]
    with (root / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (root / "dataset-manifest.json").write_text("{}\n", encoding="utf-8")


def test_join_audit_covers_all_boundaries_and_predeclared_cases(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate"
    _write_candidate(candidate)
    config = AuditConfig(segment_length=64, guard_samples=4, sampling_frequency_hz=2_000)

    metrics = analyze_join_metrics(candidate, config)
    selection = select_predeclared_cases(candidate, config)

    assert len(metrics) == 6
    assert {row["boundary_index"] for row in metrics} == {1, 2, 3}
    assert selection["full_trace_long_id"].startswith("z8w8_val_positive")
    assert selection["background_counterexample_long_id"].startswith(
        "z8w8_val_background"
    )
    assert [row["class_id"] for row in selection["nearest_safe_events"]] == [0, 1, 2]


def test_analysis_owns_metrics_with_a_recomputable_contract(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    output = tmp_path / "analysis"
    _write_candidate(candidate)
    provenance = {
        "datasets": {"candidate@v1": "manifest-sha"},
        "inputs": {"manifest": "input-sha"},
        "parameters": {"split": "val"},
        "metric_definitions": {"boundary_jump": "absolute difference"},
        "code": {"module": "module-sha"},
        "git_revision": {"repository": "revision"},
    }
    fingerprint = computation_fingerprint(provenance)

    result = write_analysis(
        candidate_root=candidate,
        output_root=output,
        config=AuditConfig(
            segment_length=64,
            guard_samples=4,
            sampling_frequency_hz=2_000,
        ),
        computation_provenance=provenance,
        computation_fingerprint=fingerprint,
        run_payload={"run_id": "join-audit-test"},
    )

    manifest = json.loads(
        (output / "metrics_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["analysis_run_id"] == "join-audit-test"
    assert manifest["computation_fingerprint"] == fingerprint
    assert [metric["path"] for metric in manifest["metrics"]] == [
        "join_metrics.csv",
        "selected_cases.json",
        "summary_metrics.json",
    ]
    assert result["outputs"][-1] == "metrics_manifest.json"
