from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from particles2snr.yeast_template_comparator import build_template_comparator


def test_template_comparator_uses_disjoint_train_only_records(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    rows = []
    for index in range(40):
        rows.append(
            {
                "signal_row": index,
                "event_id": f"e-{index}",
                "record_id": f"r-{index}",
                "source_group": "a",
                "development_split": "followup_train" if index < 36 else "followup_test",
            }
        )
    with (source / "events.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    rng = np.random.default_rng(2)
    np.save(source / "signals.npy", rng.normal(size=(40, 64)).astype(np.float32))
    output = tmp_path / "output"
    summary = build_template_comparator(
        followup_root=source, output_dir=output, n_train=20, n_validation=10
    )
    metadata = list(csv.DictReader((output / "simulation_metadata.csv").open(newline="")))
    train = {row["template_source_record_id"] for row in metadata if row["split"] == "train"}
    validation = {
        row["template_source_record_id"] for row in metadata if row["split"] == "validation"
    }
    assert not train & validation
    assert {row["template_source_split"] for row in metadata} == {"followup_train"}
    assert summary["retained_physical_factors"] is False
    assert json.loads((output / "dataset_summary.json").read_text())["template_record_crossings"] == 0
