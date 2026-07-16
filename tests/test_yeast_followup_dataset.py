from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from particles2snr.yeast_followup_dataset import (
    assign_followup_splits,
    build_followup_representation_dataset,
    validate_followup_split,
)


def _source_rows() -> list[dict[str, str]]:
    rows = []
    for group, n_blocks in (("a", 5), ("b", 2)):
        for block in range(n_blocks):
            for record in range(2):
                record_id = f"{group}-{block}-{record}"
                rows.append(
                    {
                        "record_id": record_id,
                        "source_group": group,
                        "capture_block_id": f"{group}-block-{block}",
                        "duplicate_family_id": f"family-{record_id}",
                        "is_canonical_duplicate_member": "True",
                        "development_split": "development_train",
                    }
                )
    rows.append(
        {
            "record_id": "old-final",
            "source_group": "a",
            "capture_block_id": "old-final-block",
            "duplicate_family_id": "old-final-family",
            "is_canonical_duplicate_member": "True",
            "development_split": "in_session_test",
        }
    )
    return rows


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_followup_split_is_deterministic_grouped_and_declares_two_block_limit() -> None:
    rows = _source_rows()
    first, audit = assign_followup_splits(rows)
    second, _ = assign_followup_splits(rows)
    assert first == second
    assert audit["crossing_counts"] == {
        "record_id": 0,
        "capture_block_id": 0,
        "duplicate_family_id": 0,
    }
    assert audit["status"] == "pass_with_declared_proxy_coverage_limitation"
    assert audit["limitations"][0]["source_group"] == "b"
    assert "old-final-block" not in first


def test_followup_validation_rejects_crossing_duplicate_family() -> None:
    rows = [
        {
            "record_id": f"r{index}",
            "capture_block_id": f"b{index}",
            "duplicate_family_id": "same-family",
            "source_group": "a",
            "prior_development_split": "development_train",
            "development_split": split,
        }
        for index, split in enumerate(("followup_train", "followup_test"))
    ]
    with pytest.raises(ValueError, match="leakage"):
        validate_followup_split(rows)


def test_builder_copies_only_development_and_normalizes_from_followup_train(
    tmp_path: Path,
) -> None:
    source_csv = tmp_path / "source.csv"
    source_rows = _source_rows()
    _write_csv(source_csv, source_rows)
    parent = tmp_path / "parent"
    parent.mkdir()
    events = []
    for index, source in enumerate(source_rows):
        events.append(
            {
                "event_id": f"event-{index}",
                "record_id": source["record_id"],
                "source_group": source["source_group"],
                "capture_block_id": source["capture_block_id"],
                "development_split": source["development_split"],
                "quality": "strict",
                "snr_proxy": "2.0",
                "width_ms": "1.0",
                "doppler_peak_hz": "15000",
                "energy_concentration": "0.8",
                "signal_row": str(index),
            }
        )
    _write_csv(parent / "events.csv", events)
    signals = np.stack(
        [np.linspace(-1.0, 1.0, 16, dtype=np.float32) * (index + 1) for index in range(len(events))]
    )
    np.save(parent / "signals.npy", signals)
    (parent / "input_contract.json").write_text(
        json.dumps(
            {
                "contract_id": "yeast-event-8192to4096-bandpass-global-v1",
                "output_sampling_frequency_hz": 1_000_000.0,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    summary = build_followup_representation_dataset(
        source_index_csv=source_csv,
        representation_root=parent,
        output_dir=output,
        source_dataset_id="source@v2",
        representation_dataset_id="representation@v3",
    )
    output_events = list(csv.DictReader((output / "events.csv").open(newline="")))
    assert all(row["prior_development_split"] == "development_train" for row in output_events)
    assert "old-final" not in {row["record_id"] for row in output_events}
    assert summary["n_events"] == len(source_rows) - 1
    audit = json.loads((output / "split_audit.json").read_text())
    assert audit["forbidden_event_signals_copied"] == 0
    output_signals = np.load(output / "signals.npy")
    train_rows = [int(row["signal_row"]) for row in output_events if row["development_split"] == "followup_train"]
    assert abs(float(output_signals[train_rows].mean())) < 1.0e-6
    assert abs(float(output_signals[train_rows].std()) - 1.0) < 1.0e-6
