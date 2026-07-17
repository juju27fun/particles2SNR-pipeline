from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from particles2snr.yeast_development_dataset import build_development_dataset


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_parent(tmp_path: Path) -> tuple[Path, list[dict[str, str]], np.ndarray]:
    root = tmp_path / "parent"
    root.mkdir()
    rows = [
        {"event_id": "sealed", "development_split": "sealed_acquisition_test", "signal_row": "0"},
        {"event_id": "train-a", "development_split": "development_train", "signal_row": "3"},
        {"event_id": "validation", "development_split": "development_validation", "signal_row": "1"},
        {"event_id": "in-session", "development_split": "in_session_test", "signal_row": "2"},
        {"event_id": "train-b", "development_split": "development_train", "signal_row": "4"},
    ]
    _write_csv(root / "events.csv", rows)
    signals = np.stack(
        [np.full(4096, 1000.0 + index, dtype=np.float32) for index in range(len(rows))]
    )
    np.save(root / "signals.npy", signals)
    (root / "input_contract.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "contract_id": "yeast-event-8192to4096-bandpass-global-v1",
                "output_length": 4096,
                "output_sampling_frequency_hz": 1_000_000.0,
                "normalization": {"policy": "parent normalization"},
            }
        ),
        encoding="utf-8",
    )
    split_counts = {
        "development_train": 2,
        "development_validation": 1,
        "in_session_test": 1,
        "sealed_acquisition_test": 1,
    }
    (root / "dataset_summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "n_events": len(rows),
                "split_counts": split_counts,
                "signals_shape": [len(rows), 4096],
                "signals_dtype": "float32",
            }
        ),
        encoding="utf-8",
    )
    return root, rows, signals


def test_builder_physically_excludes_sealed_signals_and_rewrites_rows(tmp_path: Path) -> None:
    parent, _, parent_signals = _make_parent(tmp_path)
    output = tmp_path / "development"

    summary = build_development_dataset(input_root=parent, output_dir=output)

    with (output / "events.csv").open(newline="", encoding="utf-8") as handle:
        output_rows = list(csv.DictReader(handle))
    assert [row["event_id"] for row in output_rows] == ["train-a", "validation", "train-b"]
    assert [row["signal_row"] for row in output_rows] == ["0", "1", "2"]
    assert [row["parent_signal_row"] for row in output_rows] == ["3", "1", "4"]
    assert {row["development_split"] for row in output_rows} == {
        "development_train",
        "development_validation",
    }

    output_signals = np.load(output / "signals.npy", allow_pickle=False)
    np.testing.assert_array_equal(output_signals, parent_signals[[3, 1, 4]])
    assert not np.any(output_signals == parent_signals[0, 0])
    assert not np.any(output_signals == parent_signals[2, 0])
    assert summary["split_counts"] == {
        "development_train": 2,
        "development_validation": 1,
    }
    assert summary["excluded_split_counts"] == {
        "in_session_test": 1,
        "sealed_acquisition_test": 1,
    }
    assert summary["sealed_splits_used"] == []

    contract = json.loads((output / "input_contract.json").read_text(encoding="utf-8"))
    assert contract["contract_id"] == "yeast-event-8192to4096-bandpass-global-v1"
    assert contract["output_length"] == 4096
    assert contract["parent_dataset"] == "yeast-events-representation@v3"
    assert contract["provenance"]["sealed_splits_used"] == []
    for name, digest in summary["source_checksums"].items():
        assert digest == _sha256(parent / name)
    for name, digest in summary["output_checksums"].items():
        assert digest == _sha256(output / name)


def test_builder_outputs_are_deterministic(tmp_path: Path) -> None:
    parent, _, _ = _make_parent(tmp_path)
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"

    summary_a = build_development_dataset(input_root=parent, output_dir=output_a)
    summary_b = build_development_dataset(input_root=parent, output_dir=output_b)

    assert summary_a == summary_b
    for name in ("signals.npy", "events.csv", "input_contract.json", "dataset_summary.json"):
        assert (output_a / name).read_bytes() == (output_b / name).read_bytes()


def test_builder_refuses_existing_output_directory(tmp_path: Path) -> None:
    parent, _, _ = _make_parent(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("untouched", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        build_development_dataset(input_root=parent, output_dir=output)

    assert marker.read_text(encoding="utf-8") == "untouched"
    assert list(output.iterdir()) == [marker]


@pytest.mark.parametrize("failure", ["shape", "unexpected_split", "empty_validation"])
def test_builder_rejects_invalid_parent_dataset(tmp_path: Path, failure: str) -> None:
    parent, rows, signals = _make_parent(tmp_path)
    if failure == "shape":
        np.save(parent / "signals.npy", signals[:, :-1])
    else:
        if failure == "unexpected_split":
            rows[0]["development_split"] = "mystery_test"
        else:
            rows[2]["development_split"] = "development_train"
        _write_csv(parent / "events.csv", rows)
        split_counts = dict(
            sorted(
                {
                    split: sum(row["development_split"] == split for row in rows)
                    for split in {row["development_split"] for row in rows}
                }.items()
            )
        )
        summary = json.loads((parent / "dataset_summary.json").read_text(encoding="utf-8"))
        summary["split_counts"] = split_counts
        (parent / "dataset_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError):
        build_development_dataset(input_root=parent, output_dir=tmp_path / "output")

    assert not (tmp_path / "output").exists()
