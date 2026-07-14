from __future__ import annotations

import csv
from pathlib import Path

from particles2snr.yeast_source_index import build_source_index, summarize_source_index, write_source_index


def _row(path: str, group: str, index: int, digest: str) -> dict[str, str]:
    return {
        "relative_path": path,
        "source_group": group,
        "filename_series": f"Yeast_{group}_0",
        "filename_index": str(index),
        "sha256": digest,
        "size_bytes": "100",
        "n_values": "8",
        "npy_dtype": "float64",
    }


def test_source_index_keeps_duplicate_family_in_one_split() -> None:
    rows = [
        _row("budding/a_63.npy", "budding", 63, "same"),
        _row("budding/a_64.npy", "budding", 64, "same"),
        _row("shmoo/b_70.npy", "shmoo", 70, "other"),
    ]
    index = build_source_index(
        rows,
        raw_dataset="yeast@v1",
        acquisition_id="session-1",
        capture_block_size=64,
    )
    duplicate_rows = [row for row in index if row["duplicate_family_id"] == "same"]
    assert len({row["development_split"] for row in duplicate_rows}) == 1
    assert len({row["capture_block_id"] for row in duplicate_rows}) == 1
    assert len({row["nominal_capture_block_id"] for row in duplicate_rows}) == 2
    assert sum(row["is_canonical_duplicate_member"] for row in duplicate_rows) == 1
    summary = summarize_source_index(index)
    assert summary["n_duplicate_excess_rows"] == 1
    assert summary["n_duplicate_families_crossing_splits"] == 0
    assert summary["n_capture_blocks_crossing_splits"] == 0
    assert summary["acquisition_ood_ready"] is False


def test_source_index_writes_boolean_rows(tmp_path: Path) -> None:
    rows = build_source_index(
        [_row("mix/a_1.npy", "mix", 1, "digest")],
        raw_dataset="yeast@v1",
        acquisition_id="session-1",
    )
    output = tmp_path / "index"
    write_source_index(output, rows, summarize_source_index(rows))
    with (output / "source_index.csv").open(newline="", encoding="utf-8") as handle:
        persisted = list(csv.DictReader(handle))
    assert persisted[0]["condition_id"] == "stationary-mixed"
    assert persisted[0]["is_canonical_duplicate_member"] == "True"


def test_source_index_stratifies_groups_with_at_least_three_blocks() -> None:
    rows = []
    for group in ("budding", "mix"):
        rows.extend(
            _row(f"{group}/a_{index}.npy", group, index, f"{group}-{index}")
            for index in range(12)
        )
    index = build_source_index(
        rows,
        raw_dataset="yeast@v1",
        acquisition_id="session-1",
        capture_block_size=3,
    )
    summary = summarize_source_index(index)
    assert summary["source_groups_with_all_development_splits"] == ["budding", "mix"]
    for group in ("budding", "mix"):
        assert set(summary["source_group_split_counts_canonical"][group]) == {
            "development_train",
            "development_validation",
            "in_session_test",
        }
