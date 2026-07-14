from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from particles2snr.yeast_source_index import (
    build_source_index,
    build_source_index_from_manifest,
    combine_acquisition_indexes,
    summarize_source_index,
    write_source_index,
)


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


def _acquisition(rows: list[dict[str, str]], acquisition_id: str, role: str) -> dict[str, object]:
    raw_dataset = f"yeast-{acquisition_id}@v1"
    return {
        "acquisition_id": acquisition_id,
        "raw_dataset": raw_dataset,
        "role": role,
        "rows": build_source_index(
            rows,
            raw_dataset=raw_dataset,
            acquisition_id=acquisition_id,
            capture_block_size=3,
        ),
    }


def test_multi_acquisition_index_namespaces_ids_and_seals_holdout() -> None:
    development = _acquisition(
        [_row(f"budding/same_{index}.npy", "budding", index, f"dev-{index}") for index in range(9)],
        "session-a",
        "development",
    )
    holdout = _acquisition(
        [_row(f"budding/same_{index}.npy", "budding", index, f"ood-{index}") for index in range(9)],
        "session-b",
        "sealed_ood_test",
    )
    combined = combine_acquisition_indexes([development, holdout])
    summary = summarize_source_index(combined)

    assert len({row["record_id"] for row in combined}) == len(combined)
    assert len({row["capture_block_id"] for row in combined}) == 6
    assert {
        row["development_split"] for row in combined if row["acquisition_id"] == "session-b"
    } == {"sealed_acquisition_test"}
    assert summary["acquisition_ood_ready"] is True
    assert summary["acquisition_roles"] == {
        "session-a": ["development"],
        "session-b": ["sealed_ood_test"],
    }


def test_multi_acquisition_index_rejects_cross_session_exact_duplicates() -> None:
    development = _acquisition(
        [_row("budding/a_0.npy", "budding", 0, "copied-signal")],
        "session-a",
        "development",
    )
    holdout = _acquisition(
        [_row("budding/b_0.npy", "budding", 0, "copied-signal")],
        "session-b",
        "sealed_ood_test",
    )
    with pytest.raises(ValueError, match="Exact signal duplicates cross"):
        combine_acquisition_indexes([development, holdout])


def test_multi_acquisition_manifest_resolves_relative_inventories(tmp_path: Path) -> None:
    acquisitions = []
    for suffix, role in (("a", "development"), ("b", "sealed_ood_test")):
        inventory = tmp_path / f"inventory-{suffix}.csv"
        rows = [
            {**_row(f"budding/a_{index}.npy", "budding", index, f"{suffix}-{index}"), "suffix": ".npy"}
            for index in range(3)
        ]
        with inventory.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        acquisitions.append(
            {
                "acquisition_id": f"session-{suffix}",
                "raw_dataset": f"raw-{suffix}@v1",
                "source_inventory": inventory.name,
                "role": role,
            }
        )
    manifest = tmp_path / "acquisitions.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "acquisitions": acquisitions}),
        encoding="utf-8",
    )

    rows = build_source_index_from_manifest(manifest, capture_block_size=1)
    assert len(rows) == 6
    assert summarize_source_index(rows)["acquisition_ood_ready"] is True
