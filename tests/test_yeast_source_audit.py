from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from particles2snr.yeast_source_audit import (
    duplicate_groups,
    inventory_source,
    summarize_inventory,
    write_inventory,
)
from particles2snr.yeast_source_import import import_verified_source


def test_inventory_detects_exact_duplicates_and_bad_arrays(tmp_path: Path) -> None:
    source = tmp_path / "source"
    budding = source / "budding"
    shmoo = source / "shmoo"
    budding.mkdir(parents=True)
    shmoo.mkdir(parents=True)
    signal = np.arange(8, dtype=np.float32)
    np.save(budding / "Yeast_budding_0_1.npy", signal)
    np.save(budding / "Yeast_budding_0_2.npy", signal)
    np.save(shmoo / "Yeast_shmoo_0_1.npy", signal + 1)
    (shmoo / "broken.npy").write_bytes(b"not-an-array")
    (source / "README.md").write_text("acquisition notes\n", encoding="utf-8")

    records = inventory_source(source)
    summary = summarize_inventory(records, source_root=source)

    assert len(records) == 5
    assert summary["n_npy_files"] == 4
    assert summary["n_load_errors"] == 1
    assert summary["n_exact_duplicate_groups"] == 1
    assert summary["n_exact_duplicate_excess_files"] == 1
    assert summary["exact_duplicate_excess_by_source_group"] == {"budding": 1}
    assert summary["n_adjacent_index_duplicate_groups"] == 1
    assert summary["n_cross_source_group_duplicate_groups"] == 0
    assert summary["split_readiness"]["status"] == "fail"
    groups = duplicate_groups(records)
    assert groups[0]["relative_paths"] == [
        "budding/Yeast_budding_0_1.npy",
        "budding/Yeast_budding_0_2.npy",
    ]


def test_inventory_writes_stable_tables_and_accepts_documented_groups(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    np.save(source / "signal_0.npy", np.asarray([1.0, 2.0], dtype=np.float32))
    records = inventory_source(source)
    summary = summarize_inventory(
        records,
        source_root=source,
        documented_acquisition_groups=["session-a", "session-b"],
    )
    output = tmp_path / "audit"
    write_inventory(output, records, summary)

    assert summary["split_readiness"]["status"] == "pass"
    with (output / "source_inventory.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["npy_shape"] == "2"
    assert rows[0]["filename_series"] == "signal"
    assert json.loads((output / "source_inventory_summary.json").read_text())["n_files"] == 1


def test_verified_import_copies_only_audited_files(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    np.save(source / "signal_0.npy", np.asarray([1.0, 2.0], dtype=np.float32))
    records = inventory_source(source)
    audit = tmp_path / "audit"
    summary = summarize_inventory(records, source_root=source)
    write_inventory(audit, records, summary)

    destination = tmp_path / "datasets" / "raw" / "yeast" / "v1"
    provenance = import_verified_source(
        source_root=source,
        destination=destination,
        inventory_csv=audit / "source_inventory.csv",
        command="test import",
    )

    assert np.load(destination / "signal_0.npy").tolist() == [1.0, 2.0]
    assert provenance["n_copied_files"] == 1
    assert json.loads((destination / "import_provenance.json").read_text())["verification"] == (
        "size_and_sha256_before_and_after_copy"
    )


def test_verified_import_rejects_source_changed_after_audit(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    signal = source / "signal_0.npy"
    np.save(signal, np.asarray([1.0], dtype=np.float32))
    records = inventory_source(source)
    audit = tmp_path / "audit"
    write_inventory(audit, records, summarize_inventory(records, source_root=source))
    np.save(signal, np.asarray([2.0], dtype=np.float32))

    destination = tmp_path / "destination"
    try:
        import_verified_source(
            source_root=source,
            destination=destination,
            inventory_csv=audit / "source_inventory.csv",
            command="test import",
        )
    except ValueError as exc:
        assert "changed since audit" in str(exc)
    else:
        raise AssertionError("Changed source should fail verified import")
    assert not destination.exists()
