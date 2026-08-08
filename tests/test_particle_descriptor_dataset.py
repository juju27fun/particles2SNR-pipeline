from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from particles2snr.particle_descriptor_dataset import (
    build_particle_descriptor_dataset,
    validate_particle_descriptor_dataset,
)


def _write_source(root: Path) -> None:
    for split in ("train", "val", "test"):
        (root / split / "signals").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
    signal = np.sin(np.linspace(0.0, 300.0, 16384)).astype(np.float32)
    np.save(root / "train/signals/sample.npy", signal)
    (root / "train/labels/sample.txt").write_text(
        "0 0.5 0.1\n1 0.1 0.05\n3 0.5 0.1\n",
        encoding="utf-8",
    )
    np.save(root / "val/signals/sample-val.npy", signal)
    (root / "val/labels/sample-val.txt").write_text("2 0.5 0.1\n", encoding="utf-8")
    np.save(root / "test/signals/never-read.npy", signal)
    (root / "test/labels/never-read.txt").write_text("0 0.5 0.1\n", encoding="utf-8")


def test_builds_only_complete_known_train_and_val_events(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_source(source)

    summary = build_particle_descriptor_dataset(
        source_root=source,
        output_dir=output,
        source_dataset_id="particles@v1",
        source_manifest_sha256="a" * 64,
        population_id="strict",
    )

    assert summary["included_events"] == 2
    assert summary["exclusion_counts"] == {
        "class_unclear_or_unknown": 1,
        "incomplete_centered_crop": 1,
    }
    assert summary["test_split_accessed"] is False
    assert validate_particle_descriptor_dataset(output)["valid"] is True
    contract = json.loads((output / "input_contract.json").read_text())
    assert contract["crop_policy"].startswith("exact centered crop")


def test_shift_window_keeps_known_boundary_events_without_opening_test(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_source(source)
    with (source / "train/labels/sample.txt").open("a", encoding="utf-8") as handle:
        handle.write("0 0.01 0.05\n")

    summary = build_particle_descriptor_dataset(
        source_root=source,
        output_dir=output,
        source_dataset_id="particles@v2",
        source_manifest_sha256="b" * 64,
        population_id="reviewed-all-known-development",
        crop_policy="shift-window",
    )

    assert summary["included_events"] == 4
    assert summary["included_counts"] == {
        "train:2um": 2,
        "train:4um": 1,
        "val:10um": 1,
    }
    assert summary["exclusion_counts"] == {"class_unclear_or_unknown": 1}
    assert summary["annotations_clipped_to_source"] == 1
    assert summary["crop_policy"] == "shift-window"
    assert summary["test_split_accessed"] is False
    contract = json.loads((output / "input_contract.json").read_text())
    assert contract["crop_policy"].startswith("fixed-length crop shifted")
    assert validate_particle_descriptor_dataset(output)["valid"] is True
