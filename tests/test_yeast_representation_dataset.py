from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from particles2snr.yeast_representation_dataset import (
    build_representation_dataset,
    clamped_crop,
    preprocess_crop,
)
from particles2snr.yeast_raw_data import read_raw_dataset_map, resolve_raw_signal


def test_clamped_crop_avoids_padding() -> None:
    signal = np.arange(10, dtype=np.float32)
    left, left_start = clamped_crop(signal, center_index=1, length=6)
    right, right_start = clamped_crop(signal, center_index=9, length=6)
    assert left_start == 0
    assert left.tolist() == [0, 1, 2, 3, 4, 5]
    assert right_start == 4
    assert right.tolist() == [4, 5, 6, 7, 8, 9]


def test_preprocess_crop_downsamples_with_finite_output() -> None:
    time = np.arange(8192) / 2_000_000.0
    signal = np.sin(2.0 * np.pi * 20_000.0 * time).astype(np.float32)
    output = preprocess_crop(signal)
    assert output.shape == (4096,)
    assert np.isfinite(output).all()
    assert float(np.std(output)) > 0.1


def test_representation_dataset_uses_train_only_global_normalization(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    (raw / "budding").mkdir(parents=True)
    time = np.arange(16384) / 2_000_000.0
    np.save(raw / "budding" / "train.npy", np.sin(2 * np.pi * 20_000 * time))
    np.save(raw / "budding" / "validation.npy", 2 * np.sin(2 * np.pi * 20_000 * time))
    candidates = tmp_path / "candidates.csv"
    rows = []
    for index, (name, split) in enumerate(
        (("train.npy", "development_train"), ("validation.npy", "development_validation"))
    ):
        rows.append(
            {
                "event_id": f"event-{index}",
                "record_id": f"record-{index}",
                "relative_path": f"budding/{name}",
                "source_group": "budding",
                "condition_id": "exponential-budding",
                "development_split": split,
                "quality": "strict",
                "center_index": 8192,
                "event_start": 7600,
                "event_end": 8800,
            }
        )
    with candidates.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output = tmp_path / "output"
    summary = build_representation_dataset(
        candidate_csv=candidates,
        raw_dataset_root=raw,
        output_dir=output,
        raw_dataset_id="raw@v1",
        candidate_dataset_id="candidates@v1",
    )
    signals = np.load(output / "signals.npy", mmap_mode="r")
    assert signals.shape == (2, 4096)
    assert abs(float(np.mean(signals[0]))) < 1.0e-4
    assert 0.99 < float(np.std(signals[0])) < 1.01
    assert float(np.std(signals[1])) > 1.9
    assert summary["split_counts"] == {"development_train": 1, "development_validation": 1}
    assert json.loads((output / "input_contract.json").read_text())["information_policy"][
        "in_band_amplitude"
    ].startswith("unresolved-preserve")


def test_representation_dataset_keeps_sealed_acquisition_out_of_normalization(
    tmp_path: Path,
) -> None:
    roots = {"raw-a@v1": tmp_path / "raw-a", "raw-b@v1": tmp_path / "raw-b"}
    time = np.arange(16384) / 2_000_000.0
    for amplitude, root in zip((1.0, 3.0), roots.values()):
        (root / "budding").mkdir(parents=True)
        np.save(
            root / "budding" / "same.npy",
            amplitude * np.sin(2 * np.pi * 20_000 * time),
        )
    candidates = tmp_path / "candidates.csv"
    rows = []
    for index, dataset_id in enumerate(roots):
        rows.append(
            {
                "event_id": f"event-{index}",
                "record_id": f"record-{index}",
                "raw_dataset": dataset_id,
                "relative_path": "budding/same.npy",
                "source_group": "budding",
                "condition_id": "exponential-budding",
                "acquisition_id": f"session-{index}",
                "development_split": (
                    "development_train" if index == 0 else "sealed_acquisition_test"
                ),
                "quality": "strict",
                "center_index": 8192,
                "event_start": 7600,
                "event_end": 8800,
            }
        )
    with candidates.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    output = tmp_path / "output"
    summary = build_representation_dataset(
        candidate_csv=candidates,
        raw_dataset_root=None,
        raw_dataset_roots=roots,
        output_dir=output,
        raw_dataset_id="multi-acquisition-manifest@v1",
        candidate_dataset_id="candidates@v1",
    )
    signals = np.load(output / "signals.npy", mmap_mode="r")
    assert 0.99 < float(np.std(signals[0])) < 1.01
    assert 2.99 < float(np.std(signals[1])) < 3.01
    assert summary["split_counts"] == {
        "development_train": 1,
        "sealed_acquisition_test": 1,
    }
    contract = json.loads((output / "input_contract.json").read_text())
    assert contract["raw_datasets"] == ["raw-a@v1", "raw-b@v1"]
    assert contract["split_scope"].startswith("sealed acquisition OOD available")


def test_raw_dataset_map_resolves_paths_relative_to_manifest(tmp_path: Path) -> None:
    (tmp_path / "raw-a").mkdir()
    manifest = tmp_path / "raw-map.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "raw_datasets": {"raw-a@v1": "raw-a"},
            }
        ),
        encoding="utf-8",
    )
    assert read_raw_dataset_map(manifest) == {"raw-a@v1": (tmp_path / "raw-a").resolve()}


def test_raw_signal_cannot_escape_registered_root(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    root.mkdir()
    with pytest.raises(ValueError, match="escapes"):
        resolve_raw_signal(
            {"relative_path": "../outside.npy"},
            single_root=root,
            roots_by_dataset={},
        )
