from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from particles2snr.yeast_representation_dataset import (
    build_representation_dataset,
    clamped_crop,
    preprocess_crop,
)


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
