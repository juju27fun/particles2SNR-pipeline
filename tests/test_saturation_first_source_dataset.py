from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from particles2snr.saturation_cleaning import (
    butter_bandpass_filter,
    repair_saturation_intervals_pre_filter,
)
from particles2snr.saturation_first_source_dataset import (
    _replacement,
    build_saturation_first_source_dataset,
    sha256_file,
)


def test_replacement_replays_frozen_carrier_slice(tmp_path: Path) -> None:
    carrier = np.arange(32, dtype=np.float64)
    carrier_path = tmp_path / "artifacts/carrier.npy"
    carrier_path.parent.mkdir(parents=True)
    np.save(carrier_path, carrier)
    expected = carrier[8:20]
    row = {
        "filename": "trace.npy",
        "core_start_sample": "10",
        "core_end_sample": "18",
        "expanded_start_sample": "8",
        "expanded_end_sample": "20",
        "historical_carrier_path": "artifacts/carrier.npy",
        "historical_carrier_slice_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
    }
    replacement = _replacement(workspace_root=tmp_path, row=row)
    assert replacement["core_interval"] == [10, 18]
    assert replacement["expanded_interval"] == [8, 20]
    np.testing.assert_array_equal(replacement["replacement"], expected)


def test_replacement_rejects_carrier_drift(tmp_path: Path) -> None:
    carrier_path = tmp_path / "carrier.npy"
    np.save(carrier_path, np.arange(16, dtype=np.float64))
    row = {
        "filename": "trace.npy",
        "core_start_sample": "3",
        "core_end_sample": "8",
        "expanded_start_sample": "2",
        "expanded_end_sample": "9",
        "historical_carrier_path": "carrier.npy",
        "historical_carrier_slice_sha256": "0" * 64,
    }
    try:
        _replacement(workspace_root=tmp_path, row=row)
    except ValueError as exc:
        assert "carrier slice drift" in str(exc)
    else:
        raise AssertionError("carrier drift was accepted")


def test_full_source_builder_copies_canonical_development_and_computes_test(
    tmp_path: Path,
) -> None:
    predecessor = tmp_path / "datasets/predecessor"
    raw_root = tmp_path / "datasets/raw/10um"
    canonical = tmp_path / "datasets/canonical"
    carrier_path = tmp_path / "artifacts/carrier.npy"
    for path in (predecessor, raw_root, canonical / "train/signals", carrier_path.parent):
        path.mkdir(parents=True, exist_ok=True)
    x = np.arange(16_384, dtype=np.float64)
    raw_dev = np.sin(2 * np.pi * 20_000 * x / 2_000_000)
    raw_test = raw_dev.copy()
    raw_test[6_000:7_000] = raw_test[6_000]
    np.save(raw_root / "dev.npy", raw_dev)
    np.save(raw_root / "test.npy", raw_test)
    carrier = 0.01 * np.cos(2 * np.pi * 18_000 * x / 2_000_000)
    np.save(carrier_path, carrier)
    canonical_signal = butter_bandpass_filter(raw_dev)
    np.save(canonical / "train/signals/dev.npy", canonical_signal)
    source_manifest = predecessor / "source_manifest.csv"
    source_manifest.write_text(
        "source_id,output_stem,source_path,source_sha256,source_class,class_id,source_split,output_split\n"
        f"dev,10um_dev,train/10um/dev.npy,{'1' * 64},10um,2,train,train\n"
        f"test,10um_test,test/10um/test.npy,{'2' * 64},10um,2,test,test\n",
        encoding="utf-8",
    )
    start, end = 5_800, 7_200
    replacement_sha = hashlib.sha256(carrier[start:end].tobytes()).hexdigest()
    repair_manifest = tmp_path / "datasets/repairs.csv"
    repair_manifest.write_text(
        "filename,class,interval_idx,core_start_sample,core_end_sample,expanded_start_sample,expanded_end_sample,raw_sha256,historical_carrier_path,historical_carrier_slice_sha256\n"
        f"test.npy,10um,0,6000,7000,{start},{end},{sha256_file(raw_root / 'test.npy')},artifacts/carrier.npy,{replacement_sha}\n",
        encoding="utf-8",
    )
    output = tmp_path / "datasets/output"
    manifest = build_saturation_first_source_dataset(
        workspace_root=tmp_path,
        predecessor_root=predecessor,
        frozen_repair_manifest=repair_manifest,
        raw_dataset_roots={"10um": raw_root},
        raw_dataset_ids={"10um": "raw-10um@v1"},
        raw_manifest_sha256s={"10um": "a" * 64},
        output_dir=output,
        dataset_id="corrected@v1",
        predecessor_dataset_id="predecessor@v1",
        predecessor_manifest_sha256="b" * 64,
        repair_reference_dataset_id="repairs@v1",
        repair_reference_manifest_sha256="c" * 64,
        canonical_development_root=canonical,
        canonical_development_dataset_id="canonical@v1",
        canonical_development_manifest_sha256="d" * 64,
        expected_traces=2,
    )
    assert manifest["counts"]["traces_total"] == 2
    assert manifest["counts"]["repaired_traces"] == 1
    assert sha256_file(output / "train/10um/dev.npy") == sha256_file(
        canonical / "train/signals/dev.npy"
    )
    expected_test = repair_saturation_intervals_pre_filter(
        raw_test,
        [
            {
                "core_interval": [6_000, 7_000],
                "expanded_interval": [start, end],
                "replacement": carrier[start:end],
            }
        ],
    )["filtered_signal"]
    np.testing.assert_array_equal(np.load(output / "test/10um/test.npy"), expected_test)
    assert json.loads((output / "dataset-manifest.json").read_text())["method"][
        "repair_before_bandpass"
    ] is True
