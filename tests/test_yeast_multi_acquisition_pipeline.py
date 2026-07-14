from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from particles2snr.yeast_event_audit import build_candidate_audit
from particles2snr.yeast_events import YeastDetectionConfig
from particles2snr.yeast_representation_dataset import build_representation_dataset
from particles2snr.yeast_source_index import (
    build_source_index_from_manifest,
    summarize_source_index,
    write_source_index,
)


def _event(phase: float) -> np.ndarray:
    rng = np.random.default_rng(7)
    index = np.arange(16384, dtype=np.float32)
    time = index / 2_000_000.0
    envelope = np.exp(-0.5 * np.square((index - 8192) / 420.0))
    return (
        envelope
        * (
            np.sin(2.0 * np.pi * 22_000.0 * time + phase)
            + 0.75 * np.sin(2.0 * np.pi * 34_000.0 * time + 0.45 + phase)
        )
        + 0.015 * rng.normal(size=index.size)
    ).astype(np.float32)


def _write_inventory(path: Path, signal_path: Path, digest: str) -> None:
    row = {
        "relative_path": "budding/event_0.npy",
        "source_group": "budding",
        "suffix": ".npy",
        "size_bytes": str(signal_path.stat().st_size),
        "sha256": digest,
        "filename_series": "event",
        "filename_index": "0",
        "npy_shape": "16384",
        "npy_dtype": "float32",
        "n_values": "16384",
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def test_multi_acquisition_pipeline_preserves_sealed_split(tmp_path: Path) -> None:
    roots: dict[str, Path] = {}
    acquisitions = []
    for suffix, (phase, role) in enumerate(
        ((0.0, "development"), (0.2, "sealed_ood_test"))
    ):
        dataset_id = f"raw-{suffix}@v1"
        root = tmp_path / f"raw-{suffix}"
        signal_path = root / "budding" / "event_0.npy"
        signal_path.parent.mkdir(parents=True)
        np.save(signal_path, _event(phase))
        digest = hashlib.sha256(signal_path.read_bytes()).hexdigest()
        inventory = tmp_path / f"inventory-{suffix}.csv"
        _write_inventory(inventory, signal_path, digest)
        roots[dataset_id] = root
        acquisitions.append(
            {
                "acquisition_id": f"session-{suffix}",
                "raw_dataset": dataset_id,
                "source_inventory": inventory.name,
                "role": role,
            }
        )

    manifest = tmp_path / "acquisitions.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "acquisitions": acquisitions}),
        encoding="utf-8",
    )
    index_rows = build_source_index_from_manifest(manifest)
    index_dir = tmp_path / "index"
    write_source_index(index_dir, index_rows, summarize_source_index(index_rows))

    candidates_dir = tmp_path / "candidates"
    build_candidate_audit(
        source_index_csv=index_dir / "source_index.csv",
        raw_dataset_root=None,
        raw_dataset_roots=roots,
        output_dir=candidates_dir,
        config=YeastDetectionConfig(
            active_snr_z=2.5,
            strict_min_snr=3.0,
            medium_min_snr=2.0,
            strict_min_concentration=0.08,
            medium_min_concentration=0.04,
            min_width_ms=0.05,
            max_width_ms=2.0,
        ),
        review_per_stratum=1,
    )
    representation_dir = tmp_path / "representation"
    summary = build_representation_dataset(
        candidate_csv=candidates_dir / "candidate_events.csv",
        raw_dataset_root=None,
        raw_dataset_roots=roots,
        output_dir=representation_dir,
        raw_dataset_id="multi-acquisition-manifest@v1",
        candidate_dataset_id="candidates@v1",
    )

    assert summary["split_counts"] == {
        "development_train": 1,
        "sealed_acquisition_test": 1,
    }
    with (representation_dir / "events.csv").open(newline="", encoding="utf-8") as handle:
        events = list(csv.DictReader(handle))
    assert {row["raw_dataset"] for row in events} == set(roots)
    assert {row["development_split"] for row in events} == {
        "development_train",
        "sealed_acquisition_test",
    }
    assert {row["acquisition_role"] for row in events} == {
        "development",
        "sealed_ood_test",
    }
