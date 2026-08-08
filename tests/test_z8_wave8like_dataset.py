from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from particles2snr.z8_wave8like_dataset import (
    EventRef,
    SourceRef,
    Z8Wave8LikeConfig,
    apply_raised_cosine_bridge,
    generate_dataset,
    match_bridge_to_local_rms,
    source_endpoint_quality,
)


def _write_fixture(root: Path) -> tuple[Path, Path, Path]:
    z8 = root / "z8"
    parent = root / "parent"
    noise = root / "noise"
    z8.mkdir()
    noise.mkdir()
    rows = []
    for split_index, split in enumerate(("train", "val")):
        signal_dir = parent / split / "signals"
        signal_dir.mkdir(parents=True)
        definitions = [
            ("p0a", 0, "2um", 20.0, 28.0),
            ("p0b", 0, "2um", 30.0, 38.0),
            ("p1a", 1, "4um", 22.0, 30.0),
            ("p1b", 1, "4um", 32.0, 40.0),
            ("p2a", 2, "10um", 24.0, 32.0),
            ("p2b", 2, "10um", 34.0, 42.0),
            ("bg0", None, None, None, None),
            ("bg1", None, None, None, None),
            ("bg2", None, None, None, None),
            ("bg3", None, None, None, None),
        ]
        for source_index, (name, class_id, class_name, left, right) in enumerate(
            definitions
        ):
            source_name = f"{name}_{split}"
            values = np.random.default_rng(
                split_index * 100 + source_index
            ).normal(size=64)
            np.save(signal_dir / f"{source_name}.npy", values.astype(np.float64))
            if class_id is not None:
                rows.append(
                    {
                        "event_id": f"event-{split}-{name}",
                        "split": split,
                        "source_filename": f"{source_name}.npy",
                        "class_id": class_id,
                        "class_name": class_name,
                        "start_sample": left,
                        "end_sample": right,
                    }
                )
    with (z8 / "events.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "event_id",
                "split",
                "source_filename",
                "class_id",
                "class_name",
                "start_sample",
                "end_sample",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    for index in range(3):
        values = np.random.default_rng(1000 + index).normal(size=64)
        np.save(noise / f"noise-{index}.npy", values.astype(np.float64))
    return z8, parent, noise


def _config() -> Z8Wave8LikeConfig:
    return Z8Wave8LikeConfig(
        output_dataset_id="fixture@v1",
        z8_dataset_id="z8@v1",
        parent_dataset_id="parent@v1",
        noise_dataset_id="noise@v1",
        segment_length=64,
        guard_samples=4,
        sampling_frequency_hz=2_000,
        bandpass_low_hz=20,
        bandpass_high_hz=400,
        bandpass_order=2,
        train_positive_groups=1,
        val_positive_groups=1,
        train_background_groups=1,
        val_background_groups=1,
        train_background_permutations=2,
        val_background_permutations=1,
        generator_revision="fixture",
    )


def test_raised_cosine_bridge_uses_exact_bridge_endpoints() -> None:
    left = np.full(16, 2.0)
    right = np.full(16, -2.0)
    bridge = np.linspace(10.0, 17.0, 8)
    joined_left, joined_right = apply_raised_cosine_bridge(
        left, right, bridge, 4
    )
    assert joined_left[-4] == pytest.approx(2.0)
    assert joined_left[-1] == pytest.approx(bridge[3])
    assert joined_right[0] == pytest.approx(bridge[4])
    assert joined_right[3] == pytest.approx(-2.0)


def test_bridge_matching_tracks_annotation_free_endpoint_rms() -> None:
    left = np.tile(np.array([-2.0, 2.0]), 8)
    right = np.tile(np.array([-4.0, 4.0]), 8)
    bridge = np.tile(np.array([-10.0, 10.0]), 4)

    matched, record = match_bridge_to_local_rms(
        left,
        right,
        bridge,
        left_events=(
            EventRef("excluded-left", 0, "2um", 0.0, 4.0),
        ),
        right_events=(
            EventRef("excluded-right", 1, "4um", 12.0, 16.0),
        ),
        guard_samples=4,
        context_samples=8,
    )

    assert record["left_target_rms"] == pytest.approx(2.0)
    assert record["right_target_rms"] == pytest.approx(4.0)
    assert record["raw_bridge_rms"] == pytest.approx(10.0)
    assert abs(matched[0]) == pytest.approx(2.0)
    assert abs(matched[-1]) == pytest.approx(4.0)


def test_global_baseline_caps_a_transient_contaminated_endpoint() -> None:
    left = np.tile(np.array([-2.0, 2.0]), 32)
    left[-12:-4] = np.tile(np.array([-20.0, 20.0]), 4)
    right = np.tile(np.array([-4.0, 4.0]), 32)
    bridge = np.tile(np.array([-10.0, 10.0]), 4)

    matched, record = match_bridge_to_local_rms(
        left,
        right,
        bridge,
        left_events=(),
        right_events=(),
        guard_samples=4,
        context_samples=8,
        cap_by_global=True,
    )

    assert record["matching"] == "robust-local-rms-global-cap"
    assert record["left_local_rms"] == pytest.approx(20.0)
    assert record["left_target_rms"] == record["left_global_rms"]
    assert float(record["left_target_rms"]) < 3.0
    assert abs(matched[0]) == pytest.approx(float(record["left_target_rms"]))


def test_endpoint_quality_rejects_an_unannotated_edge_transient(
    tmp_path: Path,
) -> None:
    path = tmp_path / "source.npy"
    values = np.tile(np.array([-2.0, 2.0]), 32)
    values[:8] = np.tile(np.array([-20.0, 20.0]), 4)
    np.save(path, values.astype(np.float64))
    sample = SourceRef(
        split="train",
        source_id="source",
        signal_path=path,
        relative_signal_path="train/signals/source.npy",
        signal_sha256="fixture",
        events=(),
    )
    config = replace(
        _config(),
        endpoint_quality_enabled=True,
        endpoint_quality_window_samples=16,
        endpoint_max_rms_ratio=2.5,
        endpoint_max_peak_robust_z=8.0,
    )

    quality = source_endpoint_quality(sample, config)

    assert quality["left"]["safe"] is False
    assert quality["right"]["safe"] is True
    assert quality["safe"] is False


def test_generation_is_atomic_deterministic_and_has_no_test_split(
    tmp_path: Path,
) -> None:
    z8, parent, noise = _write_fixture(tmp_path)
    output_a = tmp_path / "candidate-a"
    output_b = tmp_path / "candidate-b"
    manifest_a = generate_dataset(
        z8_root=z8,
        parent_root=parent,
        noise_root=noise,
        output_root=output_a,
        config=_config(),
        verify_replay=True,
    )
    manifest_b = generate_dataset(
        z8_root=z8,
        parent_root=parent,
        noise_root=noise,
        output_root=output_b,
        config=_config(),
        verify_replay=True,
    )
    assert manifest_a["audit"] == manifest_b["audit"]
    assert (output_a / "manifest.csv").read_bytes() == (
        output_b / "manifest.csv"
    ).read_bytes()
    assert not (output_a / "test").exists()
    rows = list(csv.DictReader((output_a / "manifest.csv").open()))
    assert len(rows) == 24 + 2 + 24 + 1
    assert {row["stratum"] for row in rows} == {"positive", "background"}
    assert manifest_a["deterministic_replay"]["status"] == "pass"
    assert json.loads((output_a / "dataset-contract.json").read_text())[
        "sealed_test_accessed"
    ] is False


def test_injected_failure_never_promotes_partial_candidate(tmp_path: Path) -> None:
    z8, parent, noise = _write_fixture(tmp_path)
    output = tmp_path / "candidate"
    with pytest.raises(RuntimeError, match="injected generation failure"):
        generate_dataset(
            z8_root=z8,
            parent_root=parent,
            noise_root=noise,
            output_root=output,
            config=_config(),
            verify_replay=False,
            fail_after_rows=3,
        )
    assert not output.exists()
