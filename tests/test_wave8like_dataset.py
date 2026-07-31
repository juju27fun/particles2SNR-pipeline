from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from particles2snr.wave8like_dataset import (
    Event,
    GenerationConfig,
    SourceSample,
    _background_group_count,
    apply_edge_guard,
    audit_source_split_isolation,
    draw_balanced_group,
    eligible_positive_events,
    generate_dataset,
    load_source_dataset,
    parse_yolo_labels,
)


def _write_source_dataset(root: Path, *, duplicate_across_splits: bool = False) -> None:
    length = 256
    definitions = [
        ("p0a", 0),
        ("p0b", 0),
        ("p1", 1),
        ("p2", 2),
        ("p3", 3),
        ("bg0", None),
        ("bg1", None),
        ("bg2", None),
        ("bg3", None),
    ]
    for split_index, split in enumerate(("train", "val", "test")):
        signal_dir = root / split / "signals"
        label_dir = root / split / "labels"
        signal_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        for sample_index, (name, class_id) in enumerate(definitions):
            rng_seed = sample_index if duplicate_across_splits else 1000 * split_index + sample_index
            signal = np.random.default_rng(rng_seed).normal(size=length).astype(np.float64)
            np.save(signal_dir / f"{name}_{split}.npy", signal)
            label = label_dir / f"{name}_{split}.txt"
            if class_id is None:
                label.write_text("")
            else:
                label.write_text(f"{class_id} 0.500000 0.125000\n")


def _write_noise_dataset(root: Path) -> None:
    root.mkdir(parents=True)
    for index in range(2):
        signal = np.random.default_rng(9000 + index).normal(size=256).astype(np.float64)
        np.save(root / f"noise_{index}.npy", signal)


def _config(mode: str, **overrides) -> GenerationConfig:
    values = {
        "mode": mode,
        "source_dataset_id": "synthetic-source@v1",
        "noise_dataset_id": "synthetic-noise@v1",
        "seed": 17,
        "segment_length": 256,
        "segments_per_sequence": 4,
        "noise_pad": 8,
        "join_crossfade": 8,
        "sampling_frequency_hz": 2_000,
        "bandpass_low_hz": 10.0,
        "bandpass_high_hz": 500.0,
        "bandpass_order": 2,
        "train_groups": 1,
        "val_groups": 1,
        "test_groups": 1,
        "positive_permutations": 2,
        "background_share": 0.0,
        "background_permutations": 1,
        "generator_revision": "test",
    }
    values.update(overrides)
    return GenerationConfig(**values)


def test_parse_labels_rejects_out_of_bounds_interval(tmp_path: Path) -> None:
    label = tmp_path / "bad.txt"
    label.write_text("0 0.01 0.10\n")
    with pytest.raises(ValueError, match="crosses signal boundary"):
        parse_yolo_labels(label, 256)


def test_edge_guard_drops_events_touching_modified_region() -> None:
    signal = np.arange(64, dtype=np.float64)
    events = (Event(0, 2, 12), Event(1, 16, 24), Event(2, 54, 62))
    noise = [np.zeros(8, dtype=np.float64), np.ones(8, dtype=np.float64)]
    guarded, retained, dropped = apply_edge_guard(
        signal, events, noise, 8, np.random.default_rng(1)
    )
    assert retained == (Event(1, 16, 24),)
    assert dropped == 2
    assert not np.array_equal(guarded, signal)


def test_strict_source_policy_rejects_omitted_or_edge_events() -> None:
    sample = SourceSample(
        split="test",
        source_id="mixed",
        signal_path=Path("mixed.npy"),
        relative_signal_path="test/signals/mixed.npy",
        signal_sha256="a" * 64,
        events=(Event(0, 16, 24), Event(3, 32, 40)),
        signal_length=64,
    )
    assert eligible_positive_events(
        sample, (0, 1, 2), 8, "fully_labeled_for_view"
    ) == ()
    edge_sample = SourceSample(
        split="test",
        source_id="edge",
        signal_path=Path("edge.npy"),
        relative_signal_path="test/signals/edge.npy",
        signal_sha256="b" * 64,
        events=(Event(0, 4, 16), Event(1, 24, 32)),
        signal_length=64,
    )
    assert eligible_positive_events(
        edge_sample, (0, 1, 2), 8, "fully_labeled_for_view"
    ) == ()


def test_balanced_group_uses_label_membership_not_filename(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_source_dataset(source)
    pools = load_source_dataset(source, 256)
    exposure = Counter({0: 0, 1: 0, 2: 0})
    chosen = draw_balanced_group(
        pools["train"], (0, 1, 2), 4, np.random.default_rng(3), exposure
    )
    chosen_classes = {
        event.class_id
        for index in chosen
        for event in pools["train"][index].events
        if event.class_id < 3
    }
    assert chosen_classes == {0, 1, 2}
    assert len(chosen) == len(set(chosen)) == 4


def test_balanced_group_ignores_class_event_removed_by_edge_guard(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_source_dataset(source)
    # Move the only train unclear event into the edge guard. The class must be
    # considered unavailable rather than selected and silently removed later.
    label = source / "train" / "labels" / "p3_train.txt"
    label.write_text("3 0.02 0.02\n")
    pools = load_source_dataset(source, 256)
    with pytest.raises(RuntimeError, match=r"no positive source for class ids \[3\]"):
        draw_balanced_group(
            pools["train"],
            (0, 1, 2, 3),
            4,
            np.random.default_rng(3),
            Counter({0: 0, 1: 0, 2: 0, 3: 0}),
            edge_pad=8,
        )


def test_known3_generation_is_deterministic_and_omits_unclear(tmp_path: Path) -> None:
    source = tmp_path / "source"
    noise = tmp_path / "noise"
    _write_source_dataset(source)
    _write_noise_dataset(noise)
    config = _config("known3-positive")
    output_a = tmp_path / "out-a"
    output_b = tmp_path / "out-b"
    metadata_a = generate_dataset(source, noise, output_a, config)
    metadata_b = generate_dataset(source, noise, output_b, config)

    assert metadata_a["manifest_sha256"] == metadata_b["manifest_sha256"]
    assert metadata_a["nc"] == 3
    assert metadata_a["names"] == ["2um", "4um", "10um"]
    assert (output_a / "manifest.csv").read_bytes() == (output_b / "manifest.csv").read_bytes()
    for label_path in output_a.glob("*/labels/*.txt"):
        assert all(not line.startswith("3 ") for line in label_path.read_text().splitlines())
    first_a = next((output_a / "train" / "signals").glob("*.npy"))
    first_b = output_b / "train" / "signals" / first_a.name
    assert np.array_equal(np.load(first_a), np.load(first_b))


def test_fourclass_background_generation_preserves_strata(tmp_path: Path) -> None:
    source = tmp_path / "source"
    noise = tmp_path / "noise"
    output = tmp_path / "out"
    _write_source_dataset(source)
    _write_noise_dataset(noise)
    config = _config(
        "fourclass-background",
        background_share=1 / 3,
        background_permutations=1,
    )
    metadata = generate_dataset(source, noise, output, config)

    assert metadata["nc"] == 4
    assert metadata["splits"]["train"]["positive_rows"] == 2
    assert metadata["splits"]["train"]["background_rows"] == 1
    rows = list(csv.DictReader((output / "manifest.csv").open()))
    assert {row["stratum"] for row in rows} == {"positive", "background"}
    assert any(
        line.startswith("3 ")
        for path in output.glob("*/labels/*.txt")
        for line in path.read_text().splitlines()
    )
    for row in rows:
        if row["stratum"] == "background":
            label = output / row["split"] / "labels" / f"{row['long_id']}.txt"
            assert label.read_text() == ""
    parsed_metadata = json.loads((output / "dataset.yaml").read_text())
    assert parsed_metadata["audit_results"]["generated_dataset"] == "pass"


def test_train_only_50_percent_iteration_keeps_evaluation_mixture_fixed() -> None:
    config = _config(
        "fourclass-background",
        positive_permutations=24,
        background_share=0.50,
        background_permutations=4,
        train_background_permutations=12,
        evaluation_background_share=0.25,
    )
    assert _background_group_count(config, 100, split="train") == 200
    assert _background_group_count(config, 30, split="val") == 60
    assert config.background_permutations_for_split("train") == 12
    assert config.background_permutations_for_split("test") == 4


def test_split_specific_background_contract_is_generated(tmp_path: Path) -> None:
    source = tmp_path / "source"
    noise = tmp_path / "noise"
    output = tmp_path / "out"
    _write_source_dataset(source)
    _write_noise_dataset(noise)
    config = _config(
        "fourclass-background",
        positive_permutations=2,
        background_share=0.50,
        background_permutations=1,
        train_background_permutations=2,
        evaluation_background_share=1 / 3,
    )
    metadata = generate_dataset(source, noise, output, config)
    assert metadata["splits"]["train"]["positive_rows"] == 2
    assert metadata["splits"]["train"]["background_rows"] == 2
    for split in ("val", "test"):
        assert metadata["splits"][split]["positive_rows"] == 2
        assert metadata["splits"][split]["background_rows"] == 1
    params = metadata["generation_params"]
    assert params["background_share_by_split"] == {
        "train": 0.5,
        "val": 1 / 3,
        "test": 1 / 3,
    }
    assert params["background_permutations_by_split"] == {
        "train": 2,
        "val": 1,
        "test": 1,
    }


def test_source_content_leakage_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_source_dataset(source, duplicate_across_splits=True)
    pools = {
        split: [] for split in ("train", "val", "test")
    }
    # Loading the complete source performs the same content-level audit used by
    # production generation.
    with pytest.raises(ValueError, match="content crosses splits"):
        loaded = load_source_dataset(source, 256)
        audit_source_split_isolation(loaded)


def test_generation_refuses_to_mutate_existing_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    noise = tmp_path / "noise"
    output = tmp_path / "out"
    _write_source_dataset(source)
    _write_noise_dataset(noise)
    output.mkdir()
    with pytest.raises(FileExistsError, match="refusing to mutate"):
        generate_dataset(source, noise, output, _config("known3-positive"))
