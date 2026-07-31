"""Provenance-safe Wave8-like long-sequence detection datasets.

The module deliberately separates an event-rich, three-known-class capability
view from a four-class deployment view containing explicit background-only
composites.  It never infers a class from a filename: class membership comes
only from the registered YOLO labels.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.signal import butter, sosfiltfilt


SPLITS = ("train", "val", "test")
SOURCE_CLASS_NAMES = ("2um", "4um", "10um", "unclear")
MODES = ("known3-positive", "fourclass-background")
SOURCE_ELIGIBILITY_POLICIES = (
    "legacy_any_safe_target",
    "fully_labeled_for_view",
)


@dataclass(frozen=True)
class Event:
    class_id: int
    left: int
    right: int


@dataclass(frozen=True)
class SourceSample:
    split: str
    source_id: str
    signal_path: Path
    relative_signal_path: str
    signal_sha256: str
    events: tuple[Event, ...]
    signal_length: int

    @property
    def is_background(self) -> bool:
        return not self.events

    def event_counts(self, class_ids: Iterable[int] | None = None) -> Counter:
        allowed = None if class_ids is None else set(class_ids)
        return Counter(
            event.class_id
            for event in self.events
            if allowed is None or event.class_id in allowed
        )


@dataclass(frozen=True)
class GenerationConfig:
    mode: str
    source_dataset_id: str
    noise_dataset_id: str
    output_dataset_id: str = "unregistered@v0"
    seed: int = 42
    segment_length: int = 16_384
    segments_per_sequence: int = 4
    noise_pad: int = 300
    join_crossfade: int = 300
    sampling_frequency_hz: int = 2_000_000
    bandpass_low_hz: float = 8_000.0
    bandpass_high_hz: float = 500_000.0
    bandpass_order: int = 4
    train_groups: int = 100
    val_groups: int = 30
    test_groups: int = 30
    positive_permutations: int = 24
    background_share: float = 0.25
    background_permutations: int = 4
    train_background_permutations: int | None = None
    evaluation_background_share: float | None = None
    disjoint_background_groups: bool = True
    source_eligibility_policy: str = "fully_labeled_for_view"
    generator_revision: str = "unknown"

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.source_eligibility_policy not in SOURCE_ELIGIBILITY_POLICIES:
            raise ValueError(
                "source_eligibility_policy must be one of "
                f"{SOURCE_ELIGIBILITY_POLICIES}"
            )
        if self.segment_length <= 0 or self.segments_per_sequence <= 0:
            raise ValueError("segment lengths and counts must be positive")
        if self.noise_pad < 0:
            raise ValueError("noise_pad must be non-negative")
        if not 0 <= self.join_crossfade <= self.noise_pad:
            raise ValueError("join_crossfade must be in [0, noise_pad]")
        if 2 * self.noise_pad >= self.segment_length:
            raise ValueError("noise_pad leaves no label-safe segment interior")
        max_perms = math.factorial(self.segments_per_sequence)
        if not 1 <= self.positive_permutations <= max_perms:
            raise ValueError(f"positive_permutations must be in [1, {max_perms}]")
        if not 1 <= self.background_permutations <= max_perms:
            raise ValueError(f"background_permutations must be in [1, {max_perms}]")
        if self.train_background_permutations is not None and not (
            1 <= self.train_background_permutations <= max_perms
        ):
            raise ValueError(f"train_background_permutations must be in [1, {max_perms}]")
        if not 0.0 <= self.background_share < 1.0:
            raise ValueError("background_share must be in [0, 1)")
        if self.evaluation_background_share is not None and not (
            0.0 <= self.evaluation_background_share < 1.0
        ):
            raise ValueError("evaluation_background_share must be in [0, 1)")
        nyquist = self.sampling_frequency_hz / 2.0
        if not 0 < self.bandpass_low_hz < self.bandpass_high_hz < nyquist:
            raise ValueError("bandpass cutoffs must be ordered below Nyquist")
        if self.bandpass_order <= 0:
            raise ValueError("bandpass_order must be positive")
        if any(value < 0 for value in self.groups_by_split.values()):
            raise ValueError("group counts must be non-negative")

    @property
    def class_names(self) -> tuple[str, ...]:
        return SOURCE_CLASS_NAMES[:3] if self.mode == "known3-positive" else SOURCE_CLASS_NAMES

    @property
    def target_class_ids(self) -> tuple[int, ...]:
        return tuple(range(len(self.class_names)))

    @property
    def groups_by_split(self) -> dict[str, int]:
        return {
            "train": self.train_groups,
            "val": self.val_groups,
            "test": self.test_groups,
        }

    def background_share_for_split(self, split: str) -> float:
        if split == "train" or self.evaluation_background_share is None:
            return self.background_share
        return self.evaluation_background_share

    def background_permutations_for_split(self, split: str) -> int:
        if split == "train" and self.train_background_permutations is not None:
            return self.train_background_permutations
        return self.background_permutations

    @property
    def long_length(self) -> int:
        return self.segment_length * self.segments_per_sequence


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parse_yolo_labels(
    path: Path,
    signal_length: int,
    *,
    class_count: int = len(SOURCE_CLASS_NAMES),
) -> tuple[Event, ...]:
    """Parse and validate YOLO intervals into integer sample coordinates."""

    if not path.is_file():
        raise FileNotFoundError(f"missing label file: {path}")
    events: list[Event] = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f"{path}:{line_number}: expected 3 fields")
        try:
            class_id = int(fields[0])
            center = float(fields[1])
            width = float(fields[2])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: invalid numeric label") from exc
        if not 0 <= class_id < class_count:
            raise ValueError(f"{path}:{line_number}: invalid class id {class_id}")
        if not np.isfinite(center) or not np.isfinite(width):
            raise ValueError(f"{path}:{line_number}: non-finite geometry")
        if not 0.0 <= center <= 1.0 or not 0.0 < width <= 1.0:
            raise ValueError(f"{path}:{line_number}: geometry outside normalized range")
        left_f = (center - width / 2.0) * signal_length
        right_f = (center + width / 2.0) * signal_length
        tolerance = 1e-3
        if left_f < -tolerance or right_f > signal_length + tolerance:
            raise ValueError(f"{path}:{line_number}: interval crosses signal boundary")
        left = max(0, min(signal_length, int(round(left_f))))
        right = max(0, min(signal_length, int(round(right_f))))
        if right <= left:
            raise ValueError(f"{path}:{line_number}: empty interval after sample projection")
        events.append(Event(class_id=class_id, left=left, right=right))
    return tuple(events)


def load_source_split(root: Path, split: str, segment_length: int) -> list[SourceSample]:
    signal_dir = root / split / "signals"
    label_dir = root / split / "labels"
    if not signal_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"missing YOLO split directories under {root / split}")

    samples: list[SourceSample] = []
    signal_paths = sorted(signal_dir.glob("*.npy"))
    if not signal_paths:
        raise RuntimeError(f"no signals found in {signal_dir}")
    for signal_path in signal_paths:
        label_path = label_dir / f"{signal_path.stem}.txt"
        signal = np.load(signal_path, mmap_mode="r", allow_pickle=False)
        if signal.ndim != 1 or len(signal) != segment_length:
            raise ValueError(
                f"{signal_path}: expected shape ({segment_length},), got {signal.shape}"
            )
        if not np.isfinite(signal).all():
            raise ValueError(f"{signal_path}: signal contains non-finite values")
        samples.append(
            SourceSample(
                split=split,
                source_id=signal_path.stem,
                signal_path=signal_path,
                relative_signal_path=signal_path.relative_to(root).as_posix(),
                signal_sha256=sha256_file(signal_path),
                events=parse_yolo_labels(label_path, segment_length),
                signal_length=segment_length,
            )
        )
    return samples


def load_source_dataset(root: Path, segment_length: int) -> dict[str, list[SourceSample]]:
    pools = {split: load_source_split(root, split, segment_length) for split in SPLITS}
    audit_source_split_isolation(pools)
    return pools


def audit_source_split_isolation(pools: Mapping[str, Sequence[SourceSample]]) -> None:
    id_owner: dict[str, str] = {}
    hash_owner: dict[str, str] = {}
    for split in SPLITS:
        for sample in pools[split]:
            previous = id_owner.setdefault(sample.source_id, split)
            if previous != split:
                raise ValueError(
                    f"source id crosses splits: {sample.source_id} ({previous}, {split})"
                )
            previous_hash = hash_owner.setdefault(sample.signal_sha256, split)
            if previous_hash != split:
                raise ValueError(
                    "source signal content crosses splits: "
                    f"{sample.signal_sha256} ({previous_hash}, {split})"
                )


def load_noise_chunks(noise_root: Path, chunk_length: int) -> list[np.ndarray]:
    if chunk_length <= 0:
        return []
    chunks: list[np.ndarray] = []
    for path in sorted(noise_root.rglob("*.npy")):
        signal = np.load(path, allow_pickle=False)
        if signal.ndim != 1 or not np.isfinite(signal).all():
            raise ValueError(f"invalid noise signal: {path}")
        for start in range(0, len(signal) - chunk_length + 1, chunk_length):
            chunks.append(np.asarray(signal[start : start + chunk_length], dtype=np.float64))
    if not chunks:
        raise RuntimeError(
            f"no complete {chunk_length}-sample noise chunks found under {noise_root}"
        )
    return chunks


def project_events(sample: SourceSample, class_ids: Iterable[int]) -> tuple[Event, ...]:
    allowed = set(class_ids)
    return tuple(event for event in sample.events if event.class_id in allowed)


def project_label_safe_events(
    sample: SourceSample,
    class_ids: Iterable[int],
    edge_pad: int,
) -> tuple[Event, ...]:
    """Return projected events that will survive deterministic edge guarding."""

    return tuple(
        event
        for event in project_events(sample, class_ids)
        if event.left >= edge_pad and event.right <= sample.signal_length - edge_pad
    )


def eligible_positive_events(
    sample: SourceSample,
    class_ids: Iterable[int],
    edge_pad: int,
    policy: str,
) -> tuple[Event, ...]:
    """Return events from a source that is safe for the requested label view.

    The strict policy prevents a known3 sequence from containing an omitted
    ``unclear`` event and prevents the edge guard from leaving a partially
    visible event without a label.
    """

    allowed = set(class_ids)
    if policy == "legacy_any_safe_target":
        return project_label_safe_events(sample, allowed, edge_pad)
    if policy != "fully_labeled_for_view":
        raise ValueError(f"unknown source eligibility policy: {policy}")
    if not sample.events:
        return ()
    if any(event.class_id not in allowed for event in sample.events):
        return ()
    if any(
        event.left < edge_pad
        or event.right > sample.signal_length - edge_pad
        for event in sample.events
    ):
        return ()
    return tuple(sample.events)


def apply_edge_guard(
    signal: np.ndarray,
    events: Sequence[Event],
    noise_chunks: Sequence[np.ndarray],
    pad: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, tuple[Event, ...], int]:
    """Replace both edges and drop every event touching a modified edge."""

    output = np.asarray(signal, dtype=np.float64).copy()
    if pad == 0:
        return output, tuple(events), 0
    if not noise_chunks:
        raise ValueError("noise chunks are required when pad > 0")
    output[:pad] = noise_chunks[int(rng.integers(len(noise_chunks)))][:pad]
    output[-pad:] = noise_chunks[int(rng.integers(len(noise_chunks)))][:pad]
    retained = tuple(
        event for event in events if event.left >= pad and event.right <= len(output) - pad
    )
    return output, retained, len(events) - len(retained)


def smooth_join_transitions(
    signal: np.ndarray,
    segment_length: int,
    segment_count: int,
    fade: int,
) -> np.ndarray:
    if fade <= 0:
        return signal
    output = signal.copy()
    for index in range(1, segment_count):
        boundary = index * segment_length
        start = boundary - fade
        end = boundary + fade
        left_anchor = output[start - 1]
        right_anchor = output[end]
        output[start:end] = np.linspace(
            left_anchor,
            right_anchor,
            end - start,
            endpoint=False,
            dtype=output.dtype,
        )
    return output


def build_bandpass(config: GenerationConfig) -> np.ndarray:
    nyquist = config.sampling_frequency_hz / 2.0
    return butter(
        config.bandpass_order,
        [config.bandpass_low_hz / nyquist, config.bandpass_high_hz / nyquist],
        btype="bandpass",
        output="sos",
    )


def draw_balanced_group(
    samples: Sequence[SourceSample],
    class_ids: Sequence[int],
    group_size: int,
    rng: np.random.Generator,
    exposure_counts: Counter,
    edge_pad: int = 0,
    source_eligibility_policy: str = "legacy_any_safe_target",
) -> list[int]:
    """Choose distinct sources using label-derived class membership.

    One source containing each target class is chosen first when possible. Any
    remaining slots target the class with the smallest accumulated event
    exposure. Actual selected event counts update the persistent deficit state.
    """

    eligible = [
        index
        for index, sample in enumerate(samples)
        if eligible_positive_events(
            sample, class_ids, edge_pad, source_eligibility_policy
        )
    ]
    if len(eligible) < group_size:
        raise RuntimeError(
            f"only {len(eligible)} positive sources for a group of {group_size}"
        )
    by_class = {
        class_id: [
            index
            for index in eligible
            if any(
                event.class_id == class_id
                for event in eligible_positive_events(
                    samples[index],
                    class_ids,
                    edge_pad,
                    source_eligibility_policy,
                )
            )
        ]
        for class_id in class_ids
    }
    missing = [class_id for class_id, indices in by_class.items() if not indices]
    if missing:
        raise RuntimeError(f"no positive source for class ids {missing}")

    chosen: list[int] = []
    required = list(class_ids)
    rng.shuffle(required)
    targets = required[: min(group_size, len(required))]
    while len(targets) < group_size:
        minimum = min(exposure_counts[class_id] for class_id in class_ids)
        tied = [class_id for class_id in class_ids if exposure_counts[class_id] == minimum]
        targets.append(int(rng.choice(tied)))

    for target in targets:
        candidates = [index for index in by_class[target] if index not in chosen]
        if not candidates:
            candidates = [index for index in eligible if index not in chosen]
        if not candidates:
            raise RuntimeError("positive source pool exhausted within a base group")
        # Prefer candidates with fewer off-target events; randomize exact ties.
        rng.shuffle(candidates)
        selected = min(
            candidates,
            key=lambda index: (
                sum(
                    count
                    for class_id, count in Counter(
                        event.class_id
                        for event in eligible_positive_events(
                            samples[index],
                            class_ids,
                            edge_pad,
                            source_eligibility_policy,
                        )
                    ).items()
                    if class_id != target
                ),
                len(
                    eligible_positive_events(
                        samples[index],
                        class_ids,
                        edge_pad,
                        source_eligibility_policy,
                    )
                ),
            ),
        )
        chosen.append(selected)
        exposure_counts.update(
            event.class_id
            for event in eligible_positive_events(
                samples[selected],
                class_ids,
                edge_pad,
                source_eligibility_policy,
            )
        )
    return chosen


def draw_background_group(
    samples: Sequence[SourceSample],
    group_size: int,
    rng: np.random.Generator,
) -> list[int]:
    eligible = [index for index, sample in enumerate(samples) if sample.is_background]
    if len(eligible) < group_size:
        raise RuntimeError(
            f"only {len(eligible)} background sources for a group of {group_size}"
        )
    return [int(index) for index in rng.choice(eligible, size=group_size, replace=False)]


def _assemble_sequence(
    guarded_segments: Sequence[tuple[np.ndarray, tuple[Event, ...]]],
    permutation: Sequence[int],
    config: GenerationConfig,
    bandpass_sos: np.ndarray,
) -> tuple[np.ndarray, tuple[Event, ...]]:
    ordered_signals: list[np.ndarray] = []
    long_events: list[Event] = []
    for output_position, group_position in enumerate(permutation):
        signal, events = guarded_segments[group_position]
        ordered_signals.append(signal)
        offset = output_position * config.segment_length
        long_events.extend(
            Event(event.class_id, event.left + offset, event.right + offset)
            for event in events
        )
    long_signal = np.concatenate(ordered_signals)
    long_signal = smooth_join_transitions(
        long_signal,
        config.segment_length,
        config.segments_per_sequence,
        config.join_crossfade,
    )
    long_signal = sosfiltfilt(bandpass_sos, long_signal).astype(np.float64, copy=False)
    if len(long_signal) != config.long_length or not np.isfinite(long_signal).all():
        raise RuntimeError("generated signal failed length/finite audit")
    return long_signal, tuple(long_events)


def _write_labels(path: Path, events: Sequence[Event], signal_length: int) -> None:
    lines = []
    for event in events:
        center = (event.left + event.right) / (2.0 * signal_length)
        width = (event.right - event.left) / signal_length
        lines.append(f"{event.class_id} {center:.9f} {width:.9f}\n")
    path.write_text("".join(lines))


def _background_group_count(
    config: GenerationConfig, positive_group_count: int, *, split: str
) -> int:
    background_share = config.background_share_for_split(split)
    background_permutations = config.background_permutations_for_split(split)
    if config.mode != "fourclass-background" or background_share == 0:
        return 0
    positive_rows = positive_group_count * config.positive_permutations
    background_rows = positive_rows * background_share / (1 - background_share)
    groups = background_rows / background_permutations
    rounded = int(round(groups))
    if not math.isclose(groups, rounded, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "background share does not yield an integer base-group count; "
            "adjust group or permutation counts"
        )
    return rounded


def _manifest_fieldnames() -> list[str]:
    return [
        "long_id",
        "split",
        "stratum",
        "group_id",
        "permutation_id",
        "permutation",
        "perm_order",
        "source_ids",
        "source_files",
        "ordered_source_ids",
        "source_signal_sha256",
        "n_events",
        "events_per_class",
        "dropped_edge_events",
    ]


def _process_group(
    *,
    split: str,
    stratum: str,
    group_id: int,
    selected: Sequence[int],
    samples: Sequence[SourceSample],
    permutations: Sequence[tuple[int, ...]],
    config: GenerationConfig,
    noise_chunks: Sequence[np.ndarray],
    bandpass_sos: np.ndarray,
    rng: np.random.Generator,
    output_root: Path,
    writer: csv.DictWriter,
    row_index: int,
) -> tuple[int, Counter, int]:
    guarded: list[tuple[np.ndarray, tuple[Event, ...]]] = []
    dropped_total = 0
    for source_index in selected:
        sample = samples[source_index]
        events = project_events(sample, config.target_class_ids)
        signal = np.load(sample.signal_path, allow_pickle=False)
        guarded_signal, guarded_events, dropped = apply_edge_guard(
            signal, events, noise_chunks, config.noise_pad, rng
        )
        guarded.append((guarded_signal, guarded_events))
        dropped_total += dropped

    selected_samples = [samples[index] for index in selected]
    total_counts: Counter = Counter()
    for permutation_id, permutation in enumerate(permutations):
        signal, events = _assemble_sequence(guarded, permutation, config, bandpass_sos)
        long_id = f"long_{split}_{row_index:05d}"
        np.save(output_root / split / "signals" / f"{long_id}.npy", signal)
        _write_labels(
            output_root / split / "labels" / f"{long_id}.txt",
            events,
            config.long_length,
        )
        counts = Counter(event.class_id for event in events)
        total_counts.update(counts)
        ordered = [selected_samples[index] for index in permutation]
        writer.writerow(
            {
                "long_id": long_id,
                "split": split,
                "stratum": stratum,
                "group_id": group_id,
                "permutation_id": permutation_id,
                "permutation": ",".join(map(str, permutation)),
                "perm_order": ",".join(map(str, permutation)),
                "source_ids": ";".join(sample.source_id for sample in selected_samples),
                "source_files": ";".join(sample.source_id for sample in selected_samples),
                "ordered_source_ids": ";".join(sample.source_id for sample in ordered),
                "source_signal_sha256": ";".join(
                    sample.signal_sha256 for sample in selected_samples
                ),
                "n_events": len(events),
                "events_per_class": json.dumps(
                    {str(class_id): counts[class_id] for class_id in config.target_class_ids},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "dropped_edge_events": dropped_total,
            }
        )
        row_index += 1
    return row_index, total_counts, dropped_total


def _generate_into(
    source_pools: Mapping[str, Sequence[SourceSample]],
    noise_chunks: Sequence[np.ndarray],
    output_root: Path,
    config: GenerationConfig,
) -> dict:
    for split in SPLITS:
        (output_root / split / "signals").mkdir(parents=True)
        (output_root / split / "labels").mkdir(parents=True)

    positive_permutations = list(
        itertools.islice(
            itertools.permutations(range(config.segments_per_sequence)),
            config.positive_permutations,
        )
    )
    bandpass_sos = build_bandpass(config)
    split_summaries: dict[str, dict] = {}
    manifest_path = output_root / "manifest.csv"

    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_manifest_fieldnames())
        writer.writeheader()
        for split_index, split in enumerate(SPLITS):
            samples = source_pools[split]
            rng = np.random.default_rng(np.random.SeedSequence([config.seed, split_index]))
            exposure: Counter = Counter({class_id: 0 for class_id in config.target_class_ids})
            row_index = 0
            event_counts: Counter = Counter()
            dropped_events = 0
            positive_groups = config.groups_by_split[split]
            background_permutations = list(
                itertools.islice(
                    itertools.permutations(range(config.segments_per_sequence)),
                    config.background_permutations_for_split(split),
                )
            )
            for group_id in range(positive_groups):
                selected = draw_balanced_group(
                    samples,
                    config.target_class_ids,
                    config.segments_per_sequence,
                    rng,
                    exposure,
                    config.noise_pad,
                    config.source_eligibility_policy,
                )
                row_index, counts, dropped = _process_group(
                    split=split,
                    stratum="positive",
                    group_id=group_id,
                    selected=selected,
                    samples=samples,
                    permutations=positive_permutations,
                    config=config,
                    noise_chunks=noise_chunks,
                    bandpass_sos=bandpass_sos,
                    rng=rng,
                    output_root=output_root,
                    writer=writer,
                    row_index=row_index,
                )
                event_counts.update(counts)
                dropped_events += dropped

            background_groups = _background_group_count(
                config, positive_groups, split=split
            )
            background_pool = [
                index for index, sample in enumerate(samples) if sample.is_background
            ]
            if config.disjoint_background_groups:
                required = background_groups * config.segments_per_sequence
                if len(background_pool) < required:
                    raise RuntimeError(
                        f"split {split}: disjoint background groups require {required} "
                        f"sources, found {len(background_pool)}"
                    )
                rng.shuffle(background_pool)
            for group_id in range(background_groups):
                if config.disjoint_background_groups:
                    left = group_id * config.segments_per_sequence
                    selected = background_pool[left : left + config.segments_per_sequence]
                else:
                    selected = draw_background_group(
                        samples, config.segments_per_sequence, rng
                    )
                row_index, counts, dropped = _process_group(
                    split=split,
                    stratum="background",
                    group_id=group_id,
                    selected=selected,
                    samples=samples,
                    permutations=background_permutations,
                    config=config,
                    noise_chunks=noise_chunks,
                    bandpass_sos=bandpass_sos,
                    rng=rng,
                    output_root=output_root,
                    writer=writer,
                    row_index=row_index,
                )
                event_counts.update(counts)
                dropped_events += dropped

            positive_rows = positive_groups * len(positive_permutations)
            background_rows = background_groups * len(background_permutations)
            split_summaries[split] = {
                "n_long_sequences": row_index,
                "counts": {"total": row_index},
                "positive_rows": positive_rows,
                "background_rows": background_rows,
                "positive_base_groups": positive_groups,
                "background_base_groups": background_groups,
                "source_rows": len(samples),
                "source_positive_rows": sum(
                    bool(
                        eligible_positive_events(
                            sample,
                            config.target_class_ids,
                            config.noise_pad,
                            config.source_eligibility_policy,
                        )
                    )
                    for sample in samples
                ),
                "source_background_rows": sum(sample.is_background for sample in samples),
                "events_per_class": {
                    config.class_names[class_id]: event_counts[class_id]
                    for class_id in config.target_class_ids
                },
                "dropped_edge_events_at_base_group_level": dropped_events,
            }
            if (
                config.source_eligibility_policy
                == "fully_labeled_for_view"
                and dropped_events
            ):
                raise RuntimeError(
                    f"{split}: strict source eligibility still dropped "
                    f"{dropped_events} edge events"
                )

    metadata = {
        "dataset_id": config.output_dataset_id,
        "format": "yolo-1d-long-sequence",
        "train": "train/signals",
        "val": "val/signals",
        "test": "test/signals",
        "nc": len(config.class_names),
        "names": list(config.class_names),
        "sampling_frequency_hz": config.sampling_frequency_hz,
        "signal_lengths": {
            "long_sequence": config.long_length,
            "source_segment": config.segment_length,
        },
        "generation_params": {
            "mode": config.mode,
            "source_dataset_id": config.source_dataset_id,
            "noise_dataset_id": config.noise_dataset_id,
            "seed": config.seed,
            "segments_per_sequence": config.segments_per_sequence,
            "nseq": config.segments_per_sequence,
            "segment_len": config.segment_length,
            "noise_pad": config.noise_pad,
            "join_crossfade": config.join_crossfade,
            "positive_permutations": config.positive_permutations,
            "background_share": config.background_share,
            "background_permutations": config.background_permutations,
            "train_background_permutations": config.background_permutations_for_split("train"),
            "evaluation_background_share": config.background_share_for_split("val"),
            "background_share_by_split": {
                split: config.background_share_for_split(split) for split in SPLITS
            },
            "background_permutations_by_split": {
                split: config.background_permutations_for_split(split) for split in SPLITS
            },
            "disjoint_background_groups": config.disjoint_background_groups,
            "source_eligibility_policy": config.source_eligibility_policy,
            "post_concat_bandpass": {
                "enabled": True,
                "low_hz": config.bandpass_low_hz,
                "high_hz": config.bandpass_high_hz,
                "order": config.bandpass_order,
            },
            "generator_revision": config.generator_revision,
            "class_membership_source": "YOLO labels only",
            "label_class_ids": list(config.target_class_ids),
        },
        "splits": split_summaries,
        "audit_results": {
            "source_split_identity": "pass",
            "generated_dataset": "pending",
        },
    }
    (output_root / "dataset.yaml").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def generate_dataset(
    source_root: Path,
    noise_root: Path,
    output_root: Path,
    config: GenerationConfig,
) -> dict:
    """Generate atomically and refuse to mutate an existing dataset version."""

    source_root = Path(source_root)
    noise_root = Path(noise_root)
    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(f"refusing to mutate existing output: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    source_pools = load_source_dataset(source_root, config.segment_length)
    noise_chunks = load_noise_chunks(noise_root, config.noise_pad)

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent)
    )
    try:
        metadata = _generate_into(source_pools, noise_chunks, temporary, config)
        audit = audit_generated_dataset(temporary, config)
        metadata["audit_results"]["generated_dataset"] = "pass"
        metadata["audit_results"]["details"] = audit
        metadata["manifest_sha256"] = sha256_file(temporary / "manifest.csv")
        (temporary / "dataset.yaml").write_text(json.dumps(metadata, indent=2) + "\n")
        os.replace(temporary, output_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return metadata


def audit_generated_dataset(root: Path, config: GenerationConfig) -> dict:
    """Audit every manifest row and deterministic signals from every split."""

    root = Path(root)
    rows = list(csv.DictReader((root / "manifest.csv").open(newline="")))
    if not rows:
        raise ValueError("generated manifest is empty")
    seen_ids: set[str] = set()
    split_sources: dict[str, set[str]] = {split: set() for split in SPLITS}
    split_hashes: dict[str, set[str]] = {split: set() for split in SPLITS}
    background_group_sources: dict[str, dict[int, set[str]]] = {
        split: {} for split in SPLITS
    }
    rows_by_split: dict[str, list[dict[str, str]]] = {split: [] for split in SPLITS}
    for row in rows:
        split = row["split"]
        if split not in rows_by_split:
            raise ValueError(f"manifest has unknown split {split!r}")
        long_id = row["long_id"]
        if long_id in seen_ids:
            raise ValueError(f"duplicate long id {long_id}")
        seen_ids.add(long_id)
        rows_by_split[split].append(row)
        source_ids = row["source_ids"].split(";")
        source_hashes = row["source_signal_sha256"].split(";")
        if len(source_ids) != config.segments_per_sequence or len(set(source_ids)) != len(source_ids):
            raise ValueError(f"{long_id}: repeated or wrong-count source ids")
        if len(source_hashes) != config.segments_per_sequence:
            raise ValueError(f"{long_id}: wrong-count source hashes")
        split_sources[split].update(source_ids)
        split_hashes[split].update(source_hashes)
        label_path = root / split / "labels" / f"{long_id}.txt"
        signal_path = root / split / "signals" / f"{long_id}.npy"
        events = parse_yolo_labels(label_path, config.long_length, class_count=len(config.class_names))
        if len(events) != int(row["n_events"]):
            raise ValueError(f"{long_id}: manifest/label event-count mismatch")
        if row["stratum"] == "background" and events:
            raise ValueError(f"{long_id}: background stratum has labels")
        if row["stratum"] == "background":
            background_group_sources[split].setdefault(int(row["group_id"]), set()).update(
                source_ids
            )
        if not signal_path.is_file():
            raise FileNotFoundError(signal_path)

    for left_index, left_split in enumerate(SPLITS):
        for right_split in SPLITS[left_index + 1 :]:
            duplicate_ids = split_sources[left_split] & split_sources[right_split]
            duplicate_hashes = split_hashes[left_split] & split_hashes[right_split]
            if duplicate_ids or duplicate_hashes:
                raise ValueError(
                    f"generated split leakage {left_split}/{right_split}: "
                    f"ids={len(duplicate_ids)}, hashes={len(duplicate_hashes)}"
                )

    if config.disjoint_background_groups:
        for split, groups in background_group_sources.items():
            owner: dict[str, int] = {}
            for group_id, source_ids in groups.items():
                for source_id in source_ids:
                    if source_id in owner and owner[source_id] != group_id:
                        raise ValueError(
                            f"{split}: background source {source_id} reused by groups "
                            f"{owner[source_id]} and {group_id}"
                        )
                    owner[source_id] = group_id

    sampled_signal_count = 0
    for split, split_rows in rows_by_split.items():
        if not split_rows:
            raise ValueError(f"generated split {split} is empty")
        indices = np.linspace(0, len(split_rows) - 1, min(8, len(split_rows)), dtype=int)
        for index in sorted(set(indices.tolist())):
            row = split_rows[index]
            signal = np.load(
                root / split / "signals" / f"{row['long_id']}.npy",
                allow_pickle=False,
            )
            if signal.shape != (config.long_length,) or signal.dtype != np.float64:
                raise ValueError(
                    f"{row['long_id']}: expected float64 ({config.long_length},), "
                    f"got {signal.dtype} {signal.shape}"
                )
            if not np.isfinite(signal).all():
                raise ValueError(f"{row['long_id']}: non-finite generated signal")
            sampled_signal_count += 1

    return {
        "manifest_rows_checked": len(rows),
        "signals_sampled": sampled_signal_count,
        "cross_split_source_ids": 0,
        "cross_split_source_hashes": 0,
        "status": "pass",
    }
