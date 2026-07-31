"""Development-only Z8 v2 Wave8-like dataset generation.

The generator consumes the registered Z8 v2 event-reference table and its
registered F-base parent signals. It creates source-disjoint long sequences
with separately filtered continuous-noise bridges and raised-cosine guards.
No test split is read or emitted.
"""

from __future__ import annotations

import csv
import hashlib
import io
import itertools
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.signal import butter, sosfiltfilt


SPLITS = ("train", "val")
CLASS_NAMES = ("2um", "4um", "10um")


@dataclass(frozen=True)
class EventRef:
    event_id: str
    class_id: int
    class_name: str
    left: float
    right: float


@dataclass(frozen=True)
class SourceRef:
    split: str
    source_id: str
    signal_path: Path
    relative_signal_path: str
    signal_sha256: str
    events: tuple[EventRef, ...]

    @property
    def is_background(self) -> bool:
        return not self.events


@dataclass(frozen=True)
class Z8Wave8LikeConfig:
    output_dataset_id: str
    z8_dataset_id: str
    parent_dataset_id: str
    noise_dataset_id: str
    seed: int = 42
    segment_length: int = 16_384
    segments_per_sequence: int = 4
    guard_samples: int = 300
    sampling_frequency_hz: int = 2_000_000
    bandpass_low_hz: float = 7_000.0
    bandpass_high_hz: float = 80_000.0
    bandpass_order: int = 4
    train_positive_groups: int = 100
    val_positive_groups: int = 30
    positive_permutations: int = 24
    train_background_groups: int = 200
    val_background_groups: int = 40
    train_background_permutations: int = 12
    val_background_permutations: int = 6
    bridge_matching: str = "none"
    bridge_context_samples: int = 2_400
    endpoint_quality_enabled: bool = False
    endpoint_quality_window_samples: int = 900
    endpoint_max_rms_ratio: float = 2.5
    endpoint_max_peak_robust_z: float = 8.0
    generator_revision: str = "unknown"

    def __post_init__(self) -> None:
        if self.segment_length <= 0 or self.segments_per_sequence != 4:
            raise ValueError("Z8 Wave8-like requires four positive-length segments")
        if not 0 < self.guard_samples * 2 < self.segment_length:
            raise ValueError("guard_samples must leave a non-empty segment interior")
        nyquist = self.sampling_frequency_hz / 2.0
        if not 0 < self.bandpass_low_hz < self.bandpass_high_hz < nyquist:
            raise ValueError("bandpass cutoffs must be ordered below Nyquist")
        if self.bandpass_order <= 0:
            raise ValueError("bandpass_order must be positive")
        if self.bridge_matching not in {
            "none",
            "robust-local-rms",
            "robust-local-rms-global-cap",
        }:
            raise ValueError("unsupported bridge_matching mode")
        if self.bridge_context_samples < self.guard_samples:
            raise ValueError("bridge context must be at least one guard")
        if self.endpoint_quality_enabled and not (
            self.guard_samples
            <= self.endpoint_quality_window_samples
            <= self.segment_length
        ):
            raise ValueError("invalid endpoint quality window")
        if self.endpoint_max_rms_ratio <= 0 or self.endpoint_max_peak_robust_z <= 0:
            raise ValueError("endpoint quality thresholds must be positive")
        max_permutations = 24
        for value in (
            self.positive_permutations,
            self.train_background_permutations,
            self.val_background_permutations,
        ):
            if not 1 <= value <= max_permutations:
                raise ValueError("permutation counts must lie in [1, 24]")
        if self.positive_permutations != 24:
            raise ValueError("positive groups must emit all 24 permutations")
        if any(
            value <= 0
            for value in (
                self.train_positive_groups,
                self.val_positive_groups,
                self.train_background_groups,
                self.val_background_groups,
            )
        ):
            raise ValueError("group counts must be positive")

    @property
    def bridge_length(self) -> int:
        return 2 * self.guard_samples

    @property
    def long_length(self) -> int:
        return self.segment_length * self.segments_per_sequence

    def positive_groups(self, split: str) -> int:
        return (
            self.train_positive_groups
            if split == "train"
            else self.val_positive_groups
        )

    def background_groups(self, split: str) -> int:
        return (
            self.train_background_groups
            if split == "train"
            else self.val_background_groups
        )

    def background_permutations(self, split: str) -> int:
        return (
            self.train_background_permutations
            if split == "train"
            else self.val_background_permutations
        )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_seed(seed: int, *parts: object) -> int:
    payload = "|".join((str(seed), *(str(part) for part in parts))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _npy_bytes(values: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, values, allow_pickle=False)
    return buffer.getvalue()


def _labels_text(events: Sequence[EventRef], signal_length: int) -> str:
    lines = []
    for event in events:
        center = (event.left + event.right) / (2.0 * signal_length)
        width = (event.right - event.left) / signal_length
        lines.append(f"{event.class_id} {center:.12f} {width:.12f}\n")
    return "".join(lines)


def load_event_table(path: Path) -> dict[tuple[str, str], tuple[EventRef, ...]]:
    grouped: dict[tuple[str, str], list[EventRef]] = defaultdict(list)
    seen_ids: set[str] = set()
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            split = row["split"]
            if split not in SPLITS:
                raise ValueError(f"{path}:{row_number}: sealed/unknown split {split!r}")
            event_id = row["event_id"]
            if event_id in seen_ids:
                raise ValueError(f"{path}:{row_number}: duplicate event_id {event_id}")
            seen_ids.add(event_id)
            class_name = row["class_name"]
            class_id = int(row["class_id"])
            if class_name in CLASS_NAMES:
                expected_id = CLASS_NAMES.index(class_name)
                if class_id != expected_id:
                    raise ValueError(
                        f"{path}:{row_number}: class mapping mismatch for {class_name}"
                    )
            elif class_name != "unclear" or class_id != 3:
                raise ValueError(f"{path}:{row_number}: invalid class {class_id}/{class_name}")
            left = float(row["start_sample"])
            right = float(row["end_sample"])
            if not np.isfinite(left) or not np.isfinite(right) or not 0 <= left < right:
                raise ValueError(f"{path}:{row_number}: invalid event geometry")
            grouped[(split, row["source_filename"])].append(
                EventRef(
                    event_id=event_id,
                    class_id=class_id,
                    class_name=class_name,
                    left=left,
                    right=right,
                )
            )
    return {
        key: tuple(sorted(events, key=lambda event: (event.left, event.right, event.event_id)))
        for key, events in grouped.items()
    }


def load_source_pools(
    parent_root: Path,
    event_table_path: Path,
    config: Z8Wave8LikeConfig,
) -> dict[str, list[SourceRef]]:
    events_by_source = load_event_table(event_table_path)
    pools: dict[str, list[SourceRef]] = {}
    observed_keys: set[tuple[str, str]] = set()
    id_owner: dict[str, str] = {}
    hash_owner: dict[str, str] = {}
    for split in SPLITS:
        signal_dir = Path(parent_root) / split / "signals"
        signal_paths = sorted(signal_dir.glob("*.npy"))
        if not signal_paths:
            raise RuntimeError(f"no parent signals in {signal_dir}")
        samples: list[SourceRef] = []
        for signal_path in signal_paths:
            values = np.load(signal_path, mmap_mode="r", allow_pickle=False)
            if values.shape != (config.segment_length,):
                raise ValueError(
                    f"{signal_path}: expected ({config.segment_length},), got {values.shape}"
                )
            if values.dtype != np.float64 or not np.isfinite(values).all():
                raise ValueError(f"{signal_path}: expected finite float64 signal")
            source_id = signal_path.stem
            signal_hash = sha256_file(signal_path)
            previous_split = id_owner.setdefault(source_id, split)
            if previous_split != split:
                raise ValueError(f"source ID crosses splits: {source_id}")
            previous_hash_split = hash_owner.setdefault(signal_hash, split)
            if previous_hash_split != split:
                raise ValueError(f"source signal content crosses splits: {signal_hash}")
            key = (split, signal_path.name)
            observed_keys.add(key)
            samples.append(
                SourceRef(
                    split=split,
                    source_id=source_id,
                    signal_path=signal_path,
                    relative_signal_path=signal_path.relative_to(parent_root).as_posix(),
                    signal_sha256=signal_hash,
                    events=events_by_source.get(key, ()),
                )
            )
        pools[split] = samples
    missing_sources = sorted(set(events_by_source) - observed_keys)
    if missing_sources:
        raise ValueError(f"event table references missing parent signals: {missing_sources[:3]}")
    return pools


def physical_source_is_eligible(
    sample: SourceRef, config: Z8Wave8LikeConfig
) -> bool:
    if not sample.events:
        return False
    if any(event.class_name not in CLASS_NAMES for event in sample.events):
        return False
    geometry_is_safe = all(
        event.left >= config.guard_samples
        and event.right <= config.segment_length - config.guard_samples
        for event in sample.events
    )
    return geometry_is_safe and (
        not config.endpoint_quality_enabled
        or bool(source_endpoint_quality(sample, config)["safe"])
    )


def _source_class(sample: SourceRef) -> int:
    class_ids = {event.class_id for event in sample.events}
    if len(class_ids) != 1 or not class_ids <= {0, 1, 2}:
        raise ValueError(f"{sample.source_id}: expected one physical source class")
    return next(iter(class_ids))


@lru_cache(maxsize=None)
def source_endpoint_quality(
    sample: SourceRef,
    config: Z8Wave8LikeConfig,
) -> dict[str, object]:
    signal = np.load(sample.signal_path, allow_pickle=False)
    global_values = _event_free_source_values(signal, sample.events)
    global_median = float(np.median(global_values))
    global_scale = _robust_mad_scale(global_values)
    window = config.endpoint_quality_window_samples
    endpoint_metrics: dict[str, dict[str, float | int | bool]] = {}
    for side, start, stop in (
        ("left", 0, window),
        ("right", len(signal) - window, len(signal)),
    ):
        keep = np.ones(stop - start, dtype=bool)
        for event in sample.events:
            event_start = max(start, int(np.floor(event.left)))
            event_stop = min(stop, int(np.ceil(event.right)))
            if event_start < event_stop:
                keep[event_start - start : event_stop - start] = False
        values = np.asarray(signal[start:stop])[keep]
        if len(values) < config.guard_samples:
            raise ValueError(
                f"{sample.source_id}: insufficient endpoint background on {side}"
            )
        local_scale = _robust_mad_scale(values)
        rms_ratio = local_scale / global_scale
        peak_robust_z = float(
            np.max(np.abs(values - global_median)) / global_scale
        )
        endpoint_metrics[side] = {
            "samples": len(values),
            "rms_ratio": rms_ratio,
            "peak_robust_z": peak_robust_z,
            "safe": (
                rms_ratio <= config.endpoint_max_rms_ratio
                and peak_robust_z <= config.endpoint_max_peak_robust_z
            ),
        }
    return {
        "global_robust_scale": global_scale,
        "left": endpoint_metrics["left"],
        "right": endpoint_metrics["right"],
        "safe": bool(
            endpoint_metrics["left"]["safe"]
            and endpoint_metrics["right"]["safe"]
        ),
    }


def draw_source_disjoint_positive_groups(
    samples: Sequence[SourceRef],
    group_count: int,
    *,
    seed: int,
    config: Z8Wave8LikeConfig,
) -> list[tuple[SourceRef, ...]]:
    eligible = [sample for sample in samples if physical_source_is_eligible(sample, config)]
    by_class = {
        class_id: [sample for sample in eligible if _source_class(sample) == class_id]
        for class_id in range(3)
    }
    for class_id, candidates in by_class.items():
        if len(candidates) < group_count:
            raise RuntimeError(
                f"only {len(candidates)} eligible {CLASS_NAMES[class_id]} sources "
                f"for {group_count} source-disjoint groups"
            )
    rng = np.random.default_rng(seed)
    for candidates in by_class.values():
        rng.shuffle(candidates)
    unused = {sample.source_id: sample for sample in eligible}
    exposure: Counter[int] = Counter({class_id: 0 for class_id in range(3)})
    groups: list[tuple[SourceRef, ...]] = []
    for group_index in range(group_count):
        chosen: list[SourceRef] = []
        for class_id in rng.permutation(3).tolist():
            candidates = [
                sample
                for sample in by_class[class_id]
                if sample.source_id in unused
            ]
            if not candidates:
                raise RuntimeError(f"source pool exhausted for class {class_id}")
            minimum_events = min(len(sample.events) for sample in candidates)
            tied = [
                sample for sample in candidates if len(sample.events) == minimum_events
            ]
            sample = tied[int(rng.integers(len(tied)))]
            chosen.append(sample)
            unused.pop(sample.source_id)
            exposure.update(event.class_id for event in sample.events)

        remaining_groups = group_count - group_index - 1
        optional_classes = [
            class_id
            for class_id in range(3)
            if sum(
                sample.source_id in unused for sample in by_class[class_id]
            )
            > remaining_groups
        ]
        if not optional_classes:
            raise RuntimeError("no class has a non-reserved fourth source")
        target_exposure = min(exposure[class_id] for class_id in optional_classes)
        target_classes = [
            class_id
            for class_id in optional_classes
            if exposure[class_id] == target_exposure
        ]
        target_class = target_classes[int(rng.integers(len(target_classes)))]
        candidates = [
            sample
            for sample in by_class[target_class]
            if sample.source_id in unused
        ]
        minimum_events = min(len(sample.events) for sample in candidates)
        tied = [sample for sample in candidates if len(sample.events) == minimum_events]
        fourth = tied[int(rng.integers(len(tied)))]
        chosen.append(fourth)
        unused.pop(fourth.source_id)
        exposure.update(event.class_id for event in fourth.events)
        rng.shuffle(chosen)
        groups.append(tuple(chosen))
    return groups


def draw_source_disjoint_background_groups(
    samples: Sequence[SourceRef],
    group_count: int,
    *,
    seed: int,
    config: Z8Wave8LikeConfig,
    segments_per_sequence: int = 4,
) -> list[tuple[SourceRef, ...]]:
    backgrounds = [
        sample
        for sample in samples
        if sample.is_background
        and (
            not config.endpoint_quality_enabled
            or bool(source_endpoint_quality(sample, config)["safe"])
        )
    ]
    required = group_count * segments_per_sequence
    if len(backgrounds) < required:
        raise RuntimeError(
            f"only {len(backgrounds)} annotation-relative backgrounds for "
            f"{group_count} source-disjoint groups ({required} required)"
        )
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(backgrounds))[:required]
    selected = [backgrounds[int(index)] for index in indices]
    return [
        tuple(selected[start : start + segments_per_sequence])
        for start in range(0, required, segments_per_sequence)
    ]


class FilteredNoisePool:
    def __init__(self, noise_root: Path, config: Z8Wave8LikeConfig):
        self.config = config
        self.paths = sorted(Path(noise_root).rglob("*.npy"))
        if not self.paths:
            raise RuntimeError(f"no noise arrays under {noise_root}")
        self.hashes = {path: sha256_file(path) for path in self.paths}
        self._cache: dict[Path, np.ndarray] = {}
        nyquist = config.sampling_frequency_hz / 2.0
        self._sos = butter(
            config.bandpass_order,
            [
                config.bandpass_low_hz / nyquist,
                config.bandpass_high_hz / nyquist,
            ],
            btype="bandpass",
            output="sos",
        )

    def _filtered(self, path: Path) -> np.ndarray:
        if path not in self._cache:
            values = np.load(path, allow_pickle=False)
            if values.ndim != 1 or values.dtype != np.float64:
                raise ValueError(f"{path}: expected one-dimensional float64 noise")
            if len(values) < self.config.bridge_length or not np.isfinite(values).all():
                raise ValueError(f"{path}: invalid noise bridge source")
            self._cache[path] = sosfiltfilt(self._sos, values).astype(
                np.float64, copy=False
            )
        return self._cache[path]

    def bridge(
        self, *, split: str, stratum: str, group_id: int, left_id: str, right_id: str
    ) -> tuple[np.ndarray, dict[str, object]]:
        rng = np.random.default_rng(
            _stable_seed(
                self.config.seed,
                "bridge",
                split,
                stratum,
                group_id,
                left_id,
                right_id,
            )
        )
        path = self.paths[int(rng.integers(len(self.paths)))]
        filtered = self._filtered(path)
        max_start = len(filtered) - self.config.bridge_length
        start = int(rng.integers(max_start + 1))
        bridge = filtered[start : start + self.config.bridge_length].copy()
        return bridge, {
            "path": path.name,
            "sha256": self.hashes[path],
            "crop_start": start,
        }


def apply_raised_cosine_bridge(
    left: np.ndarray,
    right: np.ndarray,
    bridge: np.ndarray,
    guard_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    if left.ndim != 1 or right.ndim != 1 or bridge.ndim != 1:
        raise ValueError("join inputs must be one-dimensional")
    if len(bridge) != 2 * guard_samples:
        raise ValueError("bridge length must equal two guards")
    if len(left) < guard_samples or len(right) < guard_samples:
        raise ValueError("guard exceeds source length")
    phase = np.linspace(0.0, np.pi, guard_samples, endpoint=True)
    enter_bridge = 0.5 - 0.5 * np.cos(phase)
    leave_bridge = 0.5 + 0.5 * np.cos(phase)
    left_out = left.copy()
    right_out = right.copy()
    left_out[-guard_samples:] = (
        (1.0 - enter_bridge) * left[-guard_samples:]
        + enter_bridge * bridge[:guard_samples]
    )
    right_out[:guard_samples] = (
        leave_bridge * bridge[guard_samples:]
        + (1.0 - leave_bridge) * right[:guard_samples]
    )
    return left_out, right_out


def _event_free_endpoint_context(
    signal: np.ndarray,
    events: Sequence[EventRef],
    *,
    side: str,
    guard_samples: int,
    context_samples: int,
) -> np.ndarray:
    if side == "left":
        primary = (
            max(0, len(signal) - guard_samples - context_samples),
            len(signal) - guard_samples,
        )
        fallback = (0, len(signal) - guard_samples)
    elif side == "right":
        primary = (
            guard_samples,
            min(len(signal), guard_samples + context_samples),
        )
        fallback = (guard_samples, len(signal))
    else:
        raise ValueError(f"invalid endpoint side: {side}")

    def select(bounds: tuple[int, int]) -> np.ndarray:
        start, stop = bounds
        keep = np.ones(stop - start, dtype=bool)
        for event in events:
            event_start = max(start, int(np.floor(event.left)))
            event_stop = min(stop, int(np.ceil(event.right)))
            if event_start < event_stop:
                keep[event_start - start : event_stop - start] = False
        return np.asarray(signal[start:stop])[keep]

    values = select(primary)
    minimum = guard_samples
    if len(values) < minimum:
        values = select(fallback)
    if len(values) < minimum or not np.isfinite(values).all():
        raise ValueError("insufficient finite annotation-free endpoint context")
    return values


def _event_free_source_values(
    signal: np.ndarray,
    events: Sequence[EventRef],
) -> np.ndarray:
    keep = np.ones(len(signal), dtype=bool)
    for event in events:
        event_start = max(0, int(np.floor(event.left)))
        event_stop = min(len(signal), int(np.ceil(event.right)))
        if event_start < event_stop:
            keep[event_start:event_stop] = False
    values = np.asarray(signal)[keep]
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("source has no finite annotation-free baseline")
    return values


def _robust_mad_scale(values: np.ndarray) -> float:
    centered = values - np.median(values)
    scale = 1.4826 * float(np.median(np.abs(centered)))
    if scale <= 1e-12:
        raise ValueError("RMS matching requires non-degenerate noise")
    return scale


def _robust_rms(values: np.ndarray) -> float:
    centered = values - np.median(values)
    mad_scale = _robust_mad_scale(values)
    clipped = np.clip(centered, -4.0 * mad_scale, 4.0 * mad_scale)
    return float(np.sqrt(np.mean(np.square(clipped))))


def match_bridge_to_local_rms(
    left: np.ndarray,
    right: np.ndarray,
    bridge: np.ndarray,
    *,
    left_events: Sequence[EventRef],
    right_events: Sequence[EventRef],
    guard_samples: int,
    context_samples: int,
    cap_by_global: bool = False,
) -> tuple[np.ndarray, dict[str, object]]:
    """Match a zero-centered bridge to annotation-free endpoint noise RMS."""
    left_context = _event_free_endpoint_context(
        left,
        left_events,
        side="left",
        guard_samples=guard_samples,
        context_samples=context_samples,
    )
    right_context = _event_free_endpoint_context(
        right,
        right_events,
        side="right",
        guard_samples=guard_samples,
        context_samples=context_samples,
    )
    left_local_rms = _robust_rms(left_context)
    right_local_rms = _robust_rms(right_context)
    if cap_by_global:
        left_global_rms = _robust_mad_scale(
            _event_free_source_values(left, left_events)
        )
        right_global_rms = _robust_mad_scale(
            _event_free_source_values(right, right_events)
        )
        left_rms = min(left_local_rms, left_global_rms)
        right_rms = min(right_local_rms, right_global_rms)
        matching = "robust-local-rms-global-cap"
    else:
        left_global_rms = None
        right_global_rms = None
        left_rms = left_local_rms
        right_rms = right_local_rms
        matching = "robust-local-rms"
    bridge_centered = bridge - np.median(bridge)
    raw_bridge_rms = _robust_rms(bridge)
    phase = np.linspace(0.0, np.pi, len(bridge), endpoint=True)
    transition = 0.5 - 0.5 * np.cos(phase)
    target_rms = (1.0 - transition) * left_rms + transition * right_rms
    matched = bridge_centered / raw_bridge_rms * target_rms
    matched_rms = float(np.sqrt(np.mean(np.square(matched))))
    return matched.astype(np.float64, copy=False), {
        "matching": matching,
        "left_local_rms": left_local_rms,
        "right_local_rms": right_local_rms,
        "left_global_rms": left_global_rms if left_global_rms is not None else "",
        "right_global_rms": (
            right_global_rms if right_global_rms is not None else ""
        ),
        "left_target_rms": left_rms,
        "right_target_rms": right_rms,
        "raw_bridge_rms": raw_bridge_rms,
        "matched_bridge_rms": matched_rms,
        "left_context_samples": len(left_context),
        "right_context_samples": len(right_context),
    }


def assemble_sequence(
    ordered_sources: Sequence[SourceRef],
    *,
    split: str,
    stratum: str,
    group_id: int,
    noise_pool: FilteredNoisePool,
    config: Z8Wave8LikeConfig,
) -> tuple[np.ndarray, tuple[EventRef, ...], list[dict[str, object]]]:
    if len(ordered_sources) != config.segments_per_sequence:
        raise ValueError("wrong source count for long sequence")
    segments = [
        np.load(sample.signal_path, allow_pickle=False).astype(np.float64, copy=True)
        for sample in ordered_sources
    ]
    bridges: list[dict[str, object]] = []
    for boundary_index in range(config.segments_per_sequence - 1):
        left_source = ordered_sources[boundary_index]
        right_source = ordered_sources[boundary_index + 1]
        bridge, bridge_record = noise_pool.bridge(
            split=split,
            stratum=stratum,
            group_id=group_id,
            left_id=left_source.source_id,
            right_id=right_source.source_id,
        )
        if config.bridge_matching in {
            "robust-local-rms",
            "robust-local-rms-global-cap",
        }:
            bridge, matching_record = match_bridge_to_local_rms(
                segments[boundary_index],
                segments[boundary_index + 1],
                bridge,
                left_events=left_source.events,
                right_events=right_source.events,
                guard_samples=config.guard_samples,
                context_samples=config.bridge_context_samples,
                cap_by_global=(
                    config.bridge_matching == "robust-local-rms-global-cap"
                ),
            )
        else:
            bridge_rms = float(
                np.sqrt(np.mean(np.square(bridge - np.median(bridge))))
            )
            matching_record = {
                "matching": "none",
                "left_local_rms": "",
                "right_local_rms": "",
                "left_global_rms": "",
                "right_global_rms": "",
                "left_target_rms": "",
                "right_target_rms": "",
                "raw_bridge_rms": bridge_rms,
                "matched_bridge_rms": bridge_rms,
                "left_context_samples": "",
                "right_context_samples": "",
            }
        bridge_record.update(matching_record)
        segments[boundary_index], segments[boundary_index + 1] = (
            apply_raised_cosine_bridge(
                segments[boundary_index],
                segments[boundary_index + 1],
                bridge,
                config.guard_samples,
            )
        )
        bridge_record["boundary_index"] = boundary_index + 1
        bridge_record["left_source_id"] = left_source.source_id
        bridge_record["right_source_id"] = right_source.source_id
        bridges.append(bridge_record)
    long_signal = np.concatenate(segments)
    if long_signal.shape != (config.long_length,) or not np.isfinite(long_signal).all():
        raise RuntimeError("assembled sequence failed shape/finite audit")
    long_events: list[EventRef] = []
    if stratum == "positive":
        for position, source in enumerate(ordered_sources):
            offset = position * config.segment_length
            long_events.extend(
                EventRef(
                    event_id=event.event_id,
                    class_id=event.class_id,
                    class_name=event.class_name,
                    left=event.left + offset,
                    right=event.right + offset,
                )
                for event in source.events
            )
    elif any(source.events for source in ordered_sources):
        raise ValueError("background sequence contains source events")
    return (
        long_signal,
        tuple(sorted(long_events, key=lambda event: (event.left, event.right))),
        bridges,
    )


def _permutations(
    *,
    split: str,
    stratum: str,
    group_id: int,
    count: int,
    config: Z8Wave8LikeConfig,
) -> list[tuple[int, ...]]:
    permutations = list(itertools.permutations(range(config.segments_per_sequence)))
    if count == len(permutations):
        return permutations
    rng = np.random.default_rng(
        _stable_seed(config.seed, "permutations", split, stratum, group_id)
    )
    indices = rng.permutation(len(permutations))[:count]
    return [permutations[int(index)] for index in indices]


def _plan_groups(
    pools: Mapping[str, Sequence[SourceRef]],
    config: Z8Wave8LikeConfig,
) -> dict[str, dict[str, list[tuple[SourceRef, ...]]]]:
    return {
        split: {
            "positive": draw_source_disjoint_positive_groups(
                pools[split],
                config.positive_groups(split),
                seed=_stable_seed(config.seed, split, "positive-groups"),
                config=config,
            ),
            "background": draw_source_disjoint_background_groups(
                pools[split],
                config.background_groups(split),
                seed=_stable_seed(config.seed, split, "background-groups"),
                config=config,
                segments_per_sequence=config.segments_per_sequence,
            ),
        }
        for split in SPLITS
    }


def endpoint_quality_pool_audit(
    pools: Mapping[str, Sequence[SourceRef]],
    config: Z8Wave8LikeConfig,
) -> dict[str, object]:
    if not config.endpoint_quality_enabled:
        return {"status": "not_enabled"}
    counts: dict[str, object] = {}
    for split in SPLITS:
        physical = [
            sample
            for sample in pools[split]
            if physical_source_is_eligible(sample, config)
        ]
        backgrounds = [
            sample
            for sample in pools[split]
            if sample.is_background
            and bool(source_endpoint_quality(sample, config)["safe"])
        ]
        by_class = Counter(_source_class(sample) for sample in physical)
        minimum_groups = config.positive_groups(split)
        required_backgrounds = (
            config.background_groups(split) * config.segments_per_sequence
        )
        if any(by_class[class_id] < minimum_groups for class_id in range(3)):
            raise RuntimeError(f"{split}: endpoint-safe physical class deficit")
        if len(physical) < minimum_groups * config.segments_per_sequence:
            raise RuntimeError(f"{split}: endpoint-safe positive source deficit")
        if len(backgrounds) < required_backgrounds:
            raise RuntimeError(f"{split}: endpoint-safe background source deficit")
        counts[split] = {
            "physical_by_class": {
                CLASS_NAMES[class_id]: by_class[class_id]
                for class_id in range(3)
            },
            "physical_total": len(physical),
            "physical_required_total": (
                minimum_groups * config.segments_per_sequence
            ),
            "background": len(backgrounds),
            "background_required": required_backgrounds,
        }
    return {
        "status": "pass",
        "window_samples": config.endpoint_quality_window_samples,
        "max_rms_ratio": config.endpoint_max_rms_ratio,
        "max_peak_robust_z": config.endpoint_max_peak_robust_z,
        "annotations_excluded": True,
        "both_endpoints_required": True,
        "counts": counts,
    }


MANIFEST_FIELDS = (
    "long_id",
    "split",
    "stratum",
    "group_id",
    "permutation_index",
    "source_ids",
    "source_signal_paths",
    "source_signal_sha256",
    "source_endpoint_quality",
    "source_order",
    "event_ids",
    "n_events",
    "event_counts",
    "bridge_noise_paths",
    "bridge_noise_sha256",
    "bridge_crop_starts",
    "bridge_matching",
    "bridge_left_local_rms",
    "bridge_right_local_rms",
    "bridge_left_global_rms",
    "bridge_right_global_rms",
    "bridge_left_target_rms",
    "bridge_right_target_rms",
    "bridge_raw_rms",
    "bridge_matched_rms",
    "bridge_left_context_samples",
    "bridge_right_context_samples",
    "signal_sha256",
    "label_sha256",
)


def _iter_generated_rows(
    plans: Mapping[str, Mapping[str, Sequence[tuple[SourceRef, ...]]]],
    noise_pool: FilteredNoisePool,
    config: Z8Wave8LikeConfig,
):
    for split in SPLITS:
        for stratum in ("positive", "background"):
            permutation_count = (
                config.positive_permutations
                if stratum == "positive"
                else config.background_permutations(split)
            )
            for group_id, group in enumerate(plans[split][stratum]):
                for permutation_index, permutation in enumerate(
                    _permutations(
                        split=split,
                        stratum=stratum,
                        group_id=group_id,
                        count=permutation_count,
                        config=config,
                    )
                ):
                    ordered = tuple(group[index] for index in permutation)
                    signal, events, bridges = assemble_sequence(
                        ordered,
                        split=split,
                        stratum=stratum,
                        group_id=group_id,
                        noise_pool=noise_pool,
                        config=config,
                    )
                    long_id = (
                        f"z8w8_{split}_{stratum}_{group_id:04d}_p"
                        f"{permutation_index:02d}"
                    )
                    label_text = _labels_text(events, config.long_length)
                    yield {
                        "long_id": long_id,
                        "split": split,
                        "stratum": stratum,
                        "group_id": str(group_id),
                        "permutation_index": str(permutation_index),
                        "source_ids": ";".join(source.source_id for source in ordered),
                        "source_signal_paths": ";".join(
                            source.relative_signal_path for source in ordered
                        ),
                        "source_signal_sha256": ";".join(
                            source.signal_sha256 for source in ordered
                        ),
                        "source_endpoint_quality": (
                            json.dumps(
                                [
                                    source_endpoint_quality(source, config)
                                    for source in ordered
                                ],
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                            if config.endpoint_quality_enabled
                            else ""
                        ),
                        "source_order": ";".join(str(index) for index in permutation),
                        "event_ids": ";".join(event.event_id for event in events),
                        "n_events": str(len(events)),
                        "event_counts": json.dumps(
                            dict(
                                sorted(
                                    Counter(
                                        event.class_name for event in events
                                    ).items()
                                )
                            ),
                            separators=(",", ":"),
                        ),
                        "bridge_noise_paths": ";".join(
                            str(record["path"]) for record in bridges
                        ),
                        "bridge_noise_sha256": ";".join(
                            str(record["sha256"]) for record in bridges
                        ),
                        "bridge_crop_starts": ";".join(
                            str(record["crop_start"]) for record in bridges
                        ),
                        "bridge_matching": ";".join(
                            str(record["matching"]) for record in bridges
                        ),
                        "bridge_left_local_rms": ";".join(
                            str(record["left_local_rms"]) for record in bridges
                        ),
                        "bridge_right_local_rms": ";".join(
                            str(record["right_local_rms"]) for record in bridges
                        ),
                        "bridge_left_global_rms": ";".join(
                            str(record["left_global_rms"]) for record in bridges
                        ),
                        "bridge_right_global_rms": ";".join(
                            str(record["right_global_rms"]) for record in bridges
                        ),
                        "bridge_left_target_rms": ";".join(
                            str(record["left_target_rms"]) for record in bridges
                        ),
                        "bridge_right_target_rms": ";".join(
                            str(record["right_target_rms"]) for record in bridges
                        ),
                        "bridge_raw_rms": ";".join(
                            str(record["raw_bridge_rms"]) for record in bridges
                        ),
                        "bridge_matched_rms": ";".join(
                            str(record["matched_bridge_rms"]) for record in bridges
                        ),
                        "bridge_left_context_samples": ";".join(
                            str(record["left_context_samples"]) for record in bridges
                        ),
                        "bridge_right_context_samples": ";".join(
                            str(record["right_context_samples"]) for record in bridges
                        ),
                        "signal_sha256": hashlib.sha256(_npy_bytes(signal)).hexdigest(),
                        "label_sha256": hashlib.sha256(label_text.encode()).hexdigest(),
                        "_signal": signal,
                        "_label_text": label_text,
                    }


def _logical_manifest_hash(rows: Iterable[Mapping[str, str]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        payload = {field: row[field] for field in MANIFEST_FIELDS}
        digest.update(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        )
        digest.update(b"\n")
    return digest.hexdigest()


def audit_generated_dataset(
    root: Path,
    config: Z8Wave8LikeConfig,
    *,
    expected_rows: Sequence[Mapping[str, str]] | None = None,
) -> dict[str, object]:
    root = Path(root)
    if (root / "test").exists():
        raise ValueError("development-only candidate must not contain a test split")
    with (root / "manifest.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("generated manifest is empty")
    if expected_rows is not None:
        expected_public = [
            {field: row[field] for field in MANIFEST_FIELDS} for row in expected_rows
        ]
        if rows != expected_public:
            raise ValueError("written manifest differs from generation plan")
    seen_ids: set[str] = set()
    source_group_owner: dict[tuple[str, str, str], int] = {}
    split_ids: dict[str, set[str]] = {split: set() for split in SPLITS}
    split_hashes: dict[str, set[str]] = {split: set() for split in SPLITS}
    counts: Counter[tuple[str, str]] = Counter()
    event_counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        long_id = row["long_id"]
        if long_id in seen_ids:
            raise ValueError(f"duplicate long_id: {long_id}")
        seen_ids.add(long_id)
        split = row["split"]
        stratum = row["stratum"]
        counts[(split, stratum)] += 1
        source_ids = row["source_ids"].split(";")
        source_hashes = row["source_signal_sha256"].split(";")
        if len(source_ids) != 4 or len(set(source_ids)) != 4:
            raise ValueError(f"{long_id}: invalid source membership")
        group_id = int(row["group_id"])
        for source_id in source_ids:
            key = (split, stratum, source_id)
            owner = source_group_owner.setdefault(key, group_id)
            if owner != group_id:
                raise ValueError(f"{long_id}: source reused across base groups")
        split_ids[split].update(source_ids)
        split_hashes[split].update(source_hashes)
        signal_path = root / split / "signals" / f"{long_id}.npy"
        label_path = root / split / "labels" / f"{long_id}.txt"
        if sha256_file(signal_path) != row["signal_sha256"]:
            raise ValueError(f"{long_id}: signal hash mismatch")
        if sha256_file(label_path) != row["label_sha256"]:
            raise ValueError(f"{long_id}: label hash mismatch")
        signal = np.load(signal_path, mmap_mode="r", allow_pickle=False)
        if signal.shape != (config.long_length,) or signal.dtype != np.float64:
            raise ValueError(f"{long_id}: invalid signal shape/dtype")
        label_lines = [line.split() for line in label_path.read_text().splitlines()]
        if len(label_lines) != int(row["n_events"]):
            raise ValueError(f"{long_id}: event count mismatch")
        if stratum == "background" and label_lines:
            raise ValueError(f"{long_id}: background row has labels")
        for fields in label_lines:
            if len(fields) != 3 or int(fields[0]) not in range(3):
                raise ValueError(f"{long_id}: invalid label row")
            class_id = int(fields[0])
            center = float(fields[1]) * config.long_length
            width = float(fields[2]) * config.long_length
            left = center - width / 2.0
            right = center + width / 2.0
            segment_index = int(min(left, config.long_length - 1)) // config.segment_length
            local_left = left - segment_index * config.segment_length
            local_right = right - segment_index * config.segment_length
            if (
                local_left < config.guard_samples - 1e-6
                or local_right > config.segment_length - config.guard_samples + 1e-6
            ):
                raise ValueError(f"{long_id}: event intersects a join guard")
            event_counts[(split, CLASS_NAMES[class_id])] += 1
    if split_ids["train"] & split_ids["val"]:
        raise ValueError("source ID crosses generated splits")
    if split_hashes["train"] & split_hashes["val"]:
        raise ValueError("source content crosses generated splits")
    expected_counts = {
        ("train", "positive"): (
            config.train_positive_groups * config.positive_permutations
        ),
        ("val", "positive"): (
            config.val_positive_groups * config.positive_permutations
        ),
        ("train", "background"): (
            config.train_background_groups * config.train_background_permutations
        ),
        ("val", "background"): (
            config.val_background_groups * config.val_background_permutations
        ),
    }
    if dict(counts) != expected_counts:
        raise ValueError(f"row count drift: expected {expected_counts}, got {dict(counts)}")
    return {
        "status": "pass",
        "row_counts": {
            split: {
                stratum: counts[(split, stratum)]
                for stratum in ("positive", "background")
            }
            for split in SPLITS
        },
        "event_counts": {
            split: {
                class_name: event_counts[(split, class_name)]
                for class_name in CLASS_NAMES
            }
            for split in SPLITS
        },
        "unique_source_counts": {
            split: len(split_ids[split]) for split in SPLITS
        },
        "cross_split_source_ids": 0,
        "cross_split_source_hashes": 0,
        "source_reuse_across_base_groups": 0,
        "events_intersecting_join_guards": 0,
        "sealed_test_accessed": False,
        "logical_manifest_sha256": _logical_manifest_hash(rows),
    }


def replay_generated_dataset(
    root: Path,
    plans: Mapping[str, Mapping[str, Sequence[tuple[SourceRef, ...]]]],
    noise_pool: FilteredNoisePool,
    config: Z8Wave8LikeConfig,
) -> dict[str, object]:
    with (Path(root) / "manifest.csv").open(encoding="utf-8", newline="") as handle:
        expected = list(csv.DictReader(handle))
    replay_hash = hashlib.sha256()
    rows_replayed = 0
    replay_iterator = _iter_generated_rows(plans, noise_pool, config)
    for rows_replayed, (expected_row, replayed_row) in enumerate(
        itertools.zip_longest(expected, replay_iterator), start=1
    ):
        if expected_row is None or replayed_row is None:
            raise ValueError("deterministic replay row count differs")
        replay_public = {
            field: replayed_row[field] for field in MANIFEST_FIELDS
        }
        if replay_public != expected_row:
            raise ValueError(
                "deterministic replay differs at row "
                f"{replayed_row.get('long_id', rows_replayed)}"
            )
        replay_hash.update(
            json.dumps(
                replay_public, sort_keys=True, separators=(",", ":")
            ).encode()
        )
        replay_hash.update(b"\n")
    return {
        "status": "pass",
        "rows_replayed": rows_replayed,
        "logical_manifest_sha256": replay_hash.hexdigest(),
    }


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def generate_dataset(
    *,
    z8_root: Path,
    parent_root: Path,
    noise_root: Path,
    output_root: Path,
    config: Z8Wave8LikeConfig,
    verify_replay: bool = True,
    fail_after_rows: int | None = None,
) -> dict[str, object]:
    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(f"refusing to mutate existing output: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    pools = load_source_pools(
        Path(parent_root), Path(z8_root) / "events.csv", config
    )
    endpoint_audit = endpoint_quality_pool_audit(pools, config)
    plans = _plan_groups(pools, config)
    noise_pool = FilteredNoisePool(Path(noise_root), config)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent)
    )
    try:
        for split in SPLITS:
            (temporary / split / "signals").mkdir(parents=True)
            (temporary / split / "labels").mkdir(parents=True)
        generated_rows: list[dict[str, object]] = []
        for row_index, row in enumerate(
            _iter_generated_rows(plans, noise_pool, config), start=1
        ):
            signal_path = (
                temporary
                / str(row["split"])
                / "signals"
                / f"{row['long_id']}.npy"
            )
            label_path = (
                temporary
                / str(row["split"])
                / "labels"
                / f"{row['long_id']}.txt"
            )
            np.save(signal_path, row.pop("_signal"), allow_pickle=False)
            label_path.write_text(str(row.pop("_label_text")), encoding="utf-8")
            generated_rows.append(row)
            if fail_after_rows is not None and row_index >= fail_after_rows:
                raise RuntimeError("injected generation failure")
        with (temporary / "manifest.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(generated_rows)

        audit = audit_generated_dataset(
            temporary, config, expected_rows=generated_rows
        )
        replay = (
            replay_generated_dataset(temporary, plans, noise_pool, config)
            if verify_replay
            else {"status": "not_run", "rows_replayed": 0}
        )
        dataset_contract = {
            "schema_version": 1,
            "dataset_id": config.output_dataset_id,
            "development_only": True,
            "sealed_test_accessed": False,
            "splits": list(SPLITS),
            "signal": {
                "format": "npy",
                "dtype": "float64",
                "shape": [config.long_length],
                "sampling_frequency_hz": config.sampling_frequency_hz,
            },
            "labels": {
                "format": "yolo-1d-class-center-width",
                "classes": list(CLASS_NAMES),
                "normalized_to": config.long_length,
                "duplicates_allowed": False,
            },
            "grain": "one long sequence per base-group permutation",
            "key": "long_id",
            "background_semantics": (
                "annotation-relative negative under registered Z8 v2; "
                "not certified particle-free"
            ),
        }
        _write_json(temporary / "dataset-contract.json", dataset_contract)
        dataset_yaml = {
            "schema_version": 1,
            "dataset_id": config.output_dataset_id,
            "format": "yolo-1d-long-sequence-z8-wave8like-development",
            "train": "train/signals",
            "val": "val/signals",
            "nc": len(CLASS_NAMES),
            "names": list(CLASS_NAMES),
            "sampling_frequency_hz": config.sampling_frequency_hz,
            "signal_lengths": {
                "source_segment": config.segment_length,
                "long_sequence": config.long_length,
            },
            "development_only": True,
            "sealed_test_accessed": False,
            "join": {
                "method": "filtered-continuous-noise-raised-cosine-bridge",
                "bridge_samples": config.bridge_length,
                "guard_samples_per_side": config.guard_samples,
                "noise_bandpass_hz": [
                    config.bandpass_low_hz,
                    config.bandpass_high_hz,
                ],
                "noise_bandpass_order": config.bandpass_order,
                "amplitude_matching": config.bridge_matching,
                "amplitude_context_samples": config.bridge_context_samples,
                "amplitude_context_excludes_annotations": True,
                "post_concat_global_filter": False,
            },
            "generation": {
                "seed": config.seed,
                "generator_revision": config.generator_revision,
                "source_disjoint_base_groups": True,
                "fully_labeled_physical_sources_only": True,
                "unclear_sources_excluded": True,
                "endpoint_quality": {
                    "enabled": config.endpoint_quality_enabled,
                    "window_samples": config.endpoint_quality_window_samples,
                    "max_rms_ratio": config.endpoint_max_rms_ratio,
                    "max_peak_robust_z": config.endpoint_max_peak_robust_z,
                    "annotations_excluded": True,
                    "both_endpoints_required": True,
                },
                "positive_groups": {
                    split: config.positive_groups(split) for split in SPLITS
                },
                "positive_permutations": config.positive_permutations,
                "background_groups": {
                    split: config.background_groups(split) for split in SPLITS
                },
                "background_permutations": {
                    split: config.background_permutations(split)
                    for split in SPLITS
                },
            },
        }
        _write_json(temporary / "dataset.yaml", dataset_yaml)
        dataset_manifest = {
            "schema_version": 1,
            "dataset_id": config.output_dataset_id,
            "status": "immutable_interim_candidate_awaiting_visual_join_audit",
            "parents": {
                "z8_event_table": config.z8_dataset_id,
                "signal_parent": config.parent_dataset_id,
                "noise_bridge_source": config.noise_dataset_id,
            },
            "generator_revision": config.generator_revision,
            "parameters": dataset_yaml["generation"],
            "join": dataset_yaml["join"],
            "payload": {
                "manifest_csv_sha256": sha256_file(temporary / "manifest.csv"),
                "dataset_contract_sha256": sha256_file(
                    temporary / "dataset-contract.json"
                ),
                "dataset_yaml_sha256": sha256_file(temporary / "dataset.yaml"),
            },
            "audit": audit,
            "endpoint_quality_audit": endpoint_audit,
            "deterministic_replay": replay,
            "promotion": {
                "status": "blocked",
                "reason": "visual join-audit result checkpoint not approved",
            },
        }
        _write_json(temporary / "dataset-manifest.json", dataset_manifest)
        os.replace(temporary, output_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return dataset_manifest
