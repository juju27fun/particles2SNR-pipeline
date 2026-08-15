"""Deterministic, fold-specific Wave8like recipes from the gradual beads ledger."""

from __future__ import annotations

import hashlib
import itertools
import json
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .z8_wave8like_dataset import (
    EventRef,
    apply_raised_cosine_bridge,
    match_bridge_to_local_rms,
)


DATASET_ID = "particles2snr-beads-gradual-wave8like-development@v2"
SCHEMA = "LedgerWave8ReplayManifestV1"
SEED = 42
SEGMENT_LENGTH = 16_384
SEGMENTS_PER_SEQUENCE = 4
LONG_LENGTH = SEGMENT_LENGTH * SEGMENTS_PER_SEQUENCE
GUARD = 300
ENDPOINT_WINDOW = 900
POSITIVE_GROUPS = 15
POSITIVE_PERMUTATIONS = 24
NEGATIVE_RECIPES = 120
NOISE_RESERVE_COUNT = 61
FOLD_SALT = "particle-ledger-wave8like-r1"
CLASS_NAMES = ("2um", "4um", "10um")


class LedgerWave8Error(ValueError):
    """Raised when a replay recipe would violate the frozen method."""


@dataclass(frozen=True)
class Source:
    source_id: str
    source_class: str
    class_id: int
    signal_path: str
    signal_sha256: str
    events: tuple[dict[str, Any], ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    value = "|".join((FOLD_SALT, str(SEED), *(str(part) for part in parts)))
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def partition_noise(noise_root: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rows = [
        {
            "noise_id": path.stem,
            "path": path.as_posix(),
            "sha256": sha256_file(path),
        }
        for path in sorted(noise_root.rglob("*.npy"))
    ]
    if len(rows) != 305:
        raise LedgerWave8Error(f"expected 305 noise traces, got {len(rows)}")
    by_hash: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_hash[row["sha256"]].append(row)
    groups = sorted(
        by_hash.values(),
        key=lambda group: hashlib.sha256(f"noise-reserve-r1:{group[0]['sha256']}".encode()).hexdigest(),
    )
    reserved_groups: list[list[dict[str, str]]] = []
    reserved_count = 0
    for group in groups:
        if reserved_count + len(group) <= NOISE_RESERVE_COUNT:
            reserved_groups.append(group)
            reserved_count += len(group)
        if reserved_count == NOISE_RESERVE_COUNT:
            break
    if reserved_count != NOISE_RESERVE_COUNT:
        raise LedgerWave8Error(f"cannot create exact hash-grouped reserve of {NOISE_RESERVE_COUNT}")
    reserved_hashes = {group[0]["sha256"] for group in reserved_groups}
    reserved = [row for group in reserved_groups for row in group]
    training = [row for row in rows if row["sha256"] not in reserved_hashes]
    if set(row["sha256"] for row in reserved) & set(row["sha256"] for row in training):
        raise LedgerWave8Error("noise hash partition overlaps")
    return training, reserved


def load_sources(ledger_root: Path) -> tuple[dict[str, Source], dict[str, int], set[str]]:
    traces = {row["source_id"]: row for row in _jsonl(ledger_root / "traces.jsonl")}
    events_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ambiguous_sources: set[str] = set()
    for event in _jsonl(ledger_root / "events.jsonl"):
        if event["identity_state"] == "ambiguous_event":
            ambiguous_sources.add(event["source_id"])
        else:
            events_by_source[event["source_id"]].append(event)
    controls = {row["source_id"] for row in _jsonl(ledger_root / "controls.jsonl")}
    sources = {
        source_id: Source(
            source_id=source_id,
            source_class=trace["source_class"],
            class_id=int(trace["class_id"]),
            signal_path=trace["signal_path"],
            signal_sha256=trace["signal_sha256"],
            events=tuple(sorted(events_by_source[source_id], key=lambda row: (row["center"], row["event_id"]))),
        )
        for source_id, trace in traces.items()
    }
    fold_by_source = {
        source_id: int(trace["audit_fold"])
        for source_id, trace in traces.items()
        if trace["audit_fold"] is not None
    }
    return sources, fold_by_source, controls | ambiguous_sources


def source_is_eligible(source: Source) -> bool:
    return bool(source.events) and all(
        event["support_start"] is not None
        and event["support_end"] is not None
        and int(event["support_start"]) >= GUARD
        and int(event["support_end"]) <= SEGMENT_LENGTH - GUARD
        for event in source.events
    )


def endpoint_quality_is_safe(source: Source, workspace: Path) -> bool:
    """Apply the exact Wave8like-v4 endpoint gate with annotations excluded."""
    signal = np.asarray(
        np.load(workspace / source.signal_path, allow_pickle=False), dtype=float
    ).reshape(-1)
    keep = np.ones(len(signal), dtype=bool)
    for event in source.events:
        keep[int(event["support_start"]) : int(event["support_end"])] = False
    baseline = signal[keep]
    median = float(np.median(baseline))
    global_scale = max(
        1.4826 * float(np.median(np.abs(baseline - median))), 1e-12
    )
    for start, end in (
        (0, ENDPOINT_WINDOW),
        (len(signal) - ENDPOINT_WINDOW, len(signal)),
    ):
        values = signal[start:end][keep[start:end]]
        if len(values) < GUARD:
            return False
        local_median = float(np.median(values))
        local_scale = 1.4826 * float(np.median(np.abs(values - local_median)))
        peak_robust_z = float(np.max(np.abs(values - median)) / global_scale)
        if local_scale / global_scale > 2.5 or peak_robust_z > 8.0:
            return False
    return True


def choose_positive_groups(sources: Sequence[Source], fold: int) -> list[tuple[Source, ...]]:
    by_class = {name: [source for source in sources if source.source_class == name] for name in CLASS_NAMES}
    for class_name, rows in by_class.items():
        rng = np.random.default_rng(stable_seed(fold, class_name, "sources"))
        rng.shuffle(rows)
        if len(rows) < POSITIVE_GROUPS + 5:
            raise LedgerWave8Error(f"insufficient {class_name} sources for fold {fold}: {len(rows)}")
    cursors = Counter({name: 0 for name in CLASS_NAMES})
    groups: list[tuple[Source, ...]] = []
    fourth_order = [CLASS_NAMES[index % 3] for index in range(POSITIVE_GROUPS)]
    for group_index, fourth_class in enumerate(fourth_order):
        chosen = []
        for class_name in (*CLASS_NAMES, fourth_class):
            chosen.append(by_class[class_name][cursors[class_name]])
            cursors[class_name] += 1
        rng = np.random.default_rng(stable_seed(fold, group_index, "group-order"))
        rng.shuffle(chosen)
        groups.append(tuple(chosen))
    if len({source.source_id for group in groups for source in group}) != POSITIVE_GROUPS * 4:
        raise LedgerWave8Error("positive groups are not source-disjoint")
    return groups


def _bridge_recipe(training_noise: Sequence[dict[str, str]], recipe_id: str, boundary: int) -> dict[str, Any]:
    rng = np.random.default_rng(stable_seed(recipe_id, boundary, "bridge"))
    source = training_noise[int(rng.integers(len(training_noise)))]
    crop_start = int(rng.integers(SEGMENT_LENGTH - 2 * GUARD + 1))
    return {**source, "crop_start": crop_start, "length": 2 * GUARD, "boundary_index": boundary}


def _masks() -> list[list[int]]:
    return [[boundary - GUARD, boundary + GUARD] for boundary in (SEGMENT_LENGTH, 2 * SEGMENT_LENGTH, 3 * SEGMENT_LENGTH)]


def _positive_recipe(
    fold: int,
    group_id: int,
    permutation_index: int,
    ordered: Sequence[Source],
    training_noise: Sequence[dict[str, str]],
) -> dict[str, Any]:
    recipe_id = f"ledgerw8_f{fold}_positive_{group_id:03d}_p{permutation_index:02d}"
    events = []
    for position, source in enumerate(ordered):
        offset = position * SEGMENT_LENGTH
        for event in source.events:
            events.append(
                {
                    **event,
                    "center": int(event["center"]) + offset,
                    "support_start": int(event["support_start"]) + offset,
                    "support_end": int(event["support_end"]) + offset,
                    "segment_position": position,
                }
            )
    return {
        "schema": SCHEMA,
        "recipe_id": recipe_id,
        "fold": fold,
        "stratum": "positive",
        "group_id": group_id,
        "permutation_index": permutation_index,
        "segments": [source.__dict__ | {"events": list(source.events)} for source in ordered],
        "bridges": [_bridge_recipe(training_noise, recipe_id, boundary) for boundary in (1, 2, 3)],
        "events": sorted(events, key=lambda row: (row["center"], row["event_id"])),
        "objectness_policy": "positive_cells_only",
        "masked_intervals": _masks(),
        "signal_length": LONG_LENGTH,
    }


def _negative_recipes(fold: int, training_noise: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    exposures = [row for row in training_noise for _ in range(2)]
    rng = np.random.default_rng(stable_seed(fold, "negative-exposures"))
    rng.shuffle(exposures)
    exposures = exposures[: NEGATIVE_RECIPES * SEGMENTS_PER_SEQUENCE]
    counts = Counter(row["noise_id"] for row in exposures)
    if max(counts.values()) > 2:
        raise LedgerWave8Error("noise exposure cap exceeded")
    recipes = []
    for index in range(NEGATIVE_RECIPES):
        segments = exposures[index * 4 : (index + 1) * 4]
        recipe_id = f"ledgerw8_f{fold}_negative_{index:03d}"
        recipes.append(
            {
                "schema": SCHEMA,
                "recipe_id": recipe_id,
                "fold": fold,
                "stratum": "negative",
                "group_id": index,
                "permutation_index": 0,
                "segments": [
                    {
                        "source_id": row["noise_id"],
                        "source_class": "noise",
                        "class_id": -1,
                        "signal_path": row["path"],
                        "signal_sha256": row["sha256"],
                        "events": [],
                    }
                    for row in segments
                ],
                "bridges": [_bridge_recipe(training_noise, recipe_id, boundary) for boundary in (1, 2, 3)],
                "events": [],
                "objectness_policy": "all_cells_negative_except_masks",
                "masked_intervals": _masks(),
                "signal_length": LONG_LENGTH,
            }
        )
    return recipes


def build_recipes(*, workspace: Path, ledger_root: Path, noise_root: Path, output_root: Path) -> dict[str, Any]:
    workspace, ledger_root, noise_root, output_root = map(Path.resolve, (workspace, ledger_root, noise_root, output_root))
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite immutable dataset: {output_root}")
    sources, fold_by_source, excluded_sources = load_sources(ledger_root)
    training_noise, reserved_noise = partition_noise(noise_root)
    for row in (*training_noise, *reserved_noise):
        path = Path(row["path"])
        row["path"] = path.relative_to(workspace).as_posix()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        all_counts = {}
        for fold in range(5):
            validation_fold = (fold + 1) % 5
            held_sources = excluded_sources | {
                source_id for source_id, assigned in fold_by_source.items() if assigned in {fold, validation_fold}
            }
            eligible = [
                source
                for source in sources.values()
                if source.source_id not in held_sources
                and source_is_eligible(source)
                and endpoint_quality_is_safe(source, workspace)
            ]
            groups = choose_positive_groups(eligible, fold)
            positives = []
            permutations = list(itertools.permutations(range(4)))
            for group_id, group in enumerate(groups):
                for permutation_index, permutation in enumerate(permutations):
                    positives.append(
                        _positive_recipe(
                            fold,
                            group_id,
                            permutation_index,
                            [group[index] for index in permutation],
                            training_noise,
                        )
                    )
            negatives = _negative_recipes(fold, training_noise)
            rows = sorted((*positives, *negatives), key=lambda row: row["recipe_id"])
            _write_jsonl(temporary / f"fold_{fold}.jsonl", rows)
            all_counts[str(fold)] = {
                "positive": len(positives),
                "negative": len(negatives),
                "total": len(rows),
                "eligible_sources": len(eligible),
                "validation_fold": validation_fold,
                "unique_positive_sources": len({segment["source_id"] for row in positives for segment in row["segments"]}),
            }
        _write_jsonl(temporary / "noise_training.jsonl", training_noise)
        _write_jsonl(temporary / "noise_reserved.jsonl", reserved_noise)
        contract = {
            "schema_version": 1,
            "dataset_id": DATASET_ID,
            "format": SCHEMA,
            "materialization": "deterministic_on_demand",
            "segment_length": SEGMENT_LENGTH,
            "long_length": LONG_LENGTH,
            "guard_samples": GUARD,
            "endpoint_window_samples": ENDPOINT_WINDOW,
            "bandpass_hz": [7_000, 80_000],
            "bridge_matching": "robust-local-rms-global-cap",
            "positive_cells_policy": "unannotated cells unknown",
            "negative_cells_policy": "noise cells negative except join guards",
            "noise_absence_basis": "protocol d’acquisition confirmé par Louis",
        }
        _write_json(temporary / "dataset-contract.json", contract)
        summary = {
            "schema_version": 1,
            "dataset_id": DATASET_ID,
            "status": "candidate_pending_join_qualification",
            "folds": all_counts,
            "noise": {"total": 305, "training": len(training_noise), "reserved": len(reserved_noise), "max_negative_exposures": 2},
            "ambiguous_sources_excluded": len(excluded_sources),
            "source_dataset_id": "particles2snr-beads-gradual-supervision-development@v2",
        }
        _write_json(temporary / "summary.json", summary)
        _write_json(
            temporary / "run.json",
            {
                "schema_version": 1,
                "run_id": "particles2snr-beads-gradual-wave8like-development-v2-build",
                "kind": "ledger-wave8like-recipe-build",
                "status": "complete_candidate",
                "dataset": DATASET_ID,
                "source_run_ids": ["particle-gradual-wave8like-method-r1", "particle-spectral-npy-targeted-ledger-analysis-r3"],
                "sealed_test_accessed": False,
                "gpu_training_authorized": False,
            },
        )
        files = [
            {"path": path.name, "sha256": sha256_file(path), "size": path.stat().st_size}
            for path in sorted(temporary.iterdir()) if path.is_file()
        ]
        _write_json(
            temporary / "dataset-manifest.json",
            {
                "schema_version": 1,
                "dataset_id": DATASET_ID,
                "parents": ["particles2snr-beads-gradual-supervision-development@v2", "noise@v1"],
                "files": files,
                "computation_fingerprint": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
            },
        )
        output_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(temporary), output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return summary


def materialize_recipe(
    recipe: dict[str, Any],
    workspace: Path,
    *,
    signal_cache: dict[str, np.ndarray] | None = None,
    filtered_noise_cache: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Materialize one recipe exactly; used by qualification and model replay."""
    signal_cache = {} if signal_cache is None else signal_cache
    filtered_noise_cache = {} if filtered_noise_cache is None else filtered_noise_cache
    segments = []
    local_events: list[list[EventRef]] = []
    for segment in recipe["segments"]:
        path = workspace / segment["signal_path"]
        if sha256_file(path) != segment["signal_sha256"]:
            raise LedgerWave8Error(f"signal hash drift: {segment['source_id']}")
        if segment["signal_sha256"] not in signal_cache:
            signal_cache[segment["signal_sha256"]] = np.asarray(
                np.load(path, allow_pickle=False), dtype=np.float64
            ).reshape(-1)
        values = signal_cache[segment["signal_sha256"]]
        if values.shape != (SEGMENT_LENGTH,) or not np.isfinite(values).all():
            raise LedgerWave8Error(f"invalid segment: {segment['source_id']}")
        segments.append(values.copy())
        local_events.append(
            [
                EventRef(
                    event_id=event["event_id"], class_id=int(event["class_id"]),
                    class_name=segment["source_class"], left=float(event["support_start"]), right=float(event["support_end"]),
                )
                for event in segment.get("events", [])
            ]
        )
    nyquist = 1_000_000.0
    sos = butter(4, [7_000 / nyquist, 80_000 / nyquist], btype="bandpass", output="sos")
    for boundary_index, bridge in enumerate(recipe["bridges"]):
        path = workspace / bridge["path"]
        if sha256_file(path) != bridge["sha256"]:
            raise LedgerWave8Error("bridge noise hash drift")
        if bridge["sha256"] not in filtered_noise_cache:
            noise = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64).reshape(-1)
            filtered_noise_cache[bridge["sha256"]] = sosfiltfilt(sos, noise)
        filtered = filtered_noise_cache[bridge["sha256"]]
        start = int(bridge["crop_start"])
        values = filtered[start : start + 2 * GUARD].copy()
        values, _ = match_bridge_to_local_rms(
            segments[boundary_index], segments[boundary_index + 1], values,
            left_events=local_events[boundary_index], right_events=local_events[boundary_index + 1],
            guard_samples=GUARD, context_samples=2_400, cap_by_global=True,
        )
        segments[boundary_index], segments[boundary_index + 1] = apply_raised_cosine_bridge(
            segments[boundary_index], segments[boundary_index + 1], values, GUARD
        )
    result = np.concatenate(segments)
    if result.shape != (LONG_LENGTH,) or not np.isfinite(result).all():
        raise LedgerWave8Error("materialized recipe is invalid")
    return result
