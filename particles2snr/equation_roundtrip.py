from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DATASET_ID = "particles2snr-z8-equation-roundtrip@v1"
SOURCE_DATASET_ID = (
    "particles2snr-fbase-dual-clean-z8-events-3class-plus-unclear-development@v1"
)
SIGNAL_DATASET_ID = "particles2snr-f-c1-yolo-4class@v1"
CHECKPOINT_SHA256 = (
    "1c5524035d3ad36e3afc5e703875e8bc5226f7d57b1cd9730cb35c984bf495a1"
)
SAMPLING_FREQUENCY_HZ = 2_000_000.0
NATIVE_LENGTH = 4096
MODEL_INPUT_LENGTH = 512
FWHM_TO_SIGMA = 2.355
WIDTH_VARIANTS = (
    "fitted_tau_as_sigma",
    "fitted_tau_as_fwhm",
    "annotation_width_as_fwhm",
    "annotation_width_as_sigma",
)
SHUFFLE_VARIANTS = (
    "shuffle_amplitude",
    "shuffle_frequency",
    "shuffle_tau",
    "shuffle_snr",
)
ALL_VARIANTS = (*WIDTH_VARIANTS, *SHUFFLE_VARIANTS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def centered_crop(signal: np.ndarray, center_sample: float, length: int = NATIVE_LENGTH) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float32).reshape(-1)
    if length <= 0:
        raise ValueError("crop length must be positive")
    start = int(round(float(center_sample))) - length // 2
    stop = start + length
    output = np.zeros(length, dtype=np.float32)
    source_start = max(0, start)
    source_stop = min(values.size, stop)
    if source_stop > source_start:
        target_start = source_start - start
        output[target_start : target_start + source_stop - source_start] = values[
            source_start:source_stop
        ]
    return output


def classifier_preprocess(signal: np.ndarray) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float32).reshape(-1)
    if values.size != NATIVE_LENGTH:
        raise ValueError(f"expected {NATIVE_LENGTH} native samples, got {values.size}")
    reduced = values.reshape(MODEL_INPUT_LENGTH, NATIVE_LENGTH // MODEL_INPUT_LENGTH).mean(axis=1)
    reduced = reduced.astype(np.float32, copy=False)
    reduced -= float(reduced.mean())
    scale = float(reduced.std())
    if not np.isfinite(scale) or scale <= 1.0e-8:
        raise ValueError("cannot z-score a constant or non-finite signal")
    reduced /= scale
    if not np.isfinite(reduced).all():
        raise ValueError("preprocessing produced non-finite values")
    return reduced


def annotation_width_ms(event: dict[str, Any]) -> float:
    return (
        float(event["end_sample"]) - float(event["start_sample"])
    ) / SAMPLING_FREQUENCY_HZ * 1000.0


def sigma_ms(event: dict[str, Any], variant: str) -> float:
    tau_ms = float(event["tau_ms"])
    width_ms = annotation_width_ms(event)
    if variant in {"fitted_tau_as_sigma", *SHUFFLE_VARIANTS}:
        value = tau_ms
    elif variant == "fitted_tau_as_fwhm":
        value = tau_ms / FWHM_TO_SIGMA
    elif variant == "annotation_width_as_fwhm":
        value = width_ms / FWHM_TO_SIGMA
    elif variant == "annotation_width_as_sigma":
        value = width_ms
    else:
        raise ValueError(f"unknown equation variant: {variant}")
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"invalid sigma for {variant}: {value}")
    return float(value)


def deterministic_seed(
    base_seed: int, event_id: str, variant: str, view_index: int
) -> int:
    payload = f"{base_seed}:{event_id}:{variant}:{view_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def synthesize_equation_view(
    *,
    amplitude: float,
    frequency_hz: float,
    sigma_ms_value: float,
    snr_db: float,
    phase_rad: float,
    seed: int,
    length: int = NATIVE_LENGTH,
    sampling_frequency_hz: float = SAMPLING_FREQUENCY_HZ,
) -> np.ndarray:
    parameters = np.asarray(
        [amplitude, frequency_hz, sigma_ms_value, snr_db, phase_rad],
        dtype=np.float64,
    )
    if not np.isfinite(parameters).all():
        raise ValueError("equation parameters must be finite")
    if amplitude <= 0.0 or frequency_hz <= 0.0 or sigma_ms_value <= 0.0:
        raise ValueError("amplitude, frequency, and sigma must be positive")
    if length <= 0 or sampling_frequency_hz <= 0.0:
        raise ValueError("length and sampling frequency must be positive")
    time = (np.arange(length, dtype=np.float64) - (length - 1) / 2.0) / sampling_frequency_hz
    sigma_seconds = sigma_ms_value / 1000.0
    envelope = np.exp(-np.square(time) / (2.0 * sigma_seconds**2))
    clean = amplitude * np.cos(2.0 * np.pi * frequency_hz * time + phase_rad) * envelope
    clean_rms = float(np.sqrt(np.mean(np.square(clean))))
    if clean_rms <= 1.0e-12:
        raise ValueError("equation clean signal has zero energy")
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=length)
    noise -= float(noise.mean())
    noise /= max(float(noise.std()), 1.0e-12)
    noise *= clean_rms / (10.0 ** (snr_db / 20.0))
    result = (clean + noise).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("equation generated non-finite values")
    return result


def _read_events(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    physical = [
        row
        for row in rows
        if row["class_name"] in {"2um", "4um", "10um"}
        and row["split"] in {"train", "val"}
    ]
    if not physical:
        raise ValueError("no eligible physical development events")
    return physical


def _shuffle_sources(
    events: list[dict[str, str]], *, seed: int
) -> dict[str, dict[str, dict[str, str]]]:
    rng = np.random.default_rng(seed)
    result: dict[str, dict[str, dict[str, str]]] = {
        name: {} for name in SHUFFLE_VARIANTS
    }
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for event in events:
        grouped[(event["split"], event["class_name"])].append(event)
    for key in sorted(grouped):
        rows = sorted(grouped[key], key=lambda row: row["event_id"])
        if len(rows) < 2:
            raise ValueError(f"shuffle stratum has fewer than two events: {key}")
        for variant in SHUFFLE_VARIANTS:
            order = np.arange(len(rows))
            for _attempt in range(100):
                rng.shuffle(order)
                if np.all(order != np.arange(len(rows))):
                    break
            else:
                order = np.roll(np.arange(len(rows)), 1)
            for target, source_index in zip(rows, order, strict=True):
                result[variant][target["event_id"]] = rows[int(source_index)]
    return result


def _variant_parameters(
    event: dict[str, str],
    variant: str,
    shuffled: dict[str, dict[str, dict[str, str]]],
) -> tuple[float, float, float, float, str | None]:
    amplitude = float(event["particles2snr_amplitude"])
    frequency = float(event["frequency_hz"])
    tau = float(event["tau_ms"])
    snr = float(event["snr_db"])
    source_event_id: str | None = None
    if variant in SHUFFLE_VARIANTS:
        source = shuffled[variant][event["event_id"]]
        source_event_id = source["event_id"]
        if variant == "shuffle_amplitude":
            amplitude = float(source["particles2snr_amplitude"])
        elif variant == "shuffle_frequency":
            frequency = float(source["frequency_hz"])
        elif variant == "shuffle_tau":
            tau = float(source["tau_ms"])
        elif variant == "shuffle_snr":
            snr = float(source["snr_db"])
    synthetic_event = dict(event)
    synthetic_event["tau_ms"] = str(tau)
    return amplitude, frequency, sigma_ms(synthetic_event, variant), snr, source_event_id


def build_equation_roundtrip_candidate(
    *,
    event_table_root: Path,
    signal_dataset_root: Path,
    output_dir: Path,
    source_manifest_sha256: str,
    signal_manifest_sha256: str,
    checkpoint_sha256: str,
    views_per_event: int = 8,
    seed: int = 20260723,
    maximum_events: int | None = None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite candidate: {output_dir}")
    if checkpoint_sha256 != CHECKPOINT_SHA256:
        raise ValueError("classifier checkpoint hash does not match the frozen contract")
    if views_per_event != 8:
        raise ValueError("the frozen protocol requires exactly eight views per event")
    events = _read_events(event_table_root / "events.csv")
    events.sort(key=lambda row: (row["split"], row["class_name"], row["event_id"]))
    shuffled = _shuffle_sources(events, seed=seed + 1)
    if maximum_events is not None:
        if maximum_events <= 0:
            raise ValueError("maximum_events must be positive")
        events = events[:maximum_events]
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    rows: list[dict[str, Any]] = []
    signals: list[np.ndarray] = []
    raw_cache: dict[str, np.ndarray] = {}
    for event in events:
        relative = str(event["source_signal_relative_path"])
        raw = raw_cache.get(relative)
        if raw is None:
            path = signal_dataset_root / relative
            raw = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
            raw_cache[relative] = raw
        real_crop = centered_crop(raw, float(event["center_norm"]) * raw.size)
        real_model_input = classifier_preprocess(real_crop)
        real_row = len(signals)
        signals.append(real_model_input)
        rows.append(
            {
                "signal_row": real_row,
                "record_kind": "real_gallery",
                "event_id": event["event_id"],
                "source_event_id": event["event_id"],
                "shuffled_parameter_source_event_id": "",
                "split": event["split"],
                "class_name": event["class_name"],
                "annotation_origin": event["annotation_origin"],
                "source_group": event["source_filename"],
                "variant": "real",
                "view_index": -1,
                "phase_rad": "",
                "amplitude": event["particles2snr_amplitude"],
                "frequency_hz": event["frequency_hz"],
                "fitted_tau_ms": event["tau_ms"],
                "annotation_width_ms": annotation_width_ms(event),
                "sigma_ms": "",
                "snr_db": event["snr_db"],
                "seed": "",
            }
        )
        for variant in ALL_VARIANTS:
            amplitude, frequency, sigma, snr, shuffle_source = _variant_parameters(
                event, variant, shuffled
            )
            for view_index in range(views_per_event):
                view_seed = deterministic_seed(
                    seed, event["event_id"], variant, view_index
                )
                phase = 2.0 * np.pi * (view_index + 0.5) / views_per_event
                native = synthesize_equation_view(
                    amplitude=amplitude,
                    frequency_hz=frequency,
                    sigma_ms_value=sigma,
                    snr_db=snr,
                    phase_rad=phase,
                    seed=view_seed,
                )
                signal_row = len(signals)
                signals.append(classifier_preprocess(native))
                rows.append(
                    {
                        "signal_row": signal_row,
                        "record_kind": "synthetic_query",
                        "event_id": f"{event['event_id']}:{variant}:view-{view_index}",
                        "source_event_id": event["event_id"],
                        "shuffled_parameter_source_event_id": shuffle_source or "",
                        "split": event["split"],
                        "class_name": event["class_name"],
                        "annotation_origin": event["annotation_origin"],
                        "source_group": event["source_filename"],
                        "variant": variant,
                        "view_index": view_index,
                        "phase_rad": phase,
                        "amplitude": amplitude,
                        "frequency_hz": frequency,
                        "fitted_tau_ms": event["tau_ms"],
                        "annotation_width_ms": annotation_width_ms(event),
                        "sigma_ms": sigma,
                        "snr_db": snr,
                        "seed": view_seed,
                    }
                )
    matrix = np.stack(signals).astype(np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("candidate contains non-finite model inputs")
    np.save(output_dir / "signals.npy", matrix, allow_pickle=False)
    with (output_dir / "records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "source_datasets": {
            SOURCE_DATASET_ID: source_manifest_sha256,
            SIGNAL_DATASET_ID: signal_manifest_sha256,
        },
        "classifier_checkpoint_sha256": checkpoint_sha256,
        "event_count": len(events),
        "record_count": len(rows),
        "signal_shape": list(matrix.shape),
        "signal_dtype": str(matrix.dtype),
        "views_per_event": views_per_event,
        "variants": list(ALL_VARIANTS),
        "seed": seed,
        "class_counts": dict(sorted(Counter(row["class_name"] for row in events).items())),
        "origin_counts": dict(
            sorted(Counter(row["annotation_origin"] for row in events).items())
        ),
        "split_counts": dict(sorted(Counter(row["split"] for row in events).items())),
        "sealed_test_accessed": False,
        "preprocessing": (
            "4096-sample crop centered on detected event; mean decimate by 8; "
            "per-window z-score"
        ),
        "claim_boundary": (
            "The candidate tests equation-to-classifier consistency. Window "
            "z-scoring prevents independent validation of absolute amplitude "
            "or physical SNR calibration."
        ),
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    contract = {
        "schema_version": 1,
        "format": "particles2snr-equation-roundtrip-classifier-input",
        "sampling_frequency_hz": SAMPLING_FREQUENCY_HZ,
        "native_length": NATIVE_LENGTH,
        "model_input_length": MODEL_INPUT_LENGTH,
        "splits": ["train", "val"],
        "sealed_splits": ["test"],
        "width_variants": list(WIDTH_VARIANTS),
        "shuffle_variants": list(SHUFFLE_VARIANTS),
        "parameter_units": {
            "amplitude": "particles2SNR P0 units",
            "frequency_hz": "Hz",
            "fitted_tau_ms": "ms; detector Gaussian sigma",
            "annotation_width_ms": "ms",
            "sigma_ms": "ms",
            "snr_db": "dB",
            "phase_rad": "radian",
        },
    }
    (output_dir / "input_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        **summary,
        "files": {
            path.name: sha256_file(path)
            for path in sorted(output_dir.iterdir())
            if path.is_file()
        },
    }


def validate_equation_roundtrip_candidate(root: Path) -> dict[str, Any]:
    summary = json.loads((root / "dataset_summary.json").read_text(encoding="utf-8"))
    contract = json.loads((root / "input_contract.json").read_text(encoding="utf-8"))
    with (root / "records.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    signals = np.load(root / "signals.npy", mmap_mode="r", allow_pickle=False)
    if summary["dataset_id"] != DATASET_ID:
        raise ValueError("candidate dataset ID changed")
    if contract["sealed_splits"] != ["test"]:
        raise ValueError("sealed split contract changed")
    if any(row["split"] == "test" for row in rows):
        raise PermissionError("candidate contains sealed test rows")
    if len(rows) != len(signals) or len(rows) != summary["record_count"]:
        raise ValueError("record/signal count mismatch")
    indices = [int(row["signal_row"]) for row in rows]
    if indices != list(range(len(rows))):
        raise ValueError("signal rows are not contiguous and aligned")
    keys = [
        (row["record_kind"], row["event_id"], row["variant"], row["view_index"])
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate candidate record key")
    if signals.shape[1:] != (MODEL_INPUT_LENGTH,) or signals.dtype != np.float32:
        raise ValueError("candidate signal contract changed")
    if not np.isfinite(np.asarray(signals)).all():
        raise ValueError("candidate contains non-finite signals")
    return {
        "valid": True,
        "record_count": len(rows),
        "event_count": summary["event_count"],
        "signal_shape": list(signals.shape),
        "sealed_test_accessed": False,
    }
