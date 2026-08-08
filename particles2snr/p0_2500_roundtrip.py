from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from scipy.signal import correlate

from particles2snr.equation_roundtrip import (
    NATIVE_LENGTH,
    SAMPLING_FREQUENCY_HZ,
    deterministic_seed,
    sha256_file,
)
from particles2snr.equation_roundtrip_v2 import (
    _analytic_clean,
    detector_spectral_calibration,
    lowest_energy_windows,
)


DATASET_ID = "particles2snr-p0-2500-equation-roundtrip@v1"
P0_DATASET_ID = "p0-baseline-3class@v1"
RAW_DATASET_ID = "particles2snr-f-c1-yolo-4class@v1"
METHOD_EVIDENCE_ID = "p0-2500-parent-retrieval-method-v1"
CHECKPOINT_SHA256 = (
    "c118a5aa593c3d9f982a3d32050ba8853143c284c33eb64ec39e4f1f18762082"
)
RAW_CROP_LENGTH = 2500
MODEL_INPUT_LENGTH = 625
VIEWS_PER_PARENT = 8
PRIMARY_MIN_CORRELATION = 0.50
PRIMARY_MAX_CENTER_DISTANCE = 625.0
CLASS_NAMES = ("2um", "4um", "10um")
_CROP_NAME = re.compile(r"^(.*\.npy)(\d+)\.npy$")


def parse_parent_crop_name(name: str) -> tuple[str, int]:
    match = _CROP_NAME.fullmatch(name)
    if match is None:
        raise ValueError(f"unexpected p0 crop filename: {name}")
    return match.group(1), int(match.group(2))


def normalized_valid_correlation(
    source: np.ndarray, crop: np.ndarray
) -> tuple[int, float]:
    raw = np.asarray(source, dtype=np.float64).reshape(-1)
    query = np.asarray(crop, dtype=np.float64).reshape(-1)
    if query.size > raw.size or query.size < 2:
        raise ValueError("invalid source/crop lengths")
    centered = query - float(query.mean())
    query_energy = float(np.sum(np.square(centered)))
    if query_energy <= 1.0e-18:
        raise ValueError("constant crop cannot be aligned")
    numerator = correlate(raw, centered, mode="valid", method="fft")
    prefix = np.concatenate(([0.0], np.cumsum(raw)))
    prefix2 = np.concatenate(([0.0], np.cumsum(np.square(raw))))
    width = query.size
    sums = prefix[width:] - prefix[:-width]
    sums2 = prefix2[width:] - prefix2[:-width]
    window_energy = np.maximum(sums2 - np.square(sums) / width, 0.0)
    denominator = np.sqrt(window_energy * query_energy)
    scores = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, -np.inf),
        where=denominator > 1.0e-18,
    )
    start = int(np.argmax(scores))
    score = float(scores[start])
    if not np.isfinite(score):
        raise ValueError("crop alignment produced no finite correlation")
    return start, score


def legacy_preprocess_2500(signals: np.ndarray) -> np.ndarray:
    values = np.asarray(signals, dtype=np.float32)
    squeeze = values.ndim == 1
    if squeeze:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != RAW_CROP_LENGTH:
        raise ValueError("legacy preprocessing expects shape (n, 2500)")
    tensor = torch.from_numpy(np.array(values, dtype=np.float32, copy=True))
    spectrum = torch.fft.fft(tensor, dim=-1)
    frequencies = torch.fft.fftfreq(
        RAW_CROP_LENGTH, d=1.0 / SAMPLING_FREQUENCY_HZ
    )
    mask = (torch.abs(frequencies) >= 5_000.0) & (
        torch.abs(frequencies) <= 100_000.0
    )
    reduced = torch.fft.ifft(spectrum * mask, dim=-1).real[..., ::4]
    result = reduced.cpu().numpy().astype(np.float32, copy=False)
    if result.shape[1] != MODEL_INPUT_LENGTH or not np.isfinite(result).all():
        raise ValueError("legacy preprocessing contract changed")
    return result[0] if squeeze else result


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _raw_file_index(raw_dataset_root: Path) -> dict[str, tuple[str, Path]]:
    output: dict[str, tuple[str, Path]] = {}
    for split in ("train", "val"):
        signal_root = raw_dataset_root / split / "signals"
        for path in sorted(signal_root.glob("*.npy")):
            if path.name in output:
                raise ValueError(f"duplicate development raw filename: {path.name}")
            output[path.name] = (split, path)
    return output


def discover_parent_provenance(
    *,
    p0_dataset_root: Path,
    raw_dataset_root: Path,
    detector_particles_csv: Path,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    if any(part.lower() == "test" for part in detector_particles_csv.parts):
        raise PermissionError("detector provenance may not come from test data")
    raw_index = _raw_file_index(raw_dataset_root)
    detector_rows = _read_csv(detector_particles_csv)
    by_filename: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in detector_rows:
        if row["snr_method"] != "peak_bin_energy_over_lowest_window_energy":
            raise ValueError("detector SNR method changed")
        by_filename[row["filename"]].append(row)

    rows: list[dict[str, Any]] = []
    raw_cache: dict[str, np.ndarray] = {}
    for class_name in CLASS_NAMES:
        for crop_path in sorted((p0_dataset_root / "train" / class_name).glob("*.npy")):
            source_filename, legacy_crop_index = parse_parent_crop_name(crop_path.name)
            raw_match = raw_index.get(source_filename)
            if raw_match is None:
                continue
            split, raw_path = raw_match
            # The frozen method uses the detector-training provenance table.
            # Files assigned to the registered raw validation split are retained
            # only in the discovery accounting and are not eligible parents.
            if split != "train":
                continue
            particles = by_filename.get(source_filename, [])
            if not particles:
                continue
            raw = raw_cache.setdefault(
                source_filename,
                np.load(raw_path, allow_pickle=False).astype(np.float32, copy=False),
            )
            crop = np.load(crop_path, allow_pickle=False).astype(np.float32, copy=False)
            if crop.shape != (RAW_CROP_LENGTH,):
                raise ValueError(f"unexpected p0 crop shape: {crop_path}")
            crop_start, correlation = normalized_valid_correlation(raw, crop)
            crop_center = crop_start + RAW_CROP_LENGTH / 2.0
            particle_samples = np.asarray(
                [float(row["t0"]) * SAMPLING_FREQUENCY_HZ for row in particles]
            )
            nearest_index = int(np.argmin(np.abs(particle_samples - crop_center)))
            nearest = particles[nearest_index]
            nearest_sample = float(particle_samples[nearest_index])
            center_distance = abs(nearest_sample - crop_center)
            multiplicity = int(
                np.sum(
                    (particle_samples >= crop_start)
                    & (particle_samples < crop_start + RAW_CROP_LENGTH)
                )
            )
            parent_id = hashlib.sha256(
                f"{class_name}/{crop_path.name}".encode()
            ).hexdigest()[:20]
            rows.append(
                {
                    "parent_id": parent_id,
                    "class_name": class_name,
                    "development_split": split,
                    "parent_crop_relative_path": (
                        f"train/{class_name}/{crop_path.name}"
                    ),
                    "source_signal_relative_path": (
                        f"{split}/signals/{source_filename}"
                    ),
                    "source_filename": source_filename,
                    "legacy_crop_index": legacy_crop_index,
                    "crop_start_sample": crop_start,
                    "crop_center_sample": crop_center,
                    "alignment_correlation": correlation,
                    "detector_center_distance_samples": center_distance,
                    "particle_multiplicity": multiplicity,
                    "multiplicity_stratum": (
                        "single" if multiplicity == 1 else "multiple"
                    ),
                    "primary_eligible": (
                        correlation >= PRIMARY_MIN_CORRELATION
                        and center_distance <= PRIMARY_MAX_CENTER_DISTANCE
                    ),
                    "detector_particle_idx": int(nearest["particle_idx"]),
                    "frequency_hz": float(nearest["frequency"]),
                    "amplitude": float(nearest["P0"]),
                    "detector_t0_sample": nearest_sample,
                    "fitted_tau_ms": float(nearest["tau"]) * 1000.0,
                    "detector_phi_rad": float(nearest["phi"]),
                    "detector_peak_energy": float(nearest["energy"]),
                    "detector_noise_floor": float(nearest["noise_floor"]),
                    "target_detector_snr_db": float(nearest["snr_db"]),
                    "parent_crop_sha256": sha256_file(crop_path),
                    "source_signal_sha256": sha256_file(raw_path),
                }
            )
    rows.sort(key=lambda row: (row["class_name"], row["parent_id"]))
    return rows, raw_cache


def _cross_source_noise_map(
    parents: Iterable[dict[str, Any]], *, seed: int
) -> dict[tuple[str, str], str]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for row in parents:
        grouped[row["class_name"]].add(row["source_filename"])
    mapping: dict[tuple[str, str], str] = {}
    for class_name, names_set in sorted(grouped.items()):
        names = sorted(names_set)
        if len(names) < 2:
            raise ValueError(f"too few noise sources for {class_name}")
        offset = 1 + deterministic_seed(seed, class_name, "noise", 0) % (
            len(names) - 1
        )
        for index, name in enumerate(names):
            other = names[(index + offset) % len(names)]
            if other == name:
                raise AssertionError("source-disjoint noise mapping failed")
            mapping[(class_name, name)] = other
    return mapping


def build_candidate(
    *,
    p0_dataset_root: Path,
    raw_dataset_root: Path,
    detector_particles_csv: Path,
    output_dir: Path,
    p0_manifest_sha256: str,
    raw_manifest_sha256: str,
    checkpoint_sha256: str,
    seed: int = 20260724,
    maximum_parents: int | None = None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite candidate: {output_dir}")
    if checkpoint_sha256 != CHECKPOINT_SHA256:
        raise ValueError("legacy classifier checkpoint hash mismatch")
    parents, raw_cache = discover_parent_provenance(
        p0_dataset_root=p0_dataset_root,
        raw_dataset_root=raw_dataset_root,
        detector_particles_csv=detector_particles_csv,
    )
    mapped_count = len(parents)
    primary_count = sum(bool(row["primary_eligible"]) for row in parents)
    if maximum_parents is None:
        if mapped_count != 683 or primary_count != 632:
            raise ValueError(
                "frozen provenance population changed: "
                f"mapped={mapped_count}, primary={primary_count}"
            )
    else:
        if maximum_parents <= 0:
            raise ValueError("maximum_parents must be positive")
        parents = parents[:maximum_parents]
    noise_map = _cross_source_noise_map(parents, seed=seed + 1)
    raw_paths = _raw_file_index(raw_dataset_root)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    row_count = len(parents) * (1 + VIEWS_PER_PARENT)
    signals = np.lib.format.open_memmap(
        output_dir / "signals.npy",
        mode="w+",
        dtype=np.float32,
        shape=(row_count, MODEL_INPUT_LENGTH),
    )
    records: list[dict[str, Any]] = []
    noise_cache: dict[str, np.ndarray] = {}
    signal_row = 0
    for parent in parents:
        crop_path = p0_dataset_root / parent["parent_crop_relative_path"]
        real_crop = np.load(crop_path, allow_pickle=False).astype(np.float32)
        signals[signal_row] = legacy_preprocess_2500(real_crop)
        base = dict(parent)
        records.append(
            {
                "signal_row": signal_row,
                "record_kind": "real_gallery",
                "view_index": -1,
                "phase_rad": "",
                "seed": "",
                "noise_source_filename": parent["source_filename"],
                "noise_source_relation": "observed",
                "spectral_amplitude_calibration_factor": "",
                "noise_spectral_calibration_factor": "",
                "predicted_clean_peak_energy": "",
                "achieved_peak_energy": "",
                "achieved_noise_floor": "",
                "achieved_detector_snr_db": "",
                **base,
            }
        )
        signal_row += 1
        noise_filename = noise_map[
            (parent["class_name"], parent["source_filename"])
        ]
        if noise_filename not in raw_cache:
            raw_cache[noise_filename] = np.load(
                raw_paths[noise_filename][1], allow_pickle=False
            ).astype(np.float32, copy=False)
        if noise_filename not in noise_cache:
            noise_cache[noise_filename] = lowest_energy_windows(
                raw_cache[noise_filename],
                stride=512 if parent["class_name"] == "2um" else 1024,
            )
        center_offset = (
            float(parent["detector_t0_sample"])
            - float(parent["crop_center_sample"])
        )
        for view_index in range(VIEWS_PER_PARENT):
            phase = 2.0 * np.pi * (view_index + 0.5) / VIEWS_PER_PARENT
            view_seed = deterministic_seed(
                seed, parent["parent_id"], "p0-2500", view_index
            )
            clean = _analytic_clean(
                amplitude=float(parent["amplitude"]),
                frequency_hz=float(parent["frequency_hz"]),
                sigma_ms=float(parent["fitted_tau_ms"]),
                center_offset_samples=center_offset,
                phase_at_crop_center_rad=phase,
            )
            native, calibration = detector_spectral_calibration(
                clean,
                noise_cache[noise_filename][view_index],
                frequency_hz=float(parent["frequency_hz"]),
                target_peak_energy=float(parent["detector_peak_energy"]),
                target_noise_floor=float(parent["detector_noise_floor"]),
            )
            crop_start = (NATIVE_LENGTH - RAW_CROP_LENGTH) // 2
            synthetic_crop = native[crop_start : crop_start + RAW_CROP_LENGTH]
            signals[signal_row] = legacy_preprocess_2500(synthetic_crop)
            records.append(
                {
                    "signal_row": signal_row,
                    "record_kind": "synthetic_query",
                    "view_index": view_index,
                    "phase_rad": phase,
                    "seed": view_seed,
                    "noise_source_filename": noise_filename,
                    "noise_source_relation": "cross_source",
                    **calibration,
                    **base,
                }
            )
            signal_row += 1
    signals.flush()
    del signals
    with (output_dir / "records.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    selected = parents
    summary = {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "source_datasets": {
            P0_DATASET_ID: p0_manifest_sha256,
            RAW_DATASET_ID: raw_manifest_sha256,
        },
        "detector_particles_csv_sha256": sha256_file(detector_particles_csv),
        "classifier_checkpoint_sha256": checkpoint_sha256,
        "method_evidence_id": METHOD_EVIDENCE_ID,
        "mapped_parent_count": len(selected),
        "primary_parent_count": sum(
            bool(row["primary_eligible"]) for row in selected
        ),
        "record_count": len(records),
        "signal_shape": [len(records), MODEL_INPUT_LENGTH],
        "signal_dtype": "float32",
        "views_per_parent": VIEWS_PER_PARENT,
        "seed": seed,
        "class_counts": dict(
            sorted(Counter(row["class_name"] for row in selected).items())
        ),
        "primary_class_counts": dict(
            sorted(
                Counter(
                    row["class_name"]
                    for row in selected
                    if row["primary_eligible"]
                ).items()
            )
        ),
        "multiplicity_counts": dict(
            sorted(
                Counter(row["multiplicity_stratum"] for row in selected).items()
            )
        ),
        "sealed_test_accessed": False,
        "preprocessing": (
            "centered 2500-sample crop; 5-100 kHz FFT bandpass at 2 MHz; "
            "stride decimation by 4 to 625; no z-score"
        ),
        "tau_semantics": "fitted tau is Gaussian sigma in milliseconds",
        "claim_boundary": (
            "The candidate includes only the registered p0 training folder and "
            "registered raw training signals with detector-training provenance. "
            "Synthetic nuisance is "
            "source-disjoint within class. No sealed test content was accessed."
        ),
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    contract = {
        "schema_version": 1,
        "format": "p0-2500-equation-parent-roundtrip",
        "native_sampling_frequency_hz": SAMPLING_FREQUENCY_HZ,
        "parent_crop_length": RAW_CROP_LENGTH,
        "model_input_length": MODEL_INPUT_LENGTH,
        "sealed_splits": ["test"],
        "primary_selection": {
            "minimum_alignment_correlation": PRIMARY_MIN_CORRELATION,
            "maximum_detector_center_distance_samples": (
                PRIMARY_MAX_CENTER_DISTANCE
            ),
        },
        "parameter_units": {
            "amplitude": "particles2SNR Hilbert P0 units",
            "frequency_hz": "Hz",
            "fitted_tau_ms": "ms; Gaussian sigma",
            "detector_t0_sample": "native 2 MHz sample",
            "detector_peak_energy": "squared FFT magnitude",
            "detector_noise_floor": "summed squared FFT magnitude",
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


def validate_candidate(root: Path) -> dict[str, Any]:
    summary = json.loads((root / "dataset_summary.json").read_text())
    contract = json.loads((root / "input_contract.json").read_text())
    records = _read_csv(root / "records.csv")
    signals = np.load(root / "signals.npy", mmap_mode="r", allow_pickle=False)
    if summary["dataset_id"] != DATASET_ID:
        raise ValueError("candidate dataset ID changed")
    if summary.get("sealed_test_accessed") is not False:
        raise PermissionError("candidate cannot prove sealed-test exclusion")
    if contract.get("sealed_splits") != ["test"]:
        raise ValueError("sealed split contract changed")
    if any(
        "test" in Path(row["parent_crop_relative_path"]).parts
        or row["development_split"] == "test"
        for row in records
    ):
        raise PermissionError("candidate contains a test row")
    if len(records) != len(signals) or len(records) != summary["record_count"]:
        raise ValueError("record/signal count mismatch")
    if signals.shape[1:] != (MODEL_INPUT_LENGTH,) or signals.dtype != np.float32:
        raise ValueError("candidate signal contract changed")
    if not np.isfinite(np.asarray(signals)).all():
        raise ValueError("candidate contains non-finite model inputs")
    if [int(row["signal_row"]) for row in records] != list(range(len(records))):
        raise ValueError("signal rows are not contiguous")
    keys = [
        (row["parent_id"], row["record_kind"], row["view_index"])
        for row in records
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate parent/view key")
    by_parent: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in records:
        by_parent[row["parent_id"]].append(row)
    if any(
        sum(row["record_kind"] == "real_gallery" for row in rows) != 1
        or sum(row["record_kind"] == "synthetic_query" for row in rows)
        != VIEWS_PER_PARENT
        for rows in by_parent.values()
    ):
        raise ValueError("incomplete parent record group")
    if any(
        row["record_kind"] == "synthetic_query"
        and row["noise_source_filename"] == row["source_filename"]
        for row in records
    ):
        raise ValueError("synthetic query leaks same-source noise")
    return {
        "valid": True,
        "parent_count": len(by_parent),
        "primary_parent_count": summary["primary_parent_count"],
        "record_count": len(records),
        "signal_shape": list(signals.shape),
        "sealed_test_accessed": False,
    }
