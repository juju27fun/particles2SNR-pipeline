from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from particles2snr.equation_roundtrip import (
    CHECKPOINT_SHA256,
    MODEL_INPUT_LENGTH,
    NATIVE_LENGTH,
    SAMPLING_FREQUENCY_HZ,
    centered_crop,
    classifier_preprocess,
    deterministic_seed,
    sha256_file,
    synthesize_equation_view,
)


DATASET_ID = "particles2snr-z8-equation-roundtrip@v2"
SOURCE_DATASET_ID = (
    "particles2snr-fbase-dual-clean-z8-events-3class-plus-unclear-development@v1"
)
SIGNAL_DATASET_ID = "particles2snr-f-c1-yolo-4class@v1"
METHOD_EVIDENCE_ID = "particles2snr-equation-latent-method-v2"
DETECTOR_SNR_METHOD = "peak_bin_energy_over_lowest_window_energy"
DETECTOR_BAND_HZ = (5_000.0, 100_000.0)
VIEWS_PER_EVENT = 8
VARIANTS = (
    "detector_bandlimited_phase_marginal",
    "detector_empirical_cross_source_phase_marginal",
    "detector_empirical_cross_source_fitted_phase",
    "detector_empirical_same_source_fitted_phase",
    "legacy_white_time_rms_phase_marginal",
    "detector_empirical_cross_source_tau_as_fwhm",
)
CLAIM_VARIANTS = (
    "detector_bandlimited_phase_marginal",
    "detector_empirical_cross_source_phase_marginal",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _physical_events(path: Path) -> list[dict[str, str]]:
    rows = _read_csv(path)
    events = [
        row
        for row in rows
        if row["class_name"] in {"2um", "4um", "10um"}
        and row["split"] in {"train", "val"}
    ]
    if not events:
        raise ValueError("no physical development events")
    return events


def join_detector_provenance(
    events: list[dict[str, str]],
    detector_rows: list[dict[str, str]],
) -> dict[str, dict[str, str]]:
    by_file: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in detector_rows:
        if row.get("snr_method") != DETECTOR_SNR_METHOD:
            raise ValueError("detector SNR method changed")
        by_file[row["filename"]].append(row)
    joined: dict[str, dict[str, str]] = {}
    for event in events:
        matches = [
            row
            for row in by_file.get(event["source_filename"], [])
            if np.isclose(
                float(row["frequency"]),
                float(event["frequency_hz"]),
                rtol=0.0,
                atol=1.0e-9,
            )
            and np.isclose(
                float(row["P0"]),
                float(event["particles2snr_amplitude"]),
                rtol=0.0,
                atol=1.0e-12,
            )
            and np.isclose(
                float(row["tau"]) * 1000.0,
                float(event["tau_ms"]),
                rtol=0.0,
                atol=1.0e-10,
            )
            and np.isclose(
                float(row["snr_db"]),
                float(event["snr_db"]),
                rtol=0.0,
                atol=1.0e-10,
            )
        ]
        if len(matches) != 1:
            raise ValueError(
                f"detector provenance join for {event['event_id']} has "
                f"{len(matches)} matches"
            )
        joined[event["event_id"]] = matches[0]
    return joined


def detector_peak_index(frequency_hz: float, length: int = NATIVE_LENGTH) -> int:
    index = int(round(float(frequency_hz) * length / SAMPLING_FREQUENCY_HZ))
    if index <= 0 or index >= length // 2 + 1:
        raise ValueError(f"detector frequency lies outside FFT support: {frequency_hz}")
    return index


def detector_spectral_calibration(
    clean: np.ndarray,
    noise_template: np.ndarray,
    *,
    frequency_hz: float,
    target_peak_energy: float,
    target_noise_floor: float,
) -> tuple[np.ndarray, dict[str, float]]:
    clean_values = np.asarray(clean, dtype=np.float64).reshape(-1)
    noise_values = np.asarray(noise_template, dtype=np.float64).reshape(-1)
    if clean_values.size != NATIVE_LENGTH or noise_values.size != NATIVE_LENGTH:
        raise ValueError("detector calibration requires 4096-sample inputs")
    targets = np.asarray([target_peak_energy, target_noise_floor], dtype=np.float64)
    if not np.isfinite(targets).all() or np.any(targets <= 0.0):
        raise ValueError("detector energy targets must be positive and finite")
    frequency = np.fft.rfftfreq(
        NATIVE_LENGTH, d=1.0 / SAMPLING_FREQUENCY_HZ
    )
    band = (frequency >= DETECTOR_BAND_HZ[0]) & (
        frequency <= DETECTOR_BAND_HZ[1]
    )
    peak_index = detector_peak_index(frequency_hz)
    if not band[peak_index]:
        raise ValueError("detector peak lies outside the calibrated noise band")

    clean_spectrum = np.fft.rfft(clean_values)
    predicted_peak_energy = float(np.abs(clean_spectrum[peak_index]) ** 2)
    if predicted_peak_energy <= 1.0e-18:
        raise ValueError("clean equation has zero detector-bin energy")
    amplitude_factor = float(
        np.sqrt(float(target_peak_energy) / predicted_peak_energy)
    )
    clean_spectrum *= amplitude_factor

    centered_noise = noise_values - float(noise_values.mean())
    noise_spectrum = np.fft.rfft(centered_noise)
    noise_spectrum[~band] = 0.0
    noise_spectrum[peak_index] = 0.0
    available_energy = float(np.sum(np.square(np.abs(noise_spectrum[band]))))
    if available_energy <= 1.0e-18:
        raise ValueError("noise template has no usable detector-band energy")
    noise_factor = float(np.sqrt(float(target_noise_floor) / available_energy))
    noise_spectrum *= noise_factor

    calibrated = np.fft.irfft(
        clean_spectrum + noise_spectrum, n=NATIVE_LENGTH
    ).astype(np.float32)
    achieved_spectrum = np.fft.rfft(calibrated.astype(np.float64))
    achieved_peak = float(np.abs(achieved_spectrum[peak_index]) ** 2)
    achieved_floor = float(np.sum(np.square(np.abs(noise_spectrum[band]))))
    achieved_snr = float(10.0 * np.log10(achieved_peak / achieved_floor))
    if not np.isfinite(calibrated).all():
        raise ValueError("detector calibration produced non-finite signal")
    return calibrated, {
        "spectral_amplitude_calibration_factor": amplitude_factor,
        "noise_spectral_calibration_factor": noise_factor,
        "predicted_clean_peak_energy": predicted_peak_energy,
        "achieved_peak_energy": achieved_peak,
        "achieved_noise_floor": achieved_floor,
        "achieved_detector_snr_db": achieved_snr,
    }


def _analytic_clean(
    *,
    amplitude: float,
    frequency_hz: float,
    sigma_ms: float,
    center_offset_samples: float,
    phase_at_crop_center_rad: float,
) -> np.ndarray:
    time = (
        np.arange(NATIVE_LENGTH, dtype=np.float64) - (NATIVE_LENGTH - 1) / 2.0
    ) / SAMPLING_FREQUENCY_HZ
    center_seconds = float(center_offset_samples) / SAMPLING_FREQUENCY_HZ
    envelope = np.exp(
        -0.5 * np.square((time - center_seconds) / (float(sigma_ms) / 1000.0))
    )
    clean = float(amplitude) * envelope * np.cos(
        2.0 * np.pi * float(frequency_hz) * time
        + float(phase_at_crop_center_rad)
    )
    if not np.isfinite(clean).all() or float(np.std(clean)) <= 1.0e-12:
        raise ValueError("invalid analytic clean signal")
    return clean.astype(np.float32)


def _band_limited_noise(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=NATIVE_LENGTH)
    spectrum = np.fft.rfft(values)
    frequency = np.fft.rfftfreq(
        NATIVE_LENGTH, d=1.0 / SAMPLING_FREQUENCY_HZ
    )
    band = (frequency >= DETECTOR_BAND_HZ[0]) & (
        frequency <= DETECTOR_BAND_HZ[1]
    )
    spectrum[~band] = 0.0
    return np.fft.irfft(spectrum, n=NATIVE_LENGTH).astype(np.float32)


def lowest_energy_windows(
    signal: np.ndarray,
    *,
    stride: int,
    count: int = VIEWS_PER_EVENT,
) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float32).reshape(-1)
    if values.size < NATIVE_LENGTH:
        raise ValueError("noise source is shorter than the detector window")
    starts = list(range(0, values.size - NATIVE_LENGTH + 1, int(stride)))
    frequency = np.fft.rfftfreq(
        NATIVE_LENGTH, d=1.0 / SAMPLING_FREQUENCY_HZ
    )
    band = frequency <= DETECTOR_BAND_HZ[1]
    ranked: list[tuple[float, int, np.ndarray]] = []
    for start in starts:
        window = np.asarray(
            values[start : start + NATIVE_LENGTH], dtype=np.float32
        )
        spectrum = np.fft.rfft(window.astype(np.float64))
        energy = float(np.sum(np.square(np.abs(spectrum[band]))))
        ranked.append((energy, start, window.copy()))
    ranked.sort(key=lambda item: (item[0], item[1]))
    if not ranked:
        raise ValueError("no complete detector noise windows")
    selected = [ranked[index % len(ranked)][2] for index in range(count)]
    return np.stack(selected).astype(np.float32)


def _noise_file_map(
    events: list[dict[str, str]], *, seed: int
) -> dict[tuple[str, str, str], dict[str, str]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for event in events:
        grouped[(event["split"], event["class_name"])][
            event["source_filename"]
        ] = event
    output: dict[tuple[str, str, str], dict[str, str]] = {}
    for (split, class_name), by_filename in sorted(grouped.items()):
        filenames = sorted(by_filename)
        if len(filenames) < 2:
            raise ValueError(
                f"cross-source noise stratum has fewer than two files: "
                f"{split}/{class_name}"
            )
        offset = 1 + deterministic_seed(seed, split, class_name, 0) % (
            len(filenames) - 1
        )
        for index, filename in enumerate(filenames):
            source_filename = filenames[(index + offset) % len(filenames)]
            if source_filename == filename:
                raise AssertionError("cross-source noise mapping retained source")
            output[(split, class_name, filename)] = by_filename[source_filename]
    return output


def _stride_for_class(class_name: str) -> int:
    return 512 if class_name == "2um" else 1024


def _phase_mode(variant: str) -> str:
    return "fitted" if "fitted_phase" in variant else "marginal"


def _noise_mode(variant: str) -> str:
    if "bandlimited" in variant:
        return "band_limited_stochastic"
    if "same_source" in variant:
        return "same_source_low_energy"
    if "empirical_cross_source" in variant:
        return "cross_source_low_energy"
    return "legacy_white_time_rms"


def _claim_role(variant: str) -> str:
    if variant in CLAIM_VARIANTS:
        return "claim_bearing"
    if "same_source" in variant:
        return "leakage_ceiling"
    return "diagnostic_control"


def build_detector_faithful_candidate(
    *,
    event_table_root: Path,
    signal_dataset_root: Path,
    detector_particles_csv: Path | tuple[Path, ...],
    output_dir: Path,
    source_manifest_sha256: str,
    signal_manifest_sha256: str,
    checkpoint_sha256: str,
    dataset_id: str = DATASET_ID,
    source_dataset_id: str = SOURCE_DATASET_ID,
    signal_dataset_id: str = SIGNAL_DATASET_ID,
    method_evidence_id: str = METHOD_EVIDENCE_ID,
    seed: int = 20260724,
    maximum_events: int | None = None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite candidate: {output_dir}")
    if checkpoint_sha256 != CHECKPOINT_SHA256:
        raise ValueError("classifier checkpoint hash mismatch")
    detector_paths = (
        (detector_particles_csv,)
        if isinstance(detector_particles_csv, Path)
        else tuple(detector_particles_csv)
    )
    if not detector_paths:
        raise ValueError("at least one detector-particle table is required")
    for detector_path in detector_paths:
        if "test" in {part.lower() for part in detector_path.parts}:
            raise PermissionError(
                "detector provenance must not use a test path"
            )
    events = _physical_events(event_table_root / "events.csv")
    events.sort(key=lambda row: (row["split"], row["class_name"], row["event_id"]))
    if maximum_events is not None:
        if maximum_events <= 0:
            raise ValueError("maximum_events must be positive")
        events = events[:maximum_events]
    detector_rows = [
        row
        for detector_path in detector_paths
        for row in _read_csv(detector_path)
    ]
    provenance = join_detector_provenance(events, detector_rows)
    cross_source = _noise_file_map(events, seed=seed + 1)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    record_count = len(events) * (1 + len(VARIANTS) * VIEWS_PER_EVENT)
    signals = np.lib.format.open_memmap(
        output_dir / "signals.npy",
        mode="w+",
        dtype=np.float32,
        shape=(record_count, MODEL_INPUT_LENGTH),
    )
    rows: list[dict[str, Any]] = []
    raw_cache: dict[str, np.ndarray] = {}
    low_window_cache: dict[str, np.ndarray] = {}

    def load_raw(event: dict[str, str]) -> np.ndarray:
        relative = event["source_signal_relative_path"]
        if relative not in raw_cache:
            raw_cache[relative] = np.load(
                signal_dataset_root / relative, allow_pickle=False
            ).astype(np.float32, copy=False)
        return raw_cache[relative]

    def noise_windows(event: dict[str, str]) -> np.ndarray:
        relative = event["source_signal_relative_path"]
        if relative not in low_window_cache:
            low_window_cache[relative] = lowest_energy_windows(
                load_raw(event),
                stride=_stride_for_class(event["class_name"]),
            )
        return low_window_cache[relative]

    signal_row = 0
    for event in events:
        detector = provenance[event["event_id"]]
        raw = load_raw(event)
        crop_center_sample = int(round(float(event["center_norm"]) * raw.size))
        real_crop = centered_crop(raw, crop_center_sample)
        signals[signal_row] = classifier_preprocess(real_crop)
        detector_t0_sample = float(detector["t0"]) * SAMPLING_FREQUENCY_HZ
        center_offset_samples = detector_t0_sample - crop_center_sample
        fitted_phase_at_crop = (
            2.0
            * np.pi
            * float(event["frequency_hz"])
            * crop_center_sample
            / SAMPLING_FREQUENCY_HZ
            + float(detector["phi"])
        ) % (2.0 * np.pi)
        base_metadata = {
            "source_event_id": event["event_id"],
            "split": event["split"],
            "class_name": event["class_name"],
            "annotation_origin": event["annotation_origin"],
            "source_group": event["source_filename"],
            "amplitude": event["particles2snr_amplitude"],
            "frequency_hz": event["frequency_hz"],
            "fitted_tau_ms": event["tau_ms"],
            "support_width_5sigma_ms": float(event["tau_ms"]) * 5.0,
            "deprecated_passage_time_ms": event["tau_ms"],
            "detector_particle_idx": detector["particle_idx"],
            "detector_t0_sample": detector_t0_sample,
            "crop_center_sample": crop_center_sample,
            "center_offset_samples": center_offset_samples,
            "detector_phi_rad": detector["phi"],
            "fitted_phase_at_crop_rad": fitted_phase_at_crop,
            "detector_peak_energy": detector["energy"],
            "detector_noise_floor": detector["noise_floor"],
            "target_detector_snr_db": detector["snr_db"],
            "detector_snr_method": detector["snr_method"],
        }
        rows.append(
            {
                "signal_row": signal_row,
                "record_kind": "real_gallery",
                "event_id": event["event_id"],
                "variant": "real",
                "view_index": -1,
                "phase_mode": "observed",
                "phase_rad": "",
                "sigma_ms": "",
                "noise_mode": "observed",
                "noise_source_filename": event["source_filename"],
                "noise_source_relation": "source",
                "claim_role": "gallery",
                "seed": "",
                "spectral_amplitude_calibration_factor": "",
                "noise_spectral_calibration_factor": "",
                "predicted_clean_peak_energy": "",
                "achieved_peak_energy": detector["energy"],
                "achieved_noise_floor": detector["noise_floor"],
                "achieved_detector_snr_db": detector["snr_db"],
                **base_metadata,
            }
        )
        signal_row += 1

        cross_event = cross_source[
            (event["split"], event["class_name"], event["source_filename"])
        ]
        for variant in VARIANTS:
            phase_mode = _phase_mode(variant)
            noise_mode = _noise_mode(variant)
            claim_role = _claim_role(variant)
            sigma_ms = float(event["tau_ms"])
            if variant.endswith("tau_as_fwhm"):
                sigma_ms /= 2.355
            for view_index in range(VIEWS_PER_EVENT):
                view_seed = deterministic_seed(
                    seed, event["event_id"], variant, view_index
                )
                phase = (
                    fitted_phase_at_crop
                    if phase_mode == "fitted"
                    else 2.0
                    * np.pi
                    * (view_index + 0.5)
                    / VIEWS_PER_EVENT
                )
                if variant == "legacy_white_time_rms_phase_marginal":
                    native = synthesize_equation_view(
                        amplitude=float(event["particles2snr_amplitude"]),
                        frequency_hz=float(event["frequency_hz"]),
                        sigma_ms_value=sigma_ms,
                        snr_db=float(event["snr_db"]),
                        phase_rad=phase,
                        seed=view_seed,
                    )
                    calibration = {
                        "spectral_amplitude_calibration_factor": "",
                        "noise_spectral_calibration_factor": "",
                        "predicted_clean_peak_energy": "",
                        "achieved_peak_energy": "",
                        "achieved_noise_floor": "",
                        "achieved_detector_snr_db": "",
                    }
                    noise_source_filename = ""
                    noise_source_relation = "synthetic_white"
                else:
                    clean = _analytic_clean(
                        amplitude=float(event["particles2snr_amplitude"]),
                        frequency_hz=float(event["frequency_hz"]),
                        sigma_ms=sigma_ms,
                        center_offset_samples=center_offset_samples,
                        phase_at_crop_center_rad=phase,
                    )
                    if noise_mode == "band_limited_stochastic":
                        template = _band_limited_noise(view_seed)
                        noise_source_filename = ""
                        noise_source_relation = "synthetic_band_limited"
                    elif noise_mode == "same_source_low_energy":
                        template = noise_windows(event)[view_index]
                        noise_source_filename = event["source_filename"]
                        noise_source_relation = "same_source"
                    else:
                        template = noise_windows(cross_event)[view_index]
                        noise_source_filename = cross_event["source_filename"]
                        noise_source_relation = "cross_source"
                    native, calibration = detector_spectral_calibration(
                        clean,
                        template,
                        frequency_hz=float(event["frequency_hz"]),
                        target_peak_energy=float(detector["energy"]),
                        target_noise_floor=float(detector["noise_floor"]),
                    )
                signals[signal_row] = classifier_preprocess(native)
                rows.append(
                    {
                        "signal_row": signal_row,
                        "record_kind": "synthetic_query",
                        "event_id": (
                            f"{event['event_id']}:{variant}:view-{view_index}"
                        ),
                        "variant": variant,
                        "view_index": view_index,
                        "phase_mode": phase_mode,
                        "phase_rad": phase,
                        "sigma_ms": sigma_ms,
                        "noise_mode": noise_mode,
                        "noise_source_filename": noise_source_filename,
                        "noise_source_relation": noise_source_relation,
                        "claim_role": claim_role,
                        "seed": view_seed,
                        **calibration,
                        **base_metadata,
                    }
                )
                signal_row += 1
    signals.flush()
    del signals
    if signal_row != record_count or len(rows) != record_count:
        raise AssertionError("candidate preallocation count mismatch")
    with (output_dir / "records.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "source_datasets": {
            source_dataset_id: source_manifest_sha256,
            signal_dataset_id: signal_manifest_sha256,
        },
        "detector_particles_csv_sha256s": {
            f"{path.parent.name}/{path.name}": sha256_file(path)
            for path in detector_paths
        },
        "classifier_checkpoint_sha256": checkpoint_sha256,
        "method_evidence_id": method_evidence_id,
        "event_count": len(events),
        "record_count": record_count,
        "signal_shape": [record_count, MODEL_INPUT_LENGTH],
        "signal_dtype": "float32",
        "views_per_event": VIEWS_PER_EVENT,
        "variants": list(VARIANTS),
        "claim_variants": list(CLAIM_VARIANTS),
        "seed": seed,
        "class_counts": dict(
            sorted(Counter(row["class_name"] for row in events).items())
        ),
        "origin_counts": dict(
            sorted(Counter(row["annotation_origin"] for row in events).items())
        ),
        "split_counts": dict(
            sorted(Counter(row["split"] for row in events).items())
        ),
        "detector_provenance_joined": len(provenance),
        "sealed_test_accessed": False,
        "preprocessing": (
            "4096-sample crop centered on registered annotation; detector t0 "
            "retained as an explicit offset; mean decimate by 8; window z-score"
        ),
        "tau_semantics": {
            "fitted_tau_ms": "Gaussian envelope sigma",
            "support_width_5sigma_ms": "annotation support width derived as 5*tau",
            "deprecated_passage_time_ms": (
                "legacy alias equal to fitted_tau_ms; forbidden for new filtering"
            ),
        },
        "snr_semantics": {
            "target_detector_snr_db": DETECTOR_SNR_METHOD,
            "calibration": (
                "clean detector-bin energy and nuisance detector-band floor are "
                "matched separately in the 4096-point FFT"
            ),
        },
        "claim_boundary": (
            "Claim-bearing noise is source-disjoint. Same-source noise is a "
            "leakage ceiling only. Spectral amplitude calibration is recorded "
            "because Hilbert P0 and detector-bin energy are not interchangeable."
        ),
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    contract = {
        "schema_version": 1,
        "format": "particles2snr-detector-faithful-equation-roundtrip",
        "sampling_frequency_hz": SAMPLING_FREQUENCY_HZ,
        "native_length": NATIVE_LENGTH,
        "model_input_length": MODEL_INPUT_LENGTH,
        "splits": ["train", "val"],
        "sealed_splits": ["test"],
        "variants": list(VARIANTS),
        "claim_variants": list(CLAIM_VARIANTS),
        "detector_band_hz": list(DETECTOR_BAND_HZ),
        "parameter_units": {
            "amplitude": "particles2SNR Hilbert P0 units",
            "frequency_hz": "Hz",
            "fitted_tau_ms": "ms; Gaussian sigma",
            "support_width_5sigma_ms": "ms",
            "deprecated_passage_time_ms": "ms; legacy alias, do not filter",
            "phase_rad": "radian",
            "detector_peak_energy": "squared FFT magnitude",
            "detector_noise_floor": "summed squared FFT magnitude",
            "target_detector_snr_db": "dB",
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


def validate_detector_faithful_candidate(
    root: Path, *, expected_dataset_id: str = DATASET_ID
) -> dict[str, Any]:
    summary = json.loads((root / "dataset_summary.json").read_text(encoding="utf-8"))
    contract = json.loads((root / "input_contract.json").read_text(encoding="utf-8"))
    rows = _read_csv(root / "records.csv")
    signals = np.load(root / "signals.npy", mmap_mode="r", allow_pickle=False)
    if summary["dataset_id"] != expected_dataset_id:
        raise ValueError("candidate dataset ID changed")
    if contract["sealed_splits"] != ["test"]:
        raise ValueError("sealed split contract changed")
    if any(row["split"] == "test" for row in rows):
        raise PermissionError("candidate contains sealed test rows")
    if len(rows) != len(signals) or len(rows) != summary["record_count"]:
        raise ValueError("record/signal count mismatch")
    if [int(row["signal_row"]) for row in rows] != list(range(len(rows))):
        raise ValueError("signal rows are not contiguous")
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
    synthetic = [row for row in rows if row["record_kind"] == "synthetic_query"]
    calibrated = [
        row
        for row in synthetic
        if row["variant"] != "legacy_white_time_rms_phase_marginal"
    ]
    for row in calibrated:
        target_peak = float(row["detector_peak_energy"])
        target_floor = float(row["detector_noise_floor"])
        if not np.isclose(
            float(row["achieved_peak_energy"]), target_peak, rtol=2.0e-5, atol=1.0e-5
        ):
            raise ValueError("detector peak calibration drifted")
        if not np.isclose(
            float(row["achieved_noise_floor"]),
            target_floor,
            rtol=1.0e-10,
            atol=1.0e-8,
        ):
            raise ValueError("detector noise-floor calibration drifted")
    for row in synthetic:
        if row["variant"] in CLAIM_VARIANTS:
            if row["noise_source_relation"] == "same_source":
                raise ValueError("claim-bearing row leaks same-source noise")
            if (
                row["noise_source_filename"]
                and row["noise_source_filename"] == row["source_group"]
            ):
                raise ValueError("claim-bearing row leaks source filename")
        if row["variant"] == "detector_empirical_same_source_fitted_phase":
            if row["claim_role"] != "leakage_ceiling":
                raise ValueError("same-source variant is not marked as a ceiling")
    if any(
        float(row["deprecated_passage_time_ms"]) != float(row["fitted_tau_ms"])
        for row in rows
    ):
        raise ValueError("deprecated passage-time alias changed value")
    return {
        "valid": True,
        "record_count": len(rows),
        "event_count": summary["event_count"],
        "signal_shape": list(signals.shape),
        "calibrated_record_count": len(calibrated),
        "sealed_test_accessed": False,
    }
