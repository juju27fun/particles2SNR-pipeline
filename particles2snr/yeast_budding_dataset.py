from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.stats import norm, rankdata

from .ssl_realism_audit import signal_descriptors
from .yeast_representation_dataset import preprocess_crop


RAW_FS = 2_000_000.0
OUTPUT_LENGTH = 4096
RAW_LENGTH = 8192
RESOLVED_DELTA_BIC = 10.0
RESOLVED_SCORE = 0.1
QUANTILE_PROBABILITIES = np.linspace(0.01, 0.99, 99)

COPULA_FEATURES = (
    "separation_ms_signed",
    "sigma1_left_ms",
    "sigma1_right_ms",
    "sigma2_left_ms",
    "sigma2_right_ms",
    "shape1",
    "shape2",
    "frequency1_khz",
    "frequency_delta_khz",
    "chirp1_khz_per_ms",
    "chirp2_khz_per_ms",
    "relative_amplitude",
)


@dataclass(frozen=True)
class BuddingLatent:
    separation_ms_signed: float
    sigma1_left_ms: float
    sigma1_right_ms: float
    sigma2_left_ms: float
    sigma2_right_ms: float
    shape1: float
    shape2: float
    frequency1_khz: float
    frequency_delta_khz: float
    chirp1_khz_per_ms: float
    chirp2_khz_per_ms: float
    relative_amplitude: float
    resolved: bool
    generator_model: str
    mother_radius_relative: float | None = None
    bud_radius_ratio: float | None = None
    orientation_cosine: float | None = None
    beam_radius_relative: float | None = None
    amplitude_size_exponent: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _quantile_contract(values: Iterable[float]) -> list[float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or array.size < 8 or not np.all(np.isfinite(array)):
        raise ValueError("Calibration vectors must contain at least eight finite values")
    return np.quantile(array, QUANTILE_PROBABILITIES).tolist()


def _fit_copula(rows: list[dict[str, float]]) -> dict[str, Any]:
    matrix = np.asarray(
        [[row[name] for name in COPULA_FEATURES] for row in rows],
        dtype=np.float64,
    )
    if matrix.shape[0] < 8 or not np.all(np.isfinite(matrix)):
        raise ValueError("Rank-Gaussian copula requires at least eight finite rows")
    gaussian = np.column_stack(
        [
            norm.ppf(
                np.clip(
                    (rankdata(matrix[:, index], method="average") - 0.5)
                    / matrix.shape[0],
                    1.0e-4,
                    1.0 - 1.0e-4,
                )
            )
            for index in range(matrix.shape[1])
        ]
    )
    correlation = np.eye(gaussian.shape[1], dtype=np.float64)
    variable = np.std(gaussian, axis=0) > 1.0e-12
    if np.count_nonzero(variable) > 1:
        correlation[np.ix_(variable, variable)] = np.corrcoef(
            gaussian[:, variable],
            rowvar=False,
        )
    correlation = 0.90 * correlation + 0.10 * np.eye(correlation.shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    correlation = (eigenvectors * np.clip(eigenvalues, 1.0e-6, None)) @ eigenvectors.T
    diagonal = np.sqrt(np.diag(correlation))
    correlation /= diagonal[:, None] * diagonal[None, :]
    return {
        "n_rows": int(matrix.shape[0]),
        "feature_names": list(COPULA_FEATURES),
        "quantile_probabilities": QUANTILE_PROBABILITIES.tolist(),
        "feature_quantiles": {
            name: _quantile_contract(matrix[:, index])
            for index, name in enumerate(COPULA_FEATURES)
        },
        "rank_gaussian_correlation": correlation.tolist(),
        "shrinkage_to_identity": 0.10,
    }


def _fit_row(row: dict[str, str]) -> dict[str, float]:
    amplitude1 = float(row["m2_c1_amplitude"])
    amplitude2 = float(row["m2_c2_amplitude"])
    return {
        "separation_ms_signed": (
            float(row["m2_c2_center_ms"]) - float(row["m2_c1_center_ms"])
        ),
        "sigma1_left_ms": float(row["m2_c1_sigma_left_ms"]),
        "sigma1_right_ms": float(row["m2_c1_sigma_right_ms"]),
        "sigma2_left_ms": float(row["m2_c2_sigma_left_ms"]),
        "sigma2_right_ms": float(row["m2_c2_sigma_right_ms"]),
        "shape1": float(row["m2_c1_shape"]),
        "shape2": float(row["m2_c2_shape"]),
        "frequency1_khz": float(row["m2_c1_frequency_khz"]),
        "frequency_delta_khz": (
            float(row["m2_c2_frequency_khz"])
            - float(row["m2_c1_frequency_khz"])
        ),
        "chirp1_khz_per_ms": float(row["m2_c1_chirp_khz_per_ms"]),
        "chirp2_khz_per_ms": float(row["m2_c2_chirp_khz_per_ms"]),
        "relative_amplitude": amplitude2 / max(amplitude1, 1.0e-12),
    }


def build_budding_calibration(
    *,
    fit_summaries_csv: Path,
    real_dataset_root: Path,
    source_dataset_id: str,
) -> dict[str, Any]:
    fit_rows = _read_csv(fit_summaries_csv)
    event_rows = _read_csv(real_dataset_root / "events.csv")
    train_events = {
        row["event_id"]: row
        for row in event_rows
        if row["source_group"] == "budding"
        and row["development_split"] == "development_train"
        and row["quality"] == "strict"
    }
    if not fit_rows or {row["event_id"] for row in fit_rows} != set(train_events):
        raise ValueError(
            "Fit summaries must exactly cover strict budding development_train"
        )
    signals = np.load(real_dataset_root / "signals.npy", mmap_mode="r")
    rms_values: list[float] = []
    snr_values: list[float] = []
    resolved_rows: list[dict[str, float]] = []
    ambiguous_rows: list[dict[str, float]] = []
    widths: list[float] = []
    frequency_deltas: list[float] = []
    common_chirps: list[float] = []
    for row in fit_rows:
        event = train_events[row["event_id"]]
        signal = np.asarray(
            signals[int(event["signal_row"])],
            dtype=np.float64,
        )
        centered = signal - float(np.mean(signal))
        rms_values.append(float(np.sqrt(np.mean(np.square(centered)))))
        start = int(float(event["event_start_input_index"]))
        end = int(float(event["event_end_input_index"]))
        outside = np.ones(centered.size, dtype=bool)
        outside[max(0, start) : min(centered.size, end)] = False
        noise_rms = float(np.sqrt(np.mean(np.square(centered[outside]))))
        event_rms = float(
            np.sqrt(np.mean(np.square(centered[max(0, start) : min(centered.size, end)])))
        )
        snr_values.append(
            float(np.clip(20.0 * np.log10(event_rms / max(noise_rms, 1.0e-12)), 0.0, 35.0))
        )
        values = _fit_row(row)
        # The full-trace observable is used for synthesis because the fitted
        # STFT ridge is conditional on a component window and is not the same
        # quantity as the downstream comparison's dominant frequency.
        values["frequency1_khz"] = signal_descriptors(centered)[
            "dominant_frequency_khz"
        ]
        is_resolved = (
            float(row["delta_bic_m1_minus_m2"]) >= RESOLVED_DELTA_BIC
            and float(row["resolvability_score"]) >= RESOLVED_SCORE
        )
        (resolved_rows if is_resolved else ambiguous_rows).append(values)
        if is_resolved:
            widths.extend(
                (
                    0.5 * (values["sigma1_left_ms"] + values["sigma1_right_ms"]),
                    0.5 * (values["sigma2_left_ms"] + values["sigma2_right_ms"]),
                )
            )
            frequency_deltas.append(values["frequency_delta_khz"])
            common_chirps.append(
                0.5
                * (
                    values["chirp1_khz_per_ms"]
                    + values["chirp2_khz_per_ms"]
                )
            )
    resolved_fraction = len(resolved_rows) / len(fit_rows)
    return {
        "schema_version": 1,
        "calibration_id": "yeast-budding-double-sphere-calibration@v1",
        "source_dataset": source_dataset_id,
        "source_split": "development_train",
        "source_group": "budding",
        "quality": "strict",
        "n_events": len(fit_rows),
        "m2_identifiability_rule": {
            "resolved": (
                f"delta_bic_m1_minus_m2 >= {RESOLVED_DELTA_BIC:g} and "
                f"resolvability_score >= {RESOLVED_SCORE:g}"
            ),
            "ambiguous_policy": (
                "retain a two-component latent but censor the second component "
                "from morphology claims"
            ),
            "resolved_count": len(resolved_rows),
            "ambiguous_count": len(ambiguous_rows),
            "resolved_fraction": resolved_fraction,
        },
        "data_oriented": {
            "model": "rank-gaussian-copula-v1",
            "resolved": _fit_copula(resolved_rows),
            "ambiguous": _fit_copula(ambiguous_rows),
        },
        "biophysics_oriented": {
            "model": "contacting-relative-double-sphere-v1",
            "relative_geometry_only": True,
            "mother_radius_distribution": "LogNormal(log_mean=0, log_sigma=0.15)",
            "bud_radius_ratio_distribution": "0.25 + 0.70 * Beta(2, 2)",
            "orientation_distribution": "cos(theta) ~ Uniform(-1, 1)",
            "empirical_median_component_width_ms": float(np.median(widths)),
            "component_width_ms_quantiles": _quantile_contract(widths),
            "resolved_separation_ms_quantiles": _quantile_contract(
                abs(row["separation_ms_signed"]) for row in resolved_rows
            ),
            "frequency1_khz_quantiles": _quantile_contract(
                row["frequency1_khz"] for row in resolved_rows
            ),
            "frequency_delta_khz_quantiles": _quantile_contract(frequency_deltas),
            "common_chirp_khz_per_ms_quantiles": _quantile_contract(common_chirps),
            "quantile_probabilities": QUANTILE_PROBABILITIES.tolist(),
        },
        "nuisance": {
            "target_rms_quantiles": _quantile_contract(rms_values),
            "snr_db_quantiles": _quantile_contract(snr_values),
            "quantile_probabilities": QUANTILE_PROBABILITIES.tolist(),
            "trace_policy": (
                "empirical scalar quantiles only; no real waveform or template "
                "is copied into simulation"
            ),
        },
        "source_checksums": {
            "fit_summaries.csv": _sha256(fit_summaries_csv),
            "events.csv": _sha256(real_dataset_root / "events.csv"),
            "signals.npy": _sha256(real_dataset_root / "signals.npy"),
        },
        "sealed_splits_used": [],
        "scientific_scope": (
            "mono-acquisition budding calibration; relative double-sphere "
            "geometry, not absolute yeast radii"
        ),
    }


def _sample_quantiles(
    rng: np.random.Generator,
    values: list[float],
    probabilities: list[float],
    *,
    probability: float | None = None,
) -> float:
    selected = (
        float(rng.uniform(probabilities[0], probabilities[-1]))
        if probability is None
        else probability
    )
    return float(np.interp(selected, probabilities, values))


def _bounded_latent(values: dict[str, float], *, resolved: bool, model: str) -> BuddingLatent:
    return BuddingLatent(
        separation_ms_signed=float(np.clip(values["separation_ms_signed"], -1.5, 1.5)),
        sigma1_left_ms=float(np.clip(values["sigma1_left_ms"], 0.015, 0.8)),
        sigma1_right_ms=float(np.clip(values["sigma1_right_ms"], 0.015, 0.8)),
        sigma2_left_ms=float(np.clip(values["sigma2_left_ms"], 0.015, 0.8)),
        sigma2_right_ms=float(np.clip(values["sigma2_right_ms"], 0.015, 0.8)),
        shape1=float(np.clip(values["shape1"], 1.0, 4.0)),
        shape2=float(np.clip(values["shape2"], 1.0, 4.0)),
        frequency1_khz=float(np.clip(values["frequency1_khz"], 5.0, 80.0)),
        frequency_delta_khz=float(np.clip(values["frequency_delta_khz"], -20.0, 20.0)),
        chirp1_khz_per_ms=float(np.clip(values["chirp1_khz_per_ms"], -40.0, 40.0)),
        chirp2_khz_per_ms=float(np.clip(values["chirp2_khz_per_ms"], -40.0, 40.0)),
        relative_amplitude=float(np.clip(values["relative_amplitude"], 0.03, 1.5)),
        resolved=resolved,
        generator_model=model,
    )


def sample_data_oriented_latent(
    rng: np.random.Generator,
    calibration: dict[str, Any],
) -> BuddingLatent:
    resolved_fraction = float(
        calibration["m2_identifiability_rule"]["resolved_fraction"]
    )
    resolved = bool(rng.random() < resolved_fraction)
    contract = calibration["data_oriented"][
        "resolved" if resolved else "ambiguous"
    ]
    correlation = np.asarray(
        contract["rank_gaussian_correlation"],
        dtype=np.float64,
    )
    gaussian = rng.multivariate_normal(
        np.zeros(len(COPULA_FEATURES)),
        correlation,
    )
    probabilities = norm.cdf(gaussian)
    values = {
        name: _sample_quantiles(
            rng,
            contract["feature_quantiles"][name],
            contract["quantile_probabilities"],
            probability=float(probabilities[index]),
        )
        for index, name in enumerate(COPULA_FEATURES)
    }
    return _bounded_latent(
        values,
        resolved=resolved,
        model="data-oriented-rank-gaussian-copula-v1",
    )


def sample_biophysics_oriented_latent(
    rng: np.random.Generator,
    calibration: dict[str, Any],
    *,
    amplitude_size_exponent: float,
    beam_radius_relative: float,
) -> BuddingLatent:
    if amplitude_size_exponent <= 0.0 or beam_radius_relative <= 0.0:
        raise ValueError("Biophysics configuration values must be positive")
    contract = calibration["biophysics_oriented"]
    mother_radius = float(rng.lognormal(mean=0.0, sigma=0.15))
    bud_ratio = float(0.25 + 0.70 * rng.beta(2.0, 2.0))
    orientation_cosine = float(rng.uniform(-1.0, 1.0))
    probabilities = contract["quantile_probabilities"]
    target_width = _sample_quantiles(
        rng,
        contract["component_width_ms_quantiles"],
        probabilities,
    )
    reference_width = np.sqrt(beam_radius_relative**2 + 1.0)
    time_scale = target_width / reference_width
    mother_width = time_scale * np.sqrt(
        beam_radius_relative**2 + mother_radius**2
    )
    bud_width = time_scale * np.sqrt(
        beam_radius_relative**2 + (mother_radius * bud_ratio) ** 2
    )
    separation = (
        time_scale
        * mother_radius
        * (1.0 + bud_ratio)
        * orientation_cosine
    )
    frequency1 = _sample_quantiles(
        rng,
        contract["frequency1_khz_quantiles"],
        probabilities,
    )
    empirical_delta = _sample_quantiles(
        rng,
        contract["frequency_delta_khz_quantiles"],
        probabilities,
    )
    common_chirp = _sample_quantiles(
        rng,
        contract["common_chirp_khz_per_ms_quantiles"],
        probabilities,
    )
    relative_amplitude = bud_ratio**amplitude_size_exponent
    proxy_resolvability = (
        abs(separation)
        / max(0.5 * (mother_width + bud_width), 1.0e-6)
        * relative_amplitude
    )
    values = {
        "separation_ms_signed": separation,
        "sigma1_left_ms": mother_width,
        "sigma1_right_ms": mother_width,
        "sigma2_left_ms": bud_width,
        "sigma2_right_ms": bud_width,
        "shape1": 2.0,
        "shape2": 2.0,
        "frequency1_khz": frequency1,
        # Contacting spheres share the common velocity; this is a small
        # calibrated residual rather than an independent second velocity.
        "frequency_delta_khz": 0.20 * empirical_delta,
        "chirp1_khz_per_ms": common_chirp,
        "chirp2_khz_per_ms": common_chirp,
        "relative_amplitude": relative_amplitude,
    }
    base = _bounded_latent(
        values,
        resolved=bool(proxy_resolvability >= RESOLVED_SCORE),
        model="biophysics-oriented-contacting-double-sphere-v1",
    )
    return BuddingLatent(
        **{
            **base.to_dict(),
            "mother_radius_relative": mother_radius,
            "bud_radius_ratio": bud_ratio,
            "orientation_cosine": orientation_cosine,
            "beam_radius_relative": beam_radius_relative,
            "amplitude_size_exponent": amplitude_size_exponent,
        }
    )


def _asymmetric_envelope(
    time_ms: np.ndarray,
    *,
    center_ms: float,
    sigma_left_ms: float,
    sigma_right_ms: float,
    shape: float,
) -> np.ndarray:
    scale = np.where(
        time_ms < center_ms,
        sigma_left_ms,
        sigma_right_ms,
    )
    distance = np.abs((time_ms - center_ms) / np.maximum(scale, 1.0e-9))
    return np.exp(-0.5 * np.power(distance, shape))


def _colored_noise(
    rng: np.random.Generator,
    length: int,
    *,
    variant: str,
) -> np.ndarray:
    white = rng.normal(size=length)
    cutoff = 55_000.0 if variant == "heldout_sensor" else 75_000.0
    order = 3 if variant == "heldout_sensor" else 2
    sos = butter(order, cutoff, btype="lowpass", fs=RAW_FS, output="sos")
    noise = sosfiltfilt(sos, white)
    return noise / max(float(np.std(noise)), 1.0e-9)


def simulate_budding_view(
    rng: np.random.Generator,
    latent: BuddingLatent,
    calibration: dict[str, Any],
    *,
    variant: str,
) -> tuple[np.ndarray, dict[str, float]]:
    time_ms = np.arange(RAW_LENGTH, dtype=np.float64) / RAW_FS * 1000.0
    midpoint = float(rng.uniform(0.9, 3.2))
    center1 = midpoint - 0.5 * latent.separation_ms_signed
    center2 = midpoint + 0.5 * latent.separation_ms_signed
    shift = float(np.clip(midpoint, 0.9, 3.2)) - midpoint
    center1 += shift
    center2 += shift
    phase1 = float(rng.uniform(0.0, 2.0 * np.pi))
    phase2 = float(rng.uniform(0.0, 2.0 * np.pi))
    envelope1 = _asymmetric_envelope(
        time_ms,
        center_ms=center1,
        sigma_left_ms=latent.sigma1_left_ms,
        sigma_right_ms=latent.sigma1_right_ms,
        shape=latent.shape1,
    )
    envelope2 = _asymmetric_envelope(
        time_ms,
        center_ms=center2,
        sigma_left_ms=latent.sigma2_left_ms,
        sigma_right_ms=latent.sigma2_right_ms,
        shape=latent.shape2,
    )
    relative1 = time_ms - center1
    relative2 = time_ms - center2
    phase_curve1 = 2.0 * np.pi * (
        latent.frequency1_khz * relative1
        + 0.5 * latent.chirp1_khz_per_ms * np.square(relative1)
    )
    phase_curve2 = 2.0 * np.pi * (
        (latent.frequency1_khz + latent.frequency_delta_khz) * relative2
        + 0.5 * latent.chirp2_khz_per_ms * np.square(relative2)
    )
    clean = envelope1 * np.cos(phase_curve1 + phase1)
    clean += (
        latent.relative_amplitude
        * envelope2
        * np.cos(phase_curve2 + phase2)
    )
    clean_rms = max(float(np.sqrt(np.mean(np.square(clean)))), 1.0e-9)
    nuisance = calibration["nuisance"]
    snr_db = _sample_quantiles(
        rng,
        nuisance["snr_db_quantiles"],
        nuisance["quantile_probabilities"],
    )
    noise = (
        _colored_noise(rng, RAW_LENGTH, variant=variant)
        * clean_rms
        / (10.0 ** (snr_db / 20.0))
    )
    baseline_strength = float(rng.uniform(0.0, 0.08)) * clean_rms
    baseline = baseline_strength * np.sin(
        2.0
        * np.pi
        * rng.uniform(80.0, 800.0)
        * (time_ms / 1000.0)
        + rng.uniform(0.0, 2.0 * np.pi)
    )
    response = float(rng.uniform(0.88, 1.12))
    processed = preprocess_crop(
        response * (clean + noise + baseline).astype(np.float32)
    )
    target_rms = _sample_quantiles(
        rng,
        nuisance["target_rms_quantiles"],
        nuisance["quantile_probabilities"],
    )
    processed *= target_rms / max(
        float(np.sqrt(np.mean(np.square(processed)))),
        1.0e-9,
    )
    return processed.astype(np.float32), {
        "event_midpoint_ms": midpoint,
        "component1_center_ms": center1,
        "component2_center_ms": center2,
        "phase1_rad": phase1,
        "phase2_rad": phase2,
        "snr_db": snr_db,
        "target_rms": target_rms,
        "baseline_strength": baseline_strength,
        "sensor_response": response,
    }


def build_budding_simulation_dataset(
    *,
    output_dir: Path,
    calibration: dict[str, Any],
    generator: str,
    n_train_latents: int,
    n_validation_latents: int,
    n_test_latents: int,
    views_per_latent: int = 2,
    seed: int = 7301,
    amplitude_size_exponent: float = 2.0,
    beam_radius_relative: float = 1.0,
) -> dict[str, Any]:
    if generator not in {"data", "biophysics"}:
        raise ValueError("generator must be 'data' or 'biophysics'")
    if min(
        n_train_latents,
        n_validation_latents,
        n_test_latents,
        views_per_latent,
    ) <= 0:
        raise ValueError("Dataset counts must be positive")
    if calibration.get("source_split") != "development_train":
        raise PermissionError("Budding calibration must use development_train only")
    if calibration.get("sealed_splits_used") != []:
        raise PermissionError("Budding calibration accessed a sealed split")
    split_specs = (
        ("train", n_train_latents, seed, "base"),
        ("validation", n_validation_latents, seed + 10_000, "base"),
        ("test", n_test_latents, seed + 20_000, "heldout_sensor"),
    )
    total = views_per_latent * sum(item[1] for item in split_specs)
    output_dir.mkdir(parents=True, exist_ok=False)
    signals = np.lib.format.open_memmap(
        output_dir / "signals.npy",
        mode="w+",
        dtype=np.float32,
        shape=(total, OUTPUT_LENGTH),
    )
    rows: list[dict[str, Any]] = []
    signal_row = 0
    for split, count, split_seed, variant in split_specs:
        latent_rng = np.random.default_rng(split_seed)
        for latent_index in range(count):
            if generator == "data":
                latent = sample_data_oriented_latent(latent_rng, calibration)
            else:
                latent = sample_biophysics_oriented_latent(
                    latent_rng,
                    calibration,
                    amplitude_size_exponent=amplitude_size_exponent,
                    beam_radius_relative=beam_radius_relative,
                )
            latent_id = f"{split}-{latent_index:07d}"
            for view_index in range(views_per_latent):
                view_rng = np.random.default_rng(
                    split_seed
                    + latent_index * 1009
                    + view_index * 1_000_003
                )
                signal, nuisance = simulate_budding_view(
                    view_rng,
                    latent,
                    calibration,
                    variant=variant,
                )
                signals[signal_row] = signal
                rows.append(
                    {
                        "signal_row": signal_row,
                        "latent_id": latent_id,
                        "view_index": view_index,
                        "split": split,
                        "generator_variant": variant,
                        **latent.to_dict(),
                        **nuisance,
                    }
                )
                signal_row += 1
    signals.flush()
    del signals
    with (output_dir / "simulation_metadata.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "calibration.json").write_text(
        json.dumps(calibration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    factor_policy = {
        "shared_between_views": [
            *COPULA_FEATURES,
            "resolved",
            "relative_double_sphere_geometry",
        ],
        "randomized_between_views": [
            "event_midpoint_ms",
            "phase1_rad",
            "phase2_rad",
            "snr_db",
            "target_rms",
            "baseline_strength",
            "sensor_response",
        ],
        "excluded_unidentified": [
            "absolute_radius_um",
            "absolute_optical_amplitude",
            "absolute_velocity",
        ],
        "ambiguous_policy": calibration["m2_identifiability_rule"][
            "ambiguous_policy"
        ],
    }
    (output_dir / "factor_policy.json").write_text(
        json.dumps(factor_policy, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    dataset_id = (
        "yeast-budding-simulations-data@v1"
        if generator == "data"
        else "yeast-budding-simulations-biophysics@v1"
    )
    summary = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "generator_id": (
            "budding-data-oriented-rank-gaussian-copula-v1"
            if generator == "data"
            else "budding-biophysics-oriented-contacting-double-sphere-v1"
        ),
        "calibration_id": calibration["calibration_id"],
        "input_contract": "yeast-event-8192to4096-bandpass-global-v1",
        "n_signals": total,
        "n_latents": sum(item[1] for item in split_specs),
        "views_per_latent": views_per_latent,
        "split_signal_counts": dict(
            sorted(Counter(row["split"] for row in rows).items())
        ),
        "resolved_signal_counts": dict(
            sorted(Counter(str(row["resolved"]) for row in rows).items())
        ),
        "seed_policy": (
            "disjoint split seeds; deterministic per-latent and per-view seeds"
        ),
        "test_policy": "held-out synthetic sensor/noise response variant",
        "scientific_scope": (
            "mono-acquisition budding; relative two-sphere passage model; "
            "ambiguous second components are censored from morphology claims"
        ),
    }
    if generator == "biophysics":
        summary["biophysics_configuration"] = {
            "amplitude_size_exponent": amplitude_size_exponent,
            "beam_radius_relative": beam_radius_relative,
        }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
