from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.stats import qmc

from .yeast_representation_dataset import preprocess_crop


RAW_FS = 2_000_000.0
RAW_LENGTH = 8192
OUTPUT_LENGTH = 4096
LAMBDA0_UM = 1.55
THETA_DEG = 80.0
FAMILIES = ("S0", "T0", "M1")
QUANTILE_PROBABILITIES = np.linspace(0.01, 0.99, 99)


@dataclass(frozen=True)
class SchmooLatent:
    family: str
    calibration_group: str
    frequency_khz: float
    velocity_mm_s_effective: float
    base_diameter_um: float
    protrusion_length_um: float
    neck_diameter_um: float
    tip_radius_um: float
    orientation_deg: float
    orientation_kappa: float
    beam_major_2w0_um: float
    beam_minor_2w0_um: float
    beam_ellipse_angle_deg: float
    lateral_offset_um: float
    medium_refractive_index: float
    cell_refractive_index: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _quantile_contract(values: list[float]) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 8 or not np.all(np.isfinite(array)):
        raise ValueError("Calibration vectors require at least eight finite values")
    return np.quantile(array, QUANTILE_PROBABILITIES).tolist()


def build_schmoo_calibration(
    *,
    real_dataset_root: Path,
    source_dataset_id: str,
) -> dict[str, Any]:
    rows = _read_csv(real_dataset_root / "events.csv")
    signals = np.load(real_dataset_root / "signals.npy", mmap_mode="r")
    strata: dict[str, dict[str, Any]] = {}
    for source_group in ("shmoo", "shmoo2"):
        eligible = [
            row
            for row in rows
            if row["source_group"] == source_group
            and row["development_split"] == "development_train"
            and row["quality"] == "strict"
        ]
        if len(eligible) < 8:
            raise ValueError(f"Insufficient development_train events for {source_group}")
        frequencies: list[float] = []
        widths: list[float] = []
        target_rms: list[float] = []
        snr_db: list[float] = []
        for row in eligible:
            values = np.asarray(signals[int(row["signal_row"])], dtype=np.float64)
            centered = values - float(np.mean(values))
            start = max(0, int(float(row["event_start_input_index"])))
            end = min(values.size, int(float(row["event_end_input_index"])))
            outside = np.ones(values.size, dtype=bool)
            outside[start:end] = False
            event_rms = float(
                np.sqrt(np.mean(np.square(centered[start:end])))
            )
            noise_rms = float(
                np.sqrt(np.mean(np.square(centered[outside])))
            )
            frequencies.append(float(row["doppler_peak_hz"]) / 1000.0)
            widths.append(float(row["width_ms"]))
            target_rms.append(float(np.sqrt(np.mean(np.square(centered)))))
            snr_db.append(
                float(
                    np.clip(
                        20.0
                        * np.log10(event_rms / max(noise_rms, 1.0e-12)),
                        2.0,
                        30.0,
                    )
                )
            )
        strata[source_group] = {
            "n_events": len(eligible),
            "frequency_khz_quantiles": _quantile_contract(frequencies),
            "width_ms_quantiles": _quantile_contract(widths),
            "target_rms_quantiles": _quantile_contract(target_rms),
            "snr_db_quantiles": _quantile_contract(snr_db),
        }
    return {
        "schema_version": 1,
        "calibration_id": "yeast-schmoo-development-train-calibration@v1",
        "source_dataset": source_dataset_id,
        "source_split": "development_train",
        "source_groups": ["shmoo", "shmoo2"],
        "quality": "strict",
        "strata": strata,
        "quantile_probabilities": QUANTILE_PROBABILITIES.tolist(),
        "sealed_splits_used": [],
        "source_checksums": {
            "events.csv": _sha256(real_dataset_root / "events.csv"),
            "signals.npy": _sha256(real_dataset_root / "signals.npy"),
        },
        "scientific_scope": (
            "frequency, width, RMS and noise calibration only; acquisition "
            "conditions are not treated as single-cell morphology labels"
        ),
    }


def _sample_quantile(
    rng: np.random.Generator,
    values: list[float],
    probabilities: list[float],
) -> float:
    probability = float(rng.uniform(probabilities[0], probabilities[-1]))
    return float(np.interp(probability, probabilities, values))


def sample_schmoo_latent(
    rng: np.random.Generator,
    calibration: dict[str, Any],
    *,
    family: str,
) -> SchmooLatent:
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}")
    source_group = str(rng.choice(("shmoo", "shmoo2")))
    contract = calibration["strata"][source_group]
    probabilities = calibration["quantile_probabilities"]
    frequency_khz = _sample_quantile(
        rng,
        contract["frequency_khz_quantiles"],
        probabilities,
    )
    theta = np.deg2rad(THETA_DEG)
    velocity_mm_s = (
        frequency_khz
        * 1000.0
        * LAMBDA0_UM
        * 1.0e-6
        / (2.0 * np.cos(theta))
        * 1000.0
    )
    kappa = float(rng.choice((0.0, 2.0, 8.0)))
    orientation = float(
        np.rad2deg(rng.uniform(-np.pi / 2.0, np.pi / 2.0))
        if kappa == 0.0
        else np.rad2deg(rng.vonmises(0.0, kappa))
    )
    orientation = ((orientation + 90.0) % 180.0) - 90.0
    base_diameter = float(rng.choice((6.0, 8.0, 10.0)))
    if family == "S0":
        protrusion_length = neck_diameter = tip_radius = 0.0
    else:
        protrusion_length = float(rng.choice((1.5, 3.5, 6.0)))
        neck_diameter = float(rng.choice((2.0, 3.5, 5.0)))
        tip_radius = (
            0.0
            if family == "T0"
            else float(rng.choice((0.75, 1.5, 2.5)))
        )
    return SchmooLatent(
        family=family,
        calibration_group=source_group,
        frequency_khz=frequency_khz,
        velocity_mm_s_effective=velocity_mm_s,
        base_diameter_um=base_diameter,
        protrusion_length_um=protrusion_length,
        neck_diameter_um=min(neck_diameter, 0.95 * base_diameter),
        tip_radius_um=min(tip_radius, 0.48 * base_diameter),
        orientation_deg=orientation,
        orientation_kappa=kappa,
        beam_major_2w0_um=float(rng.uniform(60.0, 85.0)),
        beam_minor_2w0_um=float(rng.uniform(35.0, 55.0)),
        beam_ellipse_angle_deg=float(rng.uniform(0.0, 180.0)),
        lateral_offset_um=float(rng.uniform(-13.34, 13.34)),
        medium_refractive_index=float(rng.uniform(1.310, 1.325)),
        cell_refractive_index=float(rng.uniform(1.35, 1.42)),
    )


def _halton_points(n: int) -> np.ndarray:
    return qmc.Halton(d=3, scramble=False).random(n + 1)[1:]


def _sphere_cloud(radius: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    unit = _halton_points(n)
    radial = radius * np.cbrt(unit[:, 0])
    cosine = 2.0 * unit[:, 1] - 1.0
    sine = np.sqrt(np.maximum(0.0, 1.0 - np.square(cosine)))
    azimuth = 2.0 * np.pi * unit[:, 2]
    points = np.column_stack(
        (
            radial * cosine,
            radial * sine * np.cos(azimuth),
            radial * sine * np.sin(azimuth),
        )
    )
    volume = 4.0 / 3.0 * np.pi * radius**3
    return points, np.full(n, volume / n, dtype=np.float64)


def _cone_cloud(
    *,
    start_x: float,
    length: float,
    base_radius: float,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    unit = _halton_points(n)
    axial = 1.0 - np.cbrt(1.0 - unit[:, 0])
    radius = base_radius * (1.0 - axial)
    disk_radius = radius * np.sqrt(unit[:, 1])
    azimuth = 2.0 * np.pi * unit[:, 2]
    points = np.column_stack(
        (
            start_x + length * axial,
            disk_radius * np.cos(azimuth),
            disk_radius * np.sin(azimuth),
        )
    )
    volume = np.pi * base_radius**2 * length / 3.0
    return points, np.full(n, volume / n, dtype=np.float64)


def _rounded_protrusion_cloud(
    *,
    start_x: float,
    length: float,
    neck_radius: float,
    tip_radius: float,
    n_shaft: int,
    n_tip: int,
) -> tuple[np.ndarray, np.ndarray]:
    effective_tip = min(tip_radius, 0.90 * length, neck_radius)
    shaft_length = max(length - effective_tip, 0.15)
    unit = _halton_points(n_shaft)
    axial = unit[:, 0]
    local_radius = neck_radius + (effective_tip - neck_radius) * axial
    disk_radius = local_radius * np.sqrt(unit[:, 1])
    azimuth = 2.0 * np.pi * unit[:, 2]
    shaft = np.column_stack(
        (
            start_x + shaft_length * axial,
            disk_radius * np.cos(azimuth),
            disk_radius * np.sin(azimuth),
        )
    )
    shaft_volume = (
        np.pi
        * shaft_length
        * (
            neck_radius**2
            + neck_radius * effective_tip
            + effective_tip**2
        )
        / 3.0
    )
    shaft_weights = np.full(
        n_shaft,
        shaft_volume / n_shaft,
        dtype=np.float64,
    )
    tip, tip_weights = _sphere_cloud(effective_tip, n_tip)
    tip[:, 0] = np.abs(tip[:, 0]) + start_x + shaft_length
    tip_volume = 2.0 / 3.0 * np.pi * effective_tip**3
    tip_weights[:] = tip_volume / n_tip
    return (
        np.vstack((shaft, tip)),
        np.concatenate((shaft_weights, tip_weights)),
    )


def shape_cloud(latent: SchmooLatent) -> tuple[np.ndarray, np.ndarray]:
    radius = latent.base_diameter_um / 2.0
    base, base_weights = _sphere_cloud(radius, 72 if latent.family != "S0" else 96)
    if latent.family == "S0":
        return base, base_weights
    neck_radius = latent.neck_diameter_um / 2.0
    start_x = float(np.sqrt(max(radius**2 - neck_radius**2, 0.0)))
    if latent.family == "T0":
        protrusion, protrusion_weights = _cone_cloud(
            start_x=start_x,
            length=latent.protrusion_length_um,
            base_radius=neck_radius,
            n=48,
        )
    else:
        protrusion, protrusion_weights = _rounded_protrusion_cloud(
            start_x=start_x,
            length=latent.protrusion_length_um,
            neck_radius=neck_radius,
            tip_radius=latent.tip_radius_um,
            n_shaft=36,
            n_tip=24,
        )
    return (
        np.vstack((base, protrusion)),
        np.concatenate((base_weights, protrusion_weights)),
    )


def _colored_noise(
    rng: np.random.Generator,
    length: int,
    *,
    variant: str,
) -> np.ndarray:
    white = rng.normal(size=length)
    cutoff = 52_000.0 if variant == "heldout_sensor" else 75_000.0
    order = 3 if variant == "heldout_sensor" else 2
    sos = butter(order, cutoff, btype="lowpass", fs=RAW_FS, output="sos")
    values = sosfiltfilt(sos, white)
    return values / max(float(np.std(values)), 1.0e-12)


def simulate_schmoo_view(
    rng: np.random.Generator,
    latent: SchmooLatent,
    calibration: dict[str, Any],
    *,
    variant: str,
) -> tuple[np.ndarray, dict[str, float]]:
    points, weights = shape_cloud(latent)
    orientation = np.deg2rad(latent.orientation_deg)
    x = points[:, 0] * np.cos(orientation) - points[:, 1] * np.sin(orientation)
    y = points[:, 0] * np.sin(orientation) + points[:, 1] * np.cos(orientation)
    z = points[:, 2]
    theta = np.deg2rad(THETA_DEG)
    beam_parallel = x * np.cos(theta) + y * np.sin(theta)
    beam_transverse = x * np.sin(theta) - y * np.cos(theta)
    scatter_phase = (
        4.0
        * np.pi
        * latent.medium_refractive_index
        * beam_parallel
        / LAMBDA0_UM
    )
    contrast = (
        np.square(latent.cell_refractive_index)
        / np.square(latent.medium_refractive_index)
        - 1.0
    )
    complex_weights = contrast * weights * np.exp(1j * scatter_phase)

    time_s = np.arange(RAW_LENGTH, dtype=np.float64) / RAW_FS
    midpoint_s = float(rng.uniform(0.9e-3, 3.2e-3))
    cell_transverse = (
        latent.velocity_mm_s_effective
        * 1000.0
        * np.sin(theta)
        * (time_s - midpoint_s)
    )
    p = cell_transverse[:, None] + beam_transverse[None, :]
    vertical = latent.lateral_offset_um + z
    ellipse = np.deg2rad(latent.beam_ellipse_angle_deg)
    u = p * np.cos(ellipse) + vertical[None, :] * np.sin(ellipse)
    w = -p * np.sin(ellipse) + vertical[None, :] * np.cos(ellipse)
    w_major = latent.beam_major_2w0_um / 2.0
    w_minor = latent.beam_minor_2w0_um / 2.0
    gaussian_field = np.exp(
        -np.square(u / w_major) - np.square(w / w_minor)
    )
    shape_field = gaussian_field @ complex_weights
    shape_field /= max(float(np.max(np.abs(shape_field))), 1.0e-12)
    carrier_phase = (
        2.0
        * np.pi
        * latent.frequency_khz
        * 1000.0
        * (time_s - midpoint_s)
        + float(rng.uniform(0.0, 2.0 * np.pi))
    )
    clean = np.real(shape_field * np.exp(1j * carrier_phase))
    clean_rms = max(float(np.sqrt(np.mean(np.square(clean)))), 1.0e-12)
    contract = calibration["strata"][latent.calibration_group]
    probabilities = calibration["quantile_probabilities"]
    snr_db = _sample_quantile(
        rng,
        contract["snr_db_quantiles"],
        probabilities,
    )
    noise = (
        _colored_noise(rng, RAW_LENGTH, variant=variant)
        * clean_rms
        / (10.0 ** (snr_db / 20.0))
    )
    baseline = (
        float(rng.uniform(0.0, 0.08))
        * clean_rms
        * np.sin(
            2.0 * np.pi * float(rng.uniform(80.0, 800.0)) * time_s
            + float(rng.uniform(0.0, 2.0 * np.pi))
        )
    )
    processed = preprocess_crop((clean + noise + baseline).astype(np.float32))
    target_rms = _sample_quantile(
        rng,
        contract["target_rms_quantiles"],
        probabilities,
    )
    processed *= target_rms / max(
        float(np.sqrt(np.mean(np.square(processed)))),
        1.0e-12,
    )
    return processed.astype(np.float32), {
        "event_midpoint_ms": midpoint_s * 1000.0,
        "snr_db": snr_db,
        "target_rms": target_rms,
        "shape_field_peak": float(np.max(np.abs(shape_field))),
        "shape_field_phase_span_rad": float(
            np.ptp(np.unwrap(np.angle(shape_field[np.abs(shape_field) > 0.05])))
        ),
        "n_shape_points": int(points.shape[0]),
    }


def build_schmoo_physical_sweep(
    *,
    output_dir: Path,
    calibration: dict[str, Any],
    n_train_per_family: int,
    n_validation_per_family: int,
    n_test_per_family: int,
    seed: int = 190726,
) -> dict[str, Any]:
    if calibration.get("source_split") != "development_train":
        raise PermissionError("Schmoo calibration must use development_train only")
    if calibration.get("sealed_splits_used") != []:
        raise PermissionError("Schmoo calibration accessed a sealed split")
    if min(n_train_per_family, n_validation_per_family, n_test_per_family) <= 0:
        raise ValueError("All split counts must be positive")
    split_specs = (
        ("train", n_train_per_family, seed, "base"),
        ("validation", n_validation_per_family, seed + 10_000, "base"),
        ("test", n_test_per_family, seed + 20_000, "heldout_sensor"),
    )
    total = len(FAMILIES) * sum(spec[1] for spec in split_specs)
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
        for family_index, family in enumerate(FAMILIES):
            family_seed = split_seed + family_index * 1_000_003
            latent_rng = np.random.default_rng(family_seed)
            for latent_index in range(count):
                latent = sample_schmoo_latent(
                    latent_rng,
                    calibration,
                    family=family,
                )
                view_rng = np.random.default_rng(
                    family_seed + latent_index * 1009 + 31
                )
                signal, nuisance = simulate_schmoo_view(
                    view_rng,
                    latent,
                    calibration,
                    variant=variant,
                )
                signals[signal_row] = signal
                rows.append(
                    {
                        "signal_row": signal_row,
                        "latent_id": f"{family}-{split}-{latent_index:07d}",
                        "view_index": 0,
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
    parameter_contract = {
        "schema_version": 1,
        "families": {
            "S0": "sphere control under the thesis-measured elliptical beam",
            "T0": "sphere plus sharp right-cone protrusion; triangle ablation revolved into 3D",
            "M1": "sphere plus tapered shaft and rounded hemispherical tip",
        },
        "instrument": {
            "lambda0_nm": 1550.0,
            "theta_deg": THETA_DEG,
            "beam_2w0_major_um": [60.0, 85.0],
            "beam_2w0_minor_um": [35.0, 55.0],
            "lateral_offset_um": [-13.34, 13.34],
        },
        "geometry": {
            "base_diameter_um": [6.0, 8.0, 10.0],
            "protrusion_length_um": [1.5, 3.5, 6.0],
            "neck_diameter_um": [2.0, 3.5, 5.0],
            "tip_radius_um": [0.75, 1.5, 2.5],
        },
        "scattering": {
            "model": "coherent weighted-volume Born surrogate",
            "medium_refractive_index": [1.310, 1.325],
            "cell_refractive_index": [1.35, 1.42],
            "reference_gate": "future GLMT sphere and DDA composite spot checks",
        },
        "doppler": {
            "policy": "sample f_D from real development_train strata",
            "velocity_role": "latent effective velocity from thesis Eq. 2.20 convention",
        },
    }
    (output_dir / "parameter_contract.json").write_text(
        json.dumps(parameter_contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "dataset_id": "yeast-schmoo-physical-sweep@v1",
        "generator_id": "schmoo-wide-beam-coherent-volume-sweep-v1",
        "calibration_id": calibration["calibration_id"],
        "input_contract": "yeast-event-8192to4096-bandpass-global-v1",
        "n_signals": total,
        "n_latents": total,
        "views_per_latent": 1,
        "family_counts": dict(sorted(Counter(row["family"] for row in rows).items())),
        "split_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
        "seed": seed,
        "sealed_splits_used_for_generation": [],
        "scientific_scope": (
            "small model-selection sweep only; Born surrogate is not an "
            "absolute optical cross-section model and mass SSL use is blocked"
        ),
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
