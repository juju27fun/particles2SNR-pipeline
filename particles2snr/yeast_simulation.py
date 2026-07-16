from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

from .yeast_representation_dataset import preprocess_crop


FACTOR_POLICY: dict[str, dict[str, Any]] = {
    "duration_ms": {"role": "preserve_predict", "range": [0.464, 1.424], "source": "real candidate p05-p95"},
    "doppler_khz": {"role": "preserve_predict", "range": [7.8125, 23.4375], "source": "real candidate p05-p95"},
    "component_count": {"role": "preserve_predict", "range": [1, 2], "source": "single/multi-passage hypothesis"},
    "component_separation_ms": {"role": "preserve_predict", "range": [0.08, 0.70], "source": "bounded synthetic factor"},
    "relative_component_amplitude": {"role": "preserve_predict", "range": [0.40, 1.00], "source": "bounded synthetic factor"},
    "frequency_separation_khz": {"role": "preserve_predict", "range": [0.0, 8.0], "source": "bounded synthetic factor"},
    "phase_rad": {"role": "randomize_invariant", "range": [0.0, 6.283185307179586]},
    "event_position_fraction": {"role": "randomize_invariant", "range": [0.10, 0.90]},
    "snr_db": {"role": "randomize_invariant", "range": [0.0, 30.0]},
    "target_rms": {"role": "randomize_invariant", "range": [0.40, 1.70], "source": "real normalized p05-p95"},
    "baseline_drift": {"role": "randomize_invariant", "range": [0.0, 0.30]},
    "sensor_response": {"role": "randomize_invariant", "range": [0.85, 1.15]},
    "absolute_physical_amplitude": {"role": "unresolved_excluded", "reason": "not identifiable after acquisition gain"},
    "yeast_morphology": {"role": "unresolved_excluded", "reason": "simulated components are not validated morphology labels"},
}

SUPPORT_CALIBRATION_ID = "yeast-followup-train-hilbert-support-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def envelope_support_duration_ms(
    signals: np.ndarray, *, sampling_frequency_hz: float = 1_000_000.0
) -> np.ndarray:
    values = np.asarray(signals, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("signals must have shape (n_signals, n_samples)")
    if sampling_frequency_hz <= 0.0:
        raise ValueError("sampling_frequency_hz must be positive")
    envelope = np.abs(hilbert(values, axis=1))
    threshold = 0.25 * np.max(envelope, axis=1, keepdims=True)
    duration = np.sum(envelope >= threshold, axis=1) / sampling_frequency_hz * 1000.0
    if not np.isfinite(duration).all():
        raise ValueError("Envelope support calculation produced non-finite values")
    return duration.astype(np.float64)


def fit_support_calibration(
    real_root: Path,
    *,
    quantile_knots: int = 101,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
) -> dict[str, Any]:
    if quantile_knots < 3:
        raise ValueError("quantile_knots must be at least three")
    if not 0.0 < lower_quantile < upper_quantile < 1.0:
        raise ValueError("Calibration quantiles must lie strictly inside (0, 1)")
    development_path = real_root / "development_events.csv"
    if not development_path.is_file():
        raise FileNotFoundError(
            "Support calibration requires the physically separated development_events.csv"
        )
    with development_path.open(newline="", encoding="utf-8") as handle:
        development_rows = list(csv.DictReader(handle))
    observed_splits = {row["development_split"] for row in development_rows}
    allowed_splits = {"followup_train", "followup_validation"}
    if not observed_splits <= allowed_splits:
        raise PermissionError(
            f"Support calibration development metadata contains forbidden splits: "
            f"{sorted(observed_splits - allowed_splits)}"
        )
    train_rows = [row for row in development_rows if row["development_split"] == "followup_train"]
    if not train_rows:
        raise ValueError("Support calibration requires followup_train signals")
    signals_path = real_root / "signals.npy"
    signals = np.load(signals_path, mmap_mode="r")
    indices = np.asarray([int(row["signal_row"]) for row in train_rows], dtype=np.int64)
    if indices.min() < 0 or indices.max() >= len(signals):
        raise ValueError("Training signal row lies outside signals.npy")
    durations = []
    for start in range(0, len(indices), 256):
        selected = np.asarray(signals[indices[start : start + 256]], dtype=np.float32)
        durations.append(envelope_support_duration_ms(selected))
    support = np.concatenate(durations)
    probabilities = np.linspace(lower_quantile, upper_quantile, quantile_knots)
    values = np.quantile(support, probabilities)
    if np.any(np.diff(values) < 0.0) or values[0] <= 0.0:
        raise ValueError("Invalid support-duration quantiles")
    summary_path = real_root / "dataset_summary.json"
    dataset_id = "yeast-events-followup@v2"
    if summary_path.is_file():
        dataset_id = json.loads(summary_path.read_text(encoding="utf-8")).get(
            "dataset_id", dataset_id
        )
    return {
        "schema_version": 1,
        "calibration_id": SUPPORT_CALIBRATION_ID,
        "source_dataset": dataset_id,
        "source_split": "followup_train",
        "source_metadata": "development_events.csv",
        "n_train_signals": len(train_rows),
        "observable": {
            "name": "hilbert_envelope_support_above_25pct_peak_ms",
            "sampling_frequency_hz": 1_000_000.0,
            "threshold_fraction": 0.25,
        },
        "robust_quantile_interval": [lower_quantile, upper_quantile],
        "quantile_probabilities": probabilities.tolist(),
        "support_duration_ms_quantiles": values.tolist(),
        "source_checksums": {
            "development_events.csv": _sha256(development_path),
            "signals.npy": _sha256(signals_path),
        },
        "sealed_splits_used": [],
        "trace_policy": "quantiles only; no real signal or template is copied into simulation",
    }


def _sample_calibrated_duration_ms(
    rng: np.random.Generator, calibration: dict[str, Any]
) -> float:
    probabilities = np.asarray(calibration["quantile_probabilities"], dtype=np.float64)
    values = np.asarray(calibration["support_duration_ms_quantiles"], dtype=np.float64)
    if probabilities.ndim != 1 or values.shape != probabilities.shape:
        raise ValueError("Malformed support calibration quantiles")
    probability = rng.uniform(float(probabilities[0]), float(probabilities[-1]))
    return float(np.interp(probability, probabilities, values))


def _finite_support_tukey_envelope(
    time: np.ndarray,
    *,
    center: float,
    target_support_ms: float,
    alpha: float,
) -> np.ndarray:
    if not 0.0 < alpha <= 1.0:
        raise ValueError("Tukey alpha must lie in (0, 1]")
    if target_support_ms <= 0.0:
        raise ValueError("target_support_ms must be positive")
    # At 25% height, Tukey support spans (1 - alpha / 3) of its finite support.
    total_support = target_support_ms / (1.0 - alpha / 3.0) / 1000.0
    normalized = np.abs(time - center) / max(total_support / 2.0, 1.0e-12)
    envelope = np.zeros_like(time, dtype=np.float64)
    inside = normalized <= 1.0
    flat = normalized <= 1.0 - alpha
    envelope[flat] = 1.0
    taper = inside & ~flat
    envelope[taper] = 0.5 * (
        1.0 + np.cos(np.pi * (normalized[taper] - (1.0 - alpha)) / alpha)
    )
    return envelope


def _colored_noise(rng: np.random.Generator, length: int, variant: str) -> np.ndarray:
    white = rng.normal(size=length).astype(np.float32)
    if variant == "heldout_sensor":
        cutoff = 55_000.0
        order = 3
    else:
        cutoff = 75_000.0
        order = 2
    sos = butter(order, cutoff, btype="lowpass", fs=2_000_000.0, output="sos")
    colored = sosfiltfilt(sos, white).astype(np.float32)
    colored /= max(float(np.std(colored)), 1.0e-6)
    return colored


def _latent_factors(rng: np.random.Generator) -> dict[str, float | int]:
    component_count = 2 if rng.random() < 0.30 else 1
    return {
        "duration_ms": float(rng.uniform(0.464, 1.424)),
        "doppler_khz": float(rng.uniform(7.8125, 23.4375)),
        "component_count": component_count,
        "component_separation_ms": float(rng.uniform(0.08, 0.70)) if component_count == 2 else 0.0,
        "relative_component_amplitude": float(rng.uniform(0.40, 1.00)) if component_count == 2 else 0.0,
        "frequency_separation_khz": float(rng.uniform(0.0, 8.0)) if component_count == 2 else 0.0,
    }


def simulate_view(
    rng: np.random.Generator,
    factors: dict[str, float | int],
    *,
    variant: str = "base",
    raw_length: int = 8192,
    envelope_model: str = "gaussian",
    tukey_alpha: float = 0.50,
) -> tuple[np.ndarray, dict[str, float]]:
    sampling_frequency = 2_000_000.0
    time = np.arange(raw_length, dtype=np.float64) / sampling_frequency
    position = float(rng.uniform(0.10, 0.90))
    center = position * (raw_length - 1) / sampling_frequency
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    frequency = float(factors["doppler_khz"]) * 1000.0
    if envelope_model == "gaussian":
        duration_s = float(factors["duration_ms"]) / 1000.0
        sigma = duration_s / 2.355
        envelope = np.exp(-0.5 * np.square((time - center) / sigma))
    elif envelope_model == "finite_support_tukey":
        envelope = _finite_support_tukey_envelope(
            time,
            center=center,
            target_support_ms=float(factors["duration_ms"]),
            alpha=tukey_alpha,
        )
    else:
        raise ValueError(f"Unknown envelope model: {envelope_model}")
    clean = envelope * np.cos(2.0 * np.pi * frequency * time + phase)
    if int(factors["component_count"]) == 2:
        separation = float(factors["component_separation_ms"]) / 1000.0
        second_center = np.clip(center + rng.choice((-0.5, 0.5)) * separation, time[0], time[-1])
        second_frequency = frequency + rng.choice((-1.0, 1.0)) * float(factors["frequency_separation_khz"]) * 1000.0
        second_phase = float(rng.uniform(0.0, 2.0 * np.pi))
        if envelope_model == "gaussian":
            second_envelope = np.exp(-0.5 * np.square((time - second_center) / sigma))
        else:
            second_envelope = _finite_support_tukey_envelope(
                time,
                center=float(second_center),
                target_support_ms=float(factors["duration_ms"]),
                alpha=tukey_alpha,
            )
        clean = clean + float(factors["relative_component_amplitude"]) * second_envelope * np.cos(
            2.0 * np.pi * second_frequency * time + second_phase
        )
    clean = clean.astype(np.float32)
    snr_db = float(rng.uniform(0.0, 30.0))
    clean_rms = max(float(np.sqrt(np.mean(np.square(clean)))), 1.0e-6)
    noise = _colored_noise(rng, raw_length, variant) * clean_rms / (10.0 ** (snr_db / 20.0))
    baseline_strength = float(rng.uniform(0.0, 0.30))
    baseline = baseline_strength * np.sin(
        2.0 * np.pi * rng.uniform(80.0, 800.0) * time + rng.uniform(0.0, 2.0 * np.pi)
    )
    response = float(rng.uniform(0.85, 1.15))
    raw = response * (clean + noise + baseline).astype(np.float32)
    processed = preprocess_crop(raw)
    target_rms = float(rng.uniform(0.40, 1.70))
    processed *= target_rms / max(float(np.sqrt(np.mean(np.square(processed)))), 1.0e-6)
    nuisance = {
        "phase_rad": phase,
        "event_position_fraction": position,
        "snr_db": snr_db,
        "target_rms": target_rms,
        "baseline_drift": baseline_strength,
        "sensor_response": response,
    }
    return processed.astype(np.float32), nuisance


def build_simulation_dataset(
    *,
    output_dir: Path,
    n_train_latents: int,
    n_validation_latents: int,
    n_test_latents: int,
    views_per_latent: int = 2,
    seed: int = 42,
    support_calibration: dict[str, Any] | None = None,
    envelope_model: str = "gaussian",
    tukey_alpha: float = 0.50,
) -> dict[str, Any]:
    if min(n_train_latents, n_validation_latents, n_test_latents, views_per_latent) <= 0:
        raise ValueError("Simulation counts must be positive")
    if envelope_model == "gaussian" and support_calibration is not None:
        raise ValueError("Gaussian v1 generation cannot consume support calibration")
    if envelope_model == "finite_support_tukey":
        if support_calibration is None:
            raise ValueError("Finite-support generation requires train-only calibration")
        if support_calibration.get("sealed_splits_used") != []:
            raise PermissionError("Support calibration accessed a sealed split")
        if support_calibration.get("source_split") != "followup_train":
            raise PermissionError("Support calibration must use followup_train only")
    elif envelope_model != "gaussian":
        raise ValueError(f"Unknown envelope model: {envelope_model}")
    split_specs = (
        ("train", n_train_latents, seed, "base"),
        ("validation", n_validation_latents, seed + 10_000, "base"),
        ("test", n_test_latents, seed + 20_000, "heldout_sensor"),
    )
    total = views_per_latent * sum(spec[1] for spec in split_specs)
    output_dir.mkdir(parents=True, exist_ok=False)
    signals = np.lib.format.open_memmap(
        output_dir / "signals.npy", mode="w+", dtype=np.float32, shape=(total, 4096)
    )
    rows: list[dict[str, Any]] = []
    signal_row = 0
    for split, count, split_seed, variant in split_specs:
        latent_rng = np.random.default_rng(split_seed)
        for latent_index in range(count):
            factors = _latent_factors(latent_rng)
            if support_calibration is not None:
                factors["duration_ms"] = _sample_calibrated_duration_ms(
                    latent_rng, support_calibration
                )
            latent_id = f"{split}-{latent_index:07d}"
            for view_index in range(views_per_latent):
                view_rng = np.random.default_rng(split_seed + latent_index * 1009 + view_index * 1_000_003)
                signal, nuisance = simulate_view(
                    view_rng,
                    factors,
                    variant=variant,
                    envelope_model=envelope_model,
                    tukey_alpha=tukey_alpha,
                )
                signals[signal_row] = signal
                rows.append(
                    {
                        "signal_row": signal_row,
                        "latent_id": latent_id,
                        "view_index": view_index,
                        "split": split,
                        "generator_variant": variant,
                        **(
                            {"envelope_model": envelope_model, "tukey_alpha": tukey_alpha}
                            if support_calibration is not None
                            else {}
                        ),
                        **factors,
                        **nuisance,
                    }
                )
                signal_row += 1
    signals.flush()
    del signals
    with (output_dir / "simulation_metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    factor_policy = json.loads(json.dumps(FACTOR_POLICY))
    if support_calibration is not None:
        factor_policy["duration_ms"]["source"] = (
            "followup_train Hilbert-support p05-p95 quantile calibration"
        )
        factor_policy["duration_ms"]["range"] = [
            support_calibration["support_duration_ms_quantiles"][0],
            support_calibration["support_duration_ms_quantiles"][-1],
        ]
    (output_dir / "factor_policy.json").write_text(
        json.dumps(factor_policy, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if support_calibration is not None:
        (output_dir / "support_calibration.json").write_text(
            json.dumps(support_calibration, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    summary = {
        "schema_version": 1,
        "generator_id": (
            "yeast-passage-finite-support-v2"
            if envelope_model == "finite_support_tukey"
            else "yeast-passage-identifiable-v1"
        ),
        "input_contract": "yeast-event-8192to4096-bandpass-global-v1-compatible",
        "n_signals": total,
        "n_latents": sum(spec[1] for spec in split_specs),
        "views_per_latent": views_per_latent,
        "split_signal_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
        "generator_variant_counts": dict(sorted(Counter(row["generator_variant"] for row in rows).items())),
        "seed_policy": "disjoint split seeds; deterministic per-latent per-view nuisance seeds",
        "test_policy": "held-out sensor/noise response variant",
        "scientific_limit": "components are generic passage factors and are not validated yeast morphology labels",
    }
    if support_calibration is not None:
        summary.update(
            {
                "envelope_model": envelope_model,
                "envelope_model_id": f"finite-support-tukey-alpha-{tukey_alpha:g}-v1",
                "support_calibration": support_calibration["calibration_id"],
            }
        )
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def build_support_calibrated_simulation_dataset(
    *,
    real_root: Path,
    output_dir: Path,
    n_train_latents: int,
    n_validation_latents: int,
    n_test_latents: int,
    views_per_latent: int = 2,
    seed: int = 42,
    tukey_alpha: float = 0.50,
    quantile_knots: int = 101,
) -> dict[str, Any]:
    calibration = fit_support_calibration(real_root, quantile_knots=quantile_knots)
    return build_simulation_dataset(
        output_dir=output_dir,
        n_train_latents=n_train_latents,
        n_validation_latents=n_validation_latents,
        n_test_latents=n_test_latents,
        views_per_latent=views_per_latent,
        seed=seed,
        support_calibration=calibration,
        envelope_model="finite_support_tukey",
        tukey_alpha=tukey_alpha,
    )
