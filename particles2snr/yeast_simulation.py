from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt

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
) -> tuple[np.ndarray, dict[str, float]]:
    sampling_frequency = 2_000_000.0
    time = np.arange(raw_length, dtype=np.float64) / sampling_frequency
    duration_s = float(factors["duration_ms"]) / 1000.0
    sigma = duration_s / 2.355
    position = float(rng.uniform(0.10, 0.90))
    center = position * (raw_length - 1) / sampling_frequency
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    frequency = float(factors["doppler_khz"]) * 1000.0
    envelope = np.exp(-0.5 * np.square((time - center) / sigma))
    clean = envelope * np.cos(2.0 * np.pi * frequency * time + phase)
    if int(factors["component_count"]) == 2:
        separation = float(factors["component_separation_ms"]) / 1000.0
        second_center = np.clip(center + rng.choice((-0.5, 0.5)) * separation, time[0], time[-1])
        second_frequency = frequency + rng.choice((-1.0, 1.0)) * float(factors["frequency_separation_khz"]) * 1000.0
        second_phase = float(rng.uniform(0.0, 2.0 * np.pi))
        second_envelope = np.exp(-0.5 * np.square((time - second_center) / sigma))
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
) -> dict[str, Any]:
    if min(n_train_latents, n_validation_latents, n_test_latents, views_per_latent) <= 0:
        raise ValueError("Simulation counts must be positive")
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
            latent_id = f"{split}-{latent_index:07d}"
            for view_index in range(views_per_latent):
                view_rng = np.random.default_rng(split_seed + latent_index * 1009 + view_index * 1_000_003)
                signal, nuisance = simulate_view(view_rng, factors, variant=variant)
                signals[signal_row] = signal
                rows.append(
                    {
                        "signal_row": signal_row,
                        "latent_id": latent_id,
                        "view_index": view_index,
                        "split": split,
                        "generator_variant": variant,
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
    (output_dir / "factor_policy.json").write_text(
        json.dumps(FACTOR_POLICY, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "schema_version": 1,
        "generator_id": "yeast-passage-identifiable-v1",
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
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
