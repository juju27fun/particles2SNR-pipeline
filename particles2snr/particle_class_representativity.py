from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from .particle_class_coverage import (
    CLASS_COLORS,
    CLASS_LABELS,
    CLASS_ORDER,
    FEATURE_LABELS,
    FEATURE_NAMES,
    load_real_events,
    load_simulations,
)
from .ssl_realism_audit import FS, SignalRecord


SPHERE_WEIGHTS = {
    "duration_25_ms": 0.25,
    "dominant_frequency_khz": 0.25,
    "spectral_bandwidth_khz": 0.15,
    "envelope_concentration": 0.10,
    "temporal_peak_count": 0.08,
    "spectral_peak_count": 0.05,
    "envelope_peak_over_rms": 0.06,
    "event_energy_fraction": 0.06,
}
CALIBRATION_FRACTION = 0.20
CALIBRATION_SEED = 20260719


def _matrix(records: list[SignalRecord]) -> np.ndarray:
    if not records:
        raise ValueError("Descriptor population is empty")
    return np.asarray(
        [[row.descriptors[name] for name in FEATURE_NAMES] for row in records],
        dtype=np.float64,
    )


def _group_partition(
    populations: dict[str, list[SignalRecord]],
    *,
    calibration_fraction: float,
    seed: int,
) -> tuple[dict[str, list[SignalRecord]], dict[str, list[SignalRecord]], dict[str, Any]]:
    if not 0.0 < calibration_fraction < 0.5:
        raise ValueError("calibration_fraction must be between zero and 0.5")
    all_groups = sorted(
        {
            str(row.metadata["source_group"])
            for rows in populations.values()
            for row in rows
        }
    )
    if len(all_groups) < 5:
        raise ValueError("At least five source groups are required")
    rng = np.random.default_rng(seed)
    shuffled = [all_groups[index] for index in rng.permutation(len(all_groups))]
    n_calibration = max(1, int(round(len(all_groups) * calibration_fraction)))
    calibration_groups = set(shuffled[:n_calibration])

    fit: dict[str, list[SignalRecord]] = {}
    calibration: dict[str, list[SignalRecord]] = {}
    for class_name in CLASS_ORDER:
        fit[class_name] = [
            row
            for row in populations[class_name]
            if str(row.metadata["source_group"]) not in calibration_groups
        ]
        calibration[class_name] = [
            row
            for row in populations[class_name]
            if str(row.metadata["source_group"]) in calibration_groups
        ]
        if len(fit[class_name]) < 10 or len(calibration[class_name]) < 5:
            raise ValueError(
                f"Insufficient group-disjoint fit/calibration rows for {class_name}"
            )
    return fit, calibration, {
        "seed": seed,
        "calibration_fraction": calibration_fraction,
        "fit_source_groups": sorted(set(all_groups) - calibration_groups),
        "calibration_source_groups": sorted(calibration_groups),
    }


def _robust_scale(records: list[SignalRecord]) -> dict[str, float]:
    values = _matrix(records)
    scales: dict[str, float] = {}
    for index, name in enumerate(FEATURE_NAMES):
        scale = float(
            np.quantile(values[:, index], 0.75)
            - np.quantile(values[:, index], 0.25)
        )
        if name in {"temporal_peak_count", "spectral_peak_count"}:
            scale = max(scale, 1.0)
        scales[name] = max(scale, 1.0e-6)
    return scales


def _sphere_distance(
    records: list[SignalRecord],
    *,
    center: dict[str, float],
    scales: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    values = _matrix(records)
    center_vector = np.asarray(
        [center[name] for name in FEATURE_NAMES], dtype=np.float64
    )
    scale_vector = np.asarray(
        [scales[name] for name in FEATURE_NAMES], dtype=np.float64
    )
    weight_vector = np.asarray(
        [SPHERE_WEIGHTS[name] for name in FEATURE_NAMES], dtype=np.float64
    )
    contributions = np.square((values - center_vector) / scale_vector) * weight_vector
    return np.sqrt(np.sum(contributions, axis=1)), contributions


def _nearest_real(
    simulation: SignalRecord,
    real_records: list[SignalRecord],
    *,
    scales: dict[str, float],
) -> int:
    query = _matrix([simulation])[0]
    reference = _matrix(real_records)
    scale = np.asarray([scales[name] for name in FEATURE_NAMES], dtype=np.float64)
    weight = np.asarray(
        [SPHERE_WEIGHTS[name] for name in FEATURE_NAMES], dtype=np.float64
    )
    squared = np.sum(np.square((reference - query) / scale) * weight, axis=1)
    return int(np.argmin(squared))


def _summaries(membership: np.ndarray) -> dict[str, float]:
    if membership.ndim != 2 or membership.shape[1] != len(CLASS_ORDER):
        raise ValueError("Membership must be n x 3")
    result = {
        class_name: float(np.mean(membership[:, index]))
        for index, class_name in enumerate(CLASS_ORDER)
    }
    result["any_class"] = float(np.mean(np.any(membership, axis=1)))
    result["no_class"] = float(np.mean(~np.any(membership, axis=1)))
    for left in range(len(CLASS_ORDER)):
        for right in range(left + 1, len(CLASS_ORDER)):
            key = f"{CLASS_ORDER[left]}&{CLASS_ORDER[right]}"
            result[key] = float(np.mean(membership[:, left] & membership[:, right]))
    result["all_three"] = float(np.mean(np.all(membership, axis=1)))
    return result


def _group_indices(values: list[str]) -> list[np.ndarray]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, value in enumerate(values):
        groups[value].append(index)
    return [np.asarray(groups[key], dtype=np.int64) for key in sorted(groups)]


def _bootstrap_intervals(
    *,
    simulation_distances: np.ndarray,
    calibration_distances: dict[str, np.ndarray],
    calibration_groups: dict[str, list[str]],
    simulation_groups: list[str],
    repeats: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    if repeats <= 0:
        raise ValueError("bootstrap repeats must be positive")
    rng = np.random.default_rng(seed)
    simulation_group_indices = _group_indices(simulation_groups)
    calibration_group_indices = {
        name: _group_indices(calibration_groups[name]) for name in CLASS_ORDER
    }
    keys = (
        *CLASS_ORDER,
        "any_class",
        "no_class",
        "2um&4um",
        "2um&10um",
        "4um&10um",
        "all_three",
    )
    draws = {key: np.empty(repeats, dtype=np.float64) for key in keys}
    for repeat in range(repeats):
        radii = []
        for class_name in CLASS_ORDER:
            groups = calibration_group_indices[class_name]
            chosen = rng.integers(0, len(groups), size=len(groups))
            sampled = np.concatenate([groups[index] for index in chosen])
            radii.append(
                float(np.quantile(calibration_distances[class_name][sampled], 0.95))
            )
        chosen_simulation = rng.integers(
            0, len(simulation_group_indices), size=len(simulation_group_indices)
        )
        sampled_simulation = np.concatenate(
            [simulation_group_indices[index] for index in chosen_simulation]
        )
        membership = (
            simulation_distances[sampled_simulation] <= np.asarray(radii)
        )
        summary = _summaries(membership)
        for key in keys:
            draws[key][repeat] = summary[key]
    return {
        key: {
            "low": float(np.quantile(values, 0.025)),
            "high": float(np.quantile(values, 0.975)),
        }
        for key, values in draws.items()
    }


def _population_model(
    real_root: Path,
    simulation_train: list[SignalRecord],
    *,
    bootstrap_repeats: int,
    bootstrap_seed: int,
    partition_seed: int,
    select_cases: bool,
) -> dict[str, Any]:
    real_train = load_real_events(real_root, split="train")
    real_validation = load_real_events(real_root, split="val")
    fit, calibration, partition = _group_partition(
        real_train,
        calibration_fraction=CALIBRATION_FRACTION,
        seed=partition_seed,
    )
    pooled_fit = [row for name in CLASS_ORDER for row in fit[name]]
    scales = _robust_scale(pooled_fit)
    centers: dict[str, dict[str, float]] = {}
    radii: dict[str, float] = {}
    calibration_distance: dict[str, np.ndarray] = {}
    validation_acceptance: dict[str, float] = {}
    simulation_distance = np.empty(
        (len(simulation_train), len(CLASS_ORDER)), dtype=np.float64
    )
    simulation_contributions: dict[str, np.ndarray] = {}

    for class_index, class_name in enumerate(CLASS_ORDER):
        values = _matrix(fit[class_name])
        centers[class_name] = {
            name: float(np.median(values[:, index]))
            for index, name in enumerate(FEATURE_NAMES)
        }
        calibration_distance[class_name], _ = _sphere_distance(
            calibration[class_name],
            center=centers[class_name],
            scales=scales,
        )
        radii[class_name] = float(
            np.quantile(calibration_distance[class_name], 0.95)
        )
        validation_distance, _ = _sphere_distance(
            real_validation[class_name],
            center=centers[class_name],
            scales=scales,
        )
        validation_acceptance[class_name] = float(
            np.mean(validation_distance <= radii[class_name])
        )
        distances, contributions = _sphere_distance(
            simulation_train,
            center=centers[class_name],
            scales=scales,
        )
        simulation_distance[:, class_index] = distances
        simulation_contributions[class_name] = contributions

    radius_vector = np.asarray([radii[name] for name in CLASS_ORDER])
    membership = simulation_distance <= radius_vector
    intervals = _bootstrap_intervals(
        simulation_distances=simulation_distance,
        calibration_distances=calibration_distance,
        calibration_groups={
            name: [str(row.metadata["source_group"]) for row in calibration[name]]
            for name in CLASS_ORDER
        },
        simulation_groups=[
            str(row.metadata["latent_id"]) for row in simulation_train
        ],
        repeats=bootstrap_repeats,
        seed=bootstrap_seed,
    )

    cases: dict[str, list[dict[str, Any]]] = {}
    for class_index, class_name in enumerate(CLASS_ORDER):
        distances = simulation_distance[:, class_index]
        radius = radii[class_name]
        inside = np.flatnonzero(distances <= radius)
        outside = np.flatnonzero(distances > radius)
        if not inside.size or not outside.size:
            if select_cases:
                raise ValueError(
                    f"Class {class_name} lacks inside/outside simulations"
                )
            cases[class_name] = []
            continue
        ratio = distances / radius
        representative_target = float(np.median(ratio[inside]))
        selected = (
            (
                "inside",
                int(
                    inside[
                        np.argmin(np.abs(ratio[inside] - representative_target))
                    ]
                ),
            ),
            ("boundary", int(inside[np.argmin(1.0 - ratio[inside])])),
            ("outside", int(outside[np.argmin(ratio[outside] - 1.0)])),
        )
        if len({index for _role, index in selected}) != 3:
            raise ValueError(f"Class {class_name} checkpoint cases are not distinct")
        class_cases: list[dict[str, Any]] = []
        for role, simulation_index in selected:
            real_index = _nearest_real(
                simulation_train[simulation_index],
                fit[class_name],
                scales=scales,
            )
            contribution = simulation_contributions[class_name][simulation_index]
            class_cases.append(
                {
                    "role": role,
                    "class_name": class_name,
                    "simulation_index": simulation_index,
                    "simulation_id": simulation_train[simulation_index].identifier,
                    "real_index": real_index,
                    "real_id": fit[class_name][real_index].identifier,
                    "distance": float(distances[simulation_index]),
                    "radius": radius,
                    "radius_ratio": float(ratio[simulation_index]),
                    "compatible": bool(distances[simulation_index] <= radius),
                    "feature_contributions": {
                        name: float(contribution[index])
                        for index, name in enumerate(FEATURE_NAMES)
                    },
                }
            )
        cases[class_name] = class_cases

    return {
        "centers": centers,
        "scales": scales,
        "radii": radii,
        "partition": partition,
        "real_counts": {
            "train": {name: len(real_train[name]) for name in CLASS_ORDER},
            "fit": {name: len(fit[name]) for name in CLASS_ORDER},
            "calibration": {
                name: len(calibration[name]) for name in CLASS_ORDER
            },
            "validation": {
                name: len(real_validation[name]) for name in CLASS_ORDER
            },
        },
        "real_validation_acceptance": validation_acceptance,
        "simulation_train": {
            "n": len(simulation_train),
            "rates": _summaries(membership),
            "intervals": intervals,
        },
        "cases": cases,
        "_real_fit": fit,
        "_simulation_train": simulation_train,
    }


def build_representativity_model(
    *,
    primary_real_root: Path,
    sensitivity_real_root: Path,
    simulation_root: Path,
    simulation_dataset_id: str = "yeast-passage-simulations@v1",
    bootstrap_repeats: int = 2000,
    bootstrap_seed: int = 20260719,
) -> dict[str, Any]:
    if simulation_dataset_id != "yeast-passage-simulations@v1":
        raise ValueError("Bead representativity is scoped to simulation v1")
    if not np.isclose(sum(SPHERE_WEIGHTS.values()), 1.0):
        raise ValueError("Sphere weights must sum to one")
    simulation_train = load_simulations(
        simulation_root,
        split="train",
        component_count=1,
    )
    primary = _population_model(
        primary_real_root,
        simulation_train,
        bootstrap_repeats=bootstrap_repeats,
        bootstrap_seed=bootstrap_seed,
        partition_seed=CALIBRATION_SEED,
        select_cases=True,
    )
    sensitivity = _population_model(
        sensitivity_real_root,
        simulation_train,
        bootstrap_repeats=bootstrap_repeats,
        bootstrap_seed=bootstrap_seed + 1,
        partition_seed=CALIBRATION_SEED + 1,
        select_cases=False,
    )
    return {
        "schema_version": 1,
        "method": "real-class-robust-physical-sphere-v1",
        "question": (
            "Quelle fraction des simulations v1 train mono-composante tombe "
            "dans chaque sphère de représentativité définie par le réel F ?"
        ),
        "simulation_dataset": simulation_dataset_id,
        "feature_contract": {
            "features": list(FEATURE_NAMES),
            "weights": SPHERE_WEIGHTS,
            "scaling": "pooled real-fit median/IQR; count IQR floor=1",
            "center": "per-class component-wise median on group-disjoint real fit",
            "radius": "per-class q95 on group-disjoint real calibration",
            "memberships": "independent and overlapping",
            "amplitude_policy": (
                "within-signal envelope_peak/RMS and "
                "support_energy/total_energy only"
            ),
            "historical_link": (
                "duration and dominant frequency receive the largest weights, "
                "following the approved favorable-pair search"
            ),
        },
        "bootstrap": {
            "repeats": bootstrap_repeats,
            "seed": bootstrap_seed,
            "real_cluster": "source_group calibration-radius resampling",
            "simulation_cluster": "latent_id",
            "interval": 0.95,
        },
        "simulation_counts": {
            "train_single_component": len(simulation_train),
        },
        "primary": primary,
        "sensitivity": sensitivity,
        "sealed_splits_used": [],
        "claim_boundary": (
            "Une inclusion signifie uniquement compatibilité avec une sphère "
            "empirique de descripteurs réels. Elle ne prouve ni le diamètre "
            "physique, ni l'identité des formes d'onde, ni le réalisme du "
            "générateur."
        ),
    }


def serializable_model(model: dict[str, Any]) -> dict[str, Any]:
    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: clean(item)
                for key, item in value.items()
                if not key.startswith("_")
            }
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, tuple):
            return [clean(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    return clean(model)


def _normalized(signal: np.ndarray) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float64)
    values -= float(np.mean(values))
    rms = float(np.sqrt(np.mean(np.square(values))))
    return values / max(rms, 1.0e-12)


def render_sphere_checkpoint(
    model: dict[str, Any],
    *,
    class_name: str,
    destination: Path,
) -> None:
    if class_name not in CLASS_ORDER:
        raise ValueError(f"Unknown class: {class_name}")
    population = model["primary"]
    real_rows = population["_real_fit"][class_name]
    simulations = population["_simulation_train"]
    cases = population["cases"][class_name]
    color = CLASS_COLORS[class_name]
    figure = Figure(figsize=(15.5, 10.2))
    figure.subplots_adjust(
        left=0.105,
        right=0.985,
        bottom=0.075,
        top=0.86,
        wspace=0.47,
        hspace=0.42,
    )
    grid = figure.add_gridspec(3, 3, height_ratios=(1.25, 0.9, 1.0))
    role_labels = {
        "inside": "Intérieur représentatif",
        "boundary": "À la frontière",
        "outside": "Juste hors sphère",
    }
    for column, case in enumerate(cases):
        simulation = simulations[case["simulation_index"]]
        real = real_rows[case["real_index"]]
        sim_signal = _normalized(simulation.signal)
        real_signal = _normalized(real.signal)
        time_ms = np.arange(sim_signal.size) / FS * 1000.0

        axis = figure.add_subplot(grid[0, column])
        axis.plot(
            time_ms,
            real_signal + 3.0,
            color=color,
            linewidth=0.75,
            label="réel le plus proche +3 RMS",
        )
        axis.plot(
            time_ms,
            sim_signal - 3.0,
            color="#5f6368",
            linewidth=0.75,
            label="simulation v1 −3 RMS",
        )
        axis.axhline(3.0, color=color, linewidth=0.4, alpha=0.5)
        axis.axhline(-3.0, color="#5f6368", linewidth=0.4, alpha=0.5)
        for bound in (
            float(real.metadata["event_start_index"]),
            float(real.metadata["event_end_index"]),
        ):
            axis.axvline(
                bound / FS * 1000.0,
                color=color,
                linestyle=":",
                linewidth=0.9,
            )
        axis.set_title(
            f"{role_labels[case['role']]}\n"
            f"d/r={case['radius_ratio']:.3f} "
            f"(d={case['distance']:.3f}, r={case['radius']:.3f})",
            fontsize=11,
            fontweight="bold",
        )
        axis.set_xlabel("Temps (ms)")
        axis.set_ylabel("Amplitude / RMS + décalage")
        axis.grid(alpha=0.15)
        axis.legend(fontsize=8, loc="upper right")

        spectrum_axis = figure.add_subplot(grid[1, column])
        frequencies = np.fft.rfftfreq(sim_signal.size, d=1.0 / FS) / 1000.0
        mask = (frequencies >= 5.0) & (frequencies <= 100.0)
        for signal, label, line_color in (
            (real_signal, "réel", color),
            (sim_signal, "simulation v1", "#5f6368"),
        ):
            power = np.square(
                np.abs(np.fft.rfft(signal * np.hanning(signal.size)))
            )
            db = 10.0 * np.log10(np.maximum(power, 1.0e-18))
            db -= float(np.max(db[mask]))
            spectrum_axis.plot(
                frequencies[mask],
                db[mask],
                color=line_color,
                linewidth=0.8,
                label=label,
            )
        spectrum_axis.set_ylim(-55.0, 3.0)
        spectrum_axis.set_xlabel("Fréquence (kHz)")
        spectrum_axis.set_ylabel("Puissance relative (dB)")
        spectrum_axis.grid(alpha=0.15)

        contribution_axis = figure.add_subplot(grid[2, column])
        values = np.asarray(
            [case["feature_contributions"][name] for name in FEATURE_NAMES]
        )
        contribution_axis.barh(
            np.arange(len(FEATURE_NAMES)),
            values,
            color=color if case["compatible"] else "#b24a3b",
            alpha=0.82,
        )
        contribution_axis.set_yticks(
            np.arange(len(FEATURE_NAMES)),
            [FEATURE_LABELS[name] for name in FEATURE_NAMES],
            fontsize=8,
        )
        contribution_axis.invert_yaxis()
        contribution_axis.set_xlabel("Contribution à la distance au centre²")
        contribution_axis.grid(axis="x", alpha=0.15)
    figure.suptitle(
        (
            f"Sphère de représentativité v1 · classe réelle "
            f"{CLASS_LABELS[class_name]}\n"
            "Centre réel robuste · rayon q95 sur calibration group-disjointe"
        ),
        fontsize=15,
        fontweight="bold",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160, facecolor="#fffdf8")
