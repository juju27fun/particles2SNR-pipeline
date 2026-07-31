from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from matplotlib.figure import Figure

from .ssl_realism_audit import FS, SignalRecord, signal_descriptors


CLASS_ORDER = ("2um", "4um", "10um")
CLASS_LABELS = {"2um": "2 µm", "4um": "4 µm", "10um": "10 µm"}
CLASS_COLORS = {"2um": "#2171b5", "4um": "#238b45", "10um": "#d94801"}
FEATURE_NAMES = (
    "duration_25_ms",
    "dominant_frequency_khz",
    "spectral_bandwidth_khz",
    "envelope_concentration",
    "temporal_peak_count",
    "spectral_peak_count",
    "envelope_peak_over_rms",
    "event_energy_fraction",
)
FEATURE_WEIGHTS = {
    "duration_25_ms": 0.20,
    "dominant_frequency_khz": 0.20,
    "spectral_bandwidth_khz": 0.15,
    "envelope_concentration": 0.10,
    "temporal_peak_count": 0.075,
    "spectral_peak_count": 0.075,
    "envelope_peak_over_rms": 0.10,
    "event_energy_fraction": 0.10,
}
FEATURE_LABELS = {
    "duration_25_ms": "Durée 25 %",
    "dominant_frequency_khz": "Fréq. dominante",
    "spectral_bandwidth_khz": "Bande spectrale",
    "envelope_concentration": "Concentr. enveloppe",
    "temporal_peak_count": "Pics temporels",
    "spectral_peak_count": "Pics spectraux",
    "envelope_peak_over_rms": "Pic env. / RMS",
    "event_energy_fraction": "Énergie support / crop",
}


@dataclass(frozen=True)
class DistanceResult:
    nearest_indices: np.ndarray
    distances: np.ndarray
    contributions: np.ndarray


def relative_descriptors(signal: np.ndarray) -> dict[str, float]:
    base = signal_descriptors(signal)
    values = np.asarray(signal, dtype=np.float64)
    centered = values - float(np.mean(values))
    left = max(0, int(base["support_start_index"]))
    right = min(values.size, int(base["support_end_index"]))
    total_energy = float(np.sum(np.square(centered)))
    support_energy = float(np.sum(np.square(centered[left:right])))
    if total_energy <= 1.0e-18:
        raise ValueError("Signal energy is zero")
    return {
        **{name: float(base[name]) for name in FEATURE_NAMES if name in base},
        "rms": float(base["rms"]),
        "envelope_peak": float(base["envelope_peak"]),
        "envelope_peak_over_rms": float(base["envelope_peak"] / base["rms"]),
        "event_energy_fraction": support_energy / total_energy,
        "support_start_index": float(left),
        "support_end_index": float(right),
    }


def load_real_events(root: Path, *, split: str) -> dict[str, list[SignalRecord]]:
    signals = np.load(root / "signals.npy", mmap_mode="r", allow_pickle=False)
    populations = {name: [] for name in CLASS_ORDER}
    with (root / "events.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != split:
                continue
            class_name = row["class_name"]
            if class_name not in populations:
                continue
            signal_row = int(row["signal_row"])
            signal = np.asarray(signals[signal_row], dtype=np.float32)
            populations[class_name].append(
                SignalRecord(
                    identifier=row["event_id"],
                    signal=signal,
                    metadata={
                        **row,
                        "signal_row": signal_row,
                        "event_start_index": float(row["event_start_index"]),
                        "event_end_index": float(row["event_end_index"]),
                    },
                    descriptors=relative_descriptors(signal),
                )
            )
    if any(not populations[name] for name in CLASS_ORDER):
        raise ValueError(f"Missing real class in split={split}")
    return populations


def load_simulations(
    root: Path,
    *,
    split: str,
    component_count: int = 1,
) -> list[SignalRecord]:
    signals = np.load(root / "signals.npy", mmap_mode="r", allow_pickle=False)
    records: list[SignalRecord] = []
    with (root / "simulation_metadata.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        for row in csv.DictReader(handle):
            if row["split"] != split or int(row["component_count"]) != component_count:
                continue
            signal_row = int(row["signal_row"])
            signal = np.asarray(signals[signal_row], dtype=np.float32)
            records.append(
                SignalRecord(
                    identifier=f"{row['latent_id']}:view-{row['view_index']}",
                    signal=signal,
                    metadata={**row, "signal_row": signal_row},
                    descriptors=relative_descriptors(signal),
                )
            )
    if not records:
        raise ValueError(f"No simulations for split={split}, components={component_count}")
    return records


def robust_feature_contract(
    records: Iterable[SignalRecord],
) -> tuple[dict[str, float], dict[str, float]]:
    rows = list(records)
    if not rows:
        raise ValueError("Feature reference population is empty")
    centers: dict[str, float] = {}
    scales: dict[str, float] = {}
    for name in FEATURE_NAMES:
        values = np.asarray([row.descriptors[name] for row in rows], dtype=np.float64)
        centers[name] = float(np.median(values))
        scale = float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
        if name in {"temporal_peak_count", "spectral_peak_count"}:
            scale = max(scale, 1.0)
        scales[name] = max(scale, 1.0e-6)
    return centers, scales


def nearest_distances(
    queries: list[SignalRecord],
    reference: list[SignalRecord],
    *,
    scales: dict[str, float],
    exclude_self: bool = False,
    chunk_size: int = 256,
) -> DistanceResult:
    if not queries or not reference:
        raise ValueError("Queries and reference must be non-empty")
    if exclude_self and len(queries) != len(reference):
        raise ValueError("LOO distance requires equal query and reference populations")
    scale = np.asarray([scales[name] for name in FEATURE_NAMES], dtype=np.float64)
    weight = np.asarray([FEATURE_WEIGHTS[name] for name in FEATURE_NAMES])

    def matrix(rows: list[SignalRecord]) -> np.ndarray:
        return np.asarray(
            [[row.descriptors[name] for name in FEATURE_NAMES] for row in rows],
            dtype=np.float64,
        )

    query_matrix = matrix(queries)
    reference_matrix = matrix(reference)
    nearest = np.empty(len(queries), dtype=np.int64)
    distances = np.empty(len(queries), dtype=np.float64)
    contributions = np.empty((len(queries), len(FEATURE_NAMES)), dtype=np.float64)
    for start in range(0, len(queries), chunk_size):
        stop = min(start + chunk_size, len(queries))
        delta = (
            query_matrix[start:stop, None, :] - reference_matrix[None, :, :]
        ) / scale
        weighted = np.square(delta) * weight
        squared = np.sum(weighted, axis=2)
        if exclude_self:
            local = np.arange(start, stop)
            squared[np.arange(stop - start), local] = np.inf
        indices = np.argmin(squared, axis=1)
        nearest[start:stop] = indices
        distances[start:stop] = np.sqrt(
            squared[np.arange(stop - start), indices]
        )
        contributions[start:stop] = weighted[
            np.arange(stop - start), indices, :
        ]
    return DistanceResult(nearest, distances, contributions)


def _group_indices(values: list[str]) -> list[np.ndarray]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, value in enumerate(values):
        groups[value].append(index)
    return [np.asarray(groups[key], dtype=np.int64) for key in sorted(groups)]


def _summaries_from_membership(membership: np.ndarray) -> dict[str, float]:
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


def clustered_bootstrap(
    *,
    simulation_distances: np.ndarray,
    real_loo_distances: dict[str, np.ndarray],
    real_groups: dict[str, list[str]],
    simulation_groups: list[str],
    repeats: int = 2000,
    seed: int = 20260719,
) -> dict[str, dict[str, float]]:
    if simulation_distances.shape[1] != len(CLASS_ORDER):
        raise ValueError("Simulation distances must have one column per class")
    rng = np.random.default_rng(seed)
    sim_group_indices = _group_indices(simulation_groups)
    real_group_indices = {
        name: _group_indices(real_groups[name]) for name in CLASS_ORDER
    }
    keys = (*CLASS_ORDER, "any_class", "no_class", "2um&4um", "2um&10um", "4um&10um", "all_three")
    draws = {key: np.empty(repeats, dtype=np.float64) for key in keys}
    for repeat in range(repeats):
        thresholds = []
        for class_name in CLASS_ORDER:
            groups = real_group_indices[class_name]
            chosen = rng.integers(0, len(groups), size=len(groups))
            sampled = np.concatenate([groups[index] for index in chosen])
            thresholds.append(
                float(np.quantile(real_loo_distances[class_name][sampled], 0.95))
            )
        chosen_sim = rng.integers(
            0, len(sim_group_indices), size=len(sim_group_indices)
        )
        sampled_sim = np.concatenate([sim_group_indices[index] for index in chosen_sim])
        membership = simulation_distances[sampled_sim] <= np.asarray(thresholds)
        summary = _summaries_from_membership(membership)
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
    simulation_validation: list[SignalRecord],
    *,
    bootstrap_repeats: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    real_train = load_real_events(real_root, split="train")
    real_validation = load_real_events(real_root, split="val")
    pooled = [row for name in CLASS_ORDER for row in real_train[name]]
    centers, scales = robust_feature_contract(pooled)
    thresholds: dict[str, float] = {}
    loo: dict[str, DistanceResult] = {}
    train_distances = np.empty((len(simulation_train), len(CLASS_ORDER)))
    validation_distances = np.empty(
        (len(simulation_validation), len(CLASS_ORDER))
    )
    train_nearest = np.empty_like(train_distances, dtype=np.int64)
    train_contributions: dict[str, np.ndarray] = {}
    validation_recall: dict[str, float] = {}
    for class_index, class_name in enumerate(CLASS_ORDER):
        loo[class_name] = nearest_distances(
            real_train[class_name],
            real_train[class_name],
            scales=scales,
            exclude_self=True,
        )
        thresholds[class_name] = float(
            np.quantile(loo[class_name].distances, 0.95)
        )
        train_result = nearest_distances(
            simulation_train, real_train[class_name], scales=scales
        )
        validation_result = nearest_distances(
            simulation_validation, real_train[class_name], scales=scales
        )
        train_distances[:, class_index] = train_result.distances
        validation_distances[:, class_index] = validation_result.distances
        train_nearest[:, class_index] = train_result.nearest_indices
        train_contributions[class_name] = train_result.contributions
        real_validation_result = nearest_distances(
            real_validation[class_name], real_train[class_name], scales=scales
        )
        validation_recall[class_name] = float(
            np.mean(real_validation_result.distances <= thresholds[class_name])
        )

    threshold_vector = np.asarray([thresholds[name] for name in CLASS_ORDER])
    train_membership = train_distances <= threshold_vector
    validation_membership = validation_distances <= threshold_vector
    intervals = clustered_bootstrap(
        simulation_distances=train_distances,
        real_loo_distances={
            name: loo[name].distances for name in CLASS_ORDER
        },
        real_groups={
            name: [row.metadata["source_group"] for row in real_train[name]]
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
        distances = train_distances[:, class_index]
        threshold = thresholds[class_name]
        inside = np.flatnonzero(distances <= threshold)
        outside = np.flatnonzero(distances > threshold)
        if not inside.size or not outside.size:
            raise ValueError(f"Class {class_name} lacks inside/outside cases")
        representative_target = float(np.median(distances[inside]))
        selected = (
            ("inside", int(inside[np.argmin(np.abs(distances[inside] - representative_target))])),
            ("boundary", int(inside[np.argmin(threshold - distances[inside])])),
            ("outside", int(outside[np.argmin(distances[outside] - threshold)])),
        )
        if len({index for _role, index in selected}) != len(selected):
            raise ValueError(f"Class {class_name} checkpoint cases are not distinct")
        class_cases = []
        for role, simulation_index in selected:
            real_index = int(train_nearest[simulation_index, class_index])
            contribution = train_contributions[class_name][simulation_index]
            class_cases.append(
                {
                    "role": role,
                    "class_name": class_name,
                    "simulation_index": simulation_index,
                    "simulation_id": simulation_train[simulation_index].identifier,
                    "real_index": real_index,
                    "real_id": real_train[class_name][real_index].identifier,
                    "distance": float(distances[simulation_index]),
                    "threshold": threshold,
                    "compatible": bool(distances[simulation_index] <= threshold),
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
        "thresholds": thresholds,
        "real_counts": {
            split: {
                name: len(population[name])
                for name in CLASS_ORDER
            }
            for split, population in (
                ("train", real_train),
                ("val", real_validation),
            )
        },
        "real_validation_acceptance": validation_recall,
        "simulation_train": {
            "n": len(simulation_train),
            "rates": _summaries_from_membership(train_membership),
            "intervals": intervals,
        },
        "simulation_validation": {
            "n": len(simulation_validation),
            "rates": _summaries_from_membership(validation_membership),
        },
        "cases": cases,
        "_real_train": real_train,
        "_simulation_train": simulation_train,
    }


def build_class_coverage_model(
    *,
    primary_real_root: Path,
    sensitivity_real_root: Path,
    simulation_root: Path,
    simulation_dataset_id: str = "yeast-passage-simulations@v2",
    bootstrap_repeats: int = 2000,
    bootstrap_seed: int = 20260719,
) -> dict[str, Any]:
    simulation_train = load_simulations(simulation_root, split="train")
    simulation_validation = load_simulations(simulation_root, split="validation")
    primary = _population_model(
        primary_real_root,
        simulation_train,
        simulation_validation,
        bootstrap_repeats=bootstrap_repeats,
        bootstrap_seed=bootstrap_seed,
    )
    sensitivity = _population_model(
        sensitivity_real_root,
        simulation_train,
        simulation_validation,
        bootstrap_repeats=bootstrap_repeats,
        bootstrap_seed=bootstrap_seed + 1,
    )
    return {
        "schema_version": 1,
        "method": "simulation-to-real-class-independent-nearest-neighbor-v2",
        "question": (
            f"Quelle proportion des simulations {simulation_dataset_id} "
            "mono-composante est compatible avec chaque classe réelle F dans "
            "l’espace figé des descripteurs ?"
        ),
        "simulation_dataset": simulation_dataset_id,
        "feature_contract": {
            "features": list(FEATURE_NAMES),
            "weights": FEATURE_WEIGHTS,
            "scaling": "median and IQR fitted on pooled primary real train; count IQR floor=1",
            "threshold": "per-class q95 of leave-one-out real-train nearest-neighbour distances",
            "memberships": "independent and overlapping",
            "amplitude_policy": "within-signal envelope_peak/RMS and support_energy/total_energy only",
        },
        "bootstrap": {
            "repeats": bootstrap_repeats,
            "seed": bootstrap_seed,
            "real_cluster": "source_group",
            "simulation_cluster": "latent_id",
            "interval": 0.95,
        },
        "simulation_counts": {
            "train_single_component": len(simulation_train),
            "validation_single_component": len(simulation_validation),
        },
        "primary": primary,
        "sensitivity": sensitivity,
        "claim_boundary": (
            "La compatibilité est une inclusion dans une région empirique de "
            "descripteurs relatifs. Elle ne prouve ni le diamètre physique, ni "
            "l’identité de forme d’onde, ni le réalisme biologique."
        ),
    }


def serializable_model(model: dict[str, Any]) -> dict[str, Any]:
    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items() if not key.startswith("_")}
        if isinstance(value, tuple):
            return [clean(item) for item in value]
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    return clean(model)


def _normalized(signal: np.ndarray) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float64)
    values -= float(np.mean(values))
    return values / max(float(np.sqrt(np.mean(np.square(values)))), 1.0e-12)


def render_class_checkpoint(
    model: dict[str, Any],
    *,
    class_name: str,
    destination: Path,
) -> None:
    if class_name not in CLASS_ORDER:
        raise ValueError(f"Unknown class: {class_name}")
    population = model["primary"]
    real_rows = population["_real_train"][class_name]
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
        "inside": "Dans la région",
        "boundary": "À la frontière",
        "outside": "Juste hors région",
    }
    for column, case in enumerate(cases):
        simulation = simulations[case["simulation_index"]]
        real = real_rows[case["real_index"]]
        sim_signal = _normalized(simulation.signal)
        real_signal = _normalized(real.signal)
        time_ms = np.arange(sim_signal.size) / FS * 1000.0

        axis = figure.add_subplot(grid[0, column])
        axis.plot(time_ms, real_signal + 3.0, color=color, linewidth=0.75, label="réel +3 RMS")
        axis.plot(time_ms, sim_signal - 3.0, color="#5f6368", linewidth=0.75, label="simulation −3 RMS")
        axis.axhline(3.0, color=color, linewidth=0.4, alpha=0.5)
        axis.axhline(-3.0, color="#5f6368", linewidth=0.4, alpha=0.5)
        for bound in (
            float(real.metadata["event_start_index"]),
            float(real.metadata["event_end_index"]),
        ):
            axis.axvline(bound / FS * 1000.0, color=color, linestyle=":", linewidth=0.9)
        axis.set_title(
            f"{role_labels[case['role']]}\nd={case['distance']:.3f} · seuil={case['threshold']:.3f}",
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
            (sim_signal, "simulation", "#5f6368"),
        ):
            power = np.square(np.abs(np.fft.rfft(signal * np.hanning(signal.size))))
            db = 10.0 * np.log10(np.maximum(power, 1.0e-18))
            db -= float(np.max(db[mask]))
            spectrum_axis.plot(frequencies[mask], db[mask], color=line_color, linewidth=0.8, label=label)
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
        contribution_axis.set_xlabel("Contribution à la distance²")
        contribution_axis.grid(axis="x", alpha=0.15)
    figure.suptitle(
        (
            f"Checkpoint C2 · {model.get('simulation_dataset', 'simulation')} "
            f"· classe réelle {CLASS_LABELS[class_name]}\n"
            "Trois cas déterministes autour du seuil q95 réel-réel"
        ),
        fontsize=15,
        fontweight="bold",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160, facecolor="#fffdf8")
