from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure
from scipy.ndimage import uniform_filter1d
from scipy.signal import hilbert

from .particle_class_coverage import (
    CLASS_ORDER,
    FEATURE_NAMES,
    FEATURE_WEIGHTS,
    load_real_events,
    load_simulations,
    nearest_distances,
    robust_feature_contract,
)
from .ssl_realism_audit import FS, SignalRecord


BLINDING_SEED = 20260719
BOOTSTRAP_SEED = 20260719
EXPECTED_COUNTS = {
    "real_validation_2um": 138,
    "real_validation_2um_source_groups": 101,
    "simulation_v1_train_single": 6982,
    "simulation_v2_train_single": 7020,
    "both_compatible": 47,
    "v1_only": 16,
    "v2_only": 6,
    "neither": 69,
}
CASE_SPECS = (
    ("blind-2um-case-01", "both_compatible"),
    ("blind-2um-case-02", "v1_only"),
    ("blind-2um-case-03", "v2_only"),
)


def blind_assignment(
    case_id: str,
    *,
    seed: int = BLINDING_SEED,
) -> dict[str, str]:
    digest = hashlib.sha256(f"{seed}:{case_id}".encode()).digest()
    if digest[0] % 2:
        return {"A": "v2", "B": "v1"}
    return {"A": "v1", "B": "v2"}


def _cluster_bootstrap(
    delta: np.ndarray,
    groups: list[str],
    *,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    if repeats <= 0:
        raise ValueError("bootstrap repeats must be positive")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        grouped[group].append(index)
    blocks = [
        np.asarray(grouped[key], dtype=np.int64)
        for key in sorted(grouped)
    ]
    rng = np.random.default_rng(seed)
    medians = np.empty(repeats, dtype=np.float64)
    fractions = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        indices = np.concatenate([blocks[index] for index in chosen])
        selected = delta[indices]
        medians[repeat] = float(np.median(selected))
        fractions[repeat] = float(np.mean(selected > 0.0))
    return {
        "cluster": "real source_group",
        "repeats": repeats,
        "seed": seed,
        "median_delta": {
            "estimate": float(np.median(delta)),
            "low": float(np.quantile(medians, 0.025)),
            "high": float(np.quantile(medians, 0.975)),
        },
        "v1_closer_fraction": {
            "estimate": float(np.mean(delta > 0.0)),
            "low": float(np.quantile(fractions, 0.025)),
            "high": float(np.quantile(fractions, 0.975)),
        },
    }


def _median_case_index(
    eligible: np.ndarray,
    delta: np.ndarray,
    anchors: list[SignalRecord],
) -> int:
    indices = np.flatnonzero(eligible)
    if not indices.size:
        raise ValueError("diagnostic category is empty")
    target = float(np.median(delta[indices]))
    return min(
        (int(index) for index in indices),
        key=lambda index: (
            abs(float(delta[index]) - target),
            anchors[index].identifier,
        ),
    )


def _public_record(record: SignalRecord) -> dict[str, Any]:
    return {
        "id": record.identifier,
        "metadata": {
            key: value
            for key, value in record.metadata.items()
            if key not in {"signal_row"}
        },
        "descriptors": {
            name: float(record.descriptors[name])
            for name in (
                *FEATURE_NAMES,
                "support_start_index",
                "support_end_index",
            )
        },
    }


def build_dataset_metric_diagnostic(
    *,
    primary_real_root: Path,
    simulation_v1_root: Path,
    simulation_v2_root: Path,
    class_name: str = "2um",
    bootstrap_repeats: int = 2000,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    pair_by_identifier: bool = False,
    enforce_registered_counts: bool = True,
) -> dict[str, Any]:
    if pair_by_identifier:
        raise ValueError(
            "v1/v2 latent_id rows are not physical counterfactual pairs"
        )
    if class_name != "2um":
        raise ValueError("the first diagnostic checkpoint is frozen to 2um")

    real_train = load_real_events(primary_real_root, split="train")
    real_validation = load_real_events(primary_real_root, split="val")
    pooled_train = [
        record
        for population_name in CLASS_ORDER
        for record in real_train[population_name]
    ]
    centers, scales = robust_feature_contract(pooled_train)
    anchors = real_validation[class_name]
    simulations = {
        "v1": load_simulations(
            simulation_v1_root,
            split="train",
            component_count=1,
        ),
        "v2": load_simulations(
            simulation_v2_root,
            split="train",
            component_count=1,
        ),
    }
    distances = {
        version: nearest_distances(anchors, records, scales=scales)
        for version, records in simulations.items()
    }
    real_loo = nearest_distances(
        real_train[class_name],
        real_train[class_name],
        scales=scales,
        exclude_self=True,
    )
    threshold = float(np.quantile(real_loo.distances, 0.95))
    d1 = distances["v1"].distances
    d2 = distances["v2"].distances
    delta = d2 - d1
    category_masks = {
        "both_compatible": (d1 <= threshold) & (d2 <= threshold),
        "v1_only": (d1 <= threshold) & (d2 > threshold),
        "v2_only": (d1 > threshold) & (d2 <= threshold),
        "neither": (d1 > threshold) & (d2 > threshold),
    }
    category_counts = {
        name: int(np.sum(mask))
        for name, mask in category_masks.items()
    }
    observed_counts = {
        "real_validation_2um": len(anchors),
        "real_validation_2um_source_groups": len(
            {str(record.metadata["source_group"]) for record in anchors}
        ),
        "simulation_v1_train_single": len(simulations["v1"]),
        "simulation_v2_train_single": len(simulations["v2"]),
        **category_counts,
    }
    if enforce_registered_counts and observed_counts != EXPECTED_COUNTS:
        raise ValueError(
            "registered diagnostic population changed: "
            f"{observed_counts} != {EXPECTED_COUNTS}"
        )

    cases = []
    for case_id, category in CASE_SPECS:
        anchor_index = _median_case_index(
            category_masks[category],
            delta,
            anchors,
        )
        v1_index = int(distances["v1"].nearest_indices[anchor_index])
        v2_index = int(distances["v2"].nearest_indices[anchor_index])
        cases.append(
            {
                "case_id": case_id,
                "category": category,
                "anchor_index": anchor_index,
                "real": _public_record(anchors[anchor_index]),
                "v1": _public_record(simulations["v1"][v1_index]),
                "v2": _public_record(simulations["v2"][v2_index]),
                "distance_v1": float(d1[anchor_index]),
                "distance_v2": float(d2[anchor_index]),
                "delta_v2_minus_v1": float(delta[anchor_index]),
                "threshold": threshold,
                "_real_signal": anchors[anchor_index].signal,
                "_v1_signal": simulations["v1"][v1_index].signal,
                "_v2_signal": simulations["v2"][v2_index].signal,
            }
        )

    return {
        "schema_version": 1,
        "method": "real-anchor-to-simulation-pool-nearest-neighbour-v1-v2",
        "class_name": class_name,
        "datasets": {
            "real_primary": "particles2snr-f-c1-descriptor-events-4class@v1",
            "simulation_v1": "yeast-passage-simulations@v1",
            "simulation_v2": "yeast-passage-simulations@v2",
        },
        "split_contract": {
            "feature_fit": "real train",
            "anchor_evaluation": "real val",
            "simulation_search": "simulation train mono-component",
            "sealed_splits_used": [],
        },
        "feature_contract": {
            "features": list(FEATURE_NAMES),
            "weights": FEATURE_WEIGHTS,
            "centers": centers,
            "scales": scales,
            "threshold": threshold,
            "threshold_rule": (
                "q95 leave-one-out nearest-neighbour distance within real "
                "2um train"
            ),
        },
        "population_counts": observed_counts,
        "aggregate": {
            "delta_definition": "distance_v2 - distance_v1",
            "v1_closer_count": int(np.sum(delta > 0.0)),
            "v2_closer_count": int(np.sum(delta < 0.0)),
            "tie_count": int(np.sum(delta == 0.0)),
            "bootstrap": _cluster_bootstrap(
                delta,
                [
                    str(record.metadata["source_group"])
                    for record in anchors
                ],
                repeats=bootstrap_repeats,
                seed=bootstrap_seed,
            ),
        },
        "case_selection": {
            "rule": (
                "median delta within both-compatible, v1-only, and v2-only "
                "categories; deterministic event-id tie-break"
            ),
            "selected_before_rendering": True,
            "case_ids": [case["case_id"] for case in cases],
        },
        "cases": cases,
        "claim_boundary": (
            "This diagnostic tests whether the current descriptor distance "
            "selects visually credible neighbours for fixed real anchors. It "
            "does not make v1 active, estimate simulation-to-class coverage, "
            "or validate particle diameter identity."
        ),
    }


def serializable_diagnostic(model: dict[str, Any]) -> dict[str, Any]:
    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: clean(item)
                for key, item in value.items()
                if not key.startswith("_")
            }
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    return clean(model)


def _normalized(signal: np.ndarray) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float64)
    values = values - float(np.mean(values))
    rms = float(np.sqrt(np.mean(np.square(values))))
    return values / max(rms, 1.0e-12)


def _envelope_centered(
    signal: np.ndarray,
    descriptors: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    normalized = _normalized(signal)
    envelope = uniform_filter1d(
        np.abs(hilbert(normalized)),
        size=64,
        mode="nearest",
    )
    midpoint = 0.5 * (
        float(descriptors["support_start_index"])
        + float(descriptors["support_end_index"])
    )
    time_ms = (np.arange(normalized.size) - midpoint) / FS * 1000.0
    mask = np.abs(time_ms) <= 1.5
    envelope = envelope / max(float(np.max(envelope)), 1.0e-12)
    return time_ms[mask], envelope[mask]


def _spectrum(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normalized = _normalized(signal)
    frequencies = np.fft.rfftfreq(normalized.size, d=1.0 / FS) / 1000.0
    power = np.square(
        np.abs(np.fft.rfft(normalized * np.hanning(normalized.size)))
    )
    mask = (frequencies >= 5.0) & (frequencies <= 100.0)
    db = 10.0 * np.log10(np.maximum(power[mask], 1.0e-18))
    db -= float(np.max(db))
    return frequencies[mask], db


def render_blind_case(
    case: dict[str, Any],
    *,
    assignment: dict[str, str],
    case_number: int,
    destination: Path,
) -> None:
    signals = {
        "real": case["_real_signal"],
        "v1": case["_v1_signal"],
        "v2": case["_v2_signal"],
    }
    records = {
        "real": case["real"],
        "v1": case["v1"],
        "v2": case["v2"],
    }
    colors = {"real": "#17223b", "A": "#6657a5", "B": "#c56b3f"}
    figure = Figure(figsize=(14.8, 9.4))
    figure.subplots_adjust(
        left=0.08,
        right=0.975,
        bottom=0.075,
        top=0.86,
        hspace=0.42,
    )
    grid = figure.add_gridspec(3, 1, height_ratios=(1.2, 0.9, 0.9))

    full_axis = figure.add_subplot(grid[0])
    time_ms = np.arange(4096) / FS * 1000.0
    lanes = (("real", "Réel", 6.0), ("A", "Candidat A", 0.0), ("B", "Candidat B", -6.0))
    for blind_key, label, offset in lanes:
        version = "real" if blind_key == "real" else assignment[blind_key]
        full_axis.plot(
            time_ms,
            _normalized(signals[version]) + offset,
            color=colors[blind_key],
            linewidth=0.72,
            label=label,
        )
        full_axis.axhline(
            offset,
            color=colors[blind_key],
            linewidth=0.4,
            alpha=0.5,
        )
    for bound in (
        float(case["real"]["metadata"]["event_start_index"]),
        float(case["real"]["metadata"]["event_end_index"]),
    ):
        full_axis.axvline(
            bound / FS * 1000.0,
            color=colors["real"],
            linestyle=":",
            linewidth=0.9,
        )
    full_axis.set_title(
        "Signaux prétraités complets · centrage et normalisation RMS",
        loc="left",
        fontsize=11,
        fontweight="bold",
    )
    full_axis.set_xlabel("Temps (ms)")
    full_axis.set_ylabel("Amplitude / RMS + décalage")
    full_axis.grid(alpha=0.15)
    full_axis.legend(loc="upper right", ncols=3, fontsize=9)

    envelope_axis = figure.add_subplot(grid[1])
    for blind_key, label in (("real", "Réel"), ("A", "Candidat A"), ("B", "Candidat B")):
        version = "real" if blind_key == "real" else assignment[blind_key]
        x_values, envelope = _envelope_centered(
            signals[version],
            records[version]["descriptors"],
        )
        envelope_axis.plot(
            x_values,
            envelope,
            color=colors[blind_key],
            linewidth=1.15,
            label=label,
        )
    envelope_axis.set_title(
        "Enveloppes alignées sur le milieu du support · affichage seulement",
        loc="left",
        fontsize=11,
        fontweight="bold",
    )
    envelope_axis.set_xlabel("Temps relatif au centre du support (ms)")
    envelope_axis.set_ylabel("Enveloppe / maximum")
    envelope_axis.set_xlim(-1.5, 1.5)
    envelope_axis.set_ylim(-0.03, 1.08)
    envelope_axis.grid(alpha=0.15)

    spectrum_axis = figure.add_subplot(grid[2])
    for blind_key, label in (("real", "Réel"), ("A", "Candidat A"), ("B", "Candidat B")):
        version = "real" if blind_key == "real" else assignment[blind_key]
        frequencies, db = _spectrum(signals[version])
        spectrum_axis.plot(
            frequencies,
            db,
            color=colors[blind_key],
            linewidth=0.9,
            label=label,
        )
    spectrum_axis.set_title(
        "Spectres de puissance relatifs",
        loc="left",
        fontsize=11,
        fontweight="bold",
    )
    spectrum_axis.set_xlabel("Fréquence (kHz)")
    spectrum_axis.set_ylabel("Puissance relative (dB)")
    spectrum_axis.set_xlim(5.0, 100.0)
    spectrum_axis.set_ylim(-55.0, 3.0)
    spectrum_axis.grid(alpha=0.15)

    figure.suptitle(
        (
            f"Diagnostic aveugle 2 µm · cas {case_number:02d}\n"
            "Quel candidat ressemble le plus au même signal réel ?"
        ),
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.025,
        (
            "Les versions, distances et catégories sont masquées jusqu’à la "
            "finalisation des trois décisions."
        ),
        ha="center",
        fontsize=9,
        color="#586174",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160, facecolor="#fffdf8")
