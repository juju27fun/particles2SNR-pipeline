from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from .particle_class_coverage import CLASS_ORDER, load_real_events, load_simulations
from .ssl_realism_audit import SignalRecord


CORE_FEATURE_NAMES = (
    "log_aligned_envelope_peak",
    "dominant_frequency_khz",
    "duration_25_ms",
)
CORE_FEATURE_LABELS = {
    "log_aligned_envelope_peak": "Amplitude traitée recalée (log, u.a.)",
    "dominant_frequency_khz": "Fréquence dominante (kHz)",
    "duration_25_ms": "Temps de passage à 25 % (ms)",
}
CORE_WEIGHTS = {name: 1.0 / 3.0 for name in CORE_FEATURE_NAMES}
PARTITION_SEED = 20260719
CV_SEED = 20260719


def _fit_partition(
    populations: dict[str, list[SignalRecord]],
    *,
    seed: int = PARTITION_SEED,
    holdout_fraction: float = 0.20,
) -> tuple[dict[str, list[SignalRecord]], dict[str, Any]]:
    groups = sorted(
        {
            str(row.metadata["source_group"])
            for rows in populations.values()
            for row in rows
        }
    )
    if len(groups) < 5:
        raise ValueError("At least five source groups are required")
    rng = np.random.default_rng(seed)
    shuffled = [groups[index] for index in rng.permutation(len(groups))]
    n_holdout = max(1, int(round(len(groups) * holdout_fraction)))
    holdout = set(shuffled[:n_holdout])
    fit = {
        class_name: [
            row
            for row in populations[class_name]
            if str(row.metadata["source_group"]) not in holdout
        ]
        for class_name in CLASS_ORDER
    }
    if any(len(fit[name]) < 10 for name in CLASS_ORDER):
        raise ValueError("Insufficient group-disjoint fit population")
    return fit, {
        "seed": seed,
        "holdout_fraction": holdout_fraction,
        "fit_source_groups": sorted(set(groups) - holdout),
        "holdout_source_groups": sorted(holdout),
    }


def amplitude_alignment_factor(
    real_fit: list[SignalRecord],
    simulations: list[SignalRecord],
) -> float:
    if not real_fit or not simulations:
        raise ValueError("Amplitude alignment requires non-empty populations")
    real = np.asarray(
        [row.descriptors["envelope_peak"] for row in real_fit],
        dtype=np.float64,
    )
    simulated = np.asarray(
        [row.descriptors["envelope_peak"] for row in simulations],
        dtype=np.float64,
    )
    if (
        not np.all(np.isfinite(real))
        or not np.all(np.isfinite(simulated))
        or np.any(real <= 0.0)
        or np.any(simulated <= 0.0)
    ):
        raise ValueError("Envelope peaks must be finite and strictly positive")
    return float(np.median(real) / np.median(simulated))


def core_feature_matrix(
    records: list[SignalRecord],
    *,
    amplitude_scale: float = 1.0,
) -> np.ndarray:
    if amplitude_scale <= 0.0 or not np.isfinite(amplitude_scale):
        raise ValueError("amplitude_scale must be finite and positive")
    if not records:
        raise ValueError("Core feature population is empty")
    return np.asarray(
        [
            [
                np.log10(amplitude_scale * row.descriptors["envelope_peak"]),
                row.descriptors["dominant_frequency_khz"],
                row.descriptors["duration_25_ms"],
            ]
            for row in records
        ],
        dtype=np.float64,
    )


def _group_bootstrap_interval(
    values: np.ndarray,
    groups: list[str],
    scorer,
    *,
    repeats: int,
    seed: int,
) -> list[float]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        grouped[group].append(index)
    blocks = [
        np.asarray(grouped[key], dtype=np.int64)
        for key in sorted(grouped)
    ]
    rng = np.random.default_rng(seed)
    draws = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        selected = np.concatenate([blocks[index] for index in chosen])
        draws[repeat] = float(scorer(selected))
    return [
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    ]


def grouped_real_discrimination(
    populations: dict[str, list[SignalRecord]],
    *,
    folds: int = 5,
    bootstrap_repeats: int = 1000,
    seed: int = CV_SEED,
) -> dict[str, Any]:
    records = [row for name in CLASS_ORDER for row in populations[name]]
    labels = np.asarray(
        [index for index, name in enumerate(CLASS_ORDER) for _ in populations[name]],
        dtype=np.int64,
    )
    groups = [
        str(row.metadata["source_group"])
        for row in records
    ]
    matrix = core_feature_matrix(records)
    splitter = StratifiedGroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed,
    )
    contracts = {
        CORE_FEATURE_NAMES[index]: [index]
        for index in range(len(CORE_FEATURE_NAMES))
    }
    contracts["core_three"] = list(range(len(CORE_FEATURE_NAMES)))
    result: dict[str, Any] = {}
    for contract_index, (name, columns) in enumerate(contracts.items()):
        predictions = np.full(labels.shape, -1, dtype=np.int64)
        for train, validation in splitter.split(matrix, labels, groups):
            model = make_pipeline(
                RobustScaler(),
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            )
            model.fit(matrix[train][:, columns], labels[train])
            predictions[validation] = model.predict(matrix[validation][:, columns])
        if np.any(predictions < 0):
            raise RuntimeError("Grouped CV left rows without predictions")

        def macro(selected: np.ndarray) -> float:
            return float(
                f1_score(
                    labels[selected],
                    predictions[selected],
                    labels=np.arange(len(CLASS_ORDER)),
                    average="macro",
                    zero_division=0,
                )
            )

        def balanced(selected: np.ndarray) -> float:
            return float(
                balanced_accuracy_score(labels[selected], predictions[selected])
            )

        all_indices = np.arange(labels.size, dtype=np.int64)
        result[name] = {
            "columns": [CORE_FEATURE_NAMES[index] for index in columns],
            "macro_f1": macro(all_indices),
            "macro_f1_ci95": _group_bootstrap_interval(
                predictions,
                groups,
                macro,
                repeats=bootstrap_repeats,
                seed=seed + 10 * contract_index,
            ),
            "balanced_accuracy": balanced(all_indices),
            "balanced_accuracy_ci95": _group_bootstrap_interval(
                predictions,
                groups,
                balanced,
                repeats=bootstrap_repeats,
                seed=seed + 10 * contract_index + 1,
            ),
        }
    return {
        "method": "5-fold StratifiedGroupKFold out-of-fold logistic regression",
        "group": "source_group",
        "seed": seed,
        "folds": folds,
        "bootstrap_repeats": bootstrap_repeats,
        "scores": result,
    }


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "n": int(values.size),
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "p50": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
    }


def build_metric_priority_analysis(
    *,
    real_root,
    simulation_root,
    bootstrap_repeats: int = 1000,
) -> dict[str, Any]:
    real_train = load_real_events(real_root, split="train")
    fit, partition = _fit_partition(real_train)
    simulations = load_simulations(
        simulation_root,
        split="train",
        component_count=1,
    )
    pooled_fit = [row for name in CLASS_ORDER for row in fit[name]]
    scale = amplitude_alignment_factor(pooled_fit, simulations)
    real_distributions = {}
    for class_name in CLASS_ORDER:
        matrix = core_feature_matrix(real_train[class_name])
        real_distributions[class_name] = {
            name: _summary(matrix[:, index])
            for index, name in enumerate(CORE_FEATURE_NAMES)
        }
    sim_raw = core_feature_matrix(simulations)
    sim_aligned = core_feature_matrix(simulations, amplitude_scale=scale)
    return {
        "schema_version": 2,
        "method": "bead-v1-core-three-metric-priority",
        "features": list(CORE_FEATURE_NAMES),
        "weights": CORE_WEIGHTS,
        "amplitude": {
            "name": "amplitude traitée recalée",
            "units": "unités arbitraires du signal traité",
            "descriptor": "smoothed Hilbert-envelope peak",
            "transform": "log10",
            "alignment_factor": scale,
            "alignment_rule": "median(real train-fit pooled) / median(simulation v1 train mono)",
            "physical_calibration_claim": False,
        },
        "partition": partition,
        "counts": {
            "real_train": {
                name: len(real_train[name])
                for name in CLASS_ORDER
            },
            "real_fit": {
                name: len(fit[name])
                for name in CLASS_ORDER
            },
            "simulation_train_mono": len(simulations),
        },
        "real_distributions": real_distributions,
        "simulation_amplitude": {
            "raw_log_envelope_peak": _summary(sim_raw[:, 0]),
            "aligned_log_envelope_peak": _summary(sim_aligned[:, 0]),
        },
        "real_discrimination": grouped_real_discrimination(
            real_train,
            bootstrap_repeats=bootstrap_repeats,
        ),
        "sealed_splits_used": [],
        "claim_boundary": (
            "Le recalage aligne une échelle numérique en unités arbitraires. "
            "Il ne produit ni amplitude physique étalonnée, ni identité de "
            "taille ou de forme d'onde."
        ),
    }
