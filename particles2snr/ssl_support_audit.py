from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from .ssl_realism_audit import (
    FEATURE_NAMES,
    FEATURE_UNITS,
    MATCH_WEIGHTS,
    SignalRecord,
    load_particle_population,
    load_ssl_simulation_population,
)


CLASS_ORDER = ("2um", "4um", "10um")
CLASS_LABELS = {"2um": "2 µm", "4um": "4 µm", "10um": "10 µm"}
CLASS_COLORS = {"2um": "#2c7fb8", "4um": "#31a354", "10um": "#d95f0e"}
HEATMAP_PAIRS = (
    ("duration_25_ms", "dominant_frequency_khz"),
    ("envelope_concentration", "spectral_bandwidth_khz"),
)
DISPLAY_FEATURE_LABELS = {
    "duration_25_ms": "Envelope duration at 25%",
    "envelope_concentration": "Envelope concentration (50% / 25%)",
    "dominant_frequency_khz": "Dominant frequency",
    "spectral_bandwidth_khz": "Spectral bandwidth",
    "temporal_peak_count": "Temporal peak count",
    "spectral_peak_count": "Spectral peak count",
    "amplitude_percentile": "Amplitude percentile",
}


@dataclass(frozen=True)
class SupportDistances:
    nearest_indices: np.ndarray
    distances: np.ndarray


def feature_scales(reference: list[SignalRecord]) -> dict[str, float]:
    if not reference:
        raise ValueError("reference population is empty")
    scales: dict[str, float] = {}
    for name in FEATURE_NAMES:
        if name == "amplitude_percentile":
            scales[name] = 25.0
            continue
        values = np.asarray(
            [row.descriptors[name] for row in reference], dtype=np.float64
        )
        scale = float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
        if name in {"temporal_peak_count", "spectral_peak_count"}:
            scale = max(scale, 1.0)
        scales[name] = max(scale, 1.0e-6)
    return scales


def nearest_support_distances(
    queries: list[SignalRecord],
    reference: list[SignalRecord],
    *,
    weights: dict[str, float] | None = None,
    scales: dict[str, float] | None = None,
    chunk_size: int = 512,
) -> SupportDistances:
    if not queries or not reference:
        raise ValueError("query and reference populations must be non-empty")
    active_weights = MATCH_WEIGHTS if weights is None else weights
    if set(active_weights) != set(FEATURE_NAMES):
        raise ValueError("weights must cover the feature contract")
    if not np.isclose(sum(active_weights.values()), 1.0):
        raise ValueError("weights must sum to one")
    active_scales = feature_scales(reference) if scales is None else scales
    scale = np.asarray([active_scales[name] for name in FEATURE_NAMES])
    weight = np.sqrt(
        np.asarray([active_weights[name] for name in FEATURE_NAMES])
    )

    def matrix(rows: list[SignalRecord]) -> np.ndarray:
        return np.asarray(
            [[row.descriptors[name] for name in FEATURE_NAMES] for row in rows],
            dtype=np.float64,
        )

    reference_matrix = matrix(reference) / scale * weight
    query_matrix = matrix(queries) / scale * weight
    nearest_indices = np.empty(len(queries), dtype=np.int64)
    distances = np.empty(len(queries), dtype=np.float64)
    for start in range(0, len(queries), chunk_size):
        stop = min(len(queries), start + chunk_size)
        squared = np.sum(
            np.square(
                query_matrix[start:stop, None, :]
                - reference_matrix[None, :, :]
            ),
            axis=2,
        )
        indices = np.argmin(squared, axis=1)
        nearest_indices[start:stop] = indices
        distances[start:stop] = np.sqrt(
            squared[np.arange(stop - start), indices]
        )
    return SupportDistances(nearest_indices, distances)


def quantile_edges(values: np.ndarray, n_bins: int = 4) -> np.ndarray:
    edges = np.quantile(
        np.asarray(values, dtype=np.float64),
        np.linspace(0.0, 1.0, n_bins + 1),
    )
    if np.any(np.diff(edges) <= 0.0):
        raise ValueError("quantile edges are not strictly increasing")
    edges[0] = np.nextafter(edges[0], -np.inf)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    return edges


def support_grid(
    rows: list[SignalRecord],
    supported: np.ndarray,
    *,
    x_name: str,
    y_name: str,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> dict[str, np.ndarray]:
    if len(rows) != len(supported):
        raise ValueError("rows and support mask must have equal length")
    x = np.asarray([row.descriptors[x_name] for row in rows])
    y = np.asarray([row.descriptors[y_name] for row in rows])
    x_bin = np.clip(np.digitize(x, x_edges) - 1, 0, len(x_edges) - 2)
    y_bin = np.clip(np.digitize(y, y_edges) - 1, 0, len(y_edges) - 2)
    shape = (len(y_edges) - 1, len(x_edges) - 1)
    counts = np.zeros(shape, dtype=np.int64)
    supported_counts = np.zeros(shape, dtype=np.int64)
    for xb, yb, is_supported in zip(x_bin, y_bin, supported, strict=True):
        counts[yb, xb] += 1
        supported_counts[yb, xb] += int(is_supported)
    rates = np.full(shape, np.nan, dtype=np.float64)
    np.divide(
        supported_counts,
        counts,
        out=rates,
        where=counts > 0,
    )
    return {"counts": counts, "supported_counts": supported_counts, "rates": rates}


def _serializable_record(
    row: SignalRecord,
    *,
    distance: float,
    nearest: SignalRecord,
    role: str,
    threshold: float,
) -> dict[str, Any]:
    return {
        "role": role,
        "real_id": row.identifier,
        "class_name": row.metadata["class_name"],
        "class_label": CLASS_LABELS[row.metadata["class_name"]],
        "distance": float(distance),
        "threshold": float(threshold),
        "supported": bool(distance <= threshold),
        "nearest_simulation_id": nearest.identifier,
        "real_descriptors": {
            name: float(row.descriptors[name]) for name in FEATURE_NAMES
        },
        "simulation_descriptors": {
            name: float(nearest.descriptors[name]) for name in FEATURE_NAMES
        },
        "simulation_metadata": {
            key: nearest.metadata[key]
            for key in (
                "duration_ms",
                "doppler_khz",
                "snr_db",
                "event_position_fraction",
                "target_rms",
                "signal_row",
            )
        },
        "_real_signal": row.signal,
        "_simulation_signal": nearest.signal,
        "_event_bounds": (
            float(row.metadata["event_start_index"]),
            float(row.metadata["event_end_index"]),
        ),
    }


def build_support_model(
    real_root: Path,
    simulation_root: Path,
) -> dict[str, Any]:
    real_train = load_particle_population(real_root, split="train")
    real_validation = load_particle_population(real_root, split="val")
    simulation_train = load_ssl_simulation_population(
        simulation_root, component_count=1, split="train"
    )
    simulation_validation = load_ssl_simulation_population(
        simulation_root, component_count=1, split="validation"
    )
    scales = feature_scales(simulation_train)
    simulation_reference = nearest_support_distances(
        simulation_validation, simulation_train, scales=scales
    )
    threshold = float(np.quantile(simulation_reference.distances, 0.95))

    validation_rows: list[SignalRecord] = []
    validation_classes: list[str] = []
    class_summaries: dict[str, dict[str, Any]] = {}
    class_distance_arrays: dict[str, np.ndarray] = {}
    class_nearest_arrays: dict[str, np.ndarray] = {}
    for class_name in CLASS_ORDER:
        rows = real_validation[class_name]
        distances = nearest_support_distances(rows, simulation_train, scales=scales)
        support = distances.distances <= threshold
        class_distance_arrays[class_name] = distances.distances
        class_nearest_arrays[class_name] = distances.nearest_indices
        validation_rows.extend(rows)
        validation_classes.extend([class_name] * len(rows))
        class_summaries[class_name] = {
            "n": len(rows),
            "supported": int(support.sum()),
            "support_rate": float(np.mean(support)),
            "median_distance": float(np.median(distances.distances)),
            "q95_distance": float(np.quantile(distances.distances, 0.95)),
        }

    all_distances = np.concatenate(
        [class_distance_arrays[name] for name in CLASS_ORDER]
    )
    all_nearest = np.concatenate(
        [class_nearest_arrays[name] for name in CLASS_ORDER]
    )
    all_supported = all_distances <= threshold
    supported_indices = np.flatnonzero(all_supported)
    if supported_indices.size:
        supported_median = float(np.median(all_distances[supported_indices]))
        covered_index = int(
            supported_indices[
                np.argmin(
                    np.abs(all_distances[supported_indices] - supported_median)
                )
            ]
        )
    else:
        covered_index = int(np.argmin(all_distances))
    boundary_index = int(np.argmin(np.abs(all_distances - threshold)))
    uncovered_index = int(np.argmax(all_distances))
    case_indices = (
        ("covered", covered_index),
        ("boundary", boundary_index),
        ("uncovered", uncovered_index),
    )
    cases = [
        _serializable_record(
            validation_rows[index],
            distance=float(all_distances[index]),
            nearest=simulation_train[int(all_nearest[index])],
            role=role,
            threshold=threshold,
        )
        for role, index in case_indices
    ]

    pooled_train = [
        row for class_name in CLASS_ORDER for row in real_train[class_name]
    ]
    grids: dict[str, dict[str, Any]] = {}
    for x_name, y_name in HEATMAP_PAIRS:
        x_edges = quantile_edges(
            np.asarray([row.descriptors[x_name] for row in pooled_train])
        )
        y_edges = quantile_edges(
            np.asarray([row.descriptors[y_name] for row in pooled_train])
        )
        grid = support_grid(
            validation_rows,
            all_supported,
            x_name=x_name,
            y_name=y_name,
            x_edges=x_edges,
            y_edges=y_edges,
        )
        grids[f"{x_name}__{y_name}"] = {
            "x_name": x_name,
            "y_name": y_name,
            "x_edges": x_edges,
            "y_edges": y_edges,
            **grid,
        }

    duration_edges = quantile_edges(
        np.asarray(
            [float(row.metadata["duration_ms"]) for row in simulation_train]
        )
    )
    doppler_edges = quantile_edges(
        np.asarray(
            [float(row.metadata["doppler_khz"]) for row in simulation_train]
        )
    )
    nearest_duration = np.asarray(
        [
            float(simulation_train[int(index)].metadata["duration_ms"])
            for index in all_nearest
        ]
    )
    nearest_doppler = np.asarray(
        [
            float(simulation_train[int(index)].metadata["doppler_khz"])
            for index in all_nearest
        ]
    )
    latent_counts = np.histogram2d(
        nearest_doppler,
        nearest_duration,
        bins=(doppler_edges, duration_edges),
    )[0].astype(np.int64)

    return {
        "schema_version": 1,
        "question": (
            "Does the frozen single-component simulation pool support held-out "
            "real 2/4/10 µm particle events beyond selected best pairs?"
        ),
        "claim_type": "aggregate_comparison",
        "claim_boundary": (
            "Descriptor-space nearest-neighbour support is a distributional "
            "diagnostic. It does not establish waveform identity, biological "
            "realism, causal latent recovery, or SSL utility. The held-out "
            "10 µm population contains one eligible event and cannot support "
            "a class-level coverage estimate."
        ),
        "feature_contract": {
            "names": list(FEATURE_NAMES),
            "weights": MATCH_WEIGHTS,
            "scales": scales,
            "threshold_definition": (
                "95th percentile of simulation-validation to simulation-train "
                "nearest-neighbour distances"
            ),
        },
        "populations": {
            "real_train": {
                name: len(real_train[name]) for name in CLASS_ORDER
            },
            "real_validation": {
                name: len(real_validation[name]) for name in CLASS_ORDER
            },
            "simulation_train": len(simulation_train),
            "simulation_validation": len(simulation_validation),
        },
        "threshold": threshold,
        "simulation_reference_distances": simulation_reference.distances,
        "class_distances": class_distance_arrays,
        "class_summaries": class_summaries,
        "overall": {
            "n": len(validation_rows),
            "supported": int(all_supported.sum()),
            "support_rate": float(np.mean(all_supported)),
            "median_distance": float(np.median(all_distances)),
        },
        "grids": grids,
        "latent_origin": {
            "duration_edges": duration_edges,
            "doppler_edges": doppler_edges,
            "counts": latent_counts,
        },
        "cases": cases,
        "case_selection": {
            "population": (
                f"{len(validation_rows)} eligible held-out real validation "
                "events across 2/4/10 µm"
            ),
            "rule": (
                "Covered: supported event nearest to the median supported "
                "distance; boundary: event nearest the frozen support "
                "threshold; uncovered: maximum-distance event."
            ),
            "selected_ids": [case["real_id"] for case in cases],
            "selected_before_rendering": True,
        },
    }


def serializable_model(model: dict[str, Any]) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {
                key: convert(item)
                for key, item in value.items()
                if not key.startswith("_")
            }
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        return value

    return convert(model)


def _format_edge(value: float, unit: str) -> str:
    precision = 2 if abs(value) < 10 else 1
    return f"{value:.{precision}f}{' ' + unit if unit else ''}"


def _draw_support_grid(
    axis: Any,
    grid: dict[str, Any],
    *,
    minimum_count: int = 3,
) -> None:
    rates = np.asarray(grid["rates"], dtype=np.float64)
    counts = np.asarray(grid["counts"], dtype=np.int64)
    display = rates.copy()
    display[counts < minimum_count] = np.nan
    image = axis.imshow(
        display,
        origin="lower",
        vmin=0.0,
        vmax=1.0,
        cmap="RdYlGn",
        aspect="auto",
    )
    for row in range(counts.shape[0]):
        for column in range(counts.shape[1]):
            count = int(counts[row, column])
            if count == 0:
                text = "n=0"
            elif count < minimum_count:
                text = f"n={count}\nn<3"
            else:
                text = f"{rates[row, column] * 100:.0f}%\nn={count}"
            axis.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                fontsize=8,
                color="#111111",
            )
    x_name = grid["x_name"]
    y_name = grid["y_name"]
    x_edges = np.asarray(grid["x_edges"])
    y_edges = np.asarray(grid["y_edges"])
    axis.set_xticks(
        range(len(x_edges) - 1),
        [
            f"{_format_edge(left, FEATURE_UNITS[x_name])}\n–\n"
            f"{_format_edge(right, FEATURE_UNITS[x_name])}"
            for left, right in zip(x_edges[:-1], x_edges[1:], strict=True)
        ],
        fontsize=7,
    )
    axis.set_yticks(
        range(len(y_edges) - 1),
        [
            f"{_format_edge(left, FEATURE_UNITS[y_name])}–"
            f"{_format_edge(right, FEATURE_UNITS[y_name])}"
            for left, right in zip(y_edges[:-1], y_edges[1:], strict=True)
        ],
        fontsize=7,
    )
    axis.set_xlabel(DISPLAY_FEATURE_LABELS[x_name])
    axis.set_ylabel(DISPLAY_FEATURE_LABELS[y_name])
    axis.set_title("Held-out real support rate", loc="left", fontweight="bold")
    return image


def render_support_overview(model: dict[str, Any], destination: Path) -> None:
    figure = Figure(figsize=(15.0, 10.0), constrained_layout=True)
    axes = figure.subplots(2, 2)
    distance_axis = axes[0, 0]
    reference = np.sort(model["simulation_reference_distances"])
    distance_axis.plot(
        reference,
        np.linspace(0.0, 1.0, len(reference), endpoint=True),
        color="#3b3b3b",
        linestyle="--",
        linewidth=2.0,
        label="Simulation validation",
    )
    for class_name in CLASS_ORDER:
        values = np.sort(model["class_distances"][class_name])
        distance_axis.plot(
            values,
            np.linspace(0.0, 1.0, len(values), endpoint=True),
            color=CLASS_COLORS[class_name],
            linewidth=2.2,
            marker="o",
            markersize=4.0,
            label=CLASS_LABELS[class_name],
        )
    distance_axis.axvline(
        model["threshold"],
        color="#b2182b",
        linewidth=1.8,
        label=f"Support threshold = {model['threshold']:.3f}",
    )
    distance_axis.set_xlabel("Weighted nearest-simulation distance")
    distance_axis.set_ylabel("Cumulative fraction")
    distance_axis.set_title(
        "Held-out real events are compared with the simulator's own density",
        loc="left",
        fontweight="bold",
    )
    distance_axis.legend(frameon=False, fontsize=9)
    distance_axis.grid(alpha=0.2)

    for axis, grid in zip(
        (axes[0, 1], axes[1, 0]),
        model["grids"].values(),
        strict=True,
    ):
        _draw_support_grid(axis, grid)

    latent = model["latent_origin"]
    latent_axis = axes[1, 1]
    counts = np.asarray(latent["counts"])
    latent_axis.imshow(counts, origin="lower", cmap="Blues", aspect="auto")
    for row in range(counts.shape[0]):
        for column in range(counts.shape[1]):
            latent_axis.text(
                column,
                row,
                str(int(counts[row, column])),
                ha="center",
                va="center",
                fontsize=9,
            )
    latent_axis.set_xticks(range(4), ["Q1", "Q2", "Q3", "Q4"])
    latent_axis.set_yticks(range(4), ["Q1", "Q2", "Q3", "Q4"])
    latent_axis.set_xlabel("Simulation duration quantile")
    latent_axis.set_ylabel("Simulation Doppler quantile")
    latent_axis.set_title(
        "Origin of nearest simulated neighbours: not inferred real latents",
        loc="left",
        fontweight="bold",
    )

    summary = (
        f"2 µm {model['class_summaries']['2um']['supported']}/"
        f"{model['class_summaries']['2um']['n']}  |  "
        f"4 µm {model['class_summaries']['4um']['supported']}/"
        f"{model['class_summaries']['4um']['n']}  |  "
        "10 µm n=1: inconclusive"
    )
    figure.suptitle(
        f"Only {model['overall']['supported']}/{model['overall']['n']} held-out "
        "real events meet the simulator-density support threshold\n"
        + summary,
        fontsize=16,
        fontweight="bold",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=170, facecolor="white")


def render_support_explainer(model: dict[str, Any], destination: Path) -> None:
    figure = Figure(figsize=(15.0, 9.0), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(0.9, 1.2))
    local_axis = figure.add_subplot(grid[0, 0])
    global_axis = figure.add_subplot(grid[0, 1])
    bar_axis = figure.add_subplot(grid[1, 0])
    threshold_axis = figure.add_subplot(grid[1, 1])

    for axis in (local_axis, global_axis, threshold_axis):
        axis.set_axis_off()

    local_axis.text(
        0.0,
        0.95,
        "Previous visual interface",
        transform=local_axis.transAxes,
        fontsize=18,
        fontweight="bold",
        va="top",
    )
    local_axis.text(
        0.0,
        0.70,
        "One favourable real-simulation pair\nfor each particle size",
        transform=local_axis.transAxes,
        fontsize=17,
        va="top",
        color="#174a6e",
    )
    local_axis.text(
        0.0,
        0.34,
        "Question answered:\nCan the simulator produce at least one close signal?",
        transform=local_axis.transAxes,
        fontsize=14,
        va="top",
    )
    local_axis.text(
        0.0,
        0.05,
        "Answer: yes, locally",
        transform=local_axis.transAxes,
        fontsize=16,
        fontweight="bold",
        color="#238b45",
        va="bottom",
    )

    global_axis.text(
        0.0,
        0.95,
        "New held-out coverage audit",
        transform=global_axis.transAxes,
        fontsize=18,
        fontweight="bold",
        va="top",
    )
    global_axis.text(
        0.0,
        0.70,
        f"All {model['overall']['n']} eligible validation events\n"
        "are matched to the same frozen pool",
        transform=global_axis.transAxes,
        fontsize=17,
        va="top",
        color="#174a6e",
    )
    global_axis.text(
        0.0,
        0.34,
        "Question answered:\nIs most of the measured distribution supported?",
        transform=global_axis.transAxes,
        fontsize=14,
        va="top",
    )
    global_axis.text(
        0.0,
        0.05,
        f"Answer: only {model['overall']['supported']}/"
        f"{model['overall']['n']} meet the threshold",
        transform=global_axis.transAxes,
        fontsize=16,
        fontweight="bold",
        color="#b2182b",
        va="bottom",
    )

    labels = ("All", "2 µm", "4 µm", "10 µm")
    supported = (
        model["overall"]["supported"],
        model["class_summaries"]["2um"]["supported"],
        model["class_summaries"]["4um"]["supported"],
        model["class_summaries"]["10um"]["supported"],
    )
    totals = (
        model["overall"]["n"],
        model["class_summaries"]["2um"]["n"],
        model["class_summaries"]["4um"]["n"],
        model["class_summaries"]["10um"]["n"],
    )
    rates = np.asarray(
        [
            count / total if total >= 3 else 0.0
            for count, total in zip(supported, totals, strict=True)
        ],
        dtype=float,
    )
    colors = ("#37474f", CLASS_COLORS["2um"], CLASS_COLORS["4um"], "#bdbdbd")
    positions = np.arange(len(labels))
    bars = bar_axis.barh(positions, rates * 100.0, color=colors, height=0.62)
    bar_axis.set_yticks(positions, labels)
    bar_axis.invert_yaxis()
    bar_axis.set_xlim(0.0, 100.0)
    bar_axis.set_xlabel("Events within the simulator-density threshold (%)")
    bar_axis.set_title(
        "Coverage is low for the two estimable classes",
        loc="left",
        fontsize=18,
        fontweight="bold",
    )
    bar_axis.grid(axis="x", alpha=0.2)
    for index, (bar, count, total) in enumerate(
        zip(bars, supported, totals, strict=True)
    ):
        label = (
            "n=1: insufficient"
            if labels[index] == "10 µm"
            else f"{count}/{total} ({100.0 * count / total:.1f}%)"
        )
        bar_axis.text(
            2.0 if total < 3 else min(bar.get_width() + 2.0, 84.0),
            bar.get_y() + bar.get_height() / 2.0,
            label,
            va="center",
            fontsize=13,
            fontweight="bold",
        )

    threshold_axis.text(
        0.0,
        0.95,
        "How the threshold is defined",
        transform=threshold_axis.transAxes,
        fontsize=18,
        fontweight="bold",
        va="top",
    )
    threshold_axis.text(
        0.0,
        0.73,
        "1. Take simulation validation signals.\n"
        "2. Match each one to simulation train.\n"
        "3. Keep 95% of these internal distances.\n"
        f"4. The resulting threshold is d = {model['threshold']:.3f}.",
        transform=threshold_axis.transAxes,
        fontsize=15,
        va="top",
        linespacing=1.5,
    )
    threshold_axis.text(
        0.0,
        0.27,
        "A real event is called supported only when its nearest simulation is\n"
        "at least as close as this simulator-defined reference.",
        transform=threshold_axis.transAxes,
        fontsize=14,
        va="top",
        color="#37474f",
    )
    threshold_axis.text(
        0.0,
        0.04,
        "This is descriptor-space support, not waveform identity.",
        transform=threshold_axis.transAxes,
        fontsize=14,
        fontweight="bold",
        va="bottom",
        color="#8c510a",
    )

    figure.suptitle(
        "A close example proves existence; it does not prove distribution coverage",
        fontsize=20,
        fontweight="bold",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=170, facecolor="white")


def _normalized(signal: np.ndarray) -> np.ndarray:
    centered = np.asarray(signal, dtype=np.float64) - float(np.mean(signal))
    rms = float(np.sqrt(np.mean(np.square(centered))))
    return centered / max(rms, 1.0e-12)


def render_support_cases(model: dict[str, Any], destination: Path) -> None:
    figure = Figure(figsize=(16.0, 10.0), constrained_layout=True)
    axes = figure.subplots(
        3,
        3,
        sharex="col",
        gridspec_kw={"width_ratios": (1.0, 1.0, 0.72)},
    )
    time_ms = np.arange(4096) / 1_000_000.0 * 1000.0
    role_labels = {
        "covered": "Representative supported case",
        "boundary": "Boundary case",
        "uncovered": "Largest support gap",
    }
    for row_index, case in enumerate(model["cases"]):
        real = _normalized(case["_real_signal"])
        simulation = _normalized(case["_simulation_signal"])
        limit = max(np.max(np.abs(real)), np.max(np.abs(simulation))) * 1.05
        real_axis, simulation_axis, explanation_axis = axes[row_index]
        real_axis.plot(time_ms, real, color="#2c7fb8", linewidth=0.8)
        start, end = case["_event_bounds"]
        real_axis.axvspan(
            start / 1000.0,
            end / 1000.0,
            color="#fdae61",
            alpha=0.2,
            label="Real annotation",
        )
        simulation_axis.plot(
            time_ms, simulation, color="#238b45", linewidth=0.8
        )
        for axis in (real_axis, simulation_axis):
            axis.set_ylim(-limit, limit)
            axis.axhline(0.0, color="#888888", linewidth=0.5)
            axis.grid(alpha=0.15)
            axis.set_ylabel("Amplitude / RMS")
        real_axis.set_title(
            f"{role_labels[case['role']]} · real {case['class_label']} · "
            f"d={case['distance']:.3f}",
            loc="left",
            fontweight="bold",
        )
        simulation_axis.set_title(
            f"Nearest simulation · {case['nearest_simulation_id']}",
            loc="left",
            fontweight="bold",
        )
        explanation_axis.set_axis_off()
        real_descriptors = case["real_descriptors"]
        simulation_descriptors = case["simulation_descriptors"]
        if case["role"] == "covered":
            heading = "Why supported"
            message = (
                f"Dominant frequency\n"
                f"{real_descriptors['dominant_frequency_khz']:.1f} vs "
                f"{simulation_descriptors['dominant_frequency_khz']:.1f} kHz\n\n"
                f"Envelope duration\n"
                f"{real_descriptors['duration_25_ms']:.2f} vs "
                f"{simulation_descriptors['duration_25_ms']:.2f} ms\n\n"
                "Several descriptors agree."
            )
            color = "#238b45"
        elif case["role"] == "boundary":
            heading = "Why near the boundary"
            message = (
                f"Distance {case['distance']:.3f}\n"
                f"Threshold {case['threshold']:.3f}\n\n"
                "The shapes are close, but this is the only eligible\n"
                "10 µm validation event.\n\n"
                "No class-level conclusion."
            )
            color = "#b8860b"
        else:
            heading = "Why unsupported"
            message = (
                f"Dominant frequency\n"
                f"{real_descriptors['dominant_frequency_khz']:.1f} vs "
                f"{simulation_descriptors['dominant_frequency_khz']:.1f} kHz\n\n"
                f"Envelope duration\n"
                f"{real_descriptors['duration_25_ms']:.2f} vs "
                f"{simulation_descriptors['duration_25_ms']:.2f} ms\n\n"
                "The real frequency exceeds the simulated range."
            )
            color = "#b2182b"
        explanation_axis.text(
            0.0,
            0.94,
            heading,
            transform=explanation_axis.transAxes,
            fontsize=15,
            fontweight="bold",
            color=color,
            va="top",
        )
        explanation_axis.text(
            0.0,
            0.76,
            message,
            transform=explanation_axis.transAxes,
            fontsize=12.5,
            va="top",
            linespacing=1.35,
        )
    axes[-1, 0].set_xlabel("Time (ms)")
    axes[-1, 1].set_xlabel("Time (ms)")
    figure.suptitle(
        "Prespecified held-out cases explain, but do not prove, the aggregate result",
        fontsize=16,
        fontweight="bold",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=170, facecolor="white")
