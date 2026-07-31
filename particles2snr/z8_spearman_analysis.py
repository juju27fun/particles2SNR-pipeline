from __future__ import annotations

import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

from particles2snr.z8_parameter_analysis import (
    CLASS_COLORS,
    CLASS_LABELS,
    CLASS_ORDER,
    PARAMETERS,
    parameter_value,
)


BOOTSTRAP_REPLICATES = 5_000
BOOTSTRAP_SEED = 20_260_722
CI_PERCENTILES = (2.5, 97.5)
PARAMETER_ORDER = tuple(PARAMETERS)
PARAMETER_LABELS = {
    "amplitude_p0": "Amplitude P0",
    "frequency_khz": "Frequency",
    "tau_ms": "Tau",
    "snr_effective_fbase_db": "SNR",
}
PAIR_ORDER = tuple(combinations(PARAMETER_ORDER, 2))


def pair_id(pair: tuple[str, str]) -> str:
    return f"{pair[0]}__{pair[1]}"


def pair_label(pair: tuple[str, str]) -> str:
    return f"{PARAMETER_LABELS[pair[0]]} ↔ {PARAMETER_LABELS[pair[1]]}"


def spearman_coefficient(x: np.ndarray, y: np.ndarray) -> float:
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if x_array.ndim != 1 or y_array.ndim != 1 or x_array.size != y_array.size:
        raise ValueError("Spearman inputs must be one-dimensional and equally sized")
    if x_array.size < 3 or not np.all(np.isfinite(x_array)) or not np.all(np.isfinite(y_array)):
        raise ValueError("At least three finite paired observations are required")
    x_ranks = rankdata(x_array, method="average")
    y_ranks = rankdata(y_array, method="average")
    if np.ptp(x_ranks) == 0.0 or np.ptp(y_ranks) == 0.0:
        raise ValueError("Spearman correlation is undefined for a constant variable")
    return float(np.corrcoef(x_ranks, y_ranks)[0, 1])


def _rank_correlation_matrix(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] != len(PARAMETER_ORDER):
        raise ValueError("Expected at least three observations for all four parameters")
    if not np.all(np.isfinite(values)):
        raise ValueError("Parameter matrix contains non-finite values")
    ranks = np.column_stack(
        [rankdata(values[:, index], method="average") for index in range(values.shape[1])]
    )
    if np.any(np.ptp(ranks, axis=0) == 0.0):
        raise ValueError("Rank correlation is undefined for a constant parameter")
    return np.corrcoef(ranks, rowvar=False)


def marginal_and_partial_correlations(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    marginal_matrix = _rank_correlation_matrix(matrix)
    precision = np.linalg.pinv(marginal_matrix, hermitian=True)
    scale = np.sqrt(np.outer(np.diag(precision), np.diag(precision)))
    partial_matrix = -precision / scale
    np.fill_diagonal(partial_matrix, 1.0)
    marginal = np.asarray(
        [marginal_matrix[PARAMETER_ORDER.index(x), PARAMETER_ORDER.index(y)] for x, y in PAIR_ORDER],
        dtype=np.float64,
    )
    partial = np.asarray(
        [partial_matrix[PARAMETER_ORDER.index(x), PARAMETER_ORDER.index(y)] for x, y in PAIR_ORDER],
        dtype=np.float64,
    )
    return marginal, partial


def rows_for_class(
    rows: Iterable[dict[str, str]], class_name: str, *, include_unclear: bool
) -> list[dict[str, str]]:
    if class_name not in CLASS_ORDER:
        raise ValueError(f"Unsupported physical class: {class_name}")
    if include_unclear:
        return [row for row in rows if row["physical_source_class"] == class_name]
    return [row for row in rows if row["class_name"] == class_name]


def parameter_matrix(rows: Iterable[dict[str, str]]) -> np.ndarray:
    materialized = list(rows)
    return np.asarray(
        [[parameter_value(row, parameter) for parameter in PARAMETER_ORDER] for row in materialized],
        dtype=np.float64,
    )


def _source_groups(rows: list[dict[str, str]]) -> list[np.ndarray]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["source_filename"]].append(index)
    return [np.asarray(grouped[name], dtype=np.int64) for name in sorted(grouped)]


def grouped_bootstrap_correlations(
    rows: list[dict[str, str]], *, replicates: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    if replicates < 1:
        raise ValueError("At least one bootstrap replicate is required")
    matrix = parameter_matrix(rows)
    groups = _source_groups(rows)
    if len(groups) < 3:
        raise ValueError("At least three source files are required for grouped bootstrap")
    marginal = np.empty((replicates, len(PAIR_ORDER)), dtype=np.float64)
    partial = np.empty_like(marginal)
    for replicate in range(replicates):
        sampled = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[index] for index in sampled])
        marginal[replicate], partial[replicate] = marginal_and_partial_correlations(
            matrix[indices]
        )
    return marginal, partial


def _interval(values: np.ndarray) -> tuple[float, float]:
    lower, upper = np.percentile(values, CI_PERCENTILES)
    return float(lower), float(upper)


def _correlation_rows(
    *,
    class_name: str,
    rows: list[dict[str, str]],
    marginal: np.ndarray,
    partial: np.ndarray,
    bootstrap_marginal: np.ndarray,
    bootstrap_partial: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    n_sources = len({row["source_filename"] for row in rows})
    marginal_rows: list[dict[str, Any]] = []
    partial_rows: list[dict[str, Any]] = []
    for index, pair in enumerate(PAIR_ORDER):
        marginal_lower, marginal_upper = _interval(bootstrap_marginal[:, index])
        partial_lower, partial_upper = _interval(bootstrap_partial[:, index])
        common = {
            "class_name": class_name,
            "pair_id": pair_id(pair),
            "pair_label": pair_label(pair),
            "x_parameter": pair[0],
            "y_parameter": pair[1],
            "n_events": len(rows),
            "n_source_files": n_sources,
        }
        marginal_rows.append(
            {
                **common,
                "spearman_rho": float(marginal[index]),
                "ci95_lower": marginal_lower,
                "ci95_upper": marginal_upper,
                "ci_excludes_zero": marginal_lower > 0.0 or marginal_upper < 0.0,
            }
        )
        controls = [name for name in PARAMETER_ORDER if name not in pair]
        partial_rows.append(
            {
                **common,
                "controlled_parameters": controls,
                "partial_spearman_rho": float(partial[index]),
                "ci95_lower": partial_lower,
                "ci95_upper": partial_upper,
                "ci_excludes_zero": partial_lower > 0.0 or partial_upper < 0.0,
            }
        )
    return marginal_rows, partial_rows


def rank_parameter_pairs(marginal_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in marginal_rows:
        by_pair[row["pair_id"]].append(row)
    ranking: list[dict[str, Any]] = []
    for pair in PAIR_ORDER:
        identifier = pair_id(pair)
        records = by_pair[identifier]
        if {record["class_name"] for record in records} != set(CLASS_ORDER):
            raise ValueError(f"Incomplete class coverage for {identifier}")
        ordered = sorted(records, key=lambda record: CLASS_ORDER.index(record["class_name"]))
        class_rhos = {record["class_name"]: float(record["spearman_rho"]) for record in ordered}
        ranking.append(
            {
                "pair_id": identifier,
                "pair_label": pair_label(pair),
                "x_parameter": pair[0],
                "y_parameter": pair[1],
                "rho_2um": class_rhos["2um"],
                "rho_4um": class_rhos["4um"],
                "rho_10um": class_rhos["10um"],
                "overall_score_median_absolute_rho": float(
                    np.median([abs(value) for value in class_rhos.values()])
                ),
                "stable_class_count": sum(bool(record["ci_excludes_zero"]) for record in ordered),
            }
        )
    ranking.sort(
        key=lambda row: (
            -float(row["overall_score_median_absolute_rho"]),
            row["pair_id"],
        )
    )
    for index, row in enumerate(ranking, start=1):
        row["rank"] = index
        row["selected_top_three"] = index <= 3
    return ranking


def analyze_spearman(
    rows: list[dict[str, str]],
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    rng = np.random.default_rng(bootstrap_seed)
    marginal_rows: list[dict[str, Any]] = []
    partial_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    primary_by_key: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]] = {}

    for class_name in CLASS_ORDER:
        primary = rows_for_class(rows, class_name, include_unclear=False)
        marginal, partial = marginal_and_partial_correlations(parameter_matrix(primary))
        bootstrap_marginal, bootstrap_partial = grouped_bootstrap_correlations(
            primary, replicates=bootstrap_replicates, rng=rng
        )
        class_marginal, class_partial = _correlation_rows(
            class_name=class_name,
            rows=primary,
            marginal=marginal,
            partial=partial,
            bootstrap_marginal=bootstrap_marginal,
            bootstrap_partial=bootstrap_partial,
        )
        marginal_rows.extend(class_marginal)
        partial_rows.extend(class_partial)
        for marginal_row, partial_row in zip(class_marginal, class_partial, strict=True):
            primary_by_key[(class_name, marginal_row["pair_id"])] = (
                marginal_row,
                partial_row,
            )

    snr_parameter = "snr_effective_fbase_db"
    for class_name in CLASS_ORDER:
        inclusive = rows_for_class(rows, class_name, include_unclear=True)
        inclusive_marginal, inclusive_partial = marginal_and_partial_correlations(
            parameter_matrix(inclusive)
        )
        bootstrap_marginal, bootstrap_partial = grouped_bootstrap_correlations(
            inclusive, replicates=bootstrap_replicates, rng=rng
        )
        n_primary = len(rows_for_class(rows, class_name, include_unclear=False))
        n_sources = len({row["source_filename"] for row in inclusive})
        for index, pair in enumerate(PAIR_ORDER):
            if snr_parameter not in pair:
                continue
            identifier = pair_id(pair)
            primary_marginal, primary_partial = primary_by_key[(class_name, identifier)]
            marginal_lower, marginal_upper = _interval(bootstrap_marginal[:, index])
            partial_lower, partial_upper = _interval(bootstrap_partial[:, index])
            sensitivity_rows.append(
                {
                    "class_name": class_name,
                    "pair_id": identifier,
                    "pair_label": pair_label(pair),
                    "x_parameter": pair[0],
                    "y_parameter": pair[1],
                    "n_primary_events": n_primary,
                    "n_unclear_events_added": len(inclusive) - n_primary,
                    "n_inclusive_events": len(inclusive),
                    "n_inclusive_source_files": n_sources,
                    "primary_spearman_rho": primary_marginal["spearman_rho"],
                    "inclusive_spearman_rho": float(inclusive_marginal[index]),
                    "marginal_delta_inclusive_minus_primary": float(
                        inclusive_marginal[index] - primary_marginal["spearman_rho"]
                    ),
                    "inclusive_marginal_ci95_lower": marginal_lower,
                    "inclusive_marginal_ci95_upper": marginal_upper,
                    "primary_partial_spearman_rho": primary_partial["partial_spearman_rho"],
                    "inclusive_partial_spearman_rho": float(inclusive_partial[index]),
                    "partial_delta_inclusive_minus_primary": float(
                        inclusive_partial[index] - primary_partial["partial_spearman_rho"]
                    ),
                    "inclusive_partial_ci95_lower": partial_lower,
                    "inclusive_partial_ci95_upper": partial_upper,
                }
            )

    ranking = rank_parameter_pairs(marginal_rows)
    return {
        "schema_version": 1,
        "population": "physical-class z8 development events; sealed test excluded",
        "bootstrap": {
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "unit": "source_filename",
            "ci_percentiles": list(CI_PERCENTILES),
        },
        "selection_rule": (
            "rank all six pairs by median absolute primary Spearman rho across the three "
            "physical classes; retain ranks 1 through 3"
        ),
        "marginal_correlations": marginal_rows,
        "partial_correlations": partial_rows,
        "unclear_sensitivity": sensitivity_rows,
        "ranking": ranking,
        "selected_pairs": [row for row in ranking if row["selected_top_three"]],
        "claim_boundary": (
            "Associations and source-level sampling stability in the post-processed z8 "
            "development population only; no causal or synthetic-generation rule is claimed."
        ),
    }


def _lookup(rows: list[dict[str, Any]], class_name: str, identifier: str) -> dict[str, Any]:
    return next(
        row for row in rows if row["class_name"] == class_name and row["pair_id"] == identifier
    )


def render_spearman_matrices(analysis: dict[str, Any], destination: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 5.3), constrained_layout=True)
    image = None
    for axis, class_name in zip(axes, CLASS_ORDER, strict=True):
        matrix = np.eye(len(PARAMETER_ORDER), dtype=np.float64)
        stable = np.eye(len(PARAMETER_ORDER), dtype=bool)
        for pair in PAIR_ORDER:
            record = _lookup(analysis["marginal_correlations"], class_name, pair_id(pair))
            x_index = PARAMETER_ORDER.index(pair[0])
            y_index = PARAMETER_ORDER.index(pair[1])
            matrix[x_index, y_index] = matrix[y_index, x_index] = record["spearman_rho"]
            stable[x_index, y_index] = stable[y_index, x_index] = record["ci_excludes_zero"]
        image = axis.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
        labels = ["P0", "frequency", "tau", "SNR"]
        axis.set_xticks(range(4), labels, rotation=28, ha="right")
        axis.set_yticks(range(4), labels)
        n_events = _lookup(
            analysis["marginal_correlations"], class_name, pair_id(PAIR_ORDER[0])
        )["n_events"]
        axis.set_title(f"{CLASS_LABELS[class_name]} · n={n_events:,}", fontweight="bold")
        for row_index in range(4):
            for column_index in range(4):
                marker = "*" if row_index != column_index and stable[row_index, column_index] else ""
                value = matrix[row_index, column_index]
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.2f}{marker}",
                    ha="center",
                    va="center",
                    color="white" if abs(value) > 0.55 else "#0f172a",
                    fontsize=9,
                    fontweight="bold",
                )
    assert image is not None
    figure.colorbar(image, ax=axes, shrink=0.82, label="Spearman rho")
    figure.suptitle(
        "Primary Spearman correlations · * source-bootstrap 95% CI excludes zero",
        fontsize=17,
        fontweight="bold",
    )
    figure.savefig(destination, dpi=180, facecolor="white")
    plt.close(figure)


def render_correlation_forest(analysis: dict[str, Any], destination: Path) -> None:
    ranking_ids = [row["pair_id"] for row in analysis["ranking"]]
    labels = [next(row["pair_label"] for row in analysis["ranking"] if row["pair_id"] == value) for value in ranking_ids]
    figure, axes = plt.subplots(1, 3, figsize=(17.0, 7.2), sharey=True, constrained_layout=True)
    y = np.arange(len(ranking_ids), dtype=float)
    for axis, class_name in zip(axes, CLASS_ORDER, strict=True):
        for offset, metric_key, source, color, label in (
            (-0.16, "spearman_rho", "marginal_correlations", "#2563eb", "Marginal"),
            (0.16, "partial_spearman_rho", "partial_correlations", "#ea580c", "Partial"),
        ):
            records = [_lookup(analysis[source], class_name, identifier) for identifier in ranking_ids]
            points = np.asarray([record[metric_key] for record in records])
            lower = np.asarray([record["ci95_lower"] for record in records])
            upper = np.asarray([record["ci95_upper"] for record in records])
            axis.errorbar(
                points,
                y + offset,
                xerr=np.vstack([points - lower, upper - points]),
                fmt="o",
                color=color,
                capsize=2.5,
                markersize=5,
                linewidth=1.3,
                label=label,
            )
        sensitivity = {
            row["pair_id"]: row
            for row in analysis["unclear_sensitivity"]
            if row["class_name"] == class_name
        }
        for row_index, identifier in enumerate(ranking_ids):
            if identifier not in sensitivity:
                continue
            record = sensitivity[identifier]
            point = record["inclusive_spearman_rho"]
            axis.errorbar(
                point,
                row_index,
                xerr=np.asarray(
                    [[point - record["inclusive_marginal_ci95_lower"]], [record["inclusive_marginal_ci95_upper"] - point]]
                ),
                fmt="D",
                markerfacecolor="white",
                markeredgecolor="#7c3aed",
                ecolor="#7c3aed",
                capsize=2.5,
                markersize=5,
                linewidth=1.1,
                label="Marginal + unclear" if row_index == 0 else None,
            )
        axis.axvline(0.0, color="#64748b", linewidth=1.0)
        axis.set_xlim(-1.0, 1.0)
        axis.set_xlabel("Correlation coefficient (95% source-bootstrap CI)")
        axis.set_title(CLASS_LABELS[class_name], fontweight="bold", color=CLASS_COLORS[class_name])
        axis.grid(axis="x", alpha=0.18)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    handles, legend_labels = axes[-1].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="lower center", ncol=3, frameon=False)
    figure.suptitle(
        "Marginal, partial, and unclear-sensitivity Spearman estimates",
        fontsize=17,
        fontweight="bold",
    )
    figure.savefig(destination, dpi=180, facecolor="white")
    plt.close(figure)


def _binned_median(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, 11)))
    centers: list[float] = []
    medians: list[float] = []
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        mask = (x >= left) & (x <= right if right == edges[-1] else x < right)
        if np.count_nonzero(mask) >= 3:
            centers.append(float(np.median(x[mask])))
            medians.append(float(np.median(y[mask])))
    return np.asarray(centers), np.asarray(medians)


def render_selected_joint_densities(
    analysis: dict[str, Any], rows: list[dict[str, str]], destination: Path
) -> None:
    selected = sorted(analysis["selected_pairs"], key=lambda row: row["rank"])
    figure, axes = plt.subplots(3, 3, figsize=(15.5, 13.2), constrained_layout=True)
    for row_index, selected_pair in enumerate(selected):
        pair = (selected_pair["x_parameter"], selected_pair["y_parameter"])
        for column_index, class_name in enumerate(CLASS_ORDER):
            axis = axes[row_index, column_index]
            class_rows = rows_for_class(rows, class_name, include_unclear=False)
            x = np.asarray([parameter_value(row, pair[0]) for row in class_rows])
            y = np.asarray([parameter_value(row, pair[1]) for row in class_rows])
            axis.hexbin(x, y, gridsize=32, mincnt=1, bins="log", cmap="viridis")
            centers, medians = _binned_median(x, y)
            axis.plot(centers, medians, color="#f43f5e", linewidth=2.0, marker="o", markersize=3)
            record = _lookup(analysis["marginal_correlations"], class_name, selected_pair["pair_id"])
            axis.text(
                0.03,
                0.96,
                f"rho={record['spearman_rho']:.2f} [{record['ci95_lower']:.2f}, {record['ci95_upper']:.2f}]",
                transform=axis.transAxes,
                va="top",
                fontsize=8.5,
                bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 2},
            )
            axis.set_xlabel(f"{PARAMETERS[pair[0]]['label']} ({PARAMETERS[pair[0]]['units']})")
            axis.set_ylabel(f"{PARAMETERS[pair[1]]['label']} ({PARAMETERS[pair[1]]['units']})")
            axis.set_title(
                f"#{selected_pair['rank']} · {CLASS_LABELS[class_name]}",
                fontweight="bold",
                color=CLASS_COLORS[class_name],
            )
            axis.grid(alpha=0.10)
    figure.suptitle(
        "Top-three joint relationships · full observed ranges · color is log event count",
        fontsize=17,
        fontweight="bold",
    )
    figure.savefig(destination, dpi=180, facecolor="white")
    plt.close(figure)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )


def write_spearman_outputs(
    *, analysis: dict[str, Any], rows: list[dict[str, str]], output_dir: Path
) -> list[str]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite analysis run: {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "summary_metrics.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "spearman_correlations.csv", analysis["marginal_correlations"])
    _write_csv(output_dir / "partial_spearman_correlations.csv", analysis["partial_correlations"])
    _write_csv(output_dir / "unclear_sensitivity.csv", analysis["unclear_sensitivity"])
    _write_csv(output_dir / "correlation_ranking.csv", analysis["ranking"])
    render_spearman_matrices(analysis, output_dir / "spearman_matrices.png")
    render_correlation_forest(analysis, output_dir / "spearman_forest.png")
    render_selected_joint_densities(analysis, rows, output_dir / "selected_joint_densities.png")
    return [
        "summary_metrics.json",
        "spearman_correlations.csv",
        "partial_spearman_correlations.csv",
        "unclear_sensitivity.csv",
        "correlation_ranking.csv",
        "spearman_matrices.png",
        "spearman_forest.png",
        "selected_joint_densities.png",
    ]
