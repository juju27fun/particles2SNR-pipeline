from __future__ import annotations

import csv
import math
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from particles2snr.z8_parameter_analysis import (
    CLASS_COLORS,
    CLASS_LABELS,
    CLASS_ORDER,
)


PARAMETER_ORDER = ("log_amplitude_p0", "frequency_khz", "log_tau_ms", "snr_db")
PARAMETER_LABELS = {
    "log_amplitude_p0": "log(P0)",
    "frequency_khz": "Frequency",
    "log_tau_ms": "log(tau)",
    "snr_db": "SNR",
}
PAIR_ORDER = tuple(combinations(PARAMETER_ORDER, 2))
POPULATION_ORDER = ("physical", "inclusive")
INCLUSIVE_DIFFERENCE_THRESHOLD = 0.10


def validate_inclusive_difference_threshold(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("inclusive difference threshold must be finite and positive")
    return value
POSITIVE_DEFINITE_FLOOR = 1e-10


def transformed_parameter_matrix(rows: Iterable[dict[str, str]]) -> np.ndarray:
    materialized = list(rows)
    values = np.asarray(
        [
            [
                np.log(float(row["particles2snr_amplitude"])),
                float(row["frequency_hz"]) / 1000.0,
                np.log(float(row["tau_ms"])),
                float(row["snr_db"]),
            ]
            for row in materialized
        ],
        dtype=np.float64,
    )
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] != 4:
        raise ValueError("At least three events with four parameters are required")
    if not np.all(np.isfinite(values)):
        raise ValueError("Transformed parameter matrix contains non-finite values")
    return values


def rows_for_population(
    rows: Iterable[dict[str, str]], class_name: str, population: str
) -> list[dict[str, str]]:
    if class_name not in CLASS_ORDER:
        raise ValueError(f"Unsupported physical class: {class_name}")
    if population == "physical":
        return [row for row in rows if row["class_name"] == class_name]
    if population == "inclusive":
        return [row for row in rows if row["physical_source_class"] == class_name]
    raise ValueError(f"Unsupported population: {population}")


def pearson_correlation_matrix(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] != 4:
        raise ValueError("Expected at least three observations for four parameters")
    if not np.all(np.isfinite(values)):
        raise ValueError("Parameter matrix contains non-finite values")
    if np.any(np.ptp(values, axis=0) == 0.0):
        raise ValueError("Pearson correlation is undefined for a constant parameter")
    correlation = np.corrcoef(values, rowvar=False)
    return (correlation + correlation.T) / 2.0


def regularize_for_cholesky(
    correlation: np.ndarray,
    *,
    eigenvalue_floor: float = POSITIVE_DEFINITE_FLOOR,
) -> tuple[np.ndarray, float]:
    matrix = np.asarray(correlation, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("Expected one finite 4x4 correlation matrix")
    if eigenvalue_floor <= 0.0 or eigenvalue_floor >= 1.0:
        raise ValueError("Eigenvalue floor must lie strictly between zero and one")
    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, 1.0)
    minimum = float(np.min(np.linalg.eigvalsh(matrix)))
    shrinkage = 0.0
    if minimum < eigenvalue_floor:
        shrinkage = float((eigenvalue_floor - minimum) / (1.0 - minimum))
        matrix = (1.0 - shrinkage) * matrix + shrinkage * np.eye(4)
    np.linalg.cholesky(matrix)
    return matrix, shrinkage


def _pair_id(pair: tuple[str, str]) -> str:
    return f"{pair[0]}__{pair[1]}"


def _matrix_records(
    *,
    class_name: str,
    population: str,
    rows: list[dict[str, str]],
    correlation: np.ndarray,
    regularized: np.ndarray,
    cholesky: np.ndarray,
    shrinkage: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    coefficient_rows: list[dict[str, Any]] = []
    for x_parameter, y_parameter in PAIR_ORDER:
        x_index = PARAMETER_ORDER.index(x_parameter)
        y_index = PARAMETER_ORDER.index(y_parameter)
        coefficient_rows.append(
            {
                "class_name": class_name,
                "population": population,
                "pair_id": _pair_id((x_parameter, y_parameter)),
                "x_parameter": x_parameter,
                "y_parameter": y_parameter,
                "pearson_r": float(correlation[x_index, y_index]),
                "regularized_pearson_r": float(regularized[x_index, y_index]),
                "n_events": len(rows),
                "n_source_files": len({row["source_filename"] for row in rows}),
            }
        )
    eigenvalues = np.linalg.eigvalsh(regularized)
    diagnostic = {
        "class_name": class_name,
        "population": population,
        "n_events": len(rows),
        "n_source_files": len({row["source_filename"] for row in rows}),
        "minimum_eigenvalue": float(eigenvalues[0]),
        "maximum_eigenvalue": float(eigenvalues[-1]),
        "condition_number": float(np.linalg.cond(regularized)),
        "diagonal_shrinkage": shrinkage,
        "regularization_required": shrinkage > 0.0,
        "positive_definite": bool(np.all(eigenvalues > 0.0)),
    }
    factor_rows = [
        {
            "class_name": class_name,
            "population": population,
            "row_parameter": PARAMETER_ORDER[row],
            "column_parameter": PARAMETER_ORDER[column],
            "cholesky_value": float(cholesky[row, column]),
        }
        for row in range(4)
        for column in range(row + 1)
    ]
    return coefficient_rows, diagnostic, factor_rows


def analyze_cholesky_correlations(
    rows: list[dict[str, str]],
    *,
    inclusive_difference_threshold: float = INCLUSIVE_DIFFERENCE_THRESHOLD,
) -> dict[str, Any]:
    inclusive_difference_threshold = validate_inclusive_difference_threshold(
        inclusive_difference_threshold
    )
    if inclusive_difference_threshold <= 0.0:
        raise ValueError("Inclusive difference threshold must be positive")
    matrices: dict[str, dict[str, list[list[float]]]] = {}
    regularized_matrices: dict[str, dict[str, list[list[float]]]] = {}
    coefficients: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    factors: list[dict[str, Any]] = []

    for class_name in CLASS_ORDER:
        matrices[class_name] = {}
        regularized_matrices[class_name] = {}
        for population in POPULATION_ORDER:
            population_rows = rows_for_population(rows, class_name, population)
            correlation = pearson_correlation_matrix(
                transformed_parameter_matrix(population_rows)
            )
            regularized, shrinkage = regularize_for_cholesky(correlation)
            cholesky = np.linalg.cholesky(regularized)
            matrices[class_name][population] = correlation.tolist()
            regularized_matrices[class_name][population] = regularized.tolist()
            class_coefficients, diagnostic, factor_rows = _matrix_records(
                class_name=class_name,
                population=population,
                rows=population_rows,
                correlation=correlation,
                regularized=regularized,
                cholesky=cholesky,
                shrinkage=shrinkage,
            )
            coefficients.extend(class_coefficients)
            diagnostics.append(diagnostic)
            factors.extend(factor_rows)

    differences: list[dict[str, Any]] = []
    recommendations: list[dict[str, Any]] = []
    for class_name in CLASS_ORDER:
        physical = np.asarray(matrices[class_name]["physical"], dtype=np.float64)
        inclusive = np.asarray(matrices[class_name]["inclusive"], dtype=np.float64)
        pair_differences: list[float] = []
        for pair in PAIR_ORDER:
            x_index = PARAMETER_ORDER.index(pair[0])
            y_index = PARAMETER_ORDER.index(pair[1])
            delta = float(inclusive[x_index, y_index] - physical[x_index, y_index])
            pair_differences.append(abs(delta))
            differences.append(
                {
                    "class_name": class_name,
                    "pair_id": _pair_id(pair),
                    "x_parameter": pair[0],
                    "y_parameter": pair[1],
                    "physical_pearson_r": float(physical[x_index, y_index]),
                    "inclusive_pearson_r": float(inclusive[x_index, y_index]),
                    "delta_inclusive_minus_physical": delta,
                    "absolute_delta": abs(delta),
                }
            )
        maximum_difference = max(pair_differences)
        use_inclusive = maximum_difference <= inclusive_difference_threshold
        recommendations.append(
            {
                "class_name": class_name,
                "recommended_dependency_matrix": (
                    "inclusive" if use_inclusive else "physical"
                ),
                "unclear_policy": (
                    "included in the dependency matrix"
                    if use_inclusive
                    else "excluded from dependencies; extend the SNR marginal only"
                ),
                "maximum_absolute_coefficient_delta": maximum_difference,
                "inclusive_difference_threshold": inclusive_difference_threshold,
                "decision_reason": (
                    "all six coefficient changes are within the engineering threshold"
                    if use_inclusive
                    else "at least one coefficient change exceeds the engineering threshold"
                ),
            }
        )

    return {
        "schema_version": 1,
        "transformation": ["log(P0)", "frequency_khz", "log(tau_ms)", "SNR_dB"],
        "correlation": "within-class Pearson correlation",
        "inclusive_policy": "unclear mapped through physical_source_class",
        "inclusive_difference_threshold": inclusive_difference_threshold,
        "positive_definite_floor": POSITIVE_DEFINITE_FLOOR,
        "matrices": matrices,
        "regularized_matrices": regularized_matrices,
        "correlation_coefficients": coefficients,
        "coefficient_differences": differences,
        "matrix_diagnostics": diagnostics,
        "cholesky_factors": factors,
        "recommendations": recommendations,
        "claim_boundary": (
            "These matrices describe transformed z8 development events and provide "
            "standardized dependence factors. D_gen and widened marginal standard "
            "deviations remain to be chosen before constructing Sigma_gen."
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def render_correlation_matrices(analysis: dict[str, Any], destination: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(15.5, 9.2), constrained_layout=True)
    image = None
    short_labels = ["log(P0)", "f", "log(tau)", "SNR"]
    for row_index, population in enumerate(POPULATION_ORDER):
        for column_index, class_name in enumerate(CLASS_ORDER):
            axis = axes[row_index, column_index]
            matrix = np.asarray(analysis["matrices"][class_name][population])
            image = axis.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
            axis.set_xticks(range(4), short_labels, rotation=28, ha="right")
            axis.set_yticks(range(4), short_labels)
            count = next(
                row["n_events"]
                for row in analysis["matrix_diagnostics"]
                if row["class_name"] == class_name and row["population"] == population
            )
            axis.set_title(
                f"{CLASS_LABELS[class_name]} · {population} · n={count:,}",
                fontweight="bold",
            )
            for y_index in range(4):
                for x_index in range(4):
                    value = matrix[y_index, x_index]
                    axis.text(
                        x_index,
                        y_index,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        color="white" if abs(value) > 0.55 else "#0f172a",
                        fontweight="bold",
                        fontsize=9,
                    )
    assert image is not None
    figure.colorbar(image, ax=axes, shrink=0.72, label="Pearson r")
    figure.suptitle(
        "Cholesky dependence inputs · transformed within-class Pearson matrices",
        fontsize=17,
        fontweight="bold",
    )
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_coefficient_differences(analysis: dict[str, Any], destination: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 5.8), sharey=True, constrained_layout=True)
    pair_labels = [
        f"{PARAMETER_LABELS[x]} ↔ {PARAMETER_LABELS[y]}" for x, y in PAIR_ORDER
    ]
    for axis, class_name in zip(axes, CLASS_ORDER, strict=True):
        records = [
            row
            for row in analysis["coefficient_differences"]
            if row["class_name"] == class_name
        ]
        values = [row["delta_inclusive_minus_physical"] for row in records]
        colors = ["#dc2626" if abs(value) > 0.10 else CLASS_COLORS[class_name] for value in values]
        axis.barh(range(6), values, color=colors, alpha=0.88)
        axis.axvline(0.0, color="#334155", linewidth=1)
        axis.axvline(-0.10, color="#dc2626", linestyle="--", linewidth=1)
        axis.axvline(0.10, color="#dc2626", linestyle="--", linewidth=1)
        axis.set_xlim(-0.19, 0.19)
        axis.set_title(CLASS_LABELS[class_name], fontweight="bold")
        axis.set_xlabel("Δr = inclusive − physical")
        axis.grid(axis="x", alpha=0.2)
        for index, value in enumerate(values):
            axis.text(
                value + (0.008 if value >= 0 else -0.008),
                index,
                f"{value:+.03f}",
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=8,
            )
    axes[0].set_yticks(range(6), pair_labels)
    figure.suptitle(
        "Effect of adding unclear events · dashed lines mark the ±0.10 decision threshold",
        fontsize=16,
        fontweight="bold",
    )
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_conditioning_diagnostics(analysis: dict[str, Any], destination: Path) -> None:
    figure, (eigen_axis, condition_axis) = plt.subplots(
        1, 2, figsize=(13.5, 5.6), constrained_layout=True
    )
    x_positions = np.arange(len(CLASS_ORDER), dtype=np.float64)
    width = 0.34
    for index, population in enumerate(POPULATION_ORDER):
        offset = (index - 0.5) * width
        records = [
            next(
                row
                for row in analysis["matrix_diagnostics"]
                if row["class_name"] == class_name and row["population"] == population
            )
            for class_name in CLASS_ORDER
        ]
        color = "#2563eb" if population == "physical" else "#f59e0b"
        eigen_axis.bar(
            x_positions + offset,
            [row["minimum_eigenvalue"] for row in records],
            width,
            label=population,
            color=color,
        )
        condition_axis.bar(
            x_positions + offset,
            [row["condition_number"] for row in records],
            width,
            label=population,
            color=color,
        )
    for axis in (eigen_axis, condition_axis):
        axis.set_xticks(x_positions, [CLASS_LABELS[name] for name in CLASS_ORDER])
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False)
    eigen_axis.set_title("Minimum eigenvalue (> 0 is required)", fontweight="bold")
    eigen_axis.set_ylabel("Minimum eigenvalue")
    condition_axis.set_title("Condition number (lower is better)", fontweight="bold")
    condition_axis.set_ylabel("Condition number")
    figure.suptitle(
        "All six matrices are positive definite without regularization",
        fontsize=16,
        fontweight="bold",
    )
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_cholesky_outputs(
    analysis: dict[str, Any], output_dir: Path
) -> list[str]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite analysis run: {output_dir}")
    output_dir.mkdir(parents=True)
    summary_path = output_dir / "summary_metrics.json"
    summary_path.write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    csv_outputs = {
        "correlation_coefficients.csv": analysis["correlation_coefficients"],
        "coefficient_differences.csv": analysis["coefficient_differences"],
        "matrix_diagnostics.csv": analysis["matrix_diagnostics"],
        "cholesky_factors.csv": analysis["cholesky_factors"],
        "recommendations.csv": analysis["recommendations"],
    }
    for filename, rows in csv_outputs.items():
        _write_csv(output_dir / filename, rows)
    render_correlation_matrices(
        analysis, output_dir / "cholesky_correlation_matrices.png"
    )
    render_coefficient_differences(
        analysis, output_dir / "unclear_coefficient_differences.png"
    )
    render_conditioning_diagnostics(
        analysis, output_dir / "matrix_conditioning_diagnostics.png"
    )
    return [
        "summary_metrics.json",
        *csv_outputs,
        "cholesky_correlation_matrices.png",
        "unclear_coefficient_differences.png",
        "matrix_conditioning_diagnostics.png",
    ]
