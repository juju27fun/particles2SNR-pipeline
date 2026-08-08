from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, norm

from particles2snr.z8_parameter_analysis import (
    CLASS_COLORS,
    CLASS_LABELS,
    CLASS_ORDER,
    FBASE_NOMINAL_BAND_KHZ,
    _freedman_diaconis_bins,
)


MODE_GRID_SIZE = 8_192
COVERAGE_STANDARD_DEVIATIONS = 2.0
GENERATIVE_COLOR = "#7c3aed"


def _identity(values: np.ndarray) -> np.ndarray:
    return values


def _log(values: np.ndarray) -> np.ndarray:
    return np.log(values)


def _exp(values: np.ndarray) -> np.ndarray:
    return np.exp(values)


MARGINALS: dict[str, dict[str, Any]] = {
    "amplitude_p0": {
        "source_column": "particles2snr_amplitude",
        "scale": 1.0,
        "label": "Amplitude P0",
        "units": "acquisition units",
        "population": "physical",
        "transform_name": "log",
        "transform": _log,
        "inverse": _exp,
    },
    "frequency_khz": {
        "source_column": "frequency_hz",
        "scale": 1.0 / 1000.0,
        "label": "Frequency",
        "units": "kHz",
        "population": "physical",
        "transform_name": "identity",
        "transform": _identity,
        "inverse": _identity,
    },
    "tau_ms": {
        "source_column": "tau_ms",
        "scale": 1.0,
        "label": "Tau",
        "units": "ms",
        "population": "physical",
        "transform_name": "log",
        "transform": _log,
        "inverse": _exp,
    },
    "snr_db": {
        "source_column": "snr_db",
        "scale": 1.0,
        "label": "Effective F-base SNR",
        "units": "dB",
        "population": "inclusive",
        "transform_name": "identity",
        "transform": _identity,
        "inverse": _identity,
    },
}


def rows_for_marginal(
    rows: Iterable[dict[str, str]], class_name: str, marginal: str
) -> list[dict[str, str]]:
    if class_name not in CLASS_ORDER:
        raise ValueError(f"Unsupported class: {class_name}")
    definition = MARGINALS[marginal]
    if definition["population"] == "inclusive":
        return [row for row in rows if row["physical_source_class"] == class_name]
    return [row for row in rows if row["class_name"] == class_name]


def raw_values(rows: Iterable[dict[str, str]], marginal: str) -> np.ndarray:
    definition = MARGINALS[marginal]
    values = np.asarray(
        [
            float(row[definition["source_column"]]) * float(definition["scale"])
            for row in rows
        ],
        dtype=np.float64,
    )
    if values.ndim != 1 or values.size < 3 or not np.all(np.isfinite(values)):
        raise ValueError("At least three finite marginal values are required")
    if definition["transform_name"] == "log" and np.any(values <= 0.0):
        raise ValueError(f"{marginal} contains non-positive values")
    return values


def _transformed_mode(values: np.ndarray) -> tuple[float, float]:
    if np.unique(values).size < 2:
        raise ValueError("KDE mode requires at least two unique values")
    density = gaussian_kde(values)
    grid = np.linspace(float(np.min(values)), float(np.max(values)), MODE_GRID_SIZE)
    scores = density(grid)
    index = int(np.argmax(scores))
    return float(grid[index]), float(scores[index])


def _normal_mass(lower: float, upper: float, mean: float, sigma: float) -> float:
    return float(norm.cdf((upper - mean) / sigma) - norm.cdf((lower - mean) / sigma))


def fit_gaussian_marginal(
    values: np.ndarray,
    *,
    transform: Callable[[np.ndarray], np.ndarray],
    inverse: Callable[[np.ndarray], np.ndarray],
) -> dict[str, Any]:
    raw = np.asarray(values, dtype=np.float64)
    transformed = np.asarray(transform(raw), dtype=np.float64)
    if not np.all(np.isfinite(transformed)):
        raise ValueError("Marginal transform produced non-finite values")
    minimum = float(np.min(transformed))
    maximum = float(np.max(transformed))
    mode, mode_density = _transformed_mode(transformed)
    sigma = max(mode - minimum, maximum - mode) / COVERAGE_STANDARD_DEVIATIONS
    if sigma <= 0.0:
        raise ValueError("Gaussian marginal has non-positive sigma")
    lower = mode - COVERAGE_STANDARD_DEVIATIONS * sigma
    upper = mode + COVERAGE_STANDARD_DEVIATIONS * sigma
    raw_bounds = inverse(np.asarray([lower, upper], dtype=np.float64))
    observed_mass = _normal_mass(minimum, maximum, mode, sigma)
    return {
        "n": int(raw.size),
        "observed_minimum_raw": float(np.min(raw)),
        "observed_maximum_raw": float(np.max(raw)),
        "observed_minimum_transformed": minimum,
        "observed_maximum_transformed": maximum,
        "kde_mode_transformed": mode,
        "kde_density_at_mode_transformed": mode_density,
        "gaussian_sigma_transformed": sigma,
        "gaussian_lower_2sigma_transformed": lower,
        "gaussian_upper_2sigma_transformed": upper,
        "gaussian_lower_2sigma_raw": float(raw_bounds[0]),
        "gaussian_upper_2sigma_raw": float(raw_bounds[1]),
        "gaussian_mass_inside_observed_range": observed_mass,
        "gaussian_mass_outside_observed_range": 1.0 - observed_mass,
        "observed_range_covered_by_2sigma": bool(
            lower <= minimum and upper >= maximum
        ),
    }


def analyze_gaussian_marginals(rows: list[dict[str, str]]) -> dict[str, Any]:
    fits: list[dict[str, Any]] = []
    for marginal, definition in MARGINALS.items():
        for class_name in CLASS_ORDER:
            selected = rows_for_marginal(rows, class_name, marginal)
            values = raw_values(selected, marginal)
            fit = fit_gaussian_marginal(
                values,
                transform=definition["transform"],
                inverse=definition["inverse"],
            )
            record = {
                "class_name": class_name,
                "marginal": marginal,
                "label": definition["label"],
                "units": definition["units"],
                "population": definition["population"],
                "transform": definition["transform_name"],
                **fit,
                "gaussian_mass_below_zero_raw": 0.0,
                "gaussian_mass_outside_fbase_band": 0.0,
            }
            if marginal == "frequency_khz":
                mean = float(fit["kde_mode_transformed"])
                sigma = float(fit["gaussian_sigma_transformed"])
                record["gaussian_mass_below_zero_raw"] = float(
                    norm.cdf((0.0 - mean) / sigma)
                )
                low, high = FBASE_NOMINAL_BAND_KHZ
                record["gaussian_mass_outside_fbase_band"] = 1.0 - _normal_mass(
                    low, high, mean, sigma
                )
            fits.append(record)
    return {
        "schema_version": 1,
        "rule": (
            "Within each class and marginal, center a Gaussian on the deterministic "
            "KDE mode in Cholesky space and choose sigma so the full observed min-max "
            "lies within mean plus or minus two sigma."
        ),
        "coverage_standard_deviations": COVERAGE_STANDARD_DEVIATIONS,
        "mode_grid_size": MODE_GRID_SIZE,
        "population_policy": {
            "amplitude_frequency_tau": "physical events only",
            "snr": "physical plus unclear mapped through physical_source_class",
        },
        "fits": fits,
        "claim_boundary": (
            "This is an engineering candidate for Gaussian Cholesky marginals, not a "
            "statistical fit. Density curves remain normalized and are not rescaled to "
            "visually dominate the empirical KDE. No truncation is applied."
        ),
    }


def _fit_lookup(
    analysis: dict[str, Any], class_name: str, marginal: str
) -> dict[str, Any]:
    return next(
        row
        for row in analysis["fits"]
        if row["class_name"] == class_name and row["marginal"] == marginal
    )


def _generated_density(
    x: np.ndarray, *, mean: float, sigma: float, transform_name: str
) -> np.ndarray:
    if transform_name == "log":
        density = np.zeros_like(x)
        positive = x > 0.0
        log_x = np.log(x[positive])
        density[positive] = norm.pdf((log_x - mean) / sigma) / (
            x[positive] * sigma
        )
        return density
    return norm.pdf((x - mean) / sigma) / sigma


def render_gaussian_overlay(
    rows: list[dict[str, str]], analysis: dict[str, Any], destination: Path
) -> None:
    figure, axes = plt.subplots(4, 3, figsize=(15.5, 15.5))
    for row_index, (marginal, definition) in enumerate(MARGINALS.items()):
        for column_index, class_name in enumerate(CLASS_ORDER):
            axis = axes[row_index, column_index]
            selected = rows_for_marginal(rows, class_name, marginal)
            values = raw_values(selected, marginal)
            fit = _fit_lookup(analysis, class_name, marginal)
            color = CLASS_COLORS[class_name]
            axis.hist(
                values,
                bins=_freedman_diaconis_bins(values),
                density=True,
                color=color,
                alpha=0.25,
                edgecolor="white",
                linewidth=0.4,
                label="real histogram",
            )
            real_density = gaussian_kde(values)
            width = float(np.max(values) - np.min(values))
            lower = max(0.0, float(np.min(values) - 0.03 * width)) if definition[
                "transform_name"
            ] == "log" else float(np.min(values) - 0.03 * width)
            upper = float(np.max(values) + 0.03 * width)
            x = np.linspace(lower, upper, 600)
            empirical = real_density(x)
            generated = _generated_density(
                x,
                mean=float(fit["kde_mode_transformed"]),
                sigma=float(fit["gaussian_sigma_transformed"]),
                transform_name=definition["transform_name"],
            )
            axis.plot(x, empirical, color=color, linewidth=2.1, label="real KDE")
            axis.plot(
                x,
                generated,
                color=GENERATIVE_COLOR,
                linewidth=2.2,
                linestyle="--",
                label="Gaussian candidate",
            )
            axis.fill_between(x, generated, color=GENERATIVE_COLOR, alpha=0.08)
            mode_raw = float(
                definition["inverse"](
                    np.asarray([fit["kde_mode_transformed"]], dtype=np.float64)
                )[0]
            )
            axis.axvline(
                mode_raw,
                color=GENERATIVE_COLOR,
                linestyle=":",
                linewidth=1.2,
                label="back-transformed y center",
            )
            outside = 100.0 * float(fit["gaussian_mass_outside_observed_range"])
            axis.set_title(
                f"{CLASS_LABELS[class_name]} · n={len(values):,} · outside={outside:.1f}%",
                fontweight="bold",
            )
            axis.set_xlabel(f"{definition['label']} ({definition['units']})")
            axis.set_ylabel("Normalized density")
            axis.grid(alpha=0.15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.962),
        ncol=4,
        frameon=False,
    )
    figure.suptitle(
        "Real z8 marginals versus the peak-centered ±2σ Gaussian candidate",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )
    figure.text(
        0.5,
        0.932,
        (
            "Fit space: [log(P0), frequency, log(tau), SNR] · P0 and tau appear "
            "lognormal in physical units · SNR includes mapped unclear events."
        ),
        ha="center",
        fontsize=10.2,
        color="#475569",
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.90,
        bottom=0.055,
        hspace=0.58,
        wspace=0.22,
    )
    figure.savefig(destination, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def render_candidate_diagnostics(
    analysis: dict[str, Any], destination: Path
) -> None:
    figure, (outside_axis, frequency_axis) = plt.subplots(
        1, 2, figsize=(14.5, 6.2), constrained_layout=True
    )
    labels: list[str] = []
    outside_values: list[float] = []
    colors: list[str] = []
    for marginal in MARGINALS:
        for class_name in CLASS_ORDER:
            fit = _fit_lookup(analysis, class_name, marginal)
            labels.append(f"{CLASS_LABELS[class_name]} · {fit['label']}")
            outside_values.append(
                100.0 * float(fit["gaussian_mass_outside_observed_range"])
            )
            colors.append(CLASS_COLORS[class_name])
    positions = np.arange(len(labels))
    outside_axis.barh(positions, outside_values, color=colors, alpha=0.86)
    outside_axis.set_yticks(positions, labels)
    outside_axis.invert_yaxis()
    outside_axis.set_xlabel("Generated mass outside observed min–max (%)")
    outside_axis.set_title("Broadness cost of the ±2σ rule", fontweight="bold")
    outside_axis.grid(axis="x", alpha=0.2)

    frequency_records = [
        _fit_lookup(analysis, class_name, "frequency_khz")
        for class_name in CLASS_ORDER
    ]
    x_positions = np.arange(3)
    width = 0.36
    frequency_axis.bar(
        x_positions - width / 2,
        [100.0 * row["gaussian_mass_below_zero_raw"] for row in frequency_records],
        width,
        label="frequency < 0 kHz",
        color="#dc2626",
    )
    frequency_axis.bar(
        x_positions + width / 2,
        [
            100.0 * row["gaussian_mass_outside_fbase_band"]
            for row in frequency_records
        ],
        width,
        label="outside nominal 7–80 kHz",
        color="#f59e0b",
    )
    frequency_axis.set_xticks(
        x_positions, [CLASS_LABELS[class_name] for class_name in CLASS_ORDER]
    )
    frequency_axis.set_ylabel("Generated probability mass (%)")
    frequency_axis.set_title(
        "Untruncated frequency Gaussian violates acquisition limits",
        fontweight="bold",
    )
    frequency_axis.grid(axis="y", alpha=0.2)
    frequency_axis.legend(frameon=False)
    figure.suptitle(
        "Candidate Gaussian diagnostics · no truncation or clipping applied",
        fontsize=17,
        fontweight="bold",
    )
    figure.savefig(destination, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_gaussian_marginal_outputs(
    *,
    rows: list[dict[str, str]],
    analysis: dict[str, Any],
    output_dir: Path,
) -> list[str]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite analysis run: {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "summary_metrics.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "gaussian_marginal_parameters.csv", analysis["fits"])
    render_gaussian_overlay(
        rows, analysis, output_dir / "real_vs_gaussian_marginals.png"
    )
    render_candidate_diagnostics(
        analysis, output_dir / "gaussian_candidate_diagnostics.png"
    )
    return [
        "summary_metrics.json",
        "gaussian_marginal_parameters.csv",
        "real_vs_gaussian_marginals.png",
        "gaussian_candidate_diagnostics.png",
    ]
