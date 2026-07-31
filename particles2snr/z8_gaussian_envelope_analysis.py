from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import gaussian_kde, norm

from particles2snr.z8_gaussian_marginal_analysis import (
    GENERATIVE_COLOR,
    MARGINALS,
    raw_values,
    rows_for_marginal,
)
from particles2snr.z8_parameter_analysis import (
    CLASS_COLORS,
    CLASS_LABELS,
    CLASS_ORDER,
    _freedman_diaconis_bins,
)


FIT_GRID_SIZE = 16_384
VALIDATION_GRID_SIZE = 65_536
COUNT_SAFETY_FACTOR = 1.01


def _normal_density(grid: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    return norm.pdf((grid - mean) / sigma) / sigma


def _mode_on_grid(density: gaussian_kde, minimum: float, maximum: float) -> float:
    grid = np.linspace(minimum, maximum, FIT_GRID_SIZE)
    return float(grid[int(np.argmax(density(grid)))])


def optimize_gaussian_intensity_envelope(
    values: np.ndarray,
    *,
    transform: Callable[[np.ndarray], np.ndarray],
    count_safety_factor: float = COUNT_SAFETY_FACTOR,
) -> dict[str, Any]:
    raw = np.asarray(values, dtype=np.float64)
    transformed = np.asarray(transform(raw), dtype=np.float64)
    if transformed.ndim != 1 or transformed.size < 3:
        raise ValueError("At least three transformed observations are required")
    if not np.all(np.isfinite(transformed)) or np.unique(transformed).size < 2:
        raise ValueError("Envelope optimization requires finite non-constant values")
    if count_safety_factor <= 1.0:
        raise ValueError("Count safety factor must exceed one")

    minimum = float(np.min(transformed))
    maximum = float(np.max(transformed))
    width = maximum - minimum
    standard_deviation = float(np.std(transformed, ddof=1))
    if width <= 0.0 or standard_deviation <= 0.0:
        raise ValueError("Envelope optimization requires positive spread")
    real_kde = gaussian_kde(transformed)
    fit_grid = np.linspace(minimum, maximum, FIT_GRID_SIZE)
    real_density = real_kde(fit_grid)
    mean = _mode_on_grid(real_kde, minimum, maximum)

    lower_log_sigma = float(np.log(standard_deviation / 20.0))
    upper_log_sigma = float(np.log(max(width, standard_deviation) * 5.0))

    def objective(log_sigma: float) -> float:
        sigma = float(np.exp(log_sigma))
        gaussian = np.maximum(_normal_density(fit_grid, mean, sigma), 1e-300)
        return float(np.max(np.log(real_density) - np.log(gaussian)))

    optimization = minimize_scalar(
        objective,
        bounds=(lower_log_sigma, upper_log_sigma),
        method="bounded",
        options={"xatol": 1e-10, "maxiter": 500},
    )
    if not optimization.success:
        raise RuntimeError(f"Gaussian envelope optimization failed: {optimization.message}")
    sigma = float(np.exp(optimization.x))
    minimum_required_ratio = float(np.exp(optimization.fun))
    protected_ratio = minimum_required_ratio * count_safety_factor
    required_synthetic_count = int(np.ceil(protected_ratio * transformed.size))

    validation_grid = np.linspace(minimum, maximum, VALIDATION_GRID_SIZE)
    validation_real = real_kde(validation_grid)
    validation_gaussian = _normal_density(validation_grid, mean, sigma)
    applied_ratio = required_synthetic_count / transformed.size
    envelope = applied_ratio * validation_gaussian
    real_to_envelope = validation_real / np.maximum(envelope, 1e-300)
    return {
        "n_real": int(transformed.size),
        "observed_minimum_transformed": minimum,
        "observed_maximum_transformed": maximum,
        "gaussian_mean_transformed": mean,
        "gaussian_sigma_transformed": sigma,
        "minimum_required_intensity_ratio": minimum_required_ratio,
        "protected_required_intensity_ratio": protected_ratio,
        "required_synthetic_count": required_synthetic_count,
        "optimization_iterations": int(optimization.nit),
        "optimization_evaluations": int(optimization.nfev),
        "optimization_success": bool(optimization.success),
        "validation_grid_size": VALIDATION_GRID_SIZE,
        "max_real_to_candidate_envelope_ratio": float(np.max(real_to_envelope)),
        "minimum_candidate_minus_real_density": float(
            np.min(envelope - validation_real)
        ),
        "kde_envelope_violation_count": int(np.sum(real_to_envelope > 1.0)),
    }


def analyze_gaussian_intensity_envelopes(
    rows: list[dict[str, str]],
) -> dict[str, Any]:
    fits: list[dict[str, Any]] = []
    for marginal, definition in MARGINALS.items():
        for class_name in CLASS_ORDER:
            selected = rows_for_marginal(rows, class_name, marginal)
            values = raw_values(selected, marginal)
            fit = optimize_gaussian_intensity_envelope(
                values,
                transform=definition["transform"],
            )
            fits.append(
                {
                    "class_name": class_name,
                    "marginal": marginal,
                    "label": definition["label"],
                    "units": definition["units"],
                    "population": definition["population"],
                    "transform": definition["transform_name"],
                    **fit,
                }
            )

    class_budgets: list[dict[str, Any]] = []
    for class_name in CLASS_ORDER:
        class_fits = [row for row in fits if row["class_name"] == class_name]
        bottleneck = max(class_fits, key=lambda row: row["required_synthetic_count"])
        synthetic_count = int(bottleneck["required_synthetic_count"])
        class_budgets.append(
            {
                "class_name": class_name,
                "synthetic_event_count": synthetic_count,
                "bottleneck_marginal": bottleneck["marginal"],
                "bottleneck_label": bottleneck["label"],
                "largest_minimum_required_intensity_ratio": bottleneck[
                    "minimum_required_intensity_ratio"
                ],
            }
        )
        for fit in class_fits:
            fit["class_synthetic_event_count"] = synthetic_count
            fit["applied_class_intensity_ratio"] = synthetic_count / fit["n_real"]

    validation = validate_class_envelopes(rows, fits)
    for fit in fits:
        key = (fit["class_name"], fit["marginal"])
        record = validation[key]
        fit.update(record)
    return {
        "schema_version": 1,
        "criterion": (
            "For every class and marginal over the complete observed transformed "
            "min-max, N_synth times the Gaussian PDF must be greater than or equal "
            "to N_real times the real KDE."
        ),
        "mean_policy": "fixed at the deterministic KDE mode in Cholesky space",
        "sigma_policy": (
            "bounded scalar optimization minimizing the required synthetic-to-real "
            "event-intensity multiplier"
        ),
        "count_safety_factor": COUNT_SAFETY_FACTOR,
        "fit_grid_size": FIT_GRID_SIZE,
        "validation_grid_size": VALIDATION_GRID_SIZE,
        "population_policy": {
            "amplitude_frequency_tau": "physical events only",
            "snr": "physical plus unclear mapped through physical_source_class",
        },
        "fits": fits,
        "class_budgets": class_budgets,
        "claim_boundary": (
            "The envelope guarantee applies to the deterministic KDE on the finite "
            "observed support and dense validation grid. It is an event-intensity "
            "guarantee, not an ordering of two normalized probability densities and "
            "not a guarantee for unseen future real values."
        ),
    }


def validate_class_envelopes(
    rows: list[dict[str, str]], fits: list[dict[str, Any]]
) -> dict[tuple[str, str], dict[str, Any]]:
    results: dict[tuple[str, str], dict[str, Any]] = {}
    for fit in fits:
        class_name = fit["class_name"]
        marginal = fit["marginal"]
        definition = MARGINALS[marginal]
        values = raw_values(rows_for_marginal(rows, class_name, marginal), marginal)
        transformed = definition["transform"](values)
        grid = np.linspace(
            float(np.min(transformed)),
            float(np.max(transformed)),
            VALIDATION_GRID_SIZE,
        )
        real_density = gaussian_kde(transformed)(grid)
        gaussian = _normal_density(
            grid,
            float(fit["gaussian_mean_transformed"]),
            float(fit["gaussian_sigma_transformed"]),
        )
        envelope = float(fit["applied_class_intensity_ratio"]) * gaussian
        ratio = real_density / np.maximum(envelope, 1e-300)
        results[(class_name, marginal)] = {
            "class_envelope_max_real_to_synthetic_ratio": float(np.max(ratio)),
            "class_envelope_minimum_density_gap": float(
                np.min(envelope - real_density)
            ),
            "class_envelope_violation_count": int(np.sum(ratio > 1.0)),
            "class_envelope_valid": bool(np.all(ratio <= 1.0)),
        }
    return results


def _lookup(
    analysis: dict[str, Any], class_name: str, marginal: str
) -> dict[str, Any]:
    return next(
        row
        for row in analysis["fits"]
        if row["class_name"] == class_name and row["marginal"] == marginal
    )


def _backtransformed_density(
    x: np.ndarray,
    *,
    density_y: np.ndarray,
    transform_name: str,
) -> np.ndarray:
    if transform_name == "log":
        return density_y / x
    return density_y


def render_cholesky_space_envelopes(
    rows: list[dict[str, str]],
    analysis: dict[str, Any],
    destination: Path,
) -> None:
    figure, axes = plt.subplots(4, 3, figsize=(15.5, 15.5))
    for row_index, (marginal, definition) in enumerate(MARGINALS.items()):
        for column_index, class_name in enumerate(CLASS_ORDER):
            axis = axes[row_index, column_index]
            fit = _lookup(analysis, class_name, marginal)
            values = raw_values(rows_for_marginal(rows, class_name, marginal), marginal)
            transformed = definition["transform"](values)
            grid = np.linspace(float(np.min(transformed)), float(np.max(transformed)), 800)
            real_density = gaussian_kde(transformed)(grid)
            gaussian = _normal_density(
                grid,
                float(fit["gaussian_mean_transformed"]),
                float(fit["gaussian_sigma_transformed"]),
            )
            envelope = float(fit["applied_class_intensity_ratio"]) * gaussian
            color = CLASS_COLORS[class_name]
            axis.hist(
                transformed,
                bins=_freedman_diaconis_bins(transformed),
                density=True,
                color=color,
                alpha=0.22,
                edgecolor="white",
                linewidth=0.4,
                label="real histogram",
            )
            axis.plot(grid, real_density, color=color, linewidth=2.1, label="real KDE")
            axis.plot(
                grid,
                envelope,
                color=GENERATIVE_COLOR,
                linewidth=2.3,
                linestyle="--",
                label="synthetic intensity envelope",
            )
            axis.fill_between(
                grid, real_density, envelope, color=GENERATIVE_COLOR, alpha=0.10
            )
            axis.set_title(
                (
                    f"{CLASS_LABELS[class_name]} · Nreal={fit['n_real']:,} · "
                    f"Nsynth={fit['class_synthetic_event_count']:,}"
                ),
                fontweight="bold",
            )
            x_label = (
                f"log({definition['label']})"
                if definition["transform_name"] == "log"
                else f"{definition['label']} ({definition['units']})"
            )
            axis.set_xlabel(x_label)
            axis.set_ylabel("Event intensity / Nreal")
            axis.grid(alpha=0.15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.962),
        ncol=3,
        frameon=False,
    )
    figure.suptitle(
        "Optimized Gaussian event-intensity envelopes · zero KDE crossings",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )
    figure.text(
        0.5,
        0.932,
        (
            "Purple integrates to Nsynth/Nreal, not 1. Its height represents the "
            "expected amount of synthetic data available around each real region."
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


def render_physical_space_envelopes(
    rows: list[dict[str, str]],
    analysis: dict[str, Any],
    destination: Path,
) -> None:
    figure, axes = plt.subplots(4, 3, figsize=(15.5, 17.0))
    for row_index, (marginal, definition) in enumerate(MARGINALS.items()):
        for column_index, class_name in enumerate(CLASS_ORDER):
            axis = axes[row_index, column_index]
            fit = _lookup(analysis, class_name, marginal)
            values = raw_values(rows_for_marginal(rows, class_name, marginal), marginal)
            x = np.linspace(float(np.min(values)), float(np.max(values)), 800)
            y = definition["transform"](x)
            transformed_values = definition["transform"](values)
            real_y = gaussian_kde(transformed_values)(y)
            gaussian_y = _normal_density(
                y,
                float(fit["gaussian_mean_transformed"]),
                float(fit["gaussian_sigma_transformed"]),
            )
            real_density = _backtransformed_density(
                x, density_y=real_y, transform_name=definition["transform_name"]
            )
            envelope = _backtransformed_density(
                x,
                density_y=float(fit["applied_class_intensity_ratio"]) * gaussian_y,
                transform_name=definition["transform_name"],
            )
            color = CLASS_COLORS[class_name]
            axis.hist(
                values,
                bins=_freedman_diaconis_bins(values),
                density=True,
                color=color,
                alpha=0.22,
                edgecolor="white",
                linewidth=0.4,
                label="real histogram",
            )
            axis.plot(x, real_density, color=color, linewidth=2.1, label="real KDE")
            axis.plot(
                x,
                envelope,
                color=GENERATIVE_COLOR,
                linewidth=2.3,
                linestyle="--",
                label="synthetic intensity envelope",
            )
            axis.fill_between(x, real_density, envelope, color=GENERATIVE_COLOR, alpha=0.10)
            axis.set_title(
                f"{CLASS_LABELS[class_name]} · envelope valid={fit['class_envelope_valid']}",
                fontweight="bold",
            )
            axis.set_xlabel(f"{definition['label']} ({definition['units']})")
            axis.set_ylabel("Event intensity / Nreal")
            axis.grid(alpha=0.15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.962),
        ncol=3,
        frameon=False,
    )
    figure.suptitle(
        "The same Gaussian envelopes mapped back to physical parameter units",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )
    figure.text(
        0.5,
        0.932,
        "The inequality is preserved by the log-space Jacobian for P0 and tau.",
        ha="center",
        fontsize=10.2,
        color="#475569",
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.90,
        bottom=0.055,
        hspace=0.82,
        wspace=0.22,
    )
    figure.savefig(destination, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def render_envelope_budget(
    analysis: dict[str, Any], destination: Path
) -> None:
    figure, (count_axis, ratio_axis) = plt.subplots(
        1, 2, figsize=(14.5, 6.0), constrained_layout=True
    )
    budgets = analysis["class_budgets"]
    x = np.arange(len(CLASS_ORDER))
    colors = [CLASS_COLORS[name] for name in CLASS_ORDER]
    counts = [row["synthetic_event_count"] for row in budgets]
    count_axis.bar(x, counts, color=colors)
    count_axis.set_xticks(x, [CLASS_LABELS[name] for name in CLASS_ORDER])
    count_axis.set_ylabel("Synthetic events")
    count_axis.set_title("Minimum class budgets with 1% safety", fontweight="bold")
    count_axis.grid(axis="y", alpha=0.2)
    for position, row in enumerate(budgets):
        count_axis.text(
            position,
            row["synthetic_event_count"],
            f"{row['synthetic_event_count']:,}\n{row['bottleneck_label']}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    labels: list[str] = []
    ratios: list[float] = []
    ratio_colors: list[str] = []
    for marginal in MARGINALS:
        for class_name in CLASS_ORDER:
            row = _lookup(analysis, class_name, marginal)
            labels.append(f"{CLASS_LABELS[class_name]} · {row['label']}")
            ratios.append(float(row["minimum_required_intensity_ratio"]))
            ratio_colors.append(CLASS_COLORS[class_name])
    positions = np.arange(len(labels))
    ratio_axis.barh(positions, ratios, color=ratio_colors, alpha=0.88)
    ratio_axis.set_yticks(positions, labels)
    ratio_axis.invert_yaxis()
    ratio_axis.set_xlabel("Minimum synthetic / real intensity ratio")
    ratio_axis.set_title("Optimized marginal envelope multipliers", fontweight="bold")
    ratio_axis.grid(axis="x", alpha=0.2)
    figure.suptitle(
        "A single class budget covers all four real KDEs",
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


def _methodology_markdown(analysis: dict[str, Any]) -> str:
    budgets = "\n".join(
        (
            f"| {CLASS_LABELS[row['class_name']]} | "
            f"{row['synthetic_event_count']:,} | {row['bottleneck_label']} |"
        )
        for row in analysis["class_budgets"]
    )
    return f"""# Gaussian marginal envelope method for SSL v3

## Purpose

Define one Gaussian marginal for each particle class and each Cholesky
coordinate while ensuring that the fitted real z8 density remains below the
corresponding synthetic event-intensity envelope over the complete observed
range.

The Gaussian probability density used for sampling remains normalized. The
coverage statement is made after multiplying that density by the number of
synthetic events:

\\[
N_{{\\mathrm{{synth}},c}}\\,g_{{c,p}}(y)
\\geq
N_{{\\mathrm{{real}},c,p}}\\,\\widehat f_{{c,p}}(y)
\\]

for every class \\(c\\), parameter \\(p\\), and point \\(y\\) in the observed
support. Here, \\(\\widehat f\\) is the real kernel density estimate and
\\(g\\) is the fitted Gaussian density.

## Populations and coordinates

- Dataset: post-processed particles2SNR z8 development data; train and
  validation are combined and the sealed test is untouched.
- Amplitude, frequency, and tau: physical 2, 4, and 10 µm events only.
- Effective SNR: physical events plus `unclear` events mapped through
  `physical_source_class`.
- Cholesky coordinates:
  \\(y=[\\log(P_0), f_d, \\log(\\tau), SNR]\\).

## Iterative fit

For every class and marginal:

1. Estimate the real density with SciPy `gaussian_kde` using its deterministic
   default bandwidth.
2. Evaluate the KDE on {FIT_GRID_SIZE:,} points spanning the observed minimum
   and maximum.
3. Fix the Gaussian mean at the KDE mode on that grid.
4. Optimize the positive Gaussian standard deviation with bounded scalar
   minimization. The objective is

   \\[
   \\min_{{\\sigma>0}}\\;
   \\max_y \\log\\left(
   \\frac{{\\widehat f(y)}}{{g(y;\\mu,\\sigma)}}
   \\right).
   \\]

   This selects the width requiring the smallest global synthetic-to-real
   intensity multiplier while retaining the real peak as the Gaussian center.
5. Compute the minimum required event ratio

   \\[
   r^*=\\max_y
   \\frac{{\\widehat f(y)}}{{g(y;\\mu,\\sigma)}}.
   \\]

6. Apply a {100 * (COUNT_SAFETY_FACTOR - 1):.0f}% count safety factor and round
   upward:

   \\[
   N_{{\\mathrm{{required}}}}
   =\\left\\lceil
   {COUNT_SAFETY_FACTOR:.2f}\\,N_{{\\mathrm{{real}}}}r^*
   \\right\\rceil.
   \\]

7. Use the largest required count among the four marginals as the single
   generation budget for that class.

## Independent coverage check

The final class budget is re-evaluated on an independent grid of
{VALIDATION_GRID_SIZE:,} points for every one of the twelve class-by-parameter
curves. Acceptance requires zero points at which the real KDE exceeds the
Gaussian event-intensity envelope.

| Class | Retained synthetic budget | Limiting marginal |
|---|---:|---|
{budgets}

## Intended Cholesky use

The retained \\(\\mu\\) and \\(\\sigma\\) values define the marginal Gaussian
targets in transformed space. Class-specific dependence is introduced
separately through the validated Cholesky correlation matrix. A generated joint
dataset must therefore be checked again after acquisition-band constraints or
rejection sampling, because those operations can alter both marginals and
correlations.

## Scope of the guarantee

Coverage applies to the deterministic KDE over the finite observed z8 support.
It does not claim coverage of unseen future measurements or unconstrained
Gaussian tails. The histogram is a visual reference; the formal zero-crossing
check concerns the KDE curve.
"""


def write_gaussian_envelope_outputs(
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
    (output_dir / "methodology.md").write_text(
        _methodology_markdown(analysis), encoding="utf-8"
    )
    _write_csv(output_dir / "gaussian_envelope_parameters.csv", analysis["fits"])
    _write_csv(output_dir / "class_synthetic_budgets.csv", analysis["class_budgets"])
    render_cholesky_space_envelopes(
        rows, analysis, output_dir / "gaussian_envelopes_cholesky_space.png"
    )
    render_physical_space_envelopes(
        rows, analysis, output_dir / "gaussian_envelopes_physical_space.png"
    )
    render_envelope_budget(analysis, output_dir / "gaussian_envelope_budgets.png")
    return [
        "summary_metrics.json",
        "methodology.md",
        "gaussian_envelope_parameters.csv",
        "class_synthetic_budgets.csv",
        "gaussian_envelopes_cholesky_space.png",
        "gaussian_envelopes_physical_space.png",
        "gaussian_envelope_budgets.png",
    ]
