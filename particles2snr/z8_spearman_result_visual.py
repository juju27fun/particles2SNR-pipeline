from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import rankdata

from particles2snr.z8_parameter_analysis import (
    CLASS_COLORS,
    CLASS_LABELS,
    CLASS_ORDER,
    parameter_value,
)
from particles2snr.z8_spearman_analysis import PAIR_ORDER, pair_id


BLUE = "#2563eb"
ORANGE = "#ea580c"
PURPLE = "#7c3aed"


def _lookup(rows: list[dict[str, Any]], class_name: str, identifier: str) -> dict[str, Any]:
    return next(
        row for row in rows if row["class_name"] == class_name and row["pair_id"] == identifier
    )


def select_real_rank_example(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    physical = sorted(
        (row for row in rows if row["class_name"] == "4um"),
        key=lambda row: (parameter_value(row, "frequency_khz"), row["event_id"]),
    )
    if len(physical) < 10:
        raise ValueError("The real-rank example requires at least ten 4um events")
    positions = [round(quantile * (len(physical) - 1)) for quantile in (0.1, 0.3, 0.5, 0.7, 0.9)]
    selected = [physical[position] for position in positions]
    frequency = np.asarray([parameter_value(row, "frequency_khz") for row in selected])
    tau = np.asarray([parameter_value(row, "tau_ms") for row in selected])
    frequency_rank = rankdata(frequency, method="average")
    tau_rank = rankdata(tau, method="average")
    differences = frequency_rank - tau_rank
    return [
        {
            "event_id": row["event_id"],
            "frequency_khz": float(frequency[index]),
            "frequency_rank": float(frequency_rank[index]),
            "tau_ms": float(tau[index]),
            "tau_rank": float(tau_rank[index]),
            "rank_difference": float(differences[index]),
            "rank_difference_squared": float(differences[index] ** 2),
        }
        for index, row in enumerate(selected)
    ]


def render_estimates_with_explicit_symbols(
    analysis: dict[str, Any], destination: Path
) -> None:
    ranking_ids = [row["pair_id"] for row in analysis["ranking"]]
    labels = [row["pair_label"] for row in analysis["ranking"]]
    figure, axes = plt.subplots(1, 3, figsize=(17.0, 7.5), sharey=True)
    y = np.arange(len(ranking_ids), dtype=float)

    for axis, class_name in zip(axes, CLASS_ORDER, strict=True):
        marginal = [
            _lookup(analysis["marginal_correlations"], class_name, identifier)
            for identifier in ranking_ids
        ]
        partial = [
            _lookup(analysis["partial_correlations"], class_name, identifier)
            for identifier in ranking_ids
        ]
        for records, metric, offset, color, marker in (
            (marginal, "spearman_rho", -0.18, BLUE, "o"),
            (partial, "partial_spearman_rho", 0.18, ORANGE, "s"),
        ):
            points = np.asarray([record[metric] for record in records])
            lower = np.asarray([record["ci95_lower"] for record in records])
            upper = np.asarray([record["ci95_upper"] for record in records])
            axis.errorbar(
                points,
                y + offset,
                xerr=np.vstack([points - lower, upper - points]),
                fmt=marker,
                color=color,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.8,
                capsize=3,
                markersize=6.5,
                linewidth=1.4,
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
            point = float(record["inclusive_spearman_rho"])
            axis.errorbar(
                point,
                row_index,
                xerr=np.asarray(
                    [
                        [point - float(record["inclusive_marginal_ci95_lower"])],
                        [float(record["inclusive_marginal_ci95_upper"]) - point],
                    ]
                ),
                fmt="D",
                markerfacecolor="white",
                markeredgecolor=PURPLE,
                markeredgewidth=1.8,
                ecolor=PURPLE,
                capsize=3,
                markersize=6.5,
                linewidth=1.3,
            )

        axis.axvline(0.0, color="#64748b", linewidth=1.0)
        axis.set_xlim(-1.0, 1.0)
        axis.set_xlabel("Correlation coefficient")
        axis.set_title(
            CLASS_LABELS[class_name],
            fontweight="bold",
            color=CLASS_COLORS[class_name],
            fontsize=14,
        )
        axis.grid(axis="x", alpha=0.18)

    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=BLUE,
            marker="o",
            markerfacecolor=BLUE,
            markeredgecolor="white",
            linewidth=1.4,
            label="Marginal Spearman · physical events",
        ),
        Line2D(
            [0],
            [0],
            color=ORANGE,
            marker="s",
            markerfacecolor=ORANGE,
            markeredgecolor="white",
            linewidth=1.4,
            label="Partial Spearman · physical events",
        ),
        Line2D(
            [0],
            [0],
            color=PURPLE,
            marker="D",
            markerfacecolor="white",
            markeredgecolor=PURPLE,
            linewidth=1.3,
            label="Marginal Spearman · physical + unclear",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=3,
        frameon=True,
        facecolor="white",
        edgecolor="#cbd5e1",
        fontsize=10,
        title="Marker convention",
        title_fontsize=10,
    )
    figure.suptitle(
        "Spearman estimates by particle class",
        fontsize=19,
        fontweight="bold",
        y=0.98,
    )
    figure.text(
        0.5,
        0.935,
        "Horizontal bars are 95% confidence intervals from 5,000 source-file bootstrap repetitions",
        ha="center",
        fontsize=10.5,
        color="#475569",
    )
    figure.text(
        0.5,
        0.025,
        "Hollow purple diamonds appear only for SNR-related pairs; they show the sensitivity after adding unclear events.",
        ha="center",
        fontsize=9.5,
        color="#475569",
    )
    figure.subplots_adjust(left=0.17, right=0.985, top=0.76, bottom=0.12, wspace=0.12)
    figure.savefig(destination, dpi=180, facecolor="white")
    plt.close(figure)


def render_real_pedagogic_board(
    analysis: dict[str, Any], rows: list[dict[str, str]], destination: Path
) -> None:
    example = select_real_rank_example(rows)
    sum_squared = sum(row["rank_difference_squared"] for row in example)
    example_rho = 1.0 - (6.0 * sum_squared) / (len(example) * (len(example) ** 2 - 1.0))
    frequency_tau_id = pair_id(("frequency_khz", "tau_ms"))
    real_4um = _lookup(analysis["marginal_correlations"], "4um", frequency_tau_id)

    figure = plt.figure(figsize=(18.0, 11.0), facecolor="#f8fafc")
    grid = figure.add_gridspec(2, 2, height_ratios=(1.18, 1.0), width_ratios=(1.65, 1.0))
    rank_axis = figure.add_subplot(grid[0, 0])
    uncertainty_axis = figure.add_subplot(grid[0, 1])
    ranking_axis = figure.add_subplot(grid[1, :])
    for axis in (rank_axis, uncertainty_axis, ranking_axis):
        axis.set_axis_off()

    figure.suptitle(
        "Spearman analysis applied to the real z8 particles2SNR data",
        fontsize=22,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.951,
        "Four parameters form six couples: amplitude P0 · frequency · tau · SNR",
        ha="center",
        fontsize=12,
        color="#475569",
    )

    rank_axis.text(0.0, 1.02, "1 · Replace real values with ranks", fontsize=16, fontweight="bold")
    rank_axis.text(
        0.0,
        0.955,
        "Five real 4 µm events selected deterministically at the 10%, 30%, 50%, 70% and 90% frequency-order positions.",
        fontsize=9.5,
        color="#475569",
    )
    rank_columns = ["Event ID", "Frequency\n(kHz)", "F rank", "Tau\n(ms)", "Tau rank", "d", "d²"]
    rank_cells = [
        [
            row["event_id"][:8],
            f"{row['frequency_khz']:.3f}",
            f"{row['frequency_rank']:.0f}",
            f"{row['tau_ms']:.4f}",
            f"{row['tau_rank']:.0f}",
            f"{row['rank_difference']:.0f}",
            f"{row['rank_difference_squared']:.0f}",
        ]
        for row in example
    ]
    table = rank_axis.table(
        cellText=rank_cells,
        colLabels=rank_columns,
        cellLoc="center",
        colLoc="center",
        bbox=(0.0, 0.37, 0.98, 0.52),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    for (row_index, _), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_facecolor("#dbeafe" if row_index == 0 else ("#f8fafc" if row_index % 2 else "#eff6ff"))
        if row_index == 0:
            cell.set_text_props(weight="bold", color="#1e3a8a")
    rank_axis.text(0.0, 0.27, "Without ties:", fontsize=10.5, fontweight="bold")
    rank_axis.text(
        0.16,
        0.27,
        "ρ = 1 − 6Σdᵢ² / n(n² − 1)",
        fontsize=15,
        fontweight="bold",
        color="#1e3a8a",
    )
    rank_axis.text(
        0.0,
        0.15,
        f"Real subset: Σdᵢ² = {sum_squared:.0f}  →  ρ = {example_rho:.2f}",
        fontsize=13,
        fontweight="bold",
    )
    rank_axis.text(
        0.0,
        0.065,
        "With ties, software computes the ordinary Pearson correlation between average-rank vectors.",
        fontsize=9.5,
        color="#475569",
    )

    uncertainty_axis.text(
        0.0, 1.02, "2 · Keep recordings together", fontsize=16, fontweight="bold"
    )
    uncertainty_axis.text(
        0.0,
        0.91,
        "1. Sample source_filename recordings with replacement\n"
        "2. Include every event from each sampled recording\n"
        "3. Recalculate Spearman ρ\n"
        "4. Repeat 5,000 times\n"
        "5. Read the 2.5% and 97.5% percentiles",
        fontsize=11,
        linespacing=1.65,
        va="top",
    )
    uncertainty_axis.add_patch(
        plt.Rectangle((0.02, 0.23), 0.94, 0.30, transform=uncertainty_axis.transAxes, color="#ecfdf5")
    )
    uncertainty_axis.text(
        0.07,
        0.45,
        "REAL 4 µm FREQUENCY ↔ TAU RESULT",
        fontsize=10,
        fontweight="bold",
        color="#166534",
    )
    uncertainty_axis.text(
        0.07,
        0.35,
        f"ρ = {real_4um['spearman_rho']:.3f}",
        fontsize=21,
        fontweight="bold",
        color="#166534",
    )
    uncertainty_axis.text(
        0.42,
        0.35,
        f"95% CI [{real_4um['ci95_lower']:.3f}, {real_4um['ci95_upper']:.3f}]",
        fontsize=14,
        fontweight="bold",
        color="#166534",
    )
    uncertainty_axis.text(
        0.07,
        0.265,
        "The negative relation is stable across source recordings.",
        fontsize=10,
        color="#166534",
    )
    uncertainty_axis.text(
        0.0,
        0.09,
        "This is association and stability—not causality.",
        fontsize=10,
        color="#be123c",
        fontweight="bold",
    )

    ranking_axis.text(
        0.0,
        1.03,
        "3 · Rank all six real parameter couples and retain the top three",
        fontsize=16,
        fontweight="bold",
    )
    ranking_axis.text(
        0.0,
        0.955,
        "Overall score = median(|ρ₂ µm|, |ρ₄ µm|, |ρ₁₀ µm|)",
        fontsize=11.5,
        color="#5b21b6",
        fontweight="bold",
    )
    ranking_columns = ["Rank", "Real parameter couple", "ρ 2 µm", "ρ 4 µm", "ρ 10 µm", "Overall score", "Decision"]
    ranking_cells = [
        [
            str(row["rank"]),
            row["pair_label"],
            f"{row['rho_2um']:+.3f}",
            f"{row['rho_4um']:+.3f}",
            f"{row['rho_10um']:+.3f}",
            f"{row['overall_score_median_absolute_rho']:.3f}",
            "KEEP" if row["selected_top_three"] else "—",
        ]
        for row in analysis["ranking"]
    ]
    ranking_table = ranking_axis.table(
        cellText=ranking_cells,
        colLabels=ranking_columns,
        cellLoc="center",
        colLoc="center",
        bbox=(0.0, 0.13, 1.0, 0.75),
        colWidths=[0.06, 0.28, 0.11, 0.11, 0.11, 0.15, 0.10],
    )
    ranking_table.auto_set_font_size(False)
    ranking_table.set_fontsize(9.5)
    for (row_index, _), cell in ranking_table.get_celld().items():
        cell.set_edgecolor("white")
        if row_index == 0:
            cell.set_facecolor("#ede9fe")
            cell.set_text_props(weight="bold", color="#5b21b6")
        elif row_index <= 3:
            cell.set_facecolor("#ecfdf5")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#f8fafc" if row_index % 2 else "#f1f5f9")
    ranking_axis.text(
        0.0,
        0.035,
        "Top three: amplitude P0 ↔ SNR · frequency ↔ tau · tau ↔ SNR. No synthetic-generation rule is decided here.",
        fontsize=10,
        color="#475569",
    )

    figure.subplots_adjust(left=0.045, right=0.97, top=0.90, bottom=0.045, hspace=0.25, wspace=0.10)
    figure.savefig(destination, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)
