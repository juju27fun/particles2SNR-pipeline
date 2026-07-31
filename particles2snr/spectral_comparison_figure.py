"""Render the compact spectral-comparison figure from preserved report CSVs."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BANDS = (
    ("0-1 kHz", "0-1"),
    ("1-7 kHz", "1-7"),
    ("7-10 kHz", "7-10"),
    ("10-40 kHz", "10-40"),
    ("40-80 kHz", "40-80"),
    ("80 kHz-Nyq", "80-Nyq"),
)
REFERENCE_KEY = "noise_reference"
REFERENCE_LABEL = "Standalone Noise reference"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _band_values(
    rows: list[dict[str, str]],
    *,
    pipeline: str,
    class_name: str,
    source: str,
) -> list[float]:
    values = {
        row["band"]: float(row["mean_pct"])
        for row in rows
        if row["pipeline"] == pipeline
        and row["class"] == class_name
        and row["source"] == source
    }
    missing = [band for band, _ in BANDS if band not in values]
    if missing:
        raise ValueError(
            f"Missing spectral rows for pipeline={pipeline!r}, class={class_name!r}, "
            f"source={source!r}: {missing}"
        )
    return [values[band] for band, _ in BANDS]


def _target_band_value(
    rows: list[dict[str, str]], pipeline: str, class_name: str, field: str
) -> float:
    for row in rows:
        if row["pipeline"] == pipeline and row["class"] == class_name:
            return float(row[field])
    raise ValueError(f"Missing overlap row for pipeline={pipeline!r}, class={class_name!r}")


def _coverage_value(rows: list[dict[str, str]], pipeline: str) -> float:
    values = [float(row["coverage_pct_mean"]) for row in rows if row["pipeline"] == pipeline]
    if not values:
        raise ValueError(f"Missing coverage row for pipeline={pipeline!r}")
    return float(np.mean(values))


def _format_pct(value: float) -> str:
    return f"{value:.1f}%" if math.isfinite(value) else "NaN"


def render_spectral_comparison_figure(
    *,
    band_summary: Path,
    overlap_summary: Path,
    coverage_summary: Path,
    output_png: Path,
    output_pdf: Path | None = None,
    class_name: str = "4um",
    pipeline_keys: tuple[str, str] = ("old", "c1_particles2SNR"),
    display_names: tuple[str, str] = ("initial", "particles2SNR"),
    dpi: int = 110,
) -> dict[str, object]:
    """Recreate the compact three-panel comparison while relabelling datasets."""

    if len(set(pipeline_keys)) != 2:
        raise ValueError("pipeline_keys must contain two distinct keys")
    if len(display_names) != 2 or not all(display_names):
        raise ValueError("display_names must contain two non-empty labels")

    band_rows = _read_csv(band_summary)
    overlap_rows = _read_csv(overlap_summary)
    coverage_rows = _read_csv(coverage_summary)

    labels = [short for _, short in BANDS]
    x = np.arange(len(labels))
    colors = ("#4c72b0", "#dd8452")
    bar_width = 0.35

    fig = plt.figure(figsize=(11, 7.5))
    grid = fig.add_gridspec(
        4,
        1,
        height_ratios=(0.9, 1.08, 1.08, 0.52),
        hspace=0.48,
    )
    axes = [fig.add_subplot(grid[index]) for index in range(3)]
    table_ax = fig.add_subplot(grid[3])
    table_ax.axis("off")

    reference_values = _band_values(
        band_rows,
        pipeline=REFERENCE_KEY,
        class_name=class_name,
        source="standalone_noise_folder",
    )
    axes[0].axvspan(2.5, 3.5, color="#f2d16b", alpha=0.22, zorder=0)
    axes[0].bar(x, reference_values, width=0.58, label=REFERENCE_LABEL, color="#6f777d")
    axes[0].set_title(REFERENCE_LABEL, fontsize=10.5)
    axes[0].set_ylabel("% total energy")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(loc="upper right")

    for ax, source, title in (
        (axes[1], "particle_labels", "Particle-labelled intervals"),
        (axes[2], "dataset_noise_windows", "Dataset noise windows outside labels"),
    ):
        ax.axvspan(2.5, 3.5, color="#f2d16b", alpha=0.22, zorder=0)
        for index, (pipeline, display_name) in enumerate(zip(pipeline_keys, display_names)):
            values = _band_values(
                band_rows,
                pipeline=pipeline,
                class_name=class_name,
                source=source,
            )
            offset = (index - 0.5) * bar_width
            ax.bar(
                x + offset,
                values,
                bar_width,
                label=display_name,
                color=colors[index],
            )
        ax.set_title(title, fontsize=10.5)
        ax.set_ylabel("% total energy")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper right")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Frequency band (kHz)")

    shared_ymax = max(float(ax.dataLim.ymax) for ax in axes) * 1.05
    for ax in axes:
        ax.set_ylim(0.0, shared_ymax)
        ax.set_yticks(np.arange(0.0, 81.0, 20.0))

    table_rows = []
    for pipeline, display_name in zip(pipeline_keys, display_names):
        table_rows.append(
            [
                display_name,
                _format_pct(
                    _target_band_value(
                        overlap_rows,
                        pipeline,
                        class_name,
                        "particle_label_doppler_band_energy_pct",
                    )
                ),
                _format_pct(
                    _target_band_value(
                        overlap_rows,
                        pipeline,
                        class_name,
                        "dataset_noise_doppler_band_energy_pct",
                    )
                ),
                _format_pct(
                    _target_band_value(
                        overlap_rows,
                        pipeline,
                        class_name,
                        "standalone_noise_doppler_band_energy_pct",
                    )
                ),
                _format_pct(_coverage_value(coverage_rows, pipeline)),
            ]
        )

    table = table_ax.table(
        cellText=table_rows,
        colLabels=(
            "Dataset",
            "Particle 10-40 kHz",
            "Dataset-noise 10-40 kHz",
            "Noise ref. 10-40 kHz",
            "Mean label coverage",
        ),
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.1)
    fig.subplots_adjust(left=0.06, right=0.993, top=0.953, bottom=0.01)
    table_position = table_ax.get_position()
    table_ax.set_position(
        [
            table_position.x0,
            table_position.y0 - 0.006,
            table_position.width,
            table_position.height,
        ]
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi)
    if output_pdf is not None:
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_pdf)
    plt.close(fig)

    return {
        "class": class_name,
        "pipeline_keys": list(pipeline_keys),
        "display_names": list(display_names),
        "outputs": [str(output_png)] + ([str(output_pdf)] if output_pdf is not None else []),
    }
