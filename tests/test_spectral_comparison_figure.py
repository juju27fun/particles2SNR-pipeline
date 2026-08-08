from __future__ import annotations

import csv
from pathlib import Path

from particles2snr.spectral_comparison_figure import BANDS, render_spectral_comparison_figure


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_render_spectral_comparison_figure(tmp_path: Path) -> None:
    band_summary = tmp_path / "bands.csv"
    band_rows = []
    for pipeline, source in (
        ("old", "particle_labels"),
        ("old", "dataset_noise_windows"),
        ("new", "particle_labels"),
        ("new", "dataset_noise_windows"),
        ("noise_reference", "standalone_noise_folder"),
    ):
        for index, (band, _short) in enumerate(BANDS):
            band_rows.append(
                {
                    "pipeline": pipeline,
                    "class": "4um",
                    "source": source,
                    "band": band,
                    "mean_pct": index + 1,
                }
            )
    _write_csv(
        band_summary,
        ("pipeline", "class", "source", "band", "mean_pct"),
        band_rows,
    )

    overlap_summary = tmp_path / "overlap.csv"
    _write_csv(
        overlap_summary,
        (
            "pipeline",
            "class",
            "particle_label_doppler_band_energy_pct",
            "dataset_noise_doppler_band_energy_pct",
            "standalone_noise_doppler_band_energy_pct",
        ),
        [
            {
                "pipeline": pipeline,
                "class": "4um",
                "particle_label_doppler_band_energy_pct": 50,
                "dataset_noise_doppler_band_energy_pct": 10,
                "standalone_noise_doppler_band_energy_pct": 20,
            }
            for pipeline in ("old", "new")
        ],
    )
    coverage_summary = tmp_path / "coverage.csv"
    _write_csv(
        coverage_summary,
        ("pipeline", "coverage_pct_mean"),
        [
            {"pipeline": "old", "coverage_pct_mean": 7},
            {"pipeline": "new", "coverage_pct_mean": 67},
        ],
    )

    output_png = tmp_path / "figure.png"
    output_pdf = tmp_path / "figure.pdf"
    summary = render_spectral_comparison_figure(
        band_summary=band_summary,
        overlap_summary=overlap_summary,
        coverage_summary=coverage_summary,
        output_png=output_png,
        output_pdf=output_pdf,
        pipeline_keys=("old", "new"),
        display_names=("initial", "particles2SNR"),
    )

    assert summary["display_names"] == ["initial", "particles2SNR"]
    assert output_png.stat().st_size > 0
    assert output_pdf.stat().st_size > 0
