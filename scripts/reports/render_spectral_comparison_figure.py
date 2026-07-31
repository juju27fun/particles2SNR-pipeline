#!/usr/bin/env python3
"""Re-render a compact spectral-comparison figure from preserved summaries."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from particles2snr.repo_paths import MONOREPO_ROOT, REPO_ROOT
from particles2snr.spectral_comparison_figure import render_spectral_comparison_figure


DEFAULT_SOURCE_DIR = (
    MONOREPO_ROOT
    / "artifacts"
    / "particles2SNR-pipeline"
    / "reports"
    / "yolo_detection_spectral_comparison"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-name", default="4um")
    parser.add_argument("--first-pipeline", default="old")
    parser.add_argument("--second-pipeline", default="c1_particles2SNR")
    parser.add_argument("--first-label", default="initial")
    parser.add_argument("--second-label", default="particles2SNR")
    parser.add_argument("--output-stem", default="spectral_comparison_4um")
    parser.add_argument("--run-id", default="spectral-comparison-initial-vs-particles2snr")
    return parser


def git_revision() -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_png = args.output_dir / f"{args.output_stem}.png"
    output_pdf = args.output_dir / f"{args.output_stem}.pdf"

    band_summary = args.source_dir / "yolo_spectral_band_summary.csv"
    overlap_summary = args.source_dir / "yolo_overlap_summary.csv"
    coverage_summary = args.source_dir / "yolo_label_coverage_summary.csv"
    summary = render_spectral_comparison_figure(
        band_summary=band_summary,
        overlap_summary=overlap_summary,
        coverage_summary=coverage_summary,
        output_png=output_png,
        output_pdf=output_pdf,
        class_name=args.class_name,
        pipeline_keys=(args.first_pipeline, args.second_pipeline),
        display_names=(args.first_label, args.second_label),
    )

    summary_path = args.output_dir / "figure_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    command = " ".join(
        (
            "particles2SNR-pipeline/scripts/reports/render_spectral_comparison_figure.py",
            f"--source-dir {args.source_dir}",
            f"--output-dir {args.output_dir}",
            f"--class-name {args.class_name}",
            f"--first-label {args.first_label}",
            f"--second-label {args.second_label}",
        )
    )
    manifest = {
        "schema_version": 1,
        "run_id": args.run_id,
        "project": "particles2SNR-pipeline",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "dataset": ["yolo-v3-source-named@v1", "particles2snr-f-c1-yolo-3class@v1"],
        "command": command,
        "source_artifacts": [
            str(band_summary.relative_to(MONOREPO_ROOT)),
            str(overlap_summary.relative_to(MONOREPO_ROOT)),
            str(coverage_summary.relative_to(MONOREPO_ROOT)),
        ],
        "outputs": [output_png.name, output_pdf.name, summary_path.name],
        "repositories": {"particles2SNR-pipeline": git_revision()},
    }
    (args.output_dir / "run.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output_png)
    print(output_pdf)


if __name__ == "__main__":
    main()
