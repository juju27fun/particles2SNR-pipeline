#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from particles2snr.yeast_events import YeastDetectionConfig
from particles2snr.yeast_review_calibration import (
    calibration_spec_from_row,
    count_proxy_metrics,
    detect_segmentation_variants,
    evaluate_calibration_spec,
    select_development_variant,
    sweep_count_calibration,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def candidate_diagnostics(review_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output = []
    for row in review_rows:
        if int(row["candidate_index"]) < 0:
            continue
        output.append(
            {
                "event_id": row["event_id"],
                "source_group": row["source_group"],
                "quality": row["quality"],
                "rejection_reason": row["rejection_reason"],
                "event_present": row["review_event_present"],
                "center_acceptable": row["review_center_acceptable"],
                "full_event_visible": row["review_full_event_visible"],
                "artifact": row["review_artifact"],
                "snr_proxy": float(row["snr_proxy"]),
                "energy_concentration": float(row["energy_concentration"]),
                "width_ms": float(row["width_ms"]),
            }
        )
    return output


def tier_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for quality in ("strict", "medium", "reject"):
        selected = [row for row in rows if row["quality"] == quality]
        present = [row for row in selected if row["event_present"] == "yes"]
        absent = [row for row in selected if row["event_present"] == "no"]
        output.append(
            {
                "quality": quality,
                "n_reviewed": len(selected),
                "n_event_present": len(present),
                "event_presence_fraction": len(present) / len(selected) if selected else None,
                "event_snr_median": float(np.median([row["snr_proxy"] for row in present])) if present else None,
                "non_event_snr_median": float(np.median([row["snr_proxy"] for row in absent])) if absent else None,
                "rejection_reasons": json.dumps(Counter(row["rejection_reason"] for row in selected), sort_keys=True),
            }
        )
    return output


def annotation_consistency_rows(file_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output = []
    for index, row in enumerate(file_rows):
        rejected = int(row["n_rejected_candidates"])
        true_rejected = int(row["review_true_rejected_candidate_count"])
        missed = int(row["review_missed_event_count"])
        if rejected > 0 and true_rejected == 0 and missed > 0:
            output.append(
                {
                    "human_trace_number": index + 1,
                    "record_id": row["record_id"],
                    "source_group": row["source_group"],
                    "n_rejected_candidates": rejected,
                    "true_rejected": true_rejected,
                    "missed": missed,
                    "review_notes": row["review_notes"],
                    "flag": "rejected candidate may have been counted as missed; verify semantics",
                }
            )
    return output


def quality_false_positive_bounds(
    candidate_by_id: dict[str, dict[str, str]], file_rows: list[dict[str, str]]
) -> list[dict[str, Any]]:
    counts = {"strict": 0, "medium": 0}
    lower = {"strict": 0, "medium": 0}
    upper = {"strict": 0, "medium": 0}
    for row in file_rows:
        event_ids = json.loads(row["detected_event_ids"])
        qualities = [
            candidate_by_id[event_id]["quality"]
            for event_id in event_ids
            if candidate_by_id[event_id]["quality"] in counts
        ]
        false_retained = int(row["review_false_retained_candidate_count"])
        for quality in counts:
            n_quality = qualities.count(quality)
            n_other = len(qualities) - n_quality
            counts[quality] += n_quality
            lower[quality] += max(0, false_retained - n_other)
            upper[quality] += min(n_quality, false_retained)
    return [
        {
            "quality": quality,
            "n_retained_candidates": counts[quality],
            "false_positive_lower_bound": lower[quality],
            "false_positive_upper_bound": upper[quality],
            "precision_lower_bound": (counts[quality] - upper[quality]) / counts[quality],
            "precision_upper_bound": (counts[quality] - lower[quality]) / counts[quality],
        }
        for quality in ("strict", "medium")
    ]


def plot_candidate_snr(rows: list[dict[str, Any]], output: Path) -> None:
    qualities = ("medium", "strict", "reject")
    colors = {"yes": "#16876b", "no": "#b5402f"}
    figure, axis = plt.subplots(figsize=(8.6, 5.2), constrained_layout=True)
    for quality_index, quality in enumerate(qualities):
        for truth_index, truth in enumerate(("no", "yes")):
            selected = [
                row for row in rows if row["quality"] == quality and row["event_present"] == truth
            ]
            if not selected:
                continue
            offset = -0.12 if truth_index == 0 else 0.12
            axis.scatter(
                np.full(len(selected), quality_index + offset),
                [row["snr_proxy"] for row in selected],
                color=colors[truth],
                alpha=0.8,
                s=28,
                label=("event" if truth == "yes" else "non-event") if quality_index == 0 else None,
            )
    axis.axhline(5.0, color="#555555", linestyle="--", linewidth=1.0, label="current strict SNR boundary")
    axis.set_yscale("log")
    axis.set_xticks(range(len(qualities)), qualities)
    axis.set_ylabel("Detector SNR proxy (log scale)")
    axis.set_title("Reviewed candidate truth by detector quality tier")
    axis.grid(axis="y", color="#d5d9dc", linewidth=0.45, alpha=0.7)
    axis.legend(frameon=False, ncol=3)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_tradeoff(
    rows: list[dict[str, Any]],
    baseline: dict[str, Any],
    selected: dict[str, Any] | None,
    output: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 5.5), constrained_layout=True)
    scatter = axis.scatter(
        [float(row["recall_count_proxy"]) for row in rows],
        [float(row["precision_count_proxy"]) for row in rows],
        c=[float(row["acceptance_snr_z"]) for row in rows],
        cmap="viridis",
        s=18,
        alpha=0.35,
    )
    axis.scatter(
        baseline["recall_count_proxy"],
        baseline["precision_count_proxy"],
        marker="*",
        s=150,
        color="#202428",
        label="current detector",
        zorder=4,
    )
    if selected is not None:
        axis.scatter(
            selected["recall_count_proxy"],
            selected["precision_count_proxy"],
            marker="D",
            s=75,
            color="#c94b32",
            label="development selection",
            zorder=5,
        )
    axis.axhline(0.90, color="#777777", linestyle="--", linewidth=0.8)
    axis.axvline(0.85, color="#777777", linestyle="--", linewidth=0.8)
    axis.set_xlim(0.0, 1.02)
    axis.set_ylim(0.0, 1.02)
    axis.set_xlabel("Recall count proxy")
    axis.set_ylabel("Precision count proxy")
    axis.set_title("Development-only detector calibration sweep")
    axis.grid(color="#d5d9dc", linewidth=0.4, alpha=0.6)
    axis.legend(frameon=False)
    figure.colorbar(scatter, ax=axis, label="Acceptance SNR proxy threshold")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def build_report(
    tier_rows: list[dict[str, Any]],
    bounds: list[dict[str, Any]],
    consistency: list[dict[str, Any]],
    official: dict[str, Any],
    baseline: dict[str, Any],
    selected: dict[str, Any] | None,
    group_metrics: list[dict[str, Any]],
    leave_one_group_out: list[dict[str, Any]],
) -> str:
    candidate = official["candidate_review"]
    full = official["full_trace_review"]
    lines = [
        "# Yeast detector review calibration",
        "",
        "## Status",
        "",
        f"- Gate 1 review status: `{official['event_review_status']}`.",
        f"- Retained candidate precision (balanced review): `{candidate['retained_candidate_precision_balanced']['value']:.3f}`.",
        f"- Full-trace precision cross-check: `{full['event_precision_cross_check']['value']:.3f}`.",
        f"- Full-trace recall: `{full['event_recall_balanced']['value']:.3f}`.",
        f"- Reviewed event counts: TP `{full['event_counts']['true_positive']}`, FP `{full['event_counts']['false_positive']}`, FN `{full['event_counts']['false_negative']}`.",
        "",
        "## Candidate-level SNR evidence",
        "",
        "| quality | event present | reviewed | event SNR median | non-event SNR median |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in tier_rows:
        event_median = "NA" if row["event_snr_median"] is None else f"{row['event_snr_median']:.3g}"
        non_event_median = "NA" if row["non_event_snr_median"] is None else f"{row['non_event_snr_median']:.3g}"
        lines.append(
            f"| {row['quality']} | {row['n_event_present']} | {row['n_reviewed']} | {event_median} | {non_event_median} |"
        )
    lines.extend(
        [
            "",
            "The medium tier is close to chance in this balanced sample, and its event/non-event SNR distributions overlap. The strict tier is much cleaner in the candidate-window review, but full-trace aggregate counts prove that SNR alone cannot explain all false positives.",
            "",
            "## Full-trace quality bounds",
            "",
            "The aggregate full-trace form records how many retained candidates are false, but not which numbered candidate is false. Therefore per-tier precision is bounded rather than identified.",
            "",
            "| quality | candidates | FP lower | FP upper | precision interval |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in bounds:
        lines.append(
            f"| {row['quality']} | {row['n_retained_candidates']} | {row['false_positive_lower_bound']} | {row['false_positive_upper_bound']} | {row['precision_lower_bound']:.3f}-{row['precision_upper_bound']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"`{len(consistency)}` full-trace rows require a semantic check because a rejected candidate may have been counted under missed events. This does not change total FN or recall, but it matters for diagnosing the width rejection rule.",
            "",
            "## Calibration sweep",
            "",
            "This sweep re-runs detection on the 73 reviewed traces and compares only event counts. It cannot verify localization or one-to-one event matching; its metrics are development proxies, not final validation.",
            "",
            f"Current count proxy: precision `{baseline['precision_count_proxy']:.3f}`, recall `{baseline['recall_count_proxy']:.3f}`, exact-count fraction `{baseline['exact_count_fraction']:.3f}`.",
        ]
    )
    if selected is None:
        lines.append("No swept variant reached both development proxy targets (precision >= 0.90 and recall >= 0.85).")
    else:
        lines.extend(
            [
                "",
                "Selected development variant (must be revalidated on a fresh queue):",
                "",
                f"- boundary SNR z: `{selected['boundary_snr_z']}`",
                f"- cluster gap: `{selected['cluster_gap_ms']} ms`",
                f"- acceptance SNR z: `{selected['acceptance_snr_z']}`",
                f"- maximum width: `{selected['maximum_width_ms']} ms`",
                f"- maximum events: `{selected['maximum_events']}` (`0` means unlimited)",
                f"- count-proxy precision/recall/F1: `{selected['precision_count_proxy']:.3f}` / `{selected['recall_count_proxy']:.3f}` / `{selected['f1_count_proxy']:.3f}`",
            ]
        )
    lines.extend(
        [
            "",
            "### Group stability",
            "",
            "| group | precision proxy | recall proxy | exact-count fraction |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in group_metrics:
        lines.append(
            f"| {row['source_group']} | {row['precision_count_proxy']:.3f} | {row['recall_count_proxy']:.3f} | {row['exact_count_fraction']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Leave-one-source-group-out chooses a variant on three groups and evaluates it on the fourth. It is still same-acquisition development evidence, but it exposes group-specific overfitting.",
            "",
            "| held-out group | selected SNR | width | gap | held-out precision | held-out recall |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in leave_one_group_out:
        if row["selection_status"] != "available":
            lines.append(f"| {row['held_out_group']} | NA | NA | NA | NA | NA |")
        else:
            lines.append(
                f"| {row['held_out_group']} | {row['acceptance_snr_z']} | {row['maximum_width_ms']} | {row['cluster_gap_ms']} | {row['held_out_precision_count_proxy']:.3f} | {row['held_out_recall_count_proxy']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Do not promote the current detector. Treat the completed review as calibration data, preserve the raw annotations, and validate any revised detector on a newly sampled queue. A second independent acquisition remains mandatory for sealed acquisition-OOD evaluation.",
            "",
            "![Reviewed candidate SNR](candidate_truth_vs_snr.png)",
            "",
            "![Calibration tradeoff](calibration_tradeoff.png)",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate the yeast detector from completed development reviews.")
    parser.add_argument("--candidate-dataset", type=Path, required=True)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--review-analysis", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    candidate_rows = read_csv(args.candidate_dataset / "candidate_events.csv")
    review_rows = read_csv(args.review_dir / "manual_review_queue.csv")
    file_rows = read_csv(args.review_dir / "manual_file_review_queue.csv")
    if any(not row["reviewer"].strip() for row in review_rows + file_rows):
        raise ValueError("Both review queues must be complete before calibration")
    official = json.loads(args.review_analysis.read_text(encoding="utf-8"))
    with np.load(args.candidate_dataset / "manual_file_review_signals.npz") as payload:
        signals_by_record = {
            str(record_id): np.asarray(signal, dtype=np.float32)
            for record_id, signal in zip(payload["record_id"].tolist(), payload["signals"])
        }
    true_counts = {row["record_id"]: int(row["review_true_event_count"]) for row in file_rows}
    if set(signals_by_record) != set(true_counts):
        raise ValueError("Reviewed full-trace signals and annotation rows do not match")

    diagnostics = candidate_diagnostics(review_rows)
    tiers = tier_summary(diagnostics)
    consistency = annotation_consistency_rows(file_rows)
    candidate_by_id = {row["event_id"]: row for row in candidate_rows}
    bounds = quality_false_positive_bounds(candidate_by_id, file_rows)
    baseline = count_proxy_metrics(
        [int(row["n_retained_candidates"]) for row in file_rows],
        [true_counts[row["record_id"]] for row in file_rows],
    )
    detections = detect_segmentation_variants(
        signals_by_record,
        YeastDetectionConfig(),
        boundary_values=(1.5, 2.0, 2.5, 3.0),
        cluster_gap_values=(0.0, 0.064, 0.128, 0.25),
    )
    sweep = sweep_count_calibration(
        detections,
        true_counts,
        acceptance_snr_values=(3.5, 4.0, 5.0, 8.0, 12.0),
        maximum_width_values=(1.6, 2.0, 3.2, 8.2),
        maximum_event_values=(3, 5, 0),
    )
    selected = select_development_variant(sweep)

    groups_by_record = {row["record_id"]: row["source_group"] for row in file_rows}
    group_metrics: list[dict[str, Any]] = []
    leave_one_group_out: list[dict[str, Any]] = []
    if selected is not None:
        selected_spec = calibration_spec_from_row(selected)
        for group in sorted(set(groups_by_record.values())):
            group_ids = [
                record_id for record_id, source_group in groups_by_record.items() if source_group == group
            ]
            group_metrics.append(
                {
                    "source_group": group,
                    **evaluate_calibration_spec(
                        detections, true_counts, selected_spec, record_ids=group_ids
                    ),
                }
            )

    specs = [calibration_spec_from_row(row) for row in sweep]
    for held_out_group in sorted(set(groups_by_record.values())):
        train_ids = [
            record_id
            for record_id, source_group in groups_by_record.items()
            if source_group != held_out_group
        ]
        held_out_ids = [
            record_id
            for record_id, source_group in groups_by_record.items()
            if source_group == held_out_group
        ]
        train_rows = [
            {
                **row,
                **evaluate_calibration_spec(
                    detections, true_counts, spec, record_ids=train_ids
                ),
            }
            for row, spec in zip(sweep, specs)
        ]
        train_selected = select_development_variant(train_rows)
        if train_selected is None:
            leave_one_group_out.append(
                {"held_out_group": held_out_group, "selection_status": "unavailable"}
            )
            continue
        held_out_metrics = evaluate_calibration_spec(
            detections,
            true_counts,
            calibration_spec_from_row(train_selected),
            record_ids=held_out_ids,
        )
        leave_one_group_out.append(
            {
                "held_out_group": held_out_group,
                "selection_status": "available",
                **{
                    key: train_selected[key]
                    for key in (
                        "boundary_snr_z",
                        "cluster_gap_ms",
                        "acceptance_snr_z",
                        "maximum_width_ms",
                        "maximum_events",
                        "minimum_concentration",
                    )
                },
                **{f"held_out_{key}": value for key, value in held_out_metrics.items()},
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_dir / "candidate_snr_diagnostics.csv", diagnostics)
    write_csv(args.output_dir / "candidate_tier_summary.csv", tiers)
    write_csv(args.output_dir / "full_trace_quality_bounds.csv", bounds)
    if consistency:
        write_csv(args.output_dir / "annotation_consistency_flags.csv", consistency)
    write_csv(args.output_dir / "calibration_sweep.csv", sweep)
    if group_metrics:
        write_csv(args.output_dir / "selected_variant_group_metrics.csv", group_metrics)
    write_csv(args.output_dir / "leave_one_group_out.csv", leave_one_group_out)
    summary = {
        "schema_version": 1,
        "scientific_role": "development calibration only; not independent validation",
        "official_review_status": official["event_review_status"],
        "official_gate_1_status": official["gate_1_status"],
        "candidate_tier_summary": tiers,
        "full_trace_quality_bounds": bounds,
        "annotation_consistency_flags": consistency,
        "baseline_count_proxy": baseline,
        "selected_development_variant": selected,
        "selected_variant_group_metrics": group_metrics,
        "leave_one_group_out": leave_one_group_out,
        "n_sweep_variants": len(sweep),
        "count_proxy_limitation": "Counts do not establish localization or one-to-one event matching.",
    }
    (args.output_dir / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plot_candidate_snr(diagnostics, args.output_dir / "candidate_truth_vs_snr.png")
    plot_tradeoff(sweep, baseline, selected, args.output_dir / "calibration_tradeoff.png")
    (args.output_dir / "REPORT.md").write_text(
        build_report(
            tiers,
            bounds,
            consistency,
            official,
            baseline,
            selected,
            group_metrics,
            leave_one_group_out,
        ),
        encoding="utf-8",
    )
    repo_root = Path(__file__).resolve().parents[2]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True
    ).stdout.strip()
    outputs = [
        "REPORT.md",
        "calibration_summary.json",
        "candidate_snr_diagnostics.csv",
        "candidate_tier_summary.csv",
        "full_trace_quality_bounds.csv",
        "calibration_sweep.csv",
        "leave_one_group_out.csv",
        "candidate_truth_vs_snr.png",
        "calibration_tradeoff.png",
    ]
    if consistency:
        outputs.append("annotation_consistency_flags.csv")
    if group_metrics:
        outputs.append("selected_variant_group_metrics.csv")
    run = {
        "schema_version": 1,
        "project": "particles2SNR-pipeline",
        "run_id": args.run_id,
        "dataset": args.dataset_id,
        "repositories": {"particles2SNR-pipeline": revision},
        "command": " ".join(sys.argv),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "outputs": outputs,
    }
    (args.output_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
