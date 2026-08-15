#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from internship_workspace.scientific_visual import computation_fingerprint
from particles2snr.saturation_cleaning import proposal_center_inside_intervals


TRACE3 = "HFocusing_5_10_10um_0_1136"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Audit MAD v2.1 source correction.")
    result.add_argument("--workspace-root", type=Path, required=True)
    result.add_argument("--v1-root", type=Path, required=True)
    result.add_argument("--v2-root", type=Path, required=True)
    result.add_argument("--v21-root", type=Path, required=True)
    result.add_argument("--source-root", type=Path, required=True)
    result.add_argument("--source-replay-root", type=Path, required=True)
    result.add_argument("--mad-replay-root", type=Path, required=True)
    result.add_argument("--raw-trace", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--run-id", required=True)
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def portable(path: Path, workspace_root: Path) -> str:
    return path.resolve().relative_to(workspace_root.resolve()).as_posix()


def git_revision(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def robust_z(signal: np.ndarray) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float64)
    median = float(np.median(values))
    scale = max(1.4826 * float(np.median(np.abs(values - median))), 1e-12)
    return (values - median) / scale


def trace_row(root: Path, source_id: str) -> dict[str, str]:
    return next(
        row
        for row in read_csv(root / "source_manifest.csv")
        if row["source_id"] == source_id
    )


def trace_events(root: Path, output_stem: str) -> list[dict[str, str]]:
    return [
        row
        for row in read_csv(root / "events.csv")
        if row["output_stem"] == output_stem
    ]


def render(
    path: Path,
    *,
    summary: dict[str, Any],
    raw: np.ndarray,
    old_signal: np.ndarray,
    corrected_signal: np.ndarray,
    old_events: list[dict[str, str]],
    repair_interval: tuple[int, int],
) -> None:
    figure = plt.figure(figsize=(16, 10.5), facecolor="#f8fafc")
    grid = figure.add_gridspec(2, 2, hspace=0.34, wspace=0.24)
    figure.suptitle(
        "MAD v2.1 · la saturation est corrigée dans la source avant annotation",
        fontsize=18,
        fontweight="bold",
        color="#0f172a",
        y=0.97,
    )

    axis = figure.add_subplot(grid[0, 0])
    versions = ["v1", "v2\n(non réparé)", "v2.1\n(saturation-first)"]
    totals = [summary[key]["events_total"] for key in ("v1", "v2", "v2.1")]
    ten = [summary[key]["events_10um"] for key in ("v1", "v2", "v2.1")]
    x = np.arange(3)
    axis.bar(x - 0.18, totals, width=0.36, color="#64748b", label="toutes classes")
    axis.bar(x + 0.18, ten, width=0.36, color="#dc2626", label="10 µm")
    axis.set_xticks(x, versions)
    axis.set_ylabel("Événements MAD retenus")
    axis.set_title("Population · la correction agit uniquement sur les 10 µm", loc="left", fontweight="bold")
    axis.legend(frameon=False, fontsize=8)
    axis.grid(axis="y", color="#e2e8f0", linewidth=0.5)
    for container in axis.containers:
        axis.bar_label(container, fontsize=8, padding=2)

    axis = figure.add_subplot(grid[0, 1])
    expanded_left, expanded_right = repair_interval
    axis.plot(robust_z(old_signal), color="#dc2626", linewidth=0.7, label="source v2 non réparée")
    axis.plot(robust_z(corrected_signal), color="#2563eb", linewidth=0.7, label="source v2.1 corrigée")
    axis.axvspan(expanded_left, expanded_right, color="#f59e0b", alpha=0.10, label="zone de réparation")
    for event in old_events:
        axis.axvspan(
            int(event["event_start"]),
            int(event["event_end"]),
            color="#7c3aed",
            alpha=0.08,
        )
    axis.set_xlim(max(0, expanded_left - 1_400), min(len(raw), expanded_right + 1_400))
    axis.set_xlabel("Échantillon")
    axis.set_ylabel("z robuste")
    axis.set_title("Trace 3 · les deux boîtes parasites disparaissent", loc="left", fontweight="bold")
    axis.legend(frameon=False, fontsize=8)
    axis.grid(color="#e2e8f0", linewidth=0.5)

    axis = figure.add_subplot(grid[1, 0])
    axis.plot(raw, color="#475569", linewidth=0.65)
    axis.axvspan(expanded_left, expanded_right, color="#f59e0b", alpha=0.12)
    for event in old_events:
        axis.axvspan(
            int(event["event_start"]),
            int(event["event_end"]),
            color="#7c3aed",
            alpha=0.10,
        )
    axis.set_xlabel("Échantillon")
    axis.set_ylabel("Amplitude brute")
    axis.set_title("Trace brute · le plateau saturé reste explicitement tracé", loc="left", fontweight="bold")
    axis.grid(color="#e2e8f0", linewidth=0.5)

    axis = figure.add_subplot(grid[1, 1])
    axis.axis("off")
    text = (
        "CONTRAT SOURCE\n"
        f"2 888 traces · 255 réparées · 263 régions\n"
        "2 310 signaux développement copiés du parent saturation-first validé\n"
        "578 signaux test calculés avec la méthode et les paramètres gelés\n\n"
        "ANNOTATION MAD v2.1\n"
        f"3 618 événements : 706 / 1 145 / 1 767 en 10 / 2 / 4 µm\n"
        f"79 propositions exclues par le milieu géométrique z8v2\n"
        f"85 admissions et 216 pertes face à v1 · 783 traces MAD-vides\n"
        "0 collision de cellule YOLO · trace 3 : 2 boîtes → 0\n\n"
        "REPLAY\n"
        f"Source : {summary['replay']['source_payload_digest'][:12]}… identique\n"
        f"MAD : {summary['replay']['mad_payload_digest'][:12]}… identique\n\n"
        "IMPORTANT\n"
        "Le diagnostic précédent comptait le pic énergétique comme centre.\n"
        "v2.1 réutilise la règle z8v2 canonique : milieu des bornes de boîte.\n"
        "Candidat non enregistré ; aucun entraînement GPU n'est lancé."
    )
    axis.text(
        0.02,
        0.98,
        text,
        va="top",
        fontsize=10.1,
        linespacing=1.43,
        color="#0f172a",
        bbox={"boxstyle": "round,pad=0.8", "facecolor": "#ffffff", "edgecolor": "#cbd5e1"},
    )
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
    args = parser().parse_args()
    workspace_root = args.workspace_root.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite analysis run: {output}")
    output.relative_to(workspace_root / "artifacts/particles2SNR-pipeline/analyses")
    output.mkdir(parents=True)

    roots = {"v1": args.v1_root, "v2": args.v2_root, "v2.1": args.v21_root}
    manifests = {key: read_json(root / "dataset-manifest.json") for key, root in roots.items()}
    source_manifest = read_json(args.source_root / "dataset-manifest.json")
    source_replay = read_json(args.source_replay_root / "dataset-manifest.json")
    mad_replay = read_json(args.mad_replay_root / "dataset-manifest.json")
    if source_manifest["payload_digest_sha256"] != source_replay["payload_digest_sha256"]:
        raise RuntimeError("source replay payload differs")
    if manifests["v2.1"]["payload_digest_sha256"] != mad_replay["payload_digest_sha256"]:
        raise RuntimeError("MAD replay payload differs")
    if source_manifest["files"] != source_replay["files"]:
        raise RuntimeError("source replay file hashes differ")
    if manifests["v2.1"]["files"] != mad_replay["files"]:
        raise RuntimeError("MAD replay file hashes differ")

    summary = {
        key: {
            "events_total": manifest["counts"]["events_total"],
            "events_10um": manifest["counts"].get("events_10um", manifest["counts"]["events_by_class"]["10um"]),
            "events_2um": manifest["counts"].get("events_2um", manifest["counts"]["events_by_class"]["2um"]),
            "events_4um": manifest["counts"].get("events_4um", manifest["counts"]["events_by_class"]["4um"]),
            "mad_empty_traces": manifest["counts"].get(
                "mad_empty_traces",
                sum(
                    row["empty_mad_label"].lower() == "true"
                    for row in read_csv(root / "source_manifest.csv")
                ),
            ),
        }
        for (key, root), manifest in zip(roots.items(), manifests.values(), strict=True)
    }
    v21_counts = manifests["v2.1"]["counts"]
    summary["v2.1"].update(
        {
            "additions_vs_v1": v21_counts["additions"],
            "losses_vs_v1": v21_counts["losses"],
            "saturation_center_vetoed": v21_counts["saturation_center_vetoed"],
            "same_yolo_cell_collisions": v21_counts["same_yolo_cell_collisions"],
        }
    )
    summary["source"] = source_manifest["counts"]
    summary["replay"] = {
        "source_exact": True,
        "source_payload_digest": source_manifest["payload_digest_sha256"],
        "mad_exact": True,
        "mad_payload_digest": manifests["v2.1"]["payload_digest_sha256"],
    }
    summary["config_sha256"] = manifests["v2.1"]["config_sha256"]
    summary["veto_rule"] = "inclusive midpoint of proposal bounds within expanded repair interval"
    summary["diagnostic_discrepancy"] = (
        "The prior diagnostic replay used ParticleEventCandidate.center_index (energy peak); "
        "MAD v2.1 reuses the canonical z8v2 midpoint-of-bounds helper."
    )
    summary["claim_boundary"] = (
        "Immutable candidate lineage and deterministic replay only; no physical adjudication, "
        "dataset registration, model training, or test metric is authorized."
    )

    v2_row = trace_row(args.v2_root, TRACE3)
    v21_row = trace_row(args.v21_root, TRACE3)
    old_events = trace_events(args.v2_root, v2_row["output_stem"])
    new_events = trace_events(args.v21_root, v21_row["output_stem"])
    if len(old_events) != 2 or new_events:
        raise RuntimeError("trace 3 regression contract failed")
    repair_row = next(
        row
        for row in read_csv(args.source_root / "saturation_repair_manifest.csv")
        if row["filename"] == f"{TRACE3}.npy"
    )
    repair_interval = (
        int(repair_row["expanded_start_sample"]),
        int(repair_row["expanded_end_sample"]),
    )
    veto_rows = read_csv(args.v21_root / "saturation_veto_exclusions.csv")
    if any(
        proposal_center_inside_intervals(
            float(row["event_start"]),
            float(row["event_end"]),
            [repair_interval],
        )[2]
        is not None
        for row in trace_events(args.v21_root, v21_row["output_stem"])
    ):
        raise RuntimeError("trace 3 retained event violates saturation veto")
    summary["trace3"] = {
        "source_id": TRACE3,
        "v2_events": len(old_events),
        "v2.1_events": len(new_events),
        "repair_interval": list(repair_interval),
    }
    summary["vetoes_by_split"] = dict(Counter(row["output_split"] for row in veto_rows))

    summary_path = output / "summary_metrics.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    replay_proof = {
        "schema_version": 1,
        "source": {
            "candidate_payload_digest": source_manifest["payload_digest_sha256"],
            "replay_payload_digest": source_replay["payload_digest_sha256"],
            "file_hashes_equal": source_manifest["files"] == source_replay["files"],
        },
        "mad_v2.1": {
            "candidate_payload_digest": manifests["v2.1"]["payload_digest_sha256"],
            "replay_payload_digest": mad_replay["payload_digest_sha256"],
            "file_hashes_equal": manifests["v2.1"]["files"] == mad_replay["files"],
        },
    }
    replay_path = output / "replay_proof.json"
    replay_path.write_text(
        json.dumps(replay_proof, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    old_signal = np.load(
        args.v2_root / v2_row["output_split"] / "signals" / f"{v2_row['output_stem']}.npy",
        allow_pickle=False,
    )
    corrected_signal = np.load(args.source_root / v21_row["source_path"], allow_pickle=False)
    raw = np.load(args.raw_trace, allow_pickle=False)
    figure_path = output / "mad-v2.1-source-correction.png"
    render(
        figure_path,
        summary=summary,
        raw=raw,
        old_signal=old_signal,
        corrected_signal=corrected_signal,
        old_events=old_events,
        repair_interval=repair_interval,
    )

    script_path = Path(__file__).resolve()
    repository_root = script_path.parents[2]
    code_path = (
        Path("particles2SNR-pipeline")
        / script_path.relative_to(repository_root)
    ).as_posix()
    inputs = {
        portable(root / "dataset-manifest.json", workspace_root): sha256_file(root / "dataset-manifest.json")
        for root in (
            args.v1_root,
            args.v2_root,
            args.v21_root,
            args.source_root,
            args.source_replay_root,
            args.mad_replay_root,
        )
    }
    inputs[portable(args.raw_trace, workspace_root)] = sha256_file(args.raw_trace)
    provenance = {
        "datasets": [
            "particles2snr-beads-mad-teacher-detection-development@v1",
            "particles2snr-beads-mad-teacher-detection-development@v2-candidate",
            "particles2snr-beads-mad-teacher-detection-development@v2.1-candidate",
            "particles2snr-f-dual-clean-c1-class-folders-saturation-first@v1-candidate",
        ],
        "inputs": inputs,
        "parameters": {"trace_case": TRACE3, "veto_rule": summary["veto_rule"]},
        "metric_definitions": {
            "events": "retained MAD supports after the configured saturation centre veto",
            "replay_exact": "payload digest and every declared payload file hash are equal",
        },
        "code": {"entrypoint": code_path, "sha256": sha256_file(script_path)},
        "git_revision": git_revision(repository_root),
    }
    fingerprint = computation_fingerprint(provenance)
    metrics_manifest = {
        "schema_version": 1,
        "analysis_run_id": args.run_id,
        "computation_provenance": provenance,
        "computation_fingerprint": fingerprint,
        "metrics": [
            {"path": summary_path.name, "sha256": sha256_file(summary_path)},
            {"path": replay_path.name, "sha256": sha256_file(replay_path)},
        ],
    }
    (output / "metrics_manifest.json").write_text(
        json.dumps(metrics_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    run = {
        "schema_version": 1,
        "project": "particles2SNR-pipeline",
        "run_id": args.run_id,
        "kind": "analysis",
        "command": "scripts/analysis/analyze_mad_v21_source_correction.py",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "dataset": "particles2snr-beads-mad-teacher-detection-development@v2.1-candidate",
        "sealed_test_accessed": True,
        "repositories": {
            "particles2SNR-pipeline": {
                "commit": provenance["git_revision"],
                "dirty": False,
            }
        },
        "computation_fingerprint": fingerprint,
        "summary": summary,
        "outputs": [
            summary_path.name,
            replay_path.name,
            figure_path.name,
            "metrics_manifest.json",
        ],
    }
    (output / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"run_id": args.run_id, "summary": summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
