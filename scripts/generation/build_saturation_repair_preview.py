#!/usr/bin/env python3
"""Build a two-method saturation-repair preview for one audited source trace."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from internship_workspace.config import Workspace
from internship_workspace.datasets import resolve_path, select_record
from particles2snr.saturation_cleaning import repair_saturation_interval


SOURCE_DATASET = "c1-hf-5-10-10um-doublet@v1"
CURRENT_DATASET = "particles2snr-f-dual-clean-c1-yolo-4class@v1"
NOISE_DATASET = "noise@v1"
PREVIEW_DATASET = "particles2snr-saturation-repair-preview@v1"
SOURCE_ID = "HFocusing_5_10_10um_0_325"
NOISE_FILE = "Noise_HF_20_6_0_178.npy"
NOISE_OFFSET = 10_381
CORE_INTERVAL = (3_456, 6_724)
EXPANDED_INTERVAL = (3_156, 7_024)
SAMPLING_FREQUENCY_HZ = 2_000_000.0
BANDPASS_HZ = (7_000.0, 80_000.0)
BANDPASS_ORDER = 4
RUN_ID = "saturation_repair_preview_20260717"
REVIEW_MANIFEST_RELATIVE = Path(
    "artifacts/SMI_Detection_CNN_transformers/research/"
    "wave8like-gt-review-v1-jlb/review_manifest.json"
)
HISTORICAL_CLEAN_RELATIVE = Path(
    "artifacts/particles2SNR-pipeline/runs/"
    "p0_c1_Particles2SNR_F_dual_clean_candidate/test/"
    "peak_evidence_clean_signals/10um"
) / f"{SOURCE_ID}.npy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/interim/particles2snr-saturation-repair-preview/v1"),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(f"artifacts/particles2SNR-pipeline/runs/{RUN_ID}"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def relative(workspace: Workspace, path: Path) -> str:
    return path.resolve().relative_to(workspace.root.resolve()).as_posix()


def resolve_dataset(
    workspace: Workspace, dataset_key: str
) -> tuple[Path, dict[str, Any]]:
    dataset_id, version = dataset_key.rsplit("@", 1)
    record = select_record(workspace, dataset_id, version)
    return resolve_path(workspace, record), record.payload


def robust_metrics(
    signal: np.ndarray,
    *,
    historical_interval_ms: tuple[float, float],
) -> dict[str, Any]:
    values = np.asarray(signal, dtype=float)
    median = float(np.median(values))
    sigma = max(
        float(np.median(np.abs(values - median)) * 1.4826),
        1e-12,
    )

    def maximum_z(start: int, end: int) -> float:
        start = max(0, int(start))
        end = min(len(values), int(end))
        return float(np.max(np.abs(values[start:end] - median)) / sigma)

    historical_samples = tuple(
        int(round(value / 1000.0 * SAMPLING_FREQUENCY_HZ))
        for value in historical_interval_ms
    )
    return {
        "robust_sigma": sigma,
        "historical_interval_max_abs_z": maximum_z(*historical_samples),
        "start_transition_max_abs_z": maximum_z(
            EXPANDED_INTERVAL[0] - 600, CORE_INTERVAL[0] + 600
        ),
        "end_transition_max_abs_z": maximum_z(
            CORE_INTERVAL[1] - 600, EXPANDED_INTERVAL[1] + 600
        ),
        "overall_max_abs_z": maximum_z(0, len(values)),
        "boundary_steps": {
            str(index): float(abs(values[index] - values[index - 1]))
            for index in (
                EXPANDED_INTERVAL[0],
                CORE_INTERVAL[0],
                CORE_INTERVAL[1],
                EXPANDED_INTERVAL[1],
            )
            if 0 < index < len(values)
        },
    }


def git_state(path: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty}


def main() -> None:
    args = parse_args()
    workspace = Workspace.load()
    output = (
        args.output.resolve()
        if args.output.is_absolute()
        else (workspace.root / args.output).resolve()
    )
    run_dir = (
        args.run_dir.resolve()
        if args.run_dir.is_absolute()
        else (workspace.root / args.run_dir).resolve()
    )
    output.relative_to(workspace.datasets_root / "interim")
    run_dir.relative_to(
        workspace.artifacts_root / "particles2SNR-pipeline"
    )
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise SystemExit(f"preview already exists; pass --overwrite: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "signals").mkdir(exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_root, raw_record = resolve_dataset(workspace, SOURCE_DATASET)
    current_root, current_record = resolve_dataset(workspace, CURRENT_DATASET)
    noise_root, noise_record = resolve_dataset(workspace, NOISE_DATASET)
    raw_path = raw_root / f"{SOURCE_ID}.npy"
    current_path = current_root / "test" / "signals" / f"{SOURCE_ID}.npy"
    noise_path = noise_root / NOISE_FILE
    for path in (raw_path, current_path, noise_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    raw = np.asarray(np.load(raw_path), dtype=np.float64)
    current = np.asarray(np.load(current_path), dtype=np.float64)
    noise = np.asarray(np.load(noise_path), dtype=np.float64)
    replacement_length = EXPANDED_INTERVAL[1] - EXPANDED_INTERVAL[0]
    replacement = noise[NOISE_OFFSET : NOISE_OFFSET + replacement_length]
    if len(replacement) != replacement_length:
        raise RuntimeError("registered noise carrier is too short")

    historical_clean_path = workspace.root / HISTORICAL_CLEAN_RELATIVE
    if not historical_clean_path.is_file():
        raise FileNotFoundError(historical_clean_path)
    historical_clean = np.asarray(np.load(historical_clean_path), dtype=np.float64)
    if not np.array_equal(
        historical_clean[EXPANDED_INTERVAL[0] : EXPANDED_INTERVAL[1]],
        replacement,
    ):
        raise RuntimeError("registered noise slice does not match historical repair")

    review_manifest_path = workspace.root / REVIEW_MANIFEST_RELATIVE
    review_manifest = json.loads(review_manifest_path.read_text(encoding="utf-8"))
    candidate = next(
        row
        for row in review_manifest["candidates"]
        if row["source_id"] == SOURCE_ID and row["historical_annotation_id"] == 0
    )
    historical_interval_ms = tuple(candidate["historical_interval_ms"])
    expanded_interval_ms = tuple(
        value / SAMPLING_FREQUENCY_HZ * 1000.0
        for value in EXPANDED_INTERVAL
    )
    historical_center_ms = sum(historical_interval_ms) / 2.0
    safe_interval_proposal_ms = None
    if historical_center_ms < expanded_interval_ms[0]:
        safe_interval_proposal_ms = [
            historical_interval_ms[0],
            min(historical_interval_ms[1], expanded_interval_ms[0]),
        ]
    elif historical_center_ms >= expanded_interval_ms[1]:
        safe_interval_proposal_ms = [
            max(historical_interval_ms[0], expanded_interval_ms[1]),
            historical_interval_ms[1],
        ]
    safe_interval_eligible = bool(
        safe_interval_proposal_ms
        and safe_interval_proposal_ms[1] - safe_interval_proposal_ms[0] >= 0.08
    )
    methods = {
        "cosine_pre_filter": "cosine-pre-filter",
        "cosine_filtered_domain": "cosine-filtered-domain",
    }
    variants: dict[str, dict[str, Any]] = {}
    for slug, method in methods.items():
        repaired = repair_saturation_interval(
            raw,
            replacement,
            core_interval=CORE_INTERVAL,
            expanded_interval=EXPANDED_INTERVAL,
            method=method,
            fs=SAMPLING_FREQUENCY_HZ,
            fmin=BANDPASS_HZ[0],
            fmax=BANDPASS_HZ[1],
            order=BANDPASS_ORDER,
        )
        filtered_name = f"signals/{SOURCE_ID}__{slug}.npy"
        np.save(output / filtered_name, repaired["filtered_signal"])
        variants[slug] = {
            "method": method,
            "filtered_signal_path": filtered_name,
            "filtered_signal_sha256": sha256_file(output / filtered_name),
            "metrics": robust_metrics(
                repaired["filtered_signal"],
                historical_interval_ms=historical_interval_ms,
            ),
        }
        if slug == "cosine_pre_filter":
            clean_name = f"signals/{SOURCE_ID}__cosine_clean.npy"
            np.save(output / clean_name, repaired["clean_signal"])
            variants[slug]["clean_signal_path"] = clean_name
            variants[slug]["clean_signal_sha256"] = sha256_file(
                output / clean_name
            )

    preview = {
        "schema_version": 1,
        "dataset_id": PREVIEW_DATASET,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_id": SOURCE_ID,
        "sampling_frequency_hz": SAMPLING_FREQUENCY_HZ,
        "source_length": int(len(raw)),
        "parents": {
            SOURCE_DATASET: raw_record["manifest_sha256"],
            CURRENT_DATASET: current_record["manifest_sha256"],
            NOISE_DATASET: noise_record["manifest_sha256"],
        },
        "source": {
            "raw_path": relative(workspace, raw_path),
            "raw_sha256": sha256_file(raw_path),
            "current_filtered_path": relative(workspace, current_path),
            "current_filtered_sha256": sha256_file(current_path),
            "review_manifest_path": REVIEW_MANIFEST_RELATIVE.as_posix(),
            "review_manifest_sha256": sha256_file(review_manifest_path),
        },
        "repair": {
            "core_interval_samples": list(CORE_INTERVAL),
            "expanded_interval_samples": list(EXPANDED_INTERVAL),
            "guard_samples": 300,
            "core_interval_ms": [
                value / SAMPLING_FREQUENCY_HZ * 1000.0
                for value in CORE_INTERVAL
            ],
            "expanded_interval_ms": [
                value / SAMPLING_FREQUENCY_HZ * 1000.0
                for value in EXPANDED_INTERVAL
            ],
            "noise": {
                "dataset_id": NOISE_DATASET,
                "path": NOISE_FILE,
                "file_sha256": sha256_file(noise_path),
                "offset": NOISE_OFFSET,
                "length": replacement_length,
                "slice_sha256": hashlib.sha256(replacement.tobytes()).hexdigest(),
            },
            "bandpass_hz": list(BANDPASS_HZ),
            "bandpass_order": BANDPASS_ORDER,
            "bandpass_phase": "zero-phase forward-backward",
        },
        "evidence": {
            "historical_annotation_id": candidate["historical_annotation_id"],
            "historical_class_id": candidate["historical_class_id"],
            "historical_interval_ms": candidate["historical_interval_ms"],
            "historical_center_ms": historical_center_ms,
            "safe_interval_proposal_ms": safe_interval_proposal_ms,
            "safe_interval_eligible": safe_interval_eligible,
            "consensus_prediction": candidate["consensus_prediction"],
            "current_metrics": robust_metrics(
                current,
                historical_interval_ms=historical_interval_ms,
            ),
        },
        "variants": variants,
        "acceptance": {
            "transition_max_abs_z_review_threshold": 8.0,
            "human_gate_required": True,
            "allowed_decisions": [
                "approve_A",
                "approve_B",
                "reject_both",
                "needs_adjustment",
            ],
        },
    }
    preview["preview_sha256"] = stable_hash(
        {key: value for key, value in preview.items() if key != "created_at"}
    )
    (output / "preview.json").write_text(
        json.dumps(preview, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report = [
        "# Saturation repair preview",
        "",
        f"- Source: `{SOURCE_ID}`",
        f"- Preview hash: `{preview['preview_sha256']}`",
        (
            "- Current historical-window max robust-z: "
            f"{preview['evidence']['current_metrics']['historical_interval_max_abs_z']:.3f}"
        ),
    ]
    for slug, row in variants.items():
        report.append(
            f"- `{slug}` historical-window max robust-z: "
            f"{row['metrics']['historical_interval_max_abs_z']:.3f}"
        )
    report.extend(
        [
            "",
            "No registered dataset was modified. Promotion remains blocked on "
            "the human A/B gate.",
            "",
        ]
    )
    (run_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    (run_dir / "preview_summary.json").write_text(
        json.dumps(preview, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    run = {
        "schema_version": 1,
        "project": "particles2SNR-pipeline",
        "run_id": RUN_ID,
        "dataset": PREVIEW_DATASET,
        "repositories": {
            "workspace": git_state(workspace.root),
            "particles2SNR-pipeline": git_state(
                workspace.root / "particles2SNR-pipeline"
            ),
        },
        "command": (
            "particles2SNR-pipeline/scripts/generation/"
            "build_saturation_repair_preview.py"
        ),
        "created_at": preview["created_at"],
        "status": "complete_pending_human_review",
        "outputs": [
            relative(workspace, output / "preview.json"),
            "preview_summary.json",
            "REPORT.md",
        ],
        "preview_sha256": preview["preview_sha256"],
    }
    (run_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
