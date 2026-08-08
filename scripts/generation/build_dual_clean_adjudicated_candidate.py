#!/usr/bin/env python3
"""Freeze saturation arbitration and build the adjudicated dual-clean candidate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from internship_workspace.config import Workspace
from internship_workspace.datasets import resolve_path, select_record
from internship_workspace.saturation_gt_review import (
    CANDIDATE_DATASET as PARENT_DATASET,
    current_decisions,
    load_arbitration_queue,
    read_decision_history,
)


OVERLAY_DATASET = "particles2snr-dual-clean-saturation-arbitration@v1"
OUTPUT_DATASET = (
    "particles2snr-f-dual-clean-c1-yolo-4class-adjudicated-candidate@v1"
)
DEFAULT_SESSION = Path(
    "artifacts/particles2SNR-pipeline/audits/"
    "dual-clean-saturation-gt-review-v1-jlb"
)
DEFAULT_OVERLAY = Path(
    "datasets/interim/particles2snr-dual-clean-saturation-arbitration/v1"
)
DEFAULT_OUTPUT = Path(
    "datasets/interim/"
    "particles2snr-f-dual-clean-c1-yolo-4class-adjudicated-candidate/v1"
)
RUN_ID = "dual_clean_adjudicated_candidate_20260718"
DEFAULT_RUN_DIR = Path(f"artifacts/particles2SNR-pipeline/runs/{RUN_ID}")
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-dir", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--overlay-output", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument("--dataset-output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    return parser.parse_args()


def resolve(workspace: Workspace, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (workspace.root / path).resolve()


def relative(workspace: Workspace, path: Path) -> str:
    return path.resolve().relative_to(workspace.root.resolve()).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normalize_decision(
    candidate: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "candidate_id": candidate["candidate_id"],
        "order": candidate["order"],
        "source_id": candidate["source_id"],
        "filename": candidate["filename"],
        "split": candidate["split"],
        "annotation_id": candidate["annotation_id"],
        "class_id": candidate["class_id"],
        "class_name": candidate["class_name"],
        "original_label_line": candidate["label_line"],
        "original_interval_samples": candidate["original_interval_samples"],
        "original_interval_ms": candidate["original_interval_ms"],
        "repair_regions": candidate["repair_regions"],
        "safe_fragments_ms": candidate["safe_fragments_ms"],
        "decision": decision["decision"],
        "decision_source": decision.get("decision_source", "human"),
        "reviewer": decision["reviewer"],
        "reviewed_at": decision["reviewed_at"],
        "revision": decision["revision"],
        "corrected_start_ms": decision.get("corrected_start_ms"),
        "corrected_end_ms": decision.get("corrected_end_ms"),
        "comment": decision.get("comment", ""),
        "evidence": decision.get("evidence", {}),
    }


def corrected_label_line(
    record: dict[str, Any],
    signal_length: int,
    sampling_frequency_hz: float,
) -> str:
    start_ms = record["corrected_start_ms"]
    end_ms = record["corrected_end_ms"]
    if start_ms is None or end_ms is None:
        raise RuntimeError(f"clip decision lacks bounds: {record['candidate_id']}")
    start = float(start_ms) / 1000.0 * sampling_frequency_hz
    end = float(end_ms) / 1000.0 * sampling_frequency_hz
    center = ((start + end) / 2.0) / signal_length
    width = (end - start) / signal_length
    return f"{record['class_id']} {center:.10f} {width:.10f}"


def label_counts(dataset_root: Path, class_names: list[str]) -> dict[str, Any]:
    result = {}
    for split in SPLITS:
        label_paths = sorted((dataset_root / split / "labels").glob("*.txt"))
        counts = Counter()
        event_count = 0
        empty = 0
        for path in label_paths:
            lines = [
                line
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if not lines:
                empty += 1
            for line in lines:
                class_id = int(line.split()[0])
                counts[class_names[class_id]] += 1
                event_count += 1
        result[split] = {
            "signals": len(label_paths),
            "background_signals": empty,
            "event_bearing_signals": len(label_paths) - empty,
            "events": event_count,
            "events_by_class": {
                class_name: counts[class_name] for class_name in class_names
            },
        }
    return result


def signal_inventory(dataset_root: Path) -> dict[str, dict[str, str]]:
    inventory = {}
    for split in SPLITS:
        for path in sorted((dataset_root / split / "signals").glob("*.npy")):
            inventory[f"{split}/{path.name}"] = {
                "sha256": sha256_file(path),
                "size": str(path.stat().st_size),
            }
    return inventory


def main() -> None:
    args = parse_args()
    workspace = Workspace.load()
    session_dir = resolve(workspace, args.session_dir)
    overlay_output = resolve(workspace, args.overlay_output)
    dataset_output = resolve(workspace, args.dataset_output)
    run_dir = resolve(workspace, args.run_dir)
    overlay_output.relative_to(workspace.datasets_root / "interim")
    dataset_output.relative_to(workspace.datasets_root / "interim")
    run_dir.relative_to(workspace.artifacts_root / "particles2SNR-pipeline")
    for path in (overlay_output, dataset_output, run_dir):
        if path.exists():
            raise FileExistsError(f"refusing to mutate existing output: {path}")

    parent_id, parent_version = PARENT_DATASET.rsplit("@", 1)
    parent_record = select_record(workspace, parent_id, parent_version)
    parent_root = resolve_path(workspace, parent_record)
    queue, queue_root = load_arbitration_queue(workspace, PARENT_DATASET)
    if queue_root != parent_root:
        raise RuntimeError("resolved candidate roots differ")
    session_run_path = session_dir / "run.json"
    session_summary_path = session_dir / "session_summary.json"
    session_run = json.loads(session_run_path.read_text(encoding="utf-8"))
    session_summary = json.loads(
        session_summary_path.read_text(encoding="utf-8")
    )
    if not session_summary["complete"]:
        raise RuntimeError("arbitration session is not complete")
    if session_run["queue_sha256"] != queue["queue_sha256"]:
        raise RuntimeError("session and queue hashes differ")
    latest = current_decisions(session_dir)
    if len(latest) != queue["candidate_count"]:
        raise RuntimeError("current decision population is incomplete")
    by_id = {row["candidate_id"]: row for row in queue["candidates"]}
    overlay_rows = [
        normalize_decision(by_id[candidate_id], decision)
        for candidate_id, decision in latest.items()
    ]
    overlay_rows.sort(key=lambda row: int(row["order"]))
    decision_counts = Counter(row["decision"] for row in overlay_rows)
    if decision_counts != {
        "delete": 172,
        "keep": 17,
        "needs_review": 4,
    }:
        raise RuntimeError(f"unexpected locked decisions: {decision_counts}")
    overlay_sha256 = stable_hash(overlay_rows)
    created_at = datetime.now(timezone.utc).isoformat()

    overlay_output.mkdir(parents=True)
    write_jsonl(overlay_output / "current_decisions.jsonl", overlay_rows)
    history = read_decision_history(session_dir)
    write_jsonl(overlay_output / "decision_history.jsonl", history)
    disputed = [
        row for row in overlay_rows if row["decision"] == "needs_review"
    ]
    write_jsonl(overlay_output / "disputed_intervals.jsonl", disputed)
    overlay_summary = {
        "schema_version": 1,
        "dataset_id": OVERLAY_DATASET,
        "created_at": created_at,
        "status": "complete_frozen_arbitration",
        "source_candidate": PARENT_DATASET,
        "source_candidate_manifest_sha256": parent_record.payload[
            "manifest_sha256"
        ],
        "queue_sha256": queue["queue_sha256"],
        "overlay_sha256": overlay_sha256,
        "decision_counts": dict(sorted(decision_counts.items())),
        "decision_source_counts": dict(
            sorted(Counter(row["decision_source"] for row in overlay_rows).items())
        ),
        "history_entries": len(history),
        "current_decisions": len(overlay_rows),
        "session_run_path": relative(workspace, session_run_path),
        "session_run_sha256": sha256_file(session_run_path),
        "session_summary_sha256": sha256_file(session_summary_path),
        "application_policy": {
            "delete": "remove positive label",
            "keep": "preserve positive label byte-for-byte",
            "clip": "replace positive bounds",
            "needs_review": (
                "remove from positives and preserve as disputed ignore interval"
            ),
        },
    }
    (overlay_output / "overlay_summary.json").write_text(
        json.dumps(overlay_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    shutil.copytree(parent_root, dataset_output, copy_function=shutil.copy2)
    records_by_label: dict[tuple[str, str], dict[int, dict[str, Any]]] = (
        defaultdict(dict)
    )
    for row in overlay_rows:
        key = (row["split"], Path(row["filename"]).stem)
        records_by_label[key][int(row["annotation_id"])] = row
    change_rows = []
    changed_files = set()
    for (split, stem), decisions in sorted(records_by_label.items()):
        label_path = dataset_output / split / "labels" / f"{stem}.txt"
        original_path = parent_root / split / "labels" / f"{stem}.txt"
        original_lines = original_path.read_text(encoding="utf-8").splitlines()
        output_lines = []
        for annotation_id, line in enumerate(original_lines):
            record = decisions.get(annotation_id)
            if record is None:
                output_lines.append(line)
                continue
            if line != record["original_label_line"]:
                raise RuntimeError(
                    "queue label line differs from parent label: "
                    f"{record['candidate_id']}"
                )
            decision = record["decision"]
            new_line = ""
            if decision == "keep":
                new_line = line
                output_lines.append(line)
                action = "preserved_positive"
            elif decision == "clip":
                new_line = corrected_label_line(
                    record,
                    int(by_id[record["candidate_id"]]["source_length"]),
                    float(
                        by_id[record["candidate_id"]][
                            "sampling_frequency_hz"
                        ]
                    ),
                )
                output_lines.append(new_line)
                action = "clipped_positive"
            elif decision == "delete":
                action = "removed_systematic_clipping_artifact"
            elif decision == "needs_review":
                action = "removed_positive_preserved_as_disputed_ignore"
            else:
                raise RuntimeError(f"unsupported decision: {decision}")
            if new_line != line:
                changed_files.add((split, stem))
            change_rows.append(
                {
                    "candidate_id": record["candidate_id"],
                    "split": split,
                    "filename": f"{stem}.npy",
                    "annotation_id": annotation_id,
                    "class_id": record["class_id"],
                    "decision": decision,
                    "decision_source": record["decision_source"],
                    "action": action,
                    "original_label_line": line,
                    "new_label_line": new_line,
                    "overlay_sha256": overlay_sha256,
                }
            )
        label_path.write_text(
            "\n".join(output_lines) + ("\n" if output_lines else ""),
            encoding="utf-8",
        )
    if len(change_rows) != len(overlay_rows):
        raise RuntimeError("not every arbitration row was applied")
    write_csv(dataset_output / "label_change_manifest.csv", change_rows)
    shutil.copy2(
        overlay_output / "disputed_intervals.jsonl",
        dataset_output / "disputed_intervals.jsonl",
    )

    assignment_path = dataset_output / "class_assignment_manifest.csv"
    assignment_rows = read_csv(assignment_path)
    removed_keys = {
        (row["split"], row["filename"], str(row["annotation_id"]))
        for row in overlay_rows
        if row["decision"] in {"delete", "needs_review"}
    }
    corrected_assignment = []
    for row in assignment_rows:
        key = (row["split"], row["filename"], row["annotation_id"])
        if key in removed_keys:
            continue
        row["signal_action"] = "copied_from_saturation_candidate"
        corrected_assignment.append(row)
    if len(assignment_rows) - len(corrected_assignment) != 176:
        raise RuntimeError("unexpected class-assignment removal count")
    write_csv(assignment_path, corrected_assignment)

    dataset_yaml_path = dataset_output / "dataset.yaml"
    dataset_yaml = yaml.safe_load(
        dataset_yaml_path.read_text(encoding="utf-8")
    )
    class_names = list(dataset_yaml["names"])
    parent_counts = label_counts(parent_root, class_names)
    output_counts = label_counts(dataset_output, class_names)
    for split in SPLITS:
        split_payload = dataset_yaml["splits"][split]
        split_payload.update(output_counts[split]["events_by_class"])
        split_payload["background"] = output_counts[split][
            "background_signals"
        ]
        split_payload["total"] = output_counts[split]["signals"]
    dataset_yaml["path"] = "."
    dataset_yaml["preprocessing"]["saturation_repair_method"] = (
        "cosine-filtered-domain"
    )
    dataset_yaml["provenance"] = {
        "parent_dataset": PARENT_DATASET,
        "parent_manifest_sha256": parent_record.payload["manifest_sha256"],
        "arbitration_overlay": OVERLAY_DATASET,
        "overlay_sha256": overlay_sha256,
        "queue_sha256": queue["queue_sha256"],
        "session": relative(workspace, session_dir),
        "annotation_application": {
            "delete": 172,
            "keep": 17,
            "needs_review_removed_to_ignore": 4,
            "clip": 0,
        },
        "candidate_status": (
            "reference only; pending Particle2SNR tuning and canonical v2 "
            "regeneration"
        ),
    }
    dataset_yaml["annotation_counts"] = output_counts
    dataset_yaml["disputed_intervals"] = "disputed_intervals.jsonl"
    dataset_yaml_path.write_text(
        yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8"
    )

    balance_path = dataset_output / "class_balance_summary.json"
    balance = json.loads(balance_path.read_text(encoding="utf-8"))
    balance["source_root"] = PARENT_DATASET
    balance["output_root"] = OUTPUT_DATASET
    balance["arbitration_overlay"] = OVERLAY_DATASET
    balance["overlay_sha256"] = overlay_sha256
    balance["splits"] = {
        split: {
            "files": output_counts[split]["signals"],
            "events": output_counts[split]["events"],
            "events_by_class": output_counts[split]["events_by_class"],
            "background_signals": output_counts[split]["background_signals"],
            "event_bearing_signals": output_counts[split][
                "event_bearing_signals"
            ],
        }
        for split in SPLITS
    }
    balance_path.write_text(
        json.dumps(balance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    parent_signal_inventory = signal_inventory(parent_root)
    output_signal_inventory = signal_inventory(dataset_output)
    if parent_signal_inventory != output_signal_inventory:
        raise RuntimeError("adjudication changed candidate signal payloads")
    comparison = {
        "schema_version": 1,
        "parent_dataset": PARENT_DATASET,
        "output_dataset": OUTPUT_DATASET,
        "overlay_dataset": OVERLAY_DATASET,
        "overlay_sha256": overlay_sha256,
        "signals_unchanged": True,
        "signal_count": len(output_signal_inventory),
        "changed_label_files": len(changed_files),
        "decision_counts": dict(sorted(decision_counts.items())),
        "parent_annotation_counts": parent_counts,
        "output_annotation_counts": output_counts,
        "event_delta": {
            split: output_counts[split]["events"]
            - parent_counts[split]["events"]
            for split in SPLITS
        },
        "total_event_delta": sum(
            output_counts[split]["events"] - parent_counts[split]["events"]
            for split in SPLITS
        ),
        "disputed_ignore_count": len(disputed),
    }
    (dataset_output / "adjudication_summary.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    run_dir.mkdir(parents=True)
    run = {
        "schema_version": 1,
        "project": "particles2SNR-pipeline",
        "run_id": RUN_ID,
        "kind": "dataset-generation",
        "dataset": OUTPUT_DATASET,
        "repositories": {
            "workspace": git_state(workspace.root),
            "particles2SNR-pipeline": git_state(
                workspace.root / "particles2SNR-pipeline"
            ),
        },
        "command": (
            "particles2SNR-pipeline/scripts/generation/"
            "build_dual_clean_adjudicated_candidate.py"
        ),
        "created_at": created_at,
        "status": "complete_reference_candidate",
        "parents": {
            PARENT_DATASET: parent_record.payload["manifest_sha256"],
            OVERLAY_DATASET: overlay_sha256,
        },
        "outputs": [
            relative(workspace, overlay_output / "overlay_summary.json"),
            relative(workspace, overlay_output / "current_decisions.jsonl"),
            relative(workspace, dataset_output / "dataset.yaml"),
            relative(
                workspace, dataset_output / "adjudication_summary.json"
            ),
            relative(
                workspace, dataset_output / "label_change_manifest.csv"
            ),
            relative(workspace, dataset_output / "disputed_intervals.jsonl"),
        ],
        "summary": comparison,
    }
    (run_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = [
        "# Dual-clean adjudicated candidate",
        "",
        f"- Parent: `{PARENT_DATASET}`",
        f"- Overlay: `{OVERLAY_DATASET}`",
        f"- Candidate: `{OUTPUT_DATASET}`",
        f"- Overlay hash: `{overlay_sha256}`",
        f"- Labels removed from positives: {-comparison['total_event_delta']}",
        f"- Disputed ignore intervals: {len(disputed)}",
        f"- Changed label files: {len(changed_files)}",
        f"- Signals unchanged: {comparison['signals_unchanged']}",
        "",
        "This remains a reference candidate. It is not the canonical dual-clean "
        "v2 and must not be used for training before Particle2SNR tuning and "
        "regeneration.",
        "",
    ]
    (run_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "overlay": relative(workspace, overlay_output),
                "candidate": relative(workspace, dataset_output),
                "summary": comparison,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
