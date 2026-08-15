"""Contracts and deterministic construction for the beads supervision ledger."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
FOLD_SALT = "particle-ledger-gradual-supervision-cv-r1"
CLASS_TO_ID = {"2um": 0, "4um": 1, "10um": 2}
WEIGHTS = {
    "human_confirmed": {"presence": 1.0, "class": 1.0, "center": 1.0, "box": 1.0},
    "detector_seed_unreviewed": {"presence": 1.0, "class": 1.0, "center": 1.0, "box": 0.25},
    "human_uncertain": {"presence": 1.0, "class": 1.0, "center": 1.0, "box": 0.0},
    "mad_weak": {"presence": 0.25, "class": 0.25, "center": 0.25, "box": 0.10},
}


class GradualSupervisionError(ValueError):
    """Raised when an input violates the frozen experiment contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: str) -> str:
    return hashlib.sha256(f"{FOLD_SALT}:{value}".encode()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def build_audit_folds(
    traces: list[dict[str, Any]],
    event_counts: dict[str, int],
    common_miss_counts: dict[str, int],
) -> dict[str, int]:
    """Assign exactly four traces per class to every fold.

    Common-miss traces are placed first; event burden is the secondary balance
    key and the salted source hash is the only stochastic-looking tie-breaker.
    """
    if len(traces) != 60:
        raise GradualSupervisionError(f"expected 60 audited traces, got {len(traces)}")
    folds = [
        {"class_counts": defaultdict(int), "event_count": 0, "miss_count": 0, "sources": []}
        for _ in range(5)
    ]
    assignment: dict[str, int] = {}
    for source_class in CLASS_TO_ID:
        members = [row for row in traces if row["source_class"] == source_class]
        if len(members) != 20:
            raise GradualSupervisionError(
                f"expected 20 audited {source_class} traces, got {len(members)}"
            )
        members.sort(
            key=lambda row: (
                -common_miss_counts.get(row["source_id"], 0),
                -event_counts.get(row["source_id"], 0),
                stable_hash(row["source_id"]),
            )
        )
        for row in members:
            eligible = [i for i, fold in enumerate(folds) if fold["class_counts"][source_class] < 4]
            fold_id = min(
                eligible,
                key=lambda i: (
                    folds[i]["miss_count"],
                    folds[i]["event_count"],
                    len(folds[i]["sources"]),
                    i,
                ),
            )
            assignment[row["source_id"]] = fold_id
            folds[fold_id]["class_counts"][source_class] += 1
            folds[fold_id]["event_count"] += event_counts.get(row["source_id"], 0)
            folds[fold_id]["miss_count"] += common_miss_counts.get(row["source_id"], 0)
            folds[fold_id]["sources"].append(row["source_id"])
    for fold_id, fold in enumerate(folds):
        if len(fold["sources"]) != 12 or dict(fold["class_counts"]) != {name: 4 for name in CLASS_TO_ID}:
            raise GradualSupervisionError(f"invalid fold {fold_id}: {fold}")
    miss_folds = {
        assignment[source_id]
        for source_id, count in common_miss_counts.items()
        if count
    }
    if len(miss_folds) != 5:
        raise GradualSupervisionError(
            f"common-miss source traces must cover five folds, got {sorted(miss_folds)}"
        )
    return assignment


def assign_control_folds(control_ids: Iterable[str], seeded: dict[str, int] | None = None) -> dict[str, int]:
    """Deterministically balance controls not already assigned with audit traces."""
    result = dict(seeded or {})
    loads = [0] * 5
    for fold in result.values():
        loads[fold] += 1
    for control_id in sorted(set(control_ids) - result.keys(), key=stable_hash):
        fold = min(range(5), key=lambda i: (loads[i], i))
        result[control_id] = fold
        loads[fold] += 1
    return result


def is_verified_empty_audit_row(row: dict[str, Any]) -> bool:
    """Return true only for an explicit, cardinality-evaluable empty review."""
    return (
        row["trace_status"] == "reviewed"
        and str(row["cardinality_evaluable"]) == "True"
        and int(row["confirmed_event_count"]) == 0
        and int(row["uncertain_event_count"]) == 0
    )


def _workspace_relative(workspace: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(workspace.resolve()).as_posix()
    except ValueError as exc:
        raise GradualSupervisionError(f"path escapes workspace: {path}") from exc


@dataclass(frozen=True)
class BuildInputs:
    workspace: Path
    source_matrix: Path
    proposals: Path
    audit_traces: Path
    ledger_events: Path
    ledger_summary: Path
    atlas: Path
    synthetic_manifest: Path
    synthetic_root: Path
    dual_clean_root: Path
    method_receipt: Path
    dataset_id: str = "particles2snr-beads-gradual-supervision-development@v1"


def build_dataset(inputs: BuildInputs, output: Path) -> dict[str, Any]:
    """Build a manifest-only immutable candidate dataset."""
    output.mkdir(parents=True, exist_ok=False)
    source_rows = read_csv(inputs.source_matrix)
    audit_rows = read_csv(inputs.audit_traces)
    ledger_rows = read_csv(inputs.ledger_events)
    proposal_rows = read_csv(inputs.proposals)
    ledger_summary = json.loads(inputs.ledger_summary.read_text(encoding="utf-8"))
    atlas = json.loads(inputs.atlas.read_text(encoding="utf-8"))

    source_by_id = {row["source_id"]: row for row in source_rows}
    if len(source_by_id) != 1477:
        raise GradualSupervisionError(f"expected 1,477 calibration sources, got {len(source_by_id)}")
    audit_by_source = {row["source_id"]: row for row in audit_rows}
    if len(audit_by_source) != 60:
        raise GradualSupervisionError("audit traces are not unique")

    positives = [row for row in ledger_rows if row["revised_state"] == "confirmed_positive"]
    ambiguous = [row for row in ledger_rows if row["revised_state"] == "ambiguous_event"]
    expected_positives = int(ledger_summary["population"]["confirmed_positive"])
    expected_ambiguous = int(ledger_summary["population"]["ambiguous_event"])
    if len(positives) != expected_positives or len(ambiguous) != expected_ambiguous:
        raise GradualSupervisionError(
            f"ledger population mismatch: expected {expected_positives} positives and "
            f"{expected_ambiguous} ambiguities, got {len(positives)} and {len(ambiguous)}"
        )
    event_counts: dict[str, int] = defaultdict(int)
    for row in positives:
        event_counts[row["source_id"]] += 1
    blind_to_source = {row["blind_id"]: row["source_id"] for row in audit_rows}
    common_miss_counts: dict[str, int] = defaultdict(int)
    for item in ledger_summary["all_four_common_misses"]:
        common_miss_counts[blind_to_source[item["blind_id"]]] += 1
    if sum(common_miss_counts.values()) != 6 or len(common_miss_counts) != 5:
        raise GradualSupervisionError("expected six common misses on five sources")
    audit_folds = build_audit_folds(audit_rows, event_counts, common_miss_counts)

    old_empty_cases = {
        case["case_id"]: case
        for case in atlas["calibration_review_cases"]
        if case["case_id"] in set(atlas["human_empty_trace_case_ids"])
    }
    audit_empty_sources = {
        row["source_id"] for row in audit_rows
        if is_verified_empty_audit_row(row)
    }
    old_empty_sources = {Path(case["filename"]).stem for case in old_empty_cases.values()}
    verified_empty_sources = audit_empty_sources | old_empty_sources
    if len(audit_empty_sources) != 9 or len(old_empty_sources) != 14 or len(verified_empty_sources) != 22:
        raise GradualSupervisionError("verified-empty source counts differ from the frozen contract")

    fp_loci = [row for row in atlas["stress_loci"] if row["canonical_verdict"] == "genuine_model_fp"]
    if len(fp_loci) != 6:
        raise GradualSupervisionError(f"expected six genuine FP loci, got {len(fp_loci)}")
    fp_sources = {row["source_id"] for row in fp_loci}

    split_ids = set(atlas["split_landmarks"])
    split_cases = [case for case in atlas["calibration_review_cases"] if case["case_id"] in split_ids]
    close_ids = set(atlas["close_trace_case_ids"])
    close_cases = [case for case in atlas["calibration_review_cases"] if case["case_id"] in close_ids]
    if len(split_cases) != 5 or len(close_cases) != 2:
        raise GradualSupervisionError("expected five split windows and two close-trace controls")
    joined_sources = {Path(case["filename"]).stem for case in split_cases + close_cases}
    evaluation_sources = set(audit_by_source) | verified_empty_sources | fp_sources | joined_sources

    signal_lookup: dict[str, dict[str, Any]] = {}
    for split in ("train", "val"):
        for signal_path in sorted((inputs.dual_clean_root / split / "signals").glob("*.npy")):
            signal_lookup[signal_path.stem] = {
                "signal_path": _workspace_relative(inputs.workspace, signal_path),
                "signal_sha256": sha256_file(signal_path),
                "split": split,
            }
    missing_control_signals = sorted(evaluation_sources - signal_lookup.keys())
    if missing_control_signals:
        raise GradualSupervisionError(
            f"control signals missing from dual-clean dataset: {missing_control_signals[:3]}"
        )

    seeded_empty = {source: audit_folds[source] for source in audit_empty_sources}
    empty_folds = assign_control_folds(verified_empty_sources, seeded=seeded_empty)
    fp_folds = assign_control_folds([row["locus_id"] for row in fp_loci])
    joined_folds = assign_control_folds(
        [case["case_id"] for case in split_cases + close_cases],
        seeded={
            case["case_id"]: audit_folds[Path(case["filename"]).stem]
            for case in split_cases + close_cases
            if Path(case["filename"]).stem in audit_folds
        },
    )

    trace_records: list[dict[str, Any]] = []
    for source_id, source in sorted(source_by_id.items()):
        signal_path = inputs.workspace / source["signal_path"]
        if not signal_path.is_file() or sha256_file(signal_path) != source["signal_sha256"]:
            raise GradualSupervisionError(f"signal provenance mismatch: {source_id}")
        if source_id in audit_by_source:
            state = "verified_empty" if source_id in audit_empty_sources else "human_reviewed"
            policy = "all_cells_negative" if state == "verified_empty" else "positive_cells_only"
            fold = audit_folds[source_id]
        elif source_id in evaluation_sources:
            state, policy, fold = "held_out_control", "evaluation_only", None
        else:
            state, policy, fold = "mad_weak", "positive_cells_only", None
        trace_records.append({
            "source_id": source_id,
            "source_class": source["source_class"],
            "class_id": CLASS_TO_ID[source["source_class"]],
            "signal_path": _workspace_relative(inputs.workspace, signal_path),
            "signal_sha256": source["signal_sha256"],
            "signal_length": 16384,
            "review_state": state,
            "objectness_policy": policy,
            "audit_fold": fold,
            "inference_fold": fold if fold is not None else int(stable_hash(source_id)[:8], 16) % 5,
            "training_excluded": source_id in evaluation_sources and (
                source_id not in audit_by_source or source_id in verified_empty_sources
            ),
        })

    event_records: list[dict[str, Any]] = []
    for row in positives:
        geometry = row["geometry_state"]
        weights = WEIGHTS[geometry]
        event_records.append({
            "event_id": row["event_id"], "source_id": row["source_id"],
            "origin": "human", "identity_state": "confirmed_positive",
            "geometry_state": geometry, "class_id": CLASS_TO_ID[row["source_class"]],
            "center": int(float(row["center"])),
            "support_start": int(float(row["support_start"])),
            "support_end": int(float(row["support_end"])),
            "weights": weights,
        })
    for row in ambiguous:
        support_start = int(float(row["support_start"])) if row["support_start"] else None
        support_end = int(float(row["support_end"])) if row["support_end"] else None
        event_records.append({
            "event_id": row["event_id"], "source_id": row["source_id"],
            "origin": "human", "identity_state": "ambiguous_event",
            "geometry_state": row["geometry_state"], "class_id": CLASS_TO_ID[row["source_class"]],
            "center": int(float(row["center"])),
            "support_start": support_start,
            "support_end": support_end,
            "weights": {"presence": 0.0, "class": 0.0, "center": 0.0, "box": 0.0},
        })
    for row in proposal_rows:
        if row["proposer"] != "mad_unified" or row["source_id"] in audit_by_source or row["source_id"] in evaluation_sources:
            continue
        start, end = float(row["start"]), float(row["end"])
        event_records.append({
            "event_id": row["proposal_id"], "source_id": row["source_id"],
            "origin": "mad_unified", "identity_state": "unreviewed_proposal",
            "geometry_state": "mad_weak", "class_id": CLASS_TO_ID[row["source_class"]],
            "center": int(round((start + end) / 2.0)),
            "support_start": int(round(start)), "support_end": int(round(end)),
            "weights": WEIGHTS["mad_weak"],
        })

    synthetic_rows = read_csv(inputs.synthetic_manifest)
    synthetic_subset: list[dict[str, Any]] = []
    excluded_synthetic = 0
    for row in synthetic_rows:
        source_ids = [value for value in row["source_ids"].split(";") if value]
        if evaluation_sources.intersection(source_ids):
            excluded_synthetic += 1
            continue
        signal = inputs.synthetic_root / row["split"] / "signals" / f"{row['long_id']}.npy"
        label = inputs.synthetic_root / row["split"] / "labels" / f"{row['long_id']}.txt"
        if not signal.is_file() or not label.is_file():
            raise GradualSupervisionError(f"missing synthetic pair: {row['long_id']}")
        if sha256_file(signal) != row["signal_sha256"] or sha256_file(label) != row["label_sha256"]:
            raise GradualSupervisionError(f"synthetic hash mismatch: {row['long_id']}")
        synthetic_subset.append({
            "long_id": row["long_id"], "split": row["split"], "stratum": row["stratum"],
            "group_id": int(row["group_id"]), "permutation_index": int(row["permutation_index"]),
            "signal_path": _workspace_relative(inputs.workspace, signal),
            "label_path": _workspace_relative(inputs.workspace, label),
            "signal_sha256": row["signal_sha256"], "label_sha256": row["label_sha256"],
            "source_ids": source_ids,
        })
    leaked = [row["long_id"] for row in synthetic_subset if evaluation_sources.intersection(row["source_ids"])]
    if leaked:
        raise GradualSupervisionError(f"synthetic evaluation-source leak: {leaked[:3]}")

    controls: list[dict[str, Any]] = []
    for source_id in sorted(verified_empty_sources):
        controls.append({
            "control_id": f"empty:{source_id}", "kind": "verified_empty_trace",
            "source_id": source_id, "fold": empty_folds[source_id], **signal_lookup[source_id],
        })
    for locus in fp_loci:
        controls.append({
            "control_id": locus["locus_id"], "kind": "verified_artifact_locus",
            "source_id": locus["source_id"], "start": int(float(locus["local_left"])),
            "end": int(float(locus["local_right"])), "fold": fp_folds[locus["locus_id"]],
            **signal_lookup[locus["source_id"]],
        })
    for case in split_cases + close_cases:
        controls.append({
            "control_id": case["case_id"], "kind": "joined_event_control",
            "source_id": Path(case["filename"]).stem,
            "interval": case.get("candidate_interval") or atlas["close_trace_intervals"][case["case_id"]],
            "expected_count": 2,
            "fold": joined_folds[case["case_id"]],
            **signal_lookup[Path(case["filename"]).stem],
        })

    write_jsonl(output / "traces.jsonl", trace_records)
    write_jsonl(output / "events.jsonl", sorted(event_records, key=lambda row: (row["source_id"], row["center"], row["event_id"])))
    write_jsonl(output / "synthetic_subset.jsonl", synthetic_subset)
    write_jsonl(output / "controls.jsonl", controls)
    folds = {
        str(fold): sorted(source for source, assigned in audit_folds.items() if assigned == fold)
        for fold in range(5)
    }
    write_json(output / "folds.json", {"schema_version": 1, "salt": FOLD_SALT, "folds": folds})
    contract = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": inputs.dataset_id,
        "format": "confidence-detection-manifest-v1",
        "grain": {"traces.jsonl": "one physical source trace", "events.jsonl": "one supervision event"},
        "class_names": list(CLASS_TO_ID),
        "unknown_is_negative": False,
        "weights": WEIGHTS,
        "fold_policy": "five immutable source-disjoint folds; inner validation=(outer+1)%5",
        "synthetic_policy": "manifest subset only; all evaluation and control source IDs removed",
        "verified_empty_policy": "trace_status=reviewed, cardinality_evaluable=true, and zero confirmed/uncertain events",
    }
    write_json(output / "dataset-contract.json", contract)
    summary = {
        "schema_version": 1,
        "dataset_id": contract["dataset_id"],
        "status": "candidate_reference_pending_visual_review",
        "counts": {
            "calibration_traces": len(trace_records), "confirmed_human_events": len(positives),
            "ambiguous_human_events": len(ambiguous),
            "weak_mad_events": sum(row["origin"] == "mad_unified" for row in event_records),
            "verified_empty_unique_sources": len(verified_empty_sources),
            "verified_artifact_loci": len(fp_loci), "joined_event_controls": len(split_cases) + len(close_cases),
            "synthetic_rows_retained": len(synthetic_subset), "synthetic_rows_excluded": excluded_synthetic,
        },
        "evaluation_source_count": len(evaluation_sources),
        "evaluation_source_hash": hashlib.sha256("\n".join(sorted(evaluation_sources)).encode()).hexdigest(),
        "zero_synthetic_evaluation_overlap": True,
        "fold_event_counts": {
            str(fold): sum(event_counts[source] for source in sources) for fold, sources in ((int(k), v) for k, v in folds.items())
        },
        "fold_common_miss_counts": {
            str(fold): sum(common_miss_counts[source] for source in sources) for fold, sources in ((int(k), v) for k, v in folds.items())
        },
    }
    write_json(output / "summary.json", summary)
    write_json(output / "dataset.yaml", {
        "schema_version": 1, "dataset_id": contract["dataset_id"],
        "status": "reference", "format": contract["format"], "path": ".",
        "names": list(CLASS_TO_ID), "nc": 3,
        "sampling_frequency_hz": 2_000_000,
        "signal_lengths": {"physical_trace": 16384, "synthetic_replay": 65536},
        "development_only": True, "sealed_test_accessed": False,
        "unknown_is_negative": False,
    })
    write_json(output / "run.json", {
        "schema_version": 1,
        "run_id": inputs.dataset_id.replace("@", "-").replace("/", "-") + "-build",
        "kind": "confidence-weighted-detection-manifest-build",
        "status": "complete_reference_candidate",
        "command": "particles2SNR-pipeline/scripts/generation/build_gradual_supervision_dataset.py",
        "source_run_ids": [
            "particle-event-inventory-matrix-analysis-r1",
            inputs.audit_traces.parent.name,
            inputs.ledger_events.parent.name,
            "particle-c2-mad-calibration-atlas-r3",
        ],
        "method_evidence_run_id": "particle-gradual-supervision-method-r1",
        "method_receipt": {
            "path": _workspace_relative(inputs.workspace, inputs.method_receipt),
            "sha256": sha256_file(inputs.method_receipt),
        },
        "sealed_test_accessed": False,
    })
    files = []
    for path in sorted(output.iterdir()):
        files.append({"path": path.name, "size": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {
        "schema_version": 1, "dataset_id": contract["dataset_id"],
        "parents": [
            "particles2snr-f-dual-clean-c1-yolo-4class@v2",
            "particles2snr-z8-v2-wave8like-known3-background-development@v4",
            ledger_summary["ledger"]["ledger_id"],
        ],
        "files": files,
        "computation_fingerprint": hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    write_json(output / "dataset-manifest.json", manifest)
    return summary
