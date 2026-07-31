from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import yaml


def yolo_line(
    *,
    class_id: int,
    start_ms: float,
    end_ms: float,
    duration_ms: float,
) -> str:
    if not 0.0 <= start_ms < end_ms <= duration_ms:
        raise ValueError(
            f"Invalid interval {start_ms:.6f}–{end_ms:.6f} ms "
            f"for duration {duration_ms:.6f} ms"
        )
    center = ((start_ms + end_ms) / 2.0) / duration_ms
    width = (end_ms - start_ms) / duration_ms
    return f"{int(class_id)} {center:.10f} {width:.10f}"


def apply_reviewed_labels(
    *,
    source_root: Path,
    output_root: Path,
    queue: dict[str, Any],
    decisions: dict[str, dict[str, Any]],
    proposals: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite dataset: {output_root}")
    shutil.copytree(source_root, output_root, copy_function=shutil.copy2)
    candidates = queue["candidates"]
    if set(decisions) != {row["candidate_id"] for row in candidates}:
        raise ValueError("Decisions do not cover the complete arbitration queue")

    grouped: dict[str, dict[int, dict[str, Any]]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate["label_path"], {})[
            int(candidate["annotation_id"])
        ] = candidate

    application_rows: list[dict[str, Any]] = []
    for relative_label, by_annotation in grouped.items():
        label_path = output_root / relative_label
        lines = label_path.read_text(encoding="utf-8").splitlines()
        rewritten: list[str] = []
        for annotation_id, line in enumerate(lines):
            candidate = by_annotation.get(annotation_id)
            if candidate is None:
                rewritten.append(line)
                continue
            decision = decisions[candidate["candidate_id"]]
            action = decision["decision"]
            replacement = line
            new_start = new_end = None
            if action == "delete":
                replacement = ""
            elif action == "keep":
                pass
            elif action == "clip":
                new_start = float(decision["corrected_start_ms"])
                new_end = float(decision["corrected_end_ms"])
                replacement = yolo_line(
                    class_id=int(candidate["class_id"]),
                    start_ms=new_start,
                    end_ms=new_end,
                    duration_ms=float(candidate["source_duration_ms"]),
                )
            elif action == "needs_review":
                proposal = proposals.get(candidate["candidate_id"])
                if proposal is None:
                    raise ValueError(
                        f"Missing detector box proposal for {candidate['candidate_id']}"
                    )
                if proposal["class_name"] != candidate["class_name"]:
                    raise ValueError(
                        f"Proposal changes GT class for {candidate['candidate_id']}"
                    )
                new_start = float(proposal["proposed_start_ms"])
                new_end = float(proposal["proposed_end_ms"])
                replacement = yolo_line(
                    class_id=int(candidate["class_id"]),
                    start_ms=new_start,
                    end_ms=new_end,
                    duration_ms=float(candidate["source_duration_ms"]),
                )
                action = "expand_detector_consensus"
            else:
                raise ValueError(
                    f"Unsupported decision {action!r} for {candidate['candidate_id']}"
                )
            if replacement:
                rewritten.append(replacement)
            application_rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "label_path": relative_label,
                    "annotation_id": annotation_id,
                    "class_id": candidate["class_id"],
                    "class_name": candidate["class_name"],
                    "action": action,
                    "old_line": line,
                    "new_line": replacement,
                    "new_start_ms": "" if new_start is None else new_start,
                    "new_end_ms": "" if new_end is None else new_end,
                    "decision_revision": decision["revision"],
                    "decision_source": decision.get(
                        "decision_source", "human_legacy"
                    ),
                }
            )
        if set(by_annotation) - set(range(len(lines))):
            raise ValueError(f"Annotation index out of range in {relative_label}")
        label_path.write_text(
            "\n".join(rewritten) + ("\n" if rewritten else ""),
            encoding="utf-8",
        )
    return application_rows


def write_application_metadata(
    *,
    output_root: Path,
    application_rows: list[dict[str, Any]],
    parent_dataset: str,
    parent_manifest_sha256: str,
    review_session: str,
    review_queue_sha256: str,
    proposal_artifact: str,
    proposal_run_sha256: str,
) -> dict[str, Any]:
    with (output_root / "review_application.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(application_rows[0]))
        writer.writeheader()
        writer.writerows(application_rows)
    actions: dict[str, int] = {}
    for row in application_rows:
        action = str(row["action"])
        actions[action] = actions.get(action, 0) + 1
    summary = {
        "schema_version": 1,
        "parent_dataset": parent_dataset,
        "parent_manifest_sha256": parent_manifest_sha256,
        "review_session": review_session,
        "review_queue_sha256": review_queue_sha256,
        "proposal_artifact": proposal_artifact,
        "proposal_run_sha256": proposal_run_sha256,
        "candidate_count": len(application_rows),
        "actions": dict(sorted(actions.items())),
        "status": "review_decisions_applied_reference_candidate",
    }
    (output_root / "review_application_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    yaml_path = output_root / "dataset.yaml"
    metadata = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    metadata.setdefault("provenance", {})
    metadata["provenance"].update(
        {
            "parent_candidate_dataset": parent_dataset,
            "parent_candidate_manifest_sha256": parent_manifest_sha256,
            "saturation_gt_review_session": review_session,
            "saturation_gt_review_queue_sha256": review_queue_sha256,
            "detector_box_proposal_artifact": proposal_artifact,
            "detector_box_proposal_run_sha256": proposal_run_sha256,
            "annotation_policy": (
                "human keep/delete decisions applied; four disputed 10um "
                "intervals expanded from three-seed detector geometry consensus"
            ),
            "candidate_status": "reviewed_reference_not_promoted",
        }
    )
    metadata["review"] = {
        "candidate_count": len(application_rows),
        "actions": dict(sorted(actions.items())),
        "application_csv": "review_application.csv",
        "application_summary": "review_application_summary.json",
    }
    yaml_path.write_text(
        yaml.safe_dump(metadata, sort_keys=False),
        encoding="utf-8",
    )
    return summary


def promote_reviewed_dataset(
    *,
    source_root: Path,
    output_root: Path,
    source_dataset: str,
    source_manifest_sha256: str,
    reviewer: str,
    validated_at: str,
    evidence_plots: list[str],
    boundary_truncated_candidates: list[str],
) -> dict[str, Any]:
    """Copy a reviewed reference into a new, visually approved active payload."""
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite dataset: {output_root}")
    metadata_path = source_root / "dataset.yaml"
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    provenance = metadata.get("provenance", {})
    if provenance.get("candidate_status") != "reviewed_reference_not_promoted":
        raise ValueError("Source dataset is not a reviewed reference candidate")
    review = metadata.get("review", {})
    if review.get("actions") != {
        "delete": 172,
        "expand_detector_consensus": 4,
        "keep": 17,
    }:
        raise ValueError("Unexpected review action summary")
    if len(evidence_plots) != 4:
        raise ValueError("Exactly four disputed-box evidence plots are required")

    shutil.copytree(source_root, output_root, copy_function=shutil.copy2)
    promotion = {
        "schema_version": 1,
        "status": "approved_for_active_use",
        "reviewer": reviewer,
        "validated_at": validated_at,
        "source_dataset": source_dataset,
        "source_manifest_sha256": source_manifest_sha256,
        "decision": (
            "All four detector-consensus box expansions accepted after visual "
            "inspection; human 10um classes preserved."
        ),
        "evidence_plots": evidence_plots,
        "boundary_truncated_candidates": boundary_truncated_candidates,
        "boundary_policy": (
            "A box ending at the acquisition boundary covers the available "
            "event support and is flagged as truncated, not as a measured "
            "physical event end."
        ),
    }
    (output_root / "visual_validation.json").write_text(
        json.dumps(promotion, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    active_metadata = yaml.safe_load(
        (output_root / "dataset.yaml").read_text(encoding="utf-8")
    )
    active_metadata.setdefault("provenance", {})
    active_metadata["provenance"].update(
        {
            "promoted_from_dataset": source_dataset,
            "promoted_from_manifest_sha256": source_manifest_sha256,
            "candidate_status": "active_visual_validation_approved",
            "visual_validation": "visual_validation.json",
        }
    )
    active_metadata.setdefault("review", {})
    active_metadata["review"]["visual_validation"] = "visual_validation.json"
    active_metadata["review"]["boundary_truncated_candidates"] = (
        boundary_truncated_candidates
    )
    (output_root / "dataset.yaml").write_text(
        yaml.safe_dump(active_metadata, sort_keys=False),
        encoding="utf-8",
    )
    return promotion
