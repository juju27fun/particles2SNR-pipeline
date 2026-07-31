#!/usr/bin/env python3
"""Build the approved development-only Z8 v2 Wave8-like candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from internship_workspace.config import Workspace
from internship_workspace.datasets import resolve_path, select_record
from internship_workspace.visual_review_store import ReviewStore
from particles2snr.z8_wave8like_dataset import (
    Z8Wave8LikeConfig,
    generate_dataset,
)


Z8_KEY = (
    "particles2snr-fbase-dual-clean-z8-events-3class-plus-unclear-development@v2"
)
PARENT_KEY = "particles2snr-f-dual-clean-c1-yolo-4class@v2"
NOISE_KEY = "noise@v1"
OUTPUT_DATASET = "particles2snr-z8-v2-wave8like-known3-background-development"
METHOD_RUN = Path(
    "artifacts/cross-project/reviews/"
    "particle-z8-v2-wave8like-generation-method-r1"
)


def _git_state(path: Path) -> dict[str, Any]:
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


def _write_run_atomically(
    run_root: Path,
    payload: dict[str, object],
    *,
    output_key: str,
    bridge_matching: str,
    endpoint_quality_enabled: bool,
) -> None:
    run_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{run_root.name}.tmp-", dir=run_root.parent)
    )
    try:
        (temporary / "run.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "REPORT.md").write_text(
            "\n".join(
                [
                    "# Z8 v2 Wave8-like development candidate",
                    "",
                    f"- Candidate: `{output_key}`",
                    "- Splits: train and validation only; sealed test not accessed.",
                    "- Join: separately filtered 600-sample continuous-noise bridge",
                    "  with 300-sample raised-cosine guards on each side.",
                    f"- Bridge amplitude matching: `{bridge_matching}`.",
                    (
                        "- Both source endpoints must pass the annotation-free "
                        "900-sample RMS/peak quality gate."
                        if endpoint_quality_enabled
                        else "- Source endpoint quality gate: not enabled."
                    ),
                    "- Base groups are source-disjoint within each split.",
                    "- Status: awaiting receipt-backed visual join audit.",
                    "- Training and registry promotion remain blocked.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, run_root)
    except BaseException:
        import shutil

        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-version",
        choices=("v1", "v2", "v3", "v4"),
        default="v1",
    )
    args = parser.parse_args()
    workspace = Workspace.load()
    output_key = f"{OUTPUT_DATASET}@{args.candidate_version}"
    output_path = Path(
        "datasets/interim/particles2SNR-pipeline"
    ) / OUTPUT_DATASET / args.candidate_version
    run_id = f"particle-z8-v2-wave8like-generation-{args.candidate_version}"
    run_path = Path(f"artifacts/particles2SNR-pipeline/runs/{run_id}")
    bridge_matching = {
        "v1": "none",
        "v2": "robust-local-rms",
        "v3": "robust-local-rms-global-cap",
        "v4": "robust-local-rms-global-cap",
    }[args.candidate_version]
    output_root = (workspace.root / output_path).resolve()
    run_root = (workspace.root / run_path).resolve()
    for path in (output_root, run_root):
        if path.exists():
            raise FileExistsError(f"refusing to mutate existing output: {path}")

    method_run = workspace.root / METHOD_RUN / "run.json"
    method_payload = json.loads(method_run.read_text(encoding="utf-8"))
    if (
        method_payload.get("status") != "visual_review_complete"
        or method_payload.get("visual_approval", {}).get("decision") != "approved"
        or method_payload.get("visual_checkpoint", {}).get("next_stage_blocked")
    ):
        raise RuntimeError("approved generation method checkpoint is required")
    revision_evidence = None
    if args.candidate_version != "v1":
        revision_number = int(args.candidate_version.removeprefix("v")) - 1
        revision_root = (
            workspace.root
            / "artifacts/cross-project/reviews"
            / f"particle-z8-v2-wave8like-join-audit-result-r{revision_number}"
        )
        receipt = ReviewStore(revision_root).verify_receipt()
        decisions = json.loads(
            (revision_root / "review/decisions.json").read_text(encoding="utf-8")
        )
        decision = decisions.get("decisions", {}).get(
            "particle-z8-v2-wave8like-join-audit-result", {}
        )
        expected_text = {
            "v2": "robust local RMS matching",
            "v3": "global annotation-free robust RMS",
            "v4": "endpoint-quality feasibility audit",
        }[args.candidate_version]
        if (
            decision.get("decision") != "revision_requested"
            or expected_text not in decision.get("comment", "")
        ):
            raise RuntimeError(
                f"{args.candidate_version} requires its approved join revision"
            )
        revision_evidence = {
            "run_id": receipt["run_id"],
            "receipt_sha256": hashlib.sha256(
                (revision_root / "review/receipt.json").read_bytes()
            ).hexdigest(),
            "decision": decision,
        }

    records = {}
    roots = {}
    for key in (Z8_KEY, PARENT_KEY, NOISE_KEY):
        dataset_id, version = key.rsplit("@", 1)
        record = select_record(workspace, dataset_id, version)
        records[key] = record
        roots[key] = resolve_path(workspace, record)

    module_path = (
        workspace.root
        / "particles2SNR-pipeline/particles2snr/z8_wave8like_dataset.py"
    )
    revision = f"sha256:{hashlib.sha256(module_path.read_bytes()).hexdigest()}"
    config = Z8Wave8LikeConfig(
        output_dataset_id=output_key,
        z8_dataset_id=Z8_KEY,
        parent_dataset_id=PARENT_KEY,
        noise_dataset_id=NOISE_KEY,
        bridge_matching=bridge_matching,
        endpoint_quality_enabled=args.candidate_version == "v4",
        generator_revision=revision,
    )
    manifest = generate_dataset(
        z8_root=roots[Z8_KEY],
        parent_root=roots[PARENT_KEY],
        noise_root=roots[NOISE_KEY],
        output_root=output_root,
        config=config,
        verify_replay=True,
    )
    created_at = datetime.now(timezone.utc).isoformat()
    run = {
        "schema_version": 1,
        "run_id": run_id,
        "project": "particles2SNR-pipeline",
        "kind": "dataset-generation",
        "created_at": created_at,
        "status": "complete_candidate_awaiting_visual_join_audit",
        "dataset": output_key,
        "command": (
            f"{Path(__file__).relative_to(workspace.root).as_posix()} "
            f"--candidate-version {args.candidate_version}"
        ),
        "repositories": {
            "workspace": _git_state(workspace.root),
            "particles2SNR-pipeline": _git_state(
                workspace.root / "particles2SNR-pipeline"
            ),
        },
        "method_evidence": {
            "run_id": method_payload["run_id"],
            "run_sha256": hashlib.sha256(method_run.read_bytes()).hexdigest(),
            "receipt": str(METHOD_RUN / "review/receipt.json"),
        },
        "revision_evidence": revision_evidence,
        "parents": {
            key: records[key].payload["manifest_sha256"]
            for key in (Z8_KEY, PARENT_KEY, NOISE_KEY)
        },
        "outputs": [
            output_root.relative_to(workspace.root).as_posix(),
            "REPORT.md",
        ],
        "dataset_manifest_sha256": hashlib.sha256(
            (output_root / "dataset-manifest.json").read_bytes()
        ).hexdigest(),
        "summary": {
            "audit": manifest["audit"],
            "deterministic_replay": manifest["deterministic_replay"],
            "promotion": manifest["promotion"],
            "sealed_test_accessed": False,
        },
    }
    _write_run_atomically(
        run_root,
        run,
        output_key=output_key,
        bridge_matching=bridge_matching,
        endpoint_quality_enabled=config.endpoint_quality_enabled,
    )
    print(
        json.dumps(
            {
                "candidate": output_key,
                "path": output_root.relative_to(workspace.root).as_posix(),
                "run_id": run_id,
                "audit": manifest["audit"],
                "deterministic_replay": manifest["deterministic_replay"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
