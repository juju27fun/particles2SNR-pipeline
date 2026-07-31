from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_builder_module():
    path = Path(__file__).parents[1] / "scripts/generation/build_z8_reference_event_table.py"
    spec = importlib.util.spec_from_file_location("z8_reference_builder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frozen_fixture(tmp_path: Path):
    builder = _load_builder_module()
    root = tmp_path / "workspace"
    artifacts = root / "artifacts"
    strict = artifacts / "particles2SNR-pipeline/runs/particles2snr-f-dual-clean-prefilter-v2-candidate"
    for split in ("train", "val"):
        path = strict / split / "data.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"data": [split]}), encoding="utf-8")
    run_path = strict / "run.json"
    run_path.write_text(
        json.dumps(
            {
                "run_id": builder.FROZEN_STRICT_RUN_ID,
                "status": builder.FROZEN_STRICT_RUN_STATUS,
                "dataset": builder.EXPECTED_PARENT_DATASET,
            }
        ),
        encoding="utf-8",
    )
    analysis = artifacts / "cross-project/analyses/particle-p2snrf-v2-repair-qualification-v4"
    analysis.mkdir(parents=True)
    tree = analysis / "dataset_tree_manifest.csv"
    tree.write_text(
        "kind,path,sha256\n"
        f"train_data_json,artifacts/particles2SNR-pipeline/runs/{builder.FROZEN_STRICT_RUN_ID}/train/data.json,{_sha(strict / 'train/data.json')}\n"
        f"val_data_json,artifacts/particles2SNR-pipeline/runs/{builder.FROZEN_STRICT_RUN_ID}/val/data.json,{_sha(strict / 'val/data.json')}\n",
        encoding="utf-8",
    )
    fingerprint = "frozen"
    metrics = [{"path": "dataset_tree_manifest.csv", "sha256": _sha(tree)}]
    (analysis / "run.json").write_text(
        json.dumps({"run_id": builder.P2_QUALIFICATION_RUN_ID, "status": "complete", "computation_fingerprint": fingerprint}),
        encoding="utf-8",
    )
    (analysis / "metrics_manifest.json").write_text(
        json.dumps(
            {
                "analysis_run_id": builder.P2_QUALIFICATION_RUN_ID,
                "computation_fingerprint": fingerprint,
                "metrics": metrics,
                "computation_provenance": {
                    "inputs": {f"artifacts/particles2SNR-pipeline/runs/{builder.FROZEN_STRICT_RUN_ID}/run.json": _sha(run_path)}
                },
            }
        ),
        encoding="utf-8",
    )
    result = artifacts / "cross-project/reviews/particle-p2snrf-v2-repair-result-r6"
    result.mkdir(parents=True)
    (result / "run.json").write_text(
        json.dumps(
            {
                "analysis_reference": {
                    "run_id": builder.P2_QUALIFICATION_RUN_ID,
                    "run_path": "artifacts/cross-project/analyses/particle-p2snrf-v2-repair-qualification-v4",
                    "computation_fingerprint": fingerprint,
                    "metrics": metrics,
                }
            }
        ),
        encoding="utf-8",
    )
    workspace = SimpleNamespace(root=root, artifacts_root=artifacts)
    return builder, workspace, result, strict


def test_frozen_strict_run_accepts_only_qualification_bound_candidate(tmp_path: Path) -> None:
    builder, workspace, result, strict = _frozen_fixture(tmp_path)
    proof = builder._require_frozen_strict_run(
        workspace,
        result_evidence_dir=result,
        strict_run=strict,
        strict_dataset_id=builder.EXPECTED_PARENT_DATASET,
    )
    assert proof["status"] == builder.FROZEN_STRICT_RUN_STATUS
    assert set(proof["split_data_sha256"]) == {"train_data_json", "val_data_json"}


def test_frozen_strict_run_rejects_status_or_train_val_replacement(tmp_path: Path) -> None:
    builder, workspace, result, strict = _frozen_fixture(tmp_path)
    run_path = strict / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["status"] = "complete"
    run_path.write_text(json.dumps(run), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match its frozen P2 contract"):
        builder._require_frozen_strict_run(
            workspace,
            result_evidence_dir=result,
            strict_run=strict,
            strict_dataset_id=builder.EXPECTED_PARENT_DATASET,
        )

    builder, workspace, result, strict = _frozen_fixture(tmp_path / "tampered")
    (strict / "val/data.json").write_text(json.dumps({"data": ["replacement"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="Strict train/val data differs"):
        builder._require_frozen_strict_run(
            workspace,
            result_evidence_dir=result,
            strict_run=strict,
            strict_dataset_id=builder.EXPECTED_PARENT_DATASET,
        )
