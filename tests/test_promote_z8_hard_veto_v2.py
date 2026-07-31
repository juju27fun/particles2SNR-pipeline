from __future__ import annotations

import importlib.util
import json
import shutil
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/generation/promote_z8_hard_veto_v2.py"
SPEC = importlib.util.spec_from_file_location("promote_z8_hard_veto_v2", SCRIPT)
assert SPEC and SPEC.loader
promotion = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(promotion)


def _workspace(root: Path) -> SimpleNamespace:
    (root / "datasets/registry").mkdir(parents=True)
    (root / "datasets/registry/index.yaml").write_text("version: 1\ndatasets: []\n")
    return SimpleNamespace(root=root, datasets_root=root / "datasets", artifacts_root=root / "artifacts")


def _prepare(monkeypatch: pytest.MonkeyPatch, root: Path) -> tuple[SimpleNamespace, Path, Path, Path]:
    workspace = _workspace(root)
    candidate, destination, run_dir = promotion._paths(workspace)
    candidate.mkdir(parents=True)
    for name in promotion.EXPECTED_PAYLOAD:
        (candidate / name).write_text(name)
    rows = [{"path": name, "sha256": promotion.sha256_file(candidate / name)} for name in sorted(promotion.EXPECTED_PAYLOAD)]
    proof = {"candidate_tree_hash": "test", "payload_inventory": rows, "payload_inventory_hash": promotion.canonical_tree_hash(rows)}
    monkeypatch.setattr(promotion, "assert_supported_checkpoint", lambda *_: {"checkpoint": "r3", "receipt_sha256": "receipt"})
    monkeypatch.setattr(promotion, "verify_candidate", lambda *_: proof)
    return workspace, candidate, destination, run_dir


def test_canonical_tree_hash_is_stable() -> None:
    rows = [{"path": "b", "sha256": "2"}, {"path": "a", "sha256": "1"}]
    assert promotion.canonical_tree_hash(rows) == promotion.hashlib.sha256(b"a\t1\nb\t2").hexdigest()


def test_payload_inventory_rejects_extra_or_mutated_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"; candidate.mkdir()
    for name, digest in promotion.EXPECTED_PAYLOAD.items():
        (candidate / name).write_text(name)
        monkeypatch.setitem(promotion.EXPECTED_PAYLOAD, name, promotion.sha256_file(candidate / name))
    monkeypatch.setattr(promotion, "EXPECTED_TREE_HASH", promotion.canonical_tree_hash([{"path": name, "sha256": promotion.sha256_file(candidate / name)} for name in sorted(promotion.EXPECTED_PAYLOAD)]))
    assert len(promotion._payload_inventory(candidate)) == 4
    (candidate / "late.csv").write_text("late")
    with pytest.raises(ValueError, match="inventory"):
        promotion._payload_inventory(candidate)


def test_resumes_after_payload_move_crash(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace, candidate, destination, run_dir = _prepare(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError, match="crash"):
        promotion.promote(workspace, candidate=candidate, destination=destination, run_dir=run_dir, failure_hook=lambda stage: (_ for _ in ()).throw(RuntimeError("crash")) if stage == "after_payload_moved" else None)
    assert destination.is_dir()
    assert promotion.promote(workspace, candidate=candidate, destination=destination, run_dir=run_dir)["phase"] == "registry_active"


def test_resume_rejects_destination_mutation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace, candidate, destination, run_dir = _prepare(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError):
        promotion.promote(workspace, candidate=candidate, destination=destination, run_dir=run_dir, failure_hook=lambda stage: (_ for _ in ()).throw(RuntimeError("crash")) if stage == "after_payload_moved" else None)
    (destination / "events.csv").write_text("mutated")
    with pytest.raises(RuntimeError, match="destination bytes"):
        promotion.promote(workspace, candidate=candidate, destination=destination, run_dir=run_dir)


def test_staged_candidate_toctou_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace, candidate, destination, run_dir = _prepare(monkeypatch, tmp_path)
    def mutate(stage: str) -> None:
        if stage == "before_copy": (candidate / "late.csv").write_text("late")
    with pytest.raises(ValueError, match="byte-for-byte"):
        promotion.promote(workspace, candidate=candidate, destination=destination, run_dir=run_dir, failure_hook=mutate)
    assert not destination.exists()


def test_post_staging_mutation_is_rechecked_before_move(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace, candidate, destination, run_dir = _prepare(monkeypatch, tmp_path)
    def mutate(stage: str) -> None:
        if stage == "after_payload_staged":
            staging = destination.with_name(f".{destination.name}.promotion-staging")
            (staging / "events.csv").write_text("tampered after verification")
    with pytest.raises(ValueError, match="changed after byte verification"):
        promotion.promote(workspace, candidate=candidate, destination=destination, run_dir=run_dir, failure_hook=mutate)
    staging = destination.with_name(f".{destination.name}.promotion-staging")
    assert staging.is_dir() and not destination.exists()
    with pytest.raises(ValueError, match="byte-for-byte"):
        promotion.promote(workspace, candidate=candidate, destination=destination, run_dir=run_dir)


def test_resumes_after_release_manifest_write_before_move(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace, candidate, destination, run_dir = _prepare(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError, match="metadata crash"):
        promotion.promote(workspace, candidate=candidate, destination=destination, run_dir=run_dir, failure_hook=lambda stage: (_ for _ in ()).throw(RuntimeError("metadata crash")) if stage == "after_release_metadata_written" else None)
    staging = destination.with_name(f".{destination.name}.promotion-staging")
    assert (staging / "release_manifest.json").is_file()
    assert promotion.promote(workspace, candidate=candidate, destination=destination, run_dir=run_dir)["phase"] == "registry_active"


def test_release_metadata_mutation_is_quarantined_then_rebuilt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace, candidate, destination, run_dir = _prepare(monkeypatch, tmp_path)
    def mutate(stage: str) -> None:
        if stage == "after_release_metadata_written":
            staging = destination.with_name(f".{destination.name}.promotion-staging")
            (staging / "release_manifest.json").write_text("forged release")
    with pytest.raises(RuntimeError, match="destination bytes"):
        promotion.promote(workspace, candidate=candidate, destination=destination, run_dir=run_dir, failure_hook=mutate)
    assert not destination.exists()
    assert promotion.promote(workspace, candidate=candidate, destination=destination, run_dir=run_dir)["phase"] == "registry_active"
    assert (run_dir / "quarantine-invalid-staging").is_dir()


def test_payload_mutation_after_release_is_quarantined_then_rebuilt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace, candidate, destination, run_dir = _prepare(monkeypatch, tmp_path)
    def mutate(stage: str) -> None:
        if stage == "after_release_metadata_written":
            staging = destination.with_name(f".{destination.name}.promotion-staging")
            (staging / "events.csv").write_text("forged payload")
    with pytest.raises(RuntimeError, match="destination bytes"):
        promotion.promote(workspace, candidate=candidate, destination=destination, run_dir=run_dir, failure_hook=mutate)
    assert not destination.exists()
    assert promotion.promote(workspace, candidate=candidate, destination=destination, run_dir=run_dir)["phase"] == "registry_active"
    assert (run_dir / "quarantine-invalid-staging").is_dir()


def test_nested_late_payload_is_quarantined_then_rebuilt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace, candidate, destination, run_dir = _prepare(monkeypatch, tmp_path)
    def contaminate(stage: str) -> None:
        if stage == "after_release_metadata_written":
            staging = destination.with_name(f".{destination.name}.promotion-staging")
            nested = staging / "nested"; nested.mkdir()
            (nested / "unexpected.bin").write_bytes(b"late")
    with pytest.raises(RuntimeError, match="destination bytes"):
        promotion.promote(workspace, candidate=candidate, destination=destination, run_dir=run_dir, failure_hook=contaminate)
    assert not destination.exists()
    assert promotion.promote(workspace, candidate=candidate, destination=destination, run_dir=run_dir)["phase"] == "registry_active"
    assert (run_dir / "quarantine-invalid-staging/nested/unexpected.bin").is_file()


def test_registry_api_writer_waits_on_the_same_workspace_lock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace, candidate, destination, run_dir = _prepare(monkeypatch, tmp_path)
    other = workspace.datasets_root / "processed/other/v1"; other.mkdir(parents=True); (other / "x").write_text("x")
    started, finished = threading.Event(), threading.Event()
    def writer() -> None:
        started.set()
        promotion.register_record(workspace, dataset_id="other", version="v1", relative_path="processed/other/v1", status="active", producer="other", data_format="x", command="x")
        finished.set()
    with promotion.registry_lock(workspace):
        thread = threading.Thread(target=writer); thread.start(); assert started.wait(1)
        assert not finished.wait(0.05)
    thread.join(2)
    assert finished.is_set()


def test_receipt_bound_contract_rejects_forged_qualification_bundle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    source = Path(__file__).parents[2]
    checkpoint = tmp_path / "artifacts/cross-project/reviews/particle-z8-hard-veto-v2-result-r3"
    qualification = tmp_path / "artifacts/cross-project/analyses/particle-z8-hard-veto-v2-qualification-v2"
    shutil.copytree(source / promotion.CHECKPOINT, checkpoint)
    shutil.copytree(source / promotion.QUALIFICATION, qualification)
    summary = qualification / "summary_metrics.json"
    summary.write_text(json.dumps({"forged": True}))
    (qualification / "metrics_manifest.json").write_text(json.dumps({"analysis_run_id": qualification.name, "metrics": []}))
    (qualification / "run.json").write_text(json.dumps({"status": "complete", "computation_fingerprint": "forged"}))
    monkeypatch.setattr(promotion, "CHECKPOINT", checkpoint.relative_to(tmp_path))
    monkeypatch.setattr(promotion, "QUALIFICATION", qualification.relative_to(tmp_path))
    with pytest.raises(ValueError, match="qualification"):
        promotion.assert_supported_checkpoint(workspace)


def test_existing_record_must_match_every_immutable_field(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace, candidate, destination, run_dir = _prepare(monkeypatch, tmp_path)
    promotion.promote(workspace, candidate=candidate, destination=destination, run_dir=run_dir)
    index = workspace.datasets_root / "registry/index.yaml"
    index.write_text(index.read_text().replace("event-reference-table-z8-hard-veto-development-only", "tampered"))
    with pytest.raises(RuntimeError, match="immutable Z8 v2 release contract"):
        promotion.promote(workspace, candidate=candidate, destination=destination, run_dir=run_dir)


def test_noncanonical_paths_are_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace, candidate, destination, run_dir = _prepare(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="paths are fixed"):
        promotion.promote(workspace, candidate=candidate, destination=destination, run_dir=tmp_path / "other")
