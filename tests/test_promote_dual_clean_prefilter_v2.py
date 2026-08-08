from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/generation/promote_dual_clean_prefilter_v2.py"
SPEC = importlib.util.spec_from_file_location("promote_dual_clean_prefilter_v2", SCRIPT)
assert SPEC and SPEC.loader
promotion = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(promotion)


def _workspace(root: Path) -> SimpleNamespace:
    (root / "datasets/registry").mkdir(parents=True)
    (root / "datasets/registry/index.yaml").write_text("version: 1\ndatasets: []\n")
    return SimpleNamespace(root=root, datasets_root=root / "datasets", artifacts_root=root / "artifacts")


def _review_checkpoint(root: Path) -> None:
    checkpoint = root / promotion.CHECKPOINT
    (checkpoint / "review").mkdir(parents=True)
    asset = checkpoint / "asset.txt"
    asset.write_text("approved visual")
    asset_hash = promotion.sha256_file(asset)
    contract = {
        "schema_version": 1, "run_id": checkpoint.name, "case_ids": ["gate"],
        "required_fields": {"decision": ["supported", "revision_requested"]},
        "primary_assets": [{"path": "asset.txt", "sha256": asset_hash}],
    }
    (checkpoint / "review_contract.json").write_text(json.dumps(contract))
    decisions = {"schema_version": 1, "run_id": checkpoint.name, "reviewer": "Louis Tepe", "complete": True,
                 "decisions": {"gate": {"decision": "supported"}}, "metadata": {}, "updated_at": "2026-07-25T00:00:00+00:00"}
    decisions_path = checkpoint / "review/decisions.json"
    decisions_path.write_text(json.dumps(decisions))
    receipt = {"schema_version": 1, "run_id": checkpoint.name, "reviewer": "Louis Tepe", "completed_at": "2026-07-25T00:00:00+00:00",
               "decision_count": 1, "decisions_file": "review/decisions.json", "decisions_sha256": promotion.sha256_file(decisions_path),
               "contract_file": "review_contract.json", "contract_sha256": promotion.sha256_file(checkpoint / "review_contract.json"),
               "primary_assets": [{"path": "asset.txt", "sha256": asset_hash}]}
    (checkpoint / "review/receipt.json").write_text(json.dumps(receipt))
    (checkpoint / "run.json").write_text(json.dumps({"status": "visual_review_complete", "visual_protocol": {"service": "scientific_visual_v3"}, "visual_checkpoint": {"approved": True, "next_stage_blocked": False}}))


def _minimal_candidate(workspace: SimpleNamespace) -> tuple[Path, dict[str, object]]:
    candidate, _, _ = promotion._canonical_paths(workspace)
    candidate.mkdir(parents=True)
    (candidate / "dataset.yaml").write_text("dataset_id: particles2snr-f-dual-clean-c1-yolo-4class@v2\nstatus: candidate_pending_result_validation\npath: .\n")
    (candidate / "payload.bin").write_bytes(b"qualified")
    rows = [{"path": name, "sha256": promotion.sha256_file(candidate / name)} for name in ("dataset.yaml", "payload.bin")]
    proof: dict[str, object] = {"candidate_tree_hash": "qualified-tree", "tree_entries": 2, "payload_inventory": rows,
                                "payload_inventory_hash": promotion.canonical_tree_hash(rows)}
    return candidate, proof


def _prepare_transaction(monkeypatch: pytest.MonkeyPatch, root: Path) -> tuple[SimpleNamespace, Path, Path, Path]:
    workspace = _workspace(root)
    candidate, proof = _minimal_candidate(workspace)
    monkeypatch.setattr(promotion, "assert_supported_checkpoint", lambda *_: {"checkpoint": "approved", "receipt_sha256": "receipt"})
    monkeypatch.setattr(promotion, "verify_candidate", lambda *_: proof)
    _, destination, run_dir = promotion._canonical_paths(workspace)
    return workspace, candidate, destination, run_dir


def test_canonical_tree_hash_matches_qualification_algorithm() -> None:
    rows = [{"path": "b", "sha256": "2"}, {"path": "a", "sha256": "1"}]
    assert promotion.canonical_tree_hash(rows) == hashlib.sha256(b"a\t1\nb\t2").hexdigest()


def test_review_receipt_detects_decision_contract_and_asset_mutation(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    _review_checkpoint(tmp_path)
    assert promotion.assert_supported_checkpoint(workspace, promotion.CHECKPOINT)["reviewer"] == "Louis Tepe"
    (tmp_path / promotion.CHECKPOINT / "asset.txt").write_text("mutated")
    with pytest.raises(ValueError, match="receipt is invalid"):
        promotion.assert_supported_checkpoint(workspace, promotion.CHECKPOINT)


def test_qualified_inventory_rejects_extra_candidate_file(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    (candidate / "train/signals").mkdir(parents=True)
    (candidate / "train/labels").mkdir(parents=True)
    (candidate / "dataset.yaml").write_text("x")
    (candidate / "saturation_repair_manifest.csv").write_text("x")
    (candidate / "source_inventory.csv").write_text("split,filename\ntrain,a.npy\n")
    (candidate / "train/signals/a.npy").write_bytes(b"signal")
    (candidate / "train/labels/a.txt").write_text("")
    assert len(promotion.qualified_payload_inventory(candidate)) == 5
    (candidate / "unexpected.bin").write_bytes(b"no")
    with pytest.raises(ValueError, match="unknown=.*unexpected.bin"):
        promotion.qualified_payload_inventory(candidate)


def test_staged_copy_detects_candidate_mutation_after_qualification(tmp_path: Path) -> None:
    source = tmp_path / "source"
    copied = tmp_path / "copied"
    source.mkdir()
    (source / "one").write_text("before")
    rows = [{"path": "one", "sha256": promotion.sha256_file(source / "one")}]
    (source / "one").write_text("after")
    copied.mkdir()
    (copied / "one").write_text("after")
    with pytest.raises(ValueError, match="byte-for-byte"):
        promotion._verify_copy(rows, copied)


def test_promotion_resumes_after_payload_move_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace, candidate, destination, run_dir = _prepare_transaction(monkeypatch, tmp_path)
    def fail(stage: str) -> None:
        if stage == "after_payload_moved":
            raise RuntimeError("injected move crash")
    with pytest.raises(RuntimeError, match="move crash"):
        promotion.promote(workspace, candidate=candidate, destination=destination, run_dir=run_dir, failure_hook=fail)
    assert destination.is_dir()
    result = promotion.promote(workspace, candidate=candidate, destination=destination, run_dir=run_dir)
    assert result["phase"] == "registry_active"


def test_resume_rejects_mutated_payload_after_move(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace, candidate, destination, run_dir = _prepare_transaction(monkeypatch, tmp_path)
    def fail(stage: str) -> None:
        if stage == "after_payload_moved":
            raise RuntimeError("injected move crash")
    with pytest.raises(RuntimeError, match="move crash"):
        promotion.promote(workspace, candidate=candidate, destination=destination, run_dir=run_dir, failure_hook=fail)
    (destination / "payload.bin").write_bytes(b"mutated after move")
    with pytest.raises(RuntimeError, match="destination bytes differ"):
        promotion.promote(workspace, candidate=candidate, destination=destination, run_dir=run_dir)


def test_promotion_resumes_from_byte_verified_staging_payload(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace, candidate, destination, run_dir = _prepare_transaction(monkeypatch, tmp_path)
    def fail(stage: str) -> None:
        if stage == "after_payload_staged":
            raise RuntimeError("injected staged crash")
    with pytest.raises(RuntimeError, match="staged crash"):
        promotion.promote(workspace, candidate=candidate, destination=destination, run_dir=run_dir, failure_hook=fail)
    assert not destination.exists()
    assert destination.with_name(f".{destination.name}.promotion-staging").is_dir()
    assert promotion.promote(workspace, candidate=candidate, destination=destination, run_dir=run_dir)["phase"] == "registry_active"


def test_toctou_file_added_after_qualification_is_not_copied(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace, candidate, destination, run_dir = _prepare_transaction(monkeypatch, tmp_path)
    def add_file(stage: str) -> None:
        if stage == "before_copy":
            (candidate / "late.bin").write_bytes(b"late mutation")
    with pytest.raises(ValueError, match="staged payload inventory"):
        promotion.promote(workspace, candidate=candidate, destination=destination, run_dir=run_dir, failure_hook=add_file)
    assert not destination.exists()


def test_promotion_resumes_after_registry_manifest_partial_write(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace, candidate, destination, run_dir = _prepare_transaction(monkeypatch, tmp_path)
    native_register = promotion.register_record
    def partial_register(*args: object, **kwargs: object) -> object:
        promotion.build_manifest(destination, workspace.datasets_root / "registry" / f"{promotion.DATASET_ID}-{promotion.VERSION}.jsonl")
        (workspace.datasets_root / "registry/index.yaml").write_text("not: [valid")
        raise RuntimeError("injected manifest/index crash")
    monkeypatch.setattr(promotion, "register_record", partial_register)
    with pytest.raises(RuntimeError, match="manifest/index crash"):
        promotion.promote(workspace, candidate=candidate, destination=destination, run_dir=run_dir)
    assert (workspace.datasets_root / "registry" / f"{promotion.DATASET_ID}-{promotion.VERSION}.jsonl").exists()
    # A non-API torn write is not repaired by restoring a stale whole index.
    # Production register_record now writes atomically under the shared lock.
    assert (workspace.datasets_root / "registry/index.yaml").read_text() == "not: [valid"
    monkeypatch.setattr(promotion, "register_record", native_register)


def test_before_register_mutation_fails_and_registry_is_restored(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace, candidate, destination, run_dir = _prepare_transaction(monkeypatch, tmp_path)
    def mutate(stage: str) -> None:
        if stage == "before_register":
            (destination / "payload.bin").write_bytes(b"mutated before register")
    with pytest.raises(RuntimeError, match="destination bytes differ"):
        promotion.promote(workspace, candidate=candidate, destination=destination, run_dir=run_dir, failure_hook=mutate)
    assert (workspace.datasets_root / "registry/index.yaml").read_text() == "version: 1\ndatasets: []\n"
    assert not (workspace.datasets_root / "registry" / f"{promotion.DATASET_ID}-{promotion.VERSION}.jsonl").exists()


def test_record_scoped_rollback_preserves_unrelated_registration(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace, candidate, destination, run_dir = _prepare_transaction(monkeypatch, tmp_path)
    unrelated_id = "particles2snr-fbase-dual-clean-z8-events-3class-plus-unclear-development"
    unrelated = workspace.datasets_root / f"processed/{unrelated_id}/v2"
    unrelated.mkdir(parents=True); (unrelated / "payload.bin").write_bytes(b"unrelated")
    promotion.register_record(workspace, dataset_id=unrelated_id, version="v2", relative_path=f"processed/{unrelated_id}/v2", status="active", producer="other", data_format="binary", command="other")
    native_validate = promotion.validate_record
    def invalid_record(_workspace: object, record: object, **kwargs: object) -> list[str]:
        return ["injected invalid record"] if record.key == f"{promotion.DATASET_ID}@{promotion.VERSION}" else native_validate(_workspace, record, **kwargs)
    monkeypatch.setattr(promotion, "validate_record", invalid_record)
    with pytest.raises(RuntimeError, match="registered release validation failed"):
        promotion.promote(workspace, candidate=candidate, destination=destination, run_dir=run_dir)
    assert [record.key for record in promotion.load_records(workspace)] == [f"{unrelated_id}@v2"]
    assert not (workspace.datasets_root / "registry" / f"{promotion.DATASET_ID}-{promotion.VERSION}.jsonl").exists()
    monkeypatch.setattr(promotion, "validate_record", native_validate)
    assert promotion.promote(workspace, candidate=candidate, destination=destination, run_dir=run_dir)["phase"] == "registry_active"
    assert {record.key for record in promotion.load_records(workspace)} == {f"{unrelated_id}@v2", f"{promotion.DATASET_ID}@{promotion.VERSION}"}


def test_promotion_resumes_after_record_written(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace, candidate, destination, run_dir = _prepare_transaction(monkeypatch, tmp_path)
    def fail(stage: str) -> None:
        if stage == "after_record":
            raise RuntimeError("injected after-record crash")
    with pytest.raises(RuntimeError, match="after-record crash"):
        promotion.promote(workspace, candidate=candidate, destination=destination, run_dir=run_dir, failure_hook=fail)
    assert promotion.promote(workspace, candidate=candidate, destination=destination, run_dir=run_dir)["phase"] == "registry_active"


def test_failed_retry_never_removes_preexisting_active_record(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace, candidate, destination, run_dir = _prepare_transaction(monkeypatch, tmp_path)
    promotion.promote(workspace, candidate=candidate, destination=destination, run_dir=run_dir)
    manifest = workspace.datasets_root / "registry" / f"{promotion.DATASET_ID}-{promotion.VERSION}.jsonl"
    before = manifest.read_bytes()
    with pytest.raises(RuntimeError, match="retry failure"):
        promotion.promote(workspace, candidate=candidate, destination=destination, run_dir=run_dir, failure_hook=lambda stage: (_ for _ in ()).throw(RuntimeError("retry failure")) if stage == "before_register" else None)
    assert manifest.read_bytes() == before
    assert [record.key for record in promotion.load_records(workspace)] == [f"{promotion.DATASET_ID}@{promotion.VERSION}"]
    assert promotion.promote(workspace, candidate=candidate, destination=destination, run_dir=run_dir)["phase"] == "registry_active"


def test_noncanonical_paths_are_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace, candidate, destination, run_dir = _prepare_transaction(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="paths are fixed"):
        promotion.promote(workspace, candidate=candidate, destination=destination, run_dir=tmp_path / "other")
