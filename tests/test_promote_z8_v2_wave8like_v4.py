from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/generation/promote_z8_v2_wave8like_v4.py"
SPEC = importlib.util.spec_from_file_location("promote_z8_v2_wave8like_v4", SCRIPT)
assert SPEC and SPEC.loader
promotion = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(promotion)


def _workspace(root: Path) -> SimpleNamespace:
    (root / "datasets/registry").mkdir(parents=True)
    (root / "datasets/registry/index.yaml").write_text("version: 1\ndatasets: []\n")
    return SimpleNamespace(root=root, datasets_root=root / "datasets", artifacts_root=root / "artifacts")


def _prepare(
    monkeypatch: pytest.MonkeyPatch, root: Path
) -> tuple[SimpleNamespace, Path, Path, Path]:
    workspace = _workspace(root)
    candidate, destination, run_dir = promotion._paths(workspace)
    candidate.mkdir(parents=True)
    (candidate / "payload.bin").write_bytes(b"approved")
    rows = [{"path": "payload.bin", "sha256": promotion.sha256_file(candidate / "payload.bin")}]
    monkeypatch.setattr(
        promotion,
        "assert_supported_checkpoint",
        lambda *_: {"checkpoint": "r4", "receipt_sha256": "receipt"},
    )
    monkeypatch.setattr(
        promotion,
        "verify_candidate",
        lambda *_: {
            "payload_inventory": rows,
            "payload_inventory_hash": promotion.canonical_tree_hash(rows),
        },
    )
    return workspace, candidate, destination, run_dir


def test_payload_inventory_rejects_extra_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    candidate = tmp_path / "candidate"
    for split in ("train", "val"):
        (candidate / split / "signals").mkdir(parents=True)
        (candidate / split / "labels").mkdir(parents=True)
    support = {
        "dataset-contract.json": b"contract",
        "dataset.yaml": b"yaml",
    }
    for name, payload in support.items():
        (candidate / name).write_bytes(payload)
    rows: list[dict[str, str]] = []
    for split, count in (("train", 1), ("val", 1)):
        long_id = f"{split}-0"
        signal = candidate / split / "signals" / f"{long_id}.npy"
        label = candidate / split / "labels" / f"{long_id}.txt"
        signal.write_bytes(f"{split}-signal".encode())
        label.write_bytes(f"{split}-label".encode())
        rows.append(
            {
                "split": split,
                "long_id": long_id,
                "signal_sha256": promotion.sha256_file(signal),
                "label_sha256": promotion.sha256_file(label),
            }
        )
    with (candidate / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "payload": {
            "dataset_contract_sha256": promotion.sha256_file(candidate / "dataset-contract.json"),
            "dataset_yaml_sha256": promotion.sha256_file(candidate / "dataset.yaml"),
            "manifest_csv_sha256": promotion.sha256_file(candidate / "manifest.csv"),
        }
    }
    (candidate / "dataset-manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(
        promotion,
        "EXPECTED_DATASET_MANIFEST_SHA256",
        promotion.sha256_file(candidate / "dataset-manifest.json"),
    )
    monkeypatch.setattr(promotion, "EXPECTED_ROWS", {"train": 1, "val": 1})
    assert len(promotion._payload_inventory(candidate)) == 8
    (candidate / "late.bin").write_bytes(b"late")
    with pytest.raises(ValueError, match="inventory"):
        promotion._payload_inventory(candidate)


def test_promotion_registers_reference_and_resumes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace, candidate, destination, run_dir = _prepare(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError, match="crash"):
        promotion.promote(
            workspace,
            candidate=candidate,
            destination=destination,
            run_dir=run_dir,
            failure_hook=lambda stage: (_ for _ in ()).throw(RuntimeError("crash"))
            if stage == "after_payload_moved"
            else None,
        )
    result = promotion.promote(
        workspace, candidate=candidate, destination=destination, run_dir=run_dir
    )
    assert result["status"] == "complete_reference_dataset"
    assert result["registry"]["status"] == "reference"


def test_noncanonical_paths_are_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace, candidate, destination, _ = _prepare(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="paths are fixed"):
        promotion.promote(
            workspace,
            candidate=candidate,
            destination=destination,
            run_dir=tmp_path / "other",
        )
