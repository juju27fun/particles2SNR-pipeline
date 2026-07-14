from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_inventory(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Source inventory is empty: {path}")
    required = {"relative_path", "size_bytes", "sha256"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Source inventory is missing columns: {', '.join(sorted(missing))}")
    return rows


def import_verified_source(
    *,
    source_root: Path,
    destination: Path,
    inventory_csv: Path,
    command: str,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    destination = destination.resolve()
    rows = read_inventory(inventory_csv)
    if destination.exists():
        raise ValueError(f"Refusing to overwrite existing destination: {destination}")

    destination.mkdir(parents=True)
    copied: list[str] = []
    try:
        for row in rows:
            relative = Path(row["relative_path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe inventory path: {relative}")
            source = source_root / relative
            if not source.is_file():
                raise ValueError(f"Source file is missing: {source}")
            if source.stat().st_size != int(row["size_bytes"]):
                raise ValueError(f"Source size changed since audit: {relative}")
            if _sha256(source) != row["sha256"]:
                raise ValueError(f"Source checksum changed since audit: {relative}")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if target.stat().st_size != int(row["size_bytes"]) or _sha256(target) != row["sha256"]:
                raise ValueError(f"Copied file verification failed: {relative}")
            copied.append(relative.as_posix())
    except Exception:
        shutil.rmtree(destination)
        raise

    provenance = {
        "schema_version": 1,
        "imported_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "source_inventory": str(inventory_csv.resolve()),
        "source_inventory_sha256": _sha256(inventory_csv),
        "command": command,
        "n_source_files": len(rows),
        "n_copied_files": len(copied),
        "verification": "size_and_sha256_before_and_after_copy",
        "immutability": "registered versions must not be modified in place",
    }
    (destination / "import_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return provenance
