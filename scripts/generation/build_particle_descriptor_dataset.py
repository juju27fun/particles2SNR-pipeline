#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from internship_workspace.config import Workspace
from internship_workspace.datasets import load_records
from particles2snr.particle_descriptor_dataset import (
    build_particle_descriptor_dataset,
    validate_particle_descriptor_dataset,
)


def _record(key: str) -> dict:
    workspace = Workspace.load()
    records = [record.payload for record in load_records(workspace)]
    row = next(
        (item for item in records if f"{item['id']}@{item['version']}" == key),
        None,
    )
    if row is None or row["status"] not in {"active", "reference"}:
        raise ValueError(f"Eligible registered dataset not found: {key}")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build descriptor-ready F particle events without opening test."
    )
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--population-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--crop-policy",
        choices=("exact-centered", "shift-window"),
        default="exact-centered",
    )
    args = parser.parse_args()

    workspace = Workspace.load()
    source = _record(args.source_dataset)
    summary = build_particle_descriptor_dataset(
        source_root=workspace.datasets_root / source["path"],
        output_dir=args.output_dir,
        source_dataset_id=args.source_dataset,
        source_manifest_sha256=source["manifest_sha256"],
        population_id=args.population_id,
        crop_policy=args.crop_policy,
    )
    validation = validate_particle_descriptor_dataset(args.output_dir)
    print(json.dumps({"summary": summary, "validation": validation}, indent=2))


if __name__ == "__main__":
    main()
