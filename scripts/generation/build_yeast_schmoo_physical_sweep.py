#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from internship_workspace.config import Workspace
from internship_workspace.datasets import load_records
from particles2snr.yeast_schmoo_simulation import (
    build_schmoo_calibration,
    build_schmoo_physical_sweep,
)


def _registered(records: list[dict[str, Any]], dataset_id: str) -> dict[str, Any]:
    row = next(
        (
            item
            for item in records
            if f"{item['id']}@{item['version']}" == dataset_id
        ),
        None,
    )
    if row is None or row["status"] not in {"active", "reference"}:
        raise ValueError(f"Registered usable dataset not found: {dataset_id}")
    return row


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Build the small thesis-informed S0/T0/M1 schmoo sweep."
    )
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument(
        "--source-dataset-id",
        default="yeast-events-representation@v3",
    )
    result.add_argument("--train-per-family", type=int, default=128)
    result.add_argument("--validation-per-family", type=int, default=64)
    result.add_argument("--test-per-family", type=int, default=192)
    result.add_argument("--seed", type=int, default=190726)
    return result


def main() -> None:
    args = parser().parse_args()
    workspace = Workspace.load()
    record = _registered(
        [item.payload for item in load_records(workspace)],
        args.source_dataset_id,
    )
    real_root = workspace.datasets_root / record["path"]
    calibration = build_schmoo_calibration(
        real_dataset_root=real_root,
        source_dataset_id=args.source_dataset_id,
    )
    summary = build_schmoo_physical_sweep(
        output_dir=args.output_dir,
        calibration=calibration,
        n_train_per_family=args.train_per_family,
        n_validation_per_family=args.validation_per_family,
        n_test_per_family=args.test_per_family,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
