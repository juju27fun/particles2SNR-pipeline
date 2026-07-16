#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from particles2snr.yeast_followup_dataset import build_followup_representation_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the prospective yeast SSL follow-up dataset from development data only."
    )
    parser.add_argument("--source-index-csv", type=Path, required=True)
    parser.add_argument("--source-dataset-id", required=True)
    parser.add_argument("--representation-root", type=Path, required=True)
    parser.add_argument("--representation-dataset-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()
    summary = build_followup_representation_dataset(
        source_index_csv=args.source_index_csv,
        representation_root=args.representation_root,
        output_dir=args.output_dir,
        source_dataset_id=args.source_dataset_id,
        representation_dataset_id=args.representation_dataset_id,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
