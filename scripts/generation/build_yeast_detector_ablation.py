#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from particles2snr.yeast_events_ablation import (
    TemporalEnergyConfig,
    build_temporal_ablation_comparison,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and compare a temporal-energy ablation of the yeast detector."
    )
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--raw-dataset-root", type=Path, required=True)
    parser.add_argument("--current-candidate-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-split", default="development_train")
    parser.add_argument("--validation-split", default="development_validation")
    parser.add_argument("--match-tolerance-ms", type=float, default=0.5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = build_temporal_ablation_comparison(
        source_index_csv=args.source_index,
        raw_dataset_root=args.raw_dataset_root,
        current_candidate_csv=args.current_candidate_csv,
        output_dir=args.output_dir,
        config=TemporalEnergyConfig(),
        calibration_split=args.calibration_split,
        validation_split=args.validation_split,
        match_tolerance_ms=args.match_tolerance_ms,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
