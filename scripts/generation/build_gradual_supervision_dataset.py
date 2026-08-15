#!/usr/bin/env python3
"""Build the reference-only beads gradual-supervision dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from particles2snr.gradual_supervision import BuildInputs, build_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dataset-id",
        default="particles2snr-beads-gradual-supervision-development@v1",
    )
    parser.add_argument(
        "--audit-analysis-id",
        default="particle-spectral-npy-audit60-analysis-r1",
    )
    parser.add_argument(
        "--ledger-analysis-id",
        default="particle-spectral-npy-targeted-ledger-analysis-r1",
    )
    parser.add_argument(
        "--method-receipt",
        type=Path,
        default=Path("artifacts/particles2SNR-pipeline/reviews/particle-gradual-supervision-method-r1/review/receipt.json"),
    )
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    analysis = workspace / "artifacts/particles2SNR-pipeline/analysis"
    dataset_root = workspace / "datasets/processed"
    synthetic_root = dataset_root / "particles2SNR-pipeline/particles2snr-z8-v2-wave8like-known3-background-development/v4"
    dual_root = dataset_root / "particles2snr-f-dual-clean-c1-yolo-4class/v2"
    build_dataset(
        BuildInputs(
            workspace=workspace,
            source_matrix=analysis / "particle-event-inventory-matrix-analysis-r1/source_matrix.csv",
            proposals=analysis / "particle-event-inventory-matrix-analysis-r1/proposals.csv",
            audit_traces=analysis / args.audit_analysis_id / "trace_outcomes.csv",
            ledger_events=analysis / args.ledger_analysis_id / "event_outcomes_r2.csv",
            ledger_summary=analysis / args.ledger_analysis_id / "summary.json",
            atlas=analysis / "particle-c2-mad-calibration-atlas-r3/atlas_cases.json",
            synthetic_manifest=synthetic_root / "manifest.csv",
            synthetic_root=synthetic_root,
            dual_clean_root=dual_root,
            method_receipt=(workspace / args.method_receipt).resolve(),
            dataset_id=args.dataset_id,
        ),
        args.output,
    )


if __name__ == "__main__":
    main()
