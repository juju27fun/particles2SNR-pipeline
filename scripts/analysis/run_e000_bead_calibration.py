#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from particles2snr.e000_calibration import write_preflight_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen E000 revision 2 calibration protocol by phase."
    )
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=("preflight",),
        default="preflight",
        help="Only preflight is available until all frozen grouping gates pass.",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Skip per-file content hashes; intended only for fast diagnostics.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = (
        "scripts/analysis/run_e000_bead_calibration.py "
        f"--workspace-root {args.workspace_root} --output {args.output} "
        f"--phase {args.phase}"
        + (" --manifest-only" if args.manifest_only else "")
    )
    run = write_preflight_run(
        args.workspace_root,
        args.output,
        verify_content=not args.manifest_only,
        command=command,
    )
    print(json.dumps(run["normalized_outcome"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
