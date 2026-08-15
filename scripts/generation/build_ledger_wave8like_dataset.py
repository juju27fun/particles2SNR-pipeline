#!/usr/bin/env python3
"""Build the fold-specific ledger-Wave8like recipe dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from internship_workspace.config import Workspace
from internship_workspace.datasets import resolve_path, select_record
from particles2snr.ledger_wave8like import build_recipes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--ledger-root",
        type=Path,
        default=Path("datasets/interim/particles2SNR-pipeline/particles2snr-beads-gradual-supervision-development/v2"),
    )
    args = parser.parse_args()
    workspace = Workspace.load()
    noise = resolve_path(workspace, select_record(workspace, "noise", "v1"))
    summary = build_recipes(
        workspace=workspace.root,
        ledger_root=workspace.root / args.ledger_root,
        noise_root=noise,
        output_root=workspace.root / args.output,
    )
    print(summary)


if __name__ == "__main__":
    main()
