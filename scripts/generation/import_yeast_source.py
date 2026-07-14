#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from particles2snr.yeast_source_import import import_verified_source


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import an audited yeast source as an immutable dataset payload.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-inventory", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = import_verified_source(
        source_root=args.source_root,
        destination=args.destination,
        inventory_csv=args.source_inventory,
        command=" ".join(sys.argv),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
