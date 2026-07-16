#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from particles2snr.yeast_template_comparator import build_template_comparator


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a train-only template diagnostic comparator.")
    parser.add_argument("--followup-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-validation", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()
    summary = build_template_comparator(
        followup_root=args.followup_root,
        output_dir=args.output_dir,
        n_train=args.n_train,
        n_validation=args.n_validation,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
