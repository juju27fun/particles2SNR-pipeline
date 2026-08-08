#!/usr/bin/env python3
"""Build an audited Particle2SNR Wave8-like detection dataset."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from particles2snr.wave8like_dataset import GenerationConfig, generate_dataset


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _generator_revision() -> str:
    module = Path(__file__).resolve().parents[2] / "particles2snr" / "wave8like_dataset.py"
    digest = hashlib.sha256(module.read_bytes()).hexdigest()
    return f"sha256:{digest}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("known3-positive", "fourclass-background"))
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--noise-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-dataset-id",
        default="particles2snr-f-dual-clean-c1-yolo-4class@v1",
    )
    parser.add_argument("--noise-dataset-id", default="noise@v1")
    parser.add_argument(
        "--output-dataset-id",
        default=None,
        help="Versioned registered ID written to dataset metadata. By default it is derived from --mode.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-groups", type=int, default=100)
    parser.add_argument("--val-groups", type=int, default=30)
    parser.add_argument("--test-groups", type=int, default=30)
    parser.add_argument("--positive-permutations", type=int, default=24)
    parser.add_argument("--background-share", type=float, default=0.25)
    parser.add_argument("--background-permutations", type=int, default=4)
    parser.add_argument(
        "--train-background-permutations",
        type=int,
        default=None,
        help="Optional train-only permutation count; validation/test retain --background-permutations.",
    )
    parser.add_argument(
        "--evaluation-background-share",
        type=float,
        default=None,
        help="Optional validation/test background share; defaults to --background-share.",
    )
    parser.add_argument(
        "--allow-background-source-reuse",
        action="store_true",
        help="Allow a source trace to appear in multiple background base groups. "
        "The default is disjoint groups for valid group-level uncertainty.",
    )
    parser.add_argument(
        "--source-eligibility-policy",
        choices=("fully_labeled_for_view", "legacy_any_safe_target"),
        default="fully_labeled_for_view",
        help=(
            "Strict mode excludes sources containing an omitted class or an "
            "event touched by edge replacement."
        ),
    )
    parser.add_argument("--noise-pad", type=int, default=300)
    parser.add_argument("--join-crossfade", type=int, default=300)
    parser.add_argument("--bandpass-low-hz", type=float, default=8_000.0)
    parser.add_argument("--bandpass-high-hz", type=float, default=500_000.0)
    parser.add_argument("--bandpass-order", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    workspace_root = _workspace_root().resolve()
    output = args.output.resolve()
    allowed_roots = (
        (workspace_root / "datasets" / "interim").resolve(),
        (workspace_root / "datasets" / "processed").resolve(),
    )
    if not any(output.is_relative_to(root) for root in allowed_roots):
        raise ValueError("--output must be under datasets/interim or datasets/processed")

    config = GenerationConfig(
        mode=args.mode,
        source_dataset_id=args.source_dataset_id,
        noise_dataset_id=args.noise_dataset_id,
        output_dataset_id=(
            args.output_dataset_id
            or (
                "particles2snr-wave8like-known3-positive@v1"
                if args.mode == "known3-positive"
                else "particles2snr-wave8like-fourclass-background@v1"
            )
        ),
        seed=args.seed,
        noise_pad=args.noise_pad,
        join_crossfade=args.join_crossfade,
        bandpass_low_hz=args.bandpass_low_hz,
        bandpass_high_hz=args.bandpass_high_hz,
        bandpass_order=args.bandpass_order,
        train_groups=args.train_groups,
        val_groups=args.val_groups,
        test_groups=args.test_groups,
        positive_permutations=args.positive_permutations,
        background_share=args.background_share,
        background_permutations=args.background_permutations,
        train_background_permutations=args.train_background_permutations,
        evaluation_background_share=args.evaluation_background_share,
        disjoint_background_groups=not args.allow_background_source_reuse,
        source_eligibility_policy=args.source_eligibility_policy,
        generator_revision=_generator_revision(),
    )
    metadata = generate_dataset(
        source_root=args.source_root.resolve(),
        noise_root=args.noise_root.resolve(),
        output_root=output,
        config=config,
    )
    print(f"Generated {args.mode}: {output}")
    for split, summary in metadata["splits"].items():
        print(
            f"  {split}: {summary['n_long_sequences']} rows "
            f"({summary['positive_rows']} positive, {summary['background_rows']} background)"
        )
    print(f"  manifest sha256: {metadata['manifest_sha256']}")


if __name__ == "__main__":
    main()
