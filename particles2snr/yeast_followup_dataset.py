from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SOURCE_SPLIT = "development_train"
FOLLOWUP_SPLITS = ("followup_train", "followup_validation", "followup_test")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_order(values: set[str], *, seed: int, group: str) -> list[str]:
    return sorted(
        values,
        key=lambda value: hashlib.sha256(
            f"yeast-followup-v1:{seed}:{group}:{value}".encode()
        ).hexdigest(),
    )


def assign_followup_splits(
    source_rows: list[dict[str, str]], *, seed: int = 20260716
) -> tuple[dict[str, str], dict[str, Any]]:
    """Split only the exhausted study's development pool by intact capture block."""
    eligible = [row for row in source_rows if row["development_split"] == SOURCE_SPLIT]
    if not eligible:
        raise ValueError(f"No rows from source split {SOURCE_SPLIT!r}")

    block_groups: dict[str, set[str]] = defaultdict(set)
    family_blocks: dict[str, set[str]] = defaultdict(set)
    for row in eligible:
        block_groups[row["capture_block_id"]].add(row["source_group"])
        family_blocks[row["duplicate_family_id"]].add(row["capture_block_id"])
    crossing_groups = {key: value for key, value in block_groups.items() if len(value) != 1}
    if crossing_groups:
        raise ValueError("Capture blocks cross source proxies")
    crossing_blocks = {key: value for key, value in family_blocks.items() if len(value) != 1}
    if crossing_blocks:
        raise ValueError("Duplicate families cross capture blocks")

    blocks_by_group: dict[str, set[str]] = defaultdict(set)
    for block_id, groups in block_groups.items():
        blocks_by_group[next(iter(groups))].add(block_id)

    assignments: dict[str, str] = {}
    limitations: list[dict[str, Any]] = []
    for group, block_ids in sorted(blocks_by_group.items()):
        ordered = _stable_order(block_ids, seed=seed, group=group)
        n_blocks = len(ordered)
        if n_blocks < 3:
            if n_blocks != 2:
                raise ValueError(
                    f"Source proxy {group!r} has {n_blocks} eligible block(s); at least two are required"
                )
            # Preserve one training and one prospective endpoint block. There is no
            # defensible way to manufacture an independent validation block.
            assignments[ordered[0]] = "followup_train"
            assignments[ordered[1]] = "followup_test"
            limitations.append(
                {
                    "source_group": group,
                    "n_blocks": n_blocks,
                    "missing_split": "followup_validation",
                    "consequence": "per-proxy validation metrics are unavailable",
                }
            )
            continue

        n_validation = max(1, int(round(0.20 * n_blocks)))
        n_test = max(1, int(round(0.20 * n_blocks)))
        while n_validation + n_test >= n_blocks:
            if n_validation >= n_test and n_validation > 1:
                n_validation -= 1
            else:
                n_test -= 1
        n_train = n_blocks - n_validation - n_test
        for index, block_id in enumerate(ordered):
            if index < n_train:
                split = "followup_train"
            elif index < n_train + n_validation:
                split = "followup_validation"
            else:
                split = "followup_test"
            assignments[block_id] = split

    split_rows = []
    for row in eligible:
        updated = dict(row)
        updated["prior_development_split"] = row["development_split"]
        updated["development_split"] = assignments[row["capture_block_id"]]
        split_rows.append(updated)

    audit = validate_followup_split(split_rows)
    audit.update(
        {
            "schema_version": 1,
            "policy": "deterministic-source-proxy-stratified-intact-capture-block-60-20-20-v1",
            "seed": seed,
            "source_split": SOURCE_SPLIT,
            "limitations": limitations,
            "status": "pass_with_declared_proxy_coverage_limitation" if limitations else "pass",
        }
    )
    return assignments, audit


def validate_followup_split(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Follow-up split is empty")
    invalid = sorted({row["development_split"] for row in rows} - set(FOLLOWUP_SPLITS))
    if invalid:
        raise ValueError(f"Unexpected follow-up splits: {invalid}")
    if any(row.get("prior_development_split") != SOURCE_SPLIT for row in rows):
        raise ValueError("Follow-up rows must come exclusively from development_train")

    crossings: dict[str, int] = {}
    for key in ("record_id", "capture_block_id", "duplicate_family_id"):
        grouped: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            grouped[row[key]].add(row["development_split"])
        crossings[key] = sum(len(splits) > 1 for splits in grouped.values())
    if any(crossings.values()):
        raise ValueError(f"Split leakage detected: {crossings}")

    canonical = [
        row
        for row in rows
        if str(row.get("is_canonical_duplicate_member", "")).lower() in {"true", "1"}
    ]
    if not canonical:
        canonical = rows
    coverage = {
        group: sorted({row["development_split"] for row in canonical if row["source_group"] == group})
        for group in sorted({row["source_group"] for row in canonical})
    }
    return {
        "n_rows": len(rows),
        "n_records": len({row["record_id"] for row in rows}),
        "n_capture_blocks": len({row["capture_block_id"] for row in rows}),
        "n_duplicate_families": len({row["duplicate_family_id"] for row in rows}),
        "crossing_counts": crossings,
        "split_counts": dict(sorted(Counter(row["development_split"] for row in canonical).items())),
        "source_group_split_coverage": coverage,
    }


def _distribution_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric = ("snr_proxy", "width_ms", "doppler_peak_hz", "energy_concentration")
    output: dict[str, Any] = {}
    for split in FOLLOWUP_SPLITS:
        selected = [row for row in rows if row["development_split"] == split]
        values: dict[str, Any] = {
            "n_events": len(selected),
            "n_records": len({row["record_id"] for row in selected}),
            "n_capture_blocks": len({row["capture_block_id"] for row in selected}),
            "source_group_counts": dict(sorted(Counter(row["source_group"] for row in selected).items())),
            "quality_counts": dict(sorted(Counter(row["quality"] for row in selected).items())),
        }
        for name in numeric:
            array = np.asarray([float(row[name]) for row in selected], dtype=np.float64)
            values[name] = {
                key: float(value)
                for key, value in zip(
                    ("p05", "p25", "p50", "p75", "p95"),
                    np.quantile(array, (0.05, 0.25, 0.50, 0.75, 0.95)),
                )
            }
        output[split] = values
    return output


def build_followup_representation_dataset(
    *,
    source_index_csv: Path,
    representation_root: Path,
    output_dir: Path,
    source_dataset_id: str,
    representation_dataset_id: str,
    seed: int = 20260716,
) -> dict[str, Any]:
    source_rows = _read_csv(source_index_csv)
    assignments, split_audit = assign_followup_splits(source_rows, seed=seed)
    eligible_source = {
        row["record_id"]: row
        for row in source_rows
        if row["development_split"] == SOURCE_SPLIT
    }

    event_csv = representation_root / "events.csv"
    event_rows = _read_csv(event_csv)
    selected: list[dict[str, Any]] = []
    rejected_split_counts: Counter[str] = Counter()
    for row in event_rows:
        if row["development_split"] != SOURCE_SPLIT:
            rejected_split_counts[row["development_split"]] += 1
            continue
        source = eligible_source.get(row["record_id"])
        if source is None:
            raise ValueError(f"Development event has no eligible source record: {row['record_id']}")
        if source["capture_block_id"] != row["capture_block_id"]:
            raise ValueError(f"Capture block mismatch for record {row['record_id']}")
        updated = dict(row)
        updated["prior_development_split"] = row["development_split"]
        updated["development_split"] = assignments[row["capture_block_id"]]
        selected.append(updated)
    if not selected:
        raise ValueError("No eligible development events")
    if any(row["prior_development_split"] != SOURCE_SPLIT for row in selected):
        raise AssertionError("A forbidden historical split entered the follow-up dataset")

    input_signals = np.load(representation_root / "signals.npy", mmap_mode="r")
    output_dir.mkdir(parents=True, exist_ok=False)
    output_signals = np.lib.format.open_memmap(
        output_dir / "signals.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(selected), input_signals.shape[1]),
    )
    train_sum = 0.0
    train_square_sum = 0.0
    train_count = 0
    for output_index, row in enumerate(selected):
        values = np.asarray(input_signals[int(row["signal_row"])], dtype=np.float32)
        output_signals[output_index] = values
        row["parent_signal_row"] = row["signal_row"]
        row["signal_row"] = output_index
        if row["development_split"] == "followup_train":
            values64 = values.astype(np.float64, copy=False)
            train_sum += float(values64.sum())
            train_square_sum += float(np.square(values64).sum())
            train_count += values64.size
    if train_count == 0:
        raise ValueError("followup_train has no samples for normalization")
    train_mean = train_sum / train_count
    train_variance = max(train_square_sum / train_count - train_mean**2, 0.0)
    train_std = float(np.sqrt(train_variance))
    if train_std <= 1.0e-12:
        raise ValueError("followup_train standard deviation is zero")
    for start in range(0, len(selected), 256):
        end = min(start + 256, len(selected))
        output_signals[start:end] = (output_signals[start:end] - train_mean) / train_std
    output_signals.flush()
    del output_signals

    eligible_split_rows = []
    for row in source_rows:
        if row["development_split"] != SOURCE_SPLIT:
            continue
        updated = dict(row)
        updated["prior_development_split"] = row["development_split"]
        updated["development_split"] = assignments[row["capture_block_id"]]
        eligible_split_rows.append(updated)
    with (output_dir / "source_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(eligible_split_rows[0]))
        writer.writeheader()
        writer.writerows(eligible_split_rows)
    with (output_dir / "events.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)

    parent_contract = json.loads(
        (representation_root / "input_contract.json").read_text(encoding="utf-8")
    )
    contract = {
        "schema_version": 1,
        "contract_id": "yeast-event-4096-followup-train-normalized-v2",
        "parent_contract": parent_contract["contract_id"],
        "parent_dataset": representation_dataset_id,
        "source_dataset": source_dataset_id,
        "source_split": SOURCE_SPLIT,
        "output_length": int(input_signals.shape[1]),
        "output_sampling_frequency_hz": parent_contract["output_sampling_frequency_hz"],
        "normalization": {
            "policy": "global followup_train mean and standard deviation",
            "parent_space_mean": train_mean,
            "parent_space_std": train_std,
            "algebraic_note": "Re-standardization cancels the parent global affine normalization.",
        },
        "forbidden_training_splits": [
            "in_session_test",
            "sealed_acquisition_test",
            "followup_test",
            "test",
        ],
        "historical_non_development_event_counts_not_copied": dict(sorted(rejected_split_counts.items())),
    }
    (output_dir / "input_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    event_integrity_rows = []
    for row in selected:
        source = eligible_source[row["record_id"]]
        event_integrity_rows.append(
            {
                "record_id": row["record_id"],
                "capture_block_id": row["capture_block_id"],
                "duplicate_family_id": source["duplicate_family_id"],
                "source_group": row["source_group"],
                "prior_development_split": row["prior_development_split"],
                "development_split": row["development_split"],
            }
        )
    event_integrity = validate_followup_split(event_integrity_rows)
    split_audit["event_integrity"] = event_integrity
    split_audit["source_manifest_sha256"] = _sha256(source_index_csv)
    split_audit["parent_events_sha256"] = _sha256(event_csv)
    split_audit["forbidden_event_signals_copied"] = 0
    (output_dir / "split_audit.json").write_text(
        json.dumps(split_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    distributions = _distribution_summary(selected)
    summary = {
        "schema_version": 1,
        "dataset_id": "yeast-events-followup@v1",
        "n_events": len(selected),
        "split_counts": dict(sorted(Counter(row["development_split"] for row in selected).items())),
        "source_group_counts": dict(sorted(Counter(row["source_group"] for row in selected).items())),
        "quality_counts": dict(sorted(Counter(row["quality"] for row in selected).items())),
        "n_records": len({row["record_id"] for row in selected}),
        "n_capture_blocks": len({row["capture_block_id"] for row in selected}),
        "input_contract": contract["contract_id"],
        "signals_shape": [len(selected), int(input_signals.shape[1])],
        "signals_dtype": "float32",
        "distributions": distributions,
        "split_status": split_audit["status"],
        "scientific_limitations": split_audit["limitations"],
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Yeast Follow-Up Split Audit",
        "",
        f"- Status: `{split_audit['status']}`",
        f"- Eligible source: `{representation_dataset_id}` / `{SOURCE_SPLIT}` only",
        f"- Events copied: `{len(selected)}`",
        "- Forbidden historical event signals copied: `0`",
        "- Record, capture-block, and duplicate-family crossings: `0`",
        "",
        "## Split Counts",
        "",
        "| Split | Events | Records | Blocks |",
        "|---|---:|---:|---:|",
    ]
    for split in FOLLOWUP_SPLITS:
        values = distributions[split]
        lines.append(
            f"| `{split}` | {values['n_events']} | {values['n_records']} | {values['n_capture_blocks']} |"
        )
    lines.extend(["", "## Declared Limitation", ""])
    for limitation in split_audit["limitations"]:
        lines.append(
            f"- `{limitation['source_group']}` has only {limitation['n_blocks']} eligible blocks; "
            f"`{limitation['missing_split']}` cannot contain this proxy without leakage."
        )
    (output_dir / "SPLIT_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
