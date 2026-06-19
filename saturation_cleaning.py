"""Non-destructive saturation interval cleaning helpers.

The functions here never mutate source arrays in place. They are shared by the
particles2SNR command-line cleaning tool and the P1 long-sequence generator.
"""

import argparse
import csv
import json
import os

import numpy as np

from detect_saturation import detect_saturation


def merge_intervals(intervals, signal_len):
    clipped = []
    for start, end in intervals:
        s = max(0, min(int(signal_len), int(start)))
        e = max(0, min(int(signal_len), int(end)))
        if e > s:
            clipped.append((s, e))
    if not clipped:
        return []
    clipped.sort()
    merged = [clipped[0]]
    for start, end in clipped[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def expand_intervals(intervals, signal_len, guard_before=0, guard_after=0):
    expanded = [
        (int(row["start_sample"]) - int(guard_before),
         int(row["end_sample"]) + int(guard_after))
        for row in intervals
    ]
    return merge_intervals(expanded, signal_len)


def event_overlaps_intervals(event, intervals):
    left, right, _cid = event
    for start, end in intervals:
        if left < end and right > start:
            return True
    return False


def drop_overlapping_events(events, intervals):
    kept, dropped = [], []
    for event in events:
        if event_overlaps_intervals(event, intervals):
            dropped.append(event)
        else:
            kept.append(event)
    return kept, dropped


def _noise_fill(length, noise_pool, rng):
    if not noise_pool:
        return np.zeros(length, dtype=np.float64)
    candidates = [chunk for chunk in noise_pool if len(chunk) >= length]
    if candidates:
        chunk = candidates[int(rng.integers(0, len(candidates)))]
        start_max = len(chunk) - length
        start = int(rng.integers(0, start_max + 1)) if start_max > 0 else 0
        return np.asarray(chunk[start:start + length], dtype=np.float64)
    out = np.empty(length, dtype=np.float64)
    pos = 0
    while pos < length:
        chunk = np.asarray(noise_pool[int(rng.integers(0, len(noise_pool)))], dtype=np.float64)
        take = min(len(chunk), length - pos)
        out[pos:pos + take] = chunk[:take]
        pos += take
    return out


def clean_signal_non_destructive(signal, unsafe_intervals, policy="replace",
                                 noise_pool=None, rng=None, mask_value=0.0):
    """Return a cleaned copy of ``signal`` and per-interval action rows.

    ``policy``:
      - ``replace``: fill unsafe intervals from ``noise_pool``.
      - ``mask``: fill unsafe intervals with ``mask_value``.
      - ``keep``: do not alter samples, but still report intervals.
    """
    rng = rng or np.random.default_rng(0)
    out = np.asarray(signal).copy()
    actions = []
    for idx, (start, end) in enumerate(unsafe_intervals):
        length = end - start
        action = "kept"
        if policy == "replace":
            out[start:end] = _noise_fill(length, noise_pool or [], rng)
            action = "replaced_with_noise"
        elif policy == "mask":
            out[start:end] = mask_value
            action = "masked"
        elif policy == "keep":
            action = "reported_only"
        else:
            raise ValueError(f"Unknown saturation cleaning policy: {policy}")
        actions.append({
            "interval_idx": idx,
            "start_sample": int(start),
            "end_sample": int(end),
            "duration_samples": int(length),
            "action": action,
        })
    return out, actions


def detect_unsafe_intervals(signal, fs=2_000_000, fmin=7000, fmax=80000,
                            min_flat=500, zero_threshold=1e-4,
                            guard_before=0, guard_after=0):
    sat_info = detect_saturation(
        signal,
        fs=fs,
        fmin=fmin,
        fmax=fmax,
        min_flat=min_flat,
        zero_threshold=zero_threshold,
    )
    unsafe = expand_intervals(
        sat_info["intervals"],
        len(signal),
        guard_before=guard_before,
        guard_after=guard_after,
    )
    return sat_info, unsafe


def read_noise_pool(noise_dir, chunk_len):
    chunks = []
    if not noise_dir:
        return chunks
    for fname in sorted(os.listdir(noise_dir)):
        if not fname.endswith(".npy"):
            continue
        arr = np.load(os.path.join(noise_dir, fname))
        if len(arr) < chunk_len:
            chunks.append(arr.astype(np.float64))
            continue
        n = len(arr) // chunk_len
        for i in range(n):
            chunks.append(arr[i * chunk_len:(i + 1) * chunk_len].astype(np.float64))
    return chunks


def write_manifest_csv(path, rows):
    fieldnames = [
        "source_path", "output_path", "class", "policy", "interval_idx",
        "start_sample", "end_sample", "duration_samples", "action",
        "dropped_events", "fs", "fmin", "fmax", "min_flat",
        "zero_threshold", "guard_before", "guard_after",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def iter_npy_files(input_dir):
    for root, _dirs, files in os.walk(input_dir):
        for fname in sorted(files):
            if fname.endswith(".npy"):
                yield os.path.join(root, fname)


def main():
    parser = argparse.ArgumentParser(
        description="Create a non-destructive cleaned copy of saturated .npy files."
    )
    parser.add_argument("input_dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--noise-dir", default=None,
                        help="Noise pool for --policy replace")
    parser.add_argument("--policy", choices=("replace", "mask", "keep"), default="replace")
    parser.add_argument("--fs", type=float, default=2_000_000)
    parser.add_argument("--fmin", type=float, default=7000)
    parser.add_argument("--fmax", type=float, default=80000)
    parser.add_argument("--min-flat", type=int, default=500)
    parser.add_argument("--zero-threshold", type=float, default=1e-4)
    parser.add_argument("--guard-before", type=int, default=0)
    parser.add_argument("--guard-after", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    noise_pool = read_noise_pool(args.noise_dir, max(1, args.guard_before + args.guard_after))

    manifest_rows = []
    summary = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "policy": args.policy,
        "files_seen": 0,
        "files_with_saturation": 0,
        "intervals": 0,
        "source_files_untouched": True,
    }

    for source_path in iter_npy_files(input_dir):
        summary["files_seen"] += 1
        rel = os.path.relpath(source_path, input_dir)
        output_path = os.path.join(output_dir, rel)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        signal = np.load(source_path)
        sat_info, unsafe = detect_unsafe_intervals(
            signal,
            fs=args.fs,
            fmin=args.fmin,
            fmax=args.fmax,
            min_flat=args.min_flat,
            zero_threshold=args.zero_threshold,
            guard_before=args.guard_before,
            guard_after=args.guard_after,
        )
        if unsafe:
            summary["files_with_saturation"] += 1
        summary["intervals"] += len(unsafe)
        cleaned, actions = clean_signal_non_destructive(
            signal,
            unsafe,
            policy=args.policy,
            noise_pool=noise_pool,
            rng=rng,
        )
        np.save(output_path, cleaned)
        class_name = os.path.dirname(rel).split(os.sep)[0] if os.path.dirname(rel) else ""
        for action in actions:
            manifest_rows.append({
                "source_path": source_path,
                "output_path": output_path,
                "class": class_name,
                "policy": args.policy,
                "dropped_events": 0,
                "fs": args.fs,
                "fmin": args.fmin,
                "fmax": args.fmax,
                "min_flat": args.min_flat,
                "zero_threshold": args.zero_threshold,
                "guard_before": args.guard_before,
                "guard_after": args.guard_after,
                **action,
            })
        if not sat_info["is_saturated"] and not actions:
            # Preserve a complete derived copy so downstream scripts can point
            # only at output_dir.
            pass

    manifest_path = os.path.join(output_dir, "saturation_cleaning_manifest.csv")
    summary_path = os.path.join(output_dir, "saturation_cleaning_summary.json")
    write_manifest_csv(manifest_path, manifest_rows)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Derived cleaned dataset: {output_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
