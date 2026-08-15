"""Non-destructive saturation interval cleaning helpers.

The functions here never mutate source arrays in place. They are shared by the
particles2SNR command-line cleaning tool and the P1 long-sequence generator.
"""

import argparse
import csv
import json
import os

import numpy as np
from scipy.signal import butter, filtfilt

from particles2snr.detect_saturation import detect_saturation


SATURATION_REPAIR_METHODS = {
    "direct",
    "cosine-pre-filter",
    "cosine-filtered-domain",
}


def proposal_center_inside_intervals(start_sample, end_sample, intervals):
    """Return the proposal midpoint and its first inclusive interval match.

    This is the public form of the frozen z8v2 hard-veto geometry.  It uses
    proposal bounds, not a detector-provided peak, so every caller applies the
    same decision rule.
    """
    center_sample = (float(start_sample) + float(end_sample)) / 2.0
    for index, interval in enumerate(intervals):
        left, right = (int(value) for value in interval)
        if left <= center_sample <= right:
            return center_sample, index, (left, right)
    return center_sample, None, None


def butter_bandpass_filter(signal, fs=2_000_000, fmin=7000, fmax=80000,
                           order=4):
    """Apply the same zero-phase Butterworth used by the dual-clean generator."""
    arr = np.asarray(signal)
    if arr.ndim != 1:
        arr = np.squeeze(arr)
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1D signal, got shape {arr.shape}")
    nyquist = float(fs) / 2.0
    low = max(0.001, min(float(fmin) / nyquist, 0.99))
    high = max(low + 0.001, min(float(fmax) / nyquist, 0.99))
    b, a = butter(int(order), [low, high], btype="band")
    return filtfilt(b, a, arr).astype(arr.dtype, copy=False)


def cosine_blend_replacement(signal, replacement, *, core_interval,
                             expanded_interval):
    """Blend one replacement carrier into a signal across explicit guards.

    ``replacement`` covers the entire expanded interval. The core is copied
    from it, while raised-cosine transitions blend the original and
    replacement values in the left and right guards.
    """
    source = np.asarray(signal)
    carrier = np.asarray(replacement)
    if source.ndim != 1 or carrier.ndim != 1:
        raise ValueError("signal and replacement must be one-dimensional")
    core_start, core_end = (int(value) for value in core_interval)
    expanded_start, expanded_end = (int(value) for value in expanded_interval)
    if not (
        0 <= expanded_start <= core_start < core_end <= expanded_end <= len(source)
    ):
        raise ValueError("core and expanded intervals are inconsistent")
    if len(carrier) != expanded_end - expanded_start:
        raise ValueError("replacement length must match the expanded interval")

    output = source.copy()
    left_guard = core_start - expanded_start
    right_guard = expanded_end - core_end
    core_left = core_start - expanded_start
    core_right = core_end - expanded_start

    if left_guard:
        phase = (
            np.linspace(0.0, np.pi, left_guard, endpoint=True)
            if left_guard > 1
            else np.asarray([np.pi / 2.0])
        )
        replacement_weight = 0.5 - 0.5 * np.cos(phase)
        original = source[expanded_start:core_start]
        replacement_left = carrier[:left_guard]
        output[expanded_start:core_start] = (
            (1.0 - replacement_weight) * original
            + replacement_weight * replacement_left
        )
    output[core_start:core_end] = carrier[core_left:core_right]
    if right_guard:
        phase = (
            np.linspace(0.0, np.pi, right_guard, endpoint=True)
            if right_guard > 1
            else np.asarray([np.pi / 2.0])
        )
        replacement_weight = 0.5 + 0.5 * np.cos(phase)
        original = source[core_end:expanded_end]
        replacement_right = carrier[core_right:]
        output[core_end:expanded_end] = (
            replacement_weight * replacement_right
            + (1.0 - replacement_weight) * original
        )
    return output.astype(source.dtype, copy=False)


def repair_saturation_intervals_pre_filter(
    signal,
    replacements,
    *,
    fs=2_000_000,
    fmin=7000,
    fmax=80000,
    order=4,
):
    """Repair disjoint saturation regions before one canonical bandpass pass."""
    source = np.asarray(signal)
    rows = sorted(
        list(replacements),
        key=lambda row: tuple(int(value) for value in row["expanded_interval"]),
    )
    previous_end = -1
    clean = source.copy()
    regions = []
    for row in rows:
        core_interval = tuple(int(value) for value in row["core_interval"])
        expanded_interval = tuple(
            int(value) for value in row["expanded_interval"]
        )
        if expanded_interval[0] < previous_end:
            raise ValueError("expanded saturation intervals must be disjoint")
        previous_end = expanded_interval[1]
        clean = cosine_blend_replacement(
            clean,
            np.asarray(row["replacement"]),
            core_interval=core_interval,
            expanded_interval=expanded_interval,
        )
        regions.append(
            {
                "core_interval": list(core_interval),
                "expanded_interval": list(expanded_interval),
            }
        )
    filtered = butter_bandpass_filter(
        clean, fs=fs, fmin=fmin, fmax=fmax, order=order
    )
    return {
        "method": "cosine-pre-filter",
        "clean_signal": clean.astype(source.dtype, copy=False),
        "filtered_signal": filtered.astype(source.dtype, copy=False),
        "regions": regions,
    }


def forward_backward_filter_response_radius(
    *,
    signal_length=16_384,
    fs=2_000_000,
    fmin=7000,
    fmax=80000,
    order=4,
    mass_fraction=0.999,
):
    """Return the symmetric radius containing a fixed L1 response mass.

    The centered impulse is filtered with the exact zero-phase filter used by
    the generator.  This produces a filter-derived boundary guard rather than
    calibrating one on reviewed annotations.
    """
    length = int(signal_length)
    fraction = float(mass_fraction)
    if length < 3:
        raise ValueError("signal_length must be at least 3")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("mass_fraction must be in (0, 1]")
    center = length // 2
    impulse = np.zeros(length, dtype=np.float64)
    impulse[center] = 1.0
    response = np.abs(
        butter_bandpass_filter(
            impulse, fs=fs, fmin=fmin, fmax=fmax, order=order
        )
    )
    total = float(np.sum(response))
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError("invalid forward-backward filter impulse response")
    for radius in range(max(center, length - center)):
        left = max(0, center - radius)
        right = min(length, center + radius + 1)
        if float(np.sum(response[left:right])) / total >= fraction:
            return radius
    return length - 1


def boundary_proposal_decision(
    *,
    center_sample,
    expanded_intervals,
    response_radius,
    clean_local_peak_z,
    clean_local_min_z=1.5,
    clean_peak_center_sample=None,
    clean_peak_max_alignment_samples=None,
):
    """Apply the filter-derived boundary veto to one detector proposal."""
    boundaries = [
        int(boundary)
        for interval in expanded_intervals
        for boundary in interval
    ]
    distance = (
        min(abs(float(center_sample) - boundary) for boundary in boundaries)
        if boundaries
        else None
    )
    within_guard = distance is not None and distance <= int(response_radius)
    try:
        clean_amplitude_supported = (
            np.isfinite(float(clean_local_peak_z))
            and float(clean_local_peak_z) >= float(clean_local_min_z)
        )
    except (TypeError, ValueError):
        clean_amplitude_supported = False
    if clean_peak_max_alignment_samples is None:
        clean_alignment_samples = None
        clean_aligned = True
    else:
        try:
            clean_alignment_samples = abs(
                float(clean_peak_center_sample) - float(center_sample)
            )
            clean_aligned = (
                np.isfinite(clean_alignment_samples)
                and clean_alignment_samples
                <= float(clean_peak_max_alignment_samples)
            )
        except (TypeError, ValueError):
            clean_alignment_samples = None
            clean_aligned = False
    clean_supported = clean_amplitude_supported and clean_aligned
    keep = not within_guard or clean_supported
    return {
        "keep": bool(keep),
        "within_filter_response_guard": bool(within_guard),
        "boundary_distance_samples": distance,
        "clean_amplitude_supported": bool(clean_amplitude_supported),
        "clean_peak_alignment_samples": clean_alignment_samples,
        "clean_peak_aligned": bool(clean_aligned),
        "clean_supported": bool(clean_supported),
        "reason": (
            "kept_outside_boundary_guard"
            if not within_guard
            else (
                "kept_with_clean_support"
                if clean_supported
                else "rejected_boundary_without_clean_support"
            )
        ),
    }


def repair_saturation_interval(signal, replacement, *, core_interval,
                               expanded_interval, method,
                               fs=2_000_000, fmin=7000, fmax=80000,
                               order=4):
    """Return clean and filtered views for one reproducible saturation repair.

    The filtered-domain method still returns a cosine-repaired non-bandpassed
    signal for dual-clean peak evidence. Only its model-facing filtered output
    is assembled after filtering the raw signal and carrier separately.
    """
    if method not in SATURATION_REPAIR_METHODS:
        raise ValueError(f"Unknown saturation repair method: {method}")
    source = np.asarray(signal)
    carrier = np.asarray(replacement)
    expanded_start, expanded_end = (int(value) for value in expanded_interval)
    if method == "direct":
        clean = source.copy()
        if len(carrier) != expanded_end - expanded_start:
            raise ValueError("replacement length must match the expanded interval")
        clean[expanded_start:expanded_end] = carrier
        filtered = butter_bandpass_filter(
            clean, fs=fs, fmin=fmin, fmax=fmax, order=order
        )
    else:
        clean = cosine_blend_replacement(
            source,
            carrier,
            core_interval=core_interval,
            expanded_interval=expanded_interval,
        )
        if method == "cosine-pre-filter":
            filtered = butter_bandpass_filter(
                clean, fs=fs, fmin=fmin, fmax=fmax, order=order
            )
        else:
            filtered_source = butter_bandpass_filter(
                source, fs=fs, fmin=fmin, fmax=fmax, order=order
            )
            filtered_carrier = butter_bandpass_filter(
                carrier, fs=fs, fmin=fmin, fmax=fmax, order=order
            )
            filtered = cosine_blend_replacement(
                filtered_source,
                filtered_carrier,
                core_interval=core_interval,
                expanded_interval=expanded_interval,
            )
    return {
        "method": method,
        "clean_signal": clean.astype(source.dtype, copy=False),
        "filtered_signal": filtered.astype(source.dtype, copy=False),
        "core_interval": [int(value) for value in core_interval],
        "expanded_interval": [int(value) for value in expanded_interval],
    }


def repair_saturation_intervals_filtered_domain(
    signal,
    replacements,
    *,
    fs=2_000_000,
    fmin=7000,
    fmax=80000,
    order=4,
):
    """Repair several disjoint saturation regions with the approved method B.

    ``replacements`` is an iterable of mappings with ``core_interval``,
    ``expanded_interval`` and ``replacement`` entries.  The raw trace is
    filtered once, each carrier is filtered independently, and raised-cosine
    raccords are then applied in the filtered domain.  This prevents one
    repaired interval from changing the filtering context of another one.
    """
    source = np.asarray(signal)
    rows = list(replacements)
    ordered = sorted(
        rows,
        key=lambda row: tuple(int(value) for value in row["expanded_interval"]),
    )
    previous_end = -1
    for row in ordered:
        expanded_start, expanded_end = (
            int(value) for value in row["expanded_interval"]
        )
        if expanded_start < previous_end:
            raise ValueError("expanded saturation intervals must be disjoint")
        previous_end = expanded_end

    filtered = butter_bandpass_filter(
        source, fs=fs, fmin=fmin, fmax=fmax, order=order
    )
    clean = source.copy()
    for row in ordered:
        carrier = np.asarray(row["replacement"])
        core_interval = tuple(int(value) for value in row["core_interval"])
        expanded_interval = tuple(
            int(value) for value in row["expanded_interval"]
        )
        clean = cosine_blend_replacement(
            clean,
            carrier,
            core_interval=core_interval,
            expanded_interval=expanded_interval,
        )
        filtered_carrier = butter_bandpass_filter(
            carrier, fs=fs, fmin=fmin, fmax=fmax, order=order
        )
        filtered = cosine_blend_replacement(
            filtered,
            filtered_carrier,
            core_interval=core_interval,
            expanded_interval=expanded_interval,
        )
    return {
        "method": "cosine-filtered-domain",
        "clean_signal": clean.astype(source.dtype, copy=False),
        "filtered_signal": filtered.astype(source.dtype, copy=False),
        "regions": [
            {
                "core_interval": [
                    int(value) for value in row["core_interval"]
                ],
                "expanded_interval": [
                    int(value) for value in row["expanded_interval"]
                ],
            }
            for row in ordered
        ],
    }


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
