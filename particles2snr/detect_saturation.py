"""
detect_saturation.py
--------------------
Detect saturated/clipped signals in noise datasets using derivative method.

Methodology:
1. Apply bandpass filter (default: 7-80 kHz)
2. Compute derivative of filtered signal
3. If derivative is zero for 1000+ consecutive points, signal is saturated

Usage:
    python detect_saturation.py dataset/
    python detect_saturation.py dataset/ --output saturated_files.txt
"""

import argparse
import csv
import json
import glob
import os
import shutil

import numpy as np
from scipy.signal import butter, filtfilt


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Detect saturated signals using derivative method.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
    python detect_saturation.py dataset/
    python detect_saturation.py dataset/ --output saturated_files.txt
    python detect_saturation.py dataset/ --fmin 7000 --fmax 80000 --min_flat 1000
        """
)
parser.add_argument("base_folder", help="Base folder containing noise_* subfolders")
parser.add_argument("--output", default=None, help="Output file for saturated files list")
parser.add_argument("--move", action="store_true", help="Move saturated files to saturated/ folder")
parser.add_argument("--fmin", type=float, default=7000, help="Bandpass low cutoff (default: 7000 Hz)")
parser.add_argument("--fmax", type=float, default=80000, help="Bandpass high cutoff (default: 80000 Hz)")
parser.add_argument("--fs", type=float, default=2_000_000, help="Sampling frequency (default: 2000000 Hz)")
parser.add_argument("--min-flat", type=int, default=500, help="Min consecutive zero derivative points (default: 1000)")
parser.add_argument("--zero-threshold", type=float, default=1e-4,
                    help="Derivative magnitude treated as flat (default: 1e-4)")
parser.add_argument("--intervals-csv", default=None,
                    help="CSV output path for all saturation intervals")
parser.add_argument("--summary-json", default=None,
                    help="JSON output path for saturation scan summary")


# ---------------------------------------------------------------------------
# Bandpass filter
# ---------------------------------------------------------------------------
def bandpass_filter(signal, fs, fmin, fmax, order=4):
    """Apply bandpass filter to signal."""
    nyq = fs / 2
    low = fmin / nyq
    high = fmax / nyq

    # Ensure bounds are valid
    low = max(0.01, min(low, 0.99))
    high = max(low + 0.01, min(high, 0.99))

    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, signal)
    return filtered


# ---------------------------------------------------------------------------
# Saturation detection using derivative method
# ---------------------------------------------------------------------------
def find_flat_intervals(is_zero, min_flat):
    """
    Return all consecutive flat derivative regions as signal sample intervals.

    The derivative at index i is signal[i + 1] - signal[i]. A run of c flat
    derivative points starting at i corresponds to signal samples [i, i+c+1).
    """
    intervals = []
    current_start = None
    current_len = 0

    for i, is_z in enumerate(is_zero):
        if is_z:
            if current_start is None:
                current_start = i
            current_len += 1
        else:
            if current_start is not None and current_len >= min_flat:
                intervals.append({
                    "start_sample": int(current_start),
                    "end_sample": int(current_start + current_len + 1),
                    "duration_samples": int(current_len + 1),
                    "flat_derivative_count": int(current_len),
                })
            current_start = None
            current_len = 0

    if current_start is not None and current_len >= min_flat:
        intervals.append({
            "start_sample": int(current_start),
            "end_sample": int(current_start + current_len + 1),
            "duration_samples": int(current_len + 1),
            "flat_derivative_count": int(current_len),
        })

    return intervals


def detect_saturation(signal, fs, fmin, fmax, min_flat, file_path=None,
                      zero_threshold=1e-4):
    """
    Detect saturation using derivative method.

    1. Apply bandpass filter
    2. Compute derivative
    3. Find consecutive zero derivative regions

    Returns:
        dict with saturation info
    """
    # Apply bandpass filter
    filtered = bandpass_filter(signal, fs, fmin, fmax)

    # Compute derivative (difference)
    derivative = np.diff(filtered)

    # if "HFocusing_5_10_10um_0_2087_noise_seg001" in file_path:
    #     import matplotlib.pyplot as plt
    #     plt.figure(figsize=(12, 6))
    #     plt.subplot(2, 1, 1)
    #     plt.plot(filtered, label='Filtered Signal')
    #     plt.title('Filtered Signal')
    #     plt.legend()
    #     plt.subplot(2, 1, 2)
    #     plt.plot(derivative, label='Derivative', color='orange')
    #     plt.title('Derivative of Filtered Signal')
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.show()

    # Find zero derivative (within numerical precision)
    is_zero = np.abs(derivative) < zero_threshold

    # if "HFocusing_5_10_10um_0_2087_noise_seg001" in file_path:
    #     plt.figure(figsize=(12, 4))
    #     plt.plot(is_zero.astype(int), label='Is Zero Derivative', color='red')
    #     plt.title('Zero Derivative Indicator')
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.show()

    intervals = find_flat_intervals(is_zero, min_flat)
    max_interval = max(
        intervals,
        key=lambda interval: interval["flat_derivative_count"],
        default=None,
    )
    max_consecutive = (
        max_interval["flat_derivative_count"]
        if max_interval is not None
        else 0
    )
    max_start = max_interval["start_sample"] if max_interval is not None else 0
    is_saturated = max_consecutive >= min_flat

    return {
        "is_saturated": is_saturated,
        # Backward-compatible names kept for existing reports/tests. The
        # detector is about flat derivative runs, not literal zero-valued
        # samples.
        "max_consecutive_zero": max_consecutive,
        "max_consecutive_flat": max_consecutive,
        "flat_start": max_start,
        "intervals": intervals,
        "zero_threshold": zero_threshold,
        "flat_threshold": zero_threshold,
        "requires_filtering": True
    }


def scan_class_folder(folder, class_name, fs, fmin, fmax, min_flat,
                      zero_threshold):
    """Scan all signals in a class folder for saturation."""
    npy_files = sorted(glob.glob(os.path.join(folder, "*.npy")))

    saturated_files = []
    interval_rows = []

    for npy_file in npy_files:
        sig = np.load(npy_file)
        sat_info = detect_saturation(
            sig, fs, fmin, fmax, min_flat, npy_file, zero_threshold
        )

        if sat_info["is_saturated"]:
            file_name = os.path.basename(npy_file)
            saturated_files.append({
                "file": file_name,
                "path": npy_file,
                "class": class_name,
                "max_consecutive_zero": sat_info["max_consecutive_zero"],
                "flat_start": sat_info["flat_start"]
            })
            for interval_idx, interval in enumerate(sat_info["intervals"]):
                interval_rows.append({
                    "file": file_name,
                    "path": npy_file,
                    "class": class_name,
                    "interval_idx": interval_idx,
                    "start_sample": interval["start_sample"],
                    "end_sample": interval["end_sample"],
                    "duration_samples": interval["duration_samples"],
                    "flat_derivative_count": interval["flat_derivative_count"],
                    "fs": fs,
                    "fmin": fmin,
                    "fmax": fmax,
                    "min_flat": min_flat,
                    "zero_threshold": zero_threshold,
                    "flat_threshold": zero_threshold,
                    "method": "bandpass_derivative_flat_run",
                })

    return saturated_files, interval_rows, len(npy_files)


def write_intervals_csv(interval_rows, output_path):
    """Write all saturation intervals with a stable CSV schema."""
    fieldnames = [
        "file", "path", "class", "interval_idx", "start_sample",
        "end_sample", "duration_samples", "flat_derivative_count", "fs",
        "fmin", "fmax", "min_flat", "zero_threshold", "flat_threshold", "method",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(interval_rows)


def write_summary_json(class_summary, interval_rows, args, output_path):
    """Write a machine-readable saturation scan summary."""
    total_files = sum(row["total"] for row in class_summary)
    total_saturated = sum(row["saturated"] for row in class_summary)
    output = {
        "scan_params": {
            "base_folder": args.base_folder,
            "fs": args.fs,
            "fmin": args.fmin,
            "fmax": args.fmax,
            "min_flat": args.min_flat,
            "zero_threshold": args.zero_threshold,
            "method": "bandpass_derivative_flat_run",
        },
        "summary": {
            "total_files": total_files,
            "total_saturated_files": total_saturated,
            "total_intervals": len(interval_rows),
            "saturated_file_pct": (
                100 * total_saturated / total_files if total_files > 0 else 0.0
            ),
        },
        "class_summary": class_summary,
        "intervals": interval_rows,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parser.parse_args()

    base_folder = args.base_folder.rstrip("/")
    fs = args.fs
    fmin = args.fmin
    fmax = args.fmax
    min_flat = args.min_flat
    zero_threshold = args.zero_threshold

    print(f"Scanning for saturated signals in '{base_folder}'...")
    print(f"Bandpass filter: {fmin/1000:.1f}-{fmax/1000:.1f} kHz")
    print(f"Sampling frequency: {fs/1e6:.1f} MHz")
    print(f"Min consecutive zero derivative points: {min_flat}")
    print(f"Duration of flat region: {min_flat/fs*1e6:.2f} μs\n")

    all_saturated = []
    all_intervals = []
    class_summary = []

    # Scan each class folder
    for item in sorted(os.listdir(base_folder)):
        folder = os.path.join(base_folder, item)
        if os.path.isdir(folder):
            npy_files = glob.glob(os.path.join(folder, "*.npy"))
            if npy_files:
                class_name = item.replace("noise_", "")
                saturated, intervals, total_files = scan_class_folder(
                    folder, class_name, fs, fmin, fmax, min_flat,
                    zero_threshold
                )
                all_intervals.extend(intervals)
                class_summary.append({
                    "class": class_name,
                    "saturated": len(saturated),
                    "total": total_files,
                    "intervals": len(intervals),
                })

                if saturated:
                    all_saturated.extend(saturated)

                    print(f"\n{class_name}: {len(saturated)} saturated signals")
                    for s in saturated[:20]:  # Show first 20
                        duration_us = s["max_consecutive_zero"] / fs * 1e6
                        print(f"  {s['file']}: {s['max_consecutive_zero']} points ({duration_us:.2f} μs)")
                    if len(saturated) > 20:
                        print(f"  ... and {len(saturated) - 20} more")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_saturated = sum(s["saturated"] for s in class_summary)
    total_files = sum(s["total"] for s in class_summary)

    if total_files > 0:
        print(f"\nTotal saturated signals: {total_saturated}/{total_files} ({100*total_saturated/total_files:.1f}%)")
    else:
        print(f"\nTotal saturated signals: {total_saturated}")

    if class_summary:
        print("\nPer-class breakdown:")
        print(f"{'Class':<15} {'Saturated':>12} {'Total':>10} {'Intervals':>12} {'Pct':>10}")
        print("-" * 60)
        for cs in class_summary:
            pct = 100 * cs["saturated"] / cs["total"] if cs["total"] > 0 else 0
            print(f"{cs['class']:<15} {cs['saturated']:>12} {cs['total']:>10} {cs['intervals']:>12} {pct:>9.1f}%")

    # Save to file if requested
    if args.output and all_saturated:
        with open(args.output, 'w') as f:
            f.write("# Saturated signals detected by derivative method\n")
            f.write(f"# Bandpass: {fmin/1000:.1f}-{fmax/1000:.1f} kHz\n")
            f.write(f"# Min consecutive zero points: {min_flat}\n\n")
            for s in all_saturated:
                f.write(f"{s['class']}/{s['file']}\n")
        print(f"\nSaturated files list saved to: {args.output}")

    if args.intervals_csv:
        write_intervals_csv(all_intervals, args.intervals_csv)
        print(f"\nSaturation intervals saved to: {args.intervals_csv}")

    if args.summary_json:
        write_summary_json(class_summary, all_intervals, args, args.summary_json)
        print(f"Saturation summary saved to: {args.summary_json}")

    # Move saturated files if requested
    if args.move and all_saturated:
        saturated_folder = os.path.join(base_folder, "saturated")
        os.makedirs(saturated_folder, exist_ok=True)

        moved_count = 0
        for s in all_saturated:
            src = s["path"]
            dst = os.path.join(saturated_folder, s["file"])
            # Handle duplicate filenames
            if os.path.exists(dst):
                base, ext = os.path.splitext(s["file"])
                counter = 1
                while os.path.exists(dst):
                    dst = os.path.join(saturated_folder, f"{base}_{counter}{ext}")
                    counter += 1
            shutil.move(src, dst)
            moved_count += 1

        print(f"\nMoved {moved_count} saturated files to: {saturated_folder}")

    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    if total_saturated > 0:
        print(f"\n⚠️  {total_saturated} signals show saturation (flat derivative for {min_flat}+ points)")
        print("\nThese signals have been filtered by the bandpass and show")
        print("zero derivative, indicating ADC saturation/clipping.")
        print("\nActions:")
        print("  1. Exclude these signals from analysis")
        print("  2. Check acquisition gain settings")
        print("  3. Verify ADC range is appropriate for signal amplitude")
    else:
        print("\n✓ No saturated signals detected.")


if __name__ == "__main__":
    main()
