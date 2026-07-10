"""Generate a cleaned P0 dataset and particles2SNR-derived annotations.

The command is intentionally non-destructive: source ``.npy`` files are never
modified. A derived split/class tree is written to ``--output-root`` and the
particles2SNR/SNR/annotation artifacts are written to ``--particles2SNR-output``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

from repo_paths import RESULTS_RUNS

from detect_saturation import (
    scan_class_folder,
    write_intervals_csv,
    write_summary_json,
)
from saturation_cleaning import (
    clean_signal_non_destructive,
    detect_unsafe_intervals,
    read_noise_pool,
)


DEFAULT_CLASSES = ("2um", "4um", "10um")
DEFAULT_SPLITS = ("train", "test")
DEFAULT_FS = 2_000_000.0
DEFAULT_BANDPASS_FMIN = 7_000.0
DEFAULT_BANDPASS_FMAX = 80_000.0
DEFAULT_BANDPASS_ORDER = 4


def butter_bandpass_filter(signal: np.ndarray, fs: float, fmin: float, fmax: float,
                           order: int = DEFAULT_BANDPASS_ORDER) -> np.ndarray:
    arr = np.asarray(signal)
    if arr.ndim != 1:
        arr = np.squeeze(arr)
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1D signal, got shape {arr.shape}")
    nyquist = fs / 2.0
    low = max(0.001, min(float(fmin) / nyquist, 0.99))
    high = max(low + 0.001, min(float(fmax) / nyquist, 0.99))
    b, a = butter(int(order), [low, high], btype="band")
    return filtfilt(b, a, arr).astype(arr.dtype, copy=False)


def parse_csv_arg(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_class_source_dirs(value: str) -> dict[str, Path]:
    """Parse ``class=path`` pairs separated by commas."""
    out = {}
    for item in parse_csv_arg(value):
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                "--class-source-dirs entries must be class=path"
            )
        class_name, path = item.split("=", 1)
        class_name = class_name.strip()
        path = path.strip()
        if not class_name or not path:
            raise argparse.ArgumentTypeError(
                "--class-source-dirs entries must be non-empty class=path"
            )
        out[class_name] = Path(path)
    return out


def find_zero_runs(signal: np.ndarray, zero_epsilon: float = 0.0) -> list[dict]:
    """Return consecutive zero-valued runs as half-open sample intervals."""
    if zero_epsilon < 0:
        raise ValueError("zero_epsilon must be >= 0")
    arr = np.asarray(signal)
    is_zero = np.abs(arr) <= zero_epsilon
    runs = []
    start = None
    for idx, value in enumerate(is_zero):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append({
                "start_sample": int(start),
                "end_sample": int(idx),
                "duration_samples": int(idx - start),
            })
            start = None
    if start is not None:
        runs.append({
            "start_sample": int(start),
            "end_sample": int(len(arr)),
            "duration_samples": int(len(arr) - start),
        })
    return runs


def remove_long_zero_runs(signal: np.ndarray, zero_epsilon: float = 0.0,
                          max_zero_run_after_clean: int = 2) -> tuple[np.ndarray, list[dict]]:
    """Remove zero runs longer than the allowed raccord length.

    For each zero run longer than ``max_zero_run_after_clean``, the first
    ``max_zero_run_after_clean`` samples are kept and the remainder is removed.
    This keeps a short explicit junction while preventing long dead regions
    from driving the downstream dataset.
    """
    if max_zero_run_after_clean < 0:
        raise ValueError("max_zero_run_after_clean must be >= 0")

    arr = np.asarray(signal)
    if arr.ndim != 1:
        arr = np.squeeze(arr)
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1D signal, got shape {arr.shape}")

    runs = find_zero_runs(arr, zero_epsilon=zero_epsilon)
    pieces = []
    cursor = 0
    actions = []
    removed_before = 0

    for interval_idx, run in enumerate(runs):
        start = run["start_sample"]
        end = run["end_sample"]
        duration = run["duration_samples"]
        if duration <= max_zero_run_after_clean:
            continue

        keep_end = min(start + max_zero_run_after_clean, end)
        pieces.append(arr[cursor:keep_end])
        removed = end - keep_end
        actions.append({
            "interval_idx": int(interval_idx),
            "start_sample": int(start),
            "end_sample": int(end),
            "duration_samples": int(duration),
            "kept_zero_samples": int(keep_end - start),
            "removed_samples": int(removed),
            "clean_start_sample": int(start - removed_before),
            "clean_end_sample": int(keep_end - removed_before),
            "action": "removed_zero_tail",
        })
        removed_before += removed
        cursor = end

    if not actions:
        return arr.copy(), []

    pieces.append(arr[cursor:])
    cleaned = np.concatenate(pieces) if pieces else np.asarray([], dtype=arr.dtype)
    return cleaned.astype(arr.dtype, copy=False), actions


def iter_class_files(split_dir: Path, class_names: Iterable[str]) -> Iterable[tuple[str, Path]]:
    for class_name in class_names:
        class_dir = split_dir / class_name
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.glob("*.npy")):
            yield class_name, path


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare_split_tree_from_class_sources(class_source_dirs: dict[str, Path],
                                          staging_root: Path,
                                          splits: tuple[str, ...],
                                          test_fraction: float,
                                          seed: int) -> list[dict]:
    """Create a class-folder split tree from standalone class source folders."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    if set(splits) != {"train", "test"}:
        raise ValueError("--class-source-dirs currently requires --splits train,test")

    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for class_name, source_dir in class_source_dirs.items():
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Missing class source dir for {class_name}: {source_dir}")
        files = sorted(source_dir.glob("*.npy"))
        if not files:
            raise FileNotFoundError(f"No .npy files in class source dir: {source_dir}")

        indices = np.arange(len(files))
        rng.shuffle(indices)
        n_test = max(1, int(round(len(files) * test_fraction)))
        n_test = min(n_test, len(files) - 1) if len(files) > 1 else 1
        test_indices = set(int(i) for i in indices[:n_test])

        for idx, source_path in enumerate(files):
            split = "test" if idx in test_indices else "train"
            output_path = staging_root / split / class_name / source_path.name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.exists() or output_path.is_symlink():
                output_path.unlink()
            try:
                output_path.symlink_to(source_path.resolve())
                action = "symlinked"
            except OSError:
                shutil.copy2(source_path, output_path)
                action = "copied"
            rows.append({
                "class": class_name,
                "split": split,
                "source_path": str(source_path),
                "staged_path": str(output_path),
                "action": action,
            })
    return rows


def clean_split(input_split_dir: Path, output_split_dir: Path,
                class_names: tuple[str, ...], zero_epsilon: float,
                max_zero_run_after_clean: int, args: argparse.Namespace,
                noise_pool: list[np.ndarray], rng: np.random.Generator,
                split: str, peak_evidence_clean_dir: Path | None = None) -> tuple[list[dict], list[dict], list[dict]]:
    zero_manifest_rows = []
    saturation_rows = []
    peak_evidence_rows = []
    if output_split_dir.exists():
        shutil.rmtree(output_split_dir)
    output_split_dir.mkdir(parents=True, exist_ok=True)
    if peak_evidence_clean_dir is not None:
        if peak_evidence_clean_dir.exists():
            shutil.rmtree(peak_evidence_clean_dir)
        peak_evidence_clean_dir.mkdir(parents=True, exist_ok=True)

    for class_name, source_path in iter_class_files(input_split_dir, class_names):
        rel = source_path.relative_to(input_split_dir)
        output_path = output_split_dir / rel
        output_path.parent.mkdir(parents=True, exist_ok=True)

        signal = np.load(source_path)
        zero_cleaned, zero_actions = remove_long_zero_runs(
            signal,
            zero_epsilon=zero_epsilon,
            max_zero_run_after_clean=max_zero_run_after_clean,
        )

        if not zero_actions:
            zero_manifest_rows.append({
                "source_path": str(source_path),
                "output_path": str(output_path),
                "class": class_name,
                "filename": source_path.name,
                "interval_idx": "",
                "start_sample": "",
                "end_sample": "",
                "duration_samples": "",
                "kept_zero_samples": 0,
                "removed_samples": 0,
                "source_length": int(np.asarray(signal).size),
                "clean_length": int(zero_cleaned.size),
                "clean_start_sample": "",
                "clean_end_sample": "",
                "action": "copied_unchanged",
            })
        else:
            for action in zero_actions:
                zero_manifest_rows.append({
                    "source_path": str(source_path),
                    "output_path": str(output_path),
                    "class": class_name,
                    "filename": source_path.name,
                    "source_length": int(np.asarray(signal).size),
                    "clean_length": int(zero_cleaned.size),
                    **action,
                })

        sat_info, unsafe = detect_unsafe_intervals(
            zero_cleaned,
            fs=args.fs,
            fmin=args.saturation_fmin,
            fmax=args.saturation_fmax,
            min_flat=args.saturation_min_flat,
            zero_threshold=args.saturation_zero_threshold,
            guard_before=args.saturation_guard_before,
            guard_after=args.saturation_guard_after,
        )
        sat_cleaned, sat_actions = clean_signal_non_destructive(
            zero_cleaned,
            unsafe,
            policy=args.saturation_policy,
            noise_pool=noise_pool,
            rng=rng,
            mask_value=args.saturation_mask_value,
        )
        for action in sat_actions:
            saturation_rows.append({
                "split": split,
                "source_path": str(source_path),
                "output_path": str(output_path),
                "class": class_name,
                "filename": source_path.name,
                "policy": args.saturation_policy,
                "dropped_events": 0,
                "fs": args.fs,
                "fmin": args.saturation_fmin,
                "fmax": args.saturation_fmax,
                "min_flat": args.saturation_min_flat,
                "zero_threshold": args.saturation_zero_threshold,
                "guard_before": args.saturation_guard_before,
                "guard_after": args.saturation_guard_after,
                "max_consecutive_flat": sat_info.get("max_consecutive_flat", sat_info.get("max_consecutive_zero", 0)),
                **action,
            })

        final_signal = sat_cleaned
        final_action = "saturation_cleaned"
        peak_evidence_path = None
        if peak_evidence_clean_dir is not None:
            peak_evidence_path = peak_evidence_clean_dir / rel
            peak_evidence_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(peak_evidence_path, np.asarray(sat_cleaned, dtype=np.asarray(signal).dtype))
            peak_evidence_rows.append({
                "split": split,
                "class": class_name,
                "filename": source_path.name,
                "source_path": str(source_path),
                "filtered_output_path": str(output_path),
                "peak_evidence_clean_path": str(peak_evidence_path),
                "action": "saturation_cleaned_no_bandpass",
            })
        if args.apply_bandpass_output:
            final_signal = butter_bandpass_filter(
                final_signal,
                fs=args.fs,
                fmin=args.bandpass_fmin,
                fmax=args.bandpass_fmax,
                order=args.bandpass_order,
            )
            final_action = "saturation_cleaned_bandpass_filtered"
        np.save(output_path, np.asarray(final_signal, dtype=np.asarray(signal).dtype))
        if not sat_actions:
            saturation_rows.append({
                "split": split,
                "source_path": str(source_path),
                "output_path": str(output_path),
                "class": class_name,
                "filename": source_path.name,
                "policy": args.saturation_policy,
                "interval_idx": "",
                "start_sample": "",
                "end_sample": "",
                "duration_samples": "",
                "action": final_action if args.apply_bandpass_output else "copied_no_saturation",
                "dropped_events": 0,
                "fs": args.fs,
                "fmin": args.saturation_fmin,
                "fmax": args.saturation_fmax,
                "min_flat": args.saturation_min_flat,
                "zero_threshold": args.saturation_zero_threshold,
                "guard_before": args.saturation_guard_before,
                "guard_after": args.saturation_guard_after,
                "max_consecutive_flat": sat_info.get("max_consecutive_flat", sat_info.get("max_consecutive_zero", 0)),
            })

    return zero_manifest_rows, saturation_rows, peak_evidence_rows


def scan_saturation_split(split_dir: Path, output_dir: Path,
                          class_names: tuple[str, ...], args: argparse.Namespace,
                          prefix: str = "saturation") -> dict:
    all_intervals = []
    class_summary = []
    error_rows = []
    for class_name in class_names:
        folder = split_dir / class_name
        if not folder.is_dir():
            class_summary.append({
                "class": class_name,
                "saturated": 0,
                "total": 0,
                "intervals": 0,
            })
            continue
        try:
            saturated, intervals, total_files = scan_class_folder(
                str(folder),
                class_name,
                args.fs,
                args.saturation_fmin,
                args.saturation_fmax,
                args.saturation_min_flat,
                args.saturation_zero_threshold,
            )
        except ValueError as exc:
            # scipy.signal.filtfilt can reject very short smoke-test signals.
            # Keep generation non-destructive and auditable instead of aborting.
            npy_files = sorted(folder.glob("*.npy"))
            saturated, intervals, total_files = [], [], len(npy_files)
            for path in npy_files:
                error_rows.append({
                    "file": path.name,
                    "path": str(path),
                    "class": class_name,
                    "error": str(exc),
                })
        all_intervals.extend(intervals)
        class_summary.append({
            "class": class_name,
            "saturated": len(saturated),
            "total": total_files,
            "intervals": len(intervals),
        })

    sat_args = SimpleNamespace(
        base_folder=str(split_dir),
        fs=args.fs,
        fmin=args.saturation_fmin,
        fmax=args.saturation_fmax,
        min_flat=args.saturation_min_flat,
        zero_threshold=args.saturation_zero_threshold,
    )
    write_intervals_csv(all_intervals, str(output_dir / f"{prefix}_intervals.csv"))
    write_summary_json(class_summary, all_intervals, sat_args, str(output_dir / f"{prefix}_summary.json"))
    write_csv(
        output_dir / f"{prefix}_errors.csv",
        error_rows,
        ["file", "path", "class", "error"],
    )
    return {
        "class_summary": class_summary,
        "intervals": all_intervals,
        "errors": error_rows,
        "total_saturated_files": int(sum(row["saturated"] for row in class_summary)),
    }


def run_particles2SNR_split(split_dir: Path, output_dir: Path,
                      class_names: tuple[str, ...], device: str,
                      verbose: bool, bandpass_fmin: float | None = None,
                      bandpass_fmax: float | None = None,
                      bandpass_order: int | None = None) -> None:
    # Import lazily so tests for pure cleaning/JSON helpers do not require torch.
    import run_dataset

    data_files = run_dataset.load_all_data(str(split_dir), list(class_names))
    particles2SNR_args = SimpleNamespace(device=device, verbose=verbose)
    results = []
    for signal_idx, (file_path, folder_name) in enumerate(data_files):
        config = run_dataset.get_config_for_folder(folder_name)
        if bandpass_fmin is not None:
            config.bandpass_lowcut = float(bandpass_fmin)
        if bandpass_fmax is not None:
            config.bandpass_highcut = float(bandpass_fmax)
        if bandpass_order is not None:
            config.bandpass_order = int(bandpass_order)
        result = run_dataset.process_signal(
            file_path,
            folder_name,
            config,
            particles2SNR_args,
            signal_idx,
        )
        if result is not None:
            results.append(result)
    run_dataset.export_results(results, str(output_dir), processing_time_ms=0.0)


def particle_passage_time_ms(particle: dict) -> float | None:
    try:
        tau = float(particle.get("tau"))
    except (TypeError, ValueError):
        return None
    if not np.isfinite(tau):
        return None
    return max(0.0, tau * 1000.0)


def keep_particle_by_passage_time(particle: dict, min_ms: float | None,
                                  max_ms: float | None) -> tuple[bool, str, float | None]:
    tau_ms = particle_passage_time_ms(particle)
    if tau_ms is None:
        return False, "missing_or_invalid_passage_time", tau_ms
    if min_ms is not None and tau_ms < float(min_ms):
        return False, "passage_time_below_min", tau_ms
    if max_ms is not None and tau_ms > float(max_ms):
        return False, "passage_time_above_max", tau_ms
    return True, "kept", tau_ms


def _as_finite_float(value, default: float = float("-inf")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def candidate_score(particle: dict, score_name: str = "snr_db") -> float:
    if score_name == "snr_db":
        score = _as_finite_float(particle.get("snr_db"))
        if score != float("-inf"):
            return score
    if score_name == "energy":
        score = _as_finite_float(particle.get("energy"))
        if score != float("-inf"):
            return score
    p0 = abs(_as_finite_float(particle.get("P0"), default=0.0))
    return p0


def particle_interval_samples(particle: dict, signal_length: int, fs: float) -> tuple[float, float, float]:
    t0_samples = float(particle.get("t0", 0.0)) * fs
    tau_samples = max(0.0, float(particle.get("tau", 0.0)) * fs)
    length = max(1, int(signal_length))
    mean = min(1.0, max(0.0, t0_samples / length))
    std = min(0.5, max(0.0, tau_samples / length))
    half_width = min(0.5, 2.5 * std)
    left = float(max(0.0, mean - half_width) * length)
    right = float(min(1.0, mean + half_width) * length)
    center = float(mean * length)
    return left, right, center


def interval_iou(left_a: float, right_a: float, left_b: float, right_b: float) -> float:
    inter = max(0.0, min(right_a, right_b) - max(left_a, left_b))
    union = max(right_a, right_b) - min(left_a, left_b)
    return inter / union if union > 0.0 else 0.0


def particle_frequency_hz(particle: dict) -> float | None:
    value = _as_finite_float(particle.get("frequency"), default=float("nan"))
    return value if np.isfinite(value) else None


def should_merge_candidates(
    cand: dict,
    kept_cand: dict,
    fs: float,
    iou: float,
    iou_threshold: float,
    duplicate_iou_threshold: float,
    close_center_distance_ms: float | None,
    ambiguous_center_distance_ms: float | None,
    close_frequency_hz: float | None,
    ambiguous_frequency_hz: float | None,
    snr_margin_db: float | None,
) -> tuple[bool, str | None, float, float | None, float | None, float | None]:
    center_distance = abs(float(cand["center"]) - float(kept_cand["center"]))
    center_distance_ms_value = center_distance / fs * 1000.0
    cand_freq = particle_frequency_hz(cand["particle"])
    kept_freq = particle_frequency_hz(kept_cand["particle"])
    frequency_distance_hz = None
    if cand_freq is not None and kept_freq is not None:
        frequency_distance_hz = abs(cand_freq - kept_freq)
    winner_score = float(kept_cand["score"])
    loser_score = float(cand["score"])
    score_gap = winner_score - loser_score

    if iou >= duplicate_iou_threshold:
        return True, "overlap_nms_high_iou", center_distance_ms_value, frequency_distance_hz, score_gap

    if iou < iou_threshold:
        return False, None, center_distance_ms_value, frequency_distance_hz, score_gap

    freq_close = frequency_distance_hz is not None and close_frequency_hz is not None and frequency_distance_hz <= float(close_frequency_hz)
    ambiguous_freq_close = (
        frequency_distance_hz is not None
        and ambiguous_frequency_hz is not None
        and frequency_distance_hz <= float(ambiguous_frequency_hz)
    )
    if (
        close_center_distance_ms is not None
        and center_distance_ms_value <= float(close_center_distance_ms)
        and freq_close
    ):
        return True, "overlap_nms_close_center_frequency", center_distance_ms_value, frequency_distance_hz, score_gap

    if (
        ambiguous_center_distance_ms is not None
        and snr_margin_db is not None
        and center_distance_ms_value <= float(ambiguous_center_distance_ms)
        and ambiguous_freq_close
        and score_gap >= float(snr_margin_db)
    ):
        return True, "overlap_nms_ambiguous_low_snr", center_distance_ms_value, frequency_distance_hz, score_gap

    return False, None, center_distance_ms_value, frequency_distance_hz, score_gap


def merge_overlapping_particles(
    particles: list[dict],
    signal_length: int,
    fs: float,
    iou_threshold: float = 0.4,
    score_name: str = "snr_db",
    center_distance_ms: float | None = None,
    duplicate_iou_threshold: float = 0.6,
    close_center_distance_ms: float | None = 0.20,
    ambiguous_center_distance_ms: float | None = 0.30,
    close_frequency_hz: float | None = 6000.0,
    ambiguous_frequency_hz: float | None = 8000.0,
    snr_margin_db: float | None = 4.0,
) -> tuple[list[dict], list[dict]]:
    if center_distance_ms is not None:
        ambiguous_center_distance_ms = center_distance_ms
    candidates = []
    for idx, particle in enumerate(particles):
        left, right, center = particle_interval_samples(particle, signal_length, fs)
        candidates.append({
            "idx": idx,
            "particle": particle,
            "left": left,
            "right": right,
            "center": center,
            "score": candidate_score(particle, score_name),
        })

    kept: list[dict] = []
    dropped: list[dict] = []
    for cand in sorted(candidates, key=lambda c: (c["score"], -c["idx"]), reverse=True):
        matched = None
        matched_iou = 0.0
        matched_center_distance_ms = None
        matched_frequency_distance_hz = None
        matched_score_gap = None
        matched_reason = None
        for kept_cand in kept:
            iou = interval_iou(cand["left"], cand["right"], kept_cand["left"], kept_cand["right"])
            should_merge, reason, center_distance_value, frequency_distance_value, score_gap = should_merge_candidates(
                cand,
                kept_cand,
                fs,
                iou,
                iou_threshold,
                duplicate_iou_threshold,
                close_center_distance_ms,
                ambiguous_center_distance_ms,
                close_frequency_hz,
                ambiguous_frequency_hz,
                snr_margin_db,
            )
            if should_merge and (matched is None or iou > matched_iou):
                matched = kept_cand
                matched_iou = iou
                matched_center_distance_ms = center_distance_value
                matched_frequency_distance_hz = frequency_distance_value
                matched_score_gap = score_gap
                matched_reason = reason
        if matched is None:
            kept.append(cand)
            continue
        particle = cand["particle"]
        dropped.append({
            "reason": matched_reason or "overlap_nms_conditional",
            "iou_with_kept": float(matched_iou),
            "center_distance_ms": matched_center_distance_ms,
            "frequency_distance_hz": matched_frequency_distance_hz,
            "score_gap": matched_score_gap,
            "kept_source_idx": int(matched["idx"]),
            "suppressed_source_idx": int(cand["idx"]),
            "kept_annotation_score": float(matched["score"]),
            "suppressed_score": float(cand["score"]),
            "score_name": score_name,
            "passage_time_ms": particle_passage_time_ms(particle),
            "snr_db": particle.get("snr_db"),
            "frequency": particle.get("frequency"),
            "start_ms": float(cand["left"] / fs * 1000.0),
            "end_ms": float(cand["right"] / fs * 1000.0),
        })

    kept_particles = [cand["particle"] for cand in sorted(kept, key=lambda c: c["center"])]
    return kept_particles, dropped


def particle_to_annotation(particle: dict, signal_length: int, fs: float,
                           class_id: int, ann_id: int) -> dict:
    t0_samples = float(particle.get("t0", 0.0)) * fs
    tau_samples = max(0.0, float(particle.get("tau", 0.0)) * fs)
    passage_time_ms = particle_passage_time_ms(particle)
    length = max(1, int(signal_length))
    mean = min(1.0, max(0.0, t0_samples / length))
    std = min(0.5, max(0.0, tau_samples / length))
    half_width = min(0.5, 2.5 * std)
    return {
        "id": int(ann_id),
        "class_id": int(class_id),
        "mean": float(mean),
        "std": float(std),
        "center": float(mean),
        "half_width": float(half_width),
        "start": float(max(0.0, mean - half_width)),
        "end": float(min(1.0, mean + half_width)),
        "amplitude": float(particle.get("P0", 0.0)),
        "frequency": float(particle.get("frequency", 0.0)),
        "snr_db": particle.get("snr_db"),
        "passage_time_ms": passage_time_ms,
        "peak_support": bool(particle.get("peak_support", False)),
        "peak_group_id": particle.get("peak_group_id"),
        "peak_z": particle.get("peak_z"),
        "peak_center_ms": particle.get("peak_center_ms"),
        "local_peak_z": particle.get("local_peak_z"),
        "clean_peak_support": particle.get("clean_peak_support"),
        "clean_peak_group_id": particle.get("clean_peak_group_id"),
        "clean_peak_z": particle.get("clean_peak_z"),
        "clean_peak_center_ms": particle.get("clean_peak_center_ms"),
        "clean_local_peak_z": particle.get("clean_local_peak_z"),
        "source": "particles2SNR_pipeline",
    }


def particle_yolo_width_ms(particle: dict, signal_length: int, fs: float) -> float:
    left, right, _ = particle_interval_samples(particle, signal_length, fs)
    return max(0.0, right - left) / fs * 1000.0


def filter_particles_by_yolo_width(
    particles: list[dict],
    signal_length: int,
    fs: float,
    min_width_ms: float | None = None,
    max_width_ms: float | None = None,
) -> tuple[list[dict], list[dict]]:
    kept = []
    dropped = []
    for particle in particles:
        left, right, _ = particle_interval_samples(particle, signal_length, fs)
        width_ms = max(0.0, right - left) / fs * 1000.0
        reason = None
        if min_width_ms is not None and width_ms < float(min_width_ms):
            reason = "yolo_width_below_min"
        elif max_width_ms is not None and width_ms > float(max_width_ms):
            reason = "yolo_width_above_max"
        if reason is None:
            kept.append(particle)
            continue
        dropped.append({
            "reason": reason,
            "stage": "pre_nms",
            "width_ms": float(width_ms),
            "passage_time_ms": particle_passage_time_ms(particle),
            "snr_db": particle.get("snr_db"),
            "frequency": particle.get("frequency"),
            "start_ms": float(left / fs * 1000.0),
            "end_ms": float(right / fs * 1000.0),
        })
    return kept, dropped


def moving_average_abs_envelope(signal: np.ndarray, fs: float, window_ms: float) -> np.ndarray:
    arr = np.asarray(signal, dtype=np.float64).reshape(-1)
    window = max(3, int(round(float(window_ms) / 1000.0 * fs)))
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(np.abs(arr), kernel, mode="same")


def robust_z_values(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    scale = max(mad, 1e-12)
    return (arr - median) / scale, median, scale


def group_envelope_peaks(
    envelope: np.ndarray,
    peaks: np.ndarray,
    z_values: np.ndarray,
    valley_ratio: float,
) -> list[dict]:
    if len(peaks) == 0:
        return []
    groups = []
    current = [int(peaks[0])]
    for peak in [int(p) for p in peaks[1:]]:
        previous = current[-1]
        lo, hi = sorted((previous, peak))
        valley = float(np.min(envelope[lo:hi + 1])) if hi > lo else float(envelope[lo])
        lower_peak = min(float(envelope[previous]), float(envelope[peak]))
        ratio = valley / lower_peak if lower_peak > 0 else 0.0
        if ratio >= float(valley_ratio):
            current.append(peak)
        else:
            groups.append(current)
            current = [peak]
    groups.append(current)

    out = []
    for group_id, group_peaks in enumerate(groups):
        best_peak = max(group_peaks, key=lambda p: float(z_values[p]))
        out.append({
            "id": int(group_id),
            "peaks": [int(p) for p in group_peaks],
            "peak_sample": int(best_peak),
            "peak_z": float(z_values[best_peak]),
            "start_sample": int(min(group_peaks)),
            "end_sample": int(max(group_peaks)),
        })
    return out


def detect_peak_groups(
    signal_values: np.ndarray,
    fs: float,
    envelope_window_ms: float = 0.08,
    min_z: float = 4.0,
    prominence_z: float = 2.0,
    min_separation_ms: float = 0.18,
    valley_ratio: float = 0.55,
) -> tuple[list[dict], np.ndarray, np.ndarray, float, float]:
    envelope = moving_average_abs_envelope(signal_values, fs, envelope_window_ms)
    z_values, baseline, scale = robust_z_values(envelope)
    distance = max(1, int(round(float(min_separation_ms) / 1000.0 * fs)))
    peaks, _ = find_peaks(
        envelope,
        height=baseline + float(min_z) * scale,
        prominence=float(prominence_z) * scale,
        distance=distance,
    )
    groups = group_envelope_peaks(envelope, peaks, z_values, valley_ratio)
    return groups, envelope, z_values, baseline, scale


def cluster_particle_candidates(candidates: list[dict], fs: float, cluster_gap_ms: float) -> list[list[dict]]:
    if not candidates:
        return []
    gap_samples = max(0.0, float(cluster_gap_ms) / 1000.0 * fs)
    ordered = sorted(candidates, key=lambda c: (c["left"], c["right"]))
    clusters = []
    current = [ordered[0]]
    current_right = float(ordered[0]["right"])
    for cand in ordered[1:]:
        if float(cand["left"]) <= current_right + gap_samples:
            current.append(cand)
            current_right = max(current_right, float(cand["right"]))
        else:
            clusters.append(current)
            current = [cand]
            current_right = float(cand["right"])
    clusters.append(current)
    return clusters


def annotate_particle_peak_evidence(
    particle: dict,
    signal_length: int,
    fs: float,
    peak_groups: list[dict],
    z_values: np.ndarray,
) -> dict:
    left, right, center = particle_interval_samples(particle, signal_length, fs)
    left_i = max(0, min(int(signal_length), int(np.floor(left))))
    right_i = max(left_i + 1, min(int(signal_length), int(np.ceil(right))))
    covered = [g for g in peak_groups if left_i <= int(g["peak_sample"]) <= right_i]
    local_z = float(np.max(z_values[left_i:right_i])) if right_i > left_i else 0.0
    selected = None
    if covered:
        selected = min(
            covered,
            key=lambda g: (abs(float(g["peak_sample"]) - center), -float(g["peak_z"])),
        )
    out = dict(particle)
    out["peak_support"] = bool(selected is not None)
    out["peak_group_id"] = int(selected["id"]) if selected is not None else None
    out["peak_z"] = float(selected["peak_z"]) if selected is not None else local_z
    out["peak_center_ms"] = float(selected["peak_sample"] / fs * 1000.0) if selected is not None else None
    out["local_peak_z"] = local_z
    return out


def refine_particles_with_peak_evidence(
    particles: list[dict],
    signal_values: np.ndarray,
    signal_length: int,
    fs: float,
    envelope_window_ms: float = 0.08,
    min_z: float = 4.0,
    prominence_z: float = 2.0,
    min_separation_ms: float = 0.18,
    valley_ratio: float = 0.55,
    cluster_gap_ms: float = 0.25,
    keep_high_snr_db: float = 4.0,
    weak_peak_ratio: float = 0.35,
) -> tuple[list[dict], list[dict], list[dict]]:
    peak_groups, envelope, z_values, baseline, scale = detect_peak_groups(
        signal_values,
        fs,
        envelope_window_ms=envelope_window_ms,
        min_z=min_z,
        prominence_z=prominence_z,
        min_separation_ms=min_separation_ms,
        valley_ratio=valley_ratio,
    )
    candidates = []
    for idx, particle in enumerate(particles):
        enriched = annotate_particle_peak_evidence(particle, signal_length, fs, peak_groups, z_values)
        left, right, center = particle_interval_samples(enriched, signal_length, fs)
        enriched["_peak_source_idx"] = idx
        candidates.append({
            "idx": idx,
            "particle": enriched,
            "left": left,
            "right": right,
            "center": center,
            "score": candidate_score(enriched, "snr_db"),
        })

    kept: list[dict] = []
    dropped: list[dict] = []
    for cluster in cluster_particle_candidates(candidates, fs, cluster_gap_ms):
        supported = [c for c in cluster if c["particle"].get("peak_support")]
        cluster_peak_z = max([float(c["particle"].get("peak_z") or 0.0) for c in supported] or [0.0])
        by_group: dict[int, list[dict]] = {}
        for cand in supported:
            gid = cand["particle"].get("peak_group_id")
            if gid is not None:
                by_group.setdefault(int(gid), []).append(cand)

        selected_ids = set()
        for gid, group_candidates in by_group.items():
            viable = [
                c for c in group_candidates
                if cluster_peak_z <= 0.0 or float(c["particle"].get("peak_z") or 0.0) >= cluster_peak_z * float(weak_peak_ratio)
            ]
            if not viable:
                viable = group_candidates
            best = max(
                viable,
                key=lambda c: (
                    float(c["particle"].get("peak_z") or 0.0),
                    float(c["score"]),
                    -abs(float(c["center"]) - float(c["particle"].get("peak_group_id") or 0)),
                ),
            )
            selected_ids.add(id(best))

        for cand in cluster:
            particle = cand["particle"]
            snr = _as_finite_float(particle.get("snr_db"), default=float("-inf"))
            if id(cand) in selected_ids:
                kept.append(particle)
                continue
            reason = "same_peak_group_duplicate" if particle.get("peak_support") else "no_peak_evidence"
            if particle.get("peak_support") and cluster_peak_z > 0.0 and float(particle.get("peak_z") or 0.0) < cluster_peak_z * float(weak_peak_ratio):
                reason = "weak_peak_in_cluster"
            if not particle.get("peak_support") and snr >= float(keep_high_snr_db):
                kept.append(particle)
                continue
            dropped.append({
                "reason": reason,
                "stage": "peak_evidence",
                "peak_support": bool(particle.get("peak_support")),
                "peak_group_id": particle.get("peak_group_id"),
                "peak_z": particle.get("peak_z"),
                "local_peak_z": particle.get("local_peak_z"),
                "cluster_size": int(len(cluster)),
                "cluster_peak_z": float(cluster_peak_z),
                "snr_db": particle.get("snr_db"),
                "frequency": particle.get("frequency"),
                "passage_time_ms": particle_passage_time_ms(particle),
                "start_ms": float(cand["left"] / fs * 1000.0),
                "end_ms": float(cand["right"] / fs * 1000.0),
            })

    kept.sort(key=lambda p: particle_interval_samples(p, signal_length, fs)[2])
    peak_summary = [{
        "id": int(g["id"]),
        "peak_center_ms": float(g["peak_sample"] / fs * 1000.0),
        "peak_z": float(g["peak_z"]),
        "num_peaks": int(len(g["peaks"])),
    } for g in peak_groups]
    return kept, dropped, peak_summary


def filter_annotations_by_yolo_width(
    annotations: list[dict],
    signal_length: int,
    fs: float,
    min_width_ms: float | None = None,
    max_width_ms: float | None = None,
) -> tuple[list[dict], list[dict]]:
    kept = []
    dropped = []
    for ann in annotations:
        width_ms = max(0.0, float(ann["end"]) - float(ann["start"])) * signal_length / fs * 1000.0
        reason = None
        if min_width_ms is not None and width_ms < float(min_width_ms):
            reason = "yolo_width_below_min"
        elif max_width_ms is not None and width_ms > float(max_width_ms):
            reason = "yolo_width_above_max"
        if reason is None:
            kept.append(ann)
            continue
        dropped.append({
            "reason": reason,
            "width_ms": float(width_ms),
            "passage_time_ms": ann.get("passage_time_ms"),
            "snr_db": ann.get("snr_db"),
            "frequency": ann.get("frequency"),
            "start_ms": float(ann["start"]) * signal_length / fs * 1000.0,
            "end_ms": float(ann["end"]) * signal_length / fs * 1000.0,
        })
    for ann_id, ann in enumerate(kept):
        ann["id"] = int(ann_id)
    return kept, dropped


def update_annotation_geometry(ann: dict) -> None:
    start = min(1.0, max(0.0, float(ann["start"])))
    end = min(1.0, max(start, float(ann["end"])))
    center = (start + end) / 2.0
    half_width = (end - start) / 2.0
    ann["start"] = float(start)
    ann["end"] = float(end)
    ann["center"] = float(center)
    ann["mean"] = float(center)
    ann["half_width"] = float(half_width)


def resolve_annotation_boundary_crossings(
    annotations: list[dict],
    signal_length: int,
    fs: float,
    min_width_ms: float | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    if len(annotations) < 2:
        for ann_id, ann in enumerate(annotations):
            update_annotation_geometry(ann)
            ann["id"] = int(ann_id)
        return annotations, [], []

    adjusted = [dict(ann) for ann in annotations]
    adjusted.sort(key=lambda ann: (float(ann.get("center", ann.get("mean", 0.0))), float(ann["start"])))
    edits = []
    for left, right in zip(adjusted, adjusted[1:]):
        left_end = float(left["end"])
        right_start = float(right["start"])
        if left_end <= right_start:
            continue
        boundary = min(1.0, max(0.0, (left_end + right_start) / 2.0))
        old_left_end = left_end
        old_right_start = right_start
        left["end"] = boundary
        right["start"] = boundary
        for ann, side, old_value, new_value in (
            (left, "end", old_left_end, boundary),
            (right, "start", old_right_start, boundary),
        ):
            ann["boundary_adjusted"] = True
            ann.setdefault("boundary_adjustments", []).append({
                "side": side,
                "old": float(old_value),
                "new": float(new_value),
                "reason": "adjacent_overlap_midpoint",
            })
        edits.append({
            "reason": "adjacent_overlap_midpoint",
            "left_original_id": left.get("id"),
            "right_original_id": right.get("id"),
            "left_class_id": left.get("class_id"),
            "right_class_id": right.get("class_id"),
            "old_left_end_ms": float(old_left_end) * signal_length / fs * 1000.0,
            "old_right_start_ms": float(old_right_start) * signal_length / fs * 1000.0,
            "new_boundary_ms": float(boundary) * signal_length / fs * 1000.0,
        })

    kept = []
    dropped = []
    for ann in adjusted:
        update_annotation_geometry(ann)
        width_ms = (float(ann["end"]) - float(ann["start"])) * signal_length / fs * 1000.0
        if min_width_ms is not None and width_ms < float(min_width_ms):
            dropped.append({
                "reason": "boundary_width_below_min",
                "stage": "boundary_resolution",
                "width_ms": float(width_ms),
                "passage_time_ms": ann.get("passage_time_ms"),
                "snr_db": ann.get("snr_db"),
                "frequency": ann.get("frequency"),
                "start_ms": float(ann["start"]) * signal_length / fs * 1000.0,
                "end_ms": float(ann["end"]) * signal_length / fs * 1000.0,
            })
            continue
        kept.append(ann)
    for ann_id, ann in enumerate(kept):
        ann["id"] = int(ann_id)
    return kept, dropped, edits


def peak_evidence_reference_path(signal: dict, clean_root: Path | None) -> Path | None:
    if clean_root is None:
        return None
    class_name = signal.get("class")
    filename = signal.get("filename")
    if not class_name or not filename:
        return None
    candidate = clean_root / str(class_name) / str(filename)
    return candidate if candidate.exists() else None


def annotate_clean_peak_support(
    particles: list[dict],
    signal_values: np.ndarray,
    signal_length: int,
    fs: float,
    envelope_window_ms: float,
    min_z: float,
    prominence_z: float,
    min_separation_ms: float,
    valley_ratio: float,
) -> tuple[list[dict], list[dict], list[dict]]:
    peak_groups, _envelope, z_values, _baseline, _scale = detect_peak_groups(
        signal_values,
        fs,
        envelope_window_ms=envelope_window_ms,
        min_z=min_z,
        prominence_z=prominence_z,
        min_separation_ms=min_separation_ms,
        valley_ratio=valley_ratio,
    )
    kept = []
    dropped = []
    for particle in particles:
        enriched = annotate_particle_peak_evidence(
            particle,
            signal_length,
            fs,
            peak_groups,
            z_values,
        )
        if enriched.get("peak_support"):
            out = dict(particle)
            out["clean_peak_support"] = True
            out["clean_peak_group_id"] = enriched.get("peak_group_id")
            out["clean_peak_z"] = enriched.get("peak_z")
            out["clean_peak_center_ms"] = enriched.get("peak_center_ms")
            out["clean_local_peak_z"] = enriched.get("local_peak_z")
            kept.append(out)
            continue
        dropped.append({
            "reason": "missing_clean_peak_support",
            "stage": "clean_peak_evidence",
            "peak_support": False,
            "peak_z": enriched.get("peak_z"),
            "local_peak_z": enriched.get("local_peak_z"),
            "snr_db": particle.get("snr_db"),
            "frequency": particle.get("frequency"),
            "passage_time_ms": particle_passage_time_ms(particle),
        })
    peak_summary = [{
        "id": int(g["id"]),
        "peak_center_ms": float(g["peak_sample"] / fs * 1000.0),
        "peak_z": float(g["peak_z"]),
        "num_peaks": int(len(g["peaks"])),
        "source": "clean_no_bandpass",
    } for g in peak_groups]
    return kept, dropped, peak_summary


def export_yolo_json(dataset_results_path: Path, output_json_path: Path,
                     class_names: tuple[str, ...], fs: float,
                     min_passage_time_ms: float | None = None,
                     max_passage_time_ms: float | None = None,
                     merge_overlaps: bool = True,
                     merge_iou_threshold: float = 0.4,
                     merge_center_distance_ms: float | None = None,
                     merge_duplicate_iou_threshold: float = 0.6,
                     merge_close_center_distance_ms: float | None = 0.20,
                     merge_ambiguous_center_distance_ms: float | None = 0.30,
                     merge_close_frequency_hz: float | None = 6000.0,
                     merge_ambiguous_frequency_hz: float | None = 8000.0,
                     merge_snr_margin_db: float | None = 4.0,
                     peak_evidence_filter: bool = True,
                     peak_envelope_window_ms: float = 0.08,
                     peak_min_z: float = 4.0,
                     peak_prominence_z: float = 2.0,
                     peak_min_separation_ms: float = 0.18,
                     peak_group_valley_ratio: float = 0.55,
                     peak_cluster_gap_ms: float = 0.25,
                     peak_keep_high_snr_db: float = 4.0,
                     merge_score: str = "snr_db",
                     yolo_width_filter: bool = True,
                     min_yolo_width_ms: float | None = 0.08,
                     max_yolo_width_ms: float | None = 1.5,
                     resolve_boundary_crossings: bool = True,
                     boundary_min_width_ms: float | None = None,
                     peak_evidence_signal_mode: str = "filtered",
                     peak_evidence_clean_root: Path | None = None) -> dict:
    if peak_evidence_signal_mode not in {"filtered", "clean", "dual_clean"}:
        raise ValueError(
            "peak_evidence_signal_mode must be one of: filtered, clean, dual_clean"
        )
    with dataset_results_path.open() as f:
        results = json.load(f)

    class_to_id = {class_name: idx for idx, class_name in enumerate(class_names)}
    data_rows = []
    for sample_id, signal in enumerate(results.get("signals", [])):
        class_name = signal.get("class")
        class_id = class_to_id.get(class_name)
        if class_id is None:
            continue
        length = int(signal.get("signal_length") or 0)
        filtered_particles = []
        dropped_annotations = []
        for particle in signal.get("particles", []):
            keep, reason, tau_ms = keep_particle_by_passage_time(
                particle, min_passage_time_ms, max_passage_time_ms
            )
            if not keep:
                dropped_annotations.append({
                    "reason": reason,
                    "passage_time_ms": tau_ms,
                    "snr_db": particle.get("snr_db"),
                    "frequency": particle.get("frequency"),
                })
                continue
            filtered_particles.append(particle)
        peak_summary = []
        if yolo_width_filter:
            filtered_particles, width_drops = filter_particles_by_yolo_width(
                filtered_particles,
                length,
                fs,
                min_width_ms=min_yolo_width_ms,
                max_width_ms=max_yolo_width_ms,
            )
            dropped_annotations.extend(width_drops)
        if peak_evidence_filter:
            signal_path = signal.get("path")
            clean_path = peak_evidence_reference_path(signal, peak_evidence_clean_root)
            primary_path = clean_path if peak_evidence_signal_mode == "clean" else Path(signal_path) if signal_path else None
            if primary_path and primary_path.exists():
                signal_values = np.load(primary_path)
                filtered_particles, peak_drops, peak_summary = refine_particles_with_peak_evidence(
                    filtered_particles,
                    signal_values,
                    length,
                    fs,
                    envelope_window_ms=peak_envelope_window_ms,
                    min_z=peak_min_z,
                    prominence_z=peak_prominence_z,
                    min_separation_ms=peak_min_separation_ms,
                    valley_ratio=peak_group_valley_ratio,
                    cluster_gap_ms=peak_cluster_gap_ms,
                    keep_high_snr_db=peak_keep_high_snr_db,
                )
                dropped_annotations.extend(peak_drops)
                if peak_evidence_signal_mode == "dual_clean":
                    if clean_path and clean_path.exists():
                        clean_values = np.load(clean_path)
                        filtered_particles, clean_drops, clean_peak_summary = annotate_clean_peak_support(
                            filtered_particles,
                            clean_values,
                            length,
                            fs,
                            envelope_window_ms=peak_envelope_window_ms,
                            min_z=peak_min_z,
                            prominence_z=peak_prominence_z,
                            min_separation_ms=peak_min_separation_ms,
                            valley_ratio=peak_group_valley_ratio,
                        )
                        dropped_annotations.extend(clean_drops)
                        peak_summary = [
                            {**row, "source": row.get("source", "filtered")}
                            for row in peak_summary
                        ]
                        peak_summary.extend(clean_peak_summary)
                    else:
                        dropped_annotations.append({
                            "reason": "missing_clean_signal_for_dual_peak_evidence",
                            "stage": "clean_peak_evidence",
                            "path": str(clean_path) if clean_path else None,
                        })
            else:
                dropped_annotations.append({
                    "reason": "missing_signal_for_peak_evidence",
                    "stage": "peak_evidence",
                    "path": str(primary_path) if primary_path else signal_path,
                })
        if merge_overlaps:
            filtered_particles, nms_drops = merge_overlapping_particles(
                filtered_particles,
                length,
                fs,
                iou_threshold=merge_iou_threshold,
                score_name=merge_score,
                center_distance_ms=merge_center_distance_ms,
                duplicate_iou_threshold=merge_duplicate_iou_threshold,
                close_center_distance_ms=merge_close_center_distance_ms,
                ambiguous_center_distance_ms=merge_ambiguous_center_distance_ms,
                close_frequency_hz=merge_close_frequency_hz,
                ambiguous_frequency_hz=merge_ambiguous_frequency_hz,
                snr_margin_db=merge_snr_margin_db,
            )
            dropped_annotations.extend(nms_drops)
        annotations = [
            particle_to_annotation(particle, length, fs, class_id, ann_id)
            for ann_id, particle in enumerate(filtered_particles)
        ]
        if yolo_width_filter:
            annotations, width_drops = filter_annotations_by_yolo_width(
                annotations,
                length,
                fs,
                min_width_ms=min_yolo_width_ms,
                max_width_ms=max_yolo_width_ms,
            )
            for drop in width_drops:
                drop["stage"] = "post_annotation"
            dropped_annotations.extend(width_drops)
        boundary_adjustments = []
        if resolve_boundary_crossings:
            boundary_min_ms = min_yolo_width_ms if boundary_min_width_ms is None else boundary_min_width_ms
            annotations, boundary_drops, boundary_adjustments = resolve_annotation_boundary_crossings(
                annotations,
                length,
                fs,
                min_width_ms=boundary_min_ms,
            )
            dropped_annotations.extend(boundary_drops)
        data_rows.append({
            "filename": signal.get("filename"),
            "path": signal.get("path"),
            "id": int(sample_id),
            "class_id": int(class_id),
            "class_name": class_name,
            "length": length,
            "annotations": annotations,
            "dropped_annotations": dropped_annotations,
            "peak_groups": peak_summary,
            "boundary_adjustments": boundary_adjustments,
        })

    output = {
        "info": {
            "description": "particles2SNR-derived particle detection dataset",
            "version": "1.0",
            "source_results": str(dataset_results_path),
            "annotation_source": "particles2SNR_pipeline",
            "passage_time_filter": {
                "field": "tau",
                "min_ms": min_passage_time_ms,
                "max_ms": max_passage_time_ms,
            },
            "overlap_merge": {
                "enabled": bool(merge_overlaps),
                "method": "conditional_temporal_nms",
                "iou_threshold": float(merge_iou_threshold),
                "legacy_center_distance_ms": merge_center_distance_ms,
                "duplicate_iou_threshold": float(merge_duplicate_iou_threshold),
                "close_center_distance_ms": merge_close_center_distance_ms,
                "ambiguous_center_distance_ms": merge_ambiguous_center_distance_ms,
                "close_frequency_hz": merge_close_frequency_hz,
                "ambiguous_frequency_hz": merge_ambiguous_frequency_hz,
                "snr_margin_db": merge_snr_margin_db,
                "score": merge_score,
            },
            "peak_evidence_filter": {
                "enabled": bool(peak_evidence_filter),
                "signal_mode": peak_evidence_signal_mode,
                "clean_root": str(peak_evidence_clean_root) if peak_evidence_clean_root is not None else None,
                "envelope_window_ms": peak_envelope_window_ms,
                "min_z": peak_min_z,
                "prominence_z": peak_prominence_z,
                "min_separation_ms": peak_min_separation_ms,
                "group_valley_ratio": peak_group_valley_ratio,
                "cluster_gap_ms": peak_cluster_gap_ms,
                "keep_high_snr_db": peak_keep_high_snr_db,
            },
            "yolo_width_filter": {
                "enabled": bool(yolo_width_filter),
                "min_ms": min_yolo_width_ms,
                "max_ms": max_yolo_width_ms,
            },
            "boundary_resolution": {
                "enabled": bool(resolve_boundary_crossings),
                "method": "adjacent_overlap_midpoint",
                "min_width_ms": min_yolo_width_ms if boundary_min_width_ms is None else boundary_min_width_ms,
            },
        },
        "classes": [
            {"id": idx, "name": class_name}
            for idx, class_name in enumerate(class_names)
        ],
        "data": data_rows,
    }
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with output_json_path.open("w") as f:
        json.dump(output, f, indent=2)
    return output


def split_train_val_data(data_json: dict, val_fraction: float, seed: int) -> tuple[dict, dict]:
    if not 0.0 < val_fraction < 1.0:
        return data_json, {**data_json, "data": []}
    rng = np.random.default_rng(seed)
    by_class: dict[str, list[dict]] = {}
    for row in data_json.get("data", []):
        by_class.setdefault(str(row.get("class_name", "")), []).append(row)
    train_rows = []
    val_rows = []
    for rows in by_class.values():
        rows = list(rows)
        indices = np.arange(len(rows))
        rng.shuffle(indices)
        n_val = max(1, int(round(len(rows) * val_fraction))) if len(rows) > 1 else 0
        n_val = min(n_val, len(rows) - 1) if len(rows) > 1 else 0
        val_idx = {int(i) for i in indices[:n_val]}
        for idx, row in enumerate(rows):
            if idx in val_idx:
                val_rows.append(row)
            else:
                train_rows.append(row)
    train_data = {**data_json, "data": sorted(train_rows, key=lambda row: row.get("filename", ""))}
    val_data = {**data_json, "data": sorted(val_rows, key=lambda row: row.get("filename", ""))}
    return train_data, val_data


def export_detseg_yolo_layout(data_json: dict, output_split_dir: Path) -> dict:
    """Export P1/detseg-compatible ``signals`` + ``labels`` folders."""
    signals_dir = output_split_dir / "signals"
    labels_dir = output_split_dir / "labels"
    if signals_dir.exists():
        shutil.rmtree(signals_dir)
    if labels_dir.exists():
        shutil.rmtree(labels_dir)
    signals_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    seen_names = set()
    copied = 0
    labels = 0
    for row in data_json.get("data", []):
        source_path = Path(row["path"])
        filename = source_path.name
        if filename in seen_names:
            filename = f"{row['class_name']}__{filename}"
        seen_names.add(filename)

        signal_out = signals_dir / filename
        shutil.copy2(source_path, signal_out)
        copied += 1

        label_path = labels_dir / f"{Path(filename).stem}.txt"
        lines = []
        for ann in row.get("annotations", []):
            if "start" in ann and "end" in ann:
                left = min(1.0, max(0.0, float(ann["start"])))
                right = min(1.0, max(left, float(ann["end"])))
            else:
                center = min(1.0, max(0.0, float(ann.get("center", ann.get("mean", 0.0)))))
                half_width = min(0.5, max(0.0, float(ann.get("half_width", 0.0))))
                left = max(0.0, center - half_width)
                right = min(1.0, center + half_width)
            center = (left + right) / 2.0
            width = right - left
            lines.append(f"{int(ann['class_id'])} {center:.10f} {width:.10f}")
        labels += len(lines)
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))

    return {
        "split_dir": str(output_split_dir),
        "signals": copied,
        "labels": labels,
    }


def ensure_empty_detseg_split(root: Path, split: str) -> None:
    (root / split / "signals").mkdir(parents=True, exist_ok=True)
    (root / split / "labels").mkdir(parents=True, exist_ok=True)


def write_detseg_dataset_yaml(root: Path, split_summaries: dict[str, dict],
                              class_names: tuple[str, ...]) -> None:
    """Write minimal metadata expected by P1/detseg audit tooling."""
    root.mkdir(parents=True, exist_ok=True)
    lines = [
        "names:",
        *[f"  - {name}" for name in class_names],
        "splits:",
    ]
    for split in ("train", "val", "test"):
        total = int(split_summaries.get(split, {}).get("signals", 0))
        lines.extend([
            f"  {split}:",
            f"    total: {total}",
        ])
    lines.extend([
        "generation_params:",
        "  source: particles2SNR_pipeline/generate_particles2SNR_dataset.py",
        "  annotation_source: particles2SNR_pipeline",
        "  saturation_policy: replace",
        "  min_passage_time_ms: 0.07",
        "  max_passage_time_ms: 0.65",
        "  merge_overlaps: true",
        "  merge_iou_threshold: 0.4",
        "  merge_duplicate_iou_threshold: 0.6",
        "  merge_close_center_distance_ms: 0.20",
        "  merge_ambiguous_center_distance_ms: 0.30",
        "  merge_close_frequency_hz: 6000",
        "  merge_ambiguous_frequency_hz: 8000",
        "  merge_snr_margin_db: 4.0",
        "  peak_evidence_filter: true",
        "  peak_evidence_signal_mode: dual_clean",
        "  peak_evidence_clean_root: particles2SNR split artifact / peak_evidence_clean_signals",
        "  peak_envelope_window_ms: 0.08",
        "  peak_min_z: 4.0",
        "  peak_prominence_z: 2.0",
        "  peak_min_separation_ms: 0.18",
        "  peak_group_valley_ratio: 0.55",
        "  peak_cluster_gap_ms: 0.25",
        "  peak_keep_high_snr_db: 4.0",
        "  merge_score: snr_db",
        "  yolo_width_filter: true",
        "  min_yolo_width_ms: 0.08",
        "  max_yolo_width_ms: 1.5",
        "  resolve_boundary_crossings: true",
        "  boundary_resolution_method: adjacent_overlap_midpoint",
        "preprocessing:",
        "  bandpass:",
        "    enabled: true",
        "    low_hz: 7000",
        "    high_hz: 80000",
        "    order: 4",
        "audit_results:",
        "  saturation:",
        "    status: pass",
        "    post_clean_required: true",
    ])
    (root / "dataset.yaml").write_text("\n".join(lines) + "\n")


def write_run_summary(path: Path, split_summaries: list[dict], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump({
            "input_root": str(Path(args.input_root).resolve()),
            "class_source_dirs": {
                key: str(path.resolve())
                for key, path in (args.class_source_dirs or {}).items()
            },
            "output_root": str(Path(args.output_root).resolve()),
            "particles2SNR_output": str(Path(args.particles2SNR_output).resolve()),
            "splits": list(args.splits),
            "classes": list(args.classes),
            "test_fraction": args.test_fraction,
            "val_fraction": args.val_fraction,
            "split_seed": args.split_seed,
            "zero_epsilon": args.zero_epsilon,
            "max_zero_run_after_clean": args.max_zero_run_after_clean,
            "fs": args.fs,
            "saturation_policy": args.saturation_policy,
            "noise_dir": args.noise_dir,
            "saturation_guard_before": args.saturation_guard_before,
            "saturation_guard_after": args.saturation_guard_after,
            "apply_bandpass_output": args.apply_bandpass_output,
            "bandpass_fmin": args.bandpass_fmin,
            "bandpass_fmax": args.bandpass_fmax,
            "bandpass_order": args.bandpass_order,
            "min_passage_time_ms": args.min_passage_time_ms,
            "max_passage_time_ms": args.max_passage_time_ms,
            "merge_overlaps": args.merge_overlaps,
            "merge_iou_threshold": args.merge_iou_threshold,
            "merge_center_distance_ms": args.merge_center_distance_ms,
            "merge_duplicate_iou_threshold": args.merge_duplicate_iou_threshold,
            "merge_close_center_distance_ms": args.merge_close_center_distance_ms,
            "merge_ambiguous_center_distance_ms": args.merge_ambiguous_center_distance_ms,
            "merge_close_frequency_hz": args.merge_close_frequency_hz,
            "merge_ambiguous_frequency_hz": args.merge_ambiguous_frequency_hz,
            "merge_snr_margin_db": args.merge_snr_margin_db,
            "peak_evidence_filter": args.peak_evidence_filter,
            "peak_evidence_signal_mode": args.peak_evidence_signal_mode,
            "peak_evidence_clean_root": "per-split peak_evidence_clean_signals",
            "peak_envelope_window_ms": args.peak_envelope_window_ms,
            "peak_min_z": args.peak_min_z,
            "peak_prominence_z": args.peak_prominence_z,
            "peak_min_separation_ms": args.peak_min_separation_ms,
            "peak_group_valley_ratio": args.peak_group_valley_ratio,
            "peak_cluster_gap_ms": args.peak_cluster_gap_ms,
            "peak_keep_high_snr_db": args.peak_keep_high_snr_db,
            "merge_score": args.merge_score,
            "yolo_width_filter": args.yolo_width_filter,
            "min_yolo_width_ms": args.min_yolo_width_ms,
            "max_yolo_width_ms": args.max_yolo_width_ms,
            "resolve_boundary_crossings": args.resolve_boundary_crossings,
            "boundary_min_width_ms": args.boundary_min_width_ms,
            "fail_on_post_saturation": args.fail_on_post_saturation,
            "split_summaries": split_summaries,
        }, f, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a zero-cleaned P0 dataset with particles2SNR ground truth."
    )
    parser.add_argument("--input-root", default="P0/data/processed/dataset")
    parser.add_argument("--class-source-dirs", type=parse_class_source_dirs, default=None,
                        help="Comma-separated class=dir sources; creates train/test splits before particles2SNR.")
    parser.add_argument("--staging-root", default=None,
                        help="Intermediate split tree for --class-source-dirs.")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--output-root", default="P0/data/processed/dataset_Particles2SNR_F_c1")
    parser.add_argument("--particles2SNR-output", default=str(RESULTS_RUNS / "p0_c1_Particles2SNR_F"))
    parser.add_argument("--detseg-output", default="P0/data/processed/dataset_Particles2SNR_F_c1_yolo_trainval",
                        help="P1/detseg YOLO layout output root; use '' to disable.")
    parser.add_argument("--splits", type=parse_csv_arg, default=DEFAULT_SPLITS)
    parser.add_argument("--classes", type=parse_csv_arg, default=DEFAULT_CLASSES)
    parser.add_argument("--zero-epsilon", type=float, default=0.0)
    parser.add_argument("--max-zero-run-after-clean", type=int, default=2)
    parser.add_argument("--fs", type=float, default=DEFAULT_FS)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--skip-particles2SNR", action="store_true",
                        help="Only clean/copy and saturation-scan the dataset.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--saturation-fmin", type=float, default=7000.0)
    parser.add_argument("--saturation-fmax", type=float, default=80000.0)
    parser.add_argument("--saturation-min-flat", type=int, default=500)
    parser.add_argument("--saturation-zero-threshold", type=float, default=1e-4)
    parser.add_argument("--saturation-policy", choices=("replace", "mask", "keep"), default="replace")
    parser.add_argument("--saturation-guard-before", type=int, default=300)
    parser.add_argument("--saturation-guard-after", type=int, default=300)
    parser.add_argument("--saturation-mask-value", type=float, default=0.0)
    parser.add_argument("--noise-dir", default="P0/data/processed/Noise")
    parser.add_argument("--apply-bandpass-output", dest="apply_bandpass_output", action="store_true", default=True)
    parser.add_argument("--no-apply-bandpass-output", dest="apply_bandpass_output", action="store_false")
    parser.add_argument("--bandpass-fmin", type=float, default=DEFAULT_BANDPASS_FMIN)
    parser.add_argument("--bandpass-fmax", type=float, default=DEFAULT_BANDPASS_FMAX)
    parser.add_argument("--bandpass-order", type=int, default=DEFAULT_BANDPASS_ORDER)
    parser.add_argument("--min-passage-time-ms", type=float, default=0.07)
    parser.add_argument("--max-passage-time-ms", type=float, default=0.65)
    parser.add_argument("--merge-overlaps", dest="merge_overlaps", action="store_true", default=True)
    parser.add_argument("--no-merge-overlaps", dest="merge_overlaps", action="store_false")
    parser.add_argument("--merge-iou-threshold", type=float, default=0.4)
    parser.add_argument("--merge-center-distance-ms", type=float, default=None,
                        help="Deprecated single center-distance merge gate; disabled by default.")
    parser.add_argument("--merge-duplicate-iou-threshold", type=float, default=0.6)
    parser.add_argument("--merge-close-center-distance-ms", type=float, default=0.20)
    parser.add_argument("--merge-ambiguous-center-distance-ms", type=float, default=0.30)
    parser.add_argument("--merge-close-frequency-hz", type=float, default=6000.0)
    parser.add_argument("--merge-ambiguous-frequency-hz", type=float, default=8000.0)
    parser.add_argument("--merge-snr-margin-db", type=float, default=4.0)
    parser.add_argument("--peak-evidence-filter", dest="peak_evidence_filter", action="store_true", default=True)
    parser.add_argument("--disable-peak-evidence-filter", dest="peak_evidence_filter", action="store_false")
    parser.add_argument("--peak-evidence-signal-mode", choices=("filtered", "clean", "dual_clean"), default="dual_clean",
                        help="Signal used for peak evidence. dual_clean keeps filtered evidence but also requires support in the cleaned non-bandpassed signal.")
    parser.add_argument("--peak-envelope-window-ms", type=float, default=0.08)
    parser.add_argument("--peak-min-z", type=float, default=4.0)
    parser.add_argument("--peak-prominence-z", type=float, default=2.0)
    parser.add_argument("--peak-min-separation-ms", type=float, default=0.18)
    parser.add_argument("--peak-group-valley-ratio", type=float, default=0.55)
    parser.add_argument("--peak-cluster-gap-ms", type=float, default=0.25)
    parser.add_argument("--peak-keep-high-snr-db", type=float, default=4.0)
    parser.add_argument("--merge-score", choices=("snr_db", "energy"), default="snr_db")
    parser.add_argument("--yolo-width-filter", dest="yolo_width_filter", action="store_true", default=True)
    parser.add_argument("--disable-yolo-width-filter", dest="yolo_width_filter", action="store_false")
    parser.add_argument("--min-yolo-width-ms", type=float, default=0.08)
    parser.add_argument("--max-yolo-width-ms", type=float, default=1.5)
    parser.add_argument("--resolve-boundary-crossings", dest="resolve_boundary_crossings", action="store_true", default=True)
    parser.add_argument("--disable-resolve-boundary-crossings", dest="resolve_boundary_crossings", action="store_false")
    parser.add_argument("--boundary-min-width-ms", type=float, default=None,
                        help="Minimum label width after boundary midpoint resolution; defaults to --min-yolo-width-ms.")
    parser.add_argument("--fail-on-post-saturation", dest="fail_on_post_saturation", action="store_true", default=True)
    parser.add_argument("--no-fail-on-post-saturation", dest="fail_on_post_saturation", action="store_false")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.splits = tuple(args.splits)
    args.classes = tuple(args.classes)

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    particles2SNR_output = Path(args.particles2SNR_output)
    split_summaries = []
    detseg_summaries = {}
    rng = np.random.default_rng(args.split_seed)
    noise_pool = read_noise_pool(args.noise_dir, chunk_len=16_384)
    if args.saturation_policy == "replace" and not noise_pool:
        raise FileNotFoundError(f"No .npy noise chunks found in --noise-dir {args.noise_dir!r}")

    if args.class_source_dirs:
        args.classes = tuple(args.class_source_dirs.keys())
        staging_root = Path(args.staging_root or (str(particles2SNR_output) + "_staging"))
        split_manifest = prepare_split_tree_from_class_sources(
            args.class_source_dirs,
            staging_root,
            args.splits,
            args.test_fraction,
            args.split_seed,
        )
        write_csv(
            particles2SNR_output / "source_split_manifest.csv",
            split_manifest,
            ["class", "split", "source_path", "staged_path", "action"],
        )
        input_root = staging_root

    for split in args.splits:
        input_split_dir = input_root / split
        output_split_dir = output_root / split
        split_output_dir = particles2SNR_output / split
        split_output_dir.mkdir(parents=True, exist_ok=True)
        if not input_split_dir.is_dir():
            raise FileNotFoundError(f"Missing split directory: {input_split_dir}")

        peak_evidence_clean_dir = (
            split_output_dir / "peak_evidence_clean_signals"
            if args.peak_evidence_signal_mode in {"clean", "dual_clean"}
            else None
        )
        manifest_rows, saturation_rows, peak_evidence_rows = clean_split(
            input_split_dir,
            output_split_dir,
            args.classes,
            args.zero_epsilon,
            args.max_zero_run_after_clean,
            args,
            noise_pool,
            rng,
            split,
            peak_evidence_clean_dir,
        )
        write_csv(
            split_output_dir / "zero_cleaning_manifest.csv",
            manifest_rows,
            [
                "source_path", "output_path", "class", "filename",
                "interval_idx", "start_sample", "end_sample",
                "duration_samples", "kept_zero_samples", "removed_samples",
                "source_length", "clean_length", "clean_start_sample",
                "clean_end_sample", "action",
            ],
        )
        write_csv(
            split_output_dir / "saturation_cleaning_manifest.csv",
            saturation_rows,
            [
                "split", "source_path", "output_path", "class", "filename",
                "policy", "interval_idx", "start_sample", "end_sample",
                "duration_samples", "action", "dropped_events", "fs", "fmin",
                "fmax", "min_flat", "zero_threshold", "guard_before",
                "guard_after", "max_consecutive_flat",
            ],
        )
        write_csv(
            split_output_dir / "peak_evidence_signal_manifest.csv",
            peak_evidence_rows,
            [
                "split", "class", "filename", "source_path",
                "filtered_output_path", "peak_evidence_clean_path", "action",
            ],
        )
        post_sat = scan_saturation_split(
            output_split_dir, split_output_dir, args.classes, args,
            prefix="post_clean_saturation",
        )
        if args.fail_on_post_saturation and post_sat["total_saturated_files"] > 0:
            raise RuntimeError(
                f"Post-clean saturation audit failed for {split}: "
                f"{post_sat['total_saturated_files']} files still saturated"
            )
        if not args.skip_particles2SNR:
            run_particles2SNR_split(
                output_split_dir,
                split_output_dir,
                args.classes,
                args.device,
                args.verbose,
                args.bandpass_fmin if args.apply_bandpass_output else None,
                args.bandpass_fmax if args.apply_bandpass_output else None,
                args.bandpass_order if args.apply_bandpass_output else None,
            )
            data_json = export_yolo_json(
                split_output_dir / "dataset_results.json",
                split_output_dir / "data.json",
                args.classes,
                args.fs,
                args.min_passage_time_ms,
                args.max_passage_time_ms,
                args.merge_overlaps,
                args.merge_iou_threshold,
                args.merge_center_distance_ms,
                args.merge_duplicate_iou_threshold,
                args.merge_close_center_distance_ms,
                args.merge_ambiguous_center_distance_ms,
                args.merge_close_frequency_hz,
                args.merge_ambiguous_frequency_hz,
                args.merge_snr_margin_db,
                args.peak_evidence_filter,
                args.peak_envelope_window_ms,
                args.peak_min_z,
                args.peak_prominence_z,
                args.peak_min_separation_ms,
                args.peak_group_valley_ratio,
                args.peak_cluster_gap_ms,
                args.peak_keep_high_snr_db,
                args.merge_score,
                args.yolo_width_filter,
                args.min_yolo_width_ms,
                args.max_yolo_width_ms,
                args.resolve_boundary_crossings,
                args.boundary_min_width_ms,
                args.peak_evidence_signal_mode,
                peak_evidence_clean_dir,
            )
            if args.detseg_output:
                if split == "train" and args.val_fraction > 0:
                    train_data, val_data = split_train_val_data(
                        data_json, args.val_fraction, args.split_seed
                    )
                    detseg_summaries["train"] = export_detseg_yolo_layout(
                        train_data,
                        Path(args.detseg_output) / "train",
                    )
                    detseg_summaries["val"] = export_detseg_yolo_layout(
                        val_data,
                        Path(args.detseg_output) / "val",
                    )
                else:
                    detseg_summaries[split] = export_detseg_yolo_layout(
                        data_json,
                        Path(args.detseg_output) / split,
                    )

        split_summaries.append({
            "split": split,
            "source_files": len({row["source_path"] for row in manifest_rows}),
            "manifest_rows": len(manifest_rows),
            "removed_samples": int(sum(int(row.get("removed_samples") or 0) for row in manifest_rows)),
            "saturation_actions": int(sum(1 for row in saturation_rows if row.get("interval_idx") not in ("", None))),
            "post_clean_saturated_files": int(post_sat["total_saturated_files"]),
            "cleaned_dataset_dir": str(output_split_dir),
            "artifact_dir": str(split_output_dir),
        })

    if args.detseg_output:
        detseg_root = Path(args.detseg_output)
        for split in ("train", "val", "test"):
            ensure_empty_detseg_split(detseg_root, split)
        write_detseg_dataset_yaml(detseg_root, detseg_summaries, args.classes)

    write_run_summary(particles2SNR_output / "run_summary.json", split_summaries, args)
    print(f"Cleaned dataset: {output_root}")
    print(f"particles2SNR artifacts: {particles2SNR_output}")
    print(f"Run summary: {particles2SNR_output / 'run_summary.json'}")


if __name__ == "__main__":
    main()
