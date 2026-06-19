import importlib.util
import csv
import json
import os
import sys
import types
import unittest
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PARTICLES2SNR = os.path.join(ROOT, "particles2SNR_pipeline")
P0_SCRIPT = os.path.join(ROOT, "P0", "scripts", "conv1dgap_accuracy_vs_snr.py")
P1_SCRIPT = os.path.join(ROOT, "P1", "generate_long_sequence_dataset.py")
P1_AUDIT_SCRIPT = os.path.join(ROOT, "P1", "detseg", "audit_saturation_artifacts.py")
PARTICLES2SNR_GENERATOR = os.path.join(ROOT, "particles2SNR_pipeline", "generate_particles2SNR_dataset.py")
PARTICLES2SNR_YOLO_LIM10 = os.path.join(ROOT, "particles2SNR_pipeline", "create_particles2SNR_c1_yolo_4class_lim10.py")

for path in (PARTICLES2SNR, ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from saturation_cleaning import (  # noqa: E402
    clean_signal_non_destructive,
    detect_unsafe_intervals,
    drop_overlapping_events,
    expand_intervals,
    merge_intervals,
)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module




class VisualSignalCheckTests(unittest.TestCase):
    def test_overlap_segments_counts_active_labels(self):
        mod = load_module(os.path.join(ROOT, "particles2SNR_pipeline", "generate_visual_signal_checks.py"), "visual_signal_checks_test")
        labels = [
            {"start": 0.10, "end": 0.40},
            {"start": 0.25, "end": 0.60},
            {"start": 0.30, "end": 0.35},
        ]
        segments = mod.overlap_segments(labels)
        compact = [(round(s["start"], 2), round(s["end"], 2), s["overlap"]) for s in segments]
        self.assertEqual(compact, [
            (0.10, 0.25, 1),
            (0.25, 0.30, 2),
            (0.30, 0.35, 3),
            (0.35, 0.40, 2),
            (0.40, 0.60, 1),
        ])


class EventAccuracySnrTests(unittest.TestCase):
    def test_threshold_at_target_accuracy_interpolates_crossing(self):
        mod = load_module(os.path.join(ROOT, "particles2SNR_pipeline", "event_accuracy_vs_snr.py"), "event_accuracy_snr_test")
        bins = [
            {"snr_center": -5.0, "accuracy": 0.80},
            {"snr_center": 0.0, "accuracy": 0.90},
            {"snr_center": 5.0, "accuracy": 1.00},
        ]
        self.assertAlmostEqual(mod.threshold_at_target_accuracy(bins, 0.97), 3.5)
        self.assertIsNone(mod.threshold_at_target_accuracy(bins, 1.01))

    def test_crop_centered_zero_pads_at_edges(self):
        mod = load_module(os.path.join(ROOT, "particles2SNR_pipeline", "event_accuracy_vs_snr.py"), "event_accuracy_snr_crop_test")
        signal = np.arange(5, dtype=np.float32)
        left = mod.crop_centered(signal, center_sample=0, length=7)
        right = mod.crop_centered(signal, center_sample=4, length=7)
        self.assertEqual(left.tolist(), [0, 0, 0, 0, 1, 2, 3])
        self.assertEqual(right.tolist(), [1, 2, 3, 4, 0, 0, 0])


class EventAccuracyComparisonTests(unittest.TestCase):
    def load_compare_module(self):
        return load_module(
            os.path.join(ROOT, "particles2SNR_pipeline", "compare_event_accuracy_by_snr.py"),
            "event_accuracy_comparison_test",
        )

    def test_common_bins_use_union_of_runs(self):
        mod = self.load_compare_module()
        rows = [
            {"dataset_label": "a", "snr_db": -10.0},
            {"dataset_label": "a", "snr_db": 0.0},
            {"dataset_label": "b", "snr_db": 10.0},
            {"dataset_label": "b", "snr_db": 20.0},
        ]
        bins = mod.make_common_bins(rows, 2)
        self.assertEqual(len(bins), 2)
        self.assertAlmostEqual(bins[0][0], -10.0)
        self.assertAlmostEqual(bins[-1][1], 20.0)

    def test_balanced_sampling_is_deterministic_and_class_balanced(self):
        mod = self.load_compare_module()
        subset = []
        for cls, count in (("2um", 3), ("4um", 2), ("10um", 4)):
            for idx in range(count):
                subset.append({
                    "dataset_label": "run",
                    "event_key": f"{cls}-{idx}",
                    "true_class": cls,
                    "pred_class": cls,
                    "correct": True,
                    "snr_db": float(idx),
                })
        a, counts_a, reason_a = mod.sample_balanced_subset(subset, ("2um", "4um", "10um"), 42, "run", 0)
        b, counts_b, reason_b = mod.sample_balanced_subset(subset, ("2um", "4um", "10um"), 42, "run", 0)
        self.assertIsNone(reason_a)
        self.assertEqual(counts_a, {"2um": 3, "4um": 2, "10um": 4})
        self.assertEqual(counts_a, counts_b)
        self.assertEqual([row["event_key"] for row in a], [row["event_key"] for row in b])
        self.assertEqual(Counter(row["true_class"] for row in a), {"2um": 2, "4um": 2, "10um": 2})

    def test_balanced_stats_skip_bins_with_missing_class(self):
        mod = self.load_compare_module()
        rows = [
            {"dataset_label": "run", "true_class": "2um", "pred_class": "2um", "correct": True, "snr_db": -1.0},
            {"dataset_label": "run", "true_class": "4um", "pred_class": "2um", "correct": False, "snr_db": -0.5},
        ]
        stats = mod.bin_stats_by_dataset(rows, [(-2.0, 0.0)], ("2um", "4um", "10um"), 42, "balanced_class_snr")
        self.assertEqual(len(stats), 1)
        self.assertIsNone(stats[0]["accuracy"])
        self.assertEqual(stats[0]["n"], 0)
        self.assertIn("missing_class:10um", stats[0]["skip_reason"])

    def test_threshold_ignores_skipped_bins(self):
        mod = self.load_compare_module()
        bins = [
            {"snr_center": -5.0, "accuracy": None},
            {"snr_center": 0.0, "accuracy": 0.90},
            {"snr_center": 5.0, "accuracy": 1.00},
        ]
        self.assertAlmostEqual(mod.threshold_at_target_accuracy(bins, 0.95), 2.5)
        self.assertIsNone(mod.threshold_at_target_accuracy(bins, 1.01))

    def test_comparison_csv_schema_contains_class_counts(self):
        mod = self.load_compare_module()
        out = Path("/tmp/event_accuracy_comparison_schema.csv")
        rows = [mod.stat_row(
            [{"true_class": "2um", "pred_class": "2um", "correct": True}],
            ("2um",), "run", 0, -1.0, 1.0, "available",
        )]
        mod.write_csv(out, rows)
        with out.open(newline="") as f:
            reader = csv.DictReader(f)
            self.assertIn("n_by_class", reader.fieldnames)
            self.assertIn("sampled_n_by_class", reader.fieldnames)
            self.assertIn("accuracy", reader.fieldnames)



class Particles2SNRYoloLim10Tests(unittest.TestCase):
    def test_label_for_annotation_threshold_policy(self):
        mod = load_module(PARTICLES2SNR_YOLO_LIM10, "particles2SNR_yolo_lim10_test")

        below, reason = mod.label_for_annotation({"class_id": 0, "snr_db": -10.1}, -10.0)
        self.assertEqual(below, 3)
        self.assertEqual(reason, "below_threshold")

        equal, reason = mod.label_for_annotation({"class_id": 1, "snr_db": -10.0}, -10.0)
        self.assertEqual(equal, 1)
        self.assertEqual(reason, "kept_particle")

        missing, reason = mod.label_for_annotation({"class_id": 2}, -10.0)
        self.assertEqual(missing, 3)
        self.assertEqual(reason, "missing_or_invalid_snr")

    def test_normalized_interval_clips_bounds(self):
        mod = load_module(PARTICLES2SNR_YOLO_LIM10, "particles2SNR_yolo_lim10_test2")
        left, right = mod.normalized_interval({"center": 0.1, "half_width": 0.25})
        self.assertEqual(left, 0.0)
        self.assertAlmostEqual(right, 0.35)


class SaturationCleaningTests(unittest.TestCase):
    def test_merge_and_expand_intervals_clip_and_merge(self):
        merged = merge_intervals([(-5, 5), (4, 10), (20, 30)], signal_len=25)
        self.assertEqual(merged, [(0, 10), (20, 25)])

        expanded = expand_intervals(
            [{"start_sample": 10, "end_sample": 20},
             {"start_sample": 22, "end_sample": 24}],
            signal_len=40,
            guard_before=3,
            guard_after=2,
        )
        self.assertEqual(expanded, [(7, 26)])

    def test_drop_overlapping_events(self):
        events = [(0, 5, 0), (10, 20, 1), (25, 30, 2)]
        kept, dropped = drop_overlapping_events(events, [(12, 18)])
        self.assertEqual(kept, [(0, 5, 0), (25, 30, 2)])
        self.assertEqual(dropped, [(10, 20, 1)])


    def test_detect_unsafe_intervals_detects_nonzero_plateau(self):
        signal = np.zeros(2048, dtype=np.float64)
        signal[:256] = np.sin(np.linspace(0, 4 * np.pi, 256))
        signal[256:1700] = 2.5
        signal[1700:] = np.sin(np.linspace(0, 4 * np.pi, 348))
        sat_info, unsafe = detect_unsafe_intervals(
            signal,
            fs=2_000_000,
            fmin=7000,
            fmax=80000,
            min_flat=100,
            zero_threshold=1e-4,
            guard_before=0,
            guard_after=0,
        )
        self.assertTrue(sat_info["is_saturated"])
        self.assertGreaterEqual(sat_info["max_consecutive_flat"], 100)
        self.assertTrue(unsafe)

    def test_clean_signal_policies_do_not_mutate_source(self):
        signal = np.arange(10, dtype=float)
        source_copy = signal.copy()
        cleaned, actions = clean_signal_non_destructive(
            signal,
            [(2, 5)],
            policy="mask",
            mask_value=-1.0,
        )
        np.testing.assert_array_equal(signal, source_copy)
        np.testing.assert_array_equal(cleaned[2:5], np.asarray([-1.0, -1.0, -1.0]))
        self.assertEqual(actions[0]["action"], "masked")

        noise = [np.asarray([100.0, 101.0, 102.0])]
        replaced, actions = clean_signal_non_destructive(
            signal,
            [(0, 3)],
            policy="replace",
            noise_pool=noise,
            rng=np.random.default_rng(0),
        )
        np.testing.assert_array_equal(replaced[:3], noise[0])
        self.assertEqual(actions[0]["action"], "replaced_with_noise")


class Particles2SNRDatasetGeneratorTests(unittest.TestCase):
    def test_remove_long_zero_runs_keeps_two_zero_raccord(self):
        mod = load_module(PARTICLES2SNR_GENERATOR, "particles2SNR_dataset_generator_test")
        signal = np.asarray([1, 0, 0, 0, 0, 2, 0, 0, 3], dtype=np.float32)
        cleaned, actions = mod.remove_long_zero_runs(
            signal,
            zero_epsilon=0.0,
            max_zero_run_after_clean=2,
        )
        np.testing.assert_array_equal(
            cleaned,
            np.asarray([1, 0, 0, 2, 0, 0, 3], dtype=np.float32),
        )
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["removed_samples"], 2)
        self.assertEqual(actions[0]["kept_zero_samples"], 2)


    def test_bandpass_filter_reduces_out_of_band_energy(self):
        mod = load_module(PARTICLES2SNR_GENERATOR, "particles2SNR_dataset_generator_bandpass_test")
        fs = 2_000_000.0
        t = np.arange(4096) / fs
        inband = np.sin(2 * np.pi * 20_000 * t)
        outband = np.sin(2 * np.pi * 200_000 * t)
        signal = (inband + outband).astype(np.float64)
        filtered = mod.butter_bandpass_filter(signal, fs, 7000, 80000, order=4)
        freqs = np.fft.rfftfreq(len(signal), 1 / fs)
        raw = np.abs(np.fft.rfft(signal)) ** 2
        filt = np.abs(np.fft.rfft(filtered)) ** 2
        raw_out = raw[freqs > 100_000].sum()
        filt_out = filt[freqs > 100_000].sum()
        self.assertLess(filt_out, raw_out * 0.05)

    def test_split_train_val_data_is_stratified_and_deterministic(self):
        mod = load_module(PARTICLES2SNR_GENERATOR, "particles2SNR_dataset_generator_split_test")
        rows = []
        for cls in ("2um", "4um"):
            for idx in range(5):
                rows.append({"filename": f"{cls}_{idx}.npy", "class_name": cls})
        data = {"data": rows, "classes": []}
        train_a, val_a = mod.split_train_val_data(data, val_fraction=0.4, seed=0)
        train_b, val_b = mod.split_train_val_data(data, val_fraction=0.4, seed=0)
        self.assertEqual(val_a["data"], val_b["data"])
        self.assertEqual(len(train_a["data"]), 6)
        self.assertEqual(len(val_a["data"]), 4)
        self.assertEqual(
            {row["class_name"] for row in val_a["data"]},
            {"2um", "4um"},
        )

    def test_clean_split_replaces_saturation_and_writes_filtered_signal(self):
        mod = load_module(PARTICLES2SNR_GENERATOR, "particles2SNR_dataset_generator_clean_split_test")
        tmp = Path("/tmp/Particles2SNR_F_clean_split_test")
        src = tmp / "src" / "train" / "10um"
        out = tmp / "out" / "train"
        noise_dir = tmp / "noise"
        src.mkdir(parents=True, exist_ok=True)
        noise_dir.mkdir(parents=True, exist_ok=True)
        signal = np.zeros(2048, dtype=np.float64)
        signal[:256] = np.sin(np.linspace(0, 8 * np.pi, 256))
        signal[256:1800] = 3.0
        signal[1800:] = np.sin(np.linspace(0, 8 * np.pi, 248))
        np.save(src / "sample.npy", signal)
        np.save(noise_dir / "noise.npy", np.random.default_rng(0).normal(0, 0.1, 4096))
        args = types.SimpleNamespace(
            fs=2_000_000.0,
            saturation_fmin=7000.0,
            saturation_fmax=80000.0,
            saturation_min_flat=100,
            saturation_zero_threshold=1e-4,
            saturation_guard_before=0,
            saturation_guard_after=0,
            saturation_policy="replace",
            saturation_mask_value=0.0,
            apply_bandpass_output=True,
            bandpass_fmin=7000.0,
            bandpass_fmax=80000.0,
            bandpass_order=4,
        )
        noise_pool = mod.read_noise_pool(str(noise_dir), chunk_len=2048)
        zero_rows, sat_rows = mod.clean_split(
            tmp / "src" / "train", out, ("10um",), 0.0, 2, args,
            noise_pool, np.random.default_rng(0), "train",
        )
        self.assertEqual(len(zero_rows), 1)
        self.assertTrue(any(row["action"] == "replaced_with_noise" for row in sat_rows))
        cleaned = np.load(out / "10um" / "sample.npy")
        self.assertEqual(cleaned.shape, signal.shape)
        self.assertFalse(np.allclose(cleaned[256:1800], 3.0))


    def test_passage_time_filter_uses_particles2SNR_tau_not_yolo_width(self):
        mod = load_module(PARTICLES2SNR_GENERATOR, "particles2SNR_dataset_generator_passage_test")
        self.assertEqual(
            mod.keep_particle_by_passage_time({"tau": 0.00005}, 0.07, 0.65)[1],
            "passage_time_below_min",
        )
        self.assertEqual(
            mod.keep_particle_by_passage_time({"tau": 0.00070}, 0.07, 0.65)[1],
            "passage_time_above_max",
        )
        keep, reason, tau_ms = mod.keep_particle_by_passage_time({"tau": 0.00013}, 0.07, 0.65)
        self.assertTrue(keep)
        self.assertEqual(reason, "kept")
        self.assertAlmostEqual(tau_ms, 0.13)

    def test_export_yolo_json_applies_passage_time_filter(self):
        mod = load_module(PARTICLES2SNR_GENERATOR, "particles2SNR_dataset_generator_passage_export_test")
        tmp_dir = Path("/tmp/particles2SNR_passage_export_test")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        np.save(tmp_dir / "sample.npy", np.zeros(1000, dtype=np.float32))
        results_path = tmp_dir / "dataset_results.json"
        output_path = tmp_dir / "data.json"
        with results_path.open("w") as f:
            json.dump({
                "signals": [{
                    "filename": "sample.npy",
                    "path": str(tmp_dir / "sample.npy"),
                    "class": "2um",
                    "signal_length": 1000,
                    "particles": [
                        {"t0": 0.0001, "tau": 0.00005, "P0": 0.5, "frequency": 20000, "snr_db": 3},
                        {"t0": 0.0002, "tau": 0.00010, "P0": 0.5, "frequency": 22000, "snr_db": 4},
                        {"t0": 0.0003, "tau": 0.00080, "P0": 0.5, "frequency": 24000, "snr_db": 5},
                    ],
                }]
            }, f)
        data = mod.export_yolo_json(
            results_path, output_path, ("2um", "4um", "10um"), 2_000_000.0,
            min_passage_time_ms=0.07, max_passage_time_ms=0.65,
            peak_evidence_filter=False,
        )
        row = data["data"][0]
        self.assertEqual(len(row["annotations"]), 1)
        self.assertEqual(len(row["dropped_annotations"]), 2)
        self.assertAlmostEqual(row["annotations"][0]["passage_time_ms"], 0.10)
        self.assertEqual(data["info"]["passage_time_filter"]["min_ms"], 0.07)


    def test_merge_overlapping_particles_keeps_best_snr(self):
        mod = load_module(PARTICLES2SNR_GENERATOR, "particles2SNR_dataset_generator_nms_test")
        particles = [
            {"t0": 0.0010, "tau": 0.00010, "snr_db": 1.0, "P0": 0.1, "frequency": 20000},
            {"t0": 0.00102, "tau": 0.00010, "snr_db": 8.0, "P0": 0.1, "frequency": 20000},
            {"t0": 0.0030, "tau": 0.00010, "snr_db": 2.0, "P0": 0.1, "frequency": 20000},
        ]
        kept, dropped = mod.merge_overlapping_particles(
            particles, signal_length=8192, fs=2_000_000.0, iou_threshold=0.5, score_name="snr_db"
        )
        self.assertEqual(len(kept), 2)
        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped[0]["reason"], "overlap_nms_high_iou")
        self.assertEqual(dropped[0]["kept_source_idx"], 1)
        self.assertEqual(dropped[0]["suppressed_source_idx"], 0)

    def test_merge_overlapping_particles_uses_center_distance_for_shifted_duplicates(self):
        mod = load_module(PARTICLES2SNR_GENERATOR, "particles2SNR_dataset_generator_nms_center_test")
        particles = [
            {"t0": 0.00100, "tau": 0.00020, "snr_db": 1.0, "P0": 0.1, "frequency": 20000},
            {"t0": 0.00134, "tau": 0.00020, "snr_db": 7.0, "P0": 0.1, "frequency": 21000},
        ]
        kept, dropped = mod.merge_overlapping_particles(
            particles, signal_length=8192, fs=2_000_000.0,
            iou_threshold=0.4, score_name="snr_db", center_distance_ms=0.35,
        )
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped[0]["reason"], "overlap_nms_ambiguous_low_snr")
        self.assertLess(dropped[0]["iou_with_kept"], 0.5)
        self.assertLessEqual(dropped[0]["center_distance_ms"], 0.35)
        self.assertLessEqual(dropped[0]["frequency_distance_hz"], 8000.0)

    def test_merge_overlapping_particles_keeps_close_frequency_doublet(self):
        mod = load_module(PARTICLES2SNR_GENERATOR, "particles2SNR_dataset_generator_nms_doublet_test")
        particles = [
            {"t0": 0.00100, "tau": 0.00020, "snr_db": 5.0, "P0": 0.1, "frequency": 20000},
            {"t0": 0.00130, "tau": 0.00020, "snr_db": 4.0, "P0": 0.1, "frequency": 33000},
        ]
        kept, dropped = mod.merge_overlapping_particles(
            particles, signal_length=8192, fs=2_000_000.0,
            iou_threshold=0.4, score_name="snr_db",
            duplicate_iou_threshold=0.6, close_center_distance_ms=0.20,
            ambiguous_center_distance_ms=0.30, close_frequency_hz=6000.0,
            ambiguous_frequency_hz=8000.0, snr_margin_db=4.0,
        )
        self.assertEqual(len(kept), 2)
        self.assertEqual(dropped, [])

    def test_export_yolo_json_records_nms_drops(self):
        mod = load_module(PARTICLES2SNR_GENERATOR, "particles2SNR_dataset_generator_nms_export_test")
        tmp_dir = Path("/tmp/particles2SNR_nms_export_test")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        np.save(tmp_dir / "sample.npy", np.zeros(8192, dtype=np.float32))
        results_path = tmp_dir / "dataset_results.json"
        output_path = tmp_dir / "data.json"
        with results_path.open("w") as f:
            json.dump({
                "signals": [{
                    "filename": "sample.npy",
                    "path": str(tmp_dir / "sample.npy"),
                    "class": "2um",
                    "signal_length": 8192,
                    "particles": [
                        {"t0": 0.0010, "tau": 0.00010, "P0": 0.5, "frequency": 20000, "snr_db": 1},
                        {"t0": 0.00102, "tau": 0.00010, "P0": 0.5, "frequency": 22000, "snr_db": 7},
                    ],
                }]
            }, f)
        data = mod.export_yolo_json(
            results_path, output_path, ("2um", "4um", "10um"), 2_000_000.0,
            min_passage_time_ms=0.07, max_passage_time_ms=0.65,
            merge_overlaps=True, merge_iou_threshold=0.5, merge_score="snr_db",
            peak_evidence_filter=False,
        )
        row = data["data"][0]
        self.assertEqual(len(row["annotations"]), 1)
        self.assertEqual(len(row["dropped_annotations"]), 1)
        self.assertEqual(row["dropped_annotations"][0]["reason"], "overlap_nms_high_iou")
        self.assertTrue(data["info"]["overlap_merge"]["enabled"])
        self.assertEqual(data["info"]["overlap_merge"]["method"], "conditional_temporal_nms")
        self.assertEqual(data["info"]["overlap_merge"]["duplicate_iou_threshold"], 0.6)


    def test_filter_annotations_by_yolo_width_bounds(self):
        mod = load_module(PARTICLES2SNR_GENERATOR, "particles2SNR_dataset_generator_width_filter_test")
        anns = [
            {"id": 0, "start": 0.10, "end": 0.105, "passage_time_ms": 0.1, "snr_db": 1, "frequency": 20000},
            {"id": 1, "start": 0.20, "end": 0.29, "passage_time_ms": 0.1, "snr_db": 2, "frequency": 21000},
            {"id": 2, "start": 0.30, "end": 0.50, "passage_time_ms": 0.1, "snr_db": 3, "frequency": 22000},
        ]
        kept, dropped = mod.filter_annotations_by_yolo_width(
            anns, signal_length=2000, fs=2_000_000.0,
            min_width_ms=0.08, max_width_ms=0.10,
        )
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["id"], 0)
        self.assertEqual([d["reason"] for d in dropped], ["yolo_width_below_min", "yolo_width_above_max"])

    def test_boundary_resolution_splits_crossing_annotations_at_midpoint(self):
        mod = load_module(PARTICLES2SNR_GENERATOR, "particles2SNR_dataset_generator_boundary_split_test")
        anns = [
            {"id": 0, "class_id": 1, "start": 0.10, "end": 0.30, "center": 0.20, "half_width": 0.10},
            {"id": 1, "class_id": 1, "start": 0.24, "end": 0.42, "center": 0.33, "half_width": 0.09},
        ]
        kept, dropped, edits = mod.resolve_annotation_boundary_crossings(
            anns, signal_length=2000, fs=2_000_000.0, min_width_ms=0.08,
        )
        self.assertEqual(dropped, [])
        self.assertEqual(len(edits), 1)
        self.assertEqual(len(kept), 2)
        self.assertAlmostEqual(kept[0]["end"], 0.27)
        self.assertAlmostEqual(kept[1]["start"], 0.27)
        self.assertLessEqual(kept[0]["end"], kept[1]["start"])
        self.assertTrue(kept[0]["boundary_adjusted"])
        self.assertTrue(kept[1]["boundary_adjusted"])
        self.assertAlmostEqual(kept[0]["center"], 0.185)
        self.assertAlmostEqual(kept[1]["center"], 0.345)

    def test_boundary_resolution_drops_too_narrow_after_split(self):
        mod = load_module(PARTICLES2SNR_GENERATOR, "particles2SNR_dataset_generator_boundary_drop_test")
        anns = [
            {"id": 0, "class_id": 1, "start": 0.10, "end": 0.22, "center": 0.16, "half_width": 0.06},
            {"id": 1, "class_id": 1, "start": 0.12, "end": 0.40, "center": 0.26, "half_width": 0.14},
        ]
        kept, dropped, edits = mod.resolve_annotation_boundary_crossings(
            anns, signal_length=2000, fs=2_000_000.0, min_width_ms=0.08,
        )
        self.assertEqual(len(edits), 1)
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped[0]["reason"], "boundary_width_below_min")
        self.assertEqual(dropped[0]["stage"], "boundary_resolution")

    def test_peak_evidence_drops_unsupported_low_snr_particle(self):
        mod = load_module(PARTICLES2SNR_GENERATOR, "particles2SNR_dataset_generator_peak_no_support_test")
        signal = np.zeros(4096, dtype=np.float32)
        particles = [{"t0": 0.0010, "tau": 0.00010, "snr_db": -5.0, "P0": 0.1, "frequency": 20000}]
        kept, dropped, peak_groups = mod.refine_particles_with_peak_evidence(
            particles, signal, len(signal), 2_000_000.0,
            min_z=4.0, prominence_z=2.0, min_separation_ms=0.18,
        )
        self.assertEqual(kept, [])
        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped[0]["reason"], "no_peak_evidence")
        self.assertEqual(peak_groups, [])

    def test_peak_evidence_collapses_duplicates_on_same_peak(self):
        mod = load_module(PARTICLES2SNR_GENERATOR, "particles2SNR_dataset_generator_peak_duplicate_test")
        fs = 2_000_000.0
        t = np.arange(4096) / fs
        envelope = np.exp(-0.5 * ((t - 0.0010) / 0.00008) ** 2)
        signal = (envelope * np.sin(2 * np.pi * 25000 * t)).astype(np.float32)
        particles = [
            {"t0": 0.00098, "tau": 0.00010, "snr_db": 2.0, "P0": 0.1, "frequency": 24000},
            {"t0": 0.00102, "tau": 0.00010, "snr_db": 7.0, "P0": 0.1, "frequency": 25000},
        ]
        kept, dropped, peak_groups = mod.refine_particles_with_peak_evidence(
            particles, signal, len(signal), fs,
            min_z=4.0, prominence_z=2.0, min_separation_ms=0.18,
        )
        self.assertEqual(len(peak_groups), 1)
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped[0]["reason"], "same_peak_group_duplicate")
        self.assertAlmostEqual(kept[0]["snr_db"], 7.0)

    def test_peak_evidence_preserves_two_peak_doublet(self):
        mod = load_module(PARTICLES2SNR_GENERATOR, "particles2SNR_dataset_generator_peak_doublet_test")
        fs = 2_000_000.0
        t = np.arange(8192) / fs
        envelope = (
            np.exp(-0.5 * ((t - 0.0010) / 0.00008) ** 2)
            + np.exp(-0.5 * ((t - 0.00220) / 0.00008) ** 2)
        )
        signal = (envelope * np.sin(2 * np.pi * 25000 * t)).astype(np.float32)
        particles = [
            {"t0": 0.0010, "tau": 0.00010, "snr_db": 5.0, "P0": 0.1, "frequency": 24000},
            {"t0": 0.00220, "tau": 0.00010, "snr_db": 4.0, "P0": 0.1, "frequency": 30000},
        ]
        kept, dropped, peak_groups = mod.refine_particles_with_peak_evidence(
            particles, signal, len(signal), fs,
            min_z=4.0, prominence_z=2.0, min_separation_ms=0.18,
        )
        self.assertEqual(len(peak_groups), 2)
        self.assertEqual(len(kept), 2)
        self.assertEqual(dropped, [])
        self.assertEqual({p["peak_group_id"] for p in kept}, {0, 1})

    def test_export_yolo_json_records_yolo_width_drops(self):
        mod = load_module(PARTICLES2SNR_GENERATOR, "particles2SNR_dataset_generator_width_export_test")
        tmp_dir = Path("/tmp/particles2SNR_width_export_test")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        np.save(tmp_dir / "sample.npy", np.zeros(16384, dtype=np.float32))
        results_path = tmp_dir / "dataset_results.json"
        output_path = tmp_dir / "data.json"
        with results_path.open("w") as f:
            json.dump({
                "signals": [{
                    "filename": "sample.npy",
                    "path": str(tmp_dir / "sample.npy"),
                    "class": "2um",
                    "signal_length": 16384,
                    "particles": [
                        {"t0": 0.001, "tau": 0.00010, "P0": 0.5, "frequency": 20000, "snr_db": 1},
                        {"t0": 0.003, "tau": 0.00030, "P0": 0.5, "frequency": 22000, "snr_db": 2},
                    ],
                }]
            }, f)
        data = mod.export_yolo_json(
            results_path, output_path, ("2um", "4um", "10um"), 2_000_000.0,
            min_passage_time_ms=0.07, max_passage_time_ms=0.65,
            merge_overlaps=False, yolo_width_filter=True,
            min_yolo_width_ms=0.08, max_yolo_width_ms=1.0,
            peak_evidence_filter=False,
        )
        row = data["data"][0]
        self.assertEqual(len(row["annotations"]), 1)
        self.assertEqual(len(row["dropped_annotations"]), 1)
        self.assertEqual(row["dropped_annotations"][0]["reason"], "yolo_width_above_max")
        self.assertEqual(row["dropped_annotations"][0]["stage"], "pre_nms")
        self.assertEqual(data["info"]["yolo_width_filter"]["max_ms"], 1.0)

    def test_width_filter_runs_before_nms(self):
        mod = load_module(PARTICLES2SNR_GENERATOR, "particles2SNR_dataset_generator_width_before_nms_test")
        tmp_dir = Path("/tmp/particles2SNR_width_before_nms_test")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        np.save(tmp_dir / "sample.npy", np.zeros(16384, dtype=np.float32))
        results_path = tmp_dir / "dataset_results.json"
        output_path = tmp_dir / "data.json"
        with results_path.open("w") as f:
            json.dump({
                "signals": [{
                    "filename": "sample.npy",
                    "path": str(tmp_dir / "sample.npy"),
                    "class": "2um",
                    "signal_length": 16384,
                    "particles": [
                        {"t0": 0.0030, "tau": 0.00030, "P0": 0.5, "frequency": 20000, "snr_db": 9},
                        {"t0": 0.0031, "tau": 0.00010, "P0": 0.5, "frequency": 22000, "snr_db": 1},
                    ],
                }]
            }, f)
        data = mod.export_yolo_json(
            results_path, output_path, ("2um", "4um", "10um"), 2_000_000.0,
            min_passage_time_ms=0.07, max_passage_time_ms=0.65,
            merge_overlaps=True, merge_iou_threshold=0.5, merge_score="snr_db",
            yolo_width_filter=True, min_yolo_width_ms=0.08, max_yolo_width_ms=1.0,
            peak_evidence_filter=False,
        )
        row = data["data"][0]
        self.assertEqual(len(row["annotations"]), 1)
        self.assertAlmostEqual(row["annotations"][0]["passage_time_ms"], 0.10)
        self.assertEqual(len(row["dropped_annotations"]), 1)
        self.assertEqual(row["dropped_annotations"][0]["reason"], "yolo_width_above_max")
        self.assertEqual(row["dropped_annotations"][0]["stage"], "pre_nms")

    def test_export_yolo_json_default_max_width_is_less_aggressive(self):
        mod = load_module(PARTICLES2SNR_GENERATOR, "particles2SNR_dataset_generator_default_width_test")
        tmp_dir = Path("/tmp/particles2SNR_default_width_test")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        np.save(tmp_dir / "sample.npy", np.zeros(16384, dtype=np.float32))
        results_path = tmp_dir / "dataset_results.json"
        output_path = tmp_dir / "data.json"
        with results_path.open("w") as f:
            json.dump({
                "signals": [{
                    "filename": "sample.npy",
                    "path": str(tmp_dir / "sample.npy"),
                    "class": "2um",
                    "signal_length": 16384,
                    "particles": [
                        {"t0": 0.003, "tau": 0.00025, "P0": 0.5, "frequency": 22000, "snr_db": 2},
                    ],
                }]
            }, f)
        data = mod.export_yolo_json(
            results_path, output_path, ("2um", "4um", "10um"), 2_000_000.0,
            min_passage_time_ms=0.07, max_passage_time_ms=0.65,
            merge_overlaps=False, yolo_width_filter=True,
            peak_evidence_filter=False,
        )
        row = data["data"][0]
        self.assertEqual(len(row["annotations"]), 1)
        self.assertEqual(row["dropped_annotations"], [])
        self.assertEqual(data["info"]["yolo_width_filter"]["max_ms"], 1.5)

    def test_export_yolo_json_from_particles2SNR_results(self):
        mod = load_module(PARTICLES2SNR_GENERATOR, "particles2SNR_dataset_generator_test2")
        tmp_dir = "/tmp/particles2SNR_dataset_generator_test"
        os.makedirs(tmp_dir, exist_ok=True)
        np.save(os.path.join(tmp_dir, "sample.npy"), np.zeros(1000, dtype=np.float32))
        results_path = os.path.join(tmp_dir, "dataset_results.json")
        output_path = os.path.join(tmp_dir, "data.json")
        with open(results_path, "w") as f:
            json.dump({
                "dataset_info": {"total_signals": 1},
                "signals": [{
                    "filename": "sample.npy",
                    "path": os.path.join(tmp_dir, "sample.npy"),
                    "class": "2um",
                    "signal_length": 1000,
                    "particles": [{
                        "t0": 0.0001,
                        "tau": 0.00001,
                        "P0": 0.5,
                        "frequency": 25000.0,
                        "snr_db": 12.5,
                    }],
                }],
            }, f)

        data = mod.export_yolo_json(
            Path(results_path),
            Path(output_path),
            ("2um", "4um", "10um"),
            fs=2_000_000.0,
            yolo_width_filter=False,
            peak_evidence_filter=False,
        )
        self.assertEqual(data["classes"][0]["name"], "2um")
        self.assertEqual(data["data"][0]["class_id"], 0)
        ann = data["data"][0]["annotations"][0]
        self.assertAlmostEqual(ann["mean"], 0.2)
        self.assertAlmostEqual(ann["std"], 0.02)
        self.assertGreaterEqual(ann["start"], 0.0)
        self.assertLessEqual(ann["end"], 1.0)
        self.assertTrue(os.path.isfile(output_path))

        layout = mod.export_detseg_yolo_layout(data, Path(tmp_dir) / "detseg" / "test")
        self.assertEqual(layout["signals"], 1)
        self.assertEqual(layout["labels"], 1)
        self.assertTrue(os.path.isfile(os.path.join(tmp_dir, "detseg", "test", "signals", "sample.npy")))
        label_path = os.path.join(tmp_dir, "detseg", "test", "labels", "sample.txt")
        with open(label_path) as f:
            label_line = f.read().strip().split()
        self.assertEqual(label_line[0], "0")
        self.assertAlmostEqual(float(label_line[1]), 0.2)

    def test_prepare_split_tree_from_class_sources(self):
        mod = load_module(PARTICLES2SNR_GENERATOR, "particles2SNR_dataset_generator_test3")
        tmp_dir = Path("/tmp/particles2SNR_class_sources_test")
        source = tmp_dir / "src" / "2um"
        staging = tmp_dir / "staging"
        source.mkdir(parents=True, exist_ok=True)
        for idx in range(5):
            np.save(source / f"sig_{idx}.npy", np.asarray([idx], dtype=np.float32))

        rows = mod.prepare_split_tree_from_class_sources(
            {"2um": source},
            staging,
            ("train", "test"),
            test_fraction=0.4,
            seed=0,
        )
        self.assertEqual(len(rows), 5)
        self.assertEqual(len(list((staging / "train" / "2um").glob("*.npy"))), 3)
        self.assertEqual(len(list((staging / "test" / "2um").glob("*.npy"))), 2)


class AccuracyVsSnrTests(unittest.TestCase):
    def test_threshold_detects_plateau(self):
        mod = load_module(P0_SCRIPT, "conv1dgap_accuracy_vs_snr_test")
        bins = [
            {"snr_center": 0.0, "accuracy": 0.40},
            {"snr_center": 5.0, "accuracy": 0.70},
            {"snr_center": 10.0, "accuracy": 0.90},
            {"snr_center": 15.0, "accuracy": 0.92},
            {"snr_center": 20.0, "accuracy": 0.93},
        ]
        threshold = mod.estimate_threshold(bins, derivative_frac=0.25)
        self.assertEqual(threshold["method"], "post_peak_derivative_slowdown")
        self.assertIsNotNone(threshold["unknown_snr_threshold_db"])

    def test_prediction_normalization_joins_snr_by_filename(self):
        mod = load_module(P0_SCRIPT, "conv1dgap_accuracy_vs_snr_test2")
        pred = [{"filename": "/tmp/a.npy", "y_true": "2um", "y_pred": "4um"}]
        rows = mod.normalize_predictions(pred, {"a.npy": 3.5})
        self.assertEqual(rows[0]["snr_db"], 3.5)
        self.assertFalse(rows[0]["correct"])


class P1SaturationManagementTests(unittest.TestCase):
    def test_apply_saturation_management_drops_overlapping_label(self):
        mod = load_module(P1_SCRIPT, "p1_generate_long_sequence_dataset_test")
        old_detect = mod.detect_unsafe_intervals
        try:
            mod.detect_unsafe_intervals = lambda *args, **kwargs: (
                {"is_saturated": True},
                [(10, 20)],
            )
            args = types.SimpleNamespace(
                saturation_management=True,
                saturation_policy="keep",
                fs=2_000_000,
                saturation_fmin=7000,
                saturation_fmax=80000,
                saturation_min_flat=50,
                saturation_zero_threshold=1e-4,
                saturation_guard_before=0,
                saturation_guard_after=0,
            )
            pool = [(np.zeros(64), [(12, 18, 0), (30, 40, 1)])]
            cleaned, names, audit, stats = mod.apply_saturation_management(
                pool,
                ["2um_train_0001"],
                [],
                args,
                np.random.default_rng(0),
                "train",
            )
            self.assertEqual(names, ["2um_train_0001"])
            self.assertEqual(cleaned[0][1], [(30, 40, 1)])
            self.assertEqual(stats["dropped_events"], 1)
            self.assertEqual(audit[0]["action"], "reported_only")
        finally:
            mod.detect_unsafe_intervals = old_detect


class ArtifactSchemaTests(unittest.TestCase):
    def assert_csv_columns(self, path, required):
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            self.assertIsNotNone(reader.fieldnames)
            missing = set(required) - set(reader.fieldnames)
            self.assertFalse(missing, f"{path} missing columns: {sorted(missing)}")

    @unittest.skipUnless(
        os.path.isfile("/tmp/particles2SNR_smoke_3x_report/snr_noise_report.json"),
        "bounded particles2SNR smoke report not present",
    )
    def test_particles2SNR_report_schema(self):
        with open("/tmp/particles2SNR_smoke_3x_report/snr_noise_report.json") as f:
            report = json.load(f)
        self.assertIn("metrics", report)
        self.assertIn("snr_db", report["metrics"])
        self.assertIn("comparison", report["metrics"]["snr_db"])
        self.assert_csv_columns(
            "/tmp/particles2SNR_smoke_3x_report/pairwise_comparisons.csv",
            ["metric", "left", "right", "test", "pvalue_bonferroni"],
        )

    @unittest.skipUnless(
        os.path.isfile("/tmp/saturation_smoke_out/saturation_cleaning_manifest.csv"),
        "saturation cleaning smoke manifest not present",
    )
    def test_saturation_cleaning_manifest_schema(self):
        self.assert_csv_columns(
            "/tmp/saturation_smoke_out/saturation_cleaning_manifest.csv",
            [
                "source_path", "output_path", "policy", "start_sample",
                "end_sample", "duration_samples", "action", "guard_before",
                "guard_after",
            ],
        )

    @unittest.skipUnless(
        os.path.isfile("/tmp/p1_sat_overlap_out/saturation_audit.csv"),
        "P1 saturation overlap smoke not present",
    )
    def test_p1_saturation_audit_schema_and_post_audit(self):
        self.assert_csv_columns(
            "/tmp/p1_sat_overlap_out/saturation_audit.csv",
            [
                "split", "source", "policy", "start_sample", "end_sample",
                "duration_samples", "action", "dropped_events",
            ],
        )
        audit_mod = load_module(P1_AUDIT_SCRIPT, "p1_saturation_artifact_audit_test")
        report = audit_mod.audit_dataset("/tmp/p1_sat_overlap_out")
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["counter_mismatches"], [])

    @unittest.skipUnless(
        os.path.isfile("/tmp/p0_snr_eval_out/conv1dgap_accuracy_by_snr.json"),
        "P0 SNR smoke not present",
    )
    def test_p0_snr_accuracy_schema(self):
        with open("/tmp/p0_snr_eval_out/conv1dgap_accuracy_by_snr.json") as f:
            report = json.load(f)
        self.assertEqual(report["model_family"], "Conv1DGAP")
        self.assertIn("threshold", report)
        self.assertIn("bins", report)
        self.assert_csv_columns(
            "/tmp/p0_snr_eval_out/conv1dgap_accuracy_by_snr.csv",
            ["bin_idx", "snr_left", "snr_right", "snr_center", "n", "accuracy", "macro_f1"],
        )


if __name__ == "__main__":
    unittest.main()
