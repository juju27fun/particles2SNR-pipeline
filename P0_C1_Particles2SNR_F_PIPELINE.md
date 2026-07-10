# P0 C1 particles2SNR Clean Filt Pipeline

This document describes the current C1 particles2SNR pipeline used to generate the
accepted P0 YOLO clean dataset and its P1 4-class derivative.

## Goal

The goal is to build detection labels from the particles2SNR particle detector while
cleaning obvious signal-quality issues and constraining the final YOLO labels
to a geometry that is usable for detection training.

The accepted P0 YOLO clean dataset is:

```text
P0/data/processed/dataset_Particles2SNR_F_c1_yolo_trainval
```

The main particles2SNR artifact directory is:

```text
particles2SNR_pipeline/results/runs/p0_c1_Particles2SNR_F
```

The derived P1 4-class dataset with the SNR -10 dB unclear rule is:

```text
P1/data/yolo/canonical/particles2snr_f_c1_4class_lim10_trainval
```

## Main Scripts

```text
particles2SNR_pipeline/generate_particles2SNR_dataset.py
```

Builds the cleaned P0 signal tree, runs particles2SNR detection, post-processes the
particles2SNR particles, writes `data.json`, and exports a YOLO/detseg layout.

```text
particles2SNR_pipeline/create_particles2SNR_c1_yolo_4class_lim10.py
```

Converts the 3-class P0 YOLO dataset into a P1 4-class dataset. Labels with
`snr_db < -10.0` become `unclear`.

```text
particles2SNR_pipeline/generate_visual_signal_checks.py
```

Generates PNG visual checks from the final YOLO labels and the particles2SNR metadata.

## Source Data

The current run uses explicit C1 class source folders:

```text
2um  -> P0/C1_HF_5_10_2um_doublet2
4um  -> P0/C1_HF_5_10_4um_doublet
10um -> P0/C1_HF_5_10_10um_doublet
```

The split policy is:

```text
splits: train,test
test_fraction: 0.2
val_fraction: 0.2
split_seed: 42
```

The validation split is not produced by particles2SNR directly. It is created later
from the generated train rows when exporting the YOLO/detseg layout.

## Pipeline Command

The current P0 generation command is:

```bash
P0/venv/bin/python particles2SNR_pipeline/generate_particles2SNR_dataset.py \
  --class-source-dirs 2um=P0/C1_HF_5_10_2um_doublet2,4um=P0/C1_HF_5_10_4um_doublet,10um=P0/C1_HF_5_10_10um_doublet \
  --output-root P0/data/processed/dataset_Particles2SNR_F_c1 \
  --particles2SNR-output particles2SNR_pipeline/results/runs/p0_c1_Particles2SNR_F \
  --detseg-output P0/data/processed/dataset_Particles2SNR_F_c1_yolo_trainval \
  --device cpu
```

The final P1 conversion command is:

```bash
P0/venv/bin/python particles2SNR_pipeline/create_particles2SNR_c1_yolo_4class_lim10.py
```

The visual checks are regenerated with:

```bash
P0/venv/bin/python particles2SNR_pipeline/generate_visual_signal_checks.py
```

## Signal Preprocessing

The source `.npy` files are not modified. The pipeline creates a derived signal
tree:

```text
P0/data/processed/dataset_Particles2SNR_F_c1/{train,test}/{2um,4um,10um}
```

Preprocessing steps:

1. Build train/test split from the C1 class folders.
2. Remove long zero-valued regions while allowing at most two zero samples at a
   junction.
3. Detect saturated or flat unsafe intervals.
4. Replace unsafe intervals with chunks from:

```text
P0/data/processed/Noise
```

5. Re-run the saturation audit after cleaning. The generation fails if
   saturated files remain.
6. Apply a final bandpass filter to the cleaned signal:

```text
7 kHz - 80 kHz, Butterworth order 4
```

Current saturation and filtering parameters:

```text
saturation_policy: replace
saturation_guard_before: 300 samples
saturation_guard_after: 300 samples
bandpass_fmin: 7000 Hz
bandpass_fmax: 80000 Hz
bandpass_order: 4
```

## particles2SNR Files And Artifacts

For each split, particles2SNR writes artifacts under:

```text
particles2SNR_pipeline/results/runs/p0_c1_Particles2SNR_F/{train,test}
```

Important files:

```text
dataset_results.json
```

Raw particles2SNR output for the split. It contains per-signal particles with fields
such as `t0`, `tau`, `frequency`, `P0`, `energy`, and `snr_db`. This is the
direct input used to build final YOLO annotations.

```text
data.json
```

Post-processed annotation file. This is the central file after particles2SNR. It
contains final normalized annotations, dropped-annotation metadata, peak-group
metadata, and boundary-adjustment metadata.

```text
snr_particles.csv
noise_by_file.csv
noise_by_class.csv
```

particles2SNR SNR and noise tables. These are diagnostic artifacts; the YOLO labels
come from `dataset_results.json` after post-processing, not directly from these
CSV files.

```text
zero_cleaning_manifest.csv
saturation_cleaning_manifest.csv
post_clean_saturation_intervals.csv
post_clean_saturation_summary.json
```

Cleaning and saturation audit logs.

```text
run_summary.json
source_split_manifest.csv
```

Run-level provenance and source-to-split mapping.

## particles2SNR-To-YOLO Post-Processing

The raw particles2SNR particles are not exported directly. They go through the
following post-processing steps in `generate_particles2SNR_dataset.py`.

### Passage-Time Filter

Particles are kept only if their particles2SNR passage time is in:

```text
0.07 ms <= tau <= 0.65 ms
```

This removes extremely short particles2SNR artifacts and very long windows that are
not plausible clean particle passages.

### First YOLO Width Filter

Before peak-evidence filtering, particles are converted to an expected YOLO
width and filtered with:

```text
min_yolo_width_ms: 0.08
max_yolo_width_ms: 1.5
```

### Peak-Evidence Filter

The cleaned, bandpassed signal is loaded again from `signal.path`. The pipeline
computes an absolute moving-average envelope and uses robust z-scores to group
actual signal peaks. The accepted `_F` configuration now uses `dual_clean` peak
evidence: a particle must be supported by a peak in the bandpassed signal and
also by a peak in the cleaned, non-bandpassed signal saved under each split's
`peak_evidence_clean_signals` artifact directory. This prevents accepting peaks
that are only created by the 7-80 kHz bandpass.

Current parameters:

```text
peak_evidence_filter: true
peak_evidence_signal_mode: dual_clean
peak_envelope_window_ms: 0.08
peak_min_z: 4.0
peak_prominence_z: 2.0
peak_min_separation_ms: 0.18
peak_group_valley_ratio: 0.55
peak_cluster_gap_ms: 0.25
peak_keep_high_snr_db: 4.0
```

Effects:

- particles2SNR particles without local peak support are dropped.
- particles2SNR particles whose peak support exists only after bandpass are dropped.
- Duplicate particles2SNR particles on the same envelope peak are collapsed.
- Close doublets can be preserved when they map to distinct peak groups.
- High-SNR particles can be kept even if peak support is imperfect.

Each final annotation can carry fields such as:

```text
peak_support
peak_group_id
peak_z
peak_center_ms
local_peak_z
```

Each row in `data.json` also stores `peak_groups`.

### Conditional Temporal NMS

Remaining particles are passed through conditional temporal NMS:

```text
merge_overlaps: true
method: conditional_temporal_nms
merge_iou_threshold: 0.4
merge_duplicate_iou_threshold: 0.6
merge_close_center_distance_ms: 0.20
merge_ambiguous_center_distance_ms: 0.30
merge_close_frequency_hz: 6000
merge_ambiguous_frequency_hz: 8000
merge_snr_margin_db: 4.0
merge_score: snr_db
```

This stage is intended to remove duplicates while preserving likely double
events. It is not the final geometry cleanup.

### Second YOLO Width Filter

After conversion to final annotation intervals, the YOLO width filter is
applied again:

```text
0.08 ms <= label width <= 1.5 ms
```

Drops from this stage are recorded with `stage: post_annotation`.

### Boundary Resolution

The final geometry pass resolves crossing intervals between adjacent labels.
If the end boundary of one label crosses the start boundary of the next label,
both labels are cut at the midpoint:

```text
new_boundary = (left.end + right.start) / 2
left.end = new_boundary
right.start = new_boundary
```

Current configuration:

```text
resolve_boundary_crossings: true
boundary_resolution.method: adjacent_overlap_midpoint
boundary_resolution.min_width_ms: 0.08
```

This makes final YOLO labels non-overlapping while keeping both particles when
the previous stages judged them distinct. If a label becomes narrower than
`0.08 ms` after the cut, it is dropped with:

```text
stage: boundary_resolution
reason: boundary_width_below_min
```

Boundary edits are stored in `data.json` as:

```text
boundary_adjustments
boundary_adjusted
```

## Final P0 YOLO Dataset

The accepted 3-class P0 YOLO/detseg layout is:

```text
P0/data/processed/dataset_Particles2SNR_F_c1_yolo_trainval
```

Structure:

```text
train/signals/*.npy
train/labels/*.txt
val/signals/*.npy
val/labels/*.txt
test/signals/*.npy
test/labels/*.txt
dataset.yaml
```

Labels are YOLO-style single-segment intervals:

```text
class_id center width
```

where `center` and `width` are normalized by signal length.

Class names:

```text
0: 2um
1: 4um
2: 10um
```

## P1 4-Class Dataset

The P1 4-class dataset is derived from the final P0 YOLO clean dataset:

```text
P1/data/yolo/canonical/particles2snr_f_c1_4class_lim10_trainval
```

It uses the same signal files and intervals as P0, but rewrites labels using:

```text
if snr_db < -10.0:
    class = unclear
else:
    class = original particle class
```

Class names:

```text
0: 2um
1: 4um
2: 10um
3: unclear
```

The converter reads these particles2SNR post-processed files:

```text
particles2SNR_pipeline/results/runs/p0_c1_Particles2SNR_F/train/data.json
particles2SNR_pipeline/results/runs/p0_c1_Particles2SNR_F/test/data.json
```

The `val` split is resolved by matching the P0 YOLO split files against the
particles2SNR rows by filename. The P1 `dataset.yaml` stores the inherited particles2SNR
annotation parameters, including passage-time filtering, peak evidence,
temporal NMS, YOLO width filtering, and boundary resolution.

Current P1 class counts:

```text
train: files=1848, events=3032
  2um=839, 4um=1080, 10um=773, unclear=340

val: files=462, events=763
  2um=229, 4um=266, 10um=196, unclear=72

test: files=578, events=895
  2um=260, 4um=299, 10um=231, unclear=105
```

## Visual Checks

Visual signal checks are written to:

```text
particles2SNR_pipeline/results/figures/visual_signal_checks
```

The script uses:

```text
P0/data/processed/dataset_Particles2SNR_F_c1_yolo_trainval
particles2SNR_pipeline/results/runs/p0_c1_Particles2SNR_F/train/data.json
particles2SNR_pipeline/results/runs/p0_c1_Particles2SNR_F/test/data.json
```

The PNGs overlay:

- final YOLO label intervals;
- overlap-density coloring;
- label outlines and IDs;
- particles2SNR peak-group markers when available.

The manifest is:

```text
particles2SNR_pipeline/results/figures/visual_signal_checks/visual_signal_checks_manifest.csv
```

## Validation Checks

The last validated generation had:

```text
P0 particles2SNR data.json overlaps: 0
P1 YOLO label overlaps: 0
```

Boundary-resolution activity:

```text
train boundary edits: 431
test boundary edits: 87
train boundary drops: 1
test boundary drops: 1
```

P1 checks:

```bash
env PYTHONPATH=P1 P0/venv/bin/python -m detseg.audit_dataset \
  --data-dir P1/data/yolo/canonical/particles2snr_f_c1_4class_lim10_trainval \
  --strict \
  --json-out P1/detseg_output_Particles2SNR_F_c1_4class_lim10_dataset_audit.json

env PYTHONPATH=P1 P0/venv/bin/python -m detseg.preflight_data_gate \
  --data-dir P1/data/yolo/canonical/particles2snr_f_c1_4class_lim10_trainval \
  --mode strict \
  --json-out P1/detseg_output_Particles2SNR_F_c1_4class_lim10_preflight.json
```

Both checks passed on the current generated dataset.

## Notes

- `dataset_results.json` is the raw particles2SNR detector output.
- `data.json` is the post-processed particles2SNR annotation source of truth.
- P0 YOLO labels are exported from post-processed `data.json`, not from raw
  particles2SNR particles.
- P1 4-class labels are derived from the same post-processed particles2SNR intervals,
  with only the class changed to `unclear` when `snr_db < -10.0`.
- Source C1 `.npy` files remain read-only; all modifications are in derived
  datasets and artifact folders.
