# particles2SNR Pipeline

This folder contains the FFT/particles2SNR particle detector plus downstream SNR,
noise, and saturation utilities. Source `.npy` datasets are read-only by
default; derived artifacts should be written to an explicit output directory.

For the current accepted C1 clean YOLO generation pipeline, including the
particles2SNR files used and the post-processing stages, see
[`P0_C1_Particles2SNR_F_PIPELINE.md`](P0_C1_Particles2SNR_F_PIPELINE.md).

## Dataset Analysis

Run the particles2SNR detector over the class folders:

```bash
python3 run_dataset.py --dataset-dir ../particle_detector/test --output-dir output --device cpu
```

Outputs:

- `dataset_results.json`: nested per-signal particle detections.
- `snr_particles.csv`: one row per detected particle with `snr_db`.
- `noise_by_file.csv`: raw/filtered noise summaries per source file.
- `noise_by_class.csv`: aggregate noise and SNR summary per class.

The current SNR method is `peak_bin_energy_over_lowest_window_energy`: each
particle peak-bin energy is divided by a per-file noise floor estimated from
the lowest-energy FFT windows. It is useful for ranking events, but class-wise
noise statistics should be reported next to it when class noise differs.

For short signals whose length is below the class default FFT window, the
pipeline adapts the FFT window to the largest power of two that fits the
signal. This preserves table exports for P0 benchmark files (`2500` samples)
while leaving the `16384`-sample particle-detector domain unchanged.

## SNR And Noise Report

After `run_dataset.py`, build reproducible figures and statistical tests:

```bash
python3 snr_noise_report.py --input-dir output --output-dir output/snr_noise_report
```

Outputs:

- `snr_noise_report.json`: machine-readable summaries, normality checks,
  ANOVA/Kruskal overall tests, and pairwise Mann-Whitney comparisons.
- `snr_noise_report.md`: compact human-readable report.
- `pairwise_comparisons.csv`: pairwise class comparison table.
- `snr_by_class_distribution.pdf`, `raw_std_by_class.pdf`,
  `filtered_std_by_class.pdf`, `inband_energy_ratio_by_class.pdf`.

Core schemas:

- `snr_particles.csv`: `filename`, `path`, `class`, `signal_idx`,
  `signal_length`, `particle_idx`, `frequency`, `P0`, `t0`, `tau`, `phi`,
  `energy`, `snr_db`, `noise_floor`, `noise_floor_N`, `snr_method`,
  `source_window_idx`, `source_window_center`, `source_window_energy`.
- `noise_by_file.csv`: `filename`, `path`, `class`, `signal_idx`,
  `signal_length`, `num_particles`, `num_windows`, `num_valid_windows`,
  `noise_floor`, `noise_floor_N`, `snr_method`, raw/filtered mean/std/RMS,
  raw/filtered kurtosis, `inband_energy_ratio`, `spectral_flatness`.
- `pairwise_comparisons.csv`: `metric`, `left`, `right`, `test`,
  `statistic`, raw and Bonferroni-corrected p-values, medians, and median
  delta.

## Three-Way SNR Comparison

Compare particle SNR distributions for the original particles2SNR dataset, the C1
particles2SNR dataset, and the accepted C1 clean filtered dataset:

```bash
P0/.venv/bin/python particles2SNR_pipeline/compare_snr_by_class.py \
  --dataset "P0 original particles2SNR=particles2SNR_pipeline/output/p0_dataset_particles2SNR/test/snr_particles.csv" \
  --dataset "P0 C1 particles2SNR=particles2SNR_pipeline/output/p0_c1_particles2SNR/test/snr_particles.csv" \
  --dataset "P0 C1 clean + 7-80 kHz=particles2SNR_pipeline/output/p0_c1_Particles2SNR_F/test/snr_particles.csv" \
  --output-dir particles2SNR_pipeline/output/snr_comparison \
  --output-name snr_by_class_3way \
  --classes 2um,4um,10um \
  --threshold-db -10
```

Outputs:

- `snr_by_class_3way.pdf`: three-way class-wise SNR boxplot.
- `snr_by_class_3way.png`: raster copy for quick review.
- `snr_by_class_3way_summary.csv`: counts, mean, median, p10, and p90 by dataset and class.

## Spectral Noise vs Doppler Report

Compare old/new particles2SNR outputs by separating candidate noise windows, the
standalone Noise folder, and Doppler peak frequencies:

```bash
python3 particles2SNR_pipeline/spectral_noise_particle_report.py \
  --old-output particles2SNR_pipeline/output/p0_c1_particles2SNR/test \
  --new-output particles2SNR_pipeline/output/p0_dataset_particles2SNR/test \
  --noise-dir P0/data/Noise \
  --output-dir particles2SNR_pipeline/output/spectral_noise_particle_comparison \
  --classes 2um,4um,10um \
  --doppler-band-khz 10,40
```

Candidate noise is defined as FFT windows from particle signals that do not
overlap detected particles using `t0 ± 3*tau`. If no such window remains, the
candidate-noise metrics are exported as `NaN` so the coverage issue is explicit.

Outputs:

- `spectral_comparison.pdf`: compact old/new visual report.
- `spectral_band_summary.csv`: mean band-energy percentage by pipeline, class,
  source, and frequency band.
- `doppler_peak_summary.csv`: Doppler pick distribution by frequency band.
- `overlap_summary.csv`: `10-40 kHz` energy, Doppler pick percentage, dominant
  bands, and candidate-vs-Doppler overlap metrics.

## YOLO Detection Spectral Report

For detection datasets with explicit YOLO labels, compare particle-labelled
intervals against dataset noise windows outside labels:

```bash
python3 particles2SNR_pipeline/yolo_detection_spectral_report.py \
  --old-dataset P1/yolo_dataset_v3 \
  --new-dataset P0/data/dataset_particles2SNR_c1_yolo \
  --noise-dir P0/data/Noise \
  --output-dir particles2SNR_pipeline/output/yolo_detection_spectral_comparison \
  --splits test \
  --classes 2um,4um,10um \
  --guard-samples 0 \
  --doppler-band-khz 10,40
```

This report is better suited than particles2SNR-derived masks when the goal is to use
known detection labels as the particle/noise boundary. `--guard-samples 0` is
the conservative default for the C1 particles2SNR comparison because its labels cover
a large fraction of each signal; increasing the guard can remove too many noise
windows.

Outputs:

- `yolo_detection_spectral_comparison.pdf`: compact old-vs-C1 visual report.
- `yolo_spectral_band_summary.csv`: band-energy percentages for labelled
  particles, dataset noise, and standalone Noise.
- `yolo_label_coverage_summary.csv`: how much signal is covered by labels.
- `yolo_overlap_summary.csv`: `10-40 kHz` energy and particle-vs-noise overlap.

The same script also supports a three-way comparison with repeated `--dataset`
arguments:

```bash
P0/.venv/bin/python particles2SNR_pipeline/yolo_detection_spectral_report.py \
  --dataset "YOLO v3 old pipeline=P1/yolo_dataset_v3" \
  --dataset "P0 C1 particles2SNR=P0/data/dataset_particles2SNR_c1_yolo" \
  --dataset "P0 C1 clean + 7-80 kHz=P0/data/dataset_Particles2SNR_F_c1_yolo_trainval" \
  --noise-dir P0/data/Noise \
  --output-dir particles2SNR_pipeline/output/yolo_detection_spectral_comparison_3way \
  --output-name yolo_detection_spectral_comparison_3way \
  --splits test \
  --classes 2um,4um,10um \
  --guard-samples 0 \
  --doppler-band-khz 10,40
```

Outputs use the selected prefix, for example
`yolo_detection_spectral_comparison_3way.pdf`,
`yolo_detection_spectral_comparison_3way_band_summary.csv`,
`yolo_detection_spectral_comparison_3way_label_coverage_summary.csv`, and
`yolo_detection_spectral_comparison_3way_overlap_summary.csv`.

To compare all datasets after the same analysis filter and avoid the relative
percentage bias introduced by pre-filtered signals, generate the v2 report:

```bash
P0/.venv/bin/python particles2SNR_pipeline/yolo_detection_spectral_report.py \
  --dataset "YOLO v3 old pipeline=P1/yolo_dataset_v3" \
  --dataset "P0 C1 particles2SNR=P0/data/dataset_particles2SNR_c1_yolo" \
  --dataset "P0 C1 clean + 7-80 kHz=P0/data/dataset_Particles2SNR_F_c1_yolo_trainval" \
  --noise-dir P0/data/Noise \
  --output-dir particles2SNR_pipeline/output/yolo_detection_spectral_comparison_3way_v2 \
  --output-name yolo_detection_spectral_comparison_3way_v2 \
  --splits test \
  --classes 2um,4um,10um \
  --guard-samples 0 \
  --doppler-band-khz 10,40 \
  --analysis-bandpass-khz 7,80
```

The v2 report applies the same in-memory Butterworth 7-80 kHz bandpass to all
datasets and the standalone Noise reference before spectral analysis. It also
writes `*_absolute_energy_summary.csv`, with normalized band energy in dB.

## Visual Dataset Comparison

Generate signal-level visual comparisons between the C1 particles2SNR YOLO dataset and
the accepted C1 clean filtered YOLO dataset. The P0 datasets are compared with
the same source filenames; YOLO v3 is included separately as an unaligned old
pipeline reference.

```bash
P0/.venv/bin/python particles2SNR_pipeline/compare_visual_signal_datasets.py \
  --splits test \
  --classes 2um,4um,10um \
  --max-samples 4
```

Outputs are written under
`particles2SNR_pipeline/output/p0_c1_Particles2SNR_F/visual_signal_checks/dataset_comparison/`:

- `test_<class>_comparison.png`: aligned P0 C1 vs P0 C1 clean signals.
- `yolo_v3_reference_test_<class>.png`: unaligned YOLO v3 reference examples.
- `overview_comparison.png`: compact aligned overview.
- `dataset_comparison_manifest.csv`: selected files and label/overlap summaries.

For the focused 4um presentation figure with two high-impact examples and one
YOLO v3 reference row per example:

```bash
P0/.venv/bin/python particles2SNR_pipeline/compare_visual_signal_datasets.py \
  --focused-4um-impact
```

This writes `focused_4um_impact_3row.png` and
`focused_4um_impact_manifest.csv` in the same output directory.

## P0 particles2SNR Dataset Generation

Build a non-destructive cleaned dataset from `P0/data/dataset`, remove long
zero-valued regions while keeping at most two zero samples at each junction,
run particles2SNR ground-truth generation, and export YOLO-compatible annotations:

```bash
P0/.venv/bin/python particles2SNR_pipeline/generate_particles2SNR_dataset.py \
  --input-root P0/data/dataset \
  --output-root P0/data/dataset_particles2SNR_clean \
  --particles2SNR-output particles2SNR_pipeline/output/p0_dataset_particles2SNR \
  --splits train,test \
  --classes 2um,4um,10um \
  --zero-epsilon 0 \
  --max-zero-run-after-clean 2 \
  --device cpu
```

Per split outputs are written under `particles2SNR_pipeline/output/p0_dataset_particles2SNR/{train,test}`:

- `zero_cleaning_manifest.csv`: source/output file mapping and removed zero intervals.
- `saturation_intervals.csv`, `saturation_summary.json`: derivative-based saturation scan.
- `dataset_results.json`, `snr_particles.csv`, `noise_by_file.csv`,
  `noise_by_class.csv`: particles2SNR detections and SNR/noise tables.
- `data.json`: annotation file compatible with `particle_detector` test loaders.

The cleaned signal tree is written to `P0/data/dataset_particles2SNR_clean/{train,test}`.
The P1/detseg YOLO layout is written to
`P0/data/dataset_particles2SNR_yolo/{train,val,test}/{signals,labels}` with a
minimal `dataset.yaml`; `val` is created empty unless a validation split is
explicitly generated.
Source files under `P0/data/dataset` are not modified.

Then generate the SNR/noise report on the test split:

```bash
P0/.venv/bin/python particles2SNR_pipeline/snr_noise_report.py \
  --input-dir particles2SNR_pipeline/output/p0_dataset_particles2SNR/test \
  --output-dir particles2SNR_pipeline/output/p0_dataset_particles2SNR/test/snr_noise_report
```

For Conv1DGAP accuracy-vs-SNR on the generated test split, first run the
preflight check on the selected checkpoint, then run the SNR curve script:

```bash
P0/.venv/bin/python P0/scripts/preflight_conv1dgap_snr.py \
  --checkpoint P0/results/benchmark2/checkpoints/<run>/best_model.pth \
  --data-dir P0/data/dataset_particles2SNR_clean/test \
  --snr-csv particles2SNR_pipeline/output/p0_dataset_particles2SNR/test/snr_particles.csv \
  --output-json particles2SNR_pipeline/output/p0_dataset_particles2SNR/test/conv1dgap_snr_preflight.json

P0/.venv/bin/python P0/scripts/conv1dgap_accuracy_vs_snr.py \
  --checkpoint P0/results/benchmark2/checkpoints/<run>/best_model.pth \
  --data-dir P0/data/dataset_particles2SNR_clean/test \
  --snr-csv particles2SNR_pipeline/output/p0_dataset_particles2SNR/test/snr_particles.csv \
  --output-dir particles2SNR_pipeline/output/p0_dataset_particles2SNR/test/conv1dgap_snr
```


## Event-Level Accuracy-vs-SNR Comparison

For paper-style single-event classifier comparisons, first generate event-level
predictions for each pipeline from its `data.json`. Example for the old C1
particles2SNR output:

```bash
P0/.venv/bin/python particles2SNR_pipeline/event_accuracy_vs_snr.py \
  --data-json particles2SNR_pipeline/output/p0_c1_particles2SNR/test/data.json \
  --checkpoint P0/results/particles2SNR_c1_conv1dgap_retrained/checkpoints/Conv1DGAP-L-L4096-decim-dataset_particles2SNR_c1-tier1-seed42/best_model.pth \
  --dataset-label "old particles2SNR event" \
  --output-dir particles2SNR_pipeline/output/p0_c1_particles2SNR/test/event_conv1dgap_snr \
  --model-name Conv1DGAP-L \
  --input-length 4096 \
  --crop-native-length 16384 \
  --preprocess adaptive-bandpass \
  --device cpu
```

Then compare event predictions with common SNR bins and class-balanced sampling
inside each `(pipeline, SNR bin)` group:

```bash
P0/.venv/bin/python particles2SNR_pipeline/compare_event_accuracy_by_snr.py \
  --run "old particles2SNR event=particles2SNR_pipeline/output/p0_c1_particles2SNR/test/event_conv1dgap_snr/event_predictions.csv" \
  --run "Particles2SNR_F event=particles2SNR_pipeline/output/p0_c1_Particles2SNR_F/test/event_conv1dgap_snr_retrained/event_predictions.csv" \
  --bins 10 \
  --balance class-snr \
  --seed 42 \
  --targets 0.85,0.90,0.95,0.97 \
  --output-dir particles2SNR_pipeline/output/event_accuracy_snr_comparison
```

The balanced outputs are the reference comparison. Target SNR thresholds are reported only when the target accuracy is reached and remains reached for all higher usable SNR bins:

- `event_accuracy_comparison_balanced.csv/json/pdf`
- `event_accuracy_comparison_available.csv/json/pdf` for the unbalanced natural distribution check.

## Saturation Detection

Scan a dataset and export interval-level saturation metadata:

```bash
python3 detect_saturation.py ../particle_detector/test \
  --intervals-csv output/saturation_intervals.csv \
  --summary-json output/saturation_summary.json
```

## Non-Destructive Saturation Cleaning

Create a derived cleaned copy without mutating source files:

```bash
python3 saturation_cleaning.py ../particle_detector/test \
  --output-dir output/cleaned_dataset \
  --noise-dir ../P0/data/Noise \
  --policy replace \
  --guard-before 300 \
  --guard-after 300
```

Policies:

- `replace`: fill unsafe intervals from a captured-noise pool.
- `mask`: fill unsafe intervals with zero.
- `keep`: only report intervals; samples are copied unchanged.

Outputs:

- `saturation_cleaning_manifest.csv`: one row per unsafe interval/action.
- `saturation_cleaning_summary.json`: run parameters and aggregate counts.

Manifest schema:

- `source_path`, `output_path`, `class`, `policy`, `interval_idx`,
  `start_sample`, `end_sample`, `duration_samples`, `action`,
  `dropped_events`, `fs`, `fmin`, `fmax`, `min_flat`, `zero_threshold`,
  `guard_before`, `guard_after`.

## Smoke Commands

The global `python3` environment may not have `torch`; use the project venv for
the particles2SNR detector when needed:

```bash
P0/.venv/bin/python particles2SNR_pipeline/run_dataset.py \
  --dataset-dir /tmp/particles2SNR_smoke_3x \
  --output-dir /tmp/particles2SNR_smoke_3x_out \
  --device cpu

python3 particles2SNR_pipeline/snr_noise_report.py \
  --input-dir /tmp/particles2SNR_smoke_3x_out \
  --output-dir /tmp/particles2SNR_smoke_3x_report

python3 particles2SNR_pipeline/tests/test_snr_saturation_tools.py
```

## Single-File Debug

```bash
python3 fft_analysis_pipeline_particles2SNR.py \
  --target-file ../particle_detector/test/4um/HFocusing_5_10_4um_0_788.npy \
  --output-dir output \
  --verbose
```
