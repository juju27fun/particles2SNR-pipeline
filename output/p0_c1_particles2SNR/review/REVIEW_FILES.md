# C1 particles2SNR Dataset Review Files

## 1. Annotation Quality

Open first:

- `particle_detector/visualizations/particles2SNR_c1_dataset_check.pdf`

Use this to check whether the green particles2SNR ground-truth windows align with
visible events in the signals. Red windows are predictions from the old YOLO
checkpoint, not a model retrained on this dataset.

## 2. Detailed particles2SNR Examples

Open these if you want to inspect the particles2SNR decomposition window by window:

- `particles2SNR_pipeline/output/p0_c1_particles2SNR/review/single_file_particles2SNR/fft_analysis_HFocusing_5_10_2um2_0_1.pdf`
- `particles2SNR_pipeline/output/p0_c1_particles2SNR/review/single_file_particles2SNR/fft_analysis_HFocusing_5_10_4um_0_10.pdf`
- `particles2SNR_pipeline/output/p0_c1_particles2SNR/review/single_file_particles2SNR/fft_analysis_HFocusing_5_10_10um_0_1012.pdf`

## 3. Noise And Signal Shape

Class-level visual reports:

- `particles2SNR_pipeline/output/p0_c1_particles2SNR/test/noise_2um_visual_check.pdf`
- `particles2SNR_pipeline/output/p0_c1_particles2SNR/test/noise_4um_visual_check.pdf`
- `particles2SNR_pipeline/output/p0_c1_particles2SNR/test/noise_10um_visual_check.pdf`

These show time-domain overlays, zooms, amplitude distributions, PSD, band
energy, and variability.

## 4. SNR By Class

Open:

- `particles2SNR_pipeline/output/p0_c1_particles2SNR/test/snr_noise_report/snr_by_class_distribution.pdf`
- `particles2SNR_pipeline/output/p0_c1_particles2SNR/test/snr_noise_report/raw_std_by_class.pdf`
- `particles2SNR_pipeline/output/p0_c1_particles2SNR/test/snr_noise_report/filtered_std_by_class.pdf`
- `particles2SNR_pipeline/output/p0_c1_particles2SNR/test/snr_noise_report/inband_energy_ratio_by_class.pdf`
- `particles2SNR_pipeline/output/p0_c1_particles2SNR/test/snr_noise_report/snr_noise_report.md`

Use these to decide whether SNR/noise differs meaningfully by class.

## 5. Conv1DGAP Transfer Result

Open:

- `particles2SNR_pipeline/output/p0_c1_particles2SNR/test/conv1dgap_snr/conv1dgap_accuracy_by_snr.pdf`
- `particles2SNR_pipeline/output/p0_c1_particles2SNR/test/conv1dgap_snr/conv1dgap_accuracy_by_snr.csv`
- `particles2SNR_pipeline/output/p0_c1_particles2SNR/test/conv1dgap_snr/conv1dgap_accuracy_by_snr.json`

Current result:

- rows used: `576`
- accuracy: `0.4809`
- macro F1: `0.3468`

This is a transfer check from a Conv1DGAP-L checkpoint trained on short P0
signals, not a model retrained on C1.

## 6. Dataset Integrity

Open or inspect:

- `particles2SNR_pipeline/output/p0_c1_particles2SNR/run_summary.json`
- `particles2SNR_pipeline/output/p0_c1_particles2SNR/source_split_manifest.csv`
- `particles2SNR_pipeline/output/p0_c1_particles2SNR/detseg_dataset_audit.json`
- `particles2SNR_pipeline/output/p0_c1_particles2SNR/test/zero_cleaning_manifest.csv`
- `particles2SNR_pipeline/output/p0_c1_particles2SNR/test/saturation_summary.json`

Important current counters:

- train files: `2310`
- test files: `578`
- test particles2SNR annotations: `6905`
- removed zero samples: `0`
- YOLO/detseg invalid labels: `0`
- YOLO/detseg wide labels: `0`
