# particles2SNR Pipeline

Dataset generation, FFT-based particle detection, signal cleaning, SNR
analysis, and provenance for the research workspace.

## Ownership and layout

- `particles2snr/`: reusable signal-processing package.
- `scripts/generation/`: dataset builders.
- `scripts/analysis/`: detector, noise, saturation, and accuracy studies.
- `scripts/reports/`: reproducible report and presentation builders.
- `tests/`: source-only unit tests.

Generated datasets belong under the workspace `datasets/interim/` or
`datasets/processed/` roots and must be registered with `workspace datasets
register`. Detector runs and reports belong under
`artifacts/particles2SNR-pipeline/`. Do not recreate repository-local `data/`,
`output/`, or `results/` directories.

## Development

From the workspace root:

```bash
.venv/bin/python -m pip install -e particles2SNR-pipeline
.venv/bin/python -m pytest -q particles2SNR-pipeline/tests
.venv/bin/python -m particles2snr.run_dataset --help
.venv/bin/python particles2SNR-pipeline/scripts/generation/create_event_classification_dataset.py --help
```

The final dual-clean dataset contract, counts, provenance, limitations, and
study links are documented in
[`DATASET_CARD_DUAL_CLEAN_C1.md`](DATASET_CARD_DUAL_CLEAN_C1.md). The detailed
generation history is retained in
[`P0_C1_Particles2SNR_F_PIPELINE.md`](P0_C1_Particles2SNR_F_PIPELINE.md); paths
and counts inside historical sections may refer to pre-registry candidates.

The two registered Wave8-like long-sequence derivatives, their distinct
capability/deployment estimands, exact counts, construction, and uncertainty
rules are documented in
[`DATASET_CARD_WAVE8LIKE_C1.md`](DATASET_CARD_WAVE8LIKE_C1.md).

The registered multi-acquisition ingestion and sealed OOD procedure for the
yeast representation study is in
[`YEAST_ACQUISITION_INTAKE.md`](YEAST_ACQUISITION_INTAKE.md).
The same runbook documents the local candidate/full-trace reviewer served by
`scripts/reports/serve_yeast_event_review.py`.
