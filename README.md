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

The accepted C1 clean generation history is documented in
`P0_C1_Particles2SNR_F_PIPELINE.md`; paths inside historical sections may refer
to the pre-registry layout.
