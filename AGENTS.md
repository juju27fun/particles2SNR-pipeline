# particles2SNR Pipeline agent context

- This repository owns dataset generation, signal processing, and dataset
  provenance for the parent workspace; obey the root `AGENTS.md` as well.
- Put reusable logic in `particles2snr/` and thin CLIs in the grouped `scripts/`
  directories. Do not restore flat root scripts.
- Write generated data to the workspace dataset roots and register it with a
  checksum manifest. Never mutate a registered version in place.
- Write reports and run payloads to `artifacts/particles2SNR-pipeline/` with a
  `run.json`; do not create local `data/`, `output/`, `results/`, or a venv.
- Import shared `p0` utilities from the installed package, not by path.
- Verify with `.venv/bin/python -m pytest -q particles2SNR-pipeline/tests` from
  the workspace root.
