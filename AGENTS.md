# particles2SNR data and provenance contract

- If `../workspace-repos.lock` exists, read `../AGENTS.md` first; Git roots do
  not inherit parent instructions.
- This repo alone owns dataset generation, signal processing, and provenance.
  Put reusable logic in `particles2snr/` and thin CLIs in grouped `scripts/`;
  import installed `p0`, never restore flat scripts or inject paths.
- Write data only to workspace `datasets/{interim,processed}`, register it with
  checksums, and never mutate a registered version.
- Write manifested reports and runs under `artifacts/particles2SNR-pipeline/`.
  Do not create local data/output/result trees, caches, or environments.
- Verify from the workspace root with
  `.venv/bin/python -m pytest -q particles2SNR-pipeline/tests`.
