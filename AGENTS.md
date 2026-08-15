# particles2SNR data and provenance contract

- If `../workspace-repos.lock` exists, read `../AGENTS.md` first; Git roots do
  not inherit parent instructions.
- This repo alone owns dataset generation, signal processing, and provenance.
  Put reusable logic in `particles2snr/` and thin CLIs in grouped `scripts/`;
  import installed `p0`, never restore flat scripts or inject paths.
- `notebooks/` holds executable explainers as jupytext `.py:percent` sources
  (`.ipynb` stays untracked); they import installed packages and contain no
  detector mathematics, so a notebook and the tools cannot drift apart.
- A notebook section may emit manifested evidence, but only through
  `workspace notebooks execute <source.py> --run-id <id>`, which converts and
  runs the tracked source headlessly from top to bottom. Each emitting section
  writes its own run under `artifacts/` carrying the same provenance contract as
  a tool, plus the source hash and whether it was committed. An interactive
  kernel emits nothing: the failure mode is a missing run, never a false one.
  Never emit a metric an existing run already owns.
- Write data only to workspace `datasets/{interim,processed}`, register it with
  checksums, and never mutate a registered version.
- Write manifested reports and runs under `artifacts/particles2SNR-pipeline/`.
  Do not create local data/output/result trees, caches, or environments.
- Verify from the workspace root with
  `.venv/bin/python -m pytest -q particles2SNR-pipeline/tests`.
