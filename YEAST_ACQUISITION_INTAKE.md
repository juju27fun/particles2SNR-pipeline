# Yeast Acquisition Intake

This runbook adds an independently acquired yeast session without weakening the
sealed acquisition-OOD endpoint. The scientific rationale and gate thresholds
are defined in the P3
[`YEAST_SSL_CRITIQUE_AND_REBUILD_PLAN.md`](https://github.com/juju27fun/unsupervised-learning-flow-cytometry/blob/reorg/workspace-20260710/docs/YEAST_SSL_CRITIQUE_AND_REBUILD_PLAN.md).

## Required Metadata

Before copying signals, record an acquisition ID, date, biological preparation
or batch, instrument and sensor configuration, sampling rate, acquisition
settings, concentration protocol, and any operator or environmental change.
Folder names are conditions within a session; they are not independent
acquisitions or event-level labels.

The new session must be a new acquisition, not a re-export, transformed copy,
or subset of the current `2026-06-10-hf-10-5` data. Exact duplicates across
sessions are rejected by the index builder.

## Intake Sequence

Run from the workspace root. Confirm pfcalcul Slurm and Jupyter runners are idle
and retrieve outstanding results before moving or synchronizing data.

1. Audit the staged source into a new manifested artifact:

   ```bash
   .venv/bin/python particles2SNR-pipeline/scripts/analysis/audit_yeast_source.py \
     --source-root <staged-acquisition-root> \
     --output-dir artifacts/particles2SNR-pipeline/audits/<audit-run-id> \
     --run-id <audit-run-id> \
     --documented-acquisition-group <new-acquisition-id>
   ```

   A per-session audit still reports that one session alone cannot support an
   OOD split. This is expected. Resolve load errors, non-finite signals,
   unexplained duplicate runs, and missing metadata before import.

2. Import the verified source into a new immutable raw dataset version, then
   register and validate its dataset ID. Never merge it into
   `yeast-hf-10-5-20260610@v1`.

   ```bash
   .venv/bin/python particles2SNR-pipeline/scripts/generation/import_yeast_source.py \
     --source-root <staged-acquisition-root> \
     --source-inventory artifacts/particles2SNR-pipeline/audits/<audit-run-id>/source_inventory.csv \
     --destination datasets/raw/<new-raw-dataset>/v1

   .venv/bin/workspace datasets register <new-raw-dataset> --version v1 \
     --path datasets/raw/<new-raw-dataset>/v1 \
     --status active --producer particles2SNR-pipeline --format directory \
     --command "verified yeast acquisition import"
   .venv/bin/workspace datasets validate <new-raw-dataset>@v1 --full
   ```

3. Create an acquisition manifest with paths relative to the manifest file:

   ```json
   {
     "schema_version": 1,
     "acquisitions": [
       {
         "acquisition_id": "2026-06-10-hf-10-5",
         "raw_dataset": "yeast-hf-10-5-20260610@v1",
         "source_inventory": "<relative-current-inventory.csv>",
         "role": "development"
       },
       {
         "acquisition_id": "<new-acquisition-id>",
         "raw_dataset": "<new-raw-dataset>@v1",
         "source_inventory": "<relative-new-inventory.csv>",
         "role": "sealed_ood_test"
       }
     ]
   }
   ```

4. Build and register a new source-index version. This namespaces record and
   capture-block IDs by acquisition, preserves grouped development splits, and
   forces every new-session row into `sealed_acquisition_test`.

   ```bash
   .venv/bin/python particles2SNR-pipeline/scripts/generation/build_yeast_source_index.py \
     --acquisition-manifest <acquisition-manifest.json> \
     --capture-block-size 32 \
     --output-dir datasets/interim/particles2SNR-pipeline/yeast-source-index/v3
   ```

5. Build a raw dataset map whose keys are the registered IDs in the index and
   whose values are their registry-resolved roots. Use relative paths; do not
   commit machine-specific absolute paths.

   ```json
   {
     "schema_version": 1,
     "raw_datasets": {
       "yeast-hf-10-5-20260610@v1": "<relative-resolved-root>",
       "<new-raw-dataset>@v1": "<relative-resolved-root>"
     }
   }
   ```

6. Run a new candidate audit using the frozen detector. The new queue is
   stratified by acquisition, source group, and quality or detected-count
   bucket. Render and annotate both candidate-window and full-trace queues.

   ```bash
   .venv/bin/python particles2SNR-pipeline/scripts/generation/build_yeast_event_audit.py \
     --source-index datasets/interim/particles2SNR-pipeline/yeast-source-index/v3/source_index.csv \
     --raw-dataset-map <raw-dataset-map.json> \
     --output-dir datasets/interim/particles2SNR-pipeline/yeast-event-candidates/v6
   ```

7. Analyze completed copies with `--require-complete`. Gate 1 requires the
   frozen global thresholds plus per-source and per-acquisition precision and
   recall. Register completed annotations separately; never edit the candidate
   templates in place.

8. Only after Gate 1 passes, build a new representation dataset with
   `--raw-dataset-map`. Its global normalization is fitted on
   `development_train` only. P3 training must never request
   `sealed_acquisition_test` until the model, probe, seeds, and comparisons are
   frozen.

## Decision Rule

If the frozen detector and preprocessing pass on the new acquisition, retain it
as the one-shot OOD test. If any detector threshold, crop rule, preprocessing
choice, model selection, or probe choice is changed after inspecting that
acquisition, reclassify it as development/validation data and acquire a third
independent session for the final OOD endpoint. Do not repeatedly open the same
session until the result becomes favorable.
