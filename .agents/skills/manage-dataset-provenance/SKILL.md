---
name: manage-dataset-provenance
description: Build, repair, audit, register, version, or promote workspace datasets with immutable outputs, checksums, registered IDs, and manifested evidence. Use for any particles2SNR change that can alter samples, labels, splits, or dataset status.
---

# Manage Dataset Provenance

## Workflow

1. Read the workspace and repository `AGENTS.md` files.
2. Resolve every input by registered dataset ID and record its manifest hash.
3. Record repository revisions, exact command/config, random seeds, and parent
   dataset IDs before generating output.
4. Build a new candidate under `datasets/interim`; never mutate a registered
   version. Write training-ready immutable versions under `datasets/processed`.
5. Audit counts, file/label pairing, splits, dtype/shape, NaN/Inf, class mapping,
   and task-specific signal integrity before registration.
6. Write the build or audit evidence below
   `artifacts/particles2SNR-pipeline/<kind>/<run-id>/` with a valid `run.json`.
7. Register the new version with `workspace datasets register`; use
   `reference` until every scientific or human-review gate permits `active`.
8. Run `workspace datasets validate <id> --version <version> --full`,
   `workspace artifacts validate <run-dir>`, and the narrow particles2SNR tests.

## Invariants

- Store workspace-relative paths; never persist machine-specific absolute paths
  or dataset symlink views.
- Preserve raw recordings and registered versions byte-for-byte.
- Keep generation and provenance here; downstream repositories consume and
  audit registered versions but do not promote replacements.
- Treat incomplete arbitration, disputed labels, failed gates, and missing
  provenance as blockers to active promotion.
- Never delete superseded candidates without the workspace quarantine process.

## Handoff

Report the new dataset ID/version, parent IDs and hashes, generation command,
audit results, artifact run, registry record, remaining limitations, and exact
validation commands.
