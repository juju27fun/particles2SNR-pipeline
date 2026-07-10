"""Shared repository paths for particles2SNR pipeline scripts."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MONOREPO_ROOT = REPO_ROOT.parent

DATA_DERIVED = MONOREPO_ROOT / "datasets" / "interim" / "particles2SNR-pipeline"
_ARTIFACT_ROOT = MONOREPO_ROOT / "artifacts" / "particles2SNR-pipeline"
RESULTS_RUNS = _ARTIFACT_ROOT / "runs"
RESULTS_REPORTS = _ARTIFACT_ROOT / "reports"
RESULTS_FIGURES = _ARTIFACT_ROOT / "figures"
