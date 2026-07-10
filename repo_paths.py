"""Shared repository paths for particles2SNR pipeline scripts."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
MONOREPO_ROOT = REPO_ROOT.parent

DATA_DERIVED = REPO_ROOT / "data" / "derived"
RESULTS_RUNS = REPO_ROOT / "results" / "runs"
RESULTS_REPORTS = REPO_ROOT / "results" / "reports"
RESULTS_FIGURES = REPO_ROOT / "results" / "figures"

