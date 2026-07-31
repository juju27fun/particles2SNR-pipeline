from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


SCRIPTS = (
    "analyze_z8_parameter_distributions.py",
    "analyze_z8_gaussian_marginals.py",
    "analyze_z8_gaussian_envelopes.py",
    "analyze_z8_spearman_correlations.py",
    "analyze_z8_cholesky_correlations.py",
)


@pytest.mark.parametrize("name", SCRIPTS)
def test_cli_requires_explicit_dataset_run_and_evidence(name: str) -> None:
    script = Path(__file__).parents[1] / "scripts" / "analysis" / name
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, check=False
    )

    assert result.returncode == 2
    assert "--dataset" in result.stderr
    assert "--run-id" in result.stderr
    assert "--method-evidence-id" in result.stderr
    assert "--method-evidence-run-id" in result.stderr
    assert "--evidence-contract" in result.stderr


def test_parameter_distribution_cli_requires_explicit_source_dataset() -> None:
    script = Path(__file__).parents[1] / "scripts" / "analysis" / SCRIPTS[0]
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, check=False
    )

    assert result.returncode == 2
    assert "--source-dataset" in result.stderr


@pytest.mark.parametrize("name", SCRIPTS[1:])
def test_downstream_cli_requires_approved_estimation_population(name: str) -> None:
    script = Path(__file__).parents[1] / "scripts" / "analysis" / name
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, check=False
    )

    assert result.returncode == 2
    assert "--population-run-id" in result.stderr
    assert "--population-evidence-id" in result.stderr
    assert "--population-evidence-run-id" in result.stderr


@pytest.mark.parametrize("name", SCRIPTS)
def test_cli_has_no_silent_v1_dataset_or_run_binding(name: str) -> None:
    script = Path(__file__).parents[1] / "scripts" / "analysis" / name
    assert "@v1" not in script.read_text(encoding="utf-8")
