from __future__ import annotations

from pathlib import Path

import pytest

from particles2snr.particle_dataset_metric_diagnostic import (
    EXPECTED_COUNTS,
    blind_assignment,
    build_dataset_metric_diagnostic,
    serializable_diagnostic,
)


WORKSPACE = Path(__file__).resolve().parents[2]
DATASETS = (
    WORKSPACE / "datasets/processed/particles2SNR-pipeline"
)


def test_blind_assignment_is_deterministic_and_complete() -> None:
    first = blind_assignment("blind-2um-case-01")
    second = blind_assignment("blind-2um-case-01")
    assert first == second
    assert set(first) == {"A", "B"}
    assert set(first.values()) == {"v1", "v2"}


def test_identifier_pairing_is_explicitly_forbidden() -> None:
    with pytest.raises(ValueError, match="not physical counterfactual pairs"):
        build_dataset_metric_diagnostic(
            primary_real_root=Path("unused"),
            simulation_v1_root=Path("unused"),
            simulation_v2_root=Path("unused"),
            pair_by_identifier=True,
            bootstrap_repeats=2,
        )


def test_registered_2um_diagnostic_population_is_frozen() -> None:
    model = build_dataset_metric_diagnostic(
        primary_real_root=(
            DATASETS / "particles2snr-f-c1-descriptor-events-4class/v1"
        ),
        simulation_v1_root=DATASETS / "yeast-passage-simulations/v1",
        simulation_v2_root=DATASETS / "yeast-passage-simulations/v2",
        bootstrap_repeats=20,
    )
    assert model["population_counts"] == EXPECTED_COUNTS
    assert [case["category"] for case in model["cases"]] == [
        "both_compatible",
        "v1_only",
        "v2_only",
    ]
    assert model["split_contract"]["sealed_splits_used"] == []
    assert (
        model["aggregate"]["bootstrap"]["v1_closer_fraction"]["estimate"]
        > 0.5
    )
    public = serializable_diagnostic(model)
    assert all(
        not key.startswith("_")
        for case in public["cases"]
        for key in case
    )
