from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from particles2snr.particle_class_coverage import FEATURE_NAMES
from particles2snr.particle_class_representativity import (
    SPHERE_WEIGHTS,
    _sphere_distance,
    build_representativity_model,
    serializable_model,
)
from particles2snr.ssl_realism_audit import SignalRecord


WORKSPACE = Path(__file__).resolve().parents[2]
DATASETS = WORKSPACE / "datasets/processed/particles2SNR-pipeline"


def _record(identifier: str, value: float) -> SignalRecord:
    descriptors = {name: value for name in FEATURE_NAMES}
    return SignalRecord(
        identifier=identifier,
        signal=np.ones(4096, dtype=np.float32),
        metadata={"source_group": identifier, "latent_id": identifier},
        descriptors=descriptors,
    )


def test_weighted_space_is_a_sphere_after_standardization() -> None:
    center = {name: 0.0 for name in FEATURE_NAMES}
    scales = {name: 1.0 for name in FEATURE_NAMES}
    distances, contributions = _sphere_distance(
        [_record("one", 1.0)],
        center=center,
        scales=scales,
    )
    assert np.isclose(sum(SPHERE_WEIGHTS.values()), 1.0)
    assert np.isclose(distances[0], 1.0)
    assert np.isclose(np.sum(contributions[0]), 1.0)


def test_representativity_rejects_v2_scope() -> None:
    with pytest.raises(ValueError, match="scoped to simulation v1"):
        build_representativity_model(
            primary_real_root=Path("unused"),
            sensitivity_real_root=Path("unused"),
            simulation_root=Path("unused"),
            simulation_dataset_id="yeast-passage-simulations@v2",
            bootstrap_repeats=2,
        )


def test_registered_v1_representativity_population_and_split_contract() -> None:
    model = build_representativity_model(
        primary_real_root=(
            DATASETS / "particles2snr-f-c1-descriptor-events-4class/v1"
        ),
        sensitivity_real_root=(
            DATASETS / "particles2snr-f-c1-descriptor-events-3class/v1"
        ),
        simulation_root=DATASETS / "yeast-passage-simulations/v1",
        bootstrap_repeats=20,
    )
    assert model["simulation_dataset"] == "yeast-passage-simulations@v1"
    assert model["simulation_counts"]["train_single_component"] == 6982
    assert model["primary"]["real_counts"]["train"] == {
        "2um": 470,
        "4um": 531,
        "10um": 414,
    }
    assert model["primary"]["real_counts"]["validation"] == {
        "2um": 138,
        "4um": 142,
        "10um": 91,
    }
    assert model["sealed_splits_used"] == []
    assert all(len(model["primary"]["cases"][name]) == 3 for name in ("2um", "4um", "10um"))
    public = serializable_model(model)
    assert "_simulation_train" not in public["primary"]
    assert "_real_fit" not in public["primary"]
