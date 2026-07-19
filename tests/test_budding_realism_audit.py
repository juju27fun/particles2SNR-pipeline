from __future__ import annotations

import numpy as np

from particles2snr.budding_realism_audit import (
    aggregate_metrics,
    deterministic_blind_order,
    exact_nearest_neighbor,
    stable_equalized_rows,
)


def test_equalization_is_stable_and_filters_component_count() -> None:
    rows = [
        {
            "signal_row": str(index),
            "latent_id": f"latent-{index // 2}",
            "view_index": str(index % 2),
            "split": "test",
            "component_count": "2" if index != 3 else "1",
        }
        for index in range(8)
    ]
    first = stable_equalized_rows(
        rows,
        split="test",
        count=4,
        component_count=2,
    )
    second = stable_equalized_rows(
        list(reversed(rows)),
        split="test",
        count=4,
        component_count=2,
    )
    assert [row["signal_row"] for row in first] == [
        row["signal_row"] for row in second
    ]
    assert all(row["component_count"] == "2" for row in first)


def test_metrics_and_nearest_neighbor_identify_exact_distribution() -> None:
    real = np.asarray([[0.0] * 7, [1.0] * 7, [2.0] * 7])
    center = np.zeros(7)
    scale = np.ones(7)
    metrics = aggregate_metrics(real, real.copy(), center=center, scale=scale)
    assert metrics["joint_energy_distance"] == 0.0
    assert metrics["median_real_to_sim_nearest_distance"] == 0.0
    index, distance = exact_nearest_neighbor(
        real[1],
        np.asarray([[3.0] * 7, [1.0] * 7]),
        center=center,
        scale=scale,
    )
    assert index == 1
    assert distance == 0.0


def test_blind_order_is_deterministic_and_complete() -> None:
    sources = ["v1", "data", "biophysics"]
    first = deterministic_blind_order("case-a", sources)
    assert first == deterministic_blind_order("case-a", list(reversed(sources)))
    assert sorted(first) == sorted(sources)
