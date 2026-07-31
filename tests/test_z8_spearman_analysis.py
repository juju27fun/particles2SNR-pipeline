from __future__ import annotations

import numpy as np

from particles2snr.z8_spearman_analysis import (
    PAIR_ORDER,
    analyze_spearman,
    grouped_bootstrap_correlations,
    marginal_and_partial_correlations,
    pair_id,
    rank_parameter_pairs,
    rows_for_class,
    spearman_coefficient,
)
from particles2snr.z8_spearman_result_visual import select_real_rank_example


def _row(
    *,
    event_id: str,
    class_name: str,
    physical_source_class: str,
    source_filename: str,
    index: int,
) -> dict[str, str]:
    class_offset = {"2um": 0.0, "4um": 0.7, "10um": 1.4}[physical_source_class]
    return {
        "event_id": event_id,
        "class_name": class_name,
        "physical_source_class": physical_source_class,
        "source_filename": source_filename,
        "particles2snr_amplitude": str(0.05 + class_offset + 0.013 * index + 0.003 * (index % 3)),
        "frequency_hz": str(8_000.0 + 1_500.0 * class_offset + 190.0 * index),
        "tau_ms": str(0.31 - 0.006 * index + 0.002 * (index % 4)),
        "snr_db": str(-8.0 + 0.8 * index + 0.25 * (index % 5)),
    }


def _small_population() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for class_name in ("2um", "4um", "10um"):
        for index in range(24):
            rows.append(
                _row(
                    event_id=f"{class_name}-{index}",
                    class_name=class_name,
                    physical_source_class=class_name,
                    source_filename=f"{class_name}-source-{index // 2}",
                    index=index,
                )
            )
        for index in range(3):
            row = _row(
                event_id=f"{class_name}-unclear-{index}",
                class_name="unclear",
                physical_source_class=class_name,
                source_filename=f"{class_name}-unclear-source-{index}",
                index=24 + index,
            )
            row["snr_db"] = str(-15.0 - index)
            rows.append(row)
    return rows


def test_spearman_rank_example_matches_minus_point_nine() -> None:
    frequency = np.asarray([10.0, 15.0, 20.0, 30.0, 35.0])
    tau = np.asarray([0.28, 0.25, 0.20, 0.14, 0.16])

    assert np.isclose(spearman_coefficient(frequency, tau), -0.9)


def test_spearman_uses_average_ranks_for_ties() -> None:
    x = np.asarray([1.0, 2.0, 2.0, 4.0, 5.0])
    y = np.asarray([5.0, 4.0, 3.0, 2.0, 1.0])

    assert np.isclose(spearman_coefficient(x, y), -0.9746794344808964)


def test_partial_spearman_removes_rank_confounding() -> None:
    rng = np.random.default_rng(4)
    confounder = rng.normal(size=600)
    x = confounder + rng.normal(scale=0.25, size=600)
    y = confounder + rng.normal(scale=0.25, size=600)
    unrelated = rng.normal(size=600)
    matrix = np.column_stack([x, y, confounder, unrelated])

    marginal, partial = marginal_and_partial_correlations(matrix)

    assert marginal[0] > 0.85
    assert abs(partial[0]) < 0.15


def test_grouped_bootstrap_is_deterministic() -> None:
    rows = rows_for_class(_small_population(), "2um", include_unclear=False)

    first = grouped_bootstrap_correlations(
        rows, replicates=25, rng=np.random.default_rng(20260722)
    )
    second = grouped_bootstrap_correlations(
        rows, replicates=25, rng=np.random.default_rng(20260722)
    )

    np.testing.assert_allclose(first[0], second[0])
    np.testing.assert_allclose(first[1], second[1])
    assert first[0].shape == (25, 6)


def test_unclear_rows_are_isolated_from_primary_population() -> None:
    rows = _small_population()

    primary = rows_for_class(rows, "4um", include_unclear=False)
    inclusive = rows_for_class(rows, "4um", include_unclear=True)

    assert len(primary) == 24
    assert len(inclusive) == 27
    assert all(row["class_name"] == "4um" for row in primary)
    assert sum(row["class_name"] == "unclear" for row in inclusive) == 3


def test_ranking_keeps_exactly_three_highest_median_absolute_scores() -> None:
    desired_scores = [0.85, 0.70, 0.55, 0.40, 0.25, 0.10]
    rows = []
    for pair, score in zip(PAIR_ORDER, desired_scores, strict=True):
        for class_name, multiplier in zip(("2um", "4um", "10um"), (0.9, 1.0, 1.1), strict=True):
            rows.append(
                {
                    "class_name": class_name,
                    "pair_id": pair_id(pair),
                    "spearman_rho": score * multiplier,
                    "ci_excludes_zero": score >= 0.4,
                }
            )

    ranking = rank_parameter_pairs(rows)

    assert [row["pair_id"] for row in ranking[:3]] == [pair_id(pair) for pair in PAIR_ORDER[:3]]
    assert sum(row["selected_top_three"] for row in ranking) == 3
    assert [row["rank"] for row in ranking] == list(range(1, 7))


def test_complete_analysis_emits_primary_partial_sensitivity_and_top_three() -> None:
    analysis = analyze_spearman(
        _small_population(), bootstrap_replicates=20, bootstrap_seed=20260722
    )

    assert len(analysis["marginal_correlations"]) == 18
    assert len(analysis["partial_correlations"]) == 18
    assert len(analysis["unclear_sensitivity"]) == 9
    assert len(analysis["ranking"]) == 6
    assert len(analysis["selected_pairs"]) == 3
    assert {row["n_events"] for row in analysis["marginal_correlations"]} == {24}
    assert {row["n_unclear_events_added"] for row in analysis["unclear_sensitivity"]} == {3}


def test_real_rank_example_is_deterministic_and_uses_real_rows() -> None:
    rows = _small_population()

    first = select_real_rank_example(rows)
    second = select_real_rank_example(list(reversed(rows)))

    assert first == second
    assert len(first) == 5
    assert all(row["event_id"].startswith("4um-") for row in first)
    assert [row["frequency_rank"] for row in first] == [1.0, 2.0, 3.0, 4.0, 5.0]
