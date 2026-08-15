from __future__ import annotations

from collections import Counter

from particles2snr.ledger_wave8like import (
    GUARD,
    NEGATIVE_RECIPES,
    _masks,
    _negative_recipes,
    _positive_recipe,
    partition_noise,
    source_is_eligible,
    Source,
)


def test_guard_masks_cover_all_three_boundaries() -> None:
    assert _masks() == [[16384 - GUARD, 16384 + GUARD], [32768 - GUARD, 32768 + GUARD], [49152 - GUARD, 49152 + GUARD]]


def test_support_touching_guard_is_ineligible() -> None:
    event = {"support_start": 299, "support_end": 1000}
    source = Source("s", "2um", 0, "x", "h", (event,))
    assert not source_is_eligible(source)


def test_negative_noise_exposure_cap() -> None:
    noise = [{"noise_id": f"n{i}", "path": f"n{i}.npy", "sha256": f"{i:064x}"} for i in range(244)]
    rows = _negative_recipes(0, noise)
    assert len(rows) == NEGATIVE_RECIPES
    counts = Counter(segment["signal_sha256"] for row in rows for segment in row["segments"])
    assert max(counts.values()) <= 2
    assert all(row["objectness_policy"] == "all_cells_negative_except_masks" for row in rows)


def test_positive_recipe_translates_geometry_and_preserves_weights() -> None:
    event = {
        "event_id": "event-1",
        "class_id": 0,
        "center": 1_000,
        "support_start": 800,
        "support_end": 1_200,
        "weights": {"presence": 1.0, "class": 1.0, "center": 1.0, "box": 0.25},
    }
    sources = [
        Source(f"s{i}", "2um", 0, f"s{i}.npy", f"{i:064x}", (event | {"event_id": f"event-{i}"},))
        for i in range(4)
    ]
    noise = [{"noise_id": "n", "path": "n.npy", "sha256": "f" * 64}]
    recipe = _positive_recipe(0, 0, 0, sources, noise)
    assert [row["center"] for row in recipe["events"]] == [1_000, 17_384, 33_768, 50_152]
    assert all(row["weights"] == event["weights"] for row in recipe["events"])
    assert recipe["objectness_policy"] == "positive_cells_only"
