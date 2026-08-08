from __future__ import annotations

from collections import Counter

from particles2snr.yeast_4class_dataset import (
    CLASS_NAMES,
    select_background_rows,
    select_event_rows,
)


def _event(event_id: str, source: str, split: str = "development_train", quality: str = "strict") -> dict[str, str]:
    return {
        "event_id": event_id,
        "record_id": f"record-{event_id}",
        "source_group": source,
        "development_split": split,
        "quality": quality,
    }


def test_event_mapping_excludes_low_concentration_shmoo_and_test() -> None:
    rows = [
        _event("a", "budding"),
        _event("b", "mix"),
        _event("c", "shmoo2"),
        _event("d", "shmoo"),
        _event("e", "shmoo2", split="in_session_test"),
        _event("f", "mix", quality="reject"),
        _event("g", "budding"),
    ]
    selected = select_event_rows(rows, {"record-g"})
    assert [row["class_name"] for row in selected] == ["budding", "mix", "shmoo"]
    assert [row["class_id"] for row in selected] == [1, 2, 3]
    assert CLASS_NAMES == ("background", "budding", "mix", "shmoo")
    assert all(row["source_group_original"] != "shmoo" for row in selected)


def test_event_mapping_can_merge_shmoo1_and_shmoo2() -> None:
    rows = [_event("a", "shmoo"), _event("b", "shmoo2"), _event("c", "mix")]
    selected = select_event_rows(
        rows,
        set(),
        source_to_class={"budding": "budding", "mix": "mix", "shmoo": "shmoo", "shmoo2": "shmoo"},
    )
    assert [row["class_name"] for row in selected] == ["shmoo", "shmoo", "mix"]
    assert [row["source_group_original"] for row in selected[:2]] == ["shmoo", "shmoo2"]


def test_background_selection_matches_largest_event_class_and_equal_sources() -> None:
    events = []
    for split in ("development_train", "development_validation"):
        for class_name, count in (("budding", 2), ("mix", 6), ("shmoo", 3)):
            events.extend({"development_split": split, "class_name": class_name} for _ in range(count))
    candidates = []
    for split in ("development_train", "development_validation"):
        for source in ("budding", "mix", "shmoo2"):
            for index in range(10):
                candidates.append(
                    {
                        "sample_id": f"{split}:{source}:{index}",
                        "development_split": split,
                        "source_group_original": source,
                        "background_energy": float(index),
                    }
                )
    selected = select_background_rows(candidates, events, seed=42)
    for split in ("development_train", "development_validation"):
        rows = [row for row in selected if row["development_split"] == split]
        assert len(rows) == 6
        assert Counter(row["source_group_original"] for row in rows) == {
            "budding": 2,
            "mix": 2,
            "shmoo2": 2,
        }
        assert Counter(row["background_selection"] for row in rows) == {
            "uniform": 3,
            "high_energy_clean": 3,
        }
