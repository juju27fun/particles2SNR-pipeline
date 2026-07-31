from __future__ import annotations

import numpy as np
import pytest

from particles2snr.z8_real_noise_ablation import (
    CarrierRef,
    round_robin_carriers,
    select_source_window_refs,
)


def test_window_ref_selection_is_deterministic_and_separated() -> None:
    signal = np.sin(np.arange(5000, dtype=np.float64) / 17.0)
    starts = list(range(0, 905, 32))
    first = select_source_window_refs(
        signal,
        starts,
        maximum_per_source=8,
        minimum_start_separation=64,
    )
    second = select_source_window_refs(
        signal,
        starts,
        maximum_per_source=8,
        minimum_start_separation=64,
    )
    assert first == second
    assert len(first) == 8
    assert all(
        abs(left[0] - right[0]) >= 64
        for index, left in enumerate(first)
        for right in first[index + 1 :]
    )


def test_round_robin_refs_refuses_reuse() -> None:
    refs = [
        CarrierRef(
            class_name="4um",
            split="train",
            source_relative_path=f"train/signals/source-{source}.npy",
            start_sample=round_index * 64,
            end_sample=round_index * 64 + 4096,
            source_round=round_index,
            rms=1.0,
        )
        for source in range(2)
        for round_index in range(2)
    ]
    selected = round_robin_carriers(
        refs,
        class_name="4um",
        required=4,
        seed=7,
        allow_reuse_after_exhaustion=False,
    )
    assert len(
        {(item.source_relative_path, item.start_sample) for item in selected}
    ) == 4
    with pytest.raises(ValueError, match="insufficient"):
        round_robin_carriers(
            refs,
            class_name="4um",
            required=5,
            seed=7,
            allow_reuse_after_exhaustion=False,
        )
