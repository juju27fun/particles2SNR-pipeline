from __future__ import annotations

import numpy as np
import pytest

from particles2snr.ssl_realism_audit import (
    MATCH_WEIGHTS,
    SignalRecord,
    attach_amplitude_percentiles,
)
from particles2snr.ssl_support_audit import (
    feature_scales,
    nearest_support_distances,
    quantile_edges,
    support_grid,
)


def _record(identifier: str, offset: float) -> SignalRecord:
    descriptors = {
        "duration_25_ms": 0.8 + offset,
        "envelope_concentration": 0.6 + offset * 0.1,
        "dominant_frequency_khz": 20.0 + offset,
        "spectral_bandwidth_khz": 4.0 + offset * 0.2,
        "temporal_peak_count": 1.0,
        "spectral_peak_count": 1.0,
        "rms": 1.0 + offset * 0.1,
        "envelope_peak": 1.0,
        "peak_to_peak": 2.0,
        "support_start_index": 1000.0,
        "support_end_index": 2000.0,
    }
    return SignalRecord(identifier, np.ones(4096), {}, descriptors)


def test_nearest_support_distance_uses_frozen_reference_scales() -> None:
    reference = [_record("left", 0.0), _record("middle", 1.0), _record("right", 2.0)]
    queries = [_record("near-middle", 1.05)]
    attach_amplitude_percentiles(reference)
    attach_amplitude_percentiles(queries)

    result = nearest_support_distances(
        queries,
        reference,
        weights=MATCH_WEIGHTS,
        scales=feature_scales(reference),
    )

    assert result.nearest_indices.tolist() == [1]
    assert result.distances[0] >= 0.0


def test_quantile_edges_reject_degenerate_metric() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        quantile_edges(np.ones(12))


def test_support_grid_preserves_counts_and_rates() -> None:
    rows = [_record(str(index), float(index)) for index in range(4)]
    attach_amplitude_percentiles(rows)
    grid = support_grid(
        rows,
        np.asarray([True, False, True, True]),
        x_name="duration_25_ms",
        y_name="dominant_frequency_khz",
        x_edges=np.asarray([0.0, 2.0, 5.0]),
        y_edges=np.asarray([0.0, 22.0, 30.0]),
    )

    assert int(grid["counts"].sum()) == 4
    assert int(grid["supported_counts"].sum()) == 3
    assert np.nanmax(grid["rates"]) <= 1.0
