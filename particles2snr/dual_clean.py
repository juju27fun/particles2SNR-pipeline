"""Decision helpers for dual-clean peak evidence."""

from __future__ import annotations

import math
from typing import Any


def should_rescue_missing_clean_peak(
    *,
    filtered_peak_z: float | None,
    clean_local_peak_z: float | None,
    filtered_min_z: float | None,
    clean_local_min_z: float | None,
) -> bool:
    """Return whether strong filtered evidence can rescue weak clean support.

    Both thresholds must be explicitly configured.  Keeping the defaults at
    ``None`` preserves the strict historical dual-clean behaviour.
    """

    if filtered_min_z is None or clean_local_min_z is None:
        return False
    try:
        filtered = float(filtered_peak_z)
        clean_local = float(clean_local_peak_z)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(filtered) or not math.isfinite(clean_local):
        return False
    return filtered >= float(filtered_min_z) and clean_local >= float(clean_local_min_z)


def assign_peak_groups_one_to_one(
    intervals: list[tuple[float, float, float]],
    peak_groups: list[dict[str, Any]],
    *,
    margin_samples: float,
) -> dict[int, dict[str, Any]]:
    """Assign each clean peak group to at most one particle annotation."""

    if margin_samples < 0:
        raise ValueError("margin_samples must be non-negative")
    pairs = []
    for particle_index, (left, right, center) in enumerate(intervals):
        for group in peak_groups:
            peak = float(group["peak_sample"])
            if peak < left - margin_samples or peak > right + margin_samples:
                continue
            inside = left <= peak <= right
            boundary_distance = (
                0.0
                if inside
                else min(abs(peak - left), abs(peak - right))
            )
            pairs.append(
                (
                    0 if inside else 1,
                    boundary_distance,
                    abs(peak - center),
                    -float(group["peak_z"]),
                    particle_index,
                    int(group["id"]),
                    group,
                )
            )
    assigned_particles: set[int] = set()
    assigned_groups: set[int] = set()
    result: dict[int, dict[str, Any]] = {}
    for row in sorted(pairs, key=lambda item: item[:-1]):
        particle_index, group_id, group = row[-3:]
        if particle_index in assigned_particles or group_id in assigned_groups:
            continue
        assigned_particles.add(particle_index)
        assigned_groups.add(group_id)
        result[particle_index] = group
    return result
