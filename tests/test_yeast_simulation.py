from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from particles2snr.yeast_simulation import FACTOR_POLICY, build_simulation_dataset, simulate_view


def test_paired_views_preserve_factors_and_change_nuisances(tmp_path: Path) -> None:
    output = tmp_path / "simulation"
    summary = build_simulation_dataset(
        output_dir=output,
        n_train_latents=2,
        n_validation_latents=1,
        n_test_latents=1,
        views_per_latent=2,
        seed=7,
    )
    with (output / "simulation_metadata.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    signals = np.load(output / "signals.npy", mmap_mode="r")
    assert signals.shape == (8, 4096)
    assert rows[0]["latent_id"] == rows[1]["latent_id"]
    assert rows[0]["duration_ms"] == rows[1]["duration_ms"]
    assert rows[0]["phase_rad"] != rows[1]["phase_rad"]
    assert not np.allclose(signals[0], signals[1])
    assert summary["split_signal_counts"] == {"test": 2, "train": 4, "validation": 2}
    assert json.loads((output / "factor_policy.json").read_text())["phase_rad"]["role"] == (
        "randomize_invariant"
    )


def test_simulated_view_matches_frozen_shape_and_finite_range() -> None:
    factors = {
        "duration_ms": 0.8,
        "doppler_khz": 18.0,
        "component_count": 2,
        "component_separation_ms": 0.2,
        "relative_component_amplitude": 0.7,
        "frequency_separation_khz": 3.0,
    }
    signal, nuisance = simulate_view(np.random.default_rng(3), factors)
    assert signal.shape == (4096,)
    assert np.isfinite(signal).all()
    assert 0.39 <= float(np.sqrt(np.mean(np.square(signal)))) <= 1.71
    assert nuisance["snr_db"] >= 0.0
    assert FACTOR_POLICY["yeast_morphology"]["role"] == "unresolved_excluded"
