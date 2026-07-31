from pathlib import Path

import numpy as np

from particles2snr.z8_real_noise_ablation import (
    Carrier,
    eligible_window_starts,
    inject_real_noise,
    reconstruct_clean,
    round_robin_carriers,
    yolo_blocked_intervals,
)


def test_yolo_guard_and_window_exclusion(tmp_path: Path):
    label = tmp_path / "signal.txt"
    label.write_text("0 0.5 0.1\n", encoding="utf-8")
    blocked = yolo_blocked_intervals(
        label, signal_length=1000, guard_samples=50
    )
    assert blocked == [(400, 600)]
    starts = eligible_window_starts(
        signal_length=1000,
        window_length=200,
        stride=100,
        blocked_intervals=blocked,
    )
    assert starts == [0, 100, 200, 600, 700, 800]


def _carrier(source: str, source_round: int) -> Carrier:
    values = np.linspace(-1.0, 1.0, 4096, dtype=np.float32)
    return Carrier(
        class_name="4um",
        split="train",
        source_relative_path=source,
        start_sample=source_round * 1024,
        end_sample=source_round * 1024 + 4096,
        source_round=source_round,
        rms=float(values.std()),
        sha256=f"{source}-{source_round}",
        values=values,
    )


def test_round_robin_uses_each_source_before_reuse():
    carriers = [_carrier(source, round_index) for round_index in range(2)
                for source in ("a.npy", "b.npy", "c.npy")]
    selected = round_robin_carriers(
        carriers, class_name="4um", required=5, seed=3
    )
    assert len({item.source_relative_path for item in selected[:3]}) == 3


def test_round_robin_reuses_only_after_unique_exhaustion():
    carriers = [_carrier(source, 0) for source in ("a.npy", "b.npy")]
    selected = round_robin_carriers(
        carriers,
        class_name="4um",
        required=5,
        seed=3,
        allow_reuse_after_exhaustion=True,
    )
    assert len(selected) == 5
    assert len({item.sha256 for item in selected[:2]}) == 2
    assert selected[2].sha256 == selected[0].sha256


def test_real_noise_injection_preserves_requested_snr():
    rows = [{
        "sample_id": "x",
        "class_name": "4um",
        "amplitude_p0": 0.5,
        "frequency_khz": 20.0,
        "tau_ms": 0.2,
        "snr_db": 6.0,
        "phi_rad": 0.4,
        "t0_fraction": 0.5,
    }]
    clean = reconstruct_clean(rows)
    carrier = _carrier("a.npy", 0)
    raw, model, metadata = inject_real_noise(rows, clean, [carrier])
    assert raw.shape == (1, 4096)
    assert model.shape == (1, 512)
    assert abs(metadata[0]["achieved_snr_db"] - 6.0) < 1e-10
