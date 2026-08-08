from __future__ import annotations

import numpy as np
import torch

from p0.data import BandpassFilter, Decimate
from particles2snr.p0_2500_roundtrip import (
    legacy_preprocess_2500,
    normalized_valid_correlation,
    parse_parent_crop_name,
)


def test_parent_crop_name_parses_source_and_legacy_index() -> None:
    source, index = parse_parent_crop_name(
        "HFocusing_5_10_4um_0_1297.npy981.npy"
    )
    assert source == "HFocusing_5_10_4um_0_1297.npy"
    assert index == 981


def test_normalized_valid_correlation_recovers_exact_offset() -> None:
    rng = np.random.default_rng(4)
    source = rng.normal(size=6000).astype(np.float32)
    crop = source[1234:3734].copy()
    start, score = normalized_valid_correlation(source, crop)
    assert start == 1234
    assert score > 0.999999


def test_legacy_preprocessing_matches_training_transforms() -> None:
    signal = np.random.default_rng(3).normal(size=2500).astype(np.float32)
    expected = Decimate(4)(BandpassFilter()(torch.from_numpy(signal)[None, :]))
    actual = legacy_preprocess_2500(signal)
    np.testing.assert_allclose(actual, expected.numpy()[0], rtol=0.0, atol=0.0)
    assert actual.shape == (625,)
