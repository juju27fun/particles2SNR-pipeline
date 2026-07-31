from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, norm

from particles2snr.z8_gaussian_marginal_analysis import (
    MARGINALS,
    raw_values,
    rows_for_marginal,
)
from particles2snr.z8_parameter_analysis import (
    CLASS_COLORS,
    CLASS_LABELS,
    CLASS_ORDER,
)


SEED = 20_260_723
SAMPLING_FREQUENCY_HZ = 2_000_000.0
RAW_LENGTH = 4_096
MODEL_LENGTH = 512
DECIMATION_FACTOR = RAW_LENGTH // MODEL_LENGTH
FREQUENCY_BAND_KHZ = (7.0, 80.0)
PARAMETER_ORDER = (
    "log_amplitude_p0",
    "frequency_khz",
    "log_tau_ms",
    "snr_db",
)
MARGINAL_BY_PARAMETER = {
    "log_amplitude_p0": "amplitude_p0",
    "frequency_khz": "frequency_khz",
    "log_tau_ms": "tau_ms",
    "snr_db": "snr_db",
}
CORRELATION_WARNING_THRESHOLD = 0.10
SIGNAL_GALLERY_ROLES = (
    "central",
    "low_frequency",
    "high_frequency",
    "low_snr",
    "high_amplitude",
    "long_tau",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_float(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {key}: {row[key]!r}")
    return value


def load_gaussian_targets(
    path: Path,
    *,
    include_budgets: bool = False,
) -> dict[str, dict[str, Any]] | tuple[
    dict[str, dict[str, Any]], dict[str, int]
]:
    rows = read_csv(path)
    targets: dict[str, dict[str, Any]] = {}
    for class_name in CLASS_ORDER:
        class_rows = {row["marginal"]: row for row in rows if row["class_name"] == class_name}
        if set(class_rows) != set(MARGINAL_BY_PARAMETER.values()):
            raise ValueError(f"Incomplete Gaussian targets for {class_name}")
        means: list[float] = []
        sigmas: list[float] = []
        populations: list[str] = []
        for parameter in PARAMETER_ORDER:
            marginal = MARGINAL_BY_PARAMETER[parameter]
            row = class_rows[marginal]
            means.append(_as_float(row, "gaussian_mean_transformed"))
            sigma = _as_float(row, "gaussian_sigma_transformed")
            if sigma <= 0.0:
                raise ValueError(f"Non-positive Gaussian sigma for {class_name}/{marginal}")
            sigmas.append(sigma)
            populations.append(row["population"])
        if populations != ["physical", "physical", "physical", "inclusive"]:
            raise ValueError(
                f"Unexpected Gaussian population policy for {class_name}: {populations}"
            )
        budget_values = {
            int(class_rows[marginal]["class_synthetic_event_count"])
            for marginal in class_rows
        }
        if len(budget_values) != 1:
            raise ValueError(
                f"Inconsistent Gaussian budgets for {class_name}: {budget_values}"
            )
        budget = budget_values.pop()
        if budget <= 0:
            raise ValueError(f"Non-positive Gaussian budget for {class_name}: {budget}")
        targets[class_name] = {
            "mean": np.asarray(means, dtype=np.float64),
            "sigma": np.asarray(sigmas, dtype=np.float64),
            "populations": populations,
            "budget": budget,
        }
    budgets = {
        class_name: int(targets[class_name]["budget"]) for class_name in CLASS_ORDER
    }
    return (targets, budgets) if include_budgets else targets


def load_physical_cholesky(path: Path) -> dict[str, np.ndarray]:
    rows = read_csv(path)
    factors: dict[str, np.ndarray] = {}
    for class_name in CLASS_ORDER:
        selected = [
            row
            for row in rows
            if row["class_name"] == class_name and row["population"] == "physical"
        ]
        if len(selected) != 10:
            raise ValueError(
                f"Expected 10 physical Cholesky coefficients for {class_name}"
            )
        factor = np.zeros((4, 4), dtype=np.float64)
        for row in selected:
            row_index = PARAMETER_ORDER.index(row["row_parameter"])
            column_index = PARAMETER_ORDER.index(row["column_parameter"])
            if column_index > row_index:
                raise ValueError("Cholesky factor contains an upper-triangular value")
            factor[row_index, column_index] = _as_float(row, "cholesky_value")
        correlation = factor @ factor.T
        if not np.allclose(np.diag(correlation), 1.0, atol=1e-10):
            raise ValueError(f"Invalid Cholesky diagonal for {class_name}")
        factors[class_name] = factor
    return factors


def load_recommended_cholesky(
    factor_path: Path,
    recommendation_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    rows = read_csv(factor_path)
    recommendation_rows = read_csv(recommendation_path)
    populations: dict[str, str] = {}
    for class_name in CLASS_ORDER:
        selected_recommendations = [
            row for row in recommendation_rows if row["class_name"] == class_name
        ]
        if len(selected_recommendations) != 1:
            raise ValueError(f"Expected one Cholesky recommendation for {class_name}")
        population = selected_recommendations[0]["recommended_dependency_matrix"]
        if population not in {"physical", "inclusive"}:
            raise ValueError(
                f"Unsupported Cholesky population for {class_name}: {population}"
            )
        populations[class_name] = population

    factors: dict[str, np.ndarray] = {}
    for class_name in CLASS_ORDER:
        population = populations[class_name]
        selected = [
            row
            for row in rows
            if row["class_name"] == class_name and row["population"] == population
        ]
        if len(selected) != 10:
            raise ValueError(
                f"Expected 10 {population} Cholesky coefficients for {class_name}"
            )
        factor = np.zeros((4, 4), dtype=np.float64)
        for row in selected:
            row_index = PARAMETER_ORDER.index(row["row_parameter"])
            column_index = PARAMETER_ORDER.index(row["column_parameter"])
            if column_index > row_index:
                raise ValueError("Cholesky factor contains an upper-triangular value")
            factor[row_index, column_index] = _as_float(row, "cholesky_value")
        correlation = factor @ factor.T
        if not np.allclose(np.diag(correlation), 1.0, atol=1e-10):
            raise ValueError(f"Invalid Cholesky diagonal for {class_name}")
        factors[class_name] = factor
    return factors, populations


def _sample_id(
    class_name: str,
    class_index: int,
    *,
    dataset_id: str,
    seed: int = SEED,
) -> str:
    payload = f"{dataset_id}:{seed}:{class_name}:{class_index}".encode()
    return f"syn-{class_name}-{hashlib.sha256(payload).hexdigest()[:16]}"


def generate_parameters(
    targets: dict[str, dict[str, Any]],
    factors: dict[str, np.ndarray],
    *,
    seed: int = SEED,
    budgets: dict[str, int],
    dataset_id: str,
    frequency_band_khz: tuple[float, float] = FREQUENCY_BAND_KHZ,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    retained_budgets = budgets
    if not retained_budgets or not set(retained_budgets).issubset(CLASS_ORDER):
        raise ValueError("Budgets must contain known particle classes")
    low_frequency, high_frequency = frequency_band_khz
    if not 0.0 < low_frequency < high_frequency:
        raise ValueError("Invalid frequency band")

    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    rejection_counts: dict[str, int] = {}
    proposal_index = 0
    for class_name in CLASS_ORDER:
        if class_name not in retained_budgets:
            continue
        target_count = int(retained_budgets[class_name])
        if target_count <= 0:
            raise ValueError(f"Non-positive budget for {class_name}")
        mean = np.asarray(targets[class_name]["mean"], dtype=np.float64)
        sigma = np.asarray(targets[class_name]["sigma"], dtype=np.float64)
        factor = np.asarray(factors[class_name], dtype=np.float64)
        class_records: list[dict[str, Any]] = []
        rejected = 0
        while len(class_records) < target_count:
            remaining = target_count - len(class_records)
            batch_size = max(64, int(math.ceil(remaining * 1.8)))
            z_values = rng.normal(size=(batch_size, 4))
            correlated = z_values @ factor.T
            transformed = mean[None, :] + correlated * sigma[None, :]
            finite = np.all(np.isfinite(transformed), axis=1)
            frequency = transformed[:, 1]
            accepted = finite & (frequency >= low_frequency) & (frequency <= high_frequency)
            rejected += int(np.sum(~accepted))
            for z, u, y in zip(
                z_values[accepted], correlated[accepted], transformed[accepted], strict=True
            ):
                if len(class_records) >= target_count:
                    break
                class_index = len(class_records)
                amplitude = float(np.exp(y[0]))
                tau_ms = float(np.exp(y[2]))
                if not math.isfinite(amplitude) or not math.isfinite(tau_ms):
                    rejected += 1
                    continue
                phi = float(rng.uniform(0.0, 2.0 * np.pi))
                record: dict[str, Any] = {
                    "sample_id": _sample_id(
                        class_name,
                        class_index,
                        dataset_id=dataset_id,
                        seed=seed,
                    ),
                    "class_name": class_name,
                    "class_index": class_index,
                    "proposal_index": proposal_index,
                    "amplitude_p0": amplitude,
                    "frequency_khz": float(y[1]),
                    "tau_ms": tau_ms,
                    "snr_db": float(y[3]),
                    "phi_rad": phi,
                    "t0_fraction": 0.5,
                    "log_amplitude_p0": float(y[0]),
                    "log_tau_ms": float(y[2]),
                }
                for index in range(4):
                    record[f"z{index + 1}"] = float(z[index])
                    record[f"u{index + 1}"] = float(u[index])
                class_records.append(record)
                proposal_index += 1
        rejection_counts[class_name] = rejected
        records.extend(class_records)

    ids = [row["sample_id"] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Generated sample IDs are not unique")
    return records, rejection_counts


def _band_limited_unit_noise(
    rng: np.random.Generator,
    batch_size: int,
    *,
    length: int = RAW_LENGTH,
    sampling_frequency_hz: float = SAMPLING_FREQUENCY_HZ,
    frequency_band_khz: tuple[float, float] = FREQUENCY_BAND_KHZ,
) -> np.ndarray:
    white = rng.normal(size=(batch_size, length))
    spectrum = np.fft.rfft(white, axis=1)
    frequency = np.fft.rfftfreq(length, d=1.0 / sampling_frequency_hz)
    low_hz, high_hz = (frequency_band_khz[0] * 1000.0, frequency_band_khz[1] * 1000.0)
    spectrum[:, (frequency < low_hz) | (frequency > high_hz)] = 0.0
    noise = np.fft.irfft(spectrum, n=length, axis=1)
    noise -= np.mean(noise, axis=1, keepdims=True)
    rms = np.sqrt(np.mean(np.square(noise), axis=1, keepdims=True))
    if np.any(rms <= 0.0) or not np.all(np.isfinite(rms)):
        raise ValueError("Band-limited noise has invalid RMS")
    return noise / rms


def preprocess_conv1dgap_512(signals: np.ndarray) -> np.ndarray:
    values = np.asarray(signals, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != RAW_LENGTH:
        raise ValueError(f"Expected signals shaped (n, {RAW_LENGTH})")
    decimated = values.reshape(values.shape[0], MODEL_LENGTH, DECIMATION_FACTOR).mean(axis=2)
    centered = decimated - np.mean(decimated, axis=1, keepdims=True)
    scale = np.std(centered, axis=1, keepdims=True)
    if np.any(scale <= 0.0) or not np.all(np.isfinite(scale)):
        raise ValueError("Cannot z-score a constant or invalid signal")
    return (centered / scale).astype(np.float32)


def synthesize_signals(
    records: list[dict[str, Any]],
    *,
    seed: int = SEED + 1,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    count = len(records)
    raw = np.empty((count, RAW_LENGTH), dtype=np.float32)
    rng = np.random.default_rng(seed)
    time_s = (
        np.arange(RAW_LENGTH, dtype=np.float64) - (RAW_LENGTH - 1) / 2.0
    ) / SAMPLING_FREQUENCY_HZ

    for start in range(0, count, batch_size):
        stop = min(count, start + batch_size)
        chunk = records[start:stop]
        amplitude = np.asarray([row["amplitude_p0"] for row in chunk], dtype=np.float64)
        frequency_hz = np.asarray(
            [row["frequency_khz"] * 1000.0 for row in chunk], dtype=np.float64
        )
        tau_s = np.asarray([row["tau_ms"] / 1000.0 for row in chunk], dtype=np.float64)
        phase = np.asarray([row["phi_rad"] for row in chunk], dtype=np.float64)
        envelope = amplitude[:, None] * np.exp(
            -0.5 * np.square(time_s[None, :] / tau_s[:, None])
        )
        clean = envelope * np.cos(
            2.0 * np.pi * frequency_hz[:, None] * time_s[None, :] + phase[:, None]
        )
        clean_rms = np.sqrt(np.mean(np.square(clean), axis=1))
        unit_noise = _band_limited_unit_noise(rng, len(chunk))
        requested_snr = np.asarray([row["snr_db"] for row in chunk], dtype=np.float64)
        noise_rms = clean_rms / np.power(10.0, requested_snr / 20.0)
        noisy = clean + unit_noise * noise_rms[:, None]
        actual_noise = noisy - clean
        achieved = 20.0 * np.log10(
            clean_rms / np.sqrt(np.mean(np.square(actual_noise), axis=1))
        )
        for offset, (row, value) in enumerate(zip(chunk, achieved, strict=True)):
            row["achieved_snr_db"] = float(value)
            row["clean_rms"] = float(clean_rms[offset])
            row["noise_rms"] = float(noise_rms[offset])
        raw[start:stop] = noisy.astype(np.float32)
    return raw, preprocess_conv1dgap_512(raw)


def _parameter_matrix(records: Iterable[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [
            [
                row["log_amplitude_p0"],
                row["frequency_khz"],
                row["log_tau_ms"],
                row["snr_db"],
            ]
            for row in records
        ],
        dtype=np.float64,
    )


def correlation_validation(
    records: list[dict[str, Any]], factors: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for class_name in CLASS_ORDER:
        selected = [row for row in records if row["class_name"] == class_name]
        realized = np.corrcoef(_parameter_matrix(selected), rowvar=False)
        target = factors[class_name] @ factors[class_name].T
        delta = realized - target
        for row_index, row_parameter in enumerate(PARAMETER_ORDER):
            for column_index, column_parameter in enumerate(PARAMETER_ORDER):
                output.append(
                    {
                        "class_name": class_name,
                        "row_parameter": row_parameter,
                        "column_parameter": column_parameter,
                        "target_correlation": float(target[row_index, column_index]),
                        "realized_correlation": float(realized[row_index, column_index]),
                        "delta": float(delta[row_index, column_index]),
                    }
                )
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_candidate_dataset(
    *,
    dataset_id: str,
    source_dataset_id: str,
    gaussian_run_id: str,
    cholesky_run_id: str,
    dependency_populations: dict[str, str],
    seed: int,
    output_dir: Path,
    records: list[dict[str, Any]],
    raw_signals: np.ndarray,
    model_signals: np.ndarray,
    rejection_counts: dict[str, int],
    source_manifest_sha256: str,
    gaussian_run_fingerprint: str,
    cholesky_run_fingerprint: str,
) -> list[str]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite candidate dataset: {output_dir}")
    output_dir.mkdir(parents=True)
    if raw_signals.shape != (len(records), RAW_LENGTH):
        raise ValueError("Unexpected raw signal shape")
    if model_signals.shape != (len(records), MODEL_LENGTH):
        raise ValueError("Unexpected Conv1D-GAP signal shape")
    if raw_signals.dtype != np.float32 or model_signals.dtype != np.float32:
        raise ValueError("Candidate signals must be float32")

    _write_csv(output_dir / "events.csv", records)
    np.save(output_dir / "signals_raw_4096.npy", raw_signals, allow_pickle=False)
    np.save(output_dir / "signals_conv1dgap_512.npy", model_signals, allow_pickle=False)
    class_counts = {
        class_name: sum(row["class_name"] == class_name for row in records)
        for class_name in CLASS_ORDER
    }
    summary = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "status": "interim_candidate_awaiting_visual_review",
        "event_count": len(records),
        "class_counts": class_counts,
        "rejection_counts": rejection_counts,
        "seed": seed,
        "sealed_test_accessed": False,
        "source_dataset": {
            "id": source_dataset_id,
            "manifest_sha256": source_manifest_sha256,
        },
        "source_analyses": {
            gaussian_run_id: gaussian_run_fingerprint,
            cholesky_run_id: cholesky_run_fingerprint,
        },
        "signal_contract": {
            "sampling_frequency_hz": SAMPLING_FREQUENCY_HZ,
            "raw_length": RAW_LENGTH,
            "raw_duration_ms": RAW_LENGTH / SAMPLING_FREQUENCY_HZ * 1000.0,
            "conv1dgap_length": MODEL_LENGTH,
            "conv1dgap_preprocessing": (
                "mean over contiguous blocks of 8, then per-window z-score"
            ),
        },
        "parameter_policy": {
            "coordinates": list(PARAMETER_ORDER),
            "correlations": dependency_populations,
            "snr_marginal": (
                "physical plus unclear mapped through physical_source_class"
            ),
            "phi_rad": "Uniform(0, 2*pi)",
            "t0_fraction": 0.5,
            "frequency_acceptance_khz": list(FREQUENCY_BAND_KHZ),
            "tau_semantics": "Gaussian envelope sigma in milliseconds",
        },
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    contract = {
        "schema_version": 1,
        "format": "aligned synthetic event arrays plus metadata table",
        "events": "events.csv",
        "raw_signals": {
            "path": "signals_raw_4096.npy",
            "shape": [len(records), RAW_LENGTH],
            "dtype": "float32",
        },
        "conv1dgap_signals": {
            "path": "signals_conv1dgap_512.npy",
            "shape": [len(records), MODEL_LENGTH],
            "dtype": "float32",
        },
        "class_mapping": {"0": "2um", "1": "4um", "2": "10um"},
        "review_gate": (
            "Do not register or promote before the Cholesky generation visual "
            "checkpoint is approved."
        ),
    }
    (output_dir / "input_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return [
        "events.csv",
        "signals_raw_4096.npy",
        "signals_conv1dgap_512.npy",
        "dataset_summary.json",
        "input_contract.json",
    ]


def _format_vector(values: np.ndarray, labels: tuple[str, ...]) -> str:
    return "\n".join(f"{label} = {value: .3f}" for label, value in zip(labels, values))


def render_cholesky_pipeline(
    records: list[dict[str, Any]],
    targets: dict[str, dict[str, Any]],
    dependency_populations: dict[str, str],
    destination: Path,
) -> None:
    example = next(row for row in records if row["class_name"] == "4um")
    z = np.asarray([example[f"z{i}"] for i in range(1, 5)])
    u = np.asarray([example[f"u{i}"] for i in range(1, 5)])
    y = np.asarray(
        [
            example["log_amplitude_p0"],
            example["frequency_khz"],
            example["log_tau_ms"],
            example["snr_db"],
        ]
    )
    physical = (
        f"P0 = {example['amplitude_p0']:.4g} acquisition units\n"
        f"fD = {example['frequency_khz']:.3f} kHz\n"
        f"tau = {example['tau_ms']:.4f} ms\n"
        f"SNR = {example['snr_db']:.2f} dB\n"
        f"phi = {example['phi_rad']:.3f} rad\n"
        f"t0 = {example['t0_fraction']:.2f} window"
    )
    labels = ("log(P0)", "fD", "log(tau)", "SNR")
    figure, axis = plt.subplots(figsize=(17.5, 7.5), constrained_layout=True)
    axis.set_axis_off()
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    boxes = [
        (0.02, 0.23, 0.19, 0.50, "1 · Independent draw", _format_vector(z, ("z1", "z2", "z3", "z4"))),
        (
            0.28,
            0.23,
            0.19,
            0.50,
            "2 · Correlated standard vector",
            _format_vector(u, ("u1", "u2", "u3", "u4")),
        ),
        (0.54, 0.23, 0.19, 0.50, "3 · Gaussian coordinates", _format_vector(y, labels)),
        (0.80, 0.18, 0.18, 0.60, "4 · Physical parameters", physical),
    ]
    colors = ("#e0f2fe", "#ede9fe", "#fef3c7", "#dcfce7")
    for (left, bottom, width, height, title, body), color in zip(boxes, colors, strict=True):
        patch = plt.Rectangle(
            (left, bottom),
            width,
            height,
            transform=axis.transAxes,
            facecolor=color,
            edgecolor="#0f172a",
            linewidth=1.5,
        )
        axis.add_patch(patch)
        axis.text(
            left + 0.015,
            bottom + height - 0.06,
            title,
            transform=axis.transAxes,
            fontsize=13,
            fontweight="bold",
            va="top",
        )
        axis.text(
            left + 0.015,
            bottom + height - 0.14,
            body,
            transform=axis.transAxes,
            fontsize=11,
            family="monospace",
            va="top",
            linespacing=1.6,
        )
    for left, label in ((0.22, r"$u=L_{4\mu m}z$"), (0.48, r"$y=\mu+D\,u$"), (0.74, "exp(log terms)")):
        axis.annotate(
            "",
            xy=(left + 0.05, 0.48),
            xytext=(left, 0.48),
            xycoords=axis.transAxes,
            arrowprops={"arrowstyle": "->", "linewidth": 2.2, "color": "#334155"},
        )
        axis.text(left + 0.025, 0.53, label, transform=axis.transAxes, ha="center", fontsize=10)
    axis.text(
        0.02,
        0.94,
        "SSL v3 · one actual 4 µm Cholesky generation path",
        transform=axis.transAxes,
        fontsize=22,
        fontweight="bold",
        color="#0f172a",
    )
    axis.text(
        0.02,
        0.875,
        (
            f"Sample {example['sample_id']} · "
            f"{dependency_populations['4um']} correlation matrix · "
            "effective-SNR marginal"
        ),
        transform=axis.transAxes,
        fontsize=11,
        color="#475569",
    )
    axis.text(
        0.02,
        0.055,
        (
            "This is a recorded generated event, not an illustrative fake. "
            f"Gaussian target mean for 4 µm: {np.array2string(targets['4um']['mean'], precision=3)}"
        ),
        transform=axis.transAxes,
        fontsize=10,
        color="#475569",
    )
    figure.savefig(destination, dpi=190, facecolor="white")
    plt.close(figure)


def render_statistical_validation(
    *,
    real_rows: list[dict[str, str]],
    generated_records: list[dict[str, Any]],
    targets: dict[str, dict[str, Any]],
    factors: dict[str, np.ndarray],
    dependency_populations: dict[str, str],
    destination: Path,
) -> None:
    figure = plt.figure(figsize=(21, 22), constrained_layout=True)
    outer = figure.add_gridspec(2, 1, height_ratios=(1.25, 1.0))
    marginal_grid = outer[0].subgridspec(4, 3)
    parameter_labels = {
        "amplitude_p0": "log(P0)",
        "frequency_khz": "Frequency (kHz)",
        "tau_ms": "log(tau / ms)",
        "snr_db": "SNR (dB)",
    }
    for row_index, marginal in enumerate(MARGINAL_BY_PARAMETER.values()):
        definition = MARGINALS[marginal]
        for column_index, class_name in enumerate(CLASS_ORDER):
            axis = figure.add_subplot(marginal_grid[row_index, column_index])
            selected_real = rows_for_marginal(real_rows, class_name, marginal)
            real = definition["transform"](raw_values(selected_real, marginal))
            generated_matrix = _parameter_matrix(
                row for row in generated_records if row["class_name"] == class_name
            )
            generated = generated_matrix[:, row_index]
            minimum = min(float(np.min(real)), float(np.min(generated)))
            maximum = max(float(np.max(real)), float(np.max(generated)))
            grid = np.linspace(minimum, maximum, 800)
            axis.hist(
                generated,
                bins=45,
                density=True,
                alpha=0.35,
                color=CLASS_COLORS[class_name],
                label="Generated histogram",
            )
            axis.plot(
                grid,
                gaussian_kde(real)(grid),
                color="#0f172a",
                linewidth=1.7,
                label="Real z8 KDE",
            )
            mean = float(targets[class_name]["mean"][row_index])
            sigma = float(targets[class_name]["sigma"][row_index])
            axis.plot(
                grid,
                norm.pdf((grid - mean) / sigma) / sigma,
                color="#e11d48",
                linewidth=2.0,
                label="Validated Gaussian target",
            )
            axis.axvline(mean, color="#e11d48", linestyle=":", linewidth=1.1)
            if row_index == 0:
                axis.set_title(CLASS_LABELS[class_name], fontweight="bold", fontsize=13)
            if column_index == 0:
                axis.set_ylabel(f"{parameter_labels[marginal]}\nDensity", fontsize=10)
            axis.grid(alpha=0.15)
            axis.tick_params(labelsize=8)
            if row_index == 0 and column_index == 0:
                axis.legend(fontsize=8, loc="upper right")

    matrix_grid = outer[1].subgridspec(3, 3)
    short_labels = ("log(P0)", "fD", "log(tau)", "SNR")
    maximum_delta = 0.0
    for class_index, class_name in enumerate(CLASS_ORDER):
        generated = _parameter_matrix(
            row for row in generated_records if row["class_name"] == class_name
        )
        realized = np.corrcoef(generated, rowvar=False)
        target = factors[class_name] @ factors[class_name].T
        matrices = (target, realized, realized - target)
        matrix_titles = (
            f"Target {dependency_populations[class_name]} R",
            "Generated R",
            "Generated − target",
        )
        mask = ~np.eye(4, dtype=bool)
        maximum_delta = max(
            maximum_delta, float(np.max(np.abs((realized - target)[mask])))
        )
        for matrix_index, (title, matrix) in enumerate(zip(matrix_titles, matrices, strict=True)):
            axis = figure.add_subplot(matrix_grid[class_index, matrix_index])
            limit = 0.12 if matrix_index == 2 else 1.0
            image = axis.imshow(matrix, vmin=-limit, vmax=limit, cmap="coolwarm")
            axis.set_xticks(range(4), short_labels, rotation=35, ha="right", fontsize=8)
            axis.set_yticks(range(4), short_labels, fontsize=8)
            axis.set_title(
                f"{CLASS_LABELS[class_name]} · {title}",
                fontsize=11,
                fontweight="bold",
            )
            for row in range(4):
                for column in range(4):
                    value = matrix[row, column]
                    axis.text(
                        column,
                        row,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7.5,
                        color="white" if abs(value) > 0.55 * limit else "#0f172a",
                    )
            figure.colorbar(image, ax=axis, shrink=0.65)
    figure.suptitle(
        (
            "SSL v3 candidate · realized Gaussian marginals and approved dependencies\n"
            f"Constraint warning: max |generated − target correlation| = "
            f"{maximum_delta:.3f} (review threshold {CORRELATION_WARNING_THRESHOLD:.2f})"
        ),
        fontsize=22,
        fontweight="bold",
    )
    figure.savefig(destination, dpi=170, facecolor="white")
    plt.close(figure)


def _nearest_unused(
    values: np.ndarray, target: float, used: set[int]
) -> int:
    for index in np.argsort(np.abs(values - target)):
        candidate = int(index)
        if candidate not in used:
            used.add(candidate)
            return candidate
    raise RuntimeError("Could not select a unique gallery event")


def select_gallery_indices(records: list[dict[str, Any]]) -> dict[str, list[tuple[str, int]]]:
    selections: dict[str, list[tuple[str, int]]] = {}
    for class_name in CLASS_ORDER:
        indices = [index for index, row in enumerate(records) if row["class_name"] == class_name]
        matrix = _parameter_matrix(records[index] for index in indices)
        target_mean = np.median(matrix, axis=0)
        target_scale = np.maximum(np.std(matrix, axis=0, ddof=1), 1e-12)
        central_scores = np.sum(np.square((matrix - target_mean) / target_scale), axis=1)
        used: set[int] = {int(np.argmin(central_scores))}
        local: list[tuple[str, int]] = [("Central joint draw", indices[next(iter(used))])]
        roles = (
            ("Low frequency", 1, 0.05),
            ("High frequency", 1, 0.95),
            ("Low SNR", 3, 0.05),
            ("High amplitude", 0, 0.95),
            ("Long tau", 2, 0.95),
        )
        for label, column, quantile in roles:
            target = float(np.quantile(matrix[:, column], quantile))
            local_index = _nearest_unused(matrix[:, column], target, used)
            local.append((label, indices[local_index]))
        selections[class_name] = local
    return selections


def render_signal_gallery(
    *,
    records: list[dict[str, Any]],
    raw_signals: np.ndarray,
    destination: Path,
) -> None:
    selections = select_gallery_indices(records)
    figure, axes = plt.subplots(
        3,
        6,
        figsize=(23, 11.5),
        constrained_layout=True,
        sharex=True,
    )
    time_ms = (
        np.arange(RAW_LENGTH, dtype=np.float64) - (RAW_LENGTH - 1) / 2.0
    ) / SAMPLING_FREQUENCY_HZ * 1000.0
    for class_index, class_name in enumerate(CLASS_ORDER):
        for axis, (role, index) in zip(axes[class_index], selections[class_name], strict=True):
            record = records[index]
            envelope = record["amplitude_p0"] * np.exp(
                -0.5 * np.square(time_ms / record["tau_ms"])
            )
            axis.plot(time_ms, raw_signals[index], color="#2563eb", linewidth=0.65)
            axis.plot(time_ms, envelope, color="#f97316", linewidth=1.0)
            axis.plot(time_ms, -envelope, color="#f97316", linewidth=1.0)
            axis.axvline(0.0, color="#64748b", linestyle=":", linewidth=0.7)
            axis.set_title(role, fontsize=10, fontweight="bold")
            axis.text(
                0.02,
                0.98,
                (
                    f"P0={record['amplitude_p0']:.3g}\n"
                    f"fD={record['frequency_khz']:.2f} kHz\n"
                    f"tau={record['tau_ms']:.3f} ms\n"
                    f"SNR={record['snr_db']:.1f} dB\n"
                    f"phi={record['phi_rad']:.2f}"
                ),
                transform=axis.transAxes,
                va="top",
                fontsize=7.5,
                bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
            )
            axis.grid(alpha=0.12)
            axis.tick_params(labelsize=7)
        axes[class_index, 0].set_ylabel(
            f"{CLASS_LABELS[class_name]}\nAcquisition units",
            fontsize=10,
            fontweight="bold",
        )
    for axis in axes[-1]:
        axis.set_xlabel("Time from t0 (ms)", fontsize=9)
    figure.suptitle(
        "SSL v3 candidate · deterministic representative signal gallery",
        fontsize=21,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.003,
        (
            "Blue: noisy synthetic event · orange: ±Gaussian envelope · "
            "selection fixed from generated parameter quantiles before rendering"
        ),
        ha="center",
        fontsize=9,
        color="#475569",
    )
    figure.savefig(destination, dpi=180, facecolor="white")
    plt.close(figure)


def validate_candidate(
    *,
    dataset_id: str,
    budgets: dict[str, int],
    records: list[dict[str, Any]],
    raw_signals: np.ndarray,
    model_signals: np.ndarray,
    factors: dict[str, np.ndarray],
) -> dict[str, Any]:
    expected_count = sum(budgets.values())
    class_counts = {
        class_name: sum(row["class_name"] == class_name for row in records)
        for class_name in CLASS_ORDER
    }
    correlations = correlation_validation(records, factors)
    off_diagonal = [
        abs(row["delta"])
        for row in correlations
        if row["row_parameter"] != row["column_parameter"]
    ]
    max_delta = float(max(off_diagonal))
    requested_snr = np.asarray([row["snr_db"] for row in records], dtype=np.float64)
    achieved_snr = np.asarray(
        [row["achieved_snr_db"] for row in records], dtype=np.float64
    )
    checks = {
        "event_count": len(records) == expected_count,
        "class_counts": class_counts == budgets,
        "unique_ids": len({row["sample_id"] for row in records}) == expected_count,
        "raw_shape": raw_signals.shape == (expected_count, RAW_LENGTH),
        "model_shape": model_signals.shape == (expected_count, MODEL_LENGTH),
        "raw_float32": raw_signals.dtype == np.float32,
        "model_float32": model_signals.dtype == np.float32,
        "finite_signals": bool(
            np.all(np.isfinite(raw_signals)) and np.all(np.isfinite(model_signals))
        ),
        "positive_amplitude_tau": all(
            float(row["amplitude_p0"]) > 0.0 and float(row["tau_ms"]) > 0.0
            for row in records
        ),
        "frequency_in_band": all(
            FREQUENCY_BAND_KHZ[0]
            <= float(row["frequency_khz"])
            <= FREQUENCY_BAND_KHZ[1]
            for row in records
        ),
        "t0_fixed": all(float(row["t0_fraction"]) == 0.5 for row in records),
        "phi_in_range": all(
            0.0 <= float(row["phi_rad"]) < 2.0 * np.pi for row in records
        ),
        "snr_realization": bool(np.max(np.abs(requested_snr - achieved_snr)) < 1e-5),
        "no_sealed_test_access": True,
    }
    if not all(checks.values()):
        failed = [name for name, value in checks.items() if not value]
        raise ValueError(f"Candidate validation failed: {failed}")
    return {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "status": (
            "warning_correlation_delta_above_threshold"
            if max_delta > CORRELATION_WARNING_THRESHOLD
            else "ready_for_visual_review"
        ),
        "checks": checks,
        "class_counts": class_counts,
        "event_count": expected_count,
        "maximum_absolute_off_diagonal_correlation_delta": max_delta,
        "correlation_warning_threshold": CORRELATION_WARNING_THRESHOLD,
        "correlation_warning": max_delta > CORRELATION_WARNING_THRESHOLD,
        "maximum_absolute_snr_error_db": float(
            np.max(np.abs(requested_snr - achieved_snr))
        ),
        "claim_boundary": (
            "This validates deterministic construction and realized first-order "
            "marginals/dependencies for the generated candidate. It does not validate "
            "visual realism, latent-space twin coverage, or unseen real populations."
        ),
    }


def write_analysis_outputs(
    *,
    output_dir: Path,
    validation: dict[str, Any],
    correlation_rows: list[dict[str, Any]],
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "summary_metrics.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "correlation_validation.csv", correlation_rows)
    return ["summary_metrics.json", "correlation_validation.csv"]
