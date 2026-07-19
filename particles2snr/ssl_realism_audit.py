from __future__ import annotations

import csv
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from matplotlib.figure import Figure
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, hilbert, spectrogram

from .yeast_representation_dataset import clamped_crop, preprocess_crop


FS = 1_000_000.0
RAW_FS = 2_000_000.0
FEATURE_NAMES = (
    "duration_25_ms",
    "envelope_concentration",
    "dominant_frequency_khz",
    "spectral_bandwidth_khz",
    "temporal_peak_count",
    "spectral_peak_count",
    "amplitude_percentile",
)
FEATURE_LABELS = {
    "duration_25_ms": "Durée enveloppe à 25 %",
    "envelope_concentration": "Concentration durée 50/25 %",
    "dominant_frequency_khz": "Fréquence dominante",
    "spectral_bandwidth_khz": "Largeur spectrale",
    "temporal_peak_count": "Pics temporels",
    "spectral_peak_count": "Pics fréquentiels",
    "amplitude_percentile": "Percentile d’amplitude",
}
FEATURE_UNITS = {
    "duration_25_ms": "ms",
    "envelope_concentration": "ratio",
    "dominant_frequency_khz": "kHz",
    "spectral_bandwidth_khz": "kHz",
    "temporal_peak_count": "",
    "spectral_peak_count": "",
    "amplitude_percentile": "%",
}
MATCH_WEIGHTS = {
    "duration_25_ms": 0.20,
    "envelope_concentration": 0.10,
    "dominant_frequency_khz": 0.20,
    "spectral_bandwidth_khz": 0.15,
    "temporal_peak_count": 0.075,
    "spectral_peak_count": 0.075,
    "amplitude_percentile": 0.20,
}
PHYSICAL_CORE_WEIGHTS = {
    "duration_25_ms": 0.30,
    "envelope_concentration": 0.05,
    "dominant_frequency_khz": 0.30,
    "spectral_bandwidth_khz": 0.15,
    "temporal_peak_count": 0.10,
    "spectral_peak_count": 0.05,
    "amplitude_percentile": 0.05,
}
KNOWN_CLASSES = {"0": "2um", "1": "4um", "2": "10um"}


@dataclass
class SignalRecord:
    identifier: str
    signal: np.ndarray
    metadata: dict[str, Any]
    descriptors: dict[str, float]


def _support_bounds(
    envelope: np.ndarray,
    fraction: float,
    *,
    peak_index: int | None = None,
) -> tuple[int, int]:
    peak = int(np.argmax(envelope)) if peak_index is None else int(peak_index)
    threshold = float(envelope[peak]) * fraction
    left = peak
    while left > 0 and envelope[left - 1] >= threshold:
        left -= 1
    right = peak + 1
    while right < len(envelope) and envelope[right] >= threshold:
        right += 1
    return left, right


def signal_descriptors(signal: np.ndarray, *, sampling_frequency_hz: float = FS) -> dict[str, float]:
    values = np.asarray(signal, dtype=np.float64).squeeze()
    if values.ndim != 1 or values.size < 512:
        raise ValueError(f"Expected a one-dimensional signal of at least 512 samples, got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("Signal contains non-finite values")
    centered = values - float(np.mean(values))
    rms = float(np.sqrt(np.mean(np.square(centered))))
    if rms <= 1.0e-12:
        raise ValueError("Signal RMS is zero")

    envelope = uniform_filter1d(np.abs(hilbert(centered)), size=64, mode="nearest")
    edge_guard = min(256, max(1, values.size // 10))
    interior = envelope[edge_guard : values.size - edge_guard]
    peak_index = edge_guard + int(np.argmax(interior))
    left25, right25 = _support_bounds(envelope, 0.25, peak_index=peak_index)
    left50, right50 = _support_bounds(envelope, 0.50, peak_index=peak_index)
    duration25 = (right25 - left25) / sampling_frequency_hz * 1000.0
    duration50 = (right50 - left50) / sampling_frequency_hz * 1000.0
    support_envelope = envelope[left25:right25]
    prominence = max(
        0.15 * (float(np.max(support_envelope)) - float(np.median(support_envelope))),
        1.0e-12,
    )
    temporal_peaks, _ = find_peaks(
        support_envelope,
        prominence=prominence,
        distance=max(1, int(round(0.08e-3 * sampling_frequency_hz))),
    )

    windowed = centered * np.hanning(values.size)
    frequencies = np.fft.rfftfreq(values.size, d=1.0 / sampling_frequency_hz)
    power = np.square(np.abs(np.fft.rfft(windowed)))
    mask = (frequencies >= 5_000.0) & (frequencies <= 100_000.0)
    band_frequency = frequencies[mask]
    band_power = uniform_filter1d(power[mask], size=5, mode="nearest")
    total_power = float(np.sum(band_power))
    if total_power <= 1.0e-18:
        raise ValueError("Signal has no measurable 5–100 kHz power")
    dominant = float(band_frequency[int(np.argmax(band_power))])
    centroid = float(np.sum(band_frequency * band_power) / total_power)
    bandwidth = float(
        np.sqrt(np.sum(np.square(band_frequency - centroid) * band_power) / total_power)
    )
    frequency_step = float(band_frequency[1] - band_frequency[0])
    spectral_peaks, _ = find_peaks(
        band_power,
        prominence=max(0.10 * float(np.max(band_power)), 1.0e-18),
        distance=max(1, int(round(2_000.0 / frequency_step))),
    )
    return {
        "duration_25_ms": float(duration25),
        "envelope_concentration": float(duration50 / max(duration25, 1.0e-12)),
        "dominant_frequency_khz": dominant / 1000.0,
        "spectral_bandwidth_khz": bandwidth / 1000.0,
        "temporal_peak_count": float(len(temporal_peaks)),
        "spectral_peak_count": float(len(spectral_peaks)),
        "rms": rms,
        "envelope_peak": float(np.max(envelope)),
        "peak_to_peak": float(np.ptp(centered)),
        "support_start_index": float(left25),
        "support_end_index": float(right25),
    }


def percentile_ranks(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("Percentile ranks require a non-empty vector")
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    ranks[order] = (np.arange(array.size, dtype=np.float64) + 0.5) / array.size * 100.0
    return ranks


def _feature_scale(records: list[SignalRecord], name: str) -> float:
    if name == "amplitude_percentile":
        return 25.0
    values = np.asarray([row.descriptors[name] for row in records], dtype=np.float64)
    scale = float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
    if name in {"temporal_peak_count", "spectral_peak_count"}:
        return max(scale, 1.0)
    return max(scale, 1.0e-6)


def attach_amplitude_percentiles(records: list[SignalRecord]) -> None:
    ranks = percentile_ranks(row.descriptors["rms"] for row in records)
    for row, rank in zip(records, ranks, strict=True):
        row.descriptors["amplitude_percentile"] = float(rank)


def select_real_representative(records: list[SignalRecord]) -> SignalRecord:
    if not records:
        raise ValueError("Cannot select a representative from an empty population")
    for row in records:
        if "amplitude_percentile" not in row.descriptors:
            raise ValueError("Amplitude percentiles must be attached before selection")
    medians = {
        name: float(np.median([row.descriptors[name] for row in records]))
        for name in FEATURE_NAMES
    }
    scales = {name: _feature_scale(records, name) for name in FEATURE_NAMES}

    def score(row: SignalRecord) -> tuple[float, str]:
        distance = sum(
            MATCH_WEIGHTS[name]
            * np.square((row.descriptors[name] - medians[name]) / scales[name])
            for name in FEATURE_NAMES
        )
        row.metadata["representative_distance"] = float(np.sqrt(distance))
        return float(distance), row.identifier

    return min(records, key=score)


def match_simulation(
    real: SignalRecord,
    simulations: list[SignalRecord],
    *,
    weights: dict[str, float] | None = None,
) -> tuple[SignalRecord, float]:
    if not simulations:
        raise ValueError("Simulation population is empty")
    active_weights = MATCH_WEIGHTS if weights is None else weights
    if set(active_weights) != set(FEATURE_NAMES):
        raise ValueError("Weights must cover the complete feature contract")
    if not np.isclose(sum(active_weights.values()), 1.0):
        raise ValueError("Weights must sum to one")
    scales = {name: _feature_scale(simulations, name) for name in FEATURE_NAMES}

    def distance(row: SignalRecord) -> tuple[float, str]:
        value = sum(
            active_weights[name]
            * np.square(
                (row.descriptors[name] - real.descriptors[name]) / scales[name]
            )
            for name in FEATURE_NAMES
        )
        return float(np.sqrt(value)), row.identifier

    best = min(simulations, key=distance)
    return best, distance(best)[0]


def best_cross_population_pair(
    reals: list[SignalRecord],
    simulations: list[SignalRecord],
    *,
    weights: dict[str, float],
) -> tuple[SignalRecord, SignalRecord, float]:
    if not reals:
        raise ValueError("Real population is empty")
    candidates = [
        (distance, real.identifier, matched.identifier, real, matched)
        for real in reals
        for matched, distance in [match_simulation(real, simulations, weights=weights)]
    ]
    distance, _real_id, _simulation_id, real, matched = min(candidates)
    return real, matched, float(distance)


def load_particle_population(
    dataset_root: Path,
    *,
    split: str = "train",
) -> dict[str, list[SignalRecord]]:
    repair_path = dataset_root / "saturation_repair_manifest.csv"
    with repair_path.open(newline="", encoding="utf-8") as handle:
        repaired = {
            (row["final_split"], row["filename"]) for row in csv.DictReader(handle)
        }
    populations: dict[str, list[SignalRecord]] = {
        class_name: [] for class_name in KNOWN_CLASSES.values()
    }
    label_root = dataset_root / split / "labels"
    signal_root = dataset_root / split / "signals"
    for label_path in sorted(label_root.glob("*.txt")):
        lines = [line.split() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(lines) != 1 or len(lines[0]) < 3:
            continue
        class_id, center_text, width_text = lines[0][:3]
        if class_id not in KNOWN_CLASSES:
            continue
        center = float(center_text)
        width = float(width_text)
        signal_name = f"{label_path.stem}.npy"
        if not 0.25 <= center <= 0.75:
            continue
        if (split, signal_name) in repaired:
            continue
        source = np.load(signal_root / signal_name, allow_pickle=False)
        center_index = int(round(center * len(source)))
        crop, crop_start = clamped_crop(source, center_index, 8192)
        processed = preprocess_crop(crop)
        descriptors = signal_descriptors(processed)
        record = SignalRecord(
            identifier=f"{split}:{label_path.stem}:0",
            signal=processed,
            metadata={
                "split": split,
                "filename": signal_name,
                "class_id": int(class_id),
                "class_name": KNOWN_CLASSES[class_id],
                "center_norm": center,
                "width_norm": width,
                "crop_start_raw": crop_start,
                "event_start_index": (
                    center * len(source) - width * len(source) / 2.0 - crop_start
                )
                / 2.0,
                "event_end_index": (
                    center * len(source) + width * len(source) / 2.0 - crop_start
                )
                / 2.0,
            },
            descriptors=descriptors,
        )
        populations[KNOWN_CLASSES[class_id]].append(record)
    for rows in populations.values():
        attach_amplitude_percentiles(rows)
    return populations


def load_ssl_simulation_population(
    dataset_root: Path,
    *,
    component_count: int = 1,
    split: str = "train",
) -> list[SignalRecord]:
    if component_count not in {1, 2}:
        raise ValueError("component_count must be 1 or 2")
    signals = np.load(dataset_root / "signals.npy", mmap_mode="r")
    records: list[SignalRecord] = []
    with (dataset_root / "simulation_metadata.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        for row in csv.DictReader(handle):
            if (
                row["split"] != split
                or int(row["component_count"]) != component_count
            ):
                continue
            signal_row = int(row["signal_row"])
            values = np.asarray(signals[signal_row], dtype=np.float32)
            records.append(
                SignalRecord(
                    identifier=f"{row['latent_id']}:view-{row['view_index']}",
                    signal=values,
                    metadata={
                        **row,
                        "signal_row": signal_row,
                        "component_count": component_count,
                    },
                    descriptors=signal_descriptors(values),
                )
            )
    attach_amplitude_percentiles(records)
    return records


def load_budding_population(dataset_root: Path) -> list[SignalRecord]:
    signals = np.load(dataset_root / "signals.npy", mmap_mode="r")
    records: list[SignalRecord] = []
    with (dataset_root / "events.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row["development_split"] != "development_train"
                or row["source_group"] != "budding"
                or row["quality"] != "strict"
            ):
                continue
            signal_row = int(row["signal_row"])
            values = np.asarray(signals[signal_row], dtype=np.float32)
            records.append(
                SignalRecord(
                    identifier=row["event_id"],
                    signal=values,
                    metadata={
                        **row,
                        "signal_row": signal_row,
                        "event_start_index": float(row["event_start_input_index"]),
                        "event_end_index": float(row["event_end_input_index"]),
                        "n_doppler_peaks": int(row["n_doppler_peaks"]),
                    },
                    descriptors=signal_descriptors(values),
                )
            )
    if not records:
        raise ValueError("No strict development-train budding events found")
    attach_amplitude_percentiles(records)
    return records


def _display_signal(signal: np.ndarray) -> np.ndarray:
    centered = np.asarray(signal, dtype=np.float64) - float(np.mean(signal))
    rms = float(np.sqrt(np.mean(np.square(centered))))
    return centered / max(rms, 1.0e-12)


def render_pair_figure(case: dict[str, Any], destination: Path) -> None:
    sim = np.asarray(case["_simulation_signal"], dtype=np.float64)
    real = np.asarray(case["_real_signal"], dtype=np.float64)
    sim_display = _display_signal(sim)
    real_display = _display_signal(real)
    time_ms = np.arange(sim.size) / FS * 1000.0
    y_limit = max(float(np.max(np.abs(sim_display))), float(np.max(np.abs(real_display)))) * 1.04
    sim_bounds = (
        int(case["simulation"]["descriptors"]["support_start_index"]),
        int(case["simulation"]["descriptors"]["support_end_index"]),
    )
    real_bounds = (
        float(case["real"]["metadata"]["event_start_index"]),
        float(case["real"]["metadata"]["event_end_index"]),
    )

    figure = Figure(figsize=(13.0, 8.7), constrained_layout=True)
    grid = figure.add_gridspec(4, 1, height_ratios=[1.0, 1.0, 0.82, 0.82])
    trace_axes = [figure.add_subplot(grid[0]), figure.add_subplot(grid[1])]
    colors = ("#176f5b", "#2c6386")
    labels = ("Simulation SSL la plus proche", f"Signal réel {case['class_label']}")
    displays = (sim_display, real_display)
    bounds = (
        (sim_bounds[0] / FS * 1000.0, sim_bounds[1] / FS * 1000.0),
        (real_bounds[0] / FS * 1000.0, real_bounds[1] / FS * 1000.0),
    )
    for axis, values, color, label, (left, right) in zip(
        trace_axes, displays, colors, labels, bounds, strict=True
    ):
        axis.plot(time_ms, values, color=color, linewidth=0.8)
        axis.axvspan(left, right, color="#d88b45", alpha=0.16, label="support/annotation")
        axis.axhline(0.0, color="#9a958b", linewidth=0.6)
        axis.set_ylim(-y_limit, y_limit)
        axis.set_ylabel("Amplitude / RMS")
        axis.set_title(label, loc="left", fontsize=11, fontweight="bold")
        axis.grid(alpha=0.15)
    trace_axes[0].tick_params(labelbottom=False)
    trace_axes[1].set_xlabel("Temps (ms)")

    for index, (values, label) in enumerate(zip(displays, labels, strict=True), start=2):
        axis = figure.add_subplot(grid[index])
        frequencies, times, density = spectrogram(
            values,
            fs=FS,
            nperseg=256,
            noverlap=192,
            mode="psd",
        )
        mask = (frequencies >= 5_000.0) & (frequencies <= 100_000.0)
        db = 10.0 * np.log10(np.maximum(density[mask], 1.0e-16))
        db -= float(np.max(db))
        axis.pcolormesh(
            times * 1000.0,
            frequencies[mask] / 1000.0,
            db,
            shading="auto",
            cmap="magma",
            vmin=-55.0,
            vmax=0.0,
        )
        axis.set_ylabel("kHz")
        axis.set_title(f"Spectrogramme · {label} · dB relatifs", loc="left", fontsize=10)
        axis.set_ylim(5.0, 100.0)
        if index == 3:
            axis.set_xlabel("Temps (ms)")
        else:
            axis.tick_params(labelbottom=False)
    figure.suptitle(
        (
            f"{case['class_label']} · "
            f"{'réel représentatif' if case.get('selection_role', 'representative_baseline').startswith('representative') else 'couple optimisé'} "
            f"→ voisin SSL rang 1/{case['simulation_population']}"
        ),
        fontsize=15,
        fontweight="bold",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150, facecolor="#fffdf8")


def _public_record(record: SignalRecord) -> dict[str, Any]:
    return {
        "id": record.identifier,
        "metadata": record.metadata,
        "descriptors": record.descriptors,
    }


def build_particle_checkpoint_model(
    particle_root: Path,
    simulation_root: Path,
) -> dict[str, Any]:
    real_populations = load_particle_population(particle_root)
    simulations = load_ssl_simulation_population(simulation_root)
    cases = []
    for class_name in ("2um", "4um", "10um"):
        real = select_real_representative(real_populations[class_name])
        matched, distance = match_simulation(real, simulations)
        metrics = []
        for name in FEATURE_NAMES:
            metrics.append(
                {
                    "name": name,
                    "label": FEATURE_LABELS[name],
                    "unit": FEATURE_UNITS[name],
                    "real": float(real.descriptors[name]),
                    "simulation": float(matched.descriptors[name]),
                    "delta": float(matched.descriptors[name] - real.descriptors[name]),
                    "weight": MATCH_WEIGHTS[name],
                }
            )
        cases.append(
            {
                "id": class_name.replace("um", "um-pair"),
                "class_name": class_name,
                "class_label": class_name.replace("um", " µm"),
                "real_population": len(real_populations[class_name]),
                "simulation_population": len(simulations),
                "match_distance": float(distance),
                "real": _public_record(real),
                "simulation": _public_record(matched),
                "metrics": metrics,
                "_real_signal": real.signal,
                "_simulation_signal": matched.signal,
            }
        )
    selected_ids = [
        identifier
        for case in cases
        for identifier in (case["real"]["id"], case["simulation"]["id"])
    ]
    return {
        "schema_version": 1,
        "milestone": "ssl-realism-audit-c1",
        "title": "Le pool SSL contient-il des voisins crédibles des particules réelles ?",
        "claim_type": "local diagnostic",
        "question": (
            "Pour un événement réel représentatif de chaque taille, le voisin exact "
            "du pool SSL monoparticule est-il visuellement assez proche pour rendre "
            "la source simulée crédible comme donnée d’entraînement ?"
        ),
        "datasets": {
            "real": "particles2snr-f-dual-clean-c1-yolo-4class-saturation-reviewed@v2",
            "simulation": "yeast-passage-simulations@v2",
        },
        "cases": cases,
        "feature_contract": {
            "weights": MATCH_WEIGHTS,
            "amplitude_policy": (
                "Percentile intra-population pour la distance ; RMS, pic d’enveloppe "
                "et crête-à-crête affichés dans les unités propres à chaque source."
            ),
            "distance_scaling": (
                "IQR du pool simulé train monoparticule ; 25 points de percentile "
                "pour l’amplitude."
            ),
            "excluded": [
                "SNR",
                "embedding appris",
                "corrélation point à point",
                "phase absolue",
                "position dans la fenêtre",
            ],
        },
        "display_contract": {
            "sampling_frequency_hz": FS,
            "samples": 4096,
            "duration_ms": 4.096,
            "trace_transform": "centrage puis division par le RMS de chaque signal",
            "trace_scale": "échelle verticale partagée dans chaque paire",
            "spectrogram": "5–100 kHz, dB relatifs au maximum de chaque signal, couleurs partagées [-55, 0] dB",
            "real_preprocessing": "crop 8192 à 2 MHz, passe-bande 5–100 kHz, resample_poly 2:1",
            "simulation_preprocessing": "signal 4096 stocké, aucune régénération ni transformation avant métriques",
        },
        "case_selection": {
            "population": (
                "101 événements particules train éligibles (36×2 µm, 55×4 µm, "
                "10×10 µm) et 7 020 signaux train component_count=1 de "
                "yeast-passage-simulations@v2"
            ),
            "rule": (
                "Dans chaque classe réelle, choisir indépendamment du simulateur "
                "l’événement de distance robuste minimale à la médiane de classe ; "
                "chercher ensuite l’argmin exact de la distance physique pondérée "
                "dans les 6 982 signaux SSL monoparticule."
            ),
            "selected_ids": selected_ids,
            "selected_before_rendering": True,
        },
        "claim_boundary": (
            "Ces trois paires sont des diagnostics locaux sur des ancres réelles "
            "représentatives. Elles auditent le meilleur voisin disponible, mais "
            "n’estiment ni la couverture globale du domaine réel, ni la performance "
            "du modèle SSL, ni le réalisme biologique."
        ),
    }


def _pair_case(
    *,
    case_id: str,
    class_name: str = "2um",
    class_label: str,
    selection_note: str,
    selection_role: str,
    real_source_label: str | None = None,
    real: SignalRecord,
    matched: SignalRecord,
    distance: float,
    weights: dict[str, float],
    real_population: int,
    simulation_population: int,
) -> dict[str, Any]:
    metrics = [
        {
            "name": name,
            "label": FEATURE_LABELS[name],
            "unit": FEATURE_UNITS[name],
            "real": float(real.descriptors[name]),
            "simulation": float(matched.descriptors[name]),
            "delta": float(matched.descriptors[name] - real.descriptors[name]),
            "weight": weights[name],
        }
        for name in FEATURE_NAMES
    ]
    return {
        "id": case_id,
        "class_name": class_name,
        "class_label": class_label,
        "selection_note": selection_note,
        "selection_role": selection_role,
        "real_source_label": real_source_label or (
            "Réel · sélection indépendante"
            if selection_role.startswith("representative")
            else "Réel · choisi par argmin global"
        ),
        "real_population": real_population,
        "simulation_population": simulation_population,
        "match_distance": float(distance),
        "real": _public_record(real),
        "simulation": _public_record(matched),
        "metrics": metrics,
        "_real_signal": real.signal,
        "_simulation_signal": matched.signal,
    }


def build_budding_checkpoint_model(
    budding_root: Path,
    simulation_root: Path,
) -> dict[str, Any]:
    real_population = load_budding_population(budding_root)
    double_doppler_population = [
        row for row in real_population if row.metadata["n_doppler_peaks"] >= 2
    ]
    if not double_doppler_population:
        raise ValueError("The budding pool contains no multi-Doppler event")
    simulations = load_ssl_simulation_population(
        simulation_root,
        component_count=2,
    )

    representative = select_real_representative(real_population)
    representative_simulation, representative_distance = match_simulation(
        representative,
        simulations,
        weights=MATCH_WEIGHTS,
    )
    double_doppler = select_real_representative(double_doppler_population)
    double_doppler_simulation, double_doppler_distance = match_simulation(
        double_doppler,
        simulations,
        weights=PHYSICAL_CORE_WEIGHTS,
    )
    favourable_real, favourable_simulation, favourable_distance = (
        best_cross_population_pair(
            real_population,
            simulations,
            weights=PHYSICAL_CORE_WEIGHTS,
        )
    )

    cases = [
        _pair_case(
            case_id="budding-representative",
            class_name="budding",
            class_label="Budding · ancre représentative",
            selection_note=(
                "Événement strict choisi indépendamment du simulateur comme "
                "médoïde robuste des 855 traces budding de développement."
            ),
            selection_role="representative_budding",
            real_source_label="Budding réel · ancre indépendante",
            real=representative,
            matched=representative_simulation,
            distance=representative_distance,
            weights=MATCH_WEIGHTS,
            real_population=len(real_population),
            simulation_population=len(simulations),
        ),
        _pair_case(
            case_id="budding-double-doppler",
            class_name="budding",
            class_label="Budding · ancre multi-Doppler",
            selection_note=(
                "Événement représentatif du sous-pool prédéclaré des 8 traces "
                "strictes dont le détecteur décrit au moins deux pics Doppler."
            ),
            selection_role="representative_double_doppler",
            real_source_label="Budding réel · sous-pool multi-Doppler",
            real=double_doppler,
            matched=double_doppler_simulation,
            distance=double_doppler_distance,
            weights=PHYSICAL_CORE_WEIGHTS,
            real_population=len(double_doppler_population),
            simulation_population=len(simulations),
        ),
        _pair_case(
            case_id="budding-best-physical",
            class_name="budding",
            class_label="Budding · meilleur couple physique",
            selection_note=(
                "Argmin global parmi 855×3 018 couples, avec 30 % durée, "
                "30 % Doppler et 5 % amplitude. Cas favorable, non représentatif."
            ),
            selection_role="existence_biased_physical",
            real_source_label="Budding réel · choisi par argmin global",
            real=favourable_real,
            matched=favourable_simulation,
            distance=favourable_distance,
            weights=PHYSICAL_CORE_WEIGHTS,
            real_population=len(real_population),
            simulation_population=len(simulations),
        ),
    ]
    selected_ids = [
        identifier
        for case in cases
        for identifier in (case["real"]["id"], case["simulation"]["id"])
    ]
    return {
        "schema_version": 1,
        "milestone": "ssl-realism-audit-budding-c2",
        "title": "Budding — le modèle simulé à deux composantes produit-il des voisins crédibles ?",
        "claim_type": "local diagnostic / morphology-proxy audit",
        "question": (
            "Parmi les simulations SSL déjà générées avec exactement deux "
            "composantes, trouve-t-on des voisins visuellement compatibles avec "
            "des événements propres de la condition budding ?"
        ),
        "datasets": {
            "real": "yeast-events-representation@v3",
            "simulation": "yeast-passage-simulations@v2",
            "detector_method_reference": "yeast-detector-pipeline-board-m2-r10",
        },
        "cases": cases,
        "feature_contract": {
            "representative_weights": MATCH_WEIGHTS,
            "physical_weights": PHYSICAL_CORE_WEIGHTS,
            "amplitude_policy": (
                "Percentile intra-population ; RMS, pic d’enveloppe et "
                "crête-à-crête restent affichés séparément."
            ),
            "distance_scaling": (
                "IQR des 3 018 simulations train à deux composantes ; "
                "25 points de percentile pour l’amplitude."
            ),
            "excluded": [
                "SNR",
                "embedding appris",
                "corrélation point à point",
                "phase absolue",
                "position dans la fenêtre",
                "label morphologique individuel",
            ],
        },
        "display_contract": {
            "sampling_frequency_hz": FS,
            "samples": 4096,
            "duration_ms": 4.096,
            "trace_transform": "centrage puis division par le RMS de chaque signal",
            "trace_scale": "échelle verticale partagée dans chaque paire",
            "spectrogram": "5–100 kHz, dB relatifs au maximum de chaque signal, couleurs partagées [-55, 0] dB",
            "real_preprocessing": (
                "événement v3 déjà traité par le contrat 8192→4096 : "
                "passe-bande 5–100 kHz puis sous-échantillonnage 2:1"
            ),
            "simulation_preprocessing": (
                "signal v1 stocké, généré à 8192 points/2 MHz puis soumis au "
                "même prétraitement 8192→4096 ; aucune régénération pour cet audit"
            ),
        },
        "simulation_methodology": {
            "model": (
                "Somme de deux passages sinusoïdaux sous enveloppes gaussiennes : "
                "x(t)=e₁(t)cos(2πf₁t+φ₁)+r·e₂(t)cos(2πf₂t+φ₂)+bruit."
            ),
            "geometry": (
                "Modèle générique à deux passages, pas une géométrie "
                "sphère-bourgeon validée. Le second centre est décalé de "
                "±0,5×component_separation_ms et f₂ de "
                "±frequency_separation_khz."
            ),
            "factor_ranges": (
                "durée 0,464–1,424 ms ; Doppler 7,8125–23,4375 kHz ; "
                "séparation déclarée 0,08–0,70 ms ; amplitude relative "
                "0,40–1,00 ; séparation fréquentielle 0–8 kHz"
            ),
            "nuisance": (
                "phase et position aléatoires, bruit coloré à SNR 0–30 dB, "
                "dérive de base 0–0,30, réponse capteur 0,85–1,15, puis RMS "
                "cible 0,40–1,70"
            ),
        },
        "case_selection": {
            "population": (
                "855 événements budding stricts de development_train issus de "
                "yeast-events-representation@v3 (dont 8 multi-Doppler) et "
                "2 980 signaux train component_count=2 de "
                "yeast-passage-simulations@v2"
            ),
            "rule": (
                "Prédéclarer une ancre robuste sur tout le pool budding, une "
                "ancre robuste dans le sous-pool multi-Doppler, puis un argmin "
                "global physique. Pour chaque ancre, chercher uniquement dans "
                "les simulations à deux composantes déjà générées."
            ),
            "selected_ids": selected_ids,
            "selected_before_rendering": True,
        },
        "claim_boundary": (
            "Budding est ici une condition d’acquisition, pas un label "
            "morphologique individuel. Les composantes du simulateur sont une "
            "hypothèse de passage générique : un accord visuel local ne prouve "
            "ni une géométrie bourgeon-mère ni la couverture du domaine. "
            "L’exploration shmoo reste bloquée jusqu’à décision sur ce checkpoint."
        ),
    }


def build_2um_revision_model(
    particle_root: Path,
    simulation_root: Path,
) -> dict[str, Any]:
    real_population = load_particle_population(particle_root)["2um"]
    simulations = load_ssl_simulation_population(simulation_root)
    representative = select_real_representative(real_population)
    baseline_simulation, baseline_distance = match_simulation(
        representative, simulations
    )
    balanced_real, balanced_simulation, balanced_distance = (
        best_cross_population_pair(
            real_population,
            simulations,
            weights=MATCH_WEIGHTS,
        )
    )
    physical_real, physical_simulation, physical_distance = (
        best_cross_population_pair(
            real_population,
            simulations,
            weights=PHYSICAL_CORE_WEIGHTS,
        )
    )
    cases = [
        _pair_case(
            case_id="representative-balanced",
            class_label="2 µm · ancre représentative",
            selection_note=(
                "Référence C1 : le réel est choisi indépendamment du simulateur "
                "comme événement médian de la classe."
            ),
            selection_role="representative_baseline",
            real=representative,
            matched=baseline_simulation,
            distance=baseline_distance,
            weights=MATCH_WEIGHTS,
            real_population=len(real_population),
            simulation_population=len(simulations),
        ),
        _pair_case(
            case_id="best-pair-balanced",
            class_label="2 µm · meilleur couple équilibré",
            selection_note=(
                "Argmin global parmi les 36×6 982 couples avec les poids C1. "
                "Sélection favorable à l’existence d’un bon couple, non représentative."
            ),
            selection_role="existence_biased_balanced",
            real=balanced_real,
            matched=balanced_simulation,
            distance=balanced_distance,
            weights=MATCH_WEIGHTS,
            real_population=len(real_population),
            simulation_population=len(simulations),
        ),
        _pair_case(
            case_id="best-pair-physical",
            class_label="2 µm · meilleur couple physique",
            selection_note=(
                "Argmin global avec 30 % durée, 30 % Doppler et seulement 5 % "
                "amplitude. Sélection favorable et non représentative."
            ),
            selection_role="existence_biased_physical",
            real=physical_real,
            matched=physical_simulation,
            distance=physical_distance,
            weights=PHYSICAL_CORE_WEIGHTS,
            real_population=len(real_population),
            simulation_population=len(simulations),
        ),
    ]
    selected_ids = [
        identifier
        for case in cases
        for identifier in (case["real"]["id"], case["simulation"]["id"])
    ]
    return {
        "schema_version": 1,
        "milestone": "ssl-realism-audit-2um-revision",
        "title": "2 µm — peut-on trouver un voisin SSL visuellement meilleur ?",
        "claim_type": "local diagnostic / selection sensitivity",
        "question": (
            "Le mauvais accord du cas 2 µm représentatif vient-il de la règle "
            "d’appariement, ou reste-t-il possible de trouver dans le même pool "
            "SSL un couple réel–simulé nettement plus convaincant ?"
        ),
        "datasets": {
            "real": "particles2snr-f-dual-clean-c1-yolo-4class-saturation-reviewed@v2",
            "simulation": "yeast-passage-simulations@v2",
        },
        "cases": cases,
        "feature_contract": {
            "balanced_weights": MATCH_WEIGHTS,
            "physical_weights": PHYSICAL_CORE_WEIGHTS,
            "amplitude_policy": (
                "Percentile intra-population ; le scénario physique réduit son "
                "poids à 5 % sans supprimer l’amplitude."
            ),
            "distance_scaling": (
                "IQR du pool simulé train monoparticule ; 25 points de percentile "
                "pour l’amplitude."
            ),
            "excluded": [
                "SNR",
                "embedding appris",
                "corrélation point à point",
                "phase absolue",
                "position dans la fenêtre",
            ],
        },
        "display_contract": {
            "sampling_frequency_hz": FS,
            "samples": 4096,
            "duration_ms": 4.096,
            "trace_transform": "centrage puis division par le RMS de chaque signal",
            "trace_scale": "échelle verticale partagée dans chaque paire",
            "spectrogram": "5–100 kHz, dB relatifs au maximum de chaque signal, couleurs partagées [-55, 0] dB",
            "real_preprocessing": "crop 8192 à 2 MHz, passe-bande 5–100 kHz, resample_poly 2:1",
            "simulation_preprocessing": "signal 4096 stocké, aucune régénération ni transformation avant métriques",
        },
        "case_selection": {
            "population": (
                "36 événements réels 2 µm éligibles × 6 982 signaux SSL train "
                "component_count=1, soit 251 352 couples possibles"
            ),
            "rule": (
                "Afficher la référence représentative C1, puis deux argmins globaux "
                "prédéclarés avant rendu : distance équilibrée C1 et distance "
                "physique donnant 30 % à la durée et 30 % au Doppler."
            ),
            "selected_ids": selected_ids,
            "selected_before_rendering": True,
        },
        "claim_boundary": (
            "Les deux meilleurs couples sont explicitement sélectionnés pour "
            "démontrer l’existence d’un appariement favorable. Ils ne représentent "
            "pas la classe 2 µm et ne mesurent pas la couverture du pool SSL."
        ),
    }


def serializable_model(model: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (
            [
                {
                    sub_key: sub_value
                    for sub_key, sub_value in case.items()
                    if not sub_key.startswith("_")
                }
                for case in value
            ]
            if key == "cases"
            else value
        )
        for key, value in model.items()
    }


def _json_for_html(payload: dict[str, Any]) -> str:
    return (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        .replace("</", "<\\/")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def render_checkpoint_html(
    model: dict[str, Any],
    *,
    plot_data_uris: dict[str, str] | None = None,
) -> str:
    web_model = serializable_model(model)
    if plot_data_uris is not None:
        for case in web_model["cases"]:
            case["plot_data_uri"] = plot_data_uris[case["id"]]
    payload = _json_for_html(web_model)
    title = html.escape(model["title"])
    checkpoint = "C2" if "budding" in model["milestone"] else "C1"
    return f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{ --ink:#17201f;--muted:#66716e;--paper:#f3f0e8;--card:#fffdf8;
      --line:#d8d1c3;--green:#176f5b;--blue:#2c6386;--orange:#c26a2d;
      font-family:Inter,ui-sans-serif,system-ui,sans-serif;color-scheme:light; }}
    * {{ box-sizing:border-box; }} body {{ margin:0;background:var(--paper);color:var(--ink); }}
    header {{ padding:30px max(24px,calc((100vw - 1260px)/2));background:#172824;color:white; }}
    .eyebrow {{ color:#96d8c3;font-size:12px;font-weight:800;letter-spacing:.14em;text-transform:uppercase; }}
    h1 {{ max-width:1050px;margin:8px 0 10px;font-size:clamp(30px,4.4vw,52px);letter-spacing:-.04em;line-height:1.05; }}
    header p {{ max-width:970px;margin:0;color:#d4dfdb;line-height:1.5; }}
    main {{ max-width:1260px;margin:auto;padding:22px 24px 60px; }}
    .panel {{ background:var(--card);border:1px solid var(--line);border-radius:14px;box-shadow:0 12px 34px rgba(31,43,39,.08); }}
    .selector {{ display:grid;grid-template-columns:1fr auto;gap:14px;align-items:end;padding:16px;margin-bottom:14px; }}
    label {{ display:block;margin-bottom:6px;color:var(--muted);font-size:11px;font-weight:800;letter-spacing:.08em;text-transform:uppercase; }}
    select {{ width:100%;min-height:43px;padding:8px 12px;border:1px solid #aaa397;border-radius:9px;background:white;font:inherit; }}
    .dataset {{ font:11px ui-monospace,SFMono-Regular,Consolas,monospace;color:var(--green); }}
    .case-head {{ display:grid;grid-template-columns:1fr auto;gap:18px;padding:18px 20px;border-bottom:1px solid var(--line); }}
    h2 {{ margin:0 0 5px;font-size:27px; }} .case-head p {{ margin:0;color:var(--muted);line-height:1.45; }}
    .rank {{ align-self:start;padding:8px 11px;border-radius:999px;background:#dcece5;color:var(--green);font-size:11px;font-weight:850; }}
    .pair-image {{ display:block;width:100%;height:auto;background:white; }}
    .metric-wrap {{ padding:16px 20px 20px;overflow:auto; }}
    table {{ width:100%;border-collapse:collapse;font-size:12px; }}
    th,td {{ padding:8px 10px;border-bottom:1px solid #e4dfd5;text-align:right;white-space:nowrap; }}
    th:first-child,td:first-child {{ text-align:left; }} th {{ color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.06em; }}
    .provenance {{ display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:0 20px 20px; }}
    .source {{ padding:12px;border-radius:10px;background:#f2f0ea;min-width:0; }}
    .source b {{ display:block;margin-bottom:5px; }} .source code {{ font-size:10px;overflow-wrap:anywhere;white-space:normal; }}
    .notes {{ display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px; }}
    .note {{ padding:16px;border-left:5px solid var(--orange); }} .note strong {{ color:#844317; }}
    .note p {{ margin:7px 0 0;color:var(--muted);font-size:12px;line-height:1.5; }}
    .method {{ margin-top:14px;padding:18px 20px;border-left:5px solid var(--green); }}
    .method h3 {{ margin:0 0 8px;font-size:18px; }} .method p {{ margin:5px 0;color:var(--muted);font-size:12px;line-height:1.55; }}
    .factors {{ margin-top:7px;color:var(--muted);font-size:11px;line-height:1.45; }}
    footer {{ padding:20px;text-align:center;color:var(--muted);font-size:11px; }}
    @media(max-width:800px) {{ .selector,.case-head,.provenance,.notes {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <header><div class="eyebrow">Checkpoint {checkpoint} · audit réel → simulation</div>
    <h1>{title}</h1><p id="question"></p></header>
  <main>
    <section class="panel selector"><div><label for="case-select">Classe réelle auditée</label>
      <select id="case-select"></select></div><div class="dataset">pool SSL figé · train seulement</div></section>
    <section id="case-panel" class="panel"></section>
    <section class="notes">
      <article class="panel note"><strong>Transformation d’affichage</strong><p id="display-note"></p></article>
      <article class="panel note"><strong>Limite de la preuve</strong><p id="boundary"></p></article>
    </section>
    <section id="method-panel" class="panel method" hidden></section>
  </main>
  <footer>Interface locale en lecture seule · sélection calculée avant rendu · aucun signal régénéré.</footer>
  <script id="payload" type="application/json">{payload}</script>
  <script>
    const data=JSON.parse(document.getElementById("payload").textContent);
    const fmt=(v,d=3)=>Number(v).toLocaleString("fr-FR",{{maximumFractionDigits:d}});
    document.getElementById("question").textContent=data.question;
    document.getElementById("boundary").textContent=data.claim_boundary;
    document.getElementById("display-note").textContent=data.display_contract.trace_transform+
      ". Spectrogrammes : "+data.display_contract.spectrogram+".";
    if(data.simulation_methodology) {{
      const m=data.simulation_methodology;
      const panel=document.getElementById("method-panel");
      panel.hidden=false;
      panel.innerHTML=`<h3>Générateur SSL à deux composantes</h3>
        <p><b>Équation.</b> ${{m.model}}</p><p><b>Interprétation.</b> ${{m.geometry}}</p>
        <p><b>Facteurs.</b> ${{m.factor_ranges}}</p><p><b>Nuisances.</b> ${{m.nuisance}}</p>`;
    }}
    const select=document.getElementById("case-select");
    data.cases.forEach(c=>select.add(new Option(c.class_label,c.id)));
    function render() {{
      const c=data.cases.find(x=>x.id===select.value)||data.cases[0];
      const rows=c.metrics.map(m=>`<tr><td>${{m.label}}</td><td>${{fmt(m.real)}} ${{m.unit}}</td>
        <td>${{fmt(m.simulation)}} ${{m.unit}}</td><td>${{fmt(m.delta)}} ${{m.unit}}</td><td>${{fmt(100*m.weight,1)}} %</td></tr>`).join("");
      const sm=c.simulation.metadata;
      const factors=Number(sm.component_count)===2 ? `<div class="factors">
        Facteurs latents : durée ${{fmt(sm.duration_ms)}} ms · f₁ ${{fmt(sm.doppler_khz)}} kHz ·
        séparation déclarée ${{fmt(sm.component_separation_ms)}} ms · amplitude relative ${{fmt(sm.relative_component_amplitude)}} ·
        Δf ${{fmt(sm.frequency_separation_khz)}} kHz · SNR ${{fmt(sm.snr_db,1)}} dB</div>` : "";
      document.getElementById("case-panel").innerHTML=`<div class="case-head"><div><h2>${{c.class_label}}</h2>
        <p>${{c.selection_note||`Ancre réelle représentative parmi ${{c.real_population}} événements éligibles, puis argmin exact dans le pool SSL monoparticule.`}}</p></div>
        <div class="rank">voisin 1 / ${{c.simulation_population}} · distance ${{fmt(c.match_distance)}}</div></div>
        <img class="pair-image" src="${{c.plot_data_uri||`plots/${{c.id}}.png`}}" alt="Comparaison ${{c.class_label}}">
        <div class="metric-wrap"><table><thead><tr><th>Métrique</th><th>Réel</th><th>Simulation</th><th>Δ sim−réel</th><th>Poids</th></tr></thead><tbody>${{rows}}</tbody></table></div>
        <div class="provenance"><div class="source"><b>${{c.real_source_label||"Réel · sélection indépendante"}}</b><code>${{c.real.id}}</code>
          <div>RMS ${{fmt(c.real.descriptors.rms)}} · pic enveloppe ${{fmt(c.real.descriptors.envelope_peak)}} · crête-à-crête ${{fmt(c.real.descriptors.peak_to_peak)}}</div></div>
        <div class="source"><b>Simulation SSL existante</b><code>${{c.simulation.id}} · row ${{c.simulation.metadata.signal_row}}</code>
          <div>RMS ${{fmt(c.simulation.descriptors.rms)}} · pic enveloppe ${{fmt(c.simulation.descriptors.envelope_peak)}} · crête-à-crête ${{fmt(c.simulation.descriptors.peak_to_peak)}}</div>${{factors}}</div></div>`;
    }}
    select.addEventListener("change",()=>{{history.replaceState(null,"","?case="+select.value);render();}});
    select.value=new URLSearchParams(location.search).get("case")||data.cases[0].id;render();
  </script>
</body></html>"""
