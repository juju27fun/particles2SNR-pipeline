from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import torch
from matplotlib.figure import Figure
from scipy.signal import spectrogram

from .fft_analysis_pipeline_particles2SNR import run_pipeline
from .run_dataset import get_config_for_folder
from .saturation_cleaning import detect_unsafe_intervals
from .yeast_events import (
    bandpass_yeast_signal,
    review_calibrated_detection_config_v1,
)


FS = 2_000_000.0


def _canonical_postprocess_module() -> ModuleType:
    """Load the dataset generator so this report reuses its exact filters."""

    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/generation/generate_particles2SNR_dataset.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_particles2snr_dataset_generator_for_report", source
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load canonical dual-clean helpers from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _interval_ms(module: ModuleType, particle: dict[str, Any], length: int) -> tuple[float, float]:
    left, right, _center = module.particle_interval_samples(particle, length, FS)
    return float(left / FS * 1000.0), float(right / FS * 1000.0)


def _serializable_particle(
    module: ModuleType,
    particle: dict[str, Any],
    length: int,
) -> dict[str, Any]:
    start_ms, end_ms = _interval_ms(module, particle, length)
    return {
        "frequency_khz": float(particle["frequency"]) / 1000.0,
        "t0_ms": float(particle["t0"]) * 1000.0,
        "tau_ms": float(particle["tau"]) * 1000.0,
        "snr_db": float(particle["snr_db"]),
        "start_ms": start_ms,
        "end_ms": end_ms,
        "peak_group_id": particle.get("peak_group_id"),
        "clean_peak_group_id": particle.get("clean_peak_group_id"),
    }


def replay_particles2snr_dual_clean(
    signal: np.ndarray,
    reviewed: dict[str, str],
) -> tuple[dict[str, Any], np.ndarray]:
    """Replay P0 and the full filtered/dual-clean label policy in memory."""

    post = _canonical_postprocess_module()
    values = np.asarray(signal, dtype=np.float32)
    config = get_config_for_folder("yeast")
    config.bandpass_lowcut = 7_000.0
    config.bandpass_highcut = 80_000.0
    config.bandpass_order = 4
    result, _particles, _centers, _windows, _freqs, filtered, _noise = run_pipeline(
        values,
        SimpleNamespace(device="cpu", verbose=False),
        config,
        torch.device("cpu"),
        False,
    )
    raw = list(result["particles"])
    passage = [
        particle
        for particle in raw
        if post.keep_particle_by_passage_time(particle, 0.07, 0.65)[0]
    ]
    width, width_drops = post.filter_particles_by_yolo_width(
        passage,
        len(values),
        FS,
        min_width_ms=0.08,
        max_width_ms=1.5,
    )
    filtered_evidence, filtered_drops, filtered_groups = (
        post.refine_particles_with_peak_evidence(
            width,
            filtered,
            len(values),
            FS,
            envelope_window_ms=0.08,
            min_z=4.0,
            prominence_z=2.0,
            min_separation_ms=0.18,
            valley_ratio=0.55,
            cluster_gap_ms=0.25,
            keep_high_snr_db=4.0,
        )
    )
    dual_clean, clean_drops, clean_groups = post.annotate_clean_peak_support(
        filtered_evidence,
        values,
        len(values),
        FS,
        envelope_window_ms=0.08,
        min_z=4.0,
        prominence_z=2.0,
        min_separation_ms=0.18,
        valley_ratio=0.55,
    )
    final, nms_drops = post.merge_overlapping_particles(
        dual_clean,
        len(values),
        FS,
        iou_threshold=0.4,
        score_name="snr_db",
        duplicate_iou_threshold=0.6,
        close_center_distance_ms=0.20,
        ambiguous_center_distance_ms=0.30,
        close_frequency_hz=6000.0,
        ambiguous_frequency_hz=8000.0,
        snr_margin_db=4.0,
    )

    reviewed_start_ms = int(reviewed["event_start"]) / FS * 1000.0
    reviewed_end_ms = int(reviewed["event_end"]) / FS * 1000.0
    local_raw = [
        particle
        for particle in raw
        if reviewed_start_ms
        <= float(particle["t0"]) * 1000.0
        <= reviewed_end_ms
    ]
    local_raw.sort(key=lambda row: float(row["t0"]))
    final.sort(key=lambda row: float(row["t0"]))

    sat_info, unsafe = detect_unsafe_intervals(
        values,
        fs=FS,
        fmin=7_000.0,
        fmax=80_000.0,
        min_flat=500,
        zero_threshold=1.0e-4,
        guard_before=300,
        guard_after=300,
    )
    final_rows = [
        _serializable_particle(post, particle, len(values)) for particle in final
    ]
    local_rows = [
        _serializable_particle(post, particle, len(values))
        for particle in local_raw
    ]
    for proposal_index, row in enumerate(local_rows, start=1):
        row["proposal_id"] = f"P{proposal_index}"
        row["survives_final"] = any(
            abs(row["frequency_khz"] - kept["frequency_khz"]) < 1.0e-6
            and abs(row["t0_ms"] - kept["t0_ms"]) < 1.0e-6
            for kept in final_rows
        )
    for kept in final_rows:
        source = next(
            (
                row
                for row in local_rows
                if abs(row["frequency_khz"] - kept["frequency_khz"]) < 1.0e-6
                and abs(row["t0_ms"] - kept["t0_ms"]) < 1.0e-6
            ),
            None,
        )
        kept["proposal_id"] = source["proposal_id"] if source else None
    if len(final_rows) == 2:
        left, right = final_rows
        intersection = max(
            0.0,
            min(left["end_ms"], right["end_ms"])
            - max(left["start_ms"], right["start_ms"]),
        )
        union = max(left["end_ms"], right["end_ms"]) - min(
            left["start_ms"], right["start_ms"]
        )
        final_iou = intersection / union if union > 0.0 else 0.0
        center_gap_ms = abs(left["t0_ms"] - right["t0_ms"])
    else:
        final_iou = None
        center_gap_ms = None

    def _dropped_at(particle: dict[str, Any]) -> str | None:
        def in_stage(rows: list[dict[str, Any]]) -> bool:
            return any(
                abs(float(row["t0"]) - float(particle["t0"])) < 1.0e-9
                and abs(float(row["frequency"]) - float(particle["frequency"])) < 1.0e-6
                for row in rows
            )

        for stage_rows, label in (
            (final, None),
            (dual_clean, "nms"),
            (filtered_evidence, "dual_clean"),
            (width, "filtered_evidence"),
            (passage, "width"),
        ):
            if in_stage(stage_rows):
                return label
        return "passage_time"

    raw_all_rows = []
    for particle in sorted(raw, key=lambda row: float(row["t0"])):
        serialized = _serializable_particle(post, particle, len(values))
        serialized["dropped_at"] = _dropped_at(particle)
        raw_all_rows.append(serialized)

    replay = {
        "raw_all": raw_all_rows,
        "config": {
            "sampling_rate_hz": FS,
            "bandpass_hz": [7_000.0, 80_000.0],
            "fft_window": int(config.fft_window_length),
            "fft_stride": int(config.fft_stride),
            "energy_threshold": float(config.energy_threshold),
            "max_peaks_per_window": int(config.max_peaks),
            "relative_peak_threshold": float(config.next_peak_threshold_factor),
            "narrow_bandpass_width_hz": float(config.narrow_bandpass_width),
            "tau_ms": [0.07, 0.65],
            "box_width_ms": [0.08, 1.5],
            "nms_iou": 0.4,
        },
        "human": {
            "event_id": reviewed["event_id"],
            "start_ms": reviewed_start_ms,
            "end_ms": reviewed_end_ms,
            "width_ms": float(reviewed["width_ms"]),
            "doppler_low_khz": float(reviewed["doppler_low_hz"]) / 1000.0,
            "doppler_high_khz": float(reviewed["doppler_high_hz"]) / 1000.0,
            "event_present": reviewed["review_event_present"],
            "full_event_visible": reviewed["review_full_event_visible"],
        },
        "stage_counts": {
            "raw_full_trace": len(raw),
            "raw_centered_in_human_event": len(local_raw),
            "after_passage_time": len(passage),
            "after_width": len(width),
            "after_filtered_peak_evidence": len(filtered_evidence),
            "after_dual_clean": len(dual_clean),
            "after_nms": len(final),
        },
        "raw_local": local_rows,
        "final": final_rows,
        "filtered_peak_groups": filtered_groups,
        "clean_peak_groups": clean_groups,
        "drops": {
            "width": width_drops,
            "filtered_peak_evidence": filtered_drops,
            "clean_peak_evidence": clean_drops,
            "nms": nms_drops,
        },
        "final_pair": {
            "iou": final_iou,
            "center_gap_ms": center_gap_ms,
            "nms_iou_threshold": 0.4,
            "close_center_threshold_ms": 0.20,
            "ambiguous_center_threshold_ms": 0.30,
        },
        "saturation": {
            "is_saturated": bool(sat_info["is_saturated"]),
            "unsafe_intervals": [list(row) for row in unsafe],
        },
    }
    return replay, np.asarray(filtered, dtype=np.float32)


def replay_particles2snr_on_synthetic(
    signal: np.ndarray,
    *,
    truth_start: int,
    truth_end: int,
    frequency_hz: float,
) -> dict[str, Any]:
    """Replay the legacy pipeline on a synthetic trace with known ground truth.

    The reviewed-row metadata normally comes from a manual review queue; for a
    synthetic demonstration the caller supplies the constructed truth instead.
    """
    reviewed = {
        "event_id": "synthetic",
        "event_start": str(int(truth_start)),
        "event_end": str(int(truth_end)),
        "width_ms": str((int(truth_end) - int(truth_start)) / FS * 1000.0),
        "doppler_low_hz": str(float(frequency_hz)),
        "doppler_high_hz": str(float(frequency_hz)),
        "review_event_present": "yes",
        "review_full_event_visible": "yes",
    }
    replay, _filtered = replay_particles2snr_dual_clean(signal, reviewed)
    return replay


def assert_reference_replay_contract(replay: dict[str, Any]) -> None:
    expected = {
        "raw_full_trace": 20,
        "raw_centered_in_human_event": 5,
        "after_filtered_peak_evidence": 2,
        "after_dual_clean": 2,
        "after_nms": 2,
    }
    actual = replay["stage_counts"]
    mismatches = {
        key: (actual.get(key), value)
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "Reference particles2SNR replay changed; refusing to render a stale "
            f"scientific narrative: {mismatches}"
        )
    if replay["saturation"]["unsafe_intervals"]:
        raise RuntimeError("Reference trace unexpectedly contains unsafe intervals")


_FAILURE_TEXT = {
    "fr": {
        "peak_low": "pic Doppler bas exact",
        "peak_high": "pic Doppler haut exact",
        "crest_low": " · crête basse",
        "crest_high": " · crête haute",
        "extra_max": " · maximum supplémentaire",
        "kept": " · gardée",
        "removed": " · supprimée",
        "kept_word": "gardée",
        "removed_word": "supprimée",
        "human_bar": "1 événement complet · revue humaine + pipeline yeast",
        "ytick_truth": "vérité événementielle",
        "not_merged": "Non fusionnées : IoU = {iou:.3f} < 0,400 · écart des centres = {gap:.3f} ms",
        "ylab_signal": "signal filtré\nnormalisé",
        "ylab_stft": "STFT actuelle\nfréquence (kHz)",
        "ylab_raw": "hypothèses P0\navant dual-clean",
        "ylab_final": "décision\névénementielle",
        "xlabel": "temps dans la trace (ms)",
        "title_signal": "La zone verte est un seul passage humain complet",
        "title_stft": "Deux crêtes Doppler exactes — mais cinq hypothèses fréquentielles P0 locales",
        "title_raw": "Chaque maximum devient d’abord une particule ajustée : t₀ ± 2,5τ",
        "title_final": "Après filtered + dual-clean + NMS : encore 2 boîtes pour 1 seul événement",
        "suptitle": "Pourquoi particles2SNR_F dual-clean fragmente ce yeast multi-Doppler",
    },
    "en": {
        "peak_low": "exact low Doppler peak",
        "peak_high": "exact high Doppler peak",
        "crest_low": " · low crest",
        "crest_high": " · high crest",
        "extra_max": " · extra maximum",
        "kept": " · kept",
        "removed": " · removed",
        "kept_word": "kept",
        "removed_word": "removed",
        "human_bar": "1 complete event · human review + yeast pipeline",
        "ytick_truth": "event ground truth",
        "not_merged": "Not merged: IoU = {iou:.3f} < 0.400 · centre gap = {gap:.3f} ms",
        "ylab_signal": "filtered signal\nnormalised",
        "ylab_stft": "current STFT\nfrequency (kHz)",
        "ylab_raw": "P0 hypotheses\nbefore dual-clean",
        "ylab_final": "event\ndecision",
        "xlabel": "time in trace (ms)",
        "title_signal": "The green span is one single complete human passage",
        "title_stft": "Two exact Doppler crests — but five local P0 frequency hypotheses",
        "title_raw": "Each maximum first becomes a fitted particle: t₀ ± 2.5τ",
        "title_final": "After filtered + dual-clean + NMS: still 2 boxes for 1 single event",
        "suptitle": "Why particles2SNR_F dual-clean fragments this multi-Doppler yeast",
    },
}


def render_particles2snr_failure_plot(
    *,
    signal: np.ndarray,
    filtered: np.ndarray,
    replay: dict[str, Any],
    destination: Path,
    language: str = "fr",
) -> None:
    text = _FAILURE_TEXT[language]
    human = replay["human"]
    raw = replay["raw_local"]
    final = replay["final"]
    config = review_calibrated_detection_config_v1()
    frequencies, times, complex_values = spectrogram(
        bandpass_yeast_signal(signal, config) - float(np.mean(signal)),
        fs=config.sampling_frequency_hz,
        nperseg=config.stft_nperseg,
        noverlap=config.stft_noverlap,
        window="hann",
        mode="complex",
    )
    frequency_mask = (frequencies >= 7_000.0) & (frequencies <= 28_000.0)
    magnitude_db = 20.0 * np.log10(
        np.abs(complex_values[frequency_mask]) + 1.0e-12
    )
    low, high = np.quantile(magnitude_db, [0.04, 0.997])
    time_ms = np.arange(len(signal)) / FS * 1000.0
    xlim = (2.72, 5.22)
    human_start = float(human["start_ms"])
    human_end = float(human["end_ms"])

    colors = ["#607d8b", "#00a896", "#d97706", "#2f6fed", "#7b61a8"]
    figure = Figure(figsize=(14.6, 11.4), constrained_layout=True)
    grid = figure.add_gridspec(4, 1, height_ratios=(0.75, 1.7, 1.35, 1.25))
    signal_axis = figure.add_subplot(grid[0, 0])
    stft_axis = figure.add_subplot(grid[1, 0], sharex=signal_axis)
    raw_axis = figure.add_subplot(grid[2, 0], sharex=signal_axis)
    final_axis = figure.add_subplot(grid[3, 0], sharex=signal_axis)

    centered = filtered - float(np.median(filtered))
    scale = max(float(np.quantile(np.abs(centered), 0.995)), 1.0e-12)
    signal_axis.plot(time_ms, centered / scale, color="#263640", linewidth=0.75)
    stft_axis.pcolormesh(
        times * 1000.0,
        frequencies[frequency_mask] / 1000.0,
        magnitude_db,
        shading="auto",
        cmap="magma",
        vmin=float(low),
        vmax=float(high),
    )

    for axis in (signal_axis, stft_axis, raw_axis, final_axis):
        axis.axvspan(human_start, human_end, color="#0f8a70", alpha=0.10)
        axis.axvline(human_start, color="#0f8a70", linewidth=0.9)
        axis.axvline(human_end, color="#0f8a70", linewidth=0.9)
        axis.set_xlim(*xlim)
        axis.grid(True, color="#d9e0e6", linewidth=0.5, alpha=0.55)

    peak_specs = (
        (text["peak_low"], float(human["doppler_low_khz"])),
        (text["peak_high"], float(human["doppler_high_khz"])),
    )
    for index, (label, frequency) in enumerate(peak_specs):
        stft_axis.hlines(
            frequency,
            human_start,
            human_end,
            color="#7ff5df",
            linewidth=2.2,
            linestyles="--",
        )
        stft_axis.text(
            xlim[0] + 0.04,
            frequency + (0.65 if index == 0 else -0.65),
            f"{label} · {frequency:.5g} kHz",
            color="white",
            fontsize=9,
            fontweight="bold",
            va="center",
            bbox={
                "facecolor": "#17212b",
                "edgecolor": "#7ff5df",
                "alpha": 0.86,
                "pad": 2.0,
            },
        )

    low_match = min(
        range(len(raw)),
        key=lambda index: abs(
            raw[index]["frequency_khz"] - float(human["doppler_low_khz"])
        ),
    )
    high_match = min(
        range(len(raw)),
        key=lambda index: abs(
            raw[index]["frequency_khz"] - float(human["doppler_high_khz"])
        ),
    )
    for index, row in enumerate(raw):
        color = colors[index % len(colors)]
        stft_axis.scatter(
            row["t0_ms"],
            row["frequency_khz"],
            s=60,
            color=color,
            edgecolor="white",
            linewidth=0.9,
            zorder=5,
        )
        label = f"P{index + 1}"
        if index == low_match:
            label += text["crest_low"]
        elif index == high_match:
            label += text["crest_high"]
        else:
            label += text["extra_max"]
        label += text["kept"] if row["survives_final"] else text["removed"]
        stft_axis.annotate(
            label,
            xy=(row["t0_ms"], row["frequency_khz"]),
            xytext=(5, 7),
            textcoords="offset points",
            color="white",
            fontsize=7.6,
            fontweight="bold",
        )

        y = len(raw) - index
        raw_axis.plot(
            [row["start_ms"], row["end_ms"]],
            [y, y],
            color=color,
            linewidth=10,
            alpha=1.0 if row["survives_final"] else 0.30,
            solid_capstyle="round",
        )
        raw_axis.scatter([row["t0_ms"]], [y], color="#17212b", s=20, zorder=4)
        if not row["survives_final"]:
            raw_axis.scatter(
                [row["t0_ms"]],
                [y],
                marker="x",
                color="#b34736",
                linewidth=2.0,
                s=52,
                zorder=5,
            )

    raw_axis.set_yticks(
        list(range(len(raw), 0, -1)),
        [
            (
                f"P{index + 1} · {row['frequency_khz']:.2f} kHz · "
                f"{text['kept_word'] if row['survives_final'] else text['removed_word']}"
            )
            for index, row in enumerate(raw)
        ],
    )
    raw_axis.set_ylim(0.45, len(raw) + 0.6)

    raw_colors_by_id = {
        row["proposal_id"]: colors[index % len(colors)]
        for index, row in enumerate(raw)
    }
    final_y = (2.1, 1.1)
    for row, y in zip(final, final_y):
        color = raw_colors_by_id.get(row.get("proposal_id"), "#d1493f")
        final_axis.plot(
            [row["start_ms"], row["end_ms"]],
            [y, y],
            color=color,
            linewidth=13,
            solid_capstyle="round",
        )
        final_axis.scatter([row["t0_ms"]], [y], color="#17212b", s=25, zorder=4)
        final_axis.text(
            (row["start_ms"] + row["end_ms"]) / 2.0,
            y + 0.16,
            f"{row.get('proposal_id') or ''} · {row['frequency_khz']:.2f} kHz",
            color=color,
            fontsize=8.5,
            fontweight="bold",
            ha="center",
        )
    final_axis.plot(
        [human_start, human_end],
        [0.0, 0.0],
        color="#0f8a70",
        linewidth=15,
        solid_capstyle="round",
    )
    final_axis.text(
        (human_start + human_end) / 2.0,
        0.17,
        text["human_bar"],
        color="#08745f",
        fontsize=9,
        fontweight="bold",
        ha="center",
    )
    final_axis.set_yticks(
        [0.0, 1.1, 2.1],
        [text["ytick_truth"], "dual-clean B", "dual-clean A"],
    )
    final_axis.set_ylim(-0.45, 2.65)

    pair = replay["final_pair"]
    final_axis.text(
        0.99,
        0.96,
        text["not_merged"].format(iou=pair["iou"], gap=pair["center_gap_ms"]),
        transform=final_axis.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#8e3428",
        fontweight="bold",
        bbox={
            "facecolor": "#fff0ed",
            "edgecolor": "#e9a99d",
            "alpha": 0.94,
            "pad": 4.0,
        },
    )

    signal_axis.set_ylabel(text["ylab_signal"])
    signal_axis.tick_params(labelbottom=False)
    stft_axis.set_ylabel(text["ylab_stft"])
    stft_axis.tick_params(labelbottom=False)
    raw_axis.set_ylabel(text["ylab_raw"])
    raw_axis.tick_params(labelbottom=False)
    final_axis.set_ylabel(text["ylab_final"])
    final_axis.set_xlabel(text["xlabel"])
    signal_axis.set_title(text["title_signal"], loc="left", fontweight="bold")
    stft_axis.set_title(text["title_stft"], loc="left", fontweight="bold")
    raw_axis.set_title(text["title_raw"], loc="left", fontweight="bold")
    final_axis.set_title(text["title_final"], loc="left", fontweight="bold")
    figure.suptitle(
        text["suptitle"],
        fontsize=16,
        fontweight="bold",
        color="#17212b",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=170, facecolor="white")


def build_particles2snr_legacy_section(
    *,
    signal: np.ndarray,
    reviewed: dict[str, str],
    destination: Path,
    plot_relative_path: str,
) -> dict[str, Any]:
    replay, filtered = replay_particles2snr_dual_clean(signal, reviewed)
    assert_reference_replay_contract(replay)
    render_particles2snr_failure_plot(
        signal=np.asarray(signal, dtype=np.float32),
        filtered=filtered,
        replay=replay,
        destination=destination,
    )
    return {
        **replay,
        "plot": plot_relative_path,
        "title": "L’ancienne logique : une hypothèse par maximum spectral",
        "subtitle": (
            "Sur ce même passage, particles2SNR ajuste plusieurs particules avant "
            "de savoir combien d’événements biologiques sont présents."
        ),
        "steps": [
            {
                "number": "P1",
                "title": "Nettoyage + bande",
                "body": "Contrôle saturation, puis Butterworth 7–80 kHz. Cette trace ne nécessite aucune réparation.",
            },
            {
                "number": "P2",
                "title": "Fenêtres FFT",
                "body": "Fenêtre 2048, pas 512. Les fenêtres d’énergie > 4000 sont explorées séparément.",
            },
            {
                "number": "P3",
                "title": "Maxima Doppler",
                "body": "Jusqu’à 3 maxima par fenêtre, retenus au-dessus de 50 % du maximum local.",
            },
            {
                "number": "P4",
                "title": "Une particule par pic",
                "body": "Passe-bande étroit de 4 kHz, enveloppe de Hilbert, puis estimation de t₀ et τ.",
            },
            {
                "number": "P5",
                "title": "Filtered + clean",
                "body": "L’évidence d’enveloppe doit exister dans les vues filtrée et nettoyée non filtrée.",
            },
            {
                "number": "P6",
                "title": "NMS temporel",
                "body": "Les doublons sont fusionnés selon leur recouvrement, leur distance temporelle et leur fréquence.",
            },
        ],
    }
