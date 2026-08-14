from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from .yeast_events import (
    detector_trace,
    review_calibrated_detection_config_v1,
)
from .yeast_particles2snr_comparison import build_particles2snr_legacy_section


CASE_SPECS = (
    {
        "slug": "strict-clean",
        "event_id": "214f4ce4967af98a954c:00",
        "eyebrow": "Exemple A · candidat net",
        "title": "Une structure Doppler devient une boîte temporelle",
        "takeaway": (
            "La bande passe nettoie la lecture, la STFT localise la structure, "
            "l’énergie la ramène sur l’axe du temps et le score MAD montre qu’elle "
            "est atypique relativement au bruit de cette trace."
        ),
    },
    {
        "slug": "reject-neighbour",
        "event_id": "09f788a7473797b794f6:01",
        "eyebrow": "Exemple B · proposition parasite",
        "title": "Une hausse locale ne suffit pas à faire un événement",
        "takeaway": (
            "La boîte orange ne contient pas d’événement. Le vrai événement visible à "
            "gauche est localisé séparément : la construction produit volontairement "
            "des candidats à filtrer au stade de décision."
        ),
    },
    {
        "slug": "multi-event",
        "event_id": "e1b4603f8b9de6204003:02",
        "eyebrow": "Exemple C · événements voisins",
        "title": "La lecture temps-fréquence aide à séparer deux passages",
        "takeaway": (
            "Deux structures localisées deviennent deux groupes temporels distincts. "
            "C’est précisément un cas où l’ablation énergie seule perd plus facilement "
            "la séparation offerte par la STFT."
        ),
    },
    {
        "slug": "one-event-multi-doppler",
        "event_id": "9459e76ce29342debc90:00",
        "eyebrow": "Exemple D · 1 yeast, 2 pics",
        "title": "Deux pics Doppler ne deviennent pas deux événements",
        "takeaway": (
            "La revue humaine confirme un seul événement complet. Les deux pics "
            "fréquentiels alimentent la même énergie temporelle, restent dans le "
            "même groupe de trames actives et produisent donc une seule boîte."
        ),
    },
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _normalized(values: np.ndarray) -> np.ndarray:
    centered = np.asarray(values, dtype=np.float64) - float(np.median(values))
    scale = float(np.quantile(np.abs(centered), 0.995))
    return centered / max(scale, 1.0e-12)


def _candidate_color(quality: str) -> str:
    return {
        "strict": "#0f8a70",
        "medium": "#2f6fed",
        "reject": "#d97706",
    }.get(quality, "#65737e")


def _render_case_plot(
    *,
    signal: np.ndarray,
    reviewed: dict[str, str],
    candidates: list[dict[str, str]],
    destination: Path,
) -> None:
    config = review_calibrated_detection_config_v1()
    trace = detector_trace(signal, config)
    center = int(reviewed["center_index"])
    half = 4096
    crop_start = max(0, center - half)
    crop_end = min(signal.size, center + half)
    samples = np.arange(crop_start, crop_end)
    x_ms = (samples - center) / config.sampling_frequency_hz * 1000.0
    relative_stft_ms = (
        trace.times - center / config.sampling_frequency_hz
    ) * 1000.0

    figure = Figure(figsize=(13.2, 10.2), constrained_layout=True)
    grid = figure.add_gridspec(
        5,
        1,
        height_ratios=(0.72, 0.72, 1.45, 0.82, 0.92),
    )
    raw_axis = figure.add_subplot(grid[0, 0])
    filtered_axis = figure.add_subplot(grid[1, 0], sharex=raw_axis)
    spectrum_axis = figure.add_subplot(grid[2, 0], sharex=raw_axis)
    energy_axis = figure.add_subplot(grid[3, 0], sharex=raw_axis)
    robust_axis = figure.add_subplot(grid[4, 0], sharex=raw_axis)

    raw_axis.plot(
        x_ms,
        _normalized(signal[crop_start:crop_end]),
        color="#5b6470",
        linewidth=0.75,
    )
    filtered_axis.plot(
        x_ms,
        _normalized(trace.filtered[crop_start:crop_end]),
        color="#17212b",
        linewidth=0.75,
    )
    magnitude_db = 20.0 * np.log10(np.abs(trace.complex_stft) + 1.0e-12)
    low, high = np.quantile(magnitude_db, [0.05, 0.995])
    spectrum_axis.pcolormesh(
        relative_stft_ms,
        trace.frequencies / 1000.0,
        magnitude_db,
        shading="auto",
        cmap="magma",
        vmin=float(low),
        vmax=float(high),
    )
    energy_median = trace.energy_median
    energy_scale = trace.energy_scale
    median_denominator = max(energy_median, 1.0e-12)
    relative_energy = trace.frame_energy / median_denominator
    energy_axis.plot(
        relative_stft_ms,
        relative_energy,
        color="#2f6fed",
        linewidth=1.1,
        label="énergie de bande E / médiane(E)",
    )
    energy_axis.axhline(
        1.0,
        color="#63717f",
        linestyle="--",
        linewidth=0.9,
        label="médiane : bruit typique",
    )
    energy_axis.axhspan(
        max(0.0, (energy_median - energy_scale) / median_denominator),
        (energy_median + energy_scale) / median_denominator,
        color="#9aa7b2",
        alpha=0.16,
        label="± 1.4826 MAD",
    )
    robust_axis.plot(
        relative_stft_ms,
        trace.energy_z,
        color="#6f42c1",
        linewidth=1.15,
        label="score robuste z",
    )
    robust_axis.axhline(
        config.active_snr_z,
        color="#d97706",
        linestyle="--",
        linewidth=0.9,
        label="activation z = 3,5",
    )
    active = trace.active.astype(bool)
    robust_axis.scatter(
        relative_stft_ms[active],
        np.full(int(np.count_nonzero(active)), -0.55),
        s=12,
        color="#0f8a70",
        marker="s",
        label="trames actives : z et concentration",
        zorder=4,
    )
    robust_axis.set_yscale("symlog", linthresh=1.0)

    for candidate in candidates:
        left = (
            int(candidate["event_start"]) - center
        ) / config.sampling_frequency_hz * 1000.0
        right = (
            int(candidate["event_end"]) - center
        ) / config.sampling_frequency_hz * 1000.0
        if right < -2.1 or left > 2.1:
            continue
        color = _candidate_color(candidate["quality"])
        alpha = 0.19 if candidate["event_id"] == reviewed["event_id"] else 0.10
        for axis in (filtered_axis, spectrum_axis, energy_axis, robust_axis):
            axis.axvspan(left, right, color=color, alpha=alpha)
        candidate_center = (
            int(candidate["center_index"]) - center
        ) / config.sampling_frequency_hz * 1000.0
        filtered_axis.axvline(candidate_center, color=color, linewidth=0.9)
        filtered_axis.text(
            candidate_center,
            0.91,
            f"{candidate['quality']} · z={float(candidate['snr_proxy']):.1f}",
            color=color,
            fontsize=8.5,
            fontweight="bold",
            ha="center",
            va="top",
            transform=filtered_axis.get_xaxis_transform(),
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
        )

    reviewed_left = (
        int(reviewed["event_start"]) - center
    ) / config.sampling_frequency_hz * 1000.0
    reviewed_right = (
        int(reviewed["event_end"]) - center
    ) / config.sampling_frequency_hz * 1000.0
    peak_frequency = float(reviewed["doppler_peak_hz"]) / 1000.0
    multi_doppler = int(reviewed["n_doppler_peaks"]) > 1
    if reviewed["quality"] == "reject":
        annotation = "activité locale détectée\n⇒ candidat à filtrer"
    elif multi_doppler:
        annotation = "plusieurs pics Doppler\nmême support temporel"
    else:
        annotation = "structure Doppler localisée\n⇒ candidat potentiel"
    spectrum_axis.annotate(
        annotation,
        xy=((reviewed_left + reviewed_right) / 2.0, peak_frequency),
        xytext=(reviewed_left - 1.25, min(76.0, peak_frequency + 22.0)),
        arrowprops={"arrowstyle": "->", "color": "#ffffff", "linewidth": 1.4},
        color="#ffffff",
        fontsize=9,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#17212b",
            "edgecolor": "#ffffff",
            "alpha": 0.86,
        },
    )
    if multi_doppler:
        doppler_frequencies = sorted(
            {
                float(reviewed["doppler_low_hz"]) / 1000.0,
                float(reviewed["doppler_high_hz"]) / 1000.0,
            }
        )
        for index, frequency in enumerate(doppler_frequencies, start=1):
            spectrum_axis.hlines(
                frequency,
                reviewed_left,
                reviewed_right,
                color="#54e1cf",
                linewidth=2.1,
                linestyles="--",
            )
            spectrum_axis.text(
                reviewed_right + 0.04,
                frequency,
                f"pic {index} · {frequency:.1f} kHz",
                color="#dffff8",
                fontsize=8,
                fontweight="bold",
                va="center",
                bbox={
                    "facecolor": "#17212b",
                    "edgecolor": "none",
                    "alpha": 0.78,
                    "pad": 1.5,
                },
            )
        energy_peak_index = int(np.argmax(trace.frame_energy))
        energy_axis.annotate(
            "Σ sur les fréquences\n⇒ une seule courbe E[m]",
            xy=(
                float(relative_stft_ms[energy_peak_index]),
                float(relative_energy[energy_peak_index]),
            ),
            xytext=(reviewed_right + 0.22, float(np.max(relative_energy)) * 0.72),
            arrowprops={"arrowstyle": "->", "color": "#2f6fed"},
            color="#234fba",
            fontsize=8.5,
            fontweight="bold",
        )
        robust_axis.text(
            (reviewed_left + reviewed_right) / 2.0,
            -0.18,
            "1 groupe temporel → 1 boîte",
            color="#08745f",
            fontsize=8.5,
            fontweight="bold",
            ha="center",
            va="bottom",
            bbox={
                "facecolor": "white",
                "edgecolor": "#0f8a70",
                "alpha": 0.88,
                "pad": 2.0,
            },
        )

    for axis in (
        raw_axis,
        filtered_axis,
        spectrum_axis,
        energy_axis,
        robust_axis,
    ):
        axis.axvline(0.0, color="#d1493f", linewidth=0.9, alpha=0.9)
        axis.grid(True, color="#d9e0e6", linewidth=0.5, alpha=0.65)
        axis.set_xlim(-2.048, 2.048)
    raw_axis.set_ylabel("1 · brut\nnormalisé")
    filtered_axis.set_ylabel("2 · bande\n7–80 kHz")
    spectrum_axis.set_ylabel("3 · STFT\nfréquence (kHz)")
    energy_axis.set_ylabel("4 · énergie\nrelative")
    robust_axis.set_ylabel("5 · score\nMAD")
    robust_axis.set_xlabel("temps relatif au centre contrôlé (ms)")
    raw_axis.tick_params(labelbottom=False)
    filtered_axis.tick_params(labelbottom=False)
    spectrum_axis.tick_params(labelbottom=False)
    energy_axis.tick_params(labelbottom=False)
    energy_axis.legend(
        loc="upper left",
        ncol=3,
        fontsize=8,
        frameon=False,
    )
    robust_axis.legend(
        loc="upper left",
        ncol=3,
        fontsize=8,
        frameon=False,
    )
    figure.suptitle(
        "Même trace : rendre visible, mesurer, normaliser, proposer",
        color="#17212b",
        fontsize=15,
        fontweight="bold",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150, facecolor="white")


def build_pipeline_board_model(
    *,
    candidate_dataset: Path,
    review_dir: Path,
    raw_dataset_root: Path,
    assets_dir: Path,
    supporting_candidate_dataset: Path | None = None,
    supporting_review_dir: Path | None = None,
    include_particles2snr_legacy: bool = False,
) -> dict[str, Any]:
    review_sources = [review_dir]
    candidate_sources = [candidate_dataset]
    if supporting_candidate_dataset is not None or supporting_review_dir is not None:
        if supporting_candidate_dataset is None or supporting_review_dir is None:
            raise ValueError("Supporting candidate and review sources must be paired")
        review_sources.append(supporting_review_dir)
        candidate_sources.append(supporting_candidate_dataset)
    review_rows = {
        row["event_id"]: row
        for source in review_sources
        for row in _read_csv(source / "manual_review_queue.csv")
    }
    candidates = [
        row
        for source in candidate_sources
        for row in _read_csv(source / "candidate_events.csv")
    ]
    candidates_by_record: dict[str, list[dict[str, str]]] = {}
    for row in candidates:
        candidates_by_record.setdefault(row["record_id"], []).append(row)

    cases: list[dict[str, Any]] = []
    for spec in CASE_SPECS:
        reviewed = review_rows[spec["event_id"]]
        record_candidates = candidates_by_record[reviewed["record_id"]]
        signal = np.load(
            raw_dataset_root / reviewed["relative_path"], allow_pickle=False
        )
        plot_name = f"plots/{spec['slug']}.png"
        _render_case_plot(
            signal=np.asarray(signal, dtype=np.float32),
            reviewed=reviewed,
            candidates=record_candidates,
            destination=assets_dir.parent / plot_name,
        )
        visible_neighbours = [
            {
                "event_id": row["event_id"],
                "quality": row["quality"],
                "snr_proxy": float(row["snr_proxy"]),
                "offset_ms": (
                    int(row["center_index"]) - int(reviewed["center_index"])
                )
                / 2_000_000.0
                * 1000.0,
            }
            for row in record_candidates
            if abs(int(row["center_index"]) - int(reviewed["center_index"])) <= 4096
        ]
        cases.append(
            {
                **spec,
                "plot": plot_name,
                "record_id": reviewed["record_id"],
                "relative_path": reviewed["relative_path"],
                "source_group": reviewed["source_group"],
                "quality": reviewed["quality"],
                "snr_proxy": float(reviewed["snr_proxy"]),
                "concentration": float(reviewed["energy_concentration"]),
                "phase_coherence": float(reviewed["phase_coherence"]),
                "n_doppler_peaks": int(reviewed["n_doppler_peaks"]),
                "doppler_low_khz": float(reviewed["doppler_low_hz"]) / 1000.0,
                "doppler_high_khz": float(reviewed["doppler_high_hz"]) / 1000.0,
                "width_ms": float(reviewed["width_ms"]),
                "human_event_present": reviewed["review_event_present"],
                "human_center_acceptable": reviewed["review_center_acceptable"],
                "human_full_event_visible": reviewed["review_full_event_visible"],
                "human_artifact": reviewed["review_artifact"],
                "reviewer": reviewed["reviewer"],
                "review_notes": reviewed["review_notes"],
                "visible_candidates": visible_neighbours,
                "multi_doppler_mechanism": (
                    {
                        "peak_summary": (
                            f"{int(reviewed['n_doppler_peaks'])} pics : "
                            f"{float(reviewed['doppler_low_hz']) / 1000.0:.1f} "
                            f"et {float(reviewed['doppler_high_hz']) / 1000.0:.1f} kHz"
                        ),
                        "aggregation": "Σ de la puissance excédentaire sur les fréquences",
                        "temporal_group": (
                            f"1 groupe actif, fusion des interruptions ≤ "
                            f"{review_calibrated_detection_config_v1().cluster_gap_ms:g} ms"
                        ),
                        "candidate": "1 seule boîte candidate",
                        "diagnostic": (
                            f"{int(reviewed['n_doppler_peaks'])} pics décrits "
                            "après création de la boîte"
                        ),
                        "counting_rule": (
                            "L’unité comptée est le groupe temporel, pas le nombre "
                            "de pics fréquentiels."
                        ),
                        "limit": (
                            "Si deux activités sont séparées dans le temps au-delà "
                            "du seuil de regroupement, elles deviennent deux candidats."
                        ),
                    }
                    if int(reviewed["n_doppler_peaks"]) > 1
                    else None
                ),
            }
        )

    legacy_particles2snr = None
    if include_particles2snr_legacy:
        legacy_reviewed = review_rows["9459e76ce29342debc90:00"]
        legacy_signal = np.load(
            raw_dataset_root / legacy_reviewed["relative_path"],
            allow_pickle=False,
        )
        legacy_plot = "plots/particles2snr-multi-doppler-failure.png"
        legacy_particles2snr = build_particles2snr_legacy_section(
            signal=np.asarray(legacy_signal, dtype=np.float32),
            reviewed=legacy_reviewed,
            destination=assets_dir.parent / legacy_plot,
            plot_relative_path=legacy_plot,
        )

    config = review_calibrated_detection_config_v1()
    return {
        "schema_version": 1,
        "title": "Détecteur yeast — comment naît un candidat ?",
        "subtitle": (
            "Sur une même trace : isoler la bande physique, voir une structure "
            "Doppler, mesurer son énergie, puis décider si elle est atypique."
        ),
        "scientific_precision": {
            "headline": "La méthode ne compte pas les pics Doppler",
            "badge": "agrégation spectrale → regroupement temporel",
            "body": (
                "Elle ne corrèle pas explicitement les pics entre eux. Pour chaque "
                "instant m, elle somme l’excès d’énergie sur les fréquences k afin "
                "d’obtenir une seule courbe E[m], puis elle groupe les trames actives "
                "dans le temps. Le nombre de pics Doppler est mesuré seulement après "
                "la création de la boîte : c’est un descripteur diagnostique, pas "
                "l’unité de comptage des yeasts."
            ),
        },
        "legacy_particles2snr": legacy_particles2snr,
        "cases": cases,
        "pipeline": [
            {
                "number": "01",
                "title": "Passe-bande",
                "question": "Où chercher l’information utile ?",
                "body": (
                    f"Conserver {config.low_freq_hz / 1000:.0f}–"
                    f"{config.high_freq_hz / 1000:.0f} kHz retire les composantes "
                    "hors de la bande physique étudiée."
                ),
            },
            {
                "number": "02",
                "title": "STFT",
                "question": "Quelle fréquence apparaît, et quand ?",
                "body": (
                    f"Fenêtre Hann {config.stft_nperseg}, pas "
                    f"{config.stft_nperseg - config.stft_noverlap}. Une structure "
                    "Doppler localisée indique un passage possible : un candidat "
                    "potentiel, pas encore un GT."
                ),
            },
            {
                "number": "03",
                "title": "Énergie locale",
                "question": "À quels instants la bande s’active-t-elle ?",
                "body": (
                    "Après retrait du fond Q25 par fréquence, sommer la puissance "
                    "transforme la carte 2D en une courbe temporelle E[m]."
                ),
            },
            {
                "number": "04",
                "title": "Médiane + MAD",
                "question": "Cette hausse est-elle atypique pour cette trace ?",
                "body": (
                    "La médiane décrit le bruit typique; la MAD sa dispersion robuste. "
                    "Le score z est relatif à la trace, pas un SNR absolu lu à l’œil."
                ),
            },
            {
                "number": "05",
                "title": "Trames actives",
                "question": "Activité structurée ou simple énergie diffuse ?",
                "body": (
                    f"Activer si z ≥ {config.active_snr_z:g} et si la concentration "
                    f"spectrale ≥ {config.medium_min_concentration:g}. La STFT "
                    "permet ce second contrôle."
                ),
            },
            {
                "number": "06",
                "title": "Regroupement",
                "question": "Quelles trames appartiennent au même passage ?",
                "body": (
                    f"Fusionner les trames séparées de ≤ {config.cluster_gap_ms:g} ms, "
                    "puis étendre les bords : le résultat est une boîte candidate. "
                    "Sa sélection finale vient au livrable suivant."
                ),
            },
        ],
        "math": {
            "STFT — rendre le Doppler visible": (
                "X(k,m) = Σₙ x_f[n] w[n−mH] e^(−j2πkn/N)"
            ),
            "Énergie — localiser dans le temps": (
                "E[m] = Σₖ max(|X(k,m)|² − Q₂₅,k, 0)"
            ),
            "MAD — comparer au bruit de la trace": (
                "z[m] = (E[m] − médiane(E)) / (1,4826 · MAD(E))"
            ),
            "Concentration — vérifier la structure": (
                "C[m] = énergie des 5 bins dominants / énergie totale"
            ),
        },
        "interpretation": {
            "headline": "Un pic visuel n’est pas encore une décision",
            "body": (
                "La STFT propose une localisation physique; l’énergie résume son "
                "intensité; médiane/MAD mesure son caractère inhabituel; la "
                "concentration évite de confondre une hausse diffuse avec une "
                "structure spectrale. La boîte produite reste un candidat."
            ),
        },
        "ablation": {
            "headline": "Pourquoi conserver cette chaîne ?",
            "body": (
                "L’ablation énergie+MAD reproduit l’essentiel, mais la version STFT "
                "sépare mieux plusieurs passages proches et conserve davantage de "
                "passages faibles dans la revue des désaccords (11 préférences pour "
                "l’actuel, 1 pour le simple, 1 incertaine)."
            ),
        },
    }


def render_pipeline_board_html(model: dict[str, Any]) -> str:
    pipeline_cards = "".join(
        f"""
        <article class="step">
          <span>{html.escape(step["number"])}</span>
          <h3>{html.escape(step["title"])}</h3>
          <strong>{html.escape(step["question"])}</strong>
          <p>{html.escape(step["body"])}</p>
        </article>
        """
        for step in model["pipeline"]
    )
    tabs = "".join(
        f'<button class="case-tab" data-case="{case["slug"]}">'
        f'{html.escape(case["eyebrow"])}</button>'
        for case in model["cases"]
    )
    legacy = model.get("legacy_particles2snr")
    legacy_html = ""
    if legacy is not None:
        legacy_steps = "".join(
            f"""
            <article class="legacy-step">
              <span>{html.escape(step["number"])}</span>
              <h3>{html.escape(step["title"])}</h3>
              <p>{html.escape(step["body"])}</p>
            </article>
            """
            for step in legacy["steps"]
        )
        counts = legacy["stage_counts"]
        pair = legacy["final_pair"]
        first, second = legacy["final"]
        legacy_html = f"""
        <section class="legacy-story">
          <div class="legacy-heading">
            <div>
              <div class="legacy-kicker">Pourquoi une pipeline spécifique aux yeasts ?</div>
              <h2>{html.escape(legacy["title"])}</h2>
              <p>{html.escape(legacy["subtitle"])}</p>
            </div>
            <div class="legacy-verdict">
              <span>Résultat réel</span>
              <strong>1 passage humain → 2 boîtes</strong>
              <small>après filtered + dual-clean + NMS</small>
            </div>
          </div>
          <div class="legacy-flow">{legacy_steps}</div>
          <div class="legacy-figure">
            <figure>
              <img src="{html.escape(legacy["plot"])}" alt="Rejeu particles2SNR dual-clean sur le yeast multi-Doppler">
              <figcaption>
                Même trace, mêmes bornes humaines. Les cinq hypothèses affichées sont
                uniquement celles dont le centre tombe dans l’événement contrôlé.
              </figcaption>
            </figure>
          </div>
          <div class="legacy-counts">
            <div><strong>{counts["raw_full_trace"]}</strong><span>hypothèses P0<br>sur la trace entière</span></div>
            <b>→</b>
            <div><strong>{counts["raw_centered_in_human_event"]}</strong><span>hypothèses centrées<br>dans le passage</span></div>
            <b>→</b>
            <div><strong>{counts["after_filtered_peak_evidence"]}</strong><span>après évidence<br>filtered</span></div>
            <b>→</b>
            <div><strong>{counts["after_dual_clean"]}</strong><span>après support<br>dual-clean</span></div>
            <b>→</b>
            <div class="bad-count"><strong>{counts["after_nms"]}</strong><span>boîtes finales<br>pour 1 événement</span></div>
          </div>
          <div class="legacy-explanation">
            <article>
              <span>1 · Origine fréquentielle</span>
              <h3>Deux crêtes réelles, plusieurs maxima exploités</h3>
              <p>La crête haute est reprise à 19,53 kHz. La crête basse à 11,72 kHz
              apparaît à 10,74 kHz sur la grille FFT P0. Des maxima supplémentaires
              à 7,81, 8,79 et 14,65 kHz deviennent eux aussi des hypothèses.</p>
            </article>
            <article>
              <span>2 · Dual-clean aide, mais ne compte pas les événements</span>
              <h3>Trois doublons disparaissent, deux boîtes restent</h3>
              <p>La proposition du pic haut exact à 19,53 kHz est supprimée comme
              doublon du même groupe d’enveloppe. Paradoxalement, les hypothèses
              7,81 et 10,74 kHz survivent : le contrôle améliore les propositions,
              mais ne reconstruit pas l’identité unique du yeast.</p>
            </article>
            <article class="failure">
              <span>3 · Cas limite du NMS</span>
              <h3>IoU {pair["iou"]:.3f} &lt; 0,400</h3>
              <p>Les centres restent séparés de {pair["center_gap_ms"]:.3f} ms.
              Le NMS conserve donc {first["start_ms"]:.3f}–{first["end_ms"]:.3f} ms
              et {second["start_ms"]:.3f}–{second["end_ms"]:.3f} ms : deux fragments
              mal délimités du même événement.</p>
            </article>
          </div>
          <div class="transition">
            <span>Ce que change la pipeline yeast actuelle</span>
            <strong>Elle agrège d’abord l’information fréquentielle, puis compte les groupes temporels.</strong>
            <p>Les deux pics Doppler contribuent à une même courbe d’énergie et ne
            créent donc plus deux identités candidates. Le support validé ci-dessous
            détaille cette logique sans modification.</p>
          </div>
        </section>
        <div class="current-story-title">
          <span>Pipeline actuelle</span>
          <h2>De la trace au groupe temporel yeast</h2>
        </div>
        """

    def mechanism_html(case: dict[str, Any]) -> str:
        mechanism = case["multi_doppler_mechanism"]
        if mechanism is None:
            return ""
        return f"""
        <div class="mechanism">
          <div class="wrong-model">
            <span>✕ Mauvais modèle mental</span>
            <strong>2 pics Doppler → compter les pics → 2 yeasts</strong>
            <small>Ce n’est pas ce que fait le code.</small>
          </div>
          <div class="actual-model">
            <div class="actual-title">✓ Ordre réel de la pipeline</div>
            <div class="mechanism-flow">
              <div class="mechanism-rule">
                <strong>{html.escape(mechanism["peak_summary"])}</strong>
                <span>structures observées dans la STFT</span>
              </div>
              <b>→</b>
              <div class="mechanism-rule">
                <strong>{html.escape(mechanism["aggregation"])}</strong>
                <span>E[m] = Σₖ énergie excédentaire</span>
              </div>
              <b>→</b>
              <div class="mechanism-rule">
                <strong>{html.escape(mechanism["temporal_group"])}</strong>
                <span>c’est ici que le nombre de candidats est fixé</span>
              </div>
              <b>→</b>
              <div class="mechanism-rule result">
                <strong>{html.escape(mechanism["candidate"])}</strong>
                <span>{html.escape(mechanism["counting_rule"])}</span>
              </div>
              <b>→</b>
              <div class="mechanism-rule diagnostic">
                <strong>{html.escape(mechanism["diagnostic"])}</strong>
                <span>description seulement — aucune nouvelle boîte</span>
              </div>
            </div>
          </div>
          <p class="mechanism-limit"><strong>Limite :</strong> {html.escape(mechanism["limit"])}</p>
        </div>
        """

    case_sections = "".join(
        f"""
        <section class="case-panel" id="case-{case["slug"]}" data-case-panel="{case["slug"]}" hidden>
          <div class="case-heading">
            <div>
              <div class="eyebrow">{html.escape(case["eyebrow"])}</div>
              <h2>{html.escape(case["title"])}</h2>
              <p>{html.escape(case["takeaway"])}</p>
            </div>
            <div class="decision {html.escape(case["quality"])}">{html.escape(case["quality"])}</div>
          </div>
          {mechanism_html(case)}
          <div class="case-grid">
            <figure>
              <img src="{html.escape(case["plot"])}" alt="Signal brut, filtré, spectrogramme et score énergétique">
              <figcaption>
                Rouge : centre contrôlé. Vert : candidate strict. Orange : candidate rejetée.
              </figcaption>
            </figure>
            <aside class="evidence">
              <h3>Mesures du candidat</h3>
              <dl>
                <div><dt>Score MAD <small>(pas un SNR classique)</small></dt><dd>{case["snr_proxy"]:.1f}</dd></div>
                <div><dt>Concentration STFT</dt><dd>{case["concentration"]:.3f}</dd></div>
                <div><dt>Pics Doppler décrits</dt><dd>{case["n_doppler_peaks"]}</dd></div>
                <div><dt>Phase <small>(diagnostic)</small></dt><dd>{case["phase_coherence"]:.3f}</dd></div>
                <div><dt>Largeur</dt><dd>{case["width_ms"]:.3f} ms</dd></div>
              </dl>
              <h3>Contrôle humain — contexte</h3>
              <div class="human-verdict">
                Événement dans la boîte :
                <strong>{html.escape(case["human_event_present"])}</strong>
              </div>
              <p class="quote">{html.escape(case["review_notes"] or "Aucun commentaire nécessaire.")}</p>
              <div class="source">
                <strong>{html.escape(case["reviewer"])}</strong><br>
                {html.escape(case["source_group"])} · {html.escape(case["relative_path"])}
              </div>
            </aside>
          </div>
        </section>
        """
        for case in model["cases"]
    )
    math_cards = "".join(
        f"""
        <div><span>{html.escape(label.replace("_", " "))}</span>
        <code>{html.escape(formula)}</code></div>
        """
        for label, formula in model["math"].items()
    )
    embedded = json.dumps(
        {
            "default_case": (
                "one-event-multi-doppler"
                if legacy is not None
                else model["cases"][0]["slug"]
            )
        },
        ensure_ascii=False,
    ).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(model["title"])}</title>
<style>
:root {{ --ink:#17212b; --muted:#63717f; --line:#d9e1e7; --paper:#f4f7f6;
  --green:#0f8a70; --green-pale:#dff3ed; --amber:#d97706; --amber-pale:#fff0d6;
  --blue:#2f6fed; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--paper); color:var(--ink); font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif; }}
header {{ background:#17212b; color:white; padding:34px max(32px,calc((100vw - 1460px)/2)); }}
header .kicker {{ color:#8ee0c9; text-transform:uppercase; letter-spacing:.14em; font-size:12px; font-weight:800; }}
header h1 {{ margin:8px 0 6px; font-size:34px; line-height:1.05; }}
header p {{ margin:0; color:#cbd4dc; max-width:850px; }}
main {{ max-width:1460px; margin:0 auto; padding:26px 28px 42px; }}
.legacy-story {{ background:white; border:1px solid #e2c6bf; border-radius:16px; overflow:hidden;
  box-shadow:0 10px 34px rgba(23,33,43,.08); margin-bottom:26px; }}
.legacy-heading {{ display:flex; justify-content:space-between; gap:28px; align-items:center;
  padding:22px 24px 18px; background:linear-gradient(120deg,#fff8f4,#f5f8f7); border-bottom:1px solid #ead9d4; }}
.legacy-kicker {{ color:#b34736; text-transform:uppercase; letter-spacing:.11em; font-size:11px; font-weight:900; }}
.legacy-heading h2 {{ margin:5px 0 6px; font-size:27px; }}
.legacy-heading p {{ margin:0; color:var(--muted); max-width:900px; }}
.legacy-verdict {{ flex:0 0 275px; background:#fff0ed; color:#8e3428; border:1px solid #e9a99d;
  border-radius:12px; padding:12px 15px; }}
.legacy-verdict span,.legacy-verdict small {{ display:block; font-size:10px; font-weight:850; text-transform:uppercase; letter-spacing:.06em; }}
.legacy-verdict strong {{ display:block; margin:4px 0; font-size:18px; }}
.legacy-flow {{ display:grid; grid-template-columns:repeat(6,1fr); gap:9px; padding:14px 18px 15px; background:#fffdfa; }}
.legacy-step {{ position:relative; border:1px solid #e5d6d0; border-radius:10px; padding:11px; min-height:118px; background:white; }}
.legacy-step:not(:last-child)::after {{ content:"→"; position:absolute; right:-9px; top:43px; color:#b34736; font-weight:900; z-index:2; }}
.legacy-step span {{ color:#b34736; font-size:11px; font-weight:900; }}
.legacy-step h3 {{ margin:4px 0 5px; font-size:14px; }}
.legacy-step p {{ margin:0; color:var(--muted); font-size:11.5px; line-height:1.38; }}
.legacy-figure figure {{ border:0; border-top:1px solid #ead9d4; border-bottom:1px solid #ead9d4; padding:14px 16px 10px; }}
.legacy-counts {{ display:grid; grid-template-columns:1fr auto 1fr auto 1fr auto 1fr auto 1fr; gap:12px;
  align-items:center; padding:14px 24px; background:#f6f8f8; text-align:center; }}
.legacy-counts div {{ display:flex; align-items:center; justify-content:center; gap:10px; }}
.legacy-counts strong {{ font-size:27px; color:#43525d; }}
.legacy-counts span {{ color:var(--muted); font-size:10.5px; line-height:1.25; text-align:left; }}
.legacy-counts b {{ color:#8a969e; }}
.legacy-counts .bad-count strong {{ color:#b34736; }}
.legacy-explanation {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:11px; padding:16px 18px; }}
.legacy-explanation article {{ border:1px solid var(--line); border-radius:11px; padding:13px 14px; background:#fbfcfc; }}
.legacy-explanation article.failure {{ background:#fff0ed; border-color:#e9a99d; }}
.legacy-explanation span {{ color:#b34736; text-transform:uppercase; letter-spacing:.06em; font-size:9.5px; font-weight:900; }}
.legacy-explanation h3 {{ margin:5px 0 7px; font-size:15px; }}
.legacy-explanation p {{ margin:0; color:#475661; font-size:12px; line-height:1.45; }}
.transition {{ margin:0 18px 18px; padding:15px 18px; border:2px solid var(--green); border-radius:12px;
  background:var(--green-pale); display:grid; grid-template-columns:190px 1.15fr 1fr; gap:16px; align-items:center; }}
.transition span {{ color:var(--green); text-transform:uppercase; font-size:10px; letter-spacing:.08em; font-weight:900; }}
.transition strong {{ font-size:15px; line-height:1.35; }}
.transition p {{ margin:0; color:#365248; font-size:11.5px; line-height:1.42; }}
.current-story-title {{ display:flex; align-items:baseline; gap:14px; margin:0 2px 12px; }}
.current-story-title span {{ color:var(--green); text-transform:uppercase; letter-spacing:.1em; font-size:11px; font-weight:900; }}
.current-story-title h2 {{ margin:0; font-size:24px; }}
.pipeline {{ display:grid; grid-template-columns:repeat(6,1fr); gap:10px; margin-bottom:22px; }}
.step {{ position:relative; background:white; border:1px solid var(--line); border-radius:12px; padding:15px 14px 13px; min-height:130px; }}
.step:not(:last-child)::after {{ content:"→"; position:absolute; right:-10px; top:50px; z-index:2; color:var(--green); font-weight:900; }}
.step span {{ color:var(--green); font-weight:900; font-size:12px; }}
.step h3 {{ margin:5px 0; font-size:16px; }}
.step strong {{ display:block; color:var(--ink); font-size:11.5px; line-height:1.3; margin:0 0 6px; }}
.step p {{ margin:0; color:var(--muted); font-size:12.5px; line-height:1.4; }}
.precision {{ display:grid; grid-template-columns:260px 1fr; gap:18px; align-items:center; background:#dff3ed;
  border:2px solid var(--green); border-radius:12px; padding:15px 18px; margin:0 0 14px; }}
.precision .precision-title span {{ display:block; color:var(--green); text-transform:uppercase; letter-spacing:.09em;
  font-size:10px; font-weight:900; margin-bottom:4px; }}
.precision .precision-title strong {{ display:block; font-size:17px; line-height:1.15; }}
.precision p {{ margin:0; color:#30473f; font-size:13px; line-height:1.48; }}
.precision p strong {{ color:#096b57; }}
.tabs {{ display:flex; gap:8px; margin:0 0 10px; }}
.case-tab {{ border:1px solid #aebbc5; background:white; color:var(--ink); padding:10px 16px; border-radius:9px; font-weight:750; cursor:pointer; }}
.case-tab.active {{ background:var(--ink); color:white; border-color:var(--ink); }}
.case-panel {{ background:white; border:1px solid var(--line); border-radius:15px; overflow:hidden; box-shadow:0 9px 30px rgba(23,33,43,.07); }}
.case-heading {{ display:flex; justify-content:space-between; gap:25px; padding:18px 22px 15px; border-bottom:1px solid var(--line); }}
.case-heading h2 {{ margin:3px 0 5px; font-size:24px; }}
.case-heading p {{ margin:0; color:var(--muted); max-width:970px; }}
.eyebrow {{ color:var(--green); font-weight:850; font-size:12px; text-transform:uppercase; letter-spacing:.1em; }}
.decision {{ align-self:center; border-radius:999px; padding:7px 14px; font-weight:900; text-transform:uppercase; font-size:12px; }}
.decision.strict {{ color:#096b57; background:var(--green-pale); }}
.decision.reject {{ color:#9b5500; background:var(--amber-pale); }}
.mechanism {{ padding:14px 22px; background:#f6fbf9; border-bottom:1px solid #b9dfd5; }}
.wrong-model {{ display:grid; grid-template-columns:180px 1fr auto; gap:12px; align-items:center; background:#fff0ed;
  border:1px solid #e9a99d; border-radius:9px; padding:8px 11px; margin-bottom:10px; color:#8e3428; }}
.wrong-model span {{ font-size:10px; font-weight:900; text-transform:uppercase; letter-spacing:.06em; }}
.wrong-model strong {{ font-size:12px; text-decoration:line-through; text-decoration-thickness:1.5px; }}
.wrong-model small {{ font-size:10px; font-weight:800; }}
.actual-model {{ background:#e3f6f0; border:1px solid #91cdbd; border-radius:10px; padding:10px; }}
.actual-title {{ color:#08745f; font-size:11px; font-weight:900; text-transform:uppercase; letter-spacing:.06em; margin:0 0 8px; }}
.mechanism-flow {{ display:grid; grid-template-columns:.85fr auto 1.25fr auto 1.35fr auto .9fr auto 1.05fr; gap:8px; align-items:center; }}
.mechanism-flow > b {{ color:var(--green); font-size:20px; }}
.mechanism-rule {{ background:white; border:1px solid #b9dfd5; border-radius:9px; padding:9px 11px; min-height:62px; }}
.mechanism-rule strong {{ display:block; font-size:12px; line-height:1.3; }}
.mechanism-rule span {{ display:block; color:var(--muted); font-size:10px; margin-top:4px; }}
.mechanism-rule.result {{ background:#d8f2ea; border-color:var(--green); }}
.mechanism-rule.diagnostic {{ background:#f3f0ff; border-color:#a895d1; }}
.mechanism-limit {{ margin:9px 2px 0; color:#475661; font-size:10.5px; }}
.case-grid {{ display:grid; grid-template-columns:minmax(0,1fr) 300px; }}
figure {{ margin:0; padding:12px 12px 9px; border-right:1px solid var(--line); }}
figure img {{ display:block; width:100%; height:auto; }}
figcaption {{ color:var(--muted); font-size:11.5px; padding:4px 8px 0; }}
.evidence {{ padding:20px 18px; background:#fbfcfc; }}
.evidence h3 {{ font-size:14px; margin:0 0 10px; }}
dl {{ margin:0 0 22px; }}
dl div {{ display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid var(--line); }}
dt {{ color:var(--muted); max-width:190px; }} dt small {{ display:block; font-size:9px; }} dd {{ margin:0; font-weight:800; }}
.human-verdict {{ background:var(--green-pale); color:#096b57; border-radius:9px; padding:11px; font-size:13px; }}
.quote {{ color:#3e4b57; font-style:italic; line-height:1.45; border-left:3px solid var(--amber); padding-left:10px; }}
.source {{ color:var(--muted); font-size:11px; line-height:1.45; word-break:break-word; margin-top:22px; }}
.bottom {{ display:grid; grid-template-columns:1.45fr 1fr; gap:14px; margin-top:16px; }}
.math,.message {{ background:white; border:1px solid var(--line); border-radius:12px; padding:16px; }}
.math h2,.message h2 {{ font-size:17px; margin:0 0 10px; }}
.math-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; }}
.math-grid div {{ background:#f5f7f8; padding:9px; border-radius:8px; }}
.math-grid span {{ display:block; color:var(--green); font-size:10.5px; font-weight:850; margin-bottom:4px; }}
.math-grid code {{ white-space:normal; font-size:11px; }}
.message-stack {{ display:grid; gap:10px; }}
.message {{ background:#eef7f4; }}
.message.ablation {{ background:#fff7e8; }}
.message p {{ margin:0; color:#3e4b57; font-size:12.5px; line-height:1.48; }}
footer {{ max-width:1460px; margin:0 auto; padding:0 28px 30px; color:var(--muted); font-size:11px; }}
@media (max-width:1050px) {{
  .legacy-heading {{ align-items:flex-start; flex-direction:column; }}
  .legacy-verdict {{ flex-basis:auto; width:100%; }}
  .legacy-flow {{ grid-template-columns:repeat(2,1fr); }}
  .legacy-counts,.legacy-explanation,.transition {{ grid-template-columns:1fr; }}
  .legacy-counts b {{ transform:rotate(90deg); }}
  .pipeline {{ grid-template-columns:repeat(3,1fr); }}
  .precision {{ grid-template-columns:1fr; }}
  .wrong-model,.mechanism-flow {{ grid-template-columns:1fr; }}
  .mechanism-flow > b {{ transform:rotate(90deg); justify-self:center; }}
  .case-grid,.bottom {{ grid-template-columns:1fr; }}
  figure {{ border-right:0; border-bottom:1px solid var(--line); }}
}}
</style>
</head>
<body>
<header>
  <div class="kicker">Livrable 2 · de la trace à la boîte candidate</div>
  <h1>{html.escape(model["title"])}</h1>
  <p>{html.escape(model["subtitle"])}</p>
</header>
<main>
  {legacy_html}
  <section class="pipeline">{pipeline_cards}</section>
  <section class="precision">
    <div class="precision-title">
      <span>{html.escape(model["scientific_precision"]["badge"])}</span>
      <strong>{html.escape(model["scientific_precision"]["headline"])}</strong>
    </div>
    <p>{html.escape(model["scientific_precision"]["body"])}</p>
  </section>
  <nav class="tabs" aria-label="Cas annotés">{tabs}</nav>
  {case_sections}
  <section class="bottom">
    <article class="math"><h2>Chaque formule répond à une question</h2><div class="math-grid">{math_cards}</div></article>
    <div class="message-stack">
      <article class="message">
        <h2>{html.escape(model["interpretation"]["headline"])}</h2>
        <p>{html.escape(model["interpretation"]["body"])}</p>
      </article>
      <article class="message ablation">
        <h2>{html.escape(model["ablation"]["headline"])}</h2>
        <p>{html.escape(model["ablation"]["body"])}</p>
      </article>
    </div>
  </section>
</main>
<footer>Sources : yeast-event-candidates@v7+v8 · annotations humaines v7+v8 · détecteur review-calibrated-v1.</footer>
<script id="board-config" type="application/json">{embedded}</script>
<script>
const config = JSON.parse(document.getElementById("board-config").textContent);
function showCase(slug) {{
  document.querySelectorAll("[data-case-panel]").forEach(panel => panel.hidden = panel.dataset.casePanel !== slug);
  document.querySelectorAll("[data-case]").forEach(button => button.classList.toggle("active", button.dataset.case === slug));
  const url = new URL(window.location.href); url.searchParams.set("case", slug); history.replaceState(null, "", url);
}}
document.querySelectorAll("[data-case]").forEach(button => button.onclick = () => showCase(button.dataset.case));
const requested = new URLSearchParams(location.search).get("case");
const available = [...document.querySelectorAll("[data-case]")].map(button => button.dataset.case);
showCase(available.includes(requested) ? requested : config.default_case);
</script>
</body>
</html>
"""
