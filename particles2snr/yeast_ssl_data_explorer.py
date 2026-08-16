from __future__ import annotations

import csv
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


YEAST_PREFIX = "yeast-"

VERSION_STORY = (
    {
        "version": "v1",
        "family": "v2",
        "kind": "protocol",
        "title": "Première file de candidats",
        "summary": (
            "Détecteur temps-fréquence initial. La qualité physique est encore "
            "mélangée à la faisabilité d'un crop centré de 8192 points."
        ),
        "change": "Création de la revue par fenêtres candidates.",
    },
    {
        "version": "v2",
        "family": "v2",
        "kind": "protocol",
        "title": "Ajout de la revue des traces",
        "summary": (
            "Même sortie de détection que v1. Une seconde file permet désormais "
            "d'estimer le rappel sur les traces complètes."
        ),
        "change": "Aucun changement d'équation ; protocole humain enrichi.",
    },
    {
        "version": "v3",
        "family": "v5",
        "kind": "processing",
        "title": "Qualité découplée du crop",
        "summary": (
            "La proximité d'un bord devient une métadonnée de padding et ne peut "
            "plus transformer un événement physique en rejet."
        ),
        "change": "5 360 rejets géométriques supprimés.",
    },
    {
        "version": "v4",
        "family": "v5",
        "kind": "split",
        "title": "Splits source-group stratifiés",
        "summary": (
            "Détecteur inchangé. Les capture blocks duplicate-safe sont répartis "
            "avec chaque groupe source présent dans les splits."
        ),
        "change": "Changement de source-index et de split, pas de détecteur.",
    },
    {
        "version": "v5",
        "family": "v5",
        "kind": "annotation",
        "title": "Contrat d'annotation auditable",
        "summary": (
            "Les faux retenus, vrais rejetés et événements entièrement manqués "
            "sont séparés. C'est la revue manuelle de calibration."
        ),
        "change": "Candidats identiques à v4 ; sémantique de revue corrigée.",
    },
    {
        "version": "v6",
        "family": "v7",
        "kind": "processing",
        "title": "Détecteur calibré sur v5",
        "summary": (
            "Seuil SNR renforcé, gap réduit, largeur maximale augmentée et cinq "
            "événements autorisés. Les records v5 sont exclus de la revue."
        ),
        "change": "Première file fraîche, trop petite pour conclure.",
    },
    {
        "version": "v7",
        "family": "v7",
        "kind": "validation",
        "title": "Validation indépendante du preset",
        "summary": (
            "Même détecteur que v6, avec une file plus grande et aucun record "
            "déjà inspecté en v5/v6."
        ),
        "change": "Validation humaine finale sur l'acquisition disponible.",
    },
    {
        "version": "v9",
        "family": "v9",
        "kind": "processing",
        "title": "Boîtes réduites aux frames actives",
        "summary": (
            "Extension des bornes et pad de 0,04 ms retirés, comme le contrat "
            "MAD v2.1 publié. Mêmes 12 271 candidats sur les mêmes traces ; "
            "largeur médiane 0,784 → 0,640 ms. L'extension faisait grandir deux "
            "groupes voisins l'un dans l'autre : v7 portait 143 traces à bornes "
            "dupliquées et 990 paires chevauchantes, v9 aucune."
        ),
        "change": "Aucun seuil déplacé ; deux mécanismes d'élargissement retirés.",
    },
)

FAMILY_LABELS = {
    "v2": "Rejet géométrique",
    "v5": "Avant calibration",
    "v7": "Calibré et validé",
    "v9": "Boîtes non élargies",
}

SCIENTIFIC_ROLES = {
    "yeast-hf-10-5-20260610@v1": ("source", "Source brute immuable"),
    "yeast-source-index@v1": ("metadata", "Index historique"),
    "yeast-source-index@v2": ("metadata", "Index canonique et splits"),
    "yeast-event-candidates@v1": ("candidate", "Historique : première file"),
    "yeast-event-candidates@v2": ("candidate", "Historique : revue trace ajoutée"),
    "yeast-event-candidates@v3": ("candidate", "Historique : crop découplé"),
    "yeast-event-candidates@v4": ("candidate", "Historique : nouveaux splits"),
    "yeast-event-candidates@v5": ("calibration", "Calibration manuelle"),
    "yeast-event-candidates@v6": ("candidate", "Validation sous-dimensionnée"),
    "yeast-event-candidates@v7": ("legacy", "Remplacé par v9 : boîtes élargies"),
    "yeast-event-candidates@v9": ("validation", "Détecteur gelé, boîtes non élargies"),
    "yeast-event-review-annotations@v1": ("annotation", "Annotations v7 arbitrées"),
    "yeast-events-representation@v1": ("representation", "Entrée SSL historique v3"),
    "yeast-events-representation@v2": ("representation", "Entrée SSL pré-calibration"),
    "yeast-events-representation@v3": ("legacy", "Remplacé par v4 : 108 boîtes dupliquées"),
    "yeast-events-representation@v4": ("representation", "Entrée SSL canonique"),
    "yeast-events-development@v1": ("representation", "Sous-ensemble développement"),
    "yeast-events-followup@v1": ("legacy", "Follow-up historique"),
    "yeast-events-followup@v2": ("representation", "Follow-up partitionné"),
    "yeast-passage-simulations@v1": (
        "legacy",
        "Simulation historique — résultats à réévaluer",
    ),
    "yeast-passage-simulations@v2": (
        "simulation",
        "Simulation active unique — support corrigé",
    ),
    "yeast-template-comparator@v1": ("legacy", "Comparateur historique"),
    "yeast-template-comparator@v2": ("diagnostic", "Contrôle template non physique"),
}

LINEAGE = (
    ("yeast-hf-10-5-20260610@v1", "yeast-source-index@v2", "indexe"),
    ("yeast-source-index@v2", "yeast-event-candidates@v5", "détecte"),
    ("yeast-event-candidates@v5", "yeast-event-candidates@v7", "calibre"),
    ("yeast-event-candidates@v7", "yeast-event-review-annotations@v1", "validé par"),
    ("yeast-event-candidates@v3", "yeast-events-representation@v1", "produit"),
    ("yeast-event-candidates@v4", "yeast-events-representation@v2", "produit"),
    ("yeast-event-candidates@v7", "yeast-events-representation@v3", "produit"),
    ("yeast-event-candidates@v7", "yeast-event-candidates@v9", "remplacé par"),
    ("yeast-event-candidates@v9", "yeast-events-representation@v4", "produit"),
    ("yeast-events-representation@v3", "yeast-events-development@v1", "filtre"),
    ("yeast-events-representation@v3", "yeast-events-followup@v2", "repartitionne"),
    ("yeast-passage-simulations@v2", "SSL A2/A3/A4", "entraîne"),
    ("yeast-events-representation@v4", "SSL A0–A4", "évalue/adapte"),
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dataset_key(payload: dict[str, Any]) -> str:
    return f"{payload['id']}@{payload['version']}"


def _summary_paths(dataset_path: Path) -> Iterable[Path]:
    for name in (
        "dataset_summary.json",
        "candidate_audit_summary.json",
        "review_analysis.json",
        "split_audit.json",
        "input_contract.json",
    ):
        path = dataset_path / name
        if path.is_file():
            yield path


def _compact_size(key: str, summaries: dict[str, dict[str, Any]]) -> tuple[str, str]:
    merged: dict[str, Any] = {}
    for payload in summaries.values():
        merged.update(payload)
    candidates = (
        ("n_raw_rows", "traces brutes"),
        ("n_canonical_rows", "traces canoniques"),
        ("n_candidates", "candidats"),
        ("n_events", "événements"),
        ("n_signals", "signaux"),
        ("n_source_events", "événements source"),
    )
    for field, unit in candidates:
        if field in merged:
            return f"{int(merged[field]):,}".replace(",", " "), unit
    candidate_review = merged.get("candidate_review")
    full_review = merged.get("full_trace_review")
    if isinstance(candidate_review, dict) and isinstance(full_review, dict):
        total = int(candidate_review.get("n_expected", 0)) + int(full_review.get("n_expected", 0))
        return str(total), "décisions humaines"
    return "—", "taille documentée"


def build_milestone1_model(
    records: Iterable[dict[str, Any]],
    datasets_root: Path,
) -> dict[str, Any]:
    catalog: list[dict[str, Any]] = []
    summaries_by_key: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        if not str(record.get("id", "")).startswith(YEAST_PREFIX):
            continue
        key = _dataset_key(record)
        dataset_path = datasets_root / str(record["path"])
        summaries = {path.name: _read_json(path) for path in _summary_paths(dataset_path)}
        summaries_by_key[key] = summaries
        role, role_label = SCIENTIFIC_ROLES.get(key, ("other", "Dataset auxiliaire"))
        size, size_unit = _compact_size(key, summaries)
        catalog.append(
            {
                "key": key,
                "id": record["id"],
                "version": record["version"],
                "registry_status": record["status"],
                "role": role,
                "role_label": role_label,
                "format": record["format"],
                "producer": record.get("producer", {}).get("command", ""),
                "path": record["path"],
                "manifest": record["manifest"],
                "manifest_sha256": record.get("manifest_sha256", ""),
                "file_count": record.get("file_count"),
                "size": size,
                "size_unit": size_unit,
            }
        )
    catalog.sort(key=lambda row: row["key"])

    source_summary = summaries_by_key["yeast-source-index@v2"]["dataset_summary.json"]
    stages = []
    for version in ("v2", "v5", "v7", "v9"):
        key = f"yeast-event-candidates@{version}"
        summary = summaries_by_key[key]["candidate_audit_summary.json"]
        qualities = summary["candidate_quality_counts"]
        stages.append(
            {
                "version": version,
                "family": FAMILY_LABELS[version],
                "n_candidates": int(summary["n_candidates"]),
                "retained": int(qualities.get("strict", 0) + qualities.get("medium", 0)),
                "strict": int(qualities.get("strict", 0)),
                "medium": int(qualities.get("medium", 0)),
                "rejected": int(qualities.get("reject", 0)),
                "config": summary["detection_config"],
                "rejection_geometry": 5360 if version == "v2" else 0,
            }
        )

    representation_counts = {}
    for version in ("v1", "v2", "v3", "v4"):
        summary = summaries_by_key[f"yeast-events-representation@{version}"][
            "dataset_summary.json"
        ]
        representation_counts[version] = int(summary["n_events"])

    roles = Counter(row["role"] for row in catalog)
    return {
        "schema_version": 1,
        "milestone": "m1",
        "title": "Yeast SSL — données, provenance et évolution",
        "headline": {
            "raw_traces": int(source_summary["n_raw_rows"]),
            "canonical_traces": int(source_summary["n_canonical_rows"]),
            "duplicate_excess": int(source_summary["n_duplicate_excess_rows"]),
            "acquisitions": len(source_summary["documented_acquisition_ids"]),
            "registered_datasets": len(catalog),
        },
        "scientific_limits": {
            "single_acquisition": not bool(source_summary["acquisition_ood_ready"]),
            "condition_labels": source_summary["scientific_limit"],
            "reviewer_reliability": "La répétition indépendante v7 reste à compléter.",
        },
        "timeline": list(VERSION_STORY),
        "stages": stages,
        "representations": representation_counts,
        "catalog": catalog,
        "catalog_role_counts": dict(sorted(roles.items())),
        "lineage": [
            {"source": source, "target": target, "relation": relation}
            for source, target, relation in LINEAGE
        ],
        "equations": [
            {
                "name": "Puissance temps-fréquence",
                "formula": "P(f,t) = |STFT{x filtré}(f,t)|²",
                "detail": "Butterworth 7–80 kHz, fenêtre Hann 512, overlap 384.",
            },
            {
                "name": "Énergie excédentaire",
                "formula": "E(t) = Σf max(P(f,t) − Q25(P(f,·)), 0)",
                "detail": "Baseline robuste indépendante pour chaque fréquence.",
            },
            {
                "name": "Score robuste",
                "formula": "z(t) = [E(t) − médiane(E)] / [1.4826 × MAD(E)]",
                "detail": "Frame active si z ≥ 3.5 et concentration ≥ 0.08.",
            },
            {
                "name": "Qualité candidat",
                "formula": "retenu = SNR, concentration et largeur admissibles",
                "detail": "Les seuils de qualité changent entre v5 et v7, pas la définition de z.",
            },
        ],
    }


def _json_for_html(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")


def render_milestone1_html(model: dict[str, Any]) -> str:
    data = _json_for_html(model)
    title = html.escape(model["title"])
    return f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #18201f;
      --muted: #64706d;
      --paper: #f4f1e9;
      --card: #fffdf7;
      --line: #d9d2c4;
      --green: #176f5b;
      --green-soft: #dcece5;
      --orange: #c56b2d;
      --orange-soft: #f5e3d3;
      --blue: #285f83;
      --blue-soft: #ddeaf1;
      --red: #a23f35;
      --shadow: 0 10px 30px rgba(31, 43, 39, .08);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{ margin: 0; color: var(--ink); background: var(--paper); }}
    header {{ padding: 42px max(28px, calc((100vw - 1400px) / 2)); background: #172824; color: #fff; }}
    header .eyebrow {{ color: #8fd0bd; text-transform: uppercase; font-size: 12px; font-weight: 750; letter-spacing: .16em; }}
    h1 {{ max-width: 920px; margin: 10px 0 12px; font-size: clamp(34px, 5vw, 65px); line-height: .98; letter-spacing: -.045em; }}
    header p {{ max-width: 830px; color: #d4dfdb; font-size: 17px; line-height: 1.55; }}
    nav {{ position: sticky; top: 0; z-index: 10; display: flex; gap: 6px; padding: 10px max(20px, calc((100vw - 1400px) / 2)); overflow-x: auto; background: rgba(255,253,247,.96); border-bottom: 1px solid var(--line); backdrop-filter: blur(12px); }}
    nav a {{ color: var(--ink); text-decoration: none; font-size: 13px; font-weight: 700; white-space: nowrap; padding: 8px 11px; border-radius: 8px; }}
    nav a:hover {{ background: var(--green-soft); color: var(--green); }}
    main {{ max-width: 1400px; margin: 0 auto; padding: 28px 28px 72px; }}
    section {{ scroll-margin-top: 76px; margin: 32px 0 62px; }}
    .section-kicker {{ color: var(--green); font-size: 12px; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; }}
    h2 {{ margin: 5px 0 8px; font-size: clamp(25px, 3vw, 38px); letter-spacing: -.035em; }}
    .section-intro {{ max-width: 850px; color: var(--muted); line-height: 1.55; }}
    .cards {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; margin-top: 18px; }}
    .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 18px; box-shadow: var(--shadow); }}
    .metric {{ font-size: 32px; font-weight: 800; letter-spacing: -.04em; }}
    .label {{ margin-top: 4px; color: var(--muted); font-size: 13px; line-height: 1.35; }}
    .notice {{ display: grid; grid-template-columns: auto 1fr; gap: 12px; padding: 17px; margin-top: 16px; border: 1px solid #e0bd9f; border-radius: 12px; background: var(--orange-soft); }}
    .notice strong {{ color: #7f3d17; }}
    .timeline {{ display: grid; grid-template-columns: repeat(7, minmax(150px, 1fr)); gap: 10px; margin-top: 22px; overflow-x: auto; padding: 6px 2px 20px; }}
    .step {{ position: relative; min-height: 360px; padding: 16px; border: 1px solid var(--line); border-top: 5px solid var(--green); border-radius: 12px; background: var(--card); }}
    .step[data-family="v2"] {{ border-top-color: var(--red); }}
    .step[data-family="v5"] {{ border-top-color: var(--orange); }}
    .step[data-family="v7"] {{ border-top-color: var(--green); }}
    .step .version {{ font-size: 24px; font-weight: 850; }}
    .step .kind {{ float: right; padding: 4px 7px; border-radius: 99px; background: #ece8df; color: var(--muted); font-size: 10px; font-weight: 750; text-transform: uppercase; }}
    .step h3 {{ min-height: 46px; margin: 14px 0 8px; font-size: 16px; }}
    .step p {{ color: var(--muted); font-size: 13px; line-height: 1.48; }}
    .step .change {{ position: absolute; left: 16px; right: 16px; bottom: 16px; padding-top: 10px; border-top: 1px solid var(--line); color: var(--ink); font-size: 12px; font-weight: 700; }}
    .equations {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-top: 20px; }}
    .equation {{ padding: 18px; border-radius: 12px; background: #172824; color: #fff; }}
    .equation h3 {{ margin: 0 0 14px; color: #9bd4c3; font-size: 13px; }}
    .formula {{ min-height: 56px; font-family: "SFMono-Regular", Consolas, monospace; font-size: 14px; line-height: 1.45; }}
    .equation p {{ color: #c9d5d1; font-size: 12px; line-height: 1.45; }}
    .families {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-top: 20px; }}
    .family {{ overflow: hidden; padding: 0; }}
    .family .family-head {{ padding: 17px; background: var(--blue-soft); }}
    .family:nth-child(2) .family-head {{ background: var(--orange-soft); }}
    .family:nth-child(3) .family-head {{ background: var(--green-soft); }}
    .family h3 {{ margin: 0; }}
    .family .stats {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; padding: 17px; }}
    .family .stats b {{ display: block; font-size: 22px; }}
    .params {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    .params th, .params td {{ padding: 9px 17px; border-top: 1px solid var(--line); text-align: left; }}
    .lineage {{ margin-top: 18px; padding: 18px; border-radius: 14px; background: var(--card); border: 1px solid var(--line); }}
    .edge {{ display: grid; grid-template-columns: minmax(220px, 1fr) 120px minmax(220px, 1fr); align-items: center; gap: 10px; padding: 7px 0; }}
    .node {{ padding: 9px 12px; border: 1px solid #b9cfc7; border-radius: 8px; background: #eef7f3; font-family: "SFMono-Regular", Consolas, monospace; font-size: 12px; overflow-wrap: anywhere; }}
    .relation {{ color: var(--muted); text-align: center; font-size: 11px; font-weight: 750; }}
    .relation::after {{ content: "  →"; color: var(--green); }}
    .toolbar {{ display: grid; grid-template-columns: minmax(220px, 1fr) 180px 180px; gap: 10px; margin: 18px 0 12px; }}
    input, select {{ width: 100%; min-height: 42px; border: 1px solid #bcb6aa; border-radius: 9px; background: var(--card); padding: 9px 11px; color: var(--ink); font: inherit; }}
    .table-wrap {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 12px; background: var(--card); }}
    table.catalog {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    .catalog th {{ position: sticky; top: 0; background: #e9e5db; text-align: left; padding: 11px; }}
    .catalog td {{ border-top: 1px solid var(--line); padding: 11px; vertical-align: top; }}
    .catalog .dataset {{ font-family: "SFMono-Regular", Consolas, monospace; font-weight: 750; }}
    .badge {{ display: inline-block; padding: 3px 7px; border-radius: 99px; background: #e9e5db; font-size: 10px; font-weight: 800; }}
    .badge.active {{ background: var(--green-soft); color: var(--green); }}
    .badge.reference {{ background: var(--blue-soft); color: var(--blue); }}
    .badge.legacy {{ background: #eee; color: #777; }}
    .empty {{ padding: 24px; text-align: center; color: var(--muted); }}
    footer {{ padding: 26px; border-top: 1px solid var(--line); color: var(--muted); text-align: center; font-size: 12px; }}
    body[data-capture="timeline"] header,
    body[data-capture="timeline"] nav,
    body[data-capture="timeline"] section:not(#chronologie),
    body[data-capture="timeline"] footer {{ display: none; }}
    body[data-capture="timeline"] main {{ max-width: 1560px; padding-top: 16px; }}
    body[data-capture="timeline"] #chronologie {{ margin: 0; }}
    body[data-capture="timeline"] .step {{ min-height: 380px; }}
    @media (max-width: 1000px) {{
      .cards {{ grid-template-columns: repeat(2, 1fr); }}
      .equations, .families {{ grid-template-columns: 1fr 1fr; }}
      .toolbar {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 650px) {{
      main {{ padding: 20px 14px 55px; }}
      header {{ padding: 32px 20px; }}
      .cards, .equations, .families {{ grid-template-columns: 1fr; }}
      .edge {{ grid-template-columns: 1fr; }}
      .relation {{ text-align: left; padding-left: 10px; }}
      .timeline {{ grid-template-columns: repeat(7, 250px); }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="eyebrow">Jalon 1 · inventaire et chronologie</div>
    <h1>{title}</h1>
    <p>Un état des lieux traçable avant toute présentation : ce qui est brut, ce qui est dérivé, ce qui a été annoté et ce qui a réellement changé entre les itérations.</p>
  </header>
  <nav>
    <a href="#vue-ensemble">Vue d'ensemble</a>
    <a href="#chronologie">Chronologie v1→v9</a>
    <a href="#equations">Équations</a>
    <a href="#familles">Familles comparées</a>
    <a href="#filiation">Filiation</a>
    <a href="#catalogue">Catalogue</a>
  </nav>
  <main>
    <section id="vue-ensemble">
      <div class="section-kicker">Acquisition disponible</div>
      <h2>Une source réelle, plusieurs couches de preuve</h2>
      <p class="section-intro">Les compteurs ci-dessous viennent des datasets enregistrés, pas d'une reconstruction ad hoc.</p>
      <div id="headline" class="cards"></div>
      <div class="notice">
        <strong>Limite scientifique</strong>
        <div>Une seule acquisition documentée. Les groupes <code>budding</code>, <code>mix</code>, <code>shmoo</code> et <code>shmoo2</code> sont des conditions d'acquisition, pas des ground truths biologiques indépendantes.</div>
      </div>
    </section>
    <section id="chronologie">
      <div class="section-kicker">Pourquoi chaque version existe</div>
      <h2>La chronologie distingue code, données et protocole</h2>
      <p class="section-intro">Une nouvelle version de dataset ne signifie pas toujours une nouvelle équation de détection.</p>
      <div id="timeline" class="timeline"></div>
    </section>
    <section id="equations">
      <div class="section-kicker">Cœur du traitement</div>
      <h2>Ce qui est calculé sur chaque trace</h2>
      <div id="equation-cards" class="equations"></div>
    </section>
    <section id="familles">
      <div class="section-kicker">Comparateurs retenus</div>
      <h2>Trois familles méthodologiques</h2>
      <p class="section-intro">v2, v5 et v7 sont les colonnes qui seront comparées sur les mêmes segments aux jalons suivants.</p>
      <div id="family-cards" class="families"></div>
    </section>
    <section id="filiation">
      <div class="section-kicker">Provenance</div>
      <h2>Deux relations à ne pas confondre</h2>
      <p class="section-intro"><strong>Produit à partir de</strong> décrit la transformation des données ; <strong>validé par</strong> relie un dataset à une preuve humaine sans prétendre que chaque événement a été annoté.</p>
      <div id="lineage" class="lineage"></div>
    </section>
    <section id="catalogue">
      <div class="section-kicker">Registre yeast SSL</div>
      <h2>Tous les datasets visibles au même endroit</h2>
      <div class="toolbar">
        <input id="search" type="search" placeholder="Rechercher un ID, rôle ou format">
        <select id="role-filter"><option value="">Tous les rôles</option></select>
        <select id="status-filter"><option value="">Tous les statuts registre</option></select>
      </div>
      <div class="table-wrap">
        <table class="catalog">
          <thead><tr><th>Dataset</th><th>Rôle scientifique</th><th>Statut registre</th><th>Taille</th><th>Format / production</th></tr></thead>
          <tbody id="catalog-body"></tbody>
        </table>
      </div>
    </section>
  </main>
  <footer>Explorateur local en lecture seule · aucune annotation ou donnée n'est modifiée.</footer>
  <script id="payload" type="application/json">{data}</script>
  <script>
    document.body.dataset.capture = new URLSearchParams(window.location.search).get("capture") || "";
    const data = JSON.parse(document.getElementById("payload").textContent);
    const fmt = value => new Intl.NumberFormat("fr-FR").format(value);
    const headlineLabels = {{
      raw_traces: "traces brutes",
      canonical_traces: "traces canoniques",
      duplicate_excess: "doublons excédentaires",
      acquisitions: "acquisition documentée",
      registered_datasets: "datasets yeast enregistrés"
    }};
    document.getElementById("headline").innerHTML = Object.entries(data.headline).map(([key, value]) =>
      `<article class="card"><div class="metric">${{fmt(value)}}</div><div class="label">${{headlineLabels[key]}}</div></article>`
    ).join("");
    document.getElementById("timeline").innerHTML = data.timeline.map(item =>
      `<article class="step" data-family="${{item.family}}">
        <span class="kind">${{item.kind}}</span><div class="version">${{item.version}}</div>
        <h3>${{item.title}}</h3><p>${{item.summary}}</p><div class="change">${{item.change}}</div>
      </article>`
    ).join("");
    document.getElementById("equation-cards").innerHTML = data.equations.map(item =>
      `<article class="equation"><h3>${{item.name}}</h3><div class="formula">${{item.formula}}</div><p>${{item.detail}}</p></article>`
    ).join("");
    const params = [
      ["Seuil SNR strict", "strict_min_snr"],
      ["Gap de regroupement (ms)", "cluster_gap_ms"],
      ["Largeur maximale (ms)", "max_width_ms"],
      ["Événements max / trace", "max_events_per_signal"]
    ];
    document.getElementById("family-cards").innerHTML = data.stages.map(stage =>
      `<article class="card family">
        <div class="family-head"><h3>${{stage.version}} · ${{stage.family}}</h3></div>
        <div class="stats">
          <div><b>${{fmt(stage.n_candidates)}}</b><span class="label">candidats</span></div>
          <div><b>${{fmt(stage.retained)}}</b><span class="label">retenus</span></div>
          <div><b>${{fmt(stage.rejected)}}</b><span class="label">rejetés</span></div>
          <div><b>${{fmt(stage.rejection_geometry)}}</b><span class="label">rejets géométriques</span></div>
        </div>
        <table class="params">${{params.map(([label, key]) => `<tr><th>${{label}}</th><td>${{stage.config[key]}}</td></tr>`).join("")}}</table>
      </article>`
    ).join("");
    document.getElementById("lineage").innerHTML = data.lineage.map(edge =>
      `<div class="edge"><div class="node">${{edge.source}}</div><div class="relation">${{edge.relation}}</div><div class="node">${{edge.target}}</div></div>`
    ).join("");
    const roleFilter = document.getElementById("role-filter");
    const statusFilter = document.getElementById("status-filter");
    [...new Set(data.catalog.map(row => row.role))].sort().forEach(role => roleFilter.add(new Option(role, role)));
    [...new Set(data.catalog.map(row => row.registry_status))].sort().forEach(status => statusFilter.add(new Option(status, status)));
    function renderCatalog() {{
      const query = document.getElementById("search").value.trim().toLowerCase();
      const role = roleFilter.value, status = statusFilter.value;
      const rows = data.catalog.filter(row =>
        (!query || `${{row.key}} ${{row.role_label}} ${{row.format}} ${{row.producer}}`.toLowerCase().includes(query)) &&
        (!role || row.role === role) && (!status || row.registry_status === status)
      );
      document.getElementById("catalog-body").innerHTML = rows.length ? rows.map(row =>
        `<tr>
          <td><div class="dataset">${{row.key}}</div><div class="label">${{row.path}}</div></td>
          <td><strong>${{row.role_label}}</strong><div class="label">${{row.role}}</div></td>
          <td><span class="badge ${{row.registry_status}}">${{row.registry_status}}</span></td>
          <td><strong>${{row.size}}</strong><div class="label">${{row.size_unit}}</div></td>
          <td><strong>${{row.format}}</strong><div class="label">${{row.producer}}</div></td>
        </tr>`
      ).join("") : `<tr><td class="empty" colspan="5">Aucun dataset ne correspond aux filtres.</td></tr>`;
    }}
    for (const id of ["search", "role-filter", "status-filter"]) document.getElementById(id).addEventListener("input", renderCatalog);
    renderCatalog();
  </script>
</body>
</html>
"""


PRESERVED_SIMULATION_FACTORS = (
    "duration_ms",
    "doppler_khz",
    "component_count",
    "component_separation_ms",
    "relative_component_amplitude",
    "frequency_separation_khz",
)

NUISANCE_SIMULATION_FACTORS = (
    "phase_rad",
    "event_position_fraction",
    "snr_db",
    "target_rms",
    "baseline_drift",
    "sensor_response",
)


def _simulation_rows(path: Path) -> list[dict[str, Any]]:
    integer_fields = {"signal_row", "view_index", "component_count"}
    text_fields = {
        "latent_id",
        "split",
        "generator_variant",
        "envelope_model",
    }
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    converted: list[dict[str, Any]] = []
    for row in rows:
        converted.append(
            {
                key: (
                    value
                    if key in text_fields
                    else int(value)
                    if key in integer_fields
                    else float(value)
                )
                for key, value in row.items()
            }
        )
    return converted


def _median_case(
    rows: list[dict[str, Any]],
    *,
    split: str,
    generator_variant: str,
    component_count: int,
) -> dict[str, Any]:
    candidates = [
        row
        for row in rows
        if row["view_index"] == 0
        and row["split"] == split
        and row["generator_variant"] == generator_variant
        and row["component_count"] == component_count
    ]
    if not candidates:
        raise ValueError(
            f"No simulation rows for {split=}, {generator_variant=}, "
            f"{component_count=}"
        )
    factors = ["duration_ms", "doppler_khz"]
    if component_count == 2:
        factors.extend(
            [
                "component_separation_ms",
                "relative_component_amplitude",
                "frequency_separation_khz",
            ]
        )
    values = np.asarray(
        [[float(row[factor]) for factor in factors] for row in candidates],
        dtype=np.float64,
    )
    medians = np.median(values, axis=0)
    q25, q75 = np.quantile(values, [0.25, 0.75], axis=0)
    scales = np.where((q75 - q25) > 0, q75 - q25, 1.0)
    distances = np.sum(((values - medians) / scales) ** 2, axis=1)
    best_distance = float(np.min(distances))
    tied = [
        row
        for row, distance in zip(candidates, distances, strict=True)
        if np.isclose(float(distance), best_distance)
    ]
    return min(tied, key=lambda row: str(row["latent_id"]))


def build_simulation_checkpoint_model(dataset_root: Path) -> dict[str, Any]:
    """Build a deterministic three-case view of registered yeast simulations."""
    summary = _read_json(dataset_root / "dataset_summary.json")
    factor_policy = _read_json(dataset_root / "factor_policy.json")
    rows = _simulation_rows(dataset_root / "simulation_metadata.csv")
    signals = np.load(dataset_root / "signals.npy", mmap_mode="r")
    if list(signals.shape) != [summary["n_signals"], 4096]:
        raise ValueError(
            f"Signal shape {list(signals.shape)} disagrees with registered summary"
        )

    by_latent: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_latent.setdefault(str(row["latent_id"]), []).append(row)

    specifications = (
        {
            "id": "typical-single",
            "label": "Passage simple typique",
            "split": "train",
            "variant": "base",
            "component_count": 1,
            "why": (
                "Cas à une composante le plus proche des médianes robustes "
                "durée–Doppler du train."
            ),
        },
        {
            "id": "typical-double",
            "label": "Passage double typique",
            "split": "train",
            "variant": "base",
            "component_count": 2,
            "why": (
                "Cas à deux composantes le plus proche des médianes robustes "
                "des cinq facteurs physiques du train."
            ),
        },
        {
            "id": "heldout-sensor",
            "label": "Capteur tenu à part",
            "split": "test",
            "variant": "heldout_sensor",
            "component_count": 2,
            "why": (
                "Cas à deux composantes le plus proche des médianes robustes "
                "du variant capteur réservé au test."
            ),
        },
    )

    cases = []
    for specification in specifications:
        selected = _median_case(
            rows,
            split=str(specification["split"]),
            generator_variant=str(specification["variant"]),
            component_count=int(specification["component_count"]),
        )
        paired = sorted(
            by_latent[str(selected["latent_id"])],
            key=lambda row: int(row["view_index"]),
        )
        if len(paired) != 2:
            raise ValueError(f"Expected two views for {selected['latent_id']}")
        cases.append(
            {
                **specification,
                "latent_id": selected["latent_id"],
                "preserved": {
                    factor: selected[factor]
                    for factor in PRESERVED_SIMULATION_FACTORS
                },
                "views": [
                    {
                        "view_index": row["view_index"],
                        "signal_row": row["signal_row"],
                        "nuisances": {
                            factor: row[factor]
                            for factor in NUISANCE_SIMULATION_FACTORS
                        },
                        "signal_decimation": 4,
                        "signal": [
                            round(float(value), 6)
                            for value in signals[int(row["signal_row"]), ::4]
                        ],
                    }
                    for row in paired
                ],
            }
        )

    return {
        "schema_version": 1,
        "milestone": "simulation-visual-checkpoint-1",
        "title": "Yeast SSL — trois passages simulés",
        "dataset": "yeast-passage-simulations@v2",
        "summary": summary,
        "factor_policy": factor_policy,
        "cases": cases,
        "display_contract": {
            "sampling_frequency_hz": 1_000_000,
            "full_samples": 4096,
            "duration_ms": 4.096,
            "decimation": 4,
            "decimation_method": "un échantillon sur quatre, sans autre filtrage visuel",
            "scale": "échelle verticale commune aux deux vues d'un même latent",
        },
        "claim_boundary": (
            "Ces cas illustrent la construction multi-vue et la couverture de "
            "facteurs connus. Ils ne valident ni le réalisme biologique, ni le "
            "support commun simulation–réel."
        ),
        "case_selection": {
            "population": (
                "7 000 latents / 14 000 signaux de "
                "yeast-passage-simulations@v2"
            ),
            "rule": (
                "Sélection par distance robuste aux médianes des facteurs, "
                "dans trois strates prédéclarées : train simple, train double, "
                "test heldout_sensor double."
            ),
            "selected_ids": [case["latent_id"] for case in cases],
            "selected_before_rendering": True,
        },
    }


def render_simulation_checkpoint_html(model: dict[str, Any]) -> str:
    data = _json_for_html(model)
    title = html.escape(model["title"])
    return f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17201f; --muted: #66716e; --paper: #f3f0e8;
      --card: #fffdf8; --line: #d8d1c3; --green: #176f5b;
      --green-soft: #dcece5; --blue: #2c6386; --orange: #c26a2d;
      --shadow: 0 12px 34px rgba(31, 43, 39, .09);
      font-family: Inter, ui-sans-serif, system-ui, sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; color: var(--ink); background: var(--paper); }}
    header {{ padding: 34px max(24px, calc((100vw - 1220px)/2)); background: #172824; color: white; }}
    .eyebrow {{ color: #96d8c3; font-size: 12px; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; }}
    h1 {{ margin: 8px 0 10px; font-size: clamp(34px, 5vw, 58px); letter-spacing: -.045em; }}
    header p {{ max-width: 850px; margin: 0; color: #d4dfdb; line-height: 1.55; }}
    main {{ max-width: 1220px; margin: auto; padding: 24px 24px 64px; }}
    .summary {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 0 0 20px; }}
    .metric, .panel {{ border: 1px solid var(--line); border-radius: 14px; background: var(--card); box-shadow: var(--shadow); }}
    .metric {{ padding: 15px; }}
    .metric b {{ display: block; font-size: 28px; letter-spacing: -.04em; }}
    .metric span {{ color: var(--muted); font-size: 12px; }}
    .selector {{ display: grid; grid-template-columns: 1fr auto; gap: 14px; align-items: end; padding: 18px; }}
    label {{ display: block; margin-bottom: 7px; color: var(--muted); font-size: 12px; font-weight: 750; text-transform: uppercase; letter-spacing: .08em; }}
    select {{ width: 100%; min-height: 44px; padding: 9px 12px; border: 1px solid #aaa397; border-radius: 9px; background: white; font: inherit; }}
    .dataset {{ font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 12px; color: var(--green); }}
    .case-head {{ display: grid; grid-template-columns: 1fr auto; gap: 14px; padding: 20px; border-bottom: 1px solid var(--line); }}
    h2 {{ margin: 0 0 6px; font-size: 26px; letter-spacing: -.025em; }}
    .why {{ margin: 0; color: var(--muted); line-height: 1.5; }}
    .chip {{ align-self: start; padding: 7px 10px; border-radius: 99px; background: var(--green-soft); color: var(--green); font-size: 11px; font-weight: 800; }}
    .factor-grid {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 1px; background: var(--line); border-bottom: 1px solid var(--line); }}
    .factor {{ padding: 13px; background: var(--card); }}
    .factor b {{ display: block; font-size: 16px; }}
    .factor span {{ display: block; color: var(--muted); font-size: 10px; overflow-wrap: anywhere; }}
    .views {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; padding: 18px; }}
    .view {{ border: 1px solid var(--line); border-radius: 11px; overflow: hidden; background: white; }}
    .view-head {{ display: flex; justify-content: space-between; padding: 11px 13px; background: #eef5f2; }}
    .view:nth-child(2) .view-head {{ background: #edf3f7; }}
    canvas {{ display: block; width: 100%; height: 250px; background: #fbfaf6; }}
    .nuisances {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 7px; padding: 12px; }}
    .nuisance {{ min-width: 0; padding: 7px; border-radius: 7px; background: #f3f1eb; }}
    .nuisance b {{ display: block; font-size: 11px; overflow-wrap: anywhere; }}
    .nuisance span {{ display: block; color: var(--muted); font-size: 9px; }}
    .contract {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 16px; }}
    .note {{ padding: 17px; border-left: 5px solid var(--orange); }}
    .note strong {{ color: #844317; }}
    .small {{ color: var(--muted); font-size: 12px; line-height: 1.5; }}
    footer {{ padding: 20px; border-top: 1px solid var(--line); text-align: center; color: var(--muted); font-size: 11px; }}
    @media (max-width: 850px) {{
      .summary, .factor-grid {{ grid-template-columns: repeat(2, 1fr); }}
      .views, .contract {{ grid-template-columns: 1fr; }}
      .selector {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="eyebrow">Checkpoint visuel · sélection prédéclarée</div>
    <h1>{title}</h1>
    <p>Deux vues d'un même passage partagent les facteurs physiques à préserver, tandis que phase, position, bruit, amplitude et réponse capteur sont rééchantillonnés.</p>
  </header>
  <main>
    <div id="summary" class="summary"></div>
    <section class="panel selector">
      <div>
        <label for="case-select">Cas affiché</label>
        <select id="case-select"></select>
      </div>
      <div class="dataset">yeast-passage-simulations@v2</div>
    </section>
    <section id="case-panel" class="panel"></section>
    <section class="contract">
      <article class="panel note">
        <strong>Question falsifiable</strong>
        <p class="small">Les deux vues conservent-elles visuellement une structure de passage compatible avec leurs facteurs communs, malgré les nuisances rééchantillonnées ?</p>
      </article>
      <article class="panel note">
        <strong>Limite de la preuve</strong>
        <p id="claim-boundary" class="small"></p>
      </article>
    </section>
  </main>
  <footer>Interface locale en lecture seule · affichage décimé 4:1 · aucune donnée modifiée.</footer>
  <script id="payload" type="application/json">{data}</script>
  <script>
    const data = JSON.parse(document.getElementById("payload").textContent);
    const fmt = (value, digits=3) => Number(value).toLocaleString("fr-FR", {{maximumFractionDigits: digits}});
    document.getElementById("summary").innerHTML = [
      [data.summary.n_latents, "latents physiques"],
      [data.summary.n_signals, "signaux simulés"],
      [data.summary.views_per_latent, "vues par latent"],
      [data.display_contract.duration_ms, "ms par signal"]
    ].map(([value, label]) => `<div class="metric"><b>${{fmt(value)}}</b><span>${{label}}</span></div>`).join("");
    document.getElementById("claim-boundary").textContent = data.claim_boundary;
    const select = document.getElementById("case-select");
    data.cases.forEach((item, index) => select.add(new Option(`${{index + 1}} · ${{item.label}}`, item.id)));
    const units = {{
      duration_ms: "ms", doppler_khz: "kHz", component_count: "",
      component_separation_ms: "ms", relative_component_amplitude: "ratio",
      frequency_separation_khz: "kHz", phase_rad: "rad",
      event_position_fraction: "fraction", snr_db: "dB", target_rms: "RMS",
      baseline_drift: "ratio", sensor_response: "ratio"
    }};
    const names = {{
      duration_ms: "durée", doppler_khz: "Doppler", component_count: "composantes",
      component_separation_ms: "séparation temporelle",
      relative_component_amplitude: "amplitude relative",
      frequency_separation_khz: "séparation fréquentielle",
      phase_rad: "phase", event_position_fraction: "position",
      snr_db: "SNR simulé", target_rms: "RMS cible",
      baseline_drift: "dérive", sensor_response: "réponse capteur"
    }};
    function draw(canvas, signal, position, color, yLimit) {{
      const ratio = window.devicePixelRatio || 1;
      const width = canvas.clientWidth, height = canvas.clientHeight;
      canvas.width = width * ratio; canvas.height = height * ratio;
      const ctx = canvas.getContext("2d"); ctx.scale(ratio, ratio);
      const pad = {{l: 48, r: 14, t: 14, b: 30}}, w = width-pad.l-pad.r, h = height-pad.t-pad.b;
      ctx.strokeStyle = "#ded9cf"; ctx.lineWidth = 1;
      for (let i=0; i<=4; i++) {{
        const y = pad.t + h*i/4; ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(width-pad.r,y); ctx.stroke();
      }}
      const px = pad.l + position*w;
      ctx.save(); ctx.setLineDash([5,4]); ctx.strokeStyle = "#c26a2d"; ctx.beginPath(); ctx.moveTo(px,pad.t); ctx.lineTo(px,pad.t+h); ctx.stroke(); ctx.restore();
      ctx.strokeStyle = color; ctx.lineWidth = 1.35; ctx.beginPath();
      signal.forEach((value, i) => {{
        const x = pad.l + i*w/(signal.length-1);
        const y = pad.t + h/2 - value*h/(2*yLimit);
        i ? ctx.lineTo(x,y) : ctx.moveTo(x,y);
      }}); ctx.stroke();
      ctx.fillStyle = "#66716e"; ctx.font = "10px system-ui";
      ctx.fillText("0", pad.l-3, height-10); ctx.fillText("4,096 ms", width-58, height-10);
      ctx.fillText(`±${{fmt(yLimit,2)}}`, 5, 18); ctx.fillText("position simulée", Math.min(px+5,width-95), 24);
    }}
    function render() {{
      const item = data.cases.find(row => row.id === select.value) || data.cases[0];
      const factorHtml = Object.entries(item.preserved).map(([key,value]) =>
        `<div class="factor"><b>${{fmt(value)}} ${{units[key]}}</b><span>${{names[key]}}</span></div>`
      ).join("");
      const all = item.views.flatMap(view => view.signal.map(Math.abs));
      const yLimit = Math.max(...all) * 1.04 || 1;
      const viewsHtml = item.views.map((view,index) =>
        `<article class="view">
          <div class="view-head"><strong>Vue ${{view.view_index + 1}}</strong><span class="dataset">row ${{view.signal_row}}</span></div>
          <canvas id="plot-${{index}}"></canvas>
          <div class="nuisances">${{Object.entries(view.nuisances).map(([key,value]) =>
            `<div class="nuisance"><b>${{fmt(value)}} ${{units[key]}}</b><span>${{names[key]}}</span></div>`
          ).join("")}}</div>
        </article>`
      ).join("");
      document.getElementById("case-panel").innerHTML =
        `<div class="case-head"><div><h2>${{item.label}}</h2><p class="why">${{item.why}}</p></div><div class="chip">${{item.split}} · ${{item.variant}}</div></div>
         <div class="factor-grid">${{factorHtml}}</div><div class="views">${{viewsHtml}}</div>`;
      item.views.forEach((view,index) => draw(
        document.getElementById(`plot-${{index}}`), view.signal,
        view.nuisances.event_position_fraction, index ? "#2c6386" : "#176f5b", yLimit
      ));
    }}
    select.addEventListener("change", render);
    window.addEventListener("resize", render);
    select.value = new URLSearchParams(window.location.search).get("case") || data.cases[0].id;
    render();
  </script>
</body>
</html>
"""
