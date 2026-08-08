from __future__ import annotations

import csv
import html
import json
import shutil
from pathlib import Path
from typing import Any


CASE_SPECS = (
    {
        "slug": "close-events-3040",
        "record_id": "1a35748535b2bd9b2579",
        "image": "current-better-close-events-3040.png",
        "winner": "current",
        "eyebrow": "Actuel meilleur · séparation",
        "title": "Deux passages proches restent deux GT",
        "takeaway": (
            "Le détecteur simple étend une seule zone d'énergie sur les deux "
            "passages. La représentation temps-fréquence de l'actuel conserve "
            "un centre distinct pour le premier événement."
        ),
    },
    {
        "slug": "close-events-681",
        "record_id": "343a9bb55e3e982f461a",
        "image": "current-better-close-events-681.png",
        "winner": "current",
        "eyebrow": "Actuel meilleur · voisinage",
        "title": "Un précurseur faible n'est pas absorbé par le passage fort",
        "takeaway": (
            "L'énergie temporelle simple fusionne le précurseur et le passage "
            "fort voisin. L'actuel les individualise, ce qui protège le nombre "
            "et le centrage des GT."
        ),
    },
    {
        "slug": "weak-event-3144",
        "record_id": "4f7f36e76538517bb5c0",
        "image": "current-better-weak-event-3144.png",
        "winner": "current",
        "eyebrow": "Actuel meilleur · faible signal",
        "title": "La structure spectrale rend crédible un événement faible",
        "takeaway": (
            "Le signal temporel est peu contrasté, mais le spectrogramme montre "
            "une structure localisée dans la bande utile. L'actuel la retient; "
            "le simple reste sous son seuil."
        ),
    },
    {
        "slug": "simple-recovery-660",
        "record_id": "1273f303efcd6e7f956d",
        "image": "simple-better-weak-event-660.png",
        "winner": "simple",
        "eyebrow": "Simple meilleur · contre-exemple",
        "title": "Le contrôle simple récupère aussi un passage plausible",
        "takeaway": (
            "Le candidat bleu vers 4,3 ms est jugé plausible pendant la revue "
            "humaine et n'existe pas dans la sortie actuelle. Le simple reste "
            "donc utile comme détecteur sentinelle des désaccords."
        ),
    },
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _integer(row: dict[str, str], key: str) -> int:
    return int(row.get(key, "") or 0)


def build_validation_board_model(
    *,
    candidate_review_dirs: list[Path],
    ablation_summary_path: Path,
    disagreement_review_path: Path,
    evidence_dir: Path,
    assets_dir: Path,
) -> dict[str, Any]:
    candidate_rows: list[dict[str, str]] = []
    full_trace_rows: list[dict[str, str]] = []
    for review_dir in candidate_review_dirs:
        candidate_rows.extend(_read_csv(review_dir / "manual_review_queue.csv"))
        full_trace_rows.extend(
            _read_csv(review_dir / "manual_file_review_queue.csv")
        )

    retained = [
        row for row in candidate_rows if row["quality"] in {"strict", "medium"}
    ]
    rejected = [row for row in candidate_rows if row["quality"] == "reject"]
    retained_confirmed = sum(
        row["review_event_present"] == "yes" for row in retained
    )
    rejected_true = sum(
        row["review_event_present"] == "yes" for row in rejected
    )
    true_events = sum(
        _integer(row, "review_true_event_count") for row in full_trace_rows
    )
    false_retained = sum(
        _integer(row, "review_false_retained_candidate_count")
        for row in full_trace_rows
    )
    true_rejected = sum(
        _integer(row, "review_true_rejected_candidate_count")
        for row in full_trace_rows
    )
    fully_missed = sum(
        _integer(row, "review_missed_event_count") for row in full_trace_rows
    )
    retained_true = true_events - true_rejected - fully_missed

    ablation = json.loads(ablation_summary_path.read_text(encoding="utf-8"))
    validation = ablation["split_scores"]["development_validation"]
    disagreement_rows = _read_csv(disagreement_review_path)
    reviewed_disagreements = [
        row for row in disagreement_rows if row["review_preference"].strip()
    ]
    preference_counts = {
        preference: sum(
            row["review_preference"] == preference
            for row in reviewed_disagreements
        )
        for preference in ("current", "simple", "uncertain")
    }
    disagreement_by_record = {
        row["record_id"]: row for row in disagreement_rows
    }

    assets_dir.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    for spec in CASE_SPECS:
        source = evidence_dir / "images" / spec["image"]
        destination = assets_dir / spec["image"]
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, destination)
        review = disagreement_by_record[spec["record_id"]]
        if review["review_preference"] != spec["winner"]:
            raise ValueError(
                f"{spec['record_id']}: visual winner no longer matches review"
            )
        cases.append(
            {
                **spec,
                "plot": f"plots/{spec['image']}",
                "relative_path": review["relative_path"],
                "source_group": review["source_group"],
                "reviewer": review["reviewer"],
                "review_notes": review["review_notes"],
                "review_label": (
                    "Commentaire humain enregistré"
                    if review["review_notes"].strip()
                    else "Verdict humain enregistré"
                ),
                "review_display": (
                    review["review_notes"]
                    if review["review_notes"].strip()
                    else (
                        "Pipeline actuelle préférée"
                        if spec["winner"] == "current"
                        else "Ablation simple préférée"
                    )
                ),
            }
        )

    return {
        "schema_version": 1,
        "title": "Dataset yeast final — pourquoi faire confiance aux GT ?",
        "subtitle": (
            "Validation humaine des événements, contrôle des traces complètes "
            "et ablation ciblée de la représentation temps-fréquence."
        ),
        "verdict": (
            "Dataset satisfaisant pour le développement sur cette acquisition; "
            "la généralisation inter-acquisition reste à démontrer."
        ),
        "audit": {
            "candidate_reviews": len(candidate_rows),
            "retained_reviewed": len(retained),
            "retained_confirmed": retained_confirmed,
            "rejected_reviewed": len(rejected),
            "rejected_true": rejected_true,
            "full_trace_reviews": len(full_trace_rows),
            "true_events": true_events,
            "retained_true": retained_true,
            "false_retained": false_retained,
            "true_rejected": true_rejected,
            "fully_missed": fully_missed,
            "additional_unseen_reviews": 24,
        },
        "ablation": {
            "method": ablation["method"],
            "threshold": float(ablation["selected_quality_z"]),
            "matched": int(validation["categories"]["matched"]),
            "current_only": int(validation["categories"]["current_only"]),
            "simple_only": int(validation["categories"]["simple_only"]),
            "precision": float(validation["precision_vs_current"]),
            "recall": float(validation["recall_vs_current"]),
            "f1": float(validation["f1_vs_current"]),
            "center_p50_ms": float(
                validation["center_abs_error_ms"]["p50"]
            ),
            "center_p95_ms": float(
                validation["center_abs_error_ms"]["p95"]
            ),
            "human_reviewed": len(reviewed_disagreements),
            "human_preferences": preference_counts,
        },
        "cases": cases,
        "processes": [
            {
                "name": "Bande passe 7–80 kHz",
                "status": "partagé",
                "role": (
                    "Isole la bande physique utile avant toute mesure. "
                    "L'ablation la conserve : son intérêt n'est pas contesté ici."
                ),
            },
            {
                "name": "Énergie + médiane/MAD",
                "status": "cœur commun",
                "role": (
                    "Mesure une hausse locale relativement au bruit propre à "
                    "chaque trace. Le fort accord confirme que ce bloc porte "
                    "l'essentiel de la décision."
                ),
            },
            {
                "name": "STFT et proposition temps-fréquence",
                "status": "ablaté",
                "role": (
                    "Rend le Doppler visible et aide à localiser ou séparer des "
                    "passages proches. Les cas 3040, 681 et 3144 donnent une "
                    "utilité expérimentale concrète."
                ),
            },
            {
                "name": "Concentration et phase",
                "status": "diagnostic",
                "role": (
                    "Utiles pour décrire et auditer un candidat, mais cette "
                    "comparaison ne justifie pas de les présenter comme des "
                    "portes décisionnelles indispensables."
                ),
            },
        ],
    }


def render_validation_board_html(model: dict[str, Any]) -> str:
    audit = model["audit"]
    ablation = model["ablation"]
    process_cards = "".join(
        f"""
        <article class="process-card">
          <div class="process-top">
            <h3>{html.escape(process["name"])}</h3>
            <span class="status {html.escape(process["status"].replace(" ", "-"))}">
              {html.escape(process["status"])}
            </span>
          </div>
          <p>{html.escape(process["role"])}</p>
        </article>
        """
        for process in model["processes"]
    )
    case_tabs = "".join(
        f"""
        <button class="case-tab" data-case="{html.escape(case["slug"])}">
          <span>{html.escape(case["eyebrow"])}</span>
          {html.escape(case["title"])}
        </button>
        """
        for case in model["cases"]
    )
    case_panels = "".join(
        f"""
        <article class="case-panel" data-case-panel="{html.escape(case["slug"])}" hidden>
          <div class="case-copy">
            <div>
              <div class="eyebrow">{html.escape(case["eyebrow"])}</div>
              <h3>{html.escape(case["title"])}</h3>
              <p>{html.escape(case["takeaway"])}</p>
            </div>
            <span class="winner {html.escape(case["winner"])}">
              {("actuel" if case["winner"] == "current" else "simple")}
            </span>
          </div>
          <figure>
            <img src="{html.escape(case["plot"])}" alt="{html.escape(case["title"])}">
            <figcaption>
              Vert : commun · orange : actuel seul · bleu : simple seul.
              {html.escape(case["relative_path"])}
            </figcaption>
          </figure>
          <aside class="human-note">
            <strong>{html.escape(case["review_label"])} · {html.escape(case["reviewer"])}</strong>
            <span>{html.escape(case["review_display"])}</span>
          </aside>
        </article>
        """
        for case in model["cases"]
    )
    embedded = json.dumps(
        {"default_case": model["cases"][0]["slug"]},
        ensure_ascii=False,
    ).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(model["title"])}</title>
<style>
:root {{
  --ink:#13241f; --muted:#5c6d67; --paper:#f3f4ed; --card:#fffef8;
  --line:#d8ddd2; --green:#087f65; --green-dark:#075e4d;
  --mint:#dff2e9; --orange:#e47b24; --orange-pale:#fff0df;
  --blue:#336bd6; --blue-pale:#e7eefc; --yellow:#f0c453;
}}
* {{ box-sizing:border-box; }}
html {{ scroll-behavior:smooth; }}
body {{
  margin:0; color:var(--ink); background:
  radial-gradient(circle at 90% 5%,rgba(240,196,83,.18),transparent 22rem),
  var(--paper); font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif;
}}
header {{
  color:white; background:linear-gradient(118deg,#10251f 0%,#163b31 67%,#245747 100%);
  padding:38px max(30px,calc((100vw - 1480px)/2)) 34px; position:relative; overflow:hidden;
}}
header::after {{
  content:""; position:absolute; width:360px; height:360px; border:1px solid rgba(255,255,255,.13);
  border-radius:50%; right:-80px; top:-210px; box-shadow:0 0 0 44px rgba(255,255,255,.035);
}}
.kicker {{ color:#9ce5ce; font-size:12px; font-weight:900; letter-spacing:.16em; text-transform:uppercase; }}
header h1 {{ max-width:1000px; margin:10px 0 8px; font-size:42px; line-height:1.03; letter-spacing:-.035em; }}
header p {{ max-width:930px; margin:0; color:#cfddd8; font-size:17px; line-height:1.45; }}
.verdict {{
  display:inline-flex; gap:10px; align-items:center; margin-top:20px; padding:10px 14px;
  border:1px solid rgba(156,229,206,.45); border-radius:999px; background:rgba(8,127,101,.25);
  font-weight:750; font-size:13px;
}}
.verdict::before {{ content:"✓"; display:grid; place-items:center; width:22px; height:22px; border-radius:50%; background:#9ce5ce; color:#10251f; }}
nav {{
  position:sticky; top:0; z-index:8; display:flex; gap:22px; padding:12px max(30px,calc((100vw - 1480px)/2));
  background:rgba(255,254,248,.94); border-bottom:1px solid var(--line); backdrop-filter:blur(8px);
}}
nav a {{ color:var(--muted); text-decoration:none; font-size:12px; font-weight:850; text-transform:uppercase; letter-spacing:.08em; }}
nav a:hover {{ color:var(--green); }}
main {{ max-width:1480px; margin:0 auto; padding:28px 28px 52px; }}
section {{ scroll-margin-top:55px; }}
.section-head {{ display:flex; align-items:end; justify-content:space-between; gap:24px; margin:0 0 14px; }}
.section-head .number {{ color:var(--green); font-weight:950; font-size:12px; letter-spacing:.15em; text-transform:uppercase; }}
.section-head h2 {{ margin:3px 0 0; font-size:27px; letter-spacing:-.025em; }}
.section-head p {{ max-width:600px; margin:0; color:var(--muted); font-size:13px; line-height:1.5; }}
.audit-grid {{ display:grid; grid-template-columns:1.15fr 1fr 1fr 1fr; gap:12px; }}
.audit-card {{ background:var(--card); border:1px solid var(--line); border-radius:16px; padding:18px; min-height:160px; box-shadow:0 8px 24px rgba(19,36,31,.045); }}
.audit-card.hero {{ background:var(--ink); color:white; border-color:var(--ink); }}
.audit-card .label {{ color:var(--muted); font-size:11px; font-weight:850; text-transform:uppercase; letter-spacing:.1em; }}
.audit-card.hero .label {{ color:#9ce5ce; }}
.audit-card strong {{ display:block; margin:8px 0 5px; font-size:31px; letter-spacing:-.04em; }}
.audit-card p {{ margin:0; color:var(--muted); font-size:12.5px; line-height:1.45; }}
.audit-card.hero p {{ color:#d2dfda; }}
.audit-card .mini {{ margin-top:13px; padding-top:10px; border-top:1px solid var(--line); font-size:11px; color:var(--muted); }}
.audit-card.hero .mini {{ border-color:#345249; color:#aebfba; }}
.protocol {{
  display:grid; grid-template-columns:1fr 60px 1fr 60px 1fr; align-items:stretch;
  margin:12px 0 34px; padding:14px; background:#e7ebe2; border-radius:15px;
}}
.protocol div {{ background:rgba(255,255,255,.72); border:1px solid #d3dacf; border-radius:11px; padding:13px; }}
.protocol strong {{ display:block; font-size:13px; margin-bottom:4px; }}
.protocol span {{ display:block; color:var(--muted); font-size:11.5px; line-height:1.4; }}
.protocol b {{ display:grid; place-items:center; color:var(--green); font-size:22px; }}
.split {{
  display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-bottom:34px;
}}
.method {{
  background:var(--card); border:1px solid var(--line); border-radius:16px; padding:19px;
}}
.method.current {{ border-top:4px solid var(--orange); }}
.method.simple {{ border-top:4px solid var(--blue); }}
.method h3 {{ margin:0 0 4px; font-size:18px; }}
.method > p {{ margin:0 0 15px; color:var(--muted); font-size:12px; }}
.flow {{ display:flex; align-items:stretch; gap:6px; }}
.flow span {{
  flex:1; display:grid; place-items:center; min-height:58px; padding:8px; text-align:center;
  border-radius:9px; background:#eff2eb; font-size:10.5px; font-weight:800; line-height:1.25;
}}
.flow i {{ align-self:center; color:var(--green); font-style:normal; font-weight:900; }}
.flow .distinct {{ color:#8b480e; background:var(--orange-pale); }}
.simple .flow .distinct {{ color:#224f9f; background:var(--blue-pale); }}
.comparison {{
  display:grid; grid-template-columns:1.2fr .9fr; gap:14px; margin-bottom:14px;
}}
.score-card,.human-card {{ background:var(--card); border:1px solid var(--line); border-radius:16px; padding:18px; }}
.score-line {{ display:grid; grid-template-columns:145px 1fr 68px; align-items:center; gap:10px; margin:10px 0; }}
.score-line label {{ color:var(--muted); font-size:12px; }}
.track {{ height:12px; border-radius:99px; background:#e5e9e1; overflow:hidden; }}
.track span {{ display:block; height:100%; background:linear-gradient(90deg,var(--green),#54b89f); border-radius:99px; }}
.score-line strong {{ text-align:right; font-size:13px; }}
.event-counts {{ display:grid; grid-template-columns:repeat(3,1fr); gap:8px; margin-top:14px; }}
.event-counts div {{ border-radius:10px; padding:11px; text-align:center; background:#f0f3ed; }}
.event-counts div:nth-child(2) {{ background:var(--orange-pale); }}
.event-counts div:nth-child(3) {{ background:var(--blue-pale); }}
.event-counts strong {{ display:block; font-size:23px; }}
.event-counts span {{ color:var(--muted); font-size:10px; text-transform:uppercase; }}
.vote {{
  display:grid; grid-template-columns:auto 1fr; gap:10px; align-items:center; margin:12px 0;
}}
.vote strong {{ font-size:27px; }}
.vote span {{ color:var(--muted); font-size:12px; }}
.vote.current strong {{ color:var(--orange); }}
.vote.simple strong {{ color:var(--blue); }}
.process-grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-bottom:34px; }}
.process-card {{ background:var(--card); border:1px solid var(--line); border-radius:14px; padding:15px; }}
.process-top {{ display:flex; gap:8px; align-items:flex-start; justify-content:space-between; }}
.process-card h3 {{ margin:0; font-size:14px; line-height:1.25; }}
.process-card p {{ margin:10px 0 0; color:var(--muted); font-size:11.5px; line-height:1.5; }}
.status {{ white-space:nowrap; border-radius:999px; padding:4px 7px; background:#edf0ea; color:var(--muted); font-size:9px; font-weight:900; text-transform:uppercase; }}
.status.ablaté {{ color:#8b480e; background:var(--orange-pale); }}
.status.cœur-commun {{ color:var(--green-dark); background:var(--mint); }}
.case-layout {{ display:grid; grid-template-columns:280px minmax(0,1fr); gap:12px; margin-bottom:34px; }}
.case-tabs {{ display:flex; flex-direction:column; gap:7px; }}
.case-tab {{
  appearance:none; border:1px solid var(--line); border-radius:11px; background:var(--card);
  color:var(--ink); text-align:left; padding:12px; cursor:pointer; font-weight:750; line-height:1.3;
}}
.case-tab span {{ display:block; color:var(--muted); margin-bottom:4px; font-size:9.5px; font-weight:900; text-transform:uppercase; letter-spacing:.06em; }}
.case-tab.active {{ border-color:var(--ink); background:var(--ink); color:white; }}
.case-tab.active span {{ color:#9ce5ce; }}
.case-panel {{ background:var(--card); border:1px solid var(--line); border-radius:16px; overflow:hidden; }}
.case-copy {{ display:flex; justify-content:space-between; gap:18px; padding:17px 19px 13px; border-bottom:1px solid var(--line); }}
.case-copy h3 {{ margin:3px 0 5px; font-size:20px; }}
.case-copy p {{ margin:0; color:var(--muted); font-size:12.5px; max-width:880px; line-height:1.45; }}
.eyebrow {{ color:var(--green); font-size:10px; font-weight:900; letter-spacing:.1em; text-transform:uppercase; }}
.winner {{ align-self:center; border-radius:999px; padding:7px 11px; font-size:10px; font-weight:950; text-transform:uppercase; }}
.winner.current {{ background:var(--orange-pale); color:#9b500e; }}
.winner.simple {{ background:var(--blue-pale); color:#2755a7; }}
figure {{ margin:0; padding:10px 12px 6px; }}
figure img {{ display:block; width:100%; max-height:610px; object-fit:contain; }}
figcaption {{ padding:5px 7px 0; color:var(--muted); font-size:10.5px; }}
.human-note {{ display:flex; gap:12px; align-items:baseline; margin:8px 18px 16px; border-left:3px solid var(--yellow); padding:8px 10px; background:#fffaf0; }}
.human-note strong {{ white-space:nowrap; font-size:11px; }}
.human-note span {{ color:#59655f; font-size:11px; font-style:italic; }}
.decision-grid {{ display:grid; grid-template-columns:1.2fr 1fr; gap:12px; }}
.decision-card {{ border-radius:16px; padding:21px; background:var(--ink); color:white; }}
.decision-card h3 {{ margin:0 0 8px; font-size:21px; }}
.decision-card p {{ margin:0; color:#cfddd8; line-height:1.55; font-size:13px; }}
.decision-card.secondary {{ background:#e4ebe4; color:var(--ink); }}
.decision-card.secondary p {{ color:var(--muted); }}
.caveat {{ margin-top:13px; color:var(--muted); font-size:11px; line-height:1.5; }}
footer {{ max-width:1480px; margin:0 auto; padding:0 28px 30px; color:var(--muted); font-size:10.5px; }}
body.capture-ablation header,body.capture-ablation nav,body.capture-ablation #audit,
body.capture-ablation #cas,body.capture-ablation #decision,body.capture-ablation footer {{
  display:none;
}}
body.capture-case header,body.capture-case nav,body.capture-case #audit,
body.capture-case #ablation,body.capture-case footer {{
  display:none;
}}
body.capture-ablation main,body.capture-case main {{ padding-top:24px; }}
@media(max-width:1000px) {{
  .audit-grid,.process-grid {{ grid-template-columns:1fr 1fr; }}
  .split,.comparison,.decision-grid,.case-layout {{ grid-template-columns:1fr; }}
  .protocol {{ grid-template-columns:1fr; gap:7px; }}
  .protocol b {{ transform:rotate(90deg); }}
  .case-tabs {{ display:grid; grid-template-columns:1fr 1fr; }}
}}
</style>
</head>
<body>
<header>
  <div class="kicker">Validation visuelle · dataset final</div>
  <h1>{html.escape(model["title"])}</h1>
  <p>{html.escape(model["subtitle"])}</p>
  <div class="verdict">{html.escape(model["verdict"])}</div>
</header>
<nav>
  <a href="#audit">Audit humain</a>
  <a href="#ablation">Ablation</a>
  <a href="#cas">Cas visuels</a>
  <a href="#decision">Décision</a>
</nav>
<main>
  <section id="audit">
    <div class="section-head">
      <div><div class="number">01 · confiance dans les GT</div><h2>Deux audits complémentaires</h2></div>
      <p>Regarder uniquement les boîtes mesure la précision. Relire des traces complètes est indispensable pour chercher les événements absents.</p>
    </div>
    <div class="audit-grid">
      <article class="audit-card hero">
        <div class="label">Candidats retenus</div>
        <strong>{audit["retained_confirmed"]} / {audit["retained_reviewed"]}</strong>
        <p>Chaque fenêtre retenue et inspectée contient bien un événement.</p>
        <div class="mini">{audit["candidate_reviews"]} fenêtres revues au total, dont {audit["rejected_reviewed"]} rejets.</div>
      </article>
      <article class="audit-card">
        <div class="label">Traces complètes</div>
        <strong>{audit["full_trace_reviews"]}</strong>
        <p>Inspection de bout en bout, indépendamment des seules fenêtres proposées.</p>
        <div class="mini">{audit["true_events"]} événements vrais observés.</div>
      </article>
      <article class="audit-card">
        <div class="label">Rappel observé</div>
        <strong>{audit["retained_true"]} / {audit["true_events"]}</strong>
        <p>Événements vrais associés à une détection retenue.</p>
        <div class="mini">{audit["true_rejected"]} vrai événement rejeté · {audit["fully_missed"]} totalement manqué.</div>
      </article>
      <article class="audit-card">
        <div class="label">Contrôle additionnel</div>
        <strong>+{audit["additional_unseen_reviews"]}</strong>
        <p>14 fenêtres et 10 traces jamais vues dans la revue précédente.</p>
        <div class="mini">{audit["false_retained"]} faux candidat retenu observé sur l'ensemble des traces.</div>
      </article>
    </div>
    <div class="protocol">
      <div><strong>Fenêtres candidates</strong><span>« Ce que le détecteur retient est-il réellement un événement ? »</span></div>
      <b>+</b>
      <div><strong>Traces complètes</strong><span>« Existe-t-il un événement rejeté ou absent entre les boîtes ? »</span></div>
      <b>→</b>
      <div><strong>GT auditables</strong><span>Les commentaires humains restent liés à l'identifiant du signal et à la décision.</span></div>
    </div>
  </section>

  <section id="ablation">
    <div class="section-head">
      <div><div class="number">02 · test de nécessité</div><h2>Et si un simple seuil d'énergie suffisait ?</h2></div>
      <p>Le seuil du modèle simple est réglé sur le train puis gelé sur la validation. Il partage le cœur robuste et retire la proposition temps-fréquence.</p>
    </div>
    <div class="split">
      <article class="method current">
        <h3>Pipeline actuelle</h3>
        <p>Représentation temps-fréquence et diagnostic spectral.</p>
        <div class="flow"><span>Bande<br>7–80 kHz</span><i>→</i><span class="distinct">STFT<br>énergie 2D</span><i>→</i><span>médiane<br>+ MAD</span><i>→</i><span>groupes<br>temporels</span><i>→</i><span>z +<br>largeur</span></div>
      </article>
      <article class="method simple">
        <h3>Ablation simple</h3>
        <p>Même logique de seuil, sans construction temps-fréquence.</p>
        <div class="flow"><span>Bande<br>7–80 kHz</span><i>→</i><span class="distinct">énergie<br>temporelle</span><i>→</i><span>médiane<br>+ MAD</span><i>→</i><span>groupes<br>temporels</span><i>→</i><span>z +<br>largeur</span></div>
      </article>
    </div>
    <div class="comparison">
      <article class="score-card">
        <div class="eyebrow">Validation · comparaison à l'actuel</div>
        <div class="score-line"><label>Précision</label><div class="track"><span style="width:{ablation["precision"] * 100:.2f}%"></span></div><strong>{ablation["precision"] * 100:.2f} %</strong></div>
        <div class="score-line"><label>Rappel</label><div class="track"><span style="width:{ablation["recall"] * 100:.2f}%"></span></div><strong>{ablation["recall"] * 100:.2f} %</strong></div>
        <div class="score-line"><label>F1</label><div class="track"><span style="width:{ablation["f1"] * 100:.2f}%"></span></div><strong>{ablation["f1"] * 100:.2f} %</strong></div>
        <div class="event-counts">
          <div><strong>{ablation["matched"]}</strong><span>communs</span></div>
          <div><strong>{ablation["current_only"]}</strong><span>actuel seul</span></div>
          <div><strong>{ablation["simple_only"]}</strong><span>simple seul</span></div>
        </div>
      </article>
      <article class="human-card">
        <div class="eyebrow">Arrêt après {ablation["human_reviewed"]} désaccords annotés</div>
        <div class="vote current"><strong>{ablation["human_preferences"]["current"]}</strong><span>traces où l'actuel est préféré</span></div>
        <div class="vote simple"><strong>{ablation["human_preferences"]["simple"]}</strong><span>trace où le simple est préféré</span></div>
        <div class="vote"><strong>{ablation["human_preferences"]["uncertain"]}</strong><span>cas laissé incertain</span></div>
        <p class="caveat">Échantillon séquentiel volontairement partiel : il sert à comprendre les désaccords, pas à produire un intervalle de confiance biologique.</p>
      </article>
    </div>
    <div class="process-grid">{process_cards}</div>
  </section>

  <section id="cas">
    <div class="section-head">
      <div><div class="number">03 · preuves visuelles</div><h2>Ce que les chiffres seuls ne montrent pas</h2></div>
      <p>Trois mécanismes favorisent l'actuel; le contre-exemple du simple est conservé pour éviter une conclusion à sens unique.</p>
    </div>
    <div class="case-layout">
      <div class="case-tabs">{case_tabs}</div>
      <div>{case_panels}</div>
    </div>
  </section>

  <section id="decision">
    <div class="section-head">
      <div><div class="number">04 · décision méthodologique</div><h2>Ce que l'expérience change réellement</h2></div>
    </div>
    <div class="decision-grid">
      <article class="decision-card">
        <h3>Garder l'actuel pour construire les GT</h3>
        <p>Son intérêt n'est pas un meilleur score moyen spectaculaire : c'est une meilleure lisibilité physique et, dans les désaccords contrôlés, une meilleure séparation des événements proches ou faibles. C'est précisément là que le centrage et le nombre de GT peuvent être dégradés.</p>
      </article>
      <article class="decision-card secondary">
        <h3>Garder le simple comme sentinelle</h3>
        <p>Le contre-exemple montre qu'il reste utile expérimentalement. L'union des sorties et la revue des désaccords fournissent un contrôle qualité peu coûteux pour détecter les passages plausibles manqués par la pipeline principale.</p>
      </article>
    </div>
    <p class="caveat"><strong>Portée :</strong> une seule acquisition biologique et instrumentale. Les taux observés valident ce dataset de développement; ils ne démontrent pas encore la robustesse à un changement d'acquisition.</p>
  </section>
</main>
<footer>Sources : yeast-event-candidates@v7/v8 · revues humaines candidates et traces complètes · yeast-detector-ablation@v2.</footer>
<script id="board-config" type="application/json">{embedded}</script>
<script>
const config = JSON.parse(document.getElementById("board-config").textContent);
const params = new URLSearchParams(location.search);
const capture = params.get("capture");
if (capture === "ablation" || capture === "case") document.body.classList.add(`capture-${{capture}}`);
function showCase(slug) {{
  document.querySelectorAll("[data-case-panel]").forEach(panel => panel.hidden = panel.dataset.casePanel !== slug);
  document.querySelectorAll("[data-case]").forEach(button => button.classList.toggle("active", button.dataset.case === slug));
  const url = new URL(window.location.href); url.searchParams.set("case", slug); history.replaceState(null, "", url);
}}
document.querySelectorAll("[data-case]").forEach(button => button.onclick = () => showCase(button.dataset.case));
const requested = params.get("case");
const available = [...document.querySelectorAll("[data-case]")].map(button => button.dataset.case);
showCase(available.includes(requested) ? requested : config.default_case);
</script>
</body>
</html>
"""
