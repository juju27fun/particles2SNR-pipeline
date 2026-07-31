from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scipy.signal import spectrogram

from .yeast_events import (
    bandpass_yeast_signal,
    review_calibrated_detection_config_v1,
)


PREFERENCES = {"current", "simple", "equivalent", "neither", "uncertain"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class YeastAblationReviewWorkspace:
    def __init__(self, raw_dataset_root: Path, review_dir: Path) -> None:
        self.raw_dataset_root = raw_dataset_root.resolve()
        self.review_dir = review_dir.resolve()
        self.queue_path = self.review_dir / "review_queue.csv"
        with self.queue_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            self.fields = list(reader.fieldnames or [])
            self.rows = list(reader)
        if not self.fields or not self.rows:
            raise ValueError(f"Empty ablation review queue: {self.queue_path}")
        self._lock = threading.Lock()

    def is_complete(self, row: dict[str, str]) -> bool:
        return (
            bool(row.get("reviewer", "").strip())
            and row.get("review_preference", "").strip() in PREFERENCES
        )

    def list_items(self) -> dict[str, Any]:
        items = [
            {
                "index": index,
                "record_id": row["record_id"],
                "source_group": row["source_group"],
                "relative_path": row["relative_path"],
                "current_only": int(row["n_current_only"]),
                "simple_only": int(row["n_simple_only"]),
                "complete": self.is_complete(row),
            }
            for index, row in enumerate(self.rows)
        ]
        return {
            "total": len(items),
            "complete": sum(item["complete"] for item in items),
            "items": items,
        }

    def get_item(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self.rows):
            raise IndexError(f"Review index out of range: {index}")
        row = dict(self.rows[index])
        row["current_candidates"] = json.loads(row["current_candidates_json"])
        row["simple_candidates"] = json.loads(row["simple_candidates_json"])
        return {
            "index": index,
            "total": len(self.rows),
            "complete": self.is_complete(row),
            "row": row,
            "plot_url": f"/plot/{index}.png",
        }

    def _write(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=self.review_dir,
            prefix=".review_queue.csv.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=self.fields)
            writer.writeheader()
            writer.writerows(self.rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.queue_path)

    def save_item(self, index: int, values: dict[str, Any]) -> dict[str, Any]:
        reviewer = str(values.get("reviewer", "")).strip()
        preference = str(values.get("review_preference", "")).strip()
        notes = str(values.get("review_notes", "")).strip()
        if not reviewer:
            raise ValueError("reviewer is required")
        if preference not in PREFERENCES:
            raise ValueError("review_preference is invalid")
        with self._lock:
            row = self.rows[index]
            old = {
                field: row.get(field, "")
                for field in ("reviewer", "review_preference", "review_notes")
            }
            row.update(
                {
                    "reviewer": reviewer,
                    "review_preference": preference,
                    "review_notes": notes,
                }
            )
            self._write()
            audit = {
                "schema_version": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "index": index,
                "record_id": row["record_id"],
                "old": old,
                "new": {
                    "reviewer": reviewer,
                    "review_preference": preference,
                    "review_notes": notes,
                },
                "queue_sha256": _sha256(self.queue_path),
            }
            with (self.review_dir / "annotation_audit.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(json.dumps(audit, sort_keys=True) + "\n")
        return self.get_item(index)

    @lru_cache(maxsize=96)
    def plot_png(self, index: int, signal_view: str = "filtered") -> bytes:
        if signal_view not in {"filtered", "raw"}:
            raise ValueError("signal_view must be filtered or raw")
        row = self.rows[index]
        signal = np.asarray(
            np.load(
                self.raw_dataset_root / row["relative_path"],
                allow_pickle=False,
            ),
            dtype=np.float32,
        )
        config = review_calibrated_detection_config_v1()
        displayed = (
            bandpass_yeast_signal(signal, config)
            if signal_view == "filtered"
            else signal
        )
        current = json.loads(row["current_candidates_json"])
        simple = json.loads(row["simple_candidates_json"])
        unmatched = [
            {**candidate, "method": "current"}
            for candidate in current
            if candidate["status"] == "current_only"
        ] + [
            {**candidate, "method": "simple"}
            for candidate in simple
            if candidate["status"] == "simple_only"
        ]

        zoom_count = max(1, len(unmatched))
        figure = Figure(figsize=(13.0, 8.2), constrained_layout=True)
        grid = figure.add_gridspec(
            3,
            zoom_count,
            height_ratios=(1.15, 0.9, 1.25),
        )
        full_axis = figure.add_subplot(grid[0, :])
        zoom_axes = [
            figure.add_subplot(grid[1, item])
            for item in range(zoom_count)
        ]
        spectrum_axis = figure.add_subplot(grid[2, :])
        time_ms = (
            np.arange(displayed.size)
            / config.sampling_frequency_hz
            * 1000.0
        )
        full_axis.plot(time_ms, displayed, color="#1b242c", linewidth=0.7)

        frequencies, times, power = spectrogram(
            displayed,
            fs=config.sampling_frequency_hz,
            nperseg=512,
            noverlap=384,
            window="hann",
            mode="magnitude",
        )
        band = (frequencies >= config.low_freq_hz) & (
            frequencies <= config.high_freq_hz
        )
        magnitude_db = 20.0 * np.log10(power[band] + 1.0e-12)
        low, high = np.quantile(magnitude_db, [0.05, 0.995])
        spectrum_axis.pcolormesh(
            times * 1000.0,
            frequencies[band] / 1000.0,
            magnitude_db,
            shading="auto",
            cmap="magma",
            vmin=float(low),
            vmax=float(high),
        )

        colors = {
            "matched": "#0f8a70",
            "current_only": "#d97706",
            "simple_only": "#2f6fed",
        }
        seen_matched: set[str] = set()
        for method, candidates in (("current", current), ("simple", simple)):
            for candidate in candidates:
                status = candidate["status"]
                if status == "matched":
                    match_key = str(candidate.get("match_key", ""))
                    if match_key in seen_matched:
                        continue
                    seen_matched.add(match_key)
                color = colors[status]
                left = (
                    int(candidate["event_start"])
                    / config.sampling_frequency_hz
                    * 1000.0
                )
                right = (
                    int(candidate["event_end"])
                    / config.sampling_frequency_hz
                    * 1000.0
                )
                center = (
                    int(candidate["center_index"])
                    / config.sampling_frequency_hz
                    * 1000.0
                )
                alpha = 0.15 if status != "matched" else 0.07
                for axis in (full_axis, spectrum_axis):
                    axis.axvspan(left, right, color=color, alpha=alpha)
                    axis.axvline(center, color=color, linewidth=0.85)
                full_axis.text(
                    center,
                    0.92,
                    (
                        "COMMUN"
                        if status == "matched"
                        else "ACTUEL"
                        if method == "current"
                        else "SIMPLE"
                    ),
                    color=color,
                    fontsize=8.5,
                    fontweight="bold",
                    ha="center",
                    transform=full_axis.get_xaxis_transform(),
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.8,
                        "pad": 1.5,
                    },
                )

        if unmatched:
            for axis, candidate in zip(zoom_axes, unmatched):
                center_index = int(candidate["center_index"])
                half = 2048
                start = max(0, center_index - half)
                end = min(displayed.size, center_index + half)
                local_time = (
                    np.arange(start, end) - center_index
                ) / config.sampling_frequency_hz * 1000.0
                axis.plot(
                    local_time,
                    displayed[start:end],
                    color="#1b242c",
                    linewidth=0.75,
                )
                left = (
                    int(candidate["event_start"]) - center_index
                ) / config.sampling_frequency_hz * 1000.0
                right = (
                    int(candidate["event_end"]) - center_index
                ) / config.sampling_frequency_hz * 1000.0
                color = (
                    colors["current_only"]
                    if candidate["method"] == "current"
                    else colors["simple_only"]
                )
                axis.axvspan(left, right, color=color, alpha=0.18)
                axis.axvline(0.0, color=color, linewidth=0.9)
                axis.set_title(
                    "Actuel seul"
                    if candidate["method"] == "current"
                    else "Simple seul",
                    color=color,
                    fontsize=10,
                    fontweight="bold",
                )
                axis.set_xlabel("temps relatif (ms)")
                axis.grid(True, color="#dbe2e7", linewidth=0.5)
        else:
            zoom_axes[0].text(
                0.5,
                0.5,
                "Aucun désaccord",
                ha="center",
                va="center",
                transform=zoom_axes[0].transAxes,
            )

        full_axis.set_title(
            f"{row['source_group']} · {row['relative_path']} · "
            f"actuel seul {row['n_current_only']} / simple seul {row['n_simple_only']}",
            fontsize=12,
            fontweight="bold",
        )
        full_axis.set_ylabel(
            "signal filtré" if signal_view == "filtered" else "signal brut"
        )
        full_axis.set_xlabel("temps depuis le début de la trace (ms)")
        spectrum_axis.set_ylabel("fréquence (kHz)")
        spectrum_axis.set_xlabel("temps depuis le début de la trace (ms)")
        full_axis.grid(True, color="#dbe2e7", linewidth=0.5)
        output = io.BytesIO()
        FigureCanvasAgg(figure).print_png(output)
        return output.getvalue()


def build_handler(workspace: YeastAblationReviewWorkspace) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _json(self, payload: Any, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    body = REVIEW_HTML.encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if parsed.path == "/api/items":
                    self._json(workspace.list_items())
                    return
                if parsed.path == "/api/item":
                    index = int(parse_qs(parsed.query).get("index", ["0"])[0])
                    self._json(workspace.get_item(index))
                    return
                if parsed.path.startswith("/plot/") and parsed.path.endswith(".png"):
                    index = int(Path(parsed.path).stem)
                    view = parse_qs(parsed.query).get("signal_view", ["filtered"])[0]
                    body = workspace.plot_png(index, view)
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_error(HTTPStatus.NOT_FOUND)
            except (ValueError, IndexError, FileNotFoundError) as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/api/item":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                self._json(
                    workspace.save_item(
                        int(payload["index"]),
                        dict(payload["values"]),
                    )
                )
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

        def log_message(self, _format: str, *_args: Any) -> None:
            return

    return Handler


def serve_ablation_review(
    *,
    raw_dataset_root: Path,
    review_dir: Path,
    host: str = "127.0.0.1",
    port: int = 8770,
) -> None:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Review server may bind only to loopback")
    workspace = YeastAblationReviewWorkspace(raw_dataset_root, review_dir)
    server = ThreadingHTTPServer((host, port), build_handler(workspace))
    print(f"Yeast ablation disagreement review: http://{host}:{port}", flush=True)
    print(f"Editable review directory: {review_dir.resolve()}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


REVIEW_HTML = """<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Revue des désaccords — détecteur yeast</title>
<style>
:root{--ink:#17212b;--muted:#62707d;--line:#d8e0e5;--paper:#f2f6f5;--green:#0f8a70;--orange:#d97706;--blue:#2f6fed}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:Inter,system-ui,sans-serif}
header{background:var(--ink);color:white;padding:20px 28px;display:flex;justify-content:space-between;align-items:center}
header h1{font-size:21px;margin:0}header span{color:#a9bac6;font-size:13px}
main{max-width:1450px;margin:auto;padding:20px 24px 36px}
.toolbar{display:flex;align-items:center;gap:10px;margin-bottom:12px}.toolbar #progress{margin-left:auto;font-weight:700}
button{border:1px solid #aebbc5;background:white;border-radius:8px;padding:9px 14px;font-weight:750;cursor:pointer}
button:disabled{opacity:.45}.view{display:flex;margin-left:12px}.view button{border-radius:0}.view button:first-child{border-radius:8px 0 0 8px}.view button:last-child{border-radius:0 8px 8px 0}.view button.active{background:#dff3ed;color:#076955;border-color:#0f8a70}
.card{background:white;border:1px solid var(--line);border-radius:13px;overflow:hidden}
.meta{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;padding:13px 16px;border-bottom:1px solid var(--line);font-size:12px}.meta span{display:block;color:var(--muted);font-size:10px;text-transform:uppercase}.meta strong{word-break:break-word}
#plot{display:block;width:100%;min-height:620px;object-fit:contain;background:white}
form{display:grid;grid-template-columns:1fr 230px 2fr 130px;gap:14px;padding:16px;border-top:1px solid var(--line);align-items:end}
label>span{display:block;font-size:12px;font-weight:800;margin-bottom:6px}.choices{display:flex;flex-wrap:wrap;gap:6px}.choices label{border:1px solid #aebbc5;border-radius:8px;padding:8px 10px;cursor:pointer}.choices label:has(input:checked){background:var(--ink);color:white;border-color:var(--ink)}.choices input{position:absolute;opacity:0}
input[type=text],textarea{width:100%;border:1px solid #aebbc5;border-radius:8px;padding:9px;font:inherit}textarea{min-height:55px}.save{background:var(--green);color:white;border-color:var(--green)}
.legend{font-size:12px;color:var(--muted);margin:8px 2px}.legend b:nth-child(1){color:var(--orange)}.legend b:nth-child(2){color:var(--blue)}.legend b:nth-child(3){color:var(--green)}
@media(max-width:900px){.meta{grid-template-columns:1fr 1fr}form{grid-template-columns:1fr}.toolbar{flex-wrap:wrap}}
</style></head>
<body><header><h1>Revue des désaccords — actuel vs ablation temporelle</h1><span>validation uniquement · seuil simple figé</span></header>
<main>
<div class="toolbar"><button id="prev">Précédent</button><button id="next">Suivant</button><div class="view"><button id="filtered" class="active" type="button">Filtré 7–80 kHz</button><button id="raw" type="button">Brut</button></div><div id="progress"></div></div>
<div class="legend"><b>Orange</b> : actuel seul · <b>Bleu</b> : simple seul · <b>Vert</b> : apparié</div>
<section class="card"><div class="meta" id="meta"></div><img id="plot" alt="Comparaison des détections"><form id="form">
<div><span style="display:block;font-size:12px;font-weight:800;margin-bottom:6px">Quelle lecture te paraît meilleure ?</span><div class="choices">
<label><input type="radio" name="preference" value="current">Actuel</label>
<label><input type="radio" name="preference" value="simple">Simple</label>
<label><input type="radio" name="preference" value="equivalent">Équivalent</label>
<label><input type="radio" name="preference" value="neither">Aucun</label>
<label><input type="radio" name="preference" value="uncertain">Incertain</label>
</div></div>
<label><span>Reviewer</span><input id="reviewer" type="text" required></label>
<label><span>Commentaire</span><textarea id="notes"></textarea></label>
<button class="save" type="submit">Sauver + suivant</button>
</form></section></main>
<script>
let items=[],cursor=0,current=null,signalView="filtered";const $=id=>document.getElementById(id);
async function get(url,options){const r=await fetch(url,options);const d=await r.json();if(!r.ok)throw new Error(d.error||"Erreur");return d}
function esc(v){return String(v).replace(/[&<>'"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[c]))}
async function loadItems(){const state=await get("/api/items");items=state.items;const first=items.findIndex(x=>!x.complete);cursor=first>=0?first:0;await load()}
async function load(){current=await get(`/api/item?index=${items[cursor].index}`);const r=current.row;$("progress").textContent=`${cursor+1} / ${items.length} · ${items.filter(x=>x.complete).length} revues`;$("prev").disabled=cursor===0;$("next").disabled=cursor===items.length-1;$("meta").innerHTML=[[\"Trace\",r.record_id],[\"Groupe\",r.source_group],[\"Chemin\",r.relative_path],[\"Actuel seul\",r.n_current_only],[\"Simple seul\",r.n_simple_only]].map(([k,v])=>`<div><span>${esc(k)}</span><strong>${esc(v)}</strong></div>`).join(\"\");document.querySelectorAll('[name=preference]').forEach(x=>x.checked=x.value===r.review_preference);$("reviewer").value=r.reviewer||localStorage.getItem("yeast-ablation-reviewer")||"";$("notes").value=r.review_notes||"";updatePlot()}
function updatePlot(){$("plot").src=`${current.plot_url}?signal_view=${signalView}&v=${Date.now()}`;$("filtered").classList.toggle("active",signalView==="filtered");$("raw").classList.toggle("active",signalView==="raw")}
$("prev").onclick=()=>{if(cursor>0){cursor--;load()}};$("next").onclick=()=>{if(cursor<items.length-1){cursor++;load()}};$("filtered").onclick=()=>{signalView="filtered";updatePlot()};$("raw").onclick=()=>{signalView="raw";updatePlot()};
$("form").onsubmit=async e=>{e.preventDefault();const selected=document.querySelector('[name=preference]:checked');if(!selected){alert("Choisis une préférence");return}const values={review_preference:selected.value,reviewer:$("reviewer").value,review_notes:$("notes").value};localStorage.setItem("yeast-ablation-reviewer",values.reviewer);await get("/api/item",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({index:current.index,values})});items[cursor].complete=true;if(cursor<items.length-1)cursor++;await load()};
loadItems().catch(e=>alert(e.message));
</script></body></html>
"""
