from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .yeast_budding_simulation import compare_budding_models


FINAL_CLASSES = ("budding", "mix", "shmoo")
PARAMETERS = (
    "log_A_A",
    "fD_A_khz",
    "log_tau_A_ms",
    "snr_db",
    "log_B_over_A",
    "delta_t0_ms",
    "delta_fD_khz",
    "delta_phi_rad",
    "log_tau_B_over_tau_A",
)
FIT_GUARD_SAMPLES = 250
DELTA_BIC_THRESHOLD = 10.0
RESOLVABILITY_THRESHOLD = 0.1
METHOD_EVIDENCE_ID = "yeast-physics-grounded-classifier-method"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_method_approval(review_dir: Path) -> dict[str, Any]:
    receipt_path = review_dir / "review" / "receipt.json"
    decisions_path = review_dir / "review" / "decisions.json"
    if not receipt_path.exists() or not decisions_path.exists():
        raise PermissionError("Phase 0 method review is incomplete")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    if receipt.get("run_id") != "yeast-physics-grounded-classifier-method-r1":
        raise PermissionError("Unexpected Phase 0 review receipt")
    if receipt.get("decisions_sha256") != sha256_file(decisions_path):
        raise PermissionError("Phase 0 review decision hash mismatch")
    decision = decisions.get("decisions", {}).get(METHOD_EVIDENCE_ID, {})
    if decisions.get("complete") is not True or decision.get("decision") != "approved":
        raise PermissionError("Phase 0 method is not approved")
    return {"receipt": str(receipt_path), "receipt_sha256": sha256_file(receipt_path)}


def stable_shard(event_id: str, shard_count: int) -> int:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    digest = hashlib.sha256(event_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % shard_count


def load_split_roles(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("external_holdout_status") != "closed":
        raise ValueError("The physics-grounded split must keep development_validation closed")
    assignments = payload.get("assignments")
    if not isinstance(assignments, dict):
        raise ValueError("Split manifest has no assignments")
    roles: dict[str, str] = {}
    for role in ("train_core", "model_selection"):
        values = assignments.get(role)
        if not isinstance(values, list):
            raise ValueError(f"Split manifest is missing {role} assignments")
        for sample_id in values:
            if sample_id in roles:
                raise ValueError(f"Duplicate split assignment: {sample_id}")
            roles[str(sample_id)] = role
    return roles


def select_strict_event_population(
    sample_rows: Sequence[dict[str, str]],
    split_roles: dict[str, str],
    *,
    classes: Iterable[str] = FINAL_CLASSES,
) -> list[dict[str, Any]]:
    requested = tuple(classes)
    unknown = set(requested) - set(FINAL_CLASSES)
    if unknown:
        raise ValueError(f"Unsupported event classes: {sorted(unknown)}")
    selected: list[dict[str, Any]] = []
    for row in sample_rows:
        if row.get("sample_kind") != "event" or row.get("quality") != "strict":
            continue
        if row.get("class_name") not in requested:
            continue
        if row.get("development_split") != "development_train":
            continue
        sample_id = row["sample_id"]
        role = split_roles.get(sample_id)
        if role not in {"train_core", "model_selection"}:
            raise ValueError(f"Strict development event is absent from the frozen split: {sample_id}")
        selected.append({**row, "role": role})
    selected.sort(key=lambda row: row["event_id"])
    event_ids = [row["event_id"] for row in selected]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("Strict event population contains duplicate event_id values")
    if set(row["class_name"] for row in selected) != set(requested):
        raise ValueError("Selected event population does not contain every requested class")
    return selected


def input_event_bounds(row: dict[str, str]) -> tuple[float, float]:
    crop_start = float(row["crop_start"])
    start = (float(row["event_start"]) - crop_start) / 2.0
    end = (float(row["event_end"]) - crop_start) / 2.0
    if not np.isfinite([start, end]).all() or end <= start:
        raise ValueError(f"Invalid event bounds for {row['event_id']}")
    return float(start), float(end)


def support_is_observed(
    start: float,
    end: float,
    *,
    signal_length: int = 4096,
    guard_samples: int = FIT_GUARD_SAMPLES,
) -> bool:
    return bool(start - guard_samples >= 0.0 and end + guard_samples <= signal_length)


def _snr_db(signal: np.ndarray, start: float, end: float) -> float:
    values = np.asarray(signal, dtype=np.float64).squeeze()
    left = max(0, int(math.floor(start)))
    right = min(values.size, int(math.ceil(end)))
    if right <= left:
        raise ValueError("Invalid event support for SNR")
    outside = np.ones(values.size, dtype=bool)
    outside[left:right] = False
    event_rms = float(np.sqrt(np.mean(np.square(values[left:right]))))
    noise_rms = float(np.sqrt(np.mean(np.square(values[outside]))))
    return float(20.0 * np.log10(max(event_rms, 1.0e-12) / max(noise_rms, 1.0e-12)))


def _wrap_phase(value: float) -> float:
    return float(np.angle(np.exp(1j * float(value))))


def _flatten_model(prefix: str, payload: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {
        f"{prefix}_bic": payload["bic"],
        f"{prefix}_envelope_residual_fraction": payload["envelope_residual_fraction"],
        f"{prefix}_waveform_residual_fraction": payload["waveform_residual_fraction"],
        f"{prefix}_fit_start_ms": payload["fit_start_ms"],
        f"{prefix}_fit_end_ms": payload["fit_end_ms"],
        f"{prefix}_n_fit_points": payload["n_fit_points"],
    }
    for component_index, component in enumerate(payload["components"], start=1):
        for name, value in component.items():
            output[f"{prefix}_c{component_index}_{name}"] = value
    return output


def _canonical_parameters(m2: dict[str, Any], snr_db: float) -> dict[str, float]:
    first, second = sorted(
        m2["components"],
        key=lambda component: (float(component["center_ms"]), -float(component["amplitude"])),
    )
    amplitude_a = float(first["amplitude"])
    amplitude_b = float(second["amplitude"])
    tau_a = 0.5 * (float(first["sigma_left_ms"]) + float(first["sigma_right_ms"]))
    tau_b = 0.5 * (float(second["sigma_left_ms"]) + float(second["sigma_right_ms"]))
    if min(amplitude_a, amplitude_b, tau_a, tau_b) <= 0.0:
        raise ValueError("Canonical M2 amplitudes and widths must be positive")
    return {
        "log_A_A": float(np.log(amplitude_a)),
        "fD_A_khz": float(first["frequency_khz"]),
        "log_tau_A_ms": float(np.log(tau_a)),
        "snr_db": float(snr_db),
        "log_B_over_A": float(np.log(amplitude_b / amplitude_a)),
        "delta_t0_ms": float(second["center_ms"] - first["center_ms"]),
        "delta_fD_khz": float(second["frequency_khz"] - first["frequency_khz"]),
        "delta_phi_rad": _wrap_phase(float(second["phase_rad"]) - float(first["phase_rad"])),
        "log_tau_B_over_tau_A": float(np.log(tau_b / tau_a)),
        "shape_A": float(first["shape"]),
        "shape_B": float(second["shape"]),
        "sigma_left_over_tau_A": float(first["sigma_left_ms"]) / tau_a,
        "sigma_right_over_tau_A": float(first["sigma_right_ms"]) / tau_a,
        "sigma_left_over_tau_B": float(second["sigma_left_ms"]) / tau_b,
        "sigma_right_over_tau_B": float(second["sigma_right_ms"]) / tau_b,
        "chirp_A_khz_per_ms": float(first["chirp_khz_per_ms"]),
        "chirp_B_khz_per_ms": float(second["chirp_khz_per_ms"]),
    }


def fit_event(row: dict[str, Any], signal: np.ndarray) -> dict[str, Any]:
    start, end = input_event_bounds(row)
    observable = support_is_observed(start, end)
    base: dict[str, Any] = {
        "sample_id": row["sample_id"],
        "event_id": row["event_id"],
        "record_id": row["record_id"],
        "capture_block_id": row["capture_block_id"],
        "class_name": row["class_name"],
        "source_group_original": row["source_group_original"],
        "role": row["role"],
        "signal_row": int(row["signal_row"]),
        "event_start_input": start,
        "event_end_input": end,
        "fit_guard_samples": FIT_GUARD_SAMPLES,
        "support_observable": observable,
        "source_centered_crop_8192_pad_left": int(row.get("crop_8192_pad_left", 0)),
        "source_centered_crop_8192_pad_right": int(row.get("crop_8192_pad_right", 0)),
        "fit_error": "",
    }
    try:
        comparison = compare_budding_models(
            row["event_id"],
            np.asarray(signal, dtype=np.float32),
            event_start_index=start,
            event_end_index=end,
        ).to_dict()
        delta_bic = float(comparison["delta_bic_m1_minus_m2"])
        resolvability = float(comparison["resolvability_score"])
        canonical = _canonical_parameters(comparison["m2"], _snr_db(signal, start, end))
        flattened = {
            **_flatten_model("m1", comparison["m1"]),
            **_flatten_model("m2", comparison["m2"]),
        }
        numeric = [delta_bic, resolvability, *canonical.values(), *flattened.values()]
        finite = bool(np.isfinite(np.asarray(numeric, dtype=np.float64)).all())
        resolved = bool(
            finite
            and delta_bic >= DELTA_BIC_THRESHOLD
            and resolvability >= RESOLVABILITY_THRESHOLD
        )
        eligible = bool(resolved and observable)
        fit_weight = (
            math.sqrt(
                float(np.clip(delta_bic / 50.0, 0.0, 1.0))
                * float(np.clip(resolvability, 0.0, 1.0))
            )
            if eligible
            else 0.0
        )
        reasons: list[str] = []
        if not finite:
            reasons.append("non_finite")
        if delta_bic < DELTA_BIC_THRESHOLD:
            reasons.append("delta_bic_below_10")
        if resolvability < RESOLVABILITY_THRESHOLD:
            reasons.append("resolvability_below_0.1")
        if not observable:
            reasons.append("support_guard_not_observed")
        return {
            **base,
            "fit_success": True,
            "fit_finite": finite,
            "fit_resolved": resolved,
            "fit_eligible": eligible,
            "fit_weight": fit_weight,
            "eligibility_reason": "eligible" if eligible else ";".join(reasons),
            "delta_bic_m1_minus_m2": delta_bic,
            "resolvability_score": resolvability,
            **canonical,
            **flattened,
        }
    except Exception as error:  # preserve a row for every requested event
        return {
            **base,
            "fit_success": False,
            "fit_finite": False,
            "fit_resolved": False,
            "fit_eligible": False,
            "fit_weight": 0.0,
            "eligibility_reason": "fit_error",
            "fit_error": f"{type(error).__name__}: {error}",
        }


def fit_shard(
    population: Sequence[dict[str, Any]],
    signals: np.ndarray,
    *,
    shard_index: int,
    shard_count: int,
    max_events: int = 0,
    max_events_per_class: int = 0,
) -> list[dict[str, Any]]:
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be inside [0, shard_count)")
    candidate_population = list(population)
    if max_events_per_class > 0:
        candidate_population = [
            row
            for class_name in FINAL_CLASSES
            for row in [item for item in population if item["class_name"] == class_name][
                :max_events_per_class
            ]
        ]
    selected = [
        row
        for row in candidate_population
        if stable_shard(row["event_id"], shard_count) == shard_index
    ]
    if max_events > 0:
        selected = selected[:max_events]
    return [fit_event(row, np.asarray(signals[int(row["signal_row"])])) for row in selected]


def _as_bool(value: Any) -> bool:
    return value is True or str(value).strip().lower() == "true"


def _effective_sample_size(weights: np.ndarray) -> float:
    denominator = float(np.sum(np.square(weights)))
    return float(np.square(np.sum(weights)) / denominator) if denominator > 0.0 else 0.0


def supervision_status(eligible: int, observable: int) -> str:
    rate = eligible / observable if observable else 0.0
    if eligible >= 300 and rate >= 0.5:
        return "ready"
    if eligible >= 100:
        return "partial"
    return "synthetic_only"


def summarize_groups(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[tuple[str, list[dict[str, Any]]]] = []
    for class_name in FINAL_CLASSES:
        groups.append((class_name, [row for row in rows if row["class_name"] == class_name]))
    groups.extend(
        (
            subgroup,
            [
                row
                for row in rows
                if row["class_name"] == "shmoo"
                and row["source_group_original"] == source
            ],
        )
        for subgroup, source in (("shmoo1", "shmoo"), ("shmoo2", "shmoo2"))
    )
    output: list[dict[str, Any]] = []
    for group_name, selected in groups:
        if not selected:
            continue
        observable = [row for row in selected if _as_bool(row["support_observable"])]
        eligible = [row for row in selected if _as_bool(row["fit_eligible"])]
        weights = np.asarray([float(row["fit_weight"]) for row in eligible], dtype=np.float64)
        delta = np.asarray(
            [float(row["delta_bic_m1_minus_m2"]) for row in selected if row.get("delta_bic_m1_minus_m2") not in (None, "")],
            dtype=np.float64,
        )
        resolvability = np.asarray(
            [float(row["resolvability_score"]) for row in selected if row.get("resolvability_score") not in (None, "")],
            dtype=np.float64,
        )
        output.append(
            {
                "group": group_name,
                "is_final_class": group_name in FINAL_CLASSES,
                "total_events": len(selected),
                "fit_success": sum(_as_bool(row["fit_success"]) for row in selected),
                "observable_events": len(observable),
                "eligible_fits": len(eligible),
                "eligible_fraction_of_observable": len(eligible) / len(observable) if observable else 0.0,
                "effective_sample_size": _effective_sample_size(weights),
                "median_fit_weight": float(np.median(weights)) if weights.size else 0.0,
                "median_delta_bic": float(np.median(delta)) if delta.size else float("nan"),
                "median_resolvability": float(np.median(resolvability)) if resolvability.size else float("nan"),
                "threshold_status": supervision_status(len(eligible), len(observable)),
            }
        )
    return output


def parameter_statistics(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for class_name in FINAL_CLASSES:
        selected = [
            row for row in rows if row["class_name"] == class_name and _as_bool(row["fit_eligible"])
        ]
        for parameter in PARAMETERS:
            values = np.asarray([float(row[parameter]) for row in selected], dtype=np.float64)
            if not values.size:
                continue
            output.append(
                {
                    "class_name": class_name,
                    "parameter": parameter,
                    "n": int(values.size),
                    "q01": float(np.quantile(values, 0.01)),
                    "q10": float(np.quantile(values, 0.10)),
                    "q50": float(np.quantile(values, 0.50)),
                    "q90": float(np.quantile(values, 0.90)),
                    "q99": float(np.quantile(values, 0.99)),
                }
            )
    return output


def select_gallery(rows: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for class_name in FINAL_CLASSES:
        class_rows = [row for row in rows if row["class_name"] == class_name and _as_bool(row["fit_success"])]
        eligible = [row for row in class_rows if _as_bool(row["fit_eligible"])]
        if eligible:
            median = float(np.median([float(row["m2_waveform_residual_fraction"]) for row in eligible]))
            retained = min(
                eligible,
                key=lambda row: (
                    abs(float(row["m2_waveform_residual_fraction"]) - median),
                    row["event_id"],
                ),
            )
            selected.append({"class_name": class_name, "role": "retained", "event_id": retained["event_id"]})
        observable = [row for row in class_rows if _as_bool(row["support_observable"])]
        if observable:
            limit = min(
                observable,
                key=lambda row: (
                    abs(float(row["delta_bic_m1_minus_m2"]) - DELTA_BIC_THRESHOLD)
                    + 50.0 * abs(float(row["resolvability_score"]) - RESOLVABILITY_THRESHOLD),
                    row["event_id"],
                ),
            )
            selected.append({"class_name": class_name, "role": "limit", "event_id": limit["event_id"]})
        rejected = [row for row in class_rows if not _as_bool(row["fit_eligible"])]
        if rejected:
            reject = max(
                rejected,
                key=lambda row: (
                    float(row.get("m2_waveform_residual_fraction") or -1.0),
                    row["event_id"],
                ),
            )
            selected.append({"class_name": class_name, "role": "rejected", "event_id": reject["event_id"]})
    return selected


def validate_merged_population(
    population: Sequence[dict[str, Any]],
    fit_rows: Sequence[dict[str, Any]],
) -> None:
    expected = [row["event_id"] for row in population]
    observed = [row["event_id"] for row in fit_rows]
    duplicates = sorted(event_id for event_id, count in Counter(observed).items() if count > 1)
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    if duplicates or missing or extra or len(expected) != len(observed):
        raise ValueError(
            "Shard merge mismatch: "
            f"duplicates={duplicates[:3]} missing={missing[:3]} extra={extra[:3]}"
        )
