# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Masked learning, and the method's own choices
#
# **How do we know that simulated particle signals resemble real ones — and how
# closely?**
#
# A flow-cytometry instrument records a particle crossing an acoustic beam as a
# short damped oscillation buried in noise. We want to train a model on such
# events, and there are only about two thousand real ones that anyone has
# annotated. So we simulate. This notebook is the case that the simulation is
# good enough to learn from, built in the order the case has to be made:
#
# ## What this is ultimately for
#
# Everything here is measured on **calibration beads** — 2, 4 and 10 µm spheres,
# whose size is known because someone put them in the tube. Beads are not the
# goal. They are the only population where a label can be trusted, which makes
# them the one place a simulator can be *checked* rather than believed.
#
# The goal is **yeast**: cells whose morphology varies continuously, whose class
# cannot be read off a filename, and for which no trustworthy per-event label
# exists at all. That is why the project is self-supervised, and it is why the
# bead chain has to be validated first — a method that cannot be shown to work
# where the truth is known has no business being pointed at data where it is
# not. Every number below is a bead number, and none of them transfers to yeast
# by assumption.
#
# **Part I — the argument.** The signal family and its knobs; whether a trained
# encoder recovers those knobs; how measured events become a generator; whether
# the generator's cloud covers the real one; and whether a regenerated event can
# still find the real event it came from.
#
# **Part II — the alignments.** Exploratory sections that put the method's own
# inconsistencies on one axis and measure them, rather than describing them.
#
# It replaces the `ssl-v18-complete` presentation rather than summarising it:
# every claim that deck makes should be here, with the code that produces it.
#
# ## How to read it
#
# - **Reproduce first, then explore.** Where a manifested run already owns a
#   number, the notebook recomputes it and asserts the match, so a changed digit
#   is a signal rather than a surprise. Those cells print
#   `reproduces <run-id> exactly`.
# - **Limits are content.** Where a result cannot be recomputed — an unsynced
#   checkpoint, a figure imported by hand — the notebook says so, names the
#   cause, and shows the published artefact rather than hiding the gap.
# - **The mathematics is not here.** Every method is imported from an installed
#   package (`internship_workspace`, `particles2snr`, `p3_ssl`, `p0`). Cells
#   orchestrate and plot. That is what keeps this notebook and the manifested
#   analyses from drifting apart.
# - **Sealed test rows are never loaded.** Cells raise if one appears.
#
# Detection — how a raw time series becomes a bounded event at all — is the
# companion notebook [`mad_detection_explainer`](mad_detection_explainer.py).
# This one starts where that one stops.
#
# ## Running it
#
# ```bash
# .venv/bin/python -m internship_workspace.cli notebooks execute \
#     particles2SNR-pipeline/notebooks/simulation_chain_explainer.py \
#     --run-id <run-id>
# ```
#
# That converts and executes the tracked source headlessly, top to bottom, which
# is also the acceptance test. Sections that produce a measurement no existing
# run owns emit their own manifested run under `artifacts/`; under an
# interactive kernel they print a refusal and continue, so the failure mode is a
# missing run rather than a false one.

#
# How the model is trained and what the two masking policies do, then the
# exploratory sections: the method carries choices made once and never
# compared — a window, a quantile, a basis. Each can flatter or challenge
# the simulator, and none is visible in a result quoted without it.
#
# ## The other notebooks in this series
#
# - [`simulation_chain_1_generator`](simulation_chain_1_generator.py) — the generator
# - [`simulation_chain_2_evaluation`](simulation_chain_2_evaluation.py) — the evaluation


# %% [markdown]
# ## Setup
#
# Everything below imports installed packages. No method is defined in this
# notebook: the cells orchestrate and plot, and the mathematics stays where the
# tools read it from, which is what keeps the notebook and the manifested
# analyses from drifting apart.

# %%
# The inline backend has to be requested explicitly, and then defended: several
# installed analysis modules call `matplotlib.use("Agg")` at import time, which
# is right for a headless tool and wrong here -- every figure drawn after such
# an import would be computed and silently discarded. Re-asserting the backend
# before each cell costs nothing and removes a whole class of empty notebook.
%matplotlib inline

import csv
import json
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

INLINE_BACKEND = "module://matplotlib_inline.backend_inline"


def _keep_inline_backend(*_):
    if matplotlib.get_backend() != INLINE_BACKEND:
        matplotlib.use(INLINE_BACKEND, force=True)


get_ipython().events.register("pre_run_cell", _keep_inline_backend)  # noqa: F821
plt.rcParams["figure.dpi"] = 110

from internship_workspace import notebook_evidence
from internship_workspace.config import Workspace, WorkspaceError
from internship_workspace.datasets import resolve_path, select_record
from internship_workspace.z8_domain_pca import CLASS_ORDER

workspace = Workspace.load()

CLASS_COLOUR = {"2um": "#2563eb", "4um": "#0f766e", "10um": "#b45309"}
SAMPLING_HZ = 2_000_000.0

registered = {}


def dataset_root(key: str):
    """Resolve `dataset-id@version` through the registry, never by path."""
    dataset_id, _, version = key.rpartition("@")
    record = select_record(workspace, dataset_id, version)
    registered[key] = record
    return resolve_path(workspace, record)


def dataset_provenance():
    """The manifest hash of every dataset resolved so far."""
    return {key: record.payload["manifest_sha256"] for key, record in registered.items()}


def run_dir(run_id: str):
    """Locate a manifested run, wherever its family stores it."""
    for family in ("analyses", "runs", "reports", "evaluations"):
        candidate = workspace.artifacts_root / "cross-project" / family / run_id
        if candidate.is_dir():
            return candidate
    for candidate in workspace.artifacts_root.rglob(run_id):
        if candidate.is_dir() and (candidate / "run.json").is_file():
            return candidate
    raise FileNotFoundError(f"no manifested run named {run_id}")


def published(run_id: str, name: str = "metrics.json"):
    """Read a manifested run's metrics, to check this notebook against it."""
    return json.loads((run_dir(run_id) / name).read_text())


def show_image(path, *, ax=None, title=None, figsize=(11, 5)):
    """Display a published figure that this notebook cannot recompute."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.imshow(plt.imread(str(path)))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)
    return ax


# %% [markdown]
# ### Figure helpers
#
# Gathered in one place so the argument below reads as argument. Each
# takes `ax=` or `axes=`, so a cell can redraw into an existing figure
# instead of building another one. They hold no science: the numbers
# arrive already computed, from the same functions the analysis tools
# call.

# %%
# --- masked_learning ---
"""Figures, tables and mask plumbing for the masked-reconstruction section.

No masking logic is defined here. Every mask comes from `p3_ssl.masking`, every
cycle from the shipped `build_balanced_event_mask_cycle` at the bounds
`p3_ssl.bead_ssl` expands for the trainer, and every published number from a
manifested run. What lives here is the bookkeeping around them: converting a
mask into its runs, walking a corpus sample, reading five histories, and
formatting the tables the argument reads.
"""

import statistics

from p3_ssl.bead_ssl import _expanded_event_bounds
from p3_ssl.masking import build_balanced_event_mask_cycle, mask_spans

P25_COLOUR = "#e2483f"
CYCLIC_COLOUR = "#00a3c7"
HIDDEN_COLOUR = "#f6c8c4"
EVENT_COLOUR = "#dcedf4"
BACKGROUND_COLOUR = "#f3ddb0"
INK = "#243447"
MUTED = "#617186"


def _bare(axis):
    axis.set_yticks([])
    axis.tick_params(labelsize=8, colors=MUTED)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color(MUTED)
    return axis


def mask_geometry(mask):
    """How a policy spends its budget: how many runs of hidden samples, how long."""
    spans = mask_spans(np.asarray(mask, dtype=bool))
    lengths = [end - start for start, end in spans]
    return {"hidden": int(np.sum(mask)), "spans": len(spans),
            "longest": int(max(lengths)), "median": int(np.median(lengths))}


def print_mask_geometry(measured, input_length):
    """The geometry table, and the budget check that makes the policies comparable."""
    for policy, values in measured.items():
        print(f"{policy:9s} {values['hidden']} hidden  {values['spans']:3d} runs  "
              f"longest {values['longest']:3d}  median run {values['median']}")
    print(f"budget check: both hide {next(iter(measured.values()))['hidden']} of "
          f"{input_length} samples — matched by construction, not by tuning")


def cyclic25_cycle(support, *, index, spec, settings, length, seed):
    """One CYCLIC25 cycle for a support, exactly as a training run built it.

    The bounds come from the shipped `_expanded_event_bounds`, which widens a
    support narrower than one pass's event budget, and the seed from the runs'
    own derivation `seed + sample_index * 7919`. The scheduling itself is
    `build_balanced_event_mask_cycle`; nothing about it is redefined here.
    """
    event = np.zeros(length, dtype=bool)
    event[support[0] : support[1]] = True
    start, end = _expanded_event_bounds(
        event, minimum_points=int(settings["event_windows_per_pass"])
        * int(settings["candidate_size"]),
    )
    expanded = np.zeros(length, dtype=bool)
    expanded[start:end] = True
    return build_balanced_event_mask_cycle(
        expanded,
        spec,
        np.random.default_rng(seed + index * 7919),
        event_windows_per_pass=int(settings["event_windows_per_pass"]),
        background_windows_per_pass=int(settings["background_windows_per_pass"]),
        require_context_each_side=bool(settings["require_context_each_side"]),
    )


def cycle_group(cycle, group="", *, pass_index=None):
    """A cycle's masks as booleans: every pass, or one, of `event`/`background`."""
    key = f"{group}_target_time_masks" if group else "target_time_masks"
    masks = np.asarray(cycle[key], dtype=bool)
    return masks if pass_index is None else masks[pass_index]


def scan_cycle_defect(indices, declared, true, *, spec, settings, length, seed):
    """Where each event's aimed budget actually landed, over a corpus sample.

    For every sampled event the cycle is rebuilt as trained — from the declared
    support — and once more from the support the data implies, to see what a
    correction would cost. Three quantities come back per event: the share of
    the background budget that lies inside the true event, the share of the true
    support the event group ever hides over a whole cycle, and the cycle length.
    """
    background_inside, event_coverage = [], []
    declared_passes, corrected_passes, corrected_failures = [], [], 0
    for index in indices:
        index = int(index)
        support = np.zeros(length, dtype=bool)
        support[true[0][index] : true[1][index]] = True
        cycle = cyclic25_cycle((declared[0][index], declared[1][index]), index=index,
                               spec=spec, settings=settings, length=length, seed=seed)
        background = cycle_group(cycle, "background")
        events = cycle_group(cycle, "event")
        background_inside.append(
            float((background & support).sum()) / float(background.sum())
        )
        event_coverage.append(
            float((events.any(axis=0) & support).sum()) / float(support.sum())
        )
        declared_passes.append(int(cycle["pass_count"]))
        try:
            corrected_passes.append(int(cyclic25_cycle(
                (true[0][index], true[1][index]), index=index, spec=spec,
                settings=settings, length=length, seed=seed)["pass_count"]))
        except (ValueError, RuntimeError):
            corrected_failures += 1
    return {
        "background_inside": np.asarray(background_inside),
        "event_group_coverage": np.asarray(event_coverage),
        "declared_passes": declared_passes,
        "corrected_passes": corrected_passes,
        "corrected_failures": corrected_failures,
    }


def print_cycle_defect(scan, sampled, seconds):
    """What the corpus scan found, in the four numbers the section argues from."""
    print(f"{sampled} events, cycles rebuilt from p3_ssl.masking ({seconds:.0f} s)")
    print(f"  background budget landing inside the true event: "
          f"median {100 * np.median(scan['background_inside']):.1f} %, "
          f"mean {100 * scan['background_inside'].mean():.1f} %")
    print(f"  true support ever hidden by the event group over a whole cycle: "
          f"median {100 * np.median(scan['event_group_coverage']):.1f} %, "
          f"mean {100 * scan['event_group_coverage'].mean():.1f} %")
    print(f"  passes per cycle: median {int(np.median(scan['declared_passes']))} as "
          f"trained, {int(np.median(scan['corrected_passes']))} corrected")
    print(f"  events for which the corrected support admits no cycle at all: "
          f"{scan['corrected_failures']} of {sampled} "
          f"({100 * scan['corrected_failures'] / sampled:.1f} %)")


def print_cycle_case(label, cycle, true_support):
    """The corpus scan's two quantities, for the single trace on screen."""
    background = cycle_group(cycle, "background")
    events = cycle_group(cycle, "event")
    inside = 100 * float((background & true_support).sum()) / float(background.sum())
    hidden = 100 * float((events.any(axis=0) & true_support).sum()) / float(true_support.sum())
    print(f"{label:11s} {int(cycle['pass_count'])} passes · "
          f"{inside:5.1f} % of the background budget lies inside the true event · "
          f"event group ever hides {hidden:5.1f} % of the true support")


def envelope_fwhm_samples(signals, *, smoothing):
    """Full width at half maximum of each trace's smoothed analytic envelope."""
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import hilbert

    envelope = uniform_filter1d(
        np.abs(hilbert(np.asarray(signals, dtype=np.float64), axis=1)),
        size=smoothing, axis=1,
    )
    return np.asarray(
        [np.flatnonzero(row >= 0.5 * row.max())[[0, -1]] @ [-1, 1] + 1 for row in envelope],
        dtype=float,
    )


def print_support_comparison(declared, true, captured, events):
    """The support the trainer aimed at, beside the one the data implies."""
    print(f"over all {events:,} v5 events")
    for label, statistic in (("median support", np.median),
                             ("p95 support", lambda values: np.percentile(values, 95)),
                             ("under half crop", lambda values: np.mean(values < 0.5))):
        print(f"  {label:16s} declared {100 * statistic(declared):5.1f} %   "
              f"true {100 * statistic(true):5.1f} %")
    print(f"  fraction of the true support inside the declared one: "
          f"median {np.median(captured):.3f}")


def matched_monitor_curves(run_of, policies, seeds):
    """The fixed-monitor masked MSE of every matched run, epoch by epoch.

    Reads `matched_monitor.<split>.model.masked_mse` from each run's
    `history.json` — the protocol that evaluates train and validation the same
    way, so the two differ only by the split.
    """
    monitor = {}
    for policy in policies:
        histories = [
            json.loads((run_of(policy.lower(), seed) / "history.json").read_text())
            for seed in seeds
        ]
        curves = {
            split: [[entry["matched_monitor"][split]["model"]["masked_mse"]
                     for entry in history] for history in histories]
            for split in ("train_eval", "validation")
        }
        final = {
            split: statistics.fmean(curve[-1] for curve in rows)
            for split, rows in curves.items()
        }
        monitor[policy] = {
            "epochs": [entry["epoch"] for entry in histories[0]],
            "train_eval": curves["train_eval"],
            "validation": curves["validation"],
            "final_train_eval": final["train_eval"],
            "final_validation": final["validation"],
            "gap_percent": 100.0 * (final["validation"] - final["train_eval"])
            / final["train_eval"],
        }
        print(f"{policy:9s} final fixed-monitor masked MSE  "
              f"train {final['train_eval']:.6f}  validation {final['validation']:.6f}  "
              f"gap {monitor[policy]['gap_percent']:+.4f} %")
    return monitor


def cross_mask_cells(rows, policies):
    """Every trained model on every evaluation regime, averaged over its seeds."""
    cells, zero = {}, {}
    for row in rows:
        key = (row["training_mask_policy"], row["evaluation_mask_policy"])
        cells.setdefault(key, []).append(row["real_validation"]["model"]["masked_mse"])
        zero.setdefault(row["evaluation_mask_policy"], []).append(
            row["real_validation"]["zero"]["masked_mse"]
        )
    cells = {
        key: {"mean": statistics.fmean(values), "std": statistics.pstdev(values),
              "seeds": len(values)}
        for key, values in cells.items()
    }
    zero = {key: statistics.fmean(values) for key, values in zero.items()}
    print(f"{'trained':>10} {'on P25 masks':>16} {'on CYCLIC25 masks':>19}   seeds")
    for trained in policies:
        print(f"{trained:>10} {cells[(trained, 'P25')]['mean']:16.4f} "
              f"{cells[(trained, 'CYCLIC25')]['mean']:19.4f}   "
              f"{cells[(trained, 'P25')]['seeds']}")
    print(f"{'zero':>10} {zero['P25']:16.4f} {zero['CYCLIC25']:19.4f}   —")
    return cells, zero


def _support_summary(fraction, width):
    return {
        "median_fraction": float(np.median(fraction)),
        "p95_fraction": float(np.percentile(fraction, 95)),
        "fraction_under_half_crop": float(np.mean(fraction < 0.5)),
        "median_width_samples": float(np.median(width)),
    }


def mask_rate_evidence(*, declared_hz, true_hz, probe, rate_ratio, declared, true,
                       captured, subset, class_names, class_order, scan, sampled,
                       cyclic, cycle_seed, datasets, inputs):
    """Package the sampling-rate defect as a manifested run: metrics, provenance."""
    metrics = {
        "schema_version": 1,
        "analysis": "cyclic25-event-support-sampling-rate-defect",
        "declared_sampling_frequency_hz": declared_hz,
        "generator_sampling_frequency_hz": true_hz,
        "rate_determined_from_waveforms": {
            "probe_events": int(probe["events"]),
            "minimum_achieved_snr_db": float(probe["minimum_snr_db"]),
            "measured_over_predicted_fwhm_median": {
                f"{int(declared_hz)}": rate_ratio[declared_hz],
                f"{int(true_hz)}": rate_ratio[true_hz],
            },
        },
        "event_support": {
            "events": int(len(class_names)),
            "declared": _support_summary(*declared),
            "true": _support_summary(*true),
            "declared_is_subset_of_true": bool(subset),
            "median_true_support_captured": float(np.median(captured)),
            "per_class_median_fraction": {
                name: {
                    "declared": float(np.median(declared[0][class_names == name])),
                    "true": float(np.median(true[0][class_names == name])),
                }
                for name in class_order
            },
        },
        "cycle_consequence": {
            "sampled_events": int(sampled),
            "background_inside_true_event_median": float(
                np.median(scan["background_inside"])),
            "background_inside_true_event_mean": float(scan["background_inside"].mean()),
            "true_support_covered_by_event_group_median": float(
                np.median(scan["event_group_coverage"])),
            "true_support_covered_by_event_group_mean": float(
                scan["event_group_coverage"].mean()),
            "median_passes_declared": int(np.median(scan["declared_passes"])),
            "median_passes_corrected": int(np.median(scan["corrected_passes"])),
            "corrected_cycle_failures": int(scan["corrected_failures"]),
        },
        "unaffected": {
            "p25_policy": "event_mask=None, event_biased_probability=0.0",
            "hidden_points_per_pass": 1024,
            "real_validation_masks": (
                "Z8RealValidationDataset derives the support from annotated "
                "start_sample/end_sample and never reads sampling_frequency_hz"
            ),
        },
    }
    provenance = {
        "datasets": datasets,
        "inputs": inputs,
        "parameters": {
            "event_support_formula": (
                "centre = t0_fraction * (N - 1); "
                "[centre - 3*tau*exp(-a), centre + 3*tau*exp(+a)], tau in samples"
            ),
            "cyclic25": {key: cyclic[key] for key in sorted(cyclic)},
            "seed_derivation": (
                "42 + sample_index * 7919, as in build_cyclic25_masks_for_sample"
            ),
            "cycle_sample_seed": cycle_seed,
            "cycle_sample_size": int(sampled),
            "fwhm_probe": (
                "300 highest achieved-SNR v5 events, Hilbert envelope, "
                "41-sample box smoothing"
            ),
        },
        "metric_definitions": {
            "declared/true event support": (
                "the support Z8AsymmetricSyntheticDataset builds at the config's "
                "sampling_frequency_hz, and at the generator's 2 MHz"
            ),
            "background_inside_true_event": (
                "fraction of the points CYCLIC25 hides as background, over a "
                "complete cycle, that lie inside the true event support"
            ),
            "true_support_covered_by_event_group": (
                "fraction of the true support ever hidden by the cycle's event "
                "windows -- the completeness guarantee, measured against the "
                "event the generator actually drew"
            ),
            "measured_over_predicted_fwhm": (
                "envelope full width at half maximum in samples, divided by "
                "1.1774 * tau * (e^-a + e^+a) * sampling_frequency_hz"
            ),
        },
    }
    return metrics, provenance


def draw_model_view(signal, mask, *, patch_size=16, axes=None):
    """What `sample_visibility_v1` hands the encoder, on one real trace.

    Three rows because the encoding is three things at once: a corrupted
    trace, an explicit visibility channel, and a token grid the mask does not
    respect.
    """
    if axes is None:
        _, axes = plt.subplots(
            3, 1, figsize=(12.5, 5.2), height_ratios=(2.6, 0.6, 0.6), sharex=True
        )
    hidden = np.asarray(mask, dtype=bool)
    lo, hi = 1700, 2400
    time = np.arange(lo, hi)
    for start, end in mask_spans(hidden):
        if end > lo and start < hi:
            axes[0].axvspan(max(start, lo), min(end, hi), color=HIDDEN_COLOUR, zorder=0)
    axes[0].plot(time, signal[lo:hi], color=MUTED, lw=1.0, zorder=1,
                 label="true trace (never given where hidden)")
    corrupted = np.where(hidden, 0.0, signal)
    axes[0].plot(time, corrupted[lo:hi], color=INK, lw=1.2, zorder=2,
                 label="model input · targets zero-filled")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right", labelcolor=MUTED)
    axes[0].set_title("Model input · every hidden sample is replaced by zero",
                      loc="left", fontsize=10, color=INK, fontweight="bold")
    _bare(axes[0])

    axes[1].fill_between(np.arange(signal.size), 0, (~hidden).astype(float),
                         step="mid", color=CYCLIC_COLOUR, alpha=0.85, lw=0)
    axes[1].set_ylim(0, 1.2)
    axes[1].set_title("Visibility channel · 1 observed, 0 target — the mask is itself an input",
                      loc="left", fontsize=9, color=CYCLIC_COLOUR, fontweight="bold")
    _bare(axes[1])

    starts = np.arange(0, signal.size, patch_size)
    per_token = np.array([hidden[s:s + patch_size].sum() for s in starts])
    for start, count in zip(starts, per_token):
        if start + patch_size <= lo or start >= hi:
            continue
        face = ("white" if count == 0
                else HIDDEN_COLOUR if count == patch_size else BACKGROUND_COLOUR)
        axes[2].add_patch(plt.Rectangle((start, 0.15), patch_size, 0.7,
                                        facecolor=face, edgecolor=MUTED, lw=0.5))
    axes[2].set_ylim(0, 1)
    axes[2].set_xlim(lo, hi)
    axes[2].set_title(f"{patch_size}-sample encoder tokens · white visible, pink fully hidden, "
                      "amber cut by a window edge",
                      loc="left", fontsize=9, color=INK, fontweight="bold")
    axes[2].set_xlabel("sample index within the 4,096-sample crop", fontsize=9, color=MUTED)
    _bare(axes[2])
    return axes


def draw_policies(signal, p25_mask, cyclic_mask, event_span, *, axes=None):
    """Both training policies, same trace, same 25 % budget."""
    if axes is None:
        _, axes = plt.subplots(2, 1, figsize=(12.5, 4.6), sharex=True)
    time = np.arange(signal.size)
    panels = (
        (axes[0], mask_spans(np.asarray(p25_mask, dtype=bool)),
         "P25 · targets drawn blind to the signal", P25_COLOUR),
        (axes[1], mask_spans(np.asarray(cyclic_mask, dtype=bool)),
         "CYCLIC25 · targets aimed at the annotated support", CYCLIC_COLOUR),
    )
    for axis, spans, title, accent in panels:
        axis.axvspan(*event_span, color=EVENT_COLOUR, zorder=0)
        for start, end in spans:
            axis.axvspan(start, end, color=HIDDEN_COLOUR, zorder=1)
        axis.plot(time, signal, color=INK, lw=0.5, zorder=2)
        axis.set_xlim(0, signal.size)
        axis.set_ylim(-4.6, 4.6)
        lengths = [end - start for start, end in spans]
        axis.set_title(f"{title} — {len(spans)} runs, longest {max(lengths)} samples, "
                       f"{int(sum(lengths))} hidden",
                       loc="left", fontsize=10, color=accent, fontweight="bold")
        _bare(axis)
    axes[1].set_xlabel("sample index · pale blue band is the annotated event support",
                       fontsize=9, color=MUTED)
    return axes


def draw_cycle(signal, cycle, event_span, *, axes=None):
    """The CYCLIC25 cycle for one training event: passes, then coverage."""
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(12.5, 3.4),
                               gridspec_kw={"width_ratios": (2.4, 1.0)})
    pass_spans = [mask_spans(mask)
                  for mask in np.asarray(cycle["target_time_masks"], dtype=bool)]
    coverage = np.asarray(cycle["cumulative_event_window_coverage"])
    axis = axes[0]
    axis.axvspan(*event_span, color=EVENT_COLOUR, zorder=0)
    axis.plot(np.arange(signal.size),
              0.30 * signal / np.abs(signal).max() * 2.0 + len(pass_spans) + 1.2,
              color=INK, lw=0.5, zorder=2)
    for row, spans in enumerate(pass_spans):
        level = len(pass_spans) - row - 1
        for start, end in spans:
            axis.add_patch(plt.Rectangle((start, level + 0.12), end - start, 0.76,
                                         facecolor=HIDDEN_COLOUR, edgecolor="none"))
        axis.text(-40, level + 0.5, f"pass {row + 1}", ha="right", va="center",
                  fontsize=8, color=MUTED)
    axis.set_xlim(-350, signal.size)
    axis.set_ylim(0, len(pass_spans) + 2.4)
    axis.set_yticks([])
    axis.set_xlabel("sample index", fontsize=9, color=MUTED)
    axis.set_title("One pass hides 1,024 samples; the cycle hides every event sample at least once",
                   loc="left", fontsize=10, color=CYCLIC_COLOUR, fontweight="bold")
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(labelsize=8, colors=MUTED)

    passes = np.arange(1, len(coverage) + 1)
    axes[1].plot(passes, 100 * coverage, marker="o", color=CYCLIC_COLOUR)
    axes[1].axhline(100, color=MUTED, ls="--", lw=1.0)
    axes[1].set(xlabel="pass", ylabel="% of event windows hidden",
                title="Cumulative coverage")
    axes[1].set_xticks(passes)
    axes[1].set_ylim(0, 112)
    axes[1].spines[["top", "right"]].set_visible(False)
    return axes


def draw_packing(window_spans, lane_of, selection_counts, support_span, *, axes=None):
    """How a fixed budget is packed over a variable support (deck appendix)."""
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(12.5, 3.0),
                               gridspec_kw={"width_ratios": (1.7, 1.0)})
    axis = axes[0]
    axis.axvspan(*support_span, color=EVENT_COLOUR, zorder=0)
    for (start, end), lane in zip(window_spans, lane_of):
        axis.add_patch(plt.Rectangle((start, 0.55 if lane == 0 else 0.05),
                                     end - start, 0.35,
                                     facecolor=CYCLIC_COLOUR if lane == 0 else "#7dc9dd",
                                     edgecolor="white", lw=0.4))
    axis.set_xlim(support_span[0] - 60, support_span[1] + 60)
    axis.set_ylim(0, 1.0)
    axis.set_yticks([0.22, 0.72], ["lane 1", "lane 0"], fontsize=8, color=MUTED)
    axis.set_xlabel("sample index", fontsize=9, color=MUTED)
    axis.set_title("Half-overlapping candidates split into two internally disjoint lanes",
                   loc="left", fontsize=10, color=CYCLIC_COLOUR, fontweight="bold")
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(labelsize=8, colors=MUTED)

    counts = np.asarray(selection_counts)
    once = int((counts == 1).sum())
    twice = int((counts >= 2).sum())
    axes[1].bar(["hidden once", "hidden twice"], [once, twice],
                color=[CYCLIC_COLOUR, BACKGROUND_COLOUR])
    for index, value in enumerate((once, twice)):
        axes[1].text(index, value + 1, str(value), ha="center", fontsize=9, color=INK)
    axes[1].set(ylabel="candidate windows",
                title=f"{int(counts.sum())} selections over {counts.size} windows")
    axes[1].spines[["top", "right"]].set_visible(False)
    return axes


def draw_reconstruction(signal, outputs, mask, window, *, axes=None):
    """Identical hidden samples, both trained models, one real event."""
    if axes is None:
        _, axes = plt.subplots(1, len(outputs), figsize=(12.5, 3.4), sharey=True)
    lo, hi = window
    time = np.arange(lo, hi)
    spans = mask_spans(np.asarray(mask, dtype=bool))
    for axis, (label, values, accent, error) in zip(axes, outputs):
        for start, end in spans:
            if end > lo and start < hi:
                axis.axvspan(max(start, lo), min(end, hi), color=HIDDEN_COLOUR, zorder=0)
        axis.plot(time, signal[lo:hi], color=MUTED, lw=1.5, zorder=1, label="true target")
        axis.plot(time, values[lo:hi], color=accent, lw=1.6, zorder=2, label=f"{label} output")
        axis.set_xlim(lo, hi)
        axis.set_ylim(-4.6, 4.6)
        axis.set_title(f"{label} · masked MSE {error:.4f} on this event",
                       loc="left", fontsize=10, color=accent, fontweight="bold")
        axis.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=MUTED)
        axis.set_xlabel("sample index", fontsize=9, color=MUTED)
        _bare(axis)
    return axes


def draw_learning_curves(curves, *, axes=None):
    """Matched monitoring: the same protocol on both splits, five seeds."""
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(12.5, 3.4))
    for axis, (policy, accent) in zip(axes, (("P25", P25_COLOUR), ("CYCLIC25", CYCLIC_COLOUR))):
        rows = curves[policy]
        epochs = rows["epochs"]
        for seed_index, (train, validation) in enumerate(zip(rows["train_eval"], rows["validation"])):
            axis.plot(epochs, train, color=accent, lw=1.0, alpha=0.55,
                      label="fixed train monitor" if seed_index == 0 else None)
            axis.plot(epochs, validation, color=INK, lw=1.0, alpha=0.55, ls="--",
                      label="fixed validation monitor" if seed_index == 0 else None)
        axis.set_yscale("log")
        axis.set(xlabel="epoch", ylabel="masked MSE",
                 title=f"{policy} · 5 seeds · final gap {rows['gap_percent']:+.2f} %")
        axis.legend(frameon=False, fontsize=8)
        axis.spines[["top", "right"]].set_visible(False)
    return axes


def draw_cross_mask(cells, zero, *, ax=None):
    """Every model on both regimes, against the predict-zero reference."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7.4, 3.6))
    regimes = ("P25", "CYCLIC25")
    width = 0.34
    positions = np.arange(len(regimes))
    for offset, (trained, accent) in enumerate(
        (("P25", P25_COLOUR), ("CYCLIC25", CYCLIC_COLOUR))
    ):
        values = [cells[(trained, regime)]["mean"] for regime in regimes]
        errors = [cells[(trained, regime)]["std"] for regime in regimes]
        ax.bar(positions + (offset - 0.5) * width, values, width, yerr=errors,
               color=accent, label=f"trained {trained}", capsize=3)
    for index, regime in enumerate(regimes):
        ax.plot([index - 0.75 * width, index + 0.75 * width], [zero[regime]] * 2,
                color="#111827", ls="--", lw=1.4,
                label="predicting zero" if index == 0 else None)
    ax.set_yscale("log")
    ax.set_xticks(positions, [f"evaluated on {regime} masks" for regime in regimes])
    ax.set(ylabel="masked MSE on real validation (lower is better)",
           title="Cross-mask evaluation · five seeds per cell")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def draw_support_rates(declared, true, class_names, class_order, class_colour, *, axes=None):
    """The event-support fraction the trainer used, and the one the data implies."""
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(12.5, 3.4))
    bins = np.linspace(0, 1, 60)
    axes[0].hist(declared, bins=bins, color=P25_COLOUR, alpha=0.75,
                 label="mask actually built · 1 MHz declared")
    axes[0].hist(true, bins=bins, color=CYCLIC_COLOUR, alpha=0.6,
                 label="mask the data implies · 2 MHz")
    for values, colour in ((declared, P25_COLOUR), (true, CYCLIC_COLOUR)):
        axes[0].axvline(np.median(values), color=colour, ls="--", lw=1.3)
    axes[0].axvline(0.5, color="#111827", ls=":", lw=1.2)
    axes[0].set(xlabel="event support / 4,096-sample crop", ylabel="events",
                title="Half the event was labelled background")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].spines[["top", "right"]].set_visible(False)

    positions = np.arange(len(class_order))
    width = 0.36
    for offset, (values, hatch) in enumerate(((declared, ""), (true, "//"))):
        medians = [np.median(values[class_names == name]) for name in class_order]
        axes[1].bar(positions + (offset - 0.5) * width, medians, width, hatch=hatch,
                    color=[class_colour[name] for name in class_order],
                    edgecolor="white", lw=0.6)
    axes[1].set_xticks(positions, class_order)
    axes[1].set(ylabel="median support fraction", title="Same factor of two in every class")
    axes[1].legend(
        handles=[plt.Rectangle((0, 0), 1, 1, facecolor="#94a3b8", label="1 MHz declared"),
                 plt.Rectangle((0, 0), 1, 1, facecolor="#94a3b8", hatch="//",
                               edgecolor="white", label="2 MHz true")],
        frameon=False, fontsize=8,
    )
    axes[1].spines[["top", "right"]].set_visible(False)
    return axes


def draw_defect_case(signal, declared_span, true_span, actual, corrected, *, axes=None):
    """The mask one trace got, beside the mask the data says it deserved."""
    if axes is None:
        _, axes = plt.subplots(2, 1, figsize=(12.5, 4.8), sharex=True)
    time = np.arange(signal.size)
    panels = (
        (axes[0], actual, "As trained · event support from the declared 1 MHz rate"),
        (axes[1], corrected, "As it should be · event support from the true 2 MHz rate"),
    )
    for axis, cycle, title in panels:
        axis.axvspan(*true_span, color=EVENT_COLOUR, zorder=0)
        axis.axvline(declared_span[0], color=P25_COLOUR, lw=1.2, ls="--", zorder=3)
        axis.axvline(declared_span[1], color=P25_COLOUR, lw=1.2, ls="--", zorder=3)
        for start, end in mask_spans(cycle_group(cycle, "event", pass_index=0)):
            axis.axvspan(start, end, ymin=0.5, color=CYCLIC_COLOUR, alpha=0.85, zorder=1)
        for start, end in mask_spans(cycle_group(cycle, "background", pass_index=0)):
            axis.axvspan(start, end, ymin=0.5, color=BACKGROUND_COLOUR, zorder=1)
        axis.plot(time, 0.45 * signal - 2.3, color=INK, lw=0.5, zorder=2)
        axis.set_xlim(0, signal.size)
        axis.set_ylim(-4.6, 4.6)
        axis.set_title(title, loc="left", fontsize=10, color=INK, fontweight="bold")
        _bare(axis)
    axes[1].set_xlabel(
        "sample index · blue band = true event support, dashed red = declared support, "
        "cyan = windows the cycle calls event, amber = windows it calls background",
        fontsize=8.5, color=MUTED,
    )
    return axes


def draw_defect_summary(scan, *, axes=None):
    """Where the aimed budget actually landed, over a sample of the corpus."""
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(12.5, 3.2))
    axes[0].hist(100 * scan["background_inside"], bins=40, color=BACKGROUND_COLOUR,
                 edgecolor=MUTED, lw=0.4)
    axes[0].axvline(100 * np.median(scan["background_inside"]), color=P25_COLOUR,
                    ls="--", lw=1.4)
    axes[0].set(xlabel="% of the background budget inside the true event",
                ylabel="events", title="'Background' that is event")

    axes[1].hist(100 * scan["event_group_coverage"], bins=40, color=CYCLIC_COLOUR,
                 edgecolor=MUTED, lw=0.4)
    axes[1].axvline(100 * np.median(scan["event_group_coverage"]), color=P25_COLOUR,
                    ls="--", lw=1.4)
    axes[1].axvline(100, color="#111827", ls=":", lw=1.2)
    axes[1].set(xlabel="% of the true support the event group ever hides",
                title="The completeness guarantee")

    axes[2].hist([scan["declared_passes"], scan["corrected_passes"]],
                 bins=np.arange(1.5, 17.5, 1.0),
                 color=[P25_COLOUR, CYCLIC_COLOUR], label=["as trained", "corrected"])
    axes[2].set(xlabel="passes needed for a complete cycle", title="Cycle length")
    axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    return axes


# --- window_alignment ---
"""Figures, tables and bookkeeping for the window-alignment exploration.

Nothing here measures anything. The sweep's numbers arrive already computed,
from `internship_workspace.chain_data` and the shipped coverage and domain
functions; what lives here is the work the argument does not need on screen --
assembling one sweep row, formatting two tables, and the block-RMS arithmetic
behind the realism control.
"""


def window_sweep_row(window, real, basis, chain, domain, class_order):
    """One row of the sweep: everything a window decides, already measured."""
    terminal = chain["asymmetry_5d"]
    return {
        "window": int(window),
        "descriptor_dimensions": int(real.features.shape[1]),
        "explained_variance_16": basis.explained_variance,
        "coverage": {c: terminal[c]["real_within_radius_fraction"] for c in class_order},
        "nn_median": {c: terminal[c]["real_to_synthetic_nn_median"] for c in class_order},
        "radius": {c: terminal[c]["synthetic_self_nn_radius"] for c in class_order},
        "domain_auc": {
            c: float(domain[c]["domain_classifier_auc_mean"]) for c in class_order
        },
        "local_opposite_fraction": {
            c: float(domain[c]["local_opposite_domain_fraction"]) for c in class_order
        },
        "chain": {
            label: {c: chain[label][c]["real_within_radius_fraction"] for c in class_order}
            for label in chain
        },
    }


def print_sweep_line(row, class_order, seconds):
    """The running line printed as each window finishes."""
    print(f"window {row['window']:5d}  {row['descriptor_dimensions']:3d}-D  "
          f"EV16 {row['explained_variance_16']:.4f}  coverage "
          + " ".join(f"{c} {100 * row['coverage'][c]:5.1f}%" for c in class_order)
          + f"   ({seconds:.0f} s)")


def print_window_sweep(sweep, class_order):
    """Coverage beside domain AUC, because neither is readable alone."""
    print(f"{'window':>7} {'dims':>5} {'EV16':>7}   "
          + "   ".join(f"{c:>18}" for c in class_order))
    for row in sweep:
        cells = "   ".join(
            f"{100 * row['coverage'][c]:5.1f}% / {row['domain_auc'][c]:.3f}"
            for c in class_order
        )
        print(f"{row['window']:>7} {row['descriptor_dimensions']:>5} "
              f"{row['explained_variance_16']:>7.4f}   {cells}")
    print("\ncells are coverage / domain AUC; AUC nearer 0.5 means real and synthetic "
          "are harder to tell apart")


def print_support_containment(support_widths, windows):
    """How much of the annotated support each candidate window would hold."""
    print("fraction of annotated supports wider than the window")
    for window in windows:
        print(f"  {window:5d} : {100 * np.mean(support_widths > window):5.1f} %")
    print(f"\nsupport width: median {np.median(support_widths):.0f}, "
          f"p90 {np.percentile(support_widths, 90):.0f}, max {support_widths.max():.0f}")


def block_energy_profile(traces, *, block):
    """Median RMS per block of samples, over a population of traces."""
    values = np.asarray(traces, dtype=np.float64)
    blocks = values.reshape(len(values), -1, block)
    return np.median(np.sqrt(np.mean(np.square(blocks), axis=2)), axis=0)


def core_context_ratio(traces, core):
    """Median RMS inside the descriptor window over RMS outside it."""
    values = np.asarray(traces, dtype=np.float64)
    inside = np.sqrt(np.mean(np.square(values[:, core]), axis=1))
    context = np.concatenate((values[:, : core.start], values[:, core.stop :]), axis=1)
    outside = np.sqrt(np.mean(np.square(context), axis=1))
    return float(np.median(inside / outside))


def print_core_context(ratios, class_order):
    """The one number the energy profiles are read for."""
    print(f"{'class':>6} {'core/context real':>19} {'synthetic':>10}")
    for class_name in class_order:
        real_ratio, synthetic_ratio = ratios[class_name]
        print(f"{class_name:>6} {real_ratio:19.2f} {synthetic_ratio:10.2f}")


def window_evidence(sweep, *, windows, quantile, seed, bands, envelope_bins,
                    band_hz, reproduction, datasets, inputs):
    """Package the sweep as a manifested run: metrics and provenance."""
    metrics = {
        "schema_version": 1,
        "analysis": "morphology-descriptor-window-sweep",
        "windows": list(windows),
        "quantile": quantile,
        "seed": seed,
        "descriptor": (
            "shipped window-invariant morphology_features: fixed 37-band grid on "
            "7-80 kHz, envelope smoothing in bin units, the identity at 1024"
        ),
        "reproduces": reproduction,
        "sweep": sweep,
    }
    provenance = {
        "datasets": datasets,
        "inputs": inputs,
        "parameters": {
            "windows": list(windows),
            "band_hz": list(band_hz),
            "spectral_bands": int(bands),
            "envelope_bins": int(envelope_bins),
            "components": 16,
            "quantile": quantile,
            "seed": seed,
            "basis": "one PCA fitted on every condition pooled, per window",
            "sample": "one balanced synthetic draw per class, shared by conditions",
        },
        "metric_definitions": {
            "coverage": (
                "real fraction below that condition's synthetic self-nearest-"
                "neighbour 80th percentile"
            ),
            "domain_auc": (
                "five-fold cross-validated logistic separation of real from "
                "synthetic in the 16-component basis; 0.5 is indistinguishable"
            ),
            "nn_median": (
                "median euclidean distance from a real event to its nearest synthetic"
            ),
        },
    }
    return metrics, provenance


def draw_window_sweep(sweep, class_order, class_colour, *, axes=None):
    """Coverage, domain AUC and neighbour distance against descriptor window.

    Three panels because the three quantities answer the objection jointly:
    coverage alone can be inflated by a wider window, so it is only readable
    beside a separability measure and a raw distance.
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    windows = [row["window"] for row in sweep]
    panels = (
        (axes[0], "coverage", "Coverage of the real cloud · q80", "%", 100.0),
        (axes[1], "domain_auc", "Domain AUC · 0.5 means indistinguishable", "AUC", 1.0),
        (axes[2], "nn_median", "Real→synthetic nearest neighbour · median", "distance", 1.0),
    )
    for axis, key, title, ylabel, scale in panels:
        for class_name in class_order:
            axis.plot(
                windows,
                [scale * row[key][class_name] for row in sweep],
                marker="o",
                color=class_colour[class_name],
                label=class_name,
            )
        axis.set(title=title, xlabel="descriptor window (samples)", ylabel=ylabel)
        axis.set_xscale("log", base=2)
        axis.set_xticks(windows, [str(window) for window in windows])
        axis.spines[["top", "right"]].set_visible(False)
    axes[1].axhline(0.5, color="#dc2626", ls="--", lw=1.2)
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    return axes


def draw_descriptor_widths(descriptor_shapes, *, ax=None):
    """Descriptor dimension against window, with and without the fixed band grid."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 3.0))
    windows = [row["window"] for row in descriptor_shapes]
    ax.plot(windows, [row["shipped"] for row in descriptor_shapes], marker="o",
            color="#dc2626", label="length-dependent descriptor")
    ax.plot(windows, [row["invariant"] for row in descriptor_shapes], marker="s",
            color="#0f766e", label="fixed 37-band grid · shipped")
    ax.set(xlabel="descriptor window (samples)", ylabel="dimensions",
           title="A descriptor whose size depends on its window is not one descriptor")
    ax.set_xscale("log", base=2)
    ax.set_xticks(windows, [str(window) for window in windows])
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def draw_energy_profiles(profiles, counts, class_order, class_colour, *, block, core,
                         axes=None):
    """Where each population puts its energy across the widest window."""
    if axes is None:
        figure, axes = plt.subplots(1, 3, figsize=(13, 3.2), sharex=True)
    else:
        figure = axes[0].figure
    length = block * len(next(iter(profiles.values()))[0])
    positions = np.arange(length // block) * block + block / 2
    for axis, class_name in zip(axes, class_order):
        real_profile, synthetic_profile = profiles[class_name]
        axis.plot(positions, real_profile, marker="o", color="#334155", label="real")
        axis.plot(positions, synthetic_profile, marker="s",
                  color=class_colour[class_name], label="synthetic")
        axis.axvspan(core.start, core.stop, color="#94a3b8", alpha=0.25)
        axis.set(title=f"{class_name} · n = {counts[class_name]}",
                 xlabel=f"sample within the {length} window")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("median block RMS")
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        f"Shaded band is the {core.stop - core.start}-sample window the descriptor reads",
        y=1.04, fontsize=9,
    )
    figure.tight_layout()
    return axes


# --- quantile_basis ---
"""Figures, tables and plumbing for the quantile-and-basis alignment exploration.

The measurements all come from the shipped functions -- `support_coverage`,
`domain_metrics`, `chain_data`'s pooled basis. Two helpers here are exceptions
worth naming: `domain_reference_draw` rebuilds the reference sample
`domain_metrics` draws internally, so that the shipped coverage function can
score the same cloud at another quantile, and `distance_contrast` restates the
published contrast definition so it can be extended past sixteen components.
Both are gated in the section against the published run before anything is read
from them.
"""


def coverage_row(measured, class_order):
    """One condition's coverage and radius per class, ready to plot or emit."""
    return {
        class_name: {
            "coverage": measured[class_name]["real_within_radius_fraction"],
            "radius": measured[class_name]["synthetic_self_nn_radius"],
        }
        for class_name in class_order
    }


def domain_reference_draw(real_labels, synthetic_labels, *, seed, class_order):
    """The synthetic reference `domain_metrics` samples internally, rebuilt.

    Only the draw is reproduced, so that `support_coverage` can be handed the
    same cloud at a different quantile. The real draw is consumed but discarded
    because it only advances the generator.
    """
    real_values = np.asarray(real_labels).astype(str)
    synthetic_values = np.asarray(synthetic_labels).astype(str)
    draw = {}
    for class_name in class_order:
        real_index = np.flatnonzero(real_values == class_name)
        synthetic_index = np.flatnonzero(synthetic_values == class_name)
        count = min(real_index.size, synthetic_index.size)
        generator = np.random.default_rng(seed + class_order.index(class_name))
        generator.choice(real_index, size=count, replace=False)
        draw[class_name] = generator.choice(synthetic_index, size=count, replace=False)
    return draw


def distance_contrast(scores, labels, *, dimensions, class_order, sample=2000, seed):
    """std/mean of pairwise distances, the published contrast definition."""
    from scipy.spatial.distance import pdist

    output = {}
    for class_name in class_order:
        rows = np.asarray(scores)[np.asarray(labels) == class_name][:, :dimensions]
        generator = np.random.default_rng(seed + class_order.index(class_name))
        if rows.shape[0] > sample:
            rows = rows[generator.choice(rows.shape[0], size=sample, replace=False)]
        distances = pdist(rows)
        output[class_name] = float(distances.std() / distances.mean())
    return output


def print_published_gains(published_runs, quantiles, class_order):
    """What each generator step is worth, at each published quantile."""
    print(f"{'step':>34} " + "  ".join(f"{c:>16}" for c in class_order))
    for step in ("white_noise_4d -> real_noise_4d", "real_noise_4d -> asymmetry_5d"):
        for quantile in quantiles:
            gains = published_runs[quantile]["gains_percentage_points"][step]
            print(f"{step + f'  q{quantile:.2f}':>34} "
                  + "  ".join(f"{gains[c]:+15.2f}pp" for c in class_order))
        print()


def print_crossing(published_runs, quantiles, crossing, above_bar, above_self, class_order):
    """Where the deck's "every class above 80 %" sentence is arithmetically true."""
    print("terminal condition · margin of coverage over the quantile itself")
    print(f"{'q':>6} " + "  ".join(f"{c:>16}" for c in class_order))
    for quantile in quantiles[::-1]:
        row = published_runs[quantile]["conditions"]["asymmetry_5d"]["classes"]
        print(f"{quantile:>6.2f} " + "  ".join(
            f"{100 * row[c]['real_within_radius_fraction']:6.2f}% "
            f"({100 * (row[c]['real_within_radius_fraction'] - quantile):+5.1f})"
            for c in class_order))
    below = round(above_bar - 0.01, 2)
    above = round(above_self + 0.01, 2)
    print(f"\nevery class stays above 80 % for q >= {above_bar:.2f}; at q = {below:.2f} "
          f"the 4 µm class falls to {100 * crossing[below]['4um']:.1f} %")
    print(f"every class stays above its own quantile for q <= {above_self:.2f}; at q = "
          f"{above:.2f} the 10 µm class falls "
          f"{100 * (above - crossing[above]['10um']):.1f} points short of the bar")
    print(f"the sentence therefore holds only for {above_bar:.2f} <= q <= {above_self:.2f}")


def print_reference_density(density, class_order):
    """The two populations the radius and the queries are drawn from."""
    print(f"{'class':>6} {'synthetic cloud':>16} {'reference draw':>15} "
          f"{'real events':>12} {'density ratio':>14}")
    for class_name in class_order:
        row = density[class_name]
        print(f"{class_name:>6} {row['synthetic_population']:>16d} "
              f"{row['reference_draw']:>15d} {row['real']:>12d} "
              f"{row['density_ratio']:>13.1f}×")


def print_dimension_quantiles(published_sweep, recomputed, class_order):
    """The published q95 sweep beside the same sweep on the chain's q80."""
    print(f"{'d':>3} " + "  ".join(f"{c + ' q95':>12}" for c in class_order)
          + "  " + "  ".join(f"{c + ' q80':>12}" for c in class_order) + "     gap (pp)")
    for entry in published_sweep:
        dimensions = entry["dimensions"]
        print(f"{dimensions:>3} "
              + "  ".join(f"{100 * entry['coverage'][c]:11.1f}%" for c in class_order)
              + "  " + "  ".join(
                  f"{100 * recomputed[dimensions][c]:11.1f}%" for c in class_order)
              + "   " + " ".join(
                  f"{100 * (entry['coverage'][c] - recomputed[dimensions][c]):5.1f}"
                  for c in class_order))


def quantile_basis_evidence(*, reproduces, quantile_sweep, quantile_grid, conditions,
                            class_order, violations, sentence_holds, density, control,
                            angles, variance, neighbour, basis_coverage, sweep_at_q80,
                            extended, contrast_curve, contrast_grid, contrast_measured,
                            published_extrapolation, exponent, seeds, datasets, inputs):
    """Package the alignment measurements as a manifested run."""
    metrics = {
        "schema_version": 1,
        "analysis": "z8-coverage-quantile-and-basis-alignment",
        "reproduces": reproduces,
        "quantile_sweep": {
            f"{quantile:.2f}": {
                label: {c: quantile_sweep[quantile][label][c] for c in class_order}
                for label in conditions
            }
            for quantile in quantile_grid
        },
        "condition_ordering_violations": len(violations),
        "self_coverage_sentence_holds_between": list(sentence_holds),
        "reference_density": density,
        "self_coverage_control_q80_asymmetry_5d": control,
        "basis_divergence": {
            "principal_angles_degrees": np.sort(angles).tolist(),
            "mean_squared_cosine": float(np.mean(np.cos(np.radians(angles)) ** 2)),
            "explained_variance": {
                name: {"sixteen": sixteen, "pc1_pc2": plane}
                for name, (sixteen, plane) in variance.items()
            },
            "nearest_neighbour": {
                c: {k: v for k, v in neighbour[c].items() if not isinstance(v, np.ndarray)}
                for c in class_order
            },
            "coverage_q80_per_basis": basis_coverage,
        },
        "dimension_sweep_at_q80": {str(d): row for d, row in sweep_at_q80.items()},
        "dimension_sweep_extended": {str(d): row for d, row in extended.items()},
        "distance_contrast": {
            "per_dimension": {str(d): contrast_curve[d] for d in contrast_grid},
            "class_mean_at_101": contrast_measured[-1],
            "published_sqrt_d_extrapolation": published_extrapolation,
            "measured_decay_exponent_16_to_101": float(exponent),
        },
    }
    provenance = {
        "datasets": datasets,
        "inputs": inputs,
        "parameters": {
            "quantile_grid": list(quantile_grid),
            "chain_seed": seeds["chain"],
            "intro_seed": seeds["intro"],
            "dimension_sweep_seed": seeds["sweep"],
            "window": 1024,
            "synthetic_core_slice": [1536, 2560],
            "components": {"chain": 16, "intro": 16, "extended": 101},
            "real_population": "development train+val, sealed test never read",
        },
        "metric_definitions": {
            "coverage": (
                "fraction of real events whose nearest synthetic neighbour in the full "
                "synthetic class population lies within that condition's synthetic "
                "self-nearest-neighbour quantile radius, the radius being taken over a "
                "reference draw of one synthetic event per real event"),
            "self_coverage_control": (
                "the same radius applied to synthetic events held out of the reference "
                "draw, each scored against its nearest other synthetic event of the same "
                "class in the full population, and to real events scored against the "
                "thinned reference draw instead of the full cloud"),
            "principal_angles": (
                "angles between the 16-dimensional descriptor subspaces the two "
                "projections read, standardisation included"),
            "same_nearest_fraction": (
                "fraction of real events whose nearest synthetic event is the same event "
                "in both bases"),
            "distance_contrast": (
                "std/mean of pairwise synthetic distances at d components, the published "
                "definition, extended to the full 101-component rotation"),
        },
    }
    return metrics, provenance


def draw_quantile_curve(sweep, quantiles, conditions, class_order, class_colour,
                        published_quantiles, *, axes=None):
    """Coverage against the radius quantile, one panel per class.

    The diagonal is the load-bearing annotation: at quantile q, exactly q of the
    synthetic reference events lie within the radius of another one, so a point
    above the diagonal is what the deck reads as "real events are covered better
    than the synthetic cloud covers itself".
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    styles = {conditions[0]: (":", "o"), conditions[1]: ("--", "s"),
              conditions[2]: ("-", "D")}
    for axis, class_name in zip(axes, class_order):
        axis.plot(quantiles, quantiles, color="#94a3b8", lw=1.0, ls="-",
                  label="synthetic self-coverage")
        for condition in conditions:
            line, marker = styles[condition]
            axis.plot(
                quantiles,
                [sweep[q][condition][class_name]["coverage"] for q in quantiles],
                ls=line, marker=marker, ms=3.5, lw=1.6,
                color=class_colour[class_name],
                alpha=1.0 if condition == conditions[-1] else 0.55,
                label=condition.replace("_", " "),
            )
        for q in published_quantiles:
            axis.plot([q], [sweep[q][conditions[-1]][class_name]["coverage"]],
                      marker="o", ms=9, mfc="none", mec="#0f172a", mew=1.1)
        axis.axvline(0.80, color="#0f172a", lw=0.9, ls=(0, (2, 3)))
        axis.set(title=class_name, xlabel="radius quantile")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("real events within radius")
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    axes[2].text(0.80, 0.06, " published q80", fontsize=8, color="#0f172a")
    return axes


def draw_gain_amplification(published, quantiles, class_order, class_colour, *, axes=None):
    """How much of the chain's headline gain the quantile alone decides."""
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(10.5, 3.3), sharex=True)
    steps = ("white_noise_4d -> real_noise_4d", "real_noise_4d -> asymmetry_5d")
    titles = ("step 1 · white noise → measured noise", "step 2 · symmetric → asymmetric")
    for axis, step, title in zip(axes, steps, titles):
        for class_name in class_order:
            axis.plot(quantiles,
                      [published[q]["gains_percentage_points"][step][class_name]
                       for q in quantiles],
                      marker="o", color=class_colour[class_name], label=class_name)
        axis.set(title=title, xlabel="radius quantile",
                 ylabel="coverage gain (percentage points)")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, ncol=3)
    return axes


def draw_density_control(control, class_order, class_colour, *, ax=None):
    """The three coverage numbers the same radius produces at q80."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9.2, 3.4))
    labels = ("real vs the full synthetic cloud",
              "real vs a cloud thinned to the real count",
              "synthetic vs its own full cloud (leave-one-out)")
    keys = ("real_full", "real_matched", "synthetic_leave_one_out")
    width = 0.26
    positions = np.arange(len(class_order))
    hatches = ("", "//", "..")
    alphas = (1.0, 0.62, 0.32)
    for offset, (key, hatch, alpha) in enumerate(zip(keys, hatches, alphas)):
        ax.bar(positions + (offset - 1) * width,
               [100 * control[c][key] for c in class_order],
               width, hatch=hatch, edgecolor="white",
               color=[class_colour[c] for c in class_order], alpha=alpha)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor="#475569", edgecolor="white",
                             hatch=hatch, alpha=alpha, label=label)
               for label, hatch, alpha in zip(labels, hatches, alphas)]
    ax.axhline(80.0, color="#dc2626", lw=1.2, ls="--")
    ax.text(0.995, 0.755, "the 80 % bar", color="#dc2626", fontsize=8.5,
            ha="right", transform=ax.transAxes)
    ax.set(xticks=positions, xticklabels=list(class_order), ylim=(0, 132),
           ylabel="within the q80 radius (%)", yticks=[0, 20, 40, 60, 80, 100],
           title="One radius, three populations")
    ax.title.set_fontsize(10)
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper left", ncol=1)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def draw_basis_divergence(angles, neighbour, class_order, class_colour, *, axes=None):
    """Principal angles between the two bases, and the distances they induce."""
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(11.5, 3.6))
    order = np.arange(1, len(angles) + 1)
    axes[0].bar(order, np.sort(angles), color="#334155", width=0.7)
    axes[0].axhline(90.0, color="#dc2626", lw=1.0, ls="--")
    axes[0].text(1, 84, "orthogonal", color="#dc2626", fontsize=8)
    axes[0].set(xlabel="principal direction (sorted)", ylabel="angle (degrees)",
                ylim=(0, 100),
                title="Fifteen shared directions and one the bases disagree about")
    axes[0].title.set_fontsize(10)
    for class_name in class_order:
        pair = neighbour[class_name]
        axes[1].scatter(pair["distance_chain"], pair["distance_intro"], s=6,
                        alpha=0.35, color=class_colour[class_name],
                        label=f"{class_name} · same neighbour "
                              f"{100 * pair['same_nearest_fraction']:.0f} %")
    limit = max(max(neighbour[c]["distance_chain"]) for c in class_order)
    axes[1].plot([0, limit], [0, limit], color="#0f172a", lw=0.9)
    axes[1].set(xlabel="distance to nearest synthetic · chain basis",
                ylabel="· intro basis",
                title="The two bases agree on how far, not on which one")
    axes[1].title.set_fontsize(10)
    axes[1].legend(frameon=False, fontsize=8, markerscale=2)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    return axes


def draw_dimension_quantiles(published, recomputed, extended, class_order,
                             class_colour, *, axes=None):
    """The published sweep at q95, the same sweep at q80, and past sixteen."""
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(12.4, 3.6))
    dimensions = [entry["dimensions"] for entry in published["sweep"]]
    for class_name in class_order:
        axes[0].plot(dimensions,
                     [100 * entry["coverage"][class_name] for entry in published["sweep"]],
                     marker="o", ls="--", lw=1.4, alpha=0.55,
                     color=class_colour[class_name])
        axes[0].plot(dimensions, [100 * recomputed[d][class_name] for d in dimensions],
                     marker="D", lw=1.8, color=class_colour[class_name])
    handles = [plt.Line2D([], [], color=class_colour[c], lw=2.0, label=c)
               for c in class_order]
    handles += [plt.Line2D([], [], color="#64748b", lw=1.4, ls="--", label="q95 · published"),
                plt.Line2D([], [], color="#64748b", lw=1.8, label="q80 · recomputed")]
    axes[0].axvspan(12, 16, color="#e2e8f0", zorder=0)
    axes[0].annotate("published plateau band", xy=(14, 1.0),
                     xycoords=("data", "axes fraction"), xytext=(0, 4),
                     textcoords="offset points", ha="center", fontsize=8, color="#64748b")
    axes[0].set(xlabel="retained PCA dimensions", ylabel="coverage (%)",
                xticks=dimensions)
    axes[0].set_title("The sweep still reports q95 while the chain reports q80",
                      fontsize=10, pad=16)
    axes[0].legend(handles=handles, frameon=False, fontsize=7, ncol=1, loc="lower left")

    grid = sorted(extended)
    auc_axis = axes[1].twinx()
    for class_name in class_order:
        axes[1].plot(grid, [100 * extended[d]["coverage"][class_name] for d in grid],
                     marker="D", lw=1.8, color=class_colour[class_name])
        auc_axis.plot(grid, [extended[d]["auc"][class_name] for d in grid],
                      marker="s", ms=3.5, ls=(0, (4, 3)), lw=1.2,
                      color=class_colour[class_name])
    axes[1].axvline(16, color="#0f172a", lw=1.0, ls=(0, (2, 3)))
    axes[1].annotate("the frozen choice", xy=(16, 1.0),
                     xycoords=("data", "axes fraction"), xytext=(0, 4),
                     textcoords="offset points", ha="center", fontsize=8, color="#0f172a")
    axes[1].set(xlabel="retained PCA dimensions", ylabel="coverage % at q80 (solid)",
                xticks=grid)
    axes[1].set_title("Past sixteen, both quantities resume moving", fontsize=10, pad=16)
    auc_axis.set_ylabel("domain AUC (dashed)")
    auc_axis.spines[["top"]].set_visible(False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    return axes


def draw_contrast(grid, measured, predicted, published_extrapolation, *, ax=None):
    """Measured distance contrast against the published square-root prediction."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.6, 3.4))
    ax.plot(grid, measured, marker="o", color="#0f172a", label="measured")
    ax.plot(grid, predicted, ls="--", color="#dc2626",
            label="published √d trend from d = 16")
    ax.scatter([101], [published_extrapolation], marker="*", s=160, color="#dc2626",
               zorder=3)
    ax.annotate(f"appendix quotes {published_extrapolation:.2f}",
                xy=(101, published_extrapolation), xytext=(17, 0.075),
                fontsize=8.5, color="#dc2626",
                arrowprops={"arrowstyle": "-", "color": "#dc2626", "lw": 0.8})
    ax.scatter([101], [measured[-1]], marker="o", s=70, color="#0f172a", zorder=3)
    ax.annotate(f"measured {measured[-1]:.2f}", xy=(101, measured[-1]),
                xytext=(30, 0.30), fontsize=8.5, color="#0f172a",
                arrowprops={"arrowstyle": "-", "color": "#0f172a", "lw": 0.8})
    ax.set_xscale("log", base=2)
    ax.set_xticks(grid, [str(d) for d in grid], fontsize=7.5)
    ax.set(xlabel="retained PCA dimensions", ylabel="distance contrast · std / mean",
           ylim=(0.0, 0.68),
           title="The concentration the appendix predicts does not arrive")
    ax.title.set_fontsize(10)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


# %% [markdown]
# ## Masked reconstruction · training a model with no labels
#
# Everything up to here has been about the *simulator*: whether its cloud covers
# the real one, whether a regenerated event can find its parent. This section is
# about the *model* the simulator exists to train.
#
# **The problem.** There are roughly two thousand annotated real events. That is
# far too few to train a representation, and the annotation is the very thing we
# would like not to depend on. **Self-supervised learning** (SSL) removes the
# label from the problem: part of the input is hidden, the model is asked to
# predict it, and the error on the hidden part is the loss. Nobody has to say
# what the signal *is*; the signal supervises itself. The deliverable at the end
# is the encoder, not the reconstruction — the decoder is thrown away.
#
# That leaves one design choice undecided, and it is the choice this section is
# about: not *whether* to mask, but **where**. Two policies were trained under
# identical conditions:
#
# - **P25** hides a quarter of the trace in short patches drawn blind to the
#   signal;
# - **CYCLIC25** hides the same quarter, but schedules wide windows aimed at the
#   annotated event and balances them against background.
#
# The section reproduces that comparison from the shipped code and the
# manifested runs, and then measures a defect in how CYCLIC25's "event" was
# defined — one that no run owns and that the deck does not mention.

# %%
from internship_workspace.z8_coverage import read_rows  # noqa: E402
from p3_ssl.bead_ssl import make_model  # noqa: E402
from p3_ssl.bead_ssl_v2 import load_bead_ssl_v2_config  # noqa: E402
from p3_ssl.masking import (  # noqa: E402
    PatchSpec,
    build_patch_aligned_isolated_masks,
    mask_spans,
)

SSL_ROOT = workspace.root / "unsupervised-learning-flow-cytometry"
SSL_CONFIG = SSL_ROOT / "configs/bead_ssl_z8_v5_v2.yaml"
SSL_RUNS = workspace.artifacts_root / "unsupervised-learning-flow-cytometry" / "runs"
MATCHED_SEEDS = (42, 43, 44, 45, 46)
INPUT_LENGTH = 4096


def matched_run(policy, seed):
    """The ten runs the comparison rests on: two policies × five seeds."""
    return SSL_RUNS / f"bead-ssl-v2-{policy}-full-s{seed}-e30-matched-r1"


config = load_bead_ssl_v2_config(SSL_CONFIG)
model = make_model(config)
encoder_parameters = sum(p.numel() for p in model.encoder.parameters())
total_parameters = sum(p.numel() for p in model.parameters())

assert total_parameters == 340_528, total_parameters
assert encoder_parameters == 335_520, encoder_parameters
print(f"MomentLikeReconstructor · {total_parameters:,} parameters "
      f"({encoder_parameters:,} in the encoder, which is the deliverable)")
print(f"  {model.n_tokens} tokens of {config['model']['patch_size']} samples, "
      f"stride {config['model']['patch_stride']} — the 4,096-sample crop, exactly tiled")
print(f"  {config['model']['n_layers']}-layer Transformer, d_model "
      f"{config['model']['d_model']}, {config['model']['n_heads']} heads, "
      f"FFN {config['model']['dim_feedforward']}, dropout {config['model']['dropout']}")
print(f"  mask encoding {config['model']['mask_encoding']}")

# %% [markdown]
# ### The objective, stated precisely
#
# A quarter of the 4,096 samples is hidden — **1,024 points** — and the loss is
# read only there. The visible samples cost nothing whether they are reproduced
# or not, so the model cannot win by copying its input.
#
# The loss *family* in the config is a composite: a signal term, a derivative
# term and an energy term, the latter two robustified with a pseudo-Huber of
# δ = 1 (a loss that is quadratic near zero and linear far from it, so a few
# large residuals cannot dominate). The **selected cell is B0**, and B0 sets both
# robust weights to zero.

# %%
for cell, weights in config["loss"]["cells"].items():
    marker = "  <- selected" if cell == config["loss"]["selected_cell"] else ""
    print(f"  {cell}: signal {weights['lambda_signal']}, derivative "
          f"{weights['lambda_derivative']}, energy {weights['lambda_energy']}{marker}")
print(f"\nhuber_delta {config['loss']['huber_delta']} applies to the derivative and "
      "energy terms only; under B0 both carry weight 0, so the trained objective "
      "is masked MSE on the hidden samples")

# %% [markdown]
# So the objective that actually trained these models is plain masked mean
# squared error, and saying "Huber loss" would overstate it. Worth stating
# explicitly, because the config file reads as though a robust loss were in use.
#
# `sample_visibility_v1` is the second piece that has to be right. A hidden
# sample is replaced by zero — but a genuine zero crossing is also zero, and the
# model must not confuse the two. So the mask is handed to the network as an
# explicit channel: a learned embedding of the per-sample visibility pattern is
# added to every token. This also lets a masking window sit off the token grid,
# which matters below, since CYCLIC25's windows stride 8 while tokens stride 16.

# %%
p25_examples = np.load(
    matched_run("p25", 42) / "real_reconstruction_examples.npz", allow_pickle=True
)
cyclic_examples = np.load(
    matched_run("cyclic25", 42) / "real_reconstruction_examples.npz", allow_pickle=True
)
example_index = 0
example_signal = p25_examples["signal"][example_index]
example_mask = p25_examples["mask"][example_index].astype(bool)
print(f"real validation event {p25_examples['sample_id'][example_index]} · "
      f"{int(example_mask.sum())} of {example_mask.size} samples hidden "
      f"({100 * example_mask.mean():.0f} %)")
draw_model_view(example_signal, example_mask)

# %% [markdown]
# ### Why the placement of the hole is a scientific choice
#
# **The problem.** If a trace were uniformly informative, where you hide would
# not matter. It is not. The simulator knows exactly where each event sits,
# because it put it there: the event is centred at `t0_fraction × (N − 1)` and
# extends ±3τ, stretched by the asymmetry `a` on each side.
#
# **The process.** That support is rebuilt below with the training dataset's own
# formula, over all 47,980 v5 events, and measured against the crop it lives in.

# %%
V5_KEY = ("particles2snr-fbase-z8-cholesky-physicalcorr-effective-snr-"
          "synthetic-events@v5")
v5_root = dataset_root(V5_KEY)
v5_rows = read_rows(v5_root / "events.csv")
if any(row["noise_source_split"] == "test" for row in v5_rows):
    raise PermissionError("sealed test rows are forbidden")

v5_tau_ms = np.asarray([float(row["tau_ms"]) for row in v5_rows])
v5_t0 = np.asarray([float(row["t0_fraction"]) for row in v5_rows])
v5_asymmetry = np.asarray([float(row["waveform_asymmetry"]) for row in v5_rows])
v5_class = np.asarray([row["class_name"] for row in v5_rows])
DECLARED_HZ = float(config["data"]["sampling_frequency_hz"])


def event_bounds(sampling_hz, *, length=INPUT_LENGTH):
    """The support `Z8AsymmetricSyntheticDataset` builds, for every v5 event."""
    centre = v5_t0 * (length - 1)
    tau_samples = v5_tau_ms * 1.0e-3 * sampling_hz
    start = np.maximum(0.0, np.floor(centre - 3.0 * tau_samples * np.exp(-v5_asymmetry)))
    end = np.minimum(length, np.ceil(centre + 3.0 * tau_samples * np.exp(v5_asymmetry)) + 1.0)
    return start.astype(int), end.astype(int)


declared_start, declared_end = event_bounds(DECLARED_HZ)
declared_fraction = (declared_end - declared_start) / INPUT_LENGTH
print(f"support the trainer built, over {len(v5_rows):,} v5 events "
      f"(sampling_frequency_hz = {DECLARED_HZ:,.0f})")
print(f"  median {100 * np.median(declared_fraction):.1f} % of the crop "
      f"({np.median(declared_end - declared_start):.0f} samples), "
      f"p95 {100 * np.percentile(declared_fraction, 95):.1f} %, "
      f"{100 * np.mean(declared_fraction < 0.5):.1f} % of events under half the crop")

assert round(100 * float(np.median(declared_fraction)), 1) == 22.1
assert round(100 * float(np.percentile(declared_fraction, 95)), 1) == 38.9
assert round(100 * float(np.mean(declared_fraction < 0.5)), 1) == 99.1
print("\nreproduces the deck's SSL-bridge page exactly (22.1 % / 38.9 % / 99.1 %)")

# %% [markdown]
# **The conclusion.** Roughly four fifths of every training example is
# background. A mask drawn uniformly spends four fifths of its budget there.
# That is the argument for aiming — and the hypothesis the comparison below is
# meant to test.
#
# **What it leaves open. Hold on to the 22.1 %.** It is measured correctly here,
# in the sense that it is exactly what the training code computed. The last part
# of this section shows it is not what the data says.
#
# ### The two policies, on one trace, at one budget
#
# **The problem.** Two aiming strategies cannot be compared on their loss values
# until it is clear what each one actually hides. **The process:** both masks are
# rebuilt from the shipped builders in `p3_ssl.masking`, with the production
# `cyclic25` parameters and the runs' seed. Nothing about the geometry is drawn
# by hand.
#
# - **P25** — `build_patch_aligned_isolated_masks` picks whole 16-sample tokens
#   at random and forbids two selected tokens from touching. With 256 tokens and
#   a 25 % ratio that is 64 selections, and the isolation rule makes the geometry
#   deterministic: 64 runs of exactly 16 samples, whatever the seed.
# - **CYCLIC25** — `build_balanced_event_mask_cycle` works on a finer grid
#   (16-sample windows every 8 samples), takes 32 windows intersecting the event
#   and 32 background windows per pass, and requires that windows selected within
#   one pass be mutually disjoint. Adjacent event windows merge into long
#   contiguous runs.
#
# The deck draws this on a real validation trace with an *idealised* 512-sample
# support rather than that event's own annotation, so the two policies can be
# compared on identical geometry. Reproduced exactly here, including that choice,
# so the published numbers are checkable.

# %%
DECK_EVENT = (1792, 2304)  # the deck's idealised 512-sample support
DECK_SEED = 42
CYCLIC = config["masking"]["cyclic25"]
cyclic_spec = PatchSpec(
    INPUT_LENGTH, int(CYCLIC["candidate_size"]), int(CYCLIC["candidate_stride"])
)

p25_mask = build_patch_aligned_isolated_masks(
    example_signal.astype(np.float64),
    PatchSpec(INPUT_LENGTH, config["model"]["patch_size"], config["model"]["patch_stride"]),
    np.random.default_rng(DECK_SEED),
    mask_ratio=float(config["masking"]["mask_ratio"]),
)["target_time_mask"]
deck_cycle = cyclic25_cycle(DECK_EVENT, index=0, spec=cyclic_spec, settings=CYCLIC,
                            length=INPUT_LENGTH, seed=DECK_SEED)
cyclic_mask = cycle_group(deck_cycle, pass_index=0)

measured_geometry = {"P25": mask_geometry(p25_mask), "CYCLIC25": mask_geometry(cyclic_mask)}
print_mask_geometry(measured_geometry, INPUT_LENGTH)
assert measured_geometry == {
    "P25": {"hidden": 1024, "spans": 64, "longest": 16, "median": 16},
    "CYCLIC25": {"hidden": 1024, "spans": 27, "longest": 560, "median": 16},
}, measured_geometry
print("reproduces the deck's published mask geometry exactly")

# %%
draw_policies(example_signal, p25_mask, cyclic_mask, DECK_EVENT)

# %% [markdown]
# **The conclusion.** Same 1,024 points, two different problems. P25's longest
# hole is 16 samples with visible signal on both sides — that is interpolation.
# CYCLIC25's longest hole is 560 samples covering the whole event — that is
# reconstruction from context. **What follows from it:** their raw losses are not
# comparable, so a comparison that reads one loss against the other decides
# nothing. The comparison has to cross the masks, which is what the third part
# below does.
#
# ### CYCLIC25 is a schedule, not a mask
#
# **The problem.** One CYCLIC25 pass hides only part of the event, so a single
# pass cannot be what the policy promises. **The process:** the builder produces
# a *cycle* of passes in which every candidate window intersecting the event is
# hidden at least once, while each individual pass keeps the same 25 % budget and
# stays internally disjoint. Training walks the cycle, so an event is seen many
# times through different holes. Below is the actual cycle of the matched
# CYCLIC25 run for its deck sample, rebuilt with the run's own seed derivation
# (`seed + index × 7919`).

# %%
TRAINING_SAMPLE = "syn-2um-3643708b867724bd"
training_index = next(
    i for i, row in enumerate(v5_rows) if row["sample_id"] == TRAINING_SAMPLE
)
training_signal = np.asarray(
    np.load(v5_root / "signals_raw_4096.npy", mmap_mode="r")[training_index],
    dtype=np.float64,
)
training_signal = (training_signal - training_signal.mean()) / training_signal.std()
training_support = (declared_start[training_index], declared_end[training_index])
training_cycle = cyclic25_cycle(training_support, index=training_index, spec=cyclic_spec,
                                settings=CYCLIC, length=INPUT_LENGTH, seed=DECK_SEED)

print(f"{TRAINING_SAMPLE} · declared support "
      f"[{training_support[0]}, {training_support[1]}) = "
      f"{training_support[1] - training_support[0]} samples")
print(f"  {int(training_cycle['pass_count'])} passes, "
      f"{training_cycle['event_window_indices'].size} candidate event windows, "
      f"{[int(m.sum()) for m in cycle_group(training_cycle)]} points hidden per pass")
print(f"  cumulative event-window coverage: "
      + ", ".join(f"{100 * value:.0f} %"
                  for value in training_cycle["cumulative_event_window_coverage"]))

# %%
draw_cycle(training_signal, training_cycle, training_support)

# %% [markdown]
# **The conclusion.** The completeness guarantee is real and it is what
# distinguishes CYCLIC25 from "a big hole in the middle": no event sample is
# permanently invisible, and the 25 % budget is never exceeded. That guarantee is
# the policy's whole justification — which is why the exploratory part at the end
# of this section, where the guarantee is measured against the event the
# generator actually drew, matters as much as the comparison itself.
#
# ### Appendix · how a fixed budget packs a variable support
#
# **The problem.** The guarantee is not free. Every event has a different width,
# so the number of candidate windows changes from event to event, while the
# budget is fixed at 32 event windows per pass. **The process:** the builder
# resolves this with two devices — it splits the half-overlapping catalogue into
# **lanes** (a window's index modulo `patch_size / stride` = 2), so windows
# within a lane can never overlap, and it pads the last group of a lane by
# repeating windows already scheduled. The deck states this for the median event,
# a 906-sample support; that is reproduced here from the shipped builder.

# %%
PACKING_WIDTH = int(np.median(declared_end - declared_start))
packing_support = ((INPUT_LENGTH - PACKING_WIDTH) // 2,
                   (INPUT_LENGTH - PACKING_WIDTH) // 2 + PACKING_WIDTH)
packing = cyclic25_cycle(packing_support, index=0, spec=cyclic_spec, settings=CYCLIC,
                         length=INPUT_LENGTH, seed=DECK_SEED)
packing_windows = packing["event_window_indices"]
packing_selection = np.asarray(packing["pass_event_window_indices"]).reshape(-1)
lane_sizes = sorted(int((packing_windows % 2 == lane).sum()) for lane in (0, 1))
unique_windows, selection_counts = np.unique(packing_selection, return_counts=True)

print(f"support {PACKING_WIDTH} samples (the corpus median) -> "
      f"{packing_windows.size} candidate event windows")
print(f"  two internally disjoint lanes of {lane_sizes[0]} and {lane_sizes[1]}")
print(f"  {int(packing['pass_count'])} passes x "
      f"{int(CYCLIC['event_windows_per_pass'])} = {packing_selection.size} selections: "
      f"{unique_windows.size} unique, {packing_selection.size - unique_windows.size} repeated")
assert PACKING_WIDTH == 906
assert packing_windows.size == 115 and lane_sizes == [57, 58]
assert packing_selection.size == 128 and unique_windows.size == 115
print("\nreproduces the deck's packing appendix exactly "
      "(906 -> 115 windows, lanes 57/58, 128 selections, 115 unique + 13 repeated)")

# %%
draw_packing(cyclic_spec.spans[packing_windows], packing_windows % 2, selection_counts,
             packing_support)

# %% [markdown]
# **The conclusion.** The budget is honoured exactly and the completeness comes
# from repetition, not from overspending: thirteen of the 128 selections are
# windows already scheduled. **What it leaves open** is the interaction the last
# part of this section measures — the packing is computed over the *declared*
# support, so a wrong support does not break the packing, it silently aims it
# somewhere else.
#
# ### The comparison
#
# Ten runs: two policies × five seeds, 30 epochs each, on 39,108 synthetic
# training traces with 8,872 synthetic and 444 real validation traces. The
# comparison is read in three steps, and the order matters: first check the
# monitoring is honest, then look at what the models produce, then measure which
# one survives the other's regime.
#
# #### 1 · Is the train/validation comparison meaningful at all?
#
# **The problem.** The training loss is computed under dropout with freshly drawn
# masks, so it is not comparable to a validation number, and a gap between them
# would mean nothing. **The process:** the matched-monitoring protocol fixes this
# — 2,048 class-proportional events per split, selected once by a hash of the
# sample id, evaluated under `model.eval` with the run's own policy. Train and
# validation then differ only by the split.

# %%
monitor = matched_monitor_curves(matched_run, ("P25", "CYCLIC25"), MATCHED_SEEDS)
reference_monitor = published("bead-ssl-v2-matched-monitor-analysis-r1")["final_matched_monitor"]
monitor_deviation = max(
    abs(monitor[policy]["gap_percent"]
        - reference_monitor[policy]["validation_vs_train_eval_gap_percent"])
    for policy in ("P25", "CYCLIC25")
)
assert monitor_deviation < 1.0e-9, f"reproduction drifted by {monitor_deviation:.3e}"
print(f"\nreproduces bead-ssl-v2-matched-monitor-analysis-r1 exactly "
      f"(max deviation {monitor_deviation:.1e} percentage points)")

# %%
draw_learning_curves(monitor)

# %% [markdown]
# **The conclusion.** Both policies generalise: the validation monitor sits
# **+1.49 %** (P25) and **+1.22 %** (CYCLIC25) above the train monitor at epoch
# 30, on identical protocols. Neither is memorising, and the loss curves can be
# believed. What the plot also shows is that the two levels are two orders of
# magnitude apart — P25 lands near 1.6 × 10⁻³, CYCLIC25 near 1.9 × 10⁻¹.
#
# **What it leaves open.** That gap is *not* evidence that P25 is the better
# model; it is evidence that P25 was asked an easier question, exactly as the
# geometry predicted. Comparing the two levels decides nothing, which is why the
# next two steps look at what the models produce and how they behave off their
# own regime.
#
# #### 2 · What the two models produce, on real events
#
# **Limit, named.** Every matched run's `run.json` lists `checkpoints/best.pt`
# and `checkpoints/latest.pt` among its outputs, but the checkpoint directory was
# never synced back from the cluster. The weights are not in this workspace, so
# nothing below is re-inferred: the reconstructions come from the
# `*_reconstruction_examples.npz` each run wrote at the end of training. Any
# claim requiring a forward pass — a new event, a new mask, a probe on the frozen
# encoder — is out of reach until those checkpoints are recovered.
#
# A second limit follows from the same files. Both runs exported their examples
# under the **P25** evaluation policy (`evaluate_reconstruction` defaults to it),
# so the stored masks are 64 isolated 16-sample holes in every case. That is
# precisely the deck's "identical hidden samples" comparison and it is reproduced
# faithfully — but it means the notebook cannot *show* a 560-sample CYCLIC25 gap
# being reconstructed. For that regime only the aggregate numbers below exist.

# %%
if not np.array_equal(example_mask, cyclic_examples["mask"][example_index].astype(bool)):
    raise ValueError("stored masks differ; the comparison would not be matched")
widest_hole = max(mask_spans(example_mask), key=lambda span: span[1] - span[0])
centre = (widest_hole[0] + widest_hole[1]) // 2
zoom = (max(0, centre - 320), min(example_signal.size, centre + 320))

outputs = []
for label, source, accent in (("P25", p25_examples, "#e2483f"),
                              ("CYCLIC25", cyclic_examples, "#00a3c7")):
    prediction = source["model"][example_index]
    error = float(np.mean((prediction[example_mask] - example_signal[example_mask]) ** 2))
    outputs.append((label, prediction, accent, error))
    print(f"{label:9s} masked MSE on {p25_examples['sample_id'][example_index]}: {error:.6f}")
draw_reconstruction(example_signal, outputs, example_mask, zoom)

# %% [markdown]
# **The conclusion.** On P25's own regime, P25 tracks the hidden samples closely
# and CYCLIC25 is visibly coarser — an order of magnitude worse locally. If the
# comparison stopped here it would retain P25. **It does not stop here**, because
# one regime cannot rank two policies that were trained on different problems.
#
# #### 3 · The cross-mask evaluation
#
# **The problem.** Each model has only ever been scored on the regime it trained
# on. **The process:** evaluate both trained models on **both** regimes, on the
# 444 real validation events, with `predicting zero` as the reference — a model
# that has learned nothing useful about a regime cannot beat the constant zero.
# Five seeds per cell, means recomputed here from the evaluation run's rows.

# %%
cross = published("bead-ssl-v2-matched-cross-mask-evaluation-r1")
cells, zero_reference = cross_mask_cells(cross["rows"], ("P25", "CYCLIC25"))
for key, want in {("P25", "P25"): 0.0015, ("P25", "CYCLIC25"): 1.1024,
                  ("CYCLIC25", "P25"): 0.0063, ("CYCLIC25", "CYCLIC25"): 0.0361}.items():
    assert round(cells[key]["mean"], 4) == want, (key, cells[key]["mean"])
assert round(zero_reference["P25"], 4) == 1.2765
assert round(zero_reference["CYCLIC25"], 4) == 1.1209
print("\nreproduces bead-ssl-v2-matched-cross-mask-evaluation-r1 exactly")

# %%
draw_cross_mask(cells, zero_reference)

# %% [markdown]
# **The conclusion.** The verdict is in the second column. **P25 on CYCLIC25
# masks scores 1.102 against 1.121 for predicting zero** — a 1.7 % improvement
# over outputting nothing at all. A model that hid only isolated 16-sample holes
# for 30 epochs has learned nothing that transfers to a missing event. CYCLIC25
# on P25 masks scores 0.0063 against a 1.277 zero baseline: it pays about four
# times P25's local error but stays a functioning model on a regime it never
# trained on.
#
# So the retained policy is CYCLIC25, and the honest statement of the result is
# asymmetric: CYCLIC25 is retained for robustness across missing positions, while
# P25 keeps a genuinely lower local error on its own easier task. Both halves
# belong in the claim.
#
# **What is not claimed.** All of this is *reconstruction* behaviour on
# development data. Nothing here says which encoder is the better representation
# — that requires frozen probes or fine-tuning against a from-scratch baseline,
# which needs the checkpoints, and no sealed test split was touched anywhere in
# this chain.

# %% [markdown]
# ---
# ## Exploratory · the event CYCLIC25 was aimed at is half the real one
#
# *This sub-section is exploratory and it produces a measurement no published run
# owns. It is not a correction of the published comparison; it is a measured
# statement about what that comparison actually trained.*
#
# **The problem.** The masking code never sees a waveform's timing. It converts
# the generator's `tau_ms` into samples using a number from the config:
#
# ```python
# tau_samples = float(row["tau_ms"]) * 1.0e-3 * self.sampling_frequency_hz
# ```
#
# `bead_ssl_p25_v1.yaml` declares `sampling_frequency_hz: 1000000`, and
# `bead_ssl_z8_v5_v2.yaml` extends it without overriding that key, so the merged
# config the runs used carries 1 MHz. The v5 generator writes
# `sampling_frequency_hz: 2_000_000.0` into its own provenance.
#
# **The process.** Two configuration files disagreeing is an argument, not
# evidence. The waveforms settle it: for a Gaussian envelope `exp(−½(t/τ)²)`, the
# full width at half maximum is `1.1774 · τ · (e^{−a} + e^{+a})` **seconds**, so
# measuring it in samples on the stored traces measures the sampling rate
# directly.

# %%
TRUE_HZ = SAMPLING_HZ
v5_signals = np.load(v5_root / "signals_raw_4096.npy", mmap_mode="r", allow_pickle=False)
achieved_snr = np.asarray([float(row["achieved_snr_db"]) for row in v5_rows])
probe = np.sort(np.argsort(-achieved_snr)[:300])
measured_fwhm = envelope_fwhm_samples(np.asarray(v5_signals[probe]), smoothing=41)
predicted_seconds = (
    1.1774 * v5_tau_ms[probe] * 1.0e-3
    * (np.exp(-v5_asymmetry[probe]) + np.exp(v5_asymmetry[probe]))
)
print(f"{len(probe)} highest-SNR v5 events (achieved SNR >= "
      f"{achieved_snr[probe].min():.1f} dB), envelope FWHM measured on the stored traces")
rate_ratio = {}
for label, sampling_hz in (("declared 1 MHz", DECLARED_HZ), ("generator 2 MHz", TRUE_HZ)):
    ratio = measured_fwhm / (predicted_seconds * sampling_hz)
    rate_ratio[sampling_hz] = float(np.median(ratio))
    print(f"  measured / predicted at {label:16s}: median {np.median(ratio):.3f}  "
          f"IQR {np.percentile(ratio, 25):.3f}–{np.percentile(ratio, 75):.3f}")
assert abs(rate_ratio[TRUE_HZ] - 1.0) < 0.02
assert abs(rate_ratio[DECLARED_HZ] - 2.0) < 0.04
print("\nthe data is sampled at 2 MHz, to within 0.2 % — the config value is wrong "
      "by exactly a factor of two")

# %% [markdown]
# ### What that does to the mask
#
# **The conclusion so far, and why it is not cosmetic:** this number is not
# metadata, it is the only thing converting the event's physical duration into
# array indices. At half the true rate the support is half as wide, centred on
# the same point, and therefore a **strict subset** of the real event.

# %%
true_start, true_end = event_bounds(TRUE_HZ)
true_fraction = (true_end - true_start) / INPUT_LENGTH
declared_is_subset = bool(np.all((declared_start >= true_start) & (declared_end <= true_end)))
captured = (declared_end - declared_start) / (true_end - true_start)
assert declared_is_subset
print_support_comparison(declared_fraction, true_fraction, captured, len(v5_rows))
assert round(100 * float(np.median(true_fraction)), 1) == 44.2

# %%
draw_support_rates(declared_fraction, true_fraction, v5_class, CLASS_ORDER, CLASS_COLOUR)

# %% [markdown]
# The deck's SSL bridge opens the whole masking chapter on "the annotated support
# is 22 % of the crop at the median, and 99.1 % of events sit under half the
# crop". Measured against the rate the data was generated at, the median event
# occupies **44.2 %** of the crop and **36.4 %** of events occupy more than half
# of it. The premise that motivates aiming is roughly half as strong as stated.
#
# ### What a trace actually got
#
# The clearest way to see it is on one event: the windows the cycle called
# *event* and the windows it called *background*, drawn over the support the data
# implies. The cycles are rebuilt with the runs' own seed derivation and the
# trainer's own bound expansion, so this is the mask that event received.

# %%
case = int(np.argsort(np.abs((declared_end - declared_start) - PACKING_WIDTH))[0])
case_signal = np.asarray(np.load(v5_root / "signals_raw_4096.npy",
                                 mmap_mode="r")[case], dtype=np.float64)
case_signal = (case_signal - case_signal.mean()) / case_signal.std()
case_true = np.zeros(INPUT_LENGTH, dtype=bool)
case_true[true_start[case] : true_end[case]] = True

cycles = {
    label: cyclic25_cycle(support, index=case, spec=cyclic_spec, settings=CYCLIC,
                          length=INPUT_LENGTH, seed=DECK_SEED)
    for label, support in (("as trained", (declared_start[case], declared_end[case])),
                           ("corrected", (true_start[case], true_end[case])))
}
for label, cycle in cycles.items():
    print_cycle_case(label, cycle, case_true)

draw_defect_case(
    case_signal,
    (declared_start[case], declared_end[case]),
    (true_start[case], true_end[case]),
    cycles["as trained"], cycles["corrected"],
)

# %% [markdown]
# The amber windows are the ones the cycle balanced *against* the event. Some of
# them sit on the event's own shoulders.
#
# That is the mechanism: `build_balanced_event_mask_cycle` takes its background
# candidates from `~intersects_event`, and `intersects_event` is derived from the
# halved support. Everything outside the inner half of an event was eligible to
# be sampled as background, and the completeness guarantee — "every event sample
# is hidden at least once" — was enforced on the inner half only.
#
# Run over a seeded sample of the corpus, with the runs' own seed derivation:

# %%
CYCLE_SAMPLE = 1500
CYCLE_SEED = 20260815
cycle_indices = np.sort(
    np.random.default_rng(CYCLE_SEED).choice(len(v5_rows), size=CYCLE_SAMPLE, replace=False)
)
started = time.time()
scan = scan_cycle_defect(
    cycle_indices, (declared_start, declared_end), (true_start, true_end),
    spec=cyclic_spec, settings=CYCLIC, length=INPUT_LENGTH, seed=DECK_SEED,
)
print_cycle_defect(scan, CYCLE_SAMPLE, time.time() - started)

# %%
draw_defect_summary(scan)

# %% [markdown]
# ### What this puts in doubt, and what it does not
#
# **What is established.** CYCLIC25 as trained aimed at the inner half of each
# simulated event. A median **28.6 %** of what it called *background* was in fact
# event (mean 30.7 %); its completeness guarantee — the thing that justifies the
# cycle — covered a median **51.4 %** of each true support; and a corrected cycle
# would be twice as long (median 4 passes to 8), with **0.4 %** of events
# admitting no cycle at all under the current 32-window budget. P25 is untouched:
# it passes `event_mask=None` and `event_biased_probability=0.0`, so it never
# reads the support.
#
# **What is not touched by this.** The two things the published comparison
# actually rests on:
#
# - The **budget** is unaffected. Both policies hide exactly 1,024 points per
#   pass whatever the support is, so "matched budget" remains true.
# - The **real-validation evaluation** is unaffected. `Z8RealValidationDataset`
#   builds its event mask from the annotated `start_sample` / `end_sample` of a
#   real event and never touches `sampling_frequency_hz`. So the cross-mask
#   numbers on real data — the 1.102-against-1.121 that settles the verdict —
#   were measured with *correct* event bounds.
#
# That last point cuts in a specific direction. CYCLIC25 was trained on
# half-width events and then evaluated on full-width ones: a distribution shift
# against it, not for it. It still beat P25 on that regime by a factor of thirty.
# The qualitative verdict — P25 does not generalise to missing events, CYCLIC25
# does — therefore survives the defect, and if anything is understated.
#
# **What genuinely stays unknown.** How much better a correctly-aimed CYCLIC25
# would be. Its matched masked MSE of 0.193 is measured on simulation validation,
# whose masks carry the same halved support, so that number is not a clean
# estimate of anything. Whether the encoder — the actual deliverable — is
# affected cannot be assessed at all from here: the checkpoints are not in this
# workspace, and even with them, a representation claim needs probes that were
# never run.
#
# **What would settle it.** Ten runs: two policies × five seeds, 30 epochs, on
# pfcalcul, with `sampling_frequency_hz: 2000000` in `bead_ssl_z8_v5_v2.yaml` —
# the fix is one line, since `_deep_merge` lets the child override the key. The
# 32-window budget needs raising in step, or the small fraction of wide events
# will fail to build a cycle. Re-running the same matched-monitor and cross-mask
# evaluation then makes the corrected and published comparisons directly
# commensurable. Until that exists, the deck's masking conclusion should be
# reported as it is defended here: robust in direction, unquantified in size.
#
# **A provenance note that belongs with this.** The deck's masking figures are
# resolved from `.cache/visual-evidence/ssl-v18-masking-figures-r4`, a cache path
# outside `artifacts/` with no `run.json`. Its `figure_metrics.json` holds the
# published geometry — and its `sources` block names
# `configs/bead_ssl_p25_v1.yaml`, the file carrying the wrong rate. The numbers
# happen to be reproducible, as this section shows, but nothing manifested
# guaranteed that. This notebook closes that gap for the geometry and the
# packing, and the run emitted below closes it for the defect.

# %%
metrics, provenance = mask_rate_evidence(
    declared_hz=DECLARED_HZ,
    true_hz=TRUE_HZ,
    probe={"events": len(probe), "minimum_snr_db": achieved_snr[probe].min()},
    rate_ratio=rate_ratio,
    declared=(declared_fraction, declared_end - declared_start),
    true=(true_fraction, true_end - true_start),
    captured=captured,
    subset=declared_is_subset,
    class_names=v5_class,
    class_order=CLASS_ORDER,
    scan=scan,
    sampled=CYCLE_SAMPLE,
    cyclic=CYCLIC,
    cycle_seed=CYCLE_SEED,
    datasets=dataset_provenance(),
    inputs={
        "events_csv_sha256": notebook_evidence.sha256_file(v5_root / "events.csv"),
        "config_sha256": notebook_evidence.sha256_file(SSL_CONFIG),
        "base_config_sha256": notebook_evidence.sha256_file(
            SSL_ROOT / "configs/bead_ssl_p25_v1.yaml"
        ),
    },
)
try:
    emitted = notebook_evidence.emit_run(
        workspace,
        section="masked-learning-mask-rate",
        metrics=metrics,
        provenance=provenance,
        claim_boundary=(
            "Measures the geometric consequence of the 1 MHz sampling frequency "
            "declared in bead_ssl_p25_v1.yaml against the 2 MHz the v5 waveforms "
            "are generated at: how wide the event support CYCLIC25 aimed at "
            "actually was, how much of its balanced background fell inside the "
            "real event, and how the cycle would change if corrected. It measures "
            "masks only. It does not retrain, does not re-evaluate any model, does "
            "not touch the published cross-mask or matched-monitor results, and "
            "says nothing about the quality of either learned encoder. No sealed "
            "test data is read."
        ),
    )
    print(f"emitted {emitted.name}")
except WorkspaceError as error:
    print(f"no evidence emitted ({error})")


# %% [markdown]
# ## Alignment · the descriptor window
#
# *Exploratory section. It tests a change before it is adopted, and its
# conclusion is a recommendation, not a shipped decision.*
#
# **The problem.** The morphology descriptor reads a fixed **1 024-sample
# window** centred on the event. Every distance in this notebook — coverage,
# twins, retrieval — is a distance between two such windows, so the window is
# not a detail of the method; it is the method's field of view. The audit
# earlier in the series showed that field of view is too narrow: **96.2 % of the
# real events are wider than the window that describes them.** The natural
# repair is 4 096 samples, already the raw input of the SSL model and of the
# classifier, and wide enough for 100 % of the new MAD corpus, whose widest
# event is exactly 4 000 samples.
#
# Nothing in the audit says the repair *works*. It says the window truncates;
# whether a wider one describes reality better is a separate measurement, and
# this section makes it. Two obstacles stand in the way — the descriptor used to
# change shape with its window, and a wider window buys shared silence — and
# both are settled below before the sweep is read.

# %%
from internship_workspace.chain_data import (  # noqa: E402
    Registry,
    coverage_chain,
    load_conditions,
    load_real_events,
    pooled_basis,
)
from internship_workspace.z8_domain_pca import (  # noqa: E402
    domain_metrics,
    morphology_features,
    spectrum_band_grid,
)

WINDOWS = (1024, 2048, 4096)
REFERENCE_WINDOW = 1024
QUANTILE = 0.80
SEED = 20260809
BAND_HZ = (7_000.0, 80_000.0)
ENVELOPE_BINS = 64

registry = Registry(workspace)

# %% [markdown]
# ### Step 1 · Making a window change a window change
#
# **The problem.** `morphology_features` keeps the FFT bins falling inside the
# 7–80 kHz band, and the number of such bins is set by the frequency resolution,
# which is set by the window length. Changing the window therefore used to
# change the descriptor's dimension — and with it what PCA(16) is a projection
# *of*. The envelope half had the same disease in a quieter form: the smoothing
# was applied in **samples**, before the envelope was averaged into 64 bins, so
# σ was half a bin at 1 024 and an eighth of a bin at 4 096. The same line of
# code meant something different at each window, and a sweep over it would have
# confounded a change of field of view with a change of method.

# %%
shipped_widths = []
for window in WINDOWS:
    frequencies = np.fft.rfftfreq(window, d=1.0 / SAMPLING_HZ)
    in_band = int(((frequencies >= BAND_HZ[0]) & (frequencies <= BAND_HZ[1])).sum())
    shipped_widths.append(
        {"window": window, "shipped": ENVELOPE_BINS + in_band, "invariant": ENVELOPE_BINS + 37}
    )
    print(f"window {window:5d}  Δf = {frequencies[1]:7.1f} Hz  "
          f"spectral bins {in_band:4d}  descriptor {ENVELOPE_BINS + in_band:4d}-D  "
          f"σ = {8 / (window // ENVELOPE_BINS):.2f} envelope bin")

# %% [markdown]
# **The process.** The repair is to make the descriptor a function of the
# **physics** — a band and a number of bands — rather than of the array length.
# The fine spectrum is averaged onto a **fixed grid of 37 bands** whose edges are
# exactly the bins a 1 024-sample window produces, and σ is expressed in envelope
# bins. This section prototyped that change; it has since been adopted into the
# shipped `morphology_features` under the gate that made it safe to take: the new
# descriptor is **byte-identical** to the old one at 1 024 samples on real events,
# so every published number is untouched. The prototype is gone from here — a
# second copy of a shipped method is exactly the drift this notebook exists to
# prevent — and the invariance is measured below on real event crops instead.

# %%
BAND_CENTRES, BAND_EDGES = spectrum_band_grid(sampling_frequency_hz=SAMPLING_HZ)
widest = load_real_events(registry, window=4096)  # reused by the sweep and the control
widths = {
    window: int(morphology_features(
        widest.cores[:256, (4096 - window) // 2 : (4096 + window) // 2]
    ).shape[1])
    for window in WINDOWS
}
print(f"fixed band grid: {BAND_CENTRES.size} bands from "
      f"{BAND_CENTRES[0] / 1000:.2f} to {BAND_CENTRES[-1] / 1000:.2f} kHz")
print("descriptor width on real event crops:  "
      + "  ".join(f"{window}: {width}-D" for window, width in widths.items()))
assert len(set(widths.values())) == 1, "the descriptor still depends on its window"

# %%
draw_descriptor_widths(shipped_widths)

# %% [markdown]
# **The conclusion.** One descriptor at every window — 101 numbers whatever the
# array length — and the adoption gate proves it is *the same* descriptor at
# 1 024 as the one every published number was measured with. That is what
# licenses reading the sweep below as a change of field of view rather than a
# change of method. It settles nothing else: byte-identity holds at the reference
# window only, and nothing here says a wider window describes reality better.
# That is the next measurement.
#
# ### Step 2 · Sweeping the window
#
# **The problem.** Does a wider field of view describe reality better? The
# protocol of the coverage chain is held fixed — one PCA(16) basis fitted on all
# three generator conditions pooled, one balanced synthetic draw shared across
# conditions (seed 20260809), a per-condition radius at the 80th percentile of
# synthetic self-nearest-neighbour distance, euclidean in 16 dimensions. Only
# the window changes, and the whole protocol comes from
# `internship_workspace.chain_data`, the same helpers the reproduction test
# pins against the published run.
#
# Three quantities are read together, and the second and third are there to
# answer an objection rather than to decorate:
#
# - **coverage** — the fraction of real events inside the synthetic support;
# - **domain AUC** — how well a classifier separates real from synthetic in that
#   basis. Lower is better: 0.5 means indistinguishable;
# - **median real→synthetic nearest-neighbour distance** — a raw distance, not a
#   ratio, so it cannot be gamed by a change of radius.
#
# **The objection.** A wider window adds noise-only samples that both populations
# share. That alone could raise coverage while describing the *event* no better —
# the descriptor would simply be measuring more silence. If the window were only
# buying shared silence, coverage would rise while the domain AUC stayed put or
# rose too, because the events themselves would be no better matched. Coverage
# rising *while* AUC falls is the signature of a genuinely better description.

# %%
sweep = []
for window in WINDOWS:
    started = time.time()
    real = widest if window == 4096 else load_real_events(registry, window=window)
    conditions = load_conditions(registry, window=window)
    basis = pooled_basis(conditions, real.labels, seed=SEED)
    chain = coverage_chain(basis, real, conditions, quantile=QUANTILE)
    domain = domain_metrics(
        basis.project(real.features),
        basis.project(conditions.features["asymmetry_5d"]),
        real.labels, conditions.labels, seed=SEED,
    )
    sweep.append(window_sweep_row(window, real, basis, chain, domain, CLASS_ORDER))
    print_sweep_line(sweep[-1], CLASS_ORDER, time.time() - started)

# %% [markdown]
# #### Reproduction check at the published window
#
# At 1 024 the invariant descriptor is the identity, so the whole chain must
# return the published q80 numbers unchanged. A non-zero deviation here would
# mean the sweep changed the method rather than the window, and nothing below
# would be readable.

# %%
reference = published("particle-z8-v2-coverage-conditions-q80-r1")
baseline = next(row for row in sweep if row["window"] == REFERENCE_WINDOW)
window_deviation = max(
    abs(baseline["chain"][label][class_name]
        - reference["conditions"][label]["classes"][class_name]["real_within_radius_fraction"])
    for label in baseline["chain"]
    for class_name in CLASS_ORDER
)
assert window_deviation == 0.0, f"reproduction drifted by {window_deviation:.3e}"
print("at 1024 the swept chain reproduces "
      f"particle-z8-v2-coverage-conditions-q80-r1 exactly "
      f"(max deviation {window_deviation:.1e})")

# %%
draw_window_sweep(sweep, CLASS_ORDER, CLASS_COLOUR)

# %%
print_window_sweep(sweep, CLASS_ORDER)

# %% [markdown]
# **The conclusion: the sweep refutes the plan.** The alignment note expected a
# wider window to describe reality better. It does the opposite, and both
# quantities agree, which is what makes the result hard to dismiss: coverage
# **falls** from 93.6 % to 71.1 % at 2 µm as the window widens, while the domain
# AUC **rises** from 0.66 to 0.75 — real and synthetic become *easier* to tell
# apart, not harder. A story the numbers refute is itself the finding, so the
# recommendation in the plan does not survive this section unchanged.
#
# **What it leaves open.** Two readings are still on the table, and they differ
# in what they would have the project do. Either the narrow window is genuinely
# the better description of an event, or the synthetic traces are simply
# unrealistic outside their central core and the wider window is punishing an
# artefact. The next step separates them.
#
# ### Step 3 · Is the wider window punishing an artefact?
#
# **The problem.** If the generator zero-pads, flattens or otherwise abandons the
# trace outside the event — an unmodelled edge, a noise carrier injected only
# near the centre — then widening the window measures a packaging decision rather
# than the simulator's physics, and the sweep says nothing about description
# quality at all.
#
# **The process.** One figure settles it: the energy profile of both populations
# across the same 4 096-sample window, in blocks of 512 samples. The comparison
# has to be **class-matched**. A first attempt on the leading rows of each table
# showed a 66 % amplitude gap in the event core, which dissolved once classes
# were matched — the two tables are ordered differently by class, so the gap was
# composition, not physics. Recording that here because it is the trap this
# control exists to avoid.

# %%
BLOCK = 512
CORE = slice(1536, 2560)  # the 1024-sample window the descriptor reads
terminal_root = conditions.roots["asymmetry_5d"]  # roots do not depend on the window
terminal_raw = np.load(terminal_root / "signals_raw_4096.npy",
                       mmap_mode="r", allow_pickle=False)
terminal_labels = conditions.labels  # class_name is a paired field, so it is shared
generator = np.random.default_rng(SEED)

profiles, ratios, counts = {}, {}, {}
for class_name in CLASS_ORDER:
    real_slice = widest.by_class(widest.cores, class_name)
    eligible = np.flatnonzero(terminal_labels == class_name)
    drawn = np.sort(generator.choice(eligible, size=len(real_slice), replace=False))
    synthetic_slice = np.asarray(terminal_raw[drawn])
    profiles[class_name] = (block_energy_profile(real_slice, block=BLOCK),
                            block_energy_profile(synthetic_slice, block=BLOCK))
    ratios[class_name] = (core_context_ratio(real_slice, CORE),
                          core_context_ratio(synthetic_slice, CORE))
    counts[class_name] = int(len(real_slice))

draw_energy_profiles(profiles, counts, CLASS_ORDER, CLASS_COLOUR, block=BLOCK, core=CORE)
print_core_context(ratios, CLASS_ORDER)

# %% [markdown]
# **The conclusion.** Class by class, the two populations carry the same energy
# in the same places: the synthetic traces are not padded and not flat outside
# the core, so the wider window is not punishing an artefact and the sweep is
# measuring the simulator. The one visible discrepancy is at 10 µm, where
# synthetic events stand out against their context more than real ones do — a
# core-to-context ratio of 3.7 against 2.7. That is a real mismatch, and one the
# 1 024-sample window cannot see, because at that width the window *is* the core.
#
# **What it leaves open.** The control shows the context is modelled, not that it
# is modelled *well*; the 10 µm gap is exactly the kind of thing a wider window
# would start charging the generator for, and no run measures how much.
#
# ### Step 4 · What the window actually selects
#
# **The problem.** Two honest measurements now disagree, and a reader has to be
# told which question each answers. Put the sweep beside the annotated supports:
# the 1 024-sample window covers the event's high-energy core but clips the
# low-amplitude tails the detector's support includes, while 4 096 admits those
# tails *and* several thousand samples of instrument context the generator never
# claimed to model.

# %%
print_support_containment(widest.support_widths, WINDOWS)

# %% [markdown]
# **The conclusion.** The sweep is not ranking descriptions of the event. It is
# measuring **how much of the recording you agree to be judged on**:
#
# - at 1 024 the test asks "does the simulator reproduce the core of an event?"
#   and the answer is yes, for 85–94 % of real events;
# - at 4 096 it asks "does the simulator reproduce an event *and its
#   surroundings*?" and the answer drops to 63–71 %.
#
# Neither number is wrong. The narrow one is the more flattering, and the deck
# reports it without saying which question it answers. That is the finding this
# section delivers, and it is an editorial decision as much as a technical one.
#
# #### Recommendation
#
# **Keep the descriptor window-invariant, and do not adopt 4 096 for the coverage
# claim on the strength of the original argument** — that argument was that a
# wider window describes the event better, and it is refuted here. What the
# numbers support instead:
#
# 1. **Keep the fixed 37-band grid and the bin-unit smoothing regardless.** They
#    are the identity at 1 024, so they cost nothing today, and they are what
#    makes any future window change interpretable at all.
# 2. **Report the window as part of the claim.** "85–94 % coverage of the event
#    core (1 024 samples, 0.512 ms)" is defensible; "85–94 % coverage" alone is
#    not, because the number moves by twenty points with a choice the reader
#    cannot see.
# 3. **Consider publishing both windows.** The pair — flattering and demanding —
#    is more convincing than either alone, and it pre-empts the exact objection a
#    jury would raise.
#
# **What this section does not settle.** Whether retrieval behaves the same way
# under a wider window: it is measured on the encoder's own 512-point input, a
# different path this sweep never touches. Whether the MAD corpus — whose
# supports run to 4 000 samples — shifts the balance, since every number here is
# measured on z8 detections. And whether the 10 µm context mismatch is worth
# fixing in the generator, which needs a measurement no run currently owns. All
# three belong to the redo.

# %%
metrics, provenance = window_evidence(
    sweep,
    windows=WINDOWS,
    quantile=QUANTILE,
    seed=SEED,
    bands=BAND_CENTRES.size,
    envelope_bins=ENVELOPE_BINS,
    band_hz=BAND_HZ,
    reproduction={
        "run_id": "particle-z8-v2-coverage-conditions-q80-r1",
        "max_deviation": window_deviation,
    },
    datasets=registry.provenance(),
    inputs={"events_csv_sha256": notebook_evidence.sha256_file(widest.events_csv)},
)
try:
    emitted = notebook_evidence.emit_run(
        workspace,
        section="window-alignment-sweep",
        metrics=metrics,
        provenance=provenance,
        claim_boundary=(
            "Measures how the morphology descriptor's window changes coverage, "
            "domain separability and neighbour distance on the z8 development "
            "events, with the shipped window-invariant descriptor that is the "
            "identity at 1024. It recommends a window; it does not adopt one, "
            "does not re-measure retrieval, and authorizes no dataset promotion."
        ),
    )
    print(f"emitted {emitted.name}")
except WorkspaceError as error:
    print(f"no evidence emitted ({error})")


# %% [markdown]
# ## Alignment · the quantile and the basis
#
# *Exploratory section. It tests two changes that the alignment plan proposes but
# has not adopted, and its conclusions are recommendations, not shipped
# decisions.*
#
# **The problem.** The coverage claim of this work has the form *"n % of real
# events fall inside the synthetic support"*. That sentence hides two dials the
# reader never sees:
#
# - the **radius quantile** — "inside" means *within a distance r* of some
#   synthetic event, and r is a quantile of the synthetic cloud's own
#   nearest-neighbour distances. The chain moved from the 95th percentile (q95)
#   to the 80th (q80);
# - the **basis** — that distance is measured in a 16-dimensional space obtained
#   by **PCA** (principal component analysis: an orthonormal rotation of the
#   101-number descriptor onto the directions of largest variance, truncated).
#   Which events the PCA was fitted on changes the space, and therefore changes
#   what "close" means.
#
# The companion window section already showed what such a dial can do: widening
# the descriptor window moved coverage by twenty points and refuted the change it
# was meant to support. That is the register here too. An alignment knob is not a
# formatting detail; each of these can flatter the simulator or challenge it, and
# the reader is entitled to know by how much.
#
# The plan's decisions under test are **A7** (q80 everywhere, parameterised, JSON
# keys named after the real value) and **A8** (one serialised PCA basis per
# campaign, reloaded by every consumer). Both are tested with numbers below, and
# both survive — but not for the reasons the plan gives, and one of the deck's
# sentences does not survive at all.

# %%
from internship_workspace.chain_data import (  # noqa: E402
    Registry,
    coverage_chain,
    load_conditions,
    load_real_events,
    pooled_basis,
)
from internship_workspace.z8_coverage import support_coverage  # noqa: E402
from internship_workspace.z8_domain_pca import (  # noqa: E402
    balanced_class_indices,
    domain_metrics,
    fit_synthetic_pca,
)
from scipy.linalg import subspace_angles  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

CHAIN_SEED = 20260809
started = time.time()
registry = Registry(workspace)
real = load_real_events(registry)
conditions = load_conditions(registry)
basis = pooled_basis(conditions, real.labels, seed=CHAIN_SEED)

chain_real = np.asarray(basis.project(real.features), dtype=np.float64)
chain_synthetic = {
    label: np.asarray(basis.project(conditions.features[label]), dtype=np.float64)
    for label in conditions.order
}
print(f"{len(real.rows)} real train/val events, {len(conditions.labels)} synthetic per "
      f"condition, descriptor {real.features.shape[1]}-D  ({time.time() - started:.0f} s)")


def coverage_at(label, quantile, dimensions=16):
    """The shipped coverage protocol, at one condition and one radius quantile.

    Everything except the quantile is held fixed: the pooled PCA(16) basis, the
    balanced synthetic draw the radius is taken over (seed 20260809), and the
    full synthetic class population a real event is measured against.
    """
    return support_coverage(
        chain_real, chain_synthetic[label], real.labels, conditions.labels,
        sample=basis.sample, quantile=quantile, dimensions=dimensions,
    )


# %% [markdown]
# ### Reproducing the published q80 chain
#
# Nothing below is readable unless this notebook computes the same chain the deck
# quotes. The protocol is the shipped one, loaded through
# `internship_workspace.chain_data`: one PCA(16) fitted on the three generator
# conditions pooled, one balanced synthetic draw shared across conditions, a
# radius recomputed per condition, euclidean distance in 16 dimensions, real
# events cropped to 1 024 samples and synthetic events sliced from the same
# 1 024 samples of their 4 096-sample core.

# %%
chain = coverage_chain(basis, real, conditions, quantile=0.80)
reference = published("particle-z8-v2-coverage-conditions-q80-r1")
deviation = {
    field: max(abs(chain[label][class_name][field]
                   - reference["conditions"][label]["classes"][class_name][field])
               for label in conditions.order for class_name in CLASS_ORDER)
    for field in ("real_within_radius_fraction", "synthetic_self_nn_radius")
}
chain_deviation = deviation["real_within_radius_fraction"]
radius_deviation = deviation["synthetic_self_nn_radius"]
assert chain_deviation == 0.0, f"reproduction drifted by {chain_deviation:.3e}"
assert radius_deviation == 0.0, f"radius drifted by {radius_deviation:.3e}"
for class_name in CLASS_ORDER:
    print(f"{class_name:>5}  " + " → ".join(
        f"{chain[label][class_name]['real_within_radius_fraction']:.6f}"
        for label in conditions.order))
print("\nreproduces particle-z8-v2-coverage-conditions-q80-r1 exactly "
      f"(coverage and radius, max deviation {chain_deviation:.1e})")

# %% [markdown]
# ### Part 1 · What the quantile decides
#
# **The problem.** Four runs exist on this identical pipeline, differing only by
# `--quantile`: q95 (the tool's default, and the deck's original choice), q90,
# q85 and q80. They are the cleanest controlled experiment available on this
# question, because the basis, the sample, the seed and the data are
# byte-identical across them — only the tick on the ruler moves. What has never
# been asked is whether the quantile **fabricates** the story or merely
# **amplifies** it. Those are different failures: amplification is a presentation
# problem, where the ranking is real and the reader is shown its most flattering
# scale; fabrication would be a scientific problem, where the ranking itself
# depends on an arbitrary choice.

# %%
PUBLISHED_QUANTILES = (0.80, 0.85, 0.90, 0.95)
published_runs = {
    0.95: published("particle-z8-v2-coverage-conditions-r1"),
    0.90: published("particle-z8-v2-coverage-conditions-q90-r1"),
    0.85: published("particle-z8-v2-coverage-conditions-q85-r1"),
    0.80: published("particle-z8-v2-coverage-conditions-q80-r1"),
}
for quantile, payload in published_runs.items():
    assert payload["configuration"]["quantile"] == quantile
    assert payload["pca"]["explained_variance_16"] == \
        published_runs[0.95]["pca"]["explained_variance_16"]
print("the four runs share one basis (explained variance over 16 components "
      f"{published_runs[0.95]['pca']['explained_variance_16']:.6f}); "
      "only the quantile differs\n")
print_published_gains(published_runs, PUBLISHED_QUANTILES, CLASS_ORDER)

# %% [markdown]
# **The conclusion so far.** The gains move a great deal. The first generator
# step — replacing white noise with measured instrument noise — is worth
# **+14.3 points** at 2 µm when scored at q95 and **+31.0 points** at q80: the
# headline more than doubles. The second step — the paired asymmetry — is worth
# **+3.1 points** at 4 µm under q95 and **+9.8** under q80: it triples.
#
# Both movements have the same cause, and it is not rhetorical. At q95 the radius
# is generous enough that the middle condition already covers 98.0 % of the 2 µm
# events; there is almost no room left, so the last step cannot show anything.
# **q95 measures against a ceiling.** Lowering the quantile un-saturates the
# measurement, which is a legitimate reason to prefer q80 and a better one than
# "q80 is stricter, therefore more honest".
#
# **What it leaves open.** The fabrication question. A story that only holds on
# the four quantiles someone chose to publish is not a story, so the sweep has to
# go where no run has been — below 0.80.

# %%
QUANTILE_GRID = tuple(round(float(value), 2) for value in np.arange(0.10, 0.96, 0.05))
started = time.time()
quantile_sweep = {
    quantile: {label: coverage_row(coverage_at(label, quantile), CLASS_ORDER)
               for label in conditions.order}
    for quantile in QUANTILE_GRID
}
violations = [
    (quantile, class_name)
    for quantile in QUANTILE_GRID for class_name in CLASS_ORDER
    if not (quantile_sweep[quantile]["white_noise_4d"][class_name]["coverage"]
            < quantile_sweep[quantile]["real_noise_4d"][class_name]["coverage"]
            < quantile_sweep[quantile]["asymmetry_5d"][class_name]["coverage"])
]
grid_deviation = max(
    abs(quantile_sweep[quantile][label][class_name]["coverage"]
        - published_runs[quantile]["conditions"][label]["classes"][class_name][
            "real_within_radius_fraction"])
    for quantile in PUBLISHED_QUANTILES
    for label in conditions.order for class_name in CLASS_ORDER
)
assert grid_deviation == 0.0, f"the swept grid drifted by {grid_deviation:.3e}"
print(f"swept {len(QUANTILE_GRID)} quantiles from {min(QUANTILE_GRID):.2f} to "
      f"{max(QUANTILE_GRID):.2f} in {time.time() - started:.0f} s; the four published "
      "quantiles are reproduced exactly")
print(f"orderings white < real-noise < asymmetry violated: "
      f"{len(violations)} of {len(QUANTILE_GRID) * len(CLASS_ORDER)}")

# %%
draw_quantile_curve(quantile_sweep, QUANTILE_GRID, list(conditions.order), CLASS_ORDER,
                    CLASS_COLOUR, PUBLISHED_QUANTILES)

# %% [markdown]
# **The conclusion: the quantile amplifies; it does not fabricate.** Across 0.10
# to 0.95 — a range far wider than anyone would defend — the three conditions
# never change places, in any class: 54 orderings, 0 violations. The chain's
# conclusion, that measured noise buys most of the coverage and asymmetry
# finishes it, is a property of the generator changes and not of the ruler.
#
# **What it leaves open.** Every *absolute* number the deck prints is still a
# property of the ruler, and one sentence in particular is built on one.

# %%
draw_gain_amplification(published_runs, PUBLISHED_QUANTILES, CLASS_ORDER, CLASS_COLOUR)

# %% [markdown]
# #### Where the argument breaks
#
# **The problem.** The deck's closing sentence on the chain is that **every class
# ends above the 80 % line**. At q80 that reads as a coincidence worth exploiting:
# 80 % of the synthetic reference events lie within the radius of another
# synthetic event, so a coverage above 80 % looks like "real events are covered
# better than the synthetic cloud covers itself". Two things have to be checked
# before that sentence can be used — where it stops being arithmetically true,
# and whether the two numbers being compared are measurements of the same thing.

# %%
CROSSING_GRID = tuple(round(float(value), 2) for value in np.arange(0.60, 0.96, 0.01))
crossing = {
    quantile: {c: coverage_at("asymmetry_5d", quantile)[c]["real_within_radius_fraction"]
               for c in CLASS_ORDER}
    for quantile in CROSSING_GRID
}
above_bar = min(q for q in CROSSING_GRID if all(crossing[q][c] >= 0.80 for c in CLASS_ORDER))
above_self = max(q for q in CROSSING_GRID if all(crossing[q][c] >= q for c in CLASS_ORDER))
print_crossing(published_runs, PUBLISHED_QUANTILES, crossing, above_bar, above_self,
               CLASS_ORDER)

# %% [markdown]
# **The conclusion, first half.** Arithmetically the sentence lives in a corridor,
# **0.72 ≤ q ≤ 0.91**, and the deck's previous choice was outside it. At q95 the
# sentence is simply false: 10 µm covers 93.5 % against a 95 % bar, a margin of
# −1.5 points, and 4 µm clears its own bar by 0.2. Below q = 0.72 it fails at the
# other end, the 4 µm class dropping under 80 %. So the move to q80 did not merely
# make the claim stronger — at q95 there was no claim to make. That is worth
# saying plainly, because it is the one place where a quantile change repaired
# something rather than flattering it. It also means the sentence is true only
# over a nineteen-point window of an arbitrary parameter, which is thin support
# for a closing line.
#
# #### The sharpest objection: the two numbers are not measured the same way
#
# **The problem.** The radius is the 80th percentile of nearest-neighbour
# distances **inside a thinned reference draw** — one synthetic event per real
# event of that class, because that is what makes the conditions comparable. But a
# real event's distance is measured to the **whole** synthetic class population.
# The two sides of the comparison therefore search reference sets of very
# different densities.

# %%
density = {}
for class_name in CLASS_ORDER:
    population = int((conditions.labels == class_name).sum())
    drawn = int(basis.sample[class_name].size)
    density[class_name] = {
        "synthetic_population": population,
        "reference_draw": drawn,
        "real": int((real.labels == class_name).sum()),
        "density_ratio": population / drawn,
    }
print_reference_density(density, CLASS_ORDER)

# %% [markdown]
# A real event queries a cloud sixteen to twenty-five times denser than the one
# whose spacing set the radius. Nearest-neighbour distance falls with density, so
# real events are being scored on an easier test than the 80 % they are compared
# against.
#
# **The process.** Score the synthetic events the same way real ones are scored:
# take the events *not* in the reference draw and measure each one's distance to
# its nearest **other** synthetic event of the same class in the full cloud — the
# exact query a real event gets — against the exact same radius. And score the
# real events the other way round, against the thinned draw, so both populations
# are compared at one density in both directions.

# %%
control = {}
for class_name in CLASS_ORDER:
    scores = chain_synthetic["asymmetry_5d"][:, :16]
    class_index = np.flatnonzero(conditions.labels == class_name)
    reference_cloud = scores[basis.sample[class_name]]
    self_distance = NearestNeighbors(n_neighbors=2).fit(reference_cloud).kneighbors(
        reference_cloud, return_distance=True)[0][:, 1]
    radius = float(np.quantile(self_distance, 0.80))
    held_out = np.setdiff1d(class_index, basis.sample[class_name])
    synthetic_distance = NearestNeighbors(n_neighbors=2).fit(scores[class_index]).kneighbors(
        scores[held_out], return_distance=True)[0][:, 1]
    real_slice = chain_real[real.labels == class_name][:, :16]
    real_full = NearestNeighbors(n_neighbors=1).fit(scores[class_index]).kneighbors(
        real_slice, return_distance=True)[0][:, 0]
    real_matched = NearestNeighbors(n_neighbors=1).fit(reference_cloud).kneighbors(
        real_slice, return_distance=True)[0][:, 0]
    control[class_name] = {
        "radius": radius,
        "held_out_synthetic": int(held_out.size),
        "real_full": float(np.mean(real_full <= radius)),
        "real_matched": float(np.mean(real_matched <= radius)),
        "synthetic_leave_one_out": float(np.mean(synthetic_distance <= radius)),
    }
    print(f"{class_name:>5}  radius {radius:.3f}   real vs full cloud "
          f"{100 * control[class_name]['real_full']:6.2f}%   real vs matched cloud "
          f"{100 * control[class_name]['real_matched']:6.2f}%   synthetic vs its own "
          f"cloud {100 * control[class_name]['synthetic_leave_one_out']:6.2f}%")

# %%
draw_density_control(control, CLASS_ORDER, CLASS_COLOUR)

# %% [markdown]
# **The conclusion: the sentence does not survive.** Measured the way real events
# are measured, the synthetic cloud covers **itself** at 99.4 / 97.9 / 99.2 %,
# against 93.6 / 85.2 / 87.9 % for the real events. Real events are covered
# *worse* than the synthetic cloud covers itself, by 6 to 13 points — the opposite
# of what the deck asserts. Measured the other way round, with both populations
# querying the same thinned cloud, real coverage falls to 56.8 / 68.3 / 70.1 %
# against the 80 % the reference draw achieves by construction. Either fair
# comparison inverts the claim; only the unmatched one supports it.
#
# **What this does and does not damage.** It does not touch the chain's ranking:
# the density ratio is identical across the three conditions, so it cancels out of
# every gain the chain reports, which is exactly the quantity the tool was built
# to measure and the only quantity its claim boundary asserts. What it damages is
# the *absolute* reading, and specifically the one rhetorical sentence that turns
# a coverage percentage into a statement about the cloud's own coherence. The
# 85–94 % is a real measurement of a real question — "how far is a real event from
# the nearest of ~16 000 synthetic events, compared with a radius derived from a
# much sparser draw" — but it is not the question the sentence claims it answers.
#
# **Recommendation.** Keep q80 and keep the chain. Drop the self-coverage
# sentence, or replace it with the matched-density number, which is defensible,
# unflattering and reported here for the first time. And report the quantile in
# the claim, since the same data yields 60.8 % or 83.7 % for white noise at 2 µm
# depending only on it.
#
# **What this does not settle.** Which of the two fair protocols the redo should
# adopt: the leave-one-out reading and the matched-draw reading disagree by thirty
# points and answer different questions, and choosing between them is a decision
# no measurement here makes.
#
# #### The inconsistency the plan names, and what it costs
#
# A7 also observes that `internship_workspace.z8_domain_pca.domain_metrics`
# **hard-codes** its radius:
#
# ```python
# threshold = float(np.quantile(self_distance, 0.95))   # z8_domain_pca.py, ~line 175
# ```
#
# and publishes it under the key `real_within_synthetic_self_p95_fraction`. Every
# analysis that goes through `domain_metrics` — including the dimension sweep of
# Part 3 — therefore still reports q95 while the chain reports q80. The two are
# not commensurable, and they appear in the same deck. Part 3 quantifies the gap.

# %% [markdown]
# ### Part 2 · What the basis decides
#
# **The problem.** The chain fits one PCA(16) on all three conditions pooled with
# a shared balanced draw. The two-space introduction figures — the slides that
# teach the reader what "an event is a point" means — instead reuse the **stored
# scores** of `particle-z8-v2-paired-asymmetry-pca-r2`, a basis fitted on the v4
# and v5 arms only, with a different draw and a different seed. The run is honest
# about it: its own metrics carry `"not_commensurable_with":
# "particle-z8-v2-coverage-conditions-r1"`. The two nonetheless sit on
# neighbouring slides. "Not commensurable" is a statement about licence, not about
# magnitude, and the reader deserves the magnitude.
#
# **The process.** The intro basis is refitted here from the registry and checked
# against the stored scores it is supposed to be, so that the comparison below is
# between two bases rather than between a basis and a memory.

# %%
INTRO_SEED = 20260803
intro_scores = np.load(
    run_dir("particle-z8-v2-paired-asymmetry-pca-r2") / "pca_scores.npz", allow_pickle=True
)
intro_fit_indices = balanced_class_indices(
    conditions.labels, per_class=int(intro_scores["fit_indices"].size // len(CLASS_ORDER)),
    seed=INTRO_SEED,
)
assert np.array_equal(intro_fit_indices, intro_scores["fit_indices"]), "draw differs"
intro_pool = np.concatenate(
    (conditions.features["real_noise_4d"][intro_fit_indices],
     conditions.features["asymmetry_5d"][intro_fit_indices]),
    axis=0,
)
intro_scaler = StandardScaler().fit(intro_pool)
intro_pca = PCA(n_components=16, svd_solver="full").fit(intro_scaler.transform(intro_pool))


def intro_project(values):
    return intro_pca.transform(intro_scaler.transform(values))


intro_deviation = float(np.abs(
    intro_project(conditions.features["asymmetry_5d"])
    - np.asarray(intro_scores["candidate"], dtype=np.float64)
).max())
assert intro_deviation < 1.0e-9, f"reproduction drifted by {intro_deviation:.3e}"
print("the balanced draw and the 16 stored score columns of "
      f"particle-z8-v2-paired-asymmetry-pca-r2 are reproduced from the registry "
      f"(max deviation {intro_deviation:.1e}, float32 storage)")

variance_table = {
    "chain · three conditions pooled": (basis.explained_variance,
                                        basis.explained_variance_pc1_pc2),
    "intro · v4+v5 paired arms": (float(intro_pca.explained_variance_ratio_.sum()),
                                  float(intro_pca.explained_variance_ratio_[:2].sum())),
}
print(f"\n{'basis':>34} {'16 axes':>10} {'PC1-PC2':>9}")
for name, (sixteen, plane) in variance_table.items():
    print(f"{name:>34} {100 * sixteen:9.2f}% {100 * plane:8.2f}%")

# %% [markdown]
# #### How different is "close"?
#
# Two 16-dimensional subspaces of the same 101-dimensional descriptor space can be
# compared exactly, by their **principal angles** — the sequence of angles between
# the closest pair of directions, then the closest pair orthogonal to those, and
# so on. Zero everywhere means the same subspace; 90° means a direction one basis
# reads and the other is blind to. The scaling step belongs to the projection, so
# the directions compared are the rows of the components divided by each basis's
# own standard deviations.

# %%
def read_directions(scaler, pca):
    """An orthonormal basis of the descriptor directions a projection reads."""
    return np.linalg.qr((pca.components_ / scaler.scale_[None, :]).T)[0]


principal_angles = np.degrees(
    subspace_angles(read_directions(basis.scaler, basis.pca),
                    read_directions(intro_scaler, intro_pca))
)
print("principal angles between the chain basis and the intro basis, degrees:")
print("  " + "  ".join(f"{angle:5.1f}" for angle in np.sort(principal_angles)))
print(f"\nmedian {np.median(principal_angles):.1f}°, mean {principal_angles.mean():.1f}°, "
      f"largest {principal_angles.max():.1f}°; mean cos² "
      f"{np.mean(np.cos(np.radians(principal_angles)) ** 2):.4f}")

# %%
intro_real = np.asarray(intro_project(real.features), dtype=np.float64)
intro_terminal = np.asarray(
    intro_project(conditions.features["asymmetry_5d"]), dtype=np.float64
)
neighbour = {}
for class_name in CLASS_ORDER:
    real_index = np.flatnonzero(real.labels == class_name)
    synthetic_index = np.flatnonzero(conditions.labels == class_name)
    chain_distance, chain_neighbour = NearestNeighbors(n_neighbors=1).fit(
        chain_synthetic["asymmetry_5d"][synthetic_index][:, :16]
    ).kneighbors(chain_real[real_index][:, :16])
    intro_distance, intro_neighbour = NearestNeighbors(n_neighbors=1).fit(
        intro_terminal[synthetic_index]
    ).kneighbors(intro_real[real_index])
    neighbour[class_name] = {
        "distance_chain": chain_distance[:, 0],
        "distance_intro": intro_distance[:, 0],
        "pearson": float(pearsonr(chain_distance[:, 0], intro_distance[:, 0])[0]),
        "spearman": float(spearmanr(chain_distance[:, 0], intro_distance[:, 0]).statistic),
        "same_nearest_fraction": float(np.mean(chain_neighbour[:, 0] == intro_neighbour[:, 0])),
    }
    row = neighbour[class_name]
    print(f"{class_name:>5}  n = {real_index.size:4d}   pearson {row['pearson']:.3f}   "
          f"spearman {row['spearman']:.3f}   same nearest synthetic event "
          f"{100 * row['same_nearest_fraction']:5.1f} %")

# %%
draw_basis_divergence(principal_angles, neighbour, CLASS_ORDER, CLASS_COLOUR)

# %% [markdown]
# **The conclusion is precise and two-sided.** Fifteen of the sixteen directions
# agree to within 23°, and the sixteenth is 72.5° apart: the two bases share a
# 15-dimensional core and disagree almost completely about one direction. That is
# enough for the *magnitudes* to transfer — real-to-synthetic distances correlate
# at 0.92 to 0.98 across the bases — and not nearly enough for the *identities* to
# transfer: the nearest synthetic event is the same one only **46 to 55 %** of the
# time, barely better than a coin flip.
#
# That asymmetry is exactly the wrong way round for how the two bases are used.
# The chain consumes magnitudes, so its numbers are robust to the basis; the next
# cell shows the headline coverage moving by at most 2.6 points. The intro figure
# consumes an *identity* — it draws a real event beside "its nearest synthetic
# neighbour" and asks the audience to read the pair as a like-for-like comparison.
# About half the time, the chain would have drawn a different partner. The slide's
# pedagogical claim is the fragile one, not the chain's number.

# %%
basis_coverage = {}
for name, (real_scores, synthetic_scores) in (
    ("chain · pooled", (chain_real, chain_synthetic["asymmetry_5d"])),
    ("intro · paired arms", (intro_real, intro_terminal)),
):
    measured = support_coverage(real_scores, synthetic_scores, real.labels,
                                conditions.labels, sample=basis.sample, quantile=0.80)
    basis_coverage[name] = {
        c: measured[c]["real_within_radius_fraction"] for c in CLASS_ORDER
    }
    print(f"{name:>20}: " + "  ".join(
        f"{c} {100 * basis_coverage[name][c]:6.2f}%" for c in CLASS_ORDER))
print("\nbasis-induced shift in the headline coverage, percentage points: "
      + "  ".join(
          f"{c} {100 * (basis_coverage['intro · paired arms'][c] - basis_coverage['chain · pooled'][c]):+.2f}"
          for c in CLASS_ORDER))

# %% [markdown]
# #### Checking the numbers the deck quotes
#
# The introduction slide states that 16 axes hold **60.9 %** of the descriptor
# variance and the plotted PC1–PC2 plane **21.1 %**, and attributes them to
# `particle-z8-v2-coverage-conditions-r1`. The attribution is correct to the
# digit. The picture is the problem.

# %%
chain_run = published("particle-z8-v2-coverage-conditions-r1")
sweep_variance = published("particle-z8-v2-real-synthetic-pca-r2")[
    "variants"]["Morphology · primary"]["explained_variance_ratio"]
paired_variance = published("particle-z8-v2-paired-asymmetry-pca-r2")["explained_variance_ratio"]
quoted = {"16 axes": 60.9, "PC1-PC2": 21.1}
bases = {
    "coverage-conditions-r1 · what the slide cites": (
        100 * chain_run["pca"]["explained_variance_16"],
        100 * chain_run["pca"]["explained_variance_pc1_pc2"]),
    "paired-asymmetry-pca-r2 · what the slide plots": (
        100 * sum(paired_variance[:16]), 100 * sum(paired_variance[:2])),
    "real-synthetic-pca-r2 · what Part 3 sweeps": (
        100 * sum(sweep_variance[:16]), 100 * sum(sweep_variance[:2])),
}
print(f"{'basis':>46} {'16 axes':>10} {'PC1-PC2':>10}   deviation from the quoted pair")
for name, (sixteen, plane) in bases.items():
    print(f"{name:>46} {sixteen:9.2f}% {plane:9.2f}%   "
          f"{sixteen - quoted['16 axes']:+6.2f} / {plane - quoted['PC1-PC2']:+5.2f} pp")
intro_metrics = published("ssl-v18-two-space-intro-r2")
print(f"\nthe intro figure reports coverage {100 * intro_metrics['coverage_fraction']:.2f} % "
      f"at 2 µm, against the chain's "
      f"{100 * chain['asymmetry_5d']['2um']['real_within_radius_fraction']:.2f} % "
      "— same class, same condition, same quantile, different basis and different draw")
print(f"its radius is stored under the key 'radius_p95' with the value "
      f"{intro_metrics['radius_p95']:.6f}, produced by --quantile 0.80")

# %% [markdown]
# **The conclusion.** Three bases are in play on adjoining slides, and each has a
# different variance profile. The quoted 60.9 / 21.1 belongs to the chain's pooled
# basis; the plane actually drawn on that slide belongs to the paired-arms basis
# and holds **23.4 %**, not 21.1 %, with 16 axes holding **63.2 %**, not 60.9 %.
# The figure tool hard-codes the wrong one into its own axis label. The error is
# small — 2.3 percentage points — and the defect is not: a caption that describes
# a different picture from the one above it is unfalsifiable by the reader.
#
# The naming defect A7 flags is confirmed in the same file. The tool writes its
# radius under the key **`radius_p95`** whatever `--quantile` was passed, and r2
# was run with `--quantile 0.80`; the published `radius_p95` is a p80 radius.
# Anyone reading that JSON without the run command gets the wrong protocol.
#
# **Recommendation.** A8 as written — one basis per campaign, serialised once and
# reloaded — is the right decision, and this measurement says *why*: not because
# the coverage number would change much (it moves by at most 2.6 points) but
# because example selection, twin display and any "nearest neighbour" claim are a
# coin flip between the two bases. Serialising the basis is what makes an
# illustration and a measurement talk about the same object.
#
# **What this does not settle.** Which basis to serialise. The pooled one is the
# chain's, but the intro slides were drawn in the paired one, and nothing measured
# here says the pooled basis is the better *illustration* — only that one of them
# has to win.

# %% [markdown]
# ### Part 3 · Why sixteen components
#
# **The problem.** Sixteen is not derived from anything. The variance spectrum has
# no elbow after PC2, so the choice is a frozen analysis constant, and the
# published argument for it is a sweep: `particle-z8-morphology-dimension-sweep-r2`
# truncates the stored scores to d = 2…16 and reports coverage, **domain AUC**
# (the cross-validated area under the ROC curve of a logistic classifier trying to
# tell real from synthetic; 0.5 means indistinguishable) and the **distance
# contrast** — the ratio std/mean of pairwise distances, the quantity that
# collapses when high-dimensional distances concentrate and nearest-neighbour
# queries stop discriminating. The claimed conclusion is a plateau over
# d ∈ [12, 16].
#
# It belongs in this section because it is a basis question, and because it is the
# analysis that inherits the hard-coded q95 named in Part 1. Two caveats travel
# with it, both load-bearing: it consumes the stored scores of
# `particle-z8-v2-real-synthetic-pca-r2`, a **third** basis fitted on the
# white-noise condition alone; and those scores are truncated at 16 columns, so
# the sweep structurally cannot see past the number it is meant to justify.

# %%
SWEEP_SEED = 20260724
sweep_scores = np.load(
    run_dir("particle-z8-v2-real-synthetic-pca-r2") / "pca_scores.npz", allow_pickle=True
)
sweep_real = np.asarray(sweep_scores["real_morphology"], dtype=np.float64)
sweep_synthetic = np.asarray(sweep_scores["synthetic_morphology"], dtype=np.float64)
sweep_real_labels = sweep_scores["real_class"].astype(str)
sweep_synthetic_labels = sweep_scores["synthetic_class"].astype(str)
dimension_run = published("particle-z8-morphology-dimension-sweep-r2")
DIMENSIONS = [entry["dimensions"] for entry in dimension_run["sweep"]]

started = time.time()
sweep_deviation = 0.0
for entry in dimension_run["sweep"]:
    measured = domain_metrics(sweep_real, sweep_synthetic, sweep_real_labels,
                              sweep_synthetic_labels, seed=SWEEP_SEED,
                              dimensions=entry["dimensions"])
    for class_name in CLASS_ORDER:
        sweep_deviation = max(
            sweep_deviation,
            abs(measured[class_name]["real_within_synthetic_self_p95_fraction"]
                - entry["coverage"][class_name]),
            abs(measured[class_name]["domain_classifier_auc_mean"]
                - entry["auc_mean"][class_name]),
        )
assert sweep_deviation == 0.0, f"reproduction drifted by {sweep_deviation:.3e}"
print("reproduces particle-z8-morphology-dimension-sweep-r2 exactly "
      f"(coverage and AUC, max deviation {sweep_deviation:.1e}, {time.time() - started:.0f} s)")

# %% [markdown]
# #### Putting the sweep on the chain's quantile
#
# **The process.** `domain_metrics` cannot be asked for another quantile — 0.95 is
# written into it. `support_coverage` can, but it takes the reference cloud as an
# argument, and `domain_metrics` draws its own internally. Rebuilding **that draw**
# (not the method) is what lets the shipped coverage function score the same cloud
# at another quantile. The gate is that at 0.95 it must return the published
# numbers exactly; otherwise it is scoring something else.

# %%
sweep_draw = domain_reference_draw(sweep_real_labels, sweep_synthetic_labels,
                                   seed=SWEEP_SEED, class_order=CLASS_ORDER)
gate = max(
    abs(support_coverage(sweep_real, sweep_synthetic, sweep_real_labels,
                         sweep_synthetic_labels, sample=sweep_draw, quantile=0.95,
                         dimensions=entry["dimensions"])[class_name][
            "real_within_radius_fraction"] - entry["coverage"][class_name])
    for entry in dimension_run["sweep"] for class_name in CLASS_ORDER
)
assert gate == 0.0, f"the rebuilt draw is not the published one ({gate:.3e})"
print(f"the rebuilt reference draw reproduces the published q95 sweep exactly "
      f"(max deviation {gate:.1e}); recomputing at q80 is therefore licensed\n")

sweep_at_q80 = {}
for dimensions in DIMENSIONS:
    measured = support_coverage(sweep_real, sweep_synthetic, sweep_real_labels,
                                sweep_synthetic_labels, sample=sweep_draw,
                                quantile=0.80, dimensions=dimensions)
    sweep_at_q80[dimensions] = {
        c: measured[c]["real_within_radius_fraction"] for c in CLASS_ORDER
    }
print_dimension_quantiles(dimension_run["sweep"], sweep_at_q80, CLASS_ORDER)
print("\nchain, white-noise condition at q80 (the condition this basis was fitted on): "
      + "  ".join(
          f"{c} {100 * quantile_sweep[0.80]['white_noise_4d'][c]['coverage']:.1f}%"
          for c in CLASS_ORDER))

# %% [markdown]
# **The conclusion.** At the published d = 16 the quantile alone is worth
# **21.1 / 16.1 / 27.3 percentage points**. That is the size of the
# incommensurability the deck carries between two adjoining slides, and it dwarfs
# everything else in this section.
#
# The rest of the comparison is reassuring, and worth stating because it settles
# the relative weight of A7 and A8. Once the sweep is put on q80 it reads
# 65.2 / 57.4 / 56.3 % at d = 16, against 60.8 / 56.5 / 54.5 % for the chain's
# white-noise column — the same condition, a different basis, and agreement to
# within 4.4 / 0.9 / 1.7 points. **The quantile was the whole incommensurability;
# the basis contributes a few points.** A7 is the urgent fix, A8 the structural
# one.
#
# #### Is there actually a plateau?
#
# **The problem.** The published claim is that past d ≈ 12 nothing moves. That
# holds for the AUC — the change from 12 to 16 is under 0.001 in every class — and
# the figure's title generalises it to the solid coverage curve as well. Coverage
# does not cooperate: at 4 µm it climbs 9.5 points between d = 12 and d = 16 at
# q95, and 8.0 points at q80. And because the stored scores stop at 16, the sweep
# cannot ask the obvious next question, which is whether anything settles after 16
# either.
#
# **The process.** It can be asked here. The basis is refitted from the registry
# with the full 101 components instead of 16 — the same standardisation, the same
# balanced draw, the same seed — so the first sixteen columns are the published
# ones and the extension is a strict continuation rather than a new experiment.

# %%
EXTENDED = (12, 14, 16, 20, 24, 28, 32)
sweep_per_class = min(int((conditions.labels == c).sum()) for c in CLASS_ORDER)
wide_scaler, wide_pca, wide_indices = fit_synthetic_pca(
    conditions.features["white_noise_4d"], conditions.labels,
    per_class=sweep_per_class, seed=SWEEP_SEED, components=101,
)
wide_synthetic = wide_pca.transform(wide_scaler.transform(
    conditions.features["white_noise_4d"]))
wide_real = wide_pca.transform(wide_scaler.transform(real.features))
prefix_deviation = float(np.abs(wide_synthetic[:, :16] - sweep_synthetic).max())
print(f"refitted basis reproduces the stored 16 columns to {prefix_deviation:.1e} "
      "(float32 storage)")
print("explained variance: "
      + "   ".join(f"{d} axes {100 * wide_pca.explained_variance_ratio_[:d].sum():.1f}%"
                   for d in (16, 24, 32, 101)))

wide_draw = domain_reference_draw(real.labels, conditions.labels, seed=SWEEP_SEED,
                                  class_order=CLASS_ORDER)
extended = {}
for dimensions in EXTENDED:
    coverage = support_coverage(wide_real, wide_synthetic, real.labels, conditions.labels,
                                sample=wide_draw, quantile=0.80, dimensions=dimensions)
    separability = domain_metrics(wide_real, wide_synthetic, real.labels, conditions.labels,
                                  seed=SWEEP_SEED, dimensions=dimensions)
    extended[dimensions] = {
        "coverage": {c: coverage[c]["real_within_radius_fraction"] for c in CLASS_ORDER},
        "auc": {c: float(separability[c]["domain_classifier_auc_mean"]) for c in CLASS_ORDER},
    }
    print(f"d = {dimensions:>3}   coverage@q80 " + "  ".join(
        f"{c} {100 * extended[dimensions]['coverage'][c]:5.1f}%" for c in CLASS_ORDER)
        + "    AUC " + "  ".join(
            f"{c} {extended[dimensions]['auc'][c]:.3f}" for c in CLASS_ORDER))

# %%
draw_dimension_quantiles(dimension_run, sweep_at_q80, extended, CLASS_ORDER, CLASS_COLOUR)

# %% [markdown]
# **The conclusion: the plateau is local, not terminal.** Past sixteen both
# quantities resume moving: 4 µm coverage climbs from 57.4 % to 66.8 % between
# d = 16 and d = 32, and the domain AUC — which the plateau argument rests on —
# rises from 0.963 to 0.974 at 4 µm, 0.929 to 0.944 at 2 µm and 0.859 to 0.901 at
# 10 µm. Sixteen sits on a flat stretch of the curve; it is not where the curve
# stops.
#
# Two cautions on reading that panel. The 10 µm coverage wobbles by several points
# from one d to the next because only 231 real events carry it, so its trend is
# the weakest of the three; the AUC, computed on a balanced 231-per-domain sample,
# is steadier and moves the same way. And these AUC values (0.86–0.97) are far
# above the 0.66–0.75 the window section reports, because this basis and this
# condition are the white-noise ones — the least realistic arm of the chain.
# Domain AUC is no more commensurable across runs than coverage is.
#
# #### The concentration that does not arrive
#
# **The problem.** The appendix's third argument is the strongest-sounding one:
# distance contrast falls from 0.60 at d = 2 to 0.29 at d = 16, and a √d
# extrapolation puts the raw 101-dimensional descriptor near 0.11, where
# "nearest-neighbour queries stop discriminating". That extrapolation is stated in
# the run's own metrics as "measured, not fitted" — but the quantity extrapolated
# *to* was never measured, because the stored scores stop at 16.
#
# **The process.** PCA is an orthonormal rotation, so keeping all 101 components
# preserves every distance of the standardised descriptor exactly. The endpoint is
# therefore not an extrapolation at all: it can simply be computed, on the same
# refitted basis, with the published contrast definition gated against the
# published sweep first.

# %%
CONTRAST_GRID = (2, 4, 8, 12, 16, 24, 32, 48, 64, 101)
contrast_deviation = max(
    abs(distance_contrast(wide_synthetic, conditions.labels, dimensions=entry["dimensions"],
                          class_order=CLASS_ORDER, seed=SWEEP_SEED)[class_name]
        - entry["contrast"][class_name])
    for entry in dimension_run["sweep"] for class_name in CLASS_ORDER
)
assert contrast_deviation == 0.0, f"contrast drifted by {contrast_deviation:.3e}"
print(f"the contrast definition reproduces the published sweep exactly "
      f"(max deviation {contrast_deviation:.1e})\n")

contrast_curve = {
    dimensions: distance_contrast(wide_synthetic, conditions.labels, dimensions=dimensions,
                                  class_order=CLASS_ORDER, seed=SWEEP_SEED)
    for dimensions in CONTRAST_GRID
}
measured_contrast = [float(np.mean(list(contrast_curve[d].values()))) for d in CONTRAST_GRID]
at_sixteen = measured_contrast[CONTRAST_GRID.index(16)]
predicted_contrast = [at_sixteen * np.sqrt(16 / d) for d in CONTRAST_GRID]
published_extrapolation = dimension_run["contrast_extrapolation_101d"]["class_mean"]
exponent = -np.log(measured_contrast[-1] / at_sixteen) / np.log(101 / 16)
print(f"{'d':>4} {'measured':>10} {'√d prediction':>15}")
for dimensions, measured, prediction in zip(CONTRAST_GRID, measured_contrast, predicted_contrast):
    print(f"{dimensions:>4} {measured:>10.4f} {prediction:>15.4f}")
print(f"\nmeasured at 101 components  {measured_contrast[-1]:.4f}")
print(f"published extrapolation     {published_extrapolation:.4f}  "
      f"— understates the measurement by "
      f"{100 * (measured_contrast[-1] - published_extrapolation) / measured_contrast[-1]:.0f} %")
print(f"the decay exponent between d = 16 and d = 101 is {exponent:.3f}, "
      "not the 0.5 the extrapolation assumes")

# %%
draw_contrast(CONTRAST_GRID, measured_contrast, predicted_contrast, published_extrapolation)

# %% [markdown]
# **The conclusion: the contrast does not collapse.** It falls quickly to about
# d = 12 and then flattens, reaching **0.21** at the full 101 dimensions where the
# appendix predicts 0.11 — the √d law is a bad model of this descriptor, whose
# measured decay exponent is 0.157. The raw descriptor is in essentially the same
# discrimination regime as the 16-axis ruler.
#
# So the honest answer to "why sixteen" is not the one on the slide. Each of the
# three published arguments weakens under measurement: coverage is still moving at
# 16, the AUC plateau is a local flat spot that ends by d = 20, and the
# concentration that was supposed to forbid larger d never arrives. What survives
# is weaker and defensible: **any d in [12, 16] gives the same verdict, the space
# is nowhere near the concentration regime, and 16 is a convention.** It must be
# reported as one — because it moves the 4 µm headline by 8 to 9 points between 12
# and 16, exactly like the quantile, and exactly like the window.
#
# **What is not claimed.** Nothing here says a larger d is better. Coverage rising
# with d while the domain AUC also rises is the same ambiguous signature the
# window section met: more dimensions make real and synthetic *easier* to
# separate, so the extra coverage is not evidence of a better description. This
# section shows only that the published justification for 16 does not survive its
# own extension, not that another number should replace it.

# %% [markdown]
# ### What this section adds, and what it leaves open
#
# Three alignment decisions were under test, and the numbers land differently on
# each:
#
# 1. **q80 (A7) is right, for the wrong stated reason.** The quantile never
#    reorders the conditions, over a range far wider than anyone would defend, so
#    the chain's conclusion is safe. q80 is preferable because q95 measures
#    against a ceiling, not because it is stricter. Every absolute coverage number
#    must carry its quantile.
# 2. **The self-coverage sentence must go.** Under either fair protocol the real
#    events are covered *worse* than the synthetic cloud covers itself. The
#    published percentages remain valid as a ranking and as a distance statement;
#    they do not support the sentence built on them.
# 3. **One serialised basis (A8) is right, and the reason is example selection.**
#    Coverage barely moves between bases; the nearest synthetic neighbour changes
#    identity about half the time, and that is what the introduction figures show.
# 4. **Sixteen is a convention.** Its three published justifications each weaken
#    when the sweep is put on the chain's quantile and extended past 16.
#
# **Limits, named.** The dimension sweep runs on the white-noise basis and the
# white-noise condition, so its absolute coverages are not the chain's and are not
# quoted as such anywhere above. The extension past 16 refits that basis rather
# than reading a stored one, which is a strict continuation but not a published
# run. The fairness control uses the terminal condition only; the same control on
# v3 and v4 gives the same verdict but is not plotted, to keep one figure to one
# idea. Everything here is development data — the sealed test split is never read,
# and no result below authorises a validation claim, a dataset promotion or a
# change to a shipped tool.

# %%
metrics, provenance = quantile_basis_evidence(
    reproduces={
        "particle-z8-v2-coverage-conditions-q80-r1": chain_deviation,
        "particle-z8-v2-paired-asymmetry-pca-r2": intro_deviation,
        "particle-z8-morphology-dimension-sweep-r2": sweep_deviation,
    },
    quantile_sweep=quantile_sweep,
    quantile_grid=QUANTILE_GRID,
    conditions=conditions.order,
    class_order=CLASS_ORDER,
    violations=violations,
    sentence_holds=(above_bar, above_self),
    density=density,
    control=control,
    angles=principal_angles,
    variance=variance_table,
    neighbour=neighbour,
    basis_coverage=basis_coverage,
    sweep_at_q80=sweep_at_q80,
    extended=extended,
    contrast_curve=contrast_curve,
    contrast_grid=CONTRAST_GRID,
    contrast_measured=measured_contrast,
    published_extrapolation=published_extrapolation,
    exponent=exponent,
    seeds={"chain": CHAIN_SEED, "intro": INTRO_SEED, "sweep": SWEEP_SEED},
    datasets=registry.provenance(),
    inputs={
        "real_events_sha256": notebook_evidence.sha256_file(real.events_csv),
        "paired_asymmetry_scores_sha256": notebook_evidence.sha256_file(
            run_dir("particle-z8-v2-paired-asymmetry-pca-r2") / "pca_scores.npz"),
        "real_synthetic_scores_sha256": notebook_evidence.sha256_file(
            run_dir("particle-z8-v2-real-synthetic-pca-r2") / "pca_scores.npz"),
    },
)
print(f"evidence payload serialises, {len(json.dumps(metrics))} bytes of metrics")

try:
    emitted = notebook_evidence.emit_run(
        workspace,
        section="quantile-basis-alignment",
        metrics=metrics,
        provenance=provenance,
        claim_boundary=(
            "Measures how the radius quantile, the PCA basis and the retained "
            "dimension change the z8 morphology coverage claim on development "
            "train/val events. It extends published sweeps below q0.80 and past "
            "d=16, and it tests the deck's self-coverage sentence against a "
            "density-matched control. It reproduces but does not replace any "
            "published run, recommends alignment decisions without adopting them, "
            "changes no shipped tool, and authorizes no validation claim, dataset "
            "promotion or generator decision."
        ),
    )
    print(f"emitted {emitted.name}")
except WorkspaceError as error:
    print(f"no evidence emitted ({error})")


# %% [markdown]
# # What this notebook claims, and what it does not
#
# **Claimed.** On the z8 development corpus, with the generator's five knobs and
# a real noise carrier: the synthetic cloud covers 85–94 % of real events in the
# morphology core at q80; the noise carrier both moves and broadens that cloud
# while the asymmetry coordinate mostly broadens it; a regenerated event finds
# its own parent far above chance and far below the ceiling; and the
# physics-grounded space does that better than the learned one.
#
# **Not claimed.** None of this is validation. Every number here is development
# evidence on train and validation rows; the sealed test split was never opened,
# and no cell in this notebook may read it. Nothing here authorizes promoting a
# dataset, and nothing here says the simulator is indistinguishable from reality
# — the domain classifier still separates the two populations most of the time.
#
# **Known open.** The alignment sections in Part II each end with what their
# evidence does not settle. The largest open item is upstream of all of it: this
# entire chain rests on the z8 detector, which the MAD detector replaces. The
# redo plan is `docs/experiments/2026-08-15/mad-redo-execution-plan.md`, and the
# reason this notebook exists is so that the redo can be checked against
# something executable rather than against a slide.


# %% [markdown]
# # Carrying this to yeast
#
# The bead chain exists to be transferred. Seven things have to be regenerated
# against a yeast model, and each already has a section here that says how:
#
# | To regenerate | Where the method lives |
# |---|---|
# | Observed parameter ranges, boundary-censored | notebook 1 · Cholesky |
# | Class-conditional correlations and the Cholesky factor | notebook 1 · Cholesky |
# | Generated-waveform extremes | notebook 1 · Cholesky |
# | The morphology basis and its PCA | notebook 2 · how a signal becomes a point |
# | Twin pairs and the space they were chosen in | notebook 2 · twins |
# | Matched-policy losses across seeds | notebook 3 · masked learning |
# | Reconstructions on real events | notebook 3 · masked learning |
#
# Two of them will not transfer as written, and it is better to say so here than
# to discover it later. **Labels**: a bead recording carries exactly one class,
# inherited by every event in it — notebook 1 measures that and finds 1,234
# independent label decisions behind 2,073 events. A yeast recording can hold
# more than one cell type at once, so the label channel is not merely small
# there, it is *structurally different*, and the census in notebook 1 does not
# describe it. **Geometry**: the generator's five knobs describe a sphere of
# known diameter crossing a beam. A budding cell is not that shape, and nothing
# in the Cholesky chain currently says what replaces the diameter.
#
# So the honest transfer claim is narrow: the *protocol* carries — reproduce,
# then align, then measure coverage and identity in one space — while the
# *generator* does not, and the *label census* has to be redone from scratch on
# a population where one recording can contain two answers.
