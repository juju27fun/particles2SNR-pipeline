# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # The signal and its generator
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
# Why the project exists, the analytical family and its knobs, and how
# measured events become a generator that can produce new ones.
#
# ## The other notebooks in this series
#
# - [`simulation_chain_2_evaluation`](simulation_chain_2_evaluation.py) — the evaluation
# - [`simulation_chain_3_training_and_alignments`](simulation_chain_3_training_and_alignments.py) — training and alignments


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
# --- problem ---
# Plot helpers for the opening section. Each takes ax=None or axes=None so a
# cell can redraw without rebuilding a figure. No method lives here: these
# functions receive numbers already measured by the section's cells.

import matplotlib.pyplot as plt
import numpy as np


def plot_label_inheritance(census, *, class_colour, class_order, ax=None):
    """Recordings grouped by how many events inherit their single class label.

    `census["recordings_by_event_count"]` maps events-per-recording to a
    per-class count of recordings; `census["events_by_event_count"]` maps the
    same key to the number of events that bucket accounts for.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8.0, 3.6))

    buckets = sorted(int(key) for key in census["recordings_by_event_count"])
    positions = np.arange(len(buckets))
    bottom = np.zeros(len(buckets))
    for class_name in class_order:
        heights = np.asarray([
            census["recordings_by_event_count"][str(bucket)].get(class_name, 0)
            for bucket in buckets
        ], dtype=float)
        ax.bar(positions, heights, 0.62, bottom=bottom,
               color=class_colour[class_name], label=class_name)
        bottom += heights

    for position, bucket in zip(positions, buckets):
        events = census["events_by_event_count"][str(bucket)]
        ax.text(position, bottom[position] + 12, f"{events} events",
                ha="center", fontsize=8, color="#334155")

    shared = census["events_sharing_a_label_fraction"]
    ax.axvspan(0.5, len(buckets) - 0.5, color="#dc2626", alpha=0.06)
    ax.text(len(buckets) - 0.55, bottom.max() * 0.72,
            f"{100 * shared:.0f} % of events\nshare their label",
            ha="right", fontsize=9, color="#dc2626")
    ax.set_xticks(positions, [str(bucket) for bucket in buckets])
    ax.set(xlabel="events detected in the recording",
           ylabel="recordings",
           ylim=(0, bottom.max() * 1.22),
           title="One label per recording, inherited by every event in it")
    ax.legend(frameon=False, ncol=3, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def plot_human_verdicts(verdicts, *, order=None, ax=None):
    """Blind human adjudication of detector candidates, by selection stratum.

    `verdicts` maps a stratum name to {"real_particle": n, "not_particle": n,
    "uncertain": n}.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8.0, 3.4))

    names = list(order or sorted(verdicts))
    positions = np.arange(len(names))
    colours = {"real_particle": "#0f766e",
               "not_particle": "#dc2626",
               "uncertain": "#94a3b8"}
    left = np.zeros(len(names))
    for verdict in ("real_particle", "not_particle", "uncertain"):
        widths = np.asarray([verdicts[name].get(verdict, 0) for name in names],
                            dtype=float)
        if not widths.any():
            continue
        ax.barh(positions, widths, 0.6, left=left,
                color=colours[verdict], label=verdict.replace("_", " "))
        left += widths

    for position, total in zip(positions, left):
        ax.text(total + 0.4, position, f"n={int(total)}", va="center", fontsize=8,
                color="#334155")
    ax.set_yticks(positions, names)
    ax.set(xlabel="candidate events adjudicated",
           xlim=(0, left.max() * 1.18),
           title="What one human confirmed, stratum by stratum")
    ax.legend(frameon=False, ncol=3, fontsize=9, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.invert_yaxis()
    return ax


def plot_masked_reconstruction(signal, mask, model, interpolation, *,
                               sampling_hz, zoom=None, title="", axes=None):
    """One trace as the reconstructor sees it: hidden spans, truth, prediction."""
    if axes is None:
        _, axes = plt.subplots(2, 1, figsize=(11, 4.8))
    signal = np.asarray(signal)
    mask = np.asarray(mask, dtype=bool)
    time_ms = 1000.0 * np.arange(signal.size) / sampling_hz

    edges = np.flatnonzero(np.diff(mask.astype(int)))
    borders = np.r_[0, edges + 1, mask.size]
    spans = [(borders[index], borders[index + 1])
             for index in range(len(borders) - 1)
             if mask[borders[index]]]

    upper, lower = axes
    upper.plot(time_ms, signal, lw=0.4, color="#334155")
    for start, stop in spans:
        upper.axvspan(time_ms[start], time_ms[stop - 1], color="#dc2626", alpha=0.20,
                      lw=0)
    upper.set(ylabel="amplitude", title=title)
    upper.set_xlim(time_ms[0], time_ms[-1])
    upper.spines[["top", "right"]].set_visible(False)

    if zoom is None:
        centre = int(np.argmax(np.abs(signal)))
        zoom = (max(0, centre - 160), min(signal.size, centre + 160))
    start, stop = zoom
    window = slice(start, stop)
    hidden = np.where(mask[window], signal[window], np.nan)
    predicted = np.where(mask[window], np.asarray(model)[window], np.nan)
    naive = np.where(mask[window], np.asarray(interpolation)[window], np.nan)

    lower.plot(time_ms[window], signal[window], lw=0.9, color="#94a3b8",
               label="trace (visible + hidden)")
    lower.plot(time_ms[window], hidden, lw=3.0, color="#334155",
               label="truth on hidden samples")
    lower.plot(time_ms[window], predicted, lw=1.5, color="#2563eb",
               label="encoder+decoder prediction")
    lower.plot(time_ms[window], naive, lw=1.2, ls="--", color="#b45309",
               label="linear interpolation baseline")
    for span_start, span_stop in spans:
        if span_stop > start and span_start < stop:
            lower.axvspan(time_ms[max(span_start, start)],
                          time_ms[min(span_stop, stop) - 1],
                          color="#dc2626", alpha=0.10, lw=0)
    lower.set(xlabel="ms", ylabel="amplitude")
    lower.set_xlim(time_ms[start], time_ms[stop - 1])
    lower.legend(frameon=False, fontsize=8, ncol=2)
    lower.spines[["top", "right"]].set_visible(False)
    return axes


def plot_coverage_scaling(knobs, volume, boundary, *, bins_per_knob,
                          marks=(), ax=None):
    """Cost of covering a k-knob grid against the cost of tracing a boundary."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8.0, 4.0))
    knobs = np.asarray(knobs)
    ax.plot(knobs, volume, "o-", color="#2563eb", lw=2,
            label=f"cover the joint volume  ({bins_per_knob}$^k$)")
    ax.plot(knobs, boundary, "s-", color="#0f766e", lw=2,
            label=f"trace a decision boundary  ({bins_per_knob}$^{{k-1}}$)")
    for value, label, colour in marks:
        ax.axhline(value, ls="--", lw=1.1, color=colour)
        ax.text(knobs[0] + 0.05, value * 1.35, label, fontsize=9, color=colour,
                ha="left")
    ax.set_yscale("log")
    ax.set(xlabel="k — number of independent knobs",
           ylabel=f"examples needed at {bins_per_knob} bins per knob",
           title="Covering a volume costs one factor per knob; a boundary costs one less")
    ax.set_xticks(knobs)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    return ax


# --- latent_sweep ---
"""Plotting helpers for the analytical-family latent sweep.

Every helper takes `ax=None` or `axes=None` so a cell can redraw into an
existing figure. No mathematics lives here: the signals, the envelope and the
display units all come from `p3_ssl.particle_equation_sweeps`, which is the
module the published sweep run used.

The three names imported with a leading underscore are the display helpers of
that module's own figure path. They are module-internal, so this notebook is
coupled to a private surface of `p3_ssl` -- the alternative was to re-derive the
Gaussian envelope and the unit conversions here, which is the reimplementation
the notebook contract forbids.
"""

import matplotlib.pyplot as plt
import numpy as np

from p3_ssl.particle_equation_sweeps import (
    _example_indices_by_sweep_value,
    _single_param_display,
    _single_particle_display_signal,
)

SIGNAL_COLOUR = "#1f77b4"
ENVELOPE_COLOUR = "#d95f02"
SWEEP_CMAP = "magma"


def sweep_display(panel):
    """(values in display units, axis label, short symbol, unit) for one knob."""
    return _single_param_display(panel)


def _time_axis(panel):
    return np.linspace(0.0, panel.window_duration_ms, panel.signal.shape[1])


def _thin_axis(ax):
    ax.tick_params(axis="both", labelsize=6, length=2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#666666")
    ax.grid(False)


def plot_signal_gallery(panels, *, examples_per_panel=5, axes=None):
    """One row per swept knob, five signals spanning that knob's range."""
    if axes is None:
        _, axes = plt.subplots(
            len(panels),
            examples_per_panel,
            figsize=(2.55 * examples_per_panel, 1.5 * len(panels)),
            sharex=True,
            squeeze=False,
        )
    axes = np.asarray(axes).reshape(len(panels), examples_per_panel)
    for row, panel in enumerate(panels):
        indices = _example_indices_by_sweep_value(panel, examples_per_panel)
        values, row_label, symbol, unit = sweep_display(panel)
        t_ms = _time_axis(panel)
        suffix = f" {unit}" if unit else ""
        for column, index in enumerate(indices.tolist()):
            ax = axes[row, column]
            trace, envelope = _single_particle_display_signal(panel, index)
            ax.plot(t_ms, trace, color=SIGNAL_COLOUR, linewidth=0.7)
            ax.plot(t_ms, envelope, color=ENVELOPE_COLOUR, linewidth=1.1, alpha=0.45)
            ax.set_title(f"{symbol} = {values[index]:.2f}{suffix}", fontsize=7, pad=3)
            _thin_axis(ax)
            if column == 0:
                ax.set_ylabel(row_label, fontsize=8)
            if row == len(panels) - 1:
                ax.set_xlabel("time [ms]", fontsize=7)
    return axes


def plot_normalisation_effect(panel, *, axes=None):
    """The lowest and highest value of one knob, before and after z-scoring."""
    values, _, symbol, unit = sweep_display(panel)
    order = np.argsort(panel.color_value)
    picks = (int(order[0]), int(order[-1]))
    if axes is None:
        _, axes = plt.subplots(
            2, 2, figsize=(9.6, 4.2), sharex=True, sharey="row", squeeze=False
        )
    axes = np.asarray(axes).reshape(2, 2)
    t_ms = _time_axis(panel)
    suffix = f" {unit}" if unit else ""
    for column, index in enumerate(picks):
        axes[0, column].plot(
            t_ms, panel.signal[index], color=SIGNAL_COLOUR, linewidth=0.7
        )
        axes[0, column].set_title(
            f"{symbol} = {values[index]:.2f}{suffix}", fontsize=9, pad=4
        )
        axes[1, column].plot(
            t_ms, panel.encoded_signal[index], color="#0f766e", linewidth=0.7
        )
        axes[1, column].set_xlabel("time [ms]", fontsize=8)
        for row in (0, 1):
            _thin_axis(axes[row, column])
    axes[0, 0].set_ylabel("raw signal\n[mV]", fontsize=8)
    axes[1, 0].set_ylabel("model input\n(window z-score)", fontsize=8)
    return axes


def plot_latent_pca(panels, coordinates, variances, *, axes=None, columns=3):
    """A PCA panel per knob, coloured by the value that knob was set to."""
    rows = int(np.ceil(len(panels) / columns))
    if axes is None:
        _, axes = plt.subplots(
            rows, columns, figsize=(4.6 * columns, 3.7 * rows), squeeze=False
        )
    axes = np.asarray(axes).reshape(rows, columns)
    flat = axes.reshape(-1)
    for position, panel in enumerate(panels):
        ax = flat[position]
        values, label, _, unit = sweep_display(panel)
        points = coordinates[panel.key]
        scatter = ax.scatter(
            points[:, 0],
            points[:, 1],
            c=values,
            cmap=SWEEP_CMAP,
            s=5,
            alpha=0.85,
            linewidths=0,
        )
        share = variances[panel.key]
        ax.set_title(f"{label}\nPC1-PC2 hold {share:.0%} of the latent variance", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        _thin_axis(ax)
        bar = ax.figure.colorbar(scatter, ax=ax, fraction=0.046, pad=0.03)
        bar.ax.tick_params(labelsize=6, length=2)
        if unit:
            bar.set_label(unit, fontsize=7)
    for position in range(len(panels), flat.size):
        flat[position].axis("off")
    return axes


def plot_knob_recovery(rows, *, ax=None, floor=-2.6):
    """Ten-neighbour recovery of each knob, in the latent and in the input."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9.2, 4.2))
    labels = [row["label"] for row in rows]
    positions = np.arange(len(rows))
    height = 0.38
    latent = np.clip([row["latent"] for row in rows], floor, None)
    inputs = np.clip([row["input"] for row in rows], floor, None)
    ax.barh(positions + height / 2, latent, height, color="#b45309", label="encoder latent (512-D)")
    ax.barh(positions - height / 2, inputs, height, color="#94a3b8", label="model input (512 samples)")
    ax.axvline(0.0, color="#111111", linewidth=1.0)
    ax.axvline(1.0, color="#2563eb", linewidth=0.9, linestyle="--")
    ax.set_ylim(-1.05, len(rows) - 0.4)
    ax.text(0.0, -0.85, "chance", fontsize=7, color="#111111", ha="center", va="center")
    ax.text(1.0, -0.85, "perfect", fontsize=7, color="#2563eb", ha="center", va="center")
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("cross-validated $R^2$ of the knob, from the 10 nearest neighbours", fontsize=9)
    ax.set_xlim(floor - 0.15, 1.35)
    ax.legend(fontsize=8, loc="lower left", frameon=False)
    _thin_axis(ax)
    ax.tick_params(axis="both", labelsize=8)
    return ax


# --- cholesky ---
"""Plotting helpers for the Cholesky-generator section.

Every helper takes `ax=` or `axes=` so a cell can redraw into an existing
figure without rebuilding one. No method lives here: the matrices, the
populations and the deltas are all computed by installed packages in the
section itself, and these functions only lay ink on them.
"""

import matplotlib.pyplot as plt
import numpy as np

CHOL_LABELS = ("log P₀", "f_D", "log τ", "SNR")
CHOL_CLASS_LABEL = {"2um": "2 µm", "4um": "4 µm", "10um": "10 µm"}


def _triangle(ax, matrix, *, labels, vmax, title, decimals=2, units=""):
    """One lower-triangular heat map, the deck's coolwarm grammar."""
    size = len(labels)
    shown = np.array(matrix, dtype=float).copy()
    mask = np.triu(np.ones((size, size), dtype=bool), k=1)
    shown[mask] = np.nan
    image = ax.imshow(
        np.ma.masked_invalid(shown), cmap="coolwarm", vmin=-vmax, vmax=vmax
    )
    for row in range(size):
        for column in range(row + 1):
            value = matrix[row][column]
            ax.text(
                column,
                row,
                f"{value:+.{decimals}f}".replace("-", "−"),
                ha="center",
                va="center",
                fontsize=9,
                color="white" if abs(value) > 0.62 * vmax else "#111827",
            )
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title(title, fontsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    if units:
        ax.set_xlabel(units, fontsize=8, color="#6b7280")
    return image


def plot_parameter_marginals(values_by_class, colours, *, axes=None):
    """Class-conditional marginals of the four fitted parameters.

    `values_by_class[class][parameter]` is a 1-D array of measured values.
    """
    order = ("amplitude_p0", "frequency_khz", "tau_ms", "snr_db")
    titles = (
        "Amplitude P₀ (a.u.)",
        "Doppler frequency f_D (kHz)",
        "Envelope width τ (ms)",
        "SNR (dB)",
    )
    log_scaled = {"amplitude_p0", "tau_ms"}
    if axes is None:
        _, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    axes = np.ravel(axes)
    for ax, parameter, title in zip(axes, order, titles):
        pooled = np.concatenate(
            [values_by_class[name][parameter] for name in values_by_class]
        )
        if parameter in log_scaled:
            edges = np.geomspace(pooled.min(), pooled.max(), 45)
            ax.set_xscale("log")
        else:
            edges = np.linspace(pooled.min(), pooled.max(), 45)
        for name, series in values_by_class.items():
            ax.hist(
                series[parameter],
                bins=edges,
                density=True,
                histtype="step",
                linewidth=1.6,
                color=colours[name],
                label=CHOL_CLASS_LABEL.get(name, name),
            )
        ax.set_title(title, fontsize=10)
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
    axes[0].set_ylabel("density", fontsize=9)
    axes[0].legend(frameon=False, fontsize=9)
    return axes


def plot_censoring_shift(eligible, censored, *, axes=None):
    """What the boundary-censored events look like next to the retained ones."""
    panels = (
        ("tau_ms", "Fitted envelope width τ (ms)", True),
        ("snr_db", "SNR (dB)", False),
    )
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes = np.ravel(axes)
    for ax, (parameter, title, logscale) in zip(axes, panels):
        keep, drop = eligible[parameter], censored[parameter]
        pooled = np.concatenate([keep, drop])
        if logscale:
            edges = np.geomspace(pooled.min(), pooled.max(), 45)
            ax.set_xscale("log")
        else:
            edges = np.linspace(pooled.min(), pooled.max(), 45)
        ax.hist(
            keep,
            bins=edges,
            density=True,
            color="#94a3b8",
            alpha=0.75,
            label=f"retained (n = {keep.size:,})",
        )
        ax.hist(
            drop,
            bins=edges,
            density=True,
            histtype="step",
            linewidth=2.0,
            color="#b45309",
            label=f"boundary-censored (n = {drop.size:,})",
        )
        ax.axvline(np.median(keep), color="#334155", linewidth=1.0, linestyle="--")
        ax.axvline(np.median(drop), color="#b45309", linewidth=1.0, linestyle="--")
        ax.set_title(title, fontsize=10)
        ax.set_yticks([])
        ax.legend(frameon=False, fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
    return axes


def plot_correlation_triangles(matrices, counts, populations, *, axes=None):
    """One Pearson lower triangle per class, in the transformed coordinates."""
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    axes = np.ravel(axes)
    image = None
    for ax, name in zip(axes, matrices):
        image = _triangle(
            ax,
            matrices[name],
            labels=CHOL_LABELS,
            vmax=1.0,
            title=(
                f"{CHOL_CLASS_LABEL.get(name, name)} · {populations[name]}"
                f" · n = {counts[name]:,}"
            ),
        )
    if image is not None and axes[-1].figure is not None:
        axes[-1].figure.colorbar(
            image, ax=list(axes), fraction=0.022, pad=0.02, label="Pearson r"
        )
    return axes


def plot_delta_triangles(deltas, counts, *, scale=0.15, axes=None):
    """Realised minus target Pearson r, saturating at the deck's +/- 0.15."""
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    axes = np.ravel(axes)
    image = None
    for ax, name in zip(axes, deltas):
        worst = np.max(np.abs(np.array(deltas[name]) - np.diag(np.diag(deltas[name]))))
        image = _triangle(
            ax,
            deltas[name],
            labels=CHOL_LABELS,
            vmax=scale,
            title=(
                f"{CHOL_CLASS_LABEL.get(name, name)} · n = {counts[name]:,}"
                f" · max |Δ| = {worst:.3f}"
            ),
        )
    if image is not None and axes[-1].figure is not None:
        axes[-1].figure.colorbar(
            image,
            ax=list(axes),
            fraction=0.022,
            pad=0.02,
            label="realised − target r",
        )
    return axes


def plot_signal_gallery_cholesky(selections, records, signals, sampling_hz, *, axes=None):
    """One generated waveform per (class, role), with its Gaussian envelope.

    `selections[class]` is the shipped `select_gallery_indices` output: a list
    of (role label, index) pairs into `records` and `signals`.
    """
    names = list(selections)
    width = max(len(entries) for entries in selections.values())
    if axes is None:
        _, axes = plt.subplots(
            len(names), width, figsize=(19, 8.4), sharex=True, constrained_layout=True
        )
    axes = np.asarray(axes).reshape(len(names), width)
    length = signals.shape[1]
    time_ms = (
        (np.arange(length, dtype=np.float64) - (length - 1) / 2.0) / sampling_hz * 1000.0
    )
    for row, name in enumerate(names):
        for column, (role, index) in enumerate(selections[name]):
            ax = axes[row, column]
            record = records[index]
            envelope = record["amplitude_p0"] * np.exp(
                -0.5 * np.square(time_ms / record["tau_ms"])
            )
            ax.plot(time_ms, signals[index], color="#2563eb", linewidth=0.55)
            ax.plot(time_ms, envelope, color="#f97316", linewidth=1.0)
            ax.plot(time_ms, -envelope, color="#f97316", linewidth=1.0)
            ax.text(
                0.02,
                0.97,
                (
                    f"P₀ {record['amplitude_p0']:.3g}\n"
                    f"f_D {record['frequency_khz']:.1f} kHz\n"
                    f"τ {record['tau_ms']:.3f} ms\n"
                    f"SNR {record['snr_db']:+.1f} dB"
                ),
                transform=ax.transAxes,
                va="top",
                fontsize=7,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )
            if row == 0:
                ax.set_title(role, fontsize=9)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.12)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
        axes[row, 0].set_ylabel(
            CHOL_CLASS_LABEL.get(name, name), fontsize=10, fontweight="bold"
        )
    for ax in axes[-1]:
        ax.set_xlabel("time from t₀ (ms)", fontsize=8)
    return axes


def plot_dependence_scatter(panels, *, axes=None):
    """(log P0, SNR) for one class: measured, independent draw, Cholesky draw."""
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(13, 3.9), sharex=True, sharey=True)
    axes = np.ravel(axes)
    for ax, (title, x_values, y_values, colour) in zip(axes, panels):
        correlation = float(np.corrcoef(x_values, y_values)[0, 1])
        ax.scatter(x_values, y_values, s=6, alpha=0.35, color=colour, linewidths=0)
        ax.set_title(f"{title}\nr = {correlation:+.3f}", fontsize=10)
        ax.set_xlabel("log P₀", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("SNR (dB)", fontsize=9)
    return axes


# --- asymmetry ---
"""Plotting helpers for the waveform-asymmetry section.

Every helper takes ``ax=None`` or ``axes=None`` so a cell can redraw into an
existing figure without rebuilding it. No mathematics lives here: the helpers
receive arrays that the section computed from installed packages.
"""

import matplotlib.pyplot as plt
import numpy as np

ASYMMETRY_CLASS_ORDER = ("2um", "4um", "10um")
ASYMMETRY_CLASS_COLOUR = {"2um": "#2563eb", "4um": "#0f766e", "10um": "#b45309"}
ASYMMETRY_CLASS_LABEL = {"2um": "2 µm", "4um": "4 µm", "10um": "10 µm"}
NULL_COLOUR = "#94a3b8"


def _class_axes(axes, count=3, figsize=(13.5, 4.0)):
    if axes is None:
        _, axes = plt.subplots(1, count, figsize=figsize, constrained_layout=True)
    return np.atleast_1d(axes)


def plot_envelope_skew(real, null, *, axes=None, limit=1.2):
    """Model-free envelope skew: real events against the symmetric generator.

    ``real`` and ``null`` map class name to a 1-D array of the energy-weighted
    envelope skew of each event. The null is the 4-D generator, whose envelope
    is symmetric by construction, so its spread is what the statistic returns
    on unskewed pulses at the same signal-to-noise.
    """
    axes = _class_axes(axes)
    edges = np.linspace(-limit, limit, 49)
    for axis, name in zip(axes, ASYMMETRY_CLASS_ORDER):
        observed, reference = real[name], null[name]
        axis.hist(np.clip(reference, -limit, limit), bins=edges, density=True,
                  color=NULL_COLOUR, alpha=0.55,
                  label=f"symmetric 4-D generator (n={reference.size})")
        axis.hist(np.clip(observed, -limit, limit), bins=edges, density=True,
                  histtype="step", linewidth=2.0, color=ASYMMETRY_CLASS_COLOUR[name],
                  label=f"real events (n={observed.size})")
        band = float(np.quantile(np.abs(reference), 0.95))
        for edge in (-band, band):
            axis.axvline(edge, color="#0f172a", linestyle=":", linewidth=1.2)
        beyond = float(np.mean(np.abs(observed) > band))
        axis.set_title(
            f"{ASYMMETRY_CLASS_LABEL[name]} · sd {observed.std(ddof=1):.3f} "
            f"vs {reference.std(ddof=1):.3f}",
            color=ASYMMETRY_CLASS_COLOUR[name], fontweight="bold",
        )
        axis.set_xlabel("envelope skew γ")
        axis.set_ylabel("density")
        axis.grid(alpha=0.18)
        axis.text(0.03, 0.94, f"{100 * beyond:.0f} % beyond ±{band:.2f}",
                  transform=axis.transAxes, fontsize=9, va="top")
        axis.legend(frameon=False, fontsize=7.5, loc="upper right")
    return axes


def plot_event_fit(time_us, observed, skewed, symmetric, *, label, axes=None):
    """One real event with the free fit and the symmetry-constrained fit."""
    if axes is None:
        _, axes = plt.subplots(2, 1, figsize=(12.0, 6.4), sharex=True,
                               gridspec_kw={"height_ratios": (2.0, 1.0)},
                               constrained_layout=True)
    axes = np.atleast_1d(axes)
    def root_mean_square(values):
        return float(np.sqrt(np.mean(np.square(values))))

    axes[0].plot(time_us, observed, color="#cbd5e1", linewidth=0.9, label="band-passed event")
    axes[0].plot(time_us, symmetric, color="#0f172a", linewidth=1.3, linestyle="--",
                 label="best symmetric model (a ≡ 0)")
    axes[0].plot(time_us, skewed, color="#0ea5e9", linewidth=1.5, label="best skewed model (â free)")
    axes[0].set_ylabel("amplitude")
    axes[0].set_title(label, fontweight="bold")
    axes[0].legend(frameon=False, fontsize=9, ncol=3)
    axes[0].grid(alpha=0.18)
    axes[1].plot(time_us, observed - symmetric, color="#0f172a", linewidth=0.9,
                 label=f"residual, symmetric model · rms {root_mean_square(observed - symmetric):.3f}")
    axes[1].plot(time_us, observed - skewed, color="#0ea5e9", linewidth=0.9,
                 label=f"residual, skewed model · rms {root_mean_square(observed - skewed):.3f}")
    axes[1].axhline(0.0, color="#94a3b8", linewidth=0.8)
    axes[1].set_xlabel("time from event centre (µs)")
    axes[1].set_ylabel("residual")
    axes[1].legend(frameon=False, fontsize=9, ncol=2)
    axes[1].grid(alpha=0.18)
    return axes


def plot_recovery(rows_by_class, *, axes=None):
    """Injected against recovered skew, one panel per class, coloured by SNR.

    ``rows_by_class`` maps class name to ``(truth, estimate, snr_db, r_squared)``.
    """
    axes = _class_axes(axes, figsize=(14.0, 4.4))
    image = None
    for axis, name in zip(axes, ASYMMETRY_CLASS_ORDER):
        truth, estimate, snr, determination = rows_by_class[name]
        image = axis.scatter(truth, estimate, c=snr, s=15, cmap="viridis",
                             vmin=-20.0, vmax=25.0, linewidths=0)
        axis.plot((-0.8, 0.8), (-0.8, 0.8), linestyle="--", color="#0f172a", linewidth=1.1)
        axis.set(xlim=(-0.85, 0.85), ylim=(-0.85, 0.85),
                 xlabel="injected a", ylabel="recovered â")
        axis.set_title(f"{ASYMMETRY_CLASS_LABEL[name]} · R² = {determination:.2f}",
                       color=ASYMMETRY_CLASS_COLOUR[name], fontweight="bold")
        axis.grid(alpha=0.18)
    if image is not None:
        bar = axes[0].figure.colorbar(image, ax=list(axes), shrink=0.85)
        bar.set_label("effective SNR (dB)")
    return axes


def plot_real_targets(real, noise, *, support=None, axes=None, limit=0.85):
    """Real per-event skew against the estimator's own error distribution.

    ``real`` maps class name to the accepted real â; ``noise`` maps class name
    to the estimator's signed error on domain-aligned injections. Anything the
    two distributions do not share is skew the events actually carry.
    """
    axes = _class_axes(axes, figsize=(13.5, 4.0))
    edges = np.linspace(-limit, limit, 45)
    for axis, name in zip(axes, ASYMMETRY_CLASS_ORDER):
        observed, error = real[name], noise[name]
        axis.hist(np.clip(error, -limit, limit), bins=edges, density=True,
                  color=NULL_COLOUR, alpha=0.55,
                  label=f"estimator error (n={error.size})")
        axis.hist(np.clip(observed, -limit, limit), bins=edges, density=True,
                  histtype="step", linewidth=2.0, color=ASYMMETRY_CLASS_COLOUR[name],
                  label=f"real â (n={observed.size})")
        if support is not None:
            for edge in support[name]:
                axis.axvline(edge, color=ASYMMETRY_CLASS_COLOUR[name], linewidth=1.4,
                             linestyle=":")
        axis.set_title(
            f"{ASYMMETRY_CLASS_LABEL[name]} · sd {observed.std(ddof=1):.3f} "
            f"vs {error.std(ddof=1):.3f}",
            color=ASYMMETRY_CLASS_COLOUR[name], fontweight="bold",
        )
        axis.set_xlabel("asymmetry a")
        axis.set_ylabel("density")
        axis.grid(alpha=0.18)
        axis.legend(frameon=False, fontsize=7.5, loc="upper right")
    return axes


def plot_support_bootstrap(draws, observed, *, ax=None):
    """What support 40 events buy, drawn from classes that have thousands.

    ``draws`` maps a donor class name to the half-range of many random
    40-event subsamples; ``observed`` is the 10 µm half-range actually seen.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9.0, 4.2), constrained_layout=True)
    edges = np.linspace(0.2, 0.85, 60)
    for name, values in draws.items():
        share = float(np.mean(values <= observed))
        ax.hist(values, bins=edges, density=True, histtype="step", linewidth=2.0,
                color=ASYMMETRY_CLASS_COLOUR[name],
                label=f"{ASYMMETRY_CLASS_LABEL[name]} subsampled to n=40 · "
                      f"P(≤ observed) = {share:.3f}")
    ax.axvline(observed, color="#b45309", linewidth=2.4)
    ax.text(observed, ax.get_ylim()[1] * 0.55, f"  10 µm actual: {observed:.3f}",
            color="#b45309", fontweight="bold", fontsize=10, va="center")
    ax.set_xlabel("half-range of |â| observed in the sample")
    ax.set_ylabel("density")
    ax.set_title("The support a 40-event anchor set can measure", fontweight="bold")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    return ax


def plot_delta_triptych(deltas, zscores, labels, counts, *, axes=None, scale=0.15):
    """Simulated − real Pearson r, lower triangle, with the anchor |z| shown.

    The colour carries the delta exactly as the deck figure does; the number in
    brackets on the asymmetry row is how many anchor standard errors that delta
    is worth, which is what the agreement claim actually rests on.
    """
    axes = _class_axes(axes, figsize=(15.0, 5.2))
    count = len(labels)
    image = None
    for axis, name in zip(axes, ASYMMETRY_CLASS_ORDER):
        matrix = deltas[name]
        zmatrix = zscores[name]
        masked = np.ma.masked_where(np.triu(np.ones_like(matrix, dtype=bool), k=1), matrix)
        image = axis.imshow(masked, cmap="coolwarm", vmin=-scale, vmax=scale,
                            interpolation="nearest")
        for row in range(count):
            for column in range(row + 1):
                value = matrix[row, column]
                text = f"{0.0 if abs(value) < 0.005 else value:.2f}"
                if row == count - 1 and column < row:
                    text += f"\n[{abs(zmatrix[row, column]):.1f}σ]"
                axis.text(column, row, text, ha="center", va="center", fontsize=8.5,
                          fontweight="bold",
                          color="white" if abs(value) >= 0.55 * scale else "#0f172a")
        axis.set_xticks(range(count), labels, rotation=30, ha="right", fontsize=8)
        axis.set_yticks(range(count), labels, fontsize=8)
        axis.set_title(f"{ASYMMETRY_CLASS_LABEL[name]} · {counts[name]} real anchors",
                       color=ASYMMETRY_CLASS_COLOUR[name], fontweight="bold")
    if image is not None:
        bar = axes[0].figure.colorbar(image, ax=list(axes), shrink=0.8)
        bar.set_label("simulated − real Pearson r")
    return axes


# %% [markdown]
# ## The problem — why any of this exists
#
# Before the simulation chain there is a question that has nothing to do with
# simulation: **what can actually be learned from this data?** A recording of
# an acoustic flow cytometer is a one-dimensional trace, sampled at 2 MHz, in
# which particles crossing the beam leave short oscillating bursts. The obvious
# supervised framing — label each burst with the particle that made it, train a
# classifier — collapses on contact with the corpus, and the three subsections
# below show why, with numbers rather than assertions:
#
# 1. **the labels are properties of the recording, not of the event**, so the
#    per-event supervision the framing assumes does not exist;
# 2. **self-supervised learning (SSL)** replaces the label with the signal
#    itself: hide part of a trace, predict it, keep the encoder;
# 3. **that reconstruction task is far hungrier for examples than
#    classification**, by an amount this section counts on both sides — which
#    is what forces the project into simulation, and therefore into everything
#    the rest of the notebook measures.
#
# A glossary of every acronym and symbol used in this notebook closes the
# section.

# %% [markdown]
# ### 1. The label problem — measured
#
# The real corpus is a set of 8.192 ms recordings, one file per acquisition,
# named after the bead population that was flowing: `HFocusing_5_10_2um2_0_1000.npy`
# is a 2 µm acquisition. That filename **is** the label. Nobody looked at the
# trace and decided it contained a 2 µm bead; a suspension of 2 µm beads was
# put through the instrument and the resulting file inherited the name.
#
# Events inside the file are then found by an automatic detector (`dual-clean`
# peak evidence, then the `z8` selection policy — both defined in the glossary),
# and each detected event inherits the recording's label. The first cell counts
# what that inheritance actually costs.

# %%
import collections
import math

real_key = "particles2snr-fbase-dual-clean-z8-events-3class-plus-unclear-development@v2"
real_root = dataset_root(real_key)

with (real_root / "events.csv").open(newline="") as handle:
    event_rows = list(csv.DictReader(handle))
if any(row["split"] == "test" for row in event_rows):
    raise PermissionError("sealed test rows are forbidden")

summary = json.loads((real_root / "dataset_summary.json").read_text())
contract = json.loads((real_root / "input_contract.json").read_text())

physical_rows = [row for row in event_rows if row["class_name"] in CLASS_ORDER]
recording_class = {row["source_filename"]: row["physical_source_class"]
                   for row in event_rows}
events_per_recording = collections.Counter(row["source_filename"]
                                           for row in event_rows)
classes_per_recording = collections.defaultdict(set)
for row in event_rows:
    classes_per_recording[row["source_filename"]].add(row["physical_source_class"])

distinct_classes = max(len(values) for values in classes_per_recording.values())
if distinct_classes != 1:
    raise AssertionError("a recording carries more than one physical class")

alone = sum(1 for count in events_per_recording.values() if count == 1)
shared_fraction = 1.0 - alone / len(event_rows)

print(f"development recordings          {summary['development_signal_count']}")
print(f"recordings carrying an event    {len(events_per_recording)}"
      f"  ({100 * len(events_per_recording) / summary['development_signal_count']:.1f} %)")
print(f"annotated events                {len(event_rows)}"
      f"   ({len(physical_rows)} in the three physical classes,"
      f" {len(event_rows) - len(physical_rows)} demoted to 'unclear')")
print(f"distinct physical classes in one recording   {distinct_classes}")
print(f"events sharing their label with a sibling    {100 * shared_fraction:.1f} %")
print(f"recording duration              {contract['source_length']} samples"
      f" = {1000 * contract['source_length'] / SAMPLING_HZ:.3f} ms"
      f" at {contract['sampling_frequency_hz'] / 1e6:.0f} MHz")

# %% [markdown]
# Three numbers decide the framing.
#
# **2,073** events across the three physical classes — that is the entire real
# *development* corpus of this project, train and val together, and every later
# comparison is against it. The release is development-only by construction; no
# sealed test events exist in this lineage to fall back on.
#
# **One** distinct class per recording, checked rather than assumed: no
# recording in the corpus contains events of two classes, because the class is
# not a property the detector measures. It is the name of the file.
#
# **70 %** of the annotated events share their label with at least one sibling
# found in the same recording. Their labels are not independent observations;
# they are 1,234 recording-level decisions replicated across 2,194 events. A
# classifier trained on them is partly learning to recognise the acquisition,
# which is the failure the deck states and this cell quantifies.
#
# The remaining 121 events are labelled `unclear`. That too is a rule, not a
# judgement: any event whose fitted SNR falls below −10 dB is relabelled,
# whatever it looks like.

# %%
recordings_by_event_count = collections.defaultdict(collections.Counter)
events_by_event_count = collections.Counter()
for filename, count in events_per_recording.items():
    recordings_by_event_count[str(count)][recording_class[filename]] += 1
    events_by_event_count[str(count)] += count

label_census = {
    "recordings_by_event_count": {key: dict(value)
                                  for key, value in recordings_by_event_count.items()},
    "events_by_event_count": dict(events_by_event_count),
    "events_sharing_a_label_fraction": shared_fraction,
}
plot_label_inheritance(label_census, class_colour=CLASS_COLOUR,
                       class_order=CLASS_ORDER)

# %% [markdown]
# #### How much information is in the label channel, exactly
#
# The label channel can be bounded rather than described. There are 1,234
# independent labels (one per recording that produced an event), drawn from
# three classes with the observed frequencies. The entropy of that sequence is
# an upper bound on everything supervision can ever extract from the real
# corpus.

# %%
recording_classes = collections.Counter(recording_class.values())
total_recordings = sum(recording_classes.values())
entropy_bits = -sum(
    (count / total_recordings) * math.log2(count / total_recordings)
    for count in recording_classes.values()
)
label_bits = total_recordings * entropy_bits

print(f"independent labels              {total_recordings}")
print(f"class frequencies               "
      + "  ".join(f"{name} {recording_classes[name]}" for name in CLASS_ORDER))
print(f"entropy per label               {entropy_bits:.4f} bits"
      f"  (ceiling log2(3) = {math.log2(3):.4f})")
print(f"whole real label channel        {label_bits:.0f} bits"
      f" = {label_bits / 8:.0f} bytes")

# %% [markdown]
# **231 bytes.** Everything a supervised classifier could ever be told about
# the real corpus, by these labels, fits in a quarter of a kilobyte. That is
# the size of the supervision this project is asked to work from — and it is an
# upper bound, since it credits every label as an independent observation of
# its event, which the previous cell showed is false.

# %% [markdown]
# #### What a human actually confirmed
#
# The obvious objection: a person could label the events. One did. In July 2026
# a blind review queue was built — the reviewer sees the trace and a fixed set
# of evidence fields, not the detector's verdict — and 120 candidate events,
# one per recording, were adjudicated by hand.
#
# **Sixty of those 120 sit on the sealed test split.** They are excluded here
# and their decisions are never read; only the development half is used, and
# the cell raises if a sealed row reaches the population. That halves the
# evidence and is worth stating plainly rather than quietly using the full
# tally that the session summary reports.

# %%
review_root = run_dir("dual-clean-tuning-review-120-v2-jlb")
queue = json.loads((review_root / "review_queue_snapshot.json").read_text())

development = [candidate for candidate in queue["candidates"]
               if candidate["partition"] == "development"]
if any(candidate["source_split"] == "test" or candidate["filtered_split"] == "test"
       for candidate in development):
    raise PermissionError("sealed test rows are forbidden")
sealed_excluded = len(queue["candidates"]) - len(development)

axis_of = {candidate["candidate_id"]: candidate["blind_evidence"]["failure_axis"]
           for candidate in development}
with (review_root / "current_decisions.csv").open(newline="") as handle:
    decisions = [row for row in csv.DictReader(handle)
                 if row["candidate_id"] in axis_of]

verdicts = collections.defaultdict(collections.Counter)
for row in decisions:
    verdicts[axis_of[row["candidate_id"]]][row["existence"]] += 1
overall = collections.Counter(row["existence"] for row in decisions)
classed = sum(1 for row in decisions if row["estimated_class"])
reviewers = {row["reviewer"] for row in decisions}

print(f"adjudicated on development      {len(decisions)}"
      f"   (sealed test candidates excluded: {sealed_excluded})")
print(f"reviewers                       {', '.join(sorted(reviewers))}")
print(f"verdicts                        "
      + "  ".join(f"{name} {count}" for name, count in overall.most_common()))
print(f"rejected as not a particle      "
      f"{100 * overall['not_particle'] / len(decisions):.0f} %")
print(f"decisions carrying a class      {classed} / {len(decisions)}")

# %%
plot_human_verdicts(verdicts,
                    order=["eligible", "low_filtered", "low_clean",
                           "association", "saturation"])

# %% [markdown]
# Two things are visible, and only one of them was expected.
#
# The reviewer rejected **27 of 60** detector proposals as not a particle at
# all — but read the strata, not the total. The queue was quota-sampled to
# stress the policy, and the split is sharp: on `eligible`, the only stratum
# resembling the general population, **12 of 13** were confirmed. On the strata
# built from the policy's own failure modes it collapses — `low_filtered` 7 of
# 20, `saturation` 1 of 6. So 45 % is not a corpus false-positive rate; it is
# the measured price of the thresholds, and it says that near its own limits
# the detector proposes things a human will not call particles.
#
# The second is the one that settles the section: **the class field is empty on
# all 60 decisions.** The interface offered it. A human able to say "yes, that
# is a particle" did not, on a single event, say which particle. Event-level
# class supervision does not exist in this corpus, and it is not withheld by
# accident — it is not recoverable from the trace by the person who built the
# detector.
#
# What is *not* claimed: this is one reviewer, one session, on a deliberately
# hard 60-event sample of development recordings. It bounds nothing about
# accuracy. It establishes that per-event class labels were not produced, which
# is all the argument needs.

# %% [markdown]
# The label-channel census above belongs to no existing run — the deck asserts
# the inheritance, the dataset summary counts events, but nothing measures how
# many independent labels those events actually carry. So this subsection emits
# it as manifested evidence. That happens only under
# `workspace notebooks execute`; a live kernel prints the refusal and moves on.

# %%
census_metrics = {
    "schema_version": 1,
    "analysis": "real-label-channel-census",
    "population": {
        "dataset": real_key,
        "selection": "development train and val rows, all four label values",
        "recordings_in_release": summary["development_signal_count"],
        "recordings_with_events": len(events_per_recording),
        "events": len(event_rows),
        "events_in_physical_classes": len(physical_rows),
    },
    "distinct_physical_classes_per_recording": distinct_classes,
    "recordings_by_event_count": {
        str(count): int(recordings)
        for count, recordings in sorted(
            collections.Counter(events_per_recording.values()).items()
        )
    },
    "events_sharing_a_label_fraction": shared_fraction,
    "label_channel": {
        "independent_labels": total_recordings,
        "entropy_bits_per_label": entropy_bits,
        "total_bits": label_bits,
        "class_counts": dict(recording_classes),
    },
}

try:
    emitted = notebook_evidence.emit_run(
        workspace,
        section="label-channel-census",
        metrics=census_metrics,
        provenance={
            "datasets": dataset_provenance(),
            "inputs": {
                "events_csv_sha256": notebook_evidence.sha256_file(
                    real_root / "events.csv"
                ),
            },
            "parameters": {
                "grouping_key": "source_filename",
                "class_field": "physical_source_class",
            },
            "metric_definitions": {
                "events_sharing_a_label_fraction": (
                    "fraction of annotated events found in a recording that "
                    "produced at least two events, so that their label is a "
                    "single recording-level decision replicated"
                ),
                "entropy_bits_per_label": (
                    "Shannon entropy of the physical class of the recordings "
                    "that produced at least one event, in bits"
                ),
            },
        },
        claim_boundary=(
            "Counts the structure of the real label channel on the z8 "
            "development event table: how many recordings, how many events "
            "inherit each recording label, and the entropy of that label "
            "sequence. It bounds the supervision available; it measures no "
            "classifier, no detector accuracy, and authorizes no promotion."
        ),
    )
    print(f"emitted {emitted.name}")
except WorkspaceError as error:
    print(f"no evidence emitted ({error})")

# %% [markdown]
# ### 2. Self-supervised learning, mechanically
#
# If the label cannot be trusted but the signal can, train on the signal.
#
# **Masked reconstruction** is the concrete form used here. Take a 4,096-sample
# window. Hide a quarter of it. Pass what remains through an **encoder**, which
# compresses the window into a latent vector *z*; pass *z* through a
# **decoder**, which writes samples back out. The loss reads **only the hidden
# samples**: the network is scored on what it could not see. No label enters
# anywhere.
#
# At inference the decoder is thrown away. The deliverable is the frozen
# encoder and its latent space — what later sections use for probes, retrieval
# and, eventually, classification.
#
# The deck draws that as a symmetric pair of trapezoids. The next cell
# instantiates the actual model from the training run's own configuration
# (verified byte-identical by hash) and counts what each half costs.

# %%
import hashlib  # noqa: E402  (kept beside the section that needs it)

from p3_ssl.bead_ssl import make_model  # noqa: E402
from p3_ssl.config import REPOSITORY_ROOT as P3_SSL_ROOT, load_config  # noqa: E402

SSL_RUN = "bead-ssl-v2-p25-full-s42-e30-matched-r1"
ssl_root = run_dir(SSL_RUN)
ssl_run = json.loads((ssl_root / "run.json").read_text())
ssl_metrics = json.loads((ssl_root / "metrics.json").read_text())

config_relative = "configs/bead_ssl_z8_v5_v2.yaml"
config_sha = hashlib.sha256((P3_SSL_ROOT / config_relative).read_bytes()).hexdigest()
if config_sha != ssl_run["source_sha256"]["config"]:
    raise AssertionError("the local config is not the one this run trained with")

ssl_config = load_config(config_relative)
model = make_model(ssl_config)
total_parameters = sum(parameter.numel() for parameter in model.parameters())
decoder_parameters = sum(parameter.numel()
                         for parameter in model.reconstruction_head.parameters())

print(f"config sha256 matches {SSL_RUN}")
print(f"window                          {ssl_config['data']['input_length']} samples"
      f" = {1000 * ssl_config['data']['input_length'] / SAMPLING_HZ:.3f} ms")
print(f"tokens                          {model.n_tokens}"
      f" patches of {ssl_config['model']['patch_size']} samples")
print(f"parameters                      {total_parameters:,}")
print(f"  encoder, kept at inference    {total_parameters - decoder_parameters:,}"
      f"  ({100 * (1 - decoder_parameters / total_parameters):.1f} %)")
print(f"  decoder, dropped              {decoder_parameters:,}"
      f"  ({100 * decoder_parameters / total_parameters:.1f} %)")

# %% [markdown]
# The schema is right about the direction and wrong about the proportions.
# "Training pays for a decoder" is worth **1,744 parameters out of 340,528** —
# a LayerNorm and one linear map from the 96-dimensional token back to its 16
# samples. Ninety-nine and a half percent of the model is the part that
# survives. The expensive thing about masked reconstruction is not the decoder;
# it is the data, which is the subject of subsection 3.
#
# Now the mechanism on a real trace. The run stored six reconstruction
# examples; they are validation-split 2 µm events, and the cell checks that
# before touching them.

# %%
examples = np.load(ssl_root / "real_reconstruction_examples.npz", allow_pickle=False)
event_by_id = {row["event_id"]: row for row in event_rows}
example_rows = [event_by_id[str(sample_id)] for sample_id in examples["sample_id"]]
if any(row["split"] == "test" for row in example_rows):
    raise PermissionError("sealed test rows are forbidden")

masked_per_trace = examples["mask"].sum(axis=1)
if len(set(masked_per_trace.tolist())) != 1:
    raise AssertionError("the P25 budget is not constant across examples")
hidden_per_trace = int(masked_per_trace[0])

spans = [int(np.count_nonzero(np.diff(row.astype(int)) == 1)) + int(row[0])
         for row in examples["mask"]]
budget_from_metrics = (ssl_metrics["real_validation"]["model"]["masked_points"]
                       / ssl_metrics["counts"]["real_validation"])
if hidden_per_trace != budget_from_metrics:
    raise AssertionError("stored masks disagree with the run's masked-point budget")

print(f"examples                        {len(example_rows)} "
      f"({', '.join(sorted({row['class_name'] for row in example_rows}))},"
      f" split {', '.join(sorted({row['split'] for row in example_rows}))})")
print(f"hidden per trace                {hidden_per_trace} of "
      f"{examples['signal'].shape[1]} samples"
      f" = {100 * hidden_per_trace / examples['signal'].shape[1]:.1f} %")
print(f"mask geometry                   {set(spans)} disjoint patches of "
      f"{hidden_per_trace // spans[0]} samples")
print(f"reproduces the run's budget     "
      f"{ssl_metrics['real_validation']['model']['masked_points']:.0f} masked points"
      f" / {ssl_metrics['counts']['real_validation']} traces"
      f" = {budget_from_metrics:.0f}")

# %%
example_index = 0
example_row = example_rows[example_index]
plot_masked_reconstruction(
    examples["signal"][example_index],
    examples["mask"][example_index],
    examples["model"][example_index],
    examples["interpolation"][example_index],
    sampling_hz=SAMPLING_HZ,
    title=(f"{example_row['event_id']} · {example_row['class_name']} ·"
           f" SNR {float(example_row['snr_db']):.1f} dB ·"
           f" {hidden_per_trace} samples hidden in {spans[example_index]} patches"),
)

# %%
hidden = examples["mask"].astype(bool)
model_mse = np.asarray([
    float(np.mean((examples["signal"][index] - examples["model"][index])[hidden[index]] ** 2))
    for index in range(len(example_rows))
])
interpolation_mse = np.asarray([
    float(np.mean((examples["signal"][index]
                   - examples["interpolation"][index])[hidden[index]] ** 2))
    for index in range(len(example_rows))
])
published_model = ssl_metrics["real_validation"]["model"]["masked_mse"]
published_interpolation = ssl_metrics["real_validation"]["interpolation"]["masked_mse"]

print(f"stored examples, masked MSE     model {model_mse.min():.2e}–{model_mse.max():.2e}"
      f"   interpolation {interpolation_mse.min():.2e}–{interpolation_mse.max():.2e}")
print(f"published over {ssl_metrics['counts']['real_validation']} traces"
      f"       model {published_model:.2e}"
      f"   interpolation {published_interpolation:.2e}")
if not model_mse.min() <= published_model <= model_mse.max():
    print("note: the six stored examples do not bracket the published aggregate")

# %% [markdown]
# The six examples bracket the published aggregate; they do not reproduce it,
# and cannot — the run reports over 444 real validation traces and stored six.
# The aggregate stays the property of `bead-ssl-v2-p25-full-s42-e30-matched-r1`.
#
# **Named limit.** The epoch-30 checkpoints were never repatriated from the
# compute cluster, so the trained encoder cannot be loaded in this notebook.
# What is shown above is the architecture (instantiated from the run's own
# hash-verified config) and the reconstructions the run itself wrote out.
# Re-running the encoder on a new trace would require the checkpoint or a
# retraining, which the 2026-08 redo schedules as new runs.
#
# **Second named limit.** That configuration declares
# `sampling_frequency_hz: 1000000` while the data is generated at 2 MHz. The
# sample counts above are exact and unaffected; any conversion of the mask
# geometry into microseconds is not, and the masking section takes that up as
# alignment A4.

# %% [markdown]
# ### 3. Why reconstruction is hungrier for data than classification
#
# This is the step that motivates the entire simulation chain, and it has been
# rejected once for being asserted rather than shown. So: the claim first, as a
# chain that can be checked link by link, then both sides counted.
#
# A **classifier** learns p(class | signal). There are three possible answers.
# Any factor of the signal that does not separate the classes may be ignored at
# no cost — a phase, a noise realisation, an amplitude offset that all three
# classes share. Examples matter mostly near the frontier, because that is
# where the answer changes.
#
# A **reconstructor** learns p(hidden | visible). There are 1,024 answers per
# trace, each a real number. Now every knob of the signal shapes the answer,
# and ignoring one is paid *wherever that knob varies* — not just at a
# frontier, because there is no frontier. Examples are therefore needed
# everywhere the knobs have mass, and needed **jointly**: knowing what a large
# amplitude looks like at high SNR says little about a large amplitude at low
# SNR if the two co-occur.
#
# The conclusion follows: covering a distribution scales with the **volume** of
# joint variability, not with the number of classes. Three classes cost the
# same whether the signal has one knob or five. A volume does not.
#
# Both sides are countable in this project. Here they are.

# %%
synthetic_key = "particles2snr-fbase-z8-cholesky-physicalcorr-effective-snr-synthetic-events@v5"
synthetic_root = dataset_root(synthetic_key)
synthetic_summary = json.loads((synthetic_root / "dataset_summary.json").read_text())

train_traces = ssl_metrics["counts"]["simulation_train"]
validation_traces = ssl_metrics["counts"]["simulation_validation"]
epochs = ssl_run["epochs"]
if train_traces + validation_traces != synthetic_summary["event_count"]:
    raise AssertionError("the run's splits do not exhaust the registered corpus")

hidden_per_epoch = train_traces * hidden_per_trace
problems_per_training = train_traces * epochs
hidden_per_training = problems_per_training * hidden_per_trace

print(f"synthetic corpus                {synthetic_summary['event_count']:,} events"
      f"  ({train_traces:,} train + {validation_traces:,} val)")
print(f"epochs                          {epochs}")
print("")
print("RECONSTRUCTION SIDE")
print(f"  trace-level problems / epoch  {train_traces:,}")
print(f"  trace-level problems / run    {problems_per_training:,}"
      f"   ({problems_per_training / 1e6:.2f} M)")
print(f"  hidden values / epoch         {hidden_per_epoch:,}"
      f"   ({hidden_per_epoch / 1e6:.1f} M)")
print(f"  hidden values / run           {hidden_per_training:,}"
      f"   ({hidden_per_training / 1e9:.2f} G)")
print("")
print("LABEL SIDE")
print(f"  annotated real events         {len(physical_rows):,}")
print(f"  independent real labels       {total_recordings:,}")
print(f"  human event-level classes     {classed}")
print(f"  total label information       {label_bits:.0f} bits")
print("")
print(f"reconstruction problems per annotated real event   "
      f"{problems_per_training / len(physical_rows):.0f}×")

# %% [markdown]
# **A correction, since the number circulates in two forms.** The deck's
# "≈ 1.2 M masked-reconstruction problems consumed per training" is the
# trace-level count: 39,108 training traces × 30 epochs = 1,173,240 windows
# presented to the network. It is *not* 39,108 × 1,024, which is 40.0 M — that
# is the number of hidden values in a single epoch, and over the full run the
# point-level total is 1.20 **G**. Both are correct statements of different
# things; the slide's phrasing counts problems, not points. State which one is
# meant, or the argument looks like it is off by three orders of magnitude.
#
# Against either, the label side is 2,073 events carrying 1,234 independent
# decisions and zero human class judgements. The reconstruction task consumes
# **566 trace-level problems for every annotated real event in existence**.
# There is no version of this corpus that supplies them.

# %% [markdown]
# #### The toy that shows why five knobs is the problem
#
# The generator this project builds has five knobs — amplitude P₀, Doppler
# frequency f_D, decay τ, SNR, and waveform asymmetry — plus a real noise
# carrier. Why should five be qualitatively harder than three classes?
#
# Fix a density: *m* bins along each knob, and ask for one example per cell.
# Covering the joint volume then costs mᵏ. A decision boundary between classes
# is a surface of one dimension less, so tracing it at the same density costs
# m^(k−1). The ratio is *m*, at every k: the volume task is one whole factor of
# the density more expensive, and that factor compounds with each knob added.
#
# The arithmetic is the content here, so it is computed rather than cited.

# %%
BINS_PER_KNOB = 8
MAX_KNOBS = 5

knob_counts = np.arange(1, MAX_KNOBS + 1)
volume_cost = BINS_PER_KNOB ** knob_counts
boundary_cost = BINS_PER_KNOB ** (knob_counts - 1)

real_reach = len(physical_rows) ** (1 / MAX_KNOBS)
synthetic_reach = synthetic_summary["event_count"] ** (1 / MAX_KNOBS)
real_volume_knobs = int(np.searchsorted(volume_cost, len(physical_rows), side="right"))
real_boundary_knobs = int(np.searchsorted(boundary_cost, len(physical_rows),
                                          side="right"))
synthetic_knobs = int(np.searchsorted(volume_cost, synthetic_summary["event_count"],
                                      side="right"))

for count, volume, boundary in zip(knob_counts, volume_cost, boundary_cost):
    print(f"k={count}   volume {volume:>7,}   boundary {boundary:>7,}"
          f"   ratio {volume // boundary}")
print("")
print(f"at {BINS_PER_KNOB} bins per knob, {len(physical_rows):,} real events reach"
      f" {real_volume_knobs} knobs as a volume,"
      f" {real_boundary_knobs} as a boundary")
print(f"at {BINS_PER_KNOB} bins per knob, {synthetic_summary['event_count']:,} "
      f"synthetic events reach {synthetic_knobs} knobs as a volume")
print(f"inverted at k={MAX_KNOBS}: real events buy {real_reach:.2f} bins per knob,"
      f" synthetic {synthetic_reach:.2f}")

# %%
plot_coverage_scaling(
    knob_counts, volume_cost, boundary_cost,
    bins_per_knob=BINS_PER_KNOB,
    marks=(
        (len(physical_rows), f"{len(physical_rows):,} real annotated events", "#b45309"),
        (synthetic_summary["event_count"],
         f"{synthetic_summary['event_count']:,} synthetic events", "#2563eb"),
    ),
)

# %% [markdown]
# Read the crossings, not the curves. At eight bins per knob, the same 2,073
# real events would trace a decision boundary through **four** knobs but fill a
# volume in only **three** — the one-dimension gap of the argument, priced on
# this corpus. The 47,980 synthetic events reach **five** as a volume.
# Inverted, the same fact reads: spread over five knobs, the whole real corpus
# is 4.6 bins deep per axis — under five distinguishable levels of amplitude,
# of frequency, of decay, of SNR, of asymmetry, *jointly*. The synthetic corpus
# is 8.6.
#
# That gap is the reason this project simulates. It is not a preference for
# synthetic data; it is that the task chosen to escape the label problem has an
# appetite the real corpus cannot meet, and simulation is the only source of
# density that does not require more instrument time than exists.
#
# **What is not claimed.** This is a scaling argument on a toy grid, not a
# sample-complexity bound. Three things it deliberately gets wrong in the
# conservative direction: the five knobs are correlated, so the occupied volume
# is smaller than the product of the marginals — which is exactly why the
# generator draws through a fitted Cholesky factor instead of sampling a box; a
# reconstructor does not need one example per cell, it needs enough to
# constrain a smooth function; and the class boundary being a (k−1)-dimensional
# surface is the standard geometric idealisation, not a property proven here.
# The claim that survives all three is the one the section needs: **the
# exponent differs by one, so the cost of the reconstruction task multiplies by
# the density with every knob, while the classification task does not.**
#
# **The sharpest objection.** Synthetic traces are not new information — the
# generator was fitted on those same 2,073 events, so simulation cannot conjure
# knowledge the corpus does not contain. Correct, and conceded. Simulation adds
# **density**, not knowledge: it populates the distribution the fit declares, at
# the resolution the reconstruction task requires. Whether that declared
# distribution lands where real events actually live is not an assumption this
# section is allowed to make — it is the measurement the coverage chain
# performs later, and the retrieval section tightens event by event. This
# section establishes only why those measurements are the ones that matter.

# %% [markdown]
# ### Glossary
#
# Every acronym and symbol used in this notebook, in the order the argument
# needs them. Each is also defined where it first appears.
#
# **The problem and the data**
#
# | term | meaning |
# |---|---|
# | **P2SNR** (particles2SNR) | the detection and fitting pipeline: turns a raw recording into bounded events with fitted physical parameters. |
# | **F-base** | the signal release the events index into: saturation-repaired by a cosine pre-filter, then band-pass filtered 7–80 kHz. |
# | **dual-clean** | the detection policy requiring peak evidence on both the band-filtered and the saturation-cleaned trace, so a single artefact cannot create an event. |
# | **MAD** | median absolute deviation, scaled by 1.4826 so it matches a Gaussian σ. The robust noise scale used to express amplitudes as z-scores. |
# | **z-score** | an amplitude in MAD units above the local median. |
# | **z8** | the event-selection policy of this dataset: filtered peak z ≥ 8, clean local peak z ≥ 1.5, and the annotation centre outside any saturation-repair interval. |
# | **unclear** | the fourth label, given by rule to any event whose fitted SNR is below −10 dB. Used for noise coverage only, never as a physical synthesis class. |
# | **split** | `train` / `val` are development; `test` is sealed and never read in this notebook. |
#
# **The signal model**
#
# | symbol | meaning |
# |---|---|
# | **s(t)** | the analytical event: a Gaussian envelope times a carrier, s(t) = P₀·exp(−½((t−t₀)/τ)²)·cos(2π f_D (t−t₀) + φ). |
# | **P₀** | envelope amplitude. |
# | **f_D** | Doppler (carrier) frequency of the burst, in kHz. |
# | **τ** | envelope decay time — the Gaussian half-width, in ms. Passage time, physically. |
# | **φ** | carrier phase at the event centre, in radians. Drawn uniformly; not identifiable from a fitted event, which is why retrieval averages eight phase views. |
# | **t₀** | arrival time of the event. Fixed at the window centre in generation, free in a real recording. |
# | **asymmetry a** | the fifth generated knob: τ becomes τ·e^(−a) before the centre and τ·e^(+a) after, so the envelope leans. |
# | **SNR** | signal-to-noise ratio, 20·log₁₀(clean RMS / noise RMS), in decibels. |
#
# **Learning**
#
# | term | meaning |
# |---|---|
# | **SSL** | self-supervised learning: the training target is built from the signal itself, so no annotation enters the loss. |
# | **encoder / decoder / z** | the encoder maps a window to a latent vector *z*; the decoder maps back to samples. Only the encoder survives training. |
# | **masked MSE** | mean squared error computed on the hidden samples only — the loss, and the reported metric. |
# | **B0** | the loss cell used by this run: signal MSE alone, with no derivative or energy term. |
# | **P25** | masking policy: 25 % of the 4,096 samples hidden as 64 disjoint 16-sample patches (1,024 samples), placed without reference to where the event is. |
# | **CYCLIC25** | masking policy with the same 1,024-sample budget, but drawn as event windows and background windows that cycle so every sample is eventually hidden. |
# | **Conv1D-GAP** | a supervised 1-D convolutional classifier whose penultimate global-average-pooled vector is reused as a latent space (the -L variant is 512-D). Not the SSL encoder. |
#
# **Measurement**
#
# | term | meaning |
# |---|---|
# | **PCA** | principal component analysis. Here: 16 components fitted on synthetic descriptors, defining the shared morphology space in which distances are read. |
# | **q80** | the 80th percentile of the synthetic-to-synthetic nearest-neighbour distance, used as the per-class coverage radius. |
# | **q50** | the median relative rank of a parent in a retrieval gallery — 0 % is first, ≈50 % is chance. |
# | **Recall@5** | the fraction of regenerated events whose exact parent returns in the top five neighbours. |


# %% [markdown]
# ## One analytical family, six knobs, and what a trained encoder keeps
#
# Everything else in this notebook compares a synthetic domain against a real
# one. That comparison presumes something that has not been established yet:
# that a space learned from raw signals orders the physics at all. If the
# encoder's geometry were unrelated to the generator's parameters, no distance
# measured in it would mean anything, and the coverage question would be moot.
#
# The cheapest honest way to settle it is not to argue but to build a family
# whose truth is known by construction, turn one knob at a time, and look at
# what happens to the embeddings. That is what this section does, and it does
# it live: the deck this notebook replaces showed the same experiment as a
# bitmap cropped out of an ODP file, whose provenance record literally reads
# `"command": "manual import"`. Nothing in it could be re-run, checked, or
# disagreed with.

# %% [markdown]
# ### The equation
#
# A single particle crossing the laser beam produces one Doppler burst: a
# carrier at the Doppler frequency, windowed by the Gaussian profile of the
# beam it is crossing, buried in detector noise. The generator writes that as
#
# $$s(t) \;=\; A\,\cos\!\left(2\pi f_D t + \varphi\right)\,
#   \exp\!\left(-\frac{(t-t_0)^2}{2\tau^{2}}\right)\;+\;n(t),
#   \qquad n(t)\sim\mathcal{N}\!\left(0,\sigma_n^{2}\right)$$
#
# with, on a window of duration $T$ normalised to $t \in [0, 1]$:
#
# - $A$ — peak amplitude of the carrier, in mV;
# - $f_D$ — Doppler frequency, in kHz (internally cycles per window, i.e.
#   $f_D^{\text{kHz}} \times T$); it is the beat frequency of the light fed back
#   into the laser cavity, and the quantity a velocity is read from;
# - $\varphi$ — carrier phase at the window origin, in radians;
# - $t_0$ — centre of the crossing, as a fraction of the window;
# - $\tau$ — Gaussian half-width (standard deviation) of the envelope, in ms;
#   the passage time quoted in the tables is its full width at half maximum,
#   $\mathrm{FWHM} = 2.355\,\tau$;
# - $n(t)$ — additive white Gaussian detector noise of standard deviation
#   $\sigma_n$.
#
# The sixth knob is not a term of the equation but the ratio between the two
# that are there,
#
# $$\mathrm{SNR}_{\mathrm{dB}} \;=\; 20\log_{10}
#   \frac{\mathrm{rms}\big(s - n\big)}{\sigma_n},$$
#
# swept by holding $A$ fixed and solving for $\sigma_n$. Five knobs therefore
# fix the clean waveform and the sixth fixes how much of it survives the noise.
#
# None of this is restated here as code. `particle_wave` in
# `p3_ssl.particle_equation_sweeps` is the implementation that produced the
# published sweep, and it is the one imported below; the notebook only chooses
# the arguments — and it reads even those from the published run's own
# configuration file rather than retyping them.

# %%
from p3_ssl.particle_equation_sweeps import (
    generate_single_particle_panels,
    particle_wave,
)
from internship_workspace.equation_latent_audit import (
    extract_penultimate_embeddings,
    load_frozen_classifier,
    sha256_file,
)

SWEEP_RUN = (
    workspace.artifacts_root
    / "unsupervised-learning-flow-cytometry"
    / "particle_equation_latent_sweeps"
    / "single_n1800_figure_based"
)
MODEL_RUN = SWEEP_RUN / "conv1dgap_same_input_3class"
CONFIG = json.loads((SWEEP_RUN / "run_config.json").read_text())

panels = generate_single_particle_panels(
    n_per_panel=int(CONFIG["n_per_panel"]),
    length=int(CONFIG["input_length"]),
    seed=int(CONFIG["seed"]),
    noise_std=float(CONFIG["noise_std"]),
    normalization=CONFIG["normalization"],
    shuffle=True,
    sweep_source=CONFIG["single_sweep_source"],
    phase_profile=CONFIG["phase_profile"],
    signal_window_duration_ms=float(CONFIG["signal_window_duration_ms"]),
    realistic_figure_based_sweeps=bool(CONFIG["realistic_figure_based_sweeps"]),
)
KNOBS = [panel.key for panel in panels]

published_signals = np.load(SWEEP_RUN / "synthetic_signals_encoded.npz")
signal_drift = max(
    float(np.max(np.abs(published_signals[f"{p.key}_signals"] - p.encoded_signal)))
    for p in panels
)
value_drift = max(
    float(np.max(np.abs(published_signals[f"{p.key}_color"] - p.color_value)))
    for p in panels
)
assert signal_drift < 1.0e-5, f"signal reproduction drifted by {signal_drift:.3e}"
assert value_drift == 0.0, f"swept values drifted by {value_drift:.3e}"

print(
    f"{len(panels)} knobs x {CONFIG['n_per_panel']} signals x "
    f"{CONFIG['input_length']} samples, window {CONFIG['signal_window_duration_ms']} ms"
)
print("knobs:", ", ".join(KNOBS))
print(
    "reproduces single_n1800_figure_based signal-for-signal "
    f"(max |delta| = {signal_drift:.2e} mV, swept values exact)"
)

# %% [markdown]
# ### The ranges are measured, not invented — with one exception
#
# Five of the six sweeps run over the global minimum-to-maximum of the
# descriptive statistics table of the real optical feedback interferometry
# (OFI, also called self-mixing interferometry) recordings, and the SNR sweep
# runs over the q20–q80 band measured on the visual subset of real particles.
# The sixth, $\tau$, does not: the published run was launched with
# `realistic_figure_based_sweeps=true`, which discards the passage-time range
# of the table and substitutes a hard-coded 0.05–0.25 ms. The table's own
# passage times, 0.33–1.10 ms FWHM, correspond to $\tau = 0.14$–0.47 ms, so the
# swept $\tau$ band sits mostly *below* anything measured. That is a limit of
# the sweep design, not of the encoder, and it is worth stating before reading
# the $\tau$ panel: it probes envelopes narrower than any real crossing.
#
# The knobs also interact. Only the SNR row sweeps SNR explicitly; but $A$ is
# swept against a *fixed* noise floor $\sigma_n = 0.02$ mV, and $\tau$ changes
# the energy of the burst, so both move the effective SNR too. Measuring that
# is one line, and it decides how the panels may be read.

# %%
t_norm = np.linspace(0.0, 1.0, int(CONFIG["input_length"]), dtype=np.float32)
effective_snr = {}
for panel in panels:
    clean = particle_wave(
        t_norm,
        panel.params["A"],
        panel.params["fD"],
        panel.params["phi"],
        panel.params["t0"],
        panel.params["tau"],
    )
    rms = np.sqrt(np.mean(np.square(clean), axis=1))
    sigma = (
        panel.params["snr_noise_std"]
        if panel.key == "snr_db"
        else np.full(rms.shape, float(CONFIG["noise_std"]), dtype=np.float32)
    )
    effective_snr[panel.key] = 20.0 * np.log10(rms / sigma)

snr_panel = next(p for p in panels if p.key == "snr_db")
snr_check = float(np.max(np.abs(effective_snr["snr_db"] - snr_panel.color_value)))
assert snr_check < 1.0e-3, f"effective-SNR formula disagrees by {snr_check:.3e} dB"

print(f"{'knob':<13}{'swept range':>26}{'effective SNR [dB]':>26}")
for panel in panels:
    values, _, symbol, unit = sweep_display(panel)
    span = effective_snr[panel.key]
    suffix = f" {unit}" if unit else ""
    print(
        f"{panel.key:<13}"
        f"{f'{symbol} {values.min():.2f} to {values.max():.2f}{suffix}':>26}"
        f"{f'{span.min():+.1f} to {span.max():+.1f}':>26}"
    )
print(
    f"\nthe SNR formula reproduces the generator's own SNR column to "
    f"{snr_check:.1e} dB, so the five other rows are on the same scale"
)

# %% [markdown]
# ### The gallery
#
# One row per knob, five signals spanning that knob's range, blue for the
# noised trace and orange for the Gaussian envelope. This is the figure the
# deck showed as a cropped bitmap; here it is generated from the same code that
# feeds the encoder, so the reader can change a range and look again.
#
# The rows are not equally informative, and that is the point of showing them.
# $A$ and SNR change how much noise is visible around an unchanged waveform;
# $f_D$ changes the number of fringes; $\tau$ changes the width of the packet.
# $\varphi$ and $t_0$ slide the same waveform along the window without altering
# it — with one exception visible in the row: at $t_0 = 0.2$ and $t_0 = 0.8$ the
# 1 ms window clips the tail of the envelope, so the extreme values of $t_0$ do
# change what the encoder receives. That detail comes back below.

# %%
gallery_axes = plot_signal_gallery(panels)
gallery_axes[0, 0].figure.suptitle(
    "The analytical family: five signals per swept knob", fontsize=12
)
gallery_axes[0, 0].figure.tight_layout(rect=(0, 0, 1, 0.97))
plt.show()

# %% [markdown]
# ### What the encoder is actually handed
#
# The gallery shows raw millivolts. The encoder never sees them: every window
# is z-scored to zero mean and unit variance before it reaches the network
# (`normalization = window_zscore`), because that is how the classifier was
# trained. That normalisation deletes $A$ as an amplitude. What survives is the
# amount of noise riding on the burst, since the noise floor is fixed while $A$
# is swept.

# %%
amplitude_panel = next(p for p in panels if p.key == "amplitude_A")
normalisation_axes = plot_normalisation_effect(amplitude_panel)
normalisation_axes[0, 0].figure.suptitle(
    "Window z-scoring removes the amplitude and leaves the noise", fontsize=11
)
normalisation_axes[0, 0].figure.tight_layout(rect=(0, 0, 1, 0.94))
plt.show()

order = np.argsort(amplitude_panel.color_value)
low, high = int(order[0]), int(order[-1])
for name, index in (("weakest", low), ("strongest", high)):
    raw_std = float(np.std(amplitude_panel.signal[index]))
    input_std = float(np.std(amplitude_panel.encoded_signal[index]))
    print(
        f"{name:<10} A = {amplitude_panel.color_value[index]:.2f} mV  "
        f"raw std {raw_std:.3f} mV  model-input std {input_std:.3f}  "
        f"effective SNR {effective_snr['amplitude_A'][index]:+.1f} dB"
    )
print(
    "\nafter normalisation the two windows differ only in relative noise: "
    "the A row is an SNR row with a different range"
)

# %% [markdown]
# ### The encoder
#
# The encoder is the frozen Conv1D-GAP-L classifier trained on the three
# particle sizes (2 µm, 4 µm, 10 µm), read at its 512-dimensional penultimate
# layer — the layer before the three-way decision, so the embedding is what the
# network built to separate sizes, not the decision itself.
# `load_frozen_classifier` refuses any file whose sha256, model name, input
# length or class list departs from the pinned contract, so the notebook either
# uses that exact checkpoint or none.
#
# Its path has to be passed explicitly. `DEFAULT_CONV_CHECKPOINT` in
# `p3_ssl.particle_equation_sweeps` points into `pretrained_backbones/`, a
# directory that does not exist in this workspace; the checkpoint lives in
# `pretrained_backbones-10dB/`. Two neighbouring hard-codings are worth naming
# in the same breath: `MODEL_DISPLAY` labels every conv checkpoint
# "Conv1D-GAP-L supervised same-input" without reading the file, and the shell
# launcher `launch_particle_equation_latent_sweeps.sh` passes
# `--input-length 4096` while the published figure was made at 512. The
# published run was therefore not produced by that launcher, and re-running it
# as shipped would silently produce a different experiment.
#
# Two facts about how this checkpoint was trained decide how the panels below
# may be read, so they are read off its own run configuration rather than
# recalled.

# %%
import re

BACKBONE_RUN = (
    workspace.artifacts_root
    / "unsupervised-learning-flow-cytometry"
    / "pretrained_backbones-10dB"
    / "particles2snr_f_3class_native_params_moment_patchtst_conv1dgap"
)
backbone_config = json.loads((BACKBONE_RUN / "run_config.json").read_text())
representation = backbone_config["input_representation_all_models"]
augmentation = backbone_config["augmentation"]
raw_crop = int(re.search(r"raw (\d+)", representation).group(1))

training_window_ms = raw_crop / SAMPLING_HZ * 1000.0
jitter_ms = training_window_ms * float(augmentation["jitter_frac"])
sweep_window_ms = float(CONFIG["signal_window_duration_ms"])
carrier_khz = float(np.median(panels[0].params["fD"])) / sweep_window_ms

print(f"training input      : {representation}")
print(f"training window     : {raw_crop} raw samples at {SAMPLING_HZ/1e6:.0f} MHz = {training_window_ms:.3f} ms")
print(f"sweep window        : {int(CONFIG['input_length'])} samples = {sweep_window_ms:.3f} ms")
print(f"position jitter     : +/- {augmentation['jitter_frac']:.0%} of the window = +/- {jitter_ms:.3f} ms")
print(f"amplitude scale     : {augmentation['aug_scale_min']} to {augmentation['aug_scale_max']} (removed again by the z-score)")
print(f"augmentation noise  : {augmentation['aug_snr_db']} dB")
print(
    f"\na {carrier_khz:.1f} kHz carrier shows {carrier_khz * sweep_window_ms:.0f} fringes in the swept window "
    f"against {carrier_khz * training_window_ms:.0f} in the trained one"
)
print(
    f"the jitter alone moves the carrier by {carrier_khz * jitter_ms:.0f} cycles, "
    "so training randomised arrival time and phase together"
)

# %% [markdown]
# Two consequences, one a limit and one a prediction.
#
# The limit: the sweep runs on a 1.0 ms window while the encoder was trained on
# 2.048 ms crops, so a knob value quoted in kHz produces about half the fringes
# per window that a real crossing at the same frequency would. The $f_D$ sweep
# spans 8–37.6 kHz, which are the measured frequencies, but the *waveforms* the
# encoder is shown are not the waveforms those frequencies produce in the
# geometry it was trained on. Nothing below reports an absolute frequency
# resolution, and nothing should.
#
# The prediction: training jittered the crop by a quarter of the window, which
# at these carriers is about a dozen cycles in either direction, so both arrival
# time and phase were randomised across the twelve views of every training
# event. If that recipe worked, $t_0$ and $\varphi$ must be *absent* from the
# latent space — not incidentally lost, but removed on purpose. The measurement
# below is a test of whether the augmentation took, and the reader should hold
# the deck's "$\varphi$ and $t_0$ do not come out ordered" to that standard.

# %%
CHECKPOINT = (
    workspace.artifacts_root
    / "unsupervised-learning-flow-cytometry"
    / "pretrained_backbones-10dB"
    / "particles2snr_f_3class_native_params_moment_patchtst_conv1dgap"
    / "conv1dgap_same_input_3class"
    / "best_model.pt"
)
published_embeddings = np.load(MODEL_RUN / "embeddings.npz")
published_metrics = json.loads((MODEL_RUN / "reduction_metrics.json").read_text())

embeddings, embedding_source = {}, "recomputed"
if CHECKPOINT.is_file():
    started = time.time()
    encoder = load_frozen_classifier(CHECKPOINT)
    print(f"checkpoint sha256 {sha256_file(CHECKPOINT)[:16]}... accepted")
    for panel in panels:
        panel_started = time.time()
        embeddings[panel.key] = extract_penultimate_embeddings(
            encoder, panel.encoded_signal, batch_size=256
        )
        print(f"  embedded {panel.key:<13} in {time.time() - panel_started:5.1f} s")
    print(f"total {time.time() - started:.1f} s on CPU")
else:
    embedding_source = "published"
    print(f"LIMIT: checkpoint absent at {CHECKPOINT.name}; reading published embeddings")
    for panel in panels:
        embeddings[panel.key] = published_embeddings[f"{panel.key}_embeddings"]

# %%
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

pca_coordinates, pca_variance = {}, {}
for panel in panels:
    scaled = StandardScaler().fit_transform(embeddings[panel.key])
    projector = PCA(n_components=2, random_state=int(CONFIG["seed"]))
    pca_coordinates[panel.key] = projector.fit_transform(scaled)
    pca_variance[panel.key] = float(np.sum(projector.explained_variance_ratio_))

print(f"{'knob':<13}{'PC1-PC2 variance':>20}{'published':>12}{'deviation':>12}")
worst_variance, worst_embedding = 0.0, 0.0
for panel in panels:
    got = pca_variance[panel.key]
    want = published_metrics["reduction"][panel.key]["pca_explained_variance_ratio_sum"]
    worst_variance = max(worst_variance, abs(got - want))
    worst_embedding = max(
        worst_embedding,
        float(np.max(np.abs(embeddings[panel.key] - published_embeddings[f"{panel.key}_embeddings"]))),
    )
    print(f"{panel.key:<13}{got:>20.6f}{want:>12.6f}{abs(got - want):>12.2e}")

if embedding_source == "recomputed":
    assert worst_embedding < 1.0e-4, f"embeddings drifted by {worst_embedding:.3e}"
    assert worst_variance < 1.0e-5, f"explained variance drifted by {worst_variance:.3e}"
    print(
        f"\nreproduces single_n1800_figure_based / conv1dgap_same_input_3class: "
        f"embeddings within {worst_embedding:.1e}, explained variance within "
        f"{worst_variance:.1e}"
    )

# %%
pca_axes = plot_latent_pca(panels, pca_coordinates, pca_variance)
pca_axes[0, 0].figure.suptitle(
    "The frozen encoder's latent space, one swept knob at a time (PCA)", fontsize=12
)
pca_axes[0, 0].figure.tight_layout(rect=(0, 0, 1, 0.95))
plt.show()

# %% [markdown]
# ### What the panels show, and why they are not enough
#
# Four of the six colour maps run smoothly along a trajectory: $A$, $f_D$,
# $\tau$ and SNR. Two do not: the $\varphi$ cloud is coloured at random, and the
# $t_0$ cloud very nearly so — its only structure is the pale and dark fringe
# drifting left, which is the clipped extreme $t_0$ of the gallery, not an
# ordering of arrival time. Two signals differing only in phase land in the same
# place. That is the deck's claim, and the figure supports it.
#
# It supports it weakly, though, and the objection is easy to state: these are
# two-dimensional projections of a 512-dimensional space, and the two plotted
# axes hold between 68 % and 87 % of its variance. An ordering carried by the
# discarded 13–32 % would look exactly like the $\varphi$ panel above. Reading
# "not ordered" off a PCA plane is reading an absence out of a projection.
#
# So the claim has to be re-asked in the full latent space, without any
# projection, and it has to be quantitative. The question "does the encoder
# order this knob?" is precisely: *can the knob be read back from a point's
# neighbourhood?* Fit a 10-nearest-neighbour regressor on the 512-dimensional
# embeddings, predict the knob under 5-fold cross-validation, and report $R^2$.
# One means the neighbourhood determines the knob; zero means the neighbourhood
# says no more than the global mean does.
#
# The same measurement on the encoder's *input* — the 512 z-scored samples —
# is the control that turns a description into an argument, because it
# separates "the encoder cannot see this knob" from "the encoder chose to throw
# it away". $\varphi$ is treated as the circular pair $(\cos\varphi,
# \sin\varphi)$ so that a wrap-around ordering would still score high; the test
# is deliberately generous to the hypothesis it is about to reject.

# %%
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor

NEIGHBOURS, FOLDS = 10, 5


def knob_target(panel):
    """The knob as a regression target; phase is put on the unit circle."""
    values = panel.color_value.astype(np.float64)
    if panel.key == "phase_phi":
        return np.column_stack([np.cos(values), np.sin(values)])
    return values.reshape(-1, 1)


def neighbourhood_r2(space, target, seed):
    folds = KFold(n_splits=FOLDS, shuffle=True, random_state=seed)
    predicted = cross_val_predict(
        KNeighborsRegressor(n_neighbors=NEIGHBOURS), space, target, cv=folds
    )
    return float(r2_score(target, predicted, multioutput="variance_weighted"))


recovery = []
for panel in panels:
    target = knob_target(panel)
    recovery.append(
        {
            "key": panel.key,
            "label": sweep_display(panel)[1],
            "latent": neighbourhood_r2(embeddings[panel.key], target, int(CONFIG["seed"])),
            "input": neighbourhood_r2(panel.encoded_signal, target, int(CONFIG["seed"])),
        }
    )

print(f"{'knob':<13}{'latent R2':>12}{'input R2':>12}   verdict")
for row in recovery:
    ordered = row["latent"] >= 0.5
    verdict = "ordered by the encoder" if ordered else "discarded by the encoder"
    if not ordered and row["input"] < 0.5:
        verdict = "absent from both"
    print(f"{row['key']:<13}{row['latent']:>+12.3f}{row['input']:>+12.3f}   {verdict}")

# %%
recovery_ax = plot_knob_recovery(recovery)
recovery_ax.set_title(
    "Can the knob be read back from a point's ten nearest neighbours?", fontsize=11
)
recovery_ax.figure.tight_layout()
plt.show()

# %% [markdown]
# ### The finding, sharper than the deck's
#
# The deck asserted that $A$, $f_D$, $\tau$ and SNR come out ordered while
# $\varphi$ and $t_0$ do not. Measured in the full latent space, that holds and
# is not marginal: $R^2 = 0.999$ for $f_D$, 0.990 for $\tau$, 0.988 for $A$,
# 0.963 for SNR, against 0.024 for $t_0$ and $-0.112$ for $\varphi$. A
# negative $R^2$ means the ten nearest latent neighbours are worse than a
# constant guess — the latent space carries no usable phase at all. The 0.024
# left on $t_0$ is the window clipping leaking through, not a residual ordering
# of arrival time.
#
# The control says what the deck could not. In the encoder's own input, before
# a single convolution, $\varphi$ and $t_0$ are recovered *perfectly*
# ($R^2 = 1.000$ for both): raw z-scored waveforms that differ only in phase or
# arrival time are far apart in sample space, and near-neighbours in that space
# share the value almost exactly. The information is handed to the encoder and
# does not come out. This is invariance, not blindness — and not an accident
# either: the training recipe jittered every crop by a quarter of the window, a
# dozen carrier cycles either way, across twelve views of each event. The
# network was *asked* to ignore where the burst sits and what phase it starts
# on, and the measurement says the request was granted in full. That is the one
# property a size classifier had to acquire, and exactly the property the
# phase-insensitive morphology space is built to formalise later.
#
# The control also refutes a reading of the $A$ row that the panels invite. In
# the input space $A$ scores $-1.81$ and SNR $-2.36$: far below chance, because
# window z-scoring erased the amplitude and the nearest neighbour of a very
# noisy window is a clean one, so a neighbourhood vote is systematically
# wrong. The encoder turns that same input into an ordering worth $R^2 = 0.988$.
# It is not recovering an amplitude — no amplitude survives normalisation. It is
# ordering *noise level*, and the effective-SNR table above makes the
# consequence explicit: the $A$ row sweeps +5.5 to +36.5 dB while the SNR row
# sweeps −7.8 to +6.0 dB. Two of the six knobs are, after normalisation, the
# same physical knob over adjacent ranges. The six-knob figure has five
# independent degrees of freedom.
#
# **What is not claimed.** Everything here is synthetic and analytical; no real
# recording, no dataset row and certainly no sealed-test row is touched. An
# encoder that orders a knob of a Gaussian-windowed cosine has not been shown
# to order anything on real crossings, whose envelopes are asymmetric and whose
# noise is not white. It is not even shown on the geometry it was trained in:
# the sweep window is 1.0 ms against 2.048 ms of training crop, and the $\tau$
# range dips below any measured passage time. This section establishes only that
# a learned signal space *can* be ordered by physics — enough to justify
# measuring distances in one, not enough to validate any measurement made with
# it. The $R^2$ values are a property of this family, this checkpoint and this
# normalisation; a different noise model would move them.

# %%
try:
    emitted = notebook_evidence.emit_run(
        workspace,
        section="latent-knob-recovery",
        metrics={
            "schema_version": 1,
            "analysis": "neighbourhood recovery of analytical-family knobs",
            "neighbours": NEIGHBOURS,
            "cross_validation_folds": FOLDS,
            "n_per_knob": int(CONFIG["n_per_panel"]),
            "knobs": {
                row["key"]: {
                    "latent_neighbourhood_r2": row["latent"],
                    "input_neighbourhood_r2": row["input"],
                    "pca_2d_explained_variance": pca_variance[row["key"]],
                    "effective_snr_db_min": float(np.min(effective_snr[row["key"]])),
                    "effective_snr_db_max": float(np.max(effective_snr[row["key"]])),
                }
                for row in recovery
            },
        },
        provenance={
            "datasets": dataset_provenance(),
            "inputs": {
                "checkpoint": sha256_file(CHECKPOINT),
                "run_config": sha256_file(SWEEP_RUN / "run_config.json"),
                "reduction_metrics": sha256_file(MODEL_RUN / "reduction_metrics.json"),
                "backbone_run_config": sha256_file(BACKBONE_RUN / "run_config.json"),
            },
            "parameters": {
                key: CONFIG[key]
                for key in (
                    "n_per_panel",
                    "input_length",
                    "seed",
                    "noise_std",
                    "normalization",
                    "single_sweep_source",
                    "phase_profile",
                    "signal_window_duration_ms",
                    "realistic_figure_based_sweeps",
                )
            },
            "metric_definitions": {
                "latent_neighbourhood_r2": (
                    "cross-validated R2 of a 10-nearest-neighbour regression of the "
                    "swept knob on the 512-D penultimate embedding; phase is "
                    "regressed as (cos phi, sin phi) and scored variance-weighted"
                ),
                "input_neighbourhood_r2": "the same regression on the 512 window-z-scored samples",
                "effective_snr_db": "20 log10(rms(clean signal) / noise standard deviation)",
            },
        },
        claim_boundary=(
            "Measures which knobs of the analytical single-particle family are "
            "recoverable from a neighbourhood of the frozen Conv1D-GAP-L latent "
            "space, against the same measurement on the encoder's input. "
            "Synthetic signals only; no real recording, no dataset row, no "
            "sealed-test access. Says nothing about real crossings, about any "
            "other checkpoint, or about downstream coverage."
        ),
    )
    print(f"emitted {emitted.name}")
except WorkspaceError as error:
    print(f"no evidence emitted ({error})")


# %% [markdown]
# ## From measured events to a generator
#
# The previous sections established what an event is and what a trained encoder
# can read off one. This one answers the question that makes simulation possible
# at all: **once a real event has been measured, what exactly do we keep, and
# how do we turn a few hundred kept events per class into an unlimited stream of
# new ones that behave like them?**
#
# The answer is four numbers per event, a Gaussian envelope over those four
# numbers in a transformed coordinate system, and a Cholesky factor that carries
# the dependence between them. Everything below is imported from
# `particles2snr` — the same functions the manifested analysis and generation
# runs called — so a cell that disagrees with a published run is a real
# disagreement, not a transcription difference.

# %%
import math

from particles2snr.z8_cholesky_analysis import (
    PARAMETER_ORDER,
    pearson_correlation_matrix,
    regularize_for_cholesky,
    rows_for_population,
    transformed_parameter_matrix,
)
from particles2snr.z8_cholesky_generation import (
    SEED,
    correlation_validation,
    generate_parameters,
    load_gaussian_targets,
    load_recommended_cholesky,
)
from particles2snr.z8_parameter_analysis import (
    load_approved_estimation_population,
    read_events,
)

CHOL_EVENTS = (
    "particles2snr-fbase-dual-clean-z8-events-3class-plus-unclear-development@v2"
)
CHOL_CORRELATION_RUN = "particle-z8-cholesky-correlations-v2-boundary-censored-r1"
CHOL_DISTRIBUTION_RUN = "particle-z8-parameter-distributions-v2-boundary-censored-r2"
CHOL_ENVELOPE_RUN = "particle-z8-gaussian-envelopes-v2-boundary-censored-r1"
CHOL_GENERATION_RUN = "particle-z8-cholesky-synthetic-generation-v2-r3"
CHOL_SYNTHETIC_ID = (
    "particles2snr-fbase-z8-cholesky-physicalcorr-effective-snr-synthetic-events"
)

chol_root = dataset_root(CHOL_EVENTS)
chol_summary = json.loads((chol_root / "dataset_summary.json").read_text())
chol_rows = read_events(chol_root / "events.csv", dataset_summary=chol_summary)

chol_splits = sorted({row["split"] for row in chol_rows})
if "test" in chol_splits:
    raise RuntimeError("sealed test rows reached the parameter population")
chol_counts = {
    name: sum(1 for row in chol_rows if row["class_name"] == name)
    for name in CLASS_ORDER
}
chol_unclear = sum(1 for row in chol_rows if row["class_name"] == "unclear")
print(f"{len(chol_rows):,} annotated events, splits {chol_splits} (no sealed test)")
print(
    "  classified: "
    + ", ".join(f"{name} {count:,}" for name, count in chol_counts.items())
    + f" = {sum(chol_counts.values()):,}"
)
print(f"  plus {chol_unclear:,} annotated 'unclear', kept aside for now")

# %% [markdown]
# ### The four numbers
#
# A detected event is a short damped oscillation. Fitting the instrument's
# signal model to it returns four physical parameters, and those four are the
# entire description the generator will ever use:
#
# - **P₀** — the amplitude of the envelope at its peak, in acquisition units.
#   How loud the particle was.
# - **f_D** — the Doppler frequency of the carrier, in kHz. Set by how fast the
#   particle crossed the acoustic fringe pattern.
# - **τ** — the Gaussian envelope width in milliseconds, i.e. how long the
#   particle stayed in the beam. (The dataset contract calls this
#   `tau_semantics: Gaussian envelope sigma in milliseconds`; it is a duration,
#   not an exponential decay constant.)
# - **SNR** — signal-to-noise ratio in dB against the F-base noise floor. Not a
#   property of the particle alone: it is the particle *as this instrument heard
#   it*, and it is the coordinate on which every downstream difficulty lives.
#
# Anything else about the event — its phase, its position in the window — is
# either drawn uniformly at generation time or fixed by convention, so it is not
# estimated here.

# %% [markdown]
# ### Boundary censoring: a fit is not always a measurement
#
# Not every annotated event is usable for estimating those four numbers. An
# annotation whose interval touches the edge of its source recording describes a
# *truncated* event: the crossing began before the file did, or was still going
# when it ended. The fit still converges — it always does — but what it returns
# is a property of the window, not of the particle.
#
# The published population run applies exactly one rule, recorded verbatim in
# its own summary: exclude an event when `start_sample <= 0` or
# `end_sample >= source_signal_length`. The rule is a *statement about the
# annotation geometry*, decided before any parameter is looked at, which is what
# keeps it from being a quiet quality filter on the parameters themselves.
#
# The cell below does not re-derive the censoring; it loads the approved
# population through the shipped loader, which re-checks the evidence receipt
# and the event-level census, then measures what censoring actually removed.

# %%
chol_eligible, chol_population = load_approved_estimation_population(
    workspace,
    chol_rows,
    dataset_id=CHOL_EVENTS,
    analysis_run_id=CHOL_DISTRIBUTION_RUN,
    evidence_id="particle-z8-v2-parameter-distributions-result",
    evidence_run_id="particle-z8-v2-parameter-distributions-result-r4",
)
chol_distributions = published(CHOL_DISTRIBUTION_RUN, "summary_metrics.json")
assert chol_population["eligible_event_count"] == len(chol_eligible)
assert (
    chol_distributions["boundary_censoring_policy"]
    == "exclude_from_empirical_estimation_when_start_sample<=0"
    "_or_end_sample>=source_signal_length"
)
print(
    f"{chol_population['boundary_censored_event_count']} of {len(chol_rows):,} events "
    f"censored, {chol_population['eligible_event_count']:,} retained"
)

chol_censored_ids = {
    item["event_id"] for item in chol_distributions["boundary_censored_events"]
}
chol_reason_counts = {}
for item in chol_distributions["boundary_censored_events"]:
    for reason in item["reasons"]:
        chol_reason_counts[reason] = chol_reason_counts.get(reason, 0) + 1
for reason, count in sorted(chol_reason_counts.items()):
    print(f"  {reason}: {count}")

chol_keep_rows = [row for row in chol_rows if row["event_id"] not in chol_censored_ids]
chol_drop_rows = [row for row in chol_rows if row["event_id"] in chol_censored_ids]


def chol_series(rows, column):
    return np.array([float(row[column]) for row in rows], dtype=np.float64)


chol_keep = {
    key: chol_series(chol_keep_rows, column)
    for key, column in (
        ("amplitude_p0", "particles2snr_amplitude"),
        ("tau_ms", "tau_ms"),
        ("snr_db", "snr_db"),
    )
}
chol_drop = {
    key: chol_series(chol_drop_rows, column)
    for key, column in (
        ("amplitude_p0", "particles2snr_amplitude"),
        ("tau_ms", "tau_ms"),
        ("snr_db", "snr_db"),
    )
}
print(f"\n{'class':6s} {'median τ (ms)':>26s} {'median SNR (dB)':>26s}")
chol_censoring_shift = {}
for chol_name in CLASS_ORDER:
    keep = [row for row in chol_keep_rows if row["class_name"] == chol_name]
    drop = [row for row in chol_drop_rows if row["class_name"] == chol_name]
    entry = {}
    for key, column in (("tau_ms", "tau_ms"), ("snr_db", "snr_db")):
        entry[key] = (
            float(np.median(chol_series(keep, column))),
            float(np.median(chol_series(drop, column))),
        )
    chol_censoring_shift[chol_name] = entry
    print(
        f"{chol_name:6s} {entry['tau_ms'][0]:11.3f} → {entry['tau_ms'][1]:<12.3f}"
        f" {entry['snr_db'][0]:11.2f} → {entry['snr_db'][1]:<12.2f}"
    )
print(
    f"pooled  {np.median(chol_keep['tau_ms']):11.3f} → "
    f"{np.median(chol_drop['tau_ms']):<12.3f}"
    f" {np.median(chol_keep['snr_db']):11.2f} → "
    f"{np.median(chol_drop['snr_db']):<12.2f}"
)
chol_tau_over = sum(1 for row in chol_rows if float(row["tau_ms"]) > 0.30)
chol_tau_over_censored = sum(
    1 for row in chol_drop_rows if float(row["tau_ms"]) > 0.30
)
print(
    f"  events fitted with τ > 0.30 ms: {chol_tau_over}, of which "
    f"{chol_tau_over_censored} boundary-censored"
)

# %% [markdown]
# The censored events are not a random sample. Pooled, their median SNR is 7.1
# dB lower, and in the 2 µm class their median fitted τ is 42 % longer (0.172 →
# 0.245 ms). That is the expected signature of a truncated window: with no
# falling edge to anchor the envelope against, the fit widens the envelope and
# reports a poorer signal-to-noise. The sharpest single number is that all 17
# events fitted with τ > 0.30 ms are censored, without exception — and nothing
# in the rule looks at τ. The rule is removing bad fits, not inconvenient
# particles.

# %%
chol_fig, chol_axes = plt.subplots(1, 2, figsize=(11, 3.6))
plot_censoring_shift(chol_keep, chol_drop, axes=chol_axes)
chol_fig.suptitle(
    "Boundary-censored events are systematically noisier and wider", fontsize=11
)
chol_fig.tight_layout()
chol_fig

# %% [markdown]
# ### Two populations, one of which the deck forgets
#
# The 121 events an annotator marked **unclear** still have a physical source
# class — the recording they came from was a pure suspension of one particle
# size. So each per-class estimate can be built two ways, and the shipped code
# names both:
#
# - **physical** — only events the annotator confidently assigned to that class;
# - **inclusive** — also the unclear events whose *source recording* was that
#   class.
#
# The parameter run splits them by coordinate: P₀, f_D and τ come from the
# physical population, while SNR is estimated on the inclusive one. That is
# deliberate — an unclear annotation is uninformative about *which* particle it
# was, but perfectly informative about *how loud it was against the noise*, and
# the SNR marginal is the one that must not be under-covered.

# %%
chol_values = {}
chol_class_n = {}
for chol_name in CLASS_ORDER:
    physical = rows_for_population(chol_eligible, chol_name, "physical")
    inclusive = rows_for_population(chol_eligible, chol_name, "inclusive")
    chol_values[chol_name] = {
        "amplitude_p0": chol_series(physical, "particles2snr_amplitude"),
        "frequency_khz": chol_series(physical, "frequency_hz") / 1000.0,
        "tau_ms": chol_series(physical, "tau_ms"),
        "snr_db": chol_series(inclusive, "snr_db"),
    }
    chol_class_n[chol_name] = {"physical": len(physical), "inclusive": len(inclusive)}
    print(
        f"{chol_name}: physical {len(physical):,}  inclusive {len(inclusive):,}"
        f"  (+{len(inclusive) - len(physical)} unclear)"
    )

chol_board = published("ssl-v3-v16-retrieval-and-ranges-r5", "board_values.json")
chol_range_drift = 0.0
for chol_name, entry in chol_board["observed_ranges"].items():
    assert entry["physical_events"] == chol_class_n[chol_name]["physical"]
    assert entry["snr_population"] == chol_class_n[chol_name]["inclusive"]
    for parameter, bounds in entry["ranges"].items():
        key = "snr_db" if parameter == "snr_effective_fbase_db" else parameter
        series = chol_values[chol_name][key]
        chol_range_drift = max(
            chol_range_drift,
            abs(series.min() - bounds["minimum"]),
            abs(series.max() - bounds["maximum"]),
        )
assert chol_range_drift == 0.0, f"observed ranges drifted by {chol_range_drift:.3e}"
print("\nreproduces ssl-v3-v16-retrieval-and-ranges-r5 observed ranges exactly")

# %%
chol_fig, chol_axes = plt.subplots(1, 4, figsize=(15, 3.4))
plot_parameter_marginals(chol_values, CLASS_COLOUR, axes=chol_axes)
chol_fig.suptitle(
    "The four fitted parameters, class-conditional, on the retained population",
    fontsize=11,
)
chol_fig.tight_layout()
chol_fig

# %% [markdown]
# The frequency panel is combed rather than smooth, and that is not a binning
# artefact. f_D is not measured continuously: it is read off a discrete
# transform, so it can only take grid values.

# %%
chol_grid_hz = SAMPLING_HZ / 4096.0
chol_real_f = np.unique(
    np.array([float(row["frequency_hz"]) for row in chol_eligible])
)
chol_on_grid = float(
    np.mean(
        np.abs(chol_real_f / chol_grid_hz - np.round(chol_real_f / chol_grid_hz))
        < 1e-9
    )
)
print(
    f"measured f_D: {chol_real_f.size} distinct values over "
    f"{len(chol_eligible):,} events, {100 * chol_on_grid:.0f}% on a "
    f"{chol_grid_hz:.2f} Hz grid = {SAMPLING_HZ / 1e6:.0f} MHz / 4096"
)

# %% [markdown]
# Sixty-five distinct frequencies across 1,948 retained events, every one an
# exact multiple of 488.28 Hz — the bin width of a 4096-point transform at the
# acquisition rate. The generator draws f_D from a continuous Gaussian, so it
# will produce none of them, and the comparison is made once the synthetic
# dataset is in hand further down. Hold the observation: it is the one place
# where a real and a synthetic *parameter table* are trivially separable.

# %% [markdown]
# ### Why the generator does not work in these coordinates
#
# P₀ and τ are strictly positive and right-skewed; a Gaussian fitted to them
# directly puts mass on values the instrument cannot produce. Measured on the
# retained population, that is not a rounding concern — it is a large fraction
# of the distribution for two of the three classes.

# %%
chol_negative_mass = {}
for chol_name in CLASS_ORDER:
    row = {}
    for parameter in ("amplitude_p0", "tau_ms"):
        series = chol_values[chol_name][parameter]
        mean, deviation = float(series.mean()), float(series.std(ddof=1))
        row[parameter] = 0.5 * (1.0 + math.erf(-mean / (deviation * math.sqrt(2.0))))
    chol_negative_mass[chol_name] = row
    print(
        f"{chol_name}: a Gaussian on raw P₀ puts {100 * row['amplitude_p0']:5.2f}% "
        f"of its mass below zero; on raw τ, {100 * row['tau_ms']:.2f}%"
    )
print(
    "\nIn log coordinates the constraint is structural: exp(x) > 0 for every x, "
    "so no draw can be unphysical."
)

# %% [markdown]
# So the generator works in **transformed coordinates**
# `(log P₀, f_D, log τ, SNR)` — the two positive parameters on a log scale, the
# two sign-free ones as they are. Every correlation, every Cholesky factor and
# every delta below is stated in those coordinates, and `PARAMETER_ORDER` in the
# shipped module fixes the order once and for all.

# %%
print("transformed coordinate order:", PARAMETER_ORDER)

# %% [markdown]
# ### The dependence structure
#
# The four parameters are not independent. A loud event is a high-SNR event
# almost by construction; a fast crossing is a short one. If the generator drew
# each coordinate from its own marginal, it would produce parameter vectors that
# are individually plausible and jointly impossible.
#
# The **Pearson correlation matrix** *R* of the transformed coordinates records
# that structure as one 4×4 symmetric matrix per class. Only the lower triangle
# carries information — the upper half is its mirror, and the diagonal is 1 by
# definition — so that is all that is plotted.
#
# The cell recomputes those matrices from the retained events with the shipped
# estimator and asserts they match the published run to the last bit.

# %%
chol_matrices = {}
chol_factors_recomputed = {}
chol_shrinkage = {}
for chol_name in CLASS_ORDER:
    matrix = transformed_parameter_matrix(
        rows_for_population(chol_eligible, chol_name, "physical")
    )
    correlation = pearson_correlation_matrix(matrix)
    regularized, shrinkage = regularize_for_cholesky(correlation)
    chol_matrices[chol_name] = correlation
    chol_factors_recomputed[chol_name] = np.linalg.cholesky(regularized)
    chol_shrinkage[chol_name] = shrinkage

with (run_dir(CHOL_CORRELATION_RUN) / "correlation_coefficients.csv").open(
    newline="", encoding="utf-8"
) as handle:
    chol_published_r = {
        (
            row["class_name"],
            row["population"],
            row["x_parameter"],
            row["y_parameter"],
        ): (
            float(row["pearson_r"]),
            int(row["n_events"]),
        )
        for row in csv.DictReader(handle)
    }
chol_reproduction_drift = 0.0
for chol_name in CLASS_ORDER:
    for i in range(4):
        for j in range(i + 1, 4):
            want, count = chol_published_r[
                (chol_name, "physical", PARAMETER_ORDER[i], PARAMETER_ORDER[j])
            ]
            assert count == chol_class_n[chol_name]["physical"]
            chol_reproduction_drift = max(
                chol_reproduction_drift, abs(chol_matrices[chol_name][i, j] - want)
            )
assert chol_reproduction_drift == 0.0, (
    f"reproduction drifted by {chol_reproduction_drift:.3e}"
)
print(f"reproduces {CHOL_CORRELATION_RUN} exactly")
print("diagonal shrinkage applied:", chol_shrinkage, "(none needed)")

# %%
chol_fig, chol_axes = plt.subplots(1, 3, figsize=(13, 3.9))
plot_correlation_triangles(
    chol_matrices,
    {name: chol_class_n[name]["physical"] for name in CLASS_ORDER},
    {name: "physical" for name in CLASS_ORDER},
    axes=chol_axes,
)
chol_fig.suptitle(
    "Pearson r of the transformed parameters, one triangle per class", fontsize=11
)
chol_fig

# %% [markdown]
# Two dependences hold in every class and are the ones the generator must not
# lose. **Amplitude and SNR travel together** — r = +0.84, +0.72, +0.94 from
# smallest to largest particle, which is nearly the definition of SNR on a fixed
# noise floor. And **frequency opposes width**: r = −0.59, −0.72, −0.35, a
# particle that crosses faster is in the beam for less time. The 4 µm class,
# with the most events and the tightest amplitude spread, shows the strongest
# frequency–width coupling and the only appreciable frequency–SNR term
# (−0.38).

# %% [markdown]
# ### The Cholesky construction, in two sentences
#
# Any valid correlation matrix *R* is symmetric and positive-definite, so it
# factorises uniquely as *R = L Lᵀ* with *L* lower-triangular — the **Cholesky
# factor**. That is exactly the object needed to draw correlated parameters:
# take four independent standard normal draws *z*, form *u = L z*, and *u* has
# correlation matrix *R* by construction, because
# E[u uᵀ] = L E[z zᵀ] Lᵀ = L Lᵀ = R.
#
# The generator then rescales *u* by the fitted envelope: `mean + u * sigma`, in
# the transformed coordinates. The dependence comes from *L*, the spread comes
# from the Gaussian envelope run, and neither is invented here.
#
# Positive-definiteness is not automatic — a correlation matrix estimated on few
# events can fail it, and the shipped `regularize_for_cholesky` would then shrink
# the diagonal until it holds. It never had to: all three matrices factorise as
# measured. The condition number κ(R) printed below says how close each came,
# and 10 µm is the marginal one at κ = 71 on 194 events, five times worse than
# the other two.

# %%
chol_correlation_dir = run_dir(CHOL_CORRELATION_RUN)
chol_factors, chol_populations_used = load_recommended_cholesky(
    chol_correlation_dir / "cholesky_factors.csv",
    chol_correlation_dir / "recommendations.csv",
)
chol_factor_drift = max(
    float(np.abs(chol_factors_recomputed[name] - chol_factors[name]).max())
    for name in CLASS_ORDER
    if chol_populations_used[name] == "physical"
)
assert chol_factor_drift == 0.0, f"Cholesky factor drifted by {chol_factor_drift:.3e}"
chol_identity_drift = max(
    float(np.abs(chol_factors[name] @ chol_factors[name].T - chol_matrices[name]).max())
    for name in CLASS_ORDER
    if chol_populations_used[name] == "physical"
)
print(f"L reproduces the published factor exactly; max |L Lᵀ − R| = "
      f"{chol_identity_drift:.2e}")
print("dependence population the generator was given:", chol_populations_used)

with (chol_correlation_dir / "matrix_diagnostics.csv").open(
    newline="", encoding="utf-8"
) as handle:
    chol_conditioning = {
        (row["class_name"], row["population"]): float(row["condition_number"])
        for row in csv.DictReader(handle)
        if row["positive_definite"] == "True"
    }
print("\nfactorisation exists for every class without regularisation; condition κ(R):")
for chol_name in CLASS_ORDER:
    population = chol_populations_used[chol_name]
    print(
        f"  {chol_name:5s} {population:9s} κ = "
        f"{chol_conditioning[(chol_name, population)]:6.1f}"
    )

# %% [markdown]
# ### A transcription the notebook was built to catch
#
# The deck states the three target triangles as literal constants in
# `presentation/recipes/pearson_targets.py`, transcribed by hand from the
# correlation run, under the caption *"the dependence structure the Cholesky
# generator is asked to reproduce."* Two separate things can go wrong there, and
# the cell below checks both: whether the digits match the run, and whether the
# run's *physical* population is the one the generator was actually handed.

# %%
from internship_workspace.presentation.recipes.pearson_targets import (
    MATRICES as CHOL_DECK_MATRICES,
)

chol_audit = {}
for chol_name, (chol_label, chol_deck_n, chol_triangle) in zip(
    CLASS_ORDER, CHOL_DECK_MATRICES
):
    used = chol_factors[chol_name] @ chol_factors[chol_name].T
    rounding, mismatch, worst = 0.0, 0.0, None
    for i, deck_row in enumerate(chol_triangle):
        for j, deck_value in enumerate(deck_row):
            if j >= i:
                continue
            physical = chol_published_r[
                (chol_name, "physical", PARAMETER_ORDER[j], PARAMETER_ORDER[i])
            ][0]
            rounding = max(rounding, abs(deck_value - physical))
            gap = abs(deck_value - used[i, j])
            if gap > mismatch:
                mismatch, worst = gap, (PARAMETER_ORDER[j], PARAMETER_ORDER[i])
    population = chol_populations_used[chol_name]
    chol_audit[chol_name] = {
        "deck_n": chol_deck_n,
        "generator_population": population,
        "generator_n": chol_class_n[chol_name][population],
        "max_deck_minus_physical": rounding,
        "max_deck_minus_generator_target": mismatch,
        "worst_pair": worst,
    }
    print(
        f"{chol_label:5s} deck n = {chol_deck_n:,} · generator used '{population}' "
        f"n = {chol_class_n[chol_name][population]:,}"
    )
    print(
        f"      |deck − physical r| ≤ {rounding:.4f} (2-decimal rounding) · "
        f"|deck − generator target| ≤ {mismatch:.4f}"
    )

# %% [markdown]
# The digits are clean: every deck cell is within 0.005 of the run's *physical*
# value, which is what two-decimal rounding costs. The population is not. The
# generator's own recommendation file assigns the **inclusive** matrix to 4 µm —
# the class where adding the unclear events moved no coefficient by more than
# the 0.10 engineering threshold, so they were kept — and the *physical* matrix
# to 2 µm and 10 µm, where at least one coefficient moved further than that. The
# promoted dataset records the same policy in its own contract
# (`correlations: {2um: physical, 4um: inclusive, 10um: physical}`).
#
# So for 4 µm the deck shows a triangle over 1,171 events that the generator
# never saw; it was driven by a triangle over 1,186. The largest gap is
# log P₀ × f_D, where the deck reads −0.17 and the generator's target was
# −0.072 — a factor of two, and a sign-strength reading a viewer would take
# literally. The other two classes are stated correctly.
#
# **What this is not.** It is not a defect in the generated data: the datasets
# were built from the recommendation file, and their delta tables below are
# measured against the targets actually used. It is a slide that names the wrong
# population for one class. The fix belongs in the recipe, not in the run.

# %%
chol_ranked = sorted(
    chol_audit.items(),
    key=lambda item: item[1]["max_deck_minus_generator_target"],
    reverse=True,
)
for name, entry in chol_ranked:
    flag = "MISMATCH" if entry["max_deck_minus_generator_target"] > 0.01 else "ok"
    print(
        f"{name:5s} {flag:9s} deck n={entry['deck_n']:,} vs generator "
        f"{entry['generator_population']} n={entry['generator_n']:,} · "
        f"max gap {entry['max_deck_minus_generator_target']:.3f}"
        + (f" at {entry['worst_pair']}" if flag == "MISMATCH" else "")
    )

# %% [markdown]
# ### Does the dependence structure matter? A control
#
# Before accepting the Cholesky machinery, it is worth measuring what dropping
# it would cost, on the same code path. Replacing every *L* by the identity
# matrix leaves the marginals untouched and removes only the dependence. The
# shipped `generate_parameters` takes the factors as an argument, so the control
# is one substitution rather than a second implementation.

# %%
chol_envelope_dir = run_dir(CHOL_ENVELOPE_RUN)
chol_targets, chol_budgets = load_gaussian_targets(
    chol_envelope_dir / "gaussian_envelope_parameters.csv", include_budgets=True
)
print("class budgets:", chol_budgets, f"= {sum(chol_budgets.values()):,} events")

chol_independent, _ = generate_parameters(
    chol_targets,
    {name: np.eye(4) for name in CLASS_ORDER},
    seed=SEED,
    budgets=chol_budgets,
    dataset_id="independent-marginals-control",
)
chol_control = correlation_validation(chol_independent, chol_factors)
chol_control_off = [
    row for row in chol_control if row["row_parameter"] != row["column_parameter"]
]
chol_control_worst = max(abs(row["delta"]) for row in chol_control_off)
chol_control_seen = max(abs(row["realized_correlation"]) for row in chol_control_off)
print(
    f"independent draws: max realised |r| = {chol_control_seen:.3f}, "
    f"max |realised − target| = {chol_control_worst:.3f}"
)

# %% [markdown]
# Independent draws reproduce the marginals and lose everything else: the
# strongest real dependence, log P₀ × SNR at r = +0.94 on 10 µm, comes back as
# noise around zero. A classifier trained on that would learn that a loud
# particle can be a quiet one, which is precisely the confusion the real
# instrument does not have.

# %%
chol_correlated, chol_rejections = generate_parameters(
    chol_targets,
    chol_factors,
    seed=SEED,
    budgets=chol_budgets,
    dataset_id=CHOL_SYNTHETIC_ID + "@v2",
)
chol_demo = "10um"


def chol_columns(records, name):
    selected = [row for row in records if row["class_name"] == name]
    return (
        np.array([row["log_amplitude_p0"] for row in selected]),
        np.array([row["snr_db"] for row in selected]),
    )


chol_real_matrix = transformed_parameter_matrix(
    rows_for_population(chol_eligible, chol_demo, "physical")
)
chol_fig, chol_axes = plt.subplots(1, 3, figsize=(13, 3.9), sharex=True, sharey=True)
plot_dependence_scatter(
    [
        (
            f"measured · {CHOL_CLASS_LABEL[chol_demo]} "
            f"(n = {chol_real_matrix.shape[0]:,})",
            chol_real_matrix[:, 0],
            chol_real_matrix[:, 3],
            CLASS_COLOUR[chol_demo],
        ),
        ("independent draws", *chol_columns(chol_independent, chol_demo), "#9ca3af"),
        ("Cholesky draws", *chol_columns(chol_correlated, chol_demo), "#0f766e"),
    ],
    axes=chol_axes,
)
chol_fig.suptitle(
    "Amplitude against SNR: the dependence the Cholesky factor restores",
    fontsize=11,
)
chol_fig.tight_layout()
chol_fig

# %% [markdown]
# ### The delta triangles: what was realised minus what was asked
#
# The construction is exact in expectation, not in a finite sample, and the
# generator adds one step that breaks it outright: proposals whose frequency
# falls outside the instrument's 7–80 kHz acceptance band are rejected and
# redrawn. Rejection sampling is a non-linear filter on a Gaussian copula, so
# the realised correlations *must* drift. The honest check is therefore not
# whether they drift, but by how much.
#
# The cell re-runs the promoted generator's parameter draw at seed 20260723 and
# compares its realised correlations against its targets — the same table the
# published run wrote to `correlation_validation.csv`.

# %%
chol_validation = correlation_validation(chol_correlated, chol_factors)
with (run_dir(CHOL_GENERATION_RUN) / "correlation_validation.csv").open(
    newline="", encoding="utf-8"
) as handle:
    chol_published_delta = {
        (row["class_name"], row["row_parameter"], row["column_parameter"]): float(
            row["delta"]
        )
        for row in csv.DictReader(handle)
    }
chol_delta_drift = max(
    abs(
        row["delta"]
        - chol_published_delta[
            (row["class_name"], row["row_parameter"], row["column_parameter"])
        ]
    )
    for row in chol_validation
)
assert chol_delta_drift == 0.0, f"reproduction drifted by {chol_delta_drift:.3e}"
print(f"reproduces {CHOL_GENERATION_RUN} exactly ({len(chol_correlated):,} events)")

chol_deltas = {name: np.zeros((4, 4)) for name in CLASS_ORDER}
for row in chol_validation:
    chol_deltas[row["class_name"]][
        PARAMETER_ORDER.index(row["row_parameter"]),
        PARAMETER_ORDER.index(row["column_parameter"]),
    ] = row["delta"]
chol_worst_delta = max(
    abs(row["delta"])
    for row in chol_validation
    if row["row_parameter"] != row["column_parameter"]
)
chol_figure_metrics = published(
    "ssl-v18-dependence-delta-visuals-r1", "figure_metrics.json"
)
assert (
    abs(
        chol_worst_delta
        - chol_figure_metrics["maximum_absolute_off_diagonal_delta_4d"]
    )
    < 1e-15
)
print(f"max off-diagonal |Δ| = {chol_worst_delta:.4f}, matching the deck figure run")

# %%
chol_fig, chol_axes = plt.subplots(1, 3, figsize=(13, 3.9))
plot_delta_triangles(chol_deltas, chol_budgets, axes=chol_axes)
chol_fig.suptitle(
    "Realised − target Pearson r of the 4-D generator (colour saturates at ±0.15)",
    fontsize=11,
)
chol_fig

# %% [markdown]
# The acceptance threshold the pipeline set for itself is 0.10; the run finished
# with `status: warning_correlation_delta_above_threshold` and was promoted
# anyway. The cell below tests the obvious explanation — that the two classes
# which exceeded the threshold are the two that the frequency band rejects most
# heavily — and separates it from the sampling error a finite draw carries in
# any case, using the standard error of a Pearson coefficient,
# (1 − r²)/√(n − 1).

# %%
print(f"{'class':6s} {'n':>6s} {'rejected':>9s} {'rate':>7s} {'max|Δ|':>8s} {'z':>6s}")
chol_mechanism = {}
for chol_name in CLASS_ORDER:
    count = chol_budgets[chol_name]
    rejected = chol_rejections[chol_name]
    rate = rejected / (rejected + count)
    worst, worst_z = 0.0, 0.0
    for row in chol_validation:
        if row["class_name"] != chol_name:
            continue
        if row["row_parameter"] == row["column_parameter"]:
            continue
        target = row["target_correlation"]
        error = (1.0 - target**2) / math.sqrt(count - 1)
        if abs(row["delta"]) > worst:
            worst, worst_z = abs(row["delta"]), abs(row["delta"]) / error
    chol_mechanism[chol_name] = {
        "rejection_rate": rate,
        "max_abs_delta": worst,
        "sampling_error_multiples": worst_z,
    }
    print(
        f"{chol_name:6s} {count:6,d} {rejected:9,d} {100 * rate:6.1f}% "
        f"{worst:8.3f} {worst_z:6.1f}"
    )

# %% [markdown]
# The reading is mechanistic, not statistical. 4 µm rejects 1.2 % of its
# proposals and lands within 0.030 of every target — inside two standard errors.
# 2 µm rejects 35 % (its frequency envelope is centred at 15.5 kHz with a 15.7
# kHz spread, so a third of it falls below the 7 kHz band edge) and misses by
# 0.123, which is six times its own sampling error: that is the band cut
# reshaping the joint, not bad luck. 10 µm misses by the headline 0.147 but on
# only 366 events, under three standard errors, so for that class small-sample
# noise is a sufficient explanation on its own.
#
# The practical consequence is that the deltas are *reducible by generating more
# events for 10 µm* and *not reducible that way for 2 µm*. The next cell checks
# that prediction against the dataset that was actually shipped downstream.

# %% [markdown]
# ### What the shipped dataset carries
#
# The 4,798-event run above is a pilot. The promoted 4-D white-noise dataset is
# `...-synthetic-events@v3`, which extends it to 47,980 events with the same
# targets, the same factors and the same policy, under a second seed. Its
# realised correlations are a property of the dataset the downstream sections
# consume, and no published run states them.

# %%
chol_v3_root = dataset_root(CHOL_SYNTHETIC_ID + "@v3")
chol_v3_summary = json.loads((chol_v3_root / "dataset_summary.json").read_text())
assert chol_v3_summary["sealed_test_accessed"] is False
assert chol_v3_summary["parameter_policy"]["correlations"] == chol_populations_used
assert list(chol_v3_summary["parameter_policy"]["coordinates"]) == list(PARAMETER_ORDER)

with (chol_v3_root / "events.csv").open(newline="", encoding="utf-8") as handle:
    chol_v3_records = [
        {
            "class_name": row["class_name"],
            "log_amplitude_p0": float(row["log_amplitude_p0"]),
            "frequency_khz": float(row["frequency_khz"]),
            "log_tau_ms": float(row["log_tau_ms"]),
            "snr_db": float(row["snr_db"]),
            "amplitude_p0": float(row["amplitude_p0"]),
            "tau_ms": float(row["tau_ms"]),
            "phi_rad": float(row["phi_rad"]),
        }
        for row in csv.DictReader(handle)
    ]
chol_v3_counts = {
    name: sum(1 for row in chol_v3_records if row["class_name"] == name)
    for name in CLASS_ORDER
}
chol_v3_validation = correlation_validation(chol_v3_records, chol_factors)
chol_v3_by_class = {}
for chol_name in CLASS_ORDER:
    chol_v3_by_class[chol_name] = max(
        abs(row["delta"])
        for row in chol_v3_validation
        if row["class_name"] == chol_name
        and row["row_parameter"] != row["column_parameter"]
    )
print(f"@v3: {len(chol_v3_records):,} events {chol_v3_counts}")
for chol_name in CLASS_ORDER:
    print(
        f"  {chol_name:5s} max |Δ|: pilot "
        f"{chol_mechanism[chol_name]['max_abs_delta']:.4f}"
        f"  →  shipped {chol_v3_by_class[chol_name]:.4f}"
    )
chol_v3_worst = max(chol_v3_by_class.values())
print(f"  overall max |Δ| = {chol_v3_worst:.4f} (pilot: {chol_worst_delta:.4f})")

chol_synth_f = np.array([row["frequency_khz"] * 1000.0 for row in chol_v3_records])
chol_synth_on_grid = float(
    np.mean(
        np.abs(chol_synth_f / chol_grid_hz - np.round(chol_synth_f / chol_grid_hz))
        < 1e-9
    )
)
print(
    f"\nf_D grid: measured {chol_real_f.size} distinct values, "
    f"{100 * chol_on_grid:.0f}% on the {chol_grid_hz:.2f} Hz grid; "
    f"synthetic {np.unique(chol_synth_f).size:,} distinct values, "
    f"{100 * chol_synth_on_grid:.0f}% on it"
)

# %% [markdown]
# So the frequency quantisation held: the real table has 65 frequencies, all on
# the grid; the synthetic table has 47,980, none of them. A classifier handed
# both parameter tables would separate them on that column alone and learn
# nothing about particles.
#
# The honest reading is narrower than it sounds. 488.28 Hz *is* the frequency
# resolution of the 4096-sample window the estimator reads, so an off-grid
# generated frequency is not observable *through that estimator* — re-measured,
# a synthetic waveform should snap back onto the same grid, and downstream
# models consume waveforms rather than this table.
#
# **Limit.** That last step is an argument from the transform length, not a
# measurement. Settling it means running the P2SNR frequency estimator over the
# synthetic waveforms and checking where its output lands. This section does not
# do it — the estimator is upstream of everything here — and the claim stays at
# "the parameter tables are trivially separable; whether the waveforms are is
# untested."

# %% [markdown]
# The prediction holds. Ten times the events cut 10 µm from 0.147 to 0.051 and
# 4 µm from 0.030 to 0.011, while 2 µm barely moved — 0.123 to 0.116 — because
# its miss is a band-rejection bias, which more sampling cannot remove. The
# shipped dataset's worst dependence error is therefore 0.116 on 2 µm, still
# above the 0.10 threshold, and the deck's 0.147 headline belongs to the pilot
# rather than to anything downstream consumes.

# %% [markdown]
# ### What the parameters sound like at the edges
#
# Everything above is matrices. The generator's actual product is a waveform,
# and the honest place to look at one is not the middle of the distribution —
# any construction looks plausible there — but the **edges of the domain it
# samples**, where a wrong dependence or an over-wide envelope would show up as
# something that could not be a particle crossing.
#
# The role selection is the generator run's own, imported rather than invented:
# `select_gallery_indices` picks, per class, the most central joint draw and
# then the 5th/95th percentile events on frequency, SNR, amplitude and τ, each
# forced to be a distinct event. It is deterministic, and it is fixed before any
# waveform is looked at — which is what stops a gallery from becoming a curated
# argument.

# %%
from particles2snr.z8_cholesky_generation import select_gallery_indices

chol_gallery = select_gallery_indices(chol_v3_records)
chol_signals = np.load(chol_v3_root / "signals_raw_4096.npy", mmap_mode="r")
assert chol_signals.shape[0] == len(chol_v3_records)
chol_gallery_index = [
    index for entries in chol_gallery.values() for _, index in entries
]
chol_gallery_signals = np.asarray(chol_signals[chol_gallery_index], dtype=np.float64)


def chol_join_shortfall(offset=0):
    """How far each selected waveform's peak falls short of its stated P₀."""
    worst = 0.0
    for position, index in enumerate(chol_gallery_index):
        record = chol_v3_records[(index + offset) % len(chol_v3_records)]
        peak = float(np.abs(chol_gallery_signals[position]).max())
        worst = max(worst, (record["amplitude_p0"] - peak) / peak)
    return worst


chol_join_error = chol_join_shortfall()
chol_join_control = chol_join_shortfall(offset=1)
print(
    f"{chol_signals.shape[0]:,} × {chol_signals.shape[1]} raw samples at "
    f"{SAMPLING_HZ / 1e6:.0f} MHz"
)
print(f"  aligned join   : worst P₀ shortfall {100 * chol_join_error:5.1f}%")
print(f"  join shifted +1: worst P₀ shortfall {100 * chol_join_control:5.1f}%")
for chol_name, entries in chol_gallery.items():
    roles = ", ".join(f"{role}→{index}" for role, index in entries)
    print(f"  {chol_name:5s} {roles}")

# %% [markdown]
# The join deserves its own check, because nothing in the dataset format
# guarantees that `events.csv` row *i* is `signals_raw_4096.npy` row *i* — it is
# a convention of the writer. The test is that each waveform's observed peak
# should reach the P₀ its own row claims. It can fall a little short when the
# carrier phase φ places no sample exactly at the envelope maximum, and the
# worst aligned case does, by 7 %. Shifting the join by a single row sends that
# shortfall past 1,200 % — a row whose stated P₀ is thirteen times the peak of
# the waveform it was paired with. That two-order-of-magnitude separation is
# what makes the aligned figure believable rather than merely unrefuted.

# %%
chol_local = {index: position for position, index in enumerate(chol_gallery_index)}
chol_fig, chol_axes = plt.subplots(
    3, 6, figsize=(19, 8.4), sharex=True, constrained_layout=True
)
plot_signal_gallery_cholesky(
    {
        name: [(role, chol_local[index]) for role, index in entries]
        for name, entries in chol_gallery.items()
    },
    [chol_v3_records[index] for index in chol_gallery_index],
    chol_gallery_signals,
    SAMPLING_HZ,
    axes=chol_axes,
)
chol_fig.suptitle(
    "Generated waveforms at the edges of the sampled domain "
    "(blue: synthetic event · orange: ±Gaussian envelope)",
    fontsize=12,
)
chol_fig

# %% [markdown]
# Every panel is still a damped oscillation under a Gaussian envelope, which is
# the claim. The extremes are informative in a way the deck's version does not
# say out loud: the low-SNR panels are events the *detector* would very likely
# miss — the 2 µm low-SNR draw sits at −12.2 dB, below anything a detection
# threshold tuned on real data would catch — and the high-amplitude 10 µm draw
# reaches SNR +26.5 dB against a measured 10 µm maximum of +22.3 dB.
#
# That is not a defect. It is the widening working as designed, and it is worth
# measuring rather than asserting: the Gaussian envelopes are deliberately
# broader than the measured marginals so the synthetic cloud *contains* the real
# one instead of merely resembling it.

# %%
chol_widening = {}
for chol_name in CLASS_ORDER:
    physical = transformed_parameter_matrix(
        rows_for_population(chol_eligible, chol_name, "physical")
    )
    inclusive = transformed_parameter_matrix(
        rows_for_population(chol_eligible, chol_name, "inclusive")
    )
    measured = np.array(
        [
            physical[:, 0].std(ddof=1),
            physical[:, 1].std(ddof=1),
            physical[:, 2].std(ddof=1),
            inclusive[:, 3].std(ddof=1),
        ]
    )
    ratio = np.asarray(chol_targets[chol_name]["sigma"]) / measured
    chol_widening[chol_name] = {
        parameter: float(value) for parameter, value in zip(PARAMETER_ORDER, ratio)
    }
    print(
        f"{chol_name:5s} envelope σ / measured σ: "
        + "  ".join(f"{parameter.split('_')[0]} {value:.2f}×"
                    for parameter, value in zip(("logP0", "f", "logtau", "SNR"), ratio))
    )
chol_widening_range = (
    min(v for row in chol_widening.values() for v in row.values()),
    max(v for row in chol_widening.values() for v in row.values()),
)
print(
    f"\nwidening spans {chol_widening_range[0]:.2f}× to "
    f"{chol_widening_range[1]:.2f}× across all classes and coordinates"
)

chol_outside = {}
for chol_name in CLASS_ORDER:
    bounds = chol_board["observed_ranges"][chol_name]["ranges"]
    generated = [row for row in chol_v3_records if row["class_name"] == chol_name]
    row = {}
    for parameter, column in (
        ("amplitude_p0", "amplitude_p0"),
        ("frequency_khz", "frequency_khz"),
        ("tau_ms", "tau_ms"),
        ("snr_effective_fbase_db", "snr_db"),
    ):
        series = np.array([entry[column] for entry in generated])
        low, high = bounds[parameter]["minimum"], bounds[parameter]["maximum"]
        row[parameter] = float(np.mean((series < low) | (series > high)))
    chol_outside[chol_name] = row
    print(
        f"{chol_name:5s} generated outside the measured range: "
        + "  ".join(
            f"{key.split('_')[0]} {100 * value:4.1f}%" for key, value in row.items()
        )
    )

# %% [markdown]
# ### What this section does not claim
#
# - **The marginals are not reproduced; they are deliberately widened**, by the
#   factors measured just above. Judging the generator by how closely its
#   histograms match the real histograms is therefore the wrong test — it would
#   penalise the widening that is the point. Whether the widened cloud actually
#   *contains* the real one is the right test, and it belongs to the coverage
#   section.
# - **The gallery is a plausibility check, not a validation.** Six roles per
#   class, selected by a fixed rule, shown as waveforms. It establishes that the
#   construction does not break down at the edges of its own domain; it says
#   nothing about whether those waveforms would fool the detector, the encoder,
#   or a person. Those are separate measurements in later sections.
# - **Everything here is development evidence.** The dependence targets, the
#   envelopes and the deltas are all estimated on the development split of one
#   dataset. No sealed test row was read — the cell at the top raises if one
#   appears — and nothing here validates the generator on unseen events.
# - **Correlation is all the dependence that is modelled.** A Gaussian copula
#   reproduces linear dependence in the transformed coordinates and nothing
#   else. Any curved or tail dependence in the real cloud is silently discarded,
#   and no measurement in this section would detect it.
# - **Parameters, not signals.** This section stops at the four numbers. Turning
#   them into waveforms — white noise for `@v3`, real recorded noise carriers
#   for `@v4` — is a separate step, and the fifth coordinate, waveform
#   asymmetry, extends the same construction from 4-D to 5-D in `@v5`. That is
#   the next section.

# %%
try:
    chol_emitted = notebook_evidence.emit_run(
        workspace,
        section="cholesky-generator-audit",
        metrics={
            "schema_version": 1,
            "analysis": "deck dependence-target transcription audit and shipped 4-D "
            "generator dependence error",
            "deck_transcription_audit": {
                name: {
                    "deck_event_count": entry["deck_n"],
                    "generator_population": entry["generator_population"],
                    "generator_event_count": entry["generator_n"],
                    "max_deck_minus_physical_r": entry["max_deck_minus_physical"],
                    "max_deck_minus_generator_target_r": entry[
                        "max_deck_minus_generator_target"
                    ],
                    "worst_pair": list(entry["worst_pair"]),
                }
                for name, entry in chol_audit.items()
            },
            "pilot_dependence_error": chol_mechanism,
            "shipped_v3_dependence_error": {
                "event_count": len(chol_v3_records),
                "class_counts": chol_v3_counts,
                "max_absolute_off_diagonal_delta_by_class": chol_v3_by_class,
                "max_absolute_off_diagonal_delta": chol_v3_worst,
            },
            "independent_marginal_control": {
                "max_absolute_off_diagonal_delta": chol_control_worst,
                "max_absolute_realized_correlation": chol_control_seen,
            },
            "envelope_widening_ratio": chol_widening,
            "generated_outside_measured_range_fraction": chol_outside,
            "frequency_quantisation": {
                "grid_hz": chol_grid_hz,
                "measured_distinct_values": int(chol_real_f.size),
                "measured_fraction_on_grid": chol_on_grid,
                "synthetic_distinct_values": int(np.unique(chol_synth_f).size),
                "synthetic_fraction_on_grid": chol_synth_on_grid,
            },
        },
        provenance={
            "datasets": dataset_provenance(),
            "inputs": {
                "correlation_run": CHOL_CORRELATION_RUN,
                "distribution_run": CHOL_DISTRIBUTION_RUN,
                "envelope_run": CHOL_ENVELOPE_RUN,
                "generation_run": CHOL_GENERATION_RUN,
                "approved_population": chol_population,
                "deck_recipe": "src/internship_workspace/presentation/recipes/"
                "pearson_targets.py",
            },
            "parameters": {
                "seed": SEED,
                "class_budgets": chol_budgets,
                "transformed_coordinates": list(PARAMETER_ORDER),
                "dependency_population_by_class": chol_populations_used,
            },
            "metric_definitions": {
                "max_deck_minus_physical_r": "largest absolute gap between a deck "
                "constant and the correlation run's physical-population Pearson r",
                "max_deck_minus_generator_target_r": "largest absolute gap between a "
                "deck constant and the Pearson r the generator was actually given",
                "max_absolute_off_diagonal_delta": "largest absolute realised-minus-"
                "target Pearson r over off-diagonal cells and classes",
                "sampling_error_multiples": "max |delta| divided by (1 - r^2)/"
                "sqrt(n - 1) at the target r",
                "envelope_widening_ratio": "fitted Gaussian envelope sigma divided "
                "by the measured standard deviation, per transformed coordinate",
                "generated_outside_measured_range_fraction": "share of generated "
                "events falling outside the measured per-class observed range",
                "frequency_quantisation": "share of Doppler frequencies that are "
                "exact multiples of the 4096-point transform bin at 2 MHz",
            },
        },
        claim_boundary=(
            "Audits the deck's hard-coded dependence targets against the manifested "
            "correlation run, and measures the realised-minus-target Pearson error of "
            "the shipped 4-D synthetic dataset. Development split only; no sealed test "
            "row is read. It does not validate signal realism, marginal coverage, or "
            "any 5-D construction."
        ),
    )
    print(f"emitted {chol_emitted.name}")
except WorkspaceError as chol_error:
    print(f"no evidence emitted ({chol_error})")


# %% [markdown]
# ## The fifth knob: waveform asymmetry
#
# Everything up to here has generated events from four numbers — amplitude
# $P_0$, carrier frequency $f$, envelope width $\tau$ and signal-to-noise ratio
# — and drawn the pulse itself from one fixed shape: a Gaussian envelope,
# symmetric about its own centre, multiplying a cosine. Those four numbers say
# how big, how fast, how long and how buried an event is. None of them says
# what it *looks like*.
#
# This section asks whether the fixed shape is the right one, and follows the
# answer to its end: a measurable skew of the envelope, an estimator that reads
# it off one noisy event, a calibration that says how far to trust that reading,
# and a fifth generated coordinate that makes synthetic events carry it. It also
# stops, at the point where the chain crossed a gate it had set for itself, and
# measures what that cost.
#
# **Vocabulary.** *Envelope*: the slowly varying amplitude that the fast
# oscillation rides on. *Skew* or *asymmetry*: the envelope rising and falling
# at different rates. *Anchor*: a real event whose skew was measured and kept,
# used as a target for generation. *Injection*: a synthetic event built with a
# skew we chose, so the estimator can be graded against a known truth.

# %%
import collections
import hashlib
from datetime import datetime

from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.stats import chi2, mannwhitneyu

from internship_workspace.z8_parametric_asymmetry import (
    ParametricAsymmetryConfig,
    asymmetric_gaussian_cosine,
    fit_parametric_asymmetry,
    inject_asymmetry_into_noise_carrier,
)

ASYMMETRY_BOUND = 0.8
RAW_LENGTH = 4096

signal_root = dataset_root("particles2snr-f-dual-clean-c1-yolo-4class@v2")
z8_root = dataset_root(
    "particles2snr-fbase-dual-clean-z8-events-3class-plus-unclear-development@v2"
)
v4_root = dataset_root(
    "particles2snr-fbase-z8-cholesky-physicalcorr-effective-snr-synthetic-events@v4"
)
v5_root = dataset_root(
    "particles2snr-fbase-z8-cholesky-physicalcorr-effective-snr-synthetic-events@v5"
)


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


z8_events = [row for row in read_rows(z8_root / "events.csv") if row["class_name"] in CLASS_ORDER]
if any(row["split"] == "test" for row in z8_events):
    raise RuntimeError("sealed test rows reached the asymmetry section")
print(f"real Z8 events, three physical classes: {len(z8_events)}")
print("split census:", dict(collections.Counter(row["split"] for row in z8_events)))
print("class census:", {name: sum(r["class_name"] == name for r in z8_events) for name in CLASS_ORDER})

# %% [markdown]
# ### 1. A symmetric envelope is the wrong shape
#
# The claim to test is narrow and checkable: *real acoustic events have
# envelopes that rise and fall at the same rate*. That is what the four-parameter
# generator assumes, and it can be falsified without any estimator, any fit and
# any new parameter — which matters, because an estimator that looks for skew
# will report some skew on pure noise, and we would learn nothing.
#
# The model-free statistic is the third moment of the event's own energy in
# time. Take the analytic envelope $e(t)$ of the band-passed trace — the
# instantaneous amplitude, from the Hilbert transform — weight time by $e(t)^2$
# inside a window of $\pm 3\tau$ around the peak, and compute the standardised
# third moment $\gamma$. A symmetric envelope gives $\gamma = 0$; a longer
# right-hand tail gives $\gamma > 0$. Nothing is fitted.

# %%
SAMPLING_INTERVAL_S = 1.0 / SAMPLING_HZ
bandpass = butter(4, [7_000.0, 80_000.0], btype="bandpass", fs=SAMPLING_HZ, output="sos")


def envelope_skew(values, tau_s, *, half_width=3.0, recentre=True):
    """Energy-weighted third moment of the analytic envelope. No fit involved."""
    envelope = np.abs(hilbert(np.asarray(values, dtype=np.float64)))
    count = envelope.size
    time_s = (np.arange(count) - (count - 1) / 2.0) * SAMPLING_INTERVAL_S
    centre = 0.0
    if recentre:
        span = int(max(8, round(0.5 * tau_s * SAMPLING_HZ)))
        smoothed = np.convolve(np.square(envelope), np.ones(2 * span + 1) / (2 * span + 1), mode="same")
        centre = float(time_s[int(np.argmax(smoothed))])
    inside = np.abs(time_s - centre) <= half_width * tau_s
    if int(inside.sum()) < 32:
        return np.nan
    weight = np.square(envelope[inside])
    axis = time_s[inside]
    total = float(weight.sum())
    mean = float((weight * axis).sum() / total)
    deviation = axis - mean
    variance = float((weight * deviation**2).sum() / total)
    if variance <= 0.0:
        return np.nan
    return float((weight * deviation**3).sum() / total / variance**1.5)


for probe in (-0.4, -0.2, 0.0, 0.2, 0.4):
    waveform = asymmetric_gaussian_cosine(
        RAW_LENGTH, frequency_hz=25_000.0, tau_s=4.0e-5, asymmetry=probe,
        sampling_frequency_hz=SAMPLING_HZ,
    )
    print(f"noiseless model with a = {probe:+.1f} → γ = {envelope_skew(waveform, 4.0e-5):+.4f}")

# %% [markdown]
# The statistic tracks the sign and, near zero, roughly $1.4$ times the
# magnitude of the model's own skew parameter, so it is a fair detector of skew
# even though it is not a measurement of it. Now run it on every real event, and
# on the events the four-parameter generator produced — which are symmetric by
# construction, on real recorded noise, at the same signal-to-noise. That
# synthetic population is the null: whatever spread it shows in $\gamma$ is what
# noise alone buys.

# %%
def reflect_crop(values, centre, length=RAW_LENGTH):
    """The 4,096-sample window the analyses use, reflected at file edges."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    start = int(round(float(centre))) - length // 2
    if start >= 0 and start + length <= array.size:
        return array[start : start + length].copy()
    padded = np.pad(array, (length, length), mode="reflect")
    return padded[start + length : start + length + length].copy()


started = time.time()
signal_cache = {}
real_skew = collections.defaultdict(list)
for row in z8_events:
    relative = row["source_signal_relative_path"]
    if relative not in signal_cache:
        signal_cache[relative] = np.load(signal_root / relative, allow_pickle=False)
    crop = reflect_crop(signal_cache[relative], float(row["center_sample"]))
    real_skew[row["class_name"]].append(envelope_skew(crop, float(row["tau_ms"]) / 1000.0))
signal_cache.clear()

v4_events = read_rows(v4_root / "events.csv")
v4_raw = np.load(v4_root / "signals_raw_4096.npy", mmap_mode="r", allow_pickle=False)
v4_index = collections.defaultdict(list)
for position, row in enumerate(v4_events):
    v4_index[row["class_name"]].append(position)

null_generator = np.random.default_rng(20260815)
null_skew = collections.defaultdict(list)
for name in CLASS_ORDER:
    for position in null_generator.choice(v4_index[name], 800, replace=False):
        filtered = sosfiltfilt(bandpass, np.asarray(v4_raw[int(position)], dtype=np.float64))
        null_skew[name].append(envelope_skew(filtered, float(v4_events[int(position)]["tau_ms"]) / 1000.0))

real_skew = {name: np.asarray(real_skew[name])[np.isfinite(real_skew[name])] for name in CLASS_ORDER}
null_skew = {name: np.asarray(null_skew[name])[np.isfinite(null_skew[name])] for name in CLASS_ORDER}
skew_summary = {}
for name in CLASS_ORDER:
    observed, reference = real_skew[name], null_skew[name]
    band = float(np.quantile(np.abs(reference), 0.95))
    skew_summary[name] = {
        "real_count": int(observed.size),
        "null_count": int(reference.size),
        "real_standard_deviation": float(observed.std(ddof=1)),
        "null_standard_deviation": float(reference.std(ddof=1)),
        "null_absolute_q95": band,
        "real_fraction_beyond_null_q95": float(np.mean(np.abs(observed) > band)),
        "mann_whitney_p_value": float(mannwhitneyu(np.abs(observed), np.abs(reference)).pvalue),
    }
    values = skew_summary[name]
    print(
        f"{name:>5}: real sd {values['real_standard_deviation']:.3f} vs symmetric-generator sd "
        f"{values['null_standard_deviation']:.3f} · "
        f"{100 * values['real_fraction_beyond_null_q95']:.1f} % of real events beyond ±{band:.2f} "
        f"(5 % expected) · p = {values['mann_whitney_p_value']:.1e}"
    )
print(f"[{time.time() - started:.1f} s]")

# %%
plot_envelope_skew(real_skew, null_skew)
plt.show()

# %% [markdown]
# The real distributions are one and a half to two and a half times as wide as
# a symmetric envelope on the same noise can produce, and 12 to 21 % of real
# events land outside the null's central 90 %, against the 5 % chance allows.
# The symmetric envelope is not a harmless simplification: it is measurably the
# wrong shape, on all three classes, with p-values between $10^{-10}$ and
# $10^{-32}$.
#
# So why not stop here and generate from $\gamma$? Because $\gamma$ answers the
# wrong question. It conflates skew with a mis-centred window — the annotation
# centre is not the envelope peak, and re-centring on the peak moves the median
# $\gamma$ measurably while leaving the spread intact. It is unbounded, so it
# cannot be mapped onto a Gaussian coordinate the generator can sample. And it
# is not invertible: no waveform can be built from a value of $\gamma$. What is
# needed is a *parameter of the generating model* — one bounded number per
# event, fitted jointly with the centre offset, frequency and width so that a
# timing error cannot masquerade as skew.

# %% [markdown]
# ### 2. The parametric skew estimator
#
# The shipped model keeps the Gaussian-cosine and lets the width differ on the
# two sides of the peak:
#
# $$s(t) = A\,e^{-t^{2}/2\tau(t)^{2}}\cos(2\pi f t + \varphi),
#   \qquad \tau(t) = \tau\,e^{\operatorname{sgn}(t)\,a},
#   \qquad |a| \le 0.8 .$$
#
# One new number, $a$. It is a **log half-width ratio**: the falling side has
# width $\tau e^{a}$ and the rising side $\tau e^{-a}$, so the two sides stand
# in ratio $e^{2a}$. $a = 0$ is the old symmetric envelope; $a = 0.35$ makes the
# decay twice the rise; $a = 0.8$, the bound, makes it five times.
#
# The bound is not cosmetic. Three things need it. Beyond roughly $|a| = 0.8$
# one side of the pulse stops fitting inside the 4,096-sample analysis window,
# and the model starts trading skew against the linear baseline and drift terms
# rather than against the data. The generator needs a transform that maps $a$ to
# an unbounded Gaussian coordinate, and it uses $\operatorname{atanh}(a/0.8)$,
# which requires a finite bound. And a fit that walks into the bound is not a
# measurement of a very skewed event, it is a fit that failed to localise — so
# the real-event pipeline treats contact within 1 % of the bound as an
# abstention, not as $a = \pm 0.8$.
#
# Fitting is a one-dimensional search. At fixed $(a, \Delta t, f, \tau)$ the
# model is *linear* in amplitude-cosine, amplitude-sine, offset and drift, so
# those four are solved exactly by least squares and never searched. What is
# searched is $a$, on a 65-point grid over $[-0.8, 0.8]$ followed by a bounded
# refinement, minimising a pseudo-Huber loss — quadratic on small residuals,
# linear on large ones, so a single noise spike cannot buy skew.

# %%
frozen_config = ParametricAsymmetryConfig.from_dict(
    json.loads((run_dir("particle-z8-v2-parametric-asymmetry-confirmatory-r2") / "estimator_config.json").read_text())
)
shipped_defaults = ParametricAsymmetryConfig()
deployed_config = ParametricAsymmetryConfig(
    whitening_enabled=True, multistart_enabled=True, input_pre_filtered=True
)

for field in ("band_low_hz", "band_high_hz", "filter_order", "decimation",
              "asymmetry_grid_points", "asymmetry_bound", "pseudo_huber_scale"):
    assert getattr(frozen_config, field) == getattr(shipped_defaults, field), field
print("the calibrated configuration is the shipped default for every search parameter:")
print(f"  band-pass {frozen_config.band_low_hz / 1e3:.0f}–{frozen_config.band_high_hz / 1e3:.0f} kHz, "
      f"zero-phase order {frozen_config.filter_order} · decimate by {frozen_config.decimation} · "
      f"{frozen_config.asymmetry_grid_points}-point grid on a · |a| ≤ {frozen_config.asymmetry_bound}")

differences = {
    field: (getattr(frozen_config, field), getattr(deployed_config, field))
    for field in ("whitening_enabled", "multistart_enabled", "input_pre_filtered")
}
print("\nbut the configuration graded by that calibration is not the one run on real events:")
for field, (calibrated, deployed) in differences.items():
    print(f"  {field:<20} calibration {str(calibrated):<5} → real events {deployed}")
print("this gap is measured in section 3; it does not change the model, only the search.")

# %% [markdown]
# The three differences are forced, not arbitrary. Real events arrive already
# band-passed by the detection dataset, so filtering again would double-filter
# them (`input_pre_filtered`). Real noise is not white, so the residual is
# weighted by an estimate of the noise spectrum taken from the quiet edges of
# the window (`whitening`). And real events are harder, so the nuisance solve is
# restarted from several points (`multistart`). The consequence — that the
# headline recovery numbers were measured on a configuration that never ran on a
# real event — is a limit, and section 3 puts a number on it.
#
# Here is the estimator on one real event, with the same configuration the
# 5-D targets used.

# %%
targets_run = run_dir("particle-z8-v2-asymmetry-5d-targets-r4")
real_fits = read_rows(targets_run / "real_event_asymmetry.csv")
if any(row["split"] == "test" for row in real_fits):
    raise RuntimeError("sealed test rows reached the published real-event fits")
z8_by_id = {row["event_id"]: row for row in z8_events}

candidates = sorted(
    (row for row in real_fits
     if row["annotation_origin"] == "dual_clean_strict" and row["accepted"] == "True"
     and row["class_name"] == "4um" and abs(float(row["estimated_asymmetry"])) >= 0.30
     and float(row["confidence_probability"]) >= 0.90),
    key=lambda row: row["event_id"],
)
symmetric_config = ParametricAsymmetryConfig(
    whitening_enabled=True, multistart_enabled=True, input_pre_filtered=True,
    asymmetry_bound=1.0e-3,
)


def fit_real_event(row, config):
    event = z8_by_id[row["event_id"]]
    crop = reflect_crop(
        np.load(signal_root / event["source_signal_relative_path"], allow_pickle=False),
        float(event["center_sample"]),
    )
    return crop, fit_parametric_asymmetry(
        crop, initial_frequency_hz=float(event["frequency_hz"]),
        initial_tau_s=float(event["tau_ms"]) / 1000.0, config=config,
    )


print("candidate demonstration events, refitted here against the published value:")
exact = []
for row in candidates:
    _, attempt = fit_real_event(row, deployed_config)
    gap = abs(attempt.asymmetry - float(row["estimated_asymmetry"]))
    print(f"  {row['event_id']} published â {float(row['estimated_asymmetry']):+.5f} "
          f"→ refitted {attempt.asymmetry:+.5f} (gap {gap:.1e})")
    if gap < 1.0e-9:
        exact.append(row)
assert exact, "no candidate reproduced its published â"
published_fit = max(exact, key=lambda row: float(row["confidence_probability"]))
DEMO_EVENT = published_fit["event_id"]
demo = z8_by_id[DEMO_EVENT]

demo_crop, free_fit = fit_real_event(published_fit, deployed_config)
_, symmetric_fit = fit_real_event(published_fit, symmetric_config)
deviation = abs(free_fit.asymmetry - float(published_fit["estimated_asymmetry"]))
assert deviation < 1.0e-9, f"reproduction drifted by {deviation:.3e}"

ratio = float(np.exp(2 * free_fit.asymmetry))
shape = (f"decay {ratio:.1f}× the rise" if ratio >= 1.0 else f"rise {1 / ratio:.1f}× the decay")
print(f"\nchosen: {DEMO_EVENT} · {demo['class_name']} · split {demo['split']} · "
      f"SNR {float(demo['snr_db']):.1f} dB · confidence {float(published_fit['confidence_probability']):.3f}")
print(f"reproduces particle-z8-v2-asymmetry-5d-targets-r4 exactly: â = {free_fit.asymmetry:.12f} "
      f"(gap {deviation:.1e})")
print(f"half-width ratio implied: e^(2â) = {ratio:.2f} — {shape}")
print(f"robust residual: symmetric model {symmetric_fit.objective:.4e} → free model "
      f"{free_fit.objective:.4e}, a drop of "
      f"{100 * (1 - free_fit.objective / symmetric_fit.objective):.1f} %")
print("\nsome candidates do not reproduce to the last digit; section 3 measures why.")

# %%
def reconstruct(fit):
    """The fitted model in the observed domain, from the shipped closed form."""
    core = asymmetric_gaussian_cosine(
        RAW_LENGTH, frequency_hz=fit.frequency_hz, tau_s=fit.tau_s, asymmetry=fit.asymmetry,
        amplitude=fit.amplitude, phase_rad=fit.phase_rad,
        center_offset_samples=fit.center_offset_samples, sampling_frequency_hz=SAMPLING_HZ,
    )
    axis = (np.arange(RAW_LENGTH) - (RAW_LENGTH - 1) / 2.0) / RAW_LENGTH
    return core + fit.offset + fit.drift * axis


demo_time_us = (np.arange(RAW_LENGTH) - (RAW_LENGTH - 1) / 2.0) / SAMPLING_HZ * 1e6
plot_event_fit(
    demo_time_us, demo_crop, reconstruct(free_fit), reconstruct(symmetric_fit),
    label=f"{DEMO_EVENT} · {demo['class_name']} · â = {free_fit.asymmetry:+.3f} ({shape})",
)
plt.show()

# %% [markdown]
# The symmetry-constrained model has to split the difference between a fast
# rise and a slow decay, and leaves an antisymmetric residual around the peak —
# it under-predicts on one flank and over-predicts on the other. Releasing one
# number removes it. That is the whole argument for the fifth coordinate, on one
# event; sections 3 and 4 turn it into a population.

# %% [markdown]
# ### 3. Calibration: how far can $\hat{a}$ be trusted?
#
# No real event carries a known skew, so the estimator cannot be graded on real
# data. It is graded where the truth is chosen. The protocol, from
# `tools/benchmarks/calibrate_z8_parametric_asymmetry.py`, takes a synthetic
# event, subtracts the symmetric waveform it was built from — leaving its real
# recorded noise carrier — rebuilds the *same* event with a chosen skew at the
# *same* signal-to-noise, and runs the estimator blind. 768 injections, 256 per
# class, truths spread over $[-0.6, +0.6]$, on noise sources disjoint from the
# ones used to develop the estimator. No real waveform enters the loop:
# the run records `real_z8_data_read: false`.

# %%
confirmatory = run_dir("particle-z8-v2-parametric-asymmetry-confirmatory-r2")
calibration_rows = [row for row in read_rows(confirmatory / "calibration_rows.csv") if row["fit_valid"] == "True"]
published_metrics = published("particle-z8-v2-parametric-asymmetry-confirmatory-r2")
published_r_squared = published("ssl-v18-asymmetry-recovery-visuals-r1", "figure_metrics.json")["r_squared"]

recovery_panels = {}
calibration_check = {}
for name in CLASS_ORDER:
    selected = [row for row in calibration_rows if row["class_name"] == name]
    truth = np.asarray([float(row["true_asymmetry"]) for row in selected])
    estimate = np.asarray([float(row["estimated_asymmetry"]) for row in selected])
    snr = np.asarray([float(row["snr_db"]) for row in selected])
    error = np.abs(estimate - truth)
    determination = 1.0 - float(np.sum(np.square(truth - estimate)) / np.sum(np.square(truth - truth.mean())))
    strong = np.abs(truth) >= 0.20
    recomputed = {
        "r_squared": determination,
        "median_absolute_error": float(np.median(error)),
        "q95_absolute_error": float(np.quantile(error, 0.95)),
        "median_bias": float(np.median(estimate - truth)),
        "sign_accuracy_abs_truth_ge_0p2": float(np.mean(np.sign(truth[strong]) == np.sign(estimate[strong]))),
    }
    reference = published_metrics["classes"][name]
    for key in ("median_absolute_error", "q95_absolute_error", "median_bias", "sign_accuracy_abs_truth_ge_0p2"):
        gap = abs(recomputed[key] - reference[key])
        assert gap < 1.0e-12, f"{name} {key} drifted by {gap:.3e}"
    gap = abs(determination - published_r_squared[name])
    assert gap < 1.0e-12, f"{name} R² drifted by {gap:.3e}"
    recomputed["mean_absolute_error"] = float(error.mean())
    recomputed["fraction_error_above_0p15"] = float(np.mean(error > 0.15))
    calibration_check[name] = recomputed
    recovery_panels[name] = (truth, estimate, snr, determination)

print("reproduces particle-z8-v2-parametric-asymmetry-confirmatory-r2 and "
      "ssl-v18-asymmetry-recovery-visuals-r1 exactly\n")
print(f"{'class':>6} {'R²':>6} {'median|e|':>10} {'mean|e|':>9} {'q95|e|':>8} {'sign≥0.2':>9} {'|e|>0.15':>9}")
for name in CLASS_ORDER:
    values = calibration_check[name]
    print(f"{name:>6} {values['r_squared']:6.2f} {values['median_absolute_error']:10.3f} "
          f"{values['mean_absolute_error']:9.3f} {values['q95_absolute_error']:8.3f} "
          f"{100 * values['sign_accuracy_abs_truth_ge_0p2']:8.1f}% {100 * values['fraction_error_above_0p15']:8.1f}%")

# %% [markdown]
# The deck's three numbers check out — R² $0.65 / 0.94 / 0.84$, error
# $0.068 / 0.026 / 0.024$, sign right $93 / 99 / 98$ % — with two clarifications
# the deck does not make. The error figure is the **median** absolute error, not
# the mean; the mean is $0.125 / 0.049 / 0.070$, two to three times larger,
# because the error distribution has a long tail. And the sign accuracy is
# conditional on $|a| \ge 0.2$: it says nothing about events whose true skew is
# small, where sign is close to a coin toss by construction.
#
# The sharper point is what the run says about itself.

# %%
thresholds = published_metrics["thresholds"]
print("frozen acceptance thresholds and what the run measured:\n")
print(f"{'class':>6} {'median|e|≤0.05':>16} {'q95|e|≤0.15':>14} {'|bias|≤0.03':>13} "
      f"{'sign≥0.90':>11} {'pass':>6}")
for name in CLASS_ORDER:
    reference = published_metrics["classes"][name]
    print(f"{name:>6} {reference['median_absolute_error']:16.3f} {reference['q95_absolute_error']:14.3f} "
          f"{abs(reference['median_bias']):13.3f} {reference['sign_accuracy_abs_truth_ge_0p2']:11.3f} "
          f"{str(reference['calibration_pass']):>6}")
print(f"\nrun-level calibration_pass: {published_metrics['calibration_pass']}")
print(f"q95 threshold {thresholds['q95_absolute_error_maximum']} is exceeded by every class; "
      f"2 µm also misses the median gate {thresholds['median_absolute_error_maximum']}")

# %% [markdown]
# **The calibration gate did not pass.** Not on one class — on all three, and
# on the pre-declared 95th-percentile error in every case. The deck's slide title
# is "asymmetry recovery is reliable in the central trend", which is exactly and
# only what the medians support; the run's own verdict on the full distribution
# is `calibration_pass: false`. The estimator was carried forward anyway, behind
# a per-event confidence gate that abstains rather than a gate that passes. That
# is a defensible engineering decision and an indefensible thing to leave out of
# the slide.
#
# Where does the error live? Not where the deck's scope line suggests
# ("the weakly identifiable tails"), if that is read as large $|a|$.

# %%
plot_recovery(recovery_panels)
plt.show()

print(f"{'class':>6} {'SNR quartile':>14} {'n':>4} {'median|e|':>10} {'|e|>0.15':>9}")
snr_structure = {}
for name in CLASS_ORDER:
    truth, estimate, snr, _ = recovery_panels[name]
    error = np.abs(estimate - truth)
    edges = np.quantile(snr, [0.0, 0.25, 0.5, 0.75, 1.0])
    snr_structure[name] = []
    for index in range(4):
        upper = snr <= edges[index + 1] if index == 3 else snr < edges[index + 1]
        inside = (snr >= edges[index]) & upper
        snr_structure[name].append(
            {"quartile": index + 1, "low_db": float(edges[index]), "high_db": float(edges[index + 1]),
             "count": int(inside.sum()), "median_absolute_error": float(np.median(error[inside])),
             "fraction_above_0p15": float(np.mean(error[inside] > 0.15))}
        )
        record = snr_structure[name][-1]
        print(f"{name:>6} {f'Q{index + 1} [{edges[index]:+.0f},{edges[index + 1]:+.0f}) dB':>14} "
              f"{record['count']:4d} {record['median_absolute_error']:10.3f} "
              f"{100 * record['fraction_above_0p15']:8.1f}%")

# %% [markdown]
# The error is an SNR effect, essentially entirely. In the top signal-to-noise
# quartile no class ever misses by more than $0.15$; in the bottom quartile
# 2 µm misses by more than $0.15$ in **70 %** of injections. The class ranking in
# the headline R² is a restatement of the class ranking in signal-to-noise, not
# a property of particle size. That reframes what the confidence gate is doing:
# it is an SNR-shaped filter wearing fit-quality clothes, and every downstream
# use of the accepted population inherits that selection.
#
# Can the calibration be re-run from this notebook? Almost.

# %%
started = time.time()
v4_by_id = {row["sample_id"]: position for position, row in enumerate(v4_events)}
reproduction_generator = np.random.default_rng(20260816)
by_class = collections.defaultdict(list)
for row in calibration_rows:
    by_class[row["class_name"]].append(row)

reproduction = collections.defaultdict(list)
for name in CLASS_ORDER:
    pool = by_class[name]
    for choice in reproduction_generator.choice(len(pool), 48, replace=False):
        row = pool[int(choice)]
        position = v4_by_id[row["sample_id"]]
        donor = v4_events[position]
        injected = inject_asymmetry_into_noise_carrier(
            np.asarray(v4_raw[position]),
            frequency_hz=float(donor["frequency_khz"]) * 1000.0,
            tau_s=float(donor["tau_ms"]) / 1000.0,
            asymmetry=float(row["true_asymmetry"]),
            amplitude=float(donor["amplitude_p0"]),
            phase_rad=float(donor["phi_rad"]),
            target_snr_db=float(donor["snr_db"]),
        )
        refit = fit_parametric_asymmetry(
            injected, initial_frequency_hz=float(donor["frequency_khz"]) * 1000.0,
            initial_tau_s=float(donor["tau_ms"]) / 1000.0, config=frozen_config,
        )
        reproduction[name].append(
            (float(row["true_asymmetry"]), float(row["estimated_asymmetry"]), refit.asymmetry)
        )

reproduction_summary = {}
for name in CLASS_ORDER:
    truth, published_estimate, refitted = (np.asarray(column) for column in zip(*reproduction[name]))
    def determination(estimate):
        return 1.0 - float(np.sum(np.square(truth - estimate)) / np.sum(np.square(truth - truth.mean())))
    reproduction_summary[name] = {
        "count": int(truth.size),
        "maximum_per_event_drift": float(np.max(np.abs(published_estimate - refitted))),
        "fraction_drift_above_1e_3": float(np.mean(np.abs(published_estimate - refitted) > 1.0e-3)),
        "published_median_absolute_error": float(np.median(np.abs(published_estimate - truth))),
        "refitted_median_absolute_error": float(np.median(np.abs(refitted - truth))),
        "published_r_squared": determination(published_estimate),
        "refitted_r_squared": determination(refitted),
    }
    values = reproduction_summary[name]
    print(f"{name:>5}: per-event drift up to {values['maximum_per_event_drift']:.2e} "
          f"({100 * values['fraction_drift_above_1e_3']:.0f} % of events above 1e-3) · "
          f"median|e| {values['published_median_absolute_error']:.4f} → "
          f"{values['refitted_median_absolute_error']:.4f} · "
          f"R² {values['published_r_squared']:.4f} → {values['refitted_r_squared']:.4f}")
print(f"[{time.time() - started:.1f} s]")

# %% [markdown]
# **A named limit: the estimator reproduces as an instrument, not as a
# function.** Re-running the shipped estimator today on 144 of the published
# injections gives per-event values that differ by up to $5 \times 10^{-2}$ from
# the ones the run recorded, on a fifth to a quarter of events. The fits are
# perfectly deterministic on this machine — repeated calls agree to twelve
# digits — so the drift is environmental: the robust objective is shallow near
# its minimum, and floating-point differences in the linear-algebra backend flip
# which basin the nuisance optimiser settles in. What survives is the
# aggregate: on these subsamples R² moves by less than $0.003$ and the median
# error by less than $0.008$, the largest movement being on 2 µm, whose error
# distribution is the broadest. Every population statement in this section is
# safe; no per-event $\hat{a}$ should be treated as exact beyond about two
# decimals.
#
# Now the question section 1 left open: **how much of the real spread is skew,
# and how much is the estimator's own noise?** Answering it needs the error
# distribution of the configuration that actually ran on real events, which the
# confirmatory run does not provide — but a later domain-aligned recalibration
# does, on 1,920 injections, with the same confidence gate the real events pass.

# %%
domain_run = run_dir("particle-z8-v2-parametric-asymmetry-filtered-domain-r1")
variant_rows = [row for row in read_rows(domain_run / "variant_fit_rows.csv")
                if row["variant"] == "C_prewhitened_multistart"]
confidence_rows = {row["sample_id"]: row for row in read_rows(domain_run / "confidence_oof_rows.csv")}
confidence_threshold = published("particle-z8-v2-asymmetry-5d-targets-r4")["confidence_threshold"]

estimator_noise = {}
for name in CLASS_ORDER:
    selected = [row for row in variant_rows if row["class_name"] == name]
    accepted = [row for row in selected
                if float(confidence_rows[row["sample_id"]]["confidence_probability"]) >= confidence_threshold]
    error = np.asarray([float(row["signed_error"]) for row in accepted])
    magnitude = np.abs([float(row["true_asymmetry"]) for row in accepted])
    estimator_noise[name] = error
    small = error[magnitude < 0.15]
    print(f"{name:>5}: {len(accepted):4d}/{len(selected)} injections pass the same confidence gate · "
          f"error sd {error.std(ddof=1):.4f} (sd on |a|<0.15 only: {small.std(ddof=1):.4f}, n={small.size})")
print("\nthe error spread barely moves with the injected skew, so the pooled error "
      "distribution is a fair null for real events whose skew is concentrated near zero")

domain_metrics = published("particle-z8-v2-parametric-asymmetry-filtered-domain-r1")
deployment = json.loads((domain_run / "confidence_model.json").read_text())
deployment = deployment["deployment_threshold_from_all_oof_predictions"]["classes"]
operational = {}
print(f"\nand its own operational gate, at the threshold the real-event pipeline deploys:")
print(f"{'class':>6} {'coverage':>9} {'required':>9} {'q90|e|':>8} {'allowed':>8} {'meets gate':>11}")
for name in CLASS_ORDER:
    gate = domain_metrics["operational_gates"][name]
    measured = deployment[name]
    passes = bool(measured["coverage"] >= gate["minimum_coverage"]
                  and measured["q90_error"] <= gate["maximum_conditional_q90_error"])
    operational[name] = {"coverage": measured["coverage"], "minimum_coverage": gate["minimum_coverage"],
                         "q90_absolute_error": measured["q90_error"],
                         "maximum_q90_absolute_error": gate["maximum_conditional_q90_error"],
                         "meets_gate": passes}
    print(f"{name:>6} {measured['coverage']:9.3f} {gate['minimum_coverage']:9.2f} "
          f"{measured['q90_error']:8.3f} {gate['maximum_conditional_q90_error']:8.2f} {str(passes):>11}")
print(f"run-level development_success: {domain_metrics['development_success']}")

# %% [markdown]
# Two readings. The estimator that actually runs on real events has a signed
# error with standard deviation $0.103 / 0.058 / 0.071$ for 2 / 4 / 10 µm after
# the confidence gate — that is the yardstick section 4.1 measures the real
# spread against. And this recalibration, which is the one aligned with the
# deployed configuration, does not clear its own operational gate either: at the
# deployed threshold only 4 µm meets both the coverage and the conditional-error
# requirement, and the run records `development_success: false`. The fifth
# coordinate is therefore built on an instrument that two independent gates
# declined to certify, used behind an abstention rule. That is worth stating
# plainly on any slide that shows the number.

# %% [markdown]
# ### 4. The fifth coordinate, end to end
#
# #### 4.1 What the real events say
#
# The estimator ran on all 2,073 real events of the three physical classes.
# Events whose fit the confidence model distrusts, and fits that walked into the
# $|a| = 0.8$ bound, are abstentions rather than measurements — so the
# population that defines the target is the *conditional accepted* one, and the
# claim it supports is conditional too.

# %%
strict_accepted = collections.defaultdict(list)
for row in real_fits:
    if row["annotation_origin"] == "dual_clean_strict" and row["accepted"] == "True":
        strict_accepted[row["class_name"]].append(float(row["estimated_asymmetry"]))
real_asymmetry = {name: np.asarray(strict_accepted[name]) for name in CLASS_ORDER}

targets_metrics = published("particle-z8-v2-asymmetry-5d-targets-r4")
statistics = {entry["class_name"]: entry for entry in targets_metrics["statistics"]
              if entry["population"] == "strict"}
gaussian_targets = {entry["class_name"]: entry for entry in targets_metrics["gaussian_asymmetry_targets"]}

support = {}
deconvolution = {}
for name in CLASS_ORDER:
    observed = real_asymmetry[name]
    reference = statistics[name]
    assert observed.size == reference["accepted_count"], name
    for key, value in (("standard_deviation", observed.std(ddof=1)), ("median", np.median(observed)),
                       ("minimum", observed.min()), ("maximum", observed.max())):
        gap = abs(value - reference[key])
        assert gap < 1.0e-12, f"{name} {key} drifted by {gap:.3e}"
    support[name] = (float(observed.min()), float(observed.max()))
    noise_variance = float(estimator_noise[name].var(ddof=1))
    true_variance = max(observed.var(ddof=1) - noise_variance, 0.0)
    deconvolution[name] = {
        "accepted_count": int(observed.size),
        "requested_count": int(reference["requested_count"]),
        "coverage": float(reference["coverage"]),
        "observed_standard_deviation": float(observed.std(ddof=1)),
        "estimator_noise_standard_deviation": float(np.sqrt(noise_variance)),
        "skew_standard_deviation": float(np.sqrt(true_variance)),
        "variance_share_real_skew": float(true_variance / observed.var(ddof=1)),
        "gaussian_sigma_transformed": gaussian_targets[name]["gaussian_sigma_transformed"],
        "observed_support": support[name],
    }
print("reproduces particle-z8-v2-asymmetry-5d-targets-r4 exactly\n")
print(f"{'class':>6} {'accepted':>9} {'coverage':>9} {'sd(â)':>7} {'sd(noise)':>10} {'sd(skew)':>9} {'real share':>11}")
for name in CLASS_ORDER:
    values = deconvolution[name]
    print(f"{name:>6} {values['accepted_count']:9d} {100 * values['coverage']:8.1f}% "
          f"{values['observed_standard_deviation']:7.3f} {values['estimator_noise_standard_deviation']:10.3f} "
          f"{values['skew_standard_deviation']:9.3f} {100 * values['variance_share_real_skew']:10.1f}%")

# %%
plot_real_targets(real_asymmetry, estimator_noise, support=support)
plt.show()

# %% [markdown]
# Subtracting the estimator's variance from the observed variance leaves the
# skew the events actually carry: standard deviation $0.17 / 0.16 / 0.15$ for
# 2 / 4 / 10 µm, which is $74$–$89$ % of the observed variance. Section 1 said
# the skew is real; this says how big it is, on the model's own scale, in units
# the generator can consume. That is the whole point of putting the calibration
# between the observation and the generation.
#
# Two things this does **not** say. The deconvolution assumes the estimator's
# error is independent of the true skew, which the flat error-versus-$|a|$
# reading above supports but does not prove. And every number here describes the
# *accepted* population — 72 % of 2 µm, 92 % of 4 µm, 62 % of 10 µm strict
# events. The abstained events are not measured, and section 3 showed the
# abstention is SNR-shaped, so the anchor population is biased toward the
# louder events of each class.

# %% [markdown]
# #### 4.2 The limit: a run that forbade what the next run did
#
# The targets run declared a minimum of 50 confidence-accepted events per class
# before its matrices could authorise anything. It got 40 at 10 µm. Both run
# manifests are on disk; here they are, unedited.

# %%
targets_manifest = json.loads((targets_run / "run.json").read_text())
generation_run = run_dir("particle-z8-v2-asymmetry-paired-generation-r1")
generation_manifest = json.loads((generation_run / "run.json").read_text())

print("particle-z8-v2-asymmetry-5d-targets-r4 · claim_boundary")
print(f"  {targets_manifest['claim_boundary']}\n")
print(f"  strict accepted counts: {targets_metrics['strict_accepted_counts']}")
print(f"  minimum_50_strict_per_class: {targets_metrics['minimum_50_strict_per_class']}")
print(f"  status: {targets_manifest['status']}\n")

print("particle-z8-v2-asymmetry-paired-generation-r1 · claim_boundary")
print(f"  {generation_manifest.get('claim_boundary', '(no claim_boundary field in the manifest)')}")
print(f"  status: {generation_manifest['status']}")
print(f"  targets consumed: {'particle-z8-v2-asymmetry-5d-targets-r4' in generation_manifest['command']}")

written = [datetime.fromisoformat(manifest["created_at"])
           for manifest in (targets_manifest, generation_manifest)]
elapsed_minutes = (written[1] - written[0]).total_seconds() / 60.0
print(f"  written {elapsed_minutes:.0f} minutes after the run that forbade it")

recorded = json.loads((generation_run / "metrics_manifest.json").read_text())
recorded_hash = recorded["computation_provenance"]["inputs"]["targets_metrics_manifest_sha256"]
actual_hash = sha256_file(targets_run / "metrics_manifest.json")
assert recorded_hash == actual_hash, "the consumed targets manifest is not the one on disk"
print(f"\n  the generation run recorded the targets manifest hash {recorded_hash[:16]}…, "
      f"which matches the file on disk byte for byte")

# %% [markdown]
# So the record is unambiguous. The run that measured the 5-D targets wrote, in
# its own manifest, *"the 10 µm strict population has 40 accepted events, below
# the predeclared 50-event gate, so these matrices are provisional and no
# synthetic generation is authorized by this run"*. Seventy-nine minutes later
# the paired generation run consumed exactly those matrices — verified by hash —
# and built the 47,980-event dataset on which the entire asymmetry column of the
# deck rests. Its own manifest carries **no claim boundary at all**.
#
# What rests on the under-powered class, precisely:

# %%
v5_events = read_rows(v5_root / "events.csv")
v5_summary = json.loads((v5_root / "dataset_summary.json").read_text())
generated = {name: np.asarray([float(row["waveform_asymmetry"]) for row in v5_events
                               if row["class_name"] == name]) for name in CLASS_ORDER}

print(f"{'class':>6} {'v5 events':>10} {'share':>7} {'generated a range':>22} {'sd':>7} {'|a|≥0.2':>9} {'anchors':>8}")
exposure = {}
for name in CLASS_ORDER:
    values = generated[name]
    exposure[name] = {
        "generated_count": int(values.size),
        "generated_share": float(values.size / len(v5_events)),
        "generated_minimum": float(values.min()),
        "generated_maximum": float(values.max()),
        "generated_standard_deviation": float(values.std(ddof=1)),
        "generated_fraction_abs_ge_0p2": float(np.mean(np.abs(values) >= 0.2)),
        "anchor_count": int(deconvolution[name]["accepted_count"]),
    }
    record = exposure[name]
    print(f"{name:>6} {record['generated_count']:10d} {100 * record['generated_share']:6.1f}% "
          f"{f'[{values.min():+.3f}, {values.max():+.3f}]':>22} "
          f"{record['generated_standard_deviation']:7.3f} "
          f"{100 * record['generated_fraction_abs_ge_0p2']:8.1f}% {record['anchor_count']:8d}")

transform_check = {
    name: (abs(np.arctanh(support[name][0] / ASYMMETRY_BOUND) - gaussian_targets[name]["observed_minimum_transformed"]),
           abs(np.arctanh(support[name][1] / ASYMMETRY_BOUND) - gaussian_targets[name]["observed_maximum_transformed"]))
    for name in CLASS_ORDER
}
assert max(max(pair) for pair in transform_check.values()) < 1.0e-9
print(f"\nthe generation support policy is '{v5_summary['asymmetry_policy']['support']}': "
      f"the anchors' own extremes are hard rejection bounds")
print("10 µm generated events are confined to the range 40 real events happened to span; "
      f"{100 * exposure['10um']['generated_fraction_abs_ge_0p2']:.0f} % carry |a| ≥ 0.2 "
      f"against {100 * exposure['4um']['generated_fraction_abs_ge_0p2']:.0f} % at 4 µm")

consumers = sorted(
    path.parent.name
    for path in (workspace.artifacts_root).rglob("run.json")
    if "effective-snr-synthetic-events@v5" in path.read_text()
)
print(f"\nmanifested runs that consume the v5 dataset: {len(consumers)}")
print(f"  including {sum('bead-ssl' in name for name in consumers)} bead-SSL training and "
      f"evaluation runs — the v5 dataset is the pre-training corpus of the whole SSL chain")

# %% [markdown]
# So: 3,660 generated events, 7.6 % of the corpus, carry an asymmetry
# coordinate whose spread and whose hard support bounds were set by 40 real
# measurements — and the dataset is the pre-training corpus for every model in
# the chain, not a side artefact. The visible symptom is the range: generated
# 10 µm skew is confined to $[-0.37, +0.37]$ where the other classes reach
# $\pm 0.78$, and only 28 % of 10 µm events carry $|a| \ge 0.2$ against 70 % of
# the others.
#
# Is that narrowness physics or arithmetic? It is worth measuring rather than
# asserting, so the next cell is an explicit exploration.

# %% [markdown]
# ##### Exploratory: what can 40 events measure?
#
# The two large classes have 148 and 1,129 anchors. Draw 40 of them at random,
# many times, and ask what support and what spread a 40-event sample would have
# reported — if the answer looks like 10 µm's, the narrowness is a sample-size
# artefact rather than a property of 10 µm events. This is an exploratory
# resampling of an existing published population; it authorises nothing.

# %%
bootstrap_generator = np.random.default_rng(20260817)
observed_half_range = float(max(-real_asymmetry["10um"].min(), real_asymmetry["10um"].max()))
half_range_draws = {}
support_artefact = {}
for name in ("2um", "4um"):
    pool = real_asymmetry[name]
    halves = np.empty(4000)
    spreads = np.empty(4000)
    for index in range(4000):
        draw = bootstrap_generator.choice(pool, 40, replace=False)
        halves[index] = max(-draw.min(), draw.max())
        spreads[index] = draw.std(ddof=1)
    half_range_draws[name] = halves
    support_artefact[name] = {
        "median_half_range": float(np.median(halves)),
        "probability_half_range_at_most_observed": float(np.mean(halves <= observed_half_range)),
        "median_standard_deviation": float(np.median(spreads)),
        "probability_standard_deviation_at_most_observed": float(
            np.mean(spreads <= real_asymmetry["10um"].std(ddof=1))
        ),
    }
    values = support_artefact[name]
    print(f"{name:>5} subsampled to n=40: half-range median {values['median_half_range']:.3f}, "
          f"P(≤ 10 µm's {observed_half_range:.3f}) = {values['probability_half_range_at_most_observed']:.3f} · "
          f"sd median {values['median_standard_deviation']:.3f}, "
          f"P(≤ 10 µm's) = {values['probability_standard_deviation_at_most_observed']:.3f}")

count = real_asymmetry["10um"].size
spread = float(real_asymmetry["10um"].std(ddof=1))
interval = (spread * np.sqrt((count - 1) / chi2.ppf(0.975, count - 1)),
            spread * np.sqrt((count - 1) / chi2.ppf(0.025, count - 1)))
print(f"\n10 µm sd {spread:.3f}, 95 % interval [{interval[0]:.3f}, {interval[1]:.3f}] — "
      f"contains 4 µm's {real_asymmetry['4um'].std(ddof=1):.3f} "
      f"and 2 µm's {real_asymmetry['2um'].std(ddof=1):.3f}")

# %%
plot_support_bootstrap(half_range_draws, observed_half_range)
plt.show()

# %% [markdown]
# The two readings separate cleanly, and the planned story is half wrong, which
# is the useful half. The **spread** of 10 µm skew is ordinary: its 95 %
# interval covers both other classes, and a 40-event draw from 4 µm is as narrow
# 43 % of the time. The **support** is not: only 4.9 % of 40-event draws from
# 4 µm, and 3.1 % from 2 µm, span as little as the 10 µm anchors did. So 10 µm
# events are not less skewed — the anchor set was simply too small to observe
# the tails, and the generator turned that missing observation into a hard
# rejection bound on 3,660 events. The under-powered class did not merely make
# the matrices "provisional"; it propagated a sample-size artefact into the data
# that every downstream model is pre-trained on.
#
# The redo does not need cleverness here, only counts.

# %%
mad_root = dataset_root("particles2snr-beads-mad-teacher-detection-development@v1")
mad_rows = [row for row in read_rows(mad_root / "events.csv") if row["output_split"] != "test"]
if any(row["output_split"] == "test" for row in mad_rows):
    raise RuntimeError("sealed test rows reached the MAD census")
mad_census = collections.Counter(row["source_class"] for row in mad_rows)
projected = mad_census["10um"] * deconvolution["10um"]["coverage"]
print(f"MAD detection development set, train + val only: {dict(mad_census)}")
print(f"10 µm candidates outside the sealed split: {mad_census['10um']} "
      f"(880 including the test split, which this notebook does not open)")
print(f"at the 10 µm confidence acceptance rate measured above "
      f"({100 * deconvolution['10um']['coverage']:.0f} %), that projects to roughly "
      f"{projected:.0f} anchors against a 50-event gate")

# %% [markdown]
# #### 4.3 Paired generation: change one thing
#
# With targets in hand, the generator does not build a new dataset — it rebuilds
# the existing one, event by event, changing only the skew. Every other field is
# frozen: the same amplitude, frequency, width, requested SNR, phase, timing,
# and the same slice of the same recorded noise file, identified by hash. The
# fifth coordinate is drawn conditionally on the other four through the 5-D
# correlation structure, transformed by $\operatorname{atanh}(a/0.8)$, and
# rejected if it falls outside the anchors' observed support.
#
# The one unavoidable side effect: changing the envelope changes the clean
# signal's energy, so the noise is rescaled to hold the requested SNR exactly.

# %%
v4_by_position = {position: row for position, row in enumerate(v4_events)}
frozen_fields = v5_summary["paired_contract"]["frozen_fields"]
mismatches = 0
for row in v5_events:
    baseline = v4_by_position[int(row["paired_baseline_index"])]
    if any(str(baseline[field]) != str(row[field]) for field in frozen_fields):
        mismatches += 1
assert mismatches == 0, f"{mismatches} paired events changed a frozen field"
print(f"all {len(v5_events):,} events re-verified against their v4 twin: "
      f"{len(frozen_fields)} frozen fields identical, 0 mismatches")
print(f"frozen: {', '.join(frozen_fields)}")
print(f"changed: {v5_summary['paired_contract']['changed_only']}")

generation_validation = published("particle-z8-v2-asymmetry-paired-generation-r1", "generation_validation.json")
achieved = np.asarray([float(row["achieved_snr_db"]) for row in v5_events])
requested = np.asarray([float(row["snr_db"]) for row in v5_events])
snr_error = float(np.max(np.abs(achieved - requested)))
gap = abs(snr_error - generation_validation["maximum_snr_error_db"])
assert gap < 1.0e-18, f"SNR error drifted by {gap:.3e}"
print(f"\nreproduces particle-z8-v2-asymmetry-paired-generation-r1 exactly: "
      f"maximum SNR realisation error {snr_error:.2e} dB")
print(f"seed {v5_summary['asymmetry_policy']['seed']} · "
      f"transform {v5_summary['asymmetry_policy']['transform']} · "
      f"conditional on {', '.join(v5_summary['asymmetry_policy']['conditional_coordinates'])}")
print(f"dataset status on disk: {v5_summary['status']}")

# %% [markdown]
# #### 4.4 Did the fifth coordinate land where the real events are?
#
# The test is the dependence structure. For each class, compare the correlation
# matrix realised in the 47,980 generated events against the target measured on
# the real anchors — the difference matrix, not the two matrices, is what
# carries the claim. The upper block is the four original coordinates, frozen to
# the full-population 4-D targets; the bottom row is the new one.
#
# A difference is only meaningful against the sampling error of the anchor
# correlation it is compared to, which is where the anchor count re-enters:
# Fisher's transform gives a standard error of $1/\sqrt{n-3}$, so the same
# discrepancy is worth $0.16$ standard errors at $n = 1129$ and $6$ at $n = 40$
# — or the other way round, which is the direction that matters here.

# %%
PARAMETERS = ("log_amplitude_p0", "frequency_khz", "log_tau_ms", "snr_db", "transformed_asymmetry")
LABELS = ("log P₀", "f", "log τ", "SNR", "atanh(a/0.8)")
correlation_rows = read_rows(generation_run / "correlation_validation.csv")
anchor_counts = {name: deconvolution[name]["accepted_count"] for name in CLASS_ORDER}

deltas, zscores, dependence = {}, {}, {}
for name in CLASS_ORDER:
    delta = np.zeros((5, 5))
    target = np.zeros((5, 5))
    realised = np.zeros((5, 5))
    for row in correlation_rows:
        if row["class_name"] != name:
            continue
        i, j = PARAMETERS.index(row["row_parameter"]), PARAMETERS.index(row["column_parameter"])
        delta[i, j] = float(row["delta"])
        target[i, j] = float(row["target_correlation"])
        realised[i, j] = float(row["realized_correlation"])
    standard_error = 1.0 / np.sqrt(anchor_counts[name] - 3)
    transform = lambda values: np.arctanh(np.clip(values, -0.999999, 0.999999))
    zscore = (transform(realised) - transform(target)) / standard_error
    np.fill_diagonal(zscore, 0.0)
    deltas[name], zscores[name] = delta, zscore
    off_diagonal = np.abs(delta - np.diag(np.diag(delta)))
    dependence[name] = {
        "anchor_count": anchor_counts[name],
        "anchor_standard_error": float(standard_error),
        "maximum_absolute_delta": float(off_diagonal.max()),
        "asymmetry_row_maximum_absolute_delta": float(np.abs(delta[4, :4]).max()),
        "asymmetry_row_maximum_absolute_z": float(np.abs(zscore[4, :4]).max()),
        "frozen_block_maximum_absolute_z": float(np.abs(zscore[:4, :4]).max()),
        "delta_undetectable_at_two_sigma": float(np.tanh(2.0 * standard_error)),
    }
    values = dependence[name]
    print(f"{name:>5}: n={values['anchor_count']:4d} · asymmetry row max |Δr| "
          f"{values['asymmetry_row_maximum_absolute_delta']:.3f} → |z| "
          f"{values['asymmetry_row_maximum_absolute_z']:.2f} · frozen 4-D block max |z| "
          f"{values['frozen_block_maximum_absolute_z']:.1f} · a difference up to "
          f"{values['delta_undetectable_at_two_sigma']:.2f} would pass unnoticed at 2σ")

worst = dependence["10um"]
rescaled = (worst["asymmetry_row_maximum_absolute_z"] * worst["anchor_standard_error"]
            / dependence["4um"]["anchor_standard_error"])
worst["z_at_4um_anchor_count"] = float(rescaled)
print(f"\nthe 10 µm asymmetry-row discrepancy of "
      f"{worst['asymmetry_row_maximum_absolute_delta']:.3f} is "
      f"{worst['asymmetry_row_maximum_absolute_delta'] / dependence['4um']['asymmetry_row_maximum_absolute_delta']:.0f}× "
      f"the largest at 4 µm, and would be worth |z| = {rescaled:.1f} at the 4 µm anchor count")

published_delta = published("ssl-v18-dependence-delta-visuals-r1", "figure_metrics.json")
recomputed_maximum = max(values["maximum_absolute_delta"] for values in dependence.values())
gap = abs(recomputed_maximum - published_delta["maximum_absolute_off_diagonal_delta_5d"])
assert gap < 1.0e-12, f"5-D delta maximum drifted by {gap:.3e}"
print(f"\nreproduces ssl-v18-dependence-delta-visuals-r1 exactly: "
      f"maximum off-diagonal |Δr| = {recomputed_maximum:.3f}")

# %%
plot_delta_triptych(deltas, zscores, LABELS, anchor_counts)
plt.show()

# %% [markdown]
# The deck reads this figure as "the asymmetry row matches the real anchors
# everywhere, $|z| < 1.2$", and the arithmetic is right: the largest asymmetry-row
# discrepancy is $1.07$ standard errors. But look at where it sits. That
# $1.07$ is the 10 µm skew-versus-frequency cell, and its actual discrepancy is
# $\Delta r = -0.17$ — fifteen times the largest 4 µm asymmetry-row
# discrepancy, and above the $\pm 0.15$ at which the colour scale saturates.
# It reads as agreement because $n = 40$ makes the standard error $0.164$. At
# the 4 µm anchor count the same difference would be worth $5.9$ standard
# errors. On 10 µm, a difference as large as $0.32$ — twice the figure's full
# colour range — would still pass the two-sigma test.
#
# "The asymmetry row matches everywhere" is therefore three different statements:
# a strong one at 4 µm, a fair one at 2 µm, and at 10 µm a statement about how
# few anchors there were. That is the same 40 events, surfacing a third time.
#
# The upper block is a separate matter and not a generation defect: those rows
# are frozen to the approved full-population 4-D targets, while the comparison
# here is against the confidence-gated anchor subset. The gap between them —
# up to $13.7$ standard errors on 4 µm — is the footprint of the confidence
# gate's SNR-shaped selection, measured in section 3, not noise in the generator.

# %% [markdown]
# ### What this section does not claim
#
# The skew is real, measurable and reproducible at the population level, and the
# generator carries it faithfully where there were enough anchors to say so.
# Beyond that:
#
# - Nothing here is validation. Every number comes from development and
#   train/val data; the sealed test split was never opened, and the section
#   raises rather than proceeds if a test row appears.
# - The recovery numbers grade a configuration that never ran on a real event.
#   The domain-aligned recalibration used for the noise model above is labelled
#   `complete_development_awaiting_visual_review`, and at the threshold the real
#   pipeline deploys it meets its own operational gate on 4 µm only — 2 µm and
#   10 µm miss on coverage, and its run-level `development_success` is false.
# - The calibration gate did not pass: `calibration_pass: false` on all three
#   classes, on the 95th-percentile error. The estimator is used behind an
#   abstention gate, not because it cleared the bar.
# - The 5-D targets are provisional by their own manifest, and the dataset built
#   from them is still `interim_paired_candidate_awaiting_scientific_result` —
#   yet it is registered and is the pre-training corpus of the SSL chain.
# - The anchor population is conditional and SNR-biased. Nothing is claimed
#   about the events the confidence gate abstained on, which are 28 % of 2 µm
#   and 38 % of 10 µm strict events.
# - The per-event $\hat{a}$ is reproducible to about two decimals across
#   environments, not exactly.

# %%
asymmetry_metrics = {
    "schema_version": 1,
    "analysis": "waveform-asymmetry explainer measurements",
    "model_free_envelope_skew": skew_summary,
    "estimator_noise_deconvolution": deconvolution,
    "calibration_error_by_snr_quartile": snr_structure,
    "cross_environment_reproduction": reproduction_summary,
    "generated_asymmetry_exposure": exposure,
    "small_sample_support_artefact": {
        "observed_10um_half_range": observed_half_range,
        "donors": support_artefact,
        "subsample_size": 40,
        "draws": 4000,
    },
    "dependence_delta_significance": dependence,
    "deployed_configuration_operational_gate": operational,
    "sealed_test_accessed": False,
    "real_z8_data_read": True,
}
asymmetry_provenance = {
    "datasets": dataset_provenance(),
    "inputs": {
        "targets_metrics_manifest_sha256": actual_hash,
        "calibration_rows_sha256": sha256_file(confirmatory / "calibration_rows.csv"),
        "variant_fit_rows_sha256": sha256_file(domain_run / "variant_fit_rows.csv"),
        "correlation_validation_sha256": sha256_file(generation_run / "correlation_validation.csv"),
    },
    "parameters": {
        "envelope_skew_window_tau": 3.0,
        "envelope_skew_null_per_class": 800,
        "reproduction_rows_per_class": 48,
        "subsample_size": 40,
        "subsample_draws": 4000,
        "confidence_threshold": confidence_threshold,
        "seeds": {"envelope_skew_null": 20260815, "reproduction": 20260816, "subsampling": 20260817},
    },
    "metric_definitions": {
        "envelope_skew": "energy-weighted standardised third moment of the analytic envelope within ±3τ, no fit",
        "skew_standard_deviation": "sqrt(var(accepted real â) − var(domain-aligned confidence-accepted estimator error))",
        "delta_undetectable_at_two_sigma": "tanh(2/sqrt(n−3)): correlation difference still passing a two-sigma anchor test",
        "cross_environment_reproduction": "re-running the shipped estimator on published injections in this environment",
    },
}
# Serialise here rather than inside the emission: a payload this notebook cannot
# write is a bug in the section, not a refusal by the evidence gate.
payload = json.dumps({"metrics": asymmetry_metrics, "provenance": asymmetry_provenance},
                     indent=2, sort_keys=True)
print(f"evidence payload serialises: {len(payload):,} characters, "
      f"{len(asymmetry_metrics) - 4} measurement groups")

try:
    emitted = notebook_evidence.emit_run(
        workspace,
        section="asymmetry",
        metrics=asymmetry_metrics,
        provenance=asymmetry_provenance,
        claim_boundary=(
            "Explainer measurements on development train/val data only: that real envelopes are "
            "skewed beyond a symmetric generator's noise, how much of the observed spread is skew "
            "rather than estimator error, where the calibration error lives, and what the 40-event "
            "10 µm anchor set determines in the v5 candidate. Authorises no dataset promotion, no "
            "generator change, no model training and no claim about sealed test data."
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
