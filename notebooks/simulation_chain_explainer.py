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
# # Simulating acoustic particle events, end to end
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


# %% [markdown]
# ## Setup
#
# Everything below imports installed packages. No method is defined in this
# notebook: the cells orchestrate and plot, and the mathematics stays where the
# tools read it from, which is what keeps the notebook and the manifested
# analyses from drifting apart.

# %%
import csv
import json
import time

import matplotlib.pyplot as plt
import numpy as np

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


# --- twins ---
"""Plot helpers for the `twins` section.

Every helper accepts `ax=None` or `axes=None` so a cell can redraw into an
existing figure without rebuilding it. No measurement happens here: the section
computes the numbers, these functions only draw them.

The envelope drawn on the waveform panels is the shipped descriptor envelope,
`z8_domain_pca.morphology_features(...)[:, :64]`, not a locally smoothed Hilbert
magnitude. Drawing anything else would show the reader a shape the distance does
not use.
"""

import matplotlib.pyplot as plt
import numpy as np

from internship_workspace.z8_domain_pca import morphology_features

TWIN_INK = "#243447"
TWIN_MUTED = "#6b7a8d"
TWIN_FAINT = "#eef2f7"
TWIN_ALERT = "#c2402a"
TWIN_SAMPLING_KHZ = 2_000.0
TWIN_CORE = slice(1536, 2560)


def _unit(signal):
    """Level-free, scale-free view: the only comparison the eye should make."""
    values = np.asarray(signal, dtype=np.float64)
    values = values - values.mean()
    return values / np.sqrt(np.mean(np.square(values)))


def _descriptor_envelope(signal):
    """The 64-bin Hilbert envelope the morphology descriptor is built from."""
    values = np.asarray(signal, dtype=np.float64).reshape(1, -1)
    return morphology_features(values)[0, :64]


def _quiet(axis):
    axis.set_yticks([])
    axis.tick_params(labelsize=8.5, colors=TWIN_MUTED)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(TWIN_MUTED)


def _draw_overlay(axis, real, twin, colour, *, twin_label="synthetic twin", real_label="real event"):
    """One real event and one synthetic candidate, raw faint, envelopes bold."""
    real_unit = _unit(real)
    time = np.arange(len(real_unit)) / TWIN_SAMPLING_KHZ
    bins = np.linspace(0.0, time[-1], 64, endpoint=False)
    axis.axvspan(time[TWIN_CORE.start], time[TWIN_CORE.stop - 1], color=TWIN_FAINT, zorder=0)
    scale = 3.0 * np.abs(real_unit).max()
    if twin is not None:
        twin_unit = _unit(twin)
        scale = 3.0 * max(np.abs(real_unit).max(), np.abs(twin_unit).max())
    axis.plot(time, real_unit, color=TWIN_INK, linewidth=0.4, alpha=0.28, zorder=1)
    axis.step(
        bins,
        _descriptor_envelope(real_unit) * scale,
        where="post",
        color=TWIN_INK,
        linewidth=1.8,
        label=real_label,
        zorder=3,
    )
    if twin is not None:
        axis.plot(time, twin_unit, color=colour, linewidth=0.4, alpha=0.28, zorder=1)
        axis.step(
            bins,
            _descriptor_envelope(twin_unit) * scale,
            where="post",
            color=colour,
            linewidth=1.8,
            label=twin_label,
            zorder=3,
        )
    axis.set_xlim(0.0, time[-1])
    _quiet(axis)


def plot_twin_pairs(pairs, *, axes=None):
    """Real event beside its nearest synthetic event, in each space, per class.

    `pairs` is one dict per class with keys `case_id`, `class_name`, `colour`,
    `real`, and the two candidates `morphology` and `latent`, each carrying
    `signal`, `sample_id`, `morphology_distance` and `cosine_distance`.
    """
    if axes is None:
        figure, axes = plt.subplots(len(pairs), 2, figsize=(13.6, 2.8 * len(pairs)), sharex=True)
        figure.suptitle(
            "The same real event, twinned in two different spaces",
            x=0.006,
            ha="left",
            fontsize=14,
            fontweight="bold",
            color=TWIN_INK,
        )
    axes = np.atleast_2d(axes)
    columns = (
        ("morphology", "nearest in morphology (PCA-16, euclidean)"),
        ("latent", "nearest in Conv1D-GAP latent (cosine)"),
    )
    for row, pair in enumerate(pairs):
        for column, (key, heading) in enumerate(columns):
            candidate = pair[key]
            axis = axes[row, column]
            _draw_overlay(axis, pair["real"], candidate["signal"], pair["colour"])
            axis.set_title(
                f"{pair['case_id']} · {heading}\n"
                f"morphology {candidate['morphology_distance']:.2f}  ·  "
                f"cosine {candidate['cosine_distance']:.3f}  ·  "
                f"{candidate['sample_id']}",
                loc="left",
                fontsize=9,
                color=TWIN_INK,
            )
            if row == 0 and column == 0:
                axis.legend(fontsize=7.5, frameon=False, loc="upper right")
    for axis in axes[-1]:
        axis.set_xlabel(
            "time (ms) · shaded: the 0.512 ms the morphology space compares",
            fontsize=9,
            color=TWIN_MUTED,
        )
    axes[0, 0].get_figure().tight_layout()
    return axes


def plot_parameter_triptych(cases, *, axes=None):
    """Real event, its parameter-nearest twin, its morphology-nearest twin.

    `cases` is one dict per class with `case_id`, `class_name`, `colour`,
    `real`, `real_parameters`, and the candidates `parameter` and `morphology`,
    each with `signal`, `sample_id`, `parameters`, `morphology_distance` and
    `morphology_rank`.
    """
    if axes is None:
        figure, axes = plt.subplots(len(cases), 3, figsize=(16.2, 3.0 * len(cases)), sharex=True)
        figure.suptitle(
            "Close in fitted parameters is not close in signal",
            x=0.006,
            ha="left",
            fontsize=14,
            fontweight="bold",
            color=TWIN_INK,
        )
    axes = np.atleast_2d(axes)
    for row, case in enumerate(cases):
        _draw_overlay(axes[row, 0], case["real"], None, case["colour"])
        axes[row, 0].set_title(
            f"{case['case_id']} · measured event\n{case['real_parameters']}",
            loc="left",
            fontsize=9,
            color=TWIN_INK,
        )
        for column, (key, heading) in enumerate(
            (("parameter", "nearest in fitted-parameter space"),
             ("morphology", "nearest in morphology space")),
            start=1,
        ):
            candidate = case[key]
            axis = axes[row, column]
            _draw_overlay(
                axis, case["real"], candidate["signal"], case["colour"], twin_label=heading
            )
            axis.set_title(
                f"{heading}\n{candidate['parameters']}\n"
                f"morphology {candidate['morphology_distance']:.2f} · "
                f"rank {candidate['morphology_rank']:,} of {case['gallery']:,}",
                loc="left",
                fontsize=9,
                color=TWIN_ALERT if key == "parameter" else TWIN_INK,
            )
            if row == 0:
                axis.legend(fontsize=7.5, frameon=False, loc="upper right")
    for axis in axes[-1]:
        axis.set_xlabel("time (ms)", fontsize=9, color=TWIN_MUTED)
    axes[0, 0].get_figure().tight_layout()
    return axes


def plot_metric_scatter(sample, *, ax=None):
    """One anchor's whole same-class gallery, placed on both rulers at once."""
    if ax is None:
        figure, ax = plt.subplots(figsize=(7.6, 5.6))
        figure.suptitle(
            "Two metrics, one gallery: what each calls close",
            x=0.006,
            ha="left",
            fontsize=14,
            fontweight="bold",
            color=TWIN_INK,
        )
    ax.hexbin(
        sample["cosine"],
        sample["morphology"],
        gridsize=52,
        bins="log",
        mincnt=1,
        cmap="Greys",
        zorder=0,
    )
    for key, colour, label in (
        ("latent_choice", TWIN_ALERT, "chosen by cosine"),
        ("parameter_choice", "#7c3aed", "chosen by parameters"),
        ("morphology_choice", "#0f766e", "chosen by morphology"),
    ):
        point = sample[key]
        ax.scatter(
            [point["cosine"]],
            [point["morphology"]],
            s=125,
            color=colour,
            edgecolor="white",
            linewidth=1.3,
            zorder=4,
            label=label,
        )
    ax.set_xlabel("cosine distance in the Conv1D-GAP latent", fontsize=9.5, color=TWIN_MUTED)
    ax.set_ylabel("euclidean distance in the morphology space", fontsize=9.5, color=TWIN_MUTED)
    ax.set_title(
        f"{sample['case_id']} against its {sample['gallery']:,} same-class candidates\n"
        f"Spearman rho = {sample['rho']:.2f}; a shared metric would draw a rising line",
        loc="left",
        fontsize=10.5,
        color=TWIN_INK,
    )
    ax.tick_params(labelsize=8.5, colors=TWIN_MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    ax.get_figure().tight_layout()
    return ax


def plot_rho_strip(groups, *, ax=None, title=None, ylabel="Spearman rho", figsize=(10.4, 4.8)):
    """One dot per anchor, one column per comparison, median bar per column.

    `groups` is one dict per column with `label`, `values`, and either a single
    `colour` or a per-point `colours` list.
    """
    if ax is None:
        figure, ax = plt.subplots(figsize=figsize)
        if title:
            figure.suptitle(title, x=0.006, ha="left", fontsize=14, fontweight="bold", color=TWIN_INK)
    for position, group in enumerate(groups):
        values = np.asarray(group["values"], dtype=np.float64)
        colours = group.get("colours") or [group.get("colour", TWIN_MUTED)] * len(values)
        jitter = np.linspace(-0.18, 0.18, len(values))
        ax.scatter(
            position + jitter,
            values,
            s=48,
            color=colours,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        median = float(np.median(values))
        ax.plot([position - 0.32, position + 0.32], [median] * 2, color=TWIN_INK, linewidth=2.2, zorder=4)
        ax.annotate(
            f"{median:.2f}",
            xy=(position + 0.34, median),
            fontsize=9.5,
            color=TWIN_INK,
            fontweight="bold",
            va="center",
        )
    ax.axhline(0.0, color=TWIN_MUTED, linewidth=0.9, linestyle=":", zorder=1)
    ax.axhline(1.0, color=TWIN_MUTED, linewidth=0.9, linestyle=":", zorder=1)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([group["label"] for group in groups], fontsize=9)
    ax.set_xlim(-0.6, len(groups) - 0.25)
    ax.set_ylim(-0.4, 1.08)
    ax.set_ylabel(ylabel, fontsize=9.5, color=TWIN_MUTED)
    ax.tick_params(labelsize=8.5, colors=TWIN_MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.get_figure().tight_layout()
    return ax


def plot_distance_gap(rows, *, ax=None):
    """How far the twin chosen elsewhere lands, on the morphology ruler."""
    if ax is None:
        figure, ax = plt.subplots(figsize=(11.8, 7.8))
        figure.suptitle(
            "Where the chosen twin sits, measured in the morphology space",
            x=0.006,
            ha="left",
            fontsize=14,
            fontweight="bold",
            color=TWIN_INK,
        )
    ordered = sorted(rows, key=lambda row: row["latent_morphology_distance"])
    markers = (
        ("nearest_morphology_distance", "o", None, 46),
        ("parameter_morphology_distance", "s", "#7c3aed", 34),
        ("hybrid_morphology_distance", "d", TWIN_MUTED, 34),
        ("latent_morphology_distance", "X", TWIN_ALERT, 58),
    )
    for position, row in enumerate(ordered):
        ax.plot(
            [row["nearest_morphology_distance"], row["latent_morphology_distance"]],
            [position, position],
            color=row["colour"],
            linewidth=1.5,
            alpha=0.45,
            zorder=2,
        )
        for key, marker, colour, size in markers:
            if key not in row:
                continue
            ax.scatter(
                [row[key]],
                [position],
                s=size,
                marker=marker,
                color=colour or row["colour"],
                edgecolor="white",
                linewidth=0.7,
                zorder=4 if marker in ("o", "X") else 3,
            )
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels([row["case_id"] for row in ordered], fontsize=7.6)
    ax.set_xlabel("distance in the morphology space (16 PCA axes)", fontsize=9.5, color=TWIN_MUTED)
    ax.tick_params(labelsize=8.5, colors=TWIN_MUTED)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", alpha=0.14)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=TWIN_MUTED, label="nearest synthetic in morphology"),
        plt.Line2D([], [], marker="s", linestyle="", color="#7c3aed", label="chosen by fitted parameters"),
        plt.Line2D([], [], marker="d", linestyle="", color=TWIN_MUTED, label="chosen by physical + morphology rerank"),
        plt.Line2D([], [], marker="X", linestyle="", color=TWIN_ALERT, label="chosen by latent cosine"),
    ]
    ax.legend(handles=handles, fontsize=8.5, frameon=False, loc="lower right")
    ax.get_figure().tight_layout()
    return ax


def plot_validated_counterexample(case, *, axes=None):
    """The human-accepted pair, its distance, and the candidate nobody saw.

    `axes` must be three axes when supplied: the accepted overlay, the
    nearest-neighbour overlay, and the distance ruler underneath them.
    """
    if axes is None:
        figure = plt.figure(figsize=(12.8, 6.8))
        grid = figure.add_gridspec(2, 2, height_ratios=(4.0, 1.15), hspace=0.55, wspace=0.12)
        axes = [
            figure.add_subplot(grid[0, 0]),
            figure.add_subplot(grid[0, 1]),
            figure.add_subplot(grid[1, :]),
        ]
        figure.suptitle(
            f"{case['case_id']}: a reviewer accepted this twin; the morphology space disagrees",
            x=0.006,
            ha="left",
            fontsize=14,
            fontweight="bold",
            color=TWIN_INK,
        )
    axes = list(np.asarray(axes, dtype=object).reshape(-1))
    for axis, key in zip(axes[:2], ("selected", "nearest")):
        candidate = case[key]
        _draw_overlay(
            axis, case["real"], candidate["signal"], case["colour"], twin_label=candidate["label"]
        )
        axis.set_title(
            f"{candidate['label']}\n"
            f"morphology {candidate['morphology_distance']:.2f}  ·  "
            f"rank {candidate['morphology_rank']:,} of {case['gallery']:,}  ·  "
            f"{candidate['sample_id']}",
            loc="left",
            fontsize=9.5,
            color=TWIN_INK,
        )
        axis.set_xlabel(
            "time (ms) · shaded: the window the morphology space compares",
            fontsize=9,
            color=TWIN_MUTED,
        )
        axis.legend(fontsize=8, frameon=False, loc="upper right")

    ruler = axes[2]
    limit = max(
        case["selected"]["morphology_distance"] * 1.35,
        float(np.percentile(case["gallery_distances"], 75.0)),
    )
    ruler.hist(case["gallery_distances"], bins=90, range=(0.0, limit), color="#d7dee7", zorder=1)
    height = ruler.get_ylim()[1]
    for key, colour, marker in (("nearest", case["colour"], "o"), ("selected", TWIN_ALERT, "X")):
        distance = case[key]["morphology_distance"]
        ruler.axvline(distance, color=colour, linewidth=1.5, zorder=3)
        ruler.scatter(
            [distance], [height * 0.82], s=110, marker=marker, color=colour,
            edgecolor="white", linewidth=1.0, zorder=4,
        )
        ruler.annotate(
            f"{distance:.2f}",
            xy=(distance, height * 0.82),
            xytext=(4, 6),
            textcoords="offset points",
            fontsize=9.5,
            color=colour,
            fontweight="bold",
        )
    ruler.set_xlim(0.0, limit)
    ruler.set_ylim(0.0, height)
    ruler.set_xlabel(
        f"distance to the {case['gallery']:,} same-class synthetic events, in the morphology space",
        fontsize=9,
        color=TWIN_MUTED,
    )
    ruler.set_yticks([])
    ruler.tick_params(labelsize=8.5, colors=TWIN_MUTED)
    for side in ("top", "right", "left"):
        ruler.spines[side].set_visible(False)
    return axes


# --- masked_learning ---
"""Figure helpers for the masked-reconstruction section.

Every helper takes plain arrays and pre-computed spans so a cell can redraw
without rebuilding anything, and so no masking logic leaks out of
`p3_ssl.masking` into the plotting layer.
"""

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


def draw_model_view(signal, mask, spans, *, patch_size=16, axes=None):
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
    for start, end in spans:
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


def draw_policies(signal, p25_spans, cyclic_spans, event_span, *, axes=None):
    """Both training policies, same trace, same 25 % budget."""
    if axes is None:
        _, axes = plt.subplots(2, 1, figsize=(12.5, 4.6), sharex=True)
    time = np.arange(signal.size)
    panels = (
        (axes[0], p25_spans, "P25 · targets drawn blind to the signal", P25_COLOUR),
        (axes[1], cyclic_spans, "CYCLIC25 · targets aimed at the annotated support", CYCLIC_COLOUR),
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


def draw_cycle(signal, pass_spans, coverage, event_span, *, axes=None):
    """The CYCLIC25 cycle for one training event: passes, then coverage."""
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(12.5, 3.4),
                               gridspec_kw={"width_ratios": (2.4, 1.0)})
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
    axes[1].plot(passes, 100 * np.asarray(coverage), marker="o", color=CYCLIC_COLOUR)
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


def draw_reconstruction(signal, outputs, spans, window, *, axes=None):
    """Identical hidden samples, both trained models, one real event."""
    if axes is None:
        _, axes = plt.subplots(1, len(outputs), figsize=(12.5, 3.4), sharey=True)
    lo, hi = window
    time = np.arange(lo, hi)
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
    for axis, (event_spans, background_spans), title in panels:
        axis.axvspan(*true_span, color=EVENT_COLOUR, zorder=0)
        axis.axvline(declared_span[0], color=P25_COLOUR, lw=1.2, ls="--", zorder=3)
        axis.axvline(declared_span[1], color=P25_COLOUR, lw=1.2, ls="--", zorder=3)
        for start, end in event_spans:
            axis.axvspan(start, end, ymin=0.5, color=CYCLIC_COLOUR, alpha=0.85, zorder=1)
        for start, end in background_spans:
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


def draw_defect_summary(background_in_event, event_coverage, passes, *, axes=None):
    """Where the aimed budget actually landed, over a sample of the corpus."""
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(12.5, 3.2))
    axes[0].hist(100 * np.asarray(background_in_event), bins=40, color=BACKGROUND_COLOUR,
                 edgecolor=MUTED, lw=0.4)
    axes[0].axvline(100 * np.median(background_in_event), color=P25_COLOUR, ls="--", lw=1.4)
    axes[0].set(xlabel="% of the background budget inside the true event",
                ylabel="events", title="'Background' that is event")

    axes[1].hist(100 * np.asarray(event_coverage), bins=40, color=CYCLIC_COLOUR,
                 edgecolor=MUTED, lw=0.4)
    axes[1].axvline(100 * np.median(event_coverage), color=P25_COLOUR, ls="--", lw=1.4)
    axes[1].axvline(100, color="#111827", ls=":", lw=1.2)
    axes[1].set(xlabel="% of the true support the event group ever hides",
                title="The completeness guarantee")

    declared, corrected = passes
    axes[2].hist([declared, corrected], bins=np.arange(1.5, 17.5, 1.0),
                 color=[P25_COLOUR, CYCLIC_COLOUR], label=["as trained", "corrected"])
    axes[2].set(xlabel="passes needed for a complete cycle", title="Cycle length")
    axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    return axes


# --- window_alignment ---
"""Figure helpers for the window-alignment exploration."""


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
            color="#dc2626", label="shipped descriptor")
    ax.plot(windows, [row["invariant"] for row in descriptor_shapes], marker="s",
            color="#0f766e", label="fixed 37-band grid")
    ax.set(xlabel="descriptor window (samples)", ylabel="dimensions",
           title="A descriptor whose size depends on its window is not one descriptor")
    ax.set_xscale("log", base=2)
    ax.set_xticks(windows, [str(window) for window in windows])
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


# --- quantile_basis ---
"""Figure helpers for the quantile-and-basis alignment exploration."""


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
# # Part I — The argument
#
# From the signal family to the hardest test the generator has to pass.
# Each section reproduces what a manifested run already owns before it
# adds anything of its own.


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
# ## How a signal becomes a point
#
# This is the pivot of the whole argument. Everything that follows measures
# **distances**, and a distance is only as meaningful as the space it lives in.
#
# The descriptor is deliberately **phase-insensitive**: two events with the same
# shape but a different arrival phase should be neighbours, because phase is not
# identifiable from a single crossing. It is built in four steps, all inside
# `internship_workspace.z8_domain_pca.morphology_features`:
#
# 1. crop a fixed window around the event centre, remove the mean, and normalise
#    by RMS — amplitude is a fitted parameter, not a shape;
# 2. take the Hilbert envelope, subtract its 20th-percentile floor, smooth it and
#    average into **64 bins** — the *when* of the energy;
# 3. take the windowed spectrum, keep the **7–80 kHz** band, log-compress and
#    L2-normalise — the *what* of the energy, **37 bins**;
# 4. concatenate into a **101-D** descriptor, standardise, and reduce to **16
#    principal components** fitted on synthetic events only.
#
# Why sixteen, and why that band, are basis questions; Part II answers them.

# %%
from internship_workspace import chain_figures  # noqa: E402
from internship_workspace.z8_coverage import (  # noqa: E402
    batched_features,
    load_real_core,
    read_rows,
    shared_class_sample,
    support_coverage,
    validate_pair,
)
from internship_workspace.z8_domain_pca import morphology_features  # noqa: E402

real_key = "particles2snr-fbase-dual-clean-z8-events-3class-plus-unclear-development@v2"
signal_key = "particles2snr-f-dual-clean-c1-yolo-4class@v2"

real_root = dataset_root(real_key)
signal_root = dataset_root(signal_key)

all_real_rows = read_rows(real_root / "events.csv")
if any(row["split"] == "test" for row in all_real_rows):
    raise PermissionError("sealed test rows are forbidden")
real_rows = [
    row
    for row in all_real_rows
    if row["class_name"] in CLASS_ORDER and row["split"] in {"train", "val"}
]
real_labels = np.asarray([row["class_name"] for row in real_rows])
real_cores = load_real_core(real_rows, signal_root)
real_features = batched_features(real_cores)

print(f"real events        {len(real_rows)}  "
      + "  ".join(f"{c} {int((real_labels == c).sum())}" for c in CLASS_ORDER))
print(f"crop window        {real_cores.shape[1]} samples "
      f"= {1000 * real_cores.shape[1] / SAMPLING_HZ:.3f} ms")
print(f"descriptor         {real_features.shape[1]}-D "
      f"(64 envelope + {real_features.shape[1] - 64} spectral)")

# %% [markdown]
# ### One event, three views
#
# The same event as a waveform and as the two halves of its descriptor — the
# only way "distance in the morphology space" can mean anything to a reader.

# %%
example_index = int(np.flatnonzero(real_labels == "4um")[0])
chain_figures.draw_event_views(
    real_cores[example_index],
    morphology_features(real_cores[example_index][None, :])[0],
    sampling_frequency_hz=SAMPLING_HZ,
    colour=CLASS_COLOUR["4um"],
    title=f"{real_rows[example_index]['event_id']} · 4um",
)

# %% [markdown]
# ### The window is narrower than the events it describes
#
# The descriptor window is fixed at 1 024 samples. The events are not. Putting
# both on one axis is the first thing this notebook was written to make visible,
# and it appears on no slide.

# %%
support_widths = np.sort([
    float(row["end_sample"]) - float(row["start_sample"]) for row in real_rows
])
chain_figures.draw_support_against_window(
    support_widths, window=real_cores.shape[1], candidate_window=4096
)
print(f"support width      median {np.median(support_widths):.0f}  "
      f"p90 {np.percentile(support_widths, 90):.0f}  max {support_widths.max():.0f} samples")
print(f"wider than the {real_cores.shape[1]}-sample window: "
      f"{100 * np.mean(support_widths > real_cores.shape[1]):.1f} %")

# %% [markdown]
# What that costs is not obvious from the histogram alone — a window can be
# narrow and still capture what matters. Part II sweeps it and answers with
# numbers, including one that contradicts the repair everyone expected.
#
# This measurement belongs to no existing run — it is the notebook's own — so it
# is emitted as evidence. That only happens under `workspace notebooks execute`;
# a live kernel prints the refusal and moves on.

# %%
try:
    emitted = notebook_evidence.emit_run(
        workspace,
        section="window-audit",
        metrics={
            "schema_version": 1,
            "analysis": "descriptor-window-versus-event-support",
            "sampling_frequency_hz": SAMPLING_HZ,
            "descriptor_window_samples": int(real_cores.shape[1]),
            "candidate_window_samples": 4096,
            "population": {
                "events": len(real_rows),
                "selection": "physical train and val rows of the three classes",
            },
            "support_width_samples": {
                "median": float(np.median(support_widths)),
                "p90": float(np.percentile(support_widths, 90)),
                "p99": float(np.percentile(support_widths, 99)),
                "max": float(support_widths.max()),
            },
            "wider_than_descriptor_window_fraction": float(
                np.mean(support_widths > real_cores.shape[1])
            ),
            "wider_than_candidate_window_fraction": float(np.mean(support_widths > 4096)),
            "wider_than_descriptor_window_by_class": {
                class_name: float(
                    np.mean(
                        np.asarray([
                            float(row["end_sample"]) - float(row["start_sample"])
                            for row in real_rows
                            if row["class_name"] == class_name
                        ])
                        > real_cores.shape[1]
                    )
                )
                for class_name in CLASS_ORDER
            },
        },
        provenance={
            "datasets": dataset_provenance(),
            "inputs": {
                "events_csv_sha256": notebook_evidence.sha256_file(
                    real_root / "events.csv"
                )
            },
            "parameters": {
                "descriptor_window_samples": int(real_cores.shape[1]),
                "candidate_window_samples": 4096,
                "population": "physical train and val rows of the three classes",
            },
            "metric_definitions": {
                "support_width": "end_sample minus start_sample of the detector annotation",
                "wider_than_window_fraction": (
                    "fraction of events whose annotated support exceeds the fixed "
                    "descriptor window, so the descriptor cannot see the event in full"
                ),
            },
        },
        claim_boundary=(
            "Compares annotated event support against the fixed descriptor window "
            "on the z8 development events. It measures truncation of the input, "
            "not its effect on any downstream metric, and authorizes no dataset "
            "promotion."
        ),
    )
    print(f"emitted {emitted.name}")
except WorkspaceError as error:
    print(f"no evidence emitted ({error})")


# %% [markdown]
# ## Does the synthetic cloud cover the real one?
#
# The question the whole simulation effort exists to answer, made measurable. For
# each class:
#
# - measure how tightly the generator packs its **own** events — the distance
#   from each synthetic event to its nearest synthetic neighbour — and take the
#   **80th percentile** of that distribution as a radius;
# - then ask what fraction of **real** events fall within that radius of some
#   synthetic event.
#
# The radius stays *per condition* on purpose: it measures each generator's own
# density. A radius shared across conditions would reward whichever generator
# spreads out the most.
#
# Three conditions, each adding one piece of physics, all in **one shared PCA
# basis** with **one shared synthetic draw** — which is what makes them
# comparable rather than merely similar.

# %%
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

CONDITIONS = [
    ("white_noise_4d",
     "particles2snr-fbase-z8-cholesky-physicalcorr-effective-snr-synthetic-events@v3"),
    ("real_noise_4d",
     "particles2snr-fbase-z8-cholesky-physicalcorr-effective-snr-synthetic-events@v4"),
    ("asymmetry_5d",
     "particles2snr-fbase-z8-cholesky-physicalcorr-effective-snr-synthetic-events@v5"),
]
SEED = 20260809
QUANTILE = 0.80
SYNTHETIC_CORE = slice(1536, 2560)  # the 1024-sample core of a 4096 draw

started = time.time()
condition_features = {}
reference_rows = None
for label, key in CONDITIONS:
    root = dataset_root(key)
    rows = read_rows(root / "events.csv")
    if reference_rows is None:
        reference_rows = rows
    else:
        validate_pair(reference_rows, rows)  # events must be paired one to one
    raw = np.load(root / "signals_raw_4096.npy", mmap_mode="r", allow_pickle=False)
    condition_features[label] = batched_features(np.asarray(raw[:, SYNTHETIC_CORE]))

synthetic_labels = np.asarray([row["class_name"] for row in reference_rows])
sample = shared_class_sample(synthetic_labels, real_labels, seed=SEED)
print(f"synthetic events   {len(reference_rows)} per condition "
      f"({time.time() - started:.1f} s)")

# %%
fit_rows = np.concatenate(
    [condition_features[label][np.concatenate(list(sample.values()))]
     for label, _ in CONDITIONS],
    axis=0,
)
scaler = StandardScaler().fit(fit_rows)
pca = PCA(n_components=16, svd_solver="full").fit(scaler.transform(fit_rows))


def project(values):
    return pca.transform(scaler.transform(values))


real_scores = project(real_features)
coverage = {
    label: support_coverage(
        real_scores,
        project(condition_features[label]),
        real_labels,
        synthetic_labels,
        sample=sample,
        quantile=QUANTILE,
    )
    for label, _ in CONDITIONS
}

print(f"PCA  16 components explain {pca.explained_variance_ratio_.sum():.4f} "
      f"of the variance, PC1+PC2 {pca.explained_variance_ratio_[:2].sum():.4f}")
for class_name in CLASS_ORDER:
    chain = "  →  ".join(
        f"{100 * coverage[label][class_name]['real_within_radius_fraction']:.1f} %"
        for label, _ in CONDITIONS
    )
    print(f"{class_name:>5}   {chain}")

# %% [markdown]
# ### Reproduction check
#
# Every number above must equal the published run to the last decimal. A non-zero
# deviation is a reproduction bug to fix before anything else — not a discovery.

# %%
reference = published("particle-z8-v2-coverage-conditions-q80-r1")
deviation = 0.0
for label, _ in CONDITIONS:
    for class_name in CLASS_ORDER:
        got = coverage[label][class_name]
        want = reference["conditions"][label]["classes"][class_name]
        for field in (
            "real_within_radius_fraction",
            "synthetic_self_nn_radius",
            "real_to_synthetic_nn_median",
        ):
            deviation = max(deviation, abs(got[field] - want[field]))
deviation = max(
    deviation,
    abs(pca.explained_variance_ratio_.sum() - reference["pca"]["explained_variance_16"]),
    abs(pca.explained_variance_ratio_[:2].sum()
        - reference["pca"]["explained_variance_pc1_pc2"]),
)
assert deviation == 0.0, f"reproduction drifted by {deviation:.3e}"
print(f"reproduces particle-z8-v2-coverage-conditions-q80-r1 exactly "
      f"(max deviation {deviation:.1e})")

# %% [markdown]
# ### What the chain says
#
# Read the three bars per class left to right: white noise only, then a real
# noise carrier, then the fifth asymmetry coordinate. Each step is a piece of
# physics added to the generator, and the bar is the fraction of real events it
# brings inside the synthetic support.
#
# The dashed line at 80 % is the generator's own self-coverage — the radius was
# built as the 80th percentile of synthetic-to-synthetic distances. Every class
# ending above it means the real events are covered *better than the synthetic
# cloud covers itself*.

# %%
chain_figures.draw_coverage_chain(
    coverage, [label for label, _ in CONDITIONS], quantile=QUANTILE
)

# %% [markdown]
# ### Broadening, not relocating
#
# A fair objection: a generator can raise coverage by simply spreading out, and a
# per-condition radius would partly hide that. The radii and the raw
# real-to-synthetic distances answer it directly. If the cloud had *moved* to sit
# on the real events, the median distance would drop; if it merely **broadened**,
# the median stays put while the radius grows.

# %%
print(f"{'class':>6}  {'condition':>15}  {'radius':>8}  {'real→synth median':>18}")
for class_name in CLASS_ORDER:
    for label, _ in CONDITIONS:
        cell = coverage[label][class_name]
        print(f"{class_name:>6}  {label:>15}  {cell['synthetic_self_nn_radius']:8.3f}  "
              f"{cell['real_to_synthetic_nn_median']:18.3f}")

# %% [markdown]
# The two steps turn out to be different in kind, which the deck's single reading
# flattens:
#
# - **The noise carrier moves the cloud.** The median real-to-synthetic distance
#   drops at every class — 4.54 → 4.19, 3.90 → 3.38, 5.40 → 4.66 — while the
#   radius also grows. Real events genuinely got closer to synthetic ones, and
#   the domain AUC below confirms it independently.
# - **The asymmetry coordinate mostly broadens.** The median barely moves after
#   it (4.19 → 4.26, 3.38 → 3.29, 4.66 → 4.65) while the radius keeps growing.
#   Its nine-point coverage gain at 4 µm is bought by a wider cloud, not by a
#   closer one.
#
# That distinction matters for what the chain is allowed to claim. Coverage of
# variability improved twice; proximity improved once. Retrieval, later, is the
# test that proximity has to pass, and it is much stricter than either.
#
# ### The other half of the story: the parameter space
#
# The deck's sharpest point lives in the gap between two coverage numbers. Run
# the same measurement in the space of **fitted parameters** rather than
# waveform morphology, and the picture inverts.

# %%
gap = published("particle-z8-v2-synthetic-support-gap-diagnosis-q80-r1")
print(f"{'class':>6} {'parameter space':>17} {'morphology':>12} "
      f"{'covered in parameters, not in morphology':>42}")
for class_name in CLASS_ORDER:
    cell = gap["classes"][class_name]
    print(f"{class_name:>6} {100 * cell['parameter_coverage']:16.1f} % "
          f"{100 * cell['morphology_coverage']:11.1f} % "
          f"{100 * cell['parameter_covered_morphology_uncovered']:41.1f} %")

# %% [markdown]
# **The fitted parameters of real events are almost entirely inside the synthetic
# parameter cloud — 98 to 99 % — while their waveforms are not.** A third of all
# real events sit inside the parameter support and outside the morphology
# support at the same time.
#
# That is the finding the whole deck is built around, and it is worth stating
# carefully because it is easy to overclaim. It does **not** say the parameters
# are wrong. It says that matching the parameters a model fits is not the same
# achievement as matching the signals those parameters are supposed to describe —
# the map is not the territory, measured. Everything the generator gained across
# the three conditions above was gained in the space where it was already
# weakest.
#
# One caveat on commensurability: this diagnosis run fits its own PCA basis, so
# its morphology column (64.1 / 55.6 / 63.2 %) is not the same number as the
# chain's white-noise column (60.8 / 56.5 / 54.5 %) even though both describe the
# same idea. Part II measures how much two bases can disagree.
#
# ### Can a classifier still tell the two apart?
#
# The strongest objection to any coverage claim is that a cloud can swallow
# another by being large, not by being right. Coverage cannot answer that; a
# discriminator can. This is the deck's only domain-separability measurement, and
# it is the one number that could have refuted the noise-carrier step.

# %%
ablation = published("particle-z8-v2-full-real-noise-analysis-r1")
print(f"{'class':>6} {'white noise AUC':>17} {'real noise AUC':>16}")
for class_name in CLASS_ORDER:
    white = ablation["conditions"]["white_noise_control"][class_name]
    real_noise = ablation["conditions"]["real_noise_candidate"][class_name]
    print(f"{class_name:>6} {white['domain_classifier_auc_mean']:17.3f} "
          f"{real_noise['domain_classifier_auc_mean']:16.3f}")
print("\nAUC 0.5 would mean a classifier cannot tell real from synthetic at all")

# %% [markdown]
# Replacing white noise with a real noise carrier drops the separability sharply
# — 0.94 → 0.66 at 2 µm, 0.97 → 0.81 at 4 µm, 0.89 → 0.70 at 10 µm — on a
# paired ablation where **only the noise changed**. So the coverage gain of the
# second condition is not an artefact of a wider cloud: the events genuinely
# became harder to distinguish.
#
# What is not claimed: an AUC of 0.66 is still far from 0.5. A classifier can
# still tell the populations apart most of the time, and 4 µm remains the worst
# case at 0.81. The simulator is better, not indistinguishable.


# %% [markdown]
# ## Twins
#
# A **twin** is a synthetic event paired with the real event it most resembles.
# The pairing is the most direct test a simulator can face: not "does the cloud
# of synthetic events overlap the cloud of real ones", which a coarse statistic
# can pass by accident, but "point at the one fake event that could pass for
# *this* measured event, and let a human look at both".
#
# Everything in the pairing rests on one choice, and the whole section is about
# it: **the space in which "most resembles" is measured**. Three spaces are
# available in this chain, all three already implemented and fitted by the runs
# this notebook reads, and no two of them agree. Quantifying that disagreement
# is the deliverable here.
#
# The three spaces, defined once:
#
# - **Parameter space.** The four fitted generation coordinates of an event:
#   log amplitude P0, drive frequency in kHz, log decay time τ in ms, and SNR in
#   dB. Standardised on the synthetic gallery, distance euclidean. Implemented
#   in `internship_workspace.z8_support_diagnosis` (`parameter_matrix`,
#   `nearest_indices`). This is the space the generator itself is written in.
# - **Morphology space.** Each 1024-sample window (0.512 ms at 2 MHz) becomes a
#   101-number shape descriptor: a 64-bin Hilbert envelope, which is the
#   phase-insensitive outline of the burst, concatenated with a 37-bin spectrum
#   over 7–80 kHz. The descriptors are standardised and reduced to 16 principal
#   components; distance is euclidean over those 16 axes. Implemented in
#   `internship_workspace.z8_domain_pca.morphology_features`, fitted in the run
#   `particle-z8-v2-paired-asymmetry-pca-r2`, which stores the scores read here.
# - **Latent space.** The 512-number penultimate activation of a *frozen*
#   supervised classifier (Conv1D-GAP-L, three classes: 2 µm / 4 µm / 10 µm),
#   fed a 4096-sample window decimated eightfold to 512 points and z-scored;
#   distance is cosine on L2-normalised vectors. Implemented in
#   `internship_workspace.equation_latent_audit` and
#   `internship_workspace.z8_twin_analysis.top_k_cosine`, stored by
#   `particle-z8-v2-full-real-noise-twin-latent-r1`.
#
# "Frozen" means the classifier is loaded from a fixed checkpoint whose SHA-256
# is asserted before use; it is never trained or fine-tuned here.

# %%
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from internship_workspace.z8_coverage import reflect_crop
from internship_workspace.z8_domain_pca import fit_synthetic_pca, morphology_features
from internship_workspace.z8_support_diagnosis import nearest_indices, parameter_matrix
from internship_workspace.z8_twin_analysis import top_k_cosine

TWIN_LATENT_RUN = "particle-z8-v2-full-real-noise-twin-latent-r1"
TWIN_HYBRID_RUN = "particle-z8-v2-full-real-noise-twin-physical-r1"
TWIN_PCA_RUN = "particle-z8-v2-paired-asymmetry-pca-r2"
TWIN_HUMAN_RUN = "particle-z8-v2-paired-asymmetry-twin-result-r2"

twin_events_root = dataset_root(
    "particles2snr-fbase-dual-clean-z8-events-3class-plus-unclear-development@v2"
)
twin_signal_root = dataset_root("particles2snr-f-dual-clean-c1-yolo-4class@v2")
twin_v4_root = dataset_root(
    "particles2snr-fbase-z8-cholesky-physicalcorr-effective-snr-synthetic-events@v4"
)
twin_v5_root = dataset_root(
    "particles2snr-fbase-z8-cholesky-physicalcorr-effective-snr-synthetic-events@v5"
)


def twin_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


twin_real_events = twin_rows(twin_events_root / "events.csv")
if any(row["split"] == "test" for row in twin_real_events):
    raise PermissionError("sealed test rows are forbidden in the twin section")

twin_anchors = twin_rows(run_dir(TWIN_LATENT_RUN) / "anchors.csv")
if any(row["split"] != "val" for row in twin_anchors):
    raise PermissionError("twin anchors must be validation rows only")

print(f"real development events   {len(twin_real_events):>6,}  splits present: "
      f"{sorted({row['split'] for row in twin_real_events})}")
print(f"frozen twin anchors       {len(twin_anchors):>6,}  "
      f"{ {name: sum(row['analysis_class'] == name for row in twin_anchors) for name in CLASS_ORDER} }")
print(f"synthetic gallery (v4)    {twin_v4_root.name}")

# %% [markdown]
# ### How a twin is chosen, and on what
#
# The anchors are not a fresh draw. Thirty real validation events were frozen
# once — ten per class, each from a distinct source file, spread across the SNR
# range by quantile so the panel is not silently a panel of easy events — and
# every twin run since has reused that exact set. Freezing the anchors is what
# makes runs comparable: a later retrieval cannot quietly improve by picking
# nicer events. The selection code is `z8_twin_analysis.select_anchors` /
# `reuse_frozen_physical_anchors`; this notebook consumes its output rather than
# re-deriving it.
#
# Retrieval is then a nearest-neighbour search restricted to the anchor's own
# class. Restricting to the class is deliberate: a twin is meant to answer "is
# there a credible synthetic 4 µm event for this real 4 µm event", not "can the
# metric guess the class", which is a different question with its own run.
#
# Before measuring anything, the crop convention is checked, because the spaces
# read different amounts of signal and that will matter later. A real event is
# cropped around its annotated centre — `z8_coverage.reflect_crop` — and a
# synthetic event is the central slice of its stored 4096-sample waveform. The
# check below asserts that the stored 4096-sample anchor crop and a freshly cut
# 1024-sample crop describe the same instant.

# %%
twin_real_raw = np.load(run_dir(TWIN_LATENT_RUN) / "real_signals_raw_4096.npy", allow_pickle=False)
twin_core = slice(1536, 2560)

twin_checked = 0
for twin_index, twin_anchor in enumerate(twin_anchors):
    source = np.load(
        twin_signal_root / twin_anchor["source_signal_relative_path"], allow_pickle=False
    ).astype(np.float32)
    fresh = reflect_crop(source, float(twin_anchor["center_norm"]) * source.size, 1024)
    if not np.array_equal(fresh, twin_real_raw[twin_index, twin_core]):
        raise ValueError(f"crop convention drifted for {twin_anchor['case_id']}")
    twin_checked += 1
print(f"crop convention verified on {twin_checked}/{len(twin_anchors)} anchors "
      f"(reflect_crop 1024 == centre of the stored 4096 window)")

twin_v4_signals = np.load(twin_v4_root / "signals_raw_4096.npy", mmap_mode="r", allow_pickle=False)
twin_v5_signals = np.load(twin_v5_root / "signals_raw_4096.npy", mmap_mode="r", allow_pickle=False)
twin_v4_events = twin_rows(twin_v4_root / "events.csv")
print(f"synthetic gallery {twin_v4_signals.shape[0]:,} events x {twin_v4_signals.shape[1]} samples")

# %% [markdown]
# ### Reproducing the two stored spaces before using either
#
# Neither fitted space is rebuilt from scratch here. Both are read from the runs
# that own them, and each is checked against a number that run published. If a
# reproduction drifts, the cells below fail rather than quietly measuring
# something else.
#
# The latent side is reproduced by calling the shipped retrieval,
# `z8_twin_analysis.top_k_cosine`, on the stored embeddings and asserting that
# it returns the published top-1 cosine distances *and* the published sample
# identifiers. The morphology side is reproduced by recomputing the published
# median real-to-nearest-synthetic distance per class from the stored PCA
# scores.

# %%
twin_pca = np.load(run_dir(TWIN_PCA_RUN) / "pca_scores.npz", allow_pickle=True)
twin_embeddings = np.load(run_dir(TWIN_LATENT_RUN) / "embeddings.npz", allow_pickle=True)

twin_sample_id = twin_pca["synthetic_sample_id"].astype(str)
twin_synth_class = twin_pca["synthetic_class"].astype(str)
twin_real_id = twin_pca["real_event_id"].astype(str)
twin_real_class = twin_pca["real_class"].astype(str)
twin_morph_synth = np.asarray(twin_pca["baseline"], dtype=np.float64)
twin_morph_v5 = np.asarray(twin_pca["candidate"], dtype=np.float64)
twin_morph_real = np.asarray(twin_pca["real"], dtype=np.float64)
twin_real_row = {event: index for index, event in enumerate(twin_real_id)}

if not np.array_equal(twin_sample_id, twin_embeddings["synthetic_sample_id"].astype(str)):
    raise ValueError("the two runs do not index the same synthetic gallery")
if [row["sample_id"] for row in twin_v4_events] != twin_sample_id.tolist():
    raise ValueError("the v4 event table is not aligned with the stored scores")

twin_case_order = twin_embeddings["real_case_id"].astype(str)
twin_by_case = {row["case_id"]: row for row in twin_anchors}
twin_eligible = [
    np.flatnonzero(twin_synth_class == twin_by_case[case]["analysis_class"])
    for case in twin_case_order
]
twin_top_index, twin_top_distance = top_k_cosine(
    twin_embeddings["real"],
    twin_embeddings["synthetic"],
    twin_sample_id,
    eligible_indices=twin_eligible,
)

twin_published_latent = published(TWIN_LATENT_RUN, "retrieval_summary.json")[
    "primary_top1_cosine_distance"
]
twin_got_latent = {
    "mean": float(twin_top_distance[:, 0].mean()),
    "median": float(np.median(twin_top_distance[:, 0])),
    "minimum": float(twin_top_distance[:, 0].min()),
    "maximum": float(twin_top_distance[:, 0].max()),
}
for twin_key, twin_want in twin_published_latent.items():
    twin_deviation = abs(twin_got_latent[twin_key] - twin_want)
    assert twin_deviation < 1e-12, f"latent reproduction drifted on {twin_key} by {twin_deviation:.3e}"

twin_published_neighbours = {
    row["case_id"]: row["synthetic_sample_id"]
    for row in twin_rows(run_dir(TWIN_LATENT_RUN) / "neighbors.csv")
    if row["rank"] == "1" and row["scope"] == "same_class_primary"
}
for twin_position, twin_case_id in enumerate(twin_case_order):
    twin_got = twin_sample_id[twin_top_index[twin_position, 0]]
    assert twin_got == twin_published_neighbours[twin_case_id], f"latent twin changed for {twin_case_id}"
print(f"reproduces {TWIN_LATENT_RUN} exactly: median top-1 cosine "
      f"{twin_got_latent['median']:.6f}, and all 30 selected sample IDs match")

twin_published_pca = published(TWIN_PCA_RUN)
for twin_class in CLASS_ORDER:
    twin_real_rows = np.flatnonzero(twin_real_class == twin_class)
    twin_synth_rows = np.flatnonzero(twin_synth_class == twin_class)
    twin_distances = (
        NearestNeighbors(n_neighbors=1)
        .fit(twin_morph_synth[twin_synth_rows])
        .kneighbors(twin_morph_real[twin_real_rows], return_distance=True)[0][:, 0]
    )
    twin_got = float(np.median(twin_distances))
    twin_want = twin_published_pca["baseline"][twin_class]["real_to_synthetic_nn_median"]
    assert abs(twin_got - twin_want) < 1e-12, f"morphology reproduction drifted by {abs(twin_got - twin_want):.3e}"
print(f"reproduces {TWIN_PCA_RUN} exactly: median real-to-nearest-synthetic distance "
      f"{[round(twin_published_pca['baseline'][name]['real_to_synthetic_nn_median'], 3) for name in CLASS_ORDER]} "
      f"for {list(CLASS_ORDER)}")

# %% [markdown]
# ### The retrieval, in all three spaces at once
#
# The cell below runs the same search three times for every anchor — euclidean
# on standardised fitted parameters, euclidean in the morphology space, cosine
# in the latent — over the anchor's entire same-class gallery, and records where
# each choice lands on the *other* rulers. Three quantities per anchor:
#
# - the rank of each space's chosen twin in the morphology ordering;
# - the distance of each choice on both rulers;
# - the Spearman rank correlation between distance vectors across the whole
#   same-class gallery — a single number for "do these two metrics order the
#   candidates the same way", where 1.0 means identical ordering and 0.0 means
#   no relationship at all.
#
# A fourth selection rule is carried along: the *hybrid* used by the human
# reviews, which preselects on the four generation parameters and then reranks
# on a coarser envelope-and-spectrum signature
# (`z8_twin_analysis.build_physical_morphology_neighbor_rows`). It is included
# because the pairs a human actually validated were chosen that way.
#
# The parameter-space choice comes from the shipped `nearest_indices`; the full
# distance vector needed for ranks is rebuilt with the same standardisation and
# asserted to select the identical event, so the reported ranking cannot drift
# away from the shipped selection.

# %%
twin_latent = np.asarray(twin_embeddings["synthetic"], dtype=np.float64)
twin_latent /= np.linalg.norm(twin_latent, axis=1, keepdims=True)
twin_latent_real = np.asarray(twin_embeddings["real"], dtype=np.float64)
twin_latent_real /= np.linalg.norm(twin_latent_real, axis=1, keepdims=True)

twin_parameters = parameter_matrix(twin_v4_events, real=False)
twin_parameters_real = parameter_matrix(twin_anchors, real=True)
twin_anchor_position = {row["case_id"]: index for index, row in enumerate(twin_anchors)}

twin_hybrid_choice = {
    row["case_id"]: row["synthetic_sample_id"]
    for row in twin_rows(run_dir(TWIN_HYBRID_RUN) / "neighbors.csv")
    if row["rank"] == "1"
}
twin_row_of_sample = {sample: index for index, sample in enumerate(twin_sample_id)}

twin_measurements = []
for twin_position, twin_case_id in enumerate(twin_case_order):
    anchor = twin_by_case[twin_case_id]
    class_name = anchor["analysis_class"]
    gallery = np.flatnonzero(twin_synth_class == class_name)
    anchor_row = twin_anchor_position[twin_case_id]

    morphology = np.linalg.norm(
        twin_morph_synth[gallery] - twin_morph_real[twin_real_row[anchor["event_id"]]], axis=1
    )
    cosine = 1.0 - twin_latent[gallery] @ twin_latent_real[twin_position]
    scaler = StandardScaler().fit(twin_parameters[gallery])
    parameter = np.linalg.norm(
        scaler.transform(twin_parameters[gallery])
        - scaler.transform(twin_parameters_real[anchor_row][None, :]),
        axis=1,
    )
    shipped_pick, _ = nearest_indices(
        twin_parameters_real[anchor_row][None, :], twin_parameters[gallery]
    )
    if int(np.argmin(parameter)) != int(shipped_pick[0]):
        raise ValueError(f"parameter ranking disagrees with nearest_indices for {twin_case_id}")

    morphology_order = np.argsort(morphology, kind="stable")
    cosine_order = np.argsort(cosine, kind="stable")
    morphology_rank = np.empty(len(gallery), dtype=np.int64)
    morphology_rank[morphology_order] = np.arange(1, len(gallery) + 1)
    cosine_rank = np.empty(len(gallery), dtype=np.int64)
    cosine_rank[cosine_order] = np.arange(1, len(gallery) + 1)

    latent_pick = int(cosine_order[0])
    morphology_pick = int(morphology_order[0])
    parameter_pick = int(shipped_pick[0])
    hybrid_pick = int(np.flatnonzero(gallery == twin_row_of_sample[twin_hybrid_choice[twin_case_id]])[0])

    twin_measurements.append(
        {
            "case_id": twin_case_id,
            "class_name": class_name,
            "colour": CLASS_COLOUR[class_name],
            "gallery": int(len(gallery)),
            "rho_latent": float(spearmanr(cosine, morphology).statistic),
            "rho_parameter": float(spearmanr(parameter, morphology).statistic),
            "latent_morphology_rank": int(morphology_rank[latent_pick]),
            "latent_morphology_distance": float(morphology[latent_pick]),
            "latent_cosine_distance": float(cosine[latent_pick]),
            "parameter_morphology_rank": int(morphology_rank[parameter_pick]),
            "parameter_morphology_distance": float(morphology[parameter_pick]),
            "parameter_distance": float(parameter[parameter_pick]),
            "nearest_morphology_distance": float(morphology[morphology_pick]),
            "nearest_parameter_distance": float(parameter[morphology_pick]),
            "morphology_cosine_rank": int(cosine_rank[morphology_pick]),
            "morphology_cosine_distance": float(cosine[morphology_pick]),
            "hybrid_morphology_distance": float(morphology[hybrid_pick]),
            "hybrid_morphology_rank": int(morphology_rank[hybrid_pick]),
            "latent_is_nearest": bool(latent_pick == morphology_pick),
            "parameter_is_nearest": bool(parameter_pick == morphology_pick),
            "top5_overlap": int(len(set(cosine_order[:5].tolist()) & set(morphology_order[:5].tolist()))),
            "latent_row": int(gallery[latent_pick]),
            "morphology_row": int(gallery[morphology_pick]),
            "parameter_row": int(gallery[parameter_pick]),
            "parameter_cosine_distance": float(cosine[parameter_pick]),
            "cosine_vector": cosine,
            "morphology_vector": morphology,
        }
    )

twin_by_id = {row["case_id"]: row for row in twin_measurements}
twin_latent_agreement = sum(row["latent_is_nearest"] for row in twin_measurements)
twin_parameter_agreement = sum(row["parameter_is_nearest"] for row in twin_measurements)
twin_rho_latent = np.asarray([row["rho_latent"] for row in twin_measurements])
twin_rho_parameter = np.asarray([row["rho_parameter"] for row in twin_measurements])

print(f"anchors measured                             {len(twin_measurements)}")
print(f"latent twin is also the morphology NN        {twin_latent_agreement}/{len(twin_measurements)}"
      f"  ({100.0 * twin_latent_agreement / len(twin_measurements):.1f} %)")
print(f"parameter twin is also the morphology NN     {twin_parameter_agreement}/{len(twin_measurements)}"
      f"  ({100.0 * twin_parameter_agreement / len(twin_measurements):.1f} %)")
print(f"mean shared candidates in the two top-5      "
      f"{np.mean([row['top5_overlap'] for row in twin_measurements]):.3f} of 5")
print(f"Spearman rho, latent vs morphology           median {np.median(twin_rho_latent):.3f}, "
      f"range {twin_rho_latent.min():.3f} to {twin_rho_latent.max():.3f}")
print(f"Spearman rho, parameters vs morphology       median {np.median(twin_rho_parameter):.3f}, "
      f"range {twin_rho_parameter.min():.3f} to {twin_rho_parameter.max():.3f}")

# %% [markdown]
# ### Twin pairs, displayed
#
# Before the disagreement is argued in numbers, it should be seen. The figure
# below shows, for each class, one real anchor beside the synthetic event each
# space calls its nearest neighbour. The anchor is chosen by rule and not by
# eye: within each class it is the anchor whose morphology nearest-neighbour
# distance is the class median, so the panels show a typical case rather than
# the best one.
#
# The faint traces are the raw waveforms, level-removed and RMS-normalised so
# only shape is compared; the bold step curves are the shipped 64-bin envelope
# the descriptor is built from. The shaded band is the 0.512 ms the morphology
# space compares; the latent sees the full 2.048 ms drawn.

# %%
twin_median_case = {}
for twin_class in CLASS_ORDER:
    in_class = sorted(
        (row for row in twin_measurements if row["class_name"] == twin_class),
        key=lambda row: row["nearest_morphology_distance"],
    )
    twin_median_case[twin_class] = in_class[len(in_class) // 2]

twin_pairs = []
for twin_class in CLASS_ORDER:
    chosen = twin_median_case[twin_class]
    position = int(np.flatnonzero(twin_case_order == chosen["case_id"])[0])
    twin_pairs.append(
        {
            "case_id": chosen["case_id"],
            "class_name": twin_class,
            "colour": CLASS_COLOUR[twin_class],
            "real": np.asarray(twin_real_raw[position], dtype=np.float32),
            "morphology": {
                "signal": np.asarray(twin_v4_signals[chosen["morphology_row"]], dtype=np.float32),
                "sample_id": twin_sample_id[chosen["morphology_row"]],
                "morphology_distance": chosen["nearest_morphology_distance"],
                "cosine_distance": chosen["morphology_cosine_distance"],
            },
            "latent": {
                "signal": np.asarray(twin_v4_signals[chosen["latent_row"]], dtype=np.float32),
                "sample_id": twin_sample_id[chosen["latent_row"]],
                "morphology_distance": chosen["latent_morphology_distance"],
                "cosine_distance": chosen["latent_cosine_distance"],
            },
        }
    )

plot_twin_pairs(twin_pairs)
plt.show()

for twin_class in CLASS_ORDER:
    chosen = twin_median_case[twin_class]
    print(f"{chosen['case_id']:<18} morphology twin at {chosen['nearest_morphology_distance']:5.2f} "
          f"(cosine rank {chosen['morphology_cosine_rank']:>6,})   "
          f"latent twin at {chosen['latent_morphology_distance']:5.2f} "
          f"(morphology rank {chosen['latent_morphology_rank']:>6,} of {chosen['gallery']:,})")

# %% [markdown]
# The 4 µm row is the case the chain is built on: two bursts of the same width,
# the same rise, the same decay, and a reviewer would have no reason to reject
# either candidate. The 2 µm row shows the honest limit — the real event carries
# a second burst at 1.75 ms that no single-event simulator produces, and neither
# candidate reproduces it. The 10 µm row shows the noisiest regime, where the
# envelope agrees in trend and disagrees in detail.
#
# The two columns are also already the finding. For every one of these three
# anchors the two spaces return a *different* synthetic event, and the latent's
# choice sits several units further away on the morphology ruler than the
# morphology's own choice.

# %% [markdown]
# ### Close in fitted parameters is not close in signal
#
# The same question can be asked of the space the generator is actually written
# in. Every synthetic event is drawn from four numbers — amplitude, frequency,
# decay time, target SNR — so the natural expectation is that matching those
# four numbers matches the waveform. It does not, and this is the sharpest
# single justification for measuring shape at all rather than trusting the
# parameter sheet.
#
# The figure below shows, per class, the real anchor, the synthetic event
# nearest to it in fitted-parameter space, and the synthetic event nearest to it
# in morphology space. The anchor shown is the class's **sharpest**
# counterexample, selected by rule as the anchor whose parameter-nearest
# neighbour lands worst in the morphology ordering; the class medians are
# printed underneath so the extreme is not mistaken for the typical.

# %%
def twin_parameter_text(row, *, real):
    amplitude = float(row["particles2snr_amplitude" if real else "amplitude_p0"])
    frequency = float(row["frequency_hz"]) / 1000.0 if real else float(row["frequency_khz"])
    return (f"P0 {amplitude:.3g} · f {frequency:.1f} kHz · "
            f"tau {float(row['tau_ms']):.3f} ms · SNR {float(row['snr_db']):.1f} dB")


twin_triptych = []
for twin_class in CLASS_ORDER:
    sharpest = max(
        (row for row in twin_measurements if row["class_name"] == twin_class),
        key=lambda row: row["parameter_morphology_rank"],
    )
    position = int(np.flatnonzero(twin_case_order == sharpest["case_id"])[0])
    twin_triptych.append(
        {
            "case_id": sharpest["case_id"],
            "class_name": twin_class,
            "colour": CLASS_COLOUR[twin_class],
            "gallery": sharpest["gallery"],
            "real": np.asarray(twin_real_raw[position], dtype=np.float32),
            "real_parameters": twin_parameter_text(twin_by_case[sharpest["case_id"]], real=True),
            "parameter": {
                "signal": np.asarray(twin_v4_signals[sharpest["parameter_row"]], dtype=np.float32),
                "sample_id": twin_sample_id[sharpest["parameter_row"]],
                "parameters": twin_parameter_text(twin_v4_events[sharpest["parameter_row"]], real=False),
                "morphology_distance": sharpest["parameter_morphology_distance"],
                "morphology_rank": sharpest["parameter_morphology_rank"],
            },
            "morphology": {
                "signal": np.asarray(twin_v4_signals[sharpest["morphology_row"]], dtype=np.float32),
                "sample_id": twin_sample_id[sharpest["morphology_row"]],
                "parameters": twin_parameter_text(twin_v4_events[sharpest["morphology_row"]], real=False),
                "morphology_distance": sharpest["nearest_morphology_distance"],
                "morphology_rank": 1,
            },
        }
    )

twin_sharpest_4um = twin_by_id[
    max(
        (row for row in twin_measurements if row["class_name"] == "4um"),
        key=lambda row: row["parameter_morphology_rank"],
    )["case_id"]
]
assert twin_sharpest_4um["case_id"] == "physical-4um-02", "the narrated 4 µm counterexample moved"
assert twin_sharpest_4um["parameter_morphology_rank"] == 31_793, "the narrated 4 µm rank moved"

plot_parameter_triptych(twin_triptych)
plt.show()

for twin_class in CLASS_ORDER:
    in_class = [row for row in twin_measurements if row["class_name"] == twin_class]
    ranks = np.asarray([row["parameter_morphology_rank"] for row in in_class])
    print(f"{twin_class:>5}  parameter-nearest lands at morphology rank: "
          f"median {np.median(ranks):>8,.0f}, worst {ranks.max():>8,} of {in_class[0]['gallery']:,}   "
          f"(shown: {max(in_class, key=lambda row: row['parameter_morphology_rank'])['case_id']})")

# %% [markdown]
# The 4 µm panel is the strongest form of the claim in this whole notebook, and
# it is worth reading the printed numbers rather than only the traces. The
# parameter-nearest synthetic event reproduces all four measured coordinates to
# within a few percent — P0 0.325 against 0.314, 35.3 kHz against 34.7,
# τ 0.134 ms against 0.135, SNR −0.4 dB against −1.0 — and lands at morphology
# rank 31,793 of 32,810. Better than 97 % of the same-class gallery resembles
# that real event more than the parameter match does. The morphology-nearest
# event does the opposite: it disagrees with the parameter sheet by a factor 2.3
# in amplitude and by 12 dB of SNR, and its envelope tracks the measured one.
#
# The reason is not mysterious. Two of the four coordinates are fitted from the
# real event under an ideal model, and the residual — the part of the waveform
# the ideal model does not explain, which is where noise, overlap and shape
# asymmetry live — is exactly what the eye compares and what the parameter sheet
# does not carry. The 12 dB SNR discrepancy is the same statement seen from the
# other side: the fitted SNR of this real event describes the ideal fit, not the
# burst a reviewer sees.
#
# The 2 µm and 10 µm rows are the honest limit of a visual argument. Those
# anchors are noise-dominated (fitted SNR −8.5 dB and −3.1 dB), so no reader can
# adjudicate the pairing by eye, and the figure does not ask them to — the ranks
# carry the claim there. Only the 4 µm row is decidable visually, and it is the
# row where the distances say the same thing.

# %% [markdown]
# ### The disagreement, measured across all three spaces
#
# The scatter below puts one anchor's entire same-class gallery on two rulers at
# once: each hexagon is a bin of synthetic candidates, positioned by its cosine
# distance in the latent and its euclidean distance in the morphology space. If
# the two metrics meant the same thing the cloud would be a rising line. It is a
# blob, and the three selection rules land in three different corners of it.

# %%
twin_scatter_case = twin_median_case["4um"]
twin_scatter_sample = {
    "case_id": twin_scatter_case["case_id"],
    "gallery": twin_scatter_case["gallery"],
    "cosine": twin_scatter_case["cosine_vector"],
    "morphology": twin_scatter_case["morphology_vector"],
    "rho": twin_scatter_case["rho_latent"],
    "latent_choice": {
        "cosine": twin_scatter_case["latent_cosine_distance"],
        "morphology": twin_scatter_case["latent_morphology_distance"],
    },
    "parameter_choice": {
        "cosine": twin_scatter_case["parameter_cosine_distance"],
        "morphology": twin_scatter_case["parameter_morphology_distance"],
    },
    "morphology_choice": {
        "cosine": twin_scatter_case["morphology_cosine_distance"],
        "morphology": twin_scatter_case["nearest_morphology_distance"],
    },
}
plot_metric_scatter(twin_scatter_sample)
plt.show()

# %%
plot_rho_strip(
    [
        {
            "label": "fitted parameters\nvs morphology",
            "values": [row["rho_parameter"] for row in twin_measurements],
            "colours": [row["colour"] for row in twin_measurements],
        },
        {
            "label": "Conv1D-GAP latent\nvs morphology",
            "values": [row["rho_latent"] for row in twin_measurements],
            "colours": [row["colour"] for row in twin_measurements],
        },
    ],
    title="One rank correlation per anchor, over its full same-class gallery",
    ylabel="Spearman rho against the morphology ordering",
    figsize=(8.4, 4.8),
)
plt.show()

for twin_class in CLASS_ORDER:
    in_class = [row for row in twin_measurements if row["class_name"] == twin_class]
    latent_values = np.asarray([row["rho_latent"] for row in in_class])
    parameter_values = np.asarray([row["rho_parameter"] for row in in_class])
    print(f"{twin_class:>5}  rho median: parameters {np.median(parameter_values):6.3f}   "
          f"latent {np.median(latent_values):6.3f}   "
          f"latent range {latent_values.min():6.3f} to {latent_values.max():6.3f}   "
          f"gallery {in_class[0]['gallery']:,}")

# %% [markdown]
# A median rank correlation of 0.33 for the latent and 0.48 for the parameters
# is the whole argument. None of these spaces is unrelated to the others — a
# synthetic event that is absurd in one is usually absurd in all, which is why
# the correlations are positive — but they are nowhere near interchangeable, and
# the disagreement is worst exactly where it matters, among the closest
# candidates. The 2 µm class is the extreme: its median latent correlation is
# 0.12 and two of its anchors are slightly negative, meaning that for those
# events the latent's ordering of 11,510 candidates carries essentially no
# information about the morphology ordering.
#
# The sharpest objection to this framing is that a rank correlation over an
# entire gallery is dominated by the far field, where nobody cares about the
# ordering. So the next measurement looks only at the top: where does each
# space's *chosen* twin land on the morphology ruler?

# %%
plot_distance_gap(twin_measurements)
plt.show()

twin_gallery_size = np.asarray([row["gallery"] for row in twin_measurements])


def twin_report(label, distance_key, rank_key):
    distances = np.asarray([row[distance_key] for row in twin_measurements])
    ranks = np.asarray([row[rank_key] for row in twin_measurements])
    print(f"{label:<34} median distance {np.median(distances):5.2f}   "
          f"median morphology rank {np.median(ranks):>8,.0f}   "
          f"({100.0 * np.median(ranks / twin_gallery_size):4.1f}th percentile)   "
          f"is the nearest neighbour {sum(ranks == 1)}/{len(ranks)}")


twin_nearest_distance = np.asarray([row["nearest_morphology_distance"] for row in twin_measurements])
twin_latent_distance = np.asarray([row["latent_morphology_distance"] for row in twin_measurements])
twin_reverse_rank = np.asarray([row["morphology_cosine_rank"] for row in twin_measurements])
print(f"{'morphology nearest neighbour':<34} median distance {np.median(twin_nearest_distance):5.2f}")
twin_report("twin chosen by fitted parameters", "parameter_morphology_distance", "parameter_morphology_rank")
twin_report("twin chosen by physical + rerank", "hybrid_morphology_distance", "hybrid_morphology_rank")
twin_report("twin chosen by latent cosine", "latent_morphology_distance", "latent_morphology_rank")
print()
print(f"the morphology twin, seen by the latent: median cosine rank {np.median(twin_reverse_rank):,.0f} "
      f"({100.0 * np.median(twin_reverse_rank / twin_gallery_size):.1f}th percentile) — "
      f"the disagreement is symmetric, not a defect of one space")
print(f"latent choice / nearest distance ratio: median "
      f"{np.median(twin_latent_distance / twin_nearest_distance):.2f}x")

# %% [markdown]
# ### The central finding, reproduced on a human-validated pair
#
# The carried claim behind this section is one specific pair: a twin a human
# reviewer accepted sits 9.7 away in the morphology space, where the true
# nearest neighbour of that same real event sits at 4.1. That pair comes from a
# different arm of the chain — the paired-asymmetry review, whose gallery is the
# **v5** synthetic dataset and whose thirty judgments are stored in
# `particle-z8-v2-paired-asymmetry-twin-result-r2`. The cell below recomputes
# both distances in the stored morphology basis for every pair a reviewer
# accepted.

# %%
twin_human_cases = twin_rows(run_dir(TWIN_HUMAN_RUN) / "human_paired_twin_cases.csv")
twin_validated = []
for twin_case_row in twin_human_cases:
    if twin_case_row["paired_asymmetry_success"] != "True":
        continue
    anchor = twin_by_case[twin_case_row["case_id"]]
    gallery = np.flatnonzero(twin_synth_class == anchor["analysis_class"])
    distances = np.linalg.norm(
        twin_morph_v5[gallery] - twin_morph_real[twin_real_row[anchor["event_id"]]], axis=1
    )
    order = np.argsort(distances, kind="stable")
    ranks = np.empty(len(gallery), dtype=np.int64)
    ranks[order] = np.arange(1, len(gallery) + 1)
    picked = int(
        np.flatnonzero(gallery == twin_row_of_sample[twin_case_row["paired_asymmetry_synthetic_sample_id"]])[0]
    )
    twin_validated.append(
        {
            "case_id": twin_case_row["case_id"],
            "class_name": anchor["analysis_class"],
            "gallery": int(len(gallery)),
            "selected_sample_id": twin_case_row["paired_asymmetry_synthetic_sample_id"],
            "selected_distance": float(distances[picked]),
            "selected_rank": int(ranks[picked]),
            "selected_row": int(gallery[picked]),
            "nearest_distance": float(distances[order[0]]),
            "nearest_sample_id": twin_sample_id[gallery[order[0]]],
            "nearest_row": int(gallery[order[0]]),
            "distances": distances,
        }
    )

twin_validated.sort(key=lambda row: -row["selected_distance"])
print(f"{'case':<18}{'class':>6}{'d(accepted)':>13}{'d(nearest)':>12}{'rank of accepted':>22}")
for twin_row in twin_validated:
    print(f"{twin_row['case_id']:<18}{twin_row['class_name']:>6}{twin_row['selected_distance']:13.2f}"
          f"{twin_row['nearest_distance']:12.2f}{twin_row['selected_rank']:>15,} /{twin_row['gallery']:>6,}")

twin_headline = next(row for row in twin_validated if row["case_id"] == "physical-2um-01")
print()
print("carried claim: accepted twin at 9.7, true nearest neighbour at 4.1")
print(f"measured     : accepted twin at {twin_headline['selected_distance']:.2f}, "
      f"true nearest neighbour at {twin_headline['nearest_distance']:.2f} "
      f"({twin_headline['case_id']}, rank {twin_headline['selected_rank']:,} of {twin_headline['gallery']:,})")
assert round(twin_headline["selected_distance"], 1) == 9.7
assert round(twin_headline["nearest_distance"], 1) == 4.1
print("the carried pair reproduces to the digit")

twin_accepted = np.asarray([row["selected_distance"] for row in twin_validated])
twin_accepted_nn = np.asarray([row["nearest_distance"] for row in twin_validated])
print(f"and it is not the exception: across all {len(twin_validated)} accepted pairs the median "
      f"accepted distance is {np.median(twin_accepted):.2f} against a median nearest-neighbour "
      f"distance of {np.median(twin_accepted_nn):.2f}")

# %%
twin_case = {
    "case_id": twin_headline["case_id"],
    "colour": CLASS_COLOUR[twin_headline["class_name"]],
    "gallery": twin_headline["gallery"],
    "gallery_distances": twin_headline["distances"],
    "real": np.asarray(
        twin_real_raw[int(np.flatnonzero(twin_case_order == twin_headline["case_id"])[0])],
        dtype=np.float32,
    ),
    "selected": {
        "label": "accepted by the reviewer",
        "signal": np.asarray(twin_v5_signals[twin_headline["selected_row"]], dtype=np.float32),
        "sample_id": twin_headline["selected_sample_id"],
        "morphology_distance": twin_headline["selected_distance"],
        "morphology_rank": twin_headline["selected_rank"],
    },
    "nearest": {
        "label": "nearest in the morphology space, never shown",
        "signal": np.asarray(twin_v5_signals[twin_headline["nearest_row"]], dtype=np.float32),
        "sample_id": twin_headline["nearest_sample_id"],
        "morphology_distance": twin_headline["nearest_distance"],
        "morphology_rank": 1,
    },
}
plot_validated_counterexample(twin_case)
plt.show()

# %% [markdown]
# The figure needs one caveat stated plainly: this anchor is noise-dominated at
# a fitted −8.5 dB, and neither candidate is separable from noise by eye. That
# is not a defect of the display, it is the situation the reviewer was in. When
# the waveform itself cannot settle the question, the verdict is whatever metric
# was put in front of the reviewer — which is exactly why the choice of space
# has to be made deliberately rather than inherited.
#
# One correction belongs here, because the carried claim attributes this gap to
# the wrong cause. The paired-asymmetry review did **not** select its candidates
# by latent cosine. Its tool,
# `tools/benchmarks/analyze_z8_paired_asymmetry_twins.py`, calls
# `build_physical_morphology_neighbor_rows`, which preselects on the four
# generation parameters and reranks on a coarser envelope-and-spectrum
# signature, and it records the latent cosine explicitly as
# `"latent_cosine_role": "diagnostic_only"`. The accepted pair was never a
# cosine choice.
#
# The measurement above is what settles it. On the same thirty anchors the
# parameter rule lands at a median morphology distance of 7.98, the hybrid at
# 7.56, and cosine at 8.89, against a nearest neighbour at 4.04 — and *none* of
# the three ever picks the morphology nearest neighbour, in any of the thirty
# anchors. The gap is not a property of the classifier latent. It is the generic
# consequence of choosing the twin in one space and measuring it in another.
# That is a stronger statement than the carried one, and it is the one the
# numbers support.

# %% [markdown]
# ### Is it only the window? — exploratory control
#
# The obvious objection to the latent comparison is that the two spaces are not
# being asked the same question at all: the morphology descriptor reads 1024
# samples, the classifier reads 4096. Perhaps the disagreement is a window
# mismatch rather than a representational one.
#
# This sub-section is exploratory and its basis is new, so it is labelled as
# such. It refits the morphology space from scratch with the shipped
# `fit_synthetic_pca`, once on the 1024-sample core and once on the full 4096
# window, using the same balanced fit (3660 events per class, seed 20260724) and
# the same 16 components. Three correlations then separate the effects: refit
# against stored basis (does the refit reproduce the space?), 1024 against 4096
# inside the morphology space (how much does the window alone change the
# ordering?), and 4096 morphology against the latent (does matching the window
# make them agree?).

# %%
twin_start = time.time()
twin_features = {
    "core": np.concatenate(
        [
            morphology_features(np.asarray(twin_v4_signals[start : start + 512, twin_core]))
            for start in range(0, len(twin_v4_signals), 512)
        ]
    ),
    "full": np.concatenate(
        [
            morphology_features(np.asarray(twin_v4_signals[start : start + 512]))
            for start in range(0, len(twin_v4_signals), 512)
        ]
    ),
}
twin_bases = {}
for twin_name, twin_feature_block in twin_features.items():
    twin_scaler, twin_pca_model, _ = fit_synthetic_pca(
        twin_feature_block, twin_synth_class, per_class=3660, seed=20260724
    )
    twin_anchor_features = morphology_features(
        twin_real_raw[:, twin_core] if twin_name == "core" else twin_real_raw
    )
    twin_bases[twin_name] = {
        "synthetic": twin_pca_model.transform(twin_scaler.transform(twin_feature_block)),
        "real": twin_pca_model.transform(twin_scaler.transform(twin_anchor_features)),
        "dimension": int(twin_feature_block.shape[1]),
    }
print(f"refitted both morphology bases in {time.time() - twin_start:.1f} s "
      f"({twin_bases['core']['dimension']}-D core descriptor, "
      f"{twin_bases['full']['dimension']}-D full-window descriptor)")

twin_control = {
    "refit vs stored\n(both 1024)": [],
    "1024 vs 4096\n(morphology only)": [],
    "stored 1024\nvs latent": [],
    "refit 4096\nvs latent": [],
}
twin_matched = []
for twin_position, twin_case_id in enumerate(twin_case_order):
    row = twin_by_id[twin_case_id]
    gallery = np.flatnonzero(twin_synth_class == row["class_name"])
    stored = row["morphology_vector"]
    cosine = row["cosine_vector"]
    core = np.linalg.norm(twin_bases["core"]["synthetic"][gallery] - twin_bases["core"]["real"][twin_position], axis=1)
    full = np.linalg.norm(twin_bases["full"]["synthetic"][gallery] - twin_bases["full"]["real"][twin_position], axis=1)
    twin_control["refit vs stored\n(both 1024)"].append(float(spearmanr(stored, core).statistic))
    twin_control["1024 vs 4096\n(morphology only)"].append(float(spearmanr(core, full).statistic))
    twin_control["stored 1024\nvs latent"].append(float(spearmanr(stored, cosine).statistic))
    twin_control["refit 4096\nvs latent"].append(float(spearmanr(full, cosine).statistic))

    order = np.argsort(full, kind="stable")
    ranks = np.empty(len(gallery), dtype=np.int64)
    ranks[order] = np.arange(1, len(gallery) + 1)
    pick = int(np.argmin(cosine))
    twin_matched.append(
        {
            "same_choice": bool(pick == int(order[0])),
            "rank": int(ranks[pick]),
            "distance": float(full[pick]),
            "nearest": float(full[order[0]]),
        }
    )

plot_rho_strip(
    [
        {"label": label, "values": values, "colour": colour}
        for (label, values), colour in zip(
            twin_control.items(), ("#6b7a8d", "#0f766e", "#c2402a", "#b45309")
        )
    ],
    title="Matching the window narrows the gap without closing it",
    ylabel="Spearman rho over the same-class gallery",
)
plt.show()

twin_matched_agreement = sum(row["same_choice"] for row in twin_matched)
print("with the window matched at 4096 samples:")
print(f"  latent twin is also the morphology NN     {twin_matched_agreement}/{len(twin_matched)}")
print(f"  median morphology rank of the latent twin {np.median([row['rank'] for row in twin_matched]):,.0f}")
print(f"  median distance {np.median([row['distance'] for row in twin_matched]):.2f} "
      f"against a nearest neighbour at {np.median([row['nearest'] for row in twin_matched]):.2f}")

# %% [markdown]
# The control splits the effect cleanly. The refit reproduces the stored basis
# almost perfectly (rho 0.99), so nothing here depends on a private
# reimplementation. Widening the morphology window from 1024 to 4096 samples
# changes the ordering substantially on its own (rho 0.60 between the two
# windows of the *same* descriptor) — a real and slightly uncomfortable result,
# since it means the morphology space's verdict is itself window-dependent.
# Matching the window does move the latent closer to the morphology space, from
# rho 0.33 to 0.48. So roughly a third of the disagreement is window, and the
# rest is representation.
#
# Agreement at the top does not recover at all: with the window matched, the
# latent's twin is still never the morphology nearest neighbour, 0 of 30, and
# still sits at a median 6.54 where the nearest neighbour is at 3.55. The window
# is a contributing cause, not the cause.

# %% [markdown]
# ### Why this is the expected outcome
#
# None of this makes the latent defective. It makes it *specific*. The frozen
# Conv1D-GAP network was trained on one task: separate 2 µm from 4 µm from
# 10 µm. A representation optimised for that objective is rewarded for throwing
# away every variation that does not move an event across a class boundary —
# and within-class variation in envelope shape, decay, and spectral tilt is
# exactly what it is rewarded for discarding. Twin identity is made of precisely
# that discarded variation. The latent is being asked for information its
# training deliberately removed.
#
# This is why an earlier result in the chain lands where it does: on exact-parent
# retrieval, where the target is a single known correct answer, the morphology
# space beats the latent by more than a factor of two (Recall@5 23.3 % against
# 11.0 %; that measurement belongs to its own run and is not recomputed here).
# The present section supplies the mechanism for that gap rather than repeating
# the number. A metric that collapses within-class differences cannot retrieve
# an individual, and a twin is an individual.
#
# The parameter space fails for the mirror-image reason. It carries only what
# the ideal model explains and nothing of the residual, so two events with the
# same four coordinates can differ in everything a reviewer looks at.
#
# The practical consequence for the chain is a rule, not a ranking: **choose the
# twin in the space you intend to defend it in.** A pair selected under one
# metric and reported under another will be a counterexample to itself, in
# either direction, and the disagreement is large enough — median rank between
# the 5th and the 20th percentile of a gallery of thousands, whichever pair of
# spaces is compared — that it will happen almost every time.

# %% [markdown]
# ### What this does not claim
#
# The thirty anchors are frozen **development** validation events. Nothing here
# is a validation result, no class coverage is claimed, and no sealed test row
# was read — the registered development table contains no test split at all,
# which the first cell asserts.
#
# The comparison is between *specific* fitted objects on one gallery: the
# 16-component morphology PCA of `particle-z8-v2-paired-asymmetry-pca-r2`, the
# penultimate layer of one frozen checkpoint, and the four-coordinate parameter
# sheet of the v4 generator. It does not show that morphology metrics generally
# beat learned metrics, and it does not show that a representation trained for
# retrieval rather than classification would fail the same way — that would need
# a different frozen model and a separate run.
#
# Two limits were hit and left visible. First, the review run that produced the
# human judgments, `particle-z8-v2-paired-asymmetry-twin-analysis-r2`, has been
# pruned; only its result run survives. The anchor identities were therefore
# recovered from a surviving sibling run, whose anchor table was checked against
# the human case list before use. Second, the human decisions themselves are
# read as published outcomes — this notebook cannot re-run a blind review, so
# the reviewer's acceptance is taken as given and only the distances are
# recomputed.

# %%
try:
    twin_emitted = notebook_evidence.emit_run(
        workspace,
        section="twins-space-disagreement",
        metrics={
            "schema_version": 1,
            "analysis": (
                "twin selection compared across the fitted-parameter, morphology "
                "and Conv1D-GAP latent spaces on the frozen 30-anchor panel"
            ),
            "anchors": len(twin_measurements),
            "gallery_per_class": {
                name: int(twin_median_case[name]["gallery"]) for name in CLASS_ORDER
            },
            "rank_correlation_against_morphology": {
                "latent": {
                    "median": float(np.median(twin_rho_latent)),
                    "mean": float(twin_rho_latent.mean()),
                    "minimum": float(twin_rho_latent.min()),
                    "maximum": float(twin_rho_latent.max()),
                    "by_class": {
                        name: float(np.median([
                            row["rho_latent"] for row in twin_measurements
                            if row["class_name"] == name
                        ]))
                        for name in CLASS_ORDER
                    },
                },
                "parameters": {
                    "median": float(np.median(twin_rho_parameter)),
                    "mean": float(twin_rho_parameter.mean()),
                    "minimum": float(twin_rho_parameter.min()),
                    "maximum": float(twin_rho_parameter.max()),
                    "by_class": {
                        name: float(np.median([
                            row["rho_parameter"] for row in twin_measurements
                            if row["class_name"] == name
                        ]))
                        for name in CLASS_ORDER
                    },
                },
            },
            "is_morphology_nearest_neighbour": {
                "latent": int(twin_latent_agreement),
                "parameters": int(twin_parameter_agreement),
                "hybrid": int(sum(row["hybrid_morphology_rank"] == 1 for row in twin_measurements)),
                "total": len(twin_measurements),
            },
            "mean_top5_overlap_latent_morphology": float(
                np.mean([row["top5_overlap"] for row in twin_measurements])
            ),
            "morphology_distance_median": {
                "nearest_neighbour": float(np.median(twin_nearest_distance)),
                "parameter_choice": float(np.median([
                    row["parameter_morphology_distance"] for row in twin_measurements
                ])),
                "hybrid_choice": float(np.median([
                    row["hybrid_morphology_distance"] for row in twin_measurements
                ])),
                "latent_choice": float(np.median(twin_latent_distance)),
            },
            "morphology_rank_median": {
                "parameter_choice": float(np.median([
                    row["parameter_morphology_rank"] for row in twin_measurements
                ])),
                "hybrid_choice": float(np.median([
                    row["hybrid_morphology_rank"] for row in twin_measurements
                ])),
                "latent_choice": float(np.median([
                    row["latent_morphology_rank"] for row in twin_measurements
                ])),
            },
            "latent_rank_median_of_morphology_choice": float(np.median(twin_reverse_rank)),
            "window_control_rho_median": {
                label.replace("\n", " "): float(np.median(values))
                for label, values in twin_control.items()
            },
            "matched_window_4096": {
                "is_morphology_nearest_neighbour": int(twin_matched_agreement),
                "median_rank": float(np.median([row["rank"] for row in twin_matched])),
                "median_distance": float(np.median([row["distance"] for row in twin_matched])),
                "median_nearest_distance": float(np.median([row["nearest"] for row in twin_matched])),
            },
            "human_validated_pairs": {
                "count": len(twin_validated),
                "median_accepted_distance": float(np.median(twin_accepted)),
                "median_nearest_distance": float(np.median(twin_accepted_nn)),
                "headline_case": twin_headline["case_id"],
                "headline_accepted_distance": twin_headline["selected_distance"],
                "headline_nearest_distance": twin_headline["nearest_distance"],
                "headline_accepted_rank": twin_headline["selected_rank"],
            },
        },
        provenance={
            "datasets": dataset_provenance(),
            "inputs": {
                "morphology_scores": f"{TWIN_PCA_RUN}/pca_scores.npz",
                "latent_embeddings": f"{TWIN_LATENT_RUN}/embeddings.npz",
                "latent_neighbours": f"{TWIN_LATENT_RUN}/neighbors.csv",
                "hybrid_neighbours": f"{TWIN_HYBRID_RUN}/neighbors.csv",
                "human_decisions": f"{TWIN_HUMAN_RUN}/human_paired_twin_cases.csv",
            },
            "parameters": {
                "anchors": "the frozen 30 physical validation anchors, reused unchanged",
                "gallery": "same-class synthetic events of the v4 baseline arm",
                "parameter_metric": "euclidean over the four standardised fitted coordinates",
                "morphology_metric": "euclidean over 16 PCA components of the 101-D descriptor",
                "latent_metric": "cosine over the L2-normalised 512-D penultimate activation",
                "window_control_seed": 20260724,
                "window_control_per_class": 3660,
            },
            "metric_definitions": {
                "rank_correlation": (
                    "Spearman rho between two distance vectors over the anchor's "
                    "whole same-class gallery"
                ),
                "morphology_rank": (
                    "1-based position of a candidate in the morphology distance "
                    "ordering of its same-class gallery"
                ),
                "is_morphology_nearest_neighbour": (
                    "the rule returns the same rank-1 synthetic event as the "
                    "morphology space"
                ),
            },
        },
        claim_boundary=(
            "Measures how far the twin chosen in one space lands in the others, "
            "on 30 frozen development validation anchors against the v4 synthetic "
            "gallery. It quantifies disagreement between three specific fitted "
            "spaces; it does not validate any of them, does not establish class "
            "coverage, and does not generalise beyond this checkpoint, this "
            "morphology basis and this generator."
        ),
    )
    print(f"emitted {twin_emitted.name}")
except WorkspaceError as error:
    print(f"no evidence emitted ({error})")


# %% [markdown]
# ## Masked reconstruction · training a model with no labels
#
# Everything up to here has been about the *simulator*: whether its cloud
# covers the real one, whether a regenerated event can find its parent. This
# section is about the *model* the simulator exists to train, and about the one
# design choice that separates two candidate training procedures.
#
# There are roughly two thousand annotated real events. That is far too few to
# train a representation, and the annotation is the very thing we would like
# not to depend on. **Self-supervised learning** (SSL) removes the label from
# the problem: part of the input is hidden, the model is asked to predict it,
# and the error on the hidden part is the loss. Nobody has to say what the
# signal *is*; the signal supervises itself. The deliverable at the end is the
# encoder, not the reconstruction — the decoder is thrown away.
#
# The question this section settles is not *whether* to mask, but **where**.
# Two policies were trained under identical conditions and compared:
#
# - **P25** hides a quarter of the trace in short patches drawn blind to the
#   signal;
# - **CYCLIC25** hides the same quarter, but schedules wide windows aimed at
#   the annotated event and balances them against background.
#
# The section reproduces that comparison from the shipped code and the
# manifested runs, and then measures a defect in how CYCLIC25's "event" was
# defined — one that no run owns and that the deck does not mention.

# %%
import hashlib  # noqa: E402
import statistics  # noqa: E402
from pathlib import Path  # noqa: E402

from p3_ssl.bead_ssl import make_model  # noqa: E402
from p3_ssl.bead_ssl_v2 import load_bead_ssl_v2_config  # noqa: E402
from p3_ssl.masking import (  # noqa: E402
    PatchSpec,
    build_balanced_event_mask_cycle,
    build_patch_aligned_isolated_masks,
    mask_spans,
)
from particles2snr.z8_asymmetry_generation import clean_waveforms  # noqa: E402
from scipy.ndimage import uniform_filter1d  # noqa: E402
from scipy.signal import hilbert  # noqa: E402

SSL_ROOT = workspace.root / "unsupervised-learning-flow-cytometry"
SSL_CONFIG = SSL_ROOT / "configs/bead_ssl_z8_v5_v2.yaml"
SSL_RUNS = workspace.artifacts_root / "unsupervised-learning-flow-cytometry" / "runs"
MATCHED_SEEDS = (42, 43, 44, 45, 46)
INPUT_LENGTH = 4096


def matched_run(policy, seed):
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
# read only there. The visible samples cost nothing whether they are
# reproduced or not, so the model cannot win by copying its input.
#
# The loss *family* in the config is a composite: a signal term, a derivative
# term and an energy term, the latter two robustified with a pseudo-Huber of
# δ = 1 (a loss that is quadratic near zero and linear far from it, so a few
# large residuals cannot dominate). The **selected cell is B0**, and B0 sets
# both robust weights to zero. So the objective that actually trained these
# models is plain masked mean squared error, and saying "Huber loss" would
# overstate it. Worth stating explicitly, because the config file reads as
# though a robust loss were in use.
#
# `sample_visibility_v1` is the second piece that has to be right. A hidden
# sample is replaced by zero — but a genuine zero crossing is also zero, and
# the model must not confuse the two. So the mask is handed to the network as
# an explicit channel: a learned embedding of the per-sample visibility pattern
# is added to every token. This also lets a masking window sit off the token
# grid, which matters below, since CYCLIC25's windows stride 8 while tokens
# stride 16.

# %%
for cell, weights in config["loss"]["cells"].items():
    marker = "  <- selected" if cell == config["loss"]["selected_cell"] else ""
    print(f"  {cell}: signal {weights['lambda_signal']}, derivative "
          f"{weights['lambda_derivative']}, energy {weights['lambda_energy']}{marker}")
print(f"\nhuber_delta {config['loss']['huber_delta']} applies to the derivative and "
      "energy terms only; under B0 both carry weight 0, so the trained objective "
      "is masked MSE on the hidden samples")

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
example_spans = mask_spans(example_mask)
print(f"real validation event {p25_examples['sample_id'][example_index]} · "
      f"{int(example_mask.sum())} of {example_mask.size} samples hidden "
      f"({100 * example_mask.mean():.0f} %)")
draw_model_view(example_signal, example_mask, example_spans)

# %% [markdown]
# ### Why the placement of the hole is a scientific choice
#
# If a trace were uniformly informative, where you hide would not matter. It is
# not. The simulator knows exactly where each event sits, because it put it
# there: the event is centred at `t0_fraction × (N − 1)` and extends ±3τ,
# stretched by the asymmetry `a` on each side. Measured with the training
# dataset's own formula over all 47,980 v5 events, that support is a minority
# of the crop.

# %%
V5_KEY = ("particles2snr-fbase-z8-cholesky-physicalcorr-effective-snr-"
          "synthetic-events@v5")
v5_root = dataset_root(V5_KEY)
with (v5_root / "events.csv").open(newline="", encoding="utf-8") as handle:
    v5_rows = list(csv.DictReader(handle))
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
# So roughly four fifths of every training example is background. A mask drawn
# uniformly spends four fifths of its budget there. That is the argument for
# aiming — and the hypothesis the comparison below is meant to test.
#
# **Hold on to the 22.1 %.** It is measured correctly here, in the sense that
# it is exactly what the training code computed. The last part of this section
# shows it is not what the data says.

# %% [markdown]
# ### The two policies, on one trace, at one budget
#
# Both masks are rebuilt here from the shipped builders in `p3_ssl.masking`,
# with the production `cyclic25` parameters and the runs' seed. Nothing about
# the geometry is drawn by hand.
#
# - **P25** — `build_patch_aligned_isolated_masks` picks whole 16-sample tokens
#   at random and forbids two selected tokens from touching. With 256 tokens
#   and a 25 % ratio that is 64 selections, and the isolation rule makes the
#   geometry deterministic: 64 runs of exactly 16 samples, whatever the seed.
# - **CYCLIC25** — `build_balanced_event_mask_cycle` works on a finer grid
#   (16-sample windows every 8 samples), takes 32 windows intersecting the
#   event and 32 background windows per pass, and requires that windows
#   selected within one pass be mutually disjoint. Adjacent event windows merge
#   into long contiguous runs.
#
# The deck draws this on a real validation trace with an *idealised* 512-sample
# support rather than that event's own annotation, so the two policies can be
# compared on identical geometry. Reproduced exactly here, including that
# choice, so the published numbers are checkable.

# %%
DECK_EVENT_START, DECK_EVENT_WIDTH = 1792, 512
DECK_SEED = 42
CYCLIC = config["masking"]["cyclic25"]

deck_event = np.zeros(INPUT_LENGTH, dtype=bool)
deck_event[DECK_EVENT_START : DECK_EVENT_START + DECK_EVENT_WIDTH] = True

p25_mask = build_patch_aligned_isolated_masks(
    example_signal.astype(np.float64),
    PatchSpec(INPUT_LENGTH, config["model"]["patch_size"], config["model"]["patch_stride"]),
    np.random.default_rng(DECK_SEED),
    mask_ratio=float(config["masking"]["mask_ratio"]),
)["target_time_mask"]

cyclic_spec = PatchSpec(
    INPUT_LENGTH, int(CYCLIC["candidate_size"]), int(CYCLIC["candidate_stride"])
)
deck_cycle = build_balanced_event_mask_cycle(
    deck_event,
    cyclic_spec,
    np.random.default_rng(DECK_SEED),
    event_windows_per_pass=int(CYCLIC["event_windows_per_pass"]),
    background_windows_per_pass=int(CYCLIC["background_windows_per_pass"]),
    require_context_each_side=bool(CYCLIC["require_context_each_side"]),
)
cyclic_mask = np.asarray(deck_cycle["target_time_masks"], dtype=bool)[0]


def geometry(mask):
    spans = mask_spans(mask)
    lengths = [end - start for start, end in spans]
    return {"hidden": int(mask.sum()), "spans": len(spans),
            "longest": int(max(lengths)), "median": int(np.median(lengths))}


published_geometry = {"P25": {"hidden": 1024, "spans": 64, "longest": 16, "median": 16},
                      "CYCLIC25": {"hidden": 1024, "spans": 27, "longest": 560, "median": 16}}
measured_geometry = {"P25": geometry(p25_mask), "CYCLIC25": geometry(cyclic_mask)}
for policy, values in measured_geometry.items():
    print(f"{policy:9s} {values['hidden']} hidden  {values['spans']:3d} runs  "
          f"longest {values['longest']:3d}  median run {values['median']}")
assert measured_geometry == published_geometry, measured_geometry
print("\nreproduces the deck's published mask geometry exactly")
print(f"budget check: both hide {measured_geometry['P25']['hidden']} of {INPUT_LENGTH} "
      f"samples — matched by construction, not by tuning")

# %%
draw_policies(example_signal, mask_spans(p25_mask), mask_spans(cyclic_mask),
              (DECK_EVENT_START, DECK_EVENT_START + DECK_EVENT_WIDTH))

# %% [markdown]
# Same 1,024 points, two different problems. P25's longest hole is 16 samples
# with visible signal on both sides — that is interpolation. CYCLIC25's longest
# hole is 560 samples covering the whole event — that is reconstruction from
# context. Their raw losses are therefore **not comparable**, which is the
# reason the comparison below has to be done by crossing the masks.
#
# ### CYCLIC25 is a schedule, not a mask
#
# One CYCLIC25 pass hides only part of the event. The builder produces a
# *cycle* of passes in which every candidate window intersecting the event is
# hidden at least once, while each individual pass keeps the same 25 % budget
# and stays internally disjoint. Training walks the cycle, so an event is seen
# many times through different holes.
#
# Here is the actual cycle of the matched CYCLIC25 run for its deck sample,
# rebuilt with the run's own seed derivation (`seed + index × 7919`).

# %%
SAMPLE_SEED_STRIDE = 7919
TRAINING_SAMPLE = "syn-2um-3643708b867724bd"

training_index = next(
    i for i, row in enumerate(v5_rows) if row["sample_id"] == TRAINING_SAMPLE
)
training_signal = np.asarray(
    np.load(v5_root / "signals_raw_4096.npy", mmap_mode="r")[training_index],
    dtype=np.float64,
)
training_signal = (training_signal - training_signal.mean()) / training_signal.std()
training_event = np.zeros(INPUT_LENGTH, dtype=bool)
training_event[declared_start[training_index] : declared_end[training_index]] = True

training_cycle = build_balanced_event_mask_cycle(
    training_event,
    cyclic_spec,
    np.random.default_rng(DECK_SEED + training_index * SAMPLE_SEED_STRIDE),
    event_windows_per_pass=int(CYCLIC["event_windows_per_pass"]),
    background_windows_per_pass=int(CYCLIC["background_windows_per_pass"]),
    require_context_each_side=bool(CYCLIC["require_context_each_side"]),
)
training_masks = np.asarray(training_cycle["target_time_masks"], dtype=bool)
training_coverage = np.asarray(training_cycle["cumulative_event_window_coverage"])
print(f"{TRAINING_SAMPLE} · declared support "
      f"[{declared_start[training_index]}, {declared_end[training_index]}) = "
      f"{declared_end[training_index] - declared_start[training_index]} samples")
print(f"  {int(training_cycle['pass_count'])} passes, "
      f"{training_cycle['event_window_indices'].size} candidate event windows, "
      f"{[int(m.sum()) for m in training_masks]} points hidden per pass")
print(f"  cumulative event-window coverage: "
      f"{', '.join(f'{100 * value:.0f} %' for value in training_coverage)}")

# %%
draw_cycle(training_signal, [mask_spans(mask) for mask in training_masks],
           training_coverage,
           (declared_start[training_index], declared_end[training_index]))

# %% [markdown]
# ### Appendix · how a fixed budget packs a variable support
#
# The completeness guarantee above is not free. Every event has a different
# width, so the number of candidate windows changes from event to event, while
# the budget is fixed at 32 event windows per pass. The builder resolves this
# with two devices: it splits the half-overlapping catalogue into **lanes** (a
# window's index modulo `patch_size / stride` = 2), so that windows within a
# lane can never overlap each other, and it pads the last group of a lane by
# repeating windows already scheduled.
#
# The deck states this for the median event — a 906-sample support. That is
# reproduced from the shipped builder here.

# %%
PACKING_WIDTH = int(np.median(declared_end - declared_start))
packing_start = (INPUT_LENGTH - PACKING_WIDTH) // 2
packing_event = np.zeros(INPUT_LENGTH, dtype=bool)
packing_event[packing_start : packing_start + PACKING_WIDTH] = True
packing = build_balanced_event_mask_cycle(
    packing_event,
    cyclic_spec,
    np.random.default_rng(DECK_SEED),
    event_windows_per_pass=int(CYCLIC["event_windows_per_pass"]),
    background_windows_per_pass=int(CYCLIC["background_windows_per_pass"]),
    require_context_each_side=bool(CYCLIC["require_context_each_side"]),
)
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
             (packing_start, packing_start + PACKING_WIDTH))

# %% [markdown]
# ### The comparison
#
# Ten runs: two policies × five seeds, 30 epochs each, on 39,108 synthetic
# training traces with 8,872 synthetic and 444 real validation traces. The
# comparison is read in three steps, and the order matters: first check the
# monitoring is honest, then look at what the models produce, then measure
# which one survives the other's regime.
#
# #### 1 · Is the train/validation comparison meaningful at all?
#
# The training loss is computed under dropout with freshly drawn masks, so it
# is not comparable to a validation number. The matched-monitoring protocol
# fixes this: 2,048 class-proportional events per split, selected once by a
# hash of the sample id, evaluated under `model.eval` with the run's own
# policy. Train and validation then differ only by the split.

# %%
monitor = {}
for policy in ("P25", "CYCLIC25"):
    histories = [
        json.loads((matched_run(policy.lower(), seed) / "history.json").read_text())
        for seed in MATCHED_SEEDS
    ]
    epochs = [entry["epoch"] for entry in histories[0]]
    train_curves = [[e["matched_monitor"]["train_eval"]["model"]["masked_mse"] for e in h]
                    for h in histories]
    validation_curves = [[e["matched_monitor"]["validation"]["model"]["masked_mse"] for e in h]
                         for h in histories]
    final_train = statistics.fmean(curve[-1] for curve in train_curves)
    final_validation = statistics.fmean(curve[-1] for curve in validation_curves)
    monitor[policy] = {
        "epochs": epochs,
        "train_eval": train_curves,
        "validation": validation_curves,
        "final_train_eval": final_train,
        "final_validation": final_validation,
        "gap_percent": 100.0 * (final_validation - final_train) / final_train,
    }
    print(f"{policy:9s} final fixed-monitor masked MSE  train {final_train:.6f}  "
          f"validation {final_validation:.6f}  gap {monitor[policy]['gap_percent']:+.4f} %")

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
# Both policies generalise: the validation monitor sits **+1.49 %** (P25) and
# **+1.22 %** (CYCLIC25) above the train monitor at epoch 30, on identical
# protocols. Neither is memorising, and the loss curves can be believed.
#
# What the plot also shows is that the two levels are two orders of magnitude
# apart — P25 lands near 1.6 × 10⁻³, CYCLIC25 near 1.9 × 10⁻¹. That is not
# evidence that P25 is the better model. It is evidence that P25 was asked an
# easier question, exactly as the geometry predicted.
#
# #### 2 · What the two models produce, on real events
#
# **Limit, named.** Every matched run's `run.json` lists `checkpoints/best.pt`
# and `checkpoints/latest.pt` among its outputs, but the checkpoint directory
# was never synced back from the cluster. The weights are not in this
# workspace, so nothing below is re-inferred: the reconstructions come from the
# `*_reconstruction_examples.npz` each run wrote at the end of training. Any
# claim requiring a forward pass — a new event, a new mask, a probe on the
# frozen encoder — is out of reach until those checkpoints are recovered.
#
# A second limit follows from the same files. Both runs exported their examples
# under the **P25** evaluation policy (`evaluate_reconstruction` defaults to
# it), so the stored masks are 64 isolated 16-sample holes in every case. That
# is precisely the deck's "identical hidden samples" comparison and it is
# reproduced faithfully — but it means the notebook cannot *show* a 560-sample
# CYCLIC25 gap being reconstructed. For that regime only the aggregate numbers
# below exist.

# %%
recon_signal = p25_examples["signal"][example_index]
recon_mask = p25_examples["mask"][example_index].astype(bool)
if not np.array_equal(recon_mask, cyclic_examples["mask"][example_index].astype(bool)):
    raise ValueError("stored masks differ; the comparison would not be matched")
recon_spans = mask_spans(recon_mask)
widest = max(recon_spans, key=lambda span: span[1] - span[0])
centre = (widest[0] + widest[1]) // 2
zoom = (max(0, centre - 320), min(recon_signal.size, centre + 320))

outputs = []
for label, source, accent in (("P25", p25_examples, "#e2483f"),
                              ("CYCLIC25", cyclic_examples, "#00a3c7")):
    prediction = source["model"][example_index]
    error = float(np.mean((prediction[recon_mask] - recon_signal[recon_mask]) ** 2))
    outputs.append((label, prediction, accent, error))
    print(f"{label:9s} masked MSE on {p25_examples['sample_id'][example_index]}: {error:.6f}")
draw_reconstruction(recon_signal, outputs, recon_spans, zoom)

# %% [markdown]
# On P25's own regime, P25 tracks the hidden samples closely and CYCLIC25 is
# visibly coarser — an order of magnitude worse locally. If the comparison
# stopped here it would retain P25. It does not stop here.
#
# #### 3 · The cross-mask evaluation
#
# Each trained model is evaluated on **both** regimes, on the 444 real
# validation events, with `predicting zero` as the reference: a model that has
# learned nothing useful about a regime cannot beat the constant zero. Five
# seeds per cell, means recomputed here from the evaluation run's rows.

# %%
cross = published("bead-ssl-v2-matched-cross-mask-evaluation-r1")
cells = {}
zero_reference = {}
for row in cross["rows"]:
    key = (row["training_mask_policy"], row["evaluation_mask_policy"])
    cells.setdefault(key, []).append(row["real_validation"]["model"]["masked_mse"])
    zero_reference.setdefault(
        row["evaluation_mask_policy"], []
    ).append(row["real_validation"]["zero"]["masked_mse"])
cells = {key: {"mean": statistics.fmean(values), "std": statistics.pstdev(values),
               "seeds": len(values)}
         for key, values in cells.items()}
zero_reference = {key: statistics.fmean(values) for key, values in zero_reference.items()}

print(f"{'trained':>10} {'on P25 masks':>16} {'on CYCLIC25 masks':>19}   seeds")
for trained in ("P25", "CYCLIC25"):
    print(f"{trained:>10} {cells[(trained, 'P25')]['mean']:16.4f} "
          f"{cells[(trained, 'CYCLIC25')]['mean']:19.4f}   "
          f"{cells[(trained, 'P25')]['seeds']}")
print(f"{'zero':>10} {zero_reference['P25']:16.4f} {zero_reference['CYCLIC25']:19.4f}   —")

published_cells = {("P25", "P25"): 0.0015, ("P25", "CYCLIC25"): 1.1024,
                   ("CYCLIC25", "P25"): 0.0063, ("CYCLIC25", "CYCLIC25"): 0.0361}
for key, want in published_cells.items():
    assert round(cells[key]["mean"], 4) == want, (key, cells[key]["mean"])
assert round(zero_reference["P25"], 4) == 1.2765
assert round(zero_reference["CYCLIC25"], 4) == 1.1209
print("\nreproduces bead-ssl-v2-matched-cross-mask-evaluation-r1 exactly")

# %%
draw_cross_mask(cells, zero_reference)

# %% [markdown]
# The verdict is in the second column. **P25 on CYCLIC25 masks scores 1.102
# against 1.121 for predicting zero** — a 1.7 % improvement over outputting
# nothing at all. A model that hid only isolated 16-sample holes for 30 epochs
# has learned nothing that transfers to a missing event. CYCLIC25 on P25 masks
# scores 0.0063 against a 1.277 zero baseline: it pays about four times P25's
# local error but stays a functioning model on a regime it never trained on.
#
# So the retained policy is CYCLIC25, and the honest statement of the result is
# asymmetric: CYCLIC25 is retained for robustness across missing positions,
# while P25 keeps a genuinely lower local error on its own easier task. Both
# halves belong in the claim.
#
# **What is not claimed.** All of this is *reconstruction* behaviour on
# development data. Nothing here says which encoder is the better
# representation — that requires frozen probes or fine-tuning against a
# from-scratch baseline, which needs the checkpoints, and no sealed test split
# was touched anywhere in this chain.

# %% [markdown]
# ---
# ## Exploratory · the event CYCLIC25 was aimed at is half the real one
#
# *This sub-section is exploratory and it produces a measurement no published
# run owns. It is not a correction of the published comparison; it is a
# measured statement about what that comparison actually trained.*
#
# The masking code never sees a waveform's timing. It converts the generator's
# `tau_ms` into samples using a number from the config:
#
# ```python
# tau_samples = float(row["tau_ms"]) * 1.0e-3 * self.sampling_frequency_hz
# ```
#
# `bead_ssl_p25_v1.yaml` declares `sampling_frequency_hz: 1000000`, and
# `bead_ssl_z8_v5_v2.yaml` extends it without overriding that key, so the
# merged config the runs used carries 1 MHz. The v5 generator writes
# `sampling_frequency_hz: 2_000_000.0` into its own provenance.
#
# Two configuration files disagreeing is an argument, not evidence. The
# waveforms settle it: for a Gaussian envelope `exp(−½(t/τ)²)`, the full width
# at half maximum is `1.1774 · τ · (e^{−a} + e^{+a})` **seconds**, so measuring
# it in samples on the stored traces measures the sampling rate directly.

# %%
TRUE_HZ = SAMPLING_HZ
v5_signals = np.load(v5_root / "signals_raw_4096.npy", mmap_mode="r", allow_pickle=False)
achieved_snr = np.asarray([float(row["achieved_snr_db"]) for row in v5_rows])
probe = np.sort(np.argsort(-achieved_snr)[:300])
probe_envelope = uniform_filter1d(
    np.abs(hilbert(np.asarray(v5_signals[probe], dtype=np.float64), axis=1)),
    size=41, axis=1,
)
measured_fwhm = np.asarray([
    np.flatnonzero(row >= 0.5 * row.max())[[0, -1]] @ [-1, 1] + 1 for row in probe_envelope
], dtype=float)
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
# The consequence is not cosmetic, because this number is not metadata: it is
# the only thing converting the event's physical duration into array indices.
# At half the true rate the support is half as wide, centred on the same point,
# and therefore a **strict subset** of the real event.

# %%
true_start, true_end = event_bounds(TRUE_HZ)
true_fraction = (true_end - true_start) / INPUT_LENGTH
assert bool(np.all((declared_start >= true_start) & (declared_end <= true_end)))
captured = (declared_end - declared_start) / (true_end - true_start)
print(f"over all {len(v5_rows):,} v5 events")
print(f"  median support   declared {100 * np.median(declared_fraction):5.1f} %   "
      f"true {100 * np.median(true_fraction):5.1f} %")
print(f"  p95 support      declared {100 * np.percentile(declared_fraction, 95):5.1f} %   "
      f"true {100 * np.percentile(true_fraction, 95):5.1f} %")
print(f"  under half crop  declared {100 * np.mean(declared_fraction < 0.5):5.1f} %   "
      f"true {100 * np.mean(true_fraction < 0.5):5.1f} %")
print(f"  fraction of the true support inside the declared one: "
      f"median {np.median(captured):.3f}")
assert round(100 * float(np.median(true_fraction)), 1) == 44.2

# %%
draw_support_rates(declared_fraction, true_fraction, v5_class, CLASS_ORDER, CLASS_COLOUR)

# %% [markdown]
# The deck's SSL bridge opens the whole masking chapter on "the annotated
# support is 22 % of the crop at the median, and 99.1 % of events sit under
# half the crop". Measured against the rate the data was generated at, the
# median event occupies **44.2 %** of the crop and **36.4 %** of events occupy
# more than half of it. The premise that motivates aiming is roughly half as
# strong as stated.
#
# ### What a trace actually got
#
# The clearest way to see it is on one event: the windows the cycle called
# *event* and the windows it called *background*, drawn over the support the
# data implies.

# %%
case = int(np.argsort(np.abs((declared_end - declared_start) - PACKING_WIDTH))[0])
case_signal = np.asarray(np.load(v5_root / "signals_raw_4096.npy",
                                 mmap_mode="r")[case], dtype=np.float64)
case_signal = (case_signal - case_signal.mean()) / case_signal.std()


def build_cycle(start, end, index):
    """One CYCLIC25 cycle, with the run's seed derivation, for a given support."""
    minimum = int(CYCLIC["event_windows_per_pass"]) * int(CYCLIC["candidate_size"])
    missing = max(0, minimum - (end - start))
    start = max(0, start - missing // 2)
    end = min(INPUT_LENGTH, end + missing - missing // 2)
    event = np.zeros(INPUT_LENGTH, dtype=bool)
    event[start:end] = True
    return build_balanced_event_mask_cycle(
        event, cyclic_spec,
        np.random.default_rng(DECK_SEED + index * SAMPLE_SEED_STRIDE),
        event_windows_per_pass=int(CYCLIC["event_windows_per_pass"]),
        background_windows_per_pass=int(CYCLIC["background_windows_per_pass"]),
        require_context_each_side=bool(CYCLIC["require_context_each_side"]),
    )


case_actual = build_cycle(declared_start[case], declared_end[case], case)
case_corrected = build_cycle(true_start[case], true_end[case], case)
case_true = np.zeros(INPUT_LENGTH, dtype=bool)
case_true[true_start[case] : true_end[case]] = True

for label, cycle in (("as trained", case_actual), ("corrected", case_corrected)):
    background = np.asarray(cycle["background_target_time_masks"], dtype=bool)
    intruding = float((background & case_true).sum()) / float(background.sum())
    events = np.asarray(cycle["event_target_time_masks"], dtype=bool)
    print(f"{label:11s} {int(cycle['pass_count'])} passes · "
          f"{100 * intruding:5.1f} % of the background budget lies inside the true event · "
          f"event group ever hides {100 * (events.any(0) & case_true).sum() / case_true.sum():5.1f} %"
          " of the true support")

draw_defect_case(
    case_signal,
    (declared_start[case], declared_end[case]),
    (true_start[case], true_end[case]),
    (mask_spans(np.asarray(case_actual["event_target_time_masks"], bool)[0]),
     mask_spans(np.asarray(case_actual["background_target_time_masks"], bool)[0])),
    (mask_spans(np.asarray(case_corrected["event_target_time_masks"], bool)[0]),
     mask_spans(np.asarray(case_corrected["background_target_time_masks"], bool)[0])),
)

# %% [markdown]
# The amber windows are the ones the cycle balanced *against* the event. Some
# of them sit on the event's own shoulders.
#
# That is the mechanism: `build_balanced_event_mask_cycle` takes its background
# candidates from `~intersects_event`, and `intersects_event` is derived from
# the halved support. Everything outside the inner half of an event was
# eligible to be sampled as background, and the completeness guarantee — "every
# event sample is hidden at least once" — was enforced on the inner half only.
#
# Run over a seeded sample of the corpus with the runs' own seed derivation:

# %%
CYCLE_SAMPLE = 1500
sample_generator = np.random.default_rng(20260815)
cycle_indices = np.sort(
    sample_generator.choice(len(v5_rows), size=CYCLE_SAMPLE, replace=False)
)

background_inside = []
event_group_coverage = []
declared_passes = []
corrected_passes = []
corrected_failures = 0
started = time.time()
for index in cycle_indices:
    index = int(index)
    true_support = np.zeros(INPUT_LENGTH, dtype=bool)
    true_support[true_start[index] : true_end[index]] = True
    cycle = build_cycle(declared_start[index], declared_end[index], index)
    background = np.asarray(cycle["background_target_time_masks"], dtype=bool)
    events = np.asarray(cycle["event_target_time_masks"], dtype=bool)
    background_inside.append(float((background & true_support).sum()) / float(background.sum()))
    event_group_coverage.append(
        float((events.any(axis=0) & true_support).sum()) / float(true_support.sum())
    )
    declared_passes.append(int(cycle["pass_count"]))
    try:
        corrected_passes.append(
            int(build_cycle(true_start[index], true_end[index], index)["pass_count"])
        )
    except (ValueError, RuntimeError):
        corrected_failures += 1

background_inside = np.asarray(background_inside)
event_group_coverage = np.asarray(event_group_coverage)
print(f"{CYCLE_SAMPLE} events, cycles rebuilt from p3_ssl.masking "
      f"({time.time() - started:.0f} s)")
print(f"  background budget landing inside the true event: "
      f"median {100 * np.median(background_inside):.1f} %, "
      f"mean {100 * background_inside.mean():.1f} %")
print(f"  true support ever hidden by the event group over a whole cycle: "
      f"median {100 * np.median(event_group_coverage):.1f} %, "
      f"mean {100 * event_group_coverage.mean():.1f} %")
print(f"  passes per cycle: median {int(np.median(declared_passes))} as trained, "
      f"{int(np.median(corrected_passes))} corrected")
print(f"  events for which the corrected support admits no cycle at all: "
      f"{corrected_failures} of {CYCLE_SAMPLE} "
      f"({100 * corrected_failures / CYCLE_SAMPLE:.1f} %)")

# %%
draw_defect_summary(background_inside, event_group_coverage,
                    (declared_passes, corrected_passes))

# %% [markdown]
# ### What this puts in doubt, and what it does not
#
# **What is established.** CYCLIC25 as trained aimed at the inner half of each
# simulated event. A median **28.6 %** of what it called *background* was in
# fact event (mean 30.7 %); its completeness guarantee — the thing that
# justifies the cycle — covered a median **51.4 %** of each true support; and a
# corrected cycle would be twice as long (median 4 passes to 8), with **0.4 %**
# of events admitting no cycle at all under the current 32-window budget. P25
# is untouched: it passes `event_mask=None` and `event_biased_probability=0.0`,
# so it never reads the support.
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
# against it, not for it. It still beat P25 on that regime by a factor of
# thirty. The qualitative verdict — P25 does not generalise to missing events,
# CYCLIC25 does — therefore survives the defect, and if anything is understated.
#
# **What genuinely stays unknown.** How much better a correctly-aimed CYCLIC25
# would be. Its matched masked MSE of 0.193 is measured on simulation
# validation, whose masks carry the same halved support, so that number is not
# a clean estimate of anything. Whether the encoder — the actual deliverable —
# is affected cannot be assessed at all from here: the checkpoints are not in
# this workspace, and even with them, a representation claim needs probes that
# were never run.
#
# **What would settle it.** Ten runs: two policies × five seeds, 30 epochs, on
# pfcalcul, with `sampling_frequency_hz: 2000000` in
# `bead_ssl_z8_v5_v2.yaml` — the fix is one line, since `_deep_merge` lets the
# child override the key. The 32-window budget needs raising in step, or the
# small fraction of wide events will fail to build a cycle. Re-running the same
# matched-monitor and cross-mask evaluation then makes the corrected and
# published comparisons directly commensurable. Until that exists, the deck's
# masking conclusion should be reported as it is defended here: robust in
# direction, unquantified in size.
#
# **A provenance note that belongs with this.** The deck's masking figures are
# resolved from `.cache/visual-evidence/ssl-v18-masking-figures-r4`, a cache
# path outside `artifacts/` with no `run.json`. Its `figure_metrics.json` holds
# the published geometry — and its `sources` block names
# `configs/bead_ssl_p25_v1.yaml`, the file carrying the wrong rate. The numbers
# happen to be reproducible, as this section shows, but nothing manifested
# guaranteed that. This notebook closes that gap for the geometry and the
# packing, and the run emitted below closes it for the defect.

# %%
try:
    emitted = notebook_evidence.emit_run(
        workspace,
        section="masked-learning-mask-rate",
        metrics={
            "schema_version": 1,
            "analysis": "cyclic25-event-support-sampling-rate-defect",
            "declared_sampling_frequency_hz": DECLARED_HZ,
            "generator_sampling_frequency_hz": TRUE_HZ,
            "rate_determined_from_waveforms": {
                "probe_events": int(len(probe)),
                "minimum_achieved_snr_db": float(achieved_snr[probe].min()),
                "measured_over_predicted_fwhm_median": {
                    "1000000": rate_ratio[DECLARED_HZ],
                    "2000000": rate_ratio[TRUE_HZ],
                },
            },
            "event_support": {
                "events": len(v5_rows),
                "declared": {
                    "median_fraction": float(np.median(declared_fraction)),
                    "p95_fraction": float(np.percentile(declared_fraction, 95)),
                    "fraction_under_half_crop": float(np.mean(declared_fraction < 0.5)),
                    "median_width_samples": float(np.median(declared_end - declared_start)),
                },
                "true": {
                    "median_fraction": float(np.median(true_fraction)),
                    "p95_fraction": float(np.percentile(true_fraction, 95)),
                    "fraction_under_half_crop": float(np.mean(true_fraction < 0.5)),
                    "median_width_samples": float(np.median(true_end - true_start)),
                },
                "declared_is_subset_of_true": True,
                "median_true_support_captured": float(np.median(captured)),
                "per_class_median_fraction": {
                    name: {
                        "declared": float(np.median(declared_fraction[v5_class == name])),
                        "true": float(np.median(true_fraction[v5_class == name])),
                    }
                    for name in CLASS_ORDER
                },
            },
            "cycle_consequence": {
                "sampled_events": CYCLE_SAMPLE,
                "background_inside_true_event_median": float(np.median(background_inside)),
                "background_inside_true_event_mean": float(background_inside.mean()),
                "true_support_covered_by_event_group_median": float(
                    np.median(event_group_coverage)
                ),
                "true_support_covered_by_event_group_mean": float(
                    event_group_coverage.mean()
                ),
                "median_passes_declared": int(np.median(declared_passes)),
                "median_passes_corrected": int(np.median(corrected_passes)),
                "corrected_cycle_failures": corrected_failures,
            },
            "unaffected": {
                "p25_policy": "event_mask=None, event_biased_probability=0.0",
                "hidden_points_per_pass": 1024,
                "real_validation_masks": (
                    "Z8RealValidationDataset derives the support from annotated "
                    "start_sample/end_sample and never reads sampling_frequency_hz"
                ),
            },
        },
        provenance={
            "datasets": dataset_provenance(),
            "inputs": {
                "events_csv_sha256": notebook_evidence.sha256_file(v5_root / "events.csv"),
                "config_sha256": notebook_evidence.sha256_file(SSL_CONFIG),
                "base_config_sha256": notebook_evidence.sha256_file(
                    SSL_ROOT / "configs/bead_ssl_p25_v1.yaml"
                ),
            },
            "parameters": {
                "event_support_formula": (
                    "centre = t0_fraction * (N - 1); "
                    "[centre - 3*tau*exp(-a), centre + 3*tau*exp(+a)], tau in samples"
                ),
                "cyclic25": {key: CYCLIC[key] for key in sorted(CYCLIC)},
                "seed_derivation": "42 + sample_index * 7919, as in build_cyclic25_masks_for_sample",
                "cycle_sample_seed": 20260815,
                "cycle_sample_size": CYCLE_SAMPLE,
                "fwhm_probe": "300 highest achieved-SNR v5 events, Hilbert envelope, 41-sample box smoothing",
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
        },
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
# ## Can a regenerated event find its own parent?
#
# Coverage asked whether the synthetic cloud *reaches* the real one. This asks
# something far stricter, event by event: regenerate one event from its own
# fitted parameters, and see whether the original comes back at the top of the
# neighbour list. A generator can cover a cloud by being broad; it can only pass
# this test by being accurate.
#
# Two numbers, both read against a floor and a ceiling:
#
# - **Recall@5** — how often the exact parent lands in the top five;
# - **q50** — the median *relative* rank of the parent, where 0 % is first and
#   ≈50 % is what chance gives.
#
# Each event is regenerated in eight views, because phase is not identifiable,
# and the views are averaged after normalisation. Galleries are **split-local**:
# a train query is only ever compared to train parents.

# %%
import torch  # noqa: E402,F401  (imported for the frozen classifier)
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from internship_workspace import chain_figures  # noqa: E402
from internship_workspace.equation_latent_audit import (  # noqa: E402
    average_normalized_views,
    extract_penultimate_embeddings,
    load_candidate,
    load_frozen_classifier,
)
from internship_workspace.z8_domain_pca import morphology_features  # noqa: E402
from internship_workspace.z8_parent_retrieval import (  # noqa: E402
    macro_rate,
    split_local_retrieval,
)

ROUNDTRIP_KEY = "particles2snr-z8-equation-roundtrip@v3"
CHECKPOINT = (
    workspace.root
    / "artifacts/unsupervised-learning-flow-cytometry/pretrained_backbones-10dB"
    / "particles2snr_f_3class_native_params_moment_patchtst_conv1dgap"
    / "conv1dgap_same_input_3class/best_model.pt"
)
ORIGIN = "dual_clean_strict"
SIGMA_VARIANT = "detector_empirical_cross_source_phase_marginal"
VIEWS_PER_EVENT = 8

roundtrip_signals, roundtrip_rows = load_candidate(dataset_root(ROUNDTRIP_KEY))
gallery_index = np.asarray([
    index
    for index, row in enumerate(roundtrip_rows)
    if row["record_kind"] == "real_gallery" and row["annotation_origin"] == ORIGIN
])
query_index = np.asarray([
    index
    for index, row in enumerate(roundtrip_rows)
    if row["record_kind"] == "synthetic_query"
    and row["annotation_origin"] == ORIGIN
    and row["variant"] == SIGMA_VARIANT
])
assert query_index.size == gallery_index.size * VIEWS_PER_EVENT

gallery_meta = {
    field: np.asarray([roundtrip_rows[index][field] for index in gallery_index], dtype=str)
    for field in ("source_event_id", "source_group", "class_name", "split")
}
query_event_ids = np.asarray(
    [roundtrip_rows[index]["source_event_id"] for index in query_index], dtype=str
)
print(f"gallery {gallery_index.size} real events · queries {query_index.size} views "
      f"· sources {len(set(gallery_meta['source_group']))}")


# %%
def retrieve(query_embeddings, gallery_embeddings):
    """Split-local retrieval of each event's own parent, in any space.

    One function for every space is the point: parity between experiments
    becomes structural rather than a matter of discipline.
    """
    averaged, averaged_ids = average_normalized_views(query_embeddings, query_event_ids)
    return split_local_retrieval(
        query_embeddings=averaged,
        query_ids=averaged_ids,
        gallery_embeddings=gallery_embeddings,
        gallery_ids=gallery_meta["source_event_id"],
        gallery_groups=gallery_meta["source_group"],
        gallery_classes=gallery_meta["class_name"],
        gallery_splits=gallery_meta["split"],
        top_k=50,
    )


def score(retrieval):
    recall5 = (retrieval.result.rank <= 5).astype(np.float64)
    return {
        "events": int(len(retrieval.result.rank)),
        "recall_at_5_macro": float(macro_rate(recall5, retrieval.classes)),
        "q50_relative_rank": float(np.median(retrieval.result.rank_percentile)),
    }


# %% [markdown]
# ### The published space — the frozen Conv1D-GAP latent
#
# The deck's numbers come from the 512-D penultimate activation of a frozen
# Conv1D-GAP-**L** classifier, compared by cosine distance.

# %%
model = load_frozen_classifier(CHECKPOINT)
latent_gallery = extract_penultimate_embeddings(
    model, np.asarray(roundtrip_signals[gallery_index])
)
latent_queries = extract_penultimate_embeddings(
    model, np.asarray(roundtrip_signals[query_index])
)
latent_score = score(retrieve(latent_queries, latent_gallery))
print(json.dumps(latent_score, indent=2))

# %%
reference_row = next(
    row
    for row in csv.DictReader(
        (run_dir("particle-z8-v2-exact-parent-retrieval-r1")
         / "condition_metrics.csv").open()
    )
    if row["condition_id"] == f"{ORIGIN}_tau_sigma"
)
retrieval_deviation = max(
    abs(latent_score["recall_at_5_macro"] - float(reference_row["recall_at_5_macro"])),
    abs(latent_score["q50_relative_rank"] - float(reference_row["q50_relative_rank"])),
)
assert retrieval_deviation < 1e-12, f"reproduction drifted by {retrieval_deviation:.3e}"
print(f"reproduces particle-z8-v2-exact-parent-retrieval-r1 "
      f"({ORIGIN}_tau_sigma) exactly (max deviation {retrieval_deviation:.1e})")

# %% [markdown]
# ### Where 11 % sits, between a floor and a ceiling
#
# Eleven percent means nothing on its own. The published run brackets it with
# both ends of the scale, and the brackets are what make it readable.

# %%
board = published("ssl-v3-v16-retrieval-and-ranges-r5", "board_values.json")
ranking = published("p0-conv1dgapl-z8-ranking-v1")
print(json.dumps(board, indent=2)[:1200])

# %% [markdown]
# - **Ceiling — a benign real query: 100 %, q50 0 %.** Feed a real event back as
#   its own query and it always finds itself first. The protocol works.
# - **This generator (P2SNR, τ=σ): 11.0 %, q50 9.5 %.** Far above chance, far
#   below the ceiling.
# - **An earlier detector (P0): 3.8 %, q50 17.4 %.** Same protocol, same latent
#   family, a weaker parameter fit — and the number moves the way it should. That
#   is what makes this a measurement rather than a coincidence.
# - **Floor — shuffled parents: ~2 %, q50 ≈ 50 %.** Chance.
#
# The honest reading of 11 %: a regenerated event is recognisably related to its
# parent, and is not the parent. The generator reproduces the *kind* of signal,
# not the individual. That is exactly the limitation the deck states, and it is
# why coverage and retrieval have to be reported together.

# %% [markdown]
# ### The same retrieval, in the morphology space
#
# The deck uses two different spaces for one idea of "neighbour": the learned
# latent here, the morphology descriptor for coverage. Standardising on one is a
# decision; *what it costs* is a number, and this is it.
#
# Everything is held fixed — same events, same eight views, same averaging, same
# split-local galleries, same `retrieve` function. Only the space changes. The
# descriptor is computed on the very signals the encoder sees (a 4 096 crop,
# decimated by 8 to 512 points, so an effective 250 kHz), which isolates the
# space from the preprocessing.

# %%
DECIMATED_HZ = SAMPLING_HZ / 8


def morphology_of(indices):
    return np.concatenate([
        morphology_features(
            np.asarray(roundtrip_signals[indices[start : start + 512]]),
            sampling_frequency_hz=DECIMATED_HZ,
        )
        for start in range(0, len(indices), 512)
    ])


morphology_gallery = morphology_of(gallery_index)
morphology_queries = morphology_of(query_index)
morphology_score = score(retrieve(morphology_queries, morphology_gallery))
print(f"descriptor {morphology_gallery.shape[1]}-D against a "
      f"{latent_gallery.shape[1]}-D latent")
print(json.dumps(morphology_score, indent=2))

# %% [markdown]
# A third variant makes the comparison directly transposable to the coverage
# section, which does not use the raw descriptor but its 16 principal components
# with the basis fitted on **synthetic events only**. Here the queries are the
# synthetic side, so fitting on them follows the same convention and keeps the
# gallery out of the fit.

# %%
morphology_scaler = StandardScaler().fit(morphology_queries)
morphology_pca = PCA(n_components=16, svd_solver="full").fit(
    morphology_scaler.transform(morphology_queries)
)


def reduce_morphology(values):
    return morphology_pca.transform(morphology_scaler.transform(values))


morphology_pca_score = score(
    retrieve(reduce_morphology(morphology_queries), reduce_morphology(morphology_gallery))
)
print(json.dumps(morphology_pca_score, indent=2))

# %%
CHANCE_Q50 = 0.5
SPACES = [
    ("Conv1D-GAP latent\n512-D", latent_score, "#64748b"),
    ("morphology\nfull descriptor", morphology_score, CLASS_COLOUR["4um"]),
    ("morphology\nPCA(16), as in coverage", morphology_pca_score, CLASS_COLOUR["2um"]),
]
chain_figures.draw_space_comparison(
    SPACES,
    chance_q50=CHANCE_Q50,
    suptitle=f"Same {latent_score['events']} events, same eight views, same "
             "split-local galleries — only the space changes",
)
for name, values, _ in SPACES:
    print(f"{name.replace(chr(10), ' '):<38} "
          f"R@5 {100 * values['recall_at_5_macro']:5.1f} %   "
          f"q50 {100 * values['q50_relative_rank']:5.1f} %")

# %% [markdown]
# **The physics-grounded space wins, and not narrowly**: Recall@5 more than
# doubles, from 11.0 % to 23.3 %, and the median parent rank halves, from 9.5 %
# to 5.2 %.
#
# The explanation is not mysterious, and it is the sharpest objection answered in
# advance. The Conv1D-GAP latent is the penultimate layer of a classifier trained
# to separate **three classes**. Discarding within-class variation is precisely
# what that training rewards — and within-class variation is exactly the
# information needed to tell one 4 µm event from another 4 µm event. The learned
# space is optimal for the task it was trained on and structurally wrong for this
# one.
#
# Two caveats, stated plainly. The descriptor here is computed on the encoder's
# own input — a 4 096 crop decimated to 512 points — not on the 1 024-sample
# window the coverage section uses, so this settles *which space*, not *which
# window*; Part II measures the window separately, and finds it does not behave
# as anyone expected. And the PCA is fitted on the synthetic queries, which is
# the coverage section's convention and never uses parent identity.
#
# A blocker to clear before rerunning the published tool on new data: it carries
# a hard-coded regression guard that aborts unless Recall@5 equals its historical
# z8 value (`analyze_z8_v2_exact_parent_retrieval.py`).

# %%
try:
    emitted = notebook_evidence.emit_run(
        workspace,
        section="retrieval-space-comparison",
        metrics={
            "schema_version": 1,
            "analysis": "parent-retrieval-across-neighbour-spaces",
            "population": {
                "origin": ORIGIN,
                "condition": "tau_sigma",
                "events": latent_score["events"],
                "views_per_event": VIEWS_PER_EVENT,
                "source_groups": len(set(gallery_meta["source_group"].tolist())),
            },
            "spaces": {
                "conv1dgap_latent": {
                    "dimensions": int(latent_gallery.shape[1]), **latent_score
                },
                "morphology_full": {
                    "dimensions": int(morphology_gallery.shape[1]), **morphology_score
                },
                "morphology_pca16": {"dimensions": 16, **morphology_pca_score},
            },
            "chance_q50_relative_rank": CHANCE_Q50,
            "reproduces": {
                "run_id": "particle-z8-v2-exact-parent-retrieval-r1",
                "condition_id": f"{ORIGIN}_tau_sigma",
                "max_deviation": retrieval_deviation,
            },
        },
        provenance={
            "datasets": dataset_provenance(),
            "inputs": {
                "checkpoint_sha256": notebook_evidence.sha256_file(CHECKPOINT),
            },
            "parameters": {
                "origin": ORIGIN,
                "variant": SIGMA_VARIANT,
                "views_per_event": VIEWS_PER_EVENT,
                "top_k": 50,
                "gallery": "split-local; train queries see train parents only",
                "morphology_input": (
                    "the encoder's own 512-point input, mean-decimated by 8 from a "
                    "4096 crop, so an effective 250 kHz"
                ),
                "morphology_pca_fit": "synthetic queries only, 16 components",
            },
            "metric_definitions": {
                "recall_at_5_macro": (
                    "unweighted class mean of the fraction of events whose exact "
                    "parent ranks within the first five neighbours"
                ),
                "q50_relative_rank": (
                    "median relative rank of the exact parent, where 0 is first and "
                    "0.5 is chance"
                ),
            },
        },
        claim_boundary=(
            "Compares neighbour spaces for exact-parent retrieval on one detector "
            "condition, holding events, views, averaging and galleries fixed. It "
            "measures which space ranks the parent better on the encoder's input "
            "window; it does not compare window sizes, does not evaluate coverage, "
            "and authorizes no dataset promotion."
        ),
    )
    print(f"emitted {emitted.name}")
except WorkspaceError as error:
    print(f"no evidence emitted ({error})")


# %% [markdown]
# # Part II — The alignments
#
# Exploratory sections. The method carries choices that were made once and
# never compared: a window, a quantile, a basis. Each is a knob that can
# flatter or challenge the simulator, and none of them is visible in a
# result quoted without it. These sections put each knob on an axis and
# measure what it moves.
#
# They recommend; they do not adopt. What survives review becomes an
# alignment in `docs/experiments/2026-08-15/mad-redo-execution-plan.md`.


# %% [markdown]
# ## Alignment · the descriptor window
#
# *Exploratory section. It tests a change before it is adopted, and its
# conclusion is a recommendation, not a shipped decision.*
#
# The morphology descriptor reads a fixed **1 024-sample window** centred on the
# event. Every distance in this notebook — coverage, twins, retrieval — is a
# distance between two such windows. So the window is not a detail of the
# method; it is the method's field of view.
#
# The audit earlier in this notebook showed that field of view is too narrow:
# **96.2 % of the real events are wider than the window that describes them.**
# The natural repair is 4 096 samples, which is already the raw window of the
# SSL model and of the classifier, and which covers 100 % of the new MAD corpus
# (its widest event is exactly 4 000 samples).
#
# Two things make that repair less trivial than it sounds, and one objection
# makes it less obviously desirable. This section deals with all three.

# %% [markdown]
# ### The descriptor is not window-invariant
#
# `morphology_features` keeps the FFT bins that fall inside the 7–80 kHz band.
# The number of such bins is set by the frequency resolution, which is set by the
# window length. So changing the window silently changes the descriptor's
# dimension — and with it what PCA(16) is a projection *of*.
#
# The envelope half has the same disease in a quieter form: the smoothing
# `gaussian_filter1d(sigma=8.0)` is applied in **samples**, before the envelope
# is averaged into 64 bins. At 1 024 samples a bin is 16 samples wide, so σ is
# half a bin; at 4 096 a bin is 64 samples wide and the same σ is an eighth of a
# bin. The same line of code means something different at each window.

# %%
from internship_workspace.z8_coverage import (  # noqa: E402
    load_real_core,
    read_rows,
    reflect_crop,
    shared_class_sample,
    support_coverage,
    validate_pair,
)
from internship_workspace.z8_domain_pca import domain_metrics, morphology_features  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402
from scipy.signal import hilbert  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

WINDOWS = (1024, 2048, 4096)
BAND_HZ = (7_000.0, 80_000.0)
REFERENCE_WINDOW = 1024
ENVELOPE_BINS = 64

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
# ### A window-invariant descriptor, prototyped here
#
# The repair is to make the descriptor a function of the **physics** — a band and
# a number of bands — rather than of the array length. The fine spectrum is
# averaged onto a **fixed grid of 37 bands** whose edges are exactly the bins the
# 1 024-sample window produces, and σ is expressed in envelope bins.
#
# The gate for the prototype is that it must be the *identity* at 1 024: if it
# does not reproduce the shipped descriptor there, it is a different method, not
# an aligned one, and nothing downstream would be comparable to what is
# published. This is prototyped in the notebook on purpose — it is a candidate
# change to a shipped method, and it stays here until it is adopted.

# %%
_reference_frequencies = np.fft.rfftfreq(REFERENCE_WINDOW, d=1.0 / SAMPLING_HZ)
_reference_band = (
    (_reference_frequencies >= BAND_HZ[0]) & (_reference_frequencies <= BAND_HZ[1])
)
BAND_CENTRES = _reference_frequencies[_reference_band]
_half = 0.5 * (_reference_frequencies[1] - _reference_frequencies[0])
BAND_EDGES = np.concatenate([BAND_CENTRES - _half, [BAND_CENTRES[-1] + _half]])


def invariant_morphology(signals, *, sampling_frequency_hz=SAMPLING_HZ):
    """Morphology descriptor whose dimension does not depend on the window.

    Same construction as the shipped `morphology_features`, with two changes:
    the spectrum is averaged onto the fixed band grid above, and the envelope
    smoothing is expressed in envelope bins rather than samples.
    """
    values = np.asarray(signals, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.shape[1] % ENVELOPE_BINS:
        raise ValueError("window must be divisible by the envelope bin count")
    centred = values - values.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(np.square(centred), axis=1, keepdims=True))
    normalized = centred / rms

    envelope = np.abs(hilbert(normalized, axis=1))
    envelope = np.maximum(
        envelope - np.percentile(envelope, 20.0, axis=1, keepdims=True), 0.0
    )
    bin_width = values.shape[1] // ENVELOPE_BINS
    envelope = gaussian_filter1d(envelope, sigma=0.5 * bin_width, axis=1, mode="nearest")
    envelope = envelope.reshape(len(values), ENVELOPE_BINS, bin_width).mean(axis=2)
    envelope /= np.maximum(np.linalg.norm(envelope, axis=1, keepdims=True), 1.0e-12)

    windowed = normalized * np.hanning(values.shape[1])[None, :]
    frequencies = np.fft.rfftfreq(values.shape[1], d=1.0 / sampling_frequency_hz)
    magnitude = np.abs(np.fft.rfft(windowed, axis=1))
    assignment = np.digitize(frequencies, BAND_EDGES) - 1
    banded = np.empty((len(values), BAND_CENTRES.size), dtype=np.float64)
    for band in range(BAND_CENTRES.size):
        selected = assignment == band
        if not selected.any():
            raise ValueError(f"empty band {band}: window too short for this grid")
        banded[:, band] = magnitude[:, selected].mean(axis=1)
    peak = np.maximum(banded.max(axis=1, keepdims=True), 1.0e-12)
    spectrum = np.log1p(1000.0 * np.maximum(banded / peak, 1.0e-6))
    spectrum /= np.maximum(np.linalg.norm(spectrum, axis=1, keepdims=True), 1.0e-12)
    return np.concatenate((envelope, spectrum), axis=1).astype(np.float32)


# %% [markdown]
# #### Non-regression gate at 1 024

# %%
real_key = "particles2snr-fbase-dual-clean-z8-events-3class-plus-unclear-development@v2"
signal_key = "particles2snr-f-dual-clean-c1-yolo-4class@v2"
real_root = dataset_root(real_key)
signal_root = dataset_root(signal_key)

all_rows = read_rows(real_root / "events.csv")
if any(row["split"] == "test" for row in all_rows):
    raise PermissionError("sealed test rows are forbidden")
real_rows = [
    row for row in all_rows
    if row["class_name"] in CLASS_ORDER and row["split"] in {"train", "val"}
]
real_labels = np.asarray([row["class_name"] for row in real_rows])

_probe = load_real_core(real_rows[:256], signal_root)
_shipped = morphology_features(_probe)
_invariant = invariant_morphology(_probe)
identity_deviation = float(np.abs(_shipped - _invariant).max())
assert identity_deviation == 0.0, f"prototype is not the identity: {identity_deviation:.3e}"
print(f"at {REFERENCE_WINDOW} samples the prototype reproduces the shipped "
      f"descriptor exactly (max deviation {identity_deviation:.1e}, "
      f"{_shipped.shape[1]}-D)")

# %%
draw_descriptor_widths(shipped_widths)

# %% [markdown]
# ### Sweeping the window
#
# Now the question that matters: does a wider field of view describe reality
# better? The protocol of the coverage chain is held fixed — one PCA(16) basis
# fitted on all three generator conditions pooled, one balanced synthetic draw
# shared across conditions (seed 20260809), a per-condition radius at the 80th
# percentile of synthetic self-nearest-neighbour distance, euclidean in 16
# dimensions. Only the window changes.
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
CONDITIONS = [
    ("white_noise_4d",
     "particles2snr-fbase-z8-cholesky-physicalcorr-effective-snr-synthetic-events@v3"),
    ("real_noise_4d",
     "particles2snr-fbase-z8-cholesky-physicalcorr-effective-snr-synthetic-events@v4"),
    ("asymmetry_5d",
     "particles2snr-fbase-z8-cholesky-physicalcorr-effective-snr-synthetic-events@v5"),
]
SEED = 20260809
QUANTILE = 0.80

condition_roots = {}
reference_rows = None
for label, key in CONDITIONS:
    root = dataset_root(key)
    rows = read_rows(root / "events.csv")
    if reference_rows is None:
        reference_rows = rows
    else:
        validate_pair(reference_rows, rows)
    condition_roots[label] = root
synthetic_labels = np.asarray([row["class_name"] for row in reference_rows])
sample = shared_class_sample(synthetic_labels, real_labels, seed=SEED)


def batched(signals, descriptor, batch_size=256):
    return np.concatenate(
        [descriptor(np.asarray(signals[start : start + batch_size]))
         for start in range(0, len(signals), batch_size)],
        axis=0,
    )


def real_cores(window):
    cache = {}
    output = []
    for row in real_rows:
        relative = row["source_signal_relative_path"]
        if relative not in cache:
            cache[relative] = np.load(
                signal_root / relative, allow_pickle=False
            ).astype(np.float32, copy=False)
        signal = cache[relative]
        output.append(reflect_crop(signal, float(row["center_norm"]) * signal.size, window))
    return np.stack(output)


sweep = []
for window in WINDOWS:
    started = time.time()
    core = slice((4096 - window) // 2, (4096 + window) // 2)
    real_features = batched(real_cores(window), invariant_morphology)
    condition_features = {
        label: batched(
            np.asarray(
                np.load(root / "signals_raw_4096.npy", mmap_mode="r",
                        allow_pickle=False)[:, core]
            ),
            invariant_morphology,
        )
        for label, root in condition_roots.items()
    }
    fit_rows = np.concatenate(
        [condition_features[label][np.concatenate(list(sample.values()))]
         for label, _ in CONDITIONS],
        axis=0,
    )
    scaler = StandardScaler().fit(fit_rows)
    pca = PCA(n_components=16, svd_solver="full").fit(scaler.transform(fit_rows))
    project = lambda values: pca.transform(scaler.transform(values))  # noqa: E731
    real_scores = project(real_features)

    terminal = project(condition_features["asymmetry_5d"])
    coverage = support_coverage(
        real_scores, terminal, real_labels, synthetic_labels,
        sample=sample, quantile=QUANTILE,
    )
    domain = domain_metrics(
        real_scores, terminal, real_labels, synthetic_labels, seed=SEED
    )
    chain = {
        label: support_coverage(
            real_scores, project(condition_features[label]), real_labels,
            synthetic_labels, sample=sample, quantile=QUANTILE,
        )
        for label, _ in CONDITIONS
    }
    sweep.append({
        "window": window,
        "descriptor_dimensions": int(real_features.shape[1]),
        "explained_variance_16": float(pca.explained_variance_ratio_.sum()),
        "coverage": {c: coverage[c]["real_within_radius_fraction"] for c in CLASS_ORDER},
        "nn_median": {c: coverage[c]["real_to_synthetic_nn_median"] for c in CLASS_ORDER},
        "radius": {c: coverage[c]["synthetic_self_nn_radius"] for c in CLASS_ORDER},
        "domain_auc": {
            c: float(domain[c]["domain_classifier_auc_mean"]) for c in CLASS_ORDER
        },
        "local_opposite_fraction": {
            c: float(domain[c]["local_opposite_domain_fraction"]) for c in CLASS_ORDER
        },
        "chain": {
            label: {c: chain[label][c]["real_within_radius_fraction"] for c in CLASS_ORDER}
            for label, _ in CONDITIONS
        },
    })
    print(f"window {window:5d}  {real_features.shape[1]:3d}-D  "
          f"EV16 {pca.explained_variance_ratio_.sum():.4f}  "
          f"coverage " + " ".join(
              f"{c} {100 * coverage[c]['real_within_radius_fraction']:5.1f}%"
              for c in CLASS_ORDER)
          + f"   ({time.time() - started:.0f} s)")

# %% [markdown]
# #### Reproduction check at the published window
#
# At 1 024 the invariant descriptor is the identity, so the whole chain must
# return the published q80 numbers unchanged. This is what licenses reading the
# other two windows as a change of window rather than a change of method.

# %%
reference = published("particle-z8-v2-coverage-conditions-q80-r1")
baseline = next(row for row in sweep if row["window"] == REFERENCE_WINDOW)
window_deviation = max(
    abs(baseline["chain"][label][class_name]
        - reference["conditions"][label]["classes"][class_name]["real_within_radius_fraction"])
    for label, _ in CONDITIONS
    for class_name in CLASS_ORDER
)
assert window_deviation == 0.0, f"reproduction drifted by {window_deviation:.3e}"
print("at 1024 the swept chain reproduces "
      f"particle-z8-v2-coverage-conditions-q80-r1 exactly "
      f"(max deviation {window_deviation:.1e})")

# %%
draw_window_sweep(sweep, CLASS_ORDER, CLASS_COLOUR)

# %% [markdown]
# ### The sweep refutes the plan
#
# The alignment note expected a wider window to describe reality better. It does
# the opposite, and both quantities agree, which is what makes the result hard to
# dismiss: coverage **falls** from 93.6 % to 71.1 % at 2 µm as the window widens,
# while the domain AUC **rises** from 0.66 to 0.75 — real and synthetic become
# *easier* to tell apart, not harder.
#
# A story the numbers refute is itself the finding, so the recommendation in the
# plan does not survive this section unchanged. But before reading the sweep as
# "narrow is better", the objection raised above has to be settled properly.

# %%
print(f"{'window':>7} {'dims':>5} {'EV16':>7}   "
      + "   ".join(f"{c:>18}" for c in CLASS_ORDER))
for row in sweep:
    cells = "   ".join(
        f"{100 * row['coverage'][c]:5.1f}% / {row['domain_auc'][c]:.3f}" for c in CLASS_ORDER
    )
    print(f"{row['window']:>7} {row['descriptor_dimensions']:>5} "
          f"{row['explained_variance_16']:>7.4f}   {cells}")
print("\ncells are coverage / domain AUC; AUC nearer 0.5 means real and synthetic "
      "are harder to tell apart")

# %% [markdown]
# #### Is the wider window merely adding silence the simulator gets wrong?
#
# The objection was that a wider window buys shared noise. The inverse objection
# now matters more: perhaps the synthetic traces are simply *not realistic*
# outside their central core — zero padding, an unmodelled edge, a noise carrier
# injected only near the event — in which case widening the window would punish
# the simulator for something trivial rather than measure it.
#
# That is testable in one figure: the energy profile of both populations across
# the same 4 096 window.

# %% [markdown]
# The comparison has to be **class-matched**. A first attempt on the leading rows
# of each table showed a 66 % amplitude gap in the event core, which dissolved
# once classes were matched: the two tables are ordered differently by class, so
# the gap was composition, not physics. Recording that here because it is the
# trap this control exists to avoid.

# %%
BLOCK = 512
CORE = slice(1536, 2560)
_synthetic_rows = read_rows(condition_roots["asymmetry_5d"] / "events.csv")
_synthetic_class = np.asarray([row["class_name"] for row in _synthetic_rows])
_synthetic_raw = np.load(
    condition_roots["asymmetry_5d"] / "signals_raw_4096.npy",
    mmap_mode="r", allow_pickle=False,
)
_real_raw = real_cores(4096)
_generator = np.random.default_rng(SEED)


def block_rms(values):
    blocks = np.asarray(values, dtype=np.float64).reshape(len(values), 4096 // BLOCK, BLOCK)
    return np.median(np.sqrt(np.mean(np.square(blocks), axis=2)), axis=0)


profiles = {}
core_ratios = {}
for class_name in CLASS_ORDER:
    real_slice = _real_raw[real_labels == class_name]
    eligible = np.flatnonzero(_synthetic_class == class_name)
    drawn = np.sort(_generator.choice(eligible, size=len(real_slice), replace=False))
    synthetic_slice = np.asarray(_synthetic_raw[drawn])
    profiles[class_name] = (block_rms(real_slice), block_rms(synthetic_slice))
    core_ratios[class_name] = tuple(
        float(np.median(
            np.sqrt(np.mean(np.square(values[:, CORE].astype(np.float64)), axis=1))
            / np.sqrt(np.mean(np.square(
                np.concatenate([values[:, :1536], values[:, 2560:]], axis=1).astype(np.float64)
            ), axis=1))
        ))
        for values in (real_slice, synthetic_slice)
    )

figure, axes = plt.subplots(1, 3, figsize=(13, 3.2), sharex=True)
positions = np.arange(4096 // BLOCK) * BLOCK + BLOCK / 2
for axis, class_name in zip(axes, CLASS_ORDER):
    real_profile, synthetic_profile = profiles[class_name]
    axis.plot(positions, real_profile, marker="o", color="#334155", label="real")
    axis.plot(positions, synthetic_profile, marker="s", color=CLASS_COLOUR[class_name],
              label="synthetic")
    axis.axvspan(1536, 2560, color="#94a3b8", alpha=0.25)
    axis.set(title=f"{class_name} · n = {int((real_labels == class_name).sum())}",
             xlabel="sample within the 4096 window")
    axis.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("median block RMS")
axes[0].legend(frameon=False, fontsize=8)
figure.suptitle("Shaded band is the 1024-sample window the descriptor reads", y=1.04,
                fontsize=9)
figure.tight_layout()

print(f"{'class':>6} {'core/context real':>19} {'synthetic':>10}")
for class_name in CLASS_ORDER:
    real_ratio, synthetic_ratio = core_ratios[class_name]
    print(f"{class_name:>6} {real_ratio:19.2f} {synthetic_ratio:10.2f}")

# %% [markdown]
# Class by class, the two populations carry the same energy in the same places:
# the synthetic traces are not padded and not flat outside the core, so the wider
# window is not punishing an artefact. The one visible discrepancy is at 10 µm,
# where synthetic events stand out against their context more than real ones do
# (a core-to-context ratio of 3.7 against 2.7) — a real mismatch, and one the
# 1 024-sample window cannot see because at that width the window *is* the core.
#
# #### What the window actually selects
#
# Put the two measurements together. The 1 024-sample window covers the event's
# high-energy core but clips the low-amplitude tails that the detector's support
# includes, which is why 96.2 % of annotated supports are wider than it. Widening
# to 4 096 admits those tails *and* several thousand samples of instrument
# context that the generator never claimed to model.
#
# So the sweep is not ranking descriptions of the event. It is measuring **how
# much of the recording you agree to be judged on**:
#
# - at 1 024 the test asks "does the simulator reproduce the core of an event?"
#   and the answer is yes, for 85–94 % of real events;
# - at 4 096 it asks "does the simulator reproduce an event *and its
#   surroundings*?" and the answer drops to 63–71 %.
#
# Neither number is wrong. The narrow one is the more flattering, and the deck
# currently reports it without saying which question it answers. That is the
# finding this section delivers, and it is an editorial decision as much as a
# technical one.

# %%
support_widths = np.sort([
    float(row["end_sample"]) - float(row["start_sample"]) for row in real_rows
])
print("fraction of annotated supports wider than the window")
for window in WINDOWS:
    print(f"  {window:5d} : {100 * np.mean(support_widths > window):5.1f} %")
print(f"\nsupport width: median {np.median(support_widths):.0f}, "
      f"p90 {np.percentile(support_widths, 90):.0f}, max {support_widths.max():.0f}")

# %% [markdown]
# #### Recommendation
#
# **Keep the descriptor window-invariant, and do not adopt 4 096 for the
# coverage claim on the strength of the original argument** — that argument was
# that a wider window describes the event better, and it is refuted here.
#
# What the numbers support instead:
#
# 1. **Adopt the fixed 37-band grid and the bin-unit smoothing regardless.** They
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
# What this section does **not** settle: whether retrieval behaves the same way
# under a wider window (it is measured on the encoder's own 512-point input, a
# different path), and whether the MAD corpus — whose supports run to 4 000
# samples — will shift the balance. Both belong to the redo.

# %%
try:
    emitted = notebook_evidence.emit_run(
        workspace,
        section="window-alignment-sweep",
        metrics={
            "schema_version": 1,
            "analysis": "morphology-descriptor-window-sweep",
            "windows": WINDOWS,
            "quantile": QUANTILE,
            "seed": SEED,
            "descriptor": (
                "window-invariant prototype: fixed 37-band grid on 7-80 kHz, "
                "envelope smoothing in bin units; identity at 1024"
            ),
            "identity_deviation_at_reference_window": identity_deviation,
            "reproduces": {
                "run_id": "particle-z8-v2-coverage-conditions-q80-r1",
                "max_deviation": window_deviation,
            },
            "sweep": sweep,
        },
        provenance={
            "datasets": dataset_provenance(),
            "inputs": {
                "events_csv_sha256": notebook_evidence.sha256_file(
                    real_root / "events.csv"
                )
            },
            "parameters": {
                "windows": list(WINDOWS),
                "band_hz": list(BAND_HZ),
                "spectral_bands": int(BAND_CENTRES.size),
                "envelope_bins": ENVELOPE_BINS,
                "components": 16,
                "quantile": QUANTILE,
                "seed": SEED,
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
                "nn_median": "median euclidean distance from a real event to its nearest synthetic",
            },
        },
        claim_boundary=(
            "Measures how the morphology descriptor's window changes coverage, "
            "domain separability and neighbour distance on the z8 development "
            "events, using a window-invariant descriptor prototype that is the "
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
# The coverage claim of this work has the form *"n % of real events fall inside
# the synthetic support"*. That sentence hides two dials the reader never sees:
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
from scipy.linalg import subspace_angles  # noqa: E402
from scipy.spatial.distance import pdist  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from internship_workspace.z8_coverage import (  # noqa: E402
    batched_features,
    load_real_core,
    read_rows,
    shared_class_sample,
    support_coverage,
    validate_pair,
)
from internship_workspace.z8_domain_pca import (  # noqa: E402
    balanced_class_indices,
    domain_metrics,
    fit_synthetic_pca,
)

CONDITIONS = (
    ("white_noise_4d",
     "particles2snr-fbase-z8-cholesky-physicalcorr-effective-snr-synthetic-events@v3"),
    ("real_noise_4d",
     "particles2snr-fbase-z8-cholesky-physicalcorr-effective-snr-synthetic-events@v4"),
    ("asymmetry_5d",
     "particles2snr-fbase-z8-cholesky-physicalcorr-effective-snr-synthetic-events@v5"),
)
CONDITION_LABELS = [label for label, _ in CONDITIONS]
CHAIN_SEED = 20260809
REAL_KEY = "particles2snr-fbase-dual-clean-z8-events-3class-plus-unclear-development@v2"
SIGNAL_KEY = "particles2snr-f-dual-clean-c1-yolo-4class@v2"

started = time.time()
real_root = dataset_root(REAL_KEY)
signal_root = dataset_root(SIGNAL_KEY)
all_rows = read_rows(real_root / "events.csv")
if any(row["split"] == "test" for row in all_rows):
    raise PermissionError("sealed test rows are forbidden")
real_rows = [
    row for row in all_rows
    if row["class_name"] in CLASS_ORDER and row["split"] in {"train", "val"}
]
real_labels = np.asarray([row["class_name"] for row in real_rows])
real_features = batched_features(load_real_core(real_rows, signal_root))

condition_features = {}
reference_rows = None
for label, key in CONDITIONS:
    root = dataset_root(key)
    rows = read_rows(root / "events.csv")
    if reference_rows is None:
        reference_rows = rows
    else:
        validate_pair(reference_rows, rows)
    condition_features[label] = batched_features(
        np.asarray(
            np.load(root / "signals_raw_4096.npy", mmap_mode="r",
                    allow_pickle=False)[:, 1536:2560]
        )
    )
synthetic_labels = np.asarray([row["class_name"] for row in reference_rows])
chain_sample = shared_class_sample(synthetic_labels, real_labels, seed=CHAIN_SEED)

fit_rows = np.concatenate(
    [condition_features[label][np.concatenate(list(chain_sample.values()))]
     for label in CONDITION_LABELS],
    axis=0,
)
chain_scaler = StandardScaler().fit(fit_rows)
chain_pca = PCA(n_components=16, svd_solver="full").fit(chain_scaler.transform(fit_rows))
chain_project = lambda values: chain_pca.transform(chain_scaler.transform(values))  # noqa: E731
chain_real = np.asarray(chain_project(real_features), dtype=np.float64)
chain_synthetic = {
    label: np.asarray(chain_project(condition_features[label]), dtype=np.float64)
    for label in CONDITION_LABELS
}
print(f"{len(real_rows)} real train/val events, {len(reference_rows)} synthetic per "
      f"condition, descriptor {real_features.shape[1]}-D  ({time.time() - started:.0f} s)")

# %% [markdown]
# ### Reproducing the published q80 chain
#
# Nothing below is readable unless this notebook computes the same chain the deck
# quotes. The protocol is the shipped one: one PCA(16) fitted on the three
# generator conditions pooled, one balanced synthetic draw shared across
# conditions (seed 20260809), a radius recomputed per condition, euclidean
# distance in 16 dimensions, real events cropped to 1 024 samples and synthetic
# events sliced from the same 1 024 samples of their 4 096-sample core.

# %%
chain = {
    label: support_coverage(
        chain_real, chain_synthetic[label], real_labels, synthetic_labels,
        sample=chain_sample, quantile=0.80,
    )
    for label in CONDITION_LABELS
}
reference = published("particle-z8-v2-coverage-conditions-q80-r1")
chain_deviation = max(
    abs(chain[label][class_name]["real_within_radius_fraction"]
        - reference["conditions"][label]["classes"][class_name]["real_within_radius_fraction"])
    for label in CONDITION_LABELS for class_name in CLASS_ORDER
)
radius_deviation = max(
    abs(chain[label][class_name]["synthetic_self_nn_radius"]
        - reference["conditions"][label]["classes"][class_name]["synthetic_self_nn_radius"])
    for label in CONDITION_LABELS for class_name in CLASS_ORDER
)
assert chain_deviation == 0.0, f"reproduction drifted by {chain_deviation:.3e}"
assert radius_deviation == 0.0, f"radius drifted by {radius_deviation:.3e}"
for class_name in CLASS_ORDER:
    values = [chain[label][class_name]["real_within_radius_fraction"]
              for label in CONDITION_LABELS]
    print(f"{class_name:>5}  " + " → ".join(f"{value:.6f}" for value in values))
print("\nreproduces particle-z8-v2-coverage-conditions-q80-r1 exactly "
      f"(coverage and radius, max deviation {chain_deviation:.1e})")

# %% [markdown]
# ### Part 1 · What the quantile decides
#
# Four runs exist on this identical pipeline, differing only by `--quantile`:
# q95 (the tool's default, and the deck's original choice), q90, q85 and q80.
# They are the cleanest controlled experiment available on this question, because
# the basis, the sample, the seed and the data are byte-identical across them —
# only the tick on the ruler moves.
#
# The first question is whether the quantile **fabricates** the story or merely
# **amplifies** it. Those are different failures. Amplification is a presentation
# problem: the ranking is real and the reader is being shown its most flattering
# scale. Fabrication would be a scientific problem: the ranking itself would
# depend on an arbitrary choice.

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
print("the four runs share one basis "
      f"(explained variance over 16 components "
      f"{published_runs[0.95]['pca']['explained_variance_16']:.6f}); "
      "only the quantile differs\n")
print(f"{'step':>34} " + "  ".join(f"{c:>16}" for c in CLASS_ORDER))
for step in ("white_noise_4d -> real_noise_4d", "real_noise_4d -> asymmetry_5d"):
    for quantile in PUBLISHED_QUANTILES:
        gains = published_runs[quantile]["gains_percentage_points"][step]
        print(f"{step + f'  q{quantile:.2f}':>34} "
              + "  ".join(f"{gains[c]:+15.2f}pp" for c in CLASS_ORDER))
    print()

# %% [markdown]
# The gains move a great deal. The first generator step — replacing white noise
# with measured instrument noise — is worth **+14.3 points** at 2 µm when scored
# at q95 and **+31.0 points** at q80: the headline more than doubles. The second
# step — the paired asymmetry — is worth **+3.1 points** at 4 µm under q95 and
# **+9.8** under q80: it triples.
#
# Both movements have the same cause, and it is not rhetorical. At q95 the radius
# is generous enough that the middle condition already covers 98.0 % of the 2 µm
# events; there is almost no room left, so the last step cannot show anything.
# **q95 measures against a ceiling.** Lowering the quantile un-saturates the
# measurement, which is a legitimate reason to prefer q80 and a better one than
# "q80 is stricter, therefore more honest".
#
# That still leaves the fabrication question open. Answering it needs the sweep
# to go where no run has been — below 0.80 — because a story that only holds on
# the four quantiles someone chose to publish is not a story.

# %%
QUANTILE_GRID = tuple(round(float(value), 2) for value in np.arange(0.10, 0.96, 0.05))
started = time.time()
quantile_sweep = {}
for quantile in QUANTILE_GRID:
    quantile_sweep[quantile] = {
        label: {
            class_name: {
                "coverage": measured[class_name]["real_within_radius_fraction"],
                "radius": measured[class_name]["synthetic_self_nn_radius"],
            }
            for class_name in CLASS_ORDER
        }
        for label, measured in (
            (label, support_coverage(
                chain_real, chain_synthetic[label], real_labels, synthetic_labels,
                sample=chain_sample, quantile=quantile))
            for label in CONDITION_LABELS
        )
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
    for label in CONDITION_LABELS for class_name in CLASS_ORDER
)
assert grid_deviation == 0.0, f"the swept grid drifted by {grid_deviation:.3e}"
print(f"swept {len(QUANTILE_GRID)} quantiles from {min(QUANTILE_GRID):.2f} to "
      f"{max(QUANTILE_GRID):.2f} in {time.time() - started:.0f} s; the four published "
      "quantiles are reproduced exactly")
print(f"orderings white < real-noise < asymmetry violated: "
      f"{len(violations)} of {len(QUANTILE_GRID) * len(CLASS_ORDER)}")

# %%
draw_quantile_curve(quantile_sweep, QUANTILE_GRID, CONDITION_LABELS, CLASS_ORDER,
                    CLASS_COLOUR, PUBLISHED_QUANTILES)

# %% [markdown]
# **The quantile amplifies; it does not fabricate.** Across 0.10 to 0.95 — a range
# far wider than anyone would defend — the three conditions never change places,
# in any class: 54 orderings, 0 violations. The chain's conclusion, that measured
# noise buys most of the coverage and asymmetry finishes it, is a property of the
# generator changes and not of the ruler.
#
# What *is* a property of the ruler is every absolute number the deck prints.

# %%
draw_gain_amplification(published_runs, PUBLISHED_QUANTILES, CLASS_ORDER, CLASS_COLOUR)

# %% [markdown]
# #### Where the argument breaks
#
# The deck's closing sentence on the chain is that **every class ends above the
# 80 % line**. At q80 that reads as a coincidence worth exploiting: 80 % of the
# synthetic reference events lie within the radius of another synthetic event, so
# a coverage above 80 % looks like "real events are covered better than the
# synthetic cloud covers itself".
#
# Two things have to be checked before that sentence can be used. First, where it
# stops being arithmetically true. Second — and this is the sharper objection —
# whether the two numbers being compared are measurements of the same thing.

# %%
CROSSING_GRID = tuple(round(float(value), 2) for value in np.arange(0.60, 0.96, 0.01))
crossing = {}
for quantile in CROSSING_GRID:
    measured = support_coverage(
        chain_real, chain_synthetic["asymmetry_5d"], real_labels, synthetic_labels,
        sample=chain_sample, quantile=quantile,
    )
    crossing[quantile] = {c: measured[c]["real_within_radius_fraction"] for c in CLASS_ORDER}
above_bar = min(q for q in CROSSING_GRID
                if all(crossing[q][c] >= 0.80 for c in CLASS_ORDER))
above_self = max(q for q in CROSSING_GRID
                 if all(crossing[q][c] >= q for c in CLASS_ORDER))
print("terminal condition · margin of coverage over the quantile itself")
print(f"{'q':>6} " + "  ".join(f"{c:>16}" for c in CLASS_ORDER))
for quantile in PUBLISHED_QUANTILES[::-1]:
    row = published_runs[quantile]["conditions"]["asymmetry_5d"]["classes"]
    print(f"{quantile:>6.2f} " + "  ".join(
        f"{100 * row[c]['real_within_radius_fraction']:6.2f}% "
        f"({100 * (row[c]['real_within_radius_fraction'] - quantile):+5.1f})"
        for c in CLASS_ORDER))
print(f"\nevery class stays above 80 % for q >= {above_bar:.2f}; at q = "
      f"{round(above_bar - 0.01, 2):.2f} the 4 µm class falls to "
      f"{100 * crossing[round(above_bar - 0.01, 2)]['4um']:.1f} %")
print(f"every class stays above its own quantile for q <= {above_self:.2f}; at q = "
      f"{round(above_self + 0.01, 2):.2f} the 10 µm class falls "
      f"{100 * (round(above_self + 0.01, 2) - crossing[round(above_self + 0.01, 2)]['10um']):.1f} "
      "points short of the bar")
print(f"the sentence therefore holds only for {above_bar:.2f} <= q <= {above_self:.2f}")

# %% [markdown]
# Arithmetically the sentence lives in a corridor, **0.72 ≤ q ≤ 0.91**, and the
# deck's previous choice was outside it. At q95 the sentence is simply false:
# 10 µm covers 93.5 % against a 95 % bar, a margin of −1.5 points, and 4 µm
# clears its own bar by 0.2. Below q = 0.72 it fails at the other end, the 4 µm
# class dropping under 80 %. So the move to q80 did not merely make the claim
# stronger — at q95 there was no claim to make. That is worth saying plainly,
# because it is the one place where a quantile change repaired something rather
# than flattering it.
#
# It also means the sentence is true only over a nineteen-point window of an
# arbitrary parameter, which is thin support for a closing line.
#
# #### The sharpest objection: the two numbers are not measured the same way
#
# The radius is the 80th percentile of nearest-neighbour distances **inside a
# thinned reference draw** — one synthetic event per real event of that class,
# because that is what makes the conditions comparable. But a real event's
# distance is measured to the **whole** synthetic class population. The two sides
# of the comparison therefore search reference sets of very different densities.

# %%
density = {
    class_name: {
        "synthetic_population": int((synthetic_labels == class_name).sum()),
        "reference_draw": int(chain_sample[class_name].size),
        "real": int((real_labels == class_name).sum()),
    }
    for class_name in CLASS_ORDER
}
print(f"{'class':>6} {'synthetic cloud':>16} {'reference draw':>15} "
      f"{'real events':>12} {'density ratio':>14}")
for class_name in CLASS_ORDER:
    row = density[class_name]
    row["density_ratio"] = row["synthetic_population"] / row["reference_draw"]
    print(f"{class_name:>6} {row['synthetic_population']:>16d} "
          f"{row['reference_draw']:>15d} {row['real']:>12d} {row['density_ratio']:>13.1f}×")

# %% [markdown]
# A real event queries a cloud sixteen to twenty-five times denser than the one
# whose spacing set the radius. Nearest-neighbour distance falls with density, so
# real events are being scored on an easier test than the 80 % they are compared
# against. The control is to score the synthetic events the same way: take the
# events *not* in the reference draw and measure each one's distance to its
# nearest **other** synthetic event of the same class in the full cloud — the
# exact query a real event gets — against the exact same radius.

# %%
control = {}
for class_name in CLASS_ORDER:
    scores = chain_synthetic["asymmetry_5d"][:, :16]
    class_index = np.flatnonzero(synthetic_labels == class_name)
    reference_cloud = scores[chain_sample[class_name]]
    self_distance = NearestNeighbors(n_neighbors=2).fit(reference_cloud).kneighbors(
        reference_cloud, return_distance=True)[0][:, 1]
    radius = float(np.quantile(self_distance, 0.80))
    held_out = np.setdiff1d(class_index, chain_sample[class_name])
    synthetic_distance = NearestNeighbors(n_neighbors=2).fit(scores[class_index]).kneighbors(
        scores[held_out], return_distance=True)[0][:, 1]
    real_slice = chain_real[real_labels == class_name][:, :16]
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
# The sentence does not survive. Measured the way real events are measured, the
# synthetic cloud covers **itself** at 99.4 / 97.9 / 99.2 %, against 93.6 / 85.2 /
# 87.9 % for the real events. Real events are covered *worse* than the synthetic
# cloud covers itself, by 6 to 13 points — the opposite of what the deck asserts.
# Measured the other way round, with both populations querying the same thinned
# cloud, real coverage falls to 56.8 / 68.3 / 70.1 % against the 80 % the
# reference draw achieves by construction. Either fair comparison inverts the
# claim; only the unmatched one supports it.
#
# **What this does and does not damage.** It does not touch the chain's ranking:
# the density ratio is identical across the three conditions, so it cancels out
# of every gain the chain reports, which is exactly the quantity the tool was
# built to measure and the only quantity its claim boundary asserts. What it
# damages is the *absolute* reading, and specifically the one rhetorical sentence
# that turns a coverage percentage into a statement about the cloud's own
# coherence. The 85–94 % is a real measurement of a real question — "how far is a
# real event from the nearest of ~16 000 synthetic events, compared with a radius
# derived from a much sparser draw" — but it is not the question the sentence
# claims it answers.
#
# **Recommendation.** Keep q80 and keep the chain. Drop the self-coverage
# sentence, or replace it with the matched-density number, which is defensible,
# unflattering and reported here for the first time. And report the quantile in
# the claim, since the same data yields 60.8 % or 83.7 % for white noise at 2 µm
# depending only on it.

# %% [markdown]
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
# The chain fits one PCA(16) on all three conditions pooled with a shared
# balanced draw. The two-space introduction figures — the slides that teach the
# reader what "an event is a point" means — instead reuse the **stored scores** of
# `particle-z8-v2-paired-asymmetry-pca-r2`, a basis fitted on the v4 and v5 arms
# only, with a different draw and a different seed. The run is honest about it:
# its own metrics carry `"not_commensurable_with": "particle-z8-v2-coverage-
# conditions-r1"`. The two nonetheless sit on neighbouring slides.
#
# A7 asked how much the quantile is worth. A8 deserves the same treatment: how
# much is the basis worth? "Not commensurable" is a statement about licence, not
# about magnitude, and the reader deserves the magnitude.
#
# The intro basis is reproduced here from the registry, and checked against the
# stored scores it is supposed to be.

# %%
intro_scores = np.load(
    run_dir("particle-z8-v2-paired-asymmetry-pca-r2") / "pca_scores.npz",
    allow_pickle=True,
)
INTRO_SEED = 20260803
intro_fit_indices = balanced_class_indices(
    synthetic_labels, per_class=int(intro_scores["fit_indices"].size // len(CLASS_ORDER)),
    seed=INTRO_SEED,
)
assert np.array_equal(intro_fit_indices, intro_scores["fit_indices"]), "draw differs"
intro_pool = np.concatenate(
    (condition_features["real_noise_4d"][intro_fit_indices],
     condition_features["asymmetry_5d"][intro_fit_indices]),
    axis=0,
)
intro_scaler = StandardScaler().fit(intro_pool)
intro_pca = PCA(n_components=16, svd_solver="full").fit(intro_scaler.transform(intro_pool))
intro_project = lambda values: intro_pca.transform(intro_scaler.transform(values))  # noqa: E731
intro_deviation = float(np.abs(
    intro_project(condition_features["asymmetry_5d"])
    - np.asarray(intro_scores["candidate"], dtype=np.float64)
).max())
assert intro_deviation < 1.0e-9, f"reproduction drifted by {intro_deviation:.3e}"
print("the balanced draw and the 16 stored score columns of "
      f"particle-z8-v2-paired-asymmetry-pca-r2 are reproduced from the registry "
      f"(max deviation {intro_deviation:.1e}, float32 storage)")
print(f"\n{'basis':>34} {'16 axes':>10} {'PC1-PC2':>9}")
variance_table = {
    "chain · three conditions pooled": (
        float(chain_pca.explained_variance_ratio_.sum()),
        float(chain_pca.explained_variance_ratio_[:2].sum())),
    "intro · v4+v5 paired arms": (
        float(intro_pca.explained_variance_ratio_.sum()),
        float(intro_pca.explained_variance_ratio_[:2].sum())),
}
for name, (sixteen, plane) in variance_table.items():
    print(f"{name:>34} {100 * sixteen:9.2f}% {100 * plane:8.2f}%")

# %% [markdown]
# #### How different is "close"?
#
# Two 16-dimensional subspaces of the same 101-dimensional descriptor space can
# be compared exactly, by their **principal angles** — the sequence of angles
# between the closest pair of directions, then the closest pair orthogonal to
# those, and so on. Zero everywhere means the same subspace; 90° means a
# direction one basis reads and the other is blind to. The scaling step belongs
# to the projection, so the directions compared are the rows of the components
# divided by each basis's own standard deviations.

# %%
def read_directions(scaler, pca):
    """An orthonormal basis of the descriptor directions a projection reads."""
    return np.linalg.qr((pca.components_ / scaler.scale_[None, :]).T)[0]


principal_angles = np.degrees(
    subspace_angles(read_directions(chain_scaler, chain_pca),
                    read_directions(intro_scaler, intro_pca))
)
print("principal angles between the chain basis and the intro basis, degrees:")
print("  " + "  ".join(f"{angle:5.1f}" for angle in np.sort(principal_angles)))
print(f"\nmedian {np.median(principal_angles):.1f}°, mean {principal_angles.mean():.1f}°, "
      f"largest {principal_angles.max():.1f}°; mean cos² "
      f"{np.mean(np.cos(np.radians(principal_angles)) ** 2):.4f}")

# %%
intro_real = np.asarray(intro_project(real_features), dtype=np.float64)
intro_terminal = np.asarray(
    intro_project(condition_features["asymmetry_5d"]), dtype=np.float64
)
neighbour = {}
for class_name in CLASS_ORDER:
    real_index = np.flatnonzero(real_labels == class_name)
    synthetic_index = np.flatnonzero(synthetic_labels == class_name)
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
# The answer is precise and two-sided. Fifteen of the sixteen directions agree to
# within 23°, and the sixteenth is 72.5° apart: the two bases share a
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
# About half the time, the chain would have drawn a different partner. The
# slide's pedagogical claim is the fragile one, not the chain's number.

# %%
basis_coverage = {}
for name, (real_scores, synthetic_scores) in (
    ("chain · pooled", (chain_real, chain_synthetic["asymmetry_5d"])),
    ("intro · paired arms", (intro_real, intro_terminal)),
):
    measured = support_coverage(real_scores, synthetic_scores, real_labels,
                                synthetic_labels, sample=chain_sample, quantile=0.80)
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
sweep_run = published("particle-z8-v2-real-synthetic-pca-r2")
sweep_variance = sweep_run["variants"]["Morphology · primary"]["explained_variance_ratio"]
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
      f"at 2 µm, against the chain's {100 * chain['asymmetry_5d']['2um']['real_within_radius_fraction']:.2f} % "
      "— same class, same condition, same quantile, different basis and different draw")
print(f"its radius is stored under the key 'radius_p95' with the value "
      f"{intro_metrics['radius_p95']:.6f}, produced by --quantile 0.80")

# %% [markdown]
# Three bases are in play on adjoining slides, and each has a different variance
# profile. The quoted 60.9 / 21.1 belongs to the chain's pooled basis; the plane
# actually drawn on that slide belongs to the paired-arms basis and holds
# **23.4 %**, not 21.1 %, with 16 axes holding **63.2 %**, not 60.9 %. The figure
# tool hard-codes the wrong one into its own axis label. The error is small —
# 2.3 percentage points — and the defect is not: a caption that describes a
# different picture from the one above it is unfalsifiable by the reader.
#
# The naming defect A7 flags is confirmed in the same file. The tool writes its
# radius under the key **`radius_p95`** whatever `--quantile` was passed, and r2
# was run with `--quantile 0.80`; the published `radius_p95` is a p80 radius.
# Anyone reading that JSON without the run command gets the wrong protocol.
#
# **Recommendation.** A8 as written — one basis per campaign, serialised once and
# reloaded — is the right decision, and this measurement says *why*: not because
# the coverage number would change much (it moves by at most 2.6 points) but
# because example selection, twin display and any "nearest neighbour" claim are
# a coin flip between the two bases. Serialising the basis is what makes an
# illustration and a measurement talk about the same object.

# %% [markdown]
# ### Part 3 · Why sixteen components
#
# Sixteen is not derived from anything. The variance spectrum has no elbow after
# PC2, so the choice is a frozen analysis constant, and the published argument for
# it is a sweep: `particle-z8-morphology-dimension-sweep-r2` truncates the stored
# scores to d = 2…16 and reports coverage, **domain AUC** (the cross-validated
# area under the ROC curve of a logistic classifier trying to tell real from
# synthetic; 0.5 means indistinguishable) and the **distance contrast** — the
# ratio std/mean of pairwise distances, the quantity that collapses when
# high-dimensional distances concentrate and nearest-neighbour queries stop
# discriminating. The claimed conclusion is a plateau over d ∈ [12, 16].
#
# It belongs in this section because it is a basis question, and because it is the
# analysis that inherits the hard-coded q95 named in Part 1. Two caveats travel
# with it, both load-bearing: it consumes the stored scores of
# `particle-z8-v2-real-synthetic-pca-r2`, a **third** basis fitted on the
# white-noise condition alone; and those scores are truncated at 16 columns, so
# the sweep structurally cannot see past the number it is meant to justify.

# %%
sweep_scores = np.load(
    run_dir("particle-z8-v2-real-synthetic-pca-r2") / "pca_scores.npz", allow_pickle=True
)
sweep_real = np.asarray(sweep_scores["real_morphology"], dtype=np.float64)
sweep_synthetic = np.asarray(sweep_scores["synthetic_morphology"], dtype=np.float64)
sweep_real_labels = sweep_scores["real_class"].astype(str)
sweep_synthetic_labels = sweep_scores["synthetic_class"].astype(str)
SWEEP_SEED = 20260724
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
# `domain_metrics` cannot be asked for another quantile — 0.95 is written into
# it. `support_coverage` can, but it takes the reference cloud as an argument,
# and `domain_metrics` draws its own internally. Rebuilding **that draw** (not the
# method) is what lets the shipped coverage function score the same cloud at
# another quantile. The gate is that at 0.95 it must return the published numbers
# exactly; otherwise it is scoring something else.

# %%
def domain_reference_draw(real_labels_, synthetic_labels_, *, seed):
    """The synthetic reference `domain_metrics` samples internally, rebuilt.

    Only the draw is reproduced, so that `support_coverage` can be handed the
    same cloud at a different quantile. The real draw is consumed but discarded
    because it only advances the generator.
    """
    real_values = np.asarray(real_labels_).astype(str)
    synthetic_values = np.asarray(synthetic_labels_).astype(str)
    draw = {}
    for class_name in CLASS_ORDER:
        real_index = np.flatnonzero(real_values == class_name)
        synthetic_index = np.flatnonzero(synthetic_values == class_name)
        count = min(real_index.size, synthetic_index.size)
        generator = np.random.default_rng(seed + CLASS_ORDER.index(class_name))
        generator.choice(real_index, size=count, replace=False)
        draw[class_name] = generator.choice(synthetic_index, size=count, replace=False)
    return draw


sweep_draw = domain_reference_draw(sweep_real_labels, sweep_synthetic_labels, seed=SWEEP_SEED)
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
print(f"{'d':>3} " + "  ".join(f"{c + ' q95':>12}" for c in CLASS_ORDER)
      + "  " + "  ".join(f"{c + ' q80':>12}" for c in CLASS_ORDER) + "     gap (pp)")
for entry in dimension_run["sweep"]:
    dimensions = entry["dimensions"]
    print(f"{dimensions:>3} "
          + "  ".join(f"{100 * entry['coverage'][c]:11.1f}%" for c in CLASS_ORDER)
          + "  " + "  ".join(f"{100 * sweep_at_q80[dimensions][c]:11.1f}%" for c in CLASS_ORDER)
          + "   " + " ".join(
              f"{100 * (entry['coverage'][c] - sweep_at_q80[dimensions][c]):5.1f}"
              for c in CLASS_ORDER))
print("\nchain, white-noise condition at q80 (the condition this basis was fitted on): "
      + "  ".join(
          f"{c} {100 * quantile_sweep[0.80]['white_noise_4d'][c]['coverage']:.1f}%"
          for c in CLASS_ORDER))

# %% [markdown]
# At the published d = 16 the quantile alone is worth **21.1 / 16.1 / 27.3
# percentage points**. That is the size of the incommensurability the deck
# carries between two adjoining slides, and it dwarfs everything else in this
# section.
#
# The rest of the comparison is reassuring, and worth stating because it settles
# the relative weight of A7 and A8. Once the sweep is put on q80 it reads
# 65.2 / 57.4 / 56.3 % at d = 16, against 60.8 / 56.5 / 54.5 % for the chain's
# white-noise column — the same condition, a different basis, and agreement to
# within 4.4 / 0.9 / 1.7 points. **The quantile was the whole incommensurability;
# the basis contributes a few points.** A7 is the urgent fix, A8 the structural
# one.

# %% [markdown]
# #### Is there actually a plateau?
#
# The published claim is that past d ≈ 12 nothing moves. That holds for the AUC —
# the change from 12 to 16 is under 0.001 in every class — and the figure's title
# generalises it to the solid coverage curve as well. Coverage does not cooperate:
# at 4 µm it climbs 9.5 points between d = 12 and d = 16 at q95, and 8.0 points at
# q80. And because the stored scores stop at 16, the sweep cannot ask the obvious
# next question, which is whether anything settles after 16 either.
#
# It can be asked here. The basis is refitted from the registry with 32
# components instead of 16 — the same standardisation, the same balanced draw,
# the same seed — so the first sixteen columns are the published ones and the
# extension is a strict continuation rather than a new experiment.

# %%
sweep_per_class = min(int((synthetic_labels == c).sum()) for c in CLASS_ORDER)
wide_scaler, wide_pca, wide_indices = fit_synthetic_pca(
    condition_features["white_noise_4d"], synthetic_labels,
    per_class=sweep_per_class, seed=SWEEP_SEED, components=101,
)
wide_synthetic = wide_pca.transform(wide_scaler.transform(
    condition_features["white_noise_4d"]))
wide_real = wide_pca.transform(wide_scaler.transform(real_features))
prefix_deviation = float(np.abs(wide_synthetic[:, :16] - sweep_synthetic).max())
print(f"refitted basis reproduces the stored 16 columns to {prefix_deviation:.1e} "
      "(float32 storage)")
print("explained variance: "
      + "   ".join(f"{d} axes {100 * wide_pca.explained_variance_ratio_[:d].sum():.1f}%"
                   for d in (16, 24, 32, 101)))

EXTENDED = (12, 14, 16, 20, 24, 28, 32)
wide_draw = domain_reference_draw(real_labels, synthetic_labels, seed=SWEEP_SEED)
extended = {}
for dimensions in EXTENDED:
    coverage = support_coverage(wide_real, wide_synthetic, real_labels, synthetic_labels,
                                sample=wide_draw, quantile=0.80, dimensions=dimensions)
    separability = domain_metrics(wide_real, wide_synthetic, real_labels, synthetic_labels,
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
# The plateau is local, not terminal. Past sixteen both quantities resume moving:
# 4 µm coverage climbs from 57.4 % to 66.8 % between d = 16 and d = 32, and the
# domain AUC — which the plateau argument rests on — rises from 0.963 to 0.974 at
# 4 µm, 0.929 to 0.944 at 2 µm and 0.859 to 0.901 at 10 µm. Sixteen sits on a flat
# stretch of the curve; it is not where the curve stops.
#
# Two cautions on reading that panel. The 10 µm coverage wobbles by several points
# from one d to the next because only 231 real events carry it, so its trend is
# the weakest of the three; the AUC, computed on a balanced 231-per-domain sample,
# is steadier and moves the same way. And these AUC values (0.86–0.97) are far
# above the 0.66–0.75 the window section reports, because this basis and this
# condition are the white-noise ones — the least realistic arm of the chain.
# Domain AUC is no more commensurable across runs than coverage is.

# %% [markdown]
# #### The concentration that does not arrive
#
# The appendix's third argument is the strongest-sounding one: distance contrast
# falls from 0.60 at d = 2 to 0.29 at d = 16, and a √d extrapolation puts the raw
# 101-dimensional descriptor near 0.11, where "nearest-neighbour queries stop
# discriminating". That extrapolation is stated in the run's own metrics as
# "measured, not fitted" — but the quantity extrapolated *to* was never measured,
# because the stored scores stop at 16.
#
# PCA is an orthonormal rotation, so keeping all 101 components preserves every
# distance of the standardised descriptor exactly. The endpoint is therefore not
# an extrapolation at all: it can simply be computed.

# %%
def distance_contrast(scores, labels, *, dimensions, sample=2000, seed=SWEEP_SEED):
    """std/mean of pairwise distances, the published contrast definition."""
    output = {}
    for class_name in CLASS_ORDER:
        rows = scores[labels == class_name][:, :dimensions]
        generator = np.random.default_rng(seed + CLASS_ORDER.index(class_name))
        if rows.shape[0] > sample:
            rows = rows[generator.choice(rows.shape[0], size=sample, replace=False)]
        distances = pdist(rows)
        output[class_name] = float(distances.std() / distances.mean())
    return output


contrast_deviation = max(
    abs(distance_contrast(wide_synthetic, synthetic_labels,
                          dimensions=entry["dimensions"])[class_name]
        - entry["contrast"][class_name])
    for entry in dimension_run["sweep"] for class_name in CLASS_ORDER
)
assert contrast_deviation == 0.0, f"contrast drifted by {contrast_deviation:.3e}"
print(f"the contrast definition reproduces the published sweep exactly "
      f"(max deviation {contrast_deviation:.1e})\n")

CONTRAST_GRID = (2, 4, 8, 12, 16, 24, 32, 48, 64, 101)
contrast_curve = {
    dimensions: distance_contrast(wide_synthetic, synthetic_labels, dimensions=dimensions)
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
# The contrast does not collapse. It falls quickly to about d = 12 and then
# flattens, reaching **0.21** at the full 101 dimensions where the appendix
# predicts 0.11 — the √d law is a bad model of this descriptor, whose measured
# decay exponent is 0.157. The raw descriptor is in essentially the same
# discrimination regime as the 16-axis ruler.
#
# So the honest answer to "why sixteen" is not the one on the slide. Each of the
# three published arguments weakens under measurement: coverage is still moving
# at 16, the AUC plateau is a local flat spot that ends by d = 20, and the
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
# idea. Everything here is development data — the sealed test split is never
# read, and no result below authorises a validation claim, a dataset promotion or
# a change to a shipped tool.

# %%
evidence_metrics = {
    "schema_version": 1,
    "analysis": "z8-coverage-quantile-and-basis-alignment",
    "reproduces": {
        "particle-z8-v2-coverage-conditions-q80-r1": chain_deviation,
        "particle-z8-v2-paired-asymmetry-pca-r2": intro_deviation,
        "particle-z8-morphology-dimension-sweep-r2": sweep_deviation,
    },
    "quantile_sweep": {
        f"{quantile:.2f}": {
            label: {c: quantile_sweep[quantile][label][c] for c in CLASS_ORDER}
            for label in CONDITION_LABELS
        }
        for quantile in QUANTILE_GRID
    },
    "condition_ordering_violations": len(violations),
    "self_coverage_sentence_holds_between": [above_bar, above_self],
    "reference_density": density,
    "self_coverage_control_q80_asymmetry_5d": control,
    "basis_divergence": {
        "principal_angles_degrees": np.sort(principal_angles).tolist(),
        "mean_squared_cosine": float(np.mean(np.cos(np.radians(principal_angles)) ** 2)),
        "explained_variance": {
            name: {"sixteen": sixteen, "pc1_pc2": plane}
            for name, (sixteen, plane) in variance_table.items()
        },
        "nearest_neighbour": {
            c: {k: v for k, v in neighbour[c].items() if not isinstance(v, np.ndarray)}
            for c in CLASS_ORDER
        },
        "coverage_q80_per_basis": basis_coverage,
    },
    "dimension_sweep_at_q80": {str(d): sweep_at_q80[d] for d in DIMENSIONS},
    "dimension_sweep_extended": {str(d): extended[d] for d in EXTENDED},
    "distance_contrast": {
        "per_dimension": {str(d): contrast_curve[d] for d in CONTRAST_GRID},
        "class_mean_at_101": measured_contrast[-1],
        "published_sqrt_d_extrapolation": published_extrapolation,
        "measured_decay_exponent_16_to_101": float(exponent),
    },
}
evidence_provenance = {
    "datasets": dataset_provenance(),
    "inputs": {
        "real_events_sha256": notebook_evidence.sha256_file(real_root / "events.csv"),
        "paired_asymmetry_scores_sha256": notebook_evidence.sha256_file(
            run_dir("particle-z8-v2-paired-asymmetry-pca-r2") / "pca_scores.npz"),
        "real_synthetic_scores_sha256": notebook_evidence.sha256_file(
            run_dir("particle-z8-v2-real-synthetic-pca-r2") / "pca_scores.npz"),
    },
    "parameters": {
        "quantile_grid": list(QUANTILE_GRID),
        "crossing_grid": [min(CROSSING_GRID), max(CROSSING_GRID), 0.01],
        "chain_seed": CHAIN_SEED,
        "intro_seed": INTRO_SEED,
        "dimension_sweep_seed": SWEEP_SEED,
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
print(f"evidence payload serialises, {len(json.dumps(evidence_metrics))} bytes of metrics")

try:
    emitted = notebook_evidence.emit_run(
        workspace,
        section="quantile-basis-alignment",
        metrics=evidence_metrics,
        provenance=evidence_provenance,
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
