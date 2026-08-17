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
# # Does the simulation match reality?
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
# How a signal becomes a point, whether the synthetic cloud covers the
# real one, what a twin looks like, and the strictest test of all —
# whether a regenerated event can find its own parent.
#
# ## The other notebooks in this series
#
# - [`simulation_chain_1_generator`](simulation_chain_1_generator.py) — the generator
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
# --- morphology_space ---
"""Helpers for the `morphology_space` section.

The drawings this section needs are already shipped, in
`internship_workspace.chain_figures`. What lives here is the one thing that is
bookkeeping rather than argument: the evidence payload for the descriptor-window
audit, which is forty lines of manifest around four numbers.
"""

import numpy as np

from internship_workspace import notebook_evidence
from internship_workspace.z8_domain_pca import CLASS_ORDER


def support_widths_of(real):
    """Annotated support width of every event, in samples, in row order."""
    return np.asarray(
        [float(row["end_sample"]) - float(row["start_sample"]) for row in real.rows]
    )


def window_audit_payload(real, *, candidate_window, datasets, sampling_frequency_hz):
    """`emit_run` metrics and provenance for the descriptor-window audit.

    `real` is a `chain_data.RealEvents`. The measurement is one comparison --
    annotated support against the fixed window that describes it -- and
    everything below is the manifest that makes it citable.
    """
    widths = support_widths_of(real)
    window = int(real.cores.shape[1])
    return {
        "metrics": {
            "schema_version": 1,
            "analysis": "descriptor-window-versus-event-support",
            "sampling_frequency_hz": sampling_frequency_hz,
            "descriptor_window_samples": window,
            "candidate_window_samples": candidate_window,
            "population": {
                "events": len(real.rows),
                "selection": "physical train and val rows of the three classes",
            },
            "support_width_samples": {
                "median": float(np.median(widths)),
                "p90": float(np.percentile(widths, 90)),
                "p99": float(np.percentile(widths, 99)),
                "max": float(widths.max()),
            },
            "wider_than_descriptor_window_fraction": float(np.mean(widths > window)),
            "wider_than_candidate_window_fraction": float(
                np.mean(widths > candidate_window)
            ),
            "wider_than_descriptor_window_by_class": {
                class_name: float(np.mean(real.by_class(widths, class_name) > window))
                for class_name in CLASS_ORDER
            },
        },
        "provenance": {
            "datasets": datasets,
            "inputs": {
                "events_csv_sha256": notebook_evidence.sha256_file(real.events_csv)
            },
            "parameters": {
                "descriptor_window_samples": window,
                "candidate_window_samples": candidate_window,
                "population": "physical train and val rows of the three classes",
            },
            "metric_definitions": {
                "support_width": (
                    "end_sample minus start_sample of the detector annotation"
                ),
                "wider_than_window_fraction": (
                    "fraction of events whose annotated support exceeds the fixed "
                    "descriptor window, so the descriptor cannot see the event in full"
                ),
            },
        },
    }


# --- coverage ---
"""Helpers for the `coverage` section.

The bars are drawn by the shipped `internship_workspace.chain_figures`. What is
here is the table formatting the section would otherwise carry inline, plus one
measurement the installed packages do not own: coverage with the radius
calibrated at the density it is applied to.

That one is deliberately small and deliberately visible. It exists because the
published protocol calibrates its radius on a *thinned* reference draw while
real events query the full cloud, which inflates every published coverage
number; the version below removes that asymmetry and is shown beside the
published one rather than in place of it.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors

from internship_workspace.z8_domain_pca import CLASS_ORDER


def density_consistent_coverage(real_scores, real_labels, synthetic_scores,
                                synthetic_labels, *, quantile, dimensions=16):
    """Coverage with the radius calibrated at the density it is applied to.

    The radius is the `quantile` of each synthetic event's distance to its
    nearest *other* synthetic event over the whole cloud — leave-one-out, hence
    `n_neighbors=2` and column 1 — and real events then query that same full
    cloud. Calibration and query therefore happen at one density.
    """
    real = np.asarray(real_scores)[:, :dimensions]
    synthetic = np.asarray(synthetic_scores)[:, :dimensions]
    output = {}
    for class_name in CLASS_ORDER:
        cloud = synthetic[synthetic_labels == class_name]
        queries = real[real_labels == class_name]
        self_distance = NearestNeighbors(n_neighbors=2).fit(cloud).kneighbors(
            cloud, return_distance=True
        )[0][:, 1]
        radius = float(np.quantile(self_distance, quantile))
        distance = NearestNeighbors(n_neighbors=1).fit(cloud).kneighbors(
            queries, return_distance=True
        )[0][:, 0]
        output[class_name] = {
            "radius": radius,
            "real_within_radius_fraction": float(np.mean(distance <= radius)),
            "synthetic_events": int(cloud.shape[0]),
            "reference_events": int(np.sum(real_labels == class_name)),
        }
    return output


def show_density_comparison(coverage, consistent, order):
    """The published chain and the density-consistent one, class by class."""
    print(f"{'class':>6} {'density':>9}   " + "   ".join(f"{label:>16}" for label in order))
    for class_name in CLASS_ORDER:
        thinned_row = "   ".join(
            f"{100 * coverage[label][class_name]['real_within_radius_fraction']:15.1f} %"
            for label in order
        )
        consistent_row = "   ".join(
            f"{100 * consistent[label][class_name]['real_within_radius_fraction']:15.1f} %"
            for label in order
        )
        cell = consistent[order[-1]][class_name]
        ratio = cell["synthetic_events"] / cell["reference_events"]
        print(f"{class_name:>6} {'thinned':>9}   {thinned_row}")
        print(f"{class_name:>6} {'full':>9}   {consistent_row}    "
              f"(cloud {ratio:.1f}× denser)")


def show_step_sizes(coverage, consistent, order):
    """How much each added piece of physics is worth, under both calibrations."""
    print(f"{'class':>6}   {'step':>28}   {'thinned':>9}   {'consistent':>11}")
    for class_name in CLASS_ORDER:
        for previous, current in zip(order, order[1:]):
            thin = 100 * (coverage[current][class_name]["real_within_radius_fraction"]
                          - coverage[previous][class_name]["real_within_radius_fraction"])
            cons = 100 * (consistent[current][class_name]["real_within_radius_fraction"]
                          - consistent[previous][class_name]["real_within_radius_fraction"])
            print(f"{class_name:>6}   {previous + ' → ' + current:>28}   "
                  f"{thin:+8.1f} pt   {cons:+10.1f} pt")


def show_radius_and_median(coverage, order):
    """Radius against raw distance: broadening looks different from relocating."""
    print(f"{'class':>6}  {'condition':>15}  {'radius':>8}  {'real→synth median':>18}")
    for class_name in CLASS_ORDER:
        for label in order:
            cell = coverage[label][class_name]
            print(f"{class_name:>6}  {label:>15}  {cell['synthetic_self_nn_radius']:8.3f}  "
                  f"{cell['real_to_synthetic_nn_median']:18.3f}")


def show_parameter_versus_morphology(gap):
    """The published support-gap diagnosis, one row per class."""
    print(f"{'class':>6} {'parameter space':>17} {'morphology':>12} "
          f"{'covered in parameters, not in morphology':>42}")
    for class_name in CLASS_ORDER:
        cell = gap["classes"][class_name]
        print(f"{class_name:>6} {100 * cell['parameter_coverage']:16.1f} % "
              f"{100 * cell['morphology_coverage']:11.1f} % "
              f"{100 * cell['parameter_covered_morphology_uncovered']:41.1f} %")


def show_domain_auc(ablation):
    """The paired white-noise / real-noise domain classifier, one row per class."""
    print(f"{'class':>6} {'white noise AUC':>17} {'real noise AUC':>16}")
    for class_name in CLASS_ORDER:
        white = ablation["conditions"]["white_noise_control"][class_name]
        real_noise = ablation["conditions"]["real_noise_candidate"][class_name]
        print(f"{class_name:>6} {white['domain_classifier_auc_mean']:17.3f} "
              f"{real_noise['domain_classifier_auc_mean']:16.3f}")
    print("\nAUC 0.5 would mean a classifier cannot tell real from synthetic at all")


# --- twins ---
"""Plot and plumbing helpers for the `twins` section.

Every drawing helper accepts `ax=None` or `axes=None` so a cell can redraw into
an existing figure without rebuilding it. Beside them live the loaders, the
three-space measurement loop, the table printers and the evidence payload —
everything the section needs in order to run but that a reader does not need on
screen in order to believe the result. No science is invented here: the
distances come from the shipped `z8_*` modules, the stored spaces come from the
runs that own them, and the section keeps the assertions.

The envelope drawn on the waveform panels is the shipped descriptor envelope,
`z8_domain_pca.morphology_features(...)[:, :64]`, not a locally smoothed Hilbert
magnitude. Drawing anything else would show the reader a shape the distance does
not use.
"""

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

from internship_workspace.chain_figures import CLASS_COLOUR
from internship_workspace.z8_coverage import reflect_crop
from internship_workspace.z8_domain_pca import (
    CLASS_ORDER,
    fit_synthetic_pca,
    morphology_features,
)
from internship_workspace.z8_support_diagnosis import nearest_indices, parameter_matrix

TWIN_INK = "#243447"
TWIN_MUTED = "#6b7a8d"
TWIN_FAINT = "#eef2f7"
TWIN_ALERT = "#c2402a"
TWIN_SAMPLING_KHZ = 2_000.0
TWIN_CORE = slice(1536, 2560)


# --------------------------------------------------------------------------
# loading and integrity
# --------------------------------------------------------------------------


def load_twin_spaces(pca_run, latent_run, v4_events):
    """The two stored spaces, unpacked and checked to index one gallery.

    Neither space is refitted: the morphology scores come from the run that
    fitted them and the embeddings from the run that extracted them. The two
    integrity checks are preconditions rather than results — if the runs do not
    agree on the gallery, every distance below would be comparing different
    events and the section must not run at all.
    """
    pca = np.load(pca_run / "pca_scores.npz", allow_pickle=True)
    embeddings = np.load(latent_run / "embeddings.npz", allow_pickle=True)
    sample_id = pca["synthetic_sample_id"].astype(str)
    if not np.array_equal(sample_id, embeddings["synthetic_sample_id"].astype(str)):
        raise ValueError("the two runs do not index the same synthetic gallery")
    if [row["sample_id"] for row in v4_events] != sample_id.tolist():
        raise ValueError("the v4 event table is not aligned with the stored scores")
    real_id = pca["real_event_id"].astype(str)
    return SimpleNamespace(
        sample_id=sample_id,
        synth_class=pca["synthetic_class"].astype(str),
        real_id=real_id,
        real_class=pca["real_class"].astype(str),
        morph_synth=np.asarray(pca["baseline"], dtype=np.float64),
        morph_v5=np.asarray(pca["candidate"], dtype=np.float64),
        morph_real=np.asarray(pca["real"], dtype=np.float64),
        real_row={event: index for index, event in enumerate(real_id)},
        row_of_sample={sample: index for index, sample in enumerate(sample_id)},
        latent_real=embeddings["real"],
        latent_synthetic=embeddings["synthetic"],
        case_order=embeddings["real_case_id"].astype(str),
    )


def verify_crop_convention(anchors, signal_root, stored_raw, *, window, core):
    """Check that a freshly cut crop equals the centre of the stored window.

    The spaces read different amounts of signal, which will matter later; this
    makes sure they at least start from the same instant. Raises on the first
    anchor where the two conventions disagree.
    """
    for index, anchor in enumerate(anchors):
        source = np.load(
            signal_root / anchor["source_signal_relative_path"], allow_pickle=False
        ).astype(np.float32)
        fresh = reflect_crop(source, float(anchor["center_norm"]) * source.size, window)
        if not np.array_equal(fresh, stored_raw[index, core]):
            raise ValueError(f"crop convention drifted for {anchor['case_id']}")
    return len(anchors)


# --------------------------------------------------------------------------
# the three-space measurement
# --------------------------------------------------------------------------


def measure_twin_spaces(spaces, anchors, by_case, v4_events, hybrid_choice):
    """One anchor at a time, its whole same-class gallery on three rulers.

    For each anchor: euclidean distance over the 16 stored morphology
    components, cosine distance over the L2-normalised 512-D latent, and
    euclidean distance over the four fitted generation coordinates standardised
    on that gallery. The parameter ranking is rebuilt here only to get a full
    distance vector, so it is asserted to select the identical event as the
    shipped `nearest_indices` — the reported ranking cannot drift away from the
    shipped selection.

    Returns one dict per anchor, carrying both raw distance vectors so later
    cells can re-plot without recomputing.
    """
    latent = np.asarray(spaces.latent_synthetic, dtype=np.float64)
    latent /= np.linalg.norm(latent, axis=1, keepdims=True)
    latent_real = np.asarray(spaces.latent_real, dtype=np.float64)
    latent_real /= np.linalg.norm(latent_real, axis=1, keepdims=True)

    parameters = parameter_matrix(v4_events, real=False)
    parameters_real = parameter_matrix(anchors, real=True)
    anchor_position = {row["case_id"]: index for index, row in enumerate(anchors)}

    measurements = []
    for position, case_id in enumerate(spaces.case_order):
        anchor = by_case[case_id]
        class_name = anchor["analysis_class"]
        gallery = np.flatnonzero(spaces.synth_class == class_name)
        anchor_row = anchor_position[case_id]

        morphology = np.linalg.norm(
            spaces.morph_synth[gallery] - spaces.morph_real[spaces.real_row[anchor["event_id"]]],
            axis=1,
        )
        cosine = 1.0 - latent[gallery] @ latent_real[position]
        scaler = StandardScaler().fit(parameters[gallery])
        parameter = np.linalg.norm(
            scaler.transform(parameters[gallery])
            - scaler.transform(parameters_real[anchor_row][None, :]),
            axis=1,
        )
        shipped_pick, _ = nearest_indices(
            parameters_real[anchor_row][None, :], parameters[gallery]
        )
        if int(np.argmin(parameter)) != int(shipped_pick[0]):
            raise ValueError(f"parameter ranking disagrees with nearest_indices for {case_id}")

        morphology_order = np.argsort(morphology, kind="stable")
        cosine_order = np.argsort(cosine, kind="stable")
        morphology_rank = np.empty(len(gallery), dtype=np.int64)
        morphology_rank[morphology_order] = np.arange(1, len(gallery) + 1)
        cosine_rank = np.empty(len(gallery), dtype=np.int64)
        cosine_rank[cosine_order] = np.arange(1, len(gallery) + 1)

        latent_pick = int(cosine_order[0])
        morphology_pick = int(morphology_order[0])
        parameter_pick = int(shipped_pick[0])
        hybrid_pick = int(
            np.flatnonzero(gallery == spaces.row_of_sample[hybrid_choice[case_id]])[0]
        )

        measurements.append(
            {
                "case_id": case_id,
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
                "top5_overlap": int(
                    len(set(cosine_order[:5].tolist()) & set(morphology_order[:5].tolist()))
                ),
                "latent_row": int(gallery[latent_pick]),
                "morphology_row": int(gallery[morphology_pick]),
                "parameter_row": int(gallery[parameter_pick]),
                "parameter_cosine_distance": float(cosine[parameter_pick]),
                "cosine_vector": cosine,
                "morphology_vector": morphology,
            }
        )
    return measurements


def show_space_agreement(measurements):
    """How often the three spaces agree, and how well their orderings correlate."""
    total = len(measurements)
    latent_agreement = sum(row["latent_is_nearest"] for row in measurements)
    parameter_agreement = sum(row["parameter_is_nearest"] for row in measurements)
    rho_latent = np.asarray([row["rho_latent"] for row in measurements])
    rho_parameter = np.asarray([row["rho_parameter"] for row in measurements])
    print(f"anchors measured                             {total}")
    print(f"latent twin is also the morphology NN        {latent_agreement}/{total}"
          f"  ({100.0 * latent_agreement / total:.1f} %)")
    print(f"parameter twin is also the morphology NN     {parameter_agreement}/{total}"
          f"  ({100.0 * parameter_agreement / total:.1f} %)")
    print(f"mean shared candidates in the two top-5      "
          f"{np.mean([row['top5_overlap'] for row in measurements]):.3f} of 5")
    print(f"Spearman rho, latent vs morphology           median {np.median(rho_latent):.3f}, "
          f"range {rho_latent.min():.3f} to {rho_latent.max():.3f}")
    print(f"Spearman rho, parameters vs morphology       median {np.median(rho_parameter):.3f}, "
          f"range {rho_parameter.min():.3f} to {rho_parameter.max():.3f}")


def show_rho_by_class(measurements):
    """The rank correlations again, split by class — 2 µm is the extreme."""
    for class_name in CLASS_ORDER:
        in_class = [row for row in measurements if row["class_name"] == class_name]
        latent_values = np.asarray([row["rho_latent"] for row in in_class])
        parameter_values = np.asarray([row["rho_parameter"] for row in in_class])
        print(f"{class_name:>5}  rho median: parameters {np.median(parameter_values):6.3f}   "
              f"latent {np.median(latent_values):6.3f}   "
              f"latent range {latent_values.min():6.3f} to {latent_values.max():6.3f}   "
              f"gallery {in_class[0]['gallery']:,}")


def show_chosen_twin_distances(measurements):
    """Where each selection rule's twin lands on the morphology ruler."""
    gallery_size = np.asarray([row["gallery"] for row in measurements])
    nearest = np.asarray([row["nearest_morphology_distance"] for row in measurements])
    reverse_rank = np.asarray([row["morphology_cosine_rank"] for row in measurements])
    latent_distance = np.asarray([row["latent_morphology_distance"] for row in measurements])

    print(f"{'morphology nearest neighbour':<34} median distance {np.median(nearest):5.2f}")
    for label, distance_key, rank_key in (
        ("twin chosen by fitted parameters", "parameter_morphology_distance",
         "parameter_morphology_rank"),
        ("twin chosen by physical + rerank", "hybrid_morphology_distance",
         "hybrid_morphology_rank"),
        ("twin chosen by latent cosine", "latent_morphology_distance",
         "latent_morphology_rank"),
    ):
        distances = np.asarray([row[distance_key] for row in measurements])
        ranks = np.asarray([row[rank_key] for row in measurements])
        print(f"{label:<34} median distance {np.median(distances):5.2f}   "
              f"median morphology rank {np.median(ranks):>8,.0f}   "
              f"({100.0 * np.median(ranks / gallery_size):4.1f}th percentile)   "
              f"is the nearest neighbour {sum(ranks == 1)}/{len(ranks)}")
    print()
    print(f"the morphology twin, seen by the latent: median cosine rank "
          f"{np.median(reverse_rank):,.0f} "
          f"({100.0 * np.median(reverse_rank / gallery_size):.1f}th percentile) — "
          f"the disagreement is symmetric, not a defect of one space")
    print(f"latent choice / nearest distance ratio: median "
          f"{np.median(latent_distance / nearest):.2f}x")


# --------------------------------------------------------------------------
# case selection and figure payloads
# --------------------------------------------------------------------------


def median_morphology_cases(measurements):
    """Per class, the anchor whose morphology twin sits at the class median.

    Selected by rule rather than by eye, so the panels show a typical case and
    not the best one.
    """
    chosen = {}
    for class_name in CLASS_ORDER:
        in_class = sorted(
            (row for row in measurements if row["class_name"] == class_name),
            key=lambda row: row["nearest_morphology_distance"],
        )
        chosen[class_name] = in_class[len(in_class) // 2]
    return chosen


def sharpest_parameter_cases(measurements):
    """Per class, the anchor whose parameter twin lands worst in morphology."""
    return {
        class_name: max(
            (row for row in measurements if row["class_name"] == class_name),
            key=lambda row: row["parameter_morphology_rank"],
        )
        for class_name in CLASS_ORDER
    }


def twin_pair_panels(cases, spaces, real_raw, v4_signals):
    """Payload for `plot_twin_pairs`: one real anchor beside two candidates."""
    panels = []
    for class_name in CLASS_ORDER:
        chosen = cases[class_name]
        position = int(np.flatnonzero(spaces.case_order == chosen["case_id"])[0])
        panels.append(
            {
                "case_id": chosen["case_id"],
                "class_name": class_name,
                "colour": CLASS_COLOUR[class_name],
                "real": np.asarray(real_raw[position], dtype=np.float32),
                "morphology": {
                    "signal": np.asarray(
                        v4_signals[chosen["morphology_row"]], dtype=np.float32
                    ),
                    "sample_id": spaces.sample_id[chosen["morphology_row"]],
                    "morphology_distance": chosen["nearest_morphology_distance"],
                    "cosine_distance": chosen["morphology_cosine_distance"],
                },
                "latent": {
                    "signal": np.asarray(
                        v4_signals[chosen["latent_row"]], dtype=np.float32
                    ),
                    "sample_id": spaces.sample_id[chosen["latent_row"]],
                    "morphology_distance": chosen["latent_morphology_distance"],
                    "cosine_distance": chosen["latent_cosine_distance"],
                },
            }
        )
    return panels


def show_twin_pair_distances(cases):
    """The two columns of the figure above, as numbers."""
    for class_name in CLASS_ORDER:
        chosen = cases[class_name]
        print(f"{chosen['case_id']:<18} morphology twin at "
              f"{chosen['nearest_morphology_distance']:5.2f} "
              f"(cosine rank {chosen['morphology_cosine_rank']:>6,})   "
              f"latent twin at {chosen['latent_morphology_distance']:5.2f} "
              f"(morphology rank {chosen['latent_morphology_rank']:>6,} "
              f"of {chosen['gallery']:,})")


def _parameter_text(row, *, real):
    amplitude = float(row["particles2snr_amplitude" if real else "amplitude_p0"])
    frequency = float(row["frequency_hz"]) / 1000.0 if real else float(row["frequency_khz"])
    return (f"P0 {amplitude:.3g} · f {frequency:.1f} kHz · "
            f"tau {float(row['tau_ms']):.3f} ms · SNR {float(row['snr_db']):.1f} dB")


def parameter_triptych_panels(cases, spaces, by_case, v4_events, real_raw, v4_signals):
    """Payload for `plot_parameter_triptych`: anchor, parameter twin, shape twin."""
    panels = []
    for class_name in CLASS_ORDER:
        sharpest = cases[class_name]
        position = int(np.flatnonzero(spaces.case_order == sharpest["case_id"])[0])
        panels.append(
            {
                "case_id": sharpest["case_id"],
                "class_name": class_name,
                "colour": CLASS_COLOUR[class_name],
                "gallery": sharpest["gallery"],
                "real": np.asarray(real_raw[position], dtype=np.float32),
                "real_parameters": _parameter_text(by_case[sharpest["case_id"]], real=True),
                "parameter": {
                    "signal": np.asarray(
                        v4_signals[sharpest["parameter_row"]], dtype=np.float32
                    ),
                    "sample_id": spaces.sample_id[sharpest["parameter_row"]],
                    "parameters": _parameter_text(
                        v4_events[sharpest["parameter_row"]], real=False
                    ),
                    "morphology_distance": sharpest["parameter_morphology_distance"],
                    "morphology_rank": sharpest["parameter_morphology_rank"],
                },
                "morphology": {
                    "signal": np.asarray(
                        v4_signals[sharpest["morphology_row"]], dtype=np.float32
                    ),
                    "sample_id": spaces.sample_id[sharpest["morphology_row"]],
                    "parameters": _parameter_text(
                        v4_events[sharpest["morphology_row"]], real=False
                    ),
                    "morphology_distance": sharpest["nearest_morphology_distance"],
                    "morphology_rank": 1,
                },
            }
        )
    return panels


def show_parameter_ranks(measurements, cases):
    """Class medians beside the extreme shown, so one is not read as the other."""
    for class_name in CLASS_ORDER:
        in_class = [row for row in measurements if row["class_name"] == class_name]
        ranks = np.asarray([row["parameter_morphology_rank"] for row in in_class])
        print(f"{class_name:>5}  parameter-nearest lands at morphology rank: "
              f"median {np.median(ranks):>8,.0f}, worst {ranks.max():>8,} "
              f"of {in_class[0]['gallery']:,}   (shown: {cases[class_name]['case_id']})")


def metric_scatter_sample(row):
    """Payload for `plot_metric_scatter`: one anchor's gallery on two rulers."""
    return {
        "case_id": row["case_id"],
        "gallery": row["gallery"],
        "cosine": row["cosine_vector"],
        "morphology": row["morphology_vector"],
        "rho": row["rho_latent"],
        "latent_choice": {
            "cosine": row["latent_cosine_distance"],
            "morphology": row["latent_morphology_distance"],
        },
        "parameter_choice": {
            "cosine": row["parameter_cosine_distance"],
            "morphology": row["parameter_morphology_distance"],
        },
        "morphology_choice": {
            "cosine": row["morphology_cosine_distance"],
            "morphology": row["nearest_morphology_distance"],
        },
    }


# --------------------------------------------------------------------------
# the human-validated arm
# --------------------------------------------------------------------------


def validated_pairs(human_rows, spaces, by_case):
    """Recompute, in the stored v5 basis, every pair a reviewer accepted.

    The reviewer's verdict is read as published — this notebook cannot re-run a
    blind review — and only the distances are recomputed.
    """
    rows = []
    for case_row in human_rows:
        if case_row["paired_asymmetry_success"] != "True":
            continue
        anchor = by_case[case_row["case_id"]]
        gallery = np.flatnonzero(spaces.synth_class == anchor["analysis_class"])
        distances = np.linalg.norm(
            spaces.morph_v5[gallery] - spaces.morph_real[spaces.real_row[anchor["event_id"]]],
            axis=1,
        )
        order = np.argsort(distances, kind="stable")
        ranks = np.empty(len(gallery), dtype=np.int64)
        ranks[order] = np.arange(1, len(gallery) + 1)
        picked = int(
            np.flatnonzero(
                gallery
                == spaces.row_of_sample[case_row["paired_asymmetry_synthetic_sample_id"]]
            )[0]
        )
        rows.append(
            {
                "case_id": case_row["case_id"],
                "class_name": anchor["analysis_class"],
                "gallery": int(len(gallery)),
                "selected_sample_id": case_row["paired_asymmetry_synthetic_sample_id"],
                "selected_distance": float(distances[picked]),
                "selected_rank": int(ranks[picked]),
                "selected_row": int(gallery[picked]),
                "nearest_distance": float(distances[order[0]]),
                "nearest_sample_id": spaces.sample_id[gallery[order[0]]],
                "nearest_row": int(gallery[order[0]]),
                "distances": distances,
            }
        )
    rows.sort(key=lambda row: -row["selected_distance"])
    return rows


def show_validated_pairs(rows):
    """Every accepted pair, its distance, and the distance nobody was shown."""
    print(f"{'case':<18}{'class':>6}{'d(accepted)':>13}{'d(nearest)':>12}"
          f"{'rank of accepted':>22}")
    for row in rows:
        print(f"{row['case_id']:<18}{row['class_name']:>6}{row['selected_distance']:13.2f}"
              f"{row['nearest_distance']:12.2f}{row['selected_rank']:>15,} "
              f"/{row['gallery']:>6,}")


def counterexample_panel(row, spaces, real_raw, v5_signals):
    """Payload for `plot_validated_counterexample`: the accepted pair and the NN."""
    position = int(np.flatnonzero(spaces.case_order == row["case_id"])[0])
    return {
        "case_id": row["case_id"],
        "colour": CLASS_COLOUR[row["class_name"]],
        "gallery": row["gallery"],
        "gallery_distances": row["distances"],
        "real": np.asarray(real_raw[position], dtype=np.float32),
        "selected": {
            "label": "accepted by the reviewer",
            "signal": np.asarray(v5_signals[row["selected_row"]], dtype=np.float32),
            "sample_id": row["selected_sample_id"],
            "morphology_distance": row["selected_distance"],
            "morphology_rank": row["selected_rank"],
        },
        "nearest": {
            "label": "nearest in the morphology space, never shown",
            "signal": np.asarray(v5_signals[row["nearest_row"]], dtype=np.float32),
            "sample_id": row["nearest_sample_id"],
            "morphology_distance": row["nearest_distance"],
            "morphology_rank": 1,
        },
    }


# --------------------------------------------------------------------------
# the window control
# --------------------------------------------------------------------------


def refit_window_bases(v4_signals, real_raw, synth_class, *, core, per_class, seed,
                       batch_size=512):
    """Refit the morphology space twice, on 1024 samples and on 4096.

    Uses the shipped `fit_synthetic_pca` with the same balanced draw for both,
    so the only difference between the two bases is how much signal each reads.
    """
    bases = {}
    for name, window in (("core", core), ("full", slice(None))):
        block = np.concatenate([
            morphology_features(np.asarray(v4_signals[start : start + batch_size, window]))
            for start in range(0, len(v4_signals), batch_size)
        ])
        scaler, pca_model, _ = fit_synthetic_pca(
            block, synth_class, per_class=per_class, seed=seed
        )
        anchors = morphology_features(real_raw[:, window])
        bases[name] = {
            "synthetic": pca_model.transform(scaler.transform(block)),
            "real": pca_model.transform(scaler.transform(anchors)),
            "dimension": int(block.shape[1]),
        }
    return bases


def window_control(bases, by_id, spaces):
    """Separate the window effect from the representation effect.

    Three comparisons, one per anchor: refit against stored basis (does the
    refit reproduce the space?), 1024 against 4096 inside the morphology space
    (how much does the window alone change the ordering?), and the latent
    against both windows (does matching the window make them agree?).
    """
    control = {
        "refit vs stored\n(both 1024)": [],
        "1024 vs 4096\n(morphology only)": [],
        "stored 1024\nvs latent": [],
        "refit 4096\nvs latent": [],
    }
    matched = []
    for position, case_id in enumerate(spaces.case_order):
        row = by_id[case_id]
        gallery = np.flatnonzero(spaces.synth_class == row["class_name"])
        stored = row["morphology_vector"]
        cosine = row["cosine_vector"]
        core = np.linalg.norm(
            bases["core"]["synthetic"][gallery] - bases["core"]["real"][position], axis=1
        )
        full = np.linalg.norm(
            bases["full"]["synthetic"][gallery] - bases["full"]["real"][position], axis=1
        )
        control["refit vs stored\n(both 1024)"].append(float(spearmanr(stored, core).statistic))
        control["1024 vs 4096\n(morphology only)"].append(float(spearmanr(core, full).statistic))
        control["stored 1024\nvs latent"].append(float(spearmanr(stored, cosine).statistic))
        control["refit 4096\nvs latent"].append(float(spearmanr(full, cosine).statistic))

        order = np.argsort(full, kind="stable")
        ranks = np.empty(len(gallery), dtype=np.int64)
        ranks[order] = np.arange(1, len(gallery) + 1)
        pick = int(np.argmin(cosine))
        matched.append(
            {
                "same_choice": bool(pick == int(order[0])),
                "rank": int(ranks[pick]),
                "distance": float(full[pick]),
                "nearest": float(full[order[0]]),
            }
        )
    return control, matched


def show_window_control(control, matched):
    """The four medians the prose reads, and agreement at the top."""
    for label, values in control.items():
        print(f"  {label.replace(chr(10), ' '):<36} rho median "
              f"{np.median(values):5.2f}")
    print("with the window matched at 4096 samples:")
    print(f"  latent twin is also the morphology NN     "
          f"{sum(row['same_choice'] for row in matched)}/{len(matched)}")
    print(f"  median morphology rank of the latent twin "
          f"{np.median([row['rank'] for row in matched]):,.0f}")
    print(f"  median distance {np.median([row['distance'] for row in matched]):.2f} "
          f"against a nearest neighbour at "
          f"{np.median([row['nearest'] for row in matched]):.2f}")


# --------------------------------------------------------------------------
# evidence
# --------------------------------------------------------------------------


def twin_evidence_payload(measurements, cases, control, matched, validated, headline,
                          *, datasets, runs, window_seed, window_per_class):
    """`emit_run` metrics and provenance for the three-space disagreement."""
    rho_latent = np.asarray([row["rho_latent"] for row in measurements])
    rho_parameter = np.asarray([row["rho_parameter"] for row in measurements])
    accepted = np.asarray([row["selected_distance"] for row in validated])
    accepted_nn = np.asarray([row["nearest_distance"] for row in validated])

    def _median(key):
        return float(np.median([row[key] for row in measurements]))

    def _by_class(key):
        return {
            name: float(np.median([
                row[key] for row in measurements if row["class_name"] == name
            ]))
            for name in CLASS_ORDER
        }

    return {
        "metrics": {
            "schema_version": 1,
            "analysis": (
                "twin selection compared across the fitted-parameter, morphology "
                "and Conv1D-GAP latent spaces on the frozen 30-anchor panel"
            ),
            "anchors": len(measurements),
            "gallery_per_class": {
                name: int(cases[name]["gallery"]) for name in CLASS_ORDER
            },
            "rank_correlation_against_morphology": {
                "latent": {
                    "median": float(np.median(rho_latent)),
                    "mean": float(rho_latent.mean()),
                    "minimum": float(rho_latent.min()),
                    "maximum": float(rho_latent.max()),
                    "by_class": _by_class("rho_latent"),
                },
                "parameters": {
                    "median": float(np.median(rho_parameter)),
                    "mean": float(rho_parameter.mean()),
                    "minimum": float(rho_parameter.min()),
                    "maximum": float(rho_parameter.max()),
                    "by_class": _by_class("rho_parameter"),
                },
            },
            "is_morphology_nearest_neighbour": {
                "latent": int(sum(row["latent_is_nearest"] for row in measurements)),
                "parameters": int(sum(row["parameter_is_nearest"] for row in measurements)),
                "hybrid": int(sum(row["hybrid_morphology_rank"] == 1 for row in measurements)),
                "total": len(measurements),
            },
            "mean_top5_overlap_latent_morphology": float(
                np.mean([row["top5_overlap"] for row in measurements])
            ),
            "morphology_distance_median": {
                "nearest_neighbour": _median("nearest_morphology_distance"),
                "parameter_choice": _median("parameter_morphology_distance"),
                "hybrid_choice": _median("hybrid_morphology_distance"),
                "latent_choice": _median("latent_morphology_distance"),
            },
            "morphology_rank_median": {
                "parameter_choice": _median("parameter_morphology_rank"),
                "hybrid_choice": _median("hybrid_morphology_rank"),
                "latent_choice": _median("latent_morphology_rank"),
            },
            "latent_rank_median_of_morphology_choice": _median("morphology_cosine_rank"),
            "window_control_rho_median": {
                label.replace("\n", " "): float(np.median(values))
                for label, values in control.items()
            },
            "matched_window_4096": {
                "is_morphology_nearest_neighbour": int(
                    sum(row["same_choice"] for row in matched)
                ),
                "median_rank": float(np.median([row["rank"] for row in matched])),
                "median_distance": float(np.median([row["distance"] for row in matched])),
                "median_nearest_distance": float(
                    np.median([row["nearest"] for row in matched])
                ),
            },
            "human_validated_pairs": {
                "count": len(validated),
                "median_accepted_distance": float(np.median(accepted)),
                "median_nearest_distance": float(np.median(accepted_nn)),
                "headline_case": headline["case_id"],
                "headline_accepted_distance": headline["selected_distance"],
                "headline_nearest_distance": headline["nearest_distance"],
                "headline_accepted_rank": headline["selected_rank"],
            },
        },
        "provenance": {
            "datasets": datasets,
            "inputs": {
                "morphology_scores": f"{runs['pca']}/pca_scores.npz",
                "latent_embeddings": f"{runs['latent']}/embeddings.npz",
                "latent_neighbours": f"{runs['latent']}/neighbors.csv",
                "hybrid_neighbours": f"{runs['hybrid']}/neighbors.csv",
                "human_decisions": f"{runs['human']}/human_paired_twin_cases.csv",
            },
            "parameters": {
                "anchors": "the frozen 30 physical validation anchors, reused unchanged",
                "gallery": "same-class synthetic events of the v4 baseline arm",
                "parameter_metric": "euclidean over the four standardised fitted coordinates",
                "morphology_metric": "euclidean over 16 PCA components of the 101-D descriptor",
                "latent_metric": "cosine over the L2-normalised 512-D penultimate activation",
                "window_control_seed": window_seed,
                "window_control_per_class": window_per_class,
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
    }


# --------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------


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


# --- retrieval ---
"""Helpers for the `retrieval` section.

The figure itself is shipped (`chain_figures.draw_space_comparison`). What lives
here is the bookkeeping the argument does not need on screen: selecting the
gallery and query rows out of the round-trip table, running the shipped
split-local retrieval, formatting the two published tables, and the evidence
payload.

The one function worth reading is `parent_retrieval_scores`. Every space in this
section goes through it, unchanged, so parity between the experiments is
structural rather than a matter of discipline — a space cannot accidentally be
scored under a different averaging, a different gallery scope or a different k.
"""

import csv
from types import SimpleNamespace

import numpy as np

from internship_workspace import notebook_evidence
from internship_workspace.equation_latent_audit import average_normalized_views
from internship_workspace.z8_domain_pca import morphology_features
from internship_workspace.z8_parent_retrieval import macro_rate, split_local_retrieval

GALLERY_FIELDS = ("source_event_id", "source_group", "class_name", "split")


def roundtrip_index(rows, *, origin, variant):
    """Split the round-trip table into its real gallery and its synthetic queries.

    One row is one stored waveform. `real_gallery` rows are the measured events
    a query has to find again; `synthetic_query` rows are the regenerations, one
    per view per event.
    """
    gallery = np.asarray([
        index for index, row in enumerate(rows)
        if row["record_kind"] == "real_gallery" and row["annotation_origin"] == origin
    ])
    queries = np.asarray([
        index for index, row in enumerate(rows)
        if row["record_kind"] == "synthetic_query"
        and row["annotation_origin"] == origin
        and row["variant"] == variant
    ])
    return SimpleNamespace(
        gallery=gallery,
        queries=queries,
        meta={
            field: np.asarray([rows[index][field] for index in gallery], dtype=str)
            for field in GALLERY_FIELDS
        },
        query_event_ids=np.asarray(
            [rows[index]["source_event_id"] for index in queries], dtype=str
        ),
    )


def parent_retrieval_scores(query_embeddings, gallery_embeddings, index, *, top_k=50):
    """Split-local retrieval of each event's own parent, in any space.

    The eight views of an event are L2-normalised and averaged before the search,
    because phase is not identifiable; galleries are split-local, so a train
    query is only ever compared to train parents. Returns Recall@5 as an
    unweighted class mean, and the median relative rank of the exact parent.
    """
    averaged, averaged_ids = average_normalized_views(
        query_embeddings, index.query_event_ids
    )
    retrieval = split_local_retrieval(
        query_embeddings=averaged,
        query_ids=averaged_ids,
        gallery_embeddings=gallery_embeddings,
        gallery_ids=index.meta["source_event_id"],
        gallery_groups=index.meta["source_group"],
        gallery_classes=index.meta["class_name"],
        gallery_splits=index.meta["split"],
        top_k=top_k,
    )
    recall5 = (retrieval.result.rank <= 5).astype(np.float64)
    return {
        "events": int(len(retrieval.result.rank)),
        "recall_at_5_macro": float(macro_rate(recall5, retrieval.classes)),
        "q50_relative_rank": float(np.median(retrieval.result.rank_percentile)),
    }


def decimated_morphology(signals, indices, *, sampling_frequency_hz, reference_window,
                         batch_size=512):
    """The shape descriptor of the encoder's own input, on the coverage band grid.

    `reference_window` is not cosmetic. The descriptor's band grid is invariant
    to window *length* at a fixed rate but not to the rate itself: it is defined
    as the in-band bins a reference window produces. These signals are decimated
    by 8, so the reference window has to shrink by 8 as well for the 37 bands to
    land on the same physical frequencies as the coverage chain.
    """
    return np.concatenate([
        morphology_features(
            np.asarray(signals[indices[start : start + batch_size]]),
            sampling_frequency_hz=sampling_frequency_hz,
            reference_window=reference_window,
        )
        for start in range(0, len(indices), batch_size)
    ])


def published_condition_row(path, condition_id):
    """One row of a published `condition_metrics.csv`, by condition id."""
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["condition_id"] == condition_id:
                return row
    raise KeyError(f"no condition {condition_id} in {path}")


def show_retrieval_board(board, events):
    """The published conditions, bracketed by the ceiling and the floor.

    The ceiling is a benign real query — feed a real event back as its own query
    — and the floor is what shuffled parents give. Eleven percent means nothing
    without both.
    """
    print(f"{'condition':<44} {'events':>7} {'Recall@5':>10} {'q50':>8}")
    print(f"{'benign real query (ceiling)':<44} {events:>7} {100.0:9.1f} % {0.0:7.1f} %")
    for row in sorted(board["retrieval_rows"], key=lambda item: -item["recall_at_5_macro"]):
        print(f"{row['condition']:<44} {row['events']:>7} "
              f"{100 * row['recall_at_5_macro']:9.1f} % "
              f"{100 * row['q50_relative_rank']:7.1f} %")
    print(f"{'shuffled parents (floor)':<44} {'—':>7} {2.0:9.1f} % {50.0:7.1f} %")


def show_space_scores(spaces):
    """The three spaces on one line each, in the figure's order."""
    for name, values, _ in spaces:
        print(f"{name.replace(chr(10), ' '):<38} "
              f"R@5 {100 * values['recall_at_5_macro']:5.1f} %   "
              f"q50 {100 * values['q50_relative_rank']:5.1f} %")


def retrieval_evidence_payload(scores, dimensions, *, origin, variant, views_per_event,
                               events, source_groups, chance_q50, reproduces, datasets,
                               checkpoint, top_k):
    """`emit_run` metrics and provenance for the neighbour-space comparison."""
    return {
        "metrics": {
            "schema_version": 1,
            "analysis": "parent-retrieval-across-neighbour-spaces",
            "population": {
                "origin": origin,
                "condition": "tau_sigma",
                "events": events,
                "views_per_event": views_per_event,
                "source_groups": source_groups,
            },
            "spaces": {
                "conv1dgap_latent": {
                    "dimensions": dimensions["latent"], **scores["latent"]
                },
                "morphology_full": {
                    "dimensions": dimensions["morphology"], **scores["morphology"]
                },
                "morphology_pca16": {"dimensions": 16, **scores["morphology_pca"]},
            },
            "chance_q50_relative_rank": chance_q50,
            "reproduces": reproduces,
        },
        "provenance": {
            "datasets": datasets,
            "inputs": {"checkpoint_sha256": notebook_evidence.sha256_file(checkpoint)},
            "parameters": {
                "origin": origin,
                "variant": variant,
                "views_per_event": views_per_event,
                "top_k": top_k,
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
    }


# %% [markdown]
# ## How a signal becomes a point
#
# Part I ended with a generator that can produce new events. Nothing so far says
# whether what it produces resembles what the instrument records, and the word
# *resembles* is the problem: every test in this notebook is a **distance**, and
# a distance is only as meaningful as the space it is measured in. Comparing two
# waveforms sample by sample is not an option, because the arrival phase of a
# crossing is not identifiable from the crossing itself — the same burst shifted
# by a few microseconds would come out far away from itself. So the first thing
# to settle is not a result but a ruler.
#
# The descriptor is that ruler, and it is deliberately **phase-insensitive**. It
# is built in four steps, all inside
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
# Why sixteen, and why that band, are basis questions; notebook 3 answers them.
#
# Resolving the datasets, cropping and describing them is the same setup the
# coverage step needs, so it lives once in `internship_workspace.chain_data` —
# imported here, and pinned by `tests/test_chain_data.py` to reproduce the
# published coverage chain with deviation exactly zero.

# %%
from internship_workspace import chain_data, chain_figures  # noqa: E402
from internship_workspace.z8_domain_pca import morphology_features  # noqa: E402

registry = chain_data.Registry(workspace)
real = chain_data.load_real_events(registry, window=1024)  # sealed test rows raise

for key in registry.provenance():
    print(f"dataset            {key}")
print(f"real events        {len(real.rows)}  " + "  ".join(
    f"{name} {int((real.labels == name).sum())}" for name in CLASS_ORDER))
print(f"crop window        {real.cores.shape[1]} samples "
      f"= {1000 * real.cores.shape[1] / SAMPLING_HZ:.3f} ms")
print(f"descriptor         {real.features.shape[1]}-D "
      f"(64 envelope + {real.features.shape[1] - 64} spectral)")

# %% [markdown]
# ### One event, three views
#
# The same event as a waveform and as the two halves of its descriptor — the
# only way "distance in the morphology space" can mean anything to a reader.

# %%
example = int(np.flatnonzero(real.labels == "4um")[0])
chain_figures.draw_event_views(
    real.cores[example],
    morphology_features(real.cores[example][None, :])[0],
    sampling_frequency_hz=SAMPLING_HZ,
    colour=CLASS_COLOUR["4um"],
    title=f"{real.rows[example]['event_id']} · 4um",
)

# %% [markdown]
# ### The window is narrower than the events it describes
#
# One property of that ruler is worth measuring before anything is measured with
# it. The descriptor window is fixed at 1 024 samples; the events are not. Putting
# both on one axis is the first thing this notebook was written to make visible,
# and it appears on no slide.

# %%
widths = support_widths_of(real)  # end_sample - start_sample, one per event
chain_figures.draw_support_against_window(
    widths, window=real.cores.shape[1], candidate_window=4096
)
print(f"support width      median {np.median(widths):.0f}  "
      f"p90 {np.percentile(widths, 90):.0f}  max {widths.max():.0f} samples")
print(f"wider than the {real.cores.shape[1]}-sample window: "
      f"{100 * np.mean(widths > real.cores.shape[1]):.1f} %")

# %% [markdown]
# Almost every event is wider than the window that describes it. What that costs
# is not obvious from the histogram alone — a window can be narrow and still
# capture what matters, and the descriptor is built to be scale-free precisely so
# that the tails carry little. Notebook 3 sweeps the window and answers with
# numbers, including one that contradicts the repair everyone expected.
#
# This measurement belongs to no existing run — it is the notebook's own — so it
# is emitted as evidence. That only happens under `workspace notebooks execute`;
# a live kernel prints the refusal and moves on.

# %%
audit = window_audit_payload(
    real,
    candidate_window=4096,
    datasets=registry.provenance(),
    sampling_frequency_hz=SAMPLING_HZ,
)
try:
    emitted = notebook_evidence.emit_run(
        workspace,
        section="window-audit",
        metrics=audit["metrics"],
        provenance=audit["provenance"],
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
# **Where that leaves us.** Every event, real or synthetic, is now a point in one
# fixed 16-dimensional space, and two events being close there is a statement
# about shape rather than about arrival time. That is exactly what was missing
# to ask the question the whole simulation effort exists for: the real events
# form a cloud in this space, the generator produces another one, and *do the two
# clouds occupy the same region* is now a measurement rather than an opinion.


# %% [markdown]
# ## Does the synthetic cloud cover the real one?
#
# A ruler is not a verdict. The previous step fixed the space; it said nothing
# about whether the generator's output lands where the instrument's output lands.
# That is the question the whole simulation effort exists to answer, and in this
# space it finally has an operational form. For each class:
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
# Three conditions, each adding one piece of physics — white noise only, then a
# real noise carrier, then a fifth asymmetry coordinate — all in **one shared PCA
# basis** with **one shared synthetic draw**, which is what makes them comparable
# rather than merely similar. The pairing is checked event by event, so a
# difference between two columns is the generator change and nothing else.

# %%
QUANTILE = 0.80

conditions = chain_data.load_conditions(registry, window=1024)  # pairing checked
basis = chain_data.pooled_basis(conditions, real.labels, seed=chain_data.CHAIN_SEED)
coverage = chain_data.coverage_chain(basis, real, conditions, quantile=QUANTILE)

for label, key in chain_data.CONDITION_ORDER:
    print(f"{label:>15}   {key}")
print(f"synthetic events   {len(conditions.labels)} per condition, paired one to one")
print(f"shared draw seed   {chain_data.CHAIN_SEED}   radius quantile "
      f"q{int(100 * QUANTILE)}   window 1024 samples")
print(f"PCA  16 components explain {basis.explained_variance:.4f} of the variance, "
      f"PC1+PC2 {basis.explained_variance_pc1_pc2:.4f}")
for class_name, values in chain_data.chain_table(coverage, conditions.order).items():
    print(f"{class_name:>5}   " + "  →  ".join(f"{value:.1f} %" for value in values))

# %% [markdown]
# ### Reproduction check
#
# Every number above must equal the published run to the last decimal. A non-zero
# deviation is a reproduction bug to fix before anything else — not a discovery.

# %%
reference = published("particle-z8-v2-coverage-conditions-q80-r1")
deviation = max(
    [abs(coverage[label][class_name][field]
         - reference["conditions"][label]["classes"][class_name][field])
     for label in conditions.order
     for class_name in CLASS_ORDER
     for field in ("real_within_radius_fraction", "synthetic_self_nn_radius",
                   "real_to_synthetic_nn_median")]
    + [abs(basis.explained_variance - reference["pca"]["explained_variance_16"]),
       abs(basis.explained_variance_pc1_pc2
           - reference["pca"]["explained_variance_pc1_pc2"])]
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
# The dashed line at 80 % marks the quantile the radius was built from, and it
# is a **reading aid, not a baseline**. It is tempting to say that a bar above
# it means real events are better covered than the synthetic cloud covers
# itself — the deck said exactly that, and it is wrong. The radius is calibrated
# on a *thinned* reference draw (one synthetic event per real event) while real
# events query the full cloud, 16 to 25 times denser, and a nearest-neighbour
# distance shrinks as density grows. What these bars support is the *ordering*
# of the three conditions, which is immune to that bias because it is the same
# in every column.

# %%
chain_figures.draw_coverage_chain(coverage, conditions.order, quantile=QUANTILE)

# %% [markdown]
# ### The density-consistent chain
#
# Since that bias is a real defect and not a presentational one, here is the
# chain measured without it: the radius is calibrated on the **full** synthetic
# cloud, leaving each event out of its own neighbour search, and real events
# query that same full cloud. Calibration and query then happen at one density,
# so the 80 % line becomes a baseline that means something.
#
# This is the measurement the redo should adopt. It is shown beside the
# published one rather than replacing it, because every published number in the
# deck and in the runs uses the thinned calibration.

# %%
real_scores = basis.project(real.features)
consistent = {
    label: density_consistent_coverage(
        real_scores, real.labels,
        basis.project(conditions.features[label]), conditions.labels,
        quantile=QUANTILE,
    )
    for label in conditions.order
}
show_density_comparison(coverage, consistent, conditions.order)

# %% [markdown]
# Two things to read off it. The absolute numbers fall a long way — the thinned
# calibration was inflating them — and the **ordering of the three conditions
# survives**, which is what the chain is actually for. But it does not survive
# untouched: check the size of each step, because that is where the bias was
# hiding.

# %%
show_step_sizes(coverage, consistent, conditions.order)

# %% [markdown]
# ### Broadening, not relocating
#
# A fair objection: a generator can raise coverage by simply spreading out, and a
# per-condition radius would partly hide that. The radii and the raw
# real-to-synthetic distances answer it directly. If the cloud had *moved* to sit
# on the real events, the median distance would drop; if it merely **broadened**,
# the median stays put while the radius grows.

# %%
show_radius_and_median(coverage, conditions.order)

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
# variability improved twice; proximity improved once. Retrieval, at the end of
# this notebook, is the test that proximity has to pass, and it is much stricter
# than either.
#
# ### The other half of the story: the parameter space
#
# The deck's sharpest point lives in the gap between two coverage numbers. Run
# the same measurement in the space of **fitted parameters** rather than
# waveform morphology, and the picture inverts.

# %%
show_parameter_versus_morphology(
    published("particle-z8-v2-synthetic-support-gap-diagnosis-q80-r1")
)

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
# its morphology column (64.1 / 58.6 / 63.2 %) is not the same number as the
# chain's white-noise column (60.8 / 56.5 / 54.5 %) even though both describe the
# same idea. Notebook 3 measures how much two bases can disagree.
#
# ### Can a classifier still tell the two apart?
#
# The strongest objection to any coverage claim is that a cloud can swallow
# another by being large, not by being right. Coverage cannot answer that; a
# discriminator can. This is the deck's only domain-separability measurement, and
# it is the one number that could have refuted the noise-carrier step.

# %%
show_domain_auc(published("particle-z8-v2-full-real-noise-analysis-r1"))

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
#
# **Where that leaves us.** Coverage and separability are both *population*
# statements: they describe where a cloud sits and how hard two clouds are to
# tell apart, and every one of them can be satisfied by a generator that gets the
# distribution right and every individual wrong. Nothing measured so far would
# notice that. So the next question drops from the population to the single
# event: take one real event, and point at the one synthetic event that could
# pass for it.


# %% [markdown]
# ## Twins
#
# Coverage answered a question about populations, and populations are the easy
# case: a generator that gets the distribution right and every individual wrong
# would pass everything measured so far. A **twin** is the individual test —
# a synthetic event paired with the real event it most resembles, so that a human
# can look at both and say whether the fake could pass.
#
# Except that "most resembles" is not defined yet, and that is the real subject
# of this section. Three spaces are available in this chain, all three already
# implemented and fitted by the runs this notebook reads, and no two of them
# agree about which synthetic event is the closest. Quantifying that
# disagreement is the deliverable here.
#
# The three spaces, defined once:
#
# - **Parameter space.** The four fitted generation coordinates of an event:
#   log amplitude P0, drive frequency in kHz, log decay time τ in ms, and SNR in
#   dB. Standardised on the synthetic gallery, distance euclidean. Implemented
#   in `internship_workspace.z8_support_diagnosis` (`parameter_matrix`,
#   `nearest_indices`). This is the space the generator itself is written in.
# - **Morphology space.** The descriptor of the previous two sections: each
#   1024-sample window (0.512 ms at 2 MHz) becomes a 101-number shape descriptor,
#   standardised and reduced to 16 principal components; distance is euclidean
#   over those 16 axes. Fitted in the run `particle-z8-v2-paired-asymmetry-pca-r2`,
#   which stores the scores read here.
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
from sklearn.neighbors import NearestNeighbors  # noqa: E402

from internship_workspace.z8_twin_analysis import read_csv_rows, top_k_cosine  # noqa: E402

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

twin_real_events = read_csv_rows(twin_events_root / "events.csv")
if any(row["split"] == "test" for row in twin_real_events):
    raise PermissionError("sealed test rows are forbidden in the twin section")

twin_anchors = read_csv_rows(run_dir(TWIN_LATENT_RUN) / "anchors.csv")
if any(row["split"] != "val" for row in twin_anchors):
    raise PermissionError("twin anchors must be validation rows only")
twin_by_case = {row["case_id"]: row for row in twin_anchors}

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
# 1024-sample crop describe the same instant, and raises on the first anchor
# where they do not.

# %%
twin_core = slice(1536, 2560)
twin_real_raw = np.load(
    run_dir(TWIN_LATENT_RUN) / "real_signals_raw_4096.npy", allow_pickle=False
)
twin_checked = verify_crop_convention(
    twin_anchors, twin_signal_root, twin_real_raw, window=1024, core=twin_core
)
print(f"crop convention verified on {twin_checked}/{len(twin_anchors)} anchors "
      f"(reflect_crop 1024 == centre of the stored 4096 window)")

twin_v4_signals = np.load(twin_v4_root / "signals_raw_4096.npy", mmap_mode="r", allow_pickle=False)
twin_v5_signals = np.load(twin_v5_root / "signals_raw_4096.npy", mmap_mode="r", allow_pickle=False)
twin_v4_events = read_csv_rows(twin_v4_root / "events.csv")
print(f"synthetic gallery {twin_v4_signals.shape[0]:,} events x {twin_v4_signals.shape[1]} samples")

# %% [markdown]
# ### Reproducing the two stored spaces before using either
#
# Neither fitted space is rebuilt from scratch here. Both are read from the runs
# that own them — `load_twin_spaces` also refuses to continue if the two runs do
# not index the same synthetic gallery — and each is then checked against a
# number that run published. If a reproduction drifts, the cells below fail
# rather than quietly measuring something else.
#
# The latent side is reproduced by calling the shipped retrieval,
# `z8_twin_analysis.top_k_cosine`, on the stored embeddings and asserting that
# it returns the published top-1 cosine distances *and* the published sample
# identifiers. The morphology side is reproduced by recomputing the published
# median real-to-nearest-synthetic distance per class from the stored PCA
# scores.

# %%
twin_spaces = load_twin_spaces(run_dir(TWIN_PCA_RUN), run_dir(TWIN_LATENT_RUN), twin_v4_events)

twin_top_index, twin_top_distance = top_k_cosine(
    twin_spaces.latent_real,
    twin_spaces.latent_synthetic,
    twin_spaces.sample_id,
    eligible_indices=[
        np.flatnonzero(twin_spaces.synth_class == twin_by_case[case]["analysis_class"])
        for case in twin_spaces.case_order
    ],
)
twin_got_latent = {
    "mean": float(twin_top_distance[:, 0].mean()),
    "median": float(np.median(twin_top_distance[:, 0])),
    "minimum": float(twin_top_distance[:, 0].min()),
    "maximum": float(twin_top_distance[:, 0].max()),
}
for key, want in published(TWIN_LATENT_RUN, "retrieval_summary.json")[
    "primary_top1_cosine_distance"
].items():
    assert abs(twin_got_latent[key] - want) < 1e-12, f"latent reproduction drifted on {key}"

twin_published_neighbours = {
    row["case_id"]: row["synthetic_sample_id"]
    for row in read_csv_rows(run_dir(TWIN_LATENT_RUN) / "neighbors.csv")
    if row["rank"] == "1" and row["scope"] == "same_class_primary"
}
for position, case_id in enumerate(twin_spaces.case_order):
    got = twin_spaces.sample_id[twin_top_index[position, 0]]
    assert got == twin_published_neighbours[case_id], f"latent twin changed for {case_id}"
print(f"reproduces {TWIN_LATENT_RUN} exactly: median top-1 cosine "
      f"{twin_got_latent['median']:.6f}, and all 30 selected sample IDs match")

# %%
twin_published_pca = published(TWIN_PCA_RUN)
for twin_class in CLASS_ORDER:
    twin_got = float(np.median(
        NearestNeighbors(n_neighbors=1)
        .fit(twin_spaces.morph_synth[twin_spaces.synth_class == twin_class])
        .kneighbors(twin_spaces.morph_real[twin_spaces.real_class == twin_class], return_distance=True)[0][:, 0]
    ))
    twin_want = twin_published_pca["baseline"][twin_class]["real_to_synthetic_nn_median"]
    assert abs(twin_got - twin_want) < 1e-12, f"morphology reproduction drifted by {abs(twin_got - twin_want):.3e}"
print(f"reproduces {TWIN_PCA_RUN} exactly: median real-to-nearest-synthetic distance "
      f"{[round(twin_published_pca['baseline'][name]['real_to_synthetic_nn_median'], 3) for name in CLASS_ORDER]} "
      f"for {list(CLASS_ORDER)}")

# %% [markdown]
# ### The retrieval, in all three spaces at once
#
# `measure_twin_spaces` runs the same search three times for every anchor —
# euclidean on standardised fitted parameters, euclidean in the morphology
# space, cosine in the latent — over the anchor's entire same-class gallery, and
# records where each choice lands on the *other* rulers. Three quantities per
# anchor: the rank of each space's chosen twin in the morphology ordering; the
# distance of each choice on both rulers; and the Spearman rank correlation
# between distance vectors across the whole same-class gallery — a single number
# for "do these two metrics order the candidates the same way", where 1.0 means
# identical ordering and 0.0 means no relationship at all.
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
twin_hybrid_choice = {
    row["case_id"]: row["synthetic_sample_id"]
    for row in read_csv_rows(run_dir(TWIN_HYBRID_RUN) / "neighbors.csv")
    if row["rank"] == "1"
}
twin_measurements = measure_twin_spaces(
    twin_spaces, twin_anchors, twin_by_case, twin_v4_events, twin_hybrid_choice
)
twin_by_id = {row["case_id"]: row for row in twin_measurements}
show_space_agreement(twin_measurements)

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
twin_median_case = median_morphology_cases(twin_measurements)
plot_twin_pairs(twin_pair_panels(twin_median_case, twin_spaces, twin_real_raw, twin_v4_signals))
plt.show()
show_twin_pair_distances(twin_median_case)

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
twin_sharpest = sharpest_parameter_cases(twin_measurements)
assert twin_sharpest["4um"]["case_id"] == "physical-4um-02", "the narrated 4 µm counterexample moved"
assert twin_sharpest["4um"]["parameter_morphology_rank"] == 31_793, "the narrated 4 µm rank moved"

plot_parameter_triptych(
    parameter_triptych_panels(
        twin_sharpest, twin_spaces, twin_by_case, twin_v4_events, twin_real_raw, twin_v4_signals
    )
)
plt.show()
show_parameter_ranks(twin_measurements, twin_sharpest)

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
plot_metric_scatter(metric_scatter_sample(twin_median_case["4um"]))
plt.show()

# %%
plot_rho_strip(
    [
        {"label": "fitted parameters\nvs morphology",
         "values": [row["rho_parameter"] for row in twin_measurements],
         "colours": [row["colour"] for row in twin_measurements]},
        {"label": "Conv1D-GAP latent\nvs morphology",
         "values": [row["rho_latent"] for row in twin_measurements],
         "colours": [row["colour"] for row in twin_measurements]},
    ],
    title="One rank correlation per anchor, over its full same-class gallery",
    ylabel="Spearman rho against the morphology ordering",
    figsize=(8.4, 4.8),
)
plt.show()
show_rho_by_class(twin_measurements)

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
show_chosen_twin_distances(twin_measurements)

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
twin_validated = validated_pairs(
    read_csv_rows(run_dir(TWIN_HUMAN_RUN) / "human_paired_twin_cases.csv"),
    twin_spaces,
    twin_by_case,
)
show_validated_pairs(twin_validated)

twin_headline = next(row for row in twin_validated if row["case_id"] == "physical-2um-01")
print()
print("carried claim: accepted twin at 9.7, true nearest neighbour at 4.1")
print(f"measured     : accepted twin at {twin_headline['selected_distance']:.2f}, "
      f"true nearest neighbour at {twin_headline['nearest_distance']:.2f} "
      f"({twin_headline['case_id']}, rank {twin_headline['selected_rank']:,} "
      f"of {twin_headline['gallery']:,})")
assert round(twin_headline["selected_distance"], 1) == 9.7
assert round(twin_headline["nearest_distance"], 1) == 4.1
print("the carried pair reproduces to the digit")

twin_accepted = np.asarray([row["selected_distance"] for row in twin_validated])
twin_accepted_nn = np.asarray([row["nearest_distance"] for row in twin_validated])
print(f"and it is not the exception: across all {len(twin_validated)} accepted pairs the median "
      f"accepted distance is {np.median(twin_accepted):.2f} against a median nearest-neighbour "
      f"distance of {np.median(twin_accepted_nn):.2f}")

# %%
plot_validated_counterexample(
    counterexample_panel(twin_headline, twin_spaces, twin_real_raw, twin_v5_signals)
)
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
twin_bases = refit_window_bases(
    twin_v4_signals, twin_real_raw, twin_spaces.synth_class,
    core=twin_core, per_class=3660, seed=20260724,
)
print(f"refitted both morphology bases in {time.time() - twin_start:.1f} s "
      f"({twin_bases['core']['dimension']}-D core descriptor, "
      f"{twin_bases['full']['dimension']}-D full-window descriptor)")

twin_control, twin_matched = window_control(twin_bases, twin_by_id, twin_spaces)
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
show_window_control(twin_control, twin_matched)

# %% [markdown]
# The control splits the effect cleanly. The refit reproduces the stored basis
# almost perfectly (rho 0.99), so nothing here depends on a private
# reimplementation. Widening the morphology window from 1024 to 4096 samples
# changes the ordering substantially on its own (rho 0.67 between the two
# windows of the *same* descriptor) — a real and slightly uncomfortable result,
# since it means the morphology space's verdict is itself window-dependent.
# Matching the window does move the latent closer to the morphology space, from
# rho 0.33 to 0.48, which is about a fifth of the distance left to agreement.
# Part of the disagreement is window; most of it is representation.
#
# Agreement at the top does not recover at all: with the window matched, the
# latent's twin is still never the morphology nearest neighbour, 0 of 30, and
# still sits at a median 7.49 where the nearest neighbour is at 2.72. The window
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
# That mechanism is testable, and the last section of this notebook tests it on
# a task with a known right answer, where the morphology space beats the latent
# outright. The present section supplies the mechanism rather than the number: a
# metric that collapses within-class differences cannot retrieve an individual,
# and a twin is an individual.
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
twin_payload = twin_evidence_payload(
    twin_measurements, twin_median_case, twin_control, twin_matched,
    twin_validated, twin_headline,
    datasets=dataset_provenance(),
    runs={"pca": TWIN_PCA_RUN, "latent": TWIN_LATENT_RUN,
          "hybrid": TWIN_HYBRID_RUN, "human": TWIN_HUMAN_RUN},
    window_seed=20260724,
    window_per_class=3660,
)
try:
    twin_emitted = notebook_evidence.emit_run(
        workspace,
        section="twins-space-disagreement",
        metrics=twin_payload["metrics"],
        provenance=twin_payload["provenance"],
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
# **Where that leaves us.** Three spaces, thirty anchors, and never once the same
# answer — but notice what this section could *not* do: say which space was
# right. A twin has no ground truth. "Most resembles" is a judgment, and a
# judgment is exactly what three metrics are allowed to disagree about.
#
# So the last question in this notebook removes the judgment. Take a real event,
# fit the generator's parameters to it, regenerate from those parameters alone,
# and ask the regenerated event to find the event it came from among all the
# others. There is now exactly one correct answer, known by construction, and a
# space either ranks it near the top or it does not.


# %% [markdown]
# ## Can a regenerated event find its own parent?
#
# The twins section ended without a verdict, and not by accident: a twin has no
# ground truth. Three spaces each named a different synthetic event as the
# closest match, and nothing in that setup can say which one was right, because
# "most resembles" is a judgment rather than a fact.
#
# This last question removes the judgment. Take a real event, fit the
# generator's parameters to it, regenerate a synthetic event from those
# parameters alone, and then ask the regenerated event to find the event it came
# from among all the others. There is now exactly **one** correct answer, known
# by construction. Coverage asked whether the synthetic cloud *reaches* the real
# one; this asks something far stricter, event by event. A generator can cover a
# cloud by being broad; it can only pass this test by being accurate.
#
# Two numbers, both read against a floor and a ceiling:
#
# - **Recall@5** — how often the exact parent lands in the top five;
# - **q50** — the median *relative* rank of the parent, where 0 % is first and
#   ≈50 % is what chance gives.
#
# Each event is regenerated in eight views, because phase is not identifiable,
# and the views are averaged after normalisation. Galleries are **split-local**:
# a train query is only ever compared to train parents. Both conventions live in
# `parent_retrieval_scores`, through which every space below is scored
# unchanged, so a difference between two rows can only be the space.

# %%
import torch  # noqa: E402,F401  (imported for the frozen classifier)
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from internship_workspace import chain_figures  # noqa: E402
from internship_workspace.equation_latent_audit import (  # noqa: E402
    extract_penultimate_embeddings,
    load_candidate,
    load_frozen_classifier,
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
roundtrip = roundtrip_index(roundtrip_rows, origin=ORIGIN, variant=SIGMA_VARIANT)
assert roundtrip.queries.size == roundtrip.gallery.size * VIEWS_PER_EVENT

print(f"gallery {roundtrip.gallery.size} real events · queries {roundtrip.queries.size} views "
      f"· sources {len(set(roundtrip.meta['source_group']))}")

# %% [markdown]
# ### The published space — the frozen Conv1D-GAP latent
#
# The deck's numbers come from the 512-D penultimate activation of a frozen
# Conv1D-GAP-**L** classifier, compared by cosine distance.

# %%
model = load_frozen_classifier(CHECKPOINT)
latent_gallery = extract_penultimate_embeddings(
    model, np.asarray(roundtrip_signals[roundtrip.gallery])
)
latent_queries = extract_penultimate_embeddings(
    model, np.asarray(roundtrip_signals[roundtrip.queries])
)
latent_score = parent_retrieval_scores(latent_queries, latent_gallery, roundtrip)
print(json.dumps(latent_score, indent=2))

# %%
reference_row = published_condition_row(
    run_dir("particle-z8-v2-exact-parent-retrieval-r1") / "condition_metrics.csv",
    f"{ORIGIN}_tau_sigma",
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
show_retrieval_board(
    published("ssl-v3-v16-retrieval-and-ranges-r5", "board_values.json"),
    latent_score["events"],
)

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
#
# ### The same retrieval, in the morphology space
#
# That 11 % is a property of the generator *and* of the space it was measured
# in, and the twins section already showed those two are not separable by
# assumption. The deck uses two different spaces for one idea of "neighbour":
# the learned latent here, the morphology descriptor for coverage. Standardising
# on one is a decision; *what it costs* is a number, and this is it.
#
# Everything is held fixed — same events, same eight views, same averaging, same
# split-local galleries, same scoring function. Only the space changes. The
# descriptor is computed on the very signals the encoder sees (a 4 096 crop,
# decimated by 8 to 512 points, so an effective 250 kHz), which isolates the
# space from the preprocessing. The reference window shrinks by the same factor
# of 8, so the 37 spectral bands land on the same physical frequencies as the
# coverage chain — 250 kHz / 128 is 2 MHz / 1024, and the grid was verified
# identical.

# %%
DECIMATED_HZ = SAMPLING_HZ / 8
DECIMATED_REFERENCE_WINDOW = 128

morphology_gallery = decimated_morphology(
    roundtrip_signals, roundtrip.gallery,
    sampling_frequency_hz=DECIMATED_HZ, reference_window=DECIMATED_REFERENCE_WINDOW,
)
morphology_queries = decimated_morphology(
    roundtrip_signals, roundtrip.queries,
    sampling_frequency_hz=DECIMATED_HZ, reference_window=DECIMATED_REFERENCE_WINDOW,
)
morphology_score = parent_retrieval_scores(morphology_queries, morphology_gallery, roundtrip)
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


morphology_pca_score = parent_retrieval_scores(
    reduce_morphology(morphology_queries), reduce_morphology(morphology_gallery), roundtrip
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
show_space_scores(SPACES)

# %% [markdown]
# **The physics-grounded space wins, and not narrowly**: Recall@5 rises from
# 11.0 % to 17.7 % — a 61 % relative gain — and the median parent rank falls
# from 9.5 % to 7.9 %.
#
# Those numbers moved once, and the reason is worth keeping. An earlier run of
# this section measured 23.3 % and 5.2 %, on a descriptor whose spectral half
# had 149 bins because the window's own FFT resolution was used directly. The
# alignment that made the descriptor window-invariant replaced that with the
# fixed 37-band grid the coverage chain uses, and the number fell. So the finer
# spectrum was genuinely carrying identity information, and the alignment costs
# some of it — a real trade, not a rounding change. The figure above reports the
# **aligned** descriptor, because comparing to the coverage section is the whole
# point of the comparison; the unaligned variant is the more discriminative one
# and belongs in the record.
#
# The explanation is not mysterious, and it is the sharpest objection answered in
# advance. It is also the mechanism the twins section predicted. The Conv1D-GAP
# latent is the penultimate layer of a classifier trained to separate **three
# classes**. Discarding within-class variation is precisely what that training
# rewards — and within-class variation is exactly the information needed to tell
# one 4 µm event from another 4 µm event. The learned space is optimal for the
# task it was trained on and structurally wrong for this one.
#
# Two caveats, stated plainly. The descriptor here is computed on the encoder's
# own input — a 4 096 crop decimated to 512 points — not on the 1 024-sample
# window the coverage section uses, so this settles *which space*, not *which
# window*; notebook 3 measures the window separately, and finds it does not
# behave as anyone expected. And the PCA is fitted on the synthetic queries,
# which is the coverage section's convention and never uses parent identity.
#
# A blocker to clear before rerunning the published tool on new data: it carries
# a hard-coded regression guard that aborts unless Recall@5 equals its historical
# z8 value (`analyze_z8_v2_exact_parent_retrieval.py`).

# %%
retrieval_payload = retrieval_evidence_payload(
    scores={"latent": latent_score, "morphology": morphology_score,
            "morphology_pca": morphology_pca_score},
    dimensions={"latent": int(latent_gallery.shape[1]),
                "morphology": int(morphology_gallery.shape[1])},
    origin=ORIGIN,
    variant=SIGMA_VARIANT,
    views_per_event=VIEWS_PER_EVENT,
    events=latent_score["events"],
    source_groups=len(set(roundtrip.meta["source_group"].tolist())),
    chance_q50=CHANCE_Q50,
    reproduces={
        "run_id": "particle-z8-v2-exact-parent-retrieval-r1",
        "condition_id": f"{ORIGIN}_tau_sigma",
        "max_deviation": retrieval_deviation,
    },
    datasets=dataset_provenance(),
    checkpoint=CHECKPOINT,
    top_k=50,
)
try:
    emitted = notebook_evidence.emit_run(
        workspace,
        section="retrieval-space-comparison",
        metrics=retrieval_payload["metrics"],
        provenance=retrieval_payload["provenance"],
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
# **Where that leaves the notebook.** Four steps, and each one answered the
# question the previous one could not. A signal became a point, so distance
# meant something. The clouds turned out to overlap — 85 to 94 % at q80 under
# the published calibration, less than that once the density bias is removed —
# but overlap is a population statement. Twins moved to the individual and found
# that the answer depends on the space, with three metrics never once agreeing.
# Retrieval supplied the ground truth those three metrics lacked, and named a
# winner: the physics-grounded descriptor, at 17.7 % against 11.0 %, with both
# far from the 100 % ceiling.
#
# So the simulator reproduces the *kind* of event well and the *individual* event
# poorly — which is exactly the property a pretraining corpus needs and exactly
# the property a per-event validation would fail on. Notebook 3 trains on it, and
# then measures the choices this evaluation quietly inherited: the window, the
# quantile, the basis.


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
