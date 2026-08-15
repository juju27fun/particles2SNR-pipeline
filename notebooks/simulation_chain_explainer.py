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
# # The simulation chain, end to end
#
# **How do we know that simulated particle events resemble real ones — and how
# closely?** This notebook walks the whole chain in one executable pass, in the
# order the argument is made:
#
# 1. **The latent sweep** — the analytical signal family and its six knobs, and
#    the proof that a trained encoder orders each knob.
# 2. **From events to a generator and back** — fitted parameters, Cholesky
#    correlations, the skew estimator, the 5-D end, the signal space and its
#    construction, neighbour distances at q80, twins, masked learning.
# 3. **Retrieval** — can a regenerated event find its own parent, signal by
#    signal?
#
# Section 0 is a separate notebook: [`mad_detection_explainer`](mad_detection_explainer.py)
# explains how a time series becomes a bounded event in the first place. This
# one starts where that one stops.
#
# ## What this notebook is, and is not
#
# It **imports** every method from the installed packages — `internship_workspace`,
# `particles2snr`, `p3_ssl`. It contains no method of its own. That is the point:
# the notebook and the manifested analysis tools call the same functions, so they
# cannot drift apart. Per `particles2SNR-pipeline/AGENTS.md`, a notebook is never
# manifested evidence — the runs under `artifacts/` remain the record. This is
# where the chain is *explained and audited*, not where it is *proven*.
#
# ## Reading order for the redo (2026-08)
#
# The MAD detector replaces the z8 cascade, so every experiment below is being
# redone. The rule for this first version is **reproduce, do not improve**: each
# cell must return the published number to the last decimal. Alignment fixes come
# after, one at a time, so that every changed digit has a named cause. The plan
# lives in `docs/experiments/2026-08-15/mad-redo-execution-plan.md`.
#
# Sections still to be written are marked **[à écrire]**. They are listed rather
# than omitted so the shape of the argument is visible from the start.

# %%
import csv
import json
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from internship_workspace import notebook_evidence
from internship_workspace.config import Workspace, WorkspaceError
from internship_workspace.datasets import resolve_path, select_record
from internship_workspace.z8_coverage import (
    batched_features,
    load_real_core,
    read_rows,
    reflect_crop,
    shared_class_sample,
    support_coverage,
    validate_pair,
)
from internship_workspace.z8_domain_pca import CLASS_ORDER, morphology_features

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


def published(run_id: str, name: str = "metrics.json"):
    """Read a manifested run's metrics, to check this notebook against it."""
    path = workspace.root / "artifacts/cross-project/analyses" / run_id / name
    return json.loads(path.read_text())


# %% [markdown]
# ## 1. The latent sweep — the equation, the signals, the space
#
# **[à écrire]** The analytical family s(t) and its six knobs (A, f_D, φ, t₀, τ,
# SNR); a gallery of signals as each knob turns; then the PCA of a trained
# encoder's latent, one panel per knob, showing that the encoder *orders* each
# one.
#
# Redo note: the published figure uses Conv1D-GAP-**L**, was produced by calling
# the sweep CLI directly (the launcher's defaults diverge from it), and reached
# the deck as a manual PNG crop with no generating script. The redo trains
# Conv1D-GAP-**S** and produces a tooled run. This is the one place where the
# learned latent keeps a role: the claim *is* a claim about the encoder, so it
# has to live in the encoder's space.

# %% [markdown]
# ## 2a. Fitted physical parameters
#
# **[à écrire]** Each detected event is fitted with four physical parameters —
# P₀, f_D, τ, SNR — plus the waveform asymmetry that section 2c introduces.
# Everything downstream is built on these numbers.
#
# Redo note: **MAD events carry no fitted parameters yet.** The MAD tables hold
# geometry and spectral features only. The fitting pass over MAD events is the
# first real construction site of the redo, not an acquired input.

# %% [markdown]
# ## 2b. Cholesky — the correlations and the delta triangles
#
# **[à écrire]** Class-conditional Pearson correlations between the transformed
# parameters, the Cholesky factor that lets us draw new parameter vectors with
# those correlations, and the delta triangles that check what the generator
# actually realised against what was targeted.

# %% [markdown]
# ## 2c. The skew estimator
#
# **[à écrire]** Envelope asymmetry, its parametric estimator, and the
# source-disjoint injection campaign that calibrates it (R², MAE, sign accuracy
# per class).

# %% [markdown]
# ## 2d. The 5-D end of the Cholesky chain
#
# **[à écrire]** Asymmetry becomes a fifth generated coordinate: observed
# targets, paired generation, and the 5-D delta triangles.
#
# Redo note: the published 5-D target run declared itself *provisional* — 10 µm
# had n = 40 against a pre-declared 50-event gate — and was nevertheless consumed
# to generate v5. MAD yields 880 events at 10 µm, so the gate finally clears on
# its own terms.

# %% [markdown]
# ## 2e. How a signal becomes a point — the morphology space
#
# This is the pivot of the whole argument. Everything that follows measures
# *distances*, and a distance is only as meaningful as the space it lives in.
#
# The descriptor is deliberately **phase-insensitive**: two events with the same
# shape but a different arrival phase should be neighbours. It is built in four
# steps, all inside `internship_workspace.z8_domain_pca.morphology_features`:
#
# 1. crop a fixed window around the event centre, then remove the mean and
#    normalise by RMS — amplitude is a fitted parameter, not a shape;
# 2. take the Hilbert envelope, subtract its 20th-percentile floor, smooth it,
#    and average into **64 bins** — the *when* of the energy;
# 3. take the windowed spectrum, keep the **7–80 kHz** band, log-compress and
#    L2-normalise — the *what* of the energy, currently **37 bins**;
# 4. concatenate into a **101-D** descriptor, standardise, and reduce to
#    **16 principal components** fitted on synthetic events only.

# %%
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

print(f"real events        {len(real_rows)}")
print(f"crop window        {real_cores.shape[1]} samples "
      f"= {1000 * real_cores.shape[1] / SAMPLING_HZ:.3f} ms")
print(f"descriptor         {real_features.shape[1]}-D "
      f"(64 envelope + {real_features.shape[1] - 64} spectral)")

# %% [markdown]
# ### One event, three views
#
# The same event as a waveform, as an envelope, and as a spectrum — the two
# halves of the descriptor side by side with the signal they summarise.

# %%
example_index = int(np.flatnonzero(real_labels == "4um")[0])
example = real_cores[example_index]
example_descriptor = morphology_features(example[None, :])[0]

time_ms = 1000 * np.arange(example.size) / SAMPLING_HZ
frequencies = np.fft.rfftfreq(example.size, d=1.0 / SAMPLING_HZ)
band = (frequencies >= 7_000.0) & (frequencies <= 80_000.0)

figure, axes = plt.subplots(1, 3, figsize=(13, 3.2))
axes[0].plot(time_ms, example, lw=0.6, color="#334155")
axes[0].set(title="waveform", xlabel="ms")
axes[1].plot(example_descriptor[:64], color=CLASS_COLOUR["4um"])
axes[1].set(title="envelope · 64 bins", xlabel="bin")
axes[2].plot(frequencies[band] / 1000, example_descriptor[64:],
             color=CLASS_COLOUR["4um"])
axes[2].set(title="spectrum · 7–80 kHz", xlabel="kHz")
for axis in axes:
    axis.spines[["top", "right"]].set_visible(False)
figure.suptitle(f"{real_rows[example_index]['event_id']} · 4um", y=1.04)
figure.tight_layout()

# %% [markdown]
# ### Audit — the window is narrower than the events it describes
#
# The descriptor window is fixed at 1 024 samples. The events are not. Comparing
# the two on the same axis is the first thing this notebook was written to make
# visible, and it is not visible on any slide.

# %%
z8_widths = np.sort([
    float(row["end_sample"]) - float(row["start_sample"]) for row in real_rows
])
truncated = 100 * np.mean(z8_widths > real_cores.shape[1])

figure, axis = plt.subplots(figsize=(7.5, 3.0))
axis.hist(z8_widths, bins=60, color="#94a3b8", label="z8 event support")
axis.axvline(real_cores.shape[1], color="#dc2626", lw=2,
             label=f"descriptor window ({real_cores.shape[1]})")
axis.axvline(4096, color="#0f766e", lw=2, ls="--", label="4096 (SSL window)")
axis.set(xlabel="samples", ylabel="events",
         title=f"{truncated:.1f} % of events are wider than the window that describes them")
axis.legend(frameon=False)
axis.spines[["top", "right"]].set_visible(False)
figure.tight_layout()

print(f"z8 support width   median {np.median(z8_widths):.0f}  "
      f"p90 {np.percentile(z8_widths, 90):.0f}  max {z8_widths.max():.0f} samples")
print(f"wider than the {real_cores.shape[1]}-sample window: {truncated:.1f} %")

# %% [markdown]
# The fix is scheduled as alignment **A1/A2** — window 4 096 for everything, with
# the spectrum re-binned onto a fixed 37-band grid so that the descriptor stays
# 101-D and comparable to every historical result. It is deliberately *not*
# applied in this version: the numbers below have to match the published run
# first.
#
# This measurement belongs to no existing run — it is the notebook's own finding,
# and the quantitative justification for A1. So this section emits it as
# manifested evidence. That only happens under
# `workspace notebooks execute`; a live kernel prints the refusal and moves on.

# %%
window_audit = {
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
        "median": float(np.median(z8_widths)),
        "p90": float(np.percentile(z8_widths, 90)),
        "p99": float(np.percentile(z8_widths, 99)),
        "max": float(z8_widths.max()),
    },
    "wider_than_descriptor_window_fraction": float(
        np.mean(z8_widths > real_cores.shape[1])
    ),
    "wider_than_candidate_window_fraction": float(np.mean(z8_widths > 4096)),
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
}

try:
    emitted = notebook_evidence.emit_run(
        workspace,
        section="window-audit",
        metrics=window_audit,
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
                "support_width": (
                    "end_sample minus start_sample of the detector annotation"
                ),
                "wider_than_window_fraction": (
                    "fraction of events whose annotated support exceeds the fixed "
                    "descriptor window, so that the descriptor cannot see the event "
                    "in full"
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
# ## 2f. Neighbour distance at q80 — does the synthetic cloud cover the real one?
#
# The question of the whole simulation effort, made measurable. For each class:
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

start = time.time()
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
      f"({time.time() - start:.1f} s)")

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
# The gate of this notebook version: every number above must equal the published
# run, to the last decimal. A non-zero deviation is a reproduction bug to fix
# before anything else — not a discovery.

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
figure, axis = plt.subplots(figsize=(8.5, 3.6))
width = 0.26
positions = np.arange(len(CONDITIONS))
for offset, class_name in enumerate(CLASS_ORDER):
    values = [
        100 * coverage[label][class_name]["real_within_radius_fraction"]
        for label, _ in CONDITIONS
    ]
    bars = axis.bar(positions + (offset - 1) * width, values, width,
                    color=CLASS_COLOUR[class_name], label=class_name)
    axis.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
axis.axhline(100 * QUANTILE, color="#dc2626", ls="--", lw=1.2)
axis.text(len(CONDITIONS) - 0.5, 100 * QUANTILE + 1.5,
          "synthetic self-coverage", color="#dc2626", fontsize=8, ha="right")
axis.set_xticks(positions, [label for label, _ in CONDITIONS])
axis.set(ylabel="real events covered (%)", ylim=(0, 105),
         title=f"Coverage of the real cloud at q{int(100 * QUANTILE)}")
axis.legend(frameon=False, ncol=3)
axis.spines[["top", "right"]].set_visible(False)
figure.tight_layout()

# %% [markdown]
# ## 2g. Twins — the same event, two spaces
#
# **[à écrire]** Paired baseline/candidate events, and the disagreement worth
# keeping in view: a human-validated twin sits at 9.7 in the morphology space
# where the true nearest neighbour is at 4.1 — that pair had been selected by
# cosine distance in the Conv1D-GAP latent. The two spaces do not say the same
# thing, and the redo documents the gap rather than arbitrating it silently.

# %% [markdown]
# ## 2h. Masked learning
#
# **[à écrire]** P25 against CYCLIC25: how each policy chooses what to hide, the
# geometry of the resulting masks, and the reconstruction regimes they produce.
#
# Redo note — **a defect to re-verify first.** The SSL config declares
# `sampling_frequency_hz: 1000000` while the synthetic data is generated at
# 2 MHz, and that value converts `tau_ms` into samples to build the *event mask*.
# The mask therefore covers ≈22 % of the window instead of ≈44 %. CYCLIC25 draws
# its event and background windows from that mask, so the outer half of every
# event has been sampled as background. Whether the published comparison survives
# the correction is exactly what the redo has to establish (alignment A4).

# %% [markdown]
# ## 3. Retrieval — can a regenerated event find its own parent?
#
# Section 2f asked whether the synthetic cloud *covers* the real one. This asks
# something far stricter, event by event: regenerate one event from its own
# fitted parameters, and see whether the original comes back at the top of the
# neighbour list.
#
# Two numbers, both reported against a floor and a ceiling:
#
# - **Recall@5** — how often the exact parent lands in the top five;
# - **q50** — the median *relative* rank of the parent, where 0 % is first and
#   ≈50 % is what chance would give.
#
# Each event is regenerated in eight views (phase is not identifiable), and the
# views are averaged after normalisation. Galleries are **split-local**: a train
# query is only ever compared to train parents. The tool's τ=σ arm is the one
# the deck reports.

# %%
import torch  # noqa: E402  (kept beside the section that needs it)

from internship_workspace.equation_latent_audit import (  # noqa: E402
    average_normalized_views,
    extract_penultimate_embeddings,
    load_candidate,
    load_frozen_classifier,
)
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

    The same function serves both spaces, which is the whole point of alignment
    A6: parity is structural rather than a matter of discipline.
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
# ### 3a. The published space — the frozen Conv1D-GAP latent
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

# %% [markdown]
# ### Reproduction check

# %%
reference_row = next(
    row
    for row in csv.DictReader(
        (
            workspace.root
            / "artifacts/cross-project/runs/particle-z8-v2-exact-parent-retrieval-r1"
            / "condition_metrics.csv"
        ).open()
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
# ### 3b. The same retrieval, in the morphology space
#
# This is the measurement alignment **A6** rests on. The deck currently uses two
# different spaces for the same idea of "neighbour": the learned latent here, the
# morphology descriptor in section 2f. Aligning on one is the decision; *what it
# costs* is a number, and this is it.
#
# Everything is held fixed — same events, same eight views, same averaging, same
# split-local galleries, same `retrieve` function. Only the space changes. The
# descriptor is computed on the very signals the encoder sees (4 096 crop,
# decimated ×8 to 512 points, so an effective 250 kHz), which isolates the space
# from the preprocessing.

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
# A third variant makes the comparison directly transposable to section 2f,
# which does not use the raw descriptor but its 16 principal components, with
# the basis fitted on **synthetic events only**. Here the queries are the
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

# %% [markdown]
# ### What aligning costs, or buys

# %%
CHANCE_Q50 = 0.5
SPACES = [
    ("Conv1D-GAP latent\n512-D", latent_score, "#64748b"),
    ("morphology\nfull descriptor", morphology_score, CLASS_COLOUR["4um"]),
    ("morphology\nPCA(16), as in 2f", morphology_pca_score, CLASS_COLOUR["2um"]),
]

figure, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for axis, key, title, reference in (
    (axes[0], "recall_at_5_macro", "Recall@5 · higher is better", None),
    (axes[1], "q50_relative_rank", "q50 relative rank · lower is better", CHANCE_Q50),
):
    bars = axis.bar(
        [name for name, _, _ in SPACES],
        [100 * values[key] for _, values, _ in SPACES],
        color=[colour for _, _, colour in SPACES],
        width=0.55,
    )
    axis.bar_label(bars, fmt="%.1f %%", fontsize=9, padding=2)
    if reference is not None:
        axis.axhline(100 * reference, color="#dc2626", ls="--", lw=1.2)
        axis.text(2.4, 100 * reference + 1.5, "chance", color="#dc2626",
                  fontsize=8, ha="right")
    axis.set(title=title, ylabel="%")
    axis.tick_params(axis="x", labelsize=8)
    axis.spines[["top", "right"]].set_visible(False)
figure.suptitle(
    f"Same {latent_score['events']} events, same eight views, same split-local "
    "galleries — only the space changes",
    fontsize=10, y=1.06,
)
figure.tight_layout()

for name, values, _ in SPACES:
    print(f"{name.replace(chr(10), ' '):<34} "
          f"R@5 {100 * values['recall_at_5_macro']:5.1f} %   "
          f"q50 {100 * values['q50_relative_rank']:5.1f} %")

# %% [markdown]
# The three bars describe the same 1 494 events under the same protocol, which
# is exactly what the deck could not claim before. Read them with two caveats
# stated plainly: the descriptor here is computed on the **encoder's own input**
# (4 096 crop decimated to 512 points), not on the 1 024-sample window of section
# 2f, and its spectral half is finer because of that sampling rate. So this
# settles *which space*, not *which window* — the window is alignment A1, and it
# is measured separately.
#
# A blocker to clear before rerunning the published tool on MAD: it carries a
# hard-coded regression guard that aborts unless Recall@5 equals its historical
# z8 value (`analyze_z8_v2_exact_parent_retrieval.py`, alignment A12).

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
                    "dimensions": int(latent_gallery.shape[1]),
                    **latent_score,
                },
                "morphology_full": {
                    "dimensions": int(morphology_gallery.shape[1]),
                    **morphology_score,
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
