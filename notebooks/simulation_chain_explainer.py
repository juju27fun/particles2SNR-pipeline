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
import json
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from internship_workspace.config import Workspace
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


def dataset_root(key: str):
    """Resolve `dataset-id@version` through the registry, never by path."""
    dataset_id, _, version = key.rpartition("@")
    return resolve_path(workspace, select_record(workspace, dataset_id, version))


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
# **[à écrire]** Regenerate an event from its fitted parameters, project it, and
# look for its parent among the real events: Recall@5 and the median relative
# rank q50, against chance (≈50 %) and against the ceiling (a benign parent,
# 100 %).
#
# Redo note: the published experiment measures distances in the Conv1D-GAP latent
# with cosine, on a 4 096 → 512 input, while section 2f measures distances in
# morphology with euclidean on 1 024. Same idea of "neighbour", three different
# choices. Alignment **A6** puts both experiments on the morphology space through
# one shared implementation; this section will report what that costs, measured.
#
# A blocker to clear on the way: the retrieval tool carries a hard-coded
# regression guard that aborts unless Recall@5 equals its historical z8 value.
