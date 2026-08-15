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
# # MAD detection — from a time series to a bounded event
#
# **How does a time series become a bounded training example?** — bounded
# meaning we know where the event starts and ends, which is what a
# self-supervised (SSL) model needs as input.
#
# This notebook retraces the yeast event detector
# (`particles2snr.yeast_events`) stage by stage on one reviewed record. The
# detector decides what counts as an event using the **median absolute
# deviation (MAD)** — a measure of how much a signal ordinarily fluctuates
# that a rare large event cannot distort. Section 6 builds it from nothing;
# sections 1–5 are what has to happen before it can be applied. The chain
# follows the frozen
# [v5 explainer deck](../../artifacts/cross-project/presentations/yeast-detector-explainer-v5/render-r1/Yeast_detector_explainer_v5.pdf):
#
# `ISOLATE → LOCALISE → REFERENCE → AGGREGATE → NORMALISE → ACTIVATE → GROUP → CROP`
#
# It explains a method on one record: no performance metric, no biological
# claim. The detector itself only needs a 1-D array — the workspace calls
# below just fetch a traceable example, and the last section runs the same
# chain on a signal built from scratch.

# %%
import csv
import json

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image
from internship_workspace.config import Workspace
from internship_workspace.mad_conv1dgap_training import (
    INPUT_LENGTH,
    RAW_CROP_LENGTH,
    centered_reflect_crop,
    crop_limits,
    prepare_event_signal,
    resolve_registered_dataset,
)

from particles2snr.yeast_events import (
    detect_yeast_events,
    detector_trace,
    event_bounds,
    review_calibrated_detection_config_v1,
)
from particles2snr.yeast_particles2snr_comparison import (
    assert_reference_replay_contract,
    render_particles2snr_failure_plot,
    replay_particles2snr_dual_clean,
    replay_particles2snr_on_synthetic,
)
from particles2snr.yeast_review_records import load_reviewed_event
from particles2snr import yeast_detector_figures as fig

# %%
EVENT_ID = "9459e76ce29342debc90:00"

workspace = Workspace.load()
record = load_reviewed_event(workspace, EVENT_ID)
config = review_calibrated_detection_config_v1()
trace = detector_trace(record.signal, config)
event_start, event_end = int(record.row["event_start"]), int(record.row["event_end"])
center = int(record.row["center_index"])
print(f"{EVENT_ID}: {record.signal.size} samples, "
      f"reviewed support [{event_start}, {event_end}], center {center}")

# %% [markdown]
# One call to `detector_trace` runs the whole detector front-end; every section
# below reads one of its stages instead of recomputing anything.
#
# The configuration is the frozen development preset
# (`review_calibrated_detection_config_v1`):
#
# | Stage | Setting |
# |---|---|
# | Sampling | 2 MHz |
# | Detection band | 7–80 kHz |
# | Filter | Butterworth order 4, zero-phase |
# | STFT | N = 512, overlap 384 → hop H = 128 samples (64 µs) |
# | Smoothing | 3 frames |
# | Activation | z ≥ 3.5 |
# | Boundaries | pad 0.04 ms |
# | Grouping | bridge gaps ≤ 0.128 ms (2 frames) |
# | Qualification | width 0.06–2.0 ms, max z ≥ 12, event concentration ≥ 0.12, ≤ 5 events per trace |
#
# *These values are a development choice validated on a review campaign, not a
# demonstrated optimum.* The last rows name quantities the notebook has not
# built yet — z is constructed in section 6; the table is here as a reference
# to come back to, not as something to read now.

# %% [markdown]
# ## Why energy first — the particles2SNR lesson
#
# The legacy pipeline (particles2SNR) reasons **peak-first**: every local
# Doppler maximum becomes a particle hypothesis, fitted as an arrival time
# t₀ with a decay time τ and turned into the box t₀ ± 2.5τ, then cleaned and
# merged. That representation makes one structural mistake: a
# yeast passage can produce **several exact Doppler crests for one single
# particle**, and the moment each crest becomes a separate hypothesis, the
# information that they co-occurred as one burst of energy is gone.
# Everything downstream is an attempt to reconstruct that lost information
# with heuristics, each carrying thresholds that interact with all the
# others. The five cleaning stages, in order:
#
# | Stage | Deletes a hypothesis when… |
# |---|---|
# | passage-time filter | its decay time τ is outside 0.07–0.65 ms |
# | width filter | its box t₀ ± 2.5τ is outside 0.08–1.5 ms |
# | peak-evidence refinement | its envelope (the outline of the oscillation's amplitude) shows no clear bump |
# | dual-clean | that bump is absent from the *unfiltered* view too |
# | temporal NMS | it overlaps a stronger hypothesis (NMS = non-maximum suppression: among overlapping boxes, keep the strongest) |
#
# The cell below **replays that exact failure** on this notebook's record. The
# replay is contract-checked: it refuses to render if the replayed legacy
# pipeline drifts from the audited stage counts, so the figure stays bound
# to the audited result.

# %%
failure_png = workspace.root / ".cache/notebooks/particles2snr-multi-doppler-failure-en.png"
replay, legacy_filtered = replay_particles2snr_dual_clean(record.signal, record.row)
assert_reference_replay_contract(replay)
render_particles2snr_failure_plot(signal=record.signal, filtered=legacy_filtered,
                                  replay=replay, destination=failure_png, language="en")
Image(filename=str(failure_png), width=980)

# %% [markdown]
# - One human-reviewed passage (green span), two exact Doppler crests — but
#   **five** frequency hypotheses from P0 (the legacy pipeline's first stage,
#   which proposes one particle per spectral maximum).
# - After all five cleaning stages, still **two boxes for one single event**.
#   The last stage should have merged them and did not: their overlap, scored
#   as IoU (intersection over union — shared duration divided by total
#   duration spanned, so 1 = identical boxes, 0 = disjoint), came out at
#   **0.399**, just under the 0.400 required to merge.
#
# **Could a different merge threshold fix this?** This *case*, yes — the miss
# is 0.001 wide. That is precisely the problem, not an unlucky detail. Loosen
# the threshold and genuinely distinct neighbouring events fuse; tighten it
# and multi-Doppler events fragment. No setting is principled, because the
# quantity that would decide it — *do these crests belong to the same burst
# of energy?* — was discarded when the crests became separate hypotheses.
# Tuning relocates the failure; it cannot remove it. The energy-first chain
# never faces the decision: frequencies are summed (section 5) *before* any
# hypothesis exists, and Doppler peaks are measured afterwards as
# descriptors, never as the unit of counting.
#
# Two bounded pieces of development context (beads, not yeast):
#
# - On the frozen 87-event human ledger of the bead development corpus, the
#   MAD v2.1 chain recovered **77/87 confirmed events against 41/87** for Z8v2
#   (the historical peak-first detector), and returned the *right number* of
#   events on all seven frozen two-particle loci. The v2.1 metric is recomputed
#   from its 3 618 supports below; it is not inherited from the 3 749-box v1
#   audit.
#   *Development evidence on one acquisition family, on beads. The audit also
#   reports a localisation score (average precision at IoU 0.5) but only over
#   the 11 events whose human boundaries were reliable — it is not a
#   corpus-wide performance figure.*
# - A peak threshold is also a **hard gate on a proxy**: a persistent
#   nuisance oscillation *is* a Doppler-like peak, so peak-first has to
#   threshold it away — and the same threshold can delete a real but
#   atypical particle, for example an unusually slow one with a weak crest.
#   Section 4 shows why the energy chain does not need that gate at all: a
#   persistent oscillation is *learned into the baseline* and never becomes
#   a candidate in the first place.

# %% [markdown]
# ### The bounded comparison: MAD v2.1 versus P2SNR
#
# “MAD works better” has a precise, narrower meaning here. On the same frozen
# bead audit, MAD v2.1 is the admissible operating point that preserves both
# trace-level cleanliness and two-particle cardinality. It does **not** have the
# highest raw recall: the maximal-recall P2SNR setting reaches 86/87, but it
# activates seven reviewed-empty traces and returns exactly two boxes on only
# one of seven joined-particle loci.
#
# The cell reads the manifested comparison rather than copying numbers into a
# second calculation. It refuses the old MAD v1 row.

# %%
comparison_run = (
    workspace.root
    / "artifacts/cross-project/analysis/particle-p2-noise-tradeoff-evidence-analysis-r5"
)
comparison = json.loads((comparison_run / "tradeoff_comparison.json").read_text())
assert comparison["mad_dataset_id"] == (
    "particles2snr-beads-mad-teacher-detection-development@v2.1"
)
for row in comparison["rows"]:
    print(
        f"{row['label']:<34} rappel={row['recall']:>2}/87  "
        f"propositions={row['global_predictions']:>4}  "
        f"vides={row['verified_empty_active']:>2}/22  "
        f"artefacts={row['artifact_active']}/6  "
        f"deux-particules exacts={row['joined_exact']}/7"
    )

# %%
comparison_figure = (
    workspace.root
    / "artifacts/cross-project/reviews/particle-p2-noise-pareto-closure-result-r6"
    / "assets/capture-01/source.png"
)
Image(filename=str(comparison_figure), width=1100)

# %% [markdown]
# The comparison is retrospective and limited to one acquisition family. The
# human labels establish the number of particles in each joined locus, not
# sample-precise child boundaries. It therefore supports choosing MAD as the
# pseudo-label detector; it is not an independent-generalisation claim.

# %% [markdown]
# ## 1 · The trace and the event support
#
# The record is a 1-D acoustic trace sampled at 2 MHz: one point every 0.5 µs.
# At this rate, the 16 384 samples cover about 8.2 ms.

# %%
fig.plot_trace_overview(record.signal, config, zoom_center=center);

# %% [markdown]
# - The zoom shows individual samples: the signal is smooth at the µs scale.
# - Nothing in the full view says *where* an event is — the eye picks the large
#   excursion, but that is exactly the intuition the detector must formalise.
#
# The event occupies only a fraction of the record. We call its sample
# interval the **temporal support**.
#
# **Where that support comes from, precisely** — this matters for everything
# that follows. The interval below is not a boundary a human drew. It is the
# detector's own proposal, which a reviewer then *confirmed*: the queue
# records `review_event_present = yes`, `review_center_acceptable = yes`,
# `review_full_event_visible = yes`, `reviewer = Julien`. So this record
# certifies **that one complete event is there and the proposal is
# acceptable** — not where its edges independently lie. Nothing in this
# notebook can therefore be read as boundary accuracy.

# %%
fig.plot_event_support(record.signal, config,
                       event_start=event_start, event_end=event_end);

# %% [markdown]
# - The green span is that confirmed support: an interval a reviewer looked
#   at and accepted as containing one complete event.

# %% [markdown]
# ## 2 · ISOLATE — the band-pass filter
#
# Yeast transit signatures live between 7 and 80 kHz. A Butterworth filter of
# order 4 is applied **zero-phase** (`filtfilt`, forward then backward): the
# event must not move in time, otherwise every boundary decided later would be
# systematically shifted.

# %%
fig.plot_bandpass(record.signal, trace);

# %% [markdown]
# - The slow drift and high-frequency hash disappear; the oscillatory burst
#   around 4 ms survives, unmoved.
# - Everything downstream sees only this filtered trace.

# %% [markdown]
# ## 3 · LOCALISE — the short-time Fourier transform
#
# A global FFT would say *what* frequencies the record contains but not *when*.
# The STFT slides a window of N = 512 samples (Hann-weighted) by hops of
# H = 512 − 384 = 128 samples, and takes an FFT per position: frequency content
# *and* time.
#
# **Notation for the rest of the notebook** — every later symbol builds on
# these:
#
# | Symbol | Meaning | Value here |
# |---|---|---|
# | fₛ | sampling frequency | 2 MHz |
# | N | STFT window length | 512 samples (256 µs) |
# | H | hop between windows | 128 samples (64 µs) |
# | m | **frame** index — one window position, one column | 0 … 124 |
# | k | **frequency bin** index — one row | 0 … n_bins−1, spaced fₛ/N ≈ 3.9 kHz |
# | w | Hann window — a smooth taper fading each window's edges to zero, so a signal entering the window does not create an artificial jump | — |
# | X(k, m) | complex STFT value at bin k, frame m | (n_bins × n_frames) matrix |
# | P(k, m) | power \|X(k, m)\|² | same shape, ≥ 0 |

# %%
fig.plot_stft_windows(trace, center=center);

# %% [markdown]
# Each window position becomes one **frame**: one column of the spectrogram
# every 64 µs. Overlap keeps samples near window edges (where the Hann weight
# is small) visible to neighbouring frames.

# %%
fig.plot_spectrogram(trace, annotate=True);

# %% [markdown]
# - Time runs on x, frequency on y, colour is P(k, m) in dB. The dashed white
#   column is **one frame m**; the dotted white row is **one bin k** — every
#   quantity from here on is indexed by these two letters.
# - The event appears as a localised bright patch below ~25 kHz; a single time
#   slice can contain several frequency components at once.

# %% [markdown]
# ## 4 · REFERENCE — the per-frequency Q25 baseline
#
# Each frequency has its own ordinary level: low frequencies carry more power
# than high ones even in pure noise. So the reference cannot be one global
# number — the detector learns it **per row**:
#
# - **B_k = Q25_m(P(k, m))** — for each bin k, the 25th percentile of its own
#   row over time: the level this frequency sits at when nothing happens.
# - **P⁺(k, m) = max(P(k, m) − B_k, 0)** — what remains above that level.
#
# Before the real signal, the whole idea on one toy row of 8 frames, where an
# event brightens 2 frames out of 8:

# %%
row = np.array([4, 5, 3, 6, 40, 40, 5, 4], dtype=float)
baseline_k = np.percentile(row, 25)
excess_row = np.clip(row - baseline_k, 0.0, None)
print(f"P(k, ·) one row over time : {row}")
print(f"B_k = Q25 of that row     : {baseline_k}")
print(f"P+  = max(P - B_k, 0)     : {excess_row}")

# %% [markdown]
# The 25th percentile sits at 4 — on the *quiet* level — even though a quarter
# of the frames are lit by the event. The two event frames keep almost all
# their power in P⁺; the quiet frames keep almost none. Now the same operation
# on every row of the real record:

# %%
fig.plot_frequency_baseline(trace);

# %% [markdown]
# - Middle panel: one frequency row spends ~75 % of its time near its baseline;
#   the event towers above it.
# - Why Q25 and not the median: the baseline must estimate the *quiet* level,
#   under the assumption that at least 75 % of frames are noise. A median
#   would already drift if the event covered close to half the record.
# - Why clip at zero: a power *deficit* is not an event.
#
# ### What the baseline absorbs — persistent nuisances
#
# A continuous interference tone (mains harmonic, mechanical vibration, an
# oscillating bubble) looks exactly like a Doppler peak to a peak detector:
# it has to be thresholded away, at the risk of deleting weak real peaks
# with it. For the Q25 baseline, *present most of the time* *is* the
# definition of background: the tone raises B_k for its own bin and cancels
# out of P⁺. A synthetic check — a permanent 18 kHz tone plus one real
# 30 kHz burst:

# %%
rng = np.random.default_rng(3)
t = np.arange(16384) / config.sampling_frequency_hz
tone = 0.25 * np.sin(2 * np.pi * 18e3 * t)
burst = np.exp(-0.5 * ((t - 5.0e-3) / 2.0e-4) ** 2) * np.sin(2 * np.pi * 30e3 * t)
tone_signal = (tone + burst + 0.02 * rng.normal(size=t.size)).astype(np.float32)
demo = detector_trace(tone_signal, config)
tone_bin = int(np.argmin(np.abs(demo.frequencies - 18e3)))
print(f"baseline at the 18 kHz bin: {demo.baseline[tone_bin, 0]:.2e}   "
      f"median baseline elsewhere: {float(np.median(demo.baseline)):.2e}")
print(f"excess kept at the 18 kHz bin: {demo.excess[tone_bin].mean() / demo.excess.mean():.2f}"
      " x the average bin (1.00 would be an ordinary bin)")
fig.plot_frame_energy(demo);

# %% [markdown]
# - The 18 kHz row carries a baseline **36 000 times** the typical bin's: the
#   tone was learned as background. The consequence is the whole point — that
#   row, by far the most powerful of the spectrogram, contributes **a quarter
#   of what an ordinary bin contributes** to E[m]. Power buys nothing;
#   only *departure from one's own habit* does. The excess map (top) stays
#   dim at 18 kHz and E[m] has a single bump, at the real burst.
# - So the nuisance never becomes a candidate, and **no peak threshold was
#   needed — hence no real weak event was put at risk by one**.
# - The honest limit: the detection band (7–80 kHz) itself remains a hard
#   gate. A particle slow enough to fall below 7 kHz is lost to both
#   approaches; the robustness argument starts only inside the band.
#
# ### The same trace through the peak-first pipeline
#
# What does particles2SNR do with the tone? The replay below runs the full
# legacy cascade on the identical synthetic trace:

# %%
replay_tone = replay_particles2snr_on_synthetic(
    tone_signal, truth_start=9200, truth_end=10800, frequency_hz=30e3)
tone_events, _ = detect_yeast_events(tone_signal, config)
fig.plot_legacy_vs_energy(demo, replay=replay_tone, truth_spans_ms=[(4.6, 5.4)],
                          mad_events=tone_events);

# %% [markdown]
# - The tone spawns hypotheses all along the trace: **8 raw hypotheses for 1
#   real event**. The cleaning cascade has to execute every spurious one —
#   here six fall to the box-width gate and one to peak evidence.
# - The MAD chain returns **one interval, accepted**, on the real burst and
#   nothing on the tone.
# - The final answer is right *on this toy*, but only because two of the
#   cascade's gates happened to fire on the right hypotheses. The
#   introduction's yeast record is what happens when the same cascade meets
#   a case its thresholds cannot arbitrate.
# - The energy chain (bottom) never created the problem: the tone lives in
#   the baseline, so there was nothing to clean.
#
# ### The other side of the gate — a real but atypical particle
#
# Those cleaning gates are calibrated for *typical* particles: fitted boxes
# t₀ ± 2.5τ must fall within 0.08–1.5 ms, passage times within 0.07–0.65 ms.
# Now a toy with an unusually **slow, weak particle** (9 kHz, long envelope)
# next to a normal fast one:

# %%
rng = np.random.default_rng(5)
slow = 0.35 * np.exp(-0.5 * ((t - 2.5e-3) / 4.0e-4) ** 2) * np.sin(2 * np.pi * 9e3 * t)
fast = np.exp(-0.5 * ((t - 6.0e-3) / 2.0e-4) ** 2) * np.sin(2 * np.pi * 30e3 * t)
slow_signal = (slow + fast + 0.02 * rng.normal(size=t.size)).astype(np.float32)
slow_trace = detector_trace(slow_signal, config)
replay_slow = replay_particles2snr_on_synthetic(
    slow_signal, truth_start=3400, truth_end=6600, frequency_hz=9e3)
slow_events, _ = detect_yeast_events(slow_signal, config)
fig.plot_legacy_vs_energy(slow_trace, replay=replay_slow, mad_events=slow_events,
                          truth_spans_ms=[(1.7, 3.3), (5.6, 6.4)]);

# %% [markdown]
# - The legacy pipeline *sees* the slow particle at the raw stage — then the
#   **box-width gate deletes it** (its fitted box is longer than 1.5 ms), and
#   the final output contains only the fast particle: one box, no trace of
#   the other event.
# - The MAD chain **returns both intervals** — and marks the slow one
#   `reject`, because at 2.256 ms it exceeds the 2.0 ms qualification cap.
#   Read that honestly: the energy chain does **not** rescue the atypical
#   particle either. What differs is the bookkeeping. The legacy hypothesis
#   vanished inside a cascade; here the event is localised, scored
#   (z max ≈ 1.6·10³) and rejected by **one named cap** you can read, argue
#   with, or change — instead of by the interaction of five gates.
# - Honesty notes: this is a constructed illustration, not a prevalence
#   claim. And the width cap is a proxy gate too — it is simply the only one
#   left, applied after the event exists rather than before it can form.

# %% [markdown]
# ## 5 · AGGREGATE — the frame energy E[m]
#
# All frequencies vote **before** anything is counted:
# E[m] = Σ_k P⁺(k, m), then a 3-frame smoothing. The (n_bins × n_frames)
# problem becomes one-dimensional again.

# %%
fig.plot_frame_energy(trace);

# %% [markdown]
# - The excess map is dark everywhere except the event: the sum over
#   frequencies concentrates all of it into one clean 1-D curve.
# - This sum is exactly the answer to the introduction's failure: every
#   Doppler crest of one passage pours into the *same* bump of E[m], so
#   crests can no longer be counted as separate particles.

# %% [markdown]
# ## 6 · NORMALISE — median and MAD ★
#
# Section 5 leaves us with E[m], a 1-D curve where the event is an obvious
# bump. **Why not stop here and call the bump the particle?**
#
# Because "obvious bump" is a judgement *you* make, per trace, by eye. A
# detector needs a written rule, and the obvious rule — "E[m] above some
# value V" — is written in energy units. Energy units are not a property of
# the particle: they change with amplifier gain, coupling, and the noise
# level of the day.
#
# ### 6a · Why a threshold on E[m] cannot be written down
#
# The candidate rule is: *flag every frame whose energy reaches V*. Recall
# from section 3 that a frame is one column of the spectrogram, 64 µs of
# trace, so the number of flagged frames is simply **how much of the record
# the rule calls "event"** — the detector will settle on 14 frames; this
# hand-picked V flags only the 4 around the peak.
#
# Take V at half the event's peak on this record — as good a hand-picked
# value as any — and apply that *same* V to the same trace recorded with
# three times more gain, and three times less. Nothing about the particle
# changes; only the recording chain does.

# %%
V = 0.5 * trace.frame_energy.max()
for label, factor in (("original", 1.0), ("gain x3", 3.0), ("gain /3", 1 / 3)):
    scaled = detector_trace(record.signal * factor, config)
    flagged = int((scaled.frame_energy >= V).sum())
    print(f"{label:9s}: the rule 'E >= V' flags {flagged:3d} frames out of {scaled.times.size}")

# %% [markdown]
# Same particle, same physics, three different answers — and at one third of
# the gain the rule finds **nothing at all**. V is not a bad choice; *any*
# fixed energy value has this defect. The rule has to be expressed relative
# to the trace it is applied to, which means estimating two things from the
# trace itself:
#
# 1. its **ordinary level** — what E[m] looks like when nothing happens;
# 2. its **spread** — how much E[m] normally wanders around that level.
#
# Both estimates must survive the presence of the event: if the event drags
# its own reference upwards, it hides itself. That requirement — not
# tradition — is why the next two subsections use the median and the MAD
# rather than the mean and the standard deviation.
#
# ### 6b · The ordinary level: a median the event cannot move
#
# A toy series of 7 frames, one of which contains an event:

# %%
toy = np.array([1, 1, 1, 2, 2, 2, 20], dtype=float)
calm = np.array([1, 1, 1, 2, 2, 2, 2], dtype=float)
print(f"with outlier   : mean = {toy.mean():.3f}   median = {np.median(toy):.0f}")
print(f"without outlier: mean = {calm.mean():.3f}   median = {np.median(calm):.0f}")

# %% [markdown]
# Replacing **one** value out of seven moves the mean by +164 % and the median
# by 0 %. The mean follows the event; the median stays with the majority.
# **Piece 1 acquired**: median(E) is an ordinary level the event cannot drag
# upwards.
#
# ### 6c · The spread: a MAD the event cannot inflate
#
# Knowing the ordinary level is not enough. It tells us where the middle is,
# not **how far from the middle a frame must be before it counts as
# surprising**. On a calm trace a rise of 3 units is enormous; on a restless
# one it is nothing. So we need a second number: the typical size of the
# wandering around the level.
#
# Every level comes with its natural companion measure of spread, built the
# same way the level was:
#
# | Level | Its companion spread | Built as |
# |---|---|---|
# | mean | standard deviation | square every distance to the mean, average, take the square root |
# | median | **MAD** = *median absolute deviation* | the **median** distance to the median |
#
# We rejected the mean in 6b because a single event drags it. The standard
# deviation is worse: it squares each distance before averaging, so the one
# large distance dominates the result — the event would inflate the very
# scale meant to measure it. Taking the median of the distances, exactly as
# we took the median of the values, keeps that from happening:
#
# MAD(E) = median over frames i of |E_i − median(E)|
#
# On the same seven-value toy, the distances to the median are:

# %%
distances = np.abs(toy - np.median(toy))
print("distances to the median:", np.sort(distances))
print(f"MAD = {np.median(distances):.2f}   (std of the same series = {toy.std():.2f})")

# %% [markdown]
# The large distance (18) exists but cannot occupy the middle of the sorted
# list. That is what robustness means here — not a formula, a *position in a
# sort*. Compare the two scales on the same seven values: the MAD says the
# series normally wanders by 1, the standard deviation says 6.5 — inflated
# more than sixfold by the single value we are trying to detect.
# **Piece 2 acquired.**
#
# ### 6d · z: the two pieces assembled
#
# We now have the two numbers 6a asked for, both read off the trace itself,
# both unmoved by the event they are meant to measure:
#
# - the ordinary level, **median(E)** — piece 1, from 6b;
# - the ordinary wandering, **MAD(E)** — piece 2, from 6c.
#
# Assembling them is one subtraction and one division. Measure how far a
# frame is above the level, then express that distance *in units of the
# wandering* instead of in units of energy:
#
# $$ z[m] \;=\; \underbrace{\frac{E[m] - \mathrm{median}(E)}{\vphantom{X}}}_{\text{how far above ordinary}} \Bigg/ \underbrace{\big(1.4826 \times \mathrm{MAD}(E)\big)}_{\text{one ordinary wandering}} $$
#
# (The 1.4826 is a scale convention, explained in 6f; it changes nothing to
# the reasoning here.) The essential point is what happens to the units:
# numerator and denominator are **both in energy units, so they cancel**.
# z[m] is a pure number, and that is precisely what 6a could not obtain:
#
# > **z[m] = how many ordinary wanderings above its own ordinary level this
# > frame sits.**
#
# Triple the gain and the energy, the level and the wandering all scale
# together — the ratio cannot move. So the rule "z ≥ 3.5" means the same
# thing on every recording, where "E ≥ V" did not. Here is 6a's broken
# experiment, run again with both rules side by side:

# %%
print(f"{'recording':<12}{'frames flagged by E >= V':>26}{'frames flagged by z >= 3.5':>28}")
print("-" * 66)
for label, factor in (("original", 1.0), ("gain x3", 3.0), ("gain /3", 1 / 3)):
    scaled = detector_trace(record.signal * factor, config)
    above_v = int((scaled.frame_energy >= V).sum())
    above_z = int((scaled.energy_z >= config.active_snr_z).sum())
    print(f"{label:<12}{above_v:>26}{above_z:>28}")

# %% [markdown]
# Read the two columns downwards. The energy rule answers **4, then 16, then
# 0** — it disagrees with itself about a particle that never changed. The z
# rule answers **14, 14, 14**.
#
# That is the whole reason z exists, and it is worth stating plainly: **z is
# not a new measurement of the signal. It is E[m] in a different unit** — a
# unit each trace derives from its own quiet stretches. Changing unit is what
# makes a threshold portable; it adds no information and detects nothing by
# itself.
#
# ### 6e · On the real signal
#
# Zoomed to the noise floor, because at full scale both bands are a single
# line — the event's peak sits about **112 raw-MAD widths** above the median:

# %%
fig.plot_robust_band(trace, both=True);

# %% [markdown]
# **Vocabulary, fixed once and for all.** The two purple bands are the two
# quantities that have both been called "the MAD": the inner one is
# `raw_mad`, the unscaled median absolute deviation; the outer one is
# `energy_scale = 1.4826 × raw_mad`, the divisor the detector actually uses.
# Two names, 48 % apart.
#
# ### 6f · Where 1.4826 comes from
#
# The MAD and the standard deviation both measure spread, but they do not
# come out at the same number: on Gaussian noise the MAD lands at about 0.67
# σ. The factor 1.4826 simply rescales it so that the two agree — it puts the
# MAD "on the σ ruler", which is the ruler everyone's intuition uses.
#
# Where the value comes from, for the curious: for a Gaussian, half the
# values fall within 0.6745 σ of the centre, so that distance *is* the MAD.
# Since 1 / 0.6745 ≈ 1.4826, multiplying restores σ. (0.6745 is the 75th
# percentile of the standard Gaussian, written Φ⁻¹(0.75), where Φ is the
# Gaussian cumulative distribution function — the function giving the
# probability of falling below a given value.)
#
# The check, on 100 000 Gaussian draws:

# %%
rng = np.random.default_rng(0)
draw = rng.normal(size=100_000)
mad = np.median(np.abs(draw - np.median(draw)))
print(f"std = {draw.std():.4f}   1.4826 × MAD = {1.4826 * mad:.4f}")

# %% [markdown]
# They coincide on a Gaussian draw. **The constant only has meaning under a
# Gaussian hypothesis; the detector keeps it as a scale convention, not as a
# model of the noise.**
#
# ### 6g · The invariance stated exactly — and its limit ★
#
# Subsection 6d showed the *rule* survives a gain change; here is the
# underlying identity, to numerical precision. Amplitude ×3 means power ×9,
# so E[m], its median and its MAD all scale by 9 — and the ratio does not
# move at all.

# %%
gained = detector_trace(record.signal * 3.0, config)
print(f"E[m] multiplied by {np.median(gained.frame_energy / trace.frame_energy):.3f}")
max_dz = float(np.max(np.abs(gained.energy_z - trace.energy_z)))
assert np.allclose(gained.energy_z, trace.energy_z, rtol=1e-4, atol=1e-6)
print(f"z[m] unchanged: max |Δz| = {max_dz:.2e}")

# %% [markdown]
# Now *add* noise instead of scaling — z is **not** invariant to that, and it
# should not be: the MAD follows the noise floor up, so the same event stands
# out less.

# %%
rng = np.random.default_rng(1)
extra = rng.normal(scale=3.0 * float(record.signal.std()),
                   size=record.signal.size).astype(np.float32)
noisy = detector_trace(record.signal + extra, config)
print(f"peak z: original = {trace.energy_z.max():.1f}   "
      f"with added noise = {noisy.energy_z.max():.1f}")

# %% [markdown]
# What this proves: the z threshold keeps the *same meaning* across gain
# changes, which a fixed energy threshold would not. What it does not prove:
# immunity to noise — z is a contrast against *this trace's* noise floor.
#
# ### 6h · What z is not
#
# - z is **not an SNR** — a signal-to-noise ratio compares the power of a
#   signal to the power of the noise; z compares a *distance* to a *spread*.
# - z is **not a probability**. z = 3.5 does not mean "3.5 sigmas hence
#   p < 0.001": that reading would require the noise to be Gaussian, which
#   6f explicitly declined to assume.
# - z is **not comparable between two traces** with different noise floors:
#   each trace defines its own unit, so equal z does not mean equal physics.
#
# z[m] is the distance of E[m] to this trace's own baseline, in units of this
# trace's own robust scale. Nothing more — and for thresholding, nothing less.
#
# ### 6i · Synthesis — what this buys, on the real record
#
# Everything since section 1 exists to make the following interval computable
# without a human looking at the trace. This is the detector's actual output
# on our record, back on the time series it came from:

# %%
record_events, _ = detect_yeast_events(record.signal, config)
fig.plot_detected_events(record.signal, record_events, config,
                         truth_spans_ms=[(event_start / 2e6 * 1000, event_end / 2e6 * 1000)]);

# %% [markdown]
# The green bar is the bounded event the notebook set out to produce. It sits
# inside the pale-green span but does not fill it: the two agree on the right
# edge and differ by 256 samples on the left.
#
# **That gap is worth understanding, because it is not an error.** Section 1
# said the pale-green span is a detector proposal a reviewer confirmed — and
# that proposal was produced by an *earlier* version of this detector, one
# that grew each interval outwards past the detection threshold. Section 8
# explains why that stage was removed. So the visible difference is precisely
# the region a human review judged to be margin rather than signal.
#
# Sections 7 and 8 are how the z curve becomes this bar: which frames are
# selected, and how they become an interval.
#
# *Method only · one reviewed record · no claim of detector performance or
# biological validity.*

# %% [markdown]
# ## 7 · ACTIVATE — turning z into a decision ★
#
# A frame is **active** when it is unusual enough: a[m] = 1 if z[m] ≥ 3.5.

# %%
fig.plot_activation(trace);

# %% [markdown]
# The shaded frames are the ones the threshold selects — a single contiguous
# run, which section 8 turns into the interval 6i already showed.
#
# ### 7a · The sandbox
#
# Move the threshold and watch the active line. The trace is already
# computed, so only the decision is recomputed — nothing is re-analysed.

# %%
from ipywidgets import FloatSlider, interact


@interact(z_thr=FloatSlider(value=3.5, min=1.0, max=8.0, step=0.1, description="z ≥"))
def _explore(z_thr: float) -> None:
    fig.plot_activation(trace, z_threshold=z_thr)
    plt.show()

# %% [markdown]
# **z sets the duration of the event.** Raise it and the run shrinks towards
# the peak; lower it towards 2.0 and the run extends leftwards (frame 55 to
# 54) while its right edge holds at 68, until a second run appears elsewhere
# in the trace — a candidate the detector would then have to qualify or
# reject.
#
# This is the single knob that matters at activation, and 3.5 is a
# development choice, not a demonstrated optimum. Dragging a slider on one
# record is a sensitivity check, not a calibration.
#
# *Method only · one reviewed record · no claim of detector performance or
# biological validity.*

# %% [markdown]
# ## 8 · GROUP — from selected frames to one bounded event
#
# Section 7 leaves a set of frames. Turning them into an interval takes three
# operations, each with one number from the preset.
#
# **1 · Bridge short gaps.** A passage can dip below the threshold for a frame
# or two without ending. `cluster_gap_ms = 0.128 ms` becomes a tolerance in
# frames: 0.128 ms × 2 MHz ÷ 128 samples per hop = **2 frames**. Runs
# separated by at most that are one event.
#
# **2 · Convert frames to samples.** The run spans `left × hop` to
# `right × hop + N`: a frame is a 512-sample window, not an instant, so the
# last frame contributes its whole window.
#
# **3 · Pad.** `boundary_pad_ms = 0.04 ms` = **80 samples** on each side in
# this yeast preset.
#
# **Bead dataset v2.1.** Its production profile is stricter than the yeast
# preset illustrated here: `boundary_pad_ms=0.0`. Its YOLO box is exactly the
# support of the active MAD frames after grouping.

# %%
bounds, = event_bounds(trace, record.signal.size)
group_left, group_right = bounds.group
print(f"activation run : frames {group_left}–{group_right}  ({group_right - group_left + 1} frames)")
for label, (a, b) in (("run, in samples", bounds.group_samples),
                      ("after the pad  ", (bounds.start, bounds.end))):
    print(f"{label}: [{a:5d}, {b:5d}]  {b - a:5d} samples  {(b - a) / 2e6 * 1000:.3f} ms")
print(f"detector output: [{record_events[0].event_start:5d}, {record_events[0].event_end:5d}]")

# %% [markdown]
# The interval is the activation run plus the pad — nothing else, which is why
# no figure is needed here: it is the shaded run of section 7, converted to
# samples. Every boundary in this notebook is therefore a direct consequence
# of one threshold, z ≥ 3.5, which is the whole point of section 6 having
# built z carefully.
#
# On this record the gap-bridging tolerance never fires: the 14 selected
# frames are already contiguous. It matters for events that flicker below the
# threshold mid-passage.
#
# ### 8a · Qualification — the interval still has to earn its label
#
# A bounded interval is a *candidate*. Four tests decide whether it becomes a
# usable example, and unlike the legacy cascade they run **after** the event
# exists, so a failure is reported against a localised object rather than
# deleting a hypothesis:
#
# | Test | Preset | On this event |
# |---|---|---|
# | width within 0.06–2.0 ms | `min/max_width_ms` | 1.168 ms ✔ |
# | max z over the event ≥ 12 | `strict/medium_min_snr` | 75.5 ✔ |
# | event concentration ≥ 0.12 | `strict_min_concentration` | 0.967 ✔ |
# | at most 5 events per trace | `max_events_per_signal` | 1 ✔ |
#
# All four pass, so the candidate is labelled `strict`. The slow particle of
# section 4 failed the first one at 2.256 ms and was labelled `reject` — kept,
# localised, and explained by a single named cap.
#
# *Event concentration is the share of the event's own excess held by its five
# strongest bins. It is evaluated only after localisation, as one of the
# candidate qualification tests shown in the table.*

# %% [markdown]
# ## 9 · CROP — the bounded event becomes a training example
#
# The detector's interval has a variable width; a model needs a fixed-length
# input. The dataset contract (`yeast-event-8192to4096-bandpass-global-v1`)
# resolves that in two steps, and they are different in kind:
#
# 1. **A fixed window in time** — 8192 samples (4.096 ms) centred on the
#    event's energy-weighted centre, *not* on the middle of its interval. The
#    detected event sits inside, with context on both sides.
# 2. **A change of resolution** — that window is band-passed and downsampled
#    by 2, giving **4096 points at 1 MHz**. The duration is unchanged; only
#    the sampling rate is.
#
# The "8192to4096" in the contract name is that second step. It is a
# resolution change, not a second, narrower window — an easy misreading.

# %%
fig.plot_ssl_crop(record.signal, record_events[0], config);

# %% [markdown]
# - Pale green: the detected event, 2 336 samples. Pale blue: the 8 192-sample
#   window handed to the model. Dashed line: the energy-weighted centre.
# - Why the crop is wider than the event: a model given only the event would
#   never see what ordinary signal looks like, and the boundary itself would
#   become a learnable artefact. The margin carries that context.
# - Why centring is on the energy-weighted centre rather than the interval's
#   midpoint: energy is not distributed evenly inside the interval, so the
#   midpoint (sample 8128 here) drifts from where the event actually is
#   (8249).
#
# This is the answer to the notebook's opening question. A time series went
# in; one fixed-length, bounded, centred example comes out — with a quality
# label and a documented provenance chain.
#
# *Method only · one reviewed record · no claim of detector performance or
# biological validity.*

# %% [markdown]
# ### Boxes and classifier crops in MAD v2.1
#
# The v2.1 dataset stores complete 16 384-sample traces. Its YOLO label is the
# active-frame support itself; the classifier crop is a downstream view and
# cannot change that box. Conv1DGAP-S centres 4 096 samples on the event's
# recorded energy centre (`center_index`), reflect-pads beyond a trace edge,
# then averages non-overlapping groups of eight samples to obtain 512 inputs.
# The cell calls the trainer's own preprocessing functions so this explanation
# cannot silently drift from the experiment.

# %%
_, mad_v21_root = resolve_registered_dataset(workspace)
with (mad_v21_root / "events.csv").open(newline="", encoding="utf-8") as handle:
    mad_events = list(csv.DictReader(handle))

development_events = [
    row for row in mad_events if row["output_split"] in {"train", "val"}
]
inside = lambda row: (crop_limits(int(row["center_index"]))[0] <= int(row["event_start"])
                      and int(row["event_end"]) <= crop_limits(int(row["center_index"]))[1])
edge_padded = [row for row in development_events
               if crop_limits(int(row["center_index"]))[0] < 0
               or crop_limits(int(row["center_index"]))[1] > 16_384]
print(f"{len(development_events)} train/val events · "
      f"{len(edge_padded)} crops completed by reflection")

# %%
example = next(row for row in development_events
               if row["output_split"] == "val" and inside(row)
               and 0 <= crop_limits(int(row["center_index"]))[0]
               and crop_limits(int(row["center_index"]))[1] <= 16_384)
center_index = int(example["center_index"])
signal = np.load(mad_v21_root / example["output_split"] / "signals"
                 / f"{example['output_stem']}.npy")
assert centered_reflect_crop(signal, center_index).shape == (RAW_CROP_LENGTH,)
assert prepare_event_signal(signal, center_index, noise_snr_db=None).shape == (INPUT_LENGTH,)

fig.plot_box_and_crop(
    signal, center_index=center_index,
    box=(int(example["event_start"]), int(example["event_end"])),
    crop=crop_limits(center_index), crop_length=RAW_CROP_LENGTH,
    title=f"{example['source_id']} · box and classifier view, from v2.1");

# %% [markdown]
# - Green is the v2.1 detection box; blue is the exact 4 096-sample view read
#   by Conv1DGAP-S. The dashed line is `center_index`, not the box midpoint.
# - Reflection preserves the requested crop length at trace edges without
#   moving its centre. This happens for 747 of the 2 921 train/validation
#   events.
# - The event shown is one whose box fits entirely inside its crop; a support
#   approaching 4 096 samples with an off-centre energy peak need not.
#
# This section reads development metadata and one validation signal only. The
# sealed test payload is not opened.

# %% [markdown]
# ## Running on your own data
#
# Everything above used the workspace only to fetch a traceable example. The method itself needs a plain 1-D array. The protocol:
#
# 1. A 1-D float array (any amplitude units — section 6d showed why gain does
#    not matter).
# 2. A `YeastDetectionConfig` whose `sampling_frequency_hz` matches your
#    acquisition; every other field has the development default. The preset
#    used here assumes 2 MHz.
# 3. At least one STFT window of samples (N = 512 at the default settings).
#
# Below, a synthetic trace built from scratch — a 30 kHz burst in Gaussian
# noise — goes through the identical chain:

# %%
rng = np.random.default_rng(42)
t = np.arange(16384) / config.sampling_frequency_hz
burst = np.exp(-0.5 * ((t - 4.0e-3) / 2.0e-4) ** 2) * np.sin(2 * np.pi * 30e3 * t)
your_signal = (burst + 0.02 * rng.normal(size=t.size)).astype(np.float32)
your_events, _ = detect_yeast_events(your_signal, config)
fig.plot_detected_events(your_signal, your_events, config, truth_spans_ms=[(3.6, 4.4)]);

# %% [markdown]
# No registry, no review queue, no workspace: one array, one config, the same
# eight stages, one bounded event. Replace `your_signal` with your own
# acquisition and the whole notebook's reasoning applies unchanged.
#
# ---
# **Still to come: a multi-record sandbox and the appendices (glossary,
# configuration, evidence and limits).**
