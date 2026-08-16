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
# that a rare large event cannot distort. Sections 1–5 build the quantity the
# MAD is applied to, section 6 builds the MAD itself, and section 7 turns the
# result into the bounded example. The chain follows the frozen
# [v5 explainer deck](../../artifacts/cross-project/presentations/yeast-detector-explainer-v5/render-r1/Yeast_detector_explainer_v5.pdf):
#
# `ISOLATE → LOCALISE → REFERENCE → AGGREGATE → NORMALISE → ACTIVATE → GROUP → CROP`
#
# It explains a method on one record: no performance metric, no biological
# claim. The detector itself only needs a 1-D array — the workspace calls
# below just fetch a traceable example, and the closing sections run the same
# chain on a signal built from scratch.
#
# **Colour rule for every figure**: a pale-**blue** span was *given* (a
# reviewed support, a declared truth, a context window); a pale-**green** span
# was *computed* by the chain.

# %%
import json
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image
from internship_workspace.config import Workspace
from internship_workspace.mad_conv1dgap_training import resolve_registered_dataset
from internship_workspace.mad_v21_figures import (
    read_rows,
    render_comparison,
    render_gallery,
    select_cases,
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
# | Boundaries | pad 0.04 ms = 80 samples per side |
# | Grouping | bridge gaps ≤ 0.128 ms (2 frames) |
# | Qualification | width 0.06–2.0 ms · max z ≥ 12 |
# | Cap | keep the 5 strongest events by z — a silent cut, not a reported test |
#
# *These values are a development choice validated on a review campaign, not a
# demonstrated optimum.* The last rows name quantities the notebook has not
# built yet — z is constructed in section 6, the qualification quantities in
# section 7; the table is here as a reference to come back to, not as
# something to read now.
#
# Every setting listed is one that changes an outcome. The preset also carries
# two thresholds that do not, and the table leaves them out rather than
# implying they arbitrate anything: the frame-level concentration floor, set
# to 0.0 and therefore inert by construction, and the event-level
# concentration floor, measured in §7d against 13 226 yeast events without a
# single one reaching it. Appendix A2 prints the full object, inert fields
# included.

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
# One more structural cost — section 5 demonstrates it, and the appendix
# measures it:
#
# - A peak threshold is a **hard gate on a proxy**: a persistent
#   nuisance oscillation *is* a Doppler-like peak, so peak-first has to
#   threshold it away — and the same threshold can delete a real but
#   atypical particle, for example an unusually slow one with a weak crest.
#   Section 5 shows why the energy chain does not need that gate at all: a
#   persistent oscillation is *learned into the baseline* and never becomes
#   a candidate in the first place.

# %% [markdown]
# ## 1 · The trace and the event support
#
# The record is a 1-D acoustic trace sampled at 2 MHz: one point every 0.5 µs.
# At this rate, the 16 384 samples cover about 8.2 ms. The blue span is the
# reviewed support — and **where it comes from matters for everything that
# follows**. It is not a boundary a human drew: it is a detector proposal that
# a reviewer *confirmed* (`review_event_present = yes`,
# `review_center_acceptable = yes`, `review_full_event_visible = yes`,
# `reviewer = Julien`). The record therefore certifies that one complete event
# is there and the proposal is acceptable — not where its edges independently
# lie. Nothing in this notebook can be read as boundary accuracy.

# %%
fig.plot_trace_overview(record.signal, config, zoom_center=center,
                        support=(event_start, event_end));

# %% [markdown]
# - The zoom shows individual samples: the signal is smooth at the µs scale.
# - Nothing outside the blue span says *where* the event is — the eye picks
#   the large excursion, and that intuition is exactly what the detector must
#   formalise. We call the event's sample interval its **temporal support**.

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
# - Grey behind: the raw trace of section 1. Blue: what every later stage
#   sees. The slow drift and high-frequency hash disappear; the oscillatory
#   burst around 4 ms survives, unmoved.

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
# on every row of the real record — one row in detail, then the surviving
# excess of the whole spectrogram of section 3:

# %%
fig.plot_frequency_baseline(trace);

# %% [markdown]
# - Top panel: one frequency row spends ~75 % of its time near its baseline;
#   the event towers above it.
# - Why Q25 and not the median: the baseline must estimate the *quiet* level,
#   under the assumption that at least 75 % of frames are noise. A median
#   would already drift if the event covered close to half the record.
# - Why clip at zero: a power *deficit* is not an event.

# %% [markdown]
# ## 5 · AGGREGATE — the frame energy E[m]
#
# All frequencies vote **before** anything is counted:
# E[m] = Σ_k P⁺(k, m), then a 3-frame smoothing. The excess map of section 4
# — dark everywhere except the event — collapses into one clean 1-D curve,
# and the (n_bins × n_frames) problem becomes one-dimensional again.

# %%
fig.plot_frame_energy(trace, show_map=False);

# %% [markdown]
# This sum is exactly the answer to the introduction's failure: every Doppler
# crest of one passage pours into the *same* bump of E[m], so crests can no
# longer be counted as separate particles.
#
# ### What the baseline absorbs — persistent nuisances
#
# A continuous interference tone (mains harmonic, mechanical vibration, an
# oscillating bubble) looks exactly like a Doppler peak to a peak detector:
# it has to be thresholded away, at the risk of deleting weak real peaks
# with it. For the Q25 baseline, *present most of the time* *is* the
# definition of background: the tone raises B_k for its own bin and cancels
# out of P⁺, so it never reaches E[m]. A synthetic check — a permanent
# 18 kHz tone plus one real 30 kHz burst:

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
tone_events, tone_reason = detect_yeast_events(tone_signal, config)
assert not tone_reason, tone_reason
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
slow_events, slow_reason = detect_yeast_events(slow_signal, config)
assert not slow_reason, slow_reason
fig.plot_legacy_vs_energy(slow_trace, replay=replay_slow, mad_events=slow_events,
                          truth_spans_ms=[(1.7, 3.3), (5.6, 6.4)]);

# %% [markdown]
# - The legacy pipeline *sees* the slow particle at the raw stage — then the
#   **box-width gate deletes it** (its fitted box is longer than 1.5 ms), and
#   the final output contains only the fast particle: one box, no trace of
#   the other event.
# - The MAD chain **returns both intervals** — and marks the slow one
#   `reject`, because at 2.256 ms it exceeds the 2.0 ms qualification cap.
#   What differs from the legacy is the bookkeeping: there the hypothesis
#   vanished inside a cascade; here the event is localised, scored and
#   rejected by one named cap you can read, argue with, or change.
# - **Read the widths honestly, though.** The declared truths are 1.6 ms
#   (slow) and 0.8 ms (fast); MAD's intervals are 2.256 ms and 1.552 ms. On
#   these smooth Gaussian envelopes the energy tails cross z = 3.5 well
#   beyond the declared span, so MAD's boundaries run wide — nearly ×2 on
#   the *normal* particle. The rejection above is that boundary width
#   colliding with the cap, at least as much as the particle's atypicality.
# - The z maxima printed on the bars (≈1.6·10³ and 7.3·10⁴) are what
#   near-noiseless synthetic signals produce; the real record of section 1
#   peaks at z = 75.5. Thresholds like "max z ≥ 12" live at real-noise
#   scale, not at toy scale.
# - This is a constructed illustration, not a prevalence claim.

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
# One estimation detail, stated once because every z in this notebook —
# thresholds included — lives on it: E[m] was smoothed over 3 frames back in
# section 5, *before* the median and the MAD are taken. The scale therefore
# measures the wandering of the smoothed series, which is smaller than the
# raw frame-to-frame noise — so z values are larger than they would be
# against unsmoothed frames. The convention is consistent (detection and
# thresholds share it), but z magnitudes are not comparable to a smoothing-
# free implementation.
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
# line. The event's peak sits at z = 75.5 — with the 1.4826 factor of the
# next subsection, that is 75.5 × 1.4826 ≈ **112 raw-MAD widths** above the
# median, far off this axis:

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
#   The code predates this distinction: the config fields `active_snr_z` and
#   `medium/strict_min_snr`, and the per-event field `snr_proxy`, all hold z
#   values. Wherever this codebase says *snr*, read *z*.
# - z is **not a probability**. z = 3.5 does not mean "3.5 sigmas hence
#   p < 0.001": that reading would require the noise to be Gaussian, which
#   6f explicitly declined to assume.
# - z is **not comparable between two traces** with different noise floors:
#   each trace defines its own unit, so equal z does not mean equal physics.
#
# z[m] is the distance of E[m] to this trace's own baseline, in units of this
# trace's own robust scale. Nothing more — and for thresholding, nothing less.

# %% [markdown]
# ## 7 · From z to a training example ★
#
# Four operations turn the z curve into the deliverable: **select** the
# unusual frames, **group** them into one interval, **qualify** that
# interval, and **cut** the fixed window a model can consume.
#
# ### 7a · ACTIVATE — selecting frames
#
# A frame is **active** when it is unusual enough: a[m] = 1 if z[m] ≥ 3.5.
# The shaded region below covers the exact samples the grouping step will
# assign to the run — `left × hop` to `right × hop + N`, the last frame
# contributing its whole window. Note what it does *not* yet include: the
# 80-sample pad of 7b, which the final interval adds on each side. Three
# spans therefore exist in this section, and 7b prints all three so they
# never have to be guessed from a figure.

# %%
fig.plot_activation(trace);

# %% [markdown]
# How the threshold behaves around its working point — printed, so the claims
# are checkable in the export:

# %%
for z_thr in (5.0, 3.5, 2.0):
    active = trace.energy_z >= z_thr
    spans = ", ".join(f"{a}–{b}" for a, b in fig._runs(active))
    print(f"z >= {z_thr}: {int(active.sum()):2d} frames in "
          f"{fig.count_active_runs(active)} run(s)  [{spans}]")

# %% [markdown]
# **z sets the duration of the event.** Raise it to 5.0 and the run shrinks
# towards the peak (57–67). Lower it to 2.0 and the run extends leftwards
# (55 → 54) while its right edge holds at 68 — and a **second run appears**
# at frames 72–74, a candidate the detector would then have to qualify or
# reject. 3.5 is a development choice, not a demonstrated optimum; this sweep
# on one record is a sensitivity check, not a calibration.
#
# The slider re-runs the same decision live. It opens at **z = 2.0** — the
# row of the sweep where a second candidate appears — rather than repeating
# the figure above; drag it back to 3.5 to recover the detector's own answer.
# (It needs a running kernel; in a static export the printed sweep is the
# evidence.)

# %%
from ipywidgets import FloatSlider, interact


@interact(z_thr=FloatSlider(value=2.0, min=1.0, max=8.0, step=0.1, description="z ≥"))
def _explore(z_thr: float) -> None:
    fig.plot_activation(trace, z_threshold=z_thr)
    plt.show()

# %% [markdown]
# ### 7b · GROUP — from selected frames to one bounded interval
#
# Three operations, each with one number from the preset.
#
# **1 · Bridge short gaps.** A passage can dip below the threshold for a frame
# or two without ending. `cluster_gap_ms = 0.128 ms` becomes a tolerance in
# frames: 0.128 ms × 2 MHz ÷ 128 samples per hop = **2 frames**. Runs
# separated by at most that are one event. On this record it never fires:
# the 14 selected frames are already contiguous.
#
# **2 · Convert frames to samples.** The run spans `left × hop` to
# `right × hop + N`: a frame is a 512-sample window, not an instant, so the
# last frame contributes its whole window.
#
# **3 · Pad.** `boundary_pad_ms = 0.04 ms` = **80 samples** on each side.
# This one is live and the cell below measures it: it adds 160 of the
# interval's 2 336 samples, **6.8 %** of the final width. The bead dataset
# v2.1 of the appendix sets it to 0.0 — its boxes are exactly the grouped
# active frames, with no margin at all.
#
# > **A stage that used to sit here, and no longer does.** The detector once
# > grew each interval outwards while z stayed above a second, lower threshold
# > (1.5 against 3.5 for detection), on the reasoning that an event's quiet
# > tails are still the event. Review `yeast-boundary-expansion-review-r2` put
# > that to a human: on **76 of 76** reviewed events the extended region was
# > judged `extension_margin` — margin, not signal — and never once
# > `extension_signal`. The stage is off; it remains reachable behind
# > `boundary_expansion_enabled` so that comparison stays reproducible.

# %%
bounds, = event_bounds(trace, record.signal.size)
group_left, group_right = bounds.group
print(f"activation run : frames {group_left}–{group_right}  ({group_right - group_left + 1} frames)")
for label, (a, b) in (("run, in samples", bounds.group_samples),
                      ("after the pad  ", (bounds.start, bounds.end))):
    print(f"{label}: [{a:5d}, {b:5d}]  {b - a:5d} samples  {(b - a) / 2e6 * 1000:.3f} ms")
expanded, = event_bounds(detector_trace(
    record.signal, replace(config, boundary_expansion_enabled=True)), record.signal.size)
print(f"retired expansion on: [{expanded.start:5d}, {expanded.end:5d}]  — the reviewed span")

# %% [markdown]
# The interval is the activation run plus the pad — nothing else. Every
# boundary in this notebook is a direct consequence of one threshold,
# z ≥ 3.5, which is the whole point of section 6 having built z carefully.
#
# The last line is worth pausing on: **re-enable the retired expansion and
# the bounds come out at [6704, 9296] — the reviewed support of section 1,
# to the sample.** That is a provenance check, not an accuracy result: the
# reviewed span *is* that older detector's own proposal, so reproducing it
# is an identity (the right edges agree by construction — same frame 68
# through the same formula). What the identity localises is the difference:
# the 256 samples on the left are exactly the region review r2 judged to be
# margin rather than signal.
#
# ### 7c · The output, back on the record

# %%
record_events, record_reason = detect_yeast_events(record.signal, config)
assert not record_reason, record_reason
fig.plot_detected_events(record.signal, record_events, config,
                         truth_spans_ms=[(event_start / 2e6 * 1000, event_end / 2e6 * 1000)]);

# %% [markdown]
# The green bar is the bounded event the notebook set out to produce. It sits
# inside the blue reviewed span but does not fill it: the two agree on the
# right edge and differ by the 256 left-edge samples that 7b just accounted
# for, region by region. The visible gap is the retired expansion stage —
# reviewed, and ruled margin.
#
# **Why the green bar is wider than the shading of 7a, and narrower than the
# blue span.** Three different spans have appeared, each one strictly
# containing the previous, and the whole of this section is the passage from
# one to the next:
#
# | Span | Samples | Width | Built by |
# |---|---|---|---|
# | activation run, shaded in 7a | [7040, 9216] | 1.088 ms | frames 55–68, last frame's window included |
# | **the green bar here** | [6960, 9296] | 1.168 ms | + the 80-sample pad on each side (7b) |
# | the blue span | [6704, 9296] | 1.296 ms | + 256 samples of retired expansion, on the left only |
#
# So the green bar exceeds 7a's shading by exactly the pad, and falls short of
# the blue span by exactly the retired stage. And note what the blue span is
# here: **an older detector's own proposal, confirmed by a reviewer** — not an
# independently measured boundary. The next section uses the same pale blue for
# something different in kind, which is why it says so explicitly.
#
# ### 7d · QUALIFY — the interval still has to earn its label
#
# A bounded interval is a *candidate*. Two tests decide its label, and they
# run **after** the event exists, so a failure is recorded against a localised
# object instead of deleting a hypothesis:
#
# | Reported test | Preset | On this event |
# |---|---|---|
# | width within 0.06–2.0 ms | `min/max_width_ms` | 1.168 ms ✔ |
# | max z over the event ≥ 12 | `strict/medium_min_snr` | 75.5 ✔ |
#
# The slow particle of section 5 failed the width test at 2.256 ms and was
# labelled `reject`; on the yeast corpus the z floor is what every other
# rejection comes down to.
#
# **A third test exists in the code and never binds.** `_quality` also
# requires an *event concentration* — the share of the event's own excess
# held by its five strongest frequency bins — of at least 0.12 for `strict`
# and 0.08 for `medium`. Run over the whole registered acquisition, 6 172
# traces and **13 226 events, the smallest concentration observed is 0.52**:
# four times the strict floor, and no event has ever come near either
# threshold. Two consequences follow, and the tables of this notebook reflect
# both. The floor never rejects anything. And since this preset sets the z
# floor to 12 for *both* labels, concentration is the only thing that could
# ever separate `strict` from `medium` — so **`medium` is unreachable**, and
# the label is in practice binary, `strict` or `reject`. The bead corpus of
# the appendix agrees independently: 3 618 events, minimum 0.68.
#
# The quantity is still worth computing — it is reported per event as
# `energy_concentration`, and it is what the bead pipeline's deblending rule
# tests on candidate *segments*. It is simply not a discriminator here, and
# presenting it as one would overstate what decides a label.
#
# **One limit is not a test.** `max_events_per_signal = 5` sorts candidates
# by max z and silently keeps the five strongest: events beyond it are not
# rejected with a reason, they are dropped without a label. On this record
# it is idle (one event); the bead dataset builder of the appendix runs with
# its own configuration, which carries no such cap — which is why a 7-box
# trace can appear there.
#
# ### 7e · CROP — the bounded event becomes a training example
#
# **Start from what we actually hold.** One record is **16 384 points sampled
# at 2 MHz** — one point every 0.5 µs, 8.192 ms of trace. That is the file as
# acquired: the registered dataset `yeast-hf-10-5-20260610@v1` was imported
# byte for byte, sha256-verified before and after copy, with **no resampling
# or decimation applied anywhere upstream**. Everything sections 1–7d did was
# computed on those 16 384 original points.
#
# The detector's interval has a variable width; a model needs a fixed-length
# input. The dataset contract (`yeast-event-8192to4096-bandpass-global-v1`)
# resolves that in two steps, different in kind, and each one discards
# something specific:
#
# 1. **A fixed window in time** — 8 192 points (4.096 ms) centred on the
#    event's energy-weighted centre, *not* on the middle of its interval.
#    **What is lost: half the record.** 16 384 → 8 192 points; the 4.096 ms
#    outside the window are dropped, and with them any second event living
#    there. At a trace edge the window cannot be centred, and the contract's
#    answer is to **clamp it inwards, never to pad** (`"padding": "forbidden"`)
#    — so an edge event ends up off-centre, but every sample the model sees is
#    real signal.
# 2. **A change of resolution** — that window is band-passed (5–100 kHz,
#    Butterworth order 4, zero-phase) and resampled by `resample_poly(down=2)`,
#    giving **4 096 points at 1 MHz**. The duration is unchanged; only the
#    sampling rate is. **What is lost: everything above 500 kHz**, the new
#    Nyquist limit — which the 100 kHz band-pass had already removed one line
#    earlier, so this step costs no in-band information at all. Note the
#    contract's band is *wider* than the detector's 7–80 kHz: the model is
#    handed a little more spectrum than the detector used to find the event.
#    The "8192to4096" in the contract name is this second step — a resolution
#    change, not a second narrower window, which is an easy misreading. A
#    final global normalisation (development-train mean and standard
#    deviation) is applied on top. The figure below shows step 1 only.

# %%
fig.plot_ssl_crop(record.signal, record_events[0], config);

# %% [markdown]
# - Pale green: the detected event, 2 336 samples. Pale blue: the
#   8 192-sample window handed to the model. Dashed line: the
#   energy-weighted centre.
# - Why the crop is wider than the event: a model given only the event would
#   never see what ordinary signal looks like, and the boundary itself would
#   become a learnable artefact. The margin carries that context. At a trace
#   edge the three code paths in this workspace differ, and it is worth
#   knowing which one produced a given array: the **registered yeast
#   contract** clamps the window inwards and forbids padding, the **bead
#   builder** reflect-pads (`centered_reflect_crop`, 6 144 points), and the
#   convenience helper `crop_around_index` in `yeast_events` zero-pads. Only
#   the first two ever produced a dataset.
# - Why centre on the energy-weighted centre rather than the interval's
#   midpoint: energy is not distributed evenly inside the interval. Honest
#   scale, though: both centres live on the 128-sample frame grid, and the
#   difference here (8249 vs midpoint 8128) is 121 samples — less than one
#   hop. Read the centring as a design principle, not something this record
#   proves. (A third number, the reviewed row's centre 8236, is the *old*
#   expanded interval's centre.)
#
# This closes the notebook's opening question. A time series went in; one
# fixed-length, bounded, centred example comes out — with a quality label
# and a documented provenance chain.
#
# *Method only · one reviewed record · no claim of detector performance or
# biological validity.*

# %% [markdown]
# ## Running on your own data
#
# Everything above used the workspace only to fetch a traceable example. The
# method itself needs a plain 1-D array. The protocol:
#
# 1. A 1-D float array (any amplitude units — section 6d showed why gain does
#    not matter), at least one STFT window long (512 samples here).
# 2. The configuration used throughout: `review_calibrated_detection_config_v1()`,
#    with `sampling_frequency_hz` matching your acquisition (2 MHz here). Do
#    **not** substitute bare `YeastDetectionConfig()` defaults — six fields
#    differ, and the defaults also gate activation on a concentration floor
#    the preset disables.
# 3. Check the failure channel: `detect_yeast_events` returns
#    `(events, reason)` and never raises — an empty result with a non-empty
#    `reason` is a diagnosis, not a success.

# %%
rng = np.random.default_rng(42)
t = np.arange(16384) / config.sampling_frequency_hz
burst = np.exp(-0.5 * ((t - 4.0e-3) / 2.0e-4) ** 2) * np.sin(2 * np.pi * 30e3 * t)
your_signal = (burst + 0.02 * rng.normal(size=t.size)).astype(np.float32)
your_events, your_reason = detect_yeast_events(your_signal, config)
assert not your_reason, your_reason
fig.plot_detected_events(your_signal, your_events, config, truth_spans_ms=[(3.6, 4.4)]);

# %% [markdown]
# No registry, no review queue, no workspace: one array, one config, the same
# eight stages, one bounded event. Replace `your_signal` with your own
# acquisition and the whole notebook's reasoning applies unchanged.
#
# **Here the green interval is twice the blue span — the opposite of 7c, and
# for an unrelated reason.** The two figures use the same pale blue for two
# different things, so the comparison is worth making explicit:
#
# - In 7c the blue span was *an older detector's proposal*, and the green sat
#   inside it because a stage was retired.
# - Here the blue span is **a convention I wrote by hand**: the envelope is a
#   Gaussian of σ = 0.2 ms centred at 4.0 ms, and I declared its truth to be
#   ±2σ. Nothing measured that. The detector answers a different question —
#   *where does z fall back to 3.5?* — and on a near-noiseless synthetic
#   (peak z ≈ 7·10⁴) that happens far out on the tail.
#
# The cell below decomposes the 1.680 ms the detector returned against the
# 0.800 ms I declared. Every term is a stage of section 7, and none of them is
# an error:

# %%
your_trace = detector_trace(your_signal, config)
your_bounds, = event_bounds(your_trace, your_signal.size)
left, right = your_bounds.group
half_window = config.stft_nperseg // 2
centres_ms = (right - left) * your_trace.hop / 2e6 * 1000.0
print(f"declared truth, my ±2σ convention              :   0.800 ms")
print(f"frames {left}–{right} still above z = {config.active_snr_z}, centre to centre"
      f"  :  {centres_ms:6.3f} ms")
print(f"+ half an STFT window at each end (N = {config.stft_nperseg})   : + "
      f"{2 * half_window / 2e6 * 1000:6.3f} ms")
print(f"+ the boundary pad, 80 samples per side        : + "
      f"{2 * config.boundary_pad_ms:6.3f} ms")
print(f"detector interval                              = "
      f"{your_events[0].width_ms:6.3f} ms   "
      f"({your_events[0].width_ms / 0.8:.2f} × the declared span)")
sigma_reached = (centres_ms / 2) / 0.2
window_sigma = config.stft_nperseg / 2e6 * 1000.0 / 0.2
print(f"\nthe first term reaches ±{sigma_reached:.1f}σ, where the envelope is only "
      f"{np.exp(-sigma_reached ** 2 / 2):.0e} of its peak")
print(f"— yet the frame still activates, because one STFT window spans {window_sigma:.1f}σ "
      f"and\n  3-frame smoothing widens it further: a frame collects energy from well "
      f"inside the burst.")

# %% [markdown]
# So the factor of two is a Gaussian tail plus one window plus one pad, not a
# detection failure — the same effect section 5 flagged on its toys.
#
# The last two lines are the part worth keeping. **A boundary in this chain is
# a property of frames, not of instants.** One STFT window is 1.3 σ wide here
# and the energy is smoothed over three of them, so a frame sitting where the
# envelope has all but vanished still collects enough energy from nearer the
# peak to cross the threshold. That blur is inherent to the time–frequency
# front end: no z threshold can produce a boundary sharper than the window
# that measures it. **On real traces the effect is far smaller**, because a
# real noise floor lifts the denominator — the record of section 1 peaks at
# z = 75.5, not 73 000, and its interval is 1.168 ms. Read synthetic widths as
# a property of smooth noiseless envelopes, and never calibrate a width
# threshold on them.
#
# ---

# %% [markdown]
# ## 8 · Sandbox — the same chain on four reviewed records
#
# Everything so far rests on one record. Four are manifested in the review
# queues, and they were chosen to disagree with each other: a clean single
# passage, a trace holding several, and one where a neighbouring rise is
# proposed and then rejected. The chain below is not re-implemented — it is
# the same `detect_yeast_events` call, on four different signals.

# %%
SANDBOX = {
    "principal": "9459e76ce29342debc90:00",
    "clean single passage": "214f4ce4967af98a954c:00",
    "several passages": "e1b4603f8b9de6204003:02",
    "rejected neighbour": "09f788a7473797b794f6:01",
}
entries = []
for label, event_id in SANDBOX.items():
    signal = load_reviewed_event(workspace, event_id).signal
    events, reason = detect_yeast_events(signal, config)
    assert not reason, f"{event_id}: {reason}"
    entries.append((f"{label} · {event_id}", signal, events))
fig.plot_records_overview(entries, config);

# %% [markdown]
# Read the four rows against each other:
#
# | Record | Returned | Accepted | What it shows |
# |---|---|---|---|
# | principal | 1 | 1 | the worked example of sections 1–7 |
# | clean single passage | 1 | 1 | the easy case: one passage, one box |
# | several passages | 4 | 2 | grouping keeps passages apart instead of merging them |
# | rejected neighbour | 2 | 1 | a real rise localised, then labelled `reject` |
#
# The last two rows carry the argument. **`reject` is a label, not a
# deletion**: the chain returns the interval, scores it, and records why it
# failed — which is what section 7d claimed and what the legacy cascade of the
# introduction cannot do, because its hypotheses disappear inside the cleaning
# stages. And the multi-passage record shows the grouping tolerance doing the
# opposite job from the introduction's failure: there, one event was split in
# two; here, several genuine events stay separate.
#
# Change an identifier above and the whole notebook's reasoning recomputes —
# these four are simply the ones with a confirmed human review.

# %% [markdown]
# ## Appendix · The same idea on beads
#
# **This detector was designed on yeast**, and everything above is yeast: the
# reviewed record, the four sandbox records, the review campaigns that fixed
# the preset. Yeast is also the hard case — cells vary in size, shape and
# orientation, so no boundary is ever independently known.
#
# The natural test was therefore to take the *same* chain to **calibrated
# polystyrene beads** of controlled diameter (2, 4 and 10 µm), where a passage
# is a far more repeatable object. It held: the bead corpus produced **3 618
# retained events over 2 888 traces** and became the pseudo-label reference for
# training. The two figures below are that evidence — no new argument, just
# the chain on a different particle.
#
# **What is identical, and what differs.** The front end is the same code path
# and the same numbers: 7–80 kHz, N = 512 with H = 128, 3-frame smoothing,
# activation at z ≥ 3.5, gap bridging at 0.128 ms, acceptance at max z ≥ 12,
# frame-level concentration off, boundary expansion off. The bead build then
# adds four things this notebook has not shown, each answering a bead-specific
# problem:
#
# | | Yeast preset | Bead MAD v2.1 |
# |---|---|---|
# | boundary pad | 0.04 ms (80 samples/side) | **0.0** — boxes are exactly the grouped frames |
# | splitting one group | — | **deblending**: a group whose z profile holds two peaks ≥ 0.25 ms apart across a deep enough valley is cut in two, each half kept only if narrow-band (≤ 12 kHz) and concentrated (≥ 0.8) |
# | weak events | — | **unified rescue**: an event with 7 ≤ max z < 12 is admitted if it is narrow-band and highly concentrated — which is why the bead corpus holds events down to z = 7.1 |
# | events per trace | 5 strongest kept | **no cap** — hence the 7-box trace below |
# | saturation | — | clipped traces repaired before annotation (255 of them), and 79 proposals vetoed for having their centre inside a repaired region |
# | crop | 8 192 → 4 096 pts, clamped at edges | 6 144 pts, **reflect-padded** at edges |
#
# Read the additions for what they are: beads are dense enough that two
# passages regularly land inside one activated group, so the chain needed a
# way to separate them *after* localisation — which is exactly the failure the
# introduction blamed peak-first for, solved on the energy side of the fence.

# %%
evidence = workspace.root / ".cache/notebooks"
gallery_png, comparison_png = evidence / "mad-v21-gallery.png", evidence / "mad-v21-comparison.png"
dataset_root = resolve_registered_dataset(workspace)[1]
render_gallery(select_cases(read_rows(dataset_root / "source_manifest.csv"),
                            read_rows(dataset_root / "events.csv")),
               dataset_root, gallery_png, "MAD teacher v2.1")
Image(filename=str(gallery_png), width=1050)

# %% [markdown]
# **The chain transfers.** Nothing was retuned per class: the same z ≥ 3.5 and
# the same max z ≥ 12 produce these boxes at 2, 4 and 10 µm. The six cases are
# picked by a deterministic rule before anything is drawn: the median-energy
# event of each bead class, then the densest trace, the weakest event still
# retained, and one starting at sample 0. Each box carries its own outline,
# so adjacent boxes stay countable — seven on the densest trace, which is the
# uncapped configuration and the deblending rule both visible at once.

# %%
comparison = json.loads((workspace.root / "artifacts/cross-project/analysis"
    / "particle-p2-noise-tradeoff-evidence-analysis-r5/tradeoff_comparison.json").read_text())
assert comparison["mad_dataset_id"].endswith("@v2.1")
render_comparison(comparison, workspace.root, comparison_png)
Image(filename=str(comparison_png), width=1050)

# %% [markdown]
# **And the counting failure is measurable at scale.** Each row is a locus a
# human read as holding **exactly two particles** — the multi-crest situation
# of the opening, scored here by *count* (the labels fix how many particles a
# locus holds, not where their boundaries lie). MAD returns two boxes on all
# seven loci. P2SNR tuned for recall reaches 86/87 events against MAD's 77 —
# and pays with 89 % more proposals, seven traces a human verified as empty,
# and one exact locus out of seven. Tuned for cleanliness it keeps the traces
# clean and loses the events instead. Neither setting holds both ends.
#
# *Retrospective, beads, one acquisition family. This supports choosing MAD
# as the pseudo-label detector, nothing wider.*

# %% [markdown]
# ## Appendix A1 · Glossary
#
# Every symbol the notebook introduces, in the order the chain uses them.
#
# | Symbol | Meaning | Built in |
# |---|---|---|
# | fₛ | sampling frequency, 2 MHz | §0 |
# | N | STFT window length, 512 samples | §3 |
# | H | hop between windows, 128 samples (64 µs) | §3 |
# | m | frame index — one window position, one spectrogram column | §3 |
# | k | frequency bin index — one spectrogram row | §3 |
# | w | Hann window — the taper applied before each FFT | §3 |
# | X(k, m) | complex STFT value at bin k, frame m | §3 |
# | P(k, m) | power \|X(k, m)\|² | §3 |
# | B_k | per-frequency baseline, the 25th percentile of row k over time | §4 |
# | P⁺(k, m) | positive excess, max(P − B_k, 0) | §4 |
# | E[m] | frame energy, Σ_k P⁺(k, m), smoothed over 3 frames | §5 |
# | median(E) | the trace's ordinary level — piece 1 of the scale | §6b |
# | MAD(E) | median absolute deviation — piece 2, the ordinary wandering | §6c |
# | `raw_mad` | MAD(E) unscaled | §6e |
# | `energy_scale` | 1.4826 × `raw_mad`, the divisor actually used | §6e |
# | z[m] | (E[m] − median(E)) / `energy_scale` — a pure number | §6d |
# | a[m] | activation, 1 when z[m] ≥ 3.5 | §7a |
#
# Two naming traps this codebase carries, worth repeating here: the fields
# named `*_snr*` (`active_snr_z`, `medium/strict_min_snr`, `snr_proxy`) all
# hold **z** values, not signal-to-noise ratios (§6h); and `raw_mad` and
# `energy_scale` have both been called "the MAD" while differing by 48 %
# (§6e).

# %% [markdown]
# ## Appendix A2 · The configuration, read from the object
#
# The table in §0 is prose, written by hand, and prose drifts: twice during
# this notebook's development a setting changed in the code while the table
# still described the old behaviour. The table below cannot drift — it is
# printed from the very `config` object every cell above was given.

# %%
from dataclasses import asdict

for name, value in asdict(config).items():
    default = getattr(type(config)(), name)
    mark = "" if value == default else "  ← differs from the dataclass default"
    print(f"{name:<32}{value!r:>12}{mark}")

# %% [markdown]
# The six marked lines are exactly what `review_calibrated_detection_config_v1()`
# changes relative to a bare `YeastDetectionConfig()` — the six the "Running on
# your own data" protocol warns about. Among them is
# `active_min_concentration = 0.0`, the frame-level concentration gate this
# notebook measured inert.
#
# `boundary_expansion_enabled = False` carries **no** marker, and that is
# informative rather than an omission: the reviewed retirement was applied to
# the dataclass default itself, so no preset has to opt out of it. The
# threshold it would use, `boundary_snr_z = 1.5`, is still listed — kept so
# the comparison behind the review stays reproducible (§7b).
