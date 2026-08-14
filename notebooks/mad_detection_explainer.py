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
# # MAD detection — from a raw trace to a bounded event
#
# *MAD stands for **median absolute deviation**: a way of measuring how much
# a signal normally fluctuates that a rare large event cannot distort. It is
# what lets the detector say "this is unusually high" without anyone fixing a
# value by hand. Section 6 builds it from nothing; sections 1–5 are what has
# to happen before it can be applied.*
#
# **Central question: how does a raw acoustic trace become a bounded training
# example?** — bounded meaning *we know where the event starts and ends*, which
# is what a self-supervised learning (SSL) model needs as input.
# This notebook retraces the yeast event detector
# (`particles2snr.yeast_events`) stage by stage on one manifested record,
# following the same chain as the frozen explainer deck
# ([Yeast detector explainer v5](../../artifacts/cross-project/presentations/yeast-detector-explainer-v5/render-r1/Yeast_detector_explainer_v5.pdf)):
#
# `ISOLATE → LOCALISE → REFERENCE → AGGREGATE → NORMALISE → GROUP → CROP`
#
# What this notebook **does**: expose every intermediate quantity of the
# detector on a real record, with the exact development configuration.
# *Manifested* record, below, means one whose origin is recorded in the
# workspace registry — you can trace which acquisition it came from and which
# human reviewed it.
#
# What it **does not do**: it produces no manifested artifact, no performance
# metric, no detector comparison, and no biological claim. It explains a
# method; it does not convert development evidence into validation. Every
# figure below is bound by that same claim boundary: *method only, one
# manifested record*.
#
# **What you need to run it.** Two different things, often confused:
#
# - *This notebook* loads its example record by ID through the workspace
#   dataset registry and the frozen review queues — so it needs the internship
#   workspace checkout. That machinery exists for **provenance of the
#   examples**, not for the method.
# - *The detector itself* needs none of that: `detector_trace(signal, config)`
#   accepts any 1-D numpy array. The last section, "Running on your own
#   data", shows the whole chain on a synthetic trace built from scratch.

# %%
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image
from internship_workspace.config import Workspace

from particles2snr.yeast_events import (
    detector_trace,
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
# | Activation | z ≥ 3.5 **and** C ≥ 0.08 |
# | Boundaries | extend while z ≥ 1.5, pad 0.04 ms |
# | Grouping | bridge gaps ≤ 0.128 ms (2 frames) |
# | Qualification | width 0.06–2.0 ms, max z ≥ 12, ≤ 5 events per trace |
#
# *These values are a development choice validated on a review campaign, not a
# demonstrated optimum.* The last three rows name quantities the notebook has
# not built yet — z is constructed in section 6, C in section 7; the table is
# here as a reference to come back to, not as something to read now.

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
#   MAD chain recovered **78/87 confirmed events against 41/87** for Z8v2
#   (the historical peak-first detector), and returned the *right number* of
#   events on 44 of 60 traces (manifested audit
#   `particle-mad-gt-visual-audit-r2`, frozen metrics in
#   `particle-gradual-wave8like-final-analysis-r1`).
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
# The reviewed event occupies only a fraction of the record. We call its sample
# interval the **temporal support**.

# %%
fig.plot_event_support(record.signal, config,
                       event_start=event_start, event_end=event_end);

# %% [markdown]
# - The green span is the human-reviewed support: the target the detection
#   chain must recover from the raw trace alone.

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
fig.plot_legacy_vs_energy(demo, replay=replay_tone, truth_spans_ms=[(4.6, 5.4)]);

# %% [markdown]
# - The tone spawns hypotheses all along the trace: **8 raw hypotheses for 1
#   real event**. The cleaning cascade has to execute every spurious one —
#   here six fall to the box-width gate and one to peak evidence.
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
fig.plot_legacy_vs_energy(slow_trace, replay=replay_slow,
                          truth_spans_ms=[(1.7, 3.3), (5.6, 6.4)]);

# %% [markdown]
# - The legacy pipeline *sees* the slow particle at the raw stage — then the
#   **box-width gate deletes it** (its fitted box is longer than 1.5 ms), and
#   the final output contains only the fast particle. A real event was
#   removed by a threshold calibrated on typical ones.
# - The energy chain produces a clear bump for **both** events, the slow one
#   included: no gate on speed, width or peak shape was consulted to see it.
#   Turning a bump into a decision still needs a rule — that rule is exactly
#   what section 6 builds.
# - Honesty notes: this is a constructed illustration, not a prevalence
#   claim — how often real acquisitions contain such atypical events is an
#   open question. And the MAD chain's own qualification (section 9) also
#   carries width limits (0.06–2.0 ms), wider but real: the argument is
#   about *where* selectivity lives — in a statistical contrast, or in a
#   stack of proxy gates — not about the energy chain having no limits.

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
# the rule calls "event"** — here the event is genuinely about 4 frames wide.
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

# %%
fig.plot_robust_band(trace);

# %% [markdown]
# - The purple band (median ± 1 raw MAD) hugs the noise floor so tightly it is
#   barely visible at this scale — the event's peak sits about **112 raw-MAD
#   widths** above the median.
#
# **Vocabulary, fixed once and for all:** this band uses `raw_mad`, the
# unscaled median absolute deviation. The detector divides by
# `energy_scale = 1.4826 × raw_mad`. Two quantities, two names — both have
# historically been called "MAD", 48 % apart.
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
# ### 6i · Synthesis

# %%
fig.plot_energy_and_z(trace);

# %% [markdown]
# - Top: E[m] with the median and the ± energy_scale band (now the scaled one).
# - Bottom: the *same* curve, divided by the scale — no new measurement, just
#   a change of unit. The red dashed line is the activation threshold z = 3.5
#   used in section 8, and it is only writable because of that change.
#
# *Method only · one manifested record · no claim of detector performance or
# biological validity.*

# %% [markdown]
# ## 7 · STRUCTURE — the concentration C[m]
#
# Everything up to here is shared by **two** detectors in this project: one
# for yeast (this notebook) and one for beads
# (`particles2snr.particle_events`). They are the same architecture:
#
# `STFT → E[m] → z[m] → [ C[m] mask — optional ] → grouping`
#
# The bracketed stage is the subject of this section, and the brackets are
# the point: it is a stage you switch on or off per population, not a
# mandatory link.
#
# **The question C asks.** z says a frame is unusually energetic. It does not
# say *how* that energy is arranged across frequency. A particle deposits
# energy in a few Doppler bins; a click, a saturation transient or a broadband
# knock spreads it everywhere. C measures that difference:
#
# C[m] = (power in the 5 strongest bins) / (total band power in frame m)
#
# Two frame spectra carrying exactly the same total power:

# %%
peaked = np.array([1, 1, 2, 40, 60, 45, 3, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1], dtype=float)
diffuse = np.full(19, peaked.sum() / 19)
fig.plot_concentration_toy({"structured — a Doppler signature": peaked,
                            "diffuse — a broadband knock": diffuse});

# %% [markdown]
# Same total energy, so **E[m] cannot tell them apart** — z would score both
# identically. C separates them: the peaked frame keeps most of its power in
# five bins, the flat one only the share those five bins occupy by chance.
#
# ### 7a · C on the real record

# %%
fig.plot_concentration(trace);

# %% [markdown]
# **And here is the surprise.** C never comes close to the configured floor
# of 0.08 — its lowest value on this whole trace is 0.20. The test is
# structurally satisfied everywhere. Is that specific to this record?

# %%
print(f"{'record':<26}{'lowest C':>10}{'active z & C':>14}{'active z alone':>16}")
for event_id in ("9459e76ce29342debc90:00", "214f4ce4967af98a954c:00",
                 "e1b4603f8b9de6204003:02", "09f788a7473797b794f6:01"):
    other = detector_trace(load_reviewed_event(workspace, event_id).signal, config)
    with_c = int(((other.energy_z >= 3.5) & (other.concentration >= 0.08)).sum())
    z_only = int((other.energy_z >= 3.5).sum())
    print(f"{event_id:<26}{other.concentration.min():>10.3f}{with_c:>14}{z_only:>16}")

# %% [markdown]
# On all four manifested records, **deleting the C test would change
# nothing**: the two right-hand columns are identical. At 0.08, C is not
# selecting anything — it is a *guard* that never had to fire. Section 7c
# repeats this over 1 316 traces and both detectors, with controls.
#
# There is a structural reason. The 7–80 kHz band holds only 19 STFT bins, so
# "the top 5" is already 26 % of the band by construction; C is bounded well
# above 0.08 unless the spectrum is close to perfectly flat *and* the raw
# power denominator is dominated by bins outside the top five.
#
# ### 7b · The denominator is not what you would guess
#
# > ```python
# > top_power     = top-5 of excess        # positive excess P⁺
# > concentration = top_power / (power.sum(axis=0) + 1e-12)   # raw power P
# > ```
# >
# > The numerator counts **excess** above the baseline; the denominator counts
# > **total raw power**, baseline included. C is therefore not a share of P⁺
# > within itself — that quantity would sit near 1 almost always and a 0.08
# > threshold would be meaningless. It is *the excess of the five best bins,
# > measured against everything the band carries*. Identical in all
# > implementations, so intentional — but documented nowhere until here.
#
# ### 7c · Same stage, three roles — and the threshold tells you which
#
# This is where the second pipeline matters. The bead detector runs the same
# activation line ([`particle_events.py`](../particles2snr/particle_events.py)):
#
# ```python
# active = (energy_z >= config.active_z) & (concentration >= config.active_min_concentration)
# ```
#
# with `active_min_concentration = 0.0`. The stage is present and neutralised
# — every frame passes. Across both pipelines the *value* of the threshold,
# not the presence of the code, says what C is doing:
#
# | Setting | Where |
# |---|---|
# | 0.0 | beads, main activation |
# | 0.08 | yeast activation (this notebook) |
# | 0.08 | beads, event acceptance |
# | 0.80 – 0.90 | beads, rescue and deblend paths |
#
# The natural story is that C is off for easy beads, a cheap guard for yeast,
# and a genuine selector in the rescue paths that deliberately lower z to
# reach hard cases. **Measured, that last claim is false.** An ablation —
# same detector, same data, every C threshold set to 0 — was run over:
#
# | Corpus | Scope | Traces where removing C changes the outcome |
# |---|---|---|
# | yeast `yeast-hf-10-5-20260610@v1` | 1 316 traces, 2 449 detected events | **0** |
# | beads `…dual-clean-c1-yolo-4class@v2` | 600 traces, deblend + unified rescue enabled | **0** |
#
# The bead rescue path is not dormant — it fires on 116 of those 600 traces
# and adds 73 events. It simply never rejects anything *because of C*. And
# the ablation is not blind: as positive controls, relaxing the rescue
# bandwidth limit changes 4 traces in 300 and lowering the rescue z from 7.0
# to 4.0 changes 6, so the method does detect a constraint that binds.
#
# C is not dead code either — tightening the rescue thresholds to 0.99
# changes 66 traces in 300. The test is evaluated, it is reachable, and it
# *can* bind. It simply sits far below everything the data produces: on yeast
# the lowest per-event C is 0.57 against thresholds of 0.08 and 0.12.
#
# **So should C be deleted?** The measurements say removing it would be free
# on every trace examined; they do not say it is useless. Two things to weigh:
#
# - C is **insurance against a failure mode this data does not contain** — a
#   frame with high total energy spread evenly across the band (a knock, a
#   saturation transient). Note that the bead corpus had saturation
#   *repaired upstream* on 213 traces before the detector ever sees it, which
#   plausibly removes the very cases C exists to catch.
# - What must not be claimed is that these thresholds are *tuned*. Any value
#   below roughly 0.5 gives bit-identical output. 0.08 is not an optimum, it
#   is a floor no observed frame has ever approached.
#
# Keeping a zero-cost guard is defensible; presenting it as a calibrated
# discriminator is not. That distinction is the honest content of this
# section.
#
# *This ablation is a development observation run offline over the corpora
# named above — it is not a manifested analysis run, and it says nothing
# about detection quality against ground truth.*
#
# ### 7d · One word, two quantities
#
# `concentration_by_frame` (this section, per frame, feeding activation) and
# the `concentration` reported per **event** at qualification are different
# quantities sharing a name. The per-event one divides by `event_power`
# alone, so it sits near 1 — which is why the qualification threshold of 0.12
# and the activation floor of 0.08 are not comparable numbers despite looking
# alike.

# %% [markdown]
# ## 8 · ACTIVATE — the two questions combined ★
#
# a[m] = 1 if **z[m] ≥ 3.5** *and* **C[m] ≥ 0.08**, else 0. Two independent
# questions: *unusual?* and *structured?*

# %%
fig.plot_activation(trace);

# %% [markdown]
# - Top and middle: each test with its threshold. Bottom: the frames that
#   pass both — one contiguous run, which section 9 will turn into an event.
# - Consistent with 7a, the middle panel never approaches its threshold: the
#   run is decided by z alone on this record.
#
# ### 8a · The sandbox
#
# Move the two thresholds and watch the active line. The trace is already
# computed, so only the decision is recomputed — nothing is re-analysed.

# %%
from ipywidgets import FloatSlider, interact


@interact(z_thr=FloatSlider(value=3.5, min=1.0, max=8.0, step=0.1, description="z ≥"),
          c_thr=FloatSlider(value=0.08, min=0.0, max=0.95, step=0.01, description="C ≥"))
def _explore(z_thr: float, c_thr: float) -> None:
    fig.plot_activation(trace, z_threshold=z_thr, c_threshold=c_thr)
    plt.show()

# %% [markdown]
# What the two sliders do is **not** symmetric:
#
# - **z sets the duration.** Lowering it from 3.5 to 2.0 lengthens the run at
#   both edges and eventually raises a second run elsewhere in the trace.
# - **C does nothing at all** until roughly 0.72, then starts cutting frames
#   *out of the real event* — 11 frames left at 0.80, 9 at 0.90. On this
#   record there is no setting of C that removes a false start while keeping
#   the event intact, because there is no false start for it to remove.
# - **Set C to 0.00 and you are running the bead activation rule.** The line
#   does not move. That is the two-pipeline claim of 7c, verified by dragging
#   a slider rather than asserted in a table.
#
# The values 3.5 and 0.08 are a development choice, not a demonstrated
# optimum — and this sandbox is a sensitivity check on one record, not a
# calibration.
#
# *Method only · one manifested record · no claim of detector performance or
# biological validity.*

# %% [markdown]
# ## Running on your own data
#
# Everything above used the workspace machinery only to fetch a *manifested*
# example. The method itself needs a plain 1-D array. The protocol:
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
your_trace = detector_trace(your_signal, config)
fig.plot_energy_and_z(your_trace);

# %% [markdown]
# No registry, no review queue, no workspace: one array, one config, the same
# seven stages. Replace `your_signal` with your own acquisition and the whole
# notebook's reasoning applies unchanged.
#
# ---
# **Sections 7–11 (concentration, activation with interactive thresholds,
# grouping, SSL crop, sandbox) follow after the formatting checkpoint.**
