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
# **Central question: how does a raw acoustic trace become a bounded SSL
# example?** This notebook retraces the yeast event detector
# (`particles2snr.yeast_events`) stage by stage on one manifested record,
# following the same chain as the frozen explainer deck
# ([Yeast detector explainer v5](../../artifacts/cross-project/presentations/yeast-detector-explainer-v5/render-r1/Yeast_detector_explainer_v5.pdf)):
#
# `ISOLATE → LOCALISE → REFERENCE → AGGREGATE → NORMALISE → GROUP → CROP`
#
# What this notebook **does**: expose every intermediate quantity of the
# detector on a real record, with the exact development configuration.
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
# demonstrated optimum.*

# %% [markdown]
# ## Why energy first — the particles2SNR lesson
#
# The legacy pipeline (particles2SNR) reasons **peak-first**: every local
# Doppler maximum becomes a particle hypothesis, fitted as t₀ ± 2.5τ, then
# cleaned and merged. That representation makes one structural mistake: a
# yeast passage can produce **several exact Doppler crests for one single
# particle**, and the moment each crest becomes a separate hypothesis, the
# information that they co-occurred as one burst of energy is gone.
# Everything downstream — passage-time filter, width filter, peak-evidence
# refinement, dual-clean, temporal NMS — is an attempt to reconstruct that
# lost information with heuristics, each carrying thresholds that interact
# with all the others.
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
#   **five** local P0 frequency hypotheses.
# - After every cleaning stage (filtered evidence, dual-clean, NMS), still
#   **two boxes for one single event**: the merge fails at IoU 0.399 < 0.400.
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
#   MAD chain recovered **78/87 confirmed events against 41/87** for the
#   historical Z8v2 peak stack, with exact per-trace cardinality on 44/60
#   traces (manifested audit `particle-mad-gt-visual-audit-r2`, frozen
#   metrics in `particle-gradual-wave8like-final-analysis-r1`).
#   *Development evidence on one acquisition family; the audit's local
#   AP@0.5 covers 11 reliable human supports and is not a corpus-wide mAP.*
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
# | m | **frame** index — one window position, one column | 0 … 127 |
# | k | **frequency bin** index — one row | 0 … n_bins−1, spaced fₛ/N ≈ 3.9 kHz |
# | w | Hann window | — |
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
print(f"max z outside the burst: {np.delete(demo.energy_z, np.argmax(demo.energy_z) + np.arange(-10, 10)).max():.1f}")
fig.plot_energy_and_z(demo);

# %% [markdown]
# - The 18 kHz row carries a baseline four orders of magnitude above its
#   neighbours: the tone was learned as background, its excess is ≈ 0, and z
#   stays below 2 everywhere except the true burst. The nuisance never
#   becomes a candidate — **no peak threshold was needed, so no real weak
#   event was put at risk by one**.
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
# - The final answer is right *on this toy*, but it is right because five
#   filters agreed. The introduction's yeast record is what happens when the
#   cascade meets a case its thresholds cannot arbitrate.
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
# - The energy chain activates on both: the slow event sits at z ≈ 1.5·10³,
#   three orders of magnitude above the 3.5 threshold. No gate on speed,
#   width or peak shape was consulted to *detect* it.
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
# E[m] has arbitrary units: it depends on gain, coupling, bead concentration.
# A fixed threshold on E would change meaning with every acquisition. The
# detector needs a *reference level* and a *scale* that the event itself
# cannot corrupt.
#
# ### 6a · The median does not follow events
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
#
# ### 6b · The MAD: a robust scale
#
# MAD(E) = median_i(|E_i − median(E)|) — the median of the distances to the
# median.

# %%
distances = np.abs(toy - np.median(toy))
print("distances to the median:", np.sort(distances))
print("MAD =", np.median(distances))

# %% [markdown]
# The large distance (18) exists but cannot occupy the middle of the sorted
# list. That is what robustness means here — not a formula, a *position in a
# sort*.
#
# ### 6c · On the real signal

# %%
fig.plot_robust_band(trace);

# %% [markdown]
# - The purple band (median ± 1 raw MAD) hugs the noise floor so tightly it is
#   barely visible at this scale — the event exits it by dozens of band-widths.
#
# **Vocabulary, fixed once and for all:** this band uses `raw_mad`, the
# unscaled median absolute deviation. The detector divides by
# `energy_scale = 1.4826 × raw_mad`. Two quantities, two names — both have
# historically been called "MAD", 48 % apart.
#
# ### 6d · Where 1.4826 comes from
#
# 1 / Φ⁻¹(0.75) ≈ 1.4826: for Gaussian noise, 1.4826 × MAD estimates the
# standard deviation σ.

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
# ### 6e · What the normalisation is invariant to — and what it is not ★
#
# Multiply the signal by 3 (a gain change): power scales by 9, so E[m] scales
# by 9 — and so do the median and the MAD. The ratio z[m] is unchanged.

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
# ### 6f · What z is not
#
# - z is **not an SNR** in the classical sense (no signal/noise power ratio).
# - z is **not a probability**.
# - z is **not comparable between two traces** with different noise floors.
#
# z[m] is the distance of E[m] to this trace's own baseline, in units of this
# trace's own robust scale. Nothing more — and for thresholding, nothing less.
#
# ### 6g · Synthesis

# %%
fig.plot_energy_and_z(trace);

# %% [markdown]
# - Top: E[m] with the median and the ± energy_scale band (now the scaled one).
# - Bottom: the same curve in z units. **Read z[m] as: "how many robust
#   scales does this frame sit above the ordinary level of this trace?"**
#   The red dashed line is the activation threshold z = 3.5 used in section 8.
#
# *Method only · one manifested record · no claim of detector performance or
# biological validity.*

# %% [markdown]
# ## Running on your own data
#
# Everything above used the workspace machinery only to fetch a *manifested*
# example. The method itself needs a plain 1-D array. The protocol:
#
# 1. A 1-D float array (any amplitude units — section 6e showed why gain does
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
