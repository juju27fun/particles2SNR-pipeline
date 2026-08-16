# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
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
# | Grouping | bridge gaps ≤ 0.128 ms (2 frames) |
# | Qualification | width 0.06–2.0 ms · max z ≥ 12 |
#
# *These values are a development choice validated on a review campaign, not a
# demonstrated optimum.* The last rows name quantities the notebook has not
# built yet — z is constructed in section 6, the qualification quantities in
# section 7; the table is here as a reference to come back to, not as
# something to read now.
#
# Every setting listed is one that changes an outcome. The preset also carries
# three that do not, and the table leaves them out rather than implying they
# arbitrate anything: two concentration floors that no event on this corpus
# has ever come near, and the boundary pad, now 0.0 like the published MAD
# v2.1 contract, so **a box is exactly the frames activation selected —
# nothing is ever added to it**. Appendix A2 prints the full object, inert
# fields included.

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
# The candidate rule is the obvious one: **flag every frame whose energy
# reaches some fixed value**. Recall from section 3 that a frame is one column
# of the spectrogram, 64 µs of trace, so the number of flagged frames is simply
# how much of the record the rule calls "event".
#
# Pick that value at half the event's peak on this record — as good a
# hand-picked choice as any — then apply the *same* value to the same trace
# recorded with three times more gain, and three times less. Nothing about the
# particle changes; only the recording chain does.

# %%
energy_threshold = 0.5 * trace.frame_energy.max()
gains = [(label, detector_trace(record.signal * factor, config))
         for label, factor in (("original", 1.0), ("gain x3", 3.0), ("gain /3", 1 / 3))]
fig.plot_fixed_threshold(gains, energy_threshold);

# %% [markdown]
# The three curves are the same particle. The red line is the same threshold on
# all three. Turning up the gain lifts the whole curve — noise floor included —
# over a line that stayed put, so the rule calls **16 frames** an event instead
# of 4. Turning it down drops the entire curve underneath: **the rule finds
# nothing at all**, on a trace where the particle is perfectly visible.
#
# The chosen value is not a bad one; *any* fixed energy value has this defect,
# because energy units are a property of the recording chain and not of the
# particle. The rule has to be expressed relative to the trace it is applied
# to, which means estimating two things from the trace itself:
#
# 1. its **ordinary level** — what E[m] looks like when nothing happens;
# 2. its **spread** — how much E[m] normally wanders around that level.
#
# Both estimates must survive the presence of the event: if the event drags its
# own reference upwards, it hides itself. That requirement — not tradition — is
# why the next subsection uses the median and the MAD rather than the mean and
# the standard deviation.
#
# ### 6b · Two statistics the event cannot move
#
# Both numbers must be read off the trace *while the event is in it*. So the
# test is simple: compute each one on a calm series, then again after a single
# frame lights up, and see which ones move.
#
# Every level comes with a companion measure of spread, built the same way:
#
# | Level | Its companion spread | Built as |
# |---|---|---|
# | mean | standard deviation | square every distance to the mean, average, take the square root |
# | median | **MAD** = *median absolute deviation* | the **median** distance to the median |

# %%
calm = np.array([4, 5, 5, 6, 5, 4, 6], dtype=float)
with_event = np.array([4, 5, 5, 6, 5, 4, 40], dtype=float)   # one frame lights up


def mad(values: np.ndarray) -> float:
    return float(np.median(np.abs(values - np.median(values))))


print(f"calm       : {calm.astype(int)}")
print(f"one event  : {with_event.astype(int)}   (only the last frame differs)\n")
print(f"{'':<20}{'calm':>9}{'one event':>12}{'change':>10}")
print("-" * 51)
for name, statistic in (("mean", np.mean), ("standard deviation", np.std),
                        ("median", np.median), ("MAD", mad)):
    before, after = float(statistic(calm)), float(statistic(with_event))
    print(f"{name:<20}{before:>9.2f}{after:>12.2f}{(after - before) / before * 100:>9.0f} %")
print(f"\ndistances to the median: {np.sort(np.abs(with_event - np.median(with_event)))}")

# %% [markdown]
# One value out of seven changed. The mean nearly doubled and the standard
# deviation grew fifteenfold — the event inflated the very scale meant to
# measure it. **The median and the MAD did not move at all.**
#
# The last line says why, and it is the whole idea: the large distance (35)
# exists, but it cannot occupy the *middle* of the sorted list. Robustness here
# is not a formula, it is a position in a sort. Squaring, as the standard
# deviation does, has the opposite effect — it hands the single largest
# distance control of the result.
#
# **Both pieces acquired**: median(E) is an ordinary level the event cannot
# drag upwards, MAD(E) an ordinary wandering it cannot inflate.
#
# ### 6c · z, and what it survives
#
# Assembling them is one subtraction and one division. Measure how far a frame
# is above the level, then express that distance *in units of the wandering*
# instead of in units of energy:
#
# $$ z[m] \;=\; \underbrace{\frac{E[m] - \mathrm{median}(E)}{\vphantom{X}}}_{\text{how far above ordinary}} \Bigg/ \underbrace{\big(1.4826 \times \mathrm{MAD}(E)\big)}_{\text{one ordinary wandering}} $$
#
# (The 1.4826 is a scale convention, explained in 6d; it changes nothing here.)
# The essential point is what happens to the units: numerator and denominator
# are **both in energy units, so they cancel**. z[m] is a pure number, which is
# exactly what 6a could not obtain:
#
# > **z[m] = how many ordinary wanderings above its own ordinary level this
# > frame sits.**
#
# The cell below puts that to three tests: 6a's broken experiment run again
# with both rules, then the underlying identity to numerical precision, then a
# transformation z is *not* meant to survive.

# %%
print(f"{'recording':<12}{'flagged by the fixed value':>27}{'flagged by z >= 3.5':>21}")
for label, factor in (("original", 1.0), ("gain x3", 3.0), ("gain /3", 1 / 3)):
    scaled = detector_trace(record.signal * factor, config)
    print(f"{label:<12}{int((scaled.frame_energy >= energy_threshold).sum()):>27}"
          f"{int((scaled.energy_z >= config.active_snr_z).sum()):>21}")

gained = detector_trace(record.signal * 3.0, config)
assert np.allclose(gained.energy_z, trace.energy_z, rtol=1e-4, atol=1e-6)
print(f"\namplitude x3 -> E[m] x{np.median(gained.frame_energy / trace.frame_energy):.0f}, "
      f"but max |delta z| = {float(np.max(np.abs(gained.energy_z - trace.energy_z))):.1e}")

rng = np.random.default_rng(1)
extra = rng.normal(scale=3.0 * float(record.signal.std()),
                   size=record.signal.size).astype(np.float32)
noisy = detector_trace(record.signal + extra, config)
print(f"adding noise instead -> peak z falls from {trace.energy_z.max():.1f} "
      f"to {noisy.energy_z.max():.1f}")

# %% [markdown]
# Read the two columns downwards. The energy rule answers **4, then 16, then
# 0** — it disagrees with itself about a particle that never changed. The z
# rule answers **14, 14, 14**. Amplitude ×3 means power ×9, so E[m], its median
# and its MAD all scale by 9 together and the ratio cannot move: the identity
# holds to 10⁻⁴.
#
# The last line is the honest limit. z is **not** invariant to added noise, and
# should not be: the MAD follows the noise floor up, so the same event stands
# out less. z measures contrast against *this trace's* noise floor — a gain
# change preserves it, a noisier recording does not.
#
# So z is worth stating plainly: **it is not a new measurement of the signal,
# it is E[m] in a different unit** — one each trace derives from its own quiet
# stretches. Changing unit is what makes a threshold portable; it adds no
# information and detects nothing by itself.
#
# One estimation detail, stated once because every z in this notebook lives on
# it: E[m] was smoothed over 3 frames in section 5, *before* the median and the
# MAD are taken. The scale therefore measures the wandering of the smoothed
# series, which is smaller than raw frame-to-frame noise, so z values run
# larger than they would against unsmoothed frames. The convention is
# consistent, but z magnitudes are not comparable to a smoothing-free
# implementation.
#
# ### 6d · On the real signal, and where 1.4826 comes from
#
# Zoomed to the noise floor, because at full scale both bands are a single
# line. The event's peak sits at z = 75.5 — that is 75.5 × 1.4826 ≈ **112
# raw-MAD widths** above the median, far off this axis:

# %%
fig.plot_robust_band(trace, both=True);

# %% [markdown]
# **Vocabulary, fixed once and for all.** The two purple bands are the two
# quantities that have both been called "the MAD": the inner one is `raw_mad`,
# the unscaled median absolute deviation; the outer one is
# `energy_scale = 1.4826 × raw_mad`, the divisor the detector actually uses.
# Two names, 48 % apart.
#
# Where that constant comes from: on Gaussian noise the MAD lands at about
# 0.67 σ, so 1.4826 rescales it to agree with the standard deviation — it puts
# the MAD "on the σ ruler". For a Gaussian, half the values fall within
# 0.6745 σ of the centre, so that distance *is* the MAD, and 1 / 0.6745 ≈
# 1.4826 restores σ.

# %%
draw = np.random.default_rng(0).normal(size=100_000)
print(f"on 100 000 Gaussian draws: std = {draw.std():.4f}   "
      f"1.4826 x MAD = {1.4826 * mad(draw):.4f}")

# %% [markdown]
# They coincide. **The constant only has meaning under a Gaussian hypothesis;
# the detector keeps it as a scale convention, not as a model of the noise.**

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
# contributing its whole window. That region *is* the event: no later stage
# widens it, so this figure and the interval of 7c agree to the sample.

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
# Naturally, the z threshold gives the duration of the event
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
# Two operations, and that is the whole of it.
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
# There is no third step. **The box is the active frames, and nothing is added
# to it** — the same rule the published MAD v2.1 contract states for beads.
#
# > **Two mechanisms used to sit here, and neither does now.** The detector
# > once grew each interval outwards while z stayed above a lower threshold
# > (1.5 against 3.5), and a fixed 0.04 ms pad then added 80 further samples
# > per side. Retiring them changed no event count over 6 172 traces. What it
# > removed is worse than width: growing each group outwards independently let
# > **neighbouring groups grow into each other**, so 143 traces ended up with
# > two boxes on identical bounds and 990 box pairs overlapped — 108 duplicated
# > boxes reached the training set itself. With both mechanisms off there are
# > none.

# %%
bounds, = event_bounds(trace, record.signal.size)
group_left, group_right = bounds.group
print(f"activation run : frames {group_left}–{group_right}  ({group_right - group_left + 1} frames)")
print(f"the event      : [{bounds.start:5d}, {bounds.end:5d}]  {bounds.end - bounds.start:5d} "
      f"samples  {(bounds.end - bounds.start) / 2e6 * 1000:.3f} ms")
assert (bounds.start, bounds.end) == bounds.group_samples, "a stage widened the box"

# %% [markdown]
# The assertion is the point: the interval *is* the activation run. Every
# boundary in this notebook is a direct consequence of one threshold, z ≥ 3.5,
# which is the whole point of section 6 having built z carefully.
#
# ### 7c · The output, back on the record

# %%
record_events, record_reason = detect_yeast_events(record.signal, config)
assert not record_reason, record_reason
fig.plot_detected_events(record.signal, record_events, config);

# %% [markdown]
# The green bar is the bounded event the notebook set out to produce, and it is
# **the same span the figure of 7a shaded** — [7040, 9216] — because nothing
# between the two figures widens it.
#
# ### 7d · QUALIFY — the interval still has to earn its label
#
# A bounded interval is a *candidate*. Two tests decide its label, and they
# run **after** the event exists, so a failure is recorded against a localised
# object instead of deleting a hypothesis:
#
# | Reported test | Preset | On this event |
# |---|---|---|
# | width within 0.06–2.0 ms | `min/max_width_ms` | 1.088 ms ✔ |
# | max z over the event ≥ 12 | `strict/medium_min_snr` | 75.5 ✔ |
#
# **Two z thresholds, two different jobs — this is the one thing to get
# straight.** They are easy to confuse because they are the same quantity:
#
# | | z ≥ 3.5 (§7a) | max z ≥ 12 (here) |
# |---|---|---|
# | asked of | every **frame**, one at a time | the **event**, once it exists |
# | question | is this frame part of an event? | is this event strong enough to keep? |
# | decides | *where* the boundaries fall | *whether* there is a label at all |
#
# So 3.5 draws the box and 12 accepts or rejects it. Our record answers 75.5
# to the second question, which looks like a strange thing to compare to 12 —
# **and that is the correct reading: this event is not near the line.** A
# clean single passage never is. The floor is not there for this event; it is
# there for the intervals that activation localises out of a rise that is
# real but far too weak to become a training label.
#
# **How live is that floor?** Over the whole registered acquisition, of the
# 13 226 intervals the chain localises, **3 881 — 29 % — never reach z = 12
# and are rejected**, and 826 of them land in the narrow window [10, 14)
# where the decision is genuinely close. The median accepted event sits at
# z = 47; 75.5 is an unremarkable strong passage. Here is a record where the
# floor does the work at close range:

# %%
near_floor = load_reviewed_event(workspace, "864ceffff54958c218e8:02").signal
near_events, near_reason = detect_yeast_events(near_floor, config)
assert not near_reason, near_reason
for e in near_events:
    verdict = "accepted" if e.quality != "reject" else f"rejected ({e.rejection_reason})"
    print(f"max z = {e.snr_proxy:5.1f}   vs floor {config.strict_min_snr:.0f}   → {verdict}")
fig.plot_detected_events(near_floor, near_events, config);

# %% [markdown]
# Three intervals, all genuinely localised, all straddling the floor: 7.0 and
# 9.1 are rejected, 14.5 is kept. Nothing separates them but that one number —
# which is exactly what a threshold is, and why the notebook calls 12 a
# development choice rather than a discovered constant. Elsewhere in the
# corpus an event reaches **z = 11.7** and is rejected for it; move the floor
# by 0.3 and it becomes a training label.
#
# On the yeast corpus, every rejection that is not a width comes down to
# that z floor.
#
#
# ### 7e · CROP — the bounded event becomes a training example
#
# **Start from what we actually hold.** One record is **16 384 points sampled
# at 2 MHz** — one point every 0.5 µs, 8.192 ms of trace. That is the file as
# acquired, with no resampling applied anywhere upstream.
#
# The detector's interval has a variable width; a model needs a fixed-length
# input. The dataset contract (`yeast-event-8192to4096-bandpass-global-v1`)
# resolves that in two steps, different in kind, and each one discards
# something specific:
#
# 1. **A fixed window in time** — 8 192 points (4.096 ms) centred on the
#    event's energy-weighted centre, *not* on the middle of its interval.
# 2. **A change of resolution** — that window is band-passed (5–100 kHz,
#    Butterworth order 4, zero-phase) and resampled by `resample_poly(down=2)`,
#    giving **4 096 points at 1 MHz**. The duration is unchanged; only the
#    sampling rate is. A final global normalisation (development-train mean
#    and standard deviation) is applied on top. The figure below shows step 1
#    only.

# %%
fig.plot_ssl_crop(record.signal, record_events[0], config);

# %% [markdown]
# - Pale green: the detected event, 2 176 samples. Pale blue: the
#   8 192-sample window handed to the model. Dashed line: the
#   energy-weighted centre.
# - Why the crop is wider than the event: a model given only the event would
#   never see what ordinary signal looks like, and the boundary itself would
#   become a learnable artefact. The margin carries that context.
# - Why centre on the energy-weighted centre rather than the interval's
#   midpoint: energy is not distributed evenly inside the interval. Honest
#   scale, though — both centres live on the 128-sample frame grid, and here
#   they differ by 121 samples, less than one hop. Read the centring as a
#   design principle, not something this record proves.
#
# **A crop is not one event.** The window is 4.096 ms wide and passages are not
# spread out politely, so on the 8 669 registered events **37 % of crops contain
# a second event** — 12 % contain one cut in half by the crop edge. And when two
# events are close, each gets its own crop centred on it, so the same stretch of
# signal is stored twice: 12 % of crops overlap another one at IoU ≥ 0.75.
#
# The dataset records this rather than removing it. Each row carries how many
# neighbours its crop holds and where they sit inside it, so a consumer can
# filter or deduplicate. Erasing them instead would mean painting MAD's own
# boxes into the signal — and where MAD is wrong, that deletes real events from
# an input whose whole purpose is to be unlabelled.
#
# The bead pipeline avoids the question at this stage by not cropping at all:
# it labels every box on the whole trace. It meets the same problem one stage
# later, at the proposal classifier, and answers it there — a crop overlapping
# a real event too partially to call is marked ambiguous and dropped from
# training.
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
# 1. A 1-D float array (any amplitude units — section 6c showed why gain does
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
fig.plot_detected_events(your_signal, your_events, config);

# %% [markdown]
# No registry, no review queue, no workspace: one array, one config, the same
# eight stages, one bounded event. Replace `your_signal` with your own
# acquisition and the whole notebook's reasoning applies unchanged.
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
# | splitting one group | — | **deblending**: a group whose z profile holds two peaks ≥ 0.25 ms apart across a deep enough valley is cut in two, each half kept only if narrow-band (≤ 12 kHz) and concentrated (≥ 0.8) |
# | weak events | — | **unified rescue**: an event with 7 ≤ max z < 12 is admitted if it is narrow-band and highly concentrated — which is why the bead corpus holds events down to z = 7.1 |
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
# so adjacent boxes stay countable — seven of them on the densest trace.

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
# | MAD(E) | median absolute deviation — piece 2, the ordinary wandering | §6b |
# | `raw_mad` | MAD(E) unscaled | §6d |
# | `energy_scale` | 1.4826 × `raw_mad`, the divisor actually used | §6d |
# | z[m] | (E[m] − median(E)) / `energy_scale` — a pure number | §6c |
# | a[m] | activation, 1 when z[m] ≥ 3.5 | §7a |
#
# Two naming traps this codebase carries, worth repeating here: the fields
# named `*_snr*` (`active_snr_z`, `medium/strict_min_snr`, `snr_proxy`) all
# hold **z** values, not signal-to-noise ratios; and `raw_mad` and
# `energy_scale` have both been called "the MAD" while differing by 48 %.

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
# The seven marked lines are exactly what `review_calibrated_detection_config_v1()`
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
