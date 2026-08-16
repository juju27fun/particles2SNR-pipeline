"""Screen-density figures for the MAD detection explainer notebook.

Same palette and colour semantics as the frozen v5 explainer deck, at screen
density (11 x 4.5 in, 110 dpi) instead of projection density. Colour grammar:
PURPLE = robust/MAD/z, GREEN = median and detector output, RED = threshold on
z, ORANGE = rejection, BLUE = signal/energy, GREY = raw trace. Span tints obey
one rule everywhere: PALE_BLUE = a span that was GIVEN (reviewed support,
declared truth, a context window), PALE_GREEN = a span the chain COMPUTED
(active run, detected interval, MAD box).

Every helper draws into the axes it is given (``ax=`` / ``axes=``) so
interactive cells can redraw without recreating figures; with the default
``None`` it creates its own figure.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from .yeast_events import DetectorTrace, YeastDetectionConfig

NAVY = "#0A335C"
GREY = "#5B6773"
LIGHT_GREY = "#D9E1E7"
BLUE = "#2F6FED"
PURPLE = "#7651C8"
TEAL = "#22A6A1"
GREEN = "#0F8A70"
ORANGE = "#D97706"
RED = "#FF3654"
PALE_GREEN = "#DDF2EC"
PALE_BLUE = "#EAF1FD"

FIGSIZE = (11.0, 4.5)
DPI = 110

plt.rcParams.update(
    {
        "figure.dpi": DPI,
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "mathtext.fontset": "cm",
        "axes.edgecolor": NAVY,
        "axes.labelcolor": NAVY,
        "xtick.color": NAVY,
        "ytick.color": NAVY,
        "legend.frameon": False,
    }
)


def time_axis_ms(n_samples: int, config: YeastDetectionConfig) -> np.ndarray:
    return np.arange(n_samples) / config.sampling_frequency_hz * 1000.0


def frame_axis_ms(trace: DetectorTrace) -> np.ndarray:
    return trace.times * 1000.0


def _single_axis(ax: Axes | None) -> Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=FIGSIZE)
    return ax


def _stacked_axes(axes: np.ndarray | None, nrows: int, height: float = 6.5) -> np.ndarray:
    if axes is None:
        _, axes = plt.subplots(nrows, 1, figsize=(FIGSIZE[0], height), sharex=True)
    return np.atleast_1d(np.asarray(axes))


def plot_trace_overview(
    signal: np.ndarray,
    config: YeastDetectionConfig,
    *,
    zoom_center: int | None = None,
    zoom_halfwidth: int = 25,
    support: tuple[int, int] | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Full raw trace, an inset zoom on samples, optionally a given span."""
    ax = _single_axis(ax)
    x_ms = time_axis_ms(signal.size, config)
    if support is not None:
        start, end = support
        width_ms = (end - start) / config.sampling_frequency_hz * 1000.0
        ax.axvspan(x_ms[start], x_ms[min(end, signal.size - 1)], color=PALE_BLUE,
                   zorder=0, label=f"reviewed support · {width_ms:.2f} ms")
        ax.legend(loc="upper left")
    ax.plot(x_ms, signal, color=GREY, linewidth=0.6)
    ax.set_xlabel("time [ms]")
    ax.set_ylabel("amplitude [a.u.]")
    ax.set_title(f"Raw trace — {signal.size} samples at {config.sampling_frequency_hz / 1e6:.0f} MHz")
    center = signal.size // 2 if zoom_center is None else int(zoom_center)
    lo = max(0, center - zoom_halfwidth)
    hi = min(signal.size, center + zoom_halfwidth)
    inset = ax.inset_axes([0.68, 0.60, 0.30, 0.32])
    inset.plot(x_ms[lo:hi], signal[lo:hi], color=GREY, linewidth=0.9,
               marker="o", markersize=2.4, markerfacecolor=NAVY)
    inset.text(0.03, 0.06, f"{hi - lo} samples", transform=inset.transAxes, fontsize=8, color=NAVY)
    inset.tick_params(labelsize=7)
    ax.indicate_inset_zoom(inset, edgecolor=NAVY, alpha=0.6)
    return ax




def plot_bandpass(
    signal: np.ndarray,
    trace: DetectorTrace,
    *,
    ax: Axes | None = None,
) -> Axes:
    """Band-passed trace over the raw one, on a single shared axis."""
    ax = _single_axis(ax)
    config = trace.config
    x_ms = time_axis_ms(signal.size, config)
    ax.plot(x_ms, signal, color=GREY, linewidth=0.6, alpha=0.45, label="raw")
    ax.plot(x_ms, trace.filtered, color=BLUE, linewidth=0.6,
            label=f"band-passed {config.low_freq_hz / 1000:.0f}\u2013"
                  f"{config.high_freq_hz / 1000:.0f} kHz")
    ax.set_xlabel("time [ms]")
    ax.set_ylabel("amplitude [a.u.]")
    ax.set_title(
        f"Butterworth order {config.filter_order}, zero-phase (filtfilt): "
        "the burst survives, unmoved"
    )
    ax.legend(loc="upper right")
    return ax


def plot_stft_windows(
    trace: DetectorTrace,
    *,
    center: int,
    n_windows: int = 5,
    ax: Axes | None = None,
) -> Axes:
    """Sliding Hann windows over the filtered trace around ``center``."""
    ax = _single_axis(ax)
    config = trace.config
    x_ms = time_axis_ms(trace.filtered.size, config)
    n, hop = config.stft_nperseg, trace.hop
    first = max(0, center - n // 2 - (n_windows // 2) * hop)
    span = slice(max(0, first - n // 2), min(trace.filtered.size, first + n + n_windows * hop))
    peak = float(np.max(np.abs(trace.filtered[span])) or 1.0)
    ax.plot(x_ms[span], trace.filtered[span] / peak, color=LIGHT_GREY, linewidth=0.7)
    hann = np.hanning(n)
    for index in range(n_windows):
        start = first + index * hop
        if start + n > trace.filtered.size:
            break
        colour = plt.matplotlib.colors.to_rgba(BLUE, 0.35 + 0.5 * index / max(1, n_windows - 1))
        ax.plot(x_ms[start : start + n], hann, color=colour, linewidth=1.4)
    ax.set_xlabel("time [ms]")
    ax.set_ylabel("normalised amplitude / window weight")
    ax.set_title(
        f"Hann windows of N = {n} samples, sliding by H = {hop} samples "
        f"({hop / config.sampling_frequency_hz * 1e6:.0f} µs per frame)"
    )
    return ax


def plot_spectrogram(
    trace: DetectorTrace, *, annotate: bool = False, ax: Axes | None = None
) -> Axes:
    """Band-limited power spectrogram in dB.

    With ``annotate=True``, one column and one row are outlined to anchor the
    P(k, m) notation: a column is a frame m, a row is a frequency bin k.
    """
    ax = _single_axis(ax)
    power_db = 10.0 * np.log10(trace.power + 1.0e-12)
    low, high = np.quantile(power_db, [0.05, 0.995])
    mesh = ax.pcolormesh(
        frame_axis_ms(trace), trace.frequencies / 1000.0, power_db,
        shading="auto", cmap="magma", vmin=float(low), vmax=float(high),
    )
    ax.figure.colorbar(mesh, ax=ax, label="power [dB]")
    ax.set_xlabel("time [ms]")
    ax.set_ylabel("frequency [kHz]")
    config = trace.config
    ax.set_title(
        f"Spectrogram, detection band {config.low_freq_hz / 1000:.0f}–{config.high_freq_hz / 1000:.0f} kHz"
    )
    if annotate:
        x_ms = frame_axis_ms(trace)
        frame = int(np.argmax(trace.frame_energy))
        bin_index = int(np.argmax(trace.excess.sum(axis=1)))
        ax.axvline(x_ms[frame], color="white", linestyle="--", linewidth=1.0, alpha=0.85)
        ax.axhline(trace.frequencies[bin_index] / 1000.0, color="white",
                   linestyle=":", linewidth=1.0, alpha=0.85)
        ax.annotate(f"one column = one frame m (here m = {frame})",
                    xy=(x_ms[frame], float(ax.get_ylim()[1])),
                    xytext=(4, -2), textcoords="offset points",
                    color="white", fontsize=9, va="top")
        ax.annotate(f"one row = one frequency bin k (here {trace.frequencies[bin_index] / 1000:.1f} kHz)",
                    xy=(float(x_ms[0]), trace.frequencies[bin_index] / 1000.0),
                    xytext=(4, 4), textcoords="offset points",
                    color="white", fontsize=9)
    return ax


def plot_frequency_baseline(
    trace: DetectorTrace,
    *,
    bin_index: int | None = None,
    axes: np.ndarray | None = None,
) -> np.ndarray:
    """One frequency row with its Q25 baseline, then the positive excess."""
    axes = _stacked_axes(axes, 2, height=6.0)
    x_ms = frame_axis_ms(trace)
    if bin_index is None:
        bin_index = int(np.argmax(trace.excess.sum(axis=1)))
    # Linear scale: clipped-to-zero noise stays dark, only true excess lights up.
    axes[1].pcolormesh(x_ms, trace.frequencies / 1000.0, trace.excess,
                       shading="auto", cmap="magma", vmin=0.0,
                       vmax=float(np.quantile(trace.excess, 0.999)))
    axes[1].set_ylabel("frequency [kHz]")
    axes[1].set_title(r"Positive excess $P^{+}(k, m) = \max(P - B_k, 0)$ [linear]")
    axis = axes[0]
    row = trace.power[bin_index]
    axis.plot(x_ms, row, color=BLUE, linewidth=0.9,
              label=f"P(k, m) at k = {trace.frequencies[bin_index] / 1000:.1f} kHz")
    axis.axhline(float(trace.baseline[bin_index, 0]), color=GREEN, linewidth=1.4,
                 label=r"$B_k$ = per-frequency Q25")
    axis.set_yscale("log")
    axis.set_ylabel("power")
    axis.set_title("One row of the section-3 spectrogram, against its own ordinary level")
    axis.legend(loc="upper right")
    axes[1].set_xlabel("time [ms]")
    return axes


def plot_frame_energy(
    trace: DetectorTrace,
    *,
    show_map: bool = True,
    axes: np.ndarray | None = None,
) -> np.ndarray:
    """E[m], optionally under the positive-excess map it sums."""
    axes = _stacked_axes(axes, 2 if show_map else 1, height=6.0 if show_map else 3.6)
    x_ms = frame_axis_ms(trace)
    if show_map:
        axes[0].pcolormesh(x_ms, trace.frequencies / 1000.0, trace.excess,
                           shading="auto", cmap="magma", vmin=0.0,
                           vmax=float(np.quantile(trace.excess, 0.999)))
        axes[0].set_ylabel("frequency [kHz]")
        axes[0].set_title(r"$P^{+}(k, m)$ [linear]")
    curve = axes[-1]
    curve.plot(x_ms, trace.frame_energy, color=BLUE, linewidth=1.1)
    curve.set_ylabel("E[m]")
    curve.set_xlabel("time [ms]")
    curve.set_title(
        rf"$E[m] = \sum_k P^{{+}}(k, m)$, smoothed over {trace.config.smooth_frames} frames"
    )
    return axes


def plot_robust_band(
    trace: DetectorTrace,
    *,
    scaled: bool = False,
    both: bool = False,
    ax: Axes | None = None,
) -> Axes:
    """E[m] with its median and robust band(s).

    ``both=True`` draws the unscaled MAD band and the 1.4826-scaled one
    together, which is the only place the two quantities can be told apart
    by eye.
    """
    ax = _single_axis(ax)
    x_ms = frame_axis_ms(trace)
    ax.plot(x_ms, trace.frame_energy, color=BLUE, linewidth=1.1, label="E[m]")
    ax.axhline(trace.energy_median, color=GREEN, linewidth=1.4, label="median")
    if both:
        ax.axhspan(trace.energy_median - trace.energy_scale,
                   trace.energy_median + trace.energy_scale,
                   color=PURPLE, alpha=0.12, label="median ± energy_scale (1.4826 × MAD)")
        ax.axhspan(trace.energy_median - trace.raw_mad,
                   trace.energy_median + trace.raw_mad,
                   color=PURPLE, alpha=0.30, label="median ± raw_mad")
        top = trace.energy_median + 4.0 * trace.energy_scale
        ax.set_ylim(0.0, top)
        peak_z = float(trace.energy_z.max())
        peak_ms = float(frame_axis_ms(trace)[int(np.argmax(trace.energy_z))])
        ax.annotate(f"peak continues off-axis, to z = {peak_z:.1f}",
                    xy=(peak_ms, top), xytext=(14, -30), textcoords="offset points",
                    ha="left", va="top", color=NAVY, fontsize=9,
                    arrowprops={"arrowstyle": "->", "color": NAVY})
        ax.set_title("The two bands that have both been called \"the MAD\"")
        ax.set_xlabel("time [ms]")
        ax.set_ylabel("E[m]")
        ax.legend(loc="upper left", fontsize=9)
        return ax
    else:
        half = trace.energy_scale if scaled else trace.raw_mad
        label = "median ± 1.4826 MAD (= energy_scale)" if scaled else "median ± 1 raw MAD"
        ax.axhspan(trace.energy_median - half, trace.energy_median + half,
                   color=PURPLE, alpha=0.18, label=label)
        ax.set_title("The event exits the robust band by many band-widths")
    ax.set_xlabel("time [ms]")
    ax.set_ylabel("E[m]")
    ax.legend(loc="upper right")
    return ax




def _runs(active: np.ndarray) -> list[tuple[int, int]]:
    flags = np.asarray(active, dtype=bool).astype(np.int8)
    starts = np.flatnonzero(np.diff(np.concatenate(([0], flags))) == 1)
    ends = np.flatnonzero(np.diff(np.concatenate((flags, [0]))) == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def count_active_runs(active: np.ndarray) -> int:
    """Contiguous runs of active frames, before any gap bridging."""
    flags = np.asarray(active, dtype=bool)
    if not flags.any():
        return 0
    return int(np.count_nonzero(np.diff(flags.astype(np.int8)) == 1) + int(flags[0]))


def plot_activation(
    trace: DetectorTrace,
    *,
    z_threshold: float | None = None,
    ax: Axes | None = None,
) -> Axes:
    """z[m] against its threshold, with the frames it selects shaded."""
    config = trace.config
    z_threshold = config.active_snr_z if z_threshold is None else float(z_threshold)
    ax = _single_axis(ax)
    x_ms = frame_axis_ms(trace)
    active = trace.energy_z >= z_threshold
    # Shade each run over the exact samples grouping will assign to it:
    # left*hop .. right*hop + N, so this figure and the interval agree.
    for left, right in _runs(active):
        start_ms = left * trace.hop / config.sampling_frequency_hz * 1000.0
        end_ms = ((right * trace.hop + config.stft_nperseg)
                  / config.sampling_frequency_hz * 1000.0)
        ax.axvspan(start_ms, end_ms, color=PALE_GREEN, zorder=0)
    ax.plot(x_ms, trace.energy_z, color=PURPLE, linewidth=1.2, zorder=3)
    ax.plot(x_ms[active], trace.energy_z[active], linestyle="none", marker="o",
            markersize=3.6, color=GREEN, zorder=4, label="selected frames")
    ax.axhline(z_threshold, color=RED, linestyle="--", linewidth=1.2,
               label=f"activation threshold z = {z_threshold:.2f}")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylabel("z[m]")
    ax.set_xlabel("time [ms]")
    ax.set_title(
        f"{int(active.sum())} frames of {active.size} selected, "
        f"in {count_active_runs(active)} contiguous run(s) before gap bridging"
    )
    ax.legend(loc="upper right")
    return ax




def plot_ssl_crop(
    signal: np.ndarray,
    event,
    config: YeastDetectionConfig,
    *,
    crop_length: int = 8192,
    ax: Axes | None = None,
) -> Axes:
    """The detected interval inside the fixed-length crop handed to the model."""
    ax = _single_axis(ax)
    values = np.asarray(signal)
    centre = int(event.center_index)
    crop_start = max(0, centre - crop_length // 2)
    crop_end = min(values.size, crop_start + crop_length)
    view = slice(max(0, crop_start - crop_length // 4),
                 min(values.size, crop_end + crop_length // 4))
    x_ms = (np.arange(view.start, view.stop) - centre) / config.sampling_frequency_hz * 1000.0

    def to_ms(index: int) -> float:
        return (index - centre) / config.sampling_frequency_hz * 1000.0

    ax.axvspan(to_ms(crop_start), to_ms(crop_end), color=PALE_BLUE, zorder=0,
               label=f"model crop · {crop_length} samples "
                     f"({crop_length / config.sampling_frequency_hz * 1000:.3f} ms)")
    ax.axvspan(to_ms(event.event_start), to_ms(event.event_end), color=PALE_GREEN,
               zorder=1, label=f"detected event · {event.width_samples} samples "
                               f"({event.width_ms:.3f} ms)")
    ax.plot(x_ms, values[view], color=GREY, linewidth=0.6, zorder=3)
    ax.axvline(0.0, color=NAVY, linewidth=1.0, linestyle="--", zorder=4,
               label="energy-weighted centre")
    ax.set_xlabel("time relative to the event centre [ms]")
    ax.set_ylabel("amplitude [a.u.]")
    ax.set_title(
        f"The training window: {crop_length} samples "
        f"({crop_length / config.sampling_frequency_hz * 1000:.3f} ms) "
        "centred on the energy-weighted centre"
    )
    ax.legend(loc="upper right", fontsize=9)
    return ax


def plot_detected_events(
    signal: np.ndarray,
    events,
    config: YeastDetectionConfig,
    *,
    truth_spans_ms: list[tuple[float, float]] | None = None,
    ax: Axes | None = None,
) -> Axes:
    """The time series with the intervals the MAD chain actually returned."""
    ax = _single_axis(ax)
    x_ms = time_axis_ms(np.asarray(signal).size, config)
    for start_ms, end_ms in truth_spans_ms or []:
        ax.axvspan(start_ms, end_ms, color=PALE_BLUE, zorder=0)
    ax.plot(x_ms, signal, color=GREY, linewidth=0.6)
    span = float(np.max(np.abs(signal))) or 1.0
    for index, event in enumerate(events):
        left = event.event_start / config.sampling_frequency_hz * 1000.0
        right = event.event_end / config.sampling_frequency_hz * 1000.0
        colour = GREEN if event.quality in {"strict", "medium"} else ORANGE
        ax.plot([left, right], [-1.15 * span] * 2, color=colour, linewidth=9,
                solid_capstyle="butt")
        ax.annotate(f"{event.quality} · z max {event.snr_proxy:.1f}",
                    xy=((left + right) / 2.0, -1.15 * span),
                    xytext=(0, 9 + 11 * (index % 2)),
                    textcoords="offset points", ha="center", fontsize=8.5, color=colour)
    ax.set_ylim(-1.55 * span, 1.2 * span)
    ax.set_xlabel("time [ms]")
    ax.set_ylabel("amplitude [a.u.]")
    kept = sum(1 for event in events if event.quality in {"strict", "medium"})
    ax.set_title(
        f"MAD chain output: {len(events)} interval(s), {kept} accepted"
        + (" — the given span in blue" if truth_spans_ms else "")
    )
    return ax


def plot_records_overview(
    entries,
    config: YeastDetectionConfig,
    *,
    axes: np.ndarray | None = None,
) -> np.ndarray:
    """One row per record: the trace with every interval the chain returned.

    ``entries`` is a sequence of ``(label, signal, events)``. Accepted events
    are green, rejected ones orange, so a whole cohort reads at a glance.
    """
    entries = list(entries)
    axes = _stacked_axes(axes, len(entries), height=2.05 * len(entries))
    for axis, (label, signal, events) in zip(axes, entries):
        plot_detected_events(signal, events, config, ax=axis)
        kept = sum(1 for event in events if event.quality in {"strict", "medium"})
        axis.set_title("")  # replace the single-record title
        axis.set_title(f"{label} — {len(events)} interval(s), {kept} accepted",
                       loc="left", fontsize=10, color=NAVY)
        axis.set_xlabel("")
        axis.set_ylabel("")
    axes[-1].set_xlabel("time [ms]")
    return axes


_DROP_STAGE_LABELS = {
    "passage_time": "dropped: passage time",
    "width": "dropped: box width",
    "filtered_evidence": "dropped: peak evidence",
    "dual_clean": "dropped: dual-clean",
    "nms": "dropped: NMS",
}


def plot_legacy_vs_energy(
    trace: DetectorTrace,
    *,
    replay: dict,
    truth_spans_ms: list[tuple[float, float]],
    mad_events=(),
    axes: np.ndarray | None = None,
) -> np.ndarray:
    """The two pipelines' outputs on one trace, as bars over a shared axis.

    Top: filtered trace with the true event spans. Middle: every raw legacy
    hypothesis — surviving boxes solid blue, cleaned-away ones faded orange
    with the stage that removed them. Bottom: what the MAD chain returned,
    green when accepted and orange when rejected.
    """
    axes = _stacked_axes(axes, 3, height=8.0)
    config = trace.config
    x_ms = time_axis_ms(trace.filtered.size, config)
    for axis in axes:
        for start_ms, end_ms in truth_spans_ms:
            axis.axvspan(start_ms, end_ms, color=PALE_BLUE, zorder=0)
    axes[0].plot(x_ms, trace.filtered, color=GREY, linewidth=0.6)
    axes[0].set_ylabel("filtered [a.u.]")
    axes[0].set_title("The trace — declared true events in blue")

    hypotheses = replay["raw_all"]
    survivors = sum(1 for row in hypotheses if row["dropped_at"] is None)
    for index, row in enumerate(hypotheses):
        y = len(hypotheses) - index
        survives = row["dropped_at"] is None
        axes[1].plot(
            [row["start_ms"], row["end_ms"]], [y, y],
            color=BLUE if survives else ORANGE,
            alpha=1.0 if survives else 0.4,
            linewidth=8, solid_capstyle="round",
        )
        note = f"{row['frequency_khz']:.1f} kHz"
        if not survives:
            note += f" · {_DROP_STAGE_LABELS[row['dropped_at']]}"
        axes[1].annotate(note, xy=(row["end_ms"], y), xytext=(6, -2),
                         textcoords="offset points", fontsize=8,
                         color=NAVY if survives else ORANGE)
    axes[1].set_ylim(0.2, len(hypotheses) + 0.8)
    axes[1].set_yticks([])
    axes[1].set_ylabel("hypotheses")
    axes[1].set_title(
        f"particles2SNR (peak-first): {len(hypotheses)} hypotheses "
        f"→ {survivors} box(es) after five cleaning stages"
    )

    events = list(mad_events)
    config = trace.config
    for index, event in enumerate(events):
        left = event.event_start / config.sampling_frequency_hz * 1000.0
        right = event.event_end / config.sampling_frequency_hz * 1000.0
        accepted = event.quality in {"strict", "medium"}
        axes[2].plot([left, right], [len(events) - index] * 2,
                     color=GREEN if accepted else ORANGE, linewidth=8,
                     alpha=1.0 if accepted else 0.55, solid_capstyle="round")
        axes[2].annotate(f"{event.quality} · z max {event.snr_proxy:.0f}",
                         xy=(right, len(events) - index), xytext=(6, -2),
                         textcoords="offset points", fontsize=8,
                         color=GREEN if accepted else ORANGE)
    axes[2].set_ylim(0.2, max(1, len(events)) + 0.8)
    axes[2].set_yticks([])
    axes[2].set_ylabel("MAD output")
    axes[2].set_xlabel("time [ms]")
    kept = sum(1 for event in events if event.quality in {"strict", "medium"})
    axes[2].set_title(
        f"MAD chain: no hypothesis was ever created to clean — "
        f"{len(events)} interval(s) returned, {kept} accepted"
    )
    return axes


