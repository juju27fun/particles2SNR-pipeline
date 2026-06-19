"""fft_analysis_pipeline_particles2SNR.py"""

import os
import json
import time
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import butter, filtfilt, hilbert, find_peaks, peak_widths

# ---------------- Argument parser ----------------
parser = argparse.ArgumentParser(description="FFT-based analysis pipeline for signal processing.")
parser.add_argument("--target-file", "-f", type=str, required=True, help="Target file to analyze (full path).")
parser.add_argument("--output-dir", "-o", type=str, default=".", help="Output directory for results.")
parser.add_argument("--verbose", "-v", action="store_true", help="Display intermediate processing information.")
parser.add_argument("--device", "-d", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to use for processing (default: cpu).")
parser.add_argument("--pdf", "-p", action="store_true", default=True, help="Generate PDF output (default: True).")
parser.add_argument("--no-pdf", action="store_true", help="Disable PDF generation.")

class Config:
    """Configuration for FFT analysis pipeline."""
    sampling_rate = 2000000
    sequence_length = 16384
    bandpass_lowcut = 5000
    bandpass_highcut = 100000
    bandpass_order = 4
    # --------------------------------------------------
    # --------------------------------------------------
    fft_window_length = 4096
    fft_stride = 1024
    energy_threshold = 3000.0
    # --------------------------------------------------
    # --------------------------------------------------
    max_peaks = 3
    narrow_bandpass_order = 3
    narrow_bandpass_width = 4000
    next_peak_threshold_factor = 0.5
    noise_floor_N = 3  # Number of lowest-energy windows to use for noise floor estimation

class SignalProcessor:
    """Core signal processing operations."""

    def __init__(self, config: Config, device: torch.device):
        self.config = config
        self.device = device

    def bandpass(self, data: np.ndarray, low: float, high: float, order: int, fs: float) -> np.ndarray:
        """Apply Butterworth bandpass filter."""
        nyquist = 0.5 * fs
        b, a = butter(order, [low / nyquist, high / nyquist], btype='band')
        return filtfilt(b, a, data)

    def narrow_bandpass(self, data: torch.Tensor, center: float) -> torch.Tensor:
        """Apply narrow bandpass filter."""
        fs = self.config.sampling_rate
        width = self.config.narrow_bandpass_width
        order = self.config.narrow_bandpass_order
        low = max(1.0, center - width / 2)
        high = center + width / 2
        nyquist = 0.5 * fs
        b, a = butter(order, [low / nyquist, high / nyquist], btype='band')
        y = filtfilt(b, a, data.cpu().numpy())
        return torch.from_numpy(y.copy()).float().to(data.device)

    def compute_fft_stats(self, freqs: np.ndarray, mag: np.ndarray) -> tuple:
        """Compute peak frequency, bandwidth, and energy from FFT magnitude."""
        if len(mag) < 2 or mag.max() == 0:
            return 0.0, 0.0, 0.0
        energy = float(np.sum(mag ** 2))
        peaks, properties = find_peaks(mag, prominence=0.1 * mag.max())
        if len(peaks) == 0:
            return 0.0, 0.0, energy
        peak_idx = peaks[np.argmax(mag[peaks])]
        peak_freq = float(freqs[peak_idx])
        widths, _, _, _ = peak_widths(mag, [peak_idx], rel_height=0.5)
        bandwidth = float(widths[0] * (freqs[1] - freqs[0]))
        return peak_freq, bandwidth, energy

    def find_peaks(self, fft_amp: torch.Tensor, threshold: float) -> list:
        """Find local maxima above threshold in FFT amplitude."""
        fft_np = fft_amp.cpu().numpy()
        threshold_np = float(threshold) if not isinstance(threshold, (int, float)) else threshold
        peaks, properties = find_peaks(fft_np, height=threshold_np)
        return [(int(p), float(fft_np[p])) for p in peaks]

    def extract_hilbert_params(self, signal: torch.Tensor, f_D: float,
                                center_idx: int, signal_len: int, fft_len: int) -> dict:
        """Extract signal parameters using Hilbert Transform."""
        search_factor = 0.9
        fs = self.config.sampling_rate
        expanded_len = int(fft_len * search_factor)
        half_expanded = expanded_len // 2
        expanded_start = max(0, center_idx - half_expanded)
        expanded_end = min(signal_len, center_idx + half_expanded)
        signal_expanded = signal[expanded_start:expanded_end]

        analytic = hilbert(signal_expanded.cpu().numpy())
        envelope = torch.from_numpy(np.abs(analytic)).float().to(signal.device)

        t0_idx_rel = torch.argmax(envelope).item()
        t0_idx = expanded_start + t0_idx_rel
        t0 = t0_idx / fs
        P0 = envelope.max().item()

        half_max = P0 / 2
        left_idx = right_idx = t0_idx_rel
        for i in range(t0_idx_rel, -1, -1):
            if envelope[i] < half_max:
                left_idx = i
                break
        for i in range(t0_idx_rel, len(envelope)):
            if envelope[i] < half_max:
                right_idx = i
                break

        fwhm = right_idx - left_idx
        tau = fwhm / (2 * np.sqrt(2 * np.log(2))) / fs

        envelope_safe = torch.where(envelope < 1e-10, torch.ones_like(envelope), envelope)
        y_norm = signal_expanded / envelope_safe
        V = np.clip(y_norm[t0_idx_rel].item(), -1.0, 1.0)

        if t0_idx_rel > 0 and t0_idx_rel < len(signal_expanded) - 1:
            S = (signal_expanded[t0_idx_rel + 1] - signal_expanded[t0_idx_rel - 1]).item() / 2
        elif t0_idx_rel > 0:
            S = (signal_expanded[t0_idx_rel] - signal_expanded[t0_idx_rel - 1]).item()
        else:
            S = (signal_expanded[t0_idx_rel + 1] - signal_expanded[t0_idx_rel]).item()

        sign_slope = 1 if S >= 0 else -1
        alpha = np.arccos(V)
        theta_c = 2 * np.pi * f_D * t0
        phi = sign_slope * alpha - theta_c
        phi = ((phi + np.pi) % (2 * np.pi)) - np.pi

        return {
            'P0': P0, 't0': t0, 't0_idx': t0_idx, 'tau': tau, 'phi': phi,
            'envelope': envelope.cpu().numpy(),
            'expanded_start': expanded_start,
            'expanded_end': expanded_end,
            'filtered_signal_expanded': signal_expanded.cpu().numpy()
        }


def load_data(file_path: str) -> np.ndarray:
    """Load a single signal from .npy file."""
    return np.load(file_path)


def generate_pdf(signal: np.ndarray, particles: list, window_centers: list,
                 valid_windows_with_particles: list, config: Config,
                 output_path: str, fs: float, verbose: bool) -> None:
    """Generate PDF visualization from processed results."""
    fft_len = config.fft_window_length
    color_cycle = [
        '#4169e1', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
    ]
    bandpass_low_khz = config.bandpass_lowcut / 1000
    bandpass_high_khz = config.bandpass_highcut / 1000
    signal_len = len(signal)
    time_axis_ms = np.arange(signal_len) / fs * 1000

    processor = SignalProcessor(config, torch.device('cpu'))

    with plt.ioff():
        with PdfPages(output_path) as pdf:
            # Page 1: Summary (3 panels)
            fig_summary = plt.figure(figsize=(16, 12))
            outer_gs_summary = gridspec.GridSpec(3, 1, figure=fig_summary,
                                                  height_ratios=[1, 1, 1], hspace=0.35)

            # Panel 1: Original signal only
            ax1_sum = fig_summary.add_subplot(outer_gs_summary[0])
            ax1_sum.plot(time_axis_ms, signal, color='royalblue', linewidth=0.8)
            ax1_sum.set_xlabel('Time (ms)', fontsize=9)
            ax1_sum.set_ylabel('Amplitude', fontsize=9)
            ax1_sum.set_title('Panel 1: Original Signal', fontsize=10, fontweight='bold')
            ax1_sum.grid(True, alpha=0.3)

            # Panel 2: Original signal + all particles (6σ)
            ax2_sum = fig_summary.add_subplot(outer_gs_summary[1])
            ax2_sum.plot(time_axis_ms, signal, color='gray', alpha=0.25, linewidth=0.5, label='Original signal')

            sigma_limit = 3
            for i, particle in enumerate(particles):
                color = color_cycle[i % len(color_cycle)]
                t0_ms = particle['t0'] * 1000
                tau_ms = particle['tau'] * 1000

                sigma_start = max(0, int((t0_ms - sigma_limit * tau_ms) / 1000 * fs))
                sigma_end = min(signal_len, int((t0_ms + sigma_limit * tau_ms) / 1000 * fs))

                time_6sigma = time_axis_ms[sigma_start:sigma_end]
                signal_6sigma = particle['particle_signal'][sigma_start:sigma_end]

                snr_db = particle.get('snr_db', 0.0)
                label = (f"P{i+1}: {particle['frequency']/1000:.1f} kHz  "
                        f"P₀={particle['P0']:.2f}  "
                        f"t₀={t0_ms:.3f} ms  "
                        f"τ={tau_ms:.3f} ms  "
                        f"φ={particle['phi']:.2f} rad  "
                        f"SNR={snr_db:.1f} dB")
                ax2_sum.plot(time_6sigma, signal_6sigma,
                             color=color, linewidth=1.2, alpha=0.9, label=label)
                ax2_sum.axvline(t0_ms, color=color, linestyle=':', linewidth=0.8, alpha=0.6)

            ax2_sum.set_xlabel('Time (ms)', fontsize=9)
            ax2_sum.set_ylabel('Amplitude', fontsize=9)
            ax2_sum.set_title('Panel 2: All Particles Overlay (±3σ, 99.7%)', fontsize=10, fontweight='bold')
            ax2_sum.legend(fontsize=7, loc='upper right', ncol=2)
            ax2_sum.grid(True, alpha=0.3)

            # Panel 3: Full reconstruction + original background
            ax3_sum = fig_summary.add_subplot(outer_gs_summary[2])
            ax3_sum.plot(time_axis_ms, signal, color='gray', alpha=0.3, linewidth=0.5, label='Original signal')

            reconstruction = np.zeros(signal_len)
            for particle in particles:
                reconstruction += particle['particle_signal']

            ax3_sum.plot(time_axis_ms, reconstruction, alpha=0.8,
                         color='crimson', linewidth=1.5, label='Full reconstruction')
            ax3_sum.set_xlabel('Time (ms)', fontsize=9)
            ax3_sum.set_ylabel('Amplitude', fontsize=9)
            ax3_sum.set_title('Panel 3: Full Reconstruction', fontsize=10, fontweight='bold')
            ax3_sum.legend(fontsize=9, loc='upper right')
            ax3_sum.grid(True, alpha=0.3)

            pdf.savefig(fig_summary, bbox_inches='tight')
            plt.close(fig_summary)

            # Pages 2+: Per-window analysis
            for dw in valid_windows_with_particles:
                window_start = dw['center'] - fft_len // 2
                window_end = dw['center'] + fft_len // 2
                window_time_ms = np.arange(window_start, window_end) / fs * 1000

                fig = plt.figure(figsize=(18, 14))
                outer_gs = gridspec.GridSpec(4, 1, figure=fig,
                                             height_ratios=[2, 2, 2, 2],
                                             hspace=0.45)

                # Panel 1: Original signal + window overlay
                ax1 = fig.add_subplot(outer_gs[0])
                ax1.plot(time_axis_ms, signal, color='gray', alpha=0.35, linewidth=0.5, label='Original signal')
                window_signal = signal[window_start:window_end]
                ax1.plot(window_time_ms, window_signal, color='royalblue', linewidth=1.0, label='Current window')
                ax1.axvspan(window_time_ms[0], window_time_ms[-1], alpha=0.12, color='royalblue')
                ax1.set_xlabel('Time (ms)')
                ax1.set_ylabel('Amplitude')
                status = "✔ VALID" if dw['is_valid'] else "✘ NOISE / Below threshold"
                ax1.set_title(
                    f"Window {dw['window_idx']}  |  Center sample: {dw['center']}  |"
                    f"  Energy: {dw['window_energy']:.1f}  |  Particles: {len(dw['window_particles'])}  |  {status}",
                    fontsize=9, pad=6
                )
                ax1.legend(fontsize=7, loc='upper right')
                ax1.grid(True, alpha=0.25)

                # Panel 2: FFT spectra
                inner_gs2 = gridspec.GridSpecFromSubplotSpec(
                    1, 3, subplot_spec=outer_gs[1], wspace=0.05
                )
                fft_freqs_khz = dw['fft_freqs'] / 1000
                freq_mask_bp = (
                    (fft_freqs_khz >= bandpass_low_khz)
                    & (fft_freqs_khz <= bandpass_high_khz)
                )

                detected_spectra_list = dw['detected_spectra']
                while len(detected_spectra_list) < 3:
                    detected_spectra_list.append(None)

                if detected_spectra_list[0] is not None:
                    first_amps_bp = detected_spectra_list[0]['fft_amps'][freq_mask_bp]
                    global_ymax = first_amps_bp.max() * 1.1 if first_amps_bp.max() > 0 else 1.0
                else:
                    global_ymax = dw['fft_amps'][freq_mask_bp].max() * 1.1 or 1.0

                titles_p2 = ['Original', '—', '—']
                for col, (spec, title) in enumerate(zip(detected_spectra_list, titles_p2)):
                    ax = fig.add_subplot(inner_gs2[col])
                    if spec is not None:
                        amps_bp = spec['fft_amps'][freq_mask_bp]
                        freqs_bp_khz = fft_freqs_khz[freq_mask_bp]
                        ax.plot(freqs_bp_khz, amps_bp, 'b-', linewidth=0.8)
                        ax.axhline(spec['threshold'], color='red', linestyle='--',
                                   linewidth=0.9, label=f"Thresh={spec['threshold']:.0f}")
                        energy_khz = spec['energy']
                        ax.text(0.8, 0.95, f'Energy: {energy_khz:.1f}', transform=ax.transAxes,
                                fontsize=7, verticalalignment='top', horizontalalignment='right',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
                        for pk_idx, pk_amp in spec['peaks']:
                            pk_freq_khz = dw['fft_freqs'][pk_idx] / 1000
                            if bandpass_low_khz <= pk_freq_khz <= bandpass_high_khz:
                                ax.plot(pk_freq_khz, pk_amp, 'ro', markersize=5,
                                        label=f"{pk_freq_khz:.1f} kHz")
                        ax.set_ylim(0, global_ymax)
                        ax.legend(fontsize=5, loc='upper right')
                    else:
                        ax.text(0.5, 0.5, '—', transform=ax.transAxes,
                                ha='center', va='center', color='gray')
                    ax.set_title(title, fontsize=8)
                    ax.set_xlabel('Frequency (kHz)', fontsize=7)
                    ax.grid(True, alpha=0.25)
                    if col > 0:
                        plt.setp(ax.get_yticklabels(), visible=False)
                    else:
                        ax.set_ylabel('Amplitude', fontsize=7)

                # Panel 3: Narrow bandpass filtered signals
                inner_gs3 = gridspec.GridSpecFromSubplotSpec(
                    1, 3, subplot_spec=outer_gs[2], wspace=0.05
                )
                particles_dw = dw['window_particles']

                filt_ymax = None
                filtered_signals_cache = []
                for particle in particles_dw[:3]:
                    fsig = processor.narrow_bandpass(
                        torch.from_numpy(signal.copy()).float(), particle['frequency']
                    ).cpu().numpy()
                    filtered_signals_cache.append(fsig)
                    mx = np.abs(fsig).max()
                    if filt_ymax is None or mx > filt_ymax:
                        filt_ymax = mx

                colors_p3 = ['mediumseagreen', 'darkorange', 'mediumpurple']
                for col in range(3):
                    ax = fig.add_subplot(inner_gs3[col])
                    if col < len(filtered_signals_cache):
                        fsig = filtered_signals_cache[col]
                        freq_khz = particles_dw[col]['frequency'] / 1000
                        ax.plot(time_axis_ms, fsig,
                                color=colors_p3[col], linewidth=0.8,
                                label=f"{freq_khz:.1f} kHz")
                        ax.axvspan(window_time_ms[0], window_time_ms[-1],
                                   alpha=0.1, color=colors_p3[col], label='Analysis window')
                        t0_ms = particles_dw[col]['t0'] * 1000
                        ax.axvline(t0_ms, color='red', linestyle='--', linewidth=1,
                                   label=f"t0={t0_ms:.3f} ms")
                        if filt_ymax and filt_ymax > 0:
                            ax.set_ylim(-filt_ymax * 1.1, filt_ymax * 1.1)
                        ax.legend(fontsize=6, loc='upper right')
                        ax.set_title(f'Narrow bandpass – particle {col+1}', fontsize=8)
                    else:
                        ax.text(0.5, 0.5, '—', transform=ax.transAxes,
                                ha='center', va='center', color='gray')
                        ax.set_title(f'Narrow bandpass – particle {col+1}', fontsize=8)
                    ax.set_xlabel('Time (ms)', fontsize=7)
                    ax.grid(True, alpha=0.25)
                    if col > 0:
                        plt.setp(ax.get_yticklabels(), visible=False)
                    else:
                        ax.set_ylabel('Amplitude', fontsize=7)

                # Panel 4: Cosine-Gaussian estimated particles
                ax4 = fig.add_subplot(outer_gs[3])
                ax4.plot(time_axis_ms, signal, color='gray', alpha=0.3, linewidth=0.5, label='Original signal')

                sigma_limit = 3
                for i, particle in enumerate(particles_dw):
                    t0_ms = particle['t0'] * 1000
                    tau_ms = particle['tau'] * 1000
                    t0_s = particle['t0']
                    tau_s = particle['tau']
                    sigma_start = max(0, int((t0_s - sigma_limit * tau_s) * fs))
                    sigma_end = min(len(time_axis_ms), int((t0_s + sigma_limit * tau_s) * fs))
                    time_6sigma = time_axis_ms[sigma_start:sigma_end]
                    signal_6sigma = particle['particle_signal'][sigma_start:sigma_end]

                    snr_db = particle.get('snr_db', 0.0)
                    ax4.plot(
                        time_6sigma,
                        signal_6sigma,
                        '--',
                        color=colors_p3[i] if i < len(colors_p3) else f'C{i}',
                        linewidth=0.2,
                        label=(
                            f"P{i+1}: {particle['frequency']/1000:.1f} kHz  "
                            f"P₀={particle['P0']:.2f}  "
                            f"t₀={t0_ms:.3f} ms  "
                            f"τ={tau_ms:.3f} ms  "
                            f"φ={particle['phi']:.2f} rad  "
                            f"SNR={snr_db:.1f} dB"
                        )
                    )
                    ax4.axvline(t0_ms,
                                color=colors_p3[i] if i < len(colors_p3) else f'C{i}',
                                linestyle=':', linewidth=0.8, alpha=0.7)

                ax4.set_xlabel('Time (ms)')
                ax4.set_ylabel('Amplitude')
                ax4.set_title('Cosine-Gaussian estimated particles')
                ax4.legend(fontsize=7, loc='upper right')
                ax4.grid(True, alpha=0.25)

                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)


def run_pipeline(signal_np: np.ndarray, args: argparse.Namespace, config: Config,
                 device: torch.device, verbose: bool) -> tuple:
    """Run the FFT analysis pipeline on a signal and return JSON-serializable results."""
    fs = config.sampling_rate
    fft_len = config.fft_window_length
    fft_str = config.fft_stride
    energy_thresh = config.energy_threshold
    max_peaks = config.max_peaks
    next_peak_factor = config.next_peak_threshold_factor

    processor = SignalProcessor(config, device)

    if verbose:
        print(f"Device: {device}")

    # ── Load and preprocess signal ──────────────────────────────────────────
    t0_load = time.perf_counter()
    signal_len = len(signal_np)
    filtered_signal_np = processor.bandpass(
        signal_np,
        config.bandpass_lowcut, config.bandpass_highcut,
        config.bandpass_order, fs
    )
    signal = torch.from_numpy(filtered_signal_np.copy()).float().to(device)
    original_signal = signal.clone()
    if verbose:
        print(f"  [TIMING] Load & preprocess: {(time.perf_counter() - t0_load)*1000:.2f} ms")

    if signal_len < fft_len:
        effective_fft_len = 2 ** int(np.floor(np.log2(signal_len)))
        effective_fft_len = max(256, min(signal_len, effective_fft_len))
        if effective_fft_len != fft_len:
            if verbose:
                print(
                    f"  Adapting FFT window from {fft_len} to "
                    f"{effective_fft_len} for signal length {signal_len}"
                )
            fft_len = effective_fft_len
            fft_str = min(fft_str, max(1, fft_len // 4))

    # ── Sliding-window FFT ──────────────────────────────────────────────────
    t0_fft = time.perf_counter()
    n_windows = (signal_len - fft_str) // fft_str
    windows, window_centers = [], []
    for i in range(n_windows):
        start, end = i * fft_str, i * fft_str + fft_len
        if end <= signal_len:
            windows.append(signal[start:end])
            window_centers.append(start + fft_len // 2)

    if not windows:
        print("No valid windows found.")
        result = {
            'warning': 'No valid FFT windows found',
            'signal_length': signal_len,
            'num_windows': 0,
            'num_valid_windows': 0,
            'fft_window_length': fft_len,
            'fft_stride': fft_str,
            'num_particles': 0,
            'noise_floor': None,
            'noise_floor_N': 0,
            'particles': [],
        }
        return result, [], [], [], np.asarray([]), filtered_signal_np, None

    windows = torch.stack(windows)
    hamming_window = torch.hamming_window(fft_len, device=device)
    windows_windowed = windows * hamming_window
    fft_results = torch.fft.rfft(windows_windowed)
    fft_amplitudes = torch.abs(fft_results)
    fft_freqs = torch.fft.rfftfreq(fft_len, d=1 / fs, device=device)

    freq_mask_energy = (fft_freqs <= config.bandpass_highcut)
    energies = torch.sum(fft_amplitudes[:, freq_mask_energy] ** 2, dim=1)
    valid_mask = energies > energy_thresh

    # Compute noise floor from N lowest-energy windows (more robust than single minimum)
    noise_floor_N = min(config.noise_floor_N, len(energies))
    noise_floor = torch.mean(torch.sort(energies).values[:noise_floor_N]).item()

    if verbose:
        print(f"  [TIMING] Sliding-window FFT ({n_windows} windows): {(time.perf_counter() - t0_fft)*1000:.2f} ms")
        print(f"  Noise floor (mean of {noise_floor_N} lowest windows): {noise_floor:.2f}")

    # ── Per-window processing ───────────────────────────────────────────────
    t0_process = time.perf_counter()
    detected_windows = []
    time_axis_ms = np.arange(signal_len) / fs * 1000
    fft_freqs_np = fft_freqs.cpu().numpy()

    for wi, (win_fft_amp, center) in enumerate(zip(fft_amplitudes, window_centers)):
        window_signal = windows[wi]
        is_valid = valid_mask[wi].item()

        fft_amps_np = win_fft_amp.cpu().numpy()
        freq_mask_bp = (
            (fft_freqs_np >= config.bandpass_lowcut)
            & (fft_freqs_np <= config.bandpass_highcut)
        )
        peak_freq, bandwidth, _ = processor.compute_fft_stats(
            fft_freqs_np[freq_mask_bp], fft_amps_np[freq_mask_bp]
        )

        window_particles = []
        detected_spectra = []

        if is_valid:
            window_start = center - fft_len // 2
            window_end = center + fft_len // 2

            window_signal_for_fft = original_signal[window_start:window_end]
            window_fft_amp = torch.abs(
                torch.fft.rfft(window_signal_for_fft.unsqueeze(0))
            )[0]

            max_peak = torch.max(window_fft_amp)
            threshold_init = max_peak * next_peak_factor
            peaks_init = processor.find_peaks(window_fft_amp, threshold_init)
            peaks_init.sort(key=lambda x: x[1], reverse=True)

            selected_peaks = peaks_init[:max_peaks]

            if verbose:
                print(f"  Peaks detected: {len(selected_peaks)} (threshold={threshold_init:.1f})")

            fft_amps = window_fft_amp.cpu().numpy()
            energy = float(np.sum(fft_amps ** 2))
            if verbose:
                print(f"  FFT spectrum recorded")
            detected_spectra.append({
                'fft_amps': fft_amps.copy(),
                'peaks': selected_peaks.copy(),
                'threshold': float(threshold_init.cpu().item()),
                'energy': energy,
            })

            for peak_idx, peak_amp in selected_peaks:
                best_peak_freq = fft_freqs[peak_idx].item()

                filtered = processor.narrow_bandpass(original_signal, best_peak_freq)
                params = processor.extract_hilbert_params(
                    filtered, best_peak_freq, int(center), signal_len, fft_len
                )
                t_seconds = torch.arange(signal_len, device=device, dtype=torch.float32) / fs
                cos_comp = torch.cos(2 * np.pi * best_peak_freq * t_seconds + params['phi'])
                gauss_env = torch.exp(
                    -((t_seconds - params['t0']) ** 2) / (2 * params['tau'] ** 2)
                )
                particle_signal = params['P0'] * cos_comp * gauss_env

                tau_ms = params['tau'] * 1000
                if tau_ms >= 0.05:
                    peak_energy = float(win_fft_amp[peak_idx].item() ** 2)

                    # Compute SNR: signal power / noise floor
                    snr_linear = peak_energy / noise_floor if noise_floor > 0 else 0.0
                    snr_db = 10 * np.log10(snr_linear) if snr_linear > 0 else -np.inf

                    if verbose:
                        print(f"  Particle {len(window_particles) + 1}: freq={best_peak_freq/1000:.2f} kHz, "
                              f"P0={params['P0']:.2f}, t0={params['t0']*1000:.3f} ms, "
                              f"tau={tau_ms:.3f} ms, phi={params['phi']:.2f} rad, SNR={snr_db:.1f} dB")
                    window_particles.append({
                        'frequency': best_peak_freq,
                        'P0': params['P0'],
                        't0': params['t0'],
                        'tau': params['tau'],
                        'phi': params['phi'],
                        'particle_signal': particle_signal.cpu().numpy(),
                        'energy': peak_energy,
                        'snr_db': snr_db,
                        'snr_method': 'peak_bin_energy_over_lowest_window_energy',
                        'noise_floor': noise_floor,
                        'noise_floor_N': noise_floor_N,
                        'source_window_idx': wi,
                        'source_window_center': center,
                        'source_window_energy': energies[wi].item(),
                    })
                else:
                    if verbose:
                        print(f"  Skipping particle: tau={tau_ms:.3f} ms < 0.05 ms")

        detected_windows.append({
            'window_idx': wi,
            'center': center,
            'signal': window_signal,
            'fft_freqs': fft_freqs_np,
            'fft_amps': fft_amps_np,
            'is_valid': is_valid,
            'window_energy': energies[wi].item(),
            'peak_freq': peak_freq,
            'bandwidth': bandwidth,
            'window_particles': window_particles,
            'detected_spectra': detected_spectra,
        })

    # ── Collect and deduplicate particles ───────────────────────────────────
    t0_pdf = time.perf_counter()
    all_particles = []
    for dw in detected_windows:
        for particle in dw['window_particles']:
            particle_info = {
                'frequency': particle['frequency'],
                'P0': particle['P0'],
                't0': particle['t0'],
                'tau': particle['tau'],
                'phi': particle['phi'],
                'particle_signal': particle['particle_signal'],
                'energy': particle['energy'],
                'snr_db': particle['snr_db'],
                'snr_method': particle['snr_method'],
                'noise_floor': particle['noise_floor'],
                'noise_floor_N': particle['noise_floor_N'],
                'source_window_idx': particle['source_window_idx'],
                'source_window_center': particle['source_window_center'],
                'source_window_energy': particle['source_window_energy'],
            }
            all_particles.append(particle_info)

    if verbose:
        print(f"Total particles before deduplication: {len(all_particles)}")

    freq_tolerance = 1000.0
    t0_tolerance = 0.0005

    deduplicated = []
    for particle in all_particles:
        found_duplicate = False
        for i, existing in enumerate(deduplicated):
            freq_diff = abs(particle['frequency'] - existing['frequency'])
            t0_diff = abs(particle['t0'] - existing['t0'])

            if freq_diff <= freq_tolerance and t0_diff <= t0_tolerance:
                found_duplicate = True
                if particle['energy'] > existing['energy']:
                    deduplicated[i] = particle
                break

        if not found_duplicate:
            deduplicated.append(particle)

    all_particles = deduplicated

    if verbose:
        print(f"Total particles after deduplication: {len(all_particles)}")

    valid_windows_with_particles = [
        dw for dw in detected_windows
        if dw['is_valid'] and len(dw['window_particles']) > 0
    ]

    if verbose:
        print(f"Valid windows with particles: {len(valid_windows_with_particles)} / {len(detected_windows)}")
    if verbose:
        print(f"  [TIMING] Per-window processing: {(time.perf_counter() - t0_process)*1000:.2f} ms")

    # Build JSON-serializable result (without particle_signal arrays)
    json_particles = []
    for p in all_particles:
        json_particles.append({
            'frequency': p['frequency'],
            'P0': p['P0'],
            't0': p['t0'],
            'tau': p['tau'],
            'phi': p['phi'],
            'energy': p['energy'],
            'snr_db': p['snr_db'],
            'noise_floor': noise_floor,
            'noise_floor_N': noise_floor_N,
            'snr_method': p['snr_method'],
            'source_window_idx': p['source_window_idx'],
            'source_window_center': p['source_window_center'],
            'source_window_energy': p['source_window_energy'],
        })

    result = {
        'signal_length': signal_len,
        'num_windows': n_windows,
        'num_valid_windows': len(valid_windows_with_particles),
        'fft_window_length': fft_len,
        'fft_stride': fft_str,
        'num_particles': len(all_particles),
        'noise_floor': noise_floor,
        'noise_floor_N': noise_floor_N,
        'particles': json_particles,
    }

    return result, all_particles, window_centers, valid_windows_with_particles, fft_freqs_np, filtered_signal_np, noise_floor


# ----------------- Main function -----------------
def main():
    args = parser.parse_args()

    # Handle PDF flag: --no-pdf overrides --pdf
    generate_pdf_output = args.pdf and not args.no_pdf

    if not os.path.exists(args.target_file):
        print(f"Error: File not found: {args.target_file}")
        return

    config = Config()
    device = torch.device(args.device)

    # Load data
    signal_np = load_data(args.target_file)

    # Run pipeline
    result, all_particles, window_centers, valid_windows_with_particles, fft_freqs_np, filtered_signal, noise_floor = run_pipeline(
        signal_np, args, config, device, args.verbose
    )

    if 'error' in result:
        print(f"Error: {result['error']}")
        return

    # Generate PDF if requested
    if generate_pdf_output:
        os.makedirs(args.output_dir, exist_ok=True)
        base_name = os.path.basename(args.target_file).replace('.npy', '')
        output_pdf = os.path.join(args.output_dir, f"fft_analysis_{base_name}.pdf")

        t0_pdf = time.perf_counter()
        generate_pdf(
            filtered_signal, all_particles, window_centers, valid_windows_with_particles,
            config, output_pdf, config.sampling_rate, args.verbose
        )
        if args.verbose:
            print(f"  [TIMING] PDF generation: {(time.perf_counter() - t0_pdf)*1000:.2f} ms")
        print(f"PDF saved to: {output_pdf}  ({len(valid_windows_with_particles) + 1} pages: 1 summary + {len(valid_windows_with_particles)} valid window analyses)")

    # Output JSON result
    output_json = os.path.join(args.output_dir, f"fft_results_{os.path.basename(args.target_file).replace('.npy', '')}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(result, f, indent=2)
    if args.verbose:
        print(f"JSON results saved to: {output_json}")


if __name__ == "__main__":
    main()
