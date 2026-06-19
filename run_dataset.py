"""run_dataset.py - Process entire dataset with folder-specific configurations."""

import os
import csv
import json
import math
import time
import argparse
import numpy as np
import torch
from tqdm import tqdm
from scipy.signal import welch
from scipy.stats import kurtosis as scipy_kurtosis

from fft_analysis_pipeline_particles2SNR import run_pipeline, Config, load_data

parser = argparse.ArgumentParser(description="Process entire dataset with folder-specific configurations.")
parser.add_argument("--dataset-dir", "-d", type=str, default=os.path.expanduser("~/Projects/particlesSebas/particle_detector/test"),
                    help="Path to the dataset directory (default: ~/Projects/particlesSebas/particle_detector/test)")
parser.add_argument("--output-dir", "-o", type=str, default="output", help="Output directory for results (default: output)")
parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to use for processing (default: cpu)")
parser.add_argument("--verbose", "-v", action="store_true", help="Display verbose output")

# ----------------- Folder configurations -----------------
FOLDER_CONFIGS = {
    '2um': {
        'fft_window_length': 4096,
        'fft_stride': 512,
        'energy_threshold': 3000.0,
    },
    '4um': {
        'fft_window_length': 4096,
        'fft_stride': 1024,
        'energy_threshold': 3000.0,
    },
    '10um': {
        'fft_window_length': 4096,
        'fft_stride': 1024,
        'energy_threshold': 3000.0,
    },
    'yeast': {
        'fft_window_length': 2048,
        'fft_stride': 512,
        'energy_threshold': 4000.0,
    },
}

# List of folders to process (in order)
FOLDERS_TO_PROCESS = ['2um', '4um', '10um', 'yeast']


def _safe_float(value):
    """Return a JSON/CSV-friendly float, preserving missing values as None."""
    if value is None:
        return None
    value = float(value)
    if math.isfinite(value):
        return value
    return None


def get_config_for_folder(folder_name: str) -> Config:
    """Return Config with appropriate settings based on folder name."""
    config = Config()
    
    if folder_name in FOLDER_CONFIGS:
        folder_cfg = FOLDER_CONFIGS[folder_name]
        config.fft_window_length = folder_cfg['fft_window_length']
        config.fft_stride = folder_cfg['fft_stride']
        config.energy_threshold = folder_cfg['energy_threshold']
    
    return config


def load_all_data(dataset_dir: str, folders: list) -> list:
    """Load all .npy files from the specified folders in the dataset directory."""
    data_files = []
    
    for folder_name in folders:
        folder_path = os.path.join(dataset_dir, folder_name)
        if os.path.isdir(folder_path):
            for filename in sorted(os.listdir(folder_path)):
                if filename.endswith('.npy'):
                    file_path = os.path.join(folder_path, filename)
                    data_files.append((file_path, folder_name))
        else:
            print(f"Warning: Folder not found: {folder_path}")
    
    return data_files


def compute_spectral_flatness(psd: np.ndarray) -> float:
    """Compute spectral flatness from positive PSD values."""
    psd_pos = psd[psd > 0]
    if len(psd_pos) == 0:
        return 0.0
    return float(np.exp(np.mean(np.log(psd_pos))) / np.mean(psd_pos))


def compute_noise_summary(signal_np: np.ndarray, filtered_signal_np: np.ndarray,
                          config: Config) -> dict:
    """Compute table-friendly per-file noise statistics."""
    fs = config.sampling_rate
    nperseg = min(1024, len(signal_np))
    freqs, psd = welch(signal_np, fs=fs, nperseg=nperseg)
    total_energy = float(np.sum(psd[freqs > 0]))
    inband_mask = (
        (freqs >= config.bandpass_lowcut)
        & (freqs <= config.bandpass_highcut)
    )
    inband_energy = float(np.sum(psd[inband_mask]))
    inband_ratio = inband_energy / total_energy if total_energy > 0 else 0.0

    return {
        'raw_mean': float(np.mean(signal_np)),
        'raw_std': float(np.std(signal_np)),
        'raw_rms': float(np.sqrt(np.mean(signal_np ** 2))),
        'raw_kurtosis': float(scipy_kurtosis(signal_np, fisher=True)),
        'filtered_mean': float(np.mean(filtered_signal_np)),
        'filtered_std': float(np.std(filtered_signal_np)),
        'filtered_rms': float(np.sqrt(np.mean(filtered_signal_np ** 2))),
        'filtered_kurtosis': float(scipy_kurtosis(filtered_signal_np, fisher=True)),
        'inband_energy_ratio': float(inband_ratio),
        'spectral_flatness': compute_spectral_flatness(psd),
    }


def process_signal(file_path: str, folder_name: str, config: Config, 
                   args: argparse.Namespace, signal_idx: int) -> dict:
    """Process a single signal and return the result dictionary."""
    device = torch.device(args.device)
    
    # Load and process signal
    signal_np = load_data(file_path)
    result, all_particles, _, _, _, filtered_signal_np, noise_floor = run_pipeline(
        signal_np, args, config, device, verbose=False
    )
    
    if 'error' in result:
        return None
    
    # Sort particles by t0 and add idx
    sorted_particles = sorted(all_particles, key=lambda p: p['t0'])
    particles_with_idx = []
    for idx, particle in enumerate(sorted_particles):
        particle_dict = {
            'idx': idx,
            'frequency': particle['frequency'],
            'P0': particle['P0'],
            't0': particle['t0'],
            'tau': particle['tau'],
            'phi': particle['phi'],
            'energy': particle['energy'],
            'snr_db': _safe_float(particle.get('snr_db', 0.0)),
            'noise_floor': _safe_float(particle.get('noise_floor', noise_floor)),
            'noise_floor_N': particle.get('noise_floor_N', result.get('noise_floor_N')),
            'snr_method': particle.get(
                'snr_method',
                'peak_bin_energy_over_lowest_window_energy',
            ),
            'source_window_idx': particle.get('source_window_idx'),
            'source_window_center': particle.get('source_window_center'),
            'source_window_energy': _safe_float(
                particle.get('source_window_energy')
            ),
        }
        particles_with_idx.append(particle_dict)
    
    # Build result with filename, class, and signal_idx
    filename = os.path.basename(file_path)
    noise_summary = compute_noise_summary(signal_np, filtered_signal_np, config)
    return {
        'filename': filename,
        'path': file_path,
        'class': folder_name,
        'signal_idx': signal_idx,
        'signal_length': result.get('signal_length'),
        'num_windows': result.get('num_windows'),
        'num_valid_windows': result.get('num_valid_windows'),
        'noise_floor': _safe_float(noise_floor),
        'noise_floor_N': result.get('noise_floor_N'),
        'snr_method': 'peak_bin_energy_over_lowest_window_energy',
        'noise_summary': noise_summary,
        'num_particles': len(particles_with_idx),
        'particles': particles_with_idx,
    }


def export_csv(rows: list, output_path: str, fieldnames: list) -> None:
    """Write a CSV file with a stable header, even when there are no rows."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_particle_rows(results: list) -> list:
    """Flatten nested per-signal particle output into one row per particle."""
    rows = []
    for result in results:
        for particle in result['particles']:
            rows.append({
                'filename': result['filename'],
                'path': result['path'],
                'class': result['class'],
                'signal_idx': result['signal_idx'],
                'signal_length': result['signal_length'],
                'particle_idx': particle['idx'],
                'frequency': particle['frequency'],
                'P0': particle['P0'],
                't0': particle['t0'],
                'tau': particle['tau'],
                'phi': particle['phi'],
                'energy': particle['energy'],
                'snr_db': particle['snr_db'],
                'noise_floor': particle['noise_floor'],
                'noise_floor_N': particle['noise_floor_N'],
                'snr_method': particle['snr_method'],
                'source_window_idx': particle['source_window_idx'],
                'source_window_center': particle['source_window_center'],
                'source_window_energy': particle['source_window_energy'],
            })
    return rows


def build_noise_file_rows(results: list) -> list:
    """Flatten per-signal noise summaries into one row per file."""
    rows = []
    for result in results:
        row = {
            'filename': result['filename'],
            'path': result['path'],
            'class': result['class'],
            'signal_idx': result['signal_idx'],
            'signal_length': result['signal_length'],
            'num_particles': result['num_particles'],
            'num_windows': result['num_windows'],
            'num_valid_windows': result['num_valid_windows'],
            'noise_floor': result['noise_floor'],
            'noise_floor_N': result['noise_floor_N'],
            'snr_method': result['snr_method'],
        }
        row.update(result['noise_summary'])
        rows.append(row)
    return rows


def _mean_std(values: list) -> tuple:
    clean = np.asarray([v for v in values if v is not None], dtype=float)
    if len(clean) == 0:
        return None, None
    return float(np.mean(clean)), float(np.std(clean))


def build_noise_class_rows(noise_rows: list, particle_rows: list) -> list:
    """Aggregate per-file noise and per-particle SNR by class."""
    classes = sorted({row['class'] for row in noise_rows})
    rows = []
    for class_name in classes:
        class_noise = [r for r in noise_rows if r['class'] == class_name]
        class_particles = [r for r in particle_rows if r['class'] == class_name]
        raw_std_mean, raw_std_std = _mean_std([r['raw_std'] for r in class_noise])
        filtered_std_mean, filtered_std_std = _mean_std(
            [r['filtered_std'] for r in class_noise]
        )
        inband_mean, inband_std = _mean_std(
            [r['inband_energy_ratio'] for r in class_noise]
        )
        snr_mean, snr_std = _mean_std([r['snr_db'] for r in class_particles])
        rows.append({
            'class': class_name,
            'num_files': len(class_noise),
            'num_particles': len(class_particles),
            'raw_std_mean': raw_std_mean,
            'raw_std_std': raw_std_std,
            'filtered_std_mean': filtered_std_mean,
            'filtered_std_std': filtered_std_std,
            'inband_energy_ratio_mean': inband_mean,
            'inband_energy_ratio_std': inband_std,
            'snr_db_mean': snr_mean,
            'snr_db_std': snr_std,
        })
    return rows


def export_results(results: list, output_dir: str, processing_time_ms: float) -> None:
    """Export results to a JSON file with dataset information."""
    # Collect folder information
    folders = sorted(set(r['class'] for r in results))
    output_path = os.path.join(output_dir, "dataset_results.json")
    
    output = {
        'dataset_info': {
            'total_signals': len(results),
            'folders_processed': folders,
            'processing_time_ms': processing_time_ms,
            'snr_method': 'peak_bin_energy_over_lowest_window_energy',
        },
        'signals': results,
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    particle_rows = build_particle_rows(results)
    noise_rows = build_noise_file_rows(results)
    class_rows = build_noise_class_rows(noise_rows, particle_rows)

    export_csv(
        particle_rows,
        os.path.join(output_dir, "snr_particles.csv"),
        [
            'filename', 'path', 'class', 'signal_idx', 'signal_length',
            'particle_idx', 'frequency', 'P0', 't0', 'tau', 'phi', 'energy',
            'snr_db', 'noise_floor', 'noise_floor_N', 'snr_method',
            'source_window_idx', 'source_window_center', 'source_window_energy',
        ],
    )
    export_csv(
        noise_rows,
        os.path.join(output_dir, "noise_by_file.csv"),
        [
            'filename', 'path', 'class', 'signal_idx', 'signal_length',
            'num_particles', 'num_windows', 'num_valid_windows', 'noise_floor',
            'noise_floor_N', 'snr_method', 'raw_mean', 'raw_std', 'raw_rms',
            'raw_kurtosis', 'filtered_mean', 'filtered_std', 'filtered_rms',
            'filtered_kurtosis', 'inband_energy_ratio', 'spectral_flatness',
        ],
    )
    export_csv(
        class_rows,
        os.path.join(output_dir, "noise_by_class.csv"),
        [
            'class', 'num_files', 'num_particles', 'raw_std_mean',
            'raw_std_std', 'filtered_std_mean', 'filtered_std_std',
            'inband_energy_ratio_mean', 'inband_energy_ratio_std',
            'snr_db_mean', 'snr_db_std',
        ],
    )


def main():
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset_dir):
        print(f"Error: Dataset directory not found: {args.dataset_dir}")
        return
    os.makedirs(args.output_dir, exist_ok=True)
    
    data_files = load_all_data(args.dataset_dir, FOLDERS_TO_PROCESS)
    if not data_files:
        print("No .npy files found in the dataset directory.")
        return
    
    if args.verbose:
        print(f"Found {len(data_files)} signals to process")
    
    # Process all signals
    results = []
    total_start = time.perf_counter()
    
    for signal_idx, (file_path, folder_name) in enumerate(tqdm(data_files, desc="Processing")):
        config = get_config_for_folder(folder_name)
        result = process_signal(file_path, folder_name, config, args, signal_idx)
        if result is not None:
            results.append(result)
    
    total_time = time.perf_counter() - total_start
    processing_time_ms = total_time * 1000
    
    if args.verbose:
        print(f"\nTotal processing time: {processing_time_ms:.2f} ms ({total_time:.2f} s)")
        print(f"Average per signal: {processing_time_ms / len(data_files):.2f} ms")
    
    # Export results
    export_results(results, args.output_dir, processing_time_ms)
    
    print(f"Results saved to: {os.path.join(args.output_dir, 'dataset_results.json')}")
    print(f"SNR table saved to: {os.path.join(args.output_dir, 'snr_particles.csv')}")
    print(f"Noise tables saved to: {os.path.join(args.output_dir, 'noise_by_file.csv')} and {os.path.join(args.output_dir, 'noise_by_class.csv')}")
    print(f"Total signals processed: {len(results)}")


if __name__ == "__main__":
    main()
