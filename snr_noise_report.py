"""Generate class-wise SNR and noise reports from particles2SNR CSV exports.

The script is intentionally downstream-only: it reads the CSV files produced by
``run_dataset.py`` and writes derived reports/figures without touching source
``.npy`` files.
"""

import argparse
import csv
import json
import math
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


DEFAULT_CLASSES = ("2um", "4um", "10um")


def read_csv_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def as_float(value):
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def values_by_class(rows, column, classes=None):
    grouped = defaultdict(list)
    class_filter = set(classes) if classes else None
    for row in rows:
        cls = row.get("class")
        if class_filter is not None and cls not in class_filter:
            continue
        value = as_float(row.get(column))
        if value is not None:
            grouped[cls].append(value)
    return {cls: np.asarray(vals, dtype=float) for cls, vals in grouped.items()}


def summarize(values):
    if len(values) == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "median": None,
            "iqr": None,
            "min": None,
            "max": None,
        }
    q25, q75 = np.percentile(values, [25, 75])
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "median": float(np.median(values)),
        "iqr": float(q75 - q25),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def normality_test(values):
    if len(values) < 3:
        return {"test": "not_enough_samples", "pvalue": None, "statistic": None}
    if len(values) <= 5000:
        stat, pvalue = stats.shapiro(values)
        return {"test": "shapiro", "statistic": float(stat), "pvalue": float(pvalue)}
    stat, pvalue = stats.normaltest(values)
    return {"test": "dagostino", "statistic": float(stat), "pvalue": float(pvalue)}


def compare_groups(grouped):
    clean = {cls: vals for cls, vals in grouped.items() if len(vals) >= 2}
    if len(clean) < 2:
        return {
            "overall": {"test": "not_enough_classes", "pvalue": None, "statistic": None},
            "pairwise": [],
        }

    arrays = list(clean.values())
    normal = all((normality_test(vals).get("pvalue") or 0.0) >= 0.05 for vals in arrays)
    if normal:
        stat, pvalue = stats.f_oneway(*arrays)
        overall = {"test": "anova", "statistic": float(stat), "pvalue": float(pvalue)}
    else:
        stat, pvalue = stats.kruskal(*arrays)
        overall = {"test": "kruskal", "statistic": float(stat), "pvalue": float(pvalue)}

    pairwise = []
    keys = sorted(clean)
    n_pairs = len(keys) * (len(keys) - 1) // 2
    for i, left in enumerate(keys):
        for right in keys[i + 1:]:
            stat, pvalue = stats.mannwhitneyu(clean[left], clean[right], alternative="two-sided")
            pairwise.append({
                "left": left,
                "right": right,
                "test": "mannwhitneyu",
                "statistic": float(stat),
                "pvalue": float(pvalue),
                "pvalue_bonferroni": float(min(1.0, pvalue * n_pairs)),
                "left_median": float(np.median(clean[left])),
                "right_median": float(np.median(clean[right])),
                "median_delta_left_minus_right": float(np.median(clean[left]) - np.median(clean[right])),
            })
    return {"overall": overall, "pairwise": pairwise}


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_snr_distribution(grouped, output_dir):
    if not grouped:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    classes = sorted(grouped)
    data = [grouped[cls] for cls in classes]

    try:
        axes[0].boxplot(data, tick_labels=classes, showmeans=True)
    except TypeError:
        axes[0].boxplot(data, labels=classes, showmeans=True)
    axes[0].set_title("SNR by class")
    axes[0].set_ylabel("SNR (dB)")
    axes[0].grid(axis="y", alpha=0.25)

    for cls, vals in grouped.items():
        if len(vals) > 0:
            axes[1].hist(vals, bins=min(40, max(8, int(np.sqrt(len(vals))))),
                         alpha=0.45, label=cls, density=True)
    axes[1].set_title("SNR distribution")
    axes[1].set_xlabel("SNR (dB)")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    path = os.path.join(output_dir, "snr_by_class_distribution.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_noise_metric(grouped, metric, output_dir):
    if not grouped:
        return None
    classes = sorted(grouped)
    means = [float(np.mean(grouped[cls])) for cls in classes]
    stds = [
        float(np.std(grouped[cls], ddof=1)) if len(grouped[cls]) > 1 else 0.0
        for cls in classes
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(classes, means, yerr=stds, capsize=4)
    ax.set_title(metric.replace("_", " "))
    ax.set_ylabel(metric)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = os.path.join(output_dir, f"{metric}_by_class.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path


def build_report(noise_rows, particle_rows, classes):
    metrics = {
        "snr_db": values_by_class(particle_rows, "snr_db", classes),
        "raw_std": values_by_class(noise_rows, "raw_std", classes),
        "filtered_std": values_by_class(noise_rows, "filtered_std", classes),
        "inband_energy_ratio": values_by_class(noise_rows, "inband_energy_ratio", classes),
        "spectral_flatness": values_by_class(noise_rows, "spectral_flatness", classes),
    }

    report = {
        "classes": list(classes),
        "metrics": {},
        "interpretation": {},
    }
    for metric, grouped in metrics.items():
        report["metrics"][metric] = {
            "summary_by_class": {
                cls: summarize(vals) for cls, vals in sorted(grouped.items())
            },
            "normality_by_class": {
                cls: normality_test(vals) for cls, vals in sorted(grouped.items())
            },
            "comparison": compare_groups(grouped),
        }

    noise_p = report["metrics"]["filtered_std"]["comparison"]["overall"].get("pvalue")
    snr_p = report["metrics"]["snr_db"]["comparison"]["overall"].get("pvalue")
    report["interpretation"]["same_noise_model_supported"] = (
        bool(noise_p is not None and noise_p >= 0.05)
    )
    report["interpretation"]["class_snr_difference_supported"] = (
        bool(snr_p is not None and snr_p < 0.05)
    )
    report["interpretation"]["snr_method_note"] = (
        "Current SNR is per detected particle but uses a per-file lowest-window "
        "noise floor. It is useful for ranking events within/across files; if "
        "filtered_std differs significantly by class, report class-aware noise "
        "normalization alongside the per-particle SNR."
    )
    return report, metrics


def write_markdown(report, output_path):
    lines = [
        "# particles2SNR SNR And Noise Report",
        "",
        "## Summary",
        "",
        f"- Same filtered-noise model supported: `{report['interpretation']['same_noise_model_supported']}`",
        f"- Class SNR difference supported: `{report['interpretation']['class_snr_difference_supported']}`",
        f"- SNR method note: {report['interpretation']['snr_method_note']}",
        "",
    ]
    for metric, info in report["metrics"].items():
        overall = info["comparison"]["overall"]
        lines.extend([
            f"## {metric}",
            "",
            f"- Overall test: `{overall['test']}`",
            f"- p-value: `{overall['pvalue']}`",
            "",
            "| class | n | mean | std | median | iqr |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for cls, summary in info["summary_by_class"].items():
            lines.append(
                f"| {cls} | {summary['n']} | {summary['mean']} | {summary['std']} | "
                f"{summary['median']} | {summary['iqr']} |"
            )
        lines.append("")
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Build class-wise SNR/noise figures and statistical reports."
    )
    parser.add_argument("--input-dir", default="output",
                        help="Directory containing particles2SNR CSV exports")
    parser.add_argument("--output-dir", default=None,
                        help="Derived report directory (default: input-dir/snr_noise_report)")
    parser.add_argument("--classes", default=",".join(DEFAULT_CLASSES),
                        help="Comma-separated classes to compare")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir or os.path.join(input_dir, "snr_noise_report"))
    os.makedirs(output_dir, exist_ok=True)

    noise_path = os.path.join(input_dir, "noise_by_file.csv")
    particles_path = os.path.join(input_dir, "snr_particles.csv")
    if not os.path.isfile(noise_path):
        raise FileNotFoundError(f"Missing {noise_path}; run particles2SNR_pipeline/run_dataset.py first")
    if not os.path.isfile(particles_path):
        raise FileNotFoundError(f"Missing {particles_path}; run particles2SNR_pipeline/run_dataset.py first")

    classes = tuple(item.strip() for item in args.classes.split(",") if item.strip())
    noise_rows = read_csv_rows(noise_path)
    particle_rows = read_csv_rows(particles_path)
    report, metrics = build_report(noise_rows, particle_rows, classes)

    json_path = os.path.join(output_dir, "snr_noise_report.json")
    md_path = os.path.join(output_dir, "snr_noise_report.md")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    write_markdown(report, md_path)

    pairwise_rows = []
    for metric, info in report["metrics"].items():
        for row in info["comparison"]["pairwise"]:
            pairwise_rows.append({"metric": metric, **row})
    write_csv(
        os.path.join(output_dir, "pairwise_comparisons.csv"),
        pairwise_rows,
        [
            "metric", "left", "right", "test", "statistic", "pvalue",
            "pvalue_bonferroni", "left_median", "right_median",
            "median_delta_left_minus_right",
        ],
    )

    figures = [
        plot_snr_distribution(metrics["snr_db"], output_dir),
        plot_noise_metric(metrics["raw_std"], "raw_std", output_dir),
        plot_noise_metric(metrics["filtered_std"], "filtered_std", output_dir),
        plot_noise_metric(metrics["inband_energy_ratio"], "inband_energy_ratio", output_dir),
    ]
    report["figures"] = [path for path in figures if path]
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Report JSON: {json_path}")
    print(f"Report Markdown: {md_path}")
    print(f"Figures: {len(report['figures'])}")


if __name__ == "__main__":
    main()
