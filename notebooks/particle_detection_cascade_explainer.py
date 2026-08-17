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
# # Particle-event detection in 1-D traces
# ## Why separating localisation and classification worked
#
# **Guiding question.** Why could a detector locate particle events while
# repeatedly assigning them the wrong size class, and what changed when
# localisation, event validation, and classification were separated?
#
# This executable notebook follows the shortest evidence chain needed to answer
# that question. It reads shipped models and frozen analyses; it does not train,
# tune, or select new test examples. MAD v2.1 intervals are deterministic
# teacher pseudo-labels, not independent physical truth.

# %% tags=["hide-input"] jupyter={"source_hidden": true}
import csv
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import Markdown, display

from detseg.models import build_detector
from detseg.postprocess import rebuild_model
from detseg.train_detection import remap_targets_class_agnostic
from internship_workspace.config import Workspace
from internship_workspace.datasets import resolve_path, select_record
from internship_workspace.mad_conv1dgap_training import (
    DATASET_ID,
    DATASET_KEY,
    EXPECTED_EVENT_COUNTS,
    resolve_registered_dataset,
)
from internship_workspace.particle_detection_cascade_figures import (
    factorial_freeze_facts,
    final_arm_results,
    grid_responsibility,
    inspect_detector_shapes,
    inspect_swin_stage_shapes,
    oracle_crop_facts,
    plot_b1_r1_design,
    plot_cascade_validation_cases,
    plot_factorial_design,
    plot_final_error_anatomy,
    plot_final_arm_results,
    plot_grid_responsibility,
    plot_initial_detector_results,
    plot_l1_r2_design,
    plot_localized_misclassification,
    preprocessing_results,
    r1_intermediate_facts,
    select_final_validation_cases,
    select_localized_misclassification,
    summarize_r2_development_exports,
)
from p0.models import ProposalAwareROIClassifier

# %% tags=["hide-input"] jupyter={"source_hidden": true}
workspace = Workspace.load()
dataset_record, dataset_root = resolve_registered_dataset(workspace)
assert dataset_record.key == DATASET_KEY

analysis_ids = (
    "particle-mad-causal-diagnostic-analysis-r1",
    "particle-mad-v21-b0-b1-r1-analysis-r1",
    "particle-mad-v21-final-cascade-analysis-r1",
    "particle-mad-v21-backbone-roi-fpr-analysis-r1",
    "particle-mad-v21-swin-fpr-operating-analysis-r1",
)
for run_id in analysis_ids:
    run = json.loads(
        (workspace.artifacts_root / "cross-project" / run_id / "run.json").read_text()
    )
    assert run["run_id"] == run_id and run["status"] == "complete"

print(
    f"Resolved {DATASET_KEY}: "
    + ", ".join(f"{split}={count}" for split, count in EXPECTED_EVENT_COUNTS.items())
    + " events. Frozen analyses verified."
)

backbone_summary = json.loads(
    (
        workspace.artifacts_root
        / "cross-project/particle-mad-v21-backbone-roi-fpr-analysis-r1"
        / "metrics/summary.json"
    ).read_text(encoding="utf-8")
)
operating_summary = json.loads(
    (
        workspace.artifacts_root
        / "cross-project/particle-mad-v21-swin-fpr-operating-analysis-r1"
        / "metrics/summary.json"
    ).read_text(encoding="utf-8")
)
swin_l_r2_ranking = backbone_summary["families"]["swin1d"]["L_R2"][
    "ranking_metrics"
]
swin_operating_point = operating_summary["operating_point"]
swin_l_r2_crossfit = operating_summary["primary_results"]["L_R2"]["crossfit"]
swin_crossfit_event = swin_l_r2_crossfit["held_out_event_metrics"]
swin_crossfit_ranking = swin_l_r2_crossfit[
    "held_out_ranking_metrics_after_threshold"
]
swin_crossfit_empty = swin_l_r2_crossfit["held_out_empty_traces"]

assert operating_summary["splits_loaded"] == ["val"]
assert operating_summary["test_loaded"] is False
assert swin_operating_point["selected_arm"] == "L_R2"
assert np.isclose(swin_l_r2_ranking["mAP@0.5"], 0.6302749869594989)
assert np.isclose(
    swin_l_r2_ranking["class_agnostic_event_AP@0.5"], 0.8060108577863514
)
assert np.isclose(
    swin_l_r2_ranking["per_class_AP@0.5"]["10um"], 0.5331330346781183
)
assert np.isclose(swin_crossfit_empty["activation_rate"], 7 / 86)
assert np.isclose(swin_crossfit_event["macro_f1"], 0.6205238738475591)
assert np.isclose(swin_crossfit_event["event_precision"], 0.7246376811594203)
assert np.isclose(swin_crossfit_event["event_recall"], 0.5543237250554324)
assert np.isclose(swin_crossfit_ranking["mAP@0.5"], 0.48228247784018125)
assert np.isclose(
    swin_crossfit_ranking["per_class_AP@0.5"]["10um"], 0.471413940322664
)
assert np.isclose(swin_operating_point["deployment_threshold"], 0.39910898756890506)

# %% [markdown]
# ## 0 · What was trained, on what, and how
#
# Each input is a **16,384-sample 1-D trace**. A MAD detector supplies event
# intervals, which become YOLO boxes; the acquisition folder supplies the
# 2/4/10 µm class. A trace with no retained MAD interval remains a negative
# example under that teacher reference.
#
# The initial experiment used MAD v1. The later cascade used MAD v2.1, which
# repaired saturation before annotation and removed proposals centred inside a
# repaired interval. The trace identities and split assignments stayed fixed.

# %% tags=["hide-input"] jupyter={"source_hidden": true}
with (dataset_root / "source_manifest.csv").open(newline="", encoding="utf-8") as handle:
    v21_sources = list(csv.DictReader(handle))
v21_empty_traces = sum(
    row["empty_mad_label"].lower() in {"1", "true"} for row in v21_sources
)
v21_trace_counts = {
    split: sum(row["output_split"] == split for row in v21_sources)
    for split in ("train", "val", "test")
}
assert sum(v21_trace_counts.values()) == 2888
assert v21_empty_traces == 783

v1_record = select_record(workspace, DATASET_ID, "v1")
v1_root = resolve_path(workspace, v1_record)
with (v1_root / "source_manifest.csv").open(newline="", encoding="utf-8") as handle:
    v1_sources = list(csv.DictReader(handle))
with (v1_root / "events.csv").open(newline="", encoding="utf-8") as handle:
    v1_events = list(csv.DictReader(handle))
v1_event_counts = {
    split: sum(row["output_split"] == split for row in v1_events)
    for split in ("train", "val", "test")
}
assert sum(v1_event_counts.values()) == 3749
v1_assignments = {
    row["source_id"]: (row["output_split"], row["source_class"])
    for row in v1_sources
}
v21_assignments = {
    row["source_id"]: (row["output_split"], row["source_class"])
    for row in v21_sources
}
assert v1_assignments == v21_assignments

historical_root = (
    workspace.artifacts_root
    / "cross-project/remote-pfcalcul/notebook-cascade-historical-source-r1/extra"
    / "artifacts/SMI_Detection_CNN_transformers/detection"
    / "particle-mad-teacher-swin-yolo-r1"
)
historical = json.loads(
    (
        historical_root
        / "runs/swin1d-yolo-seed42-particle-mad-teacher-swin-yolo-r1.json"
    ).read_text(encoding="utf-8")
)
assert historical["dataset"].endswith("@v1")
assert historical["train_samples"] == v21_trace_counts["train"]
assert historical["val_samples"] == v21_trace_counts["val"]
assert historical["test_samples"] == v21_trace_counts["test"]

dataset_table = [
    "| Split | Traces | MAD v1 events | MAD v2.1 events |",
    "|---|---:|---:|---:|",
]
for split in ("train", "val", "test"):
    dataset_table.append(
        f"| {split} | {v21_trace_counts[split]:,} | {v1_event_counts[split]:,} | "
        f"{EXPECTED_EVENT_COUNTS[split]:,} |"
    )
dataset_table.append(
    f"| **Total** | **{sum(v21_trace_counts.values()):,}** | "
    f"**{sum(v1_event_counts.values()):,}** | "
    f"**{sum(EXPECTED_EVENT_COUNTS.values()):,}** |"
)
display(
    Markdown(
        "\n".join(dataset_table)
        + f"\n\nMAD-empty traces in v2.1: **{v21_empty_traces:,}**."
    )
)

# %% [markdown]
# The starting model was a **Swin1D backbone with a one-scale YOLO head**. It
# jointly predicted objectness, centre, width, and three class probabilities at
# each of 512 grid cells. Training used balanced class weights and selected the
# checkpoint with the best validation performance.
#
# **Average precision (AP)** is the area under a class precision–recall curve.
# `AP@0.5` counts a prediction as localised when its intersection over union
# (IoU) with a reference box reaches 0.5; **mAP** is the mean AP over the three
# size classes.

# %% tags=["hide-input"] jupyter={"source_hidden": true}
training_table = [
    "| Training choice | Frozen value |",
    "|---|---:|",
    f"| Dataset | `{historical['dataset']}` |",
    f"| Model | `{historical['backbone']} + {historical['head']}` · "
    f"`{historical['total_params']:,}` parameters |",
    f"| Input / output grid | `{historical['input_length']:,}` samples / `512` cells |",
    "| Selection | best validation checkpoint |",
]
display(Markdown("\n".join(training_table)))

# %%
figure, axis = plt.subplots(figsize=(8.5, 3.5), constrained_layout=True)
plot_initial_detector_results(historical, ax=axis)
plt.show()

# %% [markdown]
# The baseline reached **42.3% mAP@0.5**, but AP 10 µm was only **12.2%**, far
# below AP 2 µm and AP 4 µm. The rest of the notebook explains this specific
# failure, then rebuilds the system on the corrected MAD v2.1 reference.
#
# The same short names are used from this point onward:
#
# | Name | Component | What it provides |
# |---|---|---|
# | Historical detector | Original multiclass Swin–YOLO trained on MAD v1 | starting reference |
# | **B0** | Multiclass Swin–YOLO retrained on MAD v2.1 | boxes, objectness, native classes |
# | **B1** | B0 proposals after objectness-ranked, class-agnostic NMS | fixed B0 boxes and native classes |
# | **R1** | First ROI classifier applied to B1 proposals | replacement class probabilities |
# | **L1** | Class-agnostic Swin–YOLO localiser | boxes and objectness |
# | **R2** | Proposal-aware ROI event validator and classifier | event and class probabilities |
#
# A combination names its two roles: `B1 + R1` uses B1 boxes with R1 classes;
# `L1 + R2` uses L1 boxes with R2 event and class scores.

# %% [markdown]
# ## Resolution in one page
# ### A · Start from the diagnostic
#
# **Why was good localisation not enough?** The initial Swin–YOLO found most
# 10 µm events, but frequently called them 4 µm. This dissociation showed that
# the signal required for localisation was present, while the class head had no
# aggregation explicitly aligned with the complete event.
#
# Two corrections structured the resolution. The source and pseudo-labels were
# first stabilised in **MAD v2.1**. The task was then split between a
# class-agnostic localiser and a region-of-interest (**ROI**) classifier seeing
# 6,144 samples, enough to cover a complete MAD box.

# %% [markdown]
# ### B · The useful experimental path
#
# | Stage | Question | Conclusion |
# |---|---|---|
# | B0 → B1 + R1 | Does an event-covering ROI improve classification? | Yes, but R1 does not reject background well enough. |
# | L1 + R2 | Should localisation and classification be separated? | Yes: class-agnostic localisation plus proposal-aware ROI. |
# | Backbones | Is the conclusion specific to Swin? | Class-agnostic localisation is robust; the ROI gain is most convincing with Swin. |
# | False-positive rate (FPR) | Which system should operate in practice? | `L1 + R2`, at threshold `0.3991`. |
#
# This summary replaces a chronology of jobs. The following sections retain
# only the experiments needed to understand each transition.
# In the later multi-backbone robustness check, the shorter labels `L` and `M`
# mean class-agnostic and multiclass localisation respectively; the Swin `L`
# model is the `L1` model developed here.

# %% [markdown]
# ### C · Separate ranking performance from the operating point
#
# **Ranking performance.** Before operating calibration, `L1 + R2` reaches
# **63.0%** mAP@0.5, **80.6%** Event AP, and **53.3%** AP 10 µm on validation.
# Event AP measures localisation ranking without requiring the correct size
# class. Swin remains the best backbone family for this system.
#
# **Operating point.** Thresholds are then calibrated by deterministic
# five-fold cross-validation under a primary budget of 10% activation on
# MAD-empty traces. The decision order was fixed before reading the result:
# macro-F1, precision, AP 10 µm, then the highest threshold. It selects
# `L1 + R2`.
#
# ```text
# Four folds: choose a threshold under the 10% budget
#                                        ↓
# Held-out fold: evaluate it once → rotate the held-out fold → aggregate
# ```
#
# Thus every reported cross-fit prediction is evaluated with a threshold chosen
# without that trace. A separate threshold is fitted on all validation traces
# only after the arm has been selected for deployment.

# %%
operating_result_path = (
    workspace.artifacts_root
    / "cross-project/reviews/particle-mad-v21-swin-fpr-operating-result-r1"
    / "result.png"
)
assert operating_result_path.is_file()
operating_result_image = plt.imread(operating_result_path)
figure, axis = plt.subplots(figsize=(14.0, 8.5), constrained_layout=True)
axis.imshow(operating_result_image)
axis.axis("off")
plt.show()

# %% [markdown]
# ### D · Operating conclusion and boundary
#
# | Cross-fit measurement | `L1 + R2` |
# |---|---:|
# | MAD-empty activation | **8.1%** |
# | Macro-F1 | **62.1%** |
# | Precision | **72.5%** |
# | Recall | **55.4%** |
# | mAP after thresholding | **48.2%** |
# | AP 10 µm after thresholding | **47.1%** |
# | Final threshold | **0.3991** |
#
# The change from 63.0% to 48.2% mAP is not a regression. The first number
# measures global proposal ranking; the second describes the system after
# cross-fit selection under the MAD-empty activation constraint.
#
# **Conclusion.** The results support a cascade: Swin localises without imposing
# a class, then R2 rejects background and predicts size from the complete
# support. `L1 + R2` favours macro-F1 balance and precision; multiclass
# localisation plus R2 retains
# more recall and mAP, but gives a less balanced operating classification.
#
# Here, “FPR” means activation of traces without MAD v2.1 pseudo-GT, not a rate
# measured on certified physical negatives. The next scientifically useful step
# would therefore be an independent campaign with human-reviewed negatives.
#
# ```text
# 10→4 confusion → complete ROI → separate localisation / classification
#                → multi-backbone validation → false-alarm calibration
# ```
#
# <details><summary>Reproducibility and artifacts</summary>
#
# Numbers are read from `particle-mad-v21-backbone-roi-fpr-analysis-r1` and
# `particle-mad-v21-swin-fpr-operating-analysis-r1`. The figure is checkpoint
# `particle-mad-v21-swin-fpr-operating-result-r1`. This summary is restricted to
# validation; it loads no test signal.
#
# </details>

# %% [markdown]
# ## 1 · The problem: localised but misclassified
#
# The initial failure was structured. Among **175 historical 10 µm events**,
# the detector localised **148** at intersection over union (IoU) ≥ 0.5, yet
# only **14** became correct class-aware detections: **9.5% of the events it had
# already found**. It assigned **121** of them to 4 µm.
#
# IoU measures box overlap. **Objectness** estimates whether an event exists at
# a location. Neither guarantees the correct size class. The validation example
# below makes that distinction concrete without reopening a test waveform.

# %% tags=["hide-input"] jupyter={"source_hidden": true}
ten_micron = historical["test_per_class_prf"][2]
confusion_10um = historical["test_confusion_at_f1"][2]
assert ten_micron["support"] == 175 and ten_micron["tp"] == 14
assert confusion_10um[1] == 121

development_root = (
    workspace.artifacts_root
    / "cross-project/remote-pfcalcul/notebook-cascade-section1-source-r1/extra"
    / "artifacts/cross-project/particle-mad-v21-common-proposals-b1-r1"
    / "development_detector"
)
case = select_localized_misclassification(
    development_root / "ground_truth.csv",
    development_root / "proposals.csv",
)
assert case.split == "val"
signal = np.load(dataset_root / "val/signals" / f"{case.trace_id}.npy", allow_pickle=False)

figure, axes = plt.subplots(1, 2, figsize=(13, 3.4), constrained_layout=True)
plot_localized_misclassification(signal, case, axes=axes)
figure.suptitle(
    "A confident localisation can still have the wrong class",
    x=0.01,
    ha="left",
    fontsize=13,
    fontweight="bold",
)
plt.show()

# %% [markdown]
# The box strongly overlaps the MAD interval and objectness is near one, but
# the 4 µm probability outranks 10 µm. This is why localisation recall,
# conditional classification, and class-aware average precision (AP) must be
# read separately: AP requires both the correct box and the correct class to be
# ranked ahead of errors.

# %% [markdown]
# ## 2 · Why the joint detector could fail
#
# The initial model solved everything from one deepest feature map:
#
# ```text
# 16,384 samples → Swin1D backbone → 512-cell map
#                → cell-wise YOLO head → objectness + box + 3 classes
# ```
#
# The architecture is reconstructed from its frozen run below. Hooks measure
# the tensors during a real forward pass rather than relying on copied
# dimensions.

# %% tags=["hide-input"] jupyter={"source_hidden": true}
detector = rebuild_model(historical).cpu()
shape_contract = inspect_detector_shapes(detector, input_length=historical["input_length"])
swin_contract = inspect_swin_stage_shapes(
    detector.backbone, input_length=historical["input_length"]
)
assert shape_contract.total_parameters == historical["total_params"]
assert shape_contract.output_shape[1] == 1 + historical["num_classes"] + 2
assert swin_contract.input_shape == shape_contract.input_shape
assert swin_contract.projection3_shape == shape_contract.head_cell_input_shape

patch = detector.backbone.patch_embed
merge1 = detector.backbone.merge1.reduction
projection1 = detector.backbone.projections[0]
window_size = detector.backbone.stage1[0].window

def channels_length(shape):
    return f"({shape[1]:,}, {shape[2]:,})"

display(Markdown(
    f"Measured input: `{shape_contract.input_shape}` in "
    "`(batch, channel, length)` order."
))

stage_table = [
    "| Step | Operation read from the model | Output `(channels, length)` |",
    "|---|---|---:|",
    f"| Patch embed | `Conv1d(1, {patch.out_channels}, kernel={patch.kernel_size[0]}, "
    f"stride={patch.stride[0]})` + LayerNorm | `{channels_length(swin_contract.patch_embed_shape)}` |",
    f"| Stage 1 | `{len(detector.backbone.stage1)}` Swin blocks, window `{window_size}` | "
    f"`{channels_length(swin_contract.stage1_shape)}` |",
    f"| Patch merge 1 | concatenate token pairs → `Linear({merge1.in_features}→{merge1.out_features})` "
    f"+ LayerNorm | `{channels_length(swin_contract.merge1_shape)}` |",
    f"| Pyramid projection P1 | `Conv1d({projection1[0].in_channels}, "
    f"{projection1[0].out_channels}, kernel=1)` + `GroupNorm({projection1[1].num_groups})` | "
    f"`{channels_length(swin_contract.projection1_shape)}` |",
]
display(Markdown("\n".join(stage_table)))

# %% [markdown]
# P1 is a lateral projection, not a top-down FPN fusion: the transformer trunk
# continues from the unprojected 128-channel tokens. Two further stages repeat
# the same pattern and form the measured three-level pyramid:

# %% tags=["hide-input"] jupyter={"source_hidden": true}
pyramid_rows = (
    ("P1", swin_contract.merge1_shape, swin_contract.projection1_shape, shape_contract.strides[0]),
    ("P2", swin_contract.merge2_shape, swin_contract.projection2_shape, shape_contract.strides[1]),
    ("P3", swin_contract.merge3_shape, swin_contract.projection3_shape, shape_contract.strides[2]),
)
pyramid_table = [
    "| Level | Raw Swin feature | Projected feature | Input stride | Used by this YOLO head? |",
    "|---|---:|---:|---:|---|",
]
for level, raw_shape, projected_shape, stride in pyramid_rows:
    pyramid_table.append(
        f"| {level} | `{channels_length(raw_shape)}` | "
        f"`{channels_length(projected_shape)}` | `{stride}` | "
        f"{'**yes**' if level == 'P3' else 'no'} |"
    )
display(Markdown("\n".join(pyramid_table)))

# %% [markdown]
# ### How the one-scale YOLO head uses its 512 slots
#
# This is the historical, backward-compatible **anchor-free, single-scale**
# head. It deliberately consumes only P3. Its complete measured output contract
# is displayed before the meaning of each value.

# %% tags=["hide-input"] jupyter={"source_hidden": true}
head_conv = detector.head.head
display(Markdown(
    f"`Conv1d({head_conv.in_channels}, {head_conv.out_channels}, kernel="
    f"{head_conv.kernel_size[0]})` produces `{shape_contract.output_shape}`: "
    f"{head_conv.out_channels} values at each of {shape_contract.output_shape[-1]} slots. "
    f"The detector has `{shape_contract.trainable_parameters:,}` trainable parameters."
))

head_table = [
    "| Per-slot output | Activation | Role | Supervision |",
    "|---|---|---|---|",
    "| 1 objectness logit | sigmoid | is an event centred in this slot? | every valid slot |",
    "| 3 class logits | softmax | probability of 2, 4, or 10 µm | positive slot only |",
    "| 1 centre-offset logit | sigmoid | centre position inside the slot | positive slot only |",
    "| 1 log-width | exponential at decoding | event width in input samples | positive slot only |",
]
display(Markdown("\n".join(head_table)))

# %% [markdown]
# The assignment and decoding are simple:
#
# 1. a labelled event is assigned to the stride-32 slot containing its centre;
# 2. that slot learns objectness, class, centre offset, and width, while the
#    other slots learn background objectness;
# 3. if two labels share a slot, the smaller event is retained deterministically;
# 4. at inference, each slot decodes one box and receives the native score
#    `objectness × highest class probability`;
# 5. thresholding and class-agnostic 1-D NMS remove weak and overlapping boxes.
#
# This version was chosen to reproduce the historical detector. The important
# consequence is visible below: an event may cover many slots, but only its
# centre slot owns the target, and the head never pools the predicted interval.

# %%
responsibility = grid_responsibility(6200, 10200, stride=shape_contract.strides[-1])
figure, axis = plt.subplots(figsize=(11.5, 3.2), constrained_layout=True)
plot_grid_responsibility(responsibility, ax=axis)
plt.show()

# %% [markdown]
# A long event spans many grid cells, but one centre cell emits its box and
# class. Swin attention gives that feature wider context than a single stride;
# the precise limitation is that the head performs **no explicit pooling aligned
# with the full predicted interval**. Box regression can therefore succeed
# while class evidence distributed across the waveform remains difficult to
# aggregate.

# %% [markdown]
# ## 3 · Why a crop classifier was not enough
#
# Three intermediate observations narrowed the problem:
#
# 1. changing preprocessing helped, but the best gain remained below the
#    pre-registered causal threshold;
# 2. a 6,144-sample oracle crop exposed the complete target support, making
#    classification easier than full-trace detection;
# 3. R1 improved class ranking on fixed proposals, but it neither moved boxes
#    nor learned an explicit event-versus-background decision.
#
# The cells below verify those facts from the frozen analyses. The detailed
# preprocessing sweep is intentionally not reproduced here.

# %% tags=["hide-input"] jupyter={"source_hidden": true}
preprocessing_root = workspace.artifacts_root / "cross-project/particle-preprocessing-comparison-results-r1"
preprocessing_summary = json.loads(
    (preprocessing_root / "summary.json").read_text(encoding="utf-8")
)
representations = preprocessing_results(preprocessing_summary)
baseline_preprocessing = representations[0]
best_preprocessing = max(representations[1:], key=lambda row: row.macro_f1_delta)

ceiling_root = workspace.artifacts_root / "cross-project/particle-classification-ceiling-method-analysis-r2"
ceiling_summary = json.loads((ceiling_root / "summary.json").read_text(encoding="utf-8"))
crop_facts = oracle_crop_facts(ceiling_summary)

b1_root = (
    workspace.artifacts_root
    / "cross-project/remote-pfcalcul/notebook-cascade-section1-source-r1/extra"
    / "artifacts/cross-project/particle-mad-v21-common-proposals-b1-r1"
)
intermediate_root = workspace.artifacts_root / "cross-project/particle-mad-v21-b0-b1-r1-analysis-r1"
intermediate_summary = json.loads(
    (intermediate_root / "metrics/summary.json").read_text(encoding="utf-8")
)
r1_result = r1_intermediate_facts(intermediate_summary)
bridge = intermediate_summary["historical_checkpoint_bridge"]
bridge_effects = intermediate_summary["historical_bridge_decomposition"]
b0_ranking = intermediate_summary["arms"]["B0"]["unthresholded"]

assert best_preprocessing.macro_f1_delta < 0.07
assert crop_facts.support_coverage_6144 == 1.0
assert r1_result.map_ci95_low < 0.0 < r1_result.map_ci95_high
assert np.isclose(
    bridge_effects["source_correction_effect"]["mAP@0.5"],
    bridge["v2.1_signals_v1_labels"]["mAP@0.5"]
    - bridge["v1_signals_v1_labels"]["mAP@0.5"],
)
assert np.isclose(
    bridge_effects["retraining_effect"]["mAP@0.5"],
    b0_ranking["mAP@0.5"] - bridge["v2.1_signals_v2.1_labels"]["mAP@0.5"],
)

figure, axis = plt.subplots(figsize=(11.5, 4.0), constrained_layout=True)
plot_b1_r1_design(ax=axis)
plt.show()

# %%
diagnostic_table = [
    "| Observation | Frozen result | Practical meaning |",
    "|---|---:|---|",
    f"| Best crop preprocessing | `{baseline_preprocessing.macro_f1:.1%} → "
    f"{best_preprocessing.macro_f1:.1%}` macro-F1 "
    f"(`{100 * best_preprocessing.macro_f1_delta:+.1f}` points) | useful, but below the `+7 point` gate; this is **not detector mAP** |",
    f"| Saturation/source correction, same checkpoint and v1 labels | "
    f"`{100 * bridge_effects['source_correction_effect']['mAP@0.5']:+.1f}` mAP@0.5 points | a small detector gain |",
    f"| Switch from v1 to v2.1 labels on corrected signals | "
    f"`{100 * bridge_effects['reference_change_effect']['mAP@0.5']:+.1f}` mAP@0.5 points | the reference change removes that aggregate gain |",
    f"| B0 retraining on v2.1 | "
    f"`{bridge['v2.1_signals_v2.1_labels']['mAP@0.5']:.1%} → {b0_ranking['mAP@0.5']:.1%}` mAP@0.5; "
    f"AP 10 µm `{bridge['v2.1_signals_v2.1_labels']['per_class_AP@0.5']['10um']:.1%} → "
    f"{b0_ranking['per_class_AP@0.5']['10um']:.1%}` | 10 µm improves, but end-to-end mAP does not |",
    f"| Full support inside a 6,144 crop | `{crop_facts.support_coverage_6144:.0%}` | the oracle supplies localisation |",
    f"| B1 → R1 class-aware mAP | `{r1_result.b1_map:.1%} → {r1_result.r1_map:.1%}` | ROI aggregation helps class ranking |",
    f"| B1 / R1 Event AP | `{r1_result.common_event_ap:.1%}` for both | the boxes and objectness are unchanged |",
]
display(Markdown("\n".join(diagnostic_table)))

# %% [markdown]
# R1 supported the architectural hypothesis, but not a complete solution. Its
# mAP gain had a paired 95% interval crossing zero, and at a common threshold it
# activated more MAD-empty traces than B1. The missing pieces were now clear:
# improve localisation without class competition, and teach the ROI model what
# background looks like.

# %% [markdown]
# ## 4 · The final L1 + R2 cascade
#
# **L1** is a class-agnostic detector. During training every bead class is
# remapped to one event category; objectness and box regression remain, while
# size classification is removed from the localisation head.

# %%
toy_targets = [torch.tensor([[2.0, 0.30, 0.10], [1.0, 0.70, 0.20]])]
remapped_targets = remap_targets_class_agnostic(toy_targets)
assert torch.equal(remapped_targets[0][:, 1:], toy_targets[0][:, 1:])
assert torch.equal(remapped_targets[0][:, 0], torch.zeros(2))

toy_class_names = ("2 µm", "4 µm", "10 µm")
toy_rows = [
    "| Toy event | Multiclass target | L1 target | Centre / width |",
    "|---|---|---|---:|",
]
for index, (before, after) in enumerate(zip(toy_targets[0], remapped_targets[0]), start=1):
    l1_target = "event" if int(after[0]) == 0 else f"class {int(after[0])}"
    toy_rows.append(
        f"| {index} | {toy_class_names[int(before[0])]} | {l1_target} | "
        f"{before[1]:.2f} / {before[2]:.2f} |"
    )
display(Markdown("\n".join(toy_rows)))

l1_model = build_detector("swin1d", num_classes=1, head="yolo").cpu()
l1_contract = inspect_detector_shapes(l1_model, input_length=16_384)
assert l1_contract.output_shape == (1, 4, 512)

# %% [markdown]
# **R2** is proposal-aware. It sees the same 6,144-sample, locally z-scored
# crops available at inference and learns two outputs from one encoder:
#
# - event versus background;
# - 2/4/10 µm, only when the proposal is a positive event.
#
# Proposals with IoU ≥ 0.5 are positive, those below 0.1 are background, and
# the uncertain interval between them is excluded from both losses.

# %%
r2_model = ProposalAwareROIClassifier(input_length=6_144, num_classes=3).cpu()
with torch.inference_mode():
    r2_event_logits, r2_class_logits = r2_model(torch.zeros(1, 1, 6_144))
assert r2_event_logits.shape == (1,) and r2_class_logits.shape == (1, 3)
r2_parameters = sum(parameter.numel() for parameter in r2_model.parameters())

r2_index_summary = summarize_r2_development_exports(
    b1_root / "development_detector/proposals.csv",
    b1_root / "development_detector/ground_truth.csv",
)
train_index = r2_index_summary["by_split"]["train"]
assert train_index["positive"] == 4092
assert train_index["background"] == 29274
assert train_index["ambiguous"] == 5914

figure, axis = plt.subplots(figsize=(11.5, 4.0), constrained_layout=True)
plot_l1_r2_design(
    ax=axis,
    l1_output_channels=l1_contract.output_shape[1],
    r2_parameters=r2_parameters,
)
plt.show()

# %% [markdown]
# Let \(o\) be the objectness predicted by the localiser. The final score for
# class \(k\) is
#
# \[
# s_k=o\,P(\mathrm{event}\mid\mathrm{ROI})\,
# P(k\mid\mathrm{event},\mathrm{ROI}).
# \]
#
# L1 can change proposal geometry and recall. R2 cannot move a box; it can only
# reject it or re-rank its class. This separation gives each component one
# responsibility and makes their effects independently measurable.

# %% [markdown]
# ## 5 · The causal comparison
#
# Rows change the localiser; columns change the ROI classifier. Horizontal
# comparisons keep proposal IDs, boxes, and objectness exactly fixed. Vertical
# comparisons keep the classifier and crop rule fixed.

# %%
final_result_root = workspace.artifacts_root / "cross-project/particle-mad-v21-final-cascade-analysis-r1"
final_summary = json.loads(
    (final_result_root / "metrics/summary.json").read_text(encoding="utf-8")
)
factorial_freeze = factorial_freeze_facts(final_summary)
assert factorial_freeze.test_role == "descriptive_non_independent_replication"

figure, axis = plt.subplots(figsize=(11.5, 4.8), constrained_layout=True)
plot_factorial_design(ax=axis)
plt.show()

# %% [markdown]
# The two horizontal differences isolate R2; the two vertical differences
# isolate L1. Their interaction is the difference of those differences. All
# four arms qualified on validation and were frozen before the final test was
# read. The hashes and qualification mechanics remain in the run manifest
# rather than in the pedagogical narrative.

# %% [markdown]
# ## 6 · Descriptive replication on the historical test split
#
# `B1 + R1` is the factorial reference on MAD v2.1, not the original MAD v1
# detector from section 0. The four arms below differ only by localiser and ROI
# classifier within the final experimental framework. This historically
# consumed test split checks whether the validation conclusion remains
# descriptively coherent; it does not select a second system or replace the
# validation-calibrated `L1 + R2` operating point reported above.

# %%
results = final_arm_results(final_summary)
figure, axis = plt.subplots(figsize=(11.5, 4.4), constrained_layout=True)
plot_final_arm_results(results, ax=axis)
plt.show()

# %%
baseline = results[0]
final_cascade = results[-1]
display(Markdown(
    f"**L1 + R2:** mAP@0.5 `{final_cascade.map_50:.1%}`, Event AP@0.5 "
    f"`{final_cascade.event_ap_50:.1%}`, and AP 10 µm `{final_cascade.ap_10um_50:.1%}`. "
    f"Against B1 + R1, the gains are "
    f"`{final_cascade.map_50 - baseline.map_50:+.1%}`, "
    f"`{final_cascade.event_ap_50 - baseline.event_ap_50:+.1%}`, and "
    f"`{final_cascade.ap_10um_50 - baseline.ap_10um_50:+.1%}` respectively."
))

# %% [markdown]
# This replication supports the same answer to the original paradox: explicit
# ROI aggregation recovered class information, class-agnostic training improved
# localisation, and the R2 event head supplied the background decision missing
# from R1. The full cascade reached **60.9% macro-F1** and **60.8% event
# precision** at its frozen operating threshold.
#
# The aggregate gain is real, but it does not mean every event type or operating
# condition is solved. The final section examines what remains difficult.

# %% [markdown]
# ## 7 · What the final cascade still gets wrong
#
# Three views expose the residual structure: class confusion after localisation,
# ranking performance by event width, and false activation on MAD-empty traces.

# %%
figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.2), constrained_layout=True)
plot_final_error_anatomy(final_summary, axes=axes)
plt.show()

# %%
final_arm = final_summary["arms"]["L1_R2"]
stable = final_arm["by_event_status"]["stable"]
new = final_arm["by_event_status"]["new"]
empty_controls = final_summary["mad_empty_controls"]["v2.1_empty"]["L1_R2"]
display(Markdown(
    "\n".join([
        "| Residual issue | Frozen observation |",
        "|---|---:|",
        f"| Localised 10 µm → 4 µm | `{final_arm['thresholded']['ten_to_four_rate']:.1%}` |",
        f"| Stable-event mAP / new-event mAP | `{stable['mAP@0.5']:.1%}` / `{new['mAP@0.5']:.1%}` |",
        f"| New MAD v2.1 events in test | `{new['events']}` |",
        f"| MAD-empty activation at the selected threshold | `{empty_controls['own_threshold']['activation_rate']:.1%}` |",
        f"| MAD-empty activation at the common B1+R1 threshold | `{empty_controls['common_B1_R1_threshold']['activation_rate']:.1%}` |",
    ])
))

# %% [markdown]
# The remaining errors are not uniform. Short events are hardest, the 19 newly
# admitted MAD v2.1 events are much less stable than the inherited cohort, and
# 10 µm still sometimes collapses into 4 µm. R2 strongly suppresses empty-trace
# scores at a common threshold, but its lower validation-selected threshold
# trades some of that suppression back for recall. Threshold choice therefore
# remains part of the deployed behaviour, not a cosmetic post-processing step.
#
# The following three cases are selected deterministically from **validation**:
# the highest-scoring correct 10 µm event, the highest-scoring remaining
# `10 µm → 4 µm` error, and the highest-objectness proposal rejected on a
# MAD-empty trace. They illustrate behaviour; they do not add test evidence.

# %% tags=["hide-input"] jupyter={"source_hidden": true}
r2_development_root = (
    workspace.artifacts_root
    / "cross-project/remote-pfcalcul/20260816_mad_swin_fpr_inputs/extra"
    / "artifacts/cross-project/particle-mad-v21-roi-r2-proposal-aware-s42-r1"
)
r2_development_meta = json.loads(
    (r2_development_root / "l1_r2_proposals.json").read_text(encoding="utf-8")
)
assert r2_development_meta["splits_loaded"] == ["train", "val"]
assert r2_development_meta["test_loaded"] is False

validation_empty_ids = sorted(
    row["output_stem"]
    for row in v21_sources
    if row["output_split"] == "val" and row["empty_mad_label"].lower() in {"1", "true"}
)
validation_cases = select_final_validation_cases(
    b1_root / "development_detector/ground_truth.csv",
    r2_development_root / "l1_r2_proposals.csv",
    empty_trace_ids=validation_empty_ids,
    threshold=float(final_arm["operating_threshold"]),
)
assert all(case.split == "val" for case in validation_cases)
validation_signals = {
    case.trace_id: np.load(
        dataset_root / "val/signals" / f"{case.trace_id}.npy",
        allow_pickle=False,
    )
    for case in validation_cases
}

figure, axes = plt.subplots(3, 1, figsize=(13.0, 8.8), constrained_layout=True)
plot_cascade_validation_cases(validation_cases, validation_signals, axes=axes)
plt.show()

# %% [markdown]
# The examples complete the metric-level picture. L1 can place a strong box on
# both a genuine event and structured background; R2 usually separates them,
# but a MAD-labelled 10 µm waveform can still receive a confident 4 µm
# class. The remaining limitation is therefore not one missing trick: it mixes
# difficult morphology, pseudo-label uncertainty, and operating-threshold
# tradeoffs.
#
# The conclusion remains bounded:
#
# - MAD v2.1 is a deterministic teacher reference, not independent physical GT;
# - the test had been consumed historically and is a descriptive replication;
# - the four arms use one seed and one acquisition family;
# - preprocessing, crop length, and calibration alternatives were investigated,
#   but are not reproduced here because they did not change the central answer.
#
# **Supported conclusion.** Within this dataset and model family, separating
# class-agnostic localisation, proposal validation, and ROI classification is
# more effective than asking one cell-wise head to solve all three tasks.
# Independent acquisition data with physically adjudicated labels is still
# required to establish that this advantage generalises beyond MAD v2.1.
