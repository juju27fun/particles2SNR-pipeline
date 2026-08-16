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
# ## From a joint detector to a localisation–classification cascade
#
# **Guiding question.** Why could a detector locate particle events while
# repeatedly assigning them the wrong size class, and what changed when
# localisation, proposal validation, and classification were separated?
#
# This executable account preserves the investigation in evidence order,
# including hypotheses that did not fully explain the failure. The companion
# [`mad_detection_explainer`](mad_detection_explainer.py) stops where a raw
# trace becomes a bounded MAD event; this notebook starts there.

# %% [markdown]
# ## 0 · Reading contract
#
# | This notebook does | It does not |
# |---|---|
# | replay explanations on train/validation records | train, tune, or recalibrate |
# | inspect shipped models and frozen analyses | select a new test example |
# | separate localisation, proposal validation, and class | treat MAD labels as independent physical truth |
#
# MAD v2.1 provides deterministic **teacher pseudo-labels**, not human-verified
# physical truth. The historical test is already consumed: later test numbers
# come only from frozen summaries; every new case or diagnostic stays on
# train/validation.

# %%
import json

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, display

from detseg.postprocess import rebuild_model
from internship_workspace.config import Workspace
from internship_workspace.mad_conv1dgap_training import (
    DATASET_KEY,
    EXPECTED_EVENT_COUNTS,
    resolve_registered_dataset,
)
from internship_workspace.particle_detection_cascade_figures import (
    grid_responsibility,
    inspect_detector_shapes,
    oracle_crop_facts,
    plot_ap_ranking,
    plot_b1_r1_design,
    plot_conditioned_classification_task,
    plot_grid_responsibility,
    plot_localized_misclassification,
    plot_preprocessing_deltas,
    plot_r1_tradeoffs,
    plot_zscore_scale_invariance,
    preprocessing_results,
    r1_intermediate_facts,
    roi_training_facts,
    select_localized_misclassification,
    validation_arm_results,
)

# %%
workspace = Workspace.load()
dataset_record, dataset_root = resolve_registered_dataset(workspace)
assert dataset_record.key == DATASET_KEY

analysis_ids = (
    "particle-mad-causal-diagnostic-analysis-r1",
    "particle-mad-v21-b0-b1-r1-analysis-r1",
    "particle-mad-v21-final-cascade-analysis-r1",
)
for run_id in analysis_ids:
    run = json.loads(
        (workspace.artifacts_root / "cross-project" / run_id / "run.json").read_text()
    )
    assert run["run_id"] == run_id and run["status"] == "complete"

print(
    f"Resolved {DATASET_KEY}: "
    + ", ".join(f"{split}={count}" for split, count in EXPECTED_EVENT_COUNTS.items())
    + f" events; {len(analysis_ids)} frozen analyses available. No signal loaded."
)

# %% [markdown]
# ### Roadmap
#
# | Now | Next | Then |
# |---|---|---|
# | 0. Reading contract | 2. Initial Swin1D–YOLO | 5. B1 proposals + R1 |
# | 1. Initial paradox | 3. Preprocessing and crops | 6. Why R1 was insufficient |
# |  | 4. Why crop classification is easier | 7. L1 + R2 architecture |
# |  |  | 8. Causal 2×2 experiment |
# |  |  | 9. Frozen results |
# |  |  | 10–12. Errors, cases, conclusions |
#
# New calibration, multiplicity, amplitude, and saturation-join diagnostics
# will remain development-only.

# %% [markdown]
# ## 1 · The initial paradox: localised but misclassified
#
# The first result was not random failure. Among **175 historical 10 µm
# events**, the detector localised **148** at IoU ≥ 0.5, yet only **14** were
# correct class-aware detections at the operating threshold: about **9.5% of
# the localised events**. It sent **121** 10 µm events to 4 µm.
#
# These values describe the historical MAD v1 test and are not an independent
# validation result. The cell below checks the frozen run summary for the
# population, true positives, and structured confusion; prediction rows are
# deliberately not reopened.

# %%
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
ten_micron = historical["test_per_class_prf"][2]
confusion_10um = historical["test_confusion_at_f1"][2]
historical_localized_10um = 148  # frozen class-agnostic localisation audit
assert ten_micron["support"] == 175 and ten_micron["tp"] == 14
assert confusion_10um[1] == 121

display(
    Markdown(
        f"**Historical check.** {historical_localized_10um}/175 localised · "
        f"{ten_micron['tp']}/{historical_localized_10um} correctly classified "
        f"after localisation ({100 * ten_micron['tp'] / historical_localized_10um:.1f}%) · "
        f"{confusion_10um[1]} labelled 4 µm."
    )
)

# %% [markdown]
# ### One geometry, two decisions
#
# **Intersection over union (IoU)** measures whether two intervals cover the
# same region:
#
# \[
# \operatorname{IoU}(G,P)=\frac{|G\cap P|}{|G\cup P|}.
# \]
#
# Here, \(G\) is the MAD interval and \(P\) the predicted interval. IoU ≥ 0.5
# means “localised” in this experiment. **Objectness** is the detector's score
# that an event exists at that location; it says nothing by itself about which
# size class is correct.
#
# The next case is not taken from the historical test. It is selected
# deterministically from the MAD v2.1 **validation** proposals: among 10 µm
# events that are alone on their trace and away from its edges, retain those
# whose best proposal has objectness ≥ 0.9 and is classified as 4 µm; choose
# the highest IoU, then break ties by stable IDs.

# %%
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
    f"A real validation error · {case.event_id}",
    x=0.01,
    ha="left",
    fontsize=13,
    fontweight="bold",
)
plt.show()

# %% [markdown]
# The proposal overlaps the teacher box strongly and its objectness is near
# one. The error is therefore not “the detector saw nothing”: the 4 µm class
# probability outranks the 10 µm probability on an already localised region.
#
# ### Why AP is not classification accuracy
#
# **Average precision (AP)** evaluates the whole ranked list of detections.
# After sorting predictions by score, a prediction is a true positive only if
# its class is correct *and* its IoU reaches the matching threshold. AP is high
# when correct detections appear early; a confident wrong-class box harms AP
# even when its geometry is excellent.

# %%
figure, axis = plt.subplots(figsize=(7.8, 2.8), constrained_layout=True)
plot_ap_ranking(ax=axis)
plt.show()

# %% [markdown]
# This separates the apparently contradictory observations:
#
# - **localisation:** does any proposal overlap the event?
# - **conditional classification:** is that localised proposal assigned the
#   right bead size?
# - **class-aware AP:** are correct class-and-box pairs ranked ahead of errors?
#
# At this point, four explanations were still plausible: information removed
# by preprocessing, the easier oracle-crop task, insufficient regional
# aggregation in the YOLO head, and imperfect pseudo-labels. The next section
# inspects the initial architecture before judging among them.
#
# > **Reading checkpoint.** A reader should now be able to explain how an event
# > can be localised with high IoU and objectness while still being a
# > wrong-class detection that lowers AP.

# %% [markdown]
# ## 2 · The initial joint Swin1D–YOLO detector
#
# The historical model solved all three tasks jointly:
#
# ```text
# 16,384-sample trace → Swin1D feature pyramid → deepest 512-cell map
#                      → 1×1 YOLO head → objectness + box + 3 classes
# ```
#
# The next cell reconstructs the architecture from the frozen training record.
# Temporary hooks measure the tensors seen during a real forward pass; the
# parameter count is then checked against the recorded run. The checkpoint
# itself remains remote, but its recorded size and hash identify the exact
# trained state whose results were shown in section 1.

# %%
historical_run = json.loads((historical_root / "run.json").read_text(encoding="utf-8"))
detector = rebuild_model(historical).cpu()
shape_contract = inspect_detector_shapes(detector, input_length=historical["input_length"])

assert shape_contract.total_parameters == historical["total_params"]
assert shape_contract.trainable_parameters == historical["trainable_params"]
assert shape_contract.output_shape[-1] == historical["input_length"] // shape_contract.strides[-1]
assert shape_contract.output_shape[1] == 1 + historical["num_classes"] + 2

backbone_text = " → ".join(
    f"{shape[-1]} cells × {shape[-2]} channels"
    for shape in shape_contract.backbone_shapes
)
checkpoint = historical_run["outputs"]["checkpoint"]
display(
    Markdown(
        "\n".join(
            [
                "| Measured quantity | Value |",
                "|---|---:|",
                f"| Input | `{shape_contract.input_shape}` |",
                f"| Swin1D pyramid | `{backbone_text}` |",
                f"| Strides | `{shape_contract.strides}` |",
                f"| Tensor entering the cell-wise 1×1 convolution | `{shape_contract.head_cell_input_shape}` |",
                f"| YOLO output | `{shape_contract.output_shape}` |",
                f"| Total / trainable parameters | `{shape_contract.total_parameters:,}` / `{shape_contract.trainable_parameters:,}` |",
                f"| Serialized checkpoint size recorded by the run | `{historical['size_mb']:.3f} MB` |",
                f"| Checkpoint SHA-256 | `{checkpoint['sha256']}` |",
                "| Raw / retained proposals per trace | `512` / at most `20` after NMS |",
            ]
        )
    )
)

# %% [markdown]
# The output has six channels at every deepest-grid location:
#
# \[
# [\text{objectness},\;p_{2},p_{4},p_{10},\;\Delta c,\;\log w].
# \]
#
# The three Swin maps exist, but this historical YOLO head consumes only the
# deepest one. Its `1×1` convolution independently maps each 256-dimensional
# cell vector to those six outputs. Thus the cell responsible for an event
# emits both its box and its class.

# %% [markdown]
# ### A long event and one responsible cell
#
# The toy event below is 4,000 samples long, matching the upper end of the MAD
# supports. At stride 32 it spans many deepest-grid cells, but supervision
# assigns the output to the cell containing its centre.

# %%
responsibility = grid_responsibility(6200, 10200, stride=shape_contract.strides[-1])
figure, axis = plt.subplots(figsize=(11.5, 3.2), constrained_layout=True)
plot_grid_responsibility(responsibility, ax=axis)
plt.show()

# %% [markdown]
# This does **not** mean the responsible feature sees only 32 raw samples.
# Swin attention and the hierarchical backbone give it a wider contextual
# receptive field. The narrower architectural claim is:
#
# > the class is read from one deepest feature vector; there is no explicit
# > pooling aligned with the full predicted interval.
#
# Box regression can therefore succeed whenever that vector contains enough
# information about centre and extent, while fine class evidence distributed
# across the waveform may remain difficult to extract. This is a motivated
# hypothesis, not yet a causal result: preprocessing, crop difficulty, and
# pseudo-label quality remain alternative explanations at this point in the
# story.
#
# > **Reading checkpoint.** A reader should now be able to trace the measured
# > tensor shapes, explain why there are 512 raw candidate locations, and state
# > precisely what regional aggregation the initial head does and does not
# > perform.

# %% [markdown]
# ## 3 · Did preprocessing or crop length explain the failure?
#
# Per-window z-scoring applies
#
# \[
# z(x)=\frac{x-\mu_x}{\sigma_x}.
# \]
#
# For a positive global scale factor $a$, $z(ax)=z(x)$. Absolute amplitude,
# root-mean-square level, and global energy therefore disappear. Timing,
# frequency, asymmetry, and the relative envelope can remain.

# %%
toy_time = np.linspace(-1.0, 1.0, 600)
toy_signal = np.exp(-3.0 * toy_time**2) * np.sin(2 * np.pi * 12 * toy_time)
figure, axes = plt.subplots(1, 2, figsize=(12, 3.0), constrained_layout=True)
plot_zscore_scale_invariance(toy_signal, axes=axes)
plt.show()

# %% [markdown]
# This made preprocessing a serious candidate: if bead size was encoded mainly
# by absolute signal scale, the detector never received that cue. The decisive
# comparison kept the same **2,481 isolated MAD v1 events**, Conv1DGAP-S model
# family, splits, and three seeds, and changed only the input representation.
# It used the already-consumed 475-event historical test, so it is a diagnostic
# comparison—not independent validation.

# %%
preprocessing_root = workspace.artifacts_root / "cross-project/particle-preprocessing-comparison-results-r1"
preprocessing_run = json.loads((preprocessing_root / "run.json").read_text(encoding="utf-8"))
preprocessing_summary = json.loads(
    (preprocessing_root / "summary.json").read_text(encoding="utf-8")
)
assert preprocessing_run["run_id"] == "particle-preprocessing-comparison-results-r1"
assert preprocessing_run["status"] == "complete"
assert preprocessing_summary["population"]["isolated_test_events"] == 475
representation_results = preprocessing_results(preprocessing_summary)

table = [
    "| Input representation | Macro-F1 | 10 µm recall | 10→4 µm | Gain [95% CI] |",
    "|---|---:|---:|---:|---:|",
]
for index, result in enumerate(representation_results):
    gain = "reference" if index == 0 else (
        f"{100 * result.macro_f1_delta:+.1f} "
        f"[{100 * result.ci95_low:+.1f}, {100 * result.ci95_high:+.1f}]"
    )
    table.append(
        f"| {result.label} | {result.macro_f1:.1%} | {result.recall_10um:.1%} | "
        f"{result.ten_to_four_rate:.1%} | {gain} |"
    )
display(Markdown("\n".join(table)))

# %%
figure, axis = plt.subplots(figsize=(9.5, 3.2), constrained_layout=True)
plot_preprocessing_deltas(representation_results, ax=axis)
plt.show()

# %% [markdown]
# Local z-scoring recovered much of the 10 µm signal: recall rose from **62.9%
# to 78.5%**, and `10 µm → 4 µm` fell from **17.3% to 9.9%**. The best overall
# arm—the shorter filtered P0 representation—gained **4.3 macro-F1 points**.
# Yet every arm remained below the pre-registered **+7 point** causal gate; the
# 4,096-sample arm's confidence interval also included zero.
#
# **Conclusion.** Preprocessing mattered, but it did not explain the historical
# gap by itself. The result supported preserving more local event information;
# it did not justify another preprocessing sweep or establish that amplitude
# loss was the principal cause. The next question is why an oracle crop is a
# fundamentally easier classification input than a full trace.
#
# > **Reading checkpoint.** A reader should now be able to say exactly what
# > z-scoring removes, quantify the improvement, and explain why a real gain can
# > still be a negative causal result.

# %% [markdown]
# ## 4 · Why crop classification is an easier task
#
# A joint detector must infer event presence, centre $c$, width $w$, and class
# $k$ from a complete trace $x$:
#
# \[
# P(\mathrm{event},c,w,k\mid x_{1:16384}).
# \]
#
# An oracle classifier is conditioned on information supplied by the label:
#
# \[
# P\!\left(k\mid \operatorname{crop}(x,c_{\mathrm{GT}},6144),\mathrm{event\ exists}\right).
# \]
#
# It does not search for the event, reject empty locations, or regress a box.

# %%
ceiling_root = workspace.artifacts_root / "cross-project/particle-classification-ceiling-method-analysis-r2"
ceiling_run = json.loads((ceiling_root / "run.json").read_text(encoding="utf-8"))
ceiling_summary = json.loads((ceiling_root / "summary.json").read_text(encoding="utf-8"))
assert ceiling_run["run_id"] == "particle-classification-ceiling-method-analysis-r2"
assert ceiling_run["status"] == "complete"
crop_facts = oracle_crop_facts(ceiling_summary)

figure, axis = plt.subplots(figsize=(11.5, 3.6), constrained_layout=True)
plot_conditioned_classification_task(crop_facts, ax=axis, detector_cells=shape_contract.output_shape[-1])
plt.show()

# %% [markdown]
# The geometry audit makes the advantage concrete on the same historical MAD
# v1 cohort used in section 3:
#
# | Centred crop | Complete target support visible |
# |---:|---:|
# | 2,500 samples | 54.5% |
# | 4,096 samples | 94.7% |
# | 6,144 samples | 100.0% |
#
# Thus the 6,144-sample experiment tested classification with the event already
# found and its entire annotated support visible. Its **84.0% macro-F1** cannot
# be compared directly with detector mAP: mAP additionally penalises missed
# events, background activations, box mismatch, duplicate proposals, and score
# ranking.
#
# “Oracle” does not mean “perfectly clean.” Although the 475-event cohort had no
# second MAD centre inside the crop, **81 crops (17.1%)** still intersected some
# other annotated support. The pseudo-label itself also remained a MAD teacher
# label. The experiment therefore removed localisation uncertainty; it did not
# prove physical class separability under ideal observation.
#
# This distinction motivated a separate ROI classifier: first let the detector
# propose a region of interest (ROI), then classify the crop. At inference that
# crop is proposal-centred—not GT-centred—so the next experiment still had to
# measure how much of the oracle advantage survived localisation error.
#
# > **Reading checkpoint.** A reader should now be able to list which unknowns
# > the oracle supplies, explain why macro-F1 and detector mAP are not directly
# > comparable, and state what proposal-centred ROI classification must recover.

# %% [markdown]
# ## 5 · Common proposals B1 and the first ROI classifier R1
#
# To test the class head without moving any box, the experiment created one
# **common proposal set**. The retrained B0 detector emitted 512 dense cells;
# class-agnostic non-maximum suppression (NMS) ranked them by objectness,
# removed boxes overlapping above IoU 0.5, and retained at most 20 per trace.
#
# - **B1** kept the detector's native class probabilities.
# - **R1** replaced only those probabilities with a Conv1DGAP-S prediction from
#   a 6,144-sample, proposal-centred, locally z-scored crop.
#
# Both used the same box and objectness, with class score
# $s_k=o\,p(k)$ for objectness $o$ and class $k$.

# %%
b1_root = (
    workspace.artifacts_root
    / "cross-project/remote-pfcalcul/notebook-cascade-section1-source-r1/extra"
    / "artifacts/cross-project/particle-mad-v21-common-proposals-b1-r1"
)
b1_run = json.loads((b1_root / "run.json").read_text(encoding="utf-8"))
roi_input_run = json.loads((b1_root / "roi_inputs/run.json").read_text(encoding="utf-8"))
threshold_selection = json.loads((b1_root / "threshold_selection.json").read_text(encoding="utf-8"))
assert b1_run["status"] == "complete"
assert b1_run["dataset"] == DATASET_KEY
assert roi_input_run["splits_loaded"] == ["train", "val"]
r1_crop_facts = roi_training_facts(roi_input_run)
validation_results = validation_arm_results(threshold_selection)

figure, axis = plt.subplots(figsize=(11.5, 4.0), constrained_layout=True)
plot_b1_r1_design(r1_crop_facts, ax=axis)
plt.show()

# %% [markdown]
# R1 training used the detector's distribution whenever possible: **2,856 of
# 2,921 crops** were centred on a proposal matched to the GT at IoU ≥ 0.5. The
# remaining **65** used a GT-centred fallback so every train/validation event
# contributed. All crops contained the complete support; none had zero
# variance. The fallback existed only for training—there is no GT centre at
# inference.
#
# The validation operating points were frozen before test access:

# %%
validation_table = [
    "| Arm | Event precision | Macro-F1 | 10 µm recall | 10→4 µm | Localised events |",
    "|---|---:|---:|---:|---:|---:|",
]
for result in validation_results:
    validation_table.append(
        f"| {result.arm} | {result.event_precision:.1%} | {result.macro_f1:.1%} | "
        f"{result.recall_10um:.1%} | {result.ten_to_four_rate:.1%} | "
        f"{result.localized_events} |"
    )
display(Markdown("\n".join(validation_table)))

# %% [markdown]
# On validation, replacing only the class probabilities raised event precision
# from **51.9% to 56.2%**, macro-F1 from **56.5% to 58.9%**, and 10 µm recall
# from **39.8% to 61.2%**. The localised `10 µm → 4 µm` confusion fell from
# **43.5% to 4.2%**. This was the first direct evidence that explicit regional
# aggregation recovered useful class information.
#
# It was not yet a complete detector solution. R1 was trained only on event
# crops, so it learned $P(k\mid\mathrm{event},\mathrm{ROI})$ but not whether a
# proposal was background. Its 10 µm precision at this validation threshold was
# only **35.3%**, and fewer events survived the higher operating threshold. The
# next section examines why a classifier that fixes class confusion can still
# leave the end-to-end cascade insufficient.
#
# > **Reading checkpoint.** A reader should now be able to explain why B1/R1 is
# > a causal class-head comparison, where the 65 GT fallbacks are allowed, and
# > which validation gains support ROI aggregation without proving a final
# > detector.

# %% [markdown]
# ## 6 · Why R1 remained insufficient
#
# The sealed B0/B1/R1 evaluation was opened once after thresholds and hashes
# were frozen. It is an internal replication on a historically consumed test,
# not independent confirmation. This section reads only its summary—no test
# prediction or waveform is reopened.

# %%
intermediate_root = workspace.artifacts_root / "cross-project/particle-mad-v21-b0-b1-r1-analysis-r1"
intermediate_run = json.loads((intermediate_root / "run.json").read_text(encoding="utf-8"))
intermediate_summary = json.loads(
    (intermediate_root / "metrics/summary.json").read_text(encoding="utf-8")
)
intermediate_decision = json.loads(
    (
        workspace.artifacts_root
        / "cross-project/reviews/particle-mad-v21-b0-b1-r1-result-r1/review/decisions.json"
    ).read_text(encoding="utf-8")
)
assert intermediate_run["status"] == "complete"
assert intermediate_summary["test_opened_once"] is True
assert intermediate_decision["complete"] is True
r1_result = r1_intermediate_facts(intermediate_summary)

figure, axis = plt.subplots(figsize=(10.5, 4.6), constrained_layout=True)
plot_r1_tradeoffs(r1_result, ax=axis)
plt.show()

# %% [markdown]
# The causal check passed exactly: B1 and R1 had the same **71.0% class-agnostic
# Event AP**, because they used identical boxes and objectness. R1 changed only
# the class ranking. It raised class-aware mAP from **42.4% to 46.6%** and AP
# 10 µm from **26.6% to 34.5%**, while reducing `10 µm → 4 µm` sharply.
#
# But the paired R1−B1 mAP gain was **+4.3 points with IC95
# [−0.3; +9.9]**: the interval included zero. At the selected operating point,
# proposal recall also fell from **83.8% to 75.8%**. Better conditional classes
# did not create better boxes or recover missed events.
#
# ### The background-rejection warning
#
# On 164 MAD-v2.1-empty traces, activation fell from **31.1% for B1 to 20.1%
# for R1** at each arm's own validation-selected threshold. However, R1 used a
# much higher threshold (0.522 versus 0.402). At the common B0 threshold, R1
# activated **37.8%** of empty traces versus **30.5%** for B1. The classifier
# had changed score calibration; it had not learned an explicit event-versus-
# background decision.
#
# The human checkpoint was therefore recorded as **`conditionally_supported`**:
# ROI aggregation was supported, but localisation and background rejection
# remained insufficient, and R1 was not retained as the final system. This
# diagnosis directly motivated two orthogonal changes: a class-agnostic
# localiser L1 and a proposal-aware classifier R2 with an explicit event head.
#
# > **Reading checkpoint.** A reader should now be able to separate geometry,
# > class ranking, score calibration, and background rejection—and explain why
# > correcting `10 µm → 4 µm` was necessary but not sufficient.
