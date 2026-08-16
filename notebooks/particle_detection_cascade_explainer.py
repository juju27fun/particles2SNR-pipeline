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
    plot_ap_ranking,
    plot_grid_responsibility,
    plot_localized_misclassification,
    select_localized_misclassification,
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
