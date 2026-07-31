# Particle2SNR Dual-Clean Wave8-like Detection Datasets

**Registered capability view:**
`particles2snr-wave8like-known3-positive@v1`  
**Registered deployment view:**
`particles2snr-wave8like-fourclass-background@v2`  
**Parent:** `particles2snr-f-dual-clean-c1-yolo-4class@v1`  
**Owner:** `particles2SNR-pipeline`

These two derived datasets answer different questions and must not be merged in
a single leaderboard row. The capability view reproduces the event-rich
exposure mechanism of the historical Wave 8 study. The deployment view adds
explicit empty-background composites and retains the `unclear` label so that a
validation-calibrated false-positive operating point can be tested.

## Common signal construction

Each row concatenates four 16,384-sample parent traces into one 65,536-sample
trace at 2 MHz. Generation applies:

1. 300 samples of recorded-noise replacement at both edges of every source
   segment;
2. removal of every annotation touching a replaced edge;
3. 300-sample smoothing at each artificial join, entirely inside the
   annotation-safe guard;
4. a fourth-order 8--500 kHz Butterworth bandpass after concatenation;
5. deterministic permutation of the four source segments.

Class membership is read from YOLO labels, never inferred from filenames. A
base group contains no repeated source. Parent train, validation, and test
identities are preserved, and every generated manifest records source IDs and
signal SHA-256 values. No source ID or signal content hash crosses a split.

## Capability view

`particles2snr-wave8like-known3-positive@v1` contains only sources with at
least one edge-safe physical-class event. `unclear` annotations are omitted
from this optimization/evaluation view but remain traceable through the parent
source IDs. Each base group is expanded under all 24 permutations.

| Split | Base groups | Rows | 2 um labels | 4 um labels | 10 um labels |
|---|---:|---:|---:|---:|---:|
| Train | 100 | 2,400 | 3,192 | 3,216 | 3,192 |
| Validation | 30 | 720 | 960 | 960 | 960 |
| Test | 30 | 720 | 960 | 960 | 960 |

This view is appropriate for a controlled comparison with the historical
three-class, positive-only Wave 8 experiment. It cannot estimate background
false-positive rate or deployment precision.

Internal generation-manifest SHA-256:
`eb97edd5c45aa673902ca610367c0986f9a3a62de74cedf83abd79fc95fba9cd`.

The adjudicated reference candidate
`particles2snr-wave8like-known3-positive-adjudicated-candidate@v1` uses
`source_eligibility_policy=fully_labeled_for_view`. Unlike the legacy
any-known-event rule, it rejects a parent source if this three-class view would
omit any annotation or if any annotation touches the edge-replacement guard.
Consequently, its generation audit reports zero dropped edge events in every
split. It is not a replacement benchmark until fresh inference has been run.

## Deployment view

`particles2snr-wave8like-fourclass-background@v2` retains all four class IDs.
Positive groups contain one label-safe source for each current class deficit;
background groups contain four parent rows with genuinely empty YOLO labels.
Positive groups use all 24 permutations. Background groups use four
deterministic permutations, avoiding the false effective-sample inflation that
would result from repeating 24 label-identical empty compositions.
Background sources are sampled without replacement across base groups within
each split: the 200/60/60 train/validation/test background groups use exactly
800/240/240 distinct parent traces. The four permutations within a group are
still dependent and are summarized together.

| Split | Positive rows | Background rows | Total rows | Labels per class |
|---|---:|---:|---:|---:|
| Train | 2,400 | 800 | 3,200 | 2,400 |
| Validation | 720 | 240 | 960 | 720 |
| Test | 720 | 240 | 960 | 720 |

The four label classes are `2um`, `4um`, `10um`, and `unclear`. The 25%
background share is a prespecified engineering mixture, not an estimate of
field prevalence. Calibration uses only the 240 validation backgrounds; the
240 test backgrounds are evaluated after the threshold is frozen.

Internal generation-manifest SHA-256:
`b61f6645f75c5084280928cd6accd4f0ac7befa8846e387fe6af2525bade3eab`.

Deployment v1 is retained with legacy status for provenance. It had the same
positive composites and row counts, but reused background source traces across
nominal groups; it is excluded from model evidence and uncertainty estimates.

If and only if the frozen validation gate finds the 25% training mixture
uncalibratable, the allowed iteration changes training exposure without
changing evaluation. The same 200 source-disjoint train background groups are
expanded from 4 to 12 permutations, giving 2,400 positive and 2,400 background
train rows. Validation and test remain byte-identical to v2 at 720 positive +
240 background rows. The generator options are
`--background-share 0.50 --train-background-permutations 12
--evaluation-background-share 0.25`; no such dataset is registered unless the
validation-only trigger fires.

## Effective sample size and uncertainty

The 24 permutations of a positive base group reuse the same four parent
sources. They improve optimization exposure but are not 24 independent
experiments. Statistical uncertainty must therefore resample base groups, not
individual long rows. Positive source traces are also reused across nominal
base groups and connect all 30 test groups, so even base-group resampling is a
descriptive sensitivity analysis rather than an independent confidence
interval. The frozen comparison combines:

- variation across training seeds 7, 42, and 123;
- descriptive resampling over the 30 positive test base groups;
- a row-level Wilson interval over the 240 held-out background rows;
- a conservative group FPR and Wilson interval over 60 disjoint-source base
  groups, where a group fails if any of its four permutations fires.

## Integrity and reproduction

Both registered versions pass:

- the generator's complete manifest/label/source-identity audit;
- the P1 long-sequence preflight in strict mode over every row;
- full workspace-registry checksum validation;
- deterministic regeneration covered by synthetic tests. Any independently
  generated full local or pfcalcul copy must match the registered internal
  manifest hash before its checkpoints can be evaluated against the registered
  evidence dataset.

Generation logic:
`particles2snr/wave8like_dataset.py`  
CLI:
`scripts/generation/build_wave8like_detection_dataset.py`  
The public CLI option `--source-eligibility-policy` accepts
`fully_labeled_for_view` (default) and `legacy_any_known_event` (reproduction
only).
Frozen scientific protocol:
`../docs/experiments/2026-07-16/particles2snr-wave8like-comparison-protocol.md`

## Limitations

- Both views derive from the same acquisition-internal parent split and do not
  establish new-acquisition or instrument OOD performance.
- Post-concatenation filtering and joins reproduce a historical experimental
  contract; they do not make the long traces physically acquired sequences.
- Capability mAP must not be described as background-calibrated performance.
- Deployment precision depends on the stated 25% evaluation mixture and does
  not transfer directly to another event prevalence.
- A background row is label-empty, not independently certified particle-free.
  Residual missed annotations can turn a physically plausible detection into
  an annotation-relative false positive.
- Better optimization exposure does not create new biological or physical
  observations; source-group clustering remains mandatory.
