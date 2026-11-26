# Shared experiment protocol

The [master plan](CS682-Master-Project-Document.md) defines the project design and
roles. [configs/experiment.yaml](../configs/experiment.yaml) is the shared frozen
configuration. [Data contracts](data_contracts.md) define the released inputs and
artifact paths. This document records the implementation and completed Module A work.

## Completed work and remaining delivery

Official data preparation, the 12-fit learning-rate pilot, all 15 main baseline
fits, 18 validation/test result entries and all six B1 checkpoint handoffs are
complete and verified locally. [Results and timing](../results/metrics/master_v1/baseline_summary.md)
are summarized across the three seeds. No additional baseline training is required.

Remaining delivery work is Git publication/PR, Drive transfer with receiver hash
checks, and Module D/team review of the common trainer and frozen configuration.
The original master document is preserved; this guide records implementation
status rather than rewriting the team's assignments.

## Frozen main training policy

| Component | Retained setting |
|---|---|
| Dataset | PolyAI/banking77, pinned revision in shared YAML |
| Experiment seeds | 42, 123, 456 |
| Validation | 770 fixed real queries; split seed 42 |
| Rare intents | 15 fixed IDs selected with seed 42 |
| Classifier | distilbert/distilbert-base-uncased, 77-class head |
| Pretrained revision | 12040accade4e8a0f71eabdb258fecc2e7e948be |
| Trainable weights | Full encoder and classification head |
| Optimizer / learning rate | AdamW / 1e-4 |
| Maximum epochs / patience | 30 / 5 |
| Training / evaluation batch size | 16 / 32 |
| Input length / precision | 128 tokens / float32 |
| Weight decay / gradient clipping | 0.01 / 1.0 |
| Scheduler | Linear decay; 10% warmup based on the planned epoch cap |
| Validation frequency | Every epoch |
| Checkpoint selection | Highest validation macro-F1; earliest checkpoint on exact ties |
| Final predictions | Reload the selected best checkpoint |

Parameter matrices receive weight decay; vector parameters do not. Every fit
starts from the same pinned pretrained revision with a freshly seeded head.
Prepared B3 datasets are reused rather than regenerated between learning rates.
The fixed validation split supplies 770 additional labeled examples; report that
supervision separately from the 770 training labels in the low-resource setting.

The frozen YAML's SHA-256 is
`e5e69a67dd9f5d151b01aa4a569f90bd161fbec8908c0a318916539bf1f00ede`.
Its matching copy is stored at `handoff/frozen_experiment.yaml` under the artifact
root. The latest frozen configuration and reports still await Git publication;
per-run configuration hashes and the committed trainer source identify what ran.

## Learning-rate selection

The pilot used seed 42, rates 2e-5/5e-5/1e-4, B1/B3 and both settings:
**12 fits**, separate from the core result matrix. The rate was selected using
highest equal-weight mean of four best-checkpoint validation macro-F1 scores.
An exact mean tie selects the lower rate. Test data was excluded from selection.

| Peak learning rate | Mean validation macro-F1 |
|---|---:|
| 2e-5 | 0.773046 |
| 5e-5 | 0.790588 |
| 1e-4 | 0.798107 |

The pilot selected **1e-4** under the shared maximum of 30 epochs and patience
5. It used MPS. This is a one-development-seed selection, not proof of a
universal optimum. The 2e-5 imbalanced B1 fit was still improving at epoch 30;
that limitation is recorded in the [pilot review](../results/metrics/master_v1/classifier_pilot_review.md).

Pilot artifacts use `04_checkpoints/classifier_pilot/` and
`05_results/classifier_pilot/`. Small review reports are under
`results/metrics/master_v1/`. Pilot costs are separate from completed main costs.

## Main matrix and run ownership

| Condition | Low-resource | Imbalanced | Owner |
|---|---|---|---|
| B1 | Real training data | Real training data | Manikanta |
| B2 | Exact B1 reuse | Random oversampling to 20 rows per class | Manikanta |
| B3 | Real + one EDA example per original | Real + one EDA example per original | Manikanta |
| F0 | Matched unfiltered synthetic data | Same condition for imbalanced data | Gnaneswarudu |
| F1 | Matched duplicate-filtered synthetic data | Same condition for imbalanced data | Khushboo |
| F2 | Matched classifier-filtered synthetic data | Same condition for imbalanced data | Gnaneswarudu |

Each condition has two settings × three seeds. Module A completed **18 entries
from 15 independent fits**: three balanced B2 entries reuse B1 with matching
input bytes and configuration. B2 keeps its own condition label in exported
predictions, records its B1 source and adds zero incremental training/inference cost.

The whole project calls for 36 entries and normally 33 independent main fits;
F0/F1/F2 work is outside these completed baseline totals. Danush owns generation;
Khushboo also owns filtering/matching; Rufina owns combined evaluation and cost analysis.

Module D owns the common trainer. The tested existing baseline implementation
was extracted into `src/training.py` for shared use and Module D review. It was
not supplied as an independently completed Module D deliverable. All downstream
methods should use the agreed trainer/configuration revision. B1 supplies the
F2 filtering teacher; downstream F2 starts fresh from the pinned pretrained model.

## Hardware and interrupted runs

The general trainer supports CUDA → MPS → CPU. The user stopped further CPU
model execution; the local resume wrapper requires MPS. Two main fits—B1 and
B3 for low-resource seed 42—were completed on CPU and retained. The other 13
main fits used MPS. All 15 unique test evaluations used MPS. Each run records
its actual device, precision, package versions, source/configuration hashes,
actual epochs, optimizer steps, and timing.

The interrupted CPU main seed-123 B1 attempt was preserved separately and
restarted on MPS. The final results use completed fits only; partial attempts
are not additional completed experiments. Optimizer-state or mid-epoch
resume is not implemented: completed matching fits are reused, and unfinished
fits restart from their seed. An actual Colab CUDA execution check remains unverified.

## Evaluation, summaries and integrity

Main configuration/checkpoints and all 18 validation entries were verified
before test access. Test evaluation exported 18 prediction sets of 3,080 rows,
full probability arrays, per-class metrics and inference logs. Independent
checks recomputed macro-F1, accuracy and rare-class metrics, verified ID/order,
confidence and array consistency, and confirmed unchanged model/configuration
hashes. Test scores are for reporting only and must never choose later training,
generation, duplicate references or filtering thresholds. Those choices use validation.

[Score summaries](../results/metrics/master_v1/baseline_score_summary.csv) use
arithmetic means and sample standard deviations (ddof=1) over seeds 42/123/456.
They are not confidence intervals. [Timing summaries](../results/metrics/master_v1/baseline_timing_summary.csv)
count the 15 completed main fits once: 103.07 minutes of fit elapsed time and
74.79 seconds of test inference. Pilots, interrupted attempts, EDA preparation,
generation and filtering need separate accounting. Wall time and synchronized
accelerator intervals are not hardware-utilization measurements. Mixed training
backends and sampled training data contribute to seed-to-seed variation.

The source has seven normalized train/test text overlaps; the official split is
retained and this is disclosed. Validation/test text overlap is zero. One
B3 imbalanced seed-123 generated query coincides with a validation query; it was
recorded without filtering augmentation against validation references. Real
training and validation rows/IDs/provenance remain separate.

## Verification commands

Run from the repository root in the project Python environment:

```bash
python scripts/package_data_release.py --verify-only
python scripts/export_baseline_handoff.py
python scripts/summarize_baselines.py
python -m unittest discover -s tests -v
```

The [data contracts](data_contracts.md) explain artifact transfer and receiver
checks. Model weights and detailed outputs belong on Drive; small reports,
configuration and audit/index metadata belong in Git. ZIP archives are optional.
