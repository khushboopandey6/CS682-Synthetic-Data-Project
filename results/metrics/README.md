# Results guide

## Current Module A results: `master_v1/`

Use the files in [master_v1](master_v1/) for the adopted project protocol:
seeds **42, 123, 456**, fixed official splits and rare IDs, and the shared frozen
learning rate **1e-4**, maximum **30 epochs**, patience **5**.

| File | Purpose |
|---|---|
| [baseline_summary.md](master_v1/baseline_summary.md) | Start here: test/validation mean ± sample SD, rare-class scores and timing |
| [baseline_score_summary.csv](master_v1/baseline_score_summary.csv) | Twelve score groups: six test and six validation groups, three seeds each |
| [baseline_timing_summary.csv](master_v1/baseline_timing_summary.csv) | Six groups of completed-run time, epochs and hardware; B2 reuse adds zero cost |
| [baseline_summary_manifest.json](master_v1/baseline_summary_manifest.json) | Summary input hashes, seed list and SD convention |
| [baseline_test_results.csv](master_v1/baseline_test_results.csv) | Eighteen per-seed official test result entries |
| [baseline_validation_results.csv](master_v1/baseline_validation_results.csv) | Eighteen per-seed validation entries used to document the completed matrix |
| [baseline_validation_audit.json](master_v1/baseline_validation_audit.json) | Independent verification of 18 validation entries / 15 model fits |
| [baseline_test_audit.json](master_v1/baseline_test_audit.json) | Independent verification of all 3,080-row test exports, metrics, hashes and reuse |
| [b1_checkpoint_index.json](master_v1/b1_checkpoint_index.json) | Portable paths, sizes and SHA-256 hashes for all six B1 handoffs; weights are on Drive/local artifacts |
| [data_release_manifest.json](master_v1/data_release_manifest.json) | Official dataset IDs, counts and content/file hashes |
| [classifier_pilot_review.md](master_v1/classifier_pilot_review.md) | Original 12-fit max-30 pilot decision and curve review |
| [classifier_pilot_trials.csv](master_v1/classifier_pilot_trials.csv), [classifier_learning_rates.csv](master_v1/classifier_learning_rates.csv), [classifier_selection.json](master_v1/classifier_selection.json) | Original pilot scores and selected rate |

Main training is complete: **15 distinct models, 18 result entries**. All six B1
handoffs and all 18 test/validation exports are verified. Balanced B2 reuses B1's
model and predictions for each seed; its three entries are identities, not three
additional independent fits. Pilots are development evidence, separate from the
main result matrix.

Scores use the same fixed test set across seeds. SD uses the sample denominator
n−1 and describes variation across three training seeds; it is not a confidence
interval or evidence of statistical significance. The full summary discloses
mixed CPU/MPS training and completed-main-run cost boundaries. Test scores are
for final reporting and must never guide later experimental choices.

Detailed prediction/probability files, per-class metrics, checkpoints and logs
are under `CS682_Project/` on Drive, mirrored locally at
`artifacts/CS682_Project/`. Git contains these small reports and metadata; the
latest files still await publication. See [data contracts](../../docs/data_contracts.md)
for exact paths, recipients and transfer checks.

Regenerate small reports from the released artifacts without retraining:

```bash
python scripts/export_baseline_handoff.py
python scripts/summarize_baselines.py
```

## Earlier development results: files outside `master_v1/`

The older tables in this directory use seeds **42, 43, 44**, a different
rare-class selection policy and a B1-only learning-rate search. Their reported
scores are validation scores. They are retained for traceability and must not
be combined with, or relabelled as, the adopted protocol's results.

| Historical file or group | Contents |
|---|---|
| [baseline_validation_aggregate.csv](baseline_validation_aggregate.csv) | Five historical setting/baseline means and sample SDs |
| [baseline_validation_results.csv](baseline_validation_results.csv) | Fifteen historical distinct fits |
| `b1_validation_*`, `b2_validation_*`, `b3_validation_*` | Historical per-method tables |
| [b1_b2_validation_comparison.csv](b1_b2_validation_comparison.csv), [b1_b3_validation_comparison.csv](b1_b3_validation_comparison.csv) | Historical paired comparisons |
| [baseline_validation_trials.csv](baseline_validation_trials.csv), [baseline_learning_rates.csv](baseline_learning_rates.csv) | Earlier six-trial B1 tuning search |
| [checkpoint_index.csv](checkpoint_index.csv) | Historical checkpoint index, not the current six-B1 handoff inventory |

Historical checkpoints remain under `results/baselines/distilbert_validated/`,
with tuning artifacts stored separately. `scripts/update_checkpoint_index.py`
refreshes that historical index only. Current Module A handoffs use
`scripts/export_baseline_handoff.py`. Existing local artifacts are retained.
