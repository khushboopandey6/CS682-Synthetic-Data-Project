# Shared data contracts

The [master project document](CS682-Master-Project-Document.md) defines the team
schemas. This guide describes the released Module A artifacts used by the code.

## Official data release

Artifact base is configurable through `--base-path`, `CS682_BASE_PATH`, or the
configuration's `paths.base_path`. The local default mirrors the shared Drive
layout at `artifacts/CS682_Project/` and is excluded from Git.

`01_data/` contains:

- `low_resource/seed_42.jsonl`, `seed_123.jsonl`, `seed_456.jsonl`: 770 rows each.
- `imbalanced/seed_42.jsonl`, `seed_123.jsonl`, `seed_456.jsonl`: 1,315 rows each.
- `validation.jsonl`: the fixed 770 real validation examples.
- `test.jsonl`: the unchanged official 3,080-example reporting split.
- `label_mapping.json`, `rare_classes.json`, `dataset_statistics.json`, and `manifest.json`.
- `baselines/B2/<setting>/seed_<seed>.jsonl` and corresponding B3 datasets/reports.

The manifest records exact ordered IDs and file/content hashes. Consumers should
use the released artifacts rather than independently resampling data.
Real records use the official integer label and stable `train:<row>`/`test:<row>` IDs.
Added baseline records have unique IDs and preserve the original ID in `parent_id`.

Rare classes are selected once using
`random.Random(42).sample(sorted(range(77)), 15)`. Their fixed IDs are:

```text
14, 3, 35, 31, 28, 17, 13, 69, 11, 54, 4, 75, 68, 27, 76
```

The validation set and low-resource seed-42 subset reproduce the earlier development
records exactly. The other training subsets and the rare-class policy belong to
the newly adopted protocol.

| Condition | Low-resource rows | Imbalanced rows |
|---|---:|---:|
| B1 | 770 | 1,315 |
| B2 | 770; exact B1 identity | 1,540; 225 replacement draws |
| B3 | 1,540 | 2,630 |

All six B3 datasets achieved one augmentation per original with zero shortfalls.
The saved reports record WordNet/NLTK versions, counts, retries, and provenance.

## Dataset description for the team poster

We study 77-way banking-intent classification under two sampled training
settings. The low-resource setting has 10 real examples per intent (770 total).
The imbalanced setting fixes 15 rare intents at five examples each and 62 other
intents at 20 each (1,315 total). Training samples use seeds 42, 123 and 456,
with the same rare-intent IDs across runs. A separate, fixed validation set has
10 additional real examples per intent (770 total); report this labeled
supervision explicitly. Final reporting uses the unchanged official test split
of 3,080 examples. B1 uses real data, B2 oversamples minority classes, and B3
adds one EDA example per real example. The 12-fit classifier pilot selected one
shared learning rate of 1e-4 under a 30-epoch cap and patience of five.

| Split or setting | Rare intents | Other intents | Total real examples |
|---|---:|---:|---:|
| Low-resource train, per seed | 15 × 10 | 62 × 10 | 770 |
| Imbalanced train, per seed | 15 × 5 | 62 × 20 | 1,315 |
| Fixed validation | 15 × 10 | 62 × 10 | 770 |
| Official test | 15 × 40 | 62 × 40 | 3,080 |

For a limitations note, retain the source benchmark's seven normalized
training/test text overlaps and the recorded generated-only validation
coincidence described in the experiment protocol. Do not remove rows using test
references or describe the sampled training settings as the original paper's
published evaluation protocol.

## Record formats

Real examples are JSONL records containing `id`, `text`, and the official integer
`label`. IDs retain the original official source row, such as `train:28`.
Augmented and repeated records have unique IDs and preserve `parent_id`.

Qwen records additionally require `source`, `setting`, and `seed`; the intended
label remains unchanged even if the generated question is incorrect. Filtering
must preserve synthetic IDs and record both accepted and rejected decisions.
Generation and filtering outputs are produced by the other modules and are not
part of this Module A release.

Validation prediction records contain `id`, `true_label`, `predicted_label`,
`confidence`, `condition`, `setting`, `seed`, and `split: validation`. Full
probabilities retain the same row order and IDs in the accompanying NPZ file.

Reporting-only test predictions use the same fields with `split: test` and are
exported to `05_results/predictions/<condition>/<setting>/seed_<seed>.jsonl`.
Their probability arrays, per-class metrics, result JSON, and inference logs use
`05_results/probabilities/test/`, `05_results/per_class_metrics/test/`,
`05_results/metrics/test/`, and `06_cost_logs/inference/`, respectively. The public
summary exposes `rare_class_f1` and `rare_class_recall` alongside overall
macro-F1 and accuracy. All 18 test exports are complete and verified; each
contains the unchanged 3,080 test IDs. Balanced B2 reuses B1 predictions.

## Git and Drive responsibilities

**Git stores the implementation and small review artifacts.** Include source,
scripts, notebooks, requirements, `configs/`, these documents and
`results/metrics/master_v1/`. The reports include per-seed/aggregate scores,
audits, release metadata and the B1 checkpoint index. The index contains paths
and hashes, not model weights. Latest changes still await commit/push/PR.

**Drive stores the released datasets and detailed experiment artifacts.** Upload
from the local `artifacts/CS682_Project/` folder into the team's shared
`CS682_Project/` folder, preserving the relative paths below. `--base-path`,
`CS682_BASE_PATH`, or `paths.base_path` can point code at that restored root.
The shared Drive location and recipient access have not yet been verified.

| Relative path under `CS682_Project/` | Contents / transfer purpose |
|---|---|
| `01_data/` | All real splits, B2/B3 JSONL, preparation reports, label mapping, rare IDs, statistics and manifest |
| `04_checkpoints/B1/<setting>/seed_<seed>/` | Six complete teacher run folders; required for Module C |
| `04_checkpoints/B2/imbalanced/seed_<seed>/` | Three oversampling model/run folders retained for reproducibility |
| `04_checkpoints/B3/<setting>/seed_<seed>/` | Six EDA model/run folders retained for reproducibility |
| `05_results/validation_predictions/` | Condition-labeled validation predictions for all 18 entries |
| `05_results/predictions/` | Condition-labeled official test predictions for all 18 entries |
| `05_results/probabilities/test/` | Full test probability arrays, labels and ordered IDs |
| `05_results/per_class_metrics/test/` | Per-intent test precision, recall, F1 and support |
| `05_results/metrics/` | Per-run validation/test JSON, audits, summary tables and manifests |
| `05_results/inference_runs/` | Test evaluation specifications and checkpoint/data hash bindings |
| `06_cost_logs/training/` | Epoch/stage training logs for 15 distinct fits |
| `06_cost_logs/inference/` | Test inference logs; balanced B2 records reuse and zero added cost |
| `handoff/frozen_experiment.yaml` | Exact frozen configuration copy used by the main runner |
| `handoff/master_v1_b1_checkpoints.json` | Portable inventory for all six verified B1 runs |

Each complete B1 run folder contains `best_model/` with model weights,
configuration and tokenizer, plus `run_config.json`, `results.json`,
`runtime.json`, `history.csv`, `stage_log.csv`, `per_class_metrics.csv`,
`validation_predictions.jsonl` and `validation_probabilities.npz`. Transfer the
whole run folder so all inventory paths resolve. The validation probability NPZ
files needed for the B1 teacher handoff are inside these run folders; they are
separate from the test probability directory.

The pilot folder (`classifier_pilot/`) is separate from main models. Their small reports belong in Git; their full
weights/histories may be retained as Drive archives. Interrupted runs and earlier
development artifacts remain separate from the final handoff. Existing files
are retained locally; this document does not request their deletion.

## What each teammate receives

| Recipient | Required inputs from Module A |
|---|---|
| Danush — Module B | Exact real subsets for all six setting/seed pairs, label mapping and generation/data contract |
| Khushboo — Module C | Exact real subsets and fixed validation, label/rare metadata, all six B1 model folders and matching validation predictions/probabilities |
| Gnaneswarudu — Module D | Shared trainer candidate, frozen config, official real splits and metadata; review the trainer and use it for F0/F2 |
| Rufina — Module E | Per-seed test/validation metrics and predictions, probability/per-class exports, three-seed summaries and training/inference logs |

B1 is the F2 filtering teacher. Downstream F2 starts from the pinned pretrained
DistilBERT revision, independently of the trained B1 teacher. Each member should
use the same agreed configuration revision, label mapping and released subset
IDs. Team review of the shared trainer remains pending.

Rufina coordinates final cross-method evaluation and full pipeline cost analysis.
The Module A summaries cover completed baseline fits and test inference only;
pilot, interrupted, generation, EDA preparation and filtering costs are separate.
All test scores are for reporting, never for selecting later settings/thresholds.

## Receiver checklist and verification

1. Obtain the reviewed Git revision and the actual shared Drive location. Confirm
   that each recipient can read the required artifacts.
2. Copy the artifact folders while preserving the relative paths above. Keep
   model weights out of Git; no ZIP is required.
3. Verify the official data release with the repository tool:

   ```bash
   python scripts/package_data_release.py --verify-only --base-path /path/to/CS682_Project
   ```

4. Verify all recorded B1 file sizes and SHA-256 hashes at the received root:

   ```python
   import hashlib
   import json
   from pathlib import Path

   base = Path("/path/to/CS682_Project")
   index = json.loads((base / "handoff/master_v1_b1_checkpoints.json").read_text())
   assert index["status"] == "complete" and index["completed_runs"] == 6
   records = [index["data_manifest"], index["validation_data"],
              *index["metadata_files"], *index["training_data"]]
   for checkpoint in index["checkpoints"]:
       records.extend(checkpoint["files"])
   for record in records:
       path = base / record["path"]
       assert path.stat().st_size == record["bytes"], path
       with path.open("rb") as handle:
           assert hashlib.file_digest(handle, "sha256").hexdigest() == record["sha256"], path
   print("Received B1 files and shared inputs verified")
   ```

   This inventory checks B1 and its shared inputs; it is not an inventory of
   every other model. Test result JSON files separately record hashes for their
   prediction, probability, per-class and inference-log artifacts.

5. Confirm the Git config agrees with `handoff/frozen_experiment.yaml` and the
   configuration hashes in run/audit metadata. Use the existing released data;
   do not independently resample or regenerate labels.
6. Confirm the receiver can load the B1 model/tokenizer and match its validation
   IDs before final F2 filtering. Record the shared path and successful checks
   with the team; upload/access verification is still pending locally.

The current B1 inventory is complete: **6/6 verified**. Refresh it without
training using `python scripts/export_baseline_handoff.py --base-path ...`.
Its Git review copy is `results/metrics/master_v1/b1_checkpoint_index.json`.

Local data/code ZIPs were convenience snapshots created before the Drive handoff.
They are optional and may describe an earlier preparation state. The authoritative
handoff is the reviewed Git revision plus the verified artifact folders on Drive.
There is no ZIP-refresh requirement in the current handoff workflow.
