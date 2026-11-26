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

## Record formats

Real examples are JSONL records containing `id`, `text`, and the official integer
`label`. IDs retain the original official source row, such as `train:28`.
Augmented and repeated records have unique IDs and preserve `parent_id`.

Qwen records additionally require `source`, `setting`, and `seed`; the intended
label remains unchanged even if the generated question is incorrect. Filtering
must preserve synthetic IDs and record both accepted and rejected decisions.
Generation and filtering outputs have not been produced yet.

Validation prediction records contain `id`, `true_label`, `predicted_label`,
`confidence`, `condition`, `setting`, `seed`, and `split: validation`. Full
probabilities retain the same row order and IDs in the accompanying NPZ file.

## Handoffs to other members

- **Danush:** consume the exact real JSONL subsets and label mapping for prompts;
  release each raw pool and generation log progressively.
- **Khushboo:** consume the fixed subsets, raw pool, corresponding B1 model, and
  validation predictions. Final F2 needs the official models after the freeze;
  development can use mock examples before that handoff.
- **Gnaneswarudu:** review/use the shared trainer and data contract, then execute
  F0/F2 with the same frozen configuration. Downstream F2 starts from the pinned
  pretrained initialization, independently of the B1 filtering teacher.
- **Rufina:** use the exported validation metrics/predictions and per-run logs.
  Final test exports wait until method/threshold decisions are frozen.

The data ZIP contains `01_data/` and its reports, not model weights. Later B1
handoffs must copy each entire `best_model/` folder, including tokenizer/config,
and the matching validation predictions and probabilities. Git stores metadata
and small summaries; the shared artifact folder stores weights and datasets.

## Transfer and verification

The data-only ZIP and file inventory are under `handoff/` at the artifact base.
Verify the inventory/file hashes before using a transferred release. Do not
independently regenerate the official splits for another module.

The user approved local execution first, with team sharing later. The shared
Drive folder remains a future transfer destination; it is not required for the
currently authorized local pilot.
