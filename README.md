# Does Filtering Help?

Cost-Aware LLM-Generated Synthetic Data for Low-Resource and Imbalanced Text Classification

COMPSCI 682 — University of Massachusetts Amherst

We study whether filtering Qwen-generated banking questions improves DistilBERT
classification, particularly for rare classes, and whether the gains justify the
computational cost.

## Adopted project protocol

The current team plan is [the master project document](docs/CS682-Master-Project-Document.md).
All members use [configs/experiment.yaml](configs/experiment.yaml) and the released
split artifacts. See [data_contracts.md](docs/data_contracts.md) for artifacts and interfaces,
[experiment_protocol.md](docs/experiment_protocol.md) for execution rules, and
[literature_review.md](docs/literature_review.md) for the review work outline.

The shared trainer is [src/training.py](src/training.py): one `train_classifier`
function for B1/B2/B3/F0/F1/F2. Each fit starts from the pinned pretrained model;
the condition selects its training dataset. Module D owns the common framework.
The existing baseline trainer is provided here for Module D review and the
classifier pilot, rather than waiting for a separate implementation.

- Official experiment seeds: **42, 123, 456**.
- Fixed validation set: **770** examples; official test: **3,080** examples.
- Low-resource training: **770** examples, 10 per class.
- Imbalanced training: **1,315** examples, with 15 fixed classes at 5 examples and
  the other 62 classes at 20.
- Classifier: full DistilBERT fine-tuning, float32, automatic **CUDA → MPS → CPU**.
- Classifier pilot: **12 fits**, covering three rates, B1/B3, and both settings.
- Learning rate stays unset until that pilot selects one shared value. Main runs
  are blocked until the reviewed configuration is frozen.

## Current status

- [x] Adopted plan and shared configuration recorded.
- [x] Official JSONL data release prepared and verified locally.
- [x] All official B2/B3 datasets prepared for the three seeds; EDA shortfalls are zero.
- [x] Existing training logic extracted into reusable modules for shared-trainer review.
- [x] Data/baseline checks and a small real CPU training/checkpoint roundtrip passed.
- [x] Data-only transfer ZIP and integrity inventory prepared locally.
- [ ] Share the data release and verify the team's shared storage when transferring artifacts.
- [ ] Run the official 12-fit classifier pilot and freeze the selected rate.
- [ ] Run the official baseline matrix and publish six B1 checkpoint handoffs.
- [ ] Complete Qwen generation, filtering, downstream training, and final analysis with the team.

The earlier 15 completed baseline fits used a different development protocol.
They are preserved, but are **not automatically the official results** for this
new protocol.

## Run the shared workflow

Install the tested dependencies into the environment selected for this project:

```bash
python -m pip install -r requirements.txt
```

Use these active notebooks in order:

1. [01_reproducible_data_preparation.ipynb](notebooks/01_reproducible_data_preparation.ipynb)
   prepares the pinned source, official splits, B2/B3 datasets, and transfer ZIP.
2. [02_baseline_models.ipynb](notebooks/02_baseline_models.ipynb) previews or runs
   the classifier pilot, freezes its winner, and executes official baselines.

Both notebooks support local VS Code execution and Colab setup. On a fresh Colab
runtime, check the shared Drive path before running. The pinned local versions
are tested; CUDA execution must be verified on the actual Colab GPU runtime.

For an existing local source cache:

```bash
python scripts/prepare_official_data.py
python scripts/prepare_baselines.py
python scripts/package_data_release.py
python scripts/run_classifier_pilot.py
```

The last command previews the 12 fits without training. The user approved local
execution first and team sharing later. After checking the configured output
folder, execute the pilot:

```bash
python scripts/run_classifier_pilot.py --run --base-path /path/to/CS682_Project
```

Review all 12 validation curves, especially fits still improving at the epoch cap.
Then freeze the selected rate from the verified completed fits:

```bash
python scripts/run_classifier_pilot.py --freeze --base-path /path/to/CS682_Project
```

Start the first official baseline handoff after the freeze:

```bash
python scripts/run_baselines.py --run --settings low_resource --seeds 42 --base-path /path/to/CS682_Project
```

Omit `--settings` and `--seeds` to run the full baseline matrix. It contains 18
logical entries and 15 distinct fits when all three balanced B2 conditions reuse
B1. Main predictions in this workflow are validation predictions. Official test
evaluation is reserved for the later reporting phase.

## Repository and artifact layout

```text
configs/                         Shared protocol, labels, and rare-class IDs
  experiment.yaml
  label_mapping.json
  rare_classes.json
docs/
  CS682-Master-Project-Document.md
  data_contracts.md
  experiment_protocol.md
  literature_review.md
  drafts/                        Parked older planning drafts; local only
notebooks/
  01_data_preparation.ipynb       Preserved original notebook
  01_reproducible_data_preparation.ipynb
  02_baseline_models.ipynb
scripts/                         Data, packaging, pilot, and baseline entry points
src/                             Shared implementation modules
tests/                           Focused integrity and training checks
results/metrics/
  master_v1/                     Official release manifest and future pilot/core summaries
  ...                            Earlier development score tables and checkpoint index
artifacts/CS682_Project/          Local mirror of the shared Drive layout; ignored by Git
  01_data/
  04_checkpoints/
  05_results/
  06_cost_logs/
  handoff/
```

`--base-path` or `CS682_BASE_PATH` can point to the agreed shared Drive folder.
The default local mirror is useful for preparation and verification; it does not
establish shared-folder access. Large datasets, weights, and detailed logs stay
outside Git. The [data contracts](docs/data_contracts.md) describe the release ZIP
and model-transfer requirements.

## Earlier development results

These validation results used seeds **42, 43, 44**, the earlier rare-class policy,
and a B1-only learning-rate search. They remain available as development evidence.

| Setting | Method | Validation macro-F1, mean ± sample SD |
|---|---|---:|
| Low-resource | B1 real only | 0.7809 ± 0.0150 |
| Low-resource | B3 real + EDA | 0.7846 ± 0.0208 |
| Imbalanced | B1 real only | 0.8193 ± 0.0086 |
| Imbalanced | B2 oversampling | 0.7968 ± 0.0080 |
| Imbalanced | B3 real + EDA | 0.8225 ± 0.0044 |

See the [development result guide](results/metrics/README.md),
[development summaries](results/metrics/baseline_validation_aggregate.csv), and
[historical checkpoint index](results/metrics/checkpoint_index.csv). The 15 earlier
checkpoint folders total approximately 4.03 GB and remain local under
`results/baselines/distilbert_validated/`. Older tuning checkpoints and the
[frozen development configuration](configs/distilbert_baseline.json) are preserved.
No earlier result is silently relabelled as a new seed or protocol.

## Verification

```bash
python -m unittest discover -s tests -v
```

The official source contains seven normalized train/test text overlaps; these
are disclosed without modifying the benchmark based on test references. All
real training/validation rows remain separate. Generated-only validation text
coincidences are recorded as described in the handoff guide.
