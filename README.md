# Does Filtering Help? Cost-Aware LLM-Generated Synthetic Data for Low-Resource and Imbalanced Text Classification

## Project Overview

This project investigates whether filtering Large Language Model (LLM)-generated synthetic text data improves text classification performance when the available real training data is limited or imbalanced.

We use the Banking77 dataset, which contains customer banking queries across 77 intent classes.

We generate synthetic banking questions using Qwen2.5-1.5B-Instruct and study how different filtering techniques affect classification performance, data quality, diversity, and generation cost.

## Research Objectives

- Investigate the impact of synthetic text augmentation on low-resource text classification.
- Study whether synthetic data improves classification performance for rare classes.
- Compare different synthetic data filtering techniques.
- Evaluate the quality and diversity of generated text.
- Measure the computational cost of generating and filtering synthetic examples.

## Dataset

**Banking77**

| Dataset information | Count |
|---|---:|
| Intent classes | 77 |
| Original training examples | 10,003 |
| Official test examples | 3,080 |
| Validation examples | 770 |
| Low-resource training examples | 770 |
| Imbalanced training examples | 1,315 |

The validation dataset contains 10 real examples from each intent class. The official test dataset is kept separate and untouched.

### Experimental Settings

**1. Low-resource setting**
- 10 real examples per class.
- 770 total training examples.

**2. Imbalanced setting**
- 5 examples for each of 15 rare classes.
- 20 examples for each of the remaining 62 classes.
- 1,315 total training examples.

## Methodology

### 1. Data Preparation
Prepare low-resource and imbalanced training datasets from Banking77.

### 2. Baseline Models
Compare:
- B1: Real training data only.
- B2: Real training data with random oversampling of rare classes.
- B3: Real training data with Easy Data Augmentation (EDA).

### 3. Synthetic Text Generation
Use Qwen2.5-1.5B-Instruct to generate new banking questions based on real examples and their intent labels.

### 4. Synthetic Data Filtering
Compare the following approaches:

- F0: No filtering.
- F1: Exact and near-duplicate removal.
- F2: Label agreement using a classifier.

### 5. Text Classification
Use DistilBERT to train and evaluate the classification models.

### 6. Evaluation
Compare classification performance, synthetic data quality, diversity, and generation cost.

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Qwen2.5-1.5B-Instruct
- DistilBERT
- Scikit-learn
- Sentence Transformers
- Google Colab (GPU)
- Google Drive
- GitHub

## Evaluation Metrics

- Macro-F1 (primary metric)
- Accuracy
- Rare-class F1 and recall
- Synthetic text label correctness
- Text diversity
- Cost per 1,000 accepted synthetic examples

## Project Structure

```text
CS682-Synthetic-Data-Project/
│
├── README.md
├── .gitignore
├── configs/
│   └── distilbert_baseline.json
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 01_reproducible_data_preparation.ipynb
│   └── 02_baseline_models.ipynb
│
├── data/reproducible/          # Local prepared data and manifests (ignored)
│
└── results/
    ├── baselines/             # Local checkpoints and predictions (ignored)
    └── metrics/               # Small versioned experiment summaries
```

## Current Progress

Use `notebooks/01_reproducible_data_preparation.ipynb` for the complete data preparation
workflow: fixed validation, training draws for seeds 42/43/44, official test data,
source IDs, manifests, and integrity checks. Run all cells in VS Code or Colab; no GPU
is required. New outputs are saved to `data/reproducible/`. The original notebook and
existing datasets are preserved and checked for agreement when available.

Next, run `notebooks/02_baseline_models.ipynb` for DistilBERT B1/B2/B3 baselines.
It selects CUDA, Apple MPS, or CPU automatically and loads shared settings from
`configs/distilbert_baseline.json`. The validation check compares learning rates
`2e-5`, `5e-5`, and `1e-4` on seed 42 in both data settings, with a 30-epoch ceiling
and early-stopping patience of 5. Selection uses equal-weight mean validation
macro-F1; learning rates within 0.001 of the highest mean favor the lower rate.
Set `RUN_VALIDATION_CHECK = True` to reproduce that fixed search; leave it false
for the remaining experiments. Search reports are saved in `results/metrics/`.
The completed seed-42 search selected learning rate `1e-4`, with the shared
30-epoch ceiling and patience of 5. Validation macro-F1 was 0.7728 for low-resource
data and 0.8234 for imbalanced data. See the [frozen configuration](configs/distilbert_baseline.json)
and [trial results](results/metrics/baseline_validation_trials.csv). These are
seed-42 development results used for configuration selection. Seeds 43/44 use
the frozen policy without another hyperparameter search.

All baseline training is complete on MPS: six B1 fits, three imbalanced B2 fits,
and six B3 fits. Balanced B2 reuses B1, covering all 18 logical conditions through
15 distinct training jobs. The official test set remains reserved for final evaluation.

| Setting | Baseline | Validation macro-F1 | Rare-class F1 | Rare-class recall |
|---|---|---:|---:|---:|
| Low-resource | B1: real data | 0.7809 ± 0.0150 | 0.7942 ± 0.0122 | 0.7911 ± 0.0315 |
| Low-resource | B3: real + EDA | 0.7846 ± 0.0208 | 0.7743 ± 0.0191 | 0.7733 ± 0.0267 |
| Imbalanced | B1: real data | 0.8193 ± 0.0086 | 0.7297 ± 0.0329 | 0.6644 ± 0.0444 |
| Imbalanced | B2: oversampling | 0.7968 ± 0.0080 | 0.6755 ± 0.0314 | 0.5911 ± 0.0204 |
| Imbalanced | B3: real + EDA | 0.8225 ± 0.0044 | 0.7533 ± 0.0283 | 0.6867 ± 0.0240 |

Values are means ± sample standard deviations across seeds 42/43/44. B3's mean
macro-F1 increased by 0.37 percentage points on low-resource data and 0.32 points
on imbalanced data relative to B1. Individual seed outcomes were mixed. B2 scored below B1 for all three imbalanced
seeds, averaging 2.25 macro-F1 points lower. On imbalanced data, mean rare-class F1 increased by 2.37 points; on low-resource data,
the same designated class group's F1 decreased. These are validation findings
under the shared frozen policy; official test evaluation remains pending.

All six EDA runs accepted one edited question per original, with zero shortfalls:
1,540 training rows per low-resource run and 2,630 per imbalanced run. The six
completed fits logged 54.2 minutes including their original EDA generation time.
Launcher startup and restart overhead is recorded separately under
`results/baselines/distilbert_validated/execution_restarts/`. A runner timeout
extension reused four completed fits; those fits are counted once.

See the [individual B1 results](results/metrics/b1_validation_results.csv),
[individual B2 results](results/metrics/b2_validation_results.csv), and
[paired B1–B2 comparisons](results/metrics/b1_b2_validation_comparison.csv).
The three B2 fits logged 16.8 minutes.

See [individual B3 results](results/metrics/b3_validation_results.csv), the
[B3 summary](results/metrics/b3_validation_aggregate.csv), and
[paired B1–B3 comparisons](results/metrics/b1_b3_validation_comparison.csv).
The [complete baseline results](results/metrics/baseline_validation_results.csv)
and [complete baseline summary](results/metrics/baseline_validation_aggregate.csv)
cover all 15 distinct fits. Source hashes, augmentation provenance, validation
predictions/probabilities, per-class metrics, and aggregates passed independent
checks. B1/B2 artifacts and the frozen configuration were preserved.

The default `RUN_B3_REPEATS = True` runs B3 in both settings for seeds 42/43/44,
reusing completed results. B3 targets one edited question per original, using
synonym replacement, synonym insertion, word swaps, or deletion. Edit strength
and deletion probability are 0.1, with up to 16 attempts per original. Negation
and numeric tokens are protected; unchanged text and duplicates of selected real
training or accepted EDA text are rejected. This is EDA with safeguards, rather
than an exact reproduction of upstream preprocessing. Intended labels can still
change after an edit. Actual acceptance and per-class shortfalls are recorded.

EDA text and reports are cached under `results/baselines/distilbert_validated/eda/`.
Each B3 run saves a best checkpoint, validation predictions, per-class metrics,
and timing logs. Compact B3 and complete baseline summaries are exported to
`results/metrics/`, together with paired B1–B3 score differences.

To repeat only B2, disable `RUN_B3_REPEATS` and enable `RUN_B2_REPEATS`.
B2 retains all 1,315 unique real examples and adds 225
seeded replacement draws, giving 1,540 training entries (20 per class). It uses
ordinary cross-entropy without class weights. Each run saves `balancing_report.json`
with class counts and the number of times each source example is used.
B2 has 97 optimizer batches per epoch, compared with B1's 83 on imbalanced data.
The same epoch and early-stopping policy therefore permits more training work;
score comparisons must also report optimizer steps, token counts, and elapsed time.

Balanced low-resource B2 changes no training rows, so it reuses the corresponding
B1 result as an explicit identity in `balanced_b2_identity.csv`. Set all batch-run
flags to false for one selected condition. `RUN_ALL_BASELINES = True` covers all
18 logical baseline conditions with **15 distinct training jobs**: six B1, three
imbalanced B2, and six B3. Enable only one batch-run flag at a time.
Checkpoints, validation metrics, predictions, and timing logs
are saved under `results/baselines/distilbert_validated/`. The original short run
remains under `distilbert_v1/`. Tuning cost is reported separately, with reused
training counted once. Official test examples stay reserved for final evaluation.

- [x] Downloaded Banking77 dataset.
- [x] Identified all 77 intent classes.
- [x] Created the validation dataset.
- [x] Created the low-resource training dataset.
- [x] Identified 15 rare classes.
- [x] Created the imbalanced training dataset.
- [x] Saved prepared datasets to Google Drive.
- [x] Validate training settings and freeze the shared configuration.
- [x] Complete B1 in both settings for seeds 42/43/44.
- [x] Complete the three imbalanced B2 oversampling runs.
- [x] Complete the six B3 EDA runs.
- [ ] Generate synthetic text using Qwen.
- [ ] Apply synthetic data filtering.
- [ ] Train DistilBERT models.
- [ ] Evaluate results and generation cost.

## Project Environment

The notebooks support local VS Code execution with Apple MPS, CUDA, or CPU, and
Google Colab. Current baseline runs use Apple MPS. Each notebook installs its
required pinned packages when needed. Large datasets and checkpoints stay outside
Git; compact validation summaries are versioned under `results/metrics/`.

GitHub is used for source code, notebooks, and project documentation.

## Course

CS 682 – Neural Networks  
University of Massachusetts Amherst
