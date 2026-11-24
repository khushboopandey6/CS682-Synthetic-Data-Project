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
- B2: Real training data with class balancing.
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

All six B1 conditions have now completed. Across seeds 42/43/44, validation macro-F1
is **0.7809 ± 0.0150** for low-resource data and **0.8193 ± 0.0086** for imbalanced
data (mean ± sample standard deviation). See the [individual B1 results](results/metrics/b1_validation_results.csv)
and [three-seed summary](results/metrics/b1_validation_aggregate.csv). The official
test set remains reserved for final evaluation.

The default `RUN_B1_REPEATS = True` runs B1 in both settings for seeds 42/43/44,
reusing completed results. Set it to false to run one selected condition, or set
`RUN_ALL_BASELINES = True` to run all 18 baseline conditions with the frozen settings.
Selected seed-42 B1 weights are reused. Checkpoints, validation metrics, predictions, and timing logs
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
- [ ] Complete all 18 baseline conditions (all six B1 conditions are ready; B2/B3 remain pending).
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
