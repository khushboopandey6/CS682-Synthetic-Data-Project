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
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_baseline_models.ipynb
│   ├── 03_qwen_generation.ipynb
│   ├── 04_synthetic_filtering.ipynb
│   ├── 05_distilbert_training.ipynb
│   └── 06_evaluation.ipynb
│
├── src/
│   ├── data_preparation.py
│   ├── generation.py
│   ├── filtering.py
│   └── evaluation.py
│
└── results/
    ├── figures/
    └── metrics/
```

## Current Progress

- [x] Downloaded Banking77 dataset.
- [x] Identified all 77 intent classes.
- [x] Created the validation dataset.
- [x] Created the low-resource training dataset.
- [x] Identified 15 rare classes.
- [x] Created the imbalanced training dataset.
- [x] Saved prepared datasets to Google Drive.
- [ ] Train baseline models.
- [ ] Generate synthetic text using Qwen.
- [ ] Apply synthetic data filtering.
- [ ] Train DistilBERT models.
- [ ] Evaluate results and generation cost.

## Project Environment

Experiments are conducted using Google Colab GPU resources. Google Drive is used to store datasets, generated examples, model checkpoints, and experimental results.

GitHub is used for source code, notebooks, and project documentation.

## Course

CS 682 – Neural Networks  
University of Massachusetts Amherst
