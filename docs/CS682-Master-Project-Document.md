# CS682 — Neural Networks Course Project

# Master Project Document: Context, Technical Specifications, Team Responsibilities and Execution Plan

This document is the single source of truth for our project. All five team members should use this document as the common context when working with their respective LLMs or coding assistants.

**Allocation revision · November 25, 2026.** This document supersedes all previous role and run allocations. Use the ownership table in Section 13 and the run matrix in Section 21 for every handoff.

Each team member is responsible for one module. The project is designed so that all five members can develop their modules simultaneously, with clearly defined interfaces and dependencies.

Our priority is to complete a scientifically sound, reproducible research project without introducing unnecessary engineering complexity.

---

# 1. Project Information

**Project title:**

Does Filtering Help? Cost-Aware LLM-Generated Synthetic Data for Low-Resource and Imbalanced Text Classification

**Course:** COMPSCI 682 — Neural Networks

**University:** University of Massachusetts Amherst

**Team members:**

- Khushboo Pandey
- Danush Kanchi
- Dundi Manikanta Ganneni
- Gnaneswarudu Kuna
- Rufina Lourdes Rajesh

**GitHub repository:**

https://github.com/khushboopandey6/CS682-Synthetic-Data-Project

The repository has already been created. Inspect the existing repository and reuse working implementations before creating new files.

**Development environment:** Google Colab.

**Code collaboration:** GitHub.

**Shared experimental storage:** Google Drive.

**Backup computational environment:** Kaggle.

**Target actual expenditure:** $0.

This is a neural networks research project, not a commercial software application.

We do not need a frontend, backend server, database, authentication system, paid LLM API or cloud deployment.

---

# 2. Important Deadlines

| Date | Deliverable |
| --- | --- |
| November 25 | Finalize assignments and shared configuration |
| November 26 | Official dataset splits and independent module development |
| November 27 | Qwen pilot, classifier pilot and configuration freeze, then first official B1 checkpoint |
| November 28 | First complete end-to-end experiment |
| November 29 | Preliminary results and poster draft |
| November 30 | Finalize and review poster |
| December 1 | Official poster submission |
| December 2 | Complete generation and filtering |
| December 3 | Complete majority of classifier experiments |
| December 4 | Remaining experiments, human audit and analysis |
| December 5 | Internal deadline for completing the core project |
| December 6–7 | Presentation preparation and rehearsal |
| December 8 | Official project presentation |
| December 9–14 | Final report writing and revisions |
| December 15 | Official final report submission |

**Poster dimensions:** 2×3 feet.

Recommended orientation: Landscape, 36 inches wide × 24 inches high, unless the professor specifies otherwise.

The professor explicitly permits continued project development after poster submission.

Therefore:

- The poster can contain genuine preliminary results.
- The completed experimental results should be ready by December 5.
- The presentation on December 8 should contain the latest verified results.
- The final report on December 15 should contain the complete research analysis.

Do not delay poster submission while waiting for every experiment to finish.

---

# 3. Project Overview

Machine learning classifiers generally require sufficient labeled training data to achieve good performance.

However, collecting labeled data is expensive and time-consuming.

Two common problems are:

## 3.1 Low-resource classification

Very few labeled training examples are available for each class.

For example, a classifier may need to recognize 77 banking intents but have only 10 labeled examples per intent.

This makes it difficult for the classifier to learn the different ways customers express the same intent.

## 3.2 Imbalanced classification

Some classes contain significantly fewer training examples than others.

For example:

- Common banking intents have 20 training examples.
- Rare banking intents have only 5 training examples.

The classifier may perform well on common classes but poorly on rare classes.

## 3.3 Synthetic data generation

Large language models can generate additional training examples.

For example, given the intent lost_or_stolen_card and the real training examples:

- I lost my debit card.
- Someone stole my bank card.
- I cannot find my card.

An LLM may generate:

- My bank card has gone missing.
- I think someone took my debit card.
- How do I report a stolen card?
- I misplaced my bank card yesterday.

These generated examples can be added to the real training dataset.

However, synthetic data is not necessarily reliable.

Potential problems include:

- Exact duplicates.
- Semantically repetitive sentences.
- Incorrect class labels.
- Label drift.
- Unrealistic customer queries.
- Insufficient diversity.

## 3.4 Central research problem

We want to investigate whether filtering LLM-generated synthetic data before using it for classifier training improves classification performance.

We also want to determine whether the improvement, if any, justifies the additional computational cost.

The goal is not to prove that filtering always works.

Negative, mixed or inconclusive results are scientifically acceptable if the experiments are fair and the conclusions are supported by evidence.

---

# 4. Research Questions

## RQ1 — Filtering effectiveness

Does filtering LLM-generated synthetic data improve downstream text classification performance compared with unfiltered synthetic data?

Which filtering method performs better?

This is the primary research question.

## RQ2 — Synthetic data quantity

How does classification performance change as the quantity of synthetic training data increases?

Does performance improve, plateau or decrease?

## RQ3 — Imbalanced classification

Does filtered synthetic data improve rare-class performance compared with traditional approaches such as random oversampling and Easy Data Augmentation?

## RQ4 — Computational cost

What computational cost is introduced by synthetic data generation and filtering?

Are the classification improvements worth the additional computational cost?

---

# 5. Overall Research Pipeline

The complete experimental pipeline is:

```text
Banking77 Dataset
        |
        v
Data Preparation
        |
        +---------------------------+
        |                           |
        v                           v
Real Training Subsets         Validation/Test Sets
        |
        +---------------------------+
        |                           |
        v                           v
Qwen Synthetic Generation     B1 DistilBERT Training
        |                           |
        v                           |
Shared Raw Synthetic Pool           |
        |                           |
        +-------------+-------------+
        |             |             |
        v             v             v
       F0            F1            F2
   Unfiltered    Deduplication   Classifier
                                 Agreement
        |             |             |
        +-------------+-------------+
                      |
                      v
          Matched Counts Per Class
                      |
                      v
        Real + Synthetic Training Data
                      |
                      v
           DistilBERT Fine-tuning
                      |
                      v
             Evaluation and Analysis
                      |
                      v
        Performance, Diversity and Cost
```

Traditional baseline conditions B1, B2 and B3 are trained separately and compared against F0, F1 and F2.

---

# 6. Dataset: Banking77

Dataset:

https://huggingface.co/datasets/PolyAI/banking77

Banking77 is an English banking customer service intent classification dataset.

Dataset characteristics:

- Total examples: 13,083.
- Official training examples: 10,003.
- Official test examples: 3,080.
- Number of classes: 77.
- Language: English.

Example intents include:

- Card arrival.
- Lost or stolen card.
- Card payment issues.
- Cash withdrawal issues.
- Bank transfers.
- Refund requests.
- PIN-related issues.

The task is to classify each customer query into one of the 77 banking intents.

Use the official Banking77 label mapping throughout the entire project.

Do not independently create different label mappings.

## 6.1 Validation dataset

From the official training dataset, reserve exactly 10 real examples per class for validation.

Validation size:

77 × 10 = 770 examples.

Remaining training pool:

10,003 − 770 = 9,233 examples.

Use validation split seed 42.

The validation dataset remains fixed across all experiments.

The validation dataset is used for:

- Model checkpoint selection.
- Filtering confidence threshold selection.
- Any necessary hyperparameter selection.

Report the amount of validation supervision separately because 770 validation examples represent substantial additional real-data supervision relative to the low-resource training setting.

## 6.2 Test dataset

Use the official Banking77 test dataset containing 3,080 examples.

The test dataset must remain untouched during development and model selection.

Never use test examples for:

- Training.
- Few-shot prompting.
- Synthetic generation.
- Duplicate-filtering references.
- Threshold selection.
- Hyperparameter optimization.

Test results are for reporting, not for selecting experimental configurations.

## 6.3 Experimental setting 1: Low resource

Select 10 real training examples per class.

Total training examples:

77 × 10 = 770.

All classes have equal representation.

Create three reproducible training subsets using experiment seeds:

- 42
- 123
- 456

## 6.4 Experimental setting 2: Imbalanced

Select 15 classes as artificially rare classes.

Use rare-class selection seed 42.

Keep the same rare-class identities across all three experiment seeds.

Rare classes:

15 × 5 = 75 examples.

Majority classes:

62 × 20 = 1,240 examples.

Total real training examples:

75 + 1,240 = 1,315.

Create three reproducible training subsets using experiment seeds:

- 42
- 123
- 456

## 6.5 Dataset integrity requirements

- Preserve official label IDs and names.
- Assign stable unique IDs to real examples.
- Keep validation and test data separate from training.
- Save the selected training example IDs.
- Save the fixed rare-class IDs.
- Verify class distributions.
- Ensure no unintended overlap between dataset splits.
- Use only the corresponding sampled real training subset for generation prompts.

---

# 7. Final Technical Decisions

The following are our standardized implementation choices.

They are selected to keep the project manageable while preserving research quality.

| Component | Selected implementation |
| --- | --- |
| Dataset | Banking77 |
| Generator | Qwen2.5-1.5B-Instruct |
| Classifier | DistilBERT-base-uncased |
| Embedding model | all-MiniLM-L6-v2 |
| Framework | PyTorch and Hugging Face |
| Environment | Google Colab Free |
| Backup | Kaggle |
| Experiment seeds | 42, 123, 456 |
| Validation seed | 42 |
| Rare-class selection seed | 42 |
| B2 | Random oversampling |
| B3 | Easy Data Augmentation |
| Few-shot examples | 3 per intent |
| Generation temperature | 0.7 |
| Generation top-p | 0.9 |
| Raw generation target | 50 parsed examples per class |
| Queries requested per prompt | 10 |
| Maximum generation attempts per class | 10 |
| Maximum new tokens | 512 |
| F1 cosine threshold | 0.95 |
| F2 confidence threshold | Validation-selected |
| Learning rate | One value selected by the 12-fit validation pilot in Section 16 |
| Training batch size | 16 |
| Training duration | Up to 30 epochs, with validation each epoch and early-stopping patience of 5 |
| Maximum classifier input length | 128 tokens |
| Weight decay | 0.01 |
| Primary augmentation target | 1× |
| Primary metric | Macro-F1 |
| Human audit | Approximately 200 examples |

Use the shared classifier protocol in Sections 16 and 21. Other numerical defaults may be revised through the documented pilot and team review process.

Use these defaults unless a pilot reveals a genuine technical or computational problem.

If a change is necessary:

1. Explain the problem.
2. Propose the simplest solution.
3. Identify affected modules.
4. Communicate the change to the team.
5. Update the shared configuration.
6. Record the change in the experimental documentation.

Do not independently redesign the project.

---

# 8. Models

## 8.1 Synthetic data generator

Model:

Qwen/Qwen2.5-1.5B-Instruct

https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct

Qwen is a pretrained instruction-following large language model.

We use it directly for synthetic data generation.

We do not train Qwen from scratch.

We do not fine-tune Qwen.

For each intent, provide:

- The intent name.
- Three real training examples.
- Instructions to generate diverse customer queries belonging to that intent.

Use the model's official chat template through Hugging Face Transformers.

## 8.2 Downstream classifier

Model:

distilbert-base-uncased

https://huggingface.co/distilbert/distilbert-base-uncased

DistilBERT is a smaller pretrained transformer encoder.

We fine-tune it for 77-class text classification.

The model receives a banking query and predicts its intent.

Use the same classifier architecture and training framework across all experimental conditions.

## 8.3 Sentence embedding model

Model:

sentence-transformers/all-MiniLM-L6-v2

https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

This model converts sentences into numerical vectors called embeddings.

Semantically similar sentences generally have similar embeddings.

We use cosine similarity between embeddings to detect near-duplicate synthetic examples.

## 8.4 Libraries

Use lightweight, established Python libraries:

- PyTorch.
- Transformers.
- Datasets.
- Accelerate.
- Sentence Transformers.
- NumPy.
- Pandas.
- Scikit-learn.
- Matplotlib.
- NLTK.
- PyYAML.
- Pytest.

Pin compatible dependency versions in requirements.txt after testing.

Avoid unnecessary frameworks and dependencies.

---

# 9. Mandatory Experimental Conditions

There are six core conditions.

## B1 — Real Data Only

Train DistilBERT using only the sampled real training dataset.

This establishes the baseline performance.

The B1 classifier also serves as the filtering classifier for F2.

Train one corresponding B1 classifier for every setting and experiment seed.

## B2 — Random Oversampling

Use random oversampling to bring minority classes up to the majority-class count.

For the imbalanced setting:

- Rare classes initially contain 5 examples.
- Majority classes contain 20 examples.
- Oversample rare classes to 20 examples.

Train DistilBERT using the resulting dataset.

For the balanced low-resource setting, random oversampling has no effect.

If B2 has exactly the same training inputs and configuration as B1, reuse the B1 result transparently and document that B2 is identical in this setting.

Do not waste GPU time repeating identical experiments.

## B3 — Easy Data Augmentation

Use standard EDA techniques:

- Synonym replacement.
- Random insertion.
- Random swap.
- Random deletion.

Use a simple fixed augmentation configuration, initially alpha = 0.1.

Target one augmented example per real training example for the primary experiment.

Use reproducible randomness.

Train DistilBERT using the real and EDA-generated examples.

Report the actual augmentation counts.

Because matched F0/F1/F2 datasets may contain fewer examples than B3, disclose any differences in training data quantity when comparing them.

If computationally feasible, additionally compare B3 at the matched synthetic counts.

## F0 — Unfiltered LLM-Generated Synthetic Data

Use the synthetic data generated by Qwen.

Retain all successfully parsed, nonempty customer queries before matched-count subsampling.

Do not remove duplicates.

Do not remove incorrectly labeled examples.

Do not perform semantic quality filtering.

Basic parsing of malformed model outputs is allowed as part of the shared generation process.

Train DistilBERT using real training examples plus the matched F0 synthetic examples.

## F1 — Exact and Near-Duplicate Filtering

Use the same raw synthetic pool as F0.

Apply text normalization for comparison purposes.

Suggested normalization:

- Convert text to lowercase.
- Strip leading and trailing whitespace.
- Normalize repeated whitespace.
- Apply a consistent punctuation normalization policy.

Preserve the original sentence text for classifier training.

Remove exact duplicates.

Generate sentence embeddings using MiniLM.

Calculate cosine similarity.

Use 0.95 as the initial near-duplicate threshold.

Compare synthetic examples against:

- Real training examples of the same intended class.
- Previously retained synthetic examples of the same intended class.

Process examples in a deterministic order.

Do not compare against validation or test examples.

Record acceptance and rejection decisions.

Train DistilBERT using real training examples plus matched F1 synthetic examples.

## F2 — Classifier-Based Label Agreement Filtering

Use the same raw synthetic pool as F0 and F1.

Use the corresponding B1 classifier trained only on the real training subset.

For each synthetic example:

1. Predict its class using B1.
2. Obtain the predicted class confidence using softmax.
3. Compare the predicted class with the intended class.
4. Accept only when the predicted class equals the intended class and confidence meets the selected threshold.

Select the confidence threshold using the real validation dataset.

Candidate thresholds:

- 0.3
- 0.5
- 0.7
- 0.9

Suggested selection policy:

Treat a validation prediction as correct if its predicted class matches its true class.

For each candidate threshold, treat predictions above that threshold as accepted.

Calculate precision, recall and F1 for identifying correct predictions.

Choose the threshold with the highest validation selection F1.

If tied, choose the lower threshold to preserve coverage.

Record the selected threshold for each corresponding B1 classifier.

If all thresholds provide negligible coverage, investigate using validation data only and document any necessary adjustment.

Do not select thresholds using test performance.

F2 is independent of F1.

Do not automatically apply deduplication before F2.

Train a new downstream DistilBERT classifier using real training examples plus matched F2 synthetic examples.

Do not continue training the B1 filtering classifier as the final F2 classifier.

The downstream F2 model must start from the same pretrained initialization policy as other conditions.

---

# 10. Synthetic Data Generation Methodology

## 10.1 Prompt construction

For each intent, select three real examples from the corresponding sampled training subset.

Use reproducible selection.

Example prompt:

```text
You are generating synthetic customer service queries
for a banking intent classification dataset.

Intent: lost_or_stolen_card

Examples:
1. I lost my debit card.
2. Someone stole my bank card.
3. I cannot find my card.

Generate 10 new and diverse customer service queries
belonging to the same intent.

Use natural language and vary sentence structure.

Return one customer query per line.

Do not include explanations, headings or intent labels.
```

The actual intent names must come from the official Banking77 label mapping.

## 10.2 Generation parameters

- Temperature: 0.7.
- Top-p: 0.9.
- Sampling enabled.
- Three few-shot examples.
- Request approximately 10 queries per prompt.
- Maximum new tokens: 512.
- Maximum attempts per class: 10.

Use fixed and recorded generation seeds.

Use deterministic prompt construction.

Record the exact generation configuration.

## 10.3 Generation budget

Initial target:

50 successfully parsed synthetic examples per class for every setting and seed.

Maximum target:

77 classes × 50 examples × 2 settings × 3 seeds = 23,100 parsed synthetic examples.

This is a proposed generation target and must be validated through the pilot.

If full generation is computationally impractical, reduce the common budget consistently before the final experiments.

A smaller shared generation budget is preferable to incomplete or unfair experiments.

Use the same raw generation policy across all settings and seeds.

Preserve all raw model responses.

If the attempt limit is reached before collecting the target number of examples, record the shortfall.

Do not silently keep generating unlimited examples.

Do not discard successfully parsed duplicates or incorrect-label examples during generation.

Filtering must remain a separate experimental stage.

## 10.4 Generation pilot

Before full generation:

- Select three representative banking intents.
- Generate approximately 100 examples in total.
- Manually inspect label correctness.
- Check exact and semantic uniqueness.
- Check output parsing.
- Measure generation throughput.
- Measure GPU memory usage.
- Estimate full generation runtime.

Pilot targets from the proposal:

- Approximately 70% or higher label consistency.
- Approximately 90% or higher uniqueness after near-duplicate analysis.

These are pilot feasibility targets, not final experimental findings.

If the pilot performs poorly:

1. Improve the prompt.
2. Make a minimal adjustment to generation parameters.
3. Repeat the pilot.

Only consider Qwen2.5-3B-Instruct if the smaller model remains inadequate and computational resources permit.

## 10.5 Generation checkpointing

Google Colab sessions may disconnect.

Generation must support resuming from previously completed work.

Save outputs regularly, preferably after each completed class or generation batch.

Record completed intents.

Do not regenerate completed examples unnecessarily.

Do not overwrite existing completed outputs without an explicit restart decision.

## 10.6 Shared raw generation pools

Produce six raw synthetic pools:

```text
low_resource/seed_42.jsonl
low_resource/seed_123.jsonl
low_resource/seed_456.jsonl

imbalanced/seed_42.jsonl
imbalanced/seed_123.jsonl
imbalanced/seed_456.jsonl
```

F0, F1 and F2 must use the same corresponding raw pool.

Do not independently generate synthetic data for each filter.

---

# 11. Fair Comparison and Matched Synthetic Counts

This is one of the most important parts of the research methodology.

Filtering methods retain different numbers of examples.

A classifier trained using 1,000 synthetic examples cannot be directly compared with one trained using 200 synthetic examples without considering the quantity difference.

Our primary comparison controls the number of synthetic examples per class.

For each setting, seed and class:

```text
matched_count = min(
    available_F0_examples,
    available_F1_examples,
    available_F2_examples,
    target_augmentation_count
)
```

For the primary 1× augmentation target:

- Low-resource class target: 10 synthetic examples.
- Imbalanced rare-class target: 5 synthetic examples.
- Imbalanced majority-class target: 20 synthetic examples.

Example:

For one low-resource class:

- F0 retains 50 examples.
- F1 retains 32 examples.
- F2 retains 8 examples.
- Target augmentation count is 10.

Matched count:

min(50, 32, 8, 10) = 8.

Use eight synthetic examples from each filtering condition for that class.

Use reproducible random subsampling.

If a class has zero accepted examples in any condition, use zero synthetic examples for that class across all three matched conditions.

Always preserve the real training examples.

Record:

- Raw generated counts.
- Accepted counts before matching.
- Rejected counts.
- Matched counts.
- Class-wise acceptance rates.
- Actual augmentation ratios.

Do not describe an experiment as exactly 1× augmented if the matched-count procedure results in fewer examples.

This procedure isolates filtering quality more fairly by controlling synthetic data quantity.

Also acknowledge that a restrictive filter may reduce the achievable augmentation budget for all matched conditions.

---

# 12. Experimental Scope

Core experimental results:

6 conditions × 2 settings × 3 seeds = 36 condition-setting-seed results.

Conditions:

- B1
- B2
- B3
- F0
- F1
- F2

Because B2 is identical to B1 in the balanced low-resource setting, those three results can be reused with explicit documentation.

The 36 result entries require 33 independent classifier runs when all three balanced B2 entries reuse identical B1 runs, or 36 when every entry is trained separately. Manikanta owns the baseline runs (15 with all three reuses or 18 with separate runs), and Khushboo and Gnaneswarudu own the 18 synthetic-augmentation runs.

## Optional experiments

After completing the core experiments, attempt:

1. Synthetic quantity experiments at 0.5× and 2× augmentation.
2. Additional qualitative analysis.
3. Additional rare-class analysis.

Select the best filtering strategy using validation performance, not test performance.

For quantity experiments, compare against the corresponding unfiltered condition at matched quantities where feasible.

If sufficient synthetic examples are unavailable for 2× augmentation, report the actual achievable ratio.

If quantity experiments cannot be completed, explicitly acknowledge that RQ2 remains partially unanswered.

## Do not prioritize

- F3 LLM self-verification.
- F4 combined filtering.
- Additional datasets.
- Qwen fine-tuning.
- vLLM optimization.
- Extensive hyperparameter searches.
- Larger models.
- Commercial application development.

These are unnecessary for the core project and may introduce deadline risks.

---

# 13. Team Responsibilities

| Team member | Module | Primary responsibility |
| --- | --- | --- |
| Dundi Manikanta Ganneni | A | Official data splits, baseline code PR, all B1/B2/B3 runs, B1 checkpoints and validation predictions |
| Danush Kanchi | B | Qwen pilot, six raw generation pools and generation logs |
| Khushboo Pandey | C | F0/F1/F2 filtering and matching for six pools, plus six F1 classifier runs |
| Gnaneswarudu Kuna | D | Common DistilBERT trainer, six F0 runs and six F2 runs |
| Rufina Lourdes Rajesh | E | Evaluation, cost analysis, audit coordination, research comparison and poster |

All five members should work simultaneously, contribute to the final report and participate in the human audit.

Each module must have independently testable code, documented inputs and documented outputs.

---

# MODULE A — Data Preparation and Baselines

**Owner:** Dundi Manikanta Ganneni

**Git branch:** Use Manikanta’s existing baseline work for the PR. Use `feature/data-baselines` if a separate integration branch is needed.

## Objective

Prepare the official datasets and execute all B1/B2/B3 baseline experiments with the shared Module D trainer.

## Responsibilities

1. Inspect existing repository progress.
2. Download Banking77.
3. Preserve the official label mapping.
4. Assign stable example IDs.
5. Create the fixed validation dataset.
6. Preserve the official test dataset.
7. Create three low-resource training subsets.
8. Select the 15 fixed rare classes.
9. Create three imbalanced training subsets.
10. Implement B1 real-only dataset preparation.
11. Implement B2 random oversampling.
12. Implement B3 Easy Data Augmentation.
13. Verify class distributions and dataset sizes.
14. Check dataset leakage.
15. Export all datasets using the shared data contract.
16. Document the data preparation methodology.
17. Open or update the PR for Manikanta’s existing code. Inspect the repository and his working branch, reconcile overlapping implementation, and reuse working code.
18. Run all B1/B2/B3 combinations across both settings and all three seeds, using the documented B2 reuse rule.
19. Publish the six selected B1 checkpoints and their matching validation predictions for Module C. Publish baseline test predictions, metrics and runtime logs for Module E using the shared artifact layout in Sections 15–20.

## Suggested files

```text
notebooks/01_data_preparation.ipynb
notebooks/02_baseline_models.ipynb
src/data_preparation.py
src/baselines.py
tests/test_data.py
```

Reuse existing working notebook implementations where appropriate.

## Required outputs

```text
01_data/
    low_resource/
        seed_42.jsonl
        seed_123.jsonl
        seed_456.jsonl

    imbalanced/
        seed_42.jsonl
        seed_123.jsonl
        seed_456.jsonl

    baselines/
        B2/
        B3/

    validation.jsonl
    test.jsonl
    label_mapping.json
    rare_classes.json
    dataset_statistics.json
```

## Dependencies

Data preparation can start independently. Baseline training uses Module D’s shared trainer.

Module A is the first official data provider.

Modules B and D require its official training subsets for final experiments.

## Acceptance criteria

- Exactly 77 classes.
- Exactly 770 validation examples.
- Exactly 3,080 official test examples.
- Exactly 770 real examples per low-resource seed.
- Exactly 1,315 real examples per imbalanced seed.
- Fixed validation and rare-class definitions.
- Correct label mapping.
- Reproducible sampling.
- No unintended split overlap.
- Valid B2 and B3 datasets.
- Eighteen baseline result entries with documented provenance for any B2 reuse.
- Six B1 checkpoints and six corresponding validation prediction exports with true labels, predicted labels and confidence.
- Existing baseline code submitted or updated in a PR.
- Outputs compatible with other modules.

## Deadlines

- November 26: Release official real training subsets, validation data and label mapping.
- November 27: Complete baseline dataset preparation and release the low-resource seed-42 B1 checkpoint and validation predictions.
- November 28 onward: Run the remaining B1/B2/B3 matrix and publish artifacts progressively.
- November 29: Provide dataset statistics for the poster.
- December 5: Complete all assigned experimental responsibilities.

## Literature responsibilities

- Casanueva et al. — Banking77.
- Wei and Zou — Easy Data Augmentation.

## Poster responsibility

Dataset description, class distributions and experimental settings.

---

# MODULE B — Qwen Synthetic Data Generation

**Owner:** Danush Kanchi

**Git branch:** feature/generation

## Objective

Develop a reproducible, resumable LLM inference pipeline that generates synthetic banking queries for all experimental settings and seeds.

## Responsibilities

1. Inspect existing repository progress.
2. Set up Google Colab GPU execution.
3. Load Qwen2.5-1.5B-Instruct.
4. Use the official Qwen chat template.
5. Implement class-conditioned few-shot prompting.
6. Select three real training examples per intent.
7. Implement configurable generation parameters.
8. Implement batched inference where practical.
9. Parse generated queries.
10. Handle malformed model outputs.
11. Implement the generation pilot.
12. Measure preliminary label consistency and uniqueness.
13. Measure generation throughput.
14. Implement checkpointing and resume functionality.
15. Generate all six raw synthetic pools.
16. Preserve raw model responses.
17. Record generation parameters and model version.
18. Record token counts and runtime.
19. Record GPU information.
20. Hand over generated pools progressively to Module C.
21. Document generation methodology.

## Suggested files

```text
notebooks/03_qwen_generation.ipynb
src/generation.py
src/prompts.py
tests/test_generation.py
```

## Suggested functions

```python
def load_generator(config):
    ...

def select_few_shot_examples(dataset, label, seed):
    ...

def build_prompt(intent_name, examples, num_queries):
    ...

def generate_batch(prompts, model, tokenizer, config):
    ...

def parse_generated_queries(response):
    ...

def save_generation_checkpoint(records, output_path):
    ...

def resume_generation(output_path):
    ...
```

These are suggested interfaces. Use the simplest implementation that satisfies the shared data contract.

## Required outputs

- Six raw synthetic JSONL pools.
- Raw model responses.
- Prompt configurations.
- Pilot results.
- Generation runtime logs.
- Token usage logs.
- Generation checkpoints.
- Generation methodology documentation.

## Dependencies

Official generation requires Module A's real training subsets.

Development can begin immediately using a temporary subset of Banking77's official training split.

Do not use the official test set for development prompts.

Danush owns the pilot, all six raw generation pools and their logs. Module B has no downstream classifier runs.

## Acceptance criteria

- Qwen runs successfully on Colab.
- Generation uses only allowed real training examples.
- All 77 intents are attempted.
- Raw generation budgets are fixed and documented.
- Shortfalls are recorded.
- Generated examples follow the shared format.
- Interrupted generation can resume.
- Raw outputs are preserved.
- Runtime and token counts are recorded.
- All three filters receive the same raw pools.
- No hidden quality filtering occurs during generation.

## Deadlines

- November 26: Working generation implementation.
- November 27: Complete pilot and begin official generation.
- November 28: Release first complete raw pool.
- November 29: Provide preliminary generation statistics.
- December 2: Complete all six raw pools.
- December 5: Complete assigned training runs and documentation.

## Literature responsibilities

- Qwen2.5 Technical Report.
- Kumar et al. — Data Augmentation Using Pre-trained Transformer Models.
- Yoo et al. — GPT3Mix.

Focus on synthetic generation, few-shot prompting, label preservation and generation efficiency.

## Poster responsibility

Qwen generation methodology, example prompts, synthetic query examples and generation runtime.

---

# MODULE C — Filtering, Matching and F1 Training

**Owner:** Khushboo Pandey

**Git branch:** feature/filtering

## Objective

Implement the three synthetic data filtering conditions, produce matched datasets for all six pools and train the six F1 classifiers.

## Responsibilities

1. Implement F0 unfiltered control.
2. Implement text normalization.
3. Implement exact duplicate detection.
4. Load MiniLM sentence embeddings.
5. Implement cosine similarity.
6. Implement F1 near-duplicate filtering.
7. Implement deterministic duplicate processing.
8. Implement F2 classifier inference.
9. Implement label-agreement checking.
10. Implement confidence-based filtering.
11. Select confidence thresholds using validation data.
12. Record acceptance and rejection decisions.
13. Implement matched-count subsampling.
14. Calculate class-wise acceptance rates.
15. Calculate overall acceptance rates.
16. Record filtering runtime.
17. Record GPU information.
18. Deliver filtering logs and acceptance statistics to Module E for the cost and quality analyses.
19. Export matched F0 and F2 datasets to Module D and use the matched F1 datasets for F1 training.
20. Train all six F1 classifiers with the shared Module D trainer and publish predictions, metrics and runtime logs.
21. Document the filtering and matching methodology and contribute interpretation of filtering bias and quality-diversity trade-offs.

## Suggested files

```text
notebooks/04_synthetic_filtering.ipynb
src/filtering.py
tests/test_filtering.py
```

## Suggested functions

```python
def filter_f0(samples):
    ...

def normalize_text(text):
    ...

def remove_exact_duplicates(samples, real_examples):
    ...

def remove_near_duplicates(samples, real_examples, threshold):
    ...

def select_confidence_threshold(validation_predictions):
    ...

def filter_label_agreement(samples, classifier, threshold):
    ...

def match_class_counts(filtered_datasets, target_counts, seed):
    ...

def calculate_acceptance_rate(raw_data, filtered_data):
    ...
```

## Required outputs

```text
03_filtered_data/
    F0/
    F1/
    F2/

    matched/
        F0/
        F1/
        F2/

    filtering_decisions/
    acceptance_statistics/
    filtering_logs/
```

Each condition directory must distinguish the experimental setting and seed.

## Dependencies

Development can begin immediately using mock synthetic examples.

For F2 development, use mock classifier predictions initially.

Final F0 and F1 execution requires Module B's raw synthetic data.

Final F2 execution requires:

- Module B's raw synthetic data.
- Module A's corresponding B1 classifier checkpoint and matching validation predictions.
- Module A's fixed validation data for validation IDs and ground truth.

Mock predictions must never be used for final reported results.

## Acceptance criteria

- F0 retains all valid parsed examples before matching.
- F1 correctly removes duplicates.
- F2 correctly applies label agreement and confidence thresholds.
- F2 does not automatically apply F1.
- Filtering decisions are traceable.
- Acceptance rates are recorded.
- Matched datasets contain equal synthetic counts per class.
- Real training examples are preserved.
- Filtering runtime and hardware logs are available to Module E.
- Outputs are compatible with the common trainer.
- Six F1 runs have predictions, metrics and training logs.

## Deadlines

- November 26: F0/F1 implementation and mock F2 logic.
- November 27: Complete filtering tests.
- November 28: First matched filtered datasets.
- November 29: Preliminary acceptance rates and filtering runtime logs for Rufina.
- December 2: Complete filtering for all six pools.
- December 5: Complete all six F1 classifier runs.

## Literature responsibilities

- Anaby-Tavor et al. — Do Not Have Enough Data? Deep Learning to the Rescue!
- Meng et al. — SuperGen.
- Ye et al. — ZeroGen.

Focus on synthetic data selection, classifier-based filtering and quality control.

## Poster responsibility

Filtering and matching methodology, F0/F1/F2 comparison and acceptance rates.

---

# MODULE D — DistilBERT Training Framework

**Owner:** Gnaneswarudu Kuna

**Git branch:** feature/classifier-training

## Objective

Develop one reusable classifier training framework for all experimental conditions and execute the six F0 and six F2 runs.

## Responsibilities

1. Load pretrained DistilBERT.
2. Configure a 77-class classification head.
3. Implement tokenization.
4. Implement dataset loading.
5. Implement reusable training code.
6. Implement reproducible random seeds.
7. Configure common training hyperparameters.
8. Implement validation-based checkpoint selection.
9. Give Manikanta an early working trainer for the first B1 run and ensure its checkpoint and validation prediction exports meet Module C’s needs.
10. Support Manikanta’s B1/B2/B3 and Khushboo’s F1 runs through the common training entry point.
11. Train all six F0 classifiers using matched F0 data.
12. Train all six F2 classifiers using matched F2 data.
13. Ensure consistent model initialization across conditions.
14. Export validation predictions where needed.
15. Export final test predictions.
16. Export predicted class probabilities or confidence.
17. Record training runtime.
18. Record GPU information.
19. Implement checkpointing where practical.
20. Provide execution instructions to all team members.
21. Maintain shared execution instructions and trainer compatibility across run owners.

## Important implementation requirement

Do not implement six separate training frameworks.

Create one reusable training function.

Suggested interface:

```python
def train_classifier(
    train_data,
    validation_data,
    condition,
    setting,
    seed,
    config
):
    ...
```

Use the same framework for B1, B2, B3, F0, F1 and F2.

The training framework should accept datasets produced by Modules A and C.

## Training configuration

Use the versioned policy in Section 16 for all six conditions. Fully fine-tune DistilBERT with a 77-class head and start each fit from the same pinned pretrained model revision with a freshly seeded classification head. Use B1 as the F2 filtering teacher. Initialize downstream F2 from the pinned pretrained revision.

Use dynamic padding if practical. Keep the effective training batch size at 16. If memory requires smaller microbatches, use gradient accumulation, record the microbatch and accumulation settings, and retain float32 and the agreed optimizer-step schedule.

## Suggested files

```text
notebooks/05_distilbert_training.ipynb
src/training.py
src/inference.py
tests/test_training.py
```

## Required outputs

```text
04_checkpoints/
    B1/
    B2/
    B3/
    F0/
    F1/
    F2/

05_results/
    predictions/
    validation_predictions/
    training_metrics/

06_cost_logs/
    training/
```

This is the shared artifact layout for all training owners. Manikanta retains all six B1 checkpoints and their validation predictions because Khushboo needs them for F2.

Other large checkpoints can be removed after required predictions, metrics and reproducibility information have been safely preserved, if storage is limited.

## Dependencies

Development can begin immediately using a temporary Banking77 training subset.

Module A runs B1–B3 with official splits and shared trainer code. Module C runs F1 with matched data. Module D runs F0 and F2 with Module C’s matched data. Prioritize trainer readiness for Module A’s B1 runs because F2 depends on them.

## Acceptance criteria

- DistilBERT runs successfully on Colab.
- Correct 77-class classification head.
- Reusable training implementation.
- Consistent training configuration.
- Validation-based model selection.
- No test leakage.
- Checkpoint and validation prediction exports support Module A’s six B1 runs.
- Six F0 and six F2 runs have predictions, metrics and training logs.
- Valid prediction exports.
- Training runtime logs.
- Other members can execute experiments using the same code.

## Deadlines

- November 26: Working training framework.
- November 27: Support Module A’s first B1 checkpoint and validation prediction export.
- November 28: First end-to-end classifier comparison.
- November 29: Preliminary classifier results.
- December 3: Majority of core training runs completed.
- December 5: Complete all six F0 and all six F2 runs.

## Literature responsibilities

- Sanh et al. — DistilBERT.
- Relevant low-resource text classification methodology.

## Poster responsibility

Downstream classifier architecture, training methodology and experimental conditions.

---

# MODULE E — Evaluation, Visualization and Research Comparison

**Owner:** Rufina Lourdes Rajesh

**Git branch:** feature/evaluation

## Objective

Develop the evaluation framework, consolidate results, own computational cost analysis and coordinate the audit, poster and final research comparison.

## Responsibilities

1. Implement accuracy.
2. Implement precision and recall.
3. Implement macro-F1.
4. Implement per-class F1.
5. Implement rare-class F1.
6. Implement rare-class recall.
7. Implement confusion matrices.
8. Aggregate results across three seeds.
9. Calculate mean and standard deviation.
10. Implement distinct-1 and distinct-2.
11. Implement self-BLEU.
12. Implement embedding-based diversity analysis.
13. Design the human audit rubric.
14. Coordinate approximately 200 human-audited examples.
15. Analyze label correctness.
16. Analyze filter precision.
17. Create filtering acceptance-rate visualizations.
18. Create performance versus cost visualizations.
19. Create rare-class performance comparisons.
20. Create final results tables.
21. Coordinate literature review.
22. Identify fair comparisons with prior research.
23. Coordinate poster assembly.
24. Coordinate results and discussion sections of the final report.
25. Consolidate generation, filtering and classifier training logs from Modules A–D. Own the cost calculations and comparisons in Section 22, including cost per 1,000 accepted examples and the separate treatment of B1 setup cost.

## Primary metric

Macro-F1.

Macro-F1 calculates the F1 score independently for each class and averages across all 77 classes.

This prevents majority classes from dominating the evaluation.

Use the full official label set when calculating macro-F1.

Handle zero-division cases consistently.

## Secondary metrics

- Accuracy.
- Per-class precision.
- Per-class recall.
- Per-class F1.
- Rare-class macro-F1.
- Rare-class recall.
- Filtering acceptance rate.
- Synthetic data diversity.
- Human-evaluated label correctness.
- Filter precision.
- Computational cost.

Rare-class metrics should use the fixed 15 rare classes in the imbalanced setting.

## Diversity metrics

Use:

- Distinct-1: Unique unigrams divided by total unigrams.
- Distinct-2: Unique bigrams divided by total bigrams.
- Self-BLEU: Measures similarity between generated sentences.
- Embedding spread: Measures semantic variation.

Use a reproducible, size-controlled sampling strategy for expensive diversity metrics.

Where appropriate, compare diversity within the same intent rather than allowing differences in intent distribution to dominate the result.

Use the same sampling and metric definitions across conditions.

## Suggested files

```text
notebooks/06_evaluation.ipynb
src/evaluation.py
src/visualization.py
docs/literature_review.md
tests/test_evaluation.py
```

## Required outputs

```text
05_results/
    metrics/
    figures/
    human_audit/
    comparison_tables/
    final_summary.csv
```

## Dependencies

Development can begin immediately using mock predictions and mock logs.

Final evaluation requires outputs from Modules B, C and D.

Literature review and poster preparation can begin immediately.

## Acceptance criteria

- Correct metric calculations.
- Results aggregated across all available seeds.
- Correct rare-class IDs.
- Reproducible visualizations.
- Documented human audit methodology.
- Fair comparisons.
- Transparent limitations.
- Accurate literature citations.
- Poster submitted on time.
- Final report addresses professor feedback.

## Deadlines

- November 26: Evaluation implementation and poster outline.
- November 27: Metric verification and literature review.
- November 28: Evaluate first end-to-end experiment.
- November 29: Complete initial poster draft.
- November 30: Finalize poster with team.
- December 1: Poster submission.
- December 4: Human audit and final analysis.
- December 5: Complete results tables and figures.

## Literature responsibilities

- Li et al. — Synthetic Data Generation with Large Language Models for Text Classification: Potential and Limitations.
- Shumailov et al. — AI models collapse when trained on recursively generated data.
- Cross-paper research comparisons.

The model-collapse paper provides motivation regarding synthetic data diversity. Our project is not a direct experiment on recursive model collapse.

## Poster responsibility

Research motivation, research gap, preliminary results, figures, conclusion and poster assembly.

All team members must contribute content and review the poster.

---

# 14. Parallel Development Strategy

The final execution pipeline contains dependencies, but development should happen simultaneously.

## Module A

Develop official data preparation independently. Integrate the verified existing baseline code through Manikanta’s PR, then start B1 with the shared trainer as soon as the splits are available.

## Module B

Develop generation using a temporary Banking77 training subset.

Replace temporary data with Module A's official subsets before final generation.

## Module C

Develop filtering using mock synthetic examples.

Include intentional duplicates, near-duplicates and incorrect labels.

Use simulated classifier predictions to test F2 logic until B1 is available.

## Module D

Develop the classifier training framework using a temporary Banking77 training subset.

Give Module A an early version of the trainer for B1. Start F0 and F2 runs as matched data arrives.

## Module E

Develop evaluation using mock predictions.

Begin literature review and poster preparation immediately.

## Actual execution dependencies

```text
A / Manikanta publishes official splits
    ├── B / Danush produces raw pools and generation logs
    └── A / Manikanta runs B1 with D’s common trainer
             and publishes checkpoints and validation predictions
                    ↓
C / Khushboo filters and matches F0, F1 and F2
    ├── C / Khushboo trains F1 with D’s common trainer
    └── D / Gnaneswarudu trains F0 and F2
                    ↓
E / Rufina evaluates results and analyzes costs across conditions
```

Module A’s B1 training and Module B’s generation run in parallel. Module A also runs B2 and B3. Module C can run F0 and F1 filtering before B1 artifacts are available. F2 requires the corresponding B1 checkpoint and validation predictions from Module A.

The matched-count comparison requires all three filtering conditions.

Do not wait for all six synthetic pools to finish before starting filtering and downstream training.

Release outputs progressively.

## Recommended integration order

1. Low-resource, seed 42.
2. Imbalanced, seed 42.
3. Low-resource, seed 123.
4. Low-resource, seed 456.
5. Imbalanced, seed 123.
6. Imbalanced, seed 456.

Prioritize one complete end-to-end experiment before scaling to all settings and seeds.

---

# 15. Shared Data Contracts

All modules must follow the same data formats.

Use JSONL for example-level datasets and predictions.

Use JSON or CSV for summary metrics and runtime logs.

Each JSONL file contains one JSON object per line.

The numerical label IDs, confidence values and metrics shown below are illustrative.

They are not actual Banking77 label mappings or experimental results.

## 15.1 Real dataset format

```json
{
  "id": "real_000001",
  "text": "I lost my bank card.",
  "label": 12
}
```

Required fields:

- id: Stable unique identifier.
- text: Original customer query.
- label: Official integer class ID.

Preserve the original real example IDs when sampling subsets.

For oversampled or EDA-generated records, assign unique record IDs and preserve the original example ID in an additional parent_id field where applicable.

## 15.2 Label mapping format

```json
{
  "0": "example_intent_name",
  "1": "another_intent_name"
}
```

The actual mapping must contain all 77 official Banking77 classes.

Module A owns the mapping.

## 15.3 Rare-class format

```json
{
  "selection_seed": 42,
  "rare_class_ids": [1, 5, 12]
}
```

The actual file must contain exactly 15 rare-class IDs.

## 15.4 Synthetic dataset format

```json
{
  "id": "synthetic_low_resource_42_12_001",
  "text": "My debit card seems to be missing.",
  "label": 12,
  "source": "qwen",
  "setting": "low_resource",
  "seed": 42
}
```

Required fields:

- id
- text
- label
- source
- setting
- seed

The label is the intended class supplied to Qwen.

Preserve this label even if the generated query appears incorrect.

Filtering is responsible for identifying such problems.

## 15.5 Filtering decision format

```json
{
  "id": "synthetic_low_resource_42_12_001",
  "text": "My debit card seems to be missing.",
  "label": 12,
  "setting": "low_resource",
  "seed": 42,
  "filter": "F2",
  "accepted": true,
  "predicted_label": 12,
  "confidence": 0.93,
  "rejection_reason": null
}
```

For F0 and F1, fields that are not applicable can be null.

Preserve original synthetic IDs.

Keep accepted and rejected decisions traceable.

Matched dataset files should retain the original synthetic dataset format so the shared Module D trainer can load them directly for the F0, F1 and F2 run owners.

## 15.6 Model prediction format

```json
{
  "id": "test_000001",
  "true_label": 12,
  "predicted_label": 12,
  "confidence": 0.91,
  "condition": "F2",
  "setting": "low_resource",
  "seed": 42
}
```

Validation predictions should use the same format with the appropriate example IDs and a split field.

## 15.7 Experiment metrics format

```json
{
  "condition": "F2",
  "setting": "low_resource",
  "seed": 42,
  "accuracy": 0.85,
  "macro_f1": 0.83,
  "rare_class_f1": null,
  "rare_class_recall": null,
  "real_count": 770,
  "synthetic_count": 770,
  "training_seconds": 900
}
```

Rare-class metrics can be null for the low-resource setting.

## 15.8 Computational log format

Suggested fields:

```text
run_id
stage
condition
setting
seed
model_name
model_revision
gpu_name
start_time
end_time
wall_time_seconds
gpu_allocated_seconds
measured_gpu_seconds
input_tokens
output_tokens
raw_example_count
parsed_example_count
accepted_example_count
matched_example_count
code_commit
```

Use null for unavailable or non-applicable fields.

Do not invent GPU measurements.

If only elapsed GPU allocation time is available, label it accurately.

Each run should produce a separate log file.

Avoid multiple members writing to the same CSV file simultaneously.

## 15.9 Naming conventions

Use the following exact setting names:

```text
low_resource
imbalanced
```

Use the following exact condition names:

```text
B1
B2
B3
F0
F1
F2
```

Use integer seeds:

```text
42
123
456
```

Recommended filename:

```text
seed_42.jsonl
```

Recommended unique run ID:

```text
low_resource_seed_42_F2
```

Every output must be traceable to its setting, seed and condition.

---

# 16. Shared Experimental Configuration

Maintain one common configuration file:

configs/experiment.yaml

Use the following initial configuration.

```yaml
dataset:
  name: banking77
  num_classes: 77
  validation_per_class: 10
  validation_seed: 42

seeds:
  - 42
  - 123
  - 456

settings:
  low_resource:
    samples_per_class: 10

  imbalanced:
    rare_class_seed: 42
    num_rare_classes: 15
    rare_samples_per_class: 5
    majority_samples_per_class: 20

generation:
  model: Qwen/Qwen2.5-1.5B-Instruct
  few_shot_examples: 3
  temperature: 0.7
  top_p: 0.9
  do_sample: true
  raw_budget_per_class: 50
  queries_per_prompt: 10
  max_new_tokens: 512
  max_attempts_per_class: 10

filtering:
  embedding_model: sentence-transformers/all-MiniLM-L6-v2
  duplicate_threshold: 0.95
  confidence_threshold_candidates:
    - 0.3
    - 0.5
    - 0.7
    - 0.9
  match_counts_per_class: true

training:
  model: distilbert-base-uncased
  num_labels: 77
  model_revision: null  # Pin the exact pretrained revision before the pilot.
  full_fine_tuning: true
  optimizer: adamw
  learning_rate: null  # Fill from the pilot before launching the main runs.
  batch_size: 16
  max_epochs: 30
  validation_interval_epochs: 1
  early_stopping_patience: 5
  max_length: 128
  weight_decay: 0.01
  lr_scheduler: linear
  warmup_ratio: 0.10
  max_grad_norm: 1.0
  precision: float32
  checkpoint_metric: macro_f1
  restore_best_checkpoint: true

classifier_pilot:
  development_seed: 42
  settings: [low_resource, imbalanced]
  conditions: [B1, B3]
  learning_rate_candidates: [0.00002, 0.00005, 0.0001]

baselines:
  B2: random_oversampling
  B3: easy_data_augmentation
  eda_alpha: 0.1

augmentation:
  primary_ratio: 1.0
  optional_ratios:
    - 0.5
    - 2.0

evaluation:
  primary_metric: macro_f1
  human_audit_size: 200
```

## Shared classifier pilot and configuration freeze

Manikanta runs the pilot with Gnaneswarudu’s common trainer. Use development seed 42, the released seed-42 training artifacts for each setting, and the fixed validation split. Prepare the B3 EDA data once per setting and reuse it across learning-rate candidates. Compare 2e-5, 5e-5 and 1e-4 on B1 and B3 in both settings, for 12 fits in total.

Validate after every epoch. Stop after five consecutive epochs without a strict improvement in validation macro-F1, or at 30 epochs. Retain the earlier checkpoint on an exact tie and restore the best checkpoint for predictions. Use linear decay with 10% warmup based on the planned optimizer steps for the epoch cap. Record actual optimizer steps and epochs. Review validation learning curves for cap-limited fits that are still improving. If the team changes the ceiling, record the change and repeat the learning-rate comparison under the revised policy.

For each candidate, average the best-checkpoint validation macro-F1 from the four setting/condition combinations with equal weights. Select the rate with the highest mean, using the lower rate on an exact tie. Keep test data out of the pilot and the freeze decision.

Commit the selected rate and pinned model revision to `configs/experiment.yaml` before the main B1/B2/B3/F0/F1/F2 runs. Everyone loads that same configuration revision, official split artifacts and fixed rare-class IDs. Keep the experiment seeds 42, 123 and 456. Save the pilot results separately from the core result matrix. Rerun any main experiment that differs from the frozen protocol.

---

# 17. GitHub Repository Structure

Inspect the existing repository before making changes.

Preserve existing working code.

The following is the suggested target structure, not a requirement to create every file immediately.

```text
CS682-Synthetic-Data-Project/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── experiment.yaml
│   └── label_mapping.json
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
│   ├── __init__.py
│   ├── data_preparation.py
│   ├── baselines.py
│   ├── generation.py
│   ├── prompts.py
│   ├── filtering.py
│   ├── cost_analysis.py
│   ├── training.py
│   ├── inference.py
│   ├── evaluation.py
│   └── visualization.py
│
├── tests/
│   ├── fixtures/
│   ├── test_data.py
│   ├── test_generation.py
│   ├── test_filtering.py
│   ├── test_training.py
│   └── test_evaluation.py
│
├── docs/
│   ├── data_contracts.md
│   ├── experiment_protocol.md
│   └── literature_review.md
│
└── results/
    ├── metrics/
    └── figures/
```

Keep reusable implementation logic in Python files.

Use Colab notebooks to execute experiments and display results.

Do not duplicate large amounts of code across notebooks.

Avoid creating unnecessary abstractions or additional services.

A small, readable implementation is preferable to an overengineered architecture.

---

# 18. GitHub Collaboration Workflow

Each member uses a separate working branch. Manikanta owns the PR for his existing baseline code and can reuse its branch after verifying the repository state.

| Member | Branch |
| --- | --- |
| Manikanta | Existing baseline PR branch, or `feature/data-baselines` for integration |
| Danush | feature/generation |
| Khushboo | feature/filtering |
| Gnaneswarudu | feature/classifier-training |
| Rufina | feature/evaluation |

Repository owner should provide collaborator access to all members.

Example setup:

```bash
git clone https://github.com/khushboopandey6/CS682-Synthetic-Data-Project.git

cd CS682-Synthetic-Data-Project

git checkout -b feature/generation
```

After implementation:

```bash
git add src/generation.py src/prompts.py

git commit -m "Implement Qwen synthetic generation"

git push -u origin feature/generation
```

Create a pull request for review.

## Collaboration rules

- Do not overwrite another member's work.
- Avoid directly pushing unreviewed changes to main.
- Keep commits small and meaningful.
- Review changes to shared interfaces together.
- Run relevant tests before merging.
- Do not commit credentials or access tokens.
- Do not commit large model checkpoints.
- Record the Git commit used for final experiments.
- Use the same reviewed training code for all final classifier runs.

Each member should document how to run their module.

---

# 19. Google Colab Execution Environment

Each team member uses their own Google Colab runtime.

We share code and experimental outputs, not GPU runtimes.

Recommended hardware:

| Module | Runtime |
| --- | --- |
| A — Data preparation | CPU |
| B — Qwen generation | GPU |
| C — Filtering | CPU or GPU |
| D — Classifier training | GPU |
| E — Evaluation | CPU |

Free Colab GPU availability is not guaranteed.

Use Kaggle as a backup if necessary.

## Clone repository

```python
!git clone https://github.com/khushboopandey6/CS682-Synthetic-Data-Project.git

%cd CS682-Synthetic-Data-Project
```

Check out the assigned branch when developing.

## Install dependencies

```python
!pip install -r requirements.txt
```

If requirements.txt does not yet exist, coordinate its creation.

## Mount Google Drive

```python
from google.colab import drive

drive.mount("/content/drive")
```

## Check GPU availability

```python
import torch

print(torch.cuda.is_available())

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

Additional GPU information:

```python
!nvidia-smi
```

## Colab limitations

- GPU availability is variable.
- Sessions can disconnect.
- Local runtime storage is temporary.
- GPU usage quotas are not guaranteed.
- Different members may receive different GPU types.

Therefore:

- Save outputs regularly.
- Implement resume functionality.
- Record GPU type.
- Use manageable batch sizes.
- Avoid keeping unnecessary models loaded simultaneously.
- Persist important artifacts to Google Drive.

---

# 20. Shared Google Drive Storage

Create a shared Google Drive folder named:

CS682_Project

Provide all team members with appropriate access.

Suggested structure:

```text
CS682_Project/
│
├── 01_data/
│   ├── low_resource/
│   ├── imbalanced/
│   ├── baselines/
│   │   ├── B2/
│   │   └── B3/
│   ├── validation.jsonl
│   ├── test.jsonl
│   ├── label_mapping.json
│   ├── rare_classes.json
│   └── dataset_statistics.json
│
├── 02_synthetic_raw/
│   ├── low_resource/
│   ├── imbalanced/
│   ├── raw_responses/
│   └── generation_logs/
│
├── 03_filtered_data/
│   ├── F0/
│   ├── F1/
│   ├── F2/
│   ├── matched/
│   ├── filtering_decisions/
│   └── acceptance_statistics/
│
├── 04_checkpoints/
│   ├── B1/
│   ├── B2/
│   ├── B3/
│   ├── F0/
│   ├── F1/
│   └── F2/
│
├── 05_results/
│   ├── predictions/
│   ├── validation_predictions/
│   ├── metrics/
│   ├── figures/
│   ├── human_audit/
│   └── comparison_tables/
│
├── 06_cost_logs/
│   ├── generation/
│   ├── filtering/
│   └── training/
│
└── 07_deliverables/
    ├── poster/
    ├── presentation/
    └── final_report/
```

Suggested base path:

```python
BASE_PATH = "/content/drive/MyDrive/CS682_Project"
```

The actual path may differ depending on how each member accesses the shared folder.

Make the path configurable.

Verify shared-folder access before starting expensive experiments.

Each member should write to their assigned output directory.

Use separate files for each setting, seed and condition.

Do not overwrite another member's outputs.

## Store in GitHub

- Python source code.
- Notebooks.
- Configuration files.
- Tests.
- Documentation.
- Small fixtures.
- Final summary metrics.
- Final figures.

## Store in Google Drive

- Full datasets.
- Raw synthetic outputs.
- Filtered datasets.
- Model checkpoints.
- Prediction files.
- Runtime logs.
- Human audit files.
- Poster and report files.

Do not store gigabytes of model weights in GitHub.

---

# 21. Training Execution Distribution

Module D owns the common classifier training framework.

| Member | Conditions or handoff | Result entries | Independent runs |
| --- | --- | ---: | ---: |
| Manikanta | B1, B2 and B3, six entries each | 18 | 15 with all three B2 reuses, otherwise 18 if run separately |
| Danush | Six raw pools, generation pilot and logs | 0 | 0 |
| Khushboo | Filters and matches all six pools, trains F1 | 6 | 6 |
| Gnaneswarudu | Maintains common trainer, trains F0 and F2 | 12 | 12 |
| Rufina | Verifies and aggregates results, analyzes cost | 0 | 0 |
| **Total** | **Six conditions across two settings and three seeds** | **36** | **33 with all three reuses, otherwise 36 if run separately** |

Reuse a balanced B2 result only if its training inputs and configuration match the corresponding B1 run exactly. Keep a separate B2 result entry and record the B1 run it references. If eligibility differs by seed, the independent run count increases by one for each B2 entry that needs its own run.

The 12 classifier pilot fits are separate from the core counts above. All main runs use the same reviewed trainer and frozen configuration from Section 16.

Record:

- Experimental condition.
- Setting.
- Seed.
- Real and augmentation counts.
- Code commit, frozen configuration revision and split artifact identifiers.
- Pretrained model revision.
- Actual epochs, best-checkpoint epoch and actual optimizer steps.
- Precision, hardware and runtime.

---

# 22. Computational Cost Analysis

Rufina owns the computational cost analysis using logs from generation (Danush), filtering (Khushboo) and training (Manikanta, Khushboo and Gnaneswarudu). Although actual expenditure should be zero, the project estimates the equivalent computational cost of generation and filtering.

Record:

- Generation runtime.
- Filtering runtime.
- Classifier training runtime.
- GPU type.
- GPU allocation time or measured GPU time.
- Input token count.
- Output token count.
- Raw generation count.
- Parsed example count.
- Accepted example count.
- Matched example count.

## Equivalent GPU cost

```text
Equivalent cost =
GPU hours × assumed hourly GPU rental rate
```

## Cost per 1,000 accepted examples

```text
Cost per 1,000 accepted examples =
(Generation cost + Filtering cost)
÷ Accepted example count
× 1,000
```

Include the generation cost of rejected examples.

If accepted count is zero, report the cost-per-accepted-example metric as undefined rather than zero.

For F2, report the B1 classifier training cost separately.

Explain whether B1 is treated as:

- An already-required shared baseline cost.
- An additional setup cost required for classifier filtering.

Report both interpretations where useful.

## Cost reporting

Report:

1. Generation cost.
2. Filtering cost.
3. Generation plus filtering cost.
4. Downstream classifier training cost.
5. Total pipeline cost.
6. Cost per 1,000 accepted examples.
7. Cost per 1,000 matched training examples, where useful.
8. Macro-F1 improvement relative to B1.
9. Performance versus computational cost.

Use a documented GPU rental rate with a source and access date.

Do not invent a GPU rental price.

If experiments use different GPU models, report hardware differences.

Do not present estimated equivalent cost as an actual bill.

If a CPU-only operation is not monetized, report its runtime and state that the monetary estimate excludes that CPU cost.

The actual target expenditure remains $0.

---

# 23. Human Evaluation

Manually audit approximately 200 unique synthetic examples.

Module E coordinates the audit.

All five members participate.

Each member initially annotates approximately 40 examples.

Use a reproducible sampling strategy covering:

- Low-resource setting.
- Imbalanced setting.
- Different intended classes.
- Examples accepted by F1.
- Examples rejected by F1.
- Examples accepted by F2.
- Examples rejected by F2.

Use original synthetic IDs to avoid counting the same example multiple times.

## Annotation rubric

For each generated query, record:

- Intended intent.
- Generated query.
- Correct intended label.
- Incorrect intended label.
- Ambiguous or unclear.
- Optional notes about label drift or unrealistic wording.

Where practical, hide the filtering decision from the annotator until after annotation to reduce bias.

Use a small overlapping annotation subset to check agreement if time permits.

If sampling deliberately balances accepted and rejected examples, report quality separately by sampling group or use appropriate weighting for overall estimates.

Do not treat an artificially balanced audit sample as a representative random sample without qualification.

## Human audit objectives

- Estimate raw synthetic label correctness.
- Estimate correctness among accepted examples.
- Investigate whether filters reject useful examples.
- Identify label drift.
- Analyze diversity-quality trade-offs.
- Provide qualitative examples for the poster and final report.

Do not use the final human audit to silently alter experimental datasets.

---

# 24. Evaluation and Final Results

For each condition and setting, aggregate the results across the three experiment seeds.

Report mean and standard deviation.

Suggested final results table:

| Condition | Low-resource Macro-F1 | Imbalanced Macro-F1 | Rare-class F1 |
| --- | --- | --- | --- |
| B1 | Measured result | Measured result | Measured result |
| B2 | Measured result | Measured result | Measured result |
| B3 | Measured result | Measured result | Measured result |
| F0 | Measured result | Measured result | Measured result |
| F1 | Measured result | Measured result | Measured result |
| F2 | Measured result | Measured result | Measured result |

Additional tables should include:

- Accuracy.
- Rare-class recall.
- Actual augmentation counts.
- Acceptance rates.
- Diversity metrics.
- Human-evaluated label correctness.
- Computational cost.

## Recommended figures

1. Macro-F1 comparison across conditions.
2. Rare-class F1 comparison.
3. Filtering acceptance rates.
4. Synthetic diversity comparison.
5. Performance versus equivalent computational cost.
6. Selected confusion matrices.
7. Quantity versus performance, if RQ2 experiments are completed.

Use actual experimental measurements.

Do not fabricate results or fill missing values with estimates.

## Interpretation questions

- Does F1 outperform F0?
- Does F2 outperform F0?
- Does filtering improve rare-class performance?
- Does filtering reduce diversity?
- Does F2 reject difficult but correctly labeled examples?
- Are some intents disproportionately rejected?
- Does filtering provide enough improvement to justify its cost?
- Are improvements consistent across seeds?
- How do traditional augmentation baselines compare?
- What limitations prevent broader conclusions?

Three seeds provide a limited estimate of variability.

Do not claim statistical significance without an appropriate analysis.

---

# 25. Professor's Feedback

Our project proposal received 10/10.

However, the professor provided important feedback for the final report.

The expectations are:

- Elaborate prior research in detail.
- Do not simply list relevant papers.
- Bring the reader up to speed on the current state of research.
- Connect prior research to our specific research questions.
- Clarify the research gap.
- Compare our findings with previous research when applicable.
- Ensure fair experimental comparisons.
- Demonstrate sufficient research depth for a five-person team.
- Ensure the manuscript reads naturally.

The final report should not merely describe our code.

It should explain:

- Why the research problem matters.
- What previous researchers have discovered.
- What remains uncertain.
- How our experiment investigates that uncertainty.
- What our findings demonstrate.
- What limitations remain.

---

# 26. Related Work and Research Gap

The proposal includes the following research directions.

## Anaby-Tavor et al. — Do Not Have Enough Data? Deep Learning to the Rescue!

Relevant to generating additional labeled text examples and using classifier-based selection.

This is particularly relevant to our F2 filtering condition.

## Casanueva et al. — Banking77

Provides the banking intent classification dataset and task context.

## Kumar et al. — Data Augmentation Using Pre-trained Transformer Models

Relevant to transformer-based augmentation, class-conditioned generation and label preservation.

## Kwon et al. — vLLM and PagedAttention

Relevant to efficient LLM inference.

We are not required to implement vLLM in the core experiment.

## Li et al. — Synthetic Data Generation with Large Language Models for Text Classification: Potential and Limitations

Relevant to understanding when synthetic text augmentation helps or fails.

## Meng et al. — SuperGen

Relevant to synthetic data generation and example selection.

## Qwen2.5 Technical Report

Provides background on the selected generation model.

## Sanh et al. — DistilBERT

Provides background on the selected downstream classifier.

## Shumailov et al. — AI models collapse when trained on recursively generated data

Provides motivation for investigating synthetic data quality and diversity.

Our experiment is not a direct test of recursive model collapse.

## Wei and Zou — Easy Data Augmentation

Provides the traditional augmentation baseline.

## Ye et al. — ZeroGen

Relevant to synthetic training data generation and efficient classification.

## Yoo et al. — GPT3Mix

Relevant to LLM-based text data augmentation.

## Proposed research gap

Previous research has explored LLM-generated synthetic data, data augmentation and filtering.

Our project focuses on a controlled comparison of:

- Unfiltered synthetic data.
- Semantic deduplication.
- Classifier-based label agreement.

We compare these approaches under both low-resource and imbalanced classification settings.

We control synthetic data quantity across filtering methods and explicitly analyze:

- Classification performance.
- Rare-class performance.
- Label correctness.
- Diversity.
- Acceptance rates.
- Computational cost.

Do not claim that no previous research has investigated these ideas unless the literature review establishes that claim.

Our contribution is the controlled experimental comparison and cost-aware analysis within the selected setting.

## Literature review requirements

For each assigned paper, explain:

1. Research problem.
2. Methodology.
3. Dataset and experimental setup.
4. Main findings.
5. Limitations.
6. Relationship to our project.
7. Whether a fair comparison is possible.

Read and verify original papers before citing them.

Do not invent paper findings or numerical results.

If prior work uses different datasets, models or experimental budgets, explain those differences.

Do not claim direct superiority using incomparable experimental settings.

---

# 27. Poster Requirements

Official poster submission deadline: December 1.

Poster dimensions: 2×3 feet.

Recommended orientation: Landscape, 36 inches wide × 24 inches high.

Prepare the poster as a print-ready PDF unless the professor specifies another format.

The professor is investigating printing support and will provide a Google Drive submission link.

## Poster objective

Communicate the research problem, methodology, experimental design and genuine preliminary findings clearly.

The poster does not need to contain every final experimental result.

The project can continue after poster submission.

## Suggested poster layout

Use three columns.

### Left column

- Motivation.
- Research problem.
- Related work.
- Research gap.
- Research questions.
- Dataset.
- Low-resource and imbalanced settings.

### Middle column

- Overall methodology diagram.
- Qwen generation.
- F0, F1 and F2 filtering.
- Matched-count comparison.
- DistilBERT training.
- Experimental conditions.

### Right column

- Preliminary experimental results.
- Filtering acceptance rates.
- Computational cost.
- Preliminary conclusions.
- Ongoing work.
- References.

Use readable fonts, clear figures and minimal text.

Do not overcrowd the poster.

## Minimum target before poster submission

Aim to have:

- Official dataset preparation completed.
- Qwen generation pilot completed.
- First B1 classifier trained.
- F0, F1 and F2 implemented.
- One matched-count experiment completed.
- Preliminary macro-F1 results.
- Preliminary acceptance rates.
- Preliminary runtime measurements.
- Research methodology diagram.

If some results are unavailable, mark them as ongoing work.

Do not fabricate results to fill the poster.

Do not present single-seed preliminary results as three-seed averages.

## Poster responsibilities

| Member | Poster content |
| --- | --- |
| Manikanta | Dataset, experimental settings and traditional baselines |
| Danush | Qwen generation methodology |
| Khushboo | Filtering, matching and acceptance rates |
| Gnaneswarudu | DistilBERT and experimental conditions |
| Rufina | Motivation, research gap, results, cost analysis and assembly |

All members must review the poster before submission.

Poster assembly coordinator: Rufina.

## Poster timeline

November 27:

- Prepare layout and methodology diagram.

November 28:

- Obtain first end-to-end results.

November 29:

- Complete initial poster draft.

November 30:

- Verify all results.
- Review readability.
- Verify dimensions.
- Review references.
- Obtain team feedback.
- Finalize poster.

December 1:

- Submit the poster through the professor's provided link.

---

# 28. Detailed Day-by-Day Execution Plan

## November 25 — Shared Setup

All members:

- Confirm assigned modules.
- Obtain GitHub collaborator access.
- Create feature branches.
- Create shared Google Drive folder.
- Establish shared data contracts.
- Establish common configuration.
- Inspect existing repository progress.
- Begin independent development.

Do not spend excessive time planning infrastructure.

## November 26 — Independent Development

Module A:

- Complete official data splits.
- Export validation and test datasets.
- Export label mapping and rare-class IDs.
- Begin baseline preparation.

Module B:

- Load Qwen.
- Implement prompts.
- Implement generation.
- Implement output parsing.
- Implement checkpointing.

Module C:

- Implement F0.
- Implement F1.
- Implement F2 using mock predictions.
- Implement matched-count logic.

Module D:

- Implement reusable training framework.
- Test tokenization and classifier training.

Module E:

- Implement evaluation metrics.
- Begin literature review.
- Prepare poster outline.

## November 27 — Pilot and Baseline

Module A:

- Verify official datasets.
- Complete B2 and B3 preparation.
- Run the shared 12-fit learning-rate pilot and freeze the policy with Gnaneswarudu. Then train the official low-resource seed-42 B1 and export its checkpoint and validation predictions.

Module B:

- Complete three-intent generation pilot.
- Measure throughput.
- Finalize practical generation budget.
- Begin low-resource seed 42 generation.

Module C:

- Complete filtering tests.
- Prepare first official filtering run.

Module D:

- Support the classifier pilot and configuration freeze, then verify that Module A’s first official B1 run produces the artifacts required by Module C.
- Prepare F0 and F2 run configurations.

Module E:

- Verify evaluation metrics.
- Prepare poster figures and results templates.

## November 28 — First Complete End-to-End Experiment

Complete low-resource seed 42 through:

1. Real training dataset.
2. Qwen synthetic generation.
3. B1 classifier training.
4. F0 filtering.
5. F1 filtering.
6. F2 filtering.
7. Matched-count datasets.
8. Downstream classifier training.
9. Evaluation.

Aim to obtain B1, F0, F1 and F2 results. Manikanta publishes the data and B1 artifacts, Danush publishes the raw pool and generation logs, Khushboo publishes matched datasets and trains F1, Gnaneswarudu trains F0 and F2, and Rufina evaluates the results and collects cost inputs.

Fix integration problems before scaling.

## November 29 — Preliminary Results and Poster Draft

- Continue remaining generation pools.
- Manikanta continues B1, B2 and B3 training.
- Khushboo continues filtering, matching and F1 training.
- Gnaneswarudu continues F0 and F2 training as matched datasets arrive.
- Produce preliminary macro-F1 results.
- Produce acceptance-rate statistics.
- Produce runtime measurements.
- Complete poster methodology diagram.
- Complete initial poster draft.

## November 30 — Poster Finalization

- Verify preliminary experimental results.
- Clearly label completed settings and seeds.
- Review research motivation.
- Review related work.
- Verify poster dimensions.
- Review visual readability.
- Obtain feedback from all members.
- Finalize poster.

Continue experiments in parallel.

## December 1 — Poster Submission

Submit the 2×3 ft poster.

Continue experimental execution.

Do not wait for all 36 core results before submitting the poster.

## December 2 — Generation and Filtering Completion

- Complete all six raw synthetic pools.
- Complete F0, F1 and F2 filtering.
- Complete matched-count datasets.
- Verify data consistency.
- Continue downstream classifier training.

## December 3 — Core Training and Human Audit

- Complete majority of remaining classifier runs.
- Aggregate available results.
- Begin or continue human audit.
- Analyze rare-class performance.
- Begin final cost comparison.

## December 4 — Final Experimental Analysis

- Complete remaining mandatory runs.
- Complete human audit.
- Complete diversity analysis.
- Complete computational cost analysis.
- Produce final comparison tables.
- Produce final figures.
- Attempt quantity experiments if resources permit.

## December 5 — Internal Project Completion

Target completion of:

- All core experimental conditions.
- Both experimental settings.
- Three experiment seeds.
- Final classification metrics.
- Rare-class analysis.
- Filtering acceptance rates.
- Human audit.
- Diversity analysis.
- Computational cost analysis.
- Final figures.
- Reproducibility documentation.

Freeze the main experimental results.

Document any incomplete optional experiments or unavoidable missing runs.

## December 6 — Presentation Preparation

- Prepare presentation narrative.
- Update results beyond the submitted poster where appropriate.
- Assign presentation responsibilities.
- Review research questions.
- Review methodology.
- Review findings and limitations.

## December 7 — Presentation Rehearsal

- Conduct a complete mock presentation.
- Practice anticipated professor questions.
- Verify reported numbers.
- Review literature comparisons.
- Ensure every member understands the complete project.

## December 8 — Official Presentation

Present the completed project.

If final results differ from preliminary poster results, explain the difference transparently.

## December 9–14 — Final Report

- Complete literature review.
- Write methodology.
- Write experimental results.
- Compare with prior research.
- Write discussion.
- Write limitations.
- Document reproducibility.
- Verify citations.
- Review manuscript flow.
- Complete proofreading.

## December 15 — Final Submission

Submit the final report.

---

# 29. Final Report Structure

Recommended report organization:

## Abstract

Summarize the research problem, methodology, principal findings and conclusion.

Write the abstract after the main experimental results are available.

## Introduction

Explain:

- Low-resource classification.
- Imbalanced classification.
- Synthetic data augmentation.
- Synthetic data quality problems.
- Research motivation.
- Research questions.
- Project contributions.

## Related Work

Discuss prior research in detail.

Organize by research themes rather than simply listing papers.

Explain the connection between prior work and our experiment.

## Dataset and Experimental Setup

Describe:

- Banking77.
- Validation split.
- Test split.
- Low-resource setting.
- Imbalanced setting.
- Random seeds.
- Class distributions.
- Experimental conditions.

## Methodology

Describe:

- Qwen generation.
- Prompting.
- Generation parameters.
- F0.
- F1.
- F2.
- Matched-count comparison.
- Traditional baselines.
- DistilBERT training.
- Computational cost methodology.

## Results

Present:

- Macro-F1.
- Accuracy.
- Rare-class performance.
- Acceptance rates.
- Diversity.
- Human evaluation.
- Computational cost.
- Quantity experiments, if completed.

## Discussion

Interpret the results.

Explain why filtering may help or fail.

Compare with prior research.

Discuss the trade-off between correctness, diversity and computational cost.

## Limitations

Discuss:

- Single dataset.
- English-only evaluation.
- Small number of seeds.
- Artificially constructed class imbalance.
- Additional validation supervision.
- Possible LLM exposure to Banking77 during pretraining.
- Classifier filtering bias.
- Reduced synthetic data coverage.
- Computational resource limitations.
- Incomplete optional experiments, if applicable.

## Conclusion

Answer the research questions using actual findings.

Avoid unsupported generalizations.

## References

Include verified academic references.

## Reproducibility and Contributions

Document:

- Repository.
- Model versions.
- Configuration.
- Dataset splits.
- Seeds.
- Execution instructions.
- Team contributions.

All five members contribute to the report and audit. Manikanta writes data preparation and baselines, Danush writes generation, Khushboo writes filtering and matching, Gnaneswarudu writes shared classifier training, and Rufina coordinates evaluation, cost, audit findings and discussion. Each member contributes relevant related work.

One final editor should ensure that the report reads as one coherent manuscript rather than five disconnected sections.

---

# 30. Important Research Rules

1. Never use test data for training or model selection.
2. Never use validation or test examples in generation prompts.
3. Preserve the official Banking77 label mapping.
4. Use the same raw synthetic pool across F0, F1 and F2.
5. Control synthetic data quantity in the primary filtering comparison.
6. Record actual augmentation counts.
7. Use the same classifier training framework across conditions.
8. Do not continue training from another experimental condition's checkpoint.
9. Do not fabricate metrics or experimental outputs.
10. Do not fabricate literature findings.
11. Do not assume filtering must improve performance.
12. Do not silently change shared hyperparameters.
13. Do not unnecessarily introduce paid services.
14. Do not train Qwen from scratch.
15. Do not wait for upstream modules when mock data can be used.
16. Save intermediate outputs regularly.
17. Keep experiments reproducible.
18. Record computational hardware.
19. Clearly distinguish pilot results from final results.
20. Clearly distinguish estimated cost from actual expenditure.
21. Document deviations from the approved proposal.
22. Prioritize mandatory experiments over optional extensions.
23. Do not miss the poster deadline while waiting for final results.
24. Do not introduce unnecessary project complexity.

---

# 31. Risk Management

## Risk 1: Free Colab GPU unavailable

Solution:

- Use Kaggle as backup.
- Coordinate GPU availability with the assigned run owners in Section 21.
- Execute CPU-compatible tasks while waiting.
- Prioritize the first complete experiment.

## Risk 2: Generation takes too long

Solution:

- Benchmark throughput during the pilot.
- Use batching where practical.
- Keep prompts short.
- Use checkpointing.
- Reduce the common generation budget before final runs if necessary.

## Risk 3: Poor synthetic data quality

Solution:

- Improve prompts.
- Adjust decoding parameters minimally.
- Repeat the pilot.
- Document the limitations.

Do not automatically introduce a larger model.

## Risk 4: F2 accepts very few examples

Solution:

- Inspect validation performance.
- Verify label mapping.
- Verify confidence calculation.
- Apply the documented validation-based threshold selection policy.
- Report low acceptance if it reflects the actual classifier behavior.

Do not manipulate thresholds using test results.

## Risk 5: Integration problems

Solution:

- Follow shared data contracts.
- Test with small fixtures.
- Complete one end-to-end experiment early.
- Release outputs progressively.
- Coordinate shared interface changes.

## Risk 6: Poster deadline arrives before final results

Solution:

- Submit verified preliminary results.
- Clearly label ongoing experiments.
- Include methodology and research questions.
- Continue experiments after submission.

## Risk 7: Core experiments are incomplete by December 5

Solution:

- Coordinate available compute with the assigned run owners and prioritize B1 and the matched F0/F1/F2 comparisons.
- Deprioritize optional extensions.
- Preserve both experimental settings.
- Complete as many planned seeds and conditions as possible.
- Report missing experiments transparently.

Do not invent missing results.

---

# 32. Instructions for the LLM or Coding Assistant Receiving This Document

You are assisting a member of a five-person CS682 Neural Networks research project at UMass Amherst.

This document provides the complete project context, experimental methodology, technical decisions, module responsibilities, data contracts and deadlines.

The team member will identify their name and assigned module.

Act as a technical mentor, machine learning research assistant and implementation partner.

Assume the member may be unfamiliar with the relevant machine learning concepts.

## Your responsibilities

1. Understand the complete project before proposing implementation.
2. Identify the member's assigned module.
3. Explain the module from first principles.
4. Explain relevant machine learning concepts in beginner-friendly language.
5. Inspect the existing GitHub repository if access is available.
6. Identify existing working implementations.
7. Avoid recreating existing working code.
8. Identify the member's specific responsibilities.
9. Identify tasks that can start immediately.
10. Identify dependencies on other modules.
11. Develop using mock data or temporary training subsets where necessary.
12. Create a prioritized implementation checklist.
13. Create a day-by-day execution plan.
14. Implement simple, modular and reusable Python code.
15. Ensure Google Colab compatibility.
16. Follow the shared experimental configuration.
17. Follow the shared data contracts.
18. Implement essential tests.
19. Provide execution and validation instructions.
20. Implement checkpointing where necessary.
21. Document assumptions and implementation decisions.
22. Provide integration and handover instructions.
23. Help the member understand relevant academic papers.
24. Help prepare the member's poster contribution.
25. Help prepare the member's final report contribution.
26. Identify limitations and possible experimental biases.
27. Never fabricate results or claim code was executed when it was not.

## Important restrictions

Do not attempt to implement the entire project when the member is responsible for one module.

Do not unnecessarily increase the project scope.

Do not introduce:

- Additional datasets.
- Unnecessary models.
- Paid infrastructure.
- Frontend or backend applications.
- Complex orchestration frameworks.
- Extensive hyperparameter searches.

Prioritize the existing approved research scope.

Use the technical decisions in this document as the default implementation specification.

If a genuine technical problem requires a change, explain:

1. What the problem is.
2. Why it matters.
3. The simplest proposed solution.
4. Which modules are affected.
5. Whether the change must be communicated to the team.

Do not silently change shared interfaces or experimental methodology.

Do not treat suggested starting hyperparameters as experimentally proven optimal values.

Do not claim to have inspected files, executed code or verified results unless you actually did so.

## Deadline priorities

- November 26: Official dataset splits.
- November 28: First end-to-end experiment.
- December 1: Submit the 2×3 ft poster.
- December 5: Core project completion.
- December 8: Presentation.
- December 15: Final report.

Prioritize correctness, fair comparisons, reproducibility and timely completion over unnecessary engineering complexity.

## Initial response expected from the LLM

After the member identifies themselves, provide:

1. A beginner-friendly explanation of the entire project.
2. A focused explanation of their assigned module.
3. Their exact responsibilities.
4. Existing repository progress, if accessible.
5. Tasks they can start immediately.
6. Dependencies on other modules.
7. Required inputs and outputs.
8. A prioritized implementation checklist.
9. A day-by-day plan aligned with deadlines.
10. Development, testing and Colab execution guidance.
11. Integration and handover instructions.

Guide the member through implementation in manageable stages.

---

# 33. Individual Initialization Prompts

After pasting this complete document into your LLM, use the appropriate prompt below.

## Manikanta

I am Dundi Manikanta Ganneni, responsible for Module A. Help me release the official splits by November 26, submit or update the PR for my existing baseline code, and run all B1/B2/B3 experiments using the shared trainer. Help me complete the shared 12-fit validation pilot and freeze the training policy, then publish the first official B1 checkpoint and validation predictions by November 27, then repeat for the other five setting–seed combinations. Document any balanced B2 reuse and export baseline results and logs.

## Danush

I am Danush Kanchi, responsible for Module B. Help me set up Qwen in Colab, run the three-intent pilot, implement prompt construction, parsing and resumability, and publish all six raw synthetic pools with generation logs. Prioritize the low-resource seed-42 pool for the November 28 end-to-end milestone. My classifier run count is zero.

## Khushboo

I am Khushboo Pandey, responsible for Module C. Help me develop F0, F1, F2 and matched-count sampling with mock data, then process all six official pools. Use Manikanta’s B1 checkpoints and validation predictions for F2, hand off matched F0 and F2 datasets to Gnaneswarudu, and use the shared trainer for my six F1 runs. Export filtering decisions, acceptance statistics and runtime logs for Rufina.

## Gnaneswarudu

I am Gnaneswarudu Kuna, responsible for Module D. Help me develop and test the common DistilBERT trainer for all six conditions. Prioritize enabling Manikanta’s first B1 run by November 27. Help me execute six F0 and six F2 runs from Khushboo’s matched datasets and publish predictions, metrics and training logs.

## Rufina

I am Rufina Lourdes Rajesh, responsible for Module E. Help me build evaluation with mock predictions and logs, aggregate performance and diversity results, analyze computational cost, coordinate the human audit and assemble the December 1 poster. Help me prepare final figures, literature comparisons and the report’s results and discussion with input from all five members.

---

# 34. Definition of Project Completion

The core project is considered complete when:

- Official dataset splits are prepared and verified.
- All six raw synthetic pools are generated.
- F0, F1 and F2 are implemented.
- Matched-count datasets are available.
- Core experimental conditions are evaluated across both settings and three seeds, or missing runs are explicitly documented.
- Macro-F1 is analyzed.
- Rare-class performance is analyzed.
- Synthetic data quality is evaluated.
- Synthetic data diversity is evaluated.
- Human evaluation is completed.
- Computational cost is analyzed.
- Prior research is discussed.
- Experimental results are reproducible.
- The poster is submitted.
- The presentation is completed.
- The final report is submitted.
- Every member understands their contribution and the overall research pipeline.

Our goal is not simply to generate synthetic data or train a classifier.

Our goal is to conduct a fair, reproducible research experiment that determines whether filtering LLM-generated synthetic data is beneficial under low-resource and imbalanced text classification settings, and whether any benefits justify the computational cost.

We should aim for a simple, well-executed and academically rigorous project rather than a complicated project that cannot be completed on time.
