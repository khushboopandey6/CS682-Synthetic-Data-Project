# Does Filtering Help?

Cost-Aware LLM-Generated Synthetic Data for Low-Resource and Imbalanced Text Classification
COMPSCI 682 — University of Massachusetts Amherst

The project studies whether filtering Qwen-generated banking queries improves
DistilBERT intent classification, especially for rare classes, and whether the
gains justify the computational cost. The [master project document](docs/CS682-Master-Project-Document.md)
defines the team assignments and experimental design.

## Module A status

**Owner: Dundi Manikanta Ganneni.** Local preparation, baseline training and test
evaluation are complete. Transfer to the team and Git publication remain pending.

| Deliverable | Verified state |
|---|---|
| Official real-data splits and B2/B3 datasets | Prepared for seeds 42, 123, 456 |
| Original learning-rate pilot | 12 fits completed; selected 1e-4 |
| Main B1/B2/B3 experiments | 18 result entries from 15 independent model fits |
| B1 models for Module C | All six checkpoints and matching validation exports verified |
| Test evaluation | 18 verified exports, 3,080 queries each; 15 MPS evaluations and three B2 reuses |
| Three-seed summaries | Test/validation metrics, rare-class performance and timing complete |
| Git commit, push and baseline PR | Latest changes pending publication |
| Drive transfer and receiver verification | Pending |
| Module D review of the shared trainer | Pending team review |

Generation, filtering, F0/F1/F2 training and the combined project analysis belong
to the other assigned modules. Their completion is not implied by Module A's results.

## Final baseline results

Test macro-F1, **mean ± sample standard deviation across three training seeds**:

| Setting | B1: real only | B2: oversampling | B3: real + EDA |
|---|---:|---:|---:|
| Low-resource | 0.7979 ± 0.0182 | 0.7979 ± 0.0182 | 0.8055 ± 0.0101 |
| Imbalanced | 0.8206 ± 0.0188 | 0.8143 ± 0.0063 | 0.8232 ± 0.0141 |

Balanced B2 exactly reuses B1. These are descriptive three-seed results, not
claims of statistical significance. The [full summary](results/metrics/master_v1/baseline_summary.md)
includes accuracy, rare-class F1/recall, validation scores and timing. Completed
main-fit elapsed time was **103.07 minutes**; test inference took **74.79 seconds**.
These totals exclude pilots, interrupted attempts and upstream preparation costs.

## Where to find the documentation

- [Experiment protocol](docs/experiment_protocol.md): frozen settings, pilot decisions, run ownership and evaluation rules.
- [Data contracts and transfer guide](docs/data_contracts.md): dataset counts, schemas, exact artifact paths and recipient checks.
- [Literature review](docs/literature_review.md): sourced Module A BANKING77/EDA contribution; other modules' review remains pending.
- [Results guide](results/metrics/README.md): current reports, audits and historical development results.

## Git and Google Drive

| Destination | Contents |
|---|---|
| Git | Source code, notebooks, requirements, configurations, documentation, small metric tables, manifests and audit/index files |
| Team Drive: `CS682_Project/` | Released datasets, model checkpoints, detailed predictions/probabilities, histories and runtime logs |

The local artifact root is `artifacts/CS682_Project/`, excluded from Git. Keep
its relative folder structure when transferring to Drive. The actual shared
Drive path/link and recipient access still need to be confirmed. Existing local
ZIPs are optional convenience snapshots; they are not the authoritative handoff
and do not need refreshing for the Git-and-Drive workflow.

```text
configs/experiment.yaml                Shared frozen configuration
notebooks/01_data_preparation.ipynb     Preserved original notebook
notebooks/01_reproducible_data_preparation.ipynb
notebooks/02_baseline_models.ipynb
src/                                  Shared data, baseline, training and inference code
scripts/                              Execution, verification and summary entry points
results/metrics/master_v1/             Current small reports and metadata for Git
artifacts/CS682_Project/               Local source of files to transfer to Drive
  01_data/                            Official splits and prepared B2/B3 datasets
  04_checkpoints/B1/                  Six required teacher-model folders
  04_checkpoints/B2/                  Three imbalanced oversampling models
  04_checkpoints/B3/                  Six EDA models
  05_results/                        Detailed predictions, probabilities and metrics
  06_cost_logs/                      Training and test-inference logs
  handoff/                           Frozen config copy and artifact inventories
```

## Shared settings and hardware

Everyone should load [configs/experiment.yaml](configs/experiment.yaml) and the
released split files. The retained policy uses **1e-4, maximum 30 epochs, patience
5**, full DistilBERT fine-tuning, batch size 16 and float32. Gnaneswarudu owns
Module D's common framework; [src/training.py](src/training.py) provides the tested
shared-trainer candidate used by these baselines. Team review is still pending.

The general trainer supports CUDA → MPS → CPU. For this local work, further
CPU model execution was explicitly stopped; the resume wrapper requires MPS.
Low-resource seed-42 B1/B3 were completed on CPU and retained. The other 13 main
fits and all test evaluations used MPS. Actual hardware is recorded per run.
Colab setup is supported, but an actual CUDA execution check has not been completed.

## Reproduce or verify

Install [requirements.txt](requirements.txt) in the chosen Python environment.
Restore the released Drive artifacts at the configured base path. Do not
independently resample training subsets or create another label mapping.

From the project root, these commands preview plans without training or inference:

```bash
python scripts/run_classifier_pilot.py
python scripts/run_baselines.py
python scripts/evaluate_baselines.py
```

The main runs are already complete. The local convenience command below requires
MPS and reuses matching completed fits; an interrupted fit restarts from its seed:

```bash
bash scripts/resume_baselines_mps.sh
```

That wrapper defaults to the existing local Python environment. Set
`CS682_PYTHON` to another installed project environment if needed. Generic Python
entry points use automatic device selection; check the device before execution.

Refresh metadata or summaries without retraining:

```bash
python scripts/package_data_release.py --verify-only
python scripts/export_baseline_handoff.py
python scripts/summarize_baselines.py
python -m unittest discover -s tests -v
```

The CLI `--run` flags explicitly execute training or inference. Test evaluation
loads saved main checkpoints and uses the test split for reporting only; test
scores must never guide subsequent training, generation or filtering choices.

## Historical artifacts

Earlier development results used seeds **42, 43, 44**, a different rare-class
policy and a B1-only tuning search. They remain under the older metric paths and
`results/baselines/distilbert_validated/`. They are separate from `master_v1`.
The original notebook and old local datasets/checkpoints are retained. The
[data contracts](docs/data_contracts.md) document known source overlaps and the
recorded EDA/validation coincidence.
