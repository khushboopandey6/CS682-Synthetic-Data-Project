# Result files

These small CSV reports are committed to Git. All scores are validation scores.
The official test set remains reserved for final evaluation.

| Files | Contents |
|---|---|
| [baseline_validation_aggregate.csv](baseline_validation_aggregate.csv) | Start here: means and sample standard deviations for the five setting/baseline groups. |
| [baseline_validation_results.csv](baseline_validation_results.csv) | The 15 distinct completed baseline fits, one row per setting, baseline, and seed. |
| `b1_validation_*`, `b2_validation_*`, `b3_validation_*` | Individual runs and aggregate summaries for each baseline. Balanced B2 reuses B1, so no duplicate training rows are added. |
| [b1_b2_validation_comparison.csv](b1_b2_validation_comparison.csv), [b1_b3_validation_comparison.csv](b1_b3_validation_comparison.csv) | Comparisons between matching settings and seeds. Difference columns subtract B1 from B2 or B3. |
| [baseline_validation_trials.csv](baseline_validation_trials.csv), [baseline_learning_rates.csv](baseline_learning_rates.csv) | Six development trials and the learning-rate selection summary. Two selected trials also serve as the seed-42 B1 models. |
| [checkpoint_index.csv](checkpoint_index.csv) | Paths, sizes, SHA-256 hashes, and validation macro-F1 for the 15 main local checkpoints. The model files themselves are excluded from Git. |

Checkpoint paths are relative to the project root. Cloning the branch provides
the index and reports; it does not download datasets or trained model weights.
After training or restoring models, refresh the index from the project root:

```bash
python3 scripts/update_checkpoint_index.py
```

Detailed predictions, probabilities, epoch histories, EDA text, and timing logs
remain in the ignored `results/baselines/` folder. Earlier tuning and development
checkpoints are preserved there separately from the 15 main models.
