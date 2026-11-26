# Classifier pilot review

All 12 declared fits completed on Apple M5 MPS in float32 using source commit
`62e01eee7b4bbfb05eda9637669f42ad1d5ed3f5`. This is the adopted protocol's pilot;
the earlier development experiments are separate. No classifier test evaluation
was used for this decision.

## Selection

Each candidate is scored by the equally weighted mean of its four best-checkpoint
validation macro-F1 values. All fits use development seed 42, the same fixed
770-example validation set, and identical training artifacts within each
setting/condition. B3 was prepared once and reused across learning rates.

| Learning rate | Low-resource B1 | Low-resource B3 | Imbalanced B1 | Imbalanced B3 | Mean |
|---|---:|---:|---:|---:|---:|
| 2e-5 | 0.766245 | 0.762380 | 0.769030 | 0.794530 | 0.773046 |
| 5e-5 | 0.776229 | 0.779295 | 0.797446 | 0.809382 | 0.790588 |
| **1e-4** | 0.771560 | 0.765282 | 0.819404 | 0.836183 | **0.798107** |

The declared rule selects **1e-4**, with a mean advantage of 0.007519 over 5e-5.
The low-resource cells individually favor 5e-5. One rate is retained across both
settings and every method, as specified in the master plan; separate rates are
not selected from the individual cells.

## Learning-curve review

| Rate | Setting / method | Best epoch | Epochs run | Review |
|---|---|---:|---:|---|
| 2e-5 | Low-resource B1 | 25 | 30 | Five subsequent epochs without strict improvement |
| 2e-5 | Low-resource B3 | 15 | 20 | Patience stop |
| 2e-5 | Imbalanced B1 | 30 | 30 | Cap-limited; new best at the last epoch |
| 2e-5 | Imbalanced B3 | 14 | 19 | Patience stop |
| 5e-5 | Low-resource B1 | 18 | 23 | Patience stop |
| 5e-5 | Low-resource B3 | 11 | 16 | Patience stop |
| 5e-5 | Imbalanced B1 | 20 | 25 | Patience stop |
| 5e-5 | Imbalanced B3 | 10 | 15 | Patience stop |
| 1e-4 | Low-resource B1 | 11 | 16 | Patience stop |
| 1e-4 | Low-resource B3 | 10 | 15 | Patience stop |
| 1e-4 | Imbalanced B1 | 13 | 18 | Patience stop |
| 1e-4 | Imbalanced B3 | 13 | 18 | Patience stop |

Retain the predeclared maximum of 30 epochs and patience of five. The selected
rate's four fits all stopped before the cap. The slower 2e-5 imbalanced B1 fit
was still improving at the cap, so this comparison establishes a selection under
the declared budget, not an unrestricted optimal learning rate. Extending only
that fit would change its scheduler and comparison policy; any future ceiling
change requires repeating the full pilot under a new documented policy.

## Supervision and cost

The fixed validation set contributes 10 additional labeled examples per intent.
Report this supervision separately from the 10 training examples per intent in
the low-resource setting. The pilot uses one development seed and does not
estimate seed-to-seed uncertainty.

Summed classifier elapsed time was 5,443.70 seconds (90.73 minutes), including
model setup, training, validation and checkpoint work within each fit. This is
elapsed time on the recorded MPS machine, not a measurement of accelerator
utilization. Pilot cost is separate from the main baseline experiment costs.

Detailed inputs, scores and provenance are in `classifier_pilot_trials.csv`,
`classifier_learning_rates.csv`, `classifier_selection.json`, and each local
pilot checkpoint's configuration, history, runtime and validation exports.
