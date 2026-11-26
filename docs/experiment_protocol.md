# Shared experiment protocol

Use [the adopted master plan](CS682-Master-Project-Document.md) and
[configs/experiment.yaml](../configs/experiment.yaml) for every team's run.
Data paths and record formats are in [data_contracts.md](data_contracts.md).

## Execution status

Official data and B2/B3 inputs are prepared locally. The classifier learning rate
remains unset until the 12-fit pilot completes. Existing historical runs do not
automatically count toward this protocol's official matrix.

The user approved local execution on November 25, 2026, followed by team sharing.
Device selection is CUDA, then MPS, then CPU. Each run records the actual device;
main training uses float32, batch size 16, input length 128, weight decay 0.01,
10% warmup, and gradient clipping at 1.0.

## Classifier pilot and freeze

Gnaneswarudu owns the common trainer. The working baseline implementation has
been extracted into `src/training.py` as a reusable integration candidate for
review. The existing training math, dynamic padding, fresh initialization, and
validation checkpoint selection are retained.

Manikanta uses this common function for the pilot and B1/B2/B3; Khushboo uses it
for F1, and Gnaneswarudu uses it for F0/F2. The trainer is shared code, not one
trained checkpoint reused across the six methods. Each classifier starts from
the pinned pretrained DistilBERT revision. The B1 checkpoint is used separately
as the F2 filtering teacher, not to initialize downstream F2 training.

Preview without training:

```bash
python scripts/run_classifier_pilot.py
```

After verifying the configured local or shared output folder, run the declared
pilot:

```bash
python scripts/run_classifier_pilot.py --run --base-path /path/to/CS682_Project
```

Review the validation curves before freezing, including any fit still improving
at epoch 30. If the epoch ceiling changes, repeat all 12 fits under the revised
schedule. Otherwise retain the declared budget and freeze the completed pilot:

```bash
python scripts/run_classifier_pilot.py --freeze --base-path /path/to/CS682_Project
```

The pilot uses seed 42, B1/B3, both settings, and rates 2e-5/5e-5/1e-4. EDA is
prepared once and reused. The winner maximizes the equally weighted mean of the
four best-checkpoint validation macro-F1 scores; exact ties prefer the lower rate.
No test data is supplied to training or selection. A frozen main configuration is
required before `scripts/run_baselines.py --run` will train official baselines.

Pilot checkpoints are separate under `04_checkpoints/classifier_pilot/`.
Pilot reports are under `05_results/classifier_pilot/`, with small review copies
under `results/metrics/master_v1/`. Main B1/B2/B3 models use
`04_checkpoints/<condition>/<setting>/seed_<seed>/best_model/`.

Matching complete fits are reused after configuration and artifact checks.
Interrupted fits restart from the seed; this implementation does not claim
optimizer-state or mid-epoch continuation. Do not silently modify the frozen
policy in an existing output directory.

## Official baseline matrix

Use seeds 42, 123, and 456 across low-resource and imbalanced data. Module A owns
all B1/B2/B3 runs. There are 18 logical result entries, normally 15 actual fits
because each balanced B2 entry can reuse its matching B1 input and model.

The complete project adds 18 F0/F1/F2 fits owned by Modules C and D. The classifier
pilot is recorded separately from the core matrix. Module A does not implement
or execute other members' filtering/generation responsibilities automatically.

## Integrity and reporting notes

- The source BANKING77 benchmark contains seven normalized train/test text
  overlaps. They are disclosed without changing the official split or filtering
  training data against test examples. Validation/test text overlap is zero.
- Training/validation real records and provenance remain disjoint. One B3
  imbalanced seed-123 edit coincides with a validation query. This coincidence is
  recorded; EDA was constructed from training data only and is not filtered using
  validation references. The trainer records any such generated-text coincidences.
- Training uses no test references. The early B1 handoff contains validation
  predictions; final test evaluation is a later reporting step.
- Float32 settings are shared, while exact numerical results can vary by backend.
  Record actual hardware, steps, epochs, tokens, and time. Synchronized elapsed
  intervals are not measurements of accelerator utilization.
- Pilot, historical development, and core costs remain separate. Balanced B2
  identities reference B1 and add no duplicate fit to project totals.

## Verification

Run the repository's focused data, baseline, pilot-selection, and trainer checks:

```bash
python -m unittest discover -s tests -v
```

The local environment has verified the official exports, strict reuse/refusal
behaviour, and a small real DistilBERT CPU training/checkpoint roundtrip. A Colab
CUDA execution check is still required on an actual Colab GPU runtime; it has not
been claimed as completed here.
