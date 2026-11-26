# Baseline results summary

Means and sample standard deviations (n−1 denominator) across training seeds 42, 123 and 456. All seeds use the same fixed validation and test splits. These are seed-to-seed standard deviations, not confidence intervals.

## Test scores

| Setting | Baseline | Macro-F1 | Accuracy | Rare-class F1 | Rare-class recall |
|---|---|---:|---:|---:|---:|
| low_resource | B1 | 0.7979 ± 0.0182 | 0.7978 ± 0.0185 | 0.7609 ± 0.0475 | 0.7644 ± 0.0445 |
| low_resource | B2 | 0.7979 ± 0.0182 | 0.7978 ± 0.0185 | 0.7609 ± 0.0475 | 0.7644 ± 0.0445 |
| low_resource | B3 | 0.8055 ± 0.0101 | 0.8064 ± 0.0097 | 0.7701 ± 0.0375 | 0.7767 ± 0.0493 |
| imbalanced | B1 | 0.8206 ± 0.0188 | 0.8242 ± 0.0181 | 0.6986 ± 0.0573 | 0.6172 ± 0.0488 |
| imbalanced | B2 | 0.8143 ± 0.0063 | 0.8201 ± 0.0039 | 0.6702 ± 0.0411 | 0.5600 ± 0.0400 |
| imbalanced | B3 | 0.8232 ± 0.0141 | 0.8275 ± 0.0122 | 0.7029 ± 0.0464 | 0.5994 ± 0.0467 |

## Validation scores

| Setting | Baseline | Macro-F1 | Accuracy | Rare-class F1 | Rare-class recall |
|---|---|---:|---:|---:|---:|
| low_resource | B1 | 0.7868 ± 0.0164 | 0.7870 ± 0.0195 | 0.7599 ± 0.0498 | 0.7778 ± 0.0473 |
| low_resource | B2 | 0.7868 ± 0.0164 | 0.7870 ± 0.0195 | 0.7599 ± 0.0498 | 0.7778 ± 0.0473 |
| low_resource | B3 | 0.7916 ± 0.0190 | 0.7926 ± 0.0195 | 0.7743 ± 0.0305 | 0.7889 ± 0.0491 |
| imbalanced | B1 | 0.8156 ± 0.0075 | 0.8195 ± 0.0091 | 0.7116 ± 0.0338 | 0.6311 ± 0.0204 |
| imbalanced | B2 | 0.8027 ± 0.0097 | 0.8108 ± 0.0101 | 0.6631 ± 0.0200 | 0.5578 ± 0.0269 |
| imbalanced | B3 | 0.8230 ± 0.0125 | 0.8281 ± 0.0110 | 0.7193 ± 0.0255 | 0.6244 ± 0.0139 |

## Completed-run timing

Times are mean ± sample SD per seed. Optimization excludes validation; fit elapsed time includes setup, optimization, validation and checkpoint work.

| Setting | Baseline | Optimization minutes | Fit elapsed minutes | Test inference seconds | Independent fits | Training devices |
|---|---|---:|---:|---:|---:|---|
| low_resource | B1 | 3.2194 ± 0.9530 | 3.7909 ± 0.9762 | 5.1884 ± 0.4317 | 3 | cpu:1,mps:2 |
| low_resource | B2 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0 | cpu:1,mps:2 |
| low_resource | B3 | 6.0712 ± 1.4314 | 6.6664 ± 1.6719 | 4.9342 ± 0.0159 | 3 | cpu:1,mps:2 |
| imbalanced | B1 | 5.2956 ± 1.2974 | 5.8540 ± 1.4170 | 4.9301 ± 0.0082 | 3 | mps:3 |
| imbalanced | B2 | 5.6755 ± 0.9903 | 6.1760 ± 1.0834 | 4.9427 ± 0.0295 | 3 | mps:3 |
| imbalanced | B3 | 11.2710 ± 3.6748 | 11.8695 ± 3.8683 | 4.9357 ± 0.0184 | 3 | mps:3 |

Balanced B2 is identical to B1 and reuses its scores and models. Its zero incremental time means B1's cost is counted once. This is not an independently trained B2 cohort.

Total completed main-fit elapsed time: 103.07 minutes. Total test inference elapsed time: 74.79 seconds.

Pilot experiments, interrupted attempts, EDA preparation, generation and filtering are excluded from these completed-main-fit totals. They must be accounted for separately in the full project cost analysis. Timing is elapsed time, not accelerator utilization.

Low-resource seed 42 used CPU for B1/B3; remaining main fits used MPS. All test inference used MPS. Hardware and training-sample variation accompany seed variation; this table is descriptive and does not establish statistical significance.
