# Improvement experiments

All CIFAR-10 comparisons use seed 1 and the same data partition:

- 10 clients with full participation
- 2 classes and 100 training images per client
- 1,000 test images per client
- 3 local alternating-optimization epochs
- 100 communication rounds
- `alpha=0.30`, personalization rate `p=25%`
- SGD with gradient clipping

The baseline score is the reference run at communication round 100.

| Run | Round 100 accuracy | Delta vs. reference | Decision |
| --- | ---: | ---: | --- |
| FedSelect reference | 82.0410% | - | baseline |
| Client-local Mixup | 82.1484% | +0.1074 | kept as component |
| Multi-view TTA on reference checkpoint | 82.4512% | +0.4102 | kept as component |
| Mixup + multi-view TTA | **82.6563%** | **+0.6152** | retained |

The retained condition is the combination of both switches; it improves over
the baseline by more than 0.5 percentage points. Multi-view TTA is
inference-only and was evaluated on the exact reference checkpoint, so it does
not alter model weights.

Several other candidates were rejected during screening:

| Candidate | Screening result | Decision |
| --- | --- | --- |
| Budgeted mask | 79.3262% at round 40 vs. 80.4297% baseline | rejected |
| EMA parameter-delta mask | 80.1855% at round 60 vs. 80.9766% baseline | rejected |
| Client-weight EMA through round 100 | 81.5234% vs. 82.0410% baseline | rejected |
| Cosine learning-rate decay | 81.2207% at round 80 vs. 81.2598% baseline | rejected |
| Mixup + horizontal-flip TTA | 82.2852% | rejected |

These are single-seed results from the local reproduction. Run multiple seeds
before treating the improvement as a paper-level scientific claim.
