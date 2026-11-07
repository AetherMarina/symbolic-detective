# v0.2 — Combined Confidence Rejection

## Objective

Evaluate whether combining Maximum Similarity and Weighted Vote Confidence improves unknown-motif rejection over independent confidence thresholds.

The frozen v0.2 classifier remains unchanged (Harmonics, α=0.25, Weighted K=7).

## Method

Using artifact-grouped development folds, compare:

- **Independent rejection:** Each confidence signal uses its independently calibrated threshold.
- **REJECT_OR:** Reject if either confidence score falls below its threshold.
- **REJECT_AND:** Reject only if both confidence scores fall below their thresholds.

For combined strategies, threshold pairs are jointly optimized to maximize development Balanced Accuracy.

## Development Results

| Strategy | Max Sim Threshold | Vote Threshold | TRR ↑ | FRR ↓ | Balanced Accuracy ↑ |
|---|---:|---:|---:|---:|---:|
| Max Sim | 0.6977 | — | 0.8800 | 0.3207 | 0.7797 |
| Weighted Vote | — | 0.5671 | 0.7200 | 0.3966 | 0.6617 |
| REJECT_OR | 0.6693 | 0.3091 | 0.7800 | 0.2110 | 0.7845 |
| **REJECT_AND** | **0.6977** | **0.8565** | **0.8800** | **0.2785** | **0.8008** |

## Findings

**REJECT_AND achieved the highest development Balanced Accuracy (0.8008).** Compared with Maximum Similarity alone, it reduced false rejections from 76 to 66 of 237 known queries while maintaining 44/50 correctly rejected unknown queries.

The improvement is preliminary because thresholds and performance were evaluated on the same development predictions.

**Status:** Development-only exploratory experiment. Independent held-out validation has not been performed.