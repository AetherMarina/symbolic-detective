# v0.2 — Retrieval & Classification Robustness Evaluation

## 1. Objective

Evaluate how synthetic image degradations affect the frozen v0.2 retrieval and classification pipelines, comparing Hybrid DINOv2 + geometric features against a DINOv2-only baseline.

## 2. Evaluation Setup

- **Dataset:** v0.2 development split, artifact-disjoint 5-fold cross-validation.
- **Queries:** Six known motif classes; clean gallery includes `negative_control` distractors.
- **Retrieval:** Pointedness features, α=0.25.
- **Classification:** Harmonics features, α=0.25, similarity-weighted KNN (K=7), artifact-deduplicated neighbors.
- **Baseline:** DINOv2-only (α=0), using the same classification voting strategy.
- **Degradations:** Rotation 90°, Gaussian blur 5×5 and 11×11, low light, low contrast, and 10% occlusion (five repetitions).
- **Metrics:** Retrieval Hit@1, Hit@5, MRR; classification Accuracy and six-class Macro F1.

Only queries are degraded; the gallery remains clean. Results are averaged across folds, with repetitions averaged within each fold.

## 3. Results

### Clean Baseline

| Metric | DINOv2 | Hybrid |
|---|---:|---:|
| Retrieval Hit@1 | 0.604 | **0.610** |
| Retrieval Hit@5 | 0.877 | **0.880** |
| Retrieval MRR | 0.720 | **0.722** |
| Classification Accuracy | 0.611 | **0.648** |
| Classification Macro F1 | 0.599 | **0.646** |

### Performance Change Under Degradation

Values represent absolute differences from each method's own clean baseline (Δ = degraded − clean).

| Condition | DINOv2 Δ Hit@1 | Hybrid Δ Hit@1 | DINOv2 Δ Macro F1 | Hybrid Δ Macro F1 |
|---|---:|---:|---:|---:|
| Rotation 90° | −0.024 | −0.026 | +0.024 | −0.050 |
| Blur 5×5 | −0.026 | −0.036 | −0.014 | −0.046 |
| Blur 11×11 | −0.037 | −0.067 | −0.013 | −0.077 |
| Low light | +0.002 | −0.039 | −0.011 | −0.049 |
| Low contrast | −0.008 | −0.014 | +0.005 | −0.036 |
| Occlusion 10% | −0.020 | −0.038 | +0.026 | −0.043 |

## 4. Findings

- **Clean performance:** Hybrid performs slightly better in retrieval and more substantially in classification (+0.047 Macro F1).
- **Strongest degradation:** Blur 11×11 produces the largest Hybrid decline (−0.067 Hit@1; −0.077 Macro F1).
- **Relative robustness:** DINOv2-only exhibits smaller performance declines under most tested degradations.
- **Geometric sensitivity:** The larger Hybrid declines are consistent with sensitivity of contour-based features to image degradation.
- **Positive baseline changes:** Small improvements under rotation, contrast changes, or occlusion should not be interpreted as evidence that degradation improves performance.

## 5. Conclusion

Hybrid geometric features improve clean-image classification but introduce additional sensitivity to synthetic degradations, particularly strong blur.

The frozen v0.2 configurations are retained without further optimization.

**Limitations:** Development-only evaluation, synthetic rather than real-world degradations, and no robustness testing of the unknown-query rejection mechanism.