# v0.2 — Experiment 01: Harmonic Features

## 1. Objective and Experimental Setup

### Objective

This experiment investigates whether incorporating radial symmetry harmonics into the geometric feature representation improves fine-grained discrimination between visually similar motif classes.

The primary motivation is the **point-counting problem** observed in the v0.2 development baseline, where five-, six-, and eight-pointed stars are frequently confused despite their distinct geometric structures.

### Experimental Hypothesis

Explicit radial harmonic features may provide complementary geometric information that helps distinguish motifs with different rotational symmetry patterns, potentially improving retrieval ranking and class discrimination.

The experiment evaluates whether these features improve:

- **Retrieval performance:** Hit@1, Hit@5, and Mean Reciprocal Rank (MRR).
- **Multi-class discrimination:** Precision, recall, and F1, particularly for star-shaped motifs.
- **Neighborhood voting:** Accuracy and macro F1 under majority and similarity-weighted voting.

### Feature Modification

The experiment extends the existing OpenCV geometric representation with three harmonic descriptors:

- `harmonic_5`
- `harmonic_6`
- `harmonic_8`

These features are intended to capture radial periodicity associated with five-fold, six-fold, and eight-fold geometric structures.

**Representation comparison:**

| Configuration | Geometric Features |
|---|---|
| v0.2 Development Baseline | Original geometric feature set |
| exp01_harmonics | Original geometric features + harmonic_5, harmonic_6, harmonic_8 |

DINOv2 embeddings remain unchanged. The harmonic descriptors are evaluated through the existing geometric similarity and hybrid retrieval framework.

### Evaluation Protocol

The experiment follows the same evaluation protocol as the [v0.2 Development Baseline](../../v0.2_dev_baseline/evaluation.md):

- **Dataset:** Expanded v0.2 related-motif dataset.
- **Development partition:** 251 source artifacts, with 287 successfully extracted crops.
- **Queries:** 237 motif crops across six target classes.
- **Cross-validation:** Five stratified, artifact-grouped folds to prevent source-artifact leakage.
- **Held-out test:** 74 crops remain excluded from development evaluation.
- **Retrieval configurations:** DINOv2 ($\alpha=0.0$), Hybrid ($\alpha \in \{0.25, 0.50, 0.75\}$), and geometry-only ($\alpha=1.0$).
- **Class voting:** Top-1 ($K=1$), majority voting, and similarity-weighted voting ($K \in \{3,5,7\}$), with artifact deduplication.

All comparisons are conducted against the frozen v0.2 development baseline to isolate the effect of adding harmonic features.

## 2. Retrieval Performance Comparison

Results are reported as **mean ± standard deviation across five cross-validation folds**. Δ represents experiment − baseline, calculated from unrounded means; positive values indicate improvement.

### 2.1 Overall Multi-Class Retrieval

| α | Metric | Baseline (Mean ± SD) | Harmonics (Mean ± SD) | Δ |
|---|---|---:|---:|---:|
| 0.00 | Hit@1 | 0.604 ± 0.064 | 0.604 ± 0.064 | +0.0000 |
| | Hit@5 | 0.877 ± 0.019 | 0.877 ± 0.019 | +0.0000 |
| | MRR | 0.720 ± 0.040 | 0.720 ± 0.040 | +0.0000 |
| 0.25 | Hit@1 | 0.606 ± 0.037 | 0.596 ± 0.064 | −0.0095 |
| | Hit@5 | 0.872 ± 0.043 | 0.872 ± 0.053 | +0.0000 |
| | MRR | 0.719 ± 0.036 | 0.717 ± 0.053 | −0.0022 |
| 0.50 | Hit@1 | 0.498 ± 0.075 | 0.512 ± 0.055 | +0.0138 |
| | Hit@5 | 0.891 ± 0.039 | 0.894 ± 0.046 | +0.0033 |
| | MRR | 0.665 ± 0.055 | 0.670 ± 0.047 | +0.0049 |
| 0.75 | Hit@1 | 0.354 ± 0.110 | 0.359 ± 0.073 | +0.0043 |
| | Hit@5 | 0.779 ± 0.077 | 0.798 ± 0.069 | +0.0190 |
| | MRR | 0.538 ± 0.087 | 0.546 ± 0.061 | +0.0073 |
| 1.00 | Hit@1 | 0.267 ± 0.096 | 0.237 ± 0.075 | −0.0298 |
| | Hit@5 | 0.648 ± 0.085 | 0.660 ± 0.067 | +0.0123 |
| | MRR | 0.442 ± 0.079 | 0.427 ± 0.066 | −0.0148 |

*Δ values are computed from unrounded means.*

**Observations:**

- **DINOv2 (α=0.0):** Results remain identical, as expected, since the latent embeddings are unchanged.
- **Hybrid (α=0.25):** Hit@1 decreases slightly, while Hit@5 remains unchanged and MRR decreases marginally.
- **Hybrid (α=0.50–0.75):** Small improvements appear across retrieval metrics, particularly Hit@5 at α=0.75.
- **Geometry-only (α=1.0):** Hit@1 and MRR decrease despite a modest improvement in Hit@5, indicating that the added harmonic features do not improve top-ranked retrieval on their own.

### 2.2 Target-Class Retrieval: `eight_pointed_star` (52 Queries)

| α | Metric | Baseline (Mean ± SD) | Harmonics (Mean ± SD) | Δ |
|---|---|---:|---:|---:|
| 0.00 | Hit@1 | 0.753 ± 0.059 | 0.753 ± 0.059 | +0.0000 |
| | Hit@5 | 0.960 ± 0.049 | 0.960 ± 0.049 | +0.0000 |
| | MRR | 0.829 ± 0.044 | 0.829 ± 0.044 | +0.0000 |
| 0.25 | Hit@1 | 0.746 ± 0.086 | 0.728 ± 0.078 | −0.0182 |
| | Hit@5 | 0.938 ± 0.051 | 0.938 ± 0.051 | +0.0000 |
| | MRR | 0.834 ± 0.075 | 0.825 ± 0.068 | −0.0088 |
| 0.50 | Hit@1 | 0.667 ± 0.076 | 0.687 ± 0.086 | +0.0204 |
| | Hit@5 | 0.960 ± 0.080 | 0.980 ± 0.040 | +0.0200 |
| | MRR | 0.787 ± 0.056 | 0.801 ± 0.071 | +0.0140 |
| 0.75 | Hit@1 | 0.458 ± 0.118 | 0.439 ± 0.121 | −0.0196 |
| | Hit@5 | 0.825 ± 0.076 | 0.898 ± 0.090 | +0.0730 |
| | MRR | 0.623 ± 0.079 | 0.625 ± 0.096 | +0.0020 |
| 1.00 | Hit@1 | 0.363 ± 0.043 | 0.288 ± 0.049 | −0.0748 |
| | Hit@5 | 0.684 ± 0.119 | 0.726 ± 0.113 | +0.0422 |
| | MRR | 0.521 ± 0.049 | 0.488 ± 0.061 | −0.0330 |

*Δ values are computed from unrounded means.*

**Observations:**

- **Hybrid (α=0.25):** Slight reductions in Hit@1 and MRR, with no change in Hit@5.
- **Hybrid (α=0.50):** Improvements across all three retrieval metrics, although performance remains below the DINOv2-only configuration.
- **Hybrid (α=0.75):** A noticeable Hit@5 improvement (+0.0730), accompanied by lower Hit@1.
- **Geometry-only (α=1.0):** Hit@5 increases, but Hit@1 drops substantially, indicating that more correct-class candidates appear within the top five without improving their highest-rank placement.

### 2.3 Fold-Level Consistency and Retrieval Assessment

To assess whether the observed retrieval changes are consistent across cross-validation folds, paired fold-level differences were examined for overall Hit@1.

| α | Improved folds | Worsened folds | Unchanged folds | Mean Δ Hit@1 |
|---|---:|---:|---:|---:|
| 0.00 | 0 | 0 | 5 | +0.0000 |
| 0.25 | 2 | 2 | 1 | −0.0095 |
| 0.50 | 2 | 3 | 0 | +0.0138 |
| 0.75 | 1 | 2 | 2 | +0.0043 |
| 1.00 | 0 | 4 | 1 | −0.0298 |

**Key observations:**

- **Hybrid configurations (α=0.25–0.75):** Fold-level effects are mixed. At α=0.50, mean Hit@1 improves despite three of five folds worsening, indicating that gains in the remaining folds outweigh the declines.
- **Geometry-only (α=1.00):** Hit@1 worsens in four folds and remains unchanged in one, providing the clearest evidence of a consistently negative effect from harmonic augmentation on first-rank retrieval.
- **Top-5 retrieval:** Some improvements occur at intermediate hybrid weights, but these do not consistently translate into better Hit@1 performance.

Overall, harmonic augmentation produces **mixed retrieval effects rather than a consistent improvement**, with the most systematic deterioration observed in geometry-only Top-1 retrieval.

These findings motivate further examination of per-class discrimination and confusion patterns to determine whether harmonic features improve the separation of closely related star geometries.

The differences are descriptive development-set observations and have not been established as statistically significant.

## 3. Multi-Class Discrimination Analysis

This section evaluates whether harmonic features improve Top-1 classification across the six motif classes.

Unlike the cross-validation retrieval metrics in Section 2, precision, recall, and F1 are calculated from **pooled predictions across all five cross-validation folds (237 queries)**.

Since DINOv2 embeddings remain unchanged, the analysis focuses on the Hybrid representation (α=0.25).

### 3.1 Per-Class Performance Comparison

| Motif class | Support | Baseline P | Baseline R | Baseline F1 | Harmonics P | Harmonics R | Harmonics F1 | Δ F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| eight_petal_rosette | 34 | 0.667 | 0.647 | 0.657 | 0.667 | 0.647 | 0.657 | +0.0000 |
| eight_pointed_star | 52 | 0.582 | 0.750 | 0.655 | 0.576 | 0.731 | 0.644 | −0.0114 |
| five_pointed_star | 42 | 0.585 | 0.571 | 0.578 | 0.558 | 0.571 | 0.565 | −0.0136 |
| rosette_other | 31 | 0.621 | 0.581 | 0.600 | 0.594 | 0.613 | 0.603 | +0.0032 |
| six_pointed_star | 41 | 0.571 | 0.488 | 0.526 | 0.621 | 0.439 | 0.514 | −0.0120 |
| sunburst | 37 | 0.840 | 0.568 | 0.677 | 0.875 | 0.568 | 0.689 | +0.0111 |
| **Micro avg** | **237** | 0.626 | 0.608 | 0.617 | 0.626 | 0.599 | 0.612 | −0.0046 |
| **Macro avg** | **237** | 0.644 | 0.601 | 0.616 | 0.648 | 0.595 | 0.612 | −0.0038 |
| **Weighted avg** | **237** | 0.638 | 0.608 | 0.616 | 0.643 | 0.599 | 0.611 | −0.0048 |

*P = Precision, R = Recall. Δ F1 = Harmonics − Baseline, calculated from unrounded values.*

**Observations:**

- **Overall discrimination:** Macro F1 decreases slightly from 0.6157 to 0.6119, indicating no aggregate improvement in Top-1 classification.
- **Star motifs:** All three star classes exhibit small F1 decreases after introducing harmonic features.
- **Rosette motifs:** `eight_petal_rosette` remains unchanged, while `rosette_other` shows a marginal F1 improvement.
- **Sunburst:** F1 improves from 0.677 to 0.689, primarily through increased precision rather than recall.

### 3.2 Star-Class Discrimination: The Point-Counting Problem

To assess whether harmonics improve discrimination between five-, six-, and eight-pointed stars, we examine changes in precision, recall, and F1 independently.

| Star class | Δ Precision | Δ Recall | Δ F1 |
|---|---:|---:|---:|
| five_pointed_star | −0.0272 | +0.0000 | −0.0136 |
| six_pointed_star | +0.0493 | −0.0488 | −0.0120 |
| eight_pointed_star | −0.0063 | −0.0192 | −0.0114 |

**Key observations:**
- **`five_pointed_star`:** Recall remains unchanged at 57.1%, while precision decreases. Harmonics do not improve identification of actual five-pointed stars and introduce additional false positives.
- **`six_pointed_star`:** Precision increases from 57.1% to 62.1%, but recall decreases from 48.8% to 43.9%. Predictions of this class become more precise, but fewer actual six-pointed stars are correctly identified.
- **`eight_pointed_star`:** Both precision and recall decrease slightly. The representation does not improve recognition of eight-fold star

## 4. Confusion Pattern Analysis

The confusion-matrix difference is **Harmonics − Baseline** for Hybrid Top-1 (α=0.25). Positive diagonal values indicate more correct predictions; negative off-diagonal values indicate fewer misclassifications of the corresponding type.

### 4.1 Confusion Matrix Difference

![Confusion Matrix Delta — Harmonics vs Baseline](figures/confusion_delta_hybrid.png)

### 4.2 Star-Class Confusion Changes

| True class | Baseline correct | Harmonics correct | Δ |
|---|---:|---:|---:|
| `five_pointed_star` | 24 / 42 | 24 / 42 | 0 |
| `six_pointed_star` | 20 / 41 | 18 / 41 | −2 |
| `eight_pointed_star` | 39 / 52 | 38 / 52 | −1 |

Harmonics redistribute star-class errors rather than systematically reducing them. Confusion between five- and six-pointed stars decreases by three predictions, but confusion between five- and eight-pointed stars increases by two.

Six-pointed-star correct predictions decrease by two, with two additional assignments to `negative_control`. Eight-pointed-star correct predictions decrease by one.

**Harmonics do not improve star point-count discrimination under Hybrid Top-1 classification.**

### 4.3 Other Motif Classes and Negative Controls

- **`eight_petal_rosette`:** Correct predictions remain at 22 / 34.
- **`rosette_other`:** Correct predictions improve from 18 to 19 / 31.
- **`sunburst`:** Correct predictions remain at 21 / 37.
- **Negative-control assignments:** Additional star queries are incorrectly assigned to `negative_control`.

Negative controls are retrieval candidates but not evaluation queries, explaining the all-zero `negative_control` row.

## 5. Class Voting Performance Comparison

This section compares baseline and feature-augmented class voting using artifact-deduplicated Top-K retrieval. The primary analysis focuses on Hybrid α=0.25.
Results are reported as **mean ± standard deviation across five cross-validation folds**. Δ values represent experiment − baseline.

### 5.1 Hybrid Voting Performance (α=0.25)

| Strategy | K | Baseline Accuracy | Experiment Accuracy | Δ Accuracy | Baseline Macro F1 | Experiment Macro F1 | Δ Macro F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Top-1** | 1 | 0.6058 ± 0.0374 | 0.5962 ± 0.0645 | −0.0096 | 0.6089 ± 0.0358 | 0.6069 ± 0.0638 | −0.0020 |
| Majority | 3 | 0.5944 ± 0.0559 | 0.6076 ± 0.0738 | +0.0132 | 0.5913 ± 0.0581 | 0.6071 ± 0.0767 | +0.0158 |
| Majority | 5 | 0.6076 ± 0.0131 | 0.6330 ± 0.0259 | +0.0254 | 0.6067 ± 0.0217 | 0.6300 ± 0.0299 | +0.0233 |
| Majority | 7 | 0.6187 ± 0.0447 | 0.6395 ± 0.0474 | +0.0208 | 0.6184 ± 0.0467 | 0.6375 ± 0.0509 | +0.0191 |
| Weighted | 3 | 0.5944 ± 0.0559 | 0.6076 ± 0.0738 | +0.0132 | 0.5913 ± 0.0581 | 0.6071 ± 0.0767 | +0.0158 |
| Weighted | 5 | 0.6076 ± 0.0131 | 0.6158 ± 0.0218 | +0.0082 | 0.6067 ± 0.0217 | 0.6093 ± 0.0285 | +0.0026 |
| **Weighted** | **7** | **0.6185 ± 0.0443** | **0.6483 ± 0.0407** | **+0.0298** | **0.6214 ± 0.0425** | **0.6463 ± 0.0413** | **+0.0249** |

**Key observations:**
- **Top-1 performance:** Harmonic augmentation slightly decreases both accuracy and macro F1, consistent with the retrieval and discrimination results from Sections 2 and 3.
- **Neighborhood voting:** Both majority and weighted voting improve over their corresponding baseline configurations across all tested neighborhood sizes (K=3, 5, 7).
- **Best-performing configuration:** Hybrid Weighted K=7 achieves the highest observed accuracy (0.6483) and macro F1 (0.6463), improving by 2.98 and 2.49 percentage points, respectively, over the corresponding baseline.
- **Neighborhood size:** Larger neighborhoods generally benefit the harmonic-enhanced representation, with K=7 producing the strongest results among the tested configurations.
- **Weighting strategy:** Similarity weighting does not consistently outperform majority voting. At K=5, majority voting achieves higher macro F1 (0.6300 vs. 0.6093), while weighted voting performs better at K=7 (0.6463 vs. 0.6375).

### 5.2 Per-Class Voting Performance

Per-class values below are **pooled across all five folds**, not fold-averaged macro F1. Results correspond to Hybrid α=0.25, Weighted K=7.

| Motif class | Δ Precision | Δ Recall | Δ F1 |
|---|---:|---:|---:|
| `eight_pointed_star` | −0.0039 | +0.0192 | +0.0050 |
| `five_pointed_star` | +0.0040 | +0.0476 | +0.0296 |
| `six_pointed_star` | +0.0845 | +0.0000 | +0.0351 |
| `eight_petal_rosette` | +0.0093 | +0.0294 | +0.0167 |
| `rosette_other` | +0.0267 | +0.0645 | **+0.0493** |
| `sunburst` | +0.0000 | +0.0270 | +0.0202 |

**All six motif classes improve in F1 under Weighted K=7 voting**, unlike Top-1 classification, where the three star classes show small declines.

- **Star classes:** Five-pointed-star recall increases by 0.0476 (two additional correct predictions), while eight-pointed-star recall increases by 0.0192 (one additional correct prediction). Six-pointed-star recall remains unchanged, but precision improves by 0.0845.
- **Non-star classes:** All three classes improve in F1, with the largest gain for `rosette_other` (+0.0493).

These improvements are consistent with the increase in fold-averaged macro F1 from 0.6214 to 0.6463.

### 5.3 Per-Query Prediction Transitions

To examine how harmonic features affect individual voting predictions, baseline and harmonics predictions were compared across 237 identical queries using Hybrid α=0.25, Weighted K=7.

| Transition | Query crops |
|---|---:|
| Both correct | 143 |
| Both wrong | 79 |
| Rescued | 11 |
| Regressed | 4 |

Of the 237 queries, **19 changed their predicted class**: 11 were corrected, four regressed, and four remained incorrect but switched to another class. The net improvement was seven correctly classified crops.

**Prediction transitions by true motif class:**

| Motif class | Rescued | Regressed | Net change |
|---|---:|---:|---:|
| `eight_pointed_star` | 2 | 1 | +1 |
| `five_pointed_star` | 2 | 0 | +2 |
| `six_pointed_star` | 2 | 2 | 0 |
| `eight_petal_rosette` | 1 | 0 | +1 |
| `rosette_other` | 2 | 0 | +2 |
| `sunburst` | 2 | 1 | +1 |

**Key finding:** Star classes gain three net correct predictions, while non-star classes gain four. Six previously incorrect star-to-star predictions are corrected, although three previously correct star predictions regress to another star class. The improvements extend across both motif families, supporting the observed increase in overall voting macro F1.

### 5.4 Artifact-Grouped Transition Analysis

To determine whether prediction changes are concentrated within individual artifacts, crop-level transitions were grouped by source artifact and true motif class.

Among 15 artifact-class groups with changes in prediction correctness:

- **11** show a net improvement.
- **4** show a net regression.
- **0** exhibit mixed outcomes.

All 19 changed crop-level predictions correspond to distinct source artifacts, indicating that the observed improvements are not driven by repeated crops of a single artifact.

The 11 rescued and four regressed predictions therefore span 15 distinct artifacts, with a net gain of seven correctly classified crops.

**Assessment:** Harmonic augmentation produces improvements across multiple independent source artifacts, including both star and non-star motifs, rather than concentrating gains within a small number of multiply cropped artifacts.

These summaries describe crop-level prediction changes grouped by artifact; they do not represent independently computed artifact-level classification accuracy.

### 5.5 Voting Performance Assessment

Harmonic augmentation improves neighborhood voting despite slightly weaker Top-1 classification. Hybrid Weighted K=7 achieves the strongest macro F1 (0.6463 vs. 0.6214 baseline), with F1 improvements across all six motif classes.

Per-query analysis confirms 11 rescued and four regressed predictions across distinct artifacts, including corrections of star-to-star confusions.

**Harmonics show greater promise through neighborhood aggregation than Top-1 retrieval**, although the underlying mechanism remains unverified and the results require held-out validation.

## 6. Key Findings and Interpretation

1. **Retrieval improvements are inconsistent.** Harmonic augmentation produces mixed results across hybrid configurations. Geometry-only overall Hit@1 decreases from 0.2667 to 0.2368, with four of five folds showing deterioration.

2. **Top-1 point-count discrimination remains unresolved.** Under Hybrid α=0.25, all three star classes exhibit small F1 decreases. Six-pointed-star recall declines from 0.4878 to 0.4390, and confusion patterns show a redistribution of errors rather than systematic improvement.

3. **Neighborhood voting benefits from harmonic augmentation.** All tested Hybrid majority and weighted configurations (K=3, 5, 7) improve over their corresponding baselines. Weighted K=7 achieves the highest cross-validation macro F1 (0.6463 vs. 0.6214 baseline).

4. **Weighted K=7 improves all six motif classes.** Pooled per-class F1 increases across both star and non-star motifs. Prediction transitions confirm 11 rescued and four regressed crops, including six corrected star-to-star confusions. All 19 changed predictions correspond to distinct source artifacts.

5. **The mechanism behind voting improvements remains unverified.** Per-query analysis identifies which predictions change, but Top-K neighbor analysis is needed to determine whether the gains arise from altered neighbor composition, similarity rankings, or score distributions.

### Limitations

These findings are exploratory development-set results from five artifact-grouped cross-validation folds. Multiple configurations were examined, and the observed improvements have not been established as statistically significant. The held-out test partition remains untouched.

**Overall assessment:** Harmonic features do not resolve Top-1 point-counting ambiguities, but show promising improvements through neighborhood voting without the star/non-star F1 trade-off observed with radial peaks. Harmonics remain a candidate for Phase B feature-combination experiments, pending further evaluation.
