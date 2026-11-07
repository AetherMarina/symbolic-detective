# v0.2 — Negative Query Rejection Protocol

## 1. Objective

Evaluate whether Symbolic Detective can distinguish known motif classes from unknown motifs rather than forcing every query into a known category.

This experiment extends the frozen v0.2 classification pipeline with a confidence-based rejection mechanism.

## 2. Frozen Configuration

- **Feature set:** Harmonics
- **Hybrid weight:** α=0.25
- **Classifier:** Similarity-weighted KNN, K=7
- **Known classes:** Six target motif classes
- **Unknown queries:** `negative_control`

Unlike the standard classification evaluation, the retrieval gallery contains **only known motif classes**. Negative controls are used as unknown queries, not as gallery candidates.

## 3. Rejection Methods

Two confidence signals are evaluated independently:

**Maximum Similarity**

The highest hybrid similarity score between the query and any gallery crop.

$$
C_{\mathrm{sim}}=\max_i s_{\mathrm{hybrid}}(q,g_i)
$$

**Weighted Vote Confidence**

The proportion of total KNN voting weight assigned to the predicted class.

$$
C_{\mathrm{vote}}=
\frac{\sum_{i=1}^{K}\mathbb{1}[y_i=\hat{y}]\,w_i}
{\sum_{i=1}^{K}w_i}
$$

where \(w_i=s_{\mathrm{hybrid}}(q,g_i)\), and \(\mathbb{1}[y_i=\hat{y}]\) equals 1 when neighbor \(i\) belongs to the predicted class.

The sum is computed over the **Top-7 artifact-deduplicated neighbors (K=7)**, using the same similarity weights as the frozen Weighted KNN classifier.

A query is classified as `unknown` when its confidence falls below the calibrated threshold. Otherwise, the original KNN prediction is retained.

## 4. Evaluation Protocol

**Development Calibration**

- Use artifact-disjoint development folds.
- Build galleries from known-class artifacts only.
- Evaluate known-class and `negative_control` queries.
- Calibrate a separate rejection threshold for each confidence method using development predictions.

**Held-Out Evaluation**

- Use all development known-class artifacts as the gallery.
- Evaluate both known and `negative_control` held-out queries.
- Apply the frozen thresholds without further tuning.

## 5. Metrics

| Metric | Description |
|---|---|
| True Rejection Rate (TRR) | Fraction of unknown queries correctly rejected |
| False Rejection Rate (FRR) | Fraction of known queries incorrectly rejected |
| Unknown F1 | Precision/recall balance for the `unknown` class |
| Open-set Macro F1 | Classification F1 averaged across six known classes and `unknown` |

The objective is to balance unknown detection against preserving correct classification of known motifs.

## 6. Scope and Limitations

- `negative_control` is used as a proxy for unknown motifs, not a comprehensive representation of all out-of-distribution inputs.
- Confidence scores are heuristic similarity/voting measures, not calibrated probabilities.
- Rejection thresholds are selected exclusively on development data.
- Results are reported separately from the original v0.2 closed-set classification benchmark because the gallery and query protocols differ.

## 7. Development Threshold Calibration

Rejection thresholds were selected using artifact-grouped 5-fold development evaluation, maximizing binary Balanced Accuracy:

$$
\mathrm{Balanced\ Accuracy} = \frac{(1-\mathrm{FRR})+\mathrm{TRR}}{2}
$$

| Method | Selected Threshold | TRR ↑ | FRR ↓ | Balanced Accuracy ↑ |
|---|---:|---:|---:|---:|
| Maximum Similarity | 0.6977 | **0.8800** | **0.3207** | **0.7797** |
| Weighted Vote Confidence | 0.5671 | 0.7200 | 0.3966 | 0.6617 |

**Selection:** Maximum Similarity outperformed Weighted Vote Confidence across all three rejection metrics and was selected for final held-out evaluation.

Threshold selection used development data only.

## 8. Held-Out Evaluation

The selected Maximum Similarity method was evaluated using the frozen threshold of **0.6977**.

| Metric | Held-Out Result |
|---|---:|
| Known queries | 61 |
| Unknown queries | 13 |
| True Rejection Rate (TRR) | **0.9231** (12/13) |
| False Rejection Rate (FRR) | 0.2459 (15/61) |
| Balanced Accuracy | 0.8386 |
| Unknown Precision | 0.4444 |
| Unknown F1 | 0.6000 |
| Open-set Macro F1 (7 classes) | 0.4991 |

The rejection mechanism successfully identified 12 of 13 unknown queries but incorrectly rejected 15 known queries.

### Per-Class Performance

| Class | F1 |
|---|---:|
| `eight_petal_rosette` | 0.1333 |
| `eight_pointed_star` | 0.7273 |
| `five_pointed_star` | 0.5000 |
| `rosette_other` | 0.5000 |
| `six_pointed_star` | 0.5000 |
| `sunburst` | 0.5333 |
| `unknown` | 0.6000 |

## 9. Error Analysis and Findings

**False rejections:** Among the 15 incorrectly rejected known queries, 9 had been correctly classified before applying the rejection threshold.

**Class-specific sensitivity:** `sunburst` accounted for 7 of the 15 false rejections (7/11 sunburst queries). This suggests that a global similarity threshold may reject valid motifs with relatively low gallery similarity.

**False acceptance:** One negative-control query (`701288_primary_occ_02.jpg`) was incorrectly accepted as `eight_petal_rosette`, with maximum similarity 0.7562 but weighted vote confidence only 0.2816.

### Conclusion

Maximum Similarity provides effective negative-query detection (92.3% TRR), but false rejection remains substantial (24.6% FRR).

The results demonstrate the trade-off between unknown detection and retaining valid known-class predictions. The frozen v0.2 threshold is retained without further adjustment.

**Potential future experiment:** Evaluate combined confidence rejection strategies (AND/OR) on development data to investigate whether complementary confidence signals can reduce false rejections. Any newly selected strategy would require independent confirmation before claiming an improvement over this held-out benchmark.
