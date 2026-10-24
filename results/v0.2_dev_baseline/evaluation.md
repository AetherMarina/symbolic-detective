# v0.2 Development Baseline — Expanded Related-Motif Dataset

## 1. Evaluation Methodology & Protocol

This baseline establishes retrieval performance for v0.2, moving from the primarily target-focused v0.1 evaluation to a challenging multi-class discrimination setting involving closely related geometric motifs.

* **Partitioning:** A strict 80/20 development/test split was established using `StratifiedGroupKFold`.

* **Leakage Prevention:** Artifacts were strictly grouped. Multiple crops of the same physical artifact were guaranteed to stay in the same partition. The 20% test partition (74 crops) was explicitly skipped and isolated from feature scaling, index building, and query execution.

* **Stratification Fix:** The internal 5-fold cross-validation loop was upgraded from round-robin allocation to StratifiedGroupKFold, improving class balance across folds and substantially reducing variance (e.g., target Hit@1 standard deviation decreased from ±0.207 to ±0.059).

**Dataset Context (Dev Set Only):**
* **Target Classes:** `eight_pointed_star`, `five_pointed_star`, `six_pointed_star`, `eight_petal_rosette`, `rosette_other`, `sunburst`
* **Development Partition:** 251 artifacts
* **Successfully Extracted Dev Crops:** 287
* **Queryable Motif Artifacts:** 212
* **Queries Evaluated:** 237
* **Held-Out Test Crops:** 74
* **Extraction Failures:** 2

## 2. Multi-Class Retrieval Performance (Overall)

The table below represents the average performance across all 5 stratified cross-validation folds.

| **Method** | **Alpha (α)** | **Hit@1** | **Hit@5** | **MRR** | 
| ----- | ----- | ----- | ----- | ----- | 
| Latent Embeddings (DINOv2) | 0.00 | 0.604 ± 0.064 | 0.877 ± 0.019 | **0.720 ± 0.040** | 
| Hybrid Ensemble | 0.25 | **0.606 ± 0.037** | 0.872 ± 0.043 | 0.719 ± 0.036 | 
| Hybrid Ensemble | 0.50 | 0.498 ± 0.075 | **0.891 ± 0.039** | 0.665 ± 0.055 | 
| Hybrid Ensemble | 0.75 | 0.354 ± 0.110 | 0.779 ± 0.077 | 0.538 ± 0.087 | 
| OpenCV Geometric Features | 1.00 | 0.267 ± 0.096 | 0.648 ± 0.085 | 0.442 ± 0.079 | 

## 3. Target Class Retrieval (`eight_pointed_star`)

| **Method** | **Alpha (α)** | **Hit@1** | **Hit@5** | **MRR** | 
| ----- | ----- | ----- | ----- | ----- | 
| Latent Embeddings (DINOv2) | 0.00 | **0.753 ± 0.059** | **0.960 ± 0.049** | 0.829 ± 0.044 | 
| Hybrid Ensemble | 0.25 | 0.746 ± 0.086 | 0.938 ± 0.051 | **0.834 ± 0.075** | 
| Hybrid Ensemble | 0.50 | 0.667 ± 0.076 | **0.960 ± 0.080** | 0.787 ± 0.056 | 
| Hybrid Ensemble | 0.75 | 0.458 ± 0.118 | 0.825 ± 0.076 | 0.623 ± 0.079 |
| OpenCV Geometric Features | 1.00 | 0.363 ± 0.043 | 0.684 ± 0.119 | 0.521 ± 0.049 | 

## 4. Key Behavioral Insights & Error Analysis

The introduction of closely related geometries (rosettes, sunbursts, and highly variable star points) successfully stress-tested the retrieval pipeline.

* **The Geometry vs. Semantics Trade-off:**

  * **Curved Motifs (Hybrid Advantage):** Adding the geometric signal at α=0.25 improved discrimination for the rosette classes. F1 increased from 0.57 to 0.66 for `eight_petal_rosette` and from 0.52 to 0.60 for `rosette_other`.
  
  * **Angular Motifs (DINOv2 Advantage):** DINOv2 visual embeddings retained an advantage for several angular motif classes, while the current geometric representation sometimes shifted correct star-family matches toward neighboring point-count classes.

* **The Point-Counting Problem:** Both architectures struggle heavily with the `six_pointed_star` (achieving only \~46-49% recall). The confusion matrix and error analysis reveal that `eight_pointed_star` is most frequently confused with both `five_pointed_star` and `six_pointed_star`. The system currently fails to accurately "count" points, confusing them based on similar angular density.

* **Pipeline Vulnerabilities:** Two development crops failed feature extraction (`173406_primary_occ_01.jpg` and `41417_primary_occ_01.jpg`). These cases should be inspected separately to determine whether the failure originates in preprocessing or geometric feature extraction.

## 5. Proposed Experiments: Multi-Neighbor Class Voting

To address point-counting ambiguities and stabilize predictions, the next phase of v0.2 will evaluate whether aggregating multiple top-ranked images improves classification accuracy over a strict 1-nearest-neighbor (Top-1) approach. 

### Experimental Design
The experiment will compare three retrieval strategies using the frozen Hybrid (α=0.25) configuration, evaluating across $K \in \{1, 3, 5, 7\}$. 

To prevent multi-crop artifacts from dominating the aggregation, **a strict limit of one vote per source artifact** will be enforced before applying the following voting strategies:

*   **Strategy A — Top-1 (Baseline):** The predicted class is strictly derived from the single highest-ranked image.
*   **Strategy B — K-NN Majority Voting:** The predicted class is the most frequent class among the top $K$ retrieved unique artifacts. 
    *   *Tie-Breaking Rule:* If two or more classes receive an equal number of votes, the tie is broken by selecting the tied class whose first occurrence has the highest similarity score (the highest-ranked neighbor).
*   **Strategy C — Similarity-Weighted Voting:** The predicted class is determined by summing the similarity scores ($S_{hybrid}$) for each class present in the top $K$ unique artifacts, selecting the class with the highest total score:
    $$Score(c) = \sum_{i \in TopK,\;label_i=c} S_i$$

### Success Metrics
Because standard Hit@K only measures if the correct class is *present* in the top results, this experiment will evaluate the final corrected prediction using:
1.  **Top-1 Class Accuracy:** Overall correctness of the final aggregated prediction.
2.  **Macro F1 & Per-Class Recall:** To determine if aggregation successfully rescues the highly confused `six_pointed_star` without degrading the well-performing rosette classes.
3.  **Aggregated Confusion Matrices:** To map how the class-confusion distribution shifts compared to the Top-1 baseline.