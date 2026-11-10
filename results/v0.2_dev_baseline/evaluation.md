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

## 4. Multi-Class Discrimination Performance

| Motif class         | Support |  DINO P  |  DINO R  | DINO F1  | Hybrid P | Hybrid R | Hybrid F1 |
|:--------------------| :---: |:--------:|:--------:|:--------:|:--------:|:--------:| :---: |
| eight_petal_rosette | 34 |   0.56   |   0.59   |   0.57   |   0.67   |   0.65   | 0.66 |
| eight_pointed_star  | 52 |   0.67   |   0.75   |   0.71   |   0.58   |   0.75   | 0.66 |
| five_pointed_star   | 42 |   0.60   |   0.67   |   0.63   |   0.59   |   0.57   | 0.58 |
| rosette_other       | 31 |   0.50   |   0.55   |   0.52   |   0.62   |   0.58   | 0.60 |
| six_pointed_star    | 41 |   0.61   |   0.46   |   0.53   |   0.57   |   0.49   | 0.53 |
| sunburst            | 37 |   0.88   |   0.57   |   0.69   |   0.84   |   0.57   | 0.68 |
| **Micro avg**       | **237** | **0.63** | **0.61** | **0.62** | **0.63** | **0.61** | **0.62** |
| **Macro avg**       | **237** | **0.64** | **0.60** | **0.61** | **0.64** | **0.60** | **0.62** |
| **Weighted avg**    | **237** | **0.64** | **0.61** | **0.62** | **0.64** | **0.61** | **0.62** |

Per-class precision, recall, and F1 are calculated from pooled predictions across all five cross-validation folds (237 queries). These differ from the fold-level mean ± standard deviation reported in Sections 2 and 3. DINOv2 corresponds to α=0.0, and Hybrid to α=0.25.

## 5. Key Behavioral Insights & Error Analysis

The introduction of closely related geometries (rosettes, sunbursts, and highly variable star points) successfully stress-tested the retrieval pipeline.

* **The Geometry vs. Semantics Trade-off:**

  * **Curved Motifs (Hybrid Advantage):** Adding the geometric signal at α=0.25 improved discrimination for the rosette classes. F1 increased from 0.57 to 0.66 for `eight_petal_rosette` and from 0.52 to 0.60 for `rosette_other`.
  
  * **Angular Motifs (DINOv2 Advantage):** DINOv2 visual embeddings retained an advantage for several angular motif classes, while the current geometric representation sometimes shifted correct star-family matches toward neighboring point-count classes.

* **The Point-Counting Problem:** Both architectures struggle heavily with the `six_pointed_star` (achieving only \~46-49% recall). The confusion matrix and error analysis reveal that `eight_pointed_star` is most frequently confused with both `five_pointed_star` and `six_pointed_star`. The system currently fails to accurately "count" points, confusing them based on similar angular density.

* **Pipeline Vulnerabilities:** Two development crops failed feature extraction (`173406_primary_occ_01.jpg` and `41417_primary_occ_01.jpg`). These cases should be inspected separately to determine whether the failure originates in preprocessing or geometric feature extraction.

## 6. Follow-up Experiment: Multi-Neighbor Class Voting

To investigate whether aggregating multiple retrieved neighbors improves fine-grained motif discrimination, we evaluated majority and similarity-weighted voting against the Top-1 baseline.

The follow-up experiment has been completed. See [Class Voting Experiment Evaluation](../v0.2_dev_baseline/voting_evaluation.md) for methodology, results, and per-class analysis.
