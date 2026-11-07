# Hybrid Visual Retrieval Pipeline

> **Scope:** Original Symbolic Detective retrieval baseline. This document connects the two preprocessing streams, their feature representations, corpus indexing, similarity scoring, and evaluation. Feature extraction details are documented separately in [`opencv_geometric_features.md`](opencv_geometric_features.md) and [`dinov2_embeddings.md`](dinov2_embeddings.md).

## 1. Overview

Symbolic Detective compares manually cropped motifs using two complementary representations:

| Stream | Representation | Role |
| --- | --- | --- |
| **OpenCV** (`shape_features.py`) | 26 hand-engineered, edge-based geometric features | Describes global edge geometry, edge density, radial distribution, and angular periodicity |
| **DINOv2** (`embeddings.py`) | 384-dimensional learned visual embedding | Captures visual patterns learned by a pretrained model |

Neither stream directly assigns a motif class. **Retrieval** ranks indexed examples by similarity to a query; class labels are used to evaluate whether the retrieved examples are relevant.

## 2. Image Preprocessing

The same manually cropped motif enters two different preprocessing streams:

```text
                       Original motif crop
                               |
                    +----------+----------+
                    |                     |
                    v                     v
               OpenCV stream         DINOv2 stream
                    |                     |
                 Grayscale                RGB
                    |                     |
                  CLAHE              Square padding
                    |                     |
              Neutral padding            Resize
                    |                     |
                    v                     v
             shape_features.py        embeddings.py
                    |                     |
                    v                     v
             26D raw features       384D L2-normalized
                                        embedding
```

**OpenCV stream.** The image is converted to grayscale, contrast-enhanced using **CLAHE** (*Contrast Limited Adaptive Histogram Equalization*), and padded with a neutral background. Feature extraction then applies Gaussian smoothing and Canny edge detection before computing the 26 geometric values. The resulting edges may include internal motif details and background texture; the stream does not automatically segment the motif.

**DINOv2 stream.** The crop is converted to RGB, square-padded, and resized to **224 × 224**. Input pixels are converted to tensors and normalized using the model's ImageNet channel statistics before inference. The resulting 384D embedding is **L2-normalized** for similarity comparison.

The two streams deliberately use different representations and preprocessing; they are not interchangeable.

## 3. DINOv2 vs. OpenCV

| Aspect | DINOv2 | OpenCV |
| --- | --- | --- |
| Feature type | Learned visual embedding | Explicit geometric descriptors |
| Dimensions | 384 | 26 |
| Input to extractor | Preprocessed RGB image | Preprocessed grayscale image / Canny edges |
| Similarity | Cosine-based | Euclidean distance with exponential mapping |
| Interpretability | Limited at the individual-dimension level | Feature groups have defined geometric meanings |
| Main limitation | Does not explicitly count points or petals | Affected by edge quality, texture, cropping, and centering |

The motivation for combining them is **complementarity**, not an assumption that either stream is always more robust. Whether geometric features improve retrieval over DINOv2 alone is an empirical question.

## 4. Indexing and Query Processing

```text
INDEX CORPUS                             QUERY CROP
    |                                        |
    +-- OpenCV -> raw 26D features           +-- OpenCV -> raw 26D features
    |               |                        |               |
    |          fit mean, std                 |          apply INDEX
    |          on index only                 |          mean, std
    |               |                        |               |
    |          standardized 26D              |          standardized 26D
    |                                        |
    +-- DINOv2 -> L2-normalized 384D         +-- DINOv2 -> L2-normalized 384D
                     |                                        |
                     +--------------------+-------------------+
                                          |
                              Compare query to index
                                          |
                                S_shape and S_embed
                                          |
                                 Weighted fusion
                                          |
                                      Rank Top-K
```

For OpenCV features, standardization uses the **mean and standard deviation fitted on the index corpus**. The query is transformed using those same statistics; it does **not** refit the scaler. This also prevents query-data leakage during held-out evaluation.

DINOv2 embeddings are already L2-normalized by their extractor. Both streams provide a similarity score for each query–index pair.

## 5. Similarity Computation and Hybrid Fusion

**DINOv2:** For unit-normalized embeddings, cosine similarity is their dot product. The baseline maps it from `[-1, 1]` to `[0, 1]`:

$$
S_{\mathrm{embed}}(q,x) = \frac{\hat{\mathbf{e}}_q \cdot \hat{\mathbf{e}}_x + 1}{2}
$$

**OpenCV:** Geometric similarity is calculated from **standardized** 26D feature vectors using an exponential mapping of Euclidean distance:

$$
S_{\mathrm{shape}}(q,x) = \exp\!\left(-\gamma\,\lVert \mathbf{g}_q-\mathbf{g}_x\rVert_2\right)
$$

The original feature notes document `gamma = 0.25` as the similarity parameter.

**Hybrid score:**

$$
\boxed{S_{\mathrm{hybrid}}(q,x) = (1-\alpha)S_{\mathrm{embed}}(q,x)+\alpha S_{\mathrm{shape}}(q,x)}
$$

- `α = 0`: DINOv2-only retrieval.
- `α = 1`: OpenCV-only retrieval.
- `0 < α < 1`: Weighted combination of both streams.

The highest-scoring indexed examples are returned as the **Top-K** results. Although both component scores lie in `[0, 1]`, they are similarity measures, **not calibrated probabilities**. The best weighting must be determined experimentally rather than assumed.

## 6. Evaluation Overview

The original benchmark notes describe an **artifact-level split** so that examples of the same artifact do not appear on both sides of an evaluation split.

```text
Fixed artifact-level split
           |
           v
Index: motif-family examples + generic negative distractors
           |
           v
Fit OpenCV scaler on INDEX only
           |
           v
Queries: held-out motif-family examples
           |
           v
Rank the complete index for every query
           |
           v
Compare OpenCV / DINOv2 / hybrid alpha sweep
           |
           v
        Hit@1, Hit@5, MRR
```

For this retrieval benchmark, a result is considered relevant when it belongs to the **same motif class** as the query.

| Metric | Interpretation |
| --- | --- |
| **Hit@1** | Fraction of queries whose top-ranked result is relevant |
| **Hit@5** | Fraction of queries with at least one relevant result in the first five |
| **MRR** (*Mean Reciprocal Rank*) | Mean of `1 / rank` of the first relevant result (or `0` if none is retrieved) |

Detailed splits, settings, and measured results belong in the corresponding experiment and `results/` files; this note documents only the shared retrieval workflow.

## 7. Scope and Next Steps

The original OpenCV baseline does **not** directly count contour peaks or measure individual tip widths. The separate [`Geometric Feature Ablations`](geometric_feature_ablations.md) describes the **v0.2 geometric ablation experiments** exploring harmonics, peaks, pointedness, radius relationships, and central structure. Those experimental features should not be confused with the original 26D representation.

**Implementation references:** `shape_features.py` (geometric extraction), `embeddings.py` (DINOv2 extraction), and `hybrid_search.py` (retrieval integration, as described in the project notes). The source code remains authoritative for current parameters and behavior.

## 8. v0.2 — Retrieval and Classification Integration

The v0.2 prototype separates **similarity search** from **classification and rejection**, using two complementary modules:

- **`hybrid_search.py` (`MotifIndex`):** Builds and caches feature indices, computes hybrid similarity, and ranks visually similar motifs. It supports both retrieval (Pointedness, α=0.25) and classification (Harmonics, α=0.25) configurations.
- **`motif_classifier.py`:** Uses the classification index to predict motif classes through similarity-weighted KNN (K=7, artifact-deduplicated), computes confidence scores, and applies the calibrated `max_sim` threshold to reject unknown queries.

The Streamlit interface combines both pipelines: **classification determines the predicted class or `unknown`, while retrieval independently displays visually similar motifs**, regardless of the classification outcome.

See the [v0.2 Final Summary](../../results/v0.2_final_summary.md) for the complete system configuration and evaluation findings.