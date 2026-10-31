# DINOv2 Embeddings — Design & Rationale

This note summarizes the DINOv2 representation used for visual motif retrieval in **Symbolic Detective**. It describes the existing `embeddings.py` baseline, rather than providing a general introduction to Vision Transformers.

## 1. Purpose

DINOv2 provides a **learned visual representation** that allows motif crops to be compared without pixel-by-pixel matching or manually assigned motif labels. In this pipeline, it serves as a **feature extractor**, not a classifier: it returns an embedding rather than predicting a motif category or counting star points.

| Property | Implementation |
| --- | --- |
| Model | Meta DINOv2 ViT-S/14 (`dinov2_vits14`) |
| Input | Preprocessed `224 × 224` RGB crop |
| Output | 384-dimensional `float32` embedding |
| Normalization | L2 normalization to approximately unit length |
| Comparison | Cosine similarity, rescaled to `[0, 1]` |

## 2. Extraction Pipeline

```text
Motif crop
    ↓
preprocess_for_dinov2() → 224 × 224 RGB
    ↓
ToTensor() + ImageNet channel normalization
    ↓
DINOv2 ViT-S/14 (inference)
    ↓
384-dimensional visual embedding (CLS representation)
    ↓
L2 normalization
    ↓
Unit embedding for retrieval
```

The model is loaded through PyTorch Hub, cached for reuse, and run in evaluation mode on CUDA when available (otherwise CPU). Inference uses `torch.inference_mode()`; pretrained weights are **not fine-tuned** in this baseline.

L2 normalization removes embedding magnitude while retaining its direction:

$$
\hat{\mathbf{v}} = \frac{\mathbf{v}}{\lVert\mathbf{v}\rVert_2}
$$

## 3. Similarity

Because the extracted embeddings are L2-normalized, their dot product is cosine similarity:

$$
\operatorname{cosSim}(\mathbf{a},\mathbf{b}) = \hat{\mathbf{a}} \cdot \hat{\mathbf{b}}
$$

The implementation maps cosine similarity from `[-1, 1]` to `[0, 1]`:

$$
S_{\mathrm{embed}} = \frac{\operatorname{cosSim}(\mathbf{a},\mathbf{b})+1}{2}
$$

The result is clamped to `[0, 1]` to handle numerical edge cases. **This score is a similarity measure, not a probability of motif identity**; for example, `0.5` corresponds to a cosine similarity of zero.

## 4. Role in Hybrid Retrieval

DINOv2 supplies the learned-visual component of Symbolic Detective's retrieval score. The geometric component is computed separately from OpenCV features:

$$
S_{\mathrm{hybrid}} = (1-\alpha)S_{\mathrm{embed}} + \alpha S_{\mathrm{geometry}}
$$

Here, `α = 0` is the DINOv2-only baseline; larger values assign greater weight to explicit geometry. Hybrid scoring is performed elsewhere in the retrieval pipeline, not inside `embeddings.py`.

## 5. Limitations and Research Motivation

- Embedding dimensions are **latent**: individual values do not directly represent interpretable properties such as the number of points or petals.
- General visual similarity does not guarantee **fine-grained geometric discrimination**, particularly between visually related stars and rosettes.
- Results may be influenced by crop alignment, appearance, texture, and image transformations; robustness should be evaluated rather than assumed.
- The v0.2 geometry experiments investigate whether explicit structural descriptors improve discrimination beyond the DINOv2-only baseline.


For preprocessing, indexing, and score fusion details, see [Hybrid Retrieval Pipeline](hybrid_retrieval_pipeline.md).