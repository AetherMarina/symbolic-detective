"""
Hybrid Retrieval Logic for Symbolic Detective v0.1.
Calculates visual similarity between two crops using a weighted ensemble
of explicit OpenCV geometric features and dense DINOv2 latent embeddings.
"""
from pathlib import Path
import numpy as np
import sys

from symbolic_detective.pipeline.preprocess import preprocess_for_opencv, preprocess_for_dinov2
from symbolic_detective.pipeline.shape_features import compute_shape_features, compute_shape_similarity
from symbolic_detective.pipeline.embeddings import extract_visual_embedding, compute_embedding_similarity


def compare_motifs(img_a_path: Path, img_b_path: Path, alpha: float = 0.5):
    """
    Compares two motif crops and returns their hybrid similarity score.
    alpha: Weight given to OpenCV Shape (1-alpha is given to DINOv2).
    """
    print(f"\n--- Comparing: {img_a_path.name} <-> {img_b_path.name} ---")

    # 1. OpenCV Stream
    cv_a = preprocess_for_opencv(img_a_path)
    cv_b = preprocess_for_opencv(img_b_path)

    shape_sim = 0.0
    if cv_a is not None and cv_b is not None:
        feat_a = compute_shape_features(cv_a)
        feat_b = compute_shape_features(cv_b)

        if feat_a and feat_b:
            shape_sim = compute_shape_similarity(feat_a["feature_vector"], feat_b["feature_vector"])
            print(f"S_shape (OpenCV):  {shape_sim:.4f}")
        else:
            print("S_shape (OpenCV):  FAILED (Could not extract valid contours)")

    # 2. DINOv2 Stream
    dino_a = preprocess_for_dinov2(img_a_path)
    dino_b = preprocess_for_dinov2(img_b_path)

    embed_sim = 0.0
    if dino_a is not None and dino_b is not None:
        emb_a = extract_visual_embedding(dino_a)
        emb_b = extract_visual_embedding(dino_b)

        if emb_a is not None and emb_b is not None:
            embed_sim = compute_embedding_similarity(emb_a, emb_b)
            print(f"S_embed (DINOv2):  {embed_sim:.4f}")
        else:
            print("S_embed (DINOv2):  FAILED")

    # 3. Hybrid Score Calculation
    # S_hybrid = a(S_shape) + (1-a)(S_embed)
    hybrid_score = (alpha * shape_sim) + ((1.0 - alpha) * embed_sim)

    print(f"\nS_hybrid (α={alpha:.2f}): {hybrid_score:.4f}")
    return hybrid_score


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m symbolic_detective.pipeline.hybrid_search <image_1> <image_2> [alpha]")
        sys.exit(1)

    img1 = Path(sys.argv[1])
    img2 = Path(sys.argv[2])
    alpha_val = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5

    compare_motifs(img1, img2, alpha=alpha_val)
