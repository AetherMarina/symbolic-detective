"""
Hybrid Search Engine for Symbolic Detective v0.1.
Implements a formal indexing and retrieval architecture with corpus-level
standardization and robust failure handling.
"""
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

from symbolic_detective.pipeline.preprocess import preprocess_for_opencv, preprocess_for_dinov2
from symbolic_detective.pipeline.shape_features import compute_shape_features, compute_shape_similarity
from symbolic_detective.pipeline.embeddings import extract_visual_embedding, compute_embedding_similarity


class MotifIndex:
    def __init__(self):
        self.corpus: List[Dict[str, Any]] = []
        self.shape_mean: Optional[np.ndarray] = None
        self.shape_std: Optional[np.ndarray] = None

    def build_index(self, image_paths: List[Path]) -> None:
        """
        Extracts features for all images, drops failures, and computes
        corpus-level standardization statistics for the shape vectors.
        """
        raw_corpus = []

        for path in image_paths:
            cv_img = preprocess_for_opencv(path)
            dino_img = preprocess_for_dinov2(path)

            if cv_img is None or dino_img is None:
                continue

            shape_feats = compute_shape_features(cv_img)
            dino_feats = extract_visual_embedding(dino_img)

            # Strict failure handling: only index items with complete dual-stream features
            if shape_feats is not None and dino_feats is not None:
                raw_corpus.append({
                    "path": path,
                    "raw_shape": shape_feats["feature_vector"],
                    "dino_vec": dino_feats
                })

        if not raw_corpus:
            raise ValueError("No valid features could be extracted to build the index.")

        # Compute corpus-level statistics for OpenCV standardizaton
        shape_matrix = np.array([item["raw_shape"] for item in raw_corpus], dtype=np.float32)
        self.shape_mean = shape_matrix.mean(axis=0)
        self.shape_std = shape_matrix.std(axis=0) + 1e-8  # epsilon to prevent division by zero

        # Standardize index vectors immediately
        self.corpus = []
        for item in raw_corpus:
            scaled_shape = (item["raw_shape"] - self.shape_mean) / self.shape_std
            self.corpus.append({
                "path": item["path"],
                "shape_vec": scaled_shape,
                "dino_vec": item["dino_vec"]
            })

    def search(self, query_path: Path, alpha: float = 0.5, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Executes a Top-K retrieval against the standardized index.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Alpha must be between 0.0 and 1.0.")

        if not self.corpus:
            raise RuntimeError("Index is empty. Call build_index() first.")

        # Extract query features
        cv_img = preprocess_for_opencv(query_path)
        dino_img = preprocess_for_dinov2(query_path)

        if cv_img is None or dino_img is None:
            raise ValueError(f"Failed to load or preprocess query image: {query_path}")

        q_shape = compute_shape_features(cv_img)
        q_embed = extract_visual_embedding(dino_img)

        if q_shape is None or q_embed is None:
            raise ValueError("Failed to extract complete feature set from the query image.")

        # Standardize query shape using the index's frozen statistics
        q_shape_scaled = (q_shape["feature_vector"] - self.shape_mean) / self.shape_std

        # Compute similarities
        results = []
        for item in self.corpus:
            s_shape = compute_shape_similarity(q_shape_scaled, item["shape_vec"])
            s_embed = compute_embedding_similarity(q_embed, item["dino_vec"])

            s_hybrid = (alpha * s_shape) + ((1.0 - alpha) * s_embed)

            results.append({
                "path": item["path"],
                "s_shape": s_shape,
                "s_embed": s_embed,
                "s_hybrid": s_hybrid
            })

        results.sort(key=lambda x: x["s_hybrid"], reverse=True)
        return results[:top_k]


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m symbolic_detective.pipeline.hybrid_search <query_image> <alpha>")
        sys.exit(1)

    query = Path(sys.argv[1])
    alpha_val = float(sys.argv[2])

    # Quick mock initialization for terminal testing
    crop_dir = Path("data/crops")
    all_crops = [path for path in crop_dir.rglob("*.jpg") if path.resolve() != query.resolve()]

    print(f"Building index from {len(all_crops)} images...")
    index = MotifIndex()
    index.build_index(all_crops)

    print(f"\nSearching for {query.name} (Alpha={alpha_val})...")
    top_matches = index.search(query, alpha=alpha_val, top_k=5)

    for i, res in enumerate(top_matches, start=1):
        print(f"{i}. {res['path'].name} | Hybrid: {res['s_hybrid']:.4f} (Shape: {res['s_shape']:.2f},"
              f" Embed: {res['s_embed']:.2f})")
