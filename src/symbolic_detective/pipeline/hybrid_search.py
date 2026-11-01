"""
Hybrid Search Engine.
Builds an in-memory index of motif features, compares a query image
with every indexed image using geometric and DINOv2 similarity,
and returns the highest-scoring matches.
"""
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

from symbolic_detective.pipeline.preprocess import preprocess_for_opencv, preprocess_for_dinov2
from symbolic_detective.pipeline.shape_features import compute_shape_features, compute_shape_similarity
from symbolic_detective.pipeline.embeddings import MODEL_NAME, extract_visual_embedding, compute_embedding_similarity
from symbolic_detective.utils.image_files import collect_images

INDEX_VERSION = "1"
FEATURE_VERSION = "geometry-v02-exp01"


class MotifIndex:
    def __init__(self):
        self.corpus: List[Dict[str, Any]] = []
        self.shape_mean: Optional[np.ndarray] = None
        self.shape_std: Optional[np.ndarray] = None
        self.index_version = INDEX_VERSION
        self.feature_version = FEATURE_VERSION
        self.dino_model = MODEL_NAME
        self.corpus_fingerprint = ""

    @staticmethod
    def compute_fingerprint(image_paths: List[Path]) -> str:
        """
        Generates a lightweight, deterministic hash of the corpus based on
        file paths, sizes, and modification timestamps.
        """
        hasher = hashlib.md5()
        for path in sorted(image_paths):
            try:
                stat = path.stat()
                identifier = f"{path.as_posix()}|{stat.st_size}|{stat.st_mtime_ns}\n"
                hasher.update(identifier.encode("utf-8"))
            except FileNotFoundError:
                continue
        return hasher.hexdigest()

    def build_index(self, image_paths: List[Path]) -> None:
        """
        Extracts features for all images, drops failures, and computes
        corpus-level standardization statistics for the shape vectors.
        """
        raw_corpus = []

        # Compute and store the fingerprint for this exact corpus state
        self.corpus_fingerprint = self.compute_fingerprint(image_paths)

        for path in image_paths:
            cv_img = preprocess_for_opencv(path)
            dino_img = preprocess_for_dinov2(path)

            if cv_img is None or dino_img is None:
                continue

            shape_feats = compute_shape_features(cv_img)
            dino_feats = extract_visual_embedding(dino_img)

            if shape_feats is not None and dino_feats is not None:
                raw_corpus.append({
                    "path": path,
                    "raw_shape": shape_feats["feature_vector"],
                    "dino_vec": dino_feats
                })

        if not raw_corpus:
            raise ValueError("No valid features could be extracted to build the index.")

        shape_matrix = np.array([item["raw_shape"] for item in raw_corpus], dtype=np.float32)
        self.shape_mean = shape_matrix.mean(axis=0)
        self.shape_std = shape_matrix.std(axis=0) + 1e-8

        self.corpus = []
        for item in raw_corpus:
            scaled_shape = (item["raw_shape"] - self.shape_mean) / self.shape_std
            self.corpus.append({
                "path": item["path"],
                "shape_vec": scaled_shape,
                "dino_vec": item["dino_vec"]
            })

    def save(self, filepath: Path) -> None:
        """Saves the index, statistics, and metadata to a compressed NumPy file."""
        if not self.corpus:
            raise RuntimeError("Cannot save an empty index.")

        paths = np.array([str(item["path"]) for item in self.corpus])
        shape_vecs = np.array([item["shape_vec"] for item in self.corpus], dtype=np.float32)
        dino_vecs = np.array([item["dino_vec"] for item in self.corpus], dtype=np.float32)

        filepath.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            filepath,
            paths=paths,
            shape_vecs=shape_vecs,
            dino_vecs=dino_vecs,
            shape_mean=self.shape_mean,
            shape_std=self.shape_std,
            index_version=np.array([self.index_version]),
            feature_version=np.array([self.feature_version]),
            dino_model=np.array([self.dino_model]),
            corpus_fingerprint=np.array([self.corpus_fingerprint])
        )

    def load(self, filepath: Path, expected_fingerprint: Optional[str] = None) -> None:
        """Loads a persisted motif index and corpus statistics."""

        if not filepath.exists():
            raise FileNotFoundError(f"Index file not found: {filepath}")

        with np.load(filepath, allow_pickle=False) as data:
            self.shape_mean = data["shape_mean"].astype(np.float32)
            self.shape_std = data["shape_std"].astype(np.float32)

            if data["index_version"][0] != self.index_version:
                raise RuntimeError(f"Incompatible index version: {data['index_version'][0]} != {self.index_version}")

            if data["feature_version"][0] != self.feature_version:
                raise RuntimeError(f"Incompatible feature version: {data['feature_version'][0]} != {self.feature_version}")

            if data["dino_model"][0] != self.dino_model:
                raise RuntimeError(f"Incompatible DINO model: {data['dino_model'][0]} != {self.dino_model}")

            if expected_fingerprint is not None:
                if "corpus_fingerprint" not in data or data["corpus_fingerprint"][0] != expected_fingerprint:
                    raise RuntimeError("Corpus fingerprint mismatch. Collection has been modified.")

            self.corpus_fingerprint = data.get("corpus_fingerprint", [""])[0]

            self.corpus = []
            for p, s_vec, d_vec in zip(data['paths'], data['shape_vecs'], data['dino_vecs']):
                self.corpus.append({
                    "path": Path(p),
                    "shape_vec": s_vec,
                    "dino_vec": d_vec
                })

    def load_or_build(self, filepath: Path, image_paths: List[Path]) -> bool:
        """
        Loads a valid persisted index or rebuilds it when the corpus
        or index configuration has changed.

        Returns True if the index was rebuilt.
        """
        current_fingerprint = self.compute_fingerprint(image_paths)

        if filepath.exists():
            try:
                self.load(filepath, expected_fingerprint=current_fingerprint,)
                return False
            except (RuntimeError, KeyError, ValueError, OSError):
                pass

        self.build_index(image_paths)
        self.save(filepath)
        return True

    def search(self, query_path: Path, alpha: float = 0.5, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Executes a Top-K retrieval. Reuses cached features if the query is
        already in the index, and excludes the query itself from the results.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Alpha must be between 0.0 and 1.0.")
        if not self.corpus:
            raise RuntimeError("Index is empty.")

        q_shape_scaled = None
        q_embed = None

        # 1. Check if query is already indexed (Feature Reuse)
        resolved_query_path = query_path.resolve()
        for item in self.corpus:
            if item["path"].resolve() == resolved_query_path:
                q_shape_scaled = item["shape_vec"]
                q_embed = item["dino_vec"]
                break

        # 2. If new, extract and scale dynamically
        if q_shape_scaled is None or q_embed is None:
            cv_img = preprocess_for_opencv(query_path)
            dino_img = preprocess_for_dinov2(query_path)

            if cv_img is None or dino_img is None:
                raise ValueError(f"Failed to load query image: {query_path}")

            q_shape = compute_shape_features(cv_img)
            q_embed_raw = extract_visual_embedding(dino_img)

            if q_shape is None or q_embed_raw is None:
                raise ValueError("Failed to extract complete feature set from the query image.")

            q_shape_scaled = (q_shape["feature_vector"] - self.shape_mean) / self.shape_std
            q_embed = q_embed_raw

        # 3. Compute similarities
        results = []
        for item in self.corpus:
            # Exclude self-matches
            if item["path"].resolve() == resolved_query_path:
                continue

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

    crop_dir = Path("data/crops")
    index_file = Path("data/index.npz")
    index = MotifIndex()

    # Pre-gather paths to compute the current fingerprint
    all_crops = collect_images(crop_dir, recursive=True)

    if index_file.exists():
        rebuilt = index.load_or_build(index_file, all_crops)
        if rebuilt:
            print(f"Built and saved index with {len(index.corpus)} motifs.")
        else:
            print(f"Loaded cached index with {len(index.corpus)} motifs.")

    print(f"\nSearching for {query.name} (Alpha={alpha_val})...")
    top_matches = index.search(query, alpha=alpha_val, top_k=5)

    for i, res in enumerate(top_matches, start=1):
        print(f"{i}. {res['path'].name} | Hybrid: {res['s_hybrid']:.4f} (Shape: {res['s_shape']:.2f},"
              f" Embed: {res['s_embed']:.2f})")
