"""
Benchmark & Evaluation Script for Symbolic Detective v0.1.
Calculates Hit@1, Hit@5, and MRR for OpenCV, DINOv2, and Hybrid Search.
"""
import argparse
import random
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.preprocessing import StandardScaler

from symbolic_detective.pipeline.preprocess import preprocess_for_opencv, preprocess_for_dinov2
from symbolic_detective.pipeline.shape_features import compute_shape_features, compute_shape_similarity
from symbolic_detective.pipeline.embeddings import extract_visual_embedding, compute_embedding_similarity

random.seed(42)


def calculate_metrics(rankings: List[Dict[str, Any]], true_class: str) -> Tuple[int, int, float]:
    """
    Computes Hit@1, Hit@5, and Reciprocal Rank for ranked retrieval outputs.
    """
    hit_1 = 1 if rankings[0]["class"] == true_class else 0
    hit_5 = 1 if any(r["class"] == true_class for r in rankings[:5]) else 0

    mrr = 0.0
    for rank, item in enumerate(rankings, start=1):
        if item["class"] == true_class:
            mrr = 1.0 / rank
            break

    return hit_1, hit_5, mrr


def evaluate_pipeline(data_dir: Path) -> None:
    print("--- Starting Motif Retrieval Benchmark ---")

    # -------------------------------------------------------------------------
    # 1. Deterministic Artifact-Level Partitioning
    # -------------------------------------------------------------------------
    classes = [d for d in data_dir.iterdir() if d.is_dir()]
    splits_file = data_dir / "splits.json"

    artifact_splits: Dict[str, str] = {}
    if splits_file.exists():
        with open(splits_file, "r", encoding="utf-8") as f:
            artifact_splits = json.load(f)
    else:
        print("No splits.json found. Generating deterministic artifact-level splits...")
        for cls_dir in classes:
            if cls_dir.name == "negative_control":
                continue

            images = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
            # Sort before shuffle to ensure identical splits across different Python runs
            artifact_ids = sorted({img.stem.split("_")[0] for img in images})
            random.shuffle(artifact_ids)

            split_idx = int(len(artifact_ids) * 0.8)
            for aid in artifact_ids[:split_idx]:
                artifact_splits[aid] = "index"
            for aid in artifact_ids[split_idx:]:
                artifact_splits[aid] = "query"

        with open(splits_file, "w", encoding="utf-8") as f:
            json.dump(artifact_splits, f, indent=4)

    raw_corpus_items = []
    raw_query_items = []

    for cls_dir in classes:
        images = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
        for img in images:
            aid = img.stem.split("_")[0]

            # Negative controls serve strictly as corpus distractors
            if cls_dir.name == "negative_control":
                raw_corpus_items.append({"path": img, "class": cls_dir.name})
            elif artifact_splits.get(aid) == "index":
                raw_corpus_items.append({"path": img, "class": cls_dir.name})
            elif artifact_splits.get(aid) == "query":
                raw_query_items.append({"path": img, "class": cls_dir.name})

    if not raw_corpus_items:
        raise RuntimeError("No candidate artifacts found for index corpus.")

    # -------------------------------------------------------------------------
    # 2. Extract Features for Corpus
    # -------------------------------------------------------------------------
    print(f"\nExtracting features for {len(raw_corpus_items)} candidate corpus items...")
    indexed_corpus = []
    for item in raw_corpus_items:
        cv_img = preprocess_for_opencv(item["path"])
        dino_img = preprocess_for_dinov2(item["path"])

        if cv_img is None or dino_img is None:
            continue

        shape_f = compute_shape_features(cv_img)
        dino_f = extract_visual_embedding(dino_img)

        if shape_f is not None and dino_f is not None:
            indexed_corpus.append({
                "class": item["class"],
                "path": item["path"],
                "shape_vec": shape_f["feature_vector"],
                "dino_vec": dino_f
            })

    # Validate that indexed items exist and record classes that actually survived extraction
    if not indexed_corpus:
        raise RuntimeError("No valid indexed features were extracted.")

    actual_index_classes = {doc["class"] for doc in indexed_corpus}

    # Filter queries: only retain queries whose class actually exists in the built index
    query_items = [q for q in raw_query_items if q["class"] in actual_index_classes]
    if not query_items:
        raise RuntimeError("No valid queries have a matching class in the indexed corpus.")

    print(f"Corpus successfully indexed: {len(indexed_corpus)} items across classes: {actual_index_classes}")
    print(f"Valid queries to evaluate: {len(query_items)} items")

    # Fit StandardScaler strictly on the surviving indexed corpus
    shape_scaler = StandardScaler()
    corpus_shape_vecs = [doc["shape_vec"] for doc in indexed_corpus]
    shape_scaler.fit(corpus_shape_vecs)

    for doc in indexed_corpus:
        doc["shape_vec"] = shape_scaler.transform([doc["shape_vec"]])[0]

    # -------------------------------------------------------------------------
    # 3. Query Execution & Score Calibration Audit
    # -------------------------------------------------------------------------
    print(f"\nProcessing {len(query_items)} queries...")
    query_results = []
    all_s_shape = []
    all_s_embed = []

    for q in query_items:
        q_cv = preprocess_for_opencv(q["path"])
        q_dino = preprocess_for_dinov2(q["path"])

        if q_cv is None or q_dino is None:
            continue

        q_shape = compute_shape_features(q_cv)
        q_embed = extract_visual_embedding(q_dino)

        if not q_shape or q_embed is None:
            continue

        q_shape_scaled = shape_scaler.transform([q_shape["feature_vector"]])[0]

        scores = []
        for doc in indexed_corpus:
            s_shape = compute_shape_similarity(q_shape_scaled, doc["shape_vec"])
            s_embed = compute_embedding_similarity(q_embed, doc["dino_vec"])

            all_s_shape.append(s_shape)
            all_s_embed.append(s_embed)

            scores.append({
                "class": doc["class"],
                "s_shape": s_shape,
                "s_embed": s_embed
            })

        query_results.append({"true_class": q["class"], "scores": scores})

    if not query_results:
        raise RuntimeError("All queries failed feature extraction.")

    # Score calibration audit
    print("\n--- Score Calibration Audit ---")
    print(f"S_shape Range: {np.min(all_s_shape):.3f} to {np.max(all_s_shape):.3f} (Mean: {np.mean(all_s_shape):.3f})")
    print(f"S_embed Range: {np.min(all_s_embed):.3f} to {np.max(all_s_embed):.3f} (Mean: {np.mean(all_s_embed):.3f})\n")

    # -------------------------------------------------------------------------
    # 4. Exploratory Alpha Sweep
    # -------------------------------------------------------------------------
    alphas_to_test = [0.0, 0.25, 0.50, 0.75, 1.0]

    print("### Exploratory Alpha Sweep (Validation Set)")
    print("| **Method** | **Alpha (OpenCV Weight)** | **Hit@1** | **Hit@5** | **MRR** |")
    print("|---|---|---|---|---|")

    for alpha in alphas_to_test:
        h1_list, h5_list, mrr_list = [], [], []

        for q_res in query_results:
            for s in q_res["scores"]:
                s["s_hybrid"] = (alpha * s["s_shape"]) + ((1.0 - alpha) * s["s_embed"])

            q_res["scores"].sort(key=lambda x: x["s_hybrid"], reverse=True)
            h1, h5, mrr = calculate_metrics(q_res["scores"], q_res["true_class"])

            h1_list.append(h1)
            h5_list.append(h5)
            mrr_list.append(mrr)

        if alpha == 0.0:
            method_name = "Latent Embeddings (DINOv2)"
        elif alpha == 1.0:
            method_name = "OpenCV Geometric Features"
        else:
            method_name = "Hybrid Ensemble"

        print(f"| {method_name} | {alpha:.2f} | {np.mean(h1_list):.3f} | {np.mean(h5_list):.3f} | {np.mean(mrr_list):.3f} |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Symbolic Detective Retrieval.")
    parser.add_argument("--data-dir", type=str, default="data/crops", help="Path to cropped images directory")
    args = parser.parse_args()

    evaluate_pipeline(Path(args.data_dir))
