"""
Benchmark & Evaluation Script for Symbolic Detective v0.1.
Calculates Recall@1, Recall@5, and MRR for OpenCV, DINOv2, and Hybrid Search.
"""
import argparse
import random
from pathlib import Path
import numpy as np

from symbolic_detective.pipeline.preprocess import preprocess_for_opencv, preprocess_for_dinov2
from symbolic_detective.pipeline.shape_features import compute_shape_features, compute_shape_similarity
from symbolic_detective.pipeline.embeddings import extract_visual_embedding, compute_embedding_similarity

random.seed(42)


def calculate_metrics(rankings, true_class):
    """
    Given a sorted list of retrieved items (dicts with 'class'),
    calculate if the true_class was found at Rank 1, within Rank 5, and the Reciprocal Rank.
    """
    recall_1 = 1 if rankings[0]['class'] == true_class else 0
    recall_5 = 1 if any(r['class'] == true_class for r in rankings[:5]) else 0

    mrr = 0.0
    for rank, item in enumerate(rankings, start=1):
        if item['class'] == true_class:
            mrr = 1.0 / rank
            break

    return recall_1, recall_5, mrr


def evaluate_pipeline(data_dir: Path, alpha: float):
    print(f"--- Starting Evaluation (Alpha = {alpha:.2f}) ---")

    # 1. Load and Split Data
    classes = [d for d in data_dir.iterdir() if d.is_dir() and d.name != "negative_control"]

    corpus_items = []
    query_items = []

    for cls_dir in classes:
        images = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
        random.shuffle(images)  # Shuffle for random held-out set

        split_idx = int(len(images) * 0.8)  # 80% Index, 20% Queries

        for img in images[:split_idx]:
            corpus_items.append({"path": img, "class": cls_dir.name})

        for img in images[split_idx:]:
            query_items.append({"path": img, "class": cls_dir.name})

    print(f"Dataset split: {len(corpus_items)} Corpus Indexed | {len(query_items)} Held-out Queries")

    # 2. Extract Features for Corpus
    print("\nExtracting features for indexed corpus...")
    indexed_corpus = []
    for item in corpus_items:
        cv_img = preprocess_for_opencv(item['path'])
        dino_img = preprocess_for_dinov2(item['path'])

        if cv_img is None or dino_img is None: continue

        shape_f = compute_shape_features(cv_img)
        dino_f = extract_visual_embedding(dino_img)

        if shape_f and dino_f is not None:
            indexed_corpus.append({
                "class": item["class"],
                "path": item["path"],
                "shape_vec": shape_f["feature_vector"],
                "dino_vec": dino_f
            })

    # 3. Run Queries and Accumulate Metrics
    print(f"Evaluating {len(query_items)} queries...\n")

    results = {"shape": {"r1": [], "r5": [], "mrr": []},
               "embed": {"r1": [], "r5": [], "mrr": []},
               "hybrid": {"r1": [], "r5": [], "mrr": []}}

    for q in query_items:
        q_cv = preprocess_for_opencv(q['path'])
        q_dino = preprocess_for_dinov2(q['path'])
        if q_cv is None or q_dino is None: continue

        q_shape = compute_shape_features(q_cv)
        q_embed = extract_visual_embedding(q_dino)
        if not q_shape or q_embed is None: continue

        # Compare against corpus
        scores = []
        for doc in indexed_corpus:
            s_shape = compute_shape_similarity(q_shape["feature_vector"], doc["shape_vec"])
            s_embed = compute_embedding_similarity(q_embed, doc["dino_vec"])
            s_hybrid = (alpha * s_shape) + ((1.0 - alpha) * s_embed)

            scores.append({
                "class": doc["class"],
                "s_shape": s_shape,
                "s_embed": s_embed,
                "s_hybrid": s_hybrid
            })

        # Rank by Shape
        scores.sort(key=lambda x: x["s_shape"], reverse=True)
        r1, r5, mrr = calculate_metrics(scores, q["class"])
        results["shape"]["r1"].append(r1);
        results["shape"]["r5"].append(r5);
        results["shape"]["mrr"].append(mrr)

        # Rank by Embed
        scores.sort(key=lambda x: x["s_embed"], reverse=True)
        r1, r5, mrr = calculate_metrics(scores, q["class"])
        results["embed"]["r1"].append(r1);
        results["embed"]["r5"].append(r5);
        results["embed"]["mrr"].append(mrr)

        # Rank by Hybrid
        scores.sort(key=lambda x: x["s_hybrid"], reverse=True)
        r1, r5, mrr = calculate_metrics(scores, q["class"])
        results["hybrid"]["r1"].append(r1);
        results["hybrid"]["r5"].append(r5);
        results["hybrid"]["mrr"].append(mrr)

    # 4. Print Markdown Table (Formatted for README)
    print("| **Method** | **Recall@1** | **Recall@5** | **MRR** |")
    print("|---|---|---|---|")
    for method, name in [("shape", "OpenCV Shape Moments"), ("embed", "Latent Embeddings (DINOv2)"),
                         ("hybrid", f"Hybrid Ensemble (α={alpha:.1f})")]:
        avg_r1 = np.mean(results[method]["r1"])
        avg_r5 = np.mean(results[method]["r5"])
        avg_mrr = np.mean(results[method]["mrr"])
        print(f"| {name} | {avg_r1:.3f} | {avg_r5:.3f} | {avg_mrr:.3f} |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Symbolic Detective Retrieval.")
    parser.add_argument("--data-dir", type=str, default="data/crops", help="Path to cropped images")
    parser.add_argument("--alpha", type=float, default=0.2, help="Hybrid weight for OpenCV")
    args = parser.parse_args()

    evaluate_pipeline(Path(args.data_dir), args.alpha)
