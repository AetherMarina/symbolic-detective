"""
Benchmark & Evaluation Script (K-Fold Cross-Validation).
Calculates Hit@1, Hit@5, and MRR for OpenCV, DINOv2, and Hybrid Search
using artifact-grouped k-fold cross validation.
"""
import argparse
import random
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.preprocessing import StandardScaler

from symbolic_detective.pipeline.preprocess import preprocess_for_opencv, preprocess_for_dinov2
from symbolic_detective.pipeline.shape_features import compute_shape_features, compute_shape_similarity
from symbolic_detective.pipeline.embeddings import extract_visual_embedding, compute_embedding_similarity
from symbolic_detective.utils.image_files import collect_images

random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data/crops"


def calculate_metrics(rankings: List[Dict[str, Any]], true_class: str) -> Tuple[int, int, float]:
    """Computes Hit@1, Hit@5, and Reciprocal Rank for ranked retrieval outputs."""
    hit_1 = 1 if rankings[0]["class"] == true_class else 0
    hit_5 = 1 if any(r["class"] == true_class for r in rankings[:5]) else 0

    reciprocal_rank = 0.0
    for rank, item in enumerate(rankings, start=1):
        if item["class"] == true_class:
            reciprocal_rank = 1.0 / rank
            break

    return hit_1, hit_5, reciprocal_rank


def artifact_key_from_path(path: Path) -> str:
    """
    Extracts the source-qualified artifact key robustly.
    Handles '__occ_', '_occ_', and '_primary' delimiters safely without
    breaking internal underscores in museum IDs.
    """
    parts = re.split(r'__occ_|_occ_|_primary', path.stem)
    return parts[0]


def evaluate_pipeline(data_dir: Path = DEFAULT_DATA_DIR, target_class: str = "eight_pointed_star", k_folds: int = 5) -> None:
    print(f"--- Starting {k_folds}-Fold Motif Retrieval Benchmark ---")

    classes = sorted([d for d in data_dir.iterdir() if d.is_dir()], key=lambda p: p.name)

    # 1. Global Feature Extraction
    print("Extracting raw features for the entire dataset...")
    all_items = []

    for cls_dir in classes:
        images = collect_images(cls_dir)
        for img in images:
            aid = artifact_key_from_path(img)
            cv_img = preprocess_for_opencv(img)
            dino_img = preprocess_for_dinov2(img)

            if cv_img is None or dino_img is None:
                continue

            shape_f = compute_shape_features(cv_img)
            dino_f = extract_visual_embedding(dino_img)

            if shape_f is not None and dino_f is not None:
                all_items.append({
                    "path": img,
                    "class": cls_dir.name,
                    "aid": aid,
                    "raw_shape": shape_f["feature_vector"],
                    "dino_vec": dino_f
                })
            else:
                print(f"Failed extraction on: {img.name}")

    if not all_items:
        raise RuntimeError("No valid features were extracted from the dataset.")

    # 2. Artifact-Grouped K-Fold Partitioning
    queryable_aids = sorted({item["aid"] for item in all_items if item["class"] != "negative_control"})
    random.shuffle(queryable_aids)

    folds = [queryable_aids[i::k_folds] for i in range(k_folds)]

    print(f"Dataset extracted: {len(all_items)} total crops.")
    print(f"Partitioned {len(queryable_aids)} valid artifacts into {k_folds} folds.\n")

    alphas_to_test = [0.0, 0.25, 0.50, 0.75, 1.0]

    # Data structures to aggregate fold-level means
    overall_metrics = {alpha: {"h1": [], "h5": [], "mrr": []} for alpha in alphas_to_test}
    target_metrics = {alpha: {"h1": [], "h5": [], "mrr": []} for alpha in alphas_to_test}

    total_dropped_queries = 0
    total_evaluated_queries = 0
    total_target_queries = 0

    # 3. K-Fold Evaluation Loop
    for fold_idx, query_aids in enumerate(folds):
        query_aids_set = set(query_aids)

        # Strictly partition data to prevent substrate leakage
        corpus_raw = [item for item in all_items if item["aid"] not in query_aids_set]
        queries_raw = [item for item in all_items if item["aid"] in query_aids_set and item["class"] != "negative_control"]

        scaler = StandardScaler()
        corpus_shapes = [item["raw_shape"] for item in corpus_raw]
        scaler.fit(corpus_shapes)

        corpus_scaled = []
        for item in corpus_raw:
            corpus_scaled.append({
                "class": item["class"],
                "shape_vec": scaler.transform([item["raw_shape"]])[0],
                "dino_vec": item["dino_vec"]
            })

        valid_index_classes = {doc["class"] for doc in corpus_scaled}

        queries_scaled = []
        dropped_this_fold = 0
        for item in queries_raw:
            if item["class"] in valid_index_classes:
                queries_scaled.append({
                    "class": item["class"],
                    "shape_vec": scaler.transform([item["raw_shape"]])[0],
                    "dino_vec": item["dino_vec"]
                })
            else:
                dropped_this_fold += 1

        total_dropped_queries += dropped_this_fold
        total_evaluated_queries += len(queries_scaled)

        # Temporary structures for this specific fold
        fold_overall = {alpha: {"h1": [], "h5": [], "mrr": []} for alpha in alphas_to_test}
        fold_target = {alpha: {"h1": [], "h5": [], "mrr": []} for alpha in alphas_to_test}

        # Evaluate Fold
        for q in queries_scaled:
            if q["class"] == target_class:
                total_target_queries += 1

            scores = []
            for doc in corpus_scaled:
                s_shape = compute_shape_similarity(q["shape_vec"], doc["shape_vec"])
                s_embed = compute_embedding_similarity(q["dino_vec"], doc["dino_vec"])
                scores.append({"class": doc["class"], "s_shape": s_shape, "s_embed": s_embed})

            for alpha in alphas_to_test:
                for s in scores:
                    s["s_hybrid"] = (alpha * s["s_shape"]) + ((1.0 - alpha) * s["s_embed"])

                sorted_scores = sorted(scores, key=lambda x: x["s_hybrid"], reverse=True)
                h1, h5, mrr = calculate_metrics(sorted_scores, q["class"])

                fold_overall[alpha]["h1"].append(h1)
                fold_overall[alpha]["h5"].append(h5)
                fold_overall[alpha]["mrr"].append(mrr)

                if q["class"] == target_class:
                    fold_target[alpha]["h1"].append(h1)
                    fold_target[alpha]["h5"].append(h5)
                    fold_target[alpha]["mrr"].append(mrr)

        # Aggregate fold-level means
        for alpha in alphas_to_test:
            if fold_overall[alpha]["h1"]:
                overall_metrics[alpha]["h1"].append(np.mean(fold_overall[alpha]["h1"]))
                overall_metrics[alpha]["h5"].append(np.mean(fold_overall[alpha]["h5"]))
                overall_metrics[alpha]["mrr"].append(np.mean(fold_overall[alpha]["mrr"]))

            if fold_target[alpha]["h1"]:
                target_metrics[alpha]["h1"].append(np.mean(fold_target[alpha]["h1"]))
                target_metrics[alpha]["h5"].append(np.mean(fold_target[alpha]["h5"]))
                target_metrics[alpha]["mrr"].append(np.mean(fold_target[alpha]["mrr"]))

    # 4. Aggregate & Display Results (Mean ± Std Dev Across Folds)
    print(f"Total queries evaluated: {total_evaluated_queries}")
    if total_dropped_queries > 0:
        print(f"⚠️ Queries excluded (missing from index): {total_dropped_queries}")

    print("\n### Cross-Validation Results (Overall Multi-Class)")
    print("|         **Method**         | Alpha |      Hit@1      |      Hit@5      |       MRR       |")
    print("|----------------------------|-------|-----------------|-----------------|-----------------|")

    for alpha in alphas_to_test:
        if alpha == 0.0:
            method_name = "Latent Embeddings (DINOv2)"
        elif alpha == 1.0:
            method_name = "OpenCV Geometric Features "
        else:
            method_name = "Hybrid Ensemble           "

        h1_m, h1_s = np.mean(overall_metrics[alpha]["h1"]), np.std(overall_metrics[alpha]["h1"])
        h5_m, h5_s = np.mean(overall_metrics[alpha]["h5"]), np.std(overall_metrics[alpha]["h5"])
        mrr_m, mrr_s = np.mean(overall_metrics[alpha]["mrr"]), np.std(overall_metrics[alpha]["mrr"])

        print(f"| {method_name} | {alpha:.2f}  |  {h1_m:.3f} ± {h1_s:.3f}  |  {h5_m:.3f} ± {h5_s:.3f}  |  {mrr_m:.3f} ± {mrr_s:.3f}  |")

    if total_target_queries > 0:
        print(f"\n### Cross-Validation Results (Target: {target_class} - {total_target_queries} queries evaluated)")
        print("|         **Method**         | Alpha |      Hit@1      |      Hit@5      |       MRR       |")
        print("|----------------------------|-------|-----------------|-----------------|-----------------|")

        for alpha in alphas_to_test:
            if alpha == 0.0:
                method_name = "Latent Embeddings (DINOv2)"
            elif alpha == 1.0:
                method_name = "OpenCV Geometric Features "
            else:
                method_name = "Hybrid Ensemble           "

            # Check if this alpha has target metric entries (prevents NaN warnings if a fold had 0 target queries)
            if target_metrics[alpha]["h1"]:
                t_h1_m, t_h1_s = np.mean(target_metrics[alpha]["h1"]), np.std(target_metrics[alpha]["h1"])
                t_h5_m, t_h5_s = np.mean(target_metrics[alpha]["h5"]), np.std(target_metrics[alpha]["h5"])
                t_mrr_m, t_mrr_s = np.mean(target_metrics[alpha]["mrr"]), np.std(target_metrics[alpha]["mrr"])

                print(f"| {method_name} | {alpha:.2f}  |  {t_h1_m:.3f} ± {t_h1_s:.3f}  |  {t_h5_m:.3f} ± {t_h5_s:.3f}  |  {t_mrr_m:.3f} ± {t_mrr_s:.3f}  |")
    else:
        print(f"\n⚠️ Target class '{target_class}' not found in query results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Symbolic Detective Retrieval with K-Fold CV.")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="Path to cropped images directory")
    parser.add_argument("--k-folds", type=int, default=5, help="Number of folds for cross-validation")
    args = parser.parse_args()

    evaluate_pipeline(Path(args.data_dir), k_folds=args.k_folds)
