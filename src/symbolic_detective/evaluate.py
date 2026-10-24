"""
Benchmark & Evaluation Script (K-Fold Cross-Validation).
Calculates Hit@1, Hit@5, and MRR for OpenCV, DINOv2, and Hybrid Search
using artifact-grouped k-fold cross validation on a strictly defined development split.
Includes per-query error analysis and multi-class confusion matrices.
"""
import argparse
import random
import json
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from symbolic_detective.evaluation_metrics import build_confusion_matrix, calculate_metrics
from symbolic_detective.pipeline.preprocess import preprocess_for_opencv, preprocess_for_dinov2
from symbolic_detective.pipeline.shape_features import compute_shape_features, compute_shape_similarity
from symbolic_detective.pipeline.embeddings import extract_visual_embedding, compute_embedding_similarity
from symbolic_detective.utils.image_files import collect_images, artifact_key_from_path

random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data/crops"
DEFAULT_SPLIT_FILE = PROJECT_ROOT / "data/splits/v0.2_split.json"


def evaluate_pipeline(data_dir: Path = DEFAULT_DATA_DIR,
                      split_file: Path = DEFAULT_SPLIT_FILE,
                      target_class: str = "eight_pointed_star",
                      k_folds: int = 5) -> None:

    print(f"--- Starting {k_folds}-Fold Motif Retrieval Benchmark (Dev Split Only) ---")

    # 1. Load Dev Split Manifest
    if not split_file.exists():
        raise FileNotFoundError(f"Split manifest not found at {split_file}. Run create_split.py first.")

    with open(split_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    dev_aids = set(manifest["partitions"]["dev"])
    test_aids = set(manifest["partitions"]["test"])

    overlap = dev_aids & test_aids

    if overlap:
        raise RuntimeError(
            f"Dev/test artifact leakage detected: {len(overlap)} "
            f"overlapping artifacts: {sorted(overlap)}"
        )

    print(f"Loaded {len(dev_aids)} allowed artifacts from {split_file.name}")

    classes = sorted(
        d for d in data_dir.iterdir()
        if d.is_dir() and d.name not in {"unclassified"}
    )

    # 2. Global Feature Extraction (Filtered by Dev Split)
    print("Extracting raw features for development artifacts...")
    all_items = []
    skipped_test_crops = 0

    for cls_dir in classes:
        images = collect_images(cls_dir)
        for img in images:
            aid = artifact_key_from_path(img)

            # --- STRICT LEAKAGE PREVENTION ---
            if aid not in dev_aids:
                skipped_test_crops += 1
                continue

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

    extracted_aids = {item["aid"] for item in all_items}
    unexpected_aids = extracted_aids - dev_aids
    if unexpected_aids:
        raise RuntimeError(
            f"Non-development artifacts detected in evaluation: "
            f"{sorted(unexpected_aids)}"
        )

    # 3. Artifact-Grouped K-Fold Partitioning
    queryable_aids = sorted({item["aid"] for item in all_items if item["class"] != "negative_control"})
    random.shuffle(queryable_aids)

    folds = [queryable_aids[i::k_folds] for i in range(k_folds)]

    print(f"Dataset extracted: {len(all_items)} total crops.")
    print(f"Skipped test partition crops: {skipped_test_crops}")
    print(f"Partitioned {len(queryable_aids)} valid dev artifacts into {k_folds} folds.\n")

    alphas_to_test = [0.0, 0.25, 0.50, 0.75, 1.0]

    # Data structures to aggregate fold-level means
    overall_metrics = {alpha: {"h1": [], "h5": [], "mrr": []} for alpha in alphas_to_test}
    target_metrics = {alpha: {"h1": [], "h5": [], "mrr": []} for alpha in alphas_to_test}

    # Initialize global prediction trackers for Confusion Matrix
    y_true_all = {alpha: [] for alpha in alphas_to_test}
    y_pred_all = {alpha: [] for alpha in alphas_to_test}

    total_dropped_queries = 0
    total_evaluated_queries = 0
    total_target_queries = 0
    error_analysis = []

    print("Development dataset validation passed.")

    # 4. K-Fold Evaluation Loop
    for fold_idx, query_aids in enumerate(folds):
        query_aids_set = set(query_aids)

        # Strictly partition data to prevent substrate leakage
        corpus_raw = [item for item in all_items if item["aid"] not in query_aids_set]
        queries_raw = [item for item in all_items if
                       item["aid"] in query_aids_set and item["class"] != "negative_control"]

        corpus_aids = {item["aid"] for item in corpus_raw}
        query_artifact_ids = {item["aid"] for item in queries_raw}
        overlap = corpus_aids & query_artifact_ids
        if overlap:
            raise RuntimeError(
                f"Artifact leakage detected in fold {fold_idx + 1}: "
                f"{sorted(overlap)}"
            )

        scaler = StandardScaler()
        corpus_shapes = [item["raw_shape"] for item in corpus_raw]
        scaler.fit(corpus_shapes)

        corpus_scaled = []
        for item in corpus_raw:
            corpus_scaled.append({
                "class": item["class"],
                "path": str(item["path"]),
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
                    "path": str(item["path"]),
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
                scores.append({
                    "class": doc["class"],
                    "path": doc["path"],
                    "s_shape": s_shape,
                    "s_embed": s_embed
                })

            top_results = {}

            for alpha in alphas_to_test:
                for s in scores:
                    s["s_hybrid"] = (alpha * s["s_shape"]) + ((1.0 - alpha) * s["s_embed"])

                sorted_scores = sorted(scores, key=lambda x: x["s_hybrid"], reverse=True)

                # Record true vs. predicted class globally
                top_1_class = sorted_scores[0]["class"]
                y_true_all[alpha].append(q["class"])
                y_pred_all[alpha].append(top_1_class)

                if alpha in (0.0, 0.25):
                    top_results[alpha] = sorted_scores[0]

                h1, h5, mrr = calculate_metrics(sorted_scores, q["class"])

                fold_overall[alpha]["h1"].append(h1)
                fold_overall[alpha]["h5"].append(h5)
                fold_overall[alpha]["mrr"].append(mrr)

                if q["class"] == target_class:
                    fold_target[alpha]["h1"].append(h1)
                    fold_target[alpha]["h5"].append(h5)
                    fold_target[alpha]["mrr"].append(mrr)

            # Error analysis across all motif classes
            dino = top_results[0.0]
            hybrid = top_results[0.25]

            dino_correct = dino["class"] == q["class"]
            hybrid_correct = hybrid["class"] == q["class"]

            if not dino_correct and hybrid_correct:
                outcome = "geometry_helped"
            elif dino_correct and not hybrid_correct:
                outcome = "geometry_hurt"
            elif dino_correct and hybrid_correct:
                outcome = "both_correct"
            else:
                outcome = "both_wrong"

            # Record only queries misclassified by at least one method
            if outcome != "both_correct":
                error_analysis.append({
                    "query": f"{q['class']}/{Path(q['path']).name}",
                    "true_class": q["class"],
                    "fold": fold_idx + 1,
                    "outcome": outcome,
                    "dino_top1": f"{dino['class']}/{Path(dino['path']).name}",
                    "dino_class": dino["class"],
                    "hybrid_top1": f"{hybrid['class']}/{Path(hybrid['path']).name}",
                    "hybrid_class": hybrid["class"]
                })

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

    # 5. Aggregate & Display Results (Mean ± Std Dev Across Folds)
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

        print(
            f"| {method_name} | {alpha:.2f}  |  {h1_m:.3f} ± {h1_s:.3f}  |  {h5_m:.3f} ± {h5_s:.3f}  |  {mrr_m:.3f} ± {mrr_s:.3f}  |")

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

            if target_metrics[alpha]["h1"]:
                t_h1_m, t_h1_s = np.mean(target_metrics[alpha]["h1"]), np.std(target_metrics[alpha]["h1"])
                t_h5_m, t_h5_s = np.mean(target_metrics[alpha]["h5"]), np.std(target_metrics[alpha]["h5"])
                t_mrr_m, t_mrr_s = np.mean(target_metrics[alpha]["mrr"]), np.std(target_metrics[alpha]["mrr"])

                print(
                    f"| {method_name} | {alpha:.2f}  |  {t_h1_m:.3f} ± {t_h1_s:.3f}  |  {t_h5_m:.3f} ± {t_h5_s:.3f}  |  {t_mrr_m:.3f} ± {t_mrr_s:.3f}  |")
    else:
        print(f"\n⚠️ Target class '{target_class}' not found in query results.")

    # 6. Discrimination Metrics & Confusion Matrices
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    class_names = [cls.name for cls in classes if cls.name != "negative_control"]

    for alpha in (0.0, 0.25):
        method = "DINOv2" if alpha == 0.0 else "Hybrid"
        print(f"\n### Multi-Class Discrimination Report ({method}, α={alpha})")
        report = classification_report(
            y_true_all[alpha],
            y_pred_all[alpha],
            labels=class_names,
            target_names=class_names,
            zero_division=0,
        )
        print(report)

        cm, cm_labels = build_confusion_matrix(
            y_true_all[alpha],
            y_pred_all[alpha],
            class_names,
        )

        fig, ax = plt.subplots(figsize=(11, 9))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=cm_labels,
        )
        disp.plot(
            ax=ax,
            cmap="Blues",
            xticks_rotation=45,
            values_format="d",
        )
        ax.set_title(f"Motif Confusion Matrix — {method} (α={alpha})")
        plt.tight_layout()
        path = results_dir / f"v0.2_confusion_{method.lower()}.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f"Saved: {path}")

    # 7. Export Error Analysis
    output_path = PROJECT_ROOT / "results" / "v0.2_error_analysis.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(error_analysis, f, indent=2)
    print(f"\nSaved per-query error analysis to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Symbolic Detective Retrieval with K-Fold CV (Dev Set Only).")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="Path to cropped images directory")
    parser.add_argument("--split-file", type=str, default=DEFAULT_SPLIT_FILE, help="Path to split manifest JSON")
    parser.add_argument("--k-folds", type=int, default=5, help="Number of folds for cross-validation")
    args = parser.parse_args()

    evaluate_pipeline(Path(args.data_dir), Path(args.split_file), k_folds=args.k_folds)
