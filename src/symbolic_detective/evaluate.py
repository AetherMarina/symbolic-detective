import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
import matplotlib.pyplot as plt

from symbolic_detective.evaluation_metrics import build_confusion_matrix, calculate_metrics
from symbolic_detective.pipeline.preprocess import preprocess_for_opencv, preprocess_for_dinov2
from symbolic_detective.pipeline.shape_features import compute_shape_features, compute_shape_similarity
from symbolic_detective.pipeline.embeddings import extract_visual_embedding, compute_embedding_similarity
from symbolic_detective.utils.image_files import collect_images, artifact_key_from_path
from symbolic_detective.pipeline.class_voting import predict_class

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data/crops"
DEFAULT_SPLIT_FILE = PROJECT_ROOT / "data/splits/v0.2_split.json"
RESULTS_DIR = PROJECT_ROOT / "results" / "v0.2_dev_baseline"


def evaluate_pipeline(data_dir: Path = DEFAULT_DATA_DIR,
                      split_file: Path = DEFAULT_SPLIT_FILE,
                      target_class: str = "eight_pointed_star",
                      k_folds: int = 5,
                      mode: str = "voting") -> None:
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
    class_names = [c.name for c in classes if c.name != "negative_control"]

    # 2. Global Feature Extraction (Filtered by Dev Split)
    print("Extracting raw features for development artifacts...")
    all_items = []
    skipped_test_crops = 0

    for cls_dir in classes:
        images = collect_images(cls_dir)
        for img in images:
            aid = artifact_key_from_path(img)

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

    # 3. Artifact-Grouped K-Fold Partitioning (Stratified)
    print("Stratifying development cross-validation folds...")

    artifact_to_class = {}
    for item in all_items:
        if item["class"] != "negative_control":
            if item["aid"] not in artifact_to_class:
                artifact_to_class[item["aid"]] = item["class"]

    queryable_aids = list(artifact_to_class.keys())
    y = [artifact_to_class[aid] for aid in queryable_aids]
    groups = queryable_aids

    sgkf = StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=42)

    folds = []
    for _, test_idx in sgkf.split(queryable_aids, y, groups):
        folds.append([queryable_aids[i] for i in test_idx])

    print(f"Dataset extracted: {len(all_items)} total crops.")
    print(f"Skipped test partition crops: {skipped_test_crops}")
    print(f"Partitioned {len(queryable_aids)} valid dev artifacts into {k_folds} folds.\n")

    alphas_to_test = [0.0, 0.25, 0.50, 0.75, 1.0]

    overall_metrics = {alpha: {"h1": [], "h5": [], "mrr": []} for alpha in alphas_to_test}
    target_metrics = {alpha: {"h1": [], "h5": [], "mrr": []} for alpha in alphas_to_test}

    y_true_all = {alpha: [] for alpha in alphas_to_test}
    y_pred_all = {alpha: [] for alpha in alphas_to_test}

    if mode in ["voting", "all"]:
        voting_configs = [("top1", 1)] + [(s, k) for s in ["majority", "weighted"] for k in [3, 5, 7]]
        voting_alphas = [0.0, 0.25]
        voting_fold_metrics = {(a, s, k): {"acc": [], "macro_f1": []} for a in voting_alphas for s, k in voting_configs}
        voting_y_true_all = {a: [] for a in voting_alphas}
        voting_y_pred_all = {(a, s, k): [] for a in voting_alphas for s, k in voting_configs}
        voting_error_analysis = []

    total_dropped_queries = 0
    total_evaluated_queries = 0
    total_target_queries = 0
    error_analysis = []

    print("Development dataset validation passed.")

    # 4. K-Fold Evaluation Loop
    for fold_idx, query_aids in enumerate(folds):
        query_aids_set = set(query_aids)

        corpus_raw = [item for item in all_items if item["aid"] not in query_aids_set]
        queries_raw = [item for item in all_items if
                       item["aid"] in query_aids_set and item["class"] != "negative_control"]

        corpus_aids = {item["aid"] for item in corpus_raw}
        query_artifact_ids = {item["aid"] for item in queries_raw}
        overlap = corpus_aids & query_artifact_ids
        if overlap:
            raise RuntimeError(f"Artifact leakage detected in fold {fold_idx + 1}: {sorted(overlap)}")

        scaler = StandardScaler()
        corpus_shapes = [item["raw_shape"] for item in corpus_raw]
        scaler.fit(corpus_shapes)

        corpus_scaled = []
        for item in corpus_raw:
            corpus_scaled.append({
                "class": item["class"],
                "path": str(item["path"]),
                "aid": item["aid"],
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
                    "aid": item["aid"],
                    "shape_vec": scaler.transform([item["raw_shape"]])[0],
                    "dino_vec": item["dino_vec"]
                })
            else:
                dropped_this_fold += 1

        total_dropped_queries += dropped_this_fold
        total_evaluated_queries += len(queries_scaled)

        fold_overall = {alpha: {"h1": [], "h5": [], "mrr": []} for alpha in alphas_to_test}
        fold_target = {alpha: {"h1": [], "h5": [], "mrr": []} for alpha in alphas_to_test}

        if mode in ["voting", "all"]:
            fold_voting_y_true = {a: [] for a in voting_alphas}
            fold_voting_y_pred = {(a, s, k): [] for a in voting_alphas for s, k in voting_configs}

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
                    "aid": doc["aid"],
                    "s_shape": s_shape,
                    "s_embed": s_embed
                })

            top_results = {}

            for alpha in alphas_to_test:
                for s in scores:
                    s["s_hybrid"] = (alpha * s["s_shape"]) + ((1.0 - alpha) * s["s_embed"])

                sorted_scores = sorted(scores, key=lambda x: x["s_hybrid"], reverse=True)

                top_1_class = sorted_scores[0]["class"]

                # ONLY record these metrics for baseline mode if requested
                if mode in ["baseline", "all"]:
                    y_true_all[alpha].append(q["class"])
                    y_pred_all[alpha].append(top_1_class)

                if alpha in (0.0, 0.25):
                    top_results[alpha] = sorted_scores[0]

                if mode in ["voting", "all"] and alpha in voting_alphas:
                    fold_voting_y_true[alpha].append(q["class"])
                    voting_y_true_all[alpha].append(q["class"])
                    query_preds = {}

                    for strat, k_val in voting_configs:
                        predicted_class = predict_class(
                            rankings=sorted_scores,
                            k=k_val,
                            strategy=strat,
                            score_key="s_hybrid",
                            dedup=True
                        )
                        fold_voting_y_pred[(alpha, strat, k_val)].append(predicted_class)
                        voting_y_pred_all[(alpha, strat, k_val)].append(predicted_class)
                        query_preds[f"{strat}_k{k_val}"] = predicted_class

                    if any(p != q["class"] for p in query_preds.values()):
                        voting_error_analysis.append({
                            "query": f"{q['class']}/{Path(q['path']).name}",
                            "true_class": q["class"],
                            "fold": fold_idx + 1,
                            "alpha": alpha,
                            "voting_predictions": query_preds
                        })

                h1, h5, mrr = calculate_metrics(sorted_scores, q["class"])
                fold_overall[alpha]["h1"].append(h1)
                fold_overall[alpha]["h5"].append(h5)
                fold_overall[alpha]["mrr"].append(mrr)

                if q["class"] == target_class:
                    fold_target[alpha]["h1"].append(h1)
                    fold_target[alpha]["h5"].append(h5)
                    fold_target[alpha]["mrr"].append(mrr)

            if mode in ["baseline", "all"]:
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

        for alpha in alphas_to_test:
            if fold_overall[alpha]["h1"]:
                overall_metrics[alpha]["h1"].append(np.mean(fold_overall[alpha]["h1"]))
                overall_metrics[alpha]["h5"].append(np.mean(fold_overall[alpha]["h5"]))
                overall_metrics[alpha]["mrr"].append(np.mean(fold_overall[alpha]["mrr"]))

            if fold_target[alpha]["h1"]:
                target_metrics[alpha]["h1"].append(np.mean(fold_target[alpha]["h1"]))
                target_metrics[alpha]["h5"].append(np.mean(fold_target[alpha]["h5"]))
                target_metrics[alpha]["mrr"].append(np.mean(fold_target[alpha]["mrr"]))

        if mode in ["voting", "all"]:
            for a in voting_alphas:
                if len(fold_voting_y_true[a]) > 0:
                    for strat, k_val in voting_configs:
                        cfg = (a, strat, k_val)
                        acc = accuracy_score(fold_voting_y_true[a], fold_voting_y_pred[cfg])
                        f1 = f1_score(fold_voting_y_true[a], fold_voting_y_pred[cfg], labels=class_names,
                                      average='macro',
                                      zero_division=0)
                        voting_fold_metrics[cfg]["acc"].append(acc)
                        voting_fold_metrics[cfg]["macro_f1"].append(f1)

    if mode in ["voting", "all"]:
        assert voting_y_pred_all[(0.25, "top1", 1)] == y_pred_all[
            0.25], "Voting Top-1 differs from the original Hybrid baseline"
        assert voting_y_pred_all[(0.0, "top1", 1)] == y_pred_all[
            0.0], "Voting Top-1 differs from the original DINOv2 baseline"

    # 6. Aggregate & Display Results
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

    # 7. Discrimination Metrics & Confusion Matrices
    if mode in ["baseline", "all"]:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        figs_dir = RESULTS_DIR / "figures"
        figs_dir.mkdir(parents=True, exist_ok=True)

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
            disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, values_format="d")
            ax.set_title(f"Motif Confusion Matrix — {method} (α={alpha})")
            plt.tight_layout()
            path = figs_dir / f"confusion_{method.lower()}.png"
            fig.savefig(path, dpi=300)
            plt.close(fig)
            print(f"Saved: {path}")

    # 8. Class Voting Experiment Results
    if mode in ["voting", "all"]:
        print("\n### Class Voting Experiment (Deduplicated Artifacts)")
        print(
            f"| {'Alpha':<5} | {'Strategy':<10} | {'K':<3} | {'Accuracy (Mean ± SD)':<22} | {'Macro F1 (Mean ± SD)':<22} |")
        print("|" + "-" * 7 + "|" + "-" * 12 + "|" + "-" * 5 + "|" + "-" * 24 + "|" + "-" * 24 + "|")

        voting_export_data = []

        for a in voting_alphas:
            for strat, k_val in voting_configs:
                cfg = (a, strat, k_val)
                acc_m = np.mean(voting_fold_metrics[cfg]["acc"])
                acc_s = np.std(voting_fold_metrics[cfg]["acc"])
                f1_m = np.mean(voting_fold_metrics[cfg]["macro_f1"])
                f1_s = np.std(voting_fold_metrics[cfg]["macro_f1"])

                pool_acc = accuracy_score(voting_y_true_all[a], voting_y_pred_all[cfg])
                pool_f1 = f1_score(voting_y_true_all[a], voting_y_pred_all[cfg], labels=class_names, average='macro',
                                   zero_division=0)

                print(
                    f"| {a:<5} | {strat:<10} | {k_val:<3} | {acc_m:.4f} ± {acc_s:.4f}       | {f1_m:.4f} ± {f1_s:.4f}       |")

                report_dict=classification_report(
                    voting_y_true_all[a],
                    voting_y_pred_all[cfg],
                    labels=class_names,
                    target_names=class_names,
                    output_dict=True,
                    zero_division=0,
                )

                per_class_metrics = {
                    cls: {
                        "precision": float(report_dict[cls]["precision"]),
                        "recall": float(report_dict[cls]["recall"]),
                        "f1-score": float(report_dict[cls]["f1-score"]),
                        "support": int(report_dict[cls]["support"])
                    }
                    for cls in class_names
                }

                voting_export_data.append({
                    "alpha": a,
                    "strategy": strat,
                    "k": k_val,
                    "metrics": {
                        "cv_accuracy_mean": float(acc_m),
                        "cv_accuracy_std": float(acc_s),
                        "cv_macro_f1_mean": float(f1_m),
                        "cv_macro_f1_std": float(f1_s),
                        "pooled_accuracy": float(pool_acc),
                        "pooled_macro_f1": float(pool_f1),
                        "per_class_metrics": per_class_metrics
                    }
                })

    # 9. Exports
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if mode in ["baseline", "all"]:
        error_path = RESULTS_DIR / "error_analysis.json"
        with open(error_path, "w", encoding="utf-8") as f:
            json.dump(error_analysis, f, indent=2)
        print(f"\nSaved per-query error analysis to: {error_path}")

    if mode in ["voting", "all"]:
        voting_path = RESULTS_DIR / "class_voting.json"
        with open(voting_path, "w", encoding="utf-8") as f:
            json.dump(voting_export_data, f, indent=2)

        voting_error_path = RESULTS_DIR / "voting_error_analysis.json"
        with open(voting_error_path, "w", encoding="utf-8") as f:
            json.dump(voting_error_analysis, f, indent=2)

        print(f"Saved class voting metrics to: {voting_path}")
        print(f"Saved voting error analysis to: {voting_error_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Symbolic Detective Retrieval with K-Fold CV (Dev Set Only).")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="Path to cropped images directory")
    parser.add_argument("--split-file", type=str, default=DEFAULT_SPLIT_FILE, help="Path to split manifest JSON")
    parser.add_argument("--k-folds", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--mode", choices=["baseline", "voting", "all"], default="all",
                        help="Evaluation mode: baseline retrieval and/or class voting experiment")
    args = parser.parse_args()

    evaluate_pipeline(
        Path(args.data_dir),
        Path(args.split_file),
        k_folds=args.k_folds,
        mode=args.mode
    )
