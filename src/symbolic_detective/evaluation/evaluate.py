import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
import matplotlib.pyplot as plt

from symbolic_detective.evaluation.evaluation_metrics import build_confusion_matrix, calculate_metrics
from symbolic_detective.pipeline.class_voting import predict_class, deduplicate_rankings
from symbolic_detective.evaluation.evaluation_common import (
    load_split_manifest,
    extract_evaluation_features,
    scale_evaluation_features,
    rank_candidates
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data/crops"
DEFAULT_SPLIT_FILE = PROJECT_ROOT / "data/splits/v0.2_split.json"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "v0.2_dev_baseline"


def evaluate_pipeline(data_dir: Path = DEFAULT_DATA_DIR,
                      split_file: Path = DEFAULT_SPLIT_FILE,
                      target_class: str = "eight_pointed_star",
                      k_folds: int = 5,
                      mode: str = "all",
                      results_dir: Path | None = None,
                      feature_config: dict | None = None) -> None:
    print(f"--- Starting {k_folds}-Fold Motif Retrieval Benchmark (Dev Split Only) ---")

    if results_dir is None:
        results_dir = DEFAULT_RESULTS_DIR
    if feature_config is None:
        feature_config = {}

    # 1. Load Dev Split Manifest
    dev_aids, test_aids = load_split_manifest(split_file)
    print(f"Loaded {len(dev_aids)} allowed artifacts from {split_file.name}")

    # 2. Global Feature Extraction (Filtered by Dev Split)
    active_features = [name for name, enabled in (feature_config or {}).items() if enabled]
    print(f"Active geometric features: {active_features or 'baseline only'}")
    print("Extracting raw features for development artifacts...")

    conf_key = "shape_vec"
    feature_configs = {conf_key: feature_config}

    all_items, class_names_full = extract_evaluation_features(data_dir, dev_aids, feature_configs)
    class_names = [c for c in class_names_full if c != "negative_control"]

    if not all_items:
        raise RuntimeError("No valid features were extracted from the dataset.")

    extracted_aids = {item["aid"] for item in all_items}
    unexpected_aids = extracted_aids - dev_aids
    if unexpected_aids:
        raise RuntimeError(f"Non-development artifacts detected in evaluation: {sorted(unexpected_aids)}")

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
    folds = [[queryable_aids[i] for i in test_idx] for _, test_idx in sgkf.split(queryable_aids, y, groups)]

    print(f"Dataset extracted: {len(all_items)} total valid crops.")
    print(f"Partitioned {len(queryable_aids)} valid dev artifacts into {k_folds} folds.\\n")

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
        voting_predictions_records = []

    total_evaluated_queries = 0
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

        corpus_scaled, queries_scaled = scale_evaluation_features(
            corpus_raw, queries_raw, [conf_key]
        )
        total_evaluated_queries += len(queries_scaled)

        fold_overall = {alpha: {"h1": [], "h5": [], "mrr": []} for alpha in alphas_to_test}
        fold_target = {alpha: {"h1": [], "h5": [], "mrr": []} for alpha in alphas_to_test}

        if mode in ["voting", "all"]:
            fold_voting_y_true = {a: [] for a in voting_alphas}
            fold_voting_y_pred = {(a, s, k): [] for a in voting_alphas for s, k in voting_configs}

        for q in queries_scaled:
            top_results = {}
            for alpha in alphas_to_test:
                sorted_scores = rank_candidates(q, corpus_scaled, alpha, conf_key)
                top_1_class = sorted_scores[0]["class"]

                if mode in ["baseline", "all"]:
                    y_true_all[alpha].append(q["class"])
                    y_pred_all[alpha].append(top_1_class)

                if alpha in (0.0, 0.25):
                    top_results[alpha] = sorted_scores[0]

                if mode in ["voting", "all"] and alpha in voting_alphas:
                    fold_voting_y_true[alpha].append(q["class"])
                    voting_y_true_all[alpha].append(q["class"])
                    query_preds = {}

                    deduped_rankings = deduplicate_rankings(sorted_scores)
                    top_7_neighbors = [
                        {"artifact_id": n["aid"], "class": n["class"], "rank": r, "similarity": float(n["s_hybrid"])}
                        for r, n in enumerate(deduped_rankings[:7], start=1)]

                    for strat, k_val in voting_configs:
                        predicted_class = predict_class(
                            rankings=sorted_scores, k=k_val, strategy=strat, score_key="s_hybrid", dedup=True
                        )
                        fold_voting_y_pred[(alpha, strat, k_val)].append(predicted_class)
                        voting_y_pred_all[(alpha, strat, k_val)].append(predicted_class)
                        query_preds[f"{strat}_k{k_val}"] = predicted_class

                        voting_predictions_records.append({
                            "query_id": str(Path(q["path"]).relative_to(data_dir)),
                            "artifact_id": q["aid"],
                            "fold": fold_idx,
                            "true_class": q["class"],
                            "alpha": alpha,
                            "strategy": strat,
                            "k": k_val,
                            "predicted_class": predicted_class,
                            "neighbors": top_7_neighbors
                        })

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
                                      average='macro', zero_division=0)
                        voting_fold_metrics[cfg]["acc"].append(acc)
                        voting_fold_metrics[cfg]["macro_f1"].append(f1)

    if mode == "all":
        assert voting_y_pred_all[(0.25, "top1", 1)] == y_pred_all[
            0.25], "Voting Top-1 differs from the original Hybrid baseline"
        assert voting_y_pred_all[(0.0, "top1", 1)] == y_pred_all[
            0.0], "Voting Top-1 differs from the original DINOv2 baseline"

    # 6. Aggregate & Display Results
    print(f"Total queries evaluated: {total_evaluated_queries}")

    print("\\n### Cross-Validation Results (Overall Multi-Class)")
    print("|         **Method**         | Alpha |      Hit@1      |      Hit@5      |       MRR       |")
    print("|----------------------------|-------|-----------------|-----------------|-----------------|")

    for alpha in alphas_to_test:
        method_name = "Latent Embeddings (DINOv2)" if alpha == 0.0 else "OpenCV Geometric Features " if alpha == 1.0 else "Hybrid Ensemble           "
        h1_m, h1_s = np.mean(overall_metrics[alpha]["h1"]), np.std(overall_metrics[alpha]["h1"])
        h5_m, h5_s = np.mean(overall_metrics[alpha]["h5"]), np.std(overall_metrics[alpha]["h5"])
        mrr_m, mrr_s = np.mean(overall_metrics[alpha]["mrr"]), np.std(overall_metrics[alpha]["mrr"])
        print(
            f"| {method_name} | {alpha:.2f}  |  {h1_m:.3f} ± {h1_s:.3f}  |  {h5_m:.3f} ± {h5_s:.3f}  |  {mrr_m:.3f} ± {mrr_s:.3f}  |")

    # 7. Discrimination Metrics & Confusion Matrices
    if mode in ["baseline", "all"]:
        results_dir.mkdir(parents=True, exist_ok=True)
        figs_dir = results_dir / "figures"
        figs_dir.mkdir(parents=True, exist_ok=True)

        for alpha in (0.0, 0.25):
            method = "DINOv2" if alpha == 0.0 else "Hybrid"
            print(f"\\n### Multi-Class Discrimination Report ({method}, α={alpha})")
            text_report = classification_report(y_true_all[alpha], y_pred_all[alpha], labels=class_names,
                                                target_names=class_names, zero_division=0)
            print(text_report)

            cm, cm_labels = build_confusion_matrix(y_true_all[alpha], y_pred_all[alpha], class_names)
            fig, ax = plt.subplots(figsize=(11, 9))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cm_labels)
            disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, values_format="d")
            ax.set_title(f"Motif Confusion Matrix — {method} (α={alpha})")
            plt.tight_layout()
            path = figs_dir / f"confusion_{method.lower()}.png"
            fig.savefig(path, dpi=300)
            plt.close(fig)

    # 8. Class Voting Experiment Results
    if mode in ["voting", "all"]:
        print("\\n### Class Voting Experiment (Deduplicated Artifacts)")
        print(
            f"| {'Alpha':<5} | {'Strategy':<10} | {'K':<3} | {'Accuracy (Mean ± SD)':<22} | {'Macro F1 (Mean ± SD)':<22} |")
        print("|" + "-" * 7 + "|" + "-" * 12 + "|" + "-" * 5 + "|" + "-" * 24 + "|" + "-" * 24 + "|")

        voting_export_data = []

        for a in voting_alphas:
            for strat, k_val in voting_configs:
                cfg = (a, strat, k_val)
                acc_m, acc_s = np.mean(voting_fold_metrics[cfg]["acc"]), np.std(voting_fold_metrics[cfg]["acc"])
                f1_m, f1_s = np.mean(voting_fold_metrics[cfg]["macro_f1"]), np.std(voting_fold_metrics[cfg]["macro_f1"])

                pool_acc = accuracy_score(voting_y_true_all[a], voting_y_pred_all[cfg])
                pool_f1 = f1_score(voting_y_true_all[a], voting_y_pred_all[cfg], labels=class_names, average='macro',
                                   zero_division=0)

                print(
                    f"| {a:<5} | {strat:<10} | {k_val:<3} | {acc_m:.4f} ± {acc_s:.4f}       | {f1_m:.4f} ± {f1_s:.4f}       |")

                report_dict = classification_report(voting_y_true_all[a], voting_y_pred_all[cfg], labels=class_names,
                                                    target_names=class_names, output_dict=True, zero_division=0)
                per_class_metrics = {cls: {"precision": float(report_dict[cls]["precision"]),
                                           "recall": float(report_dict[cls]["recall"]),
                                           "f1-score": float(report_dict[cls]["f1-score"]),
                                           "support": int(report_dict[cls]["support"])} for cls in class_names}

                voting_export_data.append({
                    "alpha": a,
                    "strategy": strat,
                    "k": k_val,
                    "metrics": {
                        "cv_accuracy_mean": float(acc_m), "cv_accuracy_std": float(acc_s),
                        "cv_macro_f1_mean": float(f1_m), "cv_macro_f1_std": float(f1_s),
                        "pooled_accuracy": float(pool_acc), "pooled_macro_f1": float(pool_f1),
                        "per_class_metrics": per_class_metrics
                    }
                })

    # 9. Exports
    results_dir.mkdir(parents=True, exist_ok=True)
    if mode in ["baseline", "all"]:
        with open(results_dir / "error_analysis.json", "w", encoding="utf-8") as f:
            json.dump(error_analysis, f, indent=2)

        retrieval_metrics = {"experiment": results_dir.name,
                             "cross_validation": {"overall": [], "target_eight_pointed_star": []},
                             "multi_class_discrimination": {"dinov2": {}, "hybrid_025": {}}}

        # Populate CV metrics
        for alpha in alphas_to_test:
            retrieval_metrics["cross_validation"]["overall"].append({
                "alpha": alpha,
                "hit_at_1": {"mean": float(np.mean(overall_metrics[alpha]["h1"])),
                             "std": float(np.std(overall_metrics[alpha]["h1"])),
                             "folds": [float(x) for x in overall_metrics[alpha]["h1"]]},
                "hit_at_5": {"mean": float(np.mean(overall_metrics[alpha]["h5"])),
                             "std": float(np.std(overall_metrics[alpha]["h5"])),
                             "folds": [float(x) for x in overall_metrics[alpha]["h5"]]},
                "mrr": {"mean": float(np.mean(overall_metrics[alpha]["mrr"])),
                        "std": float(np.std(overall_metrics[alpha]["mrr"])),
                        "folds": [float(x) for x in overall_metrics[alpha]["mrr"]]}
            })

            if target_metrics[alpha]["h1"]:
                retrieval_metrics["cross_validation"]["target_eight_pointed_star"].append({
                    "alpha": alpha,
                    "hit_at_1": {"mean": float(np.mean(target_metrics[alpha]["h1"])),
                                 "std": float(np.std(target_metrics[alpha]["h1"])),
                                 "folds": [float(x) for x in target_metrics[alpha]["h1"]]},
                    "hit_at_5": {"mean": float(np.mean(target_metrics[alpha]["h5"])),
                                 "std": float(np.std(target_metrics[alpha]["h5"])),
                                 "folds": [float(x) for x in target_metrics[alpha]["h5"]]},
                    "mrr": {"mean": float(np.mean(target_metrics[alpha]["mrr"])),
                            "std": float(np.std(target_metrics[alpha]["mrr"])),
                            "folds": [float(x) for x in target_metrics[alpha]["mrr"]]}
                })

        for alpha in (0.0, 0.25):
            method_key = "dinov2" if alpha == 0.0 else "hybrid_025"
            report_dict = classification_report(y_true_all[alpha], y_pred_all[alpha], labels=class_names,
                                                target_names=class_names, output_dict=True, zero_division=0)
            cm, cm_labels = build_confusion_matrix(y_true_all[alpha], y_pred_all[alpha], class_names)
            retrieval_metrics["multi_class_discrimination"][method_key] = {
                "per_class": {cls: report_dict[cls] for cls in class_names},
                "micro_avg": report_dict.get("accuracy", report_dict.get("micro avg")),
                "macro_avg": report_dict["macro avg"], "weighted_avg": report_dict["weighted avg"],
                "confusion_matrix": cm.tolist(), "confusion_matrix_labels": cm_labels}

        with open(results_dir / "retrieval_metrics.json", "w", encoding="utf-8") as f:
            json.dump(retrieval_metrics, f, indent=2)

    if mode in ["voting", "all"]:
        with open(results_dir / "class_voting.json", "w", encoding="utf-8") as f: json.dump(voting_export_data, f,
                                                                                            indent=2)
        with open(results_dir / "voting_error_analysis.json", "w", encoding="utf-8") as f: json.dump(
            voting_error_analysis, f, indent=2)
        with open(results_dir / "voting_predictions.json", "w", encoding="utf-8") as f: json.dump(
            voting_predictions_records, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Symbolic Detective Retrieval with K-Fold CV (Dev Set Only).")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="Path to cropped images directory")
    parser.add_argument("--split-file", type=str, default=DEFAULT_SPLIT_FILE, help="Path to split manifest JSON")
    parser.add_argument("--k-folds", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--mode", choices=["baseline", "voting", "all"], default="all", help="Evaluation mode")
    parser.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR, help="Path to output directory")
    args = parser.parse_args()

    evaluate_pipeline(Path(args.data_dir), Path(args.split_file), k_folds=args.k_folds, mode=args.mode, results_dir=Path(args.results_dir))
