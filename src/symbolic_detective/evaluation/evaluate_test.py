import json
from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt

from symbolic_detective.evaluation.configs import FEATURE_CONFIGS
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
RESULTS_DIR = PROJECT_ROOT / "results" / "v0.2_test_final"


def evaluate_test_set(data_dir: Path = DEFAULT_DATA_DIR, split_file: Path = DEFAULT_SPLIT_FILE):
    print("=== Running FINAL HELD-OUT Evaluation ===")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Data Splits
    dev_aids, test_aids = load_split_manifest(split_file)

    feature_configs = FEATURE_CONFIGS

    # 2. Extract and Scale Features
    print("\\nExtracting features for the Gallery (Dev) split...")
    gallery_items, class_names = extract_evaluation_features(data_dir, dev_aids, feature_configs)

    print("Extracting features for the Queries (Test) split...")
    query_items, _ = extract_evaluation_features(data_dir, test_aids, feature_configs,
                                                 exclude_classes={"negative_control"})

    print(f"\\nGallery (Dev corpus):  {len(gallery_items)} crops")
    print(f"Queries (Test set):    {len(query_items)} crops\\n")

    if not gallery_items or not query_items:
        raise RuntimeError("Missing data for gallery or queries.")

    gallery_scaled, queries_scaled = scale_evaluation_features(
        gallery_items, query_items, list(feature_configs.keys())
    )

    # 3. Models Setup
    models = {
        "dinov2_only": {
            "title": "Baseline (DINOv2 Only, Maj K=3)",
            "alpha": 0.0, "strategy": "majority", "k": 3, "shape_key": "raw_shape_base",
            "y_pred": [], "h1": [], "h5": [], "mrr": []
        },
        "hybrid_base": {
            "title": "Baseline Hybrid (Base Shape, W K=7)",
            "alpha": 0.25, "strategy": "weighted", "k": 7, "shape_key": "raw_shape_base",
            "y_pred": [], "h1": [], "h5": [], "mrr": []
        },
        "hybrid_harmonics": {
            "title": "Hybrid (Harmonics, W K=7)",
            "alpha": 0.25, "strategy": "weighted", "k": 7, "shape_key": "raw_shape_harm",
            "y_pred": [], "h1": [], "h5": [], "mrr": []
        },
        "hybrid_pointedness": {
            "title": "Hybrid (Pointedness, Top-1)",
            "alpha": 0.25, "strategy": "top1", "k": 1, "shape_key": "raw_shape_point",
            "y_pred": [], "h1": [], "h5": [], "mrr": []
        }
    }

    y_true = []
    hybrid_predictions_log = []

    # 4. Evaluation Loop
    for q in queries_scaled:
        y_true.append(q["class"])

        for mode, m_data in models.items():
            sorted_scores = rank_candidates(q, gallery_scaled, m_data["alpha"], m_data["shape_key"])
            h1, h5, mrr = calculate_metrics(sorted_scores, q["class"])

            m_data["h1"].append(h1)
            m_data["h5"].append(h5)
            m_data["mrr"].append(mrr)

            predicted_class = predict_class(
                sorted_scores, k=m_data["k"], strategy=m_data["strategy"],
                score_key="s_hybrid", dedup=True
            )
            m_data["y_pred"].append(predicted_class)

            if mode == "hybrid_harmonics":
                deduped = deduplicate_rankings(sorted_scores)
                top_neighbors = [{"aid": n["aid"], "class": n["class"], "sim": float(n["s_hybrid"])} for n in
                                 deduped[:7]]
                hybrid_predictions_log.append({
                    "query_id": str(Path(q["path"]).relative_to(data_dir)),
                    "artifact_id": q["aid"],
                    "true_class": q["class"],
                    "predicted_class": predicted_class,
                    "neighbors": top_neighbors
                })

    # 5. Calculate final metrics exclusively for the 6 target classes
    target_classes = [c for c in class_names if c != "negative_control"]

    for mode, m_data in models.items():
        report = classification_report(
            y_true, m_data["y_pred"], labels=target_classes, output_dict=True, zero_division=0
        )
        m_data["macro_f1"] = report["macro avg"]["f1-score"]
        m_data["acc"] = accuracy_score(y_true, m_data["y_pred"])
        m_data["per_class"] = {c: report[c] for c in target_classes}

    print("\\n--- Held-out Test Set Performance Comparison ---")
    print(
        f"{'Configuration':<40} | {'Alpha':<5} | {'Hit@1':<8} | {'Hit@5':<8} | {'MRR':<8} | {'Accuracy':<8} | {'Macro F1':<8}")
    print("-" * 110)

    for mode_name in ["dinov2_only", "hybrid_base", "hybrid_harmonics", "hybrid_pointedness"]:
        d = models[mode_name]
        print(
            f"{d['title']:<40} | {d['alpha']:<5.2f} | {np.mean(d['h1']):.4f}   | {np.mean(d['h5']):.4f}   | {np.mean(d['mrr']):.4f} | {d['acc']:.4f}   | {d['macro_f1']:.4f}")

    print("-" * 110)

    b_f1 = models["dinov2_only"]["macro_f1"]
    hb_f1 = models["hybrid_base"]["macro_f1"]
    hh_f1 = models["hybrid_harmonics"]["macro_f1"]
    hp_f1 = models["hybrid_pointedness"]["macro_f1"]

    print(f"\\nAnalysis:")
    print(f"- Hybrid (Base Shape) vs DINOv2 Only: {hb_f1 - b_f1:+.4f} Macro F1")
    print(f"- Hybrid (Harmonics) vs DINOv2 Only:  {hh_f1 - b_f1:+.4f} Macro F1")
    print(f"- Hybrid (Pointedness) vs DINOv2 Only:{hp_f1 - b_f1:+.4f} Macro F1")

    cm, cm_labels = build_confusion_matrix(y_true, models["hybrid_harmonics"]["y_pred"], class_names)
    fig, ax = plt.subplots(figsize=(11, 9))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cm_labels)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, values_format="d")
    ax.set_title("Test Set Confusion Matrix — Hybrid (Harmonics, α=0.25)")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "test_confusion_matrix.png", dpi=300)
    plt.close(fig)

    final_metrics = {"metrics": {}}
    for mode_name in ["dinov2_only", "hybrid_base", "hybrid_harmonics", "hybrid_pointedness"]:
        final_metrics["metrics"][mode_name] = {
            "hit_at_1": float(np.mean(models[mode_name]["h1"])),
            "hit_at_5": float(np.mean(models[mode_name]["h5"])),
            "mrr": float(np.mean(models[mode_name]["mrr"])),
            "accuracy": float(models[mode_name]["acc"]),
            "macro_f1": float(models[mode_name]["macro_f1"]),
            "per_class": models[mode_name]["per_class"]
        }

    with open(RESULTS_DIR / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)

    with open(RESULTS_DIR / "test_predictions.json", "w", encoding="utf-8") as f:
        json.dump(hybrid_predictions_log, f, indent=2)


if __name__ == "__main__":
    evaluate_test_set()
