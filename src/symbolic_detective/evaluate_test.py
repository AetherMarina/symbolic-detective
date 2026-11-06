import json
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
import matplotlib.pyplot as plt

from symbolic_detective.evaluation_metrics import build_confusion_matrix, calculate_metrics
from symbolic_detective.pipeline.preprocess import preprocess_for_opencv, preprocess_for_dinov2
from symbolic_detective.pipeline.shape_features import compute_shape_features, compute_shape_similarity
from symbolic_detective.pipeline.embeddings import extract_visual_embedding, compute_embedding_similarity
from symbolic_detective.utils.image_files import collect_images, artifact_key_from_path
from symbolic_detective.pipeline.class_voting import predict_class, deduplicate_rankings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data/crops"
DEFAULT_SPLIT_FILE = PROJECT_ROOT / "data/splits/v0.2_split.json"
RESULTS_DIR = PROJECT_ROOT / "results" / "v0.2_test_final"


def extract_features_for_split(data_dir: Path, allowed_aids: set, split_name: str, is_query: bool = False):
    print(f"Extracting features for the {split_name} split...")
    classes = sorted(d for d in data_dir.iterdir() if d.is_dir() and d.name not in {"unclassified"})
    class_names = [c.name for c in classes]

    base_config = {
        "harmonics": False,
        "peaks": False,
        "pointedness": False,
        "radius": False,
        "central_structure": False
    }

    harm_config = {
        "harmonics": True,
        "peaks": False,
        "pointedness": False,
        "radius": False,
        "central_structure": False
    }

    point_config = {
        "harmonics": False,
        "peaks": False,
        "pointedness": True,
        "radius": False,
        "central_structure": False
    }

    items = []
    for cls_dir in classes:
        if is_query and cls_dir.name == "negative_control":
            continue

        for img in collect_images(cls_dir):
            aid = artifact_key_from_path(img)
            if aid not in allowed_aids:
                continue

            cv_img = preprocess_for_opencv(img)
            dino_img = preprocess_for_dinov2(img)

            if cv_img is None or dino_img is None:
                continue

            shape_base = compute_shape_features(cv_img, config=base_config)
            shape_harm = compute_shape_features(cv_img, config=harm_config)
            shape_point = compute_shape_features(cv_img, config=point_config)
            dino_f = extract_visual_embedding(dino_img)

            if shape_base is not None and shape_harm is not None and shape_point is not None and dino_f is not None:
                items.append({
                    "path": img,
                    "class": cls_dir.name,
                    "aid": aid,
                    "raw_shape_base": shape_base["feature_vector"],
                    "raw_shape_harm": shape_harm["feature_vector"],
                    "raw_shape_point": shape_point["feature_vector"],
                    "dino_vec": dino_f
                })
    return items, class_names


def evaluate_test_set(data_dir: Path = DEFAULT_DATA_DIR, split_file: Path = DEFAULT_SPLIT_FILE):
    print("=== Running FINAL HELD-OUT Evaluation ===")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(split_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    dev_aids = set(manifest["partitions"]["dev"])
    test_aids = set(manifest["partitions"]["test"])

    assert dev_aids.isdisjoint(test_aids), "Error: Data leakage detected between dev and test sets!"

    gallery_items, class_names = extract_features_for_split(data_dir, dev_aids, "Gallery (Dev)", is_query=False)
    query_items, _ = extract_features_for_split(data_dir, test_aids, "Queries (Test)", is_query=True)

    print(f"\\nGallery (Dev corpus):  {len(gallery_items)} crops")
    print(f"Queries (Test set):    {len(query_items)} crops\\n")

    if not gallery_items or not query_items:
        raise RuntimeError("Missing data for gallery or queries.")

    scaler_base = StandardScaler()
    scaler_harm = StandardScaler()
    scaler_point = StandardScaler()

    scaler_base.fit([item["raw_shape_base"] for item in gallery_items])
    scaler_harm.fit([item["raw_shape_harm"] for item in gallery_items])
    scaler_point.fit([item["raw_shape_point"] for item in gallery_items])

    def scale_items(items):
        scaled = []
        for item in items:
            scaled.append({
                "class": item["class"],
                "path": str(item["path"]),
                "aid": item["aid"],
                "shape_base": scaler_base.transform([item["raw_shape_base"]])[0],
                "shape_harm": scaler_harm.transform([item["raw_shape_harm"]])[0],
                "shape_point": scaler_point.transform([item["raw_shape_point"]])[0],
                "dino_vec": item["dino_vec"]
            })
        return scaled

    gallery_scaled = scale_items(gallery_items)
    queries_scaled = scale_items(query_items)

    models = {
        "dinov2_only": {
            "title": "Baseline (DINOv2 Only, Maj K=3)",
            "alpha": 0.0, "strategy": "majority", "k": 3, "shape_key": "shape_base",
            "y_pred": [], "h1": [], "h5": [], "mrr": []
        },
        "hybrid_base": {
            "title": "Baseline Hybrid (Base Shape, W K=7)",
            "alpha": 0.25, "strategy": "weighted", "k": 7, "shape_key": "shape_base",
            "y_pred": [], "h1": [], "h5": [], "mrr": []
        },
        "hybrid_harmonics": {
            "title": "Hybrid (Harmonics, W K=7)",
            "alpha": 0.25, "strategy": "weighted", "k": 7, "shape_key": "shape_harm",
            "y_pred": [], "h1": [], "h5": [], "mrr": []
        },
        "hybrid_pointedness": {
            "title": "Hybrid (Pointedness, Top-1)",
            "alpha": 0.25, "strategy": "top1", "k": 1, "shape_key": "shape_point",
            "y_pred": [], "h1": [], "h5": [], "mrr": []
        }
    }

    y_true = []
    hybrid_predictions_log = []

    for q in queries_scaled:
        y_true.append(q["class"])

        for mode, m_data in models.items():
            alpha = m_data["alpha"]
            strat = m_data["strategy"]
            k_val = m_data["k"]
            shape_key = m_data["shape_key"]

            scores = []

            for doc in gallery_scaled:
                s_shape = compute_shape_similarity(q[shape_key], doc[shape_key])
                s_embed = compute_embedding_similarity(q["dino_vec"], doc["dino_vec"])
                s_combined = (alpha * s_shape) + ((1.0 - alpha) * s_embed)

                scores.append({
                    "class": doc["class"],
                    "path": doc["path"],
                    "aid": doc["aid"],
                    "s_hybrid": s_combined
                })

            sorted_scores = sorted(scores, key=lambda x: x["s_hybrid"], reverse=True)
            h1, h5, mrr = calculate_metrics(sorted_scores, q["class"])

            m_data["h1"].append(h1)
            m_data["h5"].append(h5)
            m_data["mrr"].append(mrr)

            predicted_class = predict_class(sorted_scores, k=k_val, strategy=strat, score_key="s_hybrid", dedup=True)
            m_data["y_pred"].append(predicted_class)

            # Retain prediction logging for our primary selected hybrid (Harmonics, W K=7)
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

    # Calculate final metrics exclusively for the 6 target classes
    target_classes = [c for c in class_names if c != "negative_control"]

    for mode, m_data in models.items():
        report = classification_report(
            y_true,
            m_data["y_pred"],
            labels=target_classes,
            output_dict=True,
            zero_division=0
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

    final_metrics = {
        "metrics": {}
    }
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
