"""Calibrate and evaluate unknown-motif rejection without test-set tuning.

Save as src/symbolic_detective/evaluation/evaluate_rejection.py.

The rejection gallery contains known motif classes only. This is intentionally
an open-set protocol, distinct from the closed-set v0.2 benchmark where
negative_control crops were retained as gallery distractors.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedGroupKFold

from symbolic_detective.evaluation.configs import CLASSIFICATION_FROZEN_CONFIG
from symbolic_detective.evaluation.evaluation_common import (
    extract_evaluation_features,
    load_split_manifest,
    rank_candidates,
    scale_evaluation_features
)
from symbolic_detective.pipeline.class_voting import deduplicate_rankings

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data/crops"
DEFAULT_SPLIT_FILE = PROJECT_ROOT / "data/splits/v0.2_split.json"
RESULTS_DIR = PROJECT_ROOT / "results" / "v0.2_test_rejection"


def get_confidence_scores(sorted_scores, k=7):
    deduped = deduplicate_rankings(sorted_scores)
    top_k = deduped[:k]

    if not top_k:
        return "unknown", 0.0, 0.0

    max_sim = float(top_k[0]["s_hybrid"])

    class_weights = {}
    total_weight = 0.0
    for neighbor in top_k:
        # To ensure that the weights are non-negative, weight_conf is in [0, 1]
        w = max(0.0, float(neighbor["s_hybrid"]))
        cls = neighbor["class"]
        class_weights[cls] = class_weights.get(cls, 0.0) + w
        total_weight += w

    predicted_class = max(class_weights, key=class_weights.get)
    weight_conf = class_weights[predicted_class] / total_weight if total_weight > 0 else 0.0

    return predicted_class, max_sim, weight_conf


def scale_and_score(gallery_raw, queries_raw, alpha, conf_key="shape_vec"):
    gal_scaled, q_scaled = scale_evaluation_features(gallery_raw, queries_raw, [conf_key])
    results = []
    for q in q_scaled:
        sorted_scores = rank_candidates(q, gal_scaled, alpha, conf_key)
        pred_cls, max_sim, w_conf = get_confidence_scores(sorted_scores, k=CLASSIFICATION_FROZEN_CONFIG["k"])
        results.append({
            "query_path": q["path"],
            "artifact_id": q["aid"],
            "true_class": q["class"],
            "pred_class": pred_cls,
            "is_known": q["class"] != "negative_control",
            "max_sim": max_sim,
            "weight_conf": w_conf,
        })
    return results


def find_optimal_threshold(results, score_key):
    """Pick a threshold using ONLY dev data, maximizing balanced accuracy.

    Balanced accuracy = 0.5 * (known acceptance rate + OOD rejection rate).
    Ties prefer a smaller false-rejection rate for known motifs.
    """
    if not any(r["is_known"] for r in results):
        raise ValueError("Calibration has no known queries.")
    if not any(not r["is_known"] for r in results):
        raise ValueError("Calibration has no unknown queries.")

    values = np.asarray([r[score_key] for r in results], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Non-finite confidence scores for {score_key}.")

    # Exact observed-score boundaries, rather than assuming all scores in [0, 1].
    thresholds = np.concatenate((
        [np.nextafter(values.min(), -np.inf)],
        np.unique(values),
        [np.nextafter(values.max(), np.inf)],
    ))
    known = np.asarray([r["is_known"] for r in results], dtype=bool)
    best = None

    for threshold in thresholds:
        accepted = values >= threshold
        known_acceptance = float(accepted[known].mean())
        ood_rejection = float((~accepted[~known]).mean())
        balanced_accuracy = (known_acceptance + ood_rejection) / 2
        # Lexicographic tie-break: higher balanced accuracy, then lower FRR.
        ranking = (balanced_accuracy, known_acceptance)
        if best is None or ranking > best[0]:
            best = (ranking, float(threshold))

    return best[1], {
        "calibration_balanced_accuracy": best[0][0],
        "calibration_known_acceptance": best[0][1],
        "calibration_ood_rejection": 2 * best[0][0] - best[0][1],
    }


def build_calibration_results(dev_known, dev_ood, alpha, conf_key):
    """Cross-validated calibration: each dev query artifact appears once."""
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    labels = [i["class"] for i in dev_known]
    groups = [i["aid"] for i in dev_known]

    # Negative-control artifacts are also grouped and assigned to a single fold.
    ood_aids = sorted({item["aid"] for item in dev_ood})
    rng = np.random.default_rng(42)
    rng.shuffle(ood_aids)
    ood_folds = [set(x.tolist()) for x in np.array_split(np.array(ood_aids), 5)]

    results = []
    for fold, (gallery_idx, query_idx) in enumerate(
        splitter.split(np.zeros(len(dev_known)), labels, groups)
    ):
        known_gallery_aids = {groups[i] for i in gallery_idx}
        known_query_aids = {groups[i] for i in query_idx}
        negative_query_aids = ood_folds[fold]

        # Exclude any same-artifact overlap with the unknown query partition.
        gallery_raw = [
            item for item in dev_known
            if item["aid"] in known_gallery_aids
            and item["aid"] not in negative_query_aids
        ]
        queries_raw = [
            item for item in dev_known if item["aid"] in known_query_aids
        ] + [
            item for item in dev_ood if item["aid"] in negative_query_aids
        ]
        results.extend(scale_and_score(gallery_raw, queries_raw, alpha, conf_key))

    return results


def evaluate_rejection(data_dir: Path = DEFAULT_DATA_DIR, split_file: Path = DEFAULT_SPLIT_FILE,
                       mode: str = "calibrate", method: str = "both"):
    print("=== Running Out-Of-Distribution Rejection Evaluation ===")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dev_aids, test_aids = load_split_manifest(split_file)

    conf_key = "shape_vec"
    feature_configs = {conf_key: CLASSIFICATION_FROZEN_CONFIG["feature_config"]}
    alpha = CLASSIFICATION_FROZEN_CONFIG["alpha"]
    thresholds_file = RESULTS_DIR / "calibrated_thresholds.json"

    # 1. Calibration Phase
    if mode == "calibrate":
        print("\n--- Calibration Phase (Dev Data) ---")
        all_dev, _ = extract_evaluation_features(data_dir, dev_aids, feature_configs)
        dev_known = [i for i in all_dev if i["class"] != "negative_control"]
        dev_ood = [i for i in all_dev if i["class"] == "negative_control"]

        calib_results = build_calibration_results(dev_known, dev_ood, alpha, conf_key)

        t_max_sim, max_diag = find_optimal_threshold(calib_results, "max_sim")
        t_weight, vote_diag = find_optimal_threshold(calib_results, "weight_conf")

        print(f"Optimal Maximum Similarity Threshold: {t_max_sim:.4f}")
        print(f"Optimal Weighted Vote Threshold:    {t_weight:.4f}")

        print("\n--- Calibration Metrics (Dev) ---")
        for name, diag in [("max_sim", max_diag), ("weight_conf", vote_diag)]:
            print(f"\n{name.upper()}")
            print(f"TRR: {diag['calibration_ood_rejection']:.4f}")
            print(f"FRR: {1 - diag['calibration_known_acceptance']:.4f}")
            print(f"Balanced Accuracy: {diag['calibration_balanced_accuracy']:.4f}")

        with open(thresholds_file, "w", encoding="utf-8") as f:
            json.dump({
                "max_sim": {
                    "threshold": t_max_sim,
                    "calibration_trr": max_diag["calibration_ood_rejection"],
                    "calibration_frr": 1.0 - max_diag["calibration_known_acceptance"],
                    "calibration_balanced_accuracy": max_diag["calibration_balanced_accuracy"]
                },
                "weight_conf": {
                    "threshold": t_weight,
                    "calibration_trr": vote_diag["calibration_ood_rejection"],
                    "calibration_frr": 1.0 - vote_diag["calibration_known_acceptance"],
                    "calibration_balanced_accuracy": vote_diag["calibration_balanced_accuracy"]
                }
            }, f, indent=2)
        print(f"Saved calibrated thresholds and metrics to {thresholds_file}")
    else:
        if not thresholds_file.exists():
            raise FileNotFoundError(
                f"Calibrated thresholds file not found at {thresholds_file}. "
                "Run with --mode calibrate."
            )
        with open(thresholds_file, "r", encoding="utf-8") as f:
            saved_thresholds = json.load(f)
        t_max_sim = saved_thresholds["max_sim"]["threshold"]
        t_weight = saved_thresholds["weight_conf"]["threshold"]
        print(f"Loaded calibrated thresholds: max_sim={t_max_sim:.4f}, weight_conf={t_weight:.4f}")

    # 2. Test Phase
    if mode == "test":
        print("\n--- Testing Phase (Held-out Data) ---")
        all_dev, _ = extract_evaluation_features(data_dir, dev_aids, feature_configs)
        dev_known = [i for i in all_dev if i["class"] != "negative_control"]

        all_test, _ = extract_evaluation_features(data_dir, test_aids, feature_configs)
        test_known = [i for i in all_test if i["class"] != "negative_control"]
        test_ood = [i for i in all_test if i["class"] == "negative_control"]

        test_results = scale_and_score(dev_known, test_known + test_ood, alpha, conf_key)

        metrics_export = {
            "frozen_config": CLASSIFICATION_FROZEN_CONFIG,
            "thresholds": {"max_sim": t_max_sim, "weight_conf": t_weight},
            "results": {}
        }

        methods_to_test = []
        if method in ["max_sim", "both"]:
            methods_to_test.append(("max_sim", t_max_sim))
        if method in ["weight_conf", "both"]:
            methods_to_test.append(("weight_conf", t_weight))

        predictions_export = []
        for score_type, t_val in methods_to_test:
            y_true, y_pred = [], []
            true_rejects, false_rejects, total_ood, total_known = 0, 0, 0, 0

            for r in test_results:
                is_rejected = r[score_type] < t_val
                final_pred = "unknown" if is_rejected else r["pred_class"]

                y_true.append(r["true_class"] if r["is_known"] else "unknown")
                y_pred.append(final_pred)

                predictions_export.append({
                    "query_path": r["query_path"],
                    "artifact_id": r["artifact_id"],
                    "true_class": r["true_class"],
                    "pred_class": r["pred_class"],
                    "max_sim": r["max_sim"],
                    "weight_conf": r["weight_conf"],
                    "method": score_type,
                    "threshold": t_val,
                    "rejected": is_rejected,
                    "final_prediction": final_pred,
                })

                if not r["is_known"]:
                    total_ood += 1
                    if final_pred == "unknown":
                        true_rejects += 1
                else:
                    total_known += 1
                    if final_pred == "unknown":
                        false_rejects += 1

            trr = true_rejects / total_ood if total_ood > 0 else 0
            frr = false_rejects / total_known if total_known > 0 else 0

            target_classes = sorted(list({r["true_class"] for r in test_results if r["is_known"]}))
            report = classification_report(
                y_true, y_pred, labels=target_classes + ["unknown"], output_dict=True, zero_division=0
            )
            unknown_f1 = report["unknown"]["f1-score"]

            print(f"\nMethod: {score_type.upper()} (Threshold = {t_val:.4f})")
            print(f"True Rejection Rate (TRR) : {trr:.4f} ({true_rejects}/{total_ood} OOD correctly rejected)")
            print(f"False Rejection Rate (FRR): {frr:.4f} ({false_rejects}/{total_known} Knowns wrongly rejected)")
            print(f"Macro F1 (Classification) : {report['macro avg']['f1-score']:.4f}")
            print(f"Unknown F1: {unknown_f1:.4f}")

            metrics_export["results"][score_type] = {
                "trr": float(trr),
                "frr": float(frr),
                "open_set_macro_f1": float(report['macro avg']['f1-score']),
                "unknown_f1": float(unknown_f1),
                "classification_report": report
            }

        with open(RESULTS_DIR / "rejection_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics_export, f, indent=2)

        with open(RESULTS_DIR / "rejection_predictions.json", "w", encoding="utf-8") as f:
            json.dump(predictions_export, f, indent=2)

        print(f"\nResults exported to {RESULTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate OOD Negative Query Rejection.")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="Path to cropped images directory")
    parser.add_argument("--split-file", type=str, default=DEFAULT_SPLIT_FILE, help="Path to split manifest JSON")
    parser.add_argument("--mode", choices=["calibrate", "test"], default="calibrate",
                        help="Execution mode: 'calibrate' threshold on dev, 'test' on held-out.")
    parser.add_argument("--method", choices=["max_sim", "weight_conf", "both"],
                        default="both", help="Which rejection method to evaluate in test mode")
    args = parser.parse_args()

    evaluate_rejection(Path(args.data_dir), Path(args.split_file), mode=args.mode, method=args.method)
