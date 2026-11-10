import argparse
import json
from pathlib import Path
import numpy as np

from symbolic_detective.evaluation.configs import CLASSIFICATION_FROZEN_CONFIG
from symbolic_detective.evaluation.evaluation_common import (
    load_split_manifest,
    extract_evaluation_features
)
from symbolic_detective.evaluation.evaluate_rejection import build_calibration_results, find_optimal_threshold

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data/crops"
DEFAULT_SPLIT_FILE = PROJECT_ROOT / "data/splits/v0.2_split.json"
RESULTS_DIR = PROJECT_ROOT / "results" / "v0.2_combined_rejection"


def find_optimal_thresholds_2d(results, strategy):
    best_t1, best_t2 = 0.0, 0.0
    best_diag = {}
    
    max_sims = np.array([r["max_sim"] for r in results])
    weight_confs = np.array([r["weight_conf"] for r in results])
    is_known = np.array([r["is_known"] for r in results])
    
    if not np.all(np.isfinite(max_sims)) or not np.all(np.isfinite(weight_confs)):
        raise ValueError("Non-finite confidence scores detected.")

    total_ood = np.sum(~is_known)
    total_known = np.sum(is_known)
    if total_ood == 0 or total_known == 0:
        raise ValueError("Missing OOD or Known examples for calibration.")

    # Use exact unique score values to test boundaries instead of a fixed grid
    # Include boundaries just outside the min and max observed values to allow accept-all/reject-all policies
    thresholds_sim = np.unique(np.concatenate((
        [np.nextafter(max_sims.min(), -np.inf)],
        max_sims,
        [np.nextafter(max_sims.max(), np.inf)]
    )))
    thresholds_weight = np.unique(np.concatenate((
        [np.nextafter(weight_confs.min(), -np.inf)],
        weight_confs,
        [np.nextafter(weight_confs.max(), np.inf)]
    )))
        
    best_ranking = (-1.0, -1.0)
    
    for t1 in thresholds_sim:
        for t2 in thresholds_weight:
            if strategy == "reject_or":
                # REJECT if (max_sim < t1 OR weight_conf < t2)
                # = ACCEPT if (max_sim >= t1 AND weight_conf >= t2)
                accepted = (max_sims >= t1) & (weight_confs >= t2)
            elif strategy == "reject_and":
                # REJECT if (max_sim < t1 AND weight_conf < t2)
                # = ACCEPT if (max_sim >= t1 OR weight_conf >= t2)
                accepted = (max_sims >= t1) | (weight_confs >= t2)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
                
            known_acceptance = np.sum(accepted[is_known]) / total_known
            ood_rejection = np.sum(~accepted[~is_known]) / total_ood
            
            b_acc = (known_acceptance + ood_rejection) / 2
            
            # Tie-break prefers higher known_acceptance (lower FRR).
            # In case of a perfect tie, the first pair encountered is kept (deterministic behavior).
            ranking = (b_acc, known_acceptance)
            if ranking > best_ranking:
                best_ranking = ranking
                best_t1 = t1
                best_t2 = t2
                best_diag = {
                    "calibration_ood_rejection": float(ood_rejection),
                    "calibration_known_acceptance": float(known_acceptance),
                    "calibration_balanced_accuracy": float(b_acc)
                }
                
    return float(best_t1), float(best_t2), best_diag


def evaluate_combined_rejection(data_dir: Path = DEFAULT_DATA_DIR, split_file: Path = DEFAULT_SPLIT_FILE):
    print("=== Running Development Combined Rejection Evaluation ===")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dev_aids, test_aids = load_split_manifest(split_file)
    
    conf_key = "shape_vec"
    feature_configs = {conf_key: CLASSIFICATION_FROZEN_CONFIG["feature_config"]}
    alpha = CLASSIFICATION_FROZEN_CONFIG["alpha"]

    # Dev sets for calibration
    all_dev, _ = extract_evaluation_features(data_dir, dev_aids, feature_configs)
    dev_known = [i for i in all_dev if i["class"] != "negative_control"]
    dev_ood = [i for i in all_dev if i["class"] == "negative_control"]

    # Calibration Phase
    print("\n--- Building Fold Scores (Dev Data) ---")
    calib_results = build_calibration_results(dev_known, dev_ood, alpha, conf_key)

    # 1. Independent optimal thresholds (1D)
    print("\n--- 1D Threshold Calibration (Independent) ---")
    t_max_sim, max_diag = find_optimal_threshold(calib_results, "max_sim")
    t_weight, vote_diag = find_optimal_threshold(calib_results, "weight_conf")
    
    print(f"Optimal Max Similarity: {t_max_sim:.4f} (TRR: {max_diag['calibration_ood_rejection']:.4f}, FRR: {1-max_diag['calibration_known_acceptance']:.4f})")
    print(f"Optimal Weighted Vote:  {t_weight:.4f} (TRR: {vote_diag['calibration_ood_rejection']:.4f}, FRR: {1-vote_diag['calibration_known_acceptance']:.4f})")

    # 2. Joint Logic Optimization (2D AND / OR)
    print("\n--- 2D Joint Threshold Calibration (Combined Logic) ---")
    
    print("\nStrategy: REJECT_OR (Rejects if either score is below threshold)")
    or_t1, or_t2, or_diag = find_optimal_thresholds_2d(calib_results, "reject_or")
    print(f"Optimal Thresholds -> Max Sim: {or_t1:.4f}, Weight Conf: {or_t2:.4f}")
    print(f"TRR: {or_diag['calibration_ood_rejection']:.4f} | FRR: {1-or_diag['calibration_known_acceptance']:.4f} | B-Acc: {or_diag['calibration_balanced_accuracy']:.4f}")

    print("\nStrategy: REJECT_AND (Rejects only if both scores are below threshold)")
    and_t1, and_t2, and_diag = find_optimal_thresholds_2d(calib_results, "reject_and")
    print(f"Optimal Thresholds -> Max Sim: {and_t1:.4f}, Weight Conf: {and_t2:.4f}")
    print(f"TRR: {and_diag['calibration_ood_rejection']:.4f} | FRR: {1-and_diag['calibration_known_acceptance']:.4f} | B-Acc: {and_diag['calibration_balanced_accuracy']:.4f}")

    # Exports
    out_file = RESULTS_DIR / "dev_combined_strategy_analysis.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "1D_independent_calibration": {
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
            },
            "2D_joint_calibration": {
                "reject_or": {
                    "threshold_max_sim": or_t1,
                    "threshold_weight_conf": or_t2,
                    "calibration_trr": or_diag["calibration_ood_rejection"],
                    "calibration_frr": 1.0 - or_diag["calibration_known_acceptance"],
                    "calibration_balanced_accuracy": or_diag["calibration_balanced_accuracy"]
                },
                "reject_and": {
                    "threshold_max_sim": and_t1,
                    "threshold_weight_conf": and_t2,
                    "calibration_trr": and_diag["calibration_ood_rejection"],
                    "calibration_frr": 1.0 - and_diag["calibration_known_acceptance"],
                    "calibration_balanced_accuracy": and_diag["calibration_balanced_accuracy"]
                }
            }
        }, f, indent=2)
        
    print(f"\nDev strategy analysis exported to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Symbolic Detective Combined Rejection Logic.")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="Path to cropped images directory")
    parser.add_argument("--split-file", type=str, default=DEFAULT_SPLIT_FILE, help="Path to split manifest JSON")
    args = parser.parse_args()
    
    evaluate_combined_rejection(Path(args.data_dir), Path(args.split_file))
