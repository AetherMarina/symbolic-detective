import json
import csv
from pathlib import Path
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
BASELINE_DIR = RESULTS_ROOT / "v0.2_dev_baseline"
DEV_DIR = RESULTS_ROOT / "v0.2_dev"
PHASE_B_DIR = DEV_DIR / "phase_b_combinations" / "results"

PHASE_A_EXPERIMENTS = [
    "exp01_harmonics",
    "exp02_peaks",
    "exp03_pointedness",
    "exp04_radius",
    "exp05_central_structure",
]

PHASE_B_EXPERIMENTS = [
    "b01_harmonics_peaks",
    "b02_peaks_pointedness",
    "b03_pointedness_radius",
    "b04_harmonics_peaks_pointedness",
    "b05_all_features",
    "b06_harmonics_pointedness"
]

ALL_EXPERIMENTS = PHASE_A_EXPERIMENTS + PHASE_B_EXPERIMENTS


def load_metrics(exp_dir: Path) -> Optional[Dict[str, Any]]:
    """Loads key metrics from retrieval_metrics.json and class_voting.json."""
    retrieval_path = exp_dir / "retrieval_metrics.json"
    voting_path = exp_dir / "class_voting.json"

    if not retrieval_path.exists():
        print(f"Warning: {retrieval_path} not found.")
        return None

    with open(retrieval_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Extract Cross-Validation metrics (Hit@5 and MRR) for alpha=0.25
    cv_overall = data.get("cross_validation", {}).get("overall", [])
    hit5 = 0.0
    mrr = 0.0
    for cv in cv_overall:
        if cv["alpha"] == 0.25:
            hit5 = cv["hit_at_5"]["mean"]
            mrr = cv["mrr"]["mean"]
            break

    # 2. Extract Multi-Class Discrimination for Top-1 (hybrid_025)
    hybrid_mc = data.get("multi_class_discrimination", {}).get("hybrid_025", {})

    # Calculate pooled Top-1 Accuracy directly from the confusion matrix
    cm = hybrid_mc.get("confusion_matrix", [])
    if cm:
        correct = sum(cm[i][i] for i in range(len(cm)))
        total = sum(sum(row) for row in cm)
        acc = correct / total if total > 0 else 0.0
    else:
        acc = 0.0

    macro_f1 = hybrid_mc.get("macro_avg", {}).get("f1-score", 0.0)

    per_class = {}
    for cls, metrics in hybrid_mc.get("per_class", {}).items():
        per_class[cls] = metrics.get("f1-score", 0.0)

    # 3. Extract Class Voting metrics (Weighted K=7, Alpha=0.25)
    voting_w7_f1 = 0.0
    if voting_path.exists():
        with open(voting_path, "r", encoding="utf-8") as f:
            voting_data = json.load(f)
        for row in voting_data:
            if row.get("alpha") == 0.25 and row.get("strategy") == "weighted" and row.get("k") == 7:
                # Use cv_macro_f1_mean as it is the most robust metric
                voting_w7_f1 = row.get("metrics", {}).get("cv_macro_f1_mean", 0.0)
                break

    return {
        "hit5": hit5,
        "mrr": mrr,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "voting_w7_f1": voting_w7_f1,
        "per_class": per_class
    }


def main():
    print("--- Starting Full v0.2 Metrics Comparison (Phase A & B) ---")

    baseline_metrics = load_metrics(BASELINE_DIR)
    if not baseline_metrics:
        raise RuntimeError("Baseline metrics not found. Run the baseline experiment first.")

    summary_rows = []
    per_class_rows = []

    # Add Baseline as the reference point (all deltas are 0.0)
    summary_rows.append({
        "Experiment": "baseline",
        "Hit@5": f"{baseline_metrics['hit5']:.4f}",
        "Hit@5_Delta": "0.0000",
        "MRR": f"{baseline_metrics['mrr']:.4f}",
        "MRR_Delta": "0.0000",
        "Accuracy": f"{baseline_metrics['accuracy']:.4f}",
        "Acc_Delta": "0.0000",
        "Macro_F1": f"{baseline_metrics['macro_f1']:.4f}",
        "F1_Delta": "0.0000",
        "Voting_W7_F1": f"{baseline_metrics['voting_w7_f1']:.4f}",
        "Voting_W7_Delta": "0.0000"
    })

    for cls, f1_val in baseline_metrics['per_class'].items():
        per_class_rows.append({
            "Experiment": "baseline",
            "Class": cls,
            "F1_Score": f"{f1_val:.4f}",
            "F1_Delta": "0.0000"
        })

    # Process Phase A and Phase B experiments
    for exp_name in ALL_EXPERIMENTS:
        # Route to the correct subfolder based on the experiment prefix
        if exp_name.startswith("exp"):
            exp_dir = DEV_DIR / exp_name
        else:
            exp_dir = PHASE_B_DIR / exp_name

        metrics = load_metrics(exp_dir)

        if not metrics:
            continue

        hit5_delta = metrics['hit5'] - baseline_metrics['hit5']
        mrr_delta = metrics['mrr'] - baseline_metrics['mrr']
        acc_delta = metrics['accuracy'] - baseline_metrics['accuracy']
        f1_delta = metrics['macro_f1'] - baseline_metrics['macro_f1']
        voting_delta = metrics['voting_w7_f1'] - baseline_metrics['voting_w7_f1']

        summary_rows.append({
            "Experiment": exp_name,
            "Hit@5": f"{metrics['hit5']:.4f}",
            "Hit@5_Delta": f"{hit5_delta:+.4f}",
            "MRR": f"{metrics['mrr']:.4f}",
            "MRR_Delta": f"{mrr_delta:+.4f}",
            "Accuracy": f"{metrics['accuracy']:.4f}",
            "Acc_Delta": f"{acc_delta:+.4f}",
            "Macro_F1": f"{metrics['macro_f1']:.4f}",
            "F1_Delta": f"{f1_delta:+.4f}",
            "Voting_W7_F1": f"{metrics['voting_w7_f1']:.4f}",
            "Voting_W7_Delta": f"{voting_delta:+.4f}"
        })

        for cls, f1_val in metrics['per_class'].items():
            base_f1 = baseline_metrics['per_class'].get(cls, 0.0)
            c_f1_delta = f1_val - base_f1
            per_class_rows.append({
                "Experiment": exp_name,
                "Class": cls,
                "F1_Score": f"{f1_val:.4f}",
                "F1_Delta": f"{c_f1_delta:+.4f}"
            })

    # Save results to the Phase B directory
    PHASE_B_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = PHASE_B_DIR / "comparison_summary.csv"
    per_class_path = PHASE_B_DIR / "per_class_comparison.csv"

    summary_fieldnames = ["Experiment", "Hit@5", "Hit@5_Delta", "MRR", "MRR_Delta", "Accuracy", "Acc_Delta", "Macro_F1",
                          "F1_Delta", "Voting_W7_F1", "Voting_W7_Delta"]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    per_class_fieldnames = ["Experiment", "Class", "F1_Score", "F1_Delta"]
    with open(per_class_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=per_class_fieldnames)
        writer.writeheader()
        writer.writerows(per_class_rows)

    print(f"Done! Saved summary comparison to: {summary_path}")
    print(f"Saved per-class comparison to: {per_class_path}")

if __name__ == "__main__":
    main()
