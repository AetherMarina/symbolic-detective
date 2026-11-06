import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
BASELINE_DIR = RESULTS_ROOT / "v0.2_dev_baseline"
DEV_DIR = RESULTS_ROOT / "v0.2_dev"
results_dir = Path('results/v0.2_dev')
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

ALL_EXPERIMENTS = ["baseline"] + PHASE_A_EXPERIMENTS + PHASE_B_EXPERIMENTS


def extract_per_class_metrics(exp_name: str, target_alpha: float, target_strategy: str, target_k: int):
    file_path = DEV_DIR / exp_name / "class_voting.json"

    if not file_path.exists():
        print(f"File not found: {file_path}")
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for row in data:
        if row.get("alpha") == target_alpha and row.get("strategy") == target_strategy and row.get("k") == target_k:
            return row.get("metrics", {}).get("per_class_metrics", {})

    print(f"Configuration not found in {exp_name}")
    return None


def print_per_class_metrics():
    print("--- Per-Class F1 Score Comparison ---")

    # 1. exp01_harmonics, Weighted K=7, alpha=0.25
    metrics_01 = extract_per_class_metrics("exp01_harmonics", 0.25, "weighted", 7)

    # 2. exp03_pointedness, Top-1, alpha=0.25
    metrics_03 = extract_per_class_metrics("exp03_pointedness", 0.25, "top1", 1)

    if metrics_01 is not None and metrics_03 is not None:
        classes = sorted(list(set(metrics_01.keys()) | set(metrics_03.keys())))

        print(f"\\n{'Class':<25} | {'exp01 (W, K=7)':<15} | {'exp03 (Top-1)':<15} | {'Delta (exp03 - exp01)':<25}")
        print("-" * 85)

        target_classes = ["eight_pointed_star", "eight_petal_rosette", "six_pointed_star"]

        for cls in classes:
            f1_01 = metrics_01.get(cls, {}).get("f1-score", 0.0)
            f1_03 = metrics_03.get(cls, {}).get("f1-score", 0.0)
            delta = f1_03 - f1_01

            # Highlight target classes
            marker = " *" if cls in target_classes else ""

            print(f"{cls + marker:<25} | {f1_01:<15.4f} | {f1_03:<15.4f} | {delta:+.4f}")
        print("-" * 85)


def get_fold_metrics(exp_dir: Path, alpha: float = 0.25) -> list:
    retrieval_path = exp_dir / "retrieval_metrics.json"
    if not retrieval_path.exists():
        return []
    with open(retrieval_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cv_overall = data.get("cross_validation", {}).get("overall", [])
    for cv in cv_overall:
        if cv["alpha"] == alpha:
            return cv.get("hit_at_1", {}).get("folds", [])
    return []


def extract_macro_f1(exp_name: str, target_alpha: float, target_strategy: str, target_k: int):
    file_path = DEV_DIR / exp_name / "class_voting.json"

    if not file_path.exists():
        print(f"File not found: {file_path}")
        return None, None

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for row in data:
        if row.get("alpha") == target_alpha and row.get("strategy") == target_strategy and row.get("k") == target_k:
            mean_f1 = row.get("metrics", {}).get("cv_macro_f1_mean", 0.0)
            std_f1 = row.get("metrics", {}).get("cv_macro_f1_std", 0.0)
            return mean_f1, std_f1

    print(f"Configuration not found in {exp_name}")
    return None, None


def main():
    print("--- Macro F1 Metrics Comparison ---")

    # 1. exp03_pointedness, Top-1, alpha=0.25
    mean_03, std_03 = extract_macro_f1("exp03_pointedness", 0.25, "top1", 1)

    # 2. exp01_harmonics, Weighted K=7, alpha=0.25
    mean_01, std_01 = extract_macro_f1("exp01_harmonics", 0.25, "weighted", 7)

    if mean_03 is not None and mean_01 is not None:
        delta = mean_03 - mean_01

        print(f"\\n{'Experiment':<25} | {'Configuration':<20} | {'Macro F1 (Mean ± STD)':<25}")
        print("-" * 75)
        print(f"{'exp01_harmonics':<25} | {'Weighted, K=7':<20} | {mean_01:.4f} ± {std_01:.4f}")
        print(f"{'exp03_pointedness':<25} | {'Top-1, K=1':<20} | {mean_03:.4f} ± {std_03:.4f}")
        print("-" * 75)

        if delta > 0:
            print(
                f"Conclusion: exp03_pointedness (Top-1) is BETTER by {delta:.4f} than exp01_harmonics (Weighted K=7).")
        elif delta < 0:
            print(
                f"Conclusion: exp03_pointedness (Top-1) is WORSE by {abs(delta):.4f} than exp01_harmonics (Weighted K=7).")
        else:
            print("Conclusion: Both configurations have an identical average Macro F1 score.")


def print_alpha_0_strategies(exp_name: str, dev_dir: Path):
    file_path = dev_dir / exp_name / "class_voting.json"

    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    for row in data:
        if row.get("alpha") == 0.0:
            strategy = row.get("strategy")
            k = row.get("k")
            mean_f1 = row.get("metrics", {}).get("cv_macro_f1_mean", 0.0)
            std_f1 = row.get("metrics", {}).get("cv_macro_f1_std", 0.0)
            results.append((strategy, k, mean_f1, std_f1))

    results.sort(key=lambda x: x[2], reverse=True)

    print(f"\n--- Alpha 0.0 Strategy for  {exp_name}  ---")
    print(f"{'Strategy':<15} | {'K':<3} | {'Macro F1 (Mean ± STD)':<25}")
    print("-" * 50)
    for strat, k, mean_f1, std_f1 in results:
        print(f"{strat:<15} | {k:<3} | {mean_f1:.4f} ± {std_f1:.4f}")


def find_best_configurations():
    best_per_alpha = {}

    for exp_name in ALL_EXPERIMENTS:
        if exp_name == "baseline":
            file_path = BASELINE_DIR / "class_voting.json"
        elif exp_name.startswith("exp"):
            file_path = DEV_DIR / exp_name / "class_voting.json"
        else:
            file_path = PHASE_B_DIR / exp_name / "class_voting.json"

        if not file_path.exists():
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for row in data:
            alpha = row.get("alpha")
            strategy = row.get("strategy")
            k = row.get("k")
            mean_f1 = row.get("metrics", {}).get("cv_macro_f1_mean", 0.0)
            std_f1 = row.get("metrics", {}).get("cv_macro_f1_std", 0.0)

            if alpha not in best_per_alpha or mean_f1 > best_per_alpha[alpha]["mean"]:
                best_per_alpha[alpha] = {
                    "exp": exp_name,
                    "strategy": strategy,
                    "k": k,
                    "mean": mean_f1,
                    "std": std_f1
                }

    print("\n--- Best Configuration per Alpha (All Experiments) ---\n")
    print(f"{'Alpha':<5} | {'Experiment':<31} | {'Strategy':<10} | {'K':<3} | {'CV Macro F1 (Mean ± STD)'}")
    print("-" * 85)

    global_best = None

    for alpha in sorted(best_per_alpha.keys()):
        cfg = best_per_alpha[alpha]
        print(f"{alpha:<5.2f} | {cfg['exp']:<31} | {cfg['strategy']:<10} | {cfg['k']:<3} | {cfg['mean']:.4f} ± {cfg['std']:.4f}")

        if global_best is None or cfg["mean"] > global_best["mean"]:
            global_best = cfg
            global_best["alpha"] = alpha

    print("-" * 85)

    if global_best:
        print(f"\nGLOBAL BEST:")
        print(f"Experiment:  {global_best['exp']}")
        print(f"Alpha:       {global_best['alpha']}")
        print(f"Strategy:    {global_best['strategy']}")
        print(f"K:           {global_best['k']}")
        print(f"CV Macro F1: {global_best['mean']:.4f} ± {global_best['std']:.4f}")


if __name__ == "__main__":
    # --- Fold-Level Analysis for exp03_pointedness ---
    print("\\n--- Fold-Level Analysis: exp03_pointedness vs Baseline ---")
    base_folds = get_fold_metrics(BASELINE_DIR)
    exp03_dir = DEV_DIR / "exp03_pointedness"
    exp03_folds = get_fold_metrics(exp03_dir)

    if base_folds and exp03_folds and len(base_folds) == len(exp03_folds):
        print(f"{'Fold':<6} | {'Baseline Hit@1':<16} | {'Pointedness Hit@1':<19} | {'Δ':<8}")
        print("-" * 57)
        improved = 0
        worsened = 0
        unchanged = 0

        for i, (b_val, e_val) in enumerate(zip(base_folds, exp03_folds), 1):
            delta = e_val - b_val
            # Using 0.0001 threshold for floating point comparisons
            if delta > 0.0001:
                improved += 1
            elif delta < -0.0001:
                worsened += 1
            else:
                unchanged += 1

            print(f"{i:<6} | {b_val:<16.4f} | {e_val:<19.4f} | {delta:+.4f}")

        print("-" * 57)
        print(f"Improved folds: {improved}")
        print(f"Worsened folds:  {worsened}")
        print(f"Unchanged:     {unchanged}\\n")
    else:
        print("Fold data for exp03_pointedness or baseline are not available.")

    print("*******************************************************************************")
    main()
    print("*******************************************************************************")
    print_per_class_metrics()
    print_alpha_0_strategies("exp01_harmonics", DEV_DIR)
    print("*******************************************************************************")
    find_best_configurations()
