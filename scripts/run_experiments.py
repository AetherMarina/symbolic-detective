import argparse
from pathlib import Path
from symbolic_detective.evaluate import evaluate_pipeline
from symbolic_detective.experiment_configs import EXPERIMENTS

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_experiment(experiment_name: str, data_dir: Path, split_file: Path, k_folds: int, mode: str) -> None:
    """Run one experiment with its registered feature configuration."""

    if experiment_name == "baseline":
        feature_config = {}
        results_dir = PROJECT_ROOT / "results" / "v0.2_dev_baseline"
    else:
        feature_config = EXPERIMENTS[experiment_name].copy()
        results_dir = PROJECT_ROOT / "results" / "v0.2_dev" / experiment_name

    print(f"\n{'=' * 70}")
    print(f"Experiment: {experiment_name}")
    print(f"Features: {feature_config or 'baseline only'}")
    print(f"Results: {results_dir}")
    print(f"{'=' * 70}\n")

    results_dir.mkdir(parents=True, exist_ok=True)

    evaluate_pipeline(data_dir=data_dir, split_file=split_file, k_folds=k_folds, mode=mode, results_dir=results_dir,
                      feature_config=feature_config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Symbolic Detective v0.2 geometry ablation experiments.")
    selection = parser.add_mutually_exclusive_group(required=True)

    selection.add_argument("--experiment", choices=["baseline", *EXPERIMENTS.keys()], help="Run a single experiment.")
    selection.add_argument("--all", action="store_true", help="Run all registered experiments sequentially.")

    parser.add_argument("--mode", choices=["baseline", "voting", "all"], default="all",
                        help="Evaluation mode (default: all).")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "crops")
    parser.add_argument("--split-file", type=Path, default=PROJECT_ROOT / "data" / "splits" / "v0.2_split.json")
    parser.add_argument("--k-folds", type=int, default=5)

    args = parser.parse_args()

    if args.k_folds < 2:
        parser.error("--k-folds must be at least 2.")

    experiments = list(EXPERIMENTS.keys()) if args.all else [args.experiment]

    for experiment_name in experiments:
        run_experiment(experiment_name=experiment_name, data_dir=args.data_dir, split_file=args.split_file,
                       k_folds=args.k_folds, mode=args.mode)

    print("\nAll requested experiments completed successfully.")


if __name__ == "__main__":
    main()
