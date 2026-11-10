"""
Creates a stratified, artifact-grouped development/test split.
Ensures artifacts are not leaked across partitions while maintaining class balance.
Designed to be reusable via command-line arguments.
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import StratifiedGroupKFold

from symbolic_detective.utils.image_files import collect_images, artifact_key_from_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "crops"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "splits" / "v0.2_split.json"


def create_split(data_dir: Path, output_path: Path, test_size: float, seed: int):
    print(f"Scanning dataset in {data_dir.name}/ for split generation...")

    # Exclude 'unclassified' from the partition logic
    classes = sorted([
        d for d in data_dir.iterdir()
        if d.is_dir() and d.name != "unclassified"
    ])

    # Track artifacts and their primary class for stratification
    artifact_to_class = {}

    for cls_dir in classes:
        for crop_file in collect_images(cls_dir):
            aid = artifact_key_from_path(crop_file)

            if aid not in artifact_to_class:
                artifact_to_class[aid] = cls_dir.name

    aids = list(artifact_to_class.keys())
    y = [artifact_to_class[aid] for aid in aids]
    groups = aids  # Artifact IDs are their own groups

    # Convert test_size fraction to k-fold splits (e.g., 0.20 -> 5 splits)
    n_splits = max(2, int(round(1.0 / test_size)))

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # sgkf.split returns a generator. We pull the first fold configuration.
    dev_idx, test_idx = next(sgkf.split(aids, y, groups))

    dev_aids = [aids[i] for i in dev_idx]
    test_aids = [aids[i] for i in test_idx]

    # Validate artifact-grouped split
    dev_set = set(dev_aids)
    test_set = set(test_aids)

    if not dev_set.isdisjoint(test_set):
        raise ValueError("Artifact leakage between dev and test!")
    if dev_set | test_set != set(aids):
        raise ValueError("Some artifacts are missing!")
    if len(dev_aids) + len(test_aids) != len(aids):
        raise ValueError("Artifact count mismatch!")

    print("Artifact split validation passed: no leakage or missing artifacts.")

    # Validate crop assignments and calculate actual crop distributions
    dev_crop_counts = defaultdict(int)
    test_crop_counts = defaultdict(int)

    total_crops = 0

    for cls_dir in classes:
        for crop_file in collect_images(cls_dir):
            aid = artifact_key_from_path(crop_file)
            total_crops += 1

            if aid in dev_set:
                dev_crop_counts[cls_dir.name] += 1
            elif aid in test_set:
                test_crop_counts[cls_dir.name] += 1
            else:
                raise RuntimeError(f"Unassigned crop: {crop_file}")

    if sum(dev_crop_counts.values()) + sum(test_crop_counts.values()) != total_crops:
        raise ValueError("Crop count mismatch: some crops are missing from the split!")

    print(f"Crop validation passed: {total_crops} crops assigned.")

    # Calculate class distributions for the manifest
    dev_class_counts = defaultdict(int)
    test_class_counts = defaultdict(int)

    for aid in dev_aids:
        dev_class_counts[artifact_to_class[aid]] += 1
    for aid in test_aids:
        test_class_counts[artifact_to_class[aid]] += 1

    manifest = {
        "metadata": {
            "seed": seed,
            "test_size_requested": test_size,
            "folds_calculated": n_splits,
            "description": f"Stratified artifact-grouped split ({100 - (100/n_splits):.0f}/{100/n_splits:.0f}).",
            "total_artifacts": len(aids),
            "dev_artifacts": len(dev_aids),
            "test_artifacts": len(test_aids)
        },
        "partitions": {
            "dev": sorted(dev_aids),
            "test": sorted(test_aids)
        },
        "artifact_class_distribution": {
            "dev": dict(dev_class_counts),
            "test": dict(test_class_counts)
        },
        "crop_class_distribution": {
            "dev": dict(dev_crop_counts),
            "test": dict(test_crop_counts)
        }
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✅ Split successfully created and saved to: {output_path}")
    print(f"Total Artifacts: {len(aids)}")
    print(f"Dev partition:   {len(dev_aids)} artifacts (~{100 - (100/n_splits):.0f}%)")
    print(f"Test partition:  {len(test_aids)} artifacts (~{100/n_splits:.0f}%)")

    print("\nClass distribution (by Artifact ID):")
    all_classes = set(dev_class_counts.keys()).union(set(test_class_counts.keys()))
    for c in sorted(all_classes):
        d_count = dev_class_counts.get(c, 0)
        t_count = test_class_counts.get(c, 0)
        print(f"  - {c.ljust(25)}: Dev={d_count}, Test={t_count}")

    print("\nClass distribution (by Crop):")
    for c in sorted(d.name for d in classes):
        print(
            f"  - {c.ljust(25)}: "
            f"Dev={dev_crop_counts[c]}, "
            f"Test={test_crop_counts[c]}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a stratified artifact-grouped dataset split.")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="Path to cropped images directory.")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH, help="Path to save the split manifest JSON.")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of artifacts to reserve for testing (e.g., 0.2).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()

    create_split(Path(args.data_dir), Path(args.output), args.test_size, args.seed)
