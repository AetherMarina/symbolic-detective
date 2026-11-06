"""
Robustness Stress Tests.
Evaluates Hybrid Search against synthetically degraded query images using
deterministic Albumentations transformations on a strictly defined development split.
"""
import argparse
import hashlib
import cv2
import json
import numpy as np
import logging
from pathlib import Path
import albumentations as A
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold

from symbolic_detective.pipeline.preprocess import preprocess_for_opencv, preprocess_for_dinov2
from symbolic_detective.pipeline.shape_features import compute_shape_features, compute_shape_similarity
from symbolic_detective.pipeline.embeddings import extract_visual_embedding, compute_embedding_similarity
from symbolic_detective.utils.image_files import collect_images
from symbolic_detective.evaluation.evaluate import artifact_key_from_path, calculate_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data/crops"
DEFAULT_SPLIT_FILE = PROJECT_ROOT / "data/splits/v0.2_split.json"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def apply_corruption(img_path: Path, corruption_type: str, seed: int) -> np.ndarray:
    """Applies a deterministic synthetic degradation using Albumentations."""
    image = cv2.imread(str(img_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    if corruption_type == "clean":
        return image
    elif corruption_type == "rotation_90":
        transform = A.Affine(rotate=(90, 90), fit_output=True, fill=0, p=1.0)
    elif corruption_type == "blur_5x5":
        # Fixed Gaussian sigma for deterministic moderate blur
        transform = A.GaussianBlur(blur_limit=(5, 5), sigma_limit=(1.0, 1.0), p=1.0)
    elif corruption_type == "blur_11x11":
        transform = A.GaussianBlur(blur_limit=(11, 11), sigma_limit=(2.0, 2.0), p=1.0)
    elif corruption_type == "low_light":
        transform = A.RandomBrightnessContrast(brightness_limit=(-0.4, -0.4), contrast_limit=0, p=1.0)
    elif corruption_type == "low_contrast":
        transform = A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=(-0.4, -0.4), p=1.0)
    elif corruption_type == "occlusion_10":
        drop_h, drop_w = int(h * 0.316), int(w * 0.316)
        transform = A.CoarseDropout(
            num_holes_range=(1, 1),
            hole_height_range=(drop_h, drop_h),
            hole_width_range=(drop_w, drop_w),
            fill=0,
            p=1.0
        )
    else:
        raise ValueError(f"Unknown corruption type: {corruption_type}")

    transform.set_random_seed(seed)
    return transform(image=image)["image"]


def extract_features(img_array):
    cv_img = preprocess_for_opencv(img_array)
    dino_img = preprocess_for_dinov2(img_array)

    if cv_img is None:
        logger.info("OpenCV preprocessing FAILED")
        shape_f = None
    else:
        shape_f = compute_shape_features(cv_img)
        if shape_f is None:
            logger.debug(f"Shape extraction FAILED: min={cv_img.min()}, max={cv_img.max()}, std={cv_img.std():.2f}")

    dino_f = (extract_visual_embedding(dino_img) if dino_img is not None else None)
    return shape_f, dino_f


def evaluate_robustness(data_dir: Path = DEFAULT_DATA_DIR,
                        split_file: Path = DEFAULT_SPLIT_FILE,
                        target_class: str = "eight_pointed_star",
                        k_folds: int = 5):
    print(f"--- Starting {k_folds}-Fold Robustness Evaluation (Dev Split Only) ---")

    # 1. Load Dev Split Manifest
    if not split_file.exists():
        raise FileNotFoundError(f"Split manifest not found at {split_file}. Run create_split.py first.")

    with open(split_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    dev_aids = set(manifest["partitions"]["dev"])
    print(f"Loaded {len(dev_aids)} allowed artifacts from {split_file.name}")

    classes = sorted(
        [d for d in data_dir.iterdir()
         if d.is_dir() and d.name != "unclassified"],
        key=lambda p: p.name
    )

    all_clean_items = []
    skipped_test_crops = 0

    print("Extracting clean index corpus for development artifacts...")
    for cls_dir in classes:
        for img in collect_images(cls_dir):
            aid = artifact_key_from_path(img)

            # --- STRICT LEAKAGE PREVENTION ---
            if aid not in dev_aids:
                skipped_test_crops += 1
                continue

            shape_f, dino_f = extract_features(img)

            if shape_f is not None and dino_f is not None:
                all_clean_items.append({
                    "path": img, "class": cls_dir.name, "aid": aid,
                    "raw_shape": shape_f["feature_vector"], "dino_vec": dino_f
                })

    # Artifact-Grouped K-Fold Partitioning (Stratified)
    print("\nStratifying development cross-validation folds...")

    artifact_to_class = {}
    for item in all_clean_items:
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

    print(f"Dataset extracted: {len(all_clean_items)} total crops.")
    print(f"Skipped test partition crops: {skipped_test_crops}")
    print(f"Partitioned {len(queryable_aids)} valid dev artifacts into {k_folds} folds.\n")

    corruptions = ["clean", "rotation_90", "blur_5x5", "blur_11x11", "low_light", "low_contrast", "occlusion_10"]
    alphas = {"DINOv2 (0.0)": 0.0, "Hybrid (0.25)": 0.25}

    results = {c: {name: {"h1": [], "h5": [], "mrr": []} for name in alphas} for c in corruptions}

    for fold_idx, query_aids in enumerate(folds):
        query_aids_set = set(query_aids)

        corpus_raw = [item for item in all_clean_items if item["aid"] not in query_aids_set]
        queries_metadata = [item for item in all_clean_items if
                            item["aid"] in query_aids_set and item["class"] == target_class]

        if not queries_metadata:
            continue

        corpus_aids = {doc["aid"] for doc in corpus_raw}
        if query_aids_set.intersection(corpus_aids):
            raise RuntimeError(f"Fold {fold_idx + 1}: Substrate leakage detected. Query artifact found in index.")

        if target_class not in {doc["class"] for doc in corpus_raw}:
            raise RuntimeError(f"Fold {fold_idx + 1}: Target class '{target_class}' is missing from the index.")

        scaler = StandardScaler()
        scaler.fit([item["raw_shape"] for item in corpus_raw])

        corpus_scaled = []
        for item in corpus_raw:
            corpus_scaled.append({
                "class": item["class"],
                "shape_vec": scaler.transform([item["raw_shape"]])[0],
                "dino_vec": item["dino_vec"]
            })

        for corruption in corruptions:
            repetitions = 5 if corruption == "occlusion_10" else 1

            # Temporary structures to track scores for this specific fold
            fold_rep_scores = {name: {"h1": [], "h5": [], "mrr": []} for name in alphas}

            for rep_idx in range(repetitions):
                rep_scores = {name: {"h1": [], "h5": [], "mrr": []} for name in alphas}

                for q_meta in queries_metadata:
                    # Differentiate seeds by specific crop filename to prevent identical masks on same-artifact crops
                    seed_str = f"{q_meta['path'].name}_{corruption}_{rep_idx}"
                    stable_seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2 ** 32)

                    # Clean images use exactly the main benchmark preprocessing.
                    # Degraded images use in-memory arrays.
                    if corruption == "clean":
                        image_input = q_meta["path"]
                    else:
                        image_input = apply_corruption(q_meta["path"], corruption, seed=stable_seed)

                    # Extract corrupted features ONCE per query
                    shape_f, dino_f = extract_features(image_input)

                    if shape_f is None or dino_f is None:
                        logger.warning(f"Feature extraction failed: image={q_meta['path'].name}, corruption={corruption},"
                                       f" shape_failed={shape_f is None}, dino_failed={dino_f is None}")

                    for method_name, alpha in alphas.items():
                        # DINOv2 is required for both methods.
                        # OpenCV features are required only for the hybrid.
                        if dino_f is None or (alpha > 0 and shape_f is None):
                            rep_scores[method_name]["h1"].append(0)
                            rep_scores[method_name]["h5"].append(0)
                            rep_scores[method_name]["mrr"].append(0.0)
                            continue

                        q_shape_scaled = (
                            scaler.transform([shape_f["feature_vector"]])[0]
                            if alpha > 0 else None
                        )

                        scores = []
                        for doc in corpus_scaled:
                            s_embed = compute_embedding_similarity(dino_f, doc["dino_vec"])
                            if alpha == 0.0:
                                # Pure DINOv2: no OpenCV dependency
                                score = s_embed
                            else:
                                s_shape = compute_shape_similarity(q_shape_scaled, doc["shape_vec"])
                                score = alpha * s_shape + (1.0 - alpha) * s_embed

                            scores.append({"class": doc["class"], "s_hybrid": score})

                        scores.sort(key=lambda x: x["s_hybrid"], reverse=True)

                        h1, h5, mrr = calculate_metrics(scores, target_class)
                        rep_scores[method_name]["h1"].append(h1)
                        rep_scores[method_name]["h5"].append(h5)
                        rep_scores[method_name]["mrr"].append(mrr)

                for name in alphas:
                    if rep_scores[name]["h1"]:
                        fold_rep_scores[name]["h1"].append(np.mean(rep_scores[name]["h1"]))
                        fold_rep_scores[name]["h5"].append(np.mean(rep_scores[name]["h5"]))
                        fold_rep_scores[name]["mrr"].append(np.mean(rep_scores[name]["mrr"]))

            # Average across repetitions, then store as this fold's mean score
            for name in alphas:
                if fold_rep_scores[name]["h1"]:
                    results[corruption][name]["h1"].append(np.mean(fold_rep_scores[name]["h1"]))
                    results[corruption][name]["h5"].append(np.mean(fold_rep_scores[name]["h5"]))
                    results[corruption][name]["mrr"].append(np.mean(fold_rep_scores[name]["mrr"]))

    clean_base = {}
    for name in alphas.keys():
        clean_base[name] = {
            "h1": np.mean(results["clean"][name]["h1"]),
            "h5": np.mean(results["clean"][name]["h5"]),
            "mrr": np.mean(results["clean"][name]["mrr"])
        }

    print(f"\n### Robustness Results (Target: {target_class})")
    print(
        f"| {'Condition':<15} | {'Method':<14} | {'Hit@1':<12} | {'Hit@5':<12} | {'MRR':<12} | {'Hit@1 Δ':<8} | {'MRR Δ':<8} |")
    print(
        "|" + "-" * 17 + "|" + "-" * 16 + "|" + "-" * 14 + "|" + "-" * 14 + "|" + "-" * 14 + "|" + "-" * 10 + "|" + "-" * 10 + "|")

    for corruption in corruptions:
        for method_name in alphas.keys():
            res = results[corruption][method_name]
            if not res["h1"]:
                continue

            h1_m, h1_s = np.mean(res["h1"]), np.std(res["h1"])
            h5_m, h5_s = np.mean(res["h5"]), np.std(res["h5"])
            mrr_m, mrr_s = np.mean(res["mrr"]), np.std(res["mrr"])

            if corruption == "clean":
                h1_drop, mrr_drop = "-", "-"
            else:
                h1_drop = f"{h1_m - clean_base[method_name]['h1']:+.3f}"
                mrr_drop = f"{mrr_m - clean_base[method_name]['mrr']:+.3f}"

            print(
                f"| {corruption:<15} | {method_name:<14} | {h1_m:.3f} ±{h1_s:.2f} |  {h5_m:.3f} ±{h5_s:.2f} |  {mrr_m:.3f} ±{mrr_s:.2f} | {h1_drop:<8} | {mrr_drop:<8} |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Symbolic Detective Robustness (Dev Set Only).")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="Path to cropped images directory")
    parser.add_argument("--split-file", type=str, default=DEFAULT_SPLIT_FILE, help="Path to split manifest JSON")
    parser.add_argument("--k-folds", type=int, default=5, help="Number of folds for cross-validation")
    args = parser.parse_args()

    evaluate_robustness(Path(args.data_dir), Path(args.split_file), k_folds=args.k_folds)