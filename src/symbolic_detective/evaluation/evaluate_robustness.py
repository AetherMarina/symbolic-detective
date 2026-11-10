import argparse
import hashlib
import cv2
import numpy as np
import logging
import tempfile
import os
from pathlib import Path
import albumentations as A
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score

from symbolic_detective.pipeline.preprocess import preprocess_for_opencv, preprocess_for_dinov2
from symbolic_detective.pipeline.shape_features import compute_shape_features
from symbolic_detective.pipeline.embeddings import extract_visual_embedding
from symbolic_detective.utils.image_files import collect_images, artifact_key_from_path
from symbolic_detective.evaluation.evaluation_metrics import calculate_metrics
from symbolic_detective.pipeline.class_voting import predict_class
from symbolic_detective.evaluation.evaluation_common import (
    load_split_manifest,
    scale_evaluation_features,
    rank_candidates
)
from symbolic_detective.evaluation.configs import CLASSIFICATION_FROZEN_CONFIG, RETRIEVAL_FROZEN_CONFIG

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data/crops"
DEFAULT_SPLIT_FILE = PROJECT_ROOT / "data/splits/v0.2_split.json"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def apply_corruption(img_path: Path, corruption_type: str, seed: int) -> Path:
    """Applies degradation and returns a path to the temporary degraded image."""
    if corruption_type == "clean":
        return img_path

    image = cv2.imread(str(img_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    if corruption_type == "rotation_90":
        transform = A.Affine(rotate=(90, 90), fit_output=True, fill=0, p=1.0)
    elif corruption_type == "blur_5x5":
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
            num_holes_range=(1, 1), hole_height_range=(drop_h, drop_h),
            hole_width_range=(drop_w, drop_w), fill=0, p=1.0
        )
    else:
        raise ValueError(f"Unknown corruption type: {corruption_type}")

    transform.set_random_seed(seed)
    transformed_rgb = transform(image=image)["image"]

    transformed_bgr = cv2.cvtColor(transformed_rgb, cv2.COLOR_RGB2BGR)

    tmp_fd, tmp_file = tempfile.mkstemp(suffix=".png")
    os.close(tmp_fd)
    cv2.imwrite(tmp_file, transformed_bgr)

    return Path(tmp_file)


def extract_features(img_path: Path):
    cv_img = preprocess_for_opencv(img_path)
    dino_img = preprocess_for_dinov2(img_path)

    shape_cls, shape_ret = None, None
    if cv_img is None:
        logger.info(f"OpenCV preprocessing FAILED for {img_path.name}")
    else:
        # Extract both feature sets
        shape_cls = compute_shape_features(cv_img, config=CLASSIFICATION_FROZEN_CONFIG["feature_config"])
        shape_ret = compute_shape_features(cv_img, config=RETRIEVAL_FROZEN_CONFIG["feature_config"])

        if shape_cls is None or shape_ret is None:
            logger.debug(f"Shape extraction FAILED: min={cv_img.min()}, max={cv_img.max()}, std={cv_img.std():.2f}")

    dino_f = (extract_visual_embedding(dino_img) if dino_img is not None else None)
    return shape_cls, shape_ret, dino_f


def evaluate_robustness(data_dir: Path = DEFAULT_DATA_DIR,
                        split_file: Path = DEFAULT_SPLIT_FILE,
                        k_folds: int = 5):
    print(f"--- Starting {k_folds}-Fold Retrieval & Classification Robustness Evaluation ---")

    dev_aids, _ = load_split_manifest(split_file)
    print(f"Loaded {len(dev_aids)} allowed artifacts from {split_file.name}")

    classes = sorted(
        [d for d in data_dir.iterdir() if d.is_dir() and d.name != "unclassified"],
        key=lambda p: p.name
    )
    target_classes = sorted(d.name for d in classes if d.name != "negative_control")

    all_clean_items = []
    skipped_test_crops = 0

    print("Extracting clean index corpus for development artifacts...")
    for cls_dir in classes:
        for img in collect_images(cls_dir):
            aid = artifact_key_from_path(img)
            if aid not in dev_aids:
                skipped_test_crops += 1
                continue

            shape_cls, shape_ret, dino_f = extract_features(img)

            if shape_cls is not None and shape_ret is not None and dino_f is not None:
                all_clean_items.append({
                    "path": img, "class": cls_dir.name, "aid": aid,
                    "shape_cls": shape_cls["feature_vector"], 
                    "shape_ret": shape_ret["feature_vector"], 
                    "dino_vec": dino_f
                })

    print("\\nStratifying development cross-validation folds...")
    artifact_to_class = {}
    for item in all_clean_items:
        if item["class"] != "negative_control":
            if item["aid"] not in artifact_to_class:
                artifact_to_class[item["aid"]] = item["class"]

    queryable_aids = list(artifact_to_class.keys())
    y = [artifact_to_class[aid] for aid in queryable_aids]
    groups = queryable_aids

    sgkf = StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=42)
    folds = [[queryable_aids[i] for i in test_idx] for _, test_idx in sgkf.split(queryable_aids, y, groups)]

    print(f"Dataset extracted: {len(all_clean_items)} total crops.")
    print(f"Skipped test partition crops: {skipped_test_crops}")
    print(f"Partitioned {len(queryable_aids)} valid dev artifacts into {k_folds} folds.\\n")

    corruptions = ["clean", "rotation_90", "blur_5x5", "blur_11x11", "low_light", "low_contrast", "occlusion_10"]
    methods = ["DINOv2", "Hybrid"]

    results = {c: {m: {"h1": [], "h5": [], "mrr": [], "acc": [], "macro_f1": []} for m in methods} for c in corruptions}

    for fold_idx, query_aids in enumerate(folds):
        query_aids_set = set(query_aids)

        corpus_raw = [item for item in all_clean_items if item["aid"] not in query_aids_set]
        queries_metadata = [item for item in all_clean_items if item["aid"] in query_aids_set and item["class"] != "negative_control"]

        if not queries_metadata:
            continue

        corpus_aids = {doc["aid"] for doc in corpus_raw}
        if query_aids_set.intersection(corpus_aids):
            raise RuntimeError(f"Fold {fold_idx + 1}: Leakage detected. Query artifact found in index.")

        # Fit scaling ONCE per fold for both shape configurations
        corpus_scaled, _, scalers = scale_evaluation_features(
            corpus_raw, [], ["shape_cls", "shape_ret"], return_scalers=True
        )

        for corruption in corruptions:
            repetitions = 5 if corruption == "occlusion_10" else 1
            fold_rep_scores = {m: {"h1": [], "h5": [], "mrr": [], "acc": [], "macro_f1": []} for m in methods}

            for rep_idx in range(repetitions):
                # Repetition level tracking arrays
                rep_metrics = {
                    m: {"h1": [], "h5": [], "mrr": [], "y_true": [], "y_pred": []} for m in methods
                }

                for q_meta in queries_metadata:
                    seed_str = f"{q_meta['path'].name}_{corruption}_{rep_idx}"
                    stable_seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2 ** 32)

                    image_input = apply_corruption(q_meta["path"], corruption, seed=stable_seed)

                    try:
                        shape_cls, shape_ret, dino_f = extract_features(image_input)
                    finally:
                        if corruption != "clean":
                            image_input.unlink(missing_ok=True)

                    if shape_cls is None or shape_ret is None or dino_f is None:
                        logger.warning(
                            f"Feature extraction incomplete: image={q_meta['path'].name}, corruption={corruption}")

                    # --- DINOv2 (Baseline) ---
                    if dino_f is not None:
                        q_dino = {
                            "path": str(q_meta["path"]), "class": q_meta["class"], "aid": q_meta["aid"],
                            "dino_vec": dino_f, "shape_cls": None, "shape_ret": None
                        }
                        scores_dino = rank_candidates(q_dino, corpus_scaled, 0.0, "shape_cls")
                        h1_d, h5_d, mrr_d = calculate_metrics(scores_dino, q_meta["class"])
                        pred_d = predict_class(scores_dino, k=CLASSIFICATION_FROZEN_CONFIG["k"],
                                               strategy=CLASSIFICATION_FROZEN_CONFIG["strategy"], score_key="s_hybrid",
                                               dedup=True)
                    else:
                        h1_d, h5_d, mrr_d, pred_d = 0, 0.0, 0.0, "unknown"

                    rep_metrics["DINOv2"]["h1"].append(h1_d)
                    rep_metrics["DINOv2"]["h5"].append(h5_d)
                    rep_metrics["DINOv2"]["mrr"].append(mrr_d)
                    rep_metrics["DINOv2"]["y_true"].append(q_meta["class"])
                    rep_metrics["DINOv2"]["y_pred"].append(pred_d)

                    # --- Hybrid (Retrieval & Classification Configs) ---
                    if dino_f is not None and shape_cls is not None and shape_ret is not None:
                        s_cls = scalers["shape_cls"].transform([shape_cls["feature_vector"]])[0]
                        s_ret = scalers["shape_ret"].transform([shape_ret["feature_vector"]])[0]
                        q_hyb = {
                            "path": str(q_meta["path"]), "class": q_meta["class"], "aid": q_meta["aid"],
                            "dino_vec": dino_f, "shape_cls": s_cls, "shape_ret": s_ret
                        }

                        scores_ret = rank_candidates(q_hyb, corpus_scaled, RETRIEVAL_FROZEN_CONFIG["alpha"],
                                                     "shape_ret")
                        h1_h, h5_h, mrr_h = calculate_metrics(scores_ret, q_meta["class"])

                        scores_cls = rank_candidates(q_hyb, corpus_scaled, CLASSIFICATION_FROZEN_CONFIG["alpha"],
                                                     "shape_cls")
                        pred_h = predict_class(scores_cls, k=CLASSIFICATION_FROZEN_CONFIG["k"],
                                               strategy=CLASSIFICATION_FROZEN_CONFIG["strategy"], score_key="s_hybrid",
                                               dedup=True)
                    else:
                        h1_h, h5_h, mrr_h, pred_h = 0, 0.0, 0.0, "unknown"

                    rep_metrics["Hybrid"]["h1"].append(h1_h)
                    rep_metrics["Hybrid"]["h5"].append(h5_h)
                    rep_metrics["Hybrid"]["mrr"].append(mrr_h)
                    rep_metrics["Hybrid"]["y_true"].append(q_meta["class"])
                    rep_metrics["Hybrid"]["y_pred"].append(pred_h)

                # Aggregate metrics for the whole repetition
                for m in methods:
                    if rep_metrics[m]["h1"]:
                        fold_rep_scores[m]["h1"].append(np.mean(rep_metrics[m]["h1"]))
                        fold_rep_scores[m]["h5"].append(np.mean(rep_metrics[m]["h5"]))
                        fold_rep_scores[m]["mrr"].append(np.mean(rep_metrics[m]["mrr"]))

                        y_t = rep_metrics[m]["y_true"]
                        y_p = rep_metrics[m]["y_pred"]
                        fold_rep_scores[m]["acc"].append(accuracy_score(y_t, y_p))
                        fold_rep_scores[m]["macro_f1"].append(
                            f1_score(y_t, y_p, labels=target_classes, average="macro", zero_division=0)
                        )

            # Average repetitions and store for the fold
            for m in methods:
                if fold_rep_scores[m]["h1"]:
                    results[corruption][m]["h1"].append(np.mean(fold_rep_scores[m]["h1"]))
                    results[corruption][m]["h5"].append(np.mean(fold_rep_scores[m]["h5"]))
                    results[corruption][m]["mrr"].append(np.mean(fold_rep_scores[m]["mrr"]))
                    results[corruption][m]["acc"].append(np.mean(fold_rep_scores[m]["acc"]))
                    results[corruption][m]["macro_f1"].append(np.mean(fold_rep_scores[m]["macro_f1"]))

    clean_base = {}
    for m in methods:
        clean_base[m] = {
            "h1": np.mean(results["clean"][m]["h1"]),
            "h5": np.mean(results["clean"][m]["h5"]),
            "mrr": np.mean(results["clean"][m]["mrr"]),
            "acc": np.mean(results["clean"][m]["acc"]),
            "macro_f1": np.mean(results["clean"][m]["macro_f1"])
        }

    print(f"\n### Robustness Results (Macro / All Classes)")
    print(
        f"| {'Condition':<15} | {'Method':<10} | {'R-Hit@1':<8} | {'R-Hit@5':<8} | {'R-MRR':<8} | {'C-Acc':<8} | {'C-F1':<8} | {'Δ Hit@1':<8} | {'Δ Hit@5':<8} | {'Δ MRR':<8} | {'Δ Acc':<8} | {'Δ F1':<8} |")
    print(
        "|" + "-" * 17 + "|" + "-" * 12 + "|" + "-" * 11 + "|" + "-" * 11 + "|" + "-" * 11 + "|" + "-" * 11 + "|" + "-" * 11 + "|" + "-" * 10 + "|" + "-" * 10 + "|" + "-" * 10 + "|" + "-" * 10 + "|" + "-" * 10 + "|")
    for corruption in corruptions:
        for m in methods:
            res = results[corruption][m]
            if not res["h1"]:
                continue

            h1_m = np.mean(res["h1"])
            h5_m = np.mean(res["h5"])
            mrr_m = np.mean(res["mrr"])
            acc_m = np.mean(res["acc"])
            f1_m = np.mean(res["macro_f1"])

            if corruption == "clean":
                h1_drop, h5_drop, mrr_drop, acc_drop, f1_drop = "-", "-", "-", "-", "-"
            else:
                h1_drop = f"{h1_m - clean_base[m]['h1']:+.3f}"
                h5_drop = f"{h5_m - clean_base[m]['h5']:+.3f}"
                mrr_drop = f"{mrr_m - clean_base[m]['mrr']:+.3f}"
                acc_drop = f"{acc_m - clean_base[m]['acc']:+.3f}"
                f1_drop = f"{f1_m - clean_base[m]['macro_f1']:+.3f}"

            print(
                f"| {corruption:<15} | {m:<10} | {h1_m:.3f}   | {h5_m:.3f}   | {mrr_m:.3f}   | {acc_m:.3f}   | {f1_m:.3f}   | {h1_drop:<8} | {h5_drop:<8} | {mrr_drop:<8} | {acc_drop:<8} | {f1_drop:<8} |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Symbolic Detective Robustness (Dev Set Only).")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="Path to cropped images directory")
    parser.add_argument("--split-file", type=str, default=DEFAULT_SPLIT_FILE, help="Path to split manifest JSON")
    parser.add_argument("--k-folds", type=int, default=5, help="Number of folds for cross-validation")
    args = parser.parse_args()

    evaluate_robustness(Path(args.data_dir), Path(args.split_file), k_folds=args.k_folds)
