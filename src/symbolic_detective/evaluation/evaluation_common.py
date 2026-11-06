import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from sklearn.preprocessing import StandardScaler

from symbolic_detective.pipeline.preprocess import preprocess_for_opencv, preprocess_for_dinov2
from symbolic_detective.pipeline.shape_features import compute_shape_features, compute_shape_similarity
from symbolic_detective.pipeline.embeddings import extract_visual_embedding, compute_embedding_similarity
from symbolic_detective.utils.image_files import collect_images, artifact_key_from_path


def load_split_manifest(split_file: Path) -> Tuple[Set[str], Set[str]]:
    if not split_file.exists():
        raise FileNotFoundError(f"Split manifest not found at {split_file}.")

    with open(split_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    dev_aids = set(manifest["partitions"]["dev"])
    test_aids = set(manifest["partitions"]["test"])

    overlap = dev_aids & test_aids
    if overlap:
        raise RuntimeError(f"Data leakage detected! Overlapping artifacts: {sorted(overlap)}")

    return dev_aids, test_aids


def extract_evaluation_features(data_dir: Path, allowed_aids: Set[str], feature_configs: Dict[str, dict],
                                exclude_classes: Set[str] = None) -> Tuple[List[dict], List[str]]:
    exclude_classes = exclude_classes or set()
    classes = sorted(d for d in data_dir.iterdir() if d.is_dir() and d.name not in {"unclassified"})
    class_names = [c.name for c in classes if c.name not in exclude_classes]

    items = []
    dropped_counts = {}

    for cls_dir in classes:
        if cls_dir.name in exclude_classes:
            continue

        for img in collect_images(cls_dir):
            aid = artifact_key_from_path(img)
            if aid not in allowed_aids:
                continue

            cv_img = preprocess_for_opencv(img)
            dino_img = preprocess_for_dinov2(img)

            if cv_img is None or dino_img is None:
                dropped_counts["preprocessing"] = dropped_counts.get("preprocessing", 0) + 1
                continue

            # Compute multiple geometric features based on provided configs
            extracted_shapes = {}
            failed_configs = []

            for conf_key, conf_dict in feature_configs.items():
                shape_f = compute_shape_features(cv_img, config=conf_dict)
                if shape_f is None:
                    failed_configs.append(conf_key)
                else:
                    extracted_shapes[conf_key] = shape_f["feature_vector"]

            if failed_configs:
                print(f"⚠️ Crop Skipped ({img.name}): Extraction failed for configurations {failed_configs}")
                for fc in failed_configs:
                    dropped_counts[fc] = dropped_counts.get(fc, 0) + 1
                continue

            dino_f = extract_visual_embedding(dino_img)
            if dino_f is None:
                dropped_counts["dinov2"] = dropped_counts.get("dinov2", 0) + 1
                continue

            item = {
                "path": img,
                "class": cls_dir.name,
                "aid": aid,
                "dino_vec": dino_f
            }
            for conf_key, vector in extracted_shapes.items():
                item[conf_key] = vector

            items.append(item)

    if dropped_counts:
        print(f"   -> Total number of rejected crops by reasons: {dropped_counts}")

    return items, class_names


def scale_evaluation_features(corpus_items: List[dict], query_items: List[dict],
                              feature_keys: List[str]) -> Tuple[List[dict], List[dict]]:
    scalers = {key: StandardScaler() for key in feature_keys}

    for key in feature_keys:
        scalers[key].fit([item[key] for item in corpus_items])

    def apply_scaling(items: List[dict]) -> List[dict]:
        scaled = []
        for item in items:
            s_item = {
                "class": item["class"],
                "path": str(item["path"]),
                "aid": item["aid"],
                "dino_vec": item["dino_vec"]
            }
            for key in feature_keys:
                s_item[key] = scalers[key].transform([item[key]])[0]
            scaled.append(s_item)
        return scaled

    return apply_scaling(corpus_items), apply_scaling(query_items)


def rank_candidates(query: dict, corpus: List[dict], alpha: float, shape_key: str) -> List[dict]:
    scores = []
    for doc in corpus:
        s_shape = compute_shape_similarity(query[shape_key], doc[shape_key])
        s_embed = compute_embedding_similarity(query["dino_vec"], doc["dino_vec"])
        s_combined = (alpha * s_shape) + ((1.0 - alpha) * s_embed)

        scores.append({
            "class": doc["class"],
            "path": doc["path"],
            "aid": doc["aid"],
            "s_shape": s_shape,
            "s_embed": s_embed,
            "s_hybrid": s_combined
        })

    return sorted(scores, key=lambda x: x["s_hybrid"], reverse=True)
