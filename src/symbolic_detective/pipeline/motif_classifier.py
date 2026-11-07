from pathlib import Path
from typing import Dict, Any

from symbolic_detective.pipeline.class_voting import predict_class, deduplicate_rankings
from symbolic_detective.evaluation.configs import CLASSIFICATION_FROZEN_CONFIG, MAX_SIM_THRESHOLD


def classify_motif(query_path: Path, classification_index) -> Dict[str, Any]:
    """
    Classifies a query motif using the frozen classification configuration and
    applies a calibrated maximum similarity threshold for OOD rejection.
    """
    config = CLASSIFICATION_FROZEN_CONFIG

    # Ensure the classification gallery contains only known classes
    assert all(item["class"] != "negative_control" for item in classification_index.corpus)

    # Retrieve all candidates for artifact-deduplicated voting
    ranked = classification_index.search(
        query_path,
        alpha=config["alpha"],
        top_k=None,
    )

    if not ranked:
        raise ValueError("Classification index has no candidates.")

    # Classify using frozen Weighted K=7
    predicted = predict_class(
        ranked,
        k=config["k"],
        strategy=config["strategy"],
        score_key="s_hybrid",
        dedup=True,
    )

    # Calculate confidence scores
    deduped = deduplicate_rankings(ranked)
    top_k = deduped[:config["k"]]

    max_sim = 0.0
    weight_conf = 0.0

    if top_k:
        max_sim = float(ranked[0]["s_hybrid"])

        class_weights = {}
        total_weight = 0.0
        for neighbor in top_k:
            w = max(0.0, float(neighbor["s_hybrid"]))
            cls = neighbor["class"]
            class_weights[cls] = class_weights.get(cls, 0.0) + w
            total_weight += w

        if total_weight > 0:
            weight_conf = class_weights.get(predicted, 0.0) / total_weight

    # Rejection uses the maximum similarity
    rejected = max_sim < MAX_SIM_THRESHOLD

    return {
        "predicted_class": "unknown" if rejected else predicted,
        "raw_prediction": predicted,
        "max_sim": max_sim,
        "weight_conf": weight_conf,
        "rejected": rejected,
        "threshold_used": MAX_SIM_THRESHOLD
    }
