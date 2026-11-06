from typing import List, Dict, Any, Tuple
from sklearn.metrics import confusion_matrix


def calculate_metrics(rankings: List[Dict[str, Any]], true_class: str) -> Tuple[int, int, float]:
    """Computes Hit@1, Hit@5, and Reciprocal Rank for ranked retrieval outputs."""
    if not rankings:
        return 0, 0, 0.0
    hit_1 = 1 if rankings[0]["class"] == true_class else 0
    hit_5 = 1 if any(r["class"] == true_class for r in rankings[:5]) else 0

    reciprocal_rank = 0.0
    for rank, item in enumerate(rankings, start=1):
        if item["class"] == true_class:
            reciprocal_rank = 1.0 / rank
            break

    return hit_1, hit_5, reciprocal_rank


def build_confusion_matrix(y_true: List[str], y_pred: List[str], class_names: List[str]):
    """Constructs a confusion matrix, explicitly appending the negative_control class."""
    labels = [c for c in class_names if c != "negative_control"]
    labels.append("negative_control")
    return confusion_matrix(y_true, y_pred, labels=labels,), labels
