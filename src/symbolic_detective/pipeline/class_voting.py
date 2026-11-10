from collections import Counter, defaultdict
from pathlib import Path

from symbolic_detective.utils.image_files import artifact_key_from_path


def deduplicate_rankings(rankings, max_votes_per_artifact=1):
    """
    Ensures an artifact only gets up to `max_votes_per_artifact` votes.
    """
    if max_votes_per_artifact < 1:
        raise ValueError("max_votes_per_artifact must be >= 1")

    seen_artifacts = defaultdict(int)
    deduped = []

    for r in rankings:
        aid = r.get("aid")
        if aid is None and r.get("path"):
            aid = artifact_key_from_path(Path(r["path"]))
        if aid is None:
            raise ValueError("Ranking result missing artifact ID and path")

        if seen_artifacts[aid] < max_votes_per_artifact:
            deduped.append(r)
            seen_artifacts[aid] += 1

    return deduped


def predict_class(rankings, k=5, strategy="top1", score_key="s_hybrid", dedup=True):
    """
    Predicts the class based on the ranked results using various voting strategies.

    Assumes `rankings` is already sorted by descending similarity for the selected scoring method.

    Args:
        rankings (list): Sorted list of result dictionaries.
        k (int): Number of top results to consider.
        strategy (str): 'top1', 'majority', or 'weighted'.
        score_key (str): The dictionary key to use for similarity-weighted voting.
        dedup (bool): Whether to enforce a max of 1 vote per source artifact.
    """
    if strategy not in {"top1", "majority", "weighted"}:
        raise ValueError(f"Unknown strategy: {strategy}")

    if k < 1:
        raise ValueError("k must be >= 1")

    if dedup:
        rankings = deduplicate_rankings(rankings)

    top_k = rankings[:k]
    if not top_k:
        return None

    if strategy == "top1":
        return top_k[0]["class"]

    elif strategy == "majority":
        counts = Counter(r["class"] for r in top_k)
        max_votes = max(counts.values())

        # Find all classes that share the maximum vote count
        tied_classes = {
            cls for cls, count in counts.items()
            if count == max_votes
        }

        # Tie-breaker: Highest ranked class among the ties wins
        return next(
            r["class"] for r in top_k
            if r["class"] in tied_classes
        )

    elif strategy == "weighted":
        scores = defaultdict(float)
        for result in top_k:
            scores[result["class"]] += result[score_key]

        return max(scores, key=scores.get)
