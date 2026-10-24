import pytest
from symbolic_detective.pipeline.class_voting import predict_class


@pytest.fixture
def sample_rankings():
    """Returns a pre-sorted list of mock rankings."""
    return [
        {"class": "five_pointed_star", "aid": "art1", "s_hybrid": 0.94},
        {"class": "six_pointed_star", "aid": "art2", "s_hybrid": 0.92},
        {"class": "six_pointed_star", "aid": "art3", "s_hybrid": 0.90},
        {"class": "eight_pointed_star", "aid": "art4", "s_hybrid": 0.88},
        {"class": "six_pointed_star", "aid": "art5", "s_hybrid": 0.86},
    ]


def test_top1_prediction(sample_rankings):
    """Returns first candidate's class."""
    prediction = predict_class(sample_rankings, k=5, strategy="top1")
    assert prediction == "five_pointed_star"


def test_majority_voting(sample_rankings):
    """Returns most frequent class."""
    prediction = predict_class(sample_rankings, k=5, strategy="majority")
    assert prediction == "six_pointed_star"  # 3 votes vs 1 for others


def test_majority_tie():
    """Selects highest-ranked tied class."""
    rankings = [
        {"class": "five_pointed_star", "aid": "art1", "s_hybrid": 0.95},
        {"class": "six_pointed_star", "aid": "art2", "s_hybrid": 0.94},
        {"class": "six_pointed_star", "aid": "art3", "s_hybrid": 0.90},
        {"class": "five_pointed_star", "aid": "art4", "s_hybrid": 0.80},
    ]
    # At K=4, both classes have 2 votes. 'five_pointed_star' is ranked highest (index 0).
    prediction = predict_class(rankings, k=4, strategy="majority")
    assert prediction == "five_pointed_star"


def test_weighted_voting(sample_rankings):
    """Returns class with highest summed score."""
    # five_pointed_star = 0.94
    # six_pointed_star = 0.92 + 0.90 + 0.86 = 2.68
    # eight_pointed_star = 0.88
    prediction = predict_class(sample_rankings, k=5, strategy="weighted")
    assert prediction == "six_pointed_star"


def test_weighted_tie():
    """Selects highest-ranked tied class."""
    rankings = [
        {"class": "A", "aid": "art1", "s_hybrid": 0.8},
        {"class": "B", "aid": "art2", "s_hybrid": 0.8},
    ]
    # Both sum to 0.8. Since Python dicts preserve insertion order and max()
    # returns the first maximum encountered, 'A' should win.
    prediction = predict_class(rankings, k=2, strategy="weighted")
    assert prediction == "A"


def test_duplicate_artifact():
    """Only highest-ranked crop contributes."""
    rankings = [
        {"class": "eight_pointed_star", "aid": "art1", "s_hybrid": 0.99},
        {"class": "eight_pointed_star", "aid": "art1", "s_hybrid": 0.95},  # Duplicate artifact
        {"class": "six_pointed_star", "aid": "art2", "s_hybrid": 0.90},
        {"class": "six_pointed_star", "aid": "art3", "s_hybrid": 0.85},
    ]

    # Without deduplication, top 3 would be [eight, eight, six] -> majority wins 'eight'
    # With deduplication, top 3 becomes [eight, six, six] -> majority wins 'six'
    prediction = predict_class(rankings, k=3, strategy="majority", dedup=True)
    assert prediction == "six_pointed_star"
    assert predict_class(rankings, k=3, strategy="majority", dedup=False) == "eight_pointed_star"


def test_empty_rankings():
    """Returns None for empty rankings."""
    assert predict_class([], k=5) is None


def test_invalid_strategy_and_k(sample_rankings):
    """Raises ValueError for bad parameters."""
    with pytest.raises(ValueError, match="Unknown strategy"):
        predict_class(sample_rankings, k=5, strategy="invalid_strategy")

    with pytest.raises(ValueError, match="k must be >= 1"):
        predict_class(sample_rankings, k=0, strategy="majority")


def test_missing_similarity_score():
    """Raises KeyError for weighted voting when score is missing."""
    rankings = [
        {"class": "A", "aid": "art1"},  # Missing 's_hybrid'
        {"class": "B", "aid": "art2", "s_hybrid": 0.9}
    ]
    with pytest.raises(KeyError):
        predict_class(rankings, k=2, strategy="weighted")


def test_deduplication_from_paths():
    rankings = [
        {"class": "A", "path": "123_primary_occ_01.jpg"},
        {"class": "A", "path": "123_primary_occ_02.jpg"},
        {"class": "B", "path": "456_primary_occ_01.jpg"},
        {"class": "B", "path": "789_primary_occ_01.jpg"},
    ]

    assert predict_class(rankings, k=3, strategy="majority") == "B"


def test_custom_score_key():
    rankings = [
        {"class": "A", "aid": "art1", "s_embedding": 0.9},
        {"class": "B", "aid": "art2", "s_embedding": 0.8},
        {"class": "B", "aid": "art3", "s_embedding": 0.7},
    ]

    assert predict_class(rankings, k=3, strategy="weighted", score_key="s_embedding",) == "B"
