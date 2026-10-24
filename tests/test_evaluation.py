import pytest

# Assuming evaluate.py is accessible via the package namespace
from symbolic_detective.evaluate import (
    calculate_metrics,
    build_confusion_matrix,
)


def test_hit_at_1():
    """Test: Hit@1 is 1 only when the first result has the correct class."""
    rankings_correct_first = [
        {"class": "eight_pointed_star"},
        {"class": "five_pointed_star"},
        {"class": "rosette_other"}
    ]
    h1, h5, mrr = calculate_metrics(rankings_correct_first, "eight_pointed_star")
    assert h1 == 1
    assert h5 == 1
    assert mrr == pytest.approx(1.0)

    rankings_incorrect_first = [
        {"class": "rosette_other"},
        {"class": "eight_pointed_star"},
    ]
    h1, h5, mrr = calculate_metrics(rankings_incorrect_first, "eight_pointed_star")
    assert h1 == 0
    assert h5 == 1
    assert mrr == pytest.approx(0.5)


def test_hit_at_5():
    """Test: Hit@5 is 1 when the correct class appears within the first five results."""
    # Correct match is at rank 4 (index 3)
    rankings_hit_5 = [
        {"class": "a"}, {"class": "b"}, {"class": "c"},
        {"class": "target"}, {"class": "d"}, {"class": "e"}
    ]
    h1, h5, mrr = calculate_metrics(rankings_hit_5, "target")
    assert h1 == 0
    assert h5 == 1
    assert mrr == pytest.approx(0.25)

    # Correct match is at rank 6 (index 5) - should fail Hit@5
    rankings_miss_5 = [
        {"class": "a"}, {"class": "b"}, {"class": "c"},
        {"class": "d"}, {"class": "e"}, {"class": "target"}
    ]
    h1, h5, mrr = calculate_metrics(rankings_miss_5, "target")
    assert h1 == 0
    assert h5 == 0
    assert mrr == pytest.approx(1 / 6)


def test_mrr():
    """Test: Reciprocal rank is calculated from the first correct match."""
    rankings = [
        {"class": "wrong1"},
        {"class": "wrong2"},
        {"class": "target"},  # Rank 3 -> MRR 1/3
        {"class": "target"}  # Rank 4 -> Should be ignored
    ]
    _, _, mrr = calculate_metrics(rankings, "target")
    assert mrr == pytest.approx(1 / 3)


def test_no_correct_match():
    """Test: Metrics are zero when no correct match exists."""
    rankings = [
        {"class": "wrong1"},
        {"class": "wrong2"},
        {"class": "wrong3"}
    ]
    h1, h5, mrr = calculate_metrics(rankings, "target")
    assert h1 == 0
    assert h5 == 0
    assert mrr == pytest.approx(0.0)


def test_empty_rankings():
    """Test: Empty retrieval results should produce zero metrics."""
    assert calculate_metrics([], "eight_pointed_star") == (0, 0, 0.0)


def test_confusion_matrix():
    """Test: True and predicted labels are counted correctly for valid motif classes."""
    y_true = ["eight_pointed_star", "eight_pointed_star", "rosette_other", "five_pointed_star"]
    y_pred = ["eight_pointed_star", "rosette_other", "rosette_other", "five_pointed_star"]
    class_names = ["eight_pointed_star", "five_pointed_star", "rosette_other"]

    cm, labels = build_confusion_matrix(y_true, y_pred, class_names)

    # Ensure negative_control was dynamically appended
    assert labels == ["eight_pointed_star", "five_pointed_star", "rosette_other", "negative_control"]

    # Check diagonals (correct predictions)
    assert cm[0, 0] == 1  # 1 eight_pointed_star correct
    assert cm[1, 1] == 1  # 1 five_pointed_star correct
    assert cm[2, 2] == 1  # 1 rosette_other correct

    # Check off-diagonal (incorrect prediction)
    assert cm[0, 2] == 1  # 1 eight_pointed_star predicted as rosette_other

    # Verify no predictions were silently dropped
    assert cm.sum() == len(y_true)


def test_negative_control_predictions():
    """Test: Incorrect predictions of negative_control are included in the confusion matrix."""
    y_true = ["eight_pointed_star"]
    y_pred = ["negative_control"]

    cm, labels = build_confusion_matrix(
        y_true,
        y_pred,
        ["eight_pointed_star", "five_pointed_star"],
    )

    assert labels == [
        "eight_pointed_star",
        "five_pointed_star",
        "negative_control",
    ]
    assert cm.shape == (3, 3)
    assert cm[0, 2] == 1
    assert cm.sum() == len(y_true)
