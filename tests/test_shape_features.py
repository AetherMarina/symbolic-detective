import pytest
import numpy as np
import cv2

from symbolic_detective.pipeline.preprocess import preprocess_for_opencv
from symbolic_detective.pipeline.shape_features import compute_shape_features, compute_shape_similarity


@pytest.fixture
def synthetic_star() -> np.ndarray:
    """Returns a processed synthetic star."""
    image = np.zeros((256, 256), dtype=np.uint8)
    points = np.array([
        [128, 20], [150, 90], [200, 56], [166, 106],
        [236, 128], [166, 150], [200, 200], [150, 166],
        [128, 236], [106, 166], [56, 200], [90, 150],
        [20, 128], [90, 106], [56, 56], [106, 90]
    ], dtype=np.int32)
    cv2.fillPoly(image, [points], color=255)
    return preprocess_for_opencv(image)


@pytest.fixture
def synthetic_circle() -> np.ndarray:
    """Returns a processed synthetic circle."""
    image = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(image, (128, 128), 80, 255, -1)
    return preprocess_for_opencv(image)


@pytest.fixture
def synthetic_blank() -> np.ndarray:
    """Returns a blank processed image."""
    image = np.zeros((256, 256), dtype=np.uint8)
    return preprocess_for_opencv(image)


def test_feature_vector_dimensions(synthetic_star, synthetic_circle):
    """Test: Feature vector has a fixed length for different valid images."""
    f_star = compute_shape_features(synthetic_star)
    f_circle = compute_shape_features(synthetic_circle)

    assert f_star is not None
    assert f_circle is not None

    vec_star = f_star["feature_vector"]
    vec_circle = f_circle["feature_vector"]

    assert len(vec_star) > 0
    assert len(vec_star) == len(vec_circle)


def test_finite_features(synthetic_star):
    """Test: No NaN or Inf in extracted features."""
    features = compute_shape_features(synthetic_star)["feature_vector"]

    assert np.isfinite(features).all()


def test_feature_determinism(synthetic_star):
    """Test: Identical images produce identical feature vectors."""
    f1 = compute_shape_features(synthetic_star)["feature_vector"]
    f2 = compute_shape_features(synthetic_star.copy())["feature_vector"]

    np.testing.assert_array_equal(f1, f2)


def test_identical_shape_similarity(synthetic_star):
    """Test: Identical feature vectors yield maximum similarity."""
    features = compute_shape_features(synthetic_star)["feature_vector"]
    sim = compute_shape_similarity(features, features)

    assert sim == pytest.approx(1.0)


def test_similarity_range(synthetic_star, synthetic_circle):
    """Test: Similarity scores remain within the expected [0, 1] range."""
    f_star = compute_shape_features(synthetic_star)["feature_vector"]
    f_circle = compute_shape_features(synthetic_circle)["feature_vector"]

    sim = compute_shape_similarity(f_star, f_circle)

    assert 0.0 <= sim <= 1.0


def test_rotated_shape_similarity():
    """Test: Extracting features from rotated images produces valid bounded scores."""
    image = np.zeros((256, 256), dtype=np.uint8)
    # Use an asymmetric shape to make rotation meaningful
    cv2.rectangle(image, (40, 60), (150, 100), 255, -1)

    original = preprocess_for_opencv(image)
    rotated = preprocess_for_opencv(np.rot90(image).copy())

    f_original = compute_shape_features(original)["feature_vector"]
    f_rotated = compute_shape_features(rotated)["feature_vector"]

    sim = compute_shape_similarity(f_original, f_rotated)

    assert np.isfinite(sim)
    assert 0.0 <= sim <= 1.0


def test_different_shapes(synthetic_star, synthetic_circle):
    """Test: Different shapes generate distinguishable feature vectors."""
    f_star = compute_shape_features(synthetic_star)["feature_vector"]
    f_circle = compute_shape_features(synthetic_circle)["feature_vector"]

    # Assert that the feature vectors are distinctly different arrays
    assert not np.allclose(f_star, f_circle)


def test_blank_image(synthetic_blank):
    """Test: Empty or featureless images are explicitly rejected."""
    if synthetic_blank is None:
        return

    features = compute_shape_features(synthetic_blank)
    assert features is None, "Feature extractor should return None for images without valid contours."