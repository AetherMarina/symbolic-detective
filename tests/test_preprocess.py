import pytest
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

from symbolic_detective.pipeline.preprocess import preprocess_for_opencv, preprocess_for_dinov2


@pytest.fixture
def synthetic_star() -> np.ndarray:
    """Returns a 256x256 RGB synthetic star image."""
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    points = np.array([
        [128, 20], [150, 90], [200, 56], [166, 106],
        [236, 128], [166, 150], [200, 200], [150, 166],
        [128, 236], [106, 166], [56, 200], [90, 150],
        [20, 128], [90, 106], [56, 56], [106, 90]
    ], dtype=np.int32)
    cv2.polylines(image, [points], isClosed=True, color=(255, 255, 255), thickness=3)
    return image


@pytest.fixture
def synthetic_non_square() -> np.ndarray:
    """Returns a 100x300 horizontal RGB rectangle."""
    image = np.zeros((100, 300, 3), dtype=np.uint8)
    cv2.rectangle(image, (10, 10), (290, 90), (255, 255, 255), -1)
    return image


def test_opencv_output_dimensions(synthetic_star):
    """Test: OpenCV preprocessing produces the expected dimensions and channels."""
    processed = preprocess_for_opencv(synthetic_star)
    assert processed is not None
    # Verifies it is a 2D grayscale array
    assert processed.ndim == 2
    # The pipeline adds a 20-pixel padding border on all sides (256 + 40 = 296)
    assert processed.shape == (296, 296)


def test_dinov2_output_dimensions(synthetic_star):
    """Test: DINOv2 preprocessing produces the expected 224x224 input size."""
    processed = preprocess_for_dinov2(synthetic_star)
    assert processed is not None

    if isinstance(processed, Image.Image):
        assert processed.size == (224, 224)
    else:
        assert processed.shape[:2] == (224, 224)


def test_aspect_ratio_preservation(synthetic_non_square):
    """Test: Non-square images are padded without geometric distortion."""
    processed = preprocess_for_dinov2(synthetic_non_square)
    image = np.asarray(processed)

    assert image.shape[:2] == (224, 224)

    # Identify the bright rectangular foreground against dark padding
    mask = image.mean(axis=2) > 200
    ys, xs = np.where(mask)
    assert len(xs) > 0, "Foreground object lost during preprocessing."

    width = xs.max() - xs.min() + 1
    height = ys.max() - ys.min() + 1

    # cv2.rectangle includes both endpoints, so dimensions are 281x81
    original_ratio = 281 / 81
    processed_ratio = width / height

    assert processed_ratio == pytest.approx(original_ratio, rel=0.10)


def test_valid_pixel_values(synthetic_star):
    """Test: Output contains valid finite pixel values and correct data types."""
    processed_cv = preprocess_for_opencv(synthetic_star)
    processed_dino = preprocess_for_dinov2(synthetic_star)

    dino_array = np.asarray(processed_dino)

    assert np.isfinite(processed_cv).all()
    assert np.isfinite(dino_array).all()

    assert processed_cv.dtype == np.uint8
    assert dino_array.dtype == np.uint8


def test_grayscale_input():
    """Test: Grayscale images are handled according to preprocessing contract."""
    gray_image = np.ones((256, 256), dtype=np.uint8) * 128

    processed_cv = preprocess_for_opencv(gray_image)
    processed_dino = preprocess_for_dinov2(gray_image)

    assert processed_cv is not None, "OpenCV pipeline failed on grayscale input."
    assert processed_dino is not None, "DINOv2 pipeline failed on grayscale input."

    # DINOv2 requires 3-channel input; verify it was converted up to RGB correctly
    if isinstance(processed_dino, Image.Image):
        assert processed_dino.mode == "RGB"
    else:
        assert processed_dino.shape[2] == 3


def test_invalid_image(tmp_path: Path):
    """Test: Missing or empty files are handled gracefully based on pipeline contract."""
    invalid_path = tmp_path / "non_existent.jpg"
    empty_path = tmp_path / "empty.jpg"
    empty_path.touch()

    # Assuming cv2.imread behavior (returns None).
    # If your pipeline explicitly raises exceptions, change to:
    # with pytest.raises(FileNotFoundError): preprocess_for_opencv(invalid_path)
    assert preprocess_for_opencv(invalid_path) is None
    assert preprocess_for_opencv(empty_path) is None
    assert preprocess_for_dinov2(invalid_path) is None
    assert preprocess_for_dinov2(empty_path) is None


def test_low_contrast_image():
    """Test: Uniform or nearly uniform images do not crash preprocessing (e.g., CLAHE)."""
    low_contrast = np.ones((256, 256, 3), dtype=np.uint8) * 10

    processed_cv = preprocess_for_opencv(low_contrast)
    processed_dino = preprocess_for_dinov2(low_contrast)

    assert processed_cv is not None
    assert processed_dino is not None


def test_deterministic_preprocessing(synthetic_star):
    """Test: Identical input produces identical output across both pipelines."""
    processed_cv1 = preprocess_for_opencv(synthetic_star)
    processed_cv2 = preprocess_for_opencv(synthetic_star.copy())
    np.testing.assert_array_equal(processed_cv1, processed_cv2)

    dino1 = preprocess_for_dinov2(synthetic_star)
    dino2 = preprocess_for_dinov2(synthetic_star.copy())
    np.testing.assert_array_equal(np.asarray(dino1), np.asarray(dino2))
