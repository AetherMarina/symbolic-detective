"""
Preprocessing routines
Provides dual-stream normalization:
1. OpenCV Stream: CLAHE-enhanced grayscale with boundary safety padding.
2. DINOv2 Stream: Aspect-ratio-preserving square padding resized to 224x224 RGB.
"""
from pathlib import Path
from typing import Tuple, Optional
import cv2
import numpy as np


def pad_to_square(
        image: np.ndarray,
        fill_color: Tuple[int, ...] = (0, 0, 0),
        margin_ratio: float = 0.05
) -> np.ndarray:
    """
    Pads an image to a square aspect ratio while preserving original geometry,
    adding an optional safety margin around the border.
    """
    h, w = image.shape[:2]
    max_dim = max(h, w)

    # Add small margin so edge contours aren't fused to image borders
    margin = int(max_dim * margin_ratio)
    target_dim = max_dim + (2 * margin)

    pad_top = (target_dim - h) // 2
    pad_bottom = target_dim - h - pad_top
    pad_left = (target_dim - w) // 2
    pad_right = target_dim - w - pad_left

    if len(image.shape) == 3:
        border_val = fill_color
    else:
        border_val = fill_color[0] if isinstance(fill_color, tuple) else fill_color

    padded = cv2.copyMakeBorder(
        image,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=border_val
    )
    return padded


def preprocess_for_opencv(
        crop_path: Path,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8)
) -> Optional[np.ndarray]:
    """
    Preprocesses a motif crop for geometric contour and moment analysis:
    - Reads in BGR/Grayscale.
    - Adds boundary padding.
    - Applies CLAHE to bring out low-contrast engravings and incisions.
    """
    img = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # Step 1: Add boundary buffer
    padded = pad_to_square(img, fill_color=(0,), margin_ratio=0.08)

    # Step 2: Adaptive histogram equalization for weathered/uneven textures
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(padded)

    return enhanced


def preprocess_for_dinov2(
        crop_path: Path,
        target_size: int = 224
) -> Optional[np.ndarray]:
    """
    Preprocesses a motif crop for self-supervised vision models:
    - Preserves aspect ratio with neutral letterboxing.
    - Resizes cleanly using cubic interpolation.
    - Outputs RGB format ready for torch tensor conversion.
    """
    img = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
    if img is None:
        return None

    # Step 1: Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Step 2: Pad to square with neutral gray (128, 128, 128) to avoid harsh contrast borders
    padded = pad_to_square(img_rgb, fill_color=(128, 128, 128), margin_ratio=0.0)

    # Step 3: Resize to model standard input
    resized = cv2.resize(padded, (target_size, target_size), interpolation=cv2.INTER_CUBIC)

    return resized


if __name__ == "__main__":
    import sys

    # Quick test harness on a sample crop
    if len(sys.argv) > 1:
        test_file = Path(sys.argv[1])
        cv_img = preprocess_for_opencv(test_file)
        dino_img = preprocess_for_dinov2(test_file)

        if cv_img is not None and dino_img is not None:
            print(f"Loaded: {test_file.name}")
            print(f"OpenCV processed shape: {cv_img.shape} (dtype: {cv_img.dtype})")
            print(f"DINOv2 processed shape: {dino_img.shape} (dtype: {dino_img.dtype})")
