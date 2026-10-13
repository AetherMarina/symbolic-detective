"""
Preprocessing routines
Provides dual-stream normalization:
1. OpenCV Stream: CLAHE-enhanced grayscale with boundary safety padding.
2. DINOv2 Stream: Aspect-ratio-preserving square padding resized to 224x224 RGB.
"""
from pathlib import Path
from typing import Tuple, Optional, Union
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


def estimate_border_value(image: np.ndarray) -> int:
    """Estimate a neutral padding value from the image boundary."""
    border = np.concatenate([
        image[0, :],
        image[-1, :],
        image[:, 0],
        image[:, -1],
    ])
    return int(np.median(border))


def preprocess_for_opencv(
    image_input: Union[Path, str, np.ndarray],
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> Optional[np.ndarray]:
    """
    Preprocess a motif crop for geometric contour and moment analysis:
    - Read as grayscale.
    - Apply CLAHE to enhance low-contrast engravings and incisions.
    - Add neutral boundary padding so motif contours do not touch the image border.
    """
    if isinstance(image_input, np.ndarray):
        # Albumentations passes an RGB array. Convert it to Grayscale.
        img = image_input.copy()
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img = cv2.imread(str(image_input), cv2.IMREAD_GRAYSCALE)

    if img is None:
        return None

    # Step 1: Enhance only genuine image pixels.
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size,
    )
    enhanced = clahe.apply(img)

    # Step 2: Add a neutral boundary buffer after enhancement.
    border_value = estimate_border_value(enhanced)

    padded = pad_to_square(
        enhanced,
        fill_color=(border_value,),
        margin_ratio=0.08,
    )

    return padded


def preprocess_for_dinov2(
        image_input: Union[Path, str, np.ndarray],
        target_size: int = 224
) -> Optional[np.ndarray]:
    """
    Preprocesses a motif crop for self-supervised vision models:
    - Preserves aspect ratio with neutral letterboxing.
    - Resizes using area interpolation for downsampling and cubic interpolation for upsampling.
    - Outputs RGB format ready for torch tensor conversion.
    """
    # Step 1: Get image in RGB
    if isinstance(image_input, np.ndarray):
        # Albumentations passes an RGB array.
        img_rgb = image_input.copy()
        # Guardrail in case a 2D grayscale array is accidentally passed
        if len(img_rgb.shape) == 2:
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.imread(str(image_input), cv2.IMREAD_COLOR)
        if img is None:
            return None
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Step 2: Pad to square with neutral gray (128, 128, 128) to avoid harsh contrast borders
    padded = pad_to_square(img_rgb, fill_color=(128, 128, 128), margin_ratio=0.05)
    h, w = padded.shape[:2]

    # Step 3: Resize to model standard input
    resized = cv2.resize(padded, (target_size, target_size),
                         interpolation=(cv2.INTER_AREA if max(h, w) > target_size else cv2.INTER_CUBIC))
    return resized


if __name__ == "__main__":
    import sys

    # Quick test on a sample crop
    if len(sys.argv) > 1:
        test_file = Path(sys.argv[1])
        cv_img = preprocess_for_opencv(test_file)
        dino_img = preprocess_for_dinov2(test_file)

        if cv_img is not None and dino_img is not None:
            print(f"Loaded: {test_file.name}")
            print(f"OpenCV processed shape: {cv_img.shape} (dtype: {cv_img.dtype})")
            print(f"DINOv2 processed shape: {dino_img.shape} (dtype: {dino_img.dtype})")
