"""
OpenCV Geometric Feature Extraction

Extracts interpretable geometric descriptors from manually cropped
cultural-heritage motifs.

v0.1 deliberately avoids automatic foreground segmentation.
The manual crop already defines the region of interest.

Features:
1. Log-transformed Hu moments from the complete edge mask.
2. Edge density.
3. Radial edge distribution.
4. Rotation-invariant angular symmetry spectrum.

Feature standardization is performed later in hybrid_search.py using
statistics fitted only on the indexed corpus.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np


def save_edge_debug_image(preprocessed_img: np.ndarray, edge_mask: np.ndarray, output_path: Path) -> None:
    """
    Save a three-panel debug image:

    1. CLAHE-enhanced grayscale crop
    2. Extracted Canny edge mask
    3. Edge overlay on the crop
    """
    grayscale_bgr = cv2.cvtColor(preprocessed_img, cv2.COLOR_GRAY2BGR)
    edges_bgr = cv2.cvtColor(edge_mask, cv2.COLOR_GRAY2BGR)
    overlay = grayscale_bgr.copy()
    overlay[edge_mask > 0] = (0, 0, 255)
    canvas = np.hstack([grayscale_bgr, edges_bgr, overlay])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def extract_edge_mask(
    preprocessed_img: np.ndarray,
    blur_kernel: int = 5,
    canny_sigma: float = 0.33,
    border_margin_ratio: float = 0.03,
) -> np.ndarray:
    """
    Extract geometric edge structure from the complete motif crop.

    Uses:
    - Gaussian smoothing to suppress fine surface texture.
    - Automatic Canny thresholds based on median image intensity.
    - Border suppression to remove crop/padding artifacts.
    """
    if blur_kernel % 2 == 0:
        blur_kernel += 1

    blurred = cv2.GaussianBlur(preprocessed_img, (blur_kernel, blur_kernel), 0)
    median_intensity = float(np.median(blurred))
    lower = int(max(0,(1.0 - canny_sigma)* median_intensity,))
    upper = int(min(255, (1.0 + canny_sigma) * median_intensity,))
    edges = cv2.Canny(blurred, lower, upper)

    # Remove any artificial boundary introduced by crop/padding.
    h, w = edges.shape[:2]
    margin = max(2, int(border_margin_ratio * min(h, w)))
    edges[:margin, :] = 0
    edges[-margin:, :] = 0
    edges[:, :margin] = 0
    edges[:, -margin:] = 0

    return edges


def compute_log_hu(binary_mask: np.ndarray) -> np.ndarray:
    """
    Compute sign-preserving log-transformed Hu moments.

    Hu moments provide translation-, scale-, and rotation-invariant
    global geometric descriptors.
    """
    moments = cv2.moments(binary_mask, binaryImage=True)
    hu = cv2.HuMoments(moments).flatten()
    log_hu = np.zeros(7, dtype=np.float32)

    for i, value in enumerate(hu):
        if value != 0:
            log_hu[i] = (-np.sign(value) * np.log10(abs(value)))

    return log_hu


def compute_radial_histogram(edge_mask: np.ndarray, bins: int = 6) -> np.ndarray:
    """
    Describe how edge pixels are distributed from the crop center.

    Radius is normalized, making the descriptor largely independent
    of absolute crop size.
    """
    ys, xs = np.nonzero(edge_mask)

    if len(xs) == 0:
        return np.zeros(bins, dtype=np.float32)

    h, w = edge_mask.shape[:2]
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    dx = xs.astype(np.float32) - cx
    dy = ys.astype(np.float32) - cy

    radii = np.sqrt(dx ** 2 + dy ** 2)
    max_radius = np.sqrt(cx ** 2 + cy ** 2)

    if max_radius > 0:
        radii = radii / max_radius

    histogram, _ = np.histogram(radii, bins=bins, range=(0.0, 1.0))
    histogram = histogram.astype(np.float32)
    total = histogram.sum()

    if total > 0:
        histogram /= total

    return histogram


def compute_angular_spectrum(edge_mask: np.ndarray, angular_bins: int = 36, harmonics: int = 12) -> np.ndarray:
    """
    Compute a rotation-invariant angular symmetry descriptor.

    Edge pixels are represented by their angle around the crop center.

    The angular histogram itself changes when the motif rotates, but
    the magnitude of its Fourier transform does not. This makes the
    resulting spectrum useful for comparing rotated radial motifs.

    For example, an eight-rayed motif may produce strong periodic
    structure around harmonic 8.
    """
    ys, xs = np.nonzero(edge_mask)

    if len(xs) == 0:
        return np.zeros(harmonics, dtype=np.float32)

    h, w = edge_mask.shape[:2]

    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    dx = xs.astype(np.float32) - cx
    dy = ys.astype(np.float32) - cy

    angles = np.arctan2(dy, dx)

    # Convert [-pi, pi] -> [0, 2pi].
    angles = (angles + 2.0 * np.pi) % (2.0 * np.pi)
    angular_hist, _ = np.histogram(angles, bins=angular_bins, range=(0.0, 2.0 * np.pi))
    angular_hist = angular_hist.astype(np.float32)
    total = angular_hist.sum()

    if total > 0:
        angular_hist /= total

    spectrum = np.abs(np.fft.rfft(angular_hist))
    # Ignore the DC component at index 0.
    available = spectrum[1:]
    result = np.zeros(harmonics, dtype=np.float32)
    count = min(harmonics, len(available))
    result[:count] = available[:count]

    return result


def compute_shape_features(preprocessed_img: np.ndarray, debug_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """
    Extract an interpretable geometric feature vector.
    Feature vector:
        7  log-transformed Hu moments
        1  edge density
        6  radial histogram bins
        12 angular Fourier magnitudes
    Total: 26 dimensions. The vector is returned raw.
    Standardization must later be fitted only on the indexed corpus inside the retrieval/search pipeline.
    """
    edge_mask = extract_edge_mask(preprocessed_img)
    edge_count = int(cv2.countNonZero(edge_mask))

    if edge_count == 0:
        return None

    if debug_path is not None:
        save_edge_debug_image(preprocessed_img, edge_mask, debug_path)

    # 1. Global geometry
    log_hu = compute_log_hu(edge_mask)

    # 2. Edge density
    edge_density = (edge_count / float(edge_mask.size))

    # 3. Radial structure
    radial_histogram = (compute_radial_histogram(edge_mask, bins=6))

    # 4. Rotational symmetry structure
    angular_spectrum = (compute_angular_spectrum(edge_mask, angular_bins=36, harmonics=12))

    # 5. Final descriptor
    feature_vector = np.concatenate(
        [log_hu, np.array([edge_density], dtype=np.float32), radial_histogram, angular_spectrum]
    ).astype(np.float32)

    return {
        "feature_vector": feature_vector,
        "edge_mask": edge_mask,
        "log_hu": log_hu,
        "edge_density": float(edge_density),
        "radial_histogram": radial_histogram,
        "angular_spectrum": angular_spectrum,
    }


def compute_shape_similarity(vec_a: np.ndarray, vec_b: np.ndarray, gamma: float = 0.25) -> float:
    """
    Compute geometric similarity in [0, 1].
    IMPORTANT:
    vec_a and vec_b must already be standardized using statistics fitted from the indexed corpus only.
    """
    distance = float(np.linalg.norm(vec_a - vec_b))
    similarity = float(np.exp(-gamma * distance))

    return float(np.clip(similarity, 0.0, 1.0))


if __name__ == "__main__":
    import sys
    from symbolic_detective.pipeline.preprocess import (preprocess_for_opencv)

    if len(sys.argv) < 2:
        print("Usage: python -m " "symbolic_detective.pipeline.shape_features " "<crop_path>")
        raise SystemExit(1)

    test_path = Path(sys.argv[1])
    processed = preprocess_for_opencv(test_path)

    if processed is None:
        print(f"Failed to preprocess: " f"{test_path}")
        raise SystemExit(1)

    debug_out = (Path("data/debug") / f"debug_{test_path.stem}.png")
    features = compute_shape_features(processed, debug_path=debug_out)

    if features is None:
        print("Failed to extract geometric features for " f"{test_path.name}")
        raise SystemExit(1)

    print(f"Extracted geometric features for: " f"{test_path.name}")
    print(f"Debug image saved to: " f"{debug_out}")
    print(f"Edge density: " f"{features['edge_density']:.4f}")
    print("Log Hu moments: " f"{np.round(features['log_hu'], 3)}")
    print("Radial histogram: " f"{np.round(features['radial_histogram'], 3)}")
    print("Angular spectrum: " f"{np.round(features['angular_spectrum'], 3)}")
    print(f"Feature vector dimension: " f"{len(features['feature_vector'])}")
