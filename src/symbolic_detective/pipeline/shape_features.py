"""
OpenCV Geometric Feature Extraction
Extracts interpretable shape descriptors (Hu Moments + geometric ratios)
from preprocessed cultural-heritage motif crops.
"""
from pathlib import Path
from typing import Dict, Any, Optional
import cv2
import numpy as np


def save_contour_debug_image(
    original_clahe: np.ndarray,
    binary: np.ndarray,
    contour: np.ndarray,
    output_path: Path
):
    """
    Renders a 3-panel canvas: Enhanced Grayscale, Thresholded Mask, and Extracted Contour.
    """
    vis = cv2.cvtColor(original_clahe, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(vis, [contour], -1, (0, 0, 255), 2)

    binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    clahe_bgr = cv2.cvtColor(original_clahe, cv2.COLOR_GRAY2BGR)

    canvas = np.hstack([clahe_bgr, binary_bgr, vis])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def extract_salient_contour(binary_img: np.ndarray) -> Optional[np.ndarray]:
    # Look for both external and internal contours
    contours, hierarchy = cv2.findContours(
        binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours or hierarchy is None:
        return None

    h, w = binary_img.shape[:2]
    center = np.array([w / 2.0, h / 2.0])
    total_area = float(h * w)

    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.01 * total_area:
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = (area / hull_area) if hull_area > 0 else 0.0

        # Discard bounding box frames or full-crop circles
        if area > 0.75 * total_area and solidity > 0.92:
            continue

        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            dist_from_center = np.linalg.norm(np.array([cx, cy]) - center)
            valid_contours.append((area, dist_from_center, solidity, cnt))

    if not valid_contours:
        return max(contours, key=cv2.contourArea)

    # Rank: prefer motifs near the center with lower solidity (star/radial geometry)
    valid_contours.sort(
        key=lambda x: (x[0] / (1.0 + 0.2 * (x[1] / max(w, h)))) * (1.2 - min(x[2], 0.95)),
        reverse=True
    )
    return valid_contours[0][3]


def binarize_crop_adaptive(clahe_img: np.ndarray) -> np.ndarray:
    """
    Combines adaptive bilateral filtering with Canny edge detection
    and morphological closing to isolate internal carved lines rather than outer disc borders.
    """
    # 1. Bilateral filter preserves sharp relief edges while smoothing stone noise
    filtered = cv2.bilateralFilter(clahe_img, d=7, sigmaColor=50, sigmaSpace=50)

    # 2. Adaptive thresholding captures local relief changes
    adaptive = cv2.adaptiveThreshold(
        filtered,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=21,
        C=4
    )

    # 3. Morphological close to bridge broken star rays
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)

    return closed


def compute_shape_features(clahe_img: np.ndarray, debug_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """
    Extracts a normalized geometric descriptor vector from a preprocessed crop.

    Feature Vector Components:
      - 7 Log-transformed Hu Moments (scale/rotation invariant)
      - Compactness (4 * pi * Area / Perimeter^2)
      - Solidity (Contour Area / Convex Hull Area)
      - Extent (Contour Area / Bounding Box Area)
      - Defect Count (Number of significant convexity defects)
    """
    binary = binarize_crop_adaptive(clahe_img)
    contour = extract_salient_contour(binary)
    if contour is None or len(contour) < 5:
        return None

    # Save visual debug canvas if a path was provided
    if debug_path is not None:
        save_contour_debug_image(clahe_img, binary, contour, debug_path)

    # 1. Hu Moments with sign-preserving log transformation
    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments).flatten()
    log_hu = np.zeros(7, dtype=np.float32)
    for i in range(7):
        val = hu_moments[i]
        if val != 0:
            log_hu[i] = -1.0 * np.sign(val) * np.log10(np.abs(val))

    # 2. Geometric Scalars
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, closed=True)
    compactness = (4.0 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = (area / hull_area) if hull_area > 0 else 0.0

    x, y, w, h = cv2.boundingRect(contour)
    bbox_area = float(w * h)
    extent = (area / bbox_area) if bbox_area > 0 else 0.0

    # 3. Convexity defects (captures star points vs. rounded lobes)
    defect_count = 0
    try:
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        if hull_indices is not None and len(hull_indices) > 3:
            defects = cv2.convexityDefects(contour, hull_indices)
            if defects is not None:
                # Squeeze out any extra dimensions (handles both (N, 1, 4) and (N, 4) shapes safely)
                defects_squeezed = defects.reshape(-1, 4)

                # Count defects deeper than 5% of bounding dimension
                threshold_depth = 0.05 * max(w, h) * 256
                defect_count = sum(1 for d in defects_squeezed if d[3] > threshold_depth)
    except cv2.error:
        defect_count = 0

    # Combined 11-dimensional feature vector
    scalar_features = np.array([compactness, solidity, extent, defect_count / 12.0], dtype=np.float32)
    feature_vector = np.concatenate([log_hu, scalar_features])

    return {
        "feature_vector": feature_vector,
        "contour": contour,
        "log_hu": log_hu,
        "compactness": float(compactness),
        "solidity": float(solidity),
        "extent": float(extent),
        "defects": int(defect_count)
    }


def compute_shape_similarity(vec_a: np.ndarray, vec_b: np.ndarray, gamma: float = 0.25) -> float:
    """
    Computes normalized shape similarity S_shape in [0, 1] using exponential
    decay over Euclidean distance.
    """
    distance = float(np.linalg.norm(vec_a - vec_b))
    similarity = float(np.exp(-gamma * distance))
    return max(0.0, min(1.0, similarity))


if __name__ == "__main__":
    import sys
    from symbolic_detective.pipeline.preprocess import preprocess_for_opencv

    if len(sys.argv) > 1:
        test_path = Path(sys.argv[1])
        processed = preprocess_for_opencv(test_path)

        if processed is not None:
            debug_out = Path("data/debug") / f"debug_{test_path.stem}.png"
            features = compute_shape_features(processed, debug_path=debug_out)

            if features:
                print(f"Extracted shape features for: {test_path.name}")
                print(f"Debug image saved to: {debug_out}")
                print(f"Solidity: {features['solidity']:.4f} | Compactness: {features['compactness']:.4f}")
                print(f"Significant defects: {features['defects']}")
                print(f"Log Hu Moments: {np.round(features['log_hu'][:4], 3)}")
            else:
                print(f"Failed to extract salient contour for {test_path.name}")
