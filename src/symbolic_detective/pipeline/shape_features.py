"""
OpenCV Geometric Feature Extraction

Extracts interpretable geometric descriptors from manually cropped
cultural-heritage motifs.

v0.1 Baseline Features (Always Active):
1. Log-transformed Hu moments from the complete edge mask.
2. Edge density.
3. Radial edge distribution.
4. Rotation-invariant angular symmetry spectrum.

v0.2 Ablation Modules (Toggled via Config):
Extracts the outermost contour and unwraps it into a 1D boundary signal
(radius vs. angle) to evaluate specific explicit structural properties:
- Harmonics: Fractional power of target k-fold symmetries (e.g., 5, 6, 8).
- Peaks: Count and average prominence of radial points/petals.
- Pointedness: Geometric sharpness measured via peak widths.
- Radius: Proportion of inner valley radius to outer peak radius.
- Central Structure: Internal feature density (e.g., central disk vs. empty core).

Feature standardization is performed later in hybrid_search.py using
statistics fitted only on the indexed corpus.
"""
from pathlib import Path
from typing import Any, Dict, Optional, List

import cv2
import numpy as np
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import gaussian_filter1d

FEATURE_SET = {
    "harmonics": False,
    "peaks": False,
    "pointedness": False,
    "radius": False,
    "central_structure": False
}


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


def compute_base_features(edge_mask: np.ndarray) -> List[float]:
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
    edge_count = int(cv2.countNonZero(edge_mask))
    if edge_count == 0:
        return [0.0] * 26  # 7 hu + 1 density + 6 radial + 12 angular

    log_hu = compute_log_hu(edge_mask)
    edge_density = (edge_count / float(edge_mask.size))
    radial_histogram = compute_radial_histogram(edge_mask, bins=6)
    angular_spectrum = compute_angular_spectrum(edge_mask, angular_bins=36, harmonics=12)

    base_vector = np.concatenate(
        [log_hu, np.array([edge_density], dtype=np.float32), radial_histogram, angular_spectrum]
    ).tolist()

    return base_vector


def compute_harmonic_features(contour_angles: np.ndarray, contour_radii: np.ndarray) -> List[float]:
    """
    Computes the fractional power of 5-fold, 6-fold, and 8-fold radial symmetries
    by interpolating the boundary signal onto a uniform angular grid before FFT.
    Uses the power spectrum (squared magnitude) to measure harmonic strength.
    """
    if len(contour_radii) < 10:
        return [0.0, 0.0, 0.0]

    # Ensure angles are in the [0, 2*pi) range
    contour_angles = np.mod(contour_angles, 2 * np.pi)

    # Sort radial distances by their angle
    sorted_indices = np.argsort(contour_angles)
    angles_sorted = contour_angles[sorted_indices]
    radii_sorted = contour_radii[sorted_indices]

    # Get unique angles and find the maximum radius for any duplicate angles
    # This ensures we capture the outermost boundary of the motif (peaks)
    angles_unique, inverse_indices = np.unique(angles_sorted, return_inverse=True)
    radii_max = np.zeros_like(angles_unique, dtype=np.float32)
    np.maximum.at(radii_max, inverse_indices, radii_sorted)

    if len(angles_unique) < 10:
        return [0.0, 0.0, 0.0]

    # Periodic signal padding (wrap-around) for correct interpolation at the 0/2pi seam
    angles_padded = np.concatenate([angles_unique - 2 * np.pi, angles_unique, angles_unique + 2 * np.pi])
    radii_padded = np.concatenate([radii_max, radii_max, radii_max])

    # Create a uniform angular grid of 360 points
    uniform_angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)

    # Interpolate the radial signal onto the uniform grid
    uniform_radii = np.interp(uniform_angles, angles_padded, radii_padded)

    # Detrending (remove mean) to isolate variations from the base circle
    signal = uniform_radii - np.mean(uniform_radii)

    # Compute the power spectrum (squared magnitude of the Real FFT)
    spectrum = np.abs(np.fft.rfft(signal)) ** 2

    if len(spectrum) <= 8:
        spectrum = np.pad(spectrum, (0, 9 - len(spectrum)), 'constant')

    # Normalize by total power (excluding the DC component at index 0)
    total_power = np.sum(spectrum[1:])
    if total_power > 0:
        spectrum = spectrum / total_power

    # Return the fraction of non-DC radial boundary power explained by target periodicities
    return [
        float(spectrum[5]),  # Power of 5-fold symmetry
        float(spectrum[6]),  # Power of 6-fold symmetry
        float(spectrum[8])   # Power of 8-fold symmetry
    ]


def compute_radial_peak_features(contour_angles: np.ndarray, contour_radii: np.ndarray,
                                 prominence_threshold: float = 0.05) -> List[float]:
    """
    Counts the number of structural peaks (star points / petals) and their
    average prominence along the contour boundary. Evaluates the signal on a
    uniform angular grid to map peaks directly to physical rays/petals.
    """
    if len(contour_radii) < 10:
        return [0.0, 0.0]

    # Map the contour onto a 360-bin uniform angular grid (same preprocessing as harmonics)
    contour_angles = np.mod(contour_angles, 2 * np.pi)
    sorted_indices = np.argsort(contour_angles)
    angles_sorted = contour_angles[sorted_indices]
    radii_sorted = contour_radii[sorted_indices]

    angles_unique, inverse_indices = np.unique(angles_sorted, return_inverse=True)
    radii_max = np.zeros_like(angles_unique, dtype=np.float32)
    np.maximum.at(radii_max, inverse_indices, radii_sorted)

    if len(angles_unique) < 10:
        return [0.0, 0.0]

    angles_padded = np.concatenate([angles_unique - 2 * np.pi, angles_unique, angles_unique + 2 * np.pi])
    radii_padded = np.concatenate([radii_max, radii_max, radii_max])
    uniform_angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    uniform_radii = np.interp(uniform_angles, angles_padded, radii_padded)

    max_r = np.max(uniform_radii)
    if max_r == 0:
        return [0.0, 0.0]

    # Normalize to [0, 1] so prominence is scale-invariant
    normalized_radii = uniform_radii / max_r

    # Lightly smooth the uniform 1D signal to prevent jagged contour edges from registering as double-peaks
    smooth_radii = gaussian_filter1d(normalized_radii, sigma=2, mode="wrap")

    # The array is strictly length 360. Pad to allow seamless peak detection over the 0/360 degree seam.
    pad_size = 360 // 4
    padded_radii = np.pad(smooth_radii, pad_width=pad_size, mode='wrap')

    # Distance of 10 bins (degrees) prevents over-counting
    peaks, properties = find_peaks(padded_radii, prominence=prominence_threshold, distance=10)

    # Filter peaks to only count those within the original unpadded domain
    valid_idx = (peaks >= pad_size) & (peaks < pad_size + 360)
    valid_prominences = properties["prominences"][valid_idx]

    peak_count = float(len(valid_prominences))
    avg_prominence = float(np.mean(valid_prominences)) if len(valid_prominences) > 0 else 0.0

    return [peak_count, avg_prominence]


def compute_radial_peak_width_features(contour_angles: np.ndarray, contour_radii: np.ndarray,
                                       prominence_threshold: float = 0.05) -> List[float]:
    """
    Measures the average angular width of radial peaks at half-prominence.
    Smaller values indicate sharp/pointed rays; larger values indicate rounded petals.
    Used as a proxy for motif pointedness.
    """
    if len(contour_radii) < 10:
        return [0.0]

    # Map the contour onto a 360-bin uniform angular grid
    contour_angles = np.mod(contour_angles, 2 * np.pi)
    sorted_indices = np.argsort(contour_angles)
    angles_sorted = contour_angles[sorted_indices]
    radii_sorted = contour_radii[sorted_indices]

    angles_unique, inverse_indices = np.unique(angles_sorted, return_inverse=True)
    radii_max = np.zeros_like(angles_unique, dtype=np.float32)
    np.maximum.at(radii_max, inverse_indices, radii_sorted)

    if len(angles_unique) < 10:
        return [0.0]

    angles_padded = np.concatenate([angles_unique - 2 * np.pi, angles_unique, angles_unique + 2 * np.pi])
    radii_padded = np.concatenate([radii_max, radii_max, radii_max])
    uniform_angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    uniform_radii = np.interp(uniform_angles, angles_padded, radii_padded)

    max_r = np.max(uniform_radii)
    if max_r == 0:
        return [0.0]

    # Normalize to [0, 1] for scale-invariant prominence filtering
    normalized_radii = uniform_radii / max_r

    # Lightly smooth the uniform 1D signal
    smooth_radii = gaussian_filter1d(normalized_radii, sigma=2, mode="wrap")

    # Wrap the signal to seamlessly handle peaks at the start/end seam
    pad_size = 360 // 4
    padded_radii = np.pad(smooth_radii, pad_width=pad_size, mode='wrap')

    # Use distance=10 to accommodate dense sunburst rays
    peaks, properties = find_peaks(padded_radii, prominence=prominence_threshold, distance=10)

    if len(peaks) == 0:
        return [0.0]

    # Calculate peak widths at 50% of the peak's prominence
    widths, width_heights, left_ips, right_ips = peak_widths(padded_radii, peaks, rel_height=0.5)

    # Filter peaks back to the original unpadded domain
    valid_idx = (peaks >= pad_size) & (peaks < pad_size + 360)
    valid_widths = widths[valid_idx]

    if len(valid_widths) == 0:
        return [0.0]

    # Normalize the angular width by the total angular grid length (360 degrees)
    avg_width = float(np.mean(valid_widths)) / 360.0

    return [avg_width]


def compute_radius_features(contour_angles: np.ndarray, contour_radii: np.ndarray,
                            prominence_threshold: float = 0.05) -> List[float]:
    """
    Calculates the ratio between the average inner radius (valleys) and
    average outer radius (peaks) along the 1D boundary signal.

    A low ratio indicates deep cuts (e.g., stars, sunbursts).
    A high ratio (close to 1.0) indicates shallow cuts or nearly circular shapes (e.g., shallow rosettes).
    """
    if len(contour_radii) < 10:
        return [0.0]

    # Map the contour onto a 360-bin uniform angular grid
    contour_angles = np.mod(contour_angles, 2 * np.pi)
    sorted_indices = np.argsort(contour_angles)
    angles_sorted = contour_angles[sorted_indices]
    radii_sorted = contour_radii[sorted_indices]

    angles_unique, inverse_indices = np.unique(angles_sorted, return_inverse=True)
    radii_max = np.zeros_like(angles_unique, dtype=np.float32)
    np.maximum.at(radii_max, inverse_indices, radii_sorted)

    if len(angles_unique) < 10:
        return [0.0]

    angles_padded = np.concatenate([angles_unique - 2 * np.pi, angles_unique, angles_unique + 2 * np.pi])
    radii_padded = np.concatenate([radii_max, radii_max, radii_max])
    uniform_angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    uniform_radii = np.interp(uniform_angles, angles_padded, radii_padded)

    max_r = np.max(uniform_radii)
    if max_r == 0:
        return [0.0]

    # Normalize to [0, 1] for scale-invariant prominence filtering
    normalized_radii = uniform_radii / max_r

    # Lightly smooth the uniform 1D signal
    smooth_radii = gaussian_filter1d(normalized_radii, sigma=2, mode="wrap")

    # Wrap the signal to seamlessly handle peaks at the start/end seam
    pad_size = 360 // 4
    padded_radii = np.pad(smooth_radii, pad_width=pad_size, mode='wrap')

    # Find Peaks (Outer radius)
    peaks, _ = find_peaks(padded_radii, prominence=prominence_threshold, distance=10)

    # Find Valleys (Inner radius) by inverting the signal
    valleys, _ = find_peaks(-padded_radii, prominence=prominence_threshold, distance=10)

    # Filter peaks and valleys back to the original unpadded domain
    valid_peak_idx = (peaks >= pad_size) & (peaks < pad_size + 360)
    valid_peaks = peaks[valid_peak_idx]

    valid_valley_idx = (valleys >= pad_size) & (valleys < pad_size + 360)
    valid_valleys = valleys[valid_valley_idx]

    # If no distinct peaks or valleys are found, the shape is roughly circular
    if len(valid_peaks) == 0 or len(valid_valleys) == 0:
        return [1.0]

    # Extract the actual radius values from the smoothed signal
    peak_radii_vals = smooth_radii[valid_peaks - pad_size]
    valley_radii_vals = smooth_radii[valid_valleys - pad_size]

    median_outer_r = float(np.median(peak_radii_vals))
    median_inner_r = float(np.median(valley_radii_vals))

    if median_outer_r == 0:
        return [0.0]

    # Ratio of inner radius to outer radius
    radius_ratio = median_inner_r / median_outer_r

    return [radius_ratio]


def compute_central_structure_features(edge_mask: np.ndarray, cx: float,
                                       cy: float, contour_radii: np.ndarray) -> List[float]:
    """
    Measures the edge density within the central region of the motif.
    Helps differentiate motifs with a prominent central disk, ring, or boss
    (e.g., rosettes) from those with an empty or unified core (e.g., plain stars).
    """
    if len(contour_radii) < 10:
        return [0.0]

    max_r = float(np.max(contour_radii))
    if max_r == 0:
        return [0.0]

    # Define the central zone as the inner 30% of the motif's maximum radius
    central_r = 0.3 * max_r

    # Create a spatial mask for the central circle using optimized NumPy broadcasting
    h, w = edge_mask.shape
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    center_mask = dist_from_center <= central_r

    central_edges = np.sum(edge_mask[center_mask] > 0)
    central_area = np.count_nonzero(center_mask)

    if central_area == 0:
        return [0.0]

    central_density = float(central_edges / central_area)

    return [central_density]


def compute_shape_features(preprocessed_img: np.ndarray, config: Optional[Dict[str, bool]] = None,
                           debug_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """
    Main orchestration function for geometric feature ablation.
    """
    if config is None:
        config = FEATURE_SET

    edge_mask = extract_edge_mask(preprocessed_img)
    edge_count = int(cv2.countNonZero(edge_mask))

    if edge_count == 0:
        return None

    if debug_path is not None:
        save_edge_debug_image(preprocessed_img, edge_mask, debug_path)

    features = []

    # 1. Base v0.1 features are always computed
    base_feats = compute_base_features(edge_mask)
    features.extend(base_feats)

    # 2. Extract Outer Contour for 1D Boundary Analysis
    # Ensure edge mask is contiguous uint8
    mask_uint8 = np.ascontiguousarray(edge_mask, dtype=np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours and any(config.values()):
        main_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(main_contour)

        if M["m00"] > 0:
            cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
            pts = main_contour.squeeze()
            if pts.ndim == 1:
                pts = pts.reshape(-1, 2)

            dx = pts[:, 0] - cx
            dy = pts[:, 1] - cy

            contour_radii = np.sqrt(dx**2 + dy**2)
            contour_angles = np.arctan2(dy, dx)

            # Module: Harmonics
            if config.get("harmonics", False):
                features.extend(compute_harmonic_features(contour_angles, contour_radii))

            # Module: Peaks
            if config.get("peaks", False):
                features.extend(compute_radial_peak_features(contour_angles, contour_radii))

            # Module: Pointedness
            if config.get("pointedness", False):
                features.extend(compute_radial_peak_width_features(contour_angles, contour_radii))

            # Module: Radius Ratio
            if config.get("radius", False):
                features.extend(compute_radius_features(contour_angles, contour_radii))

            # Module: Central Structure
            if config.get("central_structure", False):
                features.extend(compute_central_structure_features(edge_mask, cx, cy, contour_radii))

    feature_vector = np.array(features, dtype=np.float32)
    feature_vector = np.nan_to_num(feature_vector, nan=0.0)

    # Return dict format matching v0.1 expectation for downstream compatibility
    result_dict = {
        "feature_vector": feature_vector,
        "edge_mask": edge_mask,
        "edge_density": base_feats[7] if len(base_feats) > 7 else 0.0,
    }

    return result_dict


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
    from symbolic_detective.pipeline.preprocess import preprocess_for_opencv

    if len(sys.argv) < 2:
        print("Usage: python -m symbolic_detective.pipeline.shape_features <crop_path>")
        raise SystemExit(1)

    test_path = Path(sys.argv[1])
    processed = preprocess_for_opencv(test_path)

    if processed is None:
        print(f"Failed to preprocess: {test_path}")
        raise SystemExit(1)

    debug_out = Path("data/debug") / f"debug_{test_path.stem}.png"

    # Test standard run
    features = compute_shape_features(processed, debug_path=debug_out)

    if features is None:
        print(f"Failed to extract geometric features for {test_path.name}")
        raise SystemExit(1)

    print(f"Extracted geometric features for: {test_path.name}")
    print(f"Debug image saved to: {debug_out}")
    print(f"Feature vector dimension (Default): {len(features['feature_vector'])}")

    # Test with harmonics enabled
    config_harmonics = {"harmonics": True}
    features_harmonics = compute_shape_features(processed, config=config_harmonics)
    print(f"Feature vector dimension (Harmonics enabled): {len(features_harmonics['feature_vector'])}")
