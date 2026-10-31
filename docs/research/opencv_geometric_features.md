# OpenCV Geometric Features — Design & Rationale

> **Scope:** Original 26-dimensional OpenCV feature representation used in the Symbolic Detective motif-retrieval baseline. This document explains the design, mathematical definitions, implementation, and limitations. It is distinct from the additional contour-based features investigated in the v0.2 ablation experiments.

## 1. Motivation

Cultural-heritage motifs can appear on different materials, in different artistic styles, and under very different imaging conditions. DINOv2 provides a learned visual representation, but it does not explicitly count star points or describe a motif's geometry in human-readable terms.

The OpenCV stream complements the learned embedding with **hand-engineered descriptors** of:

- Global edge geometry (Hu moments).
- Relative amount of detected edge structure (edge density).
- Distribution of edges from the crop center outward (radial histogram).
- Periodic structure around the crop center (angular Fourier spectrum).

The aim is to test whether these complementary properties improve **visual motif retrieval**, not to infer historical or symbolic identity directly. Their usefulness must be established empirically.

**Design choice:** The baseline operates on **manually cropped regions of interest**, without automatic foreground segmentation. Cropping helps focus the input on a motif, but does **not** guarantee that the edge mask contains only the motif.

## 2. Feature Extraction Pipeline

```text
Manually cropped motif image
           |
           v
Preprocessed grayscale image (e.g., CLAHE-enhanced)
           |
           v
     Gaussian blur
           |
           v
   Canny edge detection
           |
           v
    Binary edge mask
           |
     +-----+------+----------+-----------+
     |            |          |           |
     v            v          v           v
 7 Log-Hu      1 Edge     6 Radial    12 Angular
 moments       density    histogram   harmonics
     |            |          |           |
     +------------+----------+-----------+
                        |
                        v
               Raw 26D feature vector
                        |
                        v
           Corpus-fitted standardization
                        |
                        v
         Euclidean distance and similarity
```

### 2.1 Edge mask

`extract_edge_mask(preprocessed_img, ...)` applies Gaussian smoothing and Canny edge detection, then clears a narrow margin around the crop boundary to reduce crop/padding artifacts.

The source notes describe median-based Canny thresholds:

```python
blurred = cv2.GaussianBlur(preprocessed_img, (blur_kernel, blur_kernel), 0)
median_intensity = float(np.median(blurred))
lower = int(max(0, (1.0 - canny_sigma) * median_intensity))
upper = int(min(255, (1.0 + canny_sigma) * median_intensity))
edge_mask = cv2.Canny(blurred, lower, upper)
```

The documented settings include `canny_sigma = 0.33` and a border suppression margin of approximately 3% of the shorter image dimension. Confirm current defaults against `shape_features.py` when modifying the implementation.

`edge_mask` is binary: `255` marks a detected edge; `0` marks a non-edge pixel. It can include motif boundaries **and** lines from internal decoration, cracks, substrate texture, background objects, or noise. The mask supports all four feature groups and debug visualization, but **is not itself a dimension of the 26D vector**.

## 3. The 26-Dimensional Feature Vector

The number **26 is a design choice**, not a requirement of OpenCV. It combines four groups:

| Feature group | Dimensions | Vector indices | Principal description |
| --- | ---: | --- | --- |
| Log-transformed Hu moments | 7 | `[0:7]` | Global distribution of detected edges |
| Edge density | 1 | `[7]` | Fraction of pixels marked as edges |
| Radial histogram | 6 | `[8:14]` | Edge concentration at different radii |
| Angular Fourier spectrum | 12 | `[14:26]` | Periodicity of angular edge distribution |
| **Total** | **26** | `[0:26]` | Complementary geometric descriptors |

$$
\mathbf{f} = [h_1,\ldots,h_7,\;\rho_{\mathrm{edge}},\;r_1,\ldots,r_6,\;a_1,\ldots,a_{12}] \in \mathbb{R}^{26}
$$

```python
feature_vector = np.concatenate([
    log_hu,                # 7 values
    [edge_density],        # 1 value
    radial_histogram,      # 6 values
    angular_spectrum,      # 12 values
]).astype(np.float32)
```

The extractor returns **raw, unstandardized measurements**. Standardization happens downstream in the retrieval pipeline.

### 3.1 Log-transformed Hu moments (7)

**What they measure.** Seven classical image-moment invariants summarize the two-dimensional distribution of edge pixels. In this implementation, they are calculated from the **binary edge mask**, not a filled silhouette.

```python
moments = cv2.moments(edge_mask, binaryImage=True)
hu = cv2.HuMoments(moments).flatten()
```

The Hu moment construction is designed to be invariant to translation, uniform scaling, and in-plane rotation in its mathematical formulation. In practice, discretization, interpolation, cropping, and changes in detected edges can affect the values. The moments are centered on the **edge-pixel centroid**, unlike the radial/angular features, which use the crop center.

Because Hu moments can span many orders of magnitude, the implementation uses a signed logarithmic transformation:

$$
\operatorname{logHu}(H_i) = -\operatorname{sgn}(H_i)\log_{10}|H_i|,\qquad H_i\ne 0
$$

A zero moment is handled separately. A negative transformed value is valid; it is **not** an error or a low-confidence score.

**Why it might help:** Hu moments can capture broad differences in overall spatial arrangement, elongation, and asymmetry that are not represented by a simple count of radial repetitions.

**Limitations:** They summarize *all detected edges*, including unwanted texture and interior details; they do not directly count arms, petals, or points. Classical Hu invariants are not all invariant to mirror reflection (the seventh changes sign).

### 3.2 Edge density (1)

**What it measures.** The proportion of pixels detected as edges:

$$
\rho_{\mathrm{edge}} = \frac{N_{\mathrm{edge}}}{H\,W}
$$

```python
edge_density = edge_pixel_count / float(edge_mask.size)
```

For example, a value of `0.0083` means approximately **0.83%** of the processed crop's pixels are marked as edges.

**Why it might help:** Edge density can offer a coarse indicator of linework or ornamental complexity, helping distinguish relatively sparse outlines from dense interlacing or detailed decoration.

**Limitations:** It is sensitive to image resolution, crop padding, threshold settings, cracks, background texture, and contrast. A large filled star can have low edge density because Canny detects *boundaries*, not the filled interior. It is neither a segmentation-quality score nor a recognition-confidence score.

### 3.3 Radial histogram (6)

**What it measures.** The distribution of detected edges by their distance from the **geometric center of the crop**:

$$
c_x=\frac{W-1}{2},\quad c_y=\frac{H-1}{2},\quad
r_i=\sqrt{(x_i-c_x)^2+(y_i-c_y)^2}
$$

Distances are normalized by the center-to-corner distance:

$$
\tilde r_i = \frac{r_i}{R_{\max}},\qquad
R_{\max}=\sqrt{c_x^2+c_y^2}
$$

The normalized distances are counted in **six equal-width bins** over `[0, 1]`, and the histogram is normalized by the number of edge pixels:

$$
\sum_{j=1}^{6} r_j = 1 \quad \text{(when edges are present)}
$$

Example: `[0.05, 0.10, 0.15, 0.30, 0.25, 0.15]`. Larger early-bin values indicate more edges near the crop center; larger late-bin values indicate more edges farther outward.

**Why it might help:** It describes coarse radial organization (e.g., central decoration versus peripheral borders), which may differ even when two motifs have similar angular repetition.

**Limitations:** The histogram measures the **location of edges, not filled mass**. It discards directional information, assumes a reasonably centered crop, and normalizes to crop size rather than the true motif radius. Its bin values cannot alone establish that a motif contains a solid disk, empty center, or ring.

### 3.4 Angular Fourier spectrum (12)

**What it measures.** Repeated structure in the distribution of detected edge pixels **around the crop center**.

1. Compute the polar angle of every edge pixel:

   $$
   \theta_i=\operatorname{atan2}(y_i-c_y,x_i-c_x)\bmod 2\pi
   $$

2. Count the edges in **36 angular bins** (10 degrees per bin), then normalize the resulting histogram.
3. Compute the magnitude of its real discrete Fourier transform:

   ```python
   spectrum = np.abs(np.fft.rfft(angular_hist))
   angular_spectrum = spectrum[1:13]  # k = 1, ..., 12
   ```

4. Omit the zero-frequency/DC term and retain the first **12 nonzero harmonics**.

A relatively strong fifth, sixth, or eighth harmonic can indicate prominent fivefold, sixfold, or eightfold *periodic components* in the angular edge distribution. **It does not guarantee that the visible motif has that number of points or petals.**

For an ideal continuous angular signal, a rotation changes Fourier phases but leaves magnitudes unchanged:

$$
h_{\phi}(\theta)=h(\theta-\phi),\qquad
|\mathcal{F}(h_{\phi})(k)|=|\mathcal{F}(h)(k)|
$$

With a finite 36-bin histogram, pixel sampling, and possibly imperfect centering, this property is **approximately**, not strictly, rotation invariant for arbitrary image rotations.

**Why it might help:** It explicitly represents cyclic structure that can differentiate otherwise visually similar motifs.

**Limitations:** Angular bins aggregate edges across all radii, so inner decoration, texture, and an off-center crop can create or obscure peaks. Harmonic magnitude is neither a point-count guarantee nor a class probability.

### 3.5 Complementarity of the descriptors

| Descriptor | Center used | Preserves radial distance? | Preserves angular structure? |
| --- | --- | --- | --- |
| Hu moments | Edge-pixel centroid | Implicitly, through 2D spatial moments | Implicitly, through 2D spatial moments |
| Edge density | None | No | No |
| Radial histogram | Crop center | Yes, coarsely | No |
| Angular spectrum | Crop center | No | Yes, as periodic magnitudes |

Two shapes can have a similar angular spectrum while placing edges at different radii; their radial histograms and Hu moments may help distinguish them. Conversely, similar radial distributions can conceal different angular patterns.

## 4. Feature Standardization and Similarity

`compute_shape_features(...)` emits a raw 26D vector. Because the feature groups have different numerical scales, the retrieval pipeline standardizes feature dimensions using statistics **fitted on the indexed corpus** before comparing query and indexed vectors.

For standardized feature vectors $\mathbf{a}$ and $\mathbf{b}$, geometric distance is Euclidean:

$$
d(\mathbf{a},\mathbf{b})=\|\mathbf{a}-\mathbf{b}\|_2
$$

The documented similarity function maps distance to a score in `(0, 1]`:

$$
S_{\mathrm{shape}}(\mathbf{a},\mathbf{b})=
\exp\!\left(-\gamma\,\|\mathbf{a}-\mathbf{b}\|_2\right),
\qquad \gamma=0.25
$$

This is an exponential Euclidean-distance similarity, **not** the squared-distance Gaussian RBF $\exp(-\gamma d^2)$. Identical vectors score `1`; increasingly different vectors approach `0`. The implementation clips the computed score into `[0, 1]` as a numerical safeguard.

The `gamma` parameter controls how quickly similarity decreases with distance; the result is **not a calibrated probability** that two motifs belong to the same class.

**Important for evaluation:** Standardization statistics should be fitted only on the reference/index split for each experiment, then applied unchanged to its queries. Otherwise, held-out evaluation could leak information from the query set.

## 5. Diagnostic Outputs

The debug helper `save_edge_debug_image(...)` produces three views:

1. The preprocessed grayscale crop.
2. The binary Canny edge mask.
3. An overlay highlighting detected edges in red.

These views are valuable when a surprising retrieval match may originate from crop boundaries, texture, low contrast, broken contours, or misplaced center assumptions.

The source notes also indicate that `compute_shape_features(...)` returns `None` when no edges are detected; callers should handle this case explicitly.

## 6. Known Limitations and Motivation for v0.2

The original 26D baseline is interpretable, but it does not directly isolate the motif's **outer silhouette** or count its tips:

- **No explicit segmentation:** Background and interior details can influence every feature group.
- **Crop-center dependence:** Radial and angular descriptors assume the relevant geometry is reasonably centered.
- **No direct point counting:** Fourier periodicity is evidence of repeated structure, not an explicit estimate of arm/petal count.
- **Limited local shape description:** Global moments and histograms do not directly encode peak widths, valley depths, or how sharp individual tips are.
- **Approximate invariance:** Discrete pixel and angular-bin representations are affected by interpolation, scale changes, noise, and cropping.

These limitations motivate a **separate v0.2 geometric ablation track** that tests contour-derived harmonics, peak counts/prominence, pointedness, inner-to-outer radius relationships, and central structure. Those are **additional experimental features**, not existing dimensions of the 26D baseline.

The research question is whether any such features measurably improve discrimination of visually overlapping classes—especially five-, six-, and eight-pointed stars and rosettes—under artifact-grouped evaluation. Improvement is a hypothesis to test, not an assumed outcome.

## References and Implementation Notes

- `shape_features.py` — source of truth for feature extraction, parameter defaults, output format, similarity, and debug output.
- [Hybrid retrieval pipeline](hybrid_retrieval_pipeline.md)
- [OpenCV image moments and Hu invariants](https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html).
- [NumPy real FFT (`numpy.fft.rfft`)](https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft.html).

*This document consolidates the project's original OpenCV feature notes v0.1.*
