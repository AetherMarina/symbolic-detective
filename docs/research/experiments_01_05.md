# Experiments 01–05
## Geometric Features — Theory & Ablation Rationale

These notes document the theoretical background, geometric feature definitions, and experimental rationale behind the v0.2 ablation studies. They are part of the ongoing research and learning process; feature effectiveness is evaluated empirically rather than assumed.

The v0.2 ablation study evaluates the incremental contribution of five explicit geometric feature groups:

1. **`exp01_harmonics`** — Baseline Fourier features for target rotational frequencies ($k = 5, 6, 8$).
2. **`exp02_harmonics_peaks`** — Adds radial peak counting and prominence-based noise filtering.
3. **`exp03_harmonics_peaks_pointedness`** — Adds a measure of peak sharpness based on angular width.
4. **`exp04_harmonics_peaks_pointedness_radius`** — Adds the inner-to-outer radius ratio.
5. **`exp05_full_geometry`** — Adds central-structure features to complete the evaluated feature set.

## Geometric Feature Modules

The outer contour is represented as a one-dimensional radial profile, $r(\theta)$, measuring boundary distance from an estimated center as a function of angle. The modules capture complementary geometric properties:

- **Harmonics:** Relative Fourier power at selected angular frequencies ($k = 5, 6, 8$).
- **Peaks:** Number and average prominence of detected radial maxima, representing candidate points, rays, or petals.
- **Pointedness:** Sharpness of radial maxima, estimated from peak width at half-prominence.
- **Radius:** Ratio of average valley radius to average peak radius.
- **Central structure:** Relative edge density in the motif's central and outer regions.

**Assumption:** A radial contour profile works best for a reasonably isolated, approximately star-shaped silhouette. Segmentation errors, occlusion, and inaccurate center estimation can affect all features.

### 1. Harmonics

An ideal shape has **$N$-fold rotational symmetry** if it coincides with itself under a rotation of

$$
\theta = \frac{360^\circ}{N}.
$$

Fourier analysis decomposes a periodic signal into sinusoidal components. For a radial boundary profile sampled over one full rotation:

$$
r(\theta) = a_0 + \sum_{k=1}^{\infty}\left[a_k\cos(k\theta) + b_k\sin(k\theta)\right].
$$

Here, $k$ is the **angular frequency**: the number of oscillations around a full $360^\circ$ turn. In an idealized shape, a strong component at $k=5$, $6$, or $8$ can indicate a corresponding repeated radial structure. However, an individual Fourier component does **not** by itself prove exact $N$-fold symmetry or identify the number of points; other frequencies, distortions, and noise also matter.

The term *harmonic* originated in acoustics, where a vibrating string produces frequency components at integer multiples of a fundamental frequency. Fourier analysis extends this idea to other periodic signals, including geometric boundary profiles.

### 2. Peaks

After applying a prominence threshold to suppress minor fluctuations in $r(\theta)$, detect local maxima:

- **Peak count:** The number of prominent candidate tips or petals. This can help distinguish five-, six-, and eight-pointed motifs, although segmentation artifacts or decorative details may add or remove peaks.
- **Average prominence:** The average radial height of peaks relative to their surrounding baseline. Higher prominence generally indicates more pronounced tips and deeper intervening notches; lower prominence suggests shallower radial modulation.

Prominence measures how much a peak stands out, **not** how narrow or angular its tip is. That is addressed by the pointedness feature.

### 3. Pointedness

For each detected peak, estimate its angular width at **half-prominence**, for example with `scipy.signal.peak_widths(..., rel_height=0.5)`.

- **Narrower peaks** generally correspond to sharper or more localized radial tips.
- **Wider peaks** generally correspond to broader, more rounded tips or petals.

Peak widths should be converted from sample counts to angular units (degrees or radians) if the sampling resolution can vary. This is a geometric proxy for pointedness, not a direct measurement of contour-corner angle.

### 4. Radius

- **Outer radius:** Average radial distance from the center to detected peak tips.
- **Inner radius:** Average radial distance from the center to valleys between those peaks.

Calculate:

$$
R_{\mathrm{radius}} = \frac{\overline{r}_{\mathrm{valley}}}{\overline{r}_{\mathrm{peak}}}.
$$

For valid positive radii, this ratio is typically between 0 and 1:

- **Lower ratio:** Deep notches relative to the tip radius, often seen in strongly indented stars or sunbursts.
- **Higher ratio (closer to 1):** Shallow notches, often seen in less deeply indented or rounded motifs.

The ratio is a **continuous descriptor**, not a universal star-versus-rosette classifier. Any numerical cutoffs should be calibrated using the dataset rather than treated as fixed geometric rules.

### 5. Central Structure

Use an edge map of the motif's interior (not just its external contour) to compare central and outer regions. Normalize edge counts by the area of each region:

$$
D_{\mathrm{center}} = \frac{E_{\mathrm{center}}}{A_{\mathrm{center}}},
\qquad
D_{\mathrm{outer}} = \frac{E_{\mathrm{outer}}}{A_{\mathrm{outer}}},
$$

$$
R_{\mathrm{density}} = \frac{D_{\mathrm{center}}}{D_{\mathrm{outer}} + \varepsilon},
$$

where $E$ is the number of detected edge pixels, $A$ is the region area, and $\varepsilon$ is a small constant preventing division by zero.

- **High ratio ($>1$):** The central region contains more detected edges per unit area than the outer region, potentially reflecting a central ring, emblem, or other internal detail.
- **Low ratio (near 0):** Few detected central edges relative to the outer region, consistent with a visually simple or smooth center.
- **Ratio near 1:** Similar edge densities in both regions.

**Limitation:** Edge density alone cannot reliably distinguish a filled central disk from an empty or smooth center. It is also sensitive to image texture, contrast, edge-detection settings, and the definition of the central region. Detecting a filled disk or ring as a semantic structure would require additional features.
