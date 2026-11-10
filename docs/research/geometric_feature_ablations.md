# Geometric Feature Ablations — v0.2
## Experiments 01–07: Theory & Ablation Rationale

These notes document the theoretical background, feature definitions, and experimental rationale for the v0.2 geometric ablation studies. Feature effectiveness must be measured empirically rather than assumed.

## Experimental Design

The original cumulative design added features sequentially (harmonics → peaks → pointedness → radius → central structure). That design can test whether an increasingly complex representation helps, but **cannot isolate the contribution of each individual feature group**. For example, an improvement with harmonics + peaks does not establish whether peaks would help on their own.

The revised plan evaluates **one additional feature group at a time against the same unchanged baseline**, then tests selected combinations. Here, **base** means the fixed reference retrieval configuration chosen for these experiments; define and hold its preprocessing, dataset split, candidate corpus, similarity settings, and evaluation protocol constant. Do not treat the baseline as a new feature group.

```text
v0.2_dev/
├── exp01_harmonics/          # base + harmonics
├── exp02_peaks/              # base + peak count / prominence
├── exp03_pointedness/        # base + pointedness
├── exp04_radius/             # base + inner / outer radius
├── exp05_central_structure/  # base + central edge structure
├── exp06_combined/           # base + selected useful features
└── exp07_full_geometry/      # optional: base + all five groups
```

| Experiment | Added geometric module | Primary question |
| --- | --- | --- |
| `exp01_harmonics` | Harmonics only | Do target angular frequencies help? |
| `exp02_peaks` | Peaks only | Does explicit peak count/prominence help? |
| `exp03_pointedness` | Pointedness only | Does peak width capture useful shape differences? |
| `exp04_radius` | Radius only | Does valley-to-peak depth help? |
| `exp05_central_structure` | Central structure only | Do internal edges add useful information? |
| `exp06_combined` | Selected useful modules | Do individually promising features work together? |
| `exp07_full_geometry` *(optional)* | All five modules | Does the complete representation outperform the selected subset, or introduce noise? |

### Comparison principles

- Compare each isolated experiment with the **same base** (`base` vs. `base + feature`), not only against the previous experiment.
- Keep artifact-level dev/test separation, grouped CV folds, preprocessing, and query/gallery composition fixed to make differences interpretable.
- Use the same retrieval metrics (Hit@1, Hit@5, MRR) and inspect per-class discrimination/error patterns where relevant.
- Select features for `exp06_combined` using development evidence, including consistency across folds and target classes; do **not** automatically include all five.
- Treat `exp07_full_geometry` as an optional diagnostic comparison. The held-out test set should remain untouched while choosing modules and configurations.
- Isolated ablations estimate each feature's incremental contribution relative to the base. They do **not** rule out interactions: a feature that is weak alone may still complement another one.

The purpose is to answer **which geometric features actually help**, and then whether combining them yields further improvement.

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
