# v0.2 — Phase B: Feature Combination Experiments

## 1. Objective and Setup

Evaluate whether combining complementary geometric descriptors improves fine-grained motif discrimination beyond the frozen development baseline and individual-feature ablations.

**Protocol:** Same dataset, five artifact-grouped folds, and retrieval configurations as Phase A.

## 2. Feature Combinations

| ID | Features | Purpose                                       |
|---|---|-----------------------------------------------|
| B01 | Harmonics + Peaks | Combine frequency and radial peak information |
| B02 | Peaks + Pointedness | Test peak count and angular peak width        |
| B03 | Pointedness + Radius | Test complementary radial geometry            |
| B04 | Harmonics + Peaks + Pointedness | Evaluate a richer geometric representation |
| B05 | All five features | Full-feature comparison                       |
| B06 | Harmonics + Pointedness | Combine frequency and radial peak-width information |

## 3. Results Summary

### 3.1 Consolidated Performance Comparison

All configurations are evaluated at Hybrid α=0.25.

| Experiment | Top-1 Macro F1 (pooled) | Δ F1 | Weighted K=7 CV Macro F1 | Δ Voting F1 |
|---|---:|---:|---:|---:|
| Baseline | 0.6157 | — | 0.6214 | — |
| exp01_harmonics | 0.6119 | −0.0038 | **0.6463** | **+0.0250** |
| exp03_pointedness | **0.6225** | **+0.0068** | 0.6210 | −0.0003 |
| B01 — Harmonics + Peaks | 0.6085 | −0.0072 | 0.6229 | +0.0016 |
| B02 — Peaks + Pointedness | 0.6151 | −0.0006 | 0.6233 | +0.0020 |
| B03 — Pointedness + Radius | 0.6086 | −0.0071 | 0.6172 | −0.0042 |
| B04 — Harmonics + Peaks + Pointedness | 0.6041 | −0.0116 | 0.6198 | −0.0016 |
| B05 — All Features | 0.6140 | −0.0017 | 0.6084 | −0.0130 |
| B06 — Harmonics + Pointedness | 0.6083 | −0.0074 | 0.6381 | +0.0167 |

Full results for all Phase A and Phase B configurations are available in [comparison_summary.csv](results/comparison_summary.csv) and [per_class_comparison.csv](results/per_class_comparison.csv).

*Top-1 Macro F1 is pooled across development-fold predictions; voting CV Macro F1 is averaged across five folds. Deltas are relative to the corresponding baseline metric.*

**Key findings:**

- None of the Phase B feature combinations outperforms `exp03_pointedness` in Top-1 Macro F1.
- None outperforms `exp01_harmonics` in Weighted K=7 CV Macro F1.
- B06 achieves the strongest voting performance among Phase B combinations (0.6381), but remains below `exp01_harmonics` (0.6463).
- Combining geometric features does not consistently improve discrimination over individual descriptors.

## 4. Analysis and Key Findings

### 4.1 Fold-Level Consistency — `exp03_pointedness`

To assess the consistency of the leading Top-1 configuration, paired fold-level Hit@1 differences were examined for `exp03_pointedness` at Hybrid α=0.25.

| Fold | Baseline Hit@1 | Pointedness Hit@1 | Δ |
|---|---:|---:|---:|
| 1 | 0.6667 | 0.6863 | +0.0196 |
| 2 | 0.6200 | 0.6000 | −0.0200 |
| 3 | 0.6087 | 0.5870 | −0.0217 |
| 4 | 0.5652 | 0.5870 | +0.0217 |
| 5 | 0.5682 | 0.5909 | +0.0227 |

**Observations:**

- Hit@1 improves in **3 of 5 folds**, with regressions in the remaining two.
- Mean Hit@1 increases from 0.6058 to 0.6102 (+0.0045).
- Pooled Macro F1 improves from 0.6157 to 0.6225 (+0.0068).
- Per-class F1 improvements are observed for `sunburst` (+0.0323), `rosette_other` (+0.0207), and `five_pointed_star` (+0.0071), while `six_pointed_star` regresses (−0.0196).

Assessment: `exp03_pointedness` achieves the strongest Top-1 performance among the evaluated feature configurations, although improvements are modest and inconsistent across folds. 
The final classification-strategy comparison favors `exp01_harmonics` with Weighted K=7 voting. Statistical significance has not been established.

### 4.2 Final Classification Strategy Comparison

The strongest Top-1 and neighborhood-voting candidates were compared using the same fold-averaged CV Macro F1 metric.

| Configuration | CV Macro F1 (Mean ± SD) |
|---|---:|
| exp03_pointedness, Top-1, α=0.25 | 0.6193 ± 0.0327 |
| **exp01_harmonics, Weighted K=7, α=0.25** | **0.6463 ± 0.0413** |

The voting configuration achieves a higher mean CV Macro F1 (+0.0270).

Per-class F1 also favors `exp01_harmonics` Weighted K=7 for four of the six target classes, particularly:

- `eight_pointed_star`: +0.0504
- `six_pointed_star`: +0.0413
- `sunburst`: +0.0772

Only `five_pointed_star` shows a small regression (−0.0031).

### 4.3 Comparison Against DINOv2-Only Baseline

At α=0, the best evaluated classification strategy is Majority/Weighted K=3.

| Configuration | CV Macro F1 (Mean ± SD) |
|---|---:|
| DINOv2-only (α=0), K=3 | 0.6082 ± 0.0565 |
| **Harmonics (α=0.25), Weighted K=7** | **0.6463 ± 0.0413** |

The selected hybrid configuration improves mean CV Macro F1 by **+0.0381** over the best evaluated DINOv2-only strategy.

Voting was evaluated at α ∈ {0, 0.25}. Higher hybrid weights (0.5 and 0.75) were not evaluated for voting; therefore, this selection represents the best identified candidate within the evaluated configuration space, not a guaranteed global optimum.

## 5. Selection Decisions

### 5.1 Retrieval Candidate

- **Feature set:** `exp03_pointedness`
- **Hybrid weight:** α=0.25
- **Selection metrics:**
  - Hit@1 = 0.6102
  - Hit@5 = 0.8800
  - MRR = 0.7221

**Rationale:** Best overall retrieval performance among the evaluated feature configurations at α=0.25, with modest improvements over the frozen baseline. Improvements were not consistent across all folds.

### 5.2 Classification Candidate

- **Feature set:** `exp01_harmonics`
- **Hybrid weight:** α=0.25
- **Classification strategy:** Weighted KNN voting
- **K:** 7
- **Primary metric:** CV Macro F1 = 0.6463 ± 0.0413

**Rationale:**

1. Highest observed CV Macro F1 among the compared configurations.
2. Better per-class F1 than the leading Top-1 candidate for four of six motif classes, including `eight_pointed_star`.
3. Outperforms the best evaluated DINOv2-only voting baseline.
4. Additional geometric features did not provide a consistent improvement.

### 5.3 Evaluation Decision

Retrieval and classification are treated as separate evaluation objectives, with distinct development-selected candidates.

Both configurations are selected based on development results, independently of held-out test performance.

**Next step:** Freeze both configurations and evaluate them on the untouched held-out test set. Report retrieval metrics separately from classification metrics.
