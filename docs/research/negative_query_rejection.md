# Negative Query Rejection

> **Scope:** Symbolic Detective v0.2 — detecting queries that do not belong to any known motif class. This document explains the rejection mechanism built on top of the existing hybrid retrieval and weighted KNN classification pipeline.

## 1. Why Rejection Is Needed

The standard classifier assigns every query to one of the classes represented in its retrieval gallery, even when the query depicts an unrelated motif.

**Negative Query Rejection** introduces an alternative outcome: `unknown`.

The objective is to avoid unsupported classifications while preserving correct predictions for known motifs.

## 2. How It Works

```text
Query crop
    |
    v
Hybrid retrieval (DINOv2 + OpenCV, α=0.25)
    |
    v
Top-7 artifact-deduplicated neighbors
    |
    v
Weighted class voting
    |
    +-- Predicted motif class
    +-- Confidence scores
            |
            v
    Compare with calibrated threshold
            |
       +----+----+
       |         |
  Confidence  Confidence
  >= threshold < threshold
       |         |
       v         v
  Known class   unknown
```
The diagram illustrates independent confidence-based rejection. Combined rejection strategies using two confidence signals are described in Section 4.

The gallery contains only known motif classes. Negative controls are treated as unknown queries rather than gallery candidates.

## 3. Confidence Scores

Two complementary signals are evaluated.

### Maximum Similarity

Measures how closely the query resembles its most similar gallery crop.

$$
C_{\mathrm{sim}} = \max_i S_{\mathrm{hybrid}}(q, g_i)
$$

The hybrid score combines DINOv2 and geometric similarities:

$$
S_{\mathrm{hybrid}} = (1-\alpha)S_{\mathrm{embed}} + \alpha S_{\mathrm{shape}}
$$

A low maximum similarity suggests that no gallery example closely matches the query.

### Weighted Vote Confidence

Measures how strongly the seven nearest artifacts support the predicted class.

$$
C_{\mathrm{vote}} =
\frac{\sum_{i=1}^{7} \mathbb{1}[y_i = \hat{y}]\,w_i}
{\sum_{i=1}^{7} w_i},
\qquad
w_i = S_{\mathrm{hybrid}}(q, g_i)
$$

The sum is computed over the Top-7 artifact-deduplicated neighbors, using the same similarity weights as the frozen Weighted K=7 classifier.

For example, a confidence of 0.80 means the winning class receives 80% of the total voting weight.

**Key distinction:** Maximum similarity measures closeness to known examples; weighted vote confidence measures agreement among neighbors. Strong agreement does not necessarily mean a query belongs to a known class.

Neither score is a calibrated probability.

## 4. Threshold Calibration and Rejection Strategies

A rejection threshold defines the minimum confidence required to accept a known-class prediction.

### Experiment 1 - Independent Threshold Calibration

The two confidence signals are initially calibrated independently:

- **Maximum Similarity:** Reject if \(C_{\mathrm{sim}} < t_{\mathrm{sim}}\).
- **Weighted Vote Confidence:** Reject if \(C_{\mathrm{vote}} < t_{\mathrm{vote}}\).

For each method, the optimal threshold is selected on development data by maximizing **Balanced Accuracy**.

$$
t^* = \arg\max_t \mathrm{BalancedAccuracy}(t)
$$

### Experiment 1 - Combined Confidence Rejection

The two confidence signals can also be combined to make a single accept/reject decision.

Two strategies are evaluated:

| Strategy | Rejection Rule | Interpretation |
|---|---|---|
| **REJECT_OR** | Reject if either confidence score falls below its threshold | Stricter rejection |
| **REJECT_AND** | Reject only if both confidence scores fall below their thresholds | More permissive rejection |

**REJECT_OR**

$$
\mathrm{Reject} =
(C_{\mathrm{sim}} < t_{\mathrm{sim}})
\lor
(C_{\mathrm{vote}} < t_{\mathrm{vote}})
$$

**REJECT_AND**

$$
\mathrm{Reject} =
(C_{\mathrm{sim}} < t_{\mathrm{sim}})
\land
(C_{\mathrm{vote}} < t_{\mathrm{vote}})
$$

### Joint Threshold Optimization

Unlike independent calibration, combined rejection requires finding an optimal **pair of thresholds**.

For each strategy, different pairs \((t_{\mathrm{sim}},t_{\mathrm{vote}})\) are evaluated on development data.

$$
(t_{\mathrm{sim}}^*,t_{\mathrm{vote}}^*)
=
\arg\max_{t_{\mathrm{sim}},t_{\mathrm{vote}}}
\mathrm{BalancedAccuracy}
$$

The optimization is performed separately for REJECT_OR and REJECT_AND, so each strategy may select a different threshold pair.

### Evaluation Protocol

- **Development:** Generate confidence scores using artifact-disjoint folds, calibrate independent and combined thresholds, and compare strategies.
- **Held-out evaluation:** Apply the selected strategy and frozen thresholds without further tuning.

Combined strategies are exploratory extensions of the original v0.2 rejection baseline. Since the original held-out results have already been examined, improvements from a newly selected strategy would require fresh independent evaluation to establish generalization.

## 5. Evaluation Metrics

| Metric | Meaning |
|---|---|
| **True Rejection Rate (TRR)** | Fraction of unknown queries correctly rejected |
| **False Rejection Rate (FRR)** | Fraction of known queries incorrectly rejected |
| **Known Acceptance Rate** | Fraction of known queries correctly accepted (`1 − FRR`) |
| **Balanced Accuracy** | Average of Known Acceptance Rate and TRR |
| **Unknown F1** | Precision and recall balance for detecting unknown queries |
| **Open-set Macro F1** | Classification F1 across known classes and `unknown` |

### Balanced Accuracy

Measures how well the rejection mechanism distinguishes known from unknown motifs, giving equal importance to both groups regardless of their sample counts.

$$
\mathrm{Known\ Acceptance\ Rate} = 1 - \mathrm{FRR}
$$

$$
\mathrm{Balanced\ Accuracy}
= \frac{\mathrm{Known\ Acceptance\ Rate} + \mathrm{TRR}}{2}
= \frac{(1-\mathrm{FRR})+\mathrm{TRR}}{2}
$$

For example, with TRR = 0.88 and FRR = 0.32:

$$
\mathrm{Balanced\ Accuracy}
= \frac{0.68+0.88}{2}
= 0.78
$$

**Important:** Balanced Accuracy evaluates only the *accept/reject decision*, not whether an accepted known query is assigned the correct motif class.

A higher rejection threshold typically increases TRR but also risks increasing FRR.

## 6. Key Concept

**Retrieval similarity is not classification certainty.**

A query may resemble a known motif without actually belonging to that class. Conversely, its nearest neighbors may strongly agree on a class simply because the gallery contains no suitable alternative.

Negative Query Rejection adds an explicit *abstention mechanism*: when evidence is insufficient, Symbolic Detective can return `unknown` rather than force a classification.

This is an initial form of **open-set recognition**, not full discovery or identification of previously unseen motif categories.