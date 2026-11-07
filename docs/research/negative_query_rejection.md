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

## 4. Threshold Calibration

A rejection threshold determines the minimum confidence required to accept a known-class prediction.

- **Development:** Evaluate known and negative-control queries using artifact-disjoint galleries, then select thresholds without accessing held-out test data.
- **Held-out test:** Apply the frozen thresholds to known and negative-control queries without further tuning.

The two confidence methods are evaluated independently.

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