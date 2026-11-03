# Evaluation Protocol — v0.2

> **Scope:** Development-set evaluation for Symbolic Detective v0.2, as implemented in `evaluate.py`. This document describes the evaluation design, metrics, and outputs—not experimental results. The held-out test partition is reserved and is **not evaluated by this script**.

## 1. Evaluation Objectives

The evaluation examines two related but distinct questions:

1. **Retrieval:** Does the system rank crops from the query's motif class near the top of its search results?
2. **Multi-class discrimination:** Can a motif class be inferred from the ranked examples, using either the top result or class voting?

The benchmark compares the learned DINOv2 representation, explicit OpenCV geometry, and their weighted hybrid. The v0.2 experiments investigate whether geometric changes improve fine-grained discrimination among visually related motifs, especially stars and rosettes.

For preprocessing, feature definitions, and score fusion, see [Hybrid Retrieval Pipeline](hybrid_retrieval_pipeline.md).

## 2. Dataset Partitioning

### 2.1 Fixed artifact-level dev/test split

The script reads `data/splits/v0.2_split.json`, whose `partitions.dev` and `partitions.test` entries identify **artifacts**, not individual crops. It checks that these partitions have no overlapping artifact IDs.

Only crops belonging to **development artifacts** are loaded for the present evaluation. Test artifacts are explicitly excluded before feature extraction and cross-validation; they must remain untouched during development and model/configuration selection.

The `unclassified` directory is excluded. An artifact ID is derived from each image path using `artifact_key_from_path(...)`.

### 2.2 Stratified, artifact-grouped cross-validation

Within the development partition, the default is **5-fold `StratifiedGroupKFold`** with `shuffle=True` and `random_state=42`.

- **Grouping:** All crops with the same artifact ID belong to the same fold. A physical artifact cannot appear in both the query set and reference corpus of one fold.
- **Stratification:** Folds are constructed to approximately balance motif classes at the artifact level.
- **Iteration:** In each fold, the current fold's motif artifacts supply queries; the other development artifacts supply the reference corpus (index/gallery).
- **Validation:** The script raises an error if corpus and query artifact IDs overlap.

The stratifier assigns one class to each queryable artifact when constructing folds. If an artifact has crops with multiple class labels, that representative label does not fully describe its multi-class membership. **Metrics are evaluated per query crop**, not averaged first by artifact.

### 2.3 Negative controls and valid queries

`negative_control` crops can appear in the **reference corpus as distractors**, but are **never used as queries**. Their role is to test whether irrelevant material outranks examples from a relevant motif class.

Before evaluating a query, the script checks that its true motif class is represented in that fold's corpus. Queries with no possible same-class match are excluded and counted as dropped. This makes retrieval metrics conditional on class availability in the index.

### 2.4 Leakage-safe standardization

For each fold, a new `StandardScaler` is fitted **only on the corpus's raw OpenCV feature vectors**. The same fitted scaler transforms corpus and query vectors; query data does not contribute to its statistics.

DINOv2 embeddings are produced by the separate extractor and already L2-normalized. The code extracts features for the development pool before splitting into folds, but fits the OpenCV scaler separately within each fold.

## 3. Retrieval Evaluation

### 3.1 Compared representations

Every query is scored against the entire fold-specific reference corpus. For a query `q` and indexed crop `x`:

$$
S_{\mathrm{hybrid}}(q,x) = (1-\alpha)S_{\mathrm{embed}}(q,x) + \alpha S_{\mathrm{shape}}(q,x)
$$

The current script evaluates `α ∈ {0.00, 0.25, 0.50, 0.75, 1.00}`:

| α | Configuration |
| ---: | --- |
| `0.00` | DINOv2 only |
| `0.25`, `0.50`, `0.75` | Weighted hybrid retrieval |
| `1.00` | OpenCV geometry only |

Candidates are ranked in **descending hybrid similarity**. A retrieved crop is considered relevant when its label matches the query's motif class, not when it originates from the same artifact.

### 3.2 Retrieval metrics

| Metric | Definition | Question answered |
| --- | --- | --- |
| **Hit@1** | `1` if the first result has the query's class; otherwise `0` | Is the best match relevant? |
| **Hit@5** | `1` if at least one of the first five results has the query's class; otherwise `0` | Is a relevant match in the first five? |
| **Reciprocal rank (RR)** | `1 / r`, where `r` is the rank of the first relevant result; `0` if none exists | How far down is the first relevant match? |
| **MRR** | Mean reciprocal rank over queries | How early do relevant results appear on average? |

Each metric is first averaged over the evaluated query crops **within each fold**. The script then reports **mean ± standard deviation across folds** (`np.mean`, `np.std`). The overall results and a separate report for the configured target class (default: `eight_pointed_star`) use this procedure. Fold means have equal weight in the reported cross-validation average, even if fold query counts differ.

Retrieval scoring uses ranked **crop candidates**; artifact deduplication is requested for the separate voting experiment, not for these ranking metrics.

## 4. Multi-Class Discrimination

### 4.1 Top-1 classification

The top-ranked crop's **class label** serves as the query's predicted class. This is *retrieval-derived classification*, not a trained DINOv2 classification head.

The script produces Top-1 discrimination reports for:

- **DINOv2** (`α = 0.00`)
- **Hybrid** (`α = 0.25`)

### 4.2 Precision, recall, F1, and confusion matrices

For these two configurations, pooled predictions from the evaluated development folds are used to calculate **per-class precision, recall, F1-score, and support**, along with summary metrics and a confusion matrix.

- **Precision:** Of all queries predicted as a given class, how many actually belong to it?
- **Recall:** Of all queries belonging to that class, how many were correctly identified?
- **F1:** Harmonic mean of precision and recall, balancing both.
- **Confusion matrix:** Which classes are most frequently mistaken for one another?

These metrics measure **class discrimination**, whereas Hit@5 and MRR measure **ranking quality**. They should not be treated as interchangeable.

## 5. Class Voting Experiments

Instead of assigning the class of only the first match, the voting experiment asks whether multiple high-ranked reference examples improve class prediction.

The script calls `predict_class(...)` with `dedup=True`, requesting **artifact-deduplicated voting** so repeated crops from the same indexed artifact do not simply contribute multiple votes. The details of that deduplication and the weighted-vote calculation are defined by `class_voting.py`.

Configurations tested:

| Strategy | K |
| --- | --- |
| `top1` | `1` |
| `majority` | `3`, `5`, `7` |
| `weighted` | `3`, `5`, `7` |

Voting is evaluated only for **DINOv2 (`α = 0.00`)** and **Hybrid (`α = 0.25`)**. For every combination, the script reports:

- **Accuracy:** Proportion of correctly predicted query classes.
- **Macro-F1:** Unweighted average of per-class F1 scores; gives each class equal importance.
- **Per-class precision, recall, F1, and support:** Exported from predictions pooled across development folds.

Reported voting accuracy and macro-F1 are **mean ± standard deviation of fold-level scores**; the export also includes **pooled** accuracy and macro-F1. These are different aggregations and may not be numerically identical.

### 5.1 Voting Implementation Details

For each query, ranked results are deduplicated by source artifact ID **before selecting the Top-K neighbors**. Since candidates are sorted by descending similarity, only the highest-scoring crop from each artifact is retained. The Top-K selection is then performed over these unique artifact representatives, ensuring that each source artifact contributes at most one vote.

Three voting strategies are evaluated:

- **Top-1:** Assigns the class label of the highest-ranked retrieved neighbor ($K=1$).
- **Majority Voting:** Assigns the most frequent class among the Top-K retrieved unique artifacts ($K \in \{3,5,7\}$). Ties are resolved in favor of the class whose highest-ranked representative has the greatest similarity score.
- **Similarity-Weighted Voting:** Assigns the class with the highest cumulative similarity score among the Top-K retrieved unique artifacts ($K \in \{3,5,7\}$).

For similarity-weighted voting, the score assigned to class $c$ is:

$$
\operatorname{VoteScore}(c) =
\sum_{\substack{i \in N_K(q)\\ y_i=c}} S_\alpha(q,i)
$$

where:

- $N_K(q)$ is the set of Top-K artifact-deduplicated neighbors for query $q$.
- $y_i$ is the class label of neighbor $i$.
- $S_\alpha(q,i)$ is the retrieval similarity score between query $q$ and neighbor $i$, using the selected representation ($\alpha=0.00$ for DINOv2 or $\alpha=0.25$ for Hybrid).

The final predicted class is the class with the highest voting score. **Tie-Breaking:** In similarity-weighted voting, exact ties in cumulative class scores are resolved in favor of the class whose representative appears first in the ranked Top-K neighbors.

**Evaluation objective:** Determine whether aggregating evidence from multiple retrieved examples improves motif-class discrimination compared with the Top-1 nearest-neighbor baseline.

## 6. Error Analysis and Outputs

The current script supports `--mode baseline`, `--mode voting`, or `--mode all` (CLI default: `all`). The results location is currently set in the script to:

`results/v0.2_dev/exp01_harmonics/`

| Output | Content | Mode |
| --- | --- | --- |
| Console CV tables | Overall and target-class Hit@1, Hit@5, MRR across α values | All modes |
| `retrieval_metrics.json` | Fold-level retrieval metrics and pooled Top-1 per-class discrimination summaries | `baseline`, `all` |
| `error_analysis.json` | Per-query DINOv2 vs. Hybrid Top-1 outcomes (`geometry_helped`, `geometry_hurt`, `both_correct`, `both_wrong`); `both_correct` cases are omitted | `baseline`, `all` |
| `figures/confusion_dinov2.png`, `figures/confusion_hybrid.png` | Top-1 class confusion matrices | `baseline`, `all` |
| `class_voting.json` | Voting configurations, fold-aggregated accuracy/macro-F1, pooled summaries, per-class metrics | `voting`, `all` |
| `voting_error_analysis.json` | Queries for which at least one voting configuration predicts the wrong class | `voting`, `all` |

The script stores measured results in `results/`; this protocol intentionally does **not** duplicate experiment-specific scores or conclusions.

**Implementation caveat:** In the supplied code, the voting-versus-baseline equality assertions also execute in `voting` mode, even though baseline prediction lists are populated only in `baseline`/`all`. Consequently, the **voting-only mode can fail at those assertions**. `--mode all` is the appropriate current combined run until this is fixed. The Python function's own default (`mode="voting"`) also differs from the CLI default (`all`).

## 7. Held-Out Test Evaluation and Interpretation

The fixed **test partition remains reserved** for a later final evaluation. This script currently checks separation and excludes test artifacts; it **does not produce test-set metrics**.

To preserve a meaningful final test, feature decisions, α selection, and voting settings should be finalized using development evidence before evaluating on test data. Results from repeated development experiments are not an unbiased estimate of held-out generalization.

Additional interpretation cautions:

- Same-class retrieval is a **visual-label benchmark**, not evidence that two motifs share historical meaning.
- Query crops from the same artifact remain grouped, but multiple crops from an artifact can still affect crop-weighted metrics.
- Rare or absent classes in a fold can affect stratification, query eligibility, and fold-level macro-F1.
- Negative controls test their ability to distract retrieval; because they are not queries, this protocol does **not** directly measure negative-query rejection.

## Implementation References

- `evaluate.py` — dev-only split filtering, grouped CV, retrieval, discrimination, voting, exports.
- `evaluation_metrics.py` — retrieval metric calculation and confusion-matrix helper.
- `class_voting.py` — voting strategies and artifact deduplication.
- `data/splits/v0.2_split.json` — fixed artifact-level development/test assignments.
- [Hybrid Retrieval Pipeline](hybrid_retrieval_pipeline.md) — preprocessing and similarity architecture.
- [Geometric Feature Ablations](geometric_feature_ablations.md) — geometric ablation rationale.

*This document describes the v0.2 evaluation implementation provided for review; the source code remains authoritative for subsequent changes.*
