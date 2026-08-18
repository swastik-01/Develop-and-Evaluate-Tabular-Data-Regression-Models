# Milestone 3 — Hyperparameter Tuning and Evaluation

**Project:** Binary Classification on the Bank Marketing Dataset (Kaggle Playground Series S5E8)
**Goal of this milestone:** Optimize the model using appropriate techniques and evaluate performance using relevant metrics.

---

## 1. Objective

Tune the LightGBM model for strong, stable generalization, and evaluate it with a metric appropriate to an imbalanced binary problem — **ROC-AUC**, measured out-of-fold so the score is unbiased.

## 2. Hyperparameters and rationale

```python
params = {
    'objective': 'binary',
    'metric': 'auc',
    'n_estimators': 10000,
    'learning_rate': 0.009,
    'num_leaves': 41,
    'max_depth': -1,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'subsample_freq': 1,
    'random_state': 42,
    'n_jobs': -1,
}
```

| Parameter | Value | Why this value |
|---|---|---|
| `objective` / `metric` | `binary` / `auc` | Directly optimizes and monitors the competition's ROC-AUC metric. |
| `n_estimators` | 10,000 | An **upper bound** only — early stopping selects the true count per fold. |
| `learning_rate` | 0.009 | A **small** learning rate paired with many trees generalizes better than a large one. |
| `num_leaves` | 41 | The primary capacity control; tuned to balance fit vs. overfitting. |
| `max_depth` | -1 | Unbounded depth — capacity is governed by `num_leaves` instead. |
| `min_child_samples` | 20 | Minimum samples per leaf — regularizes against fitting noise. |
| `reg_alpha` / `reg_lambda` | 0.1 / 0.1 | L1 / L2 penalties to further control overfitting. |
| `colsample_bytree` | 0.8 | Uses 80 % of features per tree — adds randomness, reduces variance. |
| `subsample` / `subsample_freq` | 0.8 / 1 | Row bagging (80 % of rows) applied every iteration. |
| `random_state` | 42 | Reproducibility. |

## 3. Tuning methodology

The parameter set reflects **informed, regularization-first tuning** for a large tabular dataset:

1. **Low learning rate + high tree budget + early stopping.** Instead of guessing the number of trees, set a very small `learning_rate` (0.009) with a large ceiling (`n_estimators = 10000`) and let **early stopping (500 rounds)** on validation AUC choose the optimal count for each fold. This is the single most effective GBDT tuning lever.
2. **Capacity control via `num_leaves` (41) with unlimited depth.** For leaf-wise boosting, `num_leaves` is the main complexity knob; 41 balances expressiveness against overfitting.
3. **Explicit regularization** — `min_child_samples`, `reg_alpha`, and `reg_lambda` all discourage over-complex trees.
4. **Stochastic training** — `colsample_bytree` and `subsample` (both 0.8) inject randomness that lowers variance and improves generalization.

Every choice is validated through the 10-fold CV described below, so improvements are measured on held-out data rather than the training set.

## 4. Evaluation

**Metric — ROC-AUC.** With a ≈ 12 % positive rate, plain accuracy is misleading (predicting "no" for everyone scores ~88 %). ROC-AUC measures **ranking quality** across all thresholds and is the competition's official metric.

**Protocol — Out-of-fold (OOF) evaluation.** Each row is scored by the fold in which it is held out, so the reported AUC reflects genuine generalization, not training-set fit.

### Per-fold out-of-fold ROC-AUC

| Fold | AUC | | Fold | AUC |
|---|---|---|---|---|
| 1 | 0.971035 | | 6 | 0.969471 |
| 2 | 0.969880 | | 7 | 0.970545 |
| 3 | 0.968747 | | 8 | 0.969767 |
| 4 | 0.969605 | | 9 | 0.970182 |
| 5 | 0.969058 | | 10 | 0.969329 |

### Overall result

| Measure | Value |
|---|---|
| **Overall out-of-fold ROC-AUC** | **0.969759** |
| Mean fold AUC | ≈ 0.9698 |
| Fold-to-fold spread | ≈ ±0.0007 |

## 5. Interpretation

- An OOF AUC of **≈ 0.97** indicates the model separates subscribers from non-subscribers **very well**.
- The **tiny variance across folds (±0.0007)** shows the model is **stable and not overfitting** to any particular split — the tuning is well-regularized.
- Because early stopping tailors the tree count per fold, the model is efficient as well as accurate.

## 6. Outcome of this milestone

- A documented, rationale-backed hyperparameter configuration.
- A robust evaluation via 10-fold OOF ROC-AUC: **0.969759**, with low variance — a reliable, well-generalizing model ready for final documentation and submission (Milestone 4).
