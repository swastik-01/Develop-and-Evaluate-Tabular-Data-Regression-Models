# Milestone 2 — Model Development and Training

**Project:** Binary Classification on the Bank Marketing Dataset (Kaggle Playground Series S5E8)
**Goal of this milestone:** Build and train the model, documenting the algorithm choice and the training procedure.

---

## 1. Objective

Train a classifier that predicts the probability of term-deposit subscription (`y`) and generalizes well to unseen data. Because the metric is **ROC-AUC** on an imbalanced target, the emphasis is on producing well-ranked probabilities and on an honest, leakage-free training/validation procedure.

## 2. Model choice and rationale

**Selected algorithm: LightGBM (`LGBMClassifier`)** — a gradient-boosted decision tree (GBDT) framework.

Why LightGBM is the right fit for this problem:

| Reason | Detail |
|---|---|
| **Best-in-class for tabular data** | GBDTs consistently outperform linear/other models on structured, mixed-type tables like this one. |
| **Captures interactions automatically** | Trees model non-linear relationships and feature interactions without manual crafting. |
| **No scaling needed** | Tree splits are scale-invariant, so numeric features can be used as-is. |
| **Handles imbalance** | The AUC training objective focuses on ranking quality rather than raw accuracy. |
| **Fast & memory-efficient** | Histogram-based training scales comfortably to 750k rows. |
| **Native missing-value handling** | The `NaN`/`-inf` values from the `balance` log feature are handled internally. |

> **Note on the "at least two models" requirement.** This notebook trains a single, strong LightGBM model. The filename (`lightgbm-xgb-catboost`) anticipates an ensemble with **XGBoost** and **CatBoost**; adding those two with the same cross-validation setup and blending the three is the natural extension (see Milestone 4 → *Next steps*).

## 3. Training strategy

The model is trained with **10-fold Stratified K-Fold cross-validation** rather than a single train/test split, which gives a more reliable performance estimate and a stronger set of test predictions.

Key elements:

- **`StratifiedKFold(n_splits=10, shuffle=True, random_state=42)`** — stratification preserves the ≈ 12 % positive rate in every fold.
- **Early stopping (500 rounds)** on validation AUC — up to 10,000 trees are available, but each fold stops once the validation score stops improving, preventing overfitting and saving time.
- **Out-of-fold (OOF) predictions** — each row is predicted by the fold in which it is held out, giving an unbiased CV score.
- **Fold-averaged test predictions** — the test set is scored by every fold model and the 10 probabilities are averaged (bag averaging), which is more stable than a single model.

## 4. Training loop (code walkthrough)

```python
NFOLDS = 10
folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42)
oof_preds = np.zeros(X.shape[0])       # out-of-fold predictions
sub_preds = np.zeros(test.shape[0])    # averaged test predictions

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

    model = LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(500, verbose=False),
                   lgb.log_evaluation(False)]
    )

    oof_preds[valid_idx] = model.predict_proba(X_valid)[:, 1]
    sub_preds += model.predict_proba(test)[:, 1] / NFOLDS

    fold_auc = roc_auc_score(y_valid, oof_preds[valid_idx])
    print(f'Fold {n_fold+1} AUC: {fold_auc:.6f}')
```

Each fold trains on **675,000** rows (81,440 positive / 593,560 negative) and validates on the remaining 75,000. LightGBM reports it uses **22 features** — confirming the engineered features are picked up.

## 5. Training output

Each fold prints its validation AUC as it completes (values ≈ 0.9687–0.9710 — see Milestone 3 for the full table). The stable per-fold scores show the training procedure is sound and the model is learning consistent signal across different splits of the data.

## 6. Outcome of this milestone

- A LightGBM classifier trained across **10 stratified folds** with early stopping.
- **OOF predictions** for honest evaluation (used in Milestone 3).
- **Fold-averaged test probabilities** ready to become the submission (Milestone 4).
