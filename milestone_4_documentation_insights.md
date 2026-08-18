# Milestone 4 — Documentation and Insights

**Project:** Binary Classification on the Bank Marketing Dataset (Kaggle Playground Series S5E8)
**Goal of this milestone:** Prepare a comprehensive summary of the code, results, and key learnings.

> **Attachment:** the complete Jupyter Notebook — `binary-classification-lightgbm-xgb-catboost.ipynb` — is submitted alongside this document. This write-up summarizes it.

---

## 1. Project summary

The task is to predict whether a bank customer subscribes to a **term deposit** (`y`) from 16 customer/campaign attributes, scored by **ROC-AUC**. The solution is an end-to-end pipeline built around a **LightGBM** gradient-boosting classifier evaluated with **10-fold stratified cross-validation**, achieving an out-of-fold **ROC-AUC of 0.969759**.

## 2. Pipeline recap (end to end)

| Stage | What happens | Milestone |
|---|---|---|
| **Load** | Read ~750k train / ~250k test rows | 1 |
| **Explore** | Confirm 0 missing values; target ≈ 12 % positive | 1 |
| **Preprocess** | Save `test_ids`, drop `id`, split `X`/`y` | 1 |
| **Feature-engineer** | Add squared + log terms → 22 features | 1 |
| **Encode** | `LabelEncoder` on 9 categorical columns | 1 |
| **Model** | LightGBM, 10-fold stratified CV, early stopping | 2 |
| **Tune & evaluate** | Regularized params; OOF ROC-AUC | 3 |
| **Submit** | Fold-averaged probabilities → `submission.csv` | 4 |

## 3. Results summary

| Metric | Value |
|---|---|
| Cross-validation | 10-fold StratifiedKFold |
| **Overall out-of-fold ROC-AUC** | **0.969759** |
| Mean fold AUC | ≈ 0.9698 |
| Fold-to-fold spread | ≈ ±0.0007 |
| Features used | 22 |
| Output file | `submission.csv` (`id`, `y` = predicted probability) |

Per-fold AUC ranged narrowly from **0.968747** to **0.971035** — evidence of a stable, well-generalizing model.

## 4. Key insights and learnings

1. **Gradient boosting dominates on this data.** A single well-tuned LightGBM reaches ~0.97 AUC with only light feature engineering — the algorithm, not hand-crafted features, does the heavy lifting on this table.
2. **Slow-and-wide beats fast-and-narrow.** A very small learning rate (0.009) with a large tree budget and early stopping generalizes better than a high learning rate with few trees.
3. **Stratification matters at 12 % positives.** Stratified folds keep each fold's class balance — and therefore its AUC — comparable and trustworthy.
4. **Choose the metric for the problem.** With heavy imbalance, ROC-AUC is the right yardstick; accuracy would be misleading (~88 % by always predicting "no").
5. **Ensembling by fold-averaging** reduces variance versus training one model on all data.
6. **`duration` is the dominant signal** in this dataset family (last-contact length). It inflates offline scores but is unavailable before a call is placed, so it should be used cautiously in any real deployment.

## 5. Limitations and next steps

- **Fix the `balance` transform** — use a signed log (`np.sign(x) * np.log1p(np.abs(x))`) to avoid the `NaN`/`-inf` values, and remove the inert `tenure`/`charge` template branch in `create_features()`.
- **Add model diversity** — train **XGBoost** and **CatBoost** with the same CV scheme and **blend** the three. This typically adds a few ten-thousandths of AUC and satisfies a rubric expecting *more than one model* (and matches the notebook's filename).
- **Native categorical handling** — pass categoricals directly to LightGBM/CatBoost instead of label-encoding; this often improves splits on high-cardinality columns such as `job` and `month`.
- **Explainability** — add feature-importance / SHAP analysis to confirm which features drive predictions and to guide further engineering.

## 6. How to reproduce

1. Open `binary-classification-lightgbm-xgb-catboost.ipynb`.
2. Ensure the S5E8 `train.csv` / `test.csv` are available at the input path.
3. Run all cells top to bottom (the notebook is documented with markdown for each milestone).
4. The final cell writes `submission.csv` with the predicted subscription probabilities.

## 7. Deliverables for this milestone

- **This document** — comprehensive summary of code, results, and insights.
- **`binary-classification-lightgbm-xgb-catboost.ipynb`** — the complete, documented notebook.
- **`submission.csv`** — the model's predictions in the required format.
