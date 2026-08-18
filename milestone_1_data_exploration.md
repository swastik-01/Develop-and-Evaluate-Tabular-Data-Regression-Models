# Milestone 1 — Data Exploration and Preprocessing

**Project:** Binary Classification on the Bank Marketing Dataset
**Dataset:** Kaggle Playground Series — Season 5, Episode 8 (S5E8)
**Goal of this milestone:** Perform initial data analysis, handle missing values, and prepare the features for modeling.

---

## 1. Objective

Predict whether a bank customer will subscribe to a **term deposit** (target column `y`, values 0/1). The competition is scored with **ROC-AUC**, so the model must output a well-ranked probability rather than a hard label.

This milestone covers everything up to a clean, fully numeric feature matrix: loading the data, exploring it, confirming quality, engineering a few features, and encoding categoricals.

## 2. Dataset description

| Property | Value |
|---|---|
| Training rows | ~750,000 |
| Test rows | ~250,000 |
| Predictors | 16 |
| Target | `y` (binary) |
| Class balance | **imbalanced** — ≈ 12.1 % positive (subscribed) vs 87.9 % negative |
| Evaluation metric | ROC-AUC |

**Feature groups**

- **Numeric (7):** `age`, `balance`, `day`, `duration`, `campaign`, `pdays`, `previous`
- **Categorical (9):** `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`
- **Identifier:** `id` (not predictive — used only to build the submission file)

## 3. Data loading

```python
train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test  = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
```

The train and test files share the same 16 predictors; `train` additionally contains the target `y`.

## 4. Exploratory data analysis (EDA)

**a) Preview the data** (`train.head()`, `test.head()`) confirms the column names and value formats — a mix of integer numeric columns and lower-case string categoricals (e.g. `job = blue-collar`, `month = may`, `poutcome = unknown`).

**b) Missing-value check** (`train.isnull().sum()`) — **every column reports 0 nulls.** The dataset is complete, so **no imputation is required**. (Note: several categoricals use the literal value `"unknown"` as a category, which is information in itself and is kept as-is rather than treated as missing.)

**c) Column comparison** confirms that `train` has all 16 predictors plus `y`, while `test` has the 16 predictors only:

```
Train columns: [... 16 features ..., 'y']
Test  columns: [... 16 features ...]
```

**d) Target distribution** — the positive class (`y = 1`) is roughly **12 %** of rows. This class imbalance is the reason we use **ROC-AUC** (threshold-independent) and **stratified** cross-validation later.

## 5. Preprocessing

1. **Preserve the test identifier, then drop `id`** from both frames — it carries no signal and would leak row position:
   ```python
   test_ids = test['id']
   train = train.drop('id', axis=1)
   test  = test.drop('id', axis=1)
   ```
2. **Split features and target:**
   ```python
   X = train.drop('y', axis=1)
   y = train['y']
   ```

## 6. Feature engineering

`create_features()` adds simple non-linear transforms — a **squared** term and a **log1p** term — for the first three numeric columns (`age`, `balance`, `day`). These give the tree model ready-made curved features to split on.

**Result:** 16 original features → **22 features** (16 + 6 engineered).

Two caveats are documented for transparency:

1. The function contains a generic `tenure`/`charge` interaction branch (template code). This dataset has no such columns, so that branch never executes — only the squared/log features are actually created.
2. `balance` contains **negative values**, so `log1p(balance)` emits *divide-by-zero / invalid-value* runtime warnings and produces `NaN`/`-inf` for those rows. LightGBM handles missing values natively, so training is unaffected; a cleaner implementation would use a **signed log**: `np.sign(x) * np.log1p(np.abs(x))`.

## 7. Categorical encoding

Every `object` (text) column is converted to integer codes with `LabelEncoder`, fit on train and applied to test so the mapping is consistent:

```python
cat_cols = X.select_dtypes(include=['object']).columns
for col in cat_cols:
    le = LabelEncoder()
    X[col]    = le.fit_transform(X[col].astype(str))
    test[col] = le.transform(test[col].astype(str))
```

## 8. Outcome of this milestone

- Data loaded and verified: **~750k rows, 0 missing values**, target imbalance ≈ 12 %.
- `id` removed; features/target separated.
- 6 engineered features added (**22 total**).
- All categoricals encoded to integers.

The dataset is now a **clean, fully numeric matrix** ready for model development in Milestone 2.
