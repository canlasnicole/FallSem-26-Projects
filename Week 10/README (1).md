# Titanic Survival Prediction: Cross-Validation & Hyperparameter Tuning with Optuna

> 인공지능모델운영 (Operation of Artificial Intelligence Model) course assignment
> **CANLAS NICOLE** · 2555039

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/<your-repo>/blob/main/titanic_cv_optuna_colab.ipynb)

This project compares six classifiers on the Titanic dataset with **Stratified 5-Fold Cross-Validation**, then tunes each one with **Optuna (TPE Sampler)**. All preprocessing is wrapped inside a scikit-learn `Pipeline`, so every statistic (medians, scaling parameters, category sets) is learned only from the training fold. This prevents data leakage.

**Result:** after tuning, XGBoost ranked first on all seven metrics, with an F1-Score of 0.7957, an AUROC of 0.8812 and 26.8 misclassifications per fold.

---

## Pipeline

```
train.csv
  │
  ├─ Stateless cleaning (before CV split; no statistics used)
  │    • Extract Title from Name → Mr / Miss / Mrs / Master / Rare
  │    • Drop PassengerId, Ticket, Cabin, Name
  │
  └─ sklearn Pipeline (fit inside each fold)
       ├─ GroupMedianImputer   Age ← median of (Pclass, Title) group
       ├─ ColumnTransformer
       │    ├─ Numeric  [Age, Fare, SibSp, Parch] → median impute → StandardScaler
       │    ├─ Category [Sex, Embarked, Title]    → mode impute   → OneHotEncoder(drop='first')
       │    └─ Pclass                             → passthrough
       └─ Model (KNN / LogReg / NB / DT / RF / XGBoost)
```

`GroupMedianImputer` is a custom transformer. Its `fit()` computes the group medians on the training fold only. If a group combination never appears in the training fold, it falls back to the global median.

## Evaluation setup

| Item | Setting |
| --- | --- |
| Cross-validation | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` |
| Tuning | Optuna `TPESampler(seed=42)`, 50 trials per model |
| Objective | Mean 5-fold F1-Score (maximize) |
| Metrics | FN+FP, FPR, Recall, Precision, F1-Score, PR_AUC, AUROC |
| Reproducibility | Fixed seeds (Python, NumPy, models, sampler), `n_jobs=1` |

FN+FP is the average number of misclassified passengers per fold (about 178 passengers per fold).

## Results

### Baseline (default hyperparameters)

| Model | FN+FP | FPR | Recall | Precision | F1-Score | PR_AUC | AUROC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| KNN | 32.2 | 0.1294 | 0.7368 | 0.7829 | 0.7581 | 0.7753 | 0.8538 |
| Logistic Regression | 31.2 | 0.1202 | 0.7366 | 0.7934 | 0.7631 | 0.8489 | 0.8702 |
| Naive Bayes | 38.6 | 0.1585 | 0.6898 | 0.7309 | 0.7087 | 0.7654 | 0.8376 |
| Decision Tree | 36.6 | 0.1621 | 0.7251 | 0.7358 | 0.7303 | 0.6388 | 0.7788 |
| Random Forest | 32.4 | 0.1311 | 0.7368 | 0.7785 | 0.7568 | 0.8237 | 0.8708 |
| XGBoost | 32.0 | 0.1293 | 0.7396 | 0.7808 | 0.7596 | 0.8495 | 0.8738 |

### Tuned (Optuna best parameters)

| Model | FN+FP | FPR | Recall | Precision | F1-Score | PR_AUC | AUROC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| KNN | 30.6 | 0.1129 | 0.7338 | 0.8021 | 0.7662 | 0.8413 | 0.8751 |
| Logistic Regression | 29.6 | 0.1166 | 0.7542 | 0.8015 | 0.7767 | 0.8549 | 0.8731 |
| Naive Bayes | 38.6 | 0.1585 | 0.6898 | 0.7309 | 0.7087 | 0.7654 | 0.8376 |
| Decision Tree | 30.4 | 0.1221 | 0.7512 | 0.7935 | 0.7710 | 0.8102 | 0.8629 |
| Random Forest | 28.0 | 0.0984 | 0.7485 | 0.8273 | 0.7853 | 0.8560 | 0.8787 |
| **XGBoost** | **26.8** | **0.0965** | **0.7631** | **0.8317** | **0.7957** | **0.8619** | **0.8812** |

### Best hyperparameters

| Model | Best CV F1 | Best parameters |
| --- | --- | --- |
| XGBoost | 0.7957 | `n_estimators=100, max_depth=8, learning_rate=0.0649, subsample=0.823, colsample_bytree=0.765, min_child_weight=2, gamma=0.157` |
| Random Forest | 0.7853 | `n_estimators=300, max_depth=15, min_samples_split=2, min_samples_leaf=2, max_features='log2'` |
| Logistic Regression | 0.7767 | `C=6.847, penalty='l2'` |
| Decision Tree | 0.7710 | `criterion='gini', max_depth=3, min_samples_split=15, min_samples_leaf=5` |
| KNN | 0.7662 | `n_neighbors=17, weights='uniform', p=1` |
| Naive Bayes | 0.7087 | `var_smoothing=1.77e-09` |

## Key findings

- **XGBoost performed best.** Boosting corrects the previous trees' residuals one tree at a time, which captures interactions such as Sex × Pclass × Title. Its depth of 8 is held in check by a low learning rate, row and column subsampling, and `gamma`.
- **Decision Tree improved the most:** 6.2 fewer errors per fold, with PR_AUC up 0.1714. The unconstrained default tree overfits and outputs near-0/1 probabilities. Limiting `max_depth=3` and `min_samples_leaf=5` acts as pre-pruning and produces smoother probability estimates.
- **Naive Bayes did not change.** Its only hyperparameter, `var_smoothing`, adds a tiny value to the variance for numerical stability and cannot change the model's structure. Its performance is limited by the conditional-independence assumption, not by tuning.
- **Most gains came from fewer false positives.** For the top models, FPR dropped (XGBoost: 0.1293 → 0.0965) and precision rose.

## Limitations

- **Optimistic bias.** Tuning and final evaluation use the same 5-fold split, so the tuned scores are somewhat optimistic. Nested CV or a separate hold-out set would give an unbiased estimate.
- **Close margins.** The F1 gap between XGBoost and Random Forest (0.0104, about 1.2 errors per fold) has not been tested for statistical significance; per-fold standard deviations were not computed.
- **Fixed threshold.** F1 is computed at a 0.5 decision threshold. Threshold tuning could shift the precision–recall balance further.

## How to run

**Google Colab (recommended):** click the badge above, or upload `titanic_cv_optuna_colab.ipynb` via *File → Upload notebook*, then select *Runtime → Run all*. The tuning step takes several minutes.

**Locally:**

```bash
pip install optuna xgboost scikit-learn pandas numpy
python titanic_cv_optuna_colab.py
```

The dataset is downloaded automatically from the [pydata-book repository](https://github.com/wesm/pydata-book/tree/3rd-edition/datasets/titanic). No local data file is needed.

## Repository structure

```
.
├── titanic_cv_optuna_colab.ipynb   # Colab notebook (cell-by-cell)
├── titanic_cv_optuna_colab.py      # Same code as a standalone script
└── README.md
```

## Tech stack

Python · scikit-learn · Optuna · XGBoost · pandas · NumPy · Google Colab
