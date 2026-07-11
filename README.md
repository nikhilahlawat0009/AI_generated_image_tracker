# Predictiva — Machine Learning Engineer Assignment

**Author:** Nikhil Ahlawat
**Email:** nikhil.ahlawat@yahoo.com
**Kaggle username:** nikhilahlawat0008

This repository contains my submission for the Predictiva ML Engineer selection process:
the Kaggle pairwise-preference challenge and the two assignment problems.

---

## Repository structure

```
.
├── README.md
├── kaggle_pairwise/           # Part 1 — Kaggle challenge
│   ├── attempt3_pnl_rule.ipynb        # final submitted notebook
│   ├── model_1_logistic_regression.ipynb
│   ├── model_2_svm_rbf.ipynb
│   ├── model_3_random_forest.ipynb
│   ├── model_4_gradient_boosting.ipynb
│   └── submission.csv
├── problem_1/                 # Part 2 — Problem 1 (compulsory)
│   └── ...
└── problem_2/                 # Part 2 — Problem 2 (bonus)
    └── ...
```

> Adjust the folder/file names above to match what you actually commit.

---

## Part 1 — Kaggle pairwise-preference challenge

### Problem
Each sample is a pair of multivariate time series `(A, B)`. Both agents act on financial OHLCV data
plus a binary `position` mask (`1` = holding, `0` = flat). The label indicates which agent performed
better under a hidden evaluation criterion (`0` = A wins, `1` = B wins). Submissions are scored on
accuracy.

### Key insight
Within every pair, **A and B share identical OHLCV** — only the `position` column differs. The market
is therefore a constant, and the task reduces to: *whose position decisions produced the better
trading outcome?* This reframes a generic time-series problem into a **realized-P&L comparison**.

### Approach
1. **EDA** — confirmed the shared-market structure and the variable sequence lengths.
2. **Feature engineering** — reconstructed each agent's trading statistics (realized P&L, Sharpe,
   drawdown, capture ratio, hold-lengths, etc.). The strongest single feature is a **trade-based P&L
   with next-bar-open execution**.
3. **Problem framing** — binary classification via **pairwise preference / learning-to-rank**:
   classify the antisymmetric difference `f(A) − f(B) → P(B wins)`, with swap-augmentation
   `(−x, 1−y)` enforcing antisymmetry.
4. **Model comparison** — tuned Logistic Regression, SVM (RBF), Random Forest, and Gradient Boosting
   with grouped, leakage-safe cross-validation and a 20% held-out test set. Training-vs-CV gaps were
   used to diagnose over/under-fitting.
5. **Model selection** — chose the model that generalizes best (highest CV with a small train–CV
   gap), rejecting the tree/SVM models that reached ~100% training accuracy (overfitting on ~208
   pairs).

### Results

| Approach | Cross-val accuracy | Notes |
|---|---|---|
| Parameter-free realized-P&L rule | ~0.83 | no parameters → cannot overfit (final submission) |
| Logistic Regression (tuned) | ~0.84 | best fitted model, small overfit gap |
| SVM / Random Forest / Gradient Boosting | 0.79–0.81 | overfit at n = 208 |

Best public leaderboard score: **0.81**, from the parameter-free P&L rule. I deliberately chose the
zero-parameter rule as the final submission because it matched the tuned models in cross-validation
while carrying no overfitting risk on the private half of the test set.

### Reproducing
```bash
pip install numpy pandas scikit-learn
```
Open `attempt3_pnl_rule.ipynb`, set the `DATA` path to the competition input folder, and run all
cells. It writes `submission.csv`.

---

## Part 2 — Problem 1 (compulsory)

**Objective:** _[one-line description of Problem 1]_

**Approach:** _[brief summary of your method and key decisions]_

**How to run:**
```bash
# e.g. pip install -r problem_1/requirements.txt
# python problem_1/main.py
```

**Results:** _[key metrics / outputs]_

---

## Part 2 — Problem 2 (bonus)

**Objective:** _[one-line description of Problem 2]_

**Approach:** _[brief summary]_

**How to run:**
```bash
# ...
```

**Results:** _[key metrics / outputs]_

---

## Notes
- Each notebook / script documents its reasoning inline.
- Cross-validation and a held-out test set were used throughout to give honest generalization
  estimates rather than optimizing the public leaderboard.
