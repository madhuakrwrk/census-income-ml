# Census Income — Classification & Marketing Segmentation

End-to-end machine-learning solution for the retail take-home project using
the 1994–1995 U.S. Current Population Survey extract (199,523 records, 40
demographic and employment features).

Two deliverables are built from the same pipeline:

1. **Classification model** — predicts whether a person earns more than
   $50,000 per year, with population-weighted evaluation and calibration.
2. **Marketing segmentation model** — unsupervised clustering over the same
   population, profiled into six actionable personas.

---

## Headline results

*(Reproduced from `artifacts/metrics/classifier_metrics.json` and
`artifacts/metrics/segmentation_profile.csv`.)*

### Classification (held-out test set, 39,905 rows, survey-weighted)

| Metric                             | Logistic Regression | **HistGradientBoosting** |
|------------------------------------|---------------------|--------------------------|
| ROC AUC                            | 0.9461              | **0.9542**               |
| Average Precision (PR AUC)         | 0.6264              | **0.6931**               |
| Brier score                        | 0.0362              | **0.0329**               |
| F1 @ tuned threshold (0.27)        | 0.5915              | **0.6225**               |
| Precision @ top 5% of scored list  | 0.6514              | **0.7105**               |
| Precision @ top 10%                | 0.4711              | **0.4837**               |
| Lift @ top 10%                     | 7.30×               | **7.49×**                |

Cross-validated HGBC: ROC AUC 0.9513 ± 0.0022, PR AUC 0.6704 ± 0.0071 over
5 stratified folds — very small variance, confirming the model is not
overfitting the training split.

### Segmentation (K-Prototypes, k=6, profiled with survey weights)

| Segment | Persona (short)                                   | % of population | P(income >$50k) | Lift vs base |
|:-------:|:--------------------------------------------------|-----------------:|----------------:|-------------:|
| C0      | Self-employed professional elite (MD/JD/DDS)      | 0.2%             | **88.1%**       | **13.8×**    |
| C2      | Affluent retired capital-holders                  | 0.2%             | 70.7%           | 11.0×        |
| C1      | Working-age high-earner professionals             | 2.0%             | 30.4%           | 4.8×         |
| C3      | Prime-age working class / broad middle            | 46.7%            | 11.2%           | 1.7×         |
| C5      | Elderly retirees, out of labor force              | 18.2%            | 1.6%            | 0.3×         |
| C4      | Children under 18 (reach via parents)             | 32.8%            | 0.03%           | 0.01×        |

Full segment profiles (demographic / employment differentiators) are in
`artifacts/metrics/segmentation_profile.csv`. The client-facing project
report in `reports/project_report.md` walks through each persona with
marketing angles and suggested product categories.

---

## Repository layout

```
census-income-ml/
├── README.md                       <- you are here
├── requirements.txt                <- pip-installable dependency list
├── data/
│   └── raw/
│       ├── census-bureau.data      <- input dataset
│       └── census-bureau.columns   <- sidecar column names
├── src/census_income/              <- importable library
│   ├── config.py                   <- paths, column metadata, feature roles
│   ├── data.py                     <- loader + stratified split
│   ├── classifier.py               <- HGBC + logreg + CV + Optuna tuning
│   ├── segmentation.py             <- K-Prototypes + persona profiling
│   └── evaluation.py               <- weighted metrics, plots, subgroup audit
├── scripts/                        <- end-to-end runnable entrypoints
│   ├── 01_run_eda.py
│   ├── 02_train_classifier.py
│   ├── 03_run_segmentation.py
│   └── 04_render_report_pdf.py     <- renders project_report.md → PDF
├── artifacts/                      <- outputs produced by the scripts
│   ├── models/                     <- joblib-serialized classifiers / cluster
│   ├── metrics/                    <- JSON / CSV metric bundles
│   └── figures/                    <- PNG plots
└── reports/
    ├── project_report.md           <- client-facing report (≤10 pages)
    ├── references.md               <- resources consulted
    └── figures/                    <- report-ready plots
```

---

## Reproducing the pipeline

### 1. Python environment

The code is developed and validated against **Python 3.11 / 3.13**. Any
reasonably recent 3.10+ CPython installation with `pip` should work.

```bash
# from the repo root
python -m venv .venv
source .venv/bin/activate           # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

No system-level dependencies are required. The gradient-boosting model is
`sklearn.ensemble.HistGradientBoostingClassifier`, which is implemented in
pure Cython with no OpenMP runtime, so there is nothing to `brew install`.

### 2. Data

The raw files ship inside `data/raw/` and the loader looks them up there
by default. Nothing to download.

### 3. Run the full pipeline

Four scripts, each runnable end-to-end from a clean clone. They are
numbered for clarity but have no hard dependency on each other — the
segmentation and EDA scripts can run on their own.

```bash
# Step 1 — Exploratory data analysis (generates EDA figures + summary JSON)
python scripts/01_run_eda.py

# Step 2 — Classification (≈1.5 min with defaults, ≈10 min with --tune)
#   Trains logistic-regression baseline + HistGradientBoosting,
#   runs 5-fold CV, Optuna tuning, threshold selection, evaluation.
python scripts/02_train_classifier.py --tune --trials 25

# Step 3 — Marketing segmentation (≈4 min)
#   K-Prototypes clustering, k-sweep, profiling with survey weights.
python scripts/03_run_segmentation.py --k 6

# Step 4 — Render the project report to PDF (optional, < 5 sec)
#   Converts reports/project_report.md → reports/project_report.pdf
python scripts/04_render_report_pdf.py
```

Each of the first three scripts writes:
- metrics to `artifacts/metrics/*.json` / `.csv`
- figures to `artifacts/figures/*.png` and `reports/figures/*.png`
- serialized models to `artifacts/models/*.joblib`

Script 04 reads `reports/project_report.md` and emits the client-ready
`reports/project_report.pdf` (≤10 pages).

Random seeds are fixed in `src/census_income/config.py::RANDOM_STATE`,
so the headline numbers reproduce bit-identically between runs.

**Quick-start (run everything end-to-end):**

```bash
python scripts/01_run_eda.py && \
python scripts/02_train_classifier.py --tune --trials 25 && \
python scripts/03_run_segmentation.py --k 6 && \
python scripts/04_render_report_pdf.py
```

### 4. Loading a saved model

```python
import joblib, pandas as pd
from census_income.data import load_raw
from census_income import config

model = joblib.load("artifacts/models/hgbc.joblib")
df = load_raw().sample(5, random_state=0)
scores = model.predict_proba(df[config.FEATURES])[:, 1]
print(scores)
```

The saved HGBC carries the categorical-feature mask so the DataFrame can
be passed in as-is, without any pre-encoding step.

---

## Design notes (why this instead of that)

A detailed write-up is in `reports/project_report.md`. The short version:

- **HistGradientBoosting over LightGBM/XGBoost** — native categorical
  and missing-value handling, identical performance characteristics on
  tabular data, zero runtime dependencies. Chosen deliberately, not by
  default.
- **Survey weights are used at fit *and* eval time.** Without them, the
  model is optimising the *sample* loss, not the *population* loss — and
  the client cares about the population.
- **Threshold is selected on out-of-fold predictions**, not on the test
  set. Reporting F1/precision/recall at a threshold chosen on test is a
  classic subtle leakage error; we avoid it.
- **"Not in universe" is treated as a category, not as a missing value.**
  It means "this question doesn't apply to this person" (e.g., children
  don't have occupations). Imputing it would destroy information.
- **Segmentation uses K-Prototypes, not K-Means over one-hot features.**
  K-Prototypes applies the right distance to each column type and
  preserves the semantics of the categorical levels.
- **k = 6 is chosen for business, not mathematical, reasons.** The elbow
  is at k=4, but k=6 preserves two tiny but extremely high-value affluent
  micro-segments (C0 and C2) that collapse into the broad middle at k=4.
  This is discussed in the report.

---

## Fairness and responsible use — a note for the client

The dataset encodes race, sex, and national origin, and the subgroup audit
(`artifacts/metrics/classifier_subgroup.csv`) shows material differences
in base rates between subgroups. The model is *discriminative*, i.e. it
reproduces the historical distribution it was trained on — a 1994
statistical artefact, not a fair distribution. Before using the classifier
in any decision that affects credit, employment, housing, or pricing, a
full fairness review is required. For purely discretionary *marketing*
targeting (e.g. catalog mailers, product recommendations) the risk is
lower but not zero, and the report lays out mitigations.

---

## License

MIT. See `LICENSE`.
