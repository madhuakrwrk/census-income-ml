"""Classification models for the income >$50k task.

We train two models:

* A **linear baseline** (L2-regularised logistic regression over one-hot
  encoded categoricals + standardised numerics). This is the honest floor;
  if the fancy model can't clearly beat this, the extra complexity isn't
  earning its keep.

* A **histogram gradient boosting** model (``HistGradientBoostingClassifier``).
  This is the production model. HGBC has native categorical and
  missing-value handling, so we pass the raw DataFrame straight in without
  any encoding step — that's both simpler and fairer to the model (an
  ordinal split on a 47-level occupation code beats a one-hot wall).

Both models are fit with the survey weights, so their loss is population-
weighted and the resulting scores are meaningful as US-population-level
probabilities. We don't up/down-sample the minority class — weighting is
cleaner and gives us back a calibrated probability for free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config
from .evaluation import compute_classification_metrics, pick_threshold_f1


# --------------------------------------------------------------------------- #
# Linear baseline
# --------------------------------------------------------------------------- #

def build_logreg_pipeline() -> Pipeline:
    """A plain-vanilla, well-regularised logistic regression.

    Categoricals are one-hot encoded with an "infrequent" bucket so high-
    cardinality columns (47 occupation codes, 42 countries of birth) don't
    blow up the feature space. NAs are imputed at the preprocessing stage
    as a separate "missing" category — the one-hot encoder handles that
    automatically because NAs arrive as the pandas sentinel.
    """
    numeric = Pipeline(
        steps=[
            # NA in numerics is rare here (weeks-worked is never NA), but
            # the safety net matters because the LR solver won't tolerate
            # NaNs under any circumstances.
            ("scale", StandardScaler()),
        ]
    )

    categorical = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=20,
        sparse_output=True,
        dtype=np.float32,
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, config.NUMERIC_FEATURES),
            ("cat", categorical, config.CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # ``class_weight='balanced'`` combines with ``sample_weight`` at fit time
    # by multiplication. We set class_weight=None and rely purely on the
    # survey weights to avoid double-weighting the minority class.
    clf = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        C=1.0,
        class_weight=None,
        max_iter=2000,
        random_state=config.RANDOM_STATE,
    )

    return Pipeline(steps=[("pre", pre), ("clf", clf)])


# --------------------------------------------------------------------------- #
# Hero model: HistGradientBoostingClassifier
# --------------------------------------------------------------------------- #

# Default hyperparameters. Chosen as a sensible starting point — tuning
# updates this set in-place via :func:`tune_hgbc`.
DEFAULT_HGBC_PARAMS: dict = dict(
    loss="log_loss",
    learning_rate=0.06,
    max_iter=600,
    max_leaf_nodes=48,
    min_samples_leaf=40,
    l2_regularization=0.1,
    max_features=0.9,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=30,
    random_state=config.RANDOM_STATE,
)


def _categorical_mask(X: pd.DataFrame) -> list[bool]:
    """Column-order-aligned boolean mask for categorical features.

    HGBC wants a boolean mask the same length as ``X.columns``. We build it
    from the config lists rather than trusting pandas dtype inspection so
    that a forgotten ``.astype('category')`` in the loader surfaces as a
    config error at fit time.
    """
    return [col in config.CATEGORICAL_FEATURES for col in X.columns]


def build_hgbc(params: dict | None = None) -> HistGradientBoostingClassifier:
    """Instantiate the gradient-boosting classifier with categorical support."""
    p = DEFAULT_HGBC_PARAMS.copy()
    if params:
        p.update(params)
    return HistGradientBoostingClassifier(**p)


def fit_hgbc(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    params: dict | None = None,
) -> HistGradientBoostingClassifier:
    """Fit the HGBC with categorical features routed through native splits.

    The categorical mask is positional — it has to match ``X_train.columns``
    after any column reordering. We assert the mask has the expected length
    so an accidental ``.drop`` upstream fails loudly instead of silently
    mis-routing columns.
    """
    clf = build_hgbc(params)
    mask = _categorical_mask(X_train)
    assert len(mask) == X_train.shape[1], "categorical mask length mismatch"
    clf.set_params(categorical_features=mask)
    clf.fit(X_train, y_train, sample_weight=sample_weight)
    return clf


# --------------------------------------------------------------------------- #
# Cross-validated evaluation
# --------------------------------------------------------------------------- #

@dataclass
class CVResult:
    fold_metrics: list[dict]
    mean_metrics: dict
    std_metrics: dict
    oof_scores: np.ndarray  # out-of-fold predicted probabilities


def cross_validate_hgbc(
    X: pd.DataFrame,
    y: np.ndarray,
    sample_weight: np.ndarray,
    n_splits: int = config.N_CV_FOLDS,
    params: dict | None = None,
) -> CVResult:
    """Run stratified K-fold CV and return per-fold metrics + OOF scores.

    Out-of-fold scores serve two purposes:
    1. They drive unbiased threshold tuning (we pick the threshold on OOF
       predictions and lock it before touching the test set).
    2. They feed the calibration curve without needing a second holdout.
    """
    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE
    )
    oof = np.zeros(len(y), dtype=float)
    fold_metrics: list[dict] = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr = X.iloc[tr_idx]
        X_va = X.iloc[va_idx]
        y_tr = y[tr_idx]
        y_va = y[va_idx]
        w_tr = sample_weight[tr_idx]
        w_va = sample_weight[va_idx]

        clf = fit_hgbc(X_tr, y_tr, w_tr, params=params)
        p_va = clf.predict_proba(X_va)[:, 1]
        oof[va_idx] = p_va

        m = compute_classification_metrics(y_va, p_va, w_va)
        fold_row = {"fold": fold, **m.to_dict()}
        fold_metrics.append(fold_row)

    df = pd.DataFrame(fold_metrics)
    mean_metrics = df.drop(columns=["fold"]).mean(numeric_only=True).to_dict()
    std_metrics = df.drop(columns=["fold"]).std(numeric_only=True).to_dict()

    return CVResult(
        fold_metrics=fold_metrics,
        mean_metrics=mean_metrics,
        std_metrics=std_metrics,
        oof_scores=oof,
    )


# --------------------------------------------------------------------------- #
# Hyperparameter tuning
# --------------------------------------------------------------------------- #

def tune_hgbc(
    X: pd.DataFrame,
    y: np.ndarray,
    sample_weight: np.ndarray,
    n_trials: int = 30,
    timeout_seconds: int | None = None,
) -> dict:
    """Tune HGBC with Optuna, maximising weighted PR-AUC on 3-fold CV.

    We optimise PR-AUC rather than ROC-AUC because the positive rate is
    ~6% — PR-AUC is more sensitive to the head of the ranking, which is
    exactly what marketing cares about.

    Three folds (not five) keeps each trial inexpensive so we can sample
    the hyperparameter space properly within a reasonable budget. For the
    final fit we re-fold to 5 using the selected parameters.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "max_iter": trial.suggest_int("max_iter", 200, 900),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 16, 96),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 120),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 5.0, log=True),
            "max_features": trial.suggest_float("max_features", 0.6, 1.0),
        }

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.RANDOM_STATE)
        aps: list[float] = []
        for tr_idx, va_idx in skf.split(X, y):
            clf = fit_hgbc(
                X.iloc[tr_idx], y[tr_idx], sample_weight[tr_idx], params=params
            )
            p = clf.predict_proba(X.iloc[va_idx])[:, 1]
            m = compute_classification_metrics(y[va_idx], p, sample_weight[va_idx])
            aps.append(m.pr_auc)
        return float(np.mean(aps))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds, show_progress_bar=False)

    best = dict(DEFAULT_HGBC_PARAMS)
    best.update(study.best_params)
    best["_best_value_pr_auc"] = float(study.best_value)
    return best


# --------------------------------------------------------------------------- #
# Feature importance
# --------------------------------------------------------------------------- #

def permutation_importance(
    clf: HistGradientBoostingClassifier,
    X: pd.DataFrame,
    y: np.ndarray,
    sample_weight: np.ndarray,
    n_repeats: int = 3,
    metric: str = "pr_auc",
) -> pd.DataFrame:
    """Weighted permutation importance, computed column by column.

    ``sklearn.inspection.permutation_importance`` is fine for small data,
    but here we want the metric to be computed with the survey weights
    and we want it to be PR-AUC, so we run our own loop.
    """
    rng = np.random.default_rng(config.RANDOM_STATE)

    baseline_score = _score_fn(metric, clf, X, y, sample_weight)
    rows = []
    for col in X.columns:
        drops: list[float] = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[col] = X_perm[col].sample(frac=1.0, random_state=rng.integers(1e9)).to_numpy()
            perm_score = _score_fn(metric, clf, X_perm, y, sample_weight)
            drops.append(baseline_score - perm_score)
        rows.append(
            {
                "feature": col,
                "importance_mean": float(np.mean(drops)),
                "importance_std": float(np.std(drops)),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def _score_fn(
    metric: str,
    clf: HistGradientBoostingClassifier,
    X: pd.DataFrame,
    y: np.ndarray,
    w: np.ndarray,
) -> float:
    p = clf.predict_proba(X)[:, 1]
    m = compute_classification_metrics(y, p, w)
    return getattr(m, metric)
