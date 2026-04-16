"""Train & evaluate the income classifier end to end.

Run:

    python scripts/02_train_classifier.py                  # default params
    python scripts/02_train_classifier.py --tune --trials 30

Pipeline:

1. Load raw data and stratified 80/20 split (target-preserving).
2. Fit the linear baseline (logistic regression, one-hot + standardised).
3. Fit HistGradientBoostingClassifier with native categorical support.
   Optionally tune via Optuna (``--tune``).
4. Pick the classification threshold on 5-fold out-of-fold predictions so
   the test-set metrics are reported at a threshold never touched by test.
5. Score both models on the held-out test set with survey weights.
6. Save:
   * ``artifacts/models/hgbc.joblib`` + ``logreg.joblib``
   * ``artifacts/metrics/classifier_metrics.json``
   * ``artifacts/figures/clf_roc.png``
   * ``artifacts/figures/clf_pr.png``
   * ``artifacts/figures/clf_calibration.png``
   * ``artifacts/figures/clf_confusion.png``
   * ``artifacts/figures/clf_feature_importance.png``
   * ``artifacts/metrics/classifier_subgroup.csv``
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from census_income import config  # noqa: E402
from census_income.classifier import (  # noqa: E402
    build_logreg_pipeline,
    cross_validate_hgbc,
    fit_hgbc,
    permutation_importance,
    tune_hgbc,
    DEFAULT_HGBC_PARAMS,
)
from census_income.data import load_and_split, describe_split  # noqa: E402
from census_income.evaluation import (  # noqa: E402
    compute_classification_metrics,
    pick_threshold_f1,
    plot_calibration,
    plot_confusion,
    plot_pr,
    plot_roc,
    subgroup_report,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tune", action="store_true", help="Run Optuna tuning on HGBC")
    p.add_argument("--trials", type=int, default=30, help="n Optuna trials")
    p.add_argument(
        "--no-logreg",
        action="store_true",
        help="Skip the logistic regression baseline (faster iteration)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()

    print("[clf] loading data...")
    ds = load_and_split()
    print(describe_split(ds).to_string(index=False))
    X_tr, y_tr, w_tr = ds.train.X, ds.train.y, ds.train.w
    X_te, y_te, w_te = ds.test.X, ds.test.y, ds.test.w

    # ------------------------------------------------------------------ #
    # 1. Linear baseline
    # ------------------------------------------------------------------ #
    baseline_metrics: dict | None = None
    logreg = None
    if not args.no_logreg:
        print("[clf] fitting logistic-regression baseline...")
        logreg = build_logreg_pipeline()
        logreg.fit(X_tr, y_tr, clf__sample_weight=w_tr)
        p_te_lr = logreg.predict_proba(X_te)[:, 1]
        baseline_metrics = compute_classification_metrics(y_te, p_te_lr, w_te).to_dict()
        print(
            f"[clf] logreg test ROC={baseline_metrics['roc_auc']:.4f}  "
            f"PR={baseline_metrics['pr_auc']:.4f}  "
            f"F1@tuned={baseline_metrics['f1']:.4f}"
        )

    # ------------------------------------------------------------------ #
    # 2. HGBC — optionally tuned
    # ------------------------------------------------------------------ #
    if args.tune:
        print(f"[clf] tuning HGBC with {args.trials} Optuna trials (PR-AUC)...")
        best = tune_hgbc(X_tr, y_tr, w_tr, n_trials=args.trials)
        best_value = best.pop("_best_value_pr_auc")
        print(f"[clf] best CV PR-AUC = {best_value:.4f}")
        print("[clf] best params:")
        for k, v in best.items():
            print(f"    {k}: {v}")
        hgbc_params = best
    else:
        print("[clf] using default HGBC params (pass --tune to run Optuna)")
        hgbc_params = dict(DEFAULT_HGBC_PARAMS)

    # 3. CV on train for OOF threshold selection
    print("[clf] 5-fold CV on train to get OOF predictions + threshold...")
    cv = cross_validate_hgbc(X_tr, y_tr, w_tr, n_splits=config.N_CV_FOLDS, params=hgbc_params)
    print(
        f"[clf] CV mean — ROC={cv.mean_metrics['roc_auc']:.4f} ± "
        f"{cv.std_metrics['roc_auc']:.4f}  "
        f"PR={cv.mean_metrics['pr_auc']:.4f} ± {cv.std_metrics['pr_auc']:.4f}"
    )

    tuned_threshold = pick_threshold_f1(y_tr, cv.oof_scores, w_tr)
    print(f"[clf] OOF-tuned threshold (max F1): {tuned_threshold:.3f}")

    # 4. Final fit on all train data
    print("[clf] fitting final HGBC on full training split...")
    hgbc = fit_hgbc(X_tr, y_tr, w_tr, params=hgbc_params)
    p_te_hgbc = hgbc.predict_proba(X_te)[:, 1]
    hgbc_metrics = compute_classification_metrics(
        y_te, p_te_hgbc, w_te, threshold=tuned_threshold
    ).to_dict()
    print(
        f"[clf] HGBC test ROC={hgbc_metrics['roc_auc']:.4f}  "
        f"PR={hgbc_metrics['pr_auc']:.4f}  "
        f"F1={hgbc_metrics['f1']:.4f}  "
        f"Prec@10%={hgbc_metrics['precision_at_top_10pct']:.3f}  "
        f"Lift@10%={hgbc_metrics['lift_at_top_10pct']:.2f}"
    )

    # ------------------------------------------------------------------ #
    # 5. Save figures
    # ------------------------------------------------------------------ #
    curves: dict = {"HistGradientBoosting": (y_te, p_te_hgbc, w_te)}
    if logreg is not None:
        curves["Logistic Regression"] = (y_te, p_te_lr, w_te)

    plot_roc(curves).savefig(config.FIGURES_DIR / "clf_roc.png", dpi=160)
    plt.close("all")
    plot_pr(curves).savefig(config.FIGURES_DIR / "clf_pr.png", dpi=160)
    plt.close("all")
    plot_calibration(y_te, p_te_hgbc, w_te, label="HistGradientBoosting").savefig(
        config.FIGURES_DIR / "clf_calibration.png", dpi=160
    )
    plt.close("all")
    y_hat = (p_te_hgbc >= tuned_threshold).astype(np.int8)
    plot_confusion(y_te, y_hat, w_te).savefig(
        config.FIGURES_DIR / "clf_confusion.png", dpi=160
    )
    plt.close("all")

    # Subgroup audit — honest fairness read-out, not a full audit.
    subgroups = subgroup_report(X_te, y_te, y_hat, p_te_hgbc, w_te)
    subgroups.to_csv(config.METRICS_DIR / "classifier_subgroup.csv", index=False)
    print("[clf] subgroup audit:")
    print(subgroups.to_string(index=False))

    # ------------------------------------------------------------------ #
    # 6. Feature importance (permutation on a subsample for speed)
    # ------------------------------------------------------------------ #
    print("[clf] computing weighted permutation importance (PR-AUC delta)...")
    rng = np.random.default_rng(config.RANDOM_STATE)
    sub_idx = rng.choice(len(X_te), size=min(20_000, len(X_te)), replace=False)
    perm = permutation_importance(
        hgbc,
        X_te.iloc[sub_idx].reset_index(drop=True),
        y_te[sub_idx],
        w_te[sub_idx],
        n_repeats=3,
        metric="pr_auc",
    )
    perm.to_csv(config.METRICS_DIR / "classifier_permutation_importance.csv", index=False)

    top_n = 15
    fig, ax = plt.subplots(figsize=(7, 5.5))
    top = perm.head(top_n).iloc[::-1]
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#3B7DDD")
    ax.set_xlabel("PR-AUC drop when feature is shuffled")
    ax.set_title(f"Top {top_n} permutation-importance features")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "clf_feature_importance.png", dpi=160)
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # 7. Persist artifacts
    # ------------------------------------------------------------------ #
    joblib.dump(hgbc, config.MODELS_DIR / "hgbc.joblib")
    if logreg is not None:
        joblib.dump(logreg, config.MODELS_DIR / "logreg.joblib")

    artefact: dict = {
        "runtime_seconds": round(time.perf_counter() - t0, 1),
        "hgbc_params": {k: v for k, v in hgbc_params.items() if not k.startswith("_")},
        "cv": {
            "mean": cv.mean_metrics,
            "std": cv.std_metrics,
            "fold_metrics": cv.fold_metrics,
        },
        "tuned_threshold": tuned_threshold,
        "test_metrics_hgbc": hgbc_metrics,
        "test_metrics_logreg": baseline_metrics,
    }
    (config.METRICS_DIR / "classifier_metrics.json").write_text(json.dumps(artefact, indent=2, default=float))
    print(f"[clf] wrote {config.METRICS_DIR / 'classifier_metrics.json'}")
    print(f"[clf] done in {artefact['runtime_seconds']}s")


if __name__ == "__main__":
    main()
