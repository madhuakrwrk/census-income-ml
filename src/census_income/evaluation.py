"""Model evaluation utilities.

The evaluation story we want to tell is built around four beats:

1. **Threshold-free discrimination.** ROC AUC and (more importantly for a
   6% base rate) PR AUC, both computed with the survey weights so the
   numbers reflect the actual US population rather than the sampled set.
2. **Business-realistic thresholds.** Marketing cannot ship a probability;
   it needs a decision. We pick the threshold that maximises F1 on the
   validation split, and also report precision@top-k so the client can
   read "if we send mailers to the top 10% most likely high-earners, here
   is what we'd see".
3. **Calibration.** For downstream expected-value math to work, ``P(y=1|x)``
   has to mean something. We report Brier score and a reliability curve.
4. **Fairness sanity check.** The data encodes race and sex, which lets us
   do a very quick subgroup disparity read-out. This isn't a full audit,
   just a flag in case something is obviously off.

All plotting helpers return the ``Figure`` and the caller is responsible
for saving it — that separation keeps the module easy to use both from the
training script and from a notebook.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# --------------------------------------------------------------------------- #
# Metric bundles
# --------------------------------------------------------------------------- #

@dataclass
class ClassificationMetrics:
    """A single flat bag of headline metrics, easy to serialise to JSON."""

    n: int
    weighted_n: float
    positive_rate: float
    weighted_positive_rate: float
    roc_auc: float
    pr_auc: float
    brier: float
    # Threshold-dependent — evaluated at the tuned threshold.
    threshold: float
    precision: float
    recall: float
    f1: float
    # For the client: "if we reach out to the top k% of the scored list…"
    precision_at_top_5pct: float
    precision_at_top_10pct: float
    precision_at_top_20pct: float
    lift_at_top_10pct: float

    def to_dict(self) -> dict:
        return asdict(self)


def pick_threshold_f1(
    y_true: np.ndarray,
    y_score: np.ndarray,
    sample_weight: np.ndarray | None = None,
    grid: np.ndarray | None = None,
) -> float:
    """Pick the probability threshold that maximises weighted F1.

    We do a grid search rather than an analytic PR-curve argmax because the
    curve has tiny cliffs near the top of the score distribution at this
    class imbalance, and F1 computed on a grid with sample weights is both
    more numerically stable and easier to reason about.
    """
    if grid is None:
        grid = np.linspace(0.01, 0.95, 95)

    best_thr, best_f1 = 0.5, -1.0
    for thr in grid:
        y_hat = (y_score >= thr).astype(np.int8)
        f1 = f1_score(y_true, y_hat, sample_weight=sample_weight, zero_division=0)
        if f1 > best_f1:
            best_thr, best_f1 = float(thr), float(f1)
    return best_thr


def _precision_at_top_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k_fraction: float,
    sample_weight: np.ndarray | None = None,
) -> float:
    """Precision among the ``k_fraction`` highest-scoring rows (weighted)."""
    if sample_weight is None:
        sample_weight = np.ones_like(y_true, dtype=float)

    order = np.argsort(-y_score)
    cum_w = np.cumsum(sample_weight[order])
    target_mass = k_fraction * cum_w[-1]
    # ``searchsorted`` returns the first index with cum_w >= target_mass,
    # which is the smallest top-k window that spans the requested fraction
    # of survey weight.
    k = int(np.searchsorted(cum_w, target_mass) + 1)
    k = min(max(k, 1), len(y_true))
    top_idx = order[:k]
    top_w = sample_weight[top_idx]
    return float((top_w * y_true[top_idx]).sum() / top_w.sum())


def compute_classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    sample_weight: np.ndarray | None = None,
    threshold: float | None = None,
) -> ClassificationMetrics:
    """Compute the full headline metric bundle from scores + labels.

    If ``threshold`` is ``None`` we pick it here to maximise weighted F1.
    Leakage note: the caller is expected to have picked the threshold on
    a *validation* fold before passing it in when scoring the test set —
    picking on test would inflate precision/recall/F1.
    """
    if sample_weight is None:
        sample_weight = np.ones_like(y_true, dtype=float)

    thr = pick_threshold_f1(y_true, y_score, sample_weight) if threshold is None else threshold
    y_hat = (y_score >= thr).astype(np.int8)

    # Lift at top decile — a classic marketing number. "How many times more
    # likely is someone in the top 10% of scores to be high-income than a
    # random person in the base population?"
    base_rate = float((sample_weight * y_true).sum() / sample_weight.sum())
    prec_top10 = _precision_at_top_k(y_true, y_score, 0.10, sample_weight)
    lift_top10 = prec_top10 / base_rate if base_rate else 0.0

    return ClassificationMetrics(
        n=int(len(y_true)),
        weighted_n=float(sample_weight.sum()),
        positive_rate=float(y_true.mean()),
        weighted_positive_rate=base_rate,
        roc_auc=float(roc_auc_score(y_true, y_score, sample_weight=sample_weight)),
        pr_auc=float(average_precision_score(y_true, y_score, sample_weight=sample_weight)),
        brier=float(brier_score_loss(y_true, y_score, sample_weight=sample_weight)),
        threshold=thr,
        precision=float(
            precision_score(y_true, y_hat, sample_weight=sample_weight, zero_division=0)
        ),
        recall=float(recall_score(y_true, y_hat, sample_weight=sample_weight, zero_division=0)),
        f1=float(f1_score(y_true, y_hat, sample_weight=sample_weight, zero_division=0)),
        precision_at_top_5pct=_precision_at_top_k(y_true, y_score, 0.05, sample_weight),
        precision_at_top_10pct=prec_top10,
        precision_at_top_20pct=_precision_at_top_k(y_true, y_score, 0.20, sample_weight),
        lift_at_top_10pct=float(lift_top10),
    )


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #

def plot_roc(
    curves: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray | None]],
) -> plt.Figure:
    """Plot one or more ROC curves on the same axes.

    ``curves`` maps a model label to ``(y_true, y_score, sample_weight)``
    so that we can overlay multiple models in a single chart — this is
    how the report shows logistic-regression vs. HistGradientBoosting.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1, label="chance")

    for name, (y_true, y_score, sw) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, y_score, sample_weight=sw)
        auc = roc_auc_score(y_true, y_score, sample_weight=sw)
        ax.plot(fpr, tpr, lw=1.6, label=f"{name} (AUC={auc:.3f})")

    ax.set_xlabel("False positive rate (weighted)")
    ax.set_ylabel("True positive rate (weighted)")
    ax.set_title("ROC — income >$50k classifier")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_pr(
    curves: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray | None]],
) -> plt.Figure:
    """Plot one or more precision-recall curves.

    PR is the honest story at a 6% base rate; ROC can look impressive even
    when actual precision at any operating point is weak.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (y_true, y_score, sw) in curves.items():
        prec, rec, _ = precision_recall_curve(y_true, y_score, sample_weight=sw)
        ap = average_precision_score(y_true, y_score, sample_weight=sw)
        ax.plot(rec, prec, lw=1.6, label=f"{name} (AP={ap:.3f})")

    # A dashed baseline at the positive class prevalence — the "do nothing"
    # line that any honest model has to beat.
    base = float((list(curves.values())[0][0]).mean())
    ax.axhline(base, ls="--", c="grey", lw=1, label=f"base rate ({base:.3f})")
    ax.set_xlabel("Recall (weighted)")
    ax.set_ylabel("Precision (weighted)")
    ax.set_title("Precision-Recall — income >$50k classifier")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_calibration(
    y_true: np.ndarray,
    y_score: np.ndarray,
    sample_weight: np.ndarray | None = None,
    n_bins: int = 15,
    label: str = "model",
) -> plt.Figure:
    """Reliability diagram.

    ``calibration_curve`` doesn't natively take sample weights, so we do
    the weighted version by hand — important because the survey weights
    change the effective bin sizes by 3-4× in places.
    """
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1, label="perfect")

    if sample_weight is None:
        frac_pos, mean_pred = calibration_curve(y_true, y_score, n_bins=n_bins, strategy="quantile")
    else:
        # Quantile bins on the weighted score distribution.
        order = np.argsort(y_score)
        y_sorted = y_true[order]
        s_sorted = y_score[order]
        w_sorted = sample_weight[order]
        cum = np.cumsum(w_sorted)
        edges_mass = np.linspace(0, cum[-1], n_bins + 1)
        bin_ids = np.searchsorted(cum, edges_mass[1:-1])
        bins = np.split(np.arange(len(y_sorted)), bin_ids)
        mean_pred, frac_pos = [], []
        for b in bins:
            if len(b) == 0:
                continue
            w = w_sorted[b]
            mean_pred.append((s_sorted[b] * w).sum() / w.sum())
            frac_pos.append((y_sorted[b] * w).sum() / w.sum())
        mean_pred, frac_pos = np.array(mean_pred), np.array(frac_pos)

    ax.plot(mean_pred, frac_pos, "o-", lw=1.6, label=label)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive rate")
    ax.set_title("Calibration — reliability diagram")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> plt.Figure:
    """Weighted confusion matrix as counts, annotated with row-normalised rates."""
    cm = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1e-9)

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i,
                f"{int(cm[i, j]):,}\n({cm_norm[i, j]:.1%})",
                ha="center", va="center",
                color="white" if cm_norm[i, j] > 0.5 else "black",
                fontsize=10,
            )
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["<=50k", ">50k"])
    ax.set_yticklabels(["<=50k", ">50k"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (weighted)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Subgroup audit
# --------------------------------------------------------------------------- #

def subgroup_report(
    X: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    sample_weight: np.ndarray,
    groups: tuple[str, ...] = ("sex", "race"),
) -> pd.DataFrame:
    """Compute per-subgroup positive rates, precision, recall, and AUC.

    Not a full fairness audit — this is the "is anything obviously wrong"
    check that a sensible data scientist does before handing a model to
    marketing. A disparity here should trigger a deeper audit, not an
    automatic block.
    """
    rows: list[dict] = []
    for g in groups:
        vals = X[g].astype(str)
        for v in sorted(vals.unique()):
            mask = (vals == v).to_numpy()
            if mask.sum() < 50:
                continue
            w = sample_weight[mask]
            yt = y_true[mask]
            yp = y_pred[mask]
            ys = y_score[mask]
            # roc_auc fails if the subgroup is single-class; guard it.
            try:
                auc = roc_auc_score(yt, ys, sample_weight=w)
            except ValueError:
                auc = np.nan
            rows.append(
                {
                    "group": g,
                    "value": v,
                    "n": int(mask.sum()),
                    "weighted_n": float(w.sum()),
                    "pos_rate": float((w * yt).sum() / w.sum()),
                    "pred_pos_rate": float((w * yp).sum() / w.sum()),
                    "precision": float(
                        precision_score(yt, yp, sample_weight=w, zero_division=0)
                    ),
                    "recall": float(
                        recall_score(yt, yp, sample_weight=w, zero_division=0)
                    ),
                    "roc_auc": auc,
                }
            )
    return pd.DataFrame(rows)
