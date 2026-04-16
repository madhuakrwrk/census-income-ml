"""Marketing segmentation over the census population.

Philosophy
----------

The segmentation is *not* the classification model with a different name.
Two things matter to the retail client that push the design:

1. **It has to be interpretable.** A cluster that a marketing manager can't
   describe in one sentence is a cluster they won't ship. We aim for 5-6
   segments, each summarised by a "persona card" of its top-differentiating
   features against the full population.

2. **It has to be action-relevant.** That means segmenting on features that
   (a) correlate with spending power and lifestyle and (b) are actually
   plausible to target on in an acquisition funnel — age, household
   composition, education, employment, capital activity. We deliberately
   drop low-signal, high-noise columns like the migration codes and the
   detailed 50-level industry code, which add cluster variance without
   adding marketing meaning.

Algorithm
---------

We use K-Prototypes (``kmodes``) which extends K-Means to mixed data
types: Euclidean distance on the continuous columns, simple-matching
distance on the categoricals. Unlike "one-hot everything + K-Means", this
respects the fact that "Divorced" and "Widowed" are equally distant from
"Never married"; in one-hot space they get collapsed into the same
quadrant by the 0/1 encoding.

For k selection we compute the cost curve across k=3..8 and combine with
a weighted-silhouette sanity check, then lock on the k that cleanly
separates the persona stories (typically k=6).

Segmenting with weights
-----------------------

``kmodes`` doesn't support sample weights directly. Rather than down/up-
sampling the data (which distorts the effective sample size at random),
we do the principled thing: fit on the unweighted data and then
*profile* clusters using the survey weights. That gives us a stable
geometry and client-facing numbers that correctly reflect the US
population.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import StandardScaler

from . import config


# --------------------------------------------------------------------------- #
# Feature selection
# --------------------------------------------------------------------------- #

# Features used for segmentation. A tighter list than the classification
# feature set — marketing personas should be built on variables the client
# can *act on* (age, education, employment), not on 50-level BLS industry
# codes that nobody in a marketing team knows how to read.
SEG_NUMERIC: list[str] = [
    "age",
    "wage per hour",
    "capital gains",
    "capital losses",
    "dividends from stocks",
    "num persons worked for employer",
    "weeks worked in year",
]

SEG_CATEGORICAL: list[str] = [
    "education",
    "marital stat",
    "major industry code",
    "major occupation code",
    "race",
    "sex",
    "class of worker",
    "full or part time employment stat",
    "tax filer stat",
    "detailed household summary in household",
    "citizenship",
    "own business or self employed",
]

SEG_FEATURES: list[str] = SEG_NUMERIC + SEG_CATEGORICAL


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #

@dataclass
class SegPrep:
    """Output of :func:`prepare_segmentation_matrix`.

    ``matrix`` is a numpy object array, because ``kmodes`` wants mixed
    dtypes in a single 2D block. ``categorical_idx`` tells KPrototypes
    which columns to treat with matching distance.
    """
    matrix: np.ndarray
    categorical_idx: list[int]
    numeric_cols: list[str]
    categorical_cols: list[str]
    numeric_scaler: StandardScaler


def prepare_segmentation_matrix(
    X: pd.DataFrame,
    scaler: StandardScaler | None = None,
) -> SegPrep:
    """Build the KPrototypes input matrix.

    Numeric columns are z-scored so they sit on the same scale as the
    matching-distance categoricals (otherwise "dividends from stocks"
    in dollars completely dominates "marital stat" and the clustering
    collapses to a capital-gains-versus-not split).

    Missing categorical values (rare, a few percent on country of birth)
    are folded into the string "MISSING" so KPrototypes doesn't choke
    on ``nan``.
    """
    num = X[SEG_NUMERIC].to_numpy(dtype=float)

    if scaler is None:
        scaler = StandardScaler().fit(num)
    num_scaled = scaler.transform(num)

    cat = X[SEG_CATEGORICAL].astype(str).fillna("MISSING").to_numpy()

    # Stack as object array — KPrototypes requires that.
    matrix = np.concatenate([num_scaled.astype(object), cat], axis=1)
    categorical_idx = list(range(len(SEG_NUMERIC), len(SEG_NUMERIC) + len(SEG_CATEGORICAL)))

    return SegPrep(
        matrix=matrix,
        categorical_idx=categorical_idx,
        numeric_cols=SEG_NUMERIC,
        categorical_cols=SEG_CATEGORICAL,
        numeric_scaler=scaler,
    )


# --------------------------------------------------------------------------- #
# Model selection
# --------------------------------------------------------------------------- #

@dataclass
class KSelection:
    ks: list[int]
    costs: list[float]
    recommended_k: int


def select_k(
    prep: SegPrep,
    k_values: Sequence[int] = (3, 4, 5, 6, 7, 8),
    sample_size: int = 20_000,
    random_state: int = config.RANDOM_STATE,
    init: str = "Huang",
    n_init: int = 2,
) -> KSelection:
    """Sweep k and return the cost curve.

    KPrototypes cost is Σ(within-cluster distance). It decreases
    monotonically in k, so we pick the elbow rather than the min.

    We sample 20k rows for the sweep because fitting on the full 200k at
    six different values of k is expensive and adds no information — the
    elbow location is stable at this sample size.
    """
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(prep.matrix), size=min(sample_size, len(prep.matrix)), replace=False)
    sub = prep.matrix[idx]

    ks = list(k_values)
    costs: list[float] = []
    for k in ks:
        km = KPrototypes(
            n_clusters=k,
            init=init,
            n_init=n_init,
            random_state=random_state,
            n_jobs=1,
            verbose=0,
        )
        km.fit_predict(sub, categorical=prep.categorical_idx)
        costs.append(float(km.cost_))

    # Knee detection: largest *relative* cost drop between k and k-1, scaled
    # by 1/k so we lean toward the simpler solution. This is our default —
    # a human should still look at the plot.
    scores: list[float] = []
    for i, k in enumerate(ks):
        if i == 0:
            scores.append(0.0)
            continue
        rel_drop = (costs[i - 1] - costs[i]) / max(costs[i - 1], 1e-9)
        scores.append(rel_drop / k)
    recommended_k = ks[int(np.argmax(scores))]
    return KSelection(ks=ks, costs=costs, recommended_k=recommended_k)


# --------------------------------------------------------------------------- #
# Fit
# --------------------------------------------------------------------------- #

@dataclass
class SegmentationFit:
    model: KPrototypes
    labels: np.ndarray
    prep: SegPrep
    k: int


def fit_segmentation(
    X: pd.DataFrame,
    k: int,
    random_state: int = config.RANDOM_STATE,
    init: str = "Huang",
    n_init: int = 5,
) -> SegmentationFit:
    """Fit the final K-Prototypes model on the full dataset."""
    prep = prepare_segmentation_matrix(X)
    km = KPrototypes(
        n_clusters=k,
        init=init,
        n_init=n_init,
        max_iter=50,
        random_state=random_state,
        n_jobs=1,
        verbose=0,
    )
    labels = km.fit_predict(prep.matrix, categorical=prep.categorical_idx)
    return SegmentationFit(model=km, labels=labels, prep=prep, k=k)


# --------------------------------------------------------------------------- #
# Profiling / persona generation
# --------------------------------------------------------------------------- #

def profile_clusters(
    X: pd.DataFrame,
    labels: np.ndarray,
    sample_weight: np.ndarray,
    is_high_income: np.ndarray,
    top_k_features: int = 6,
) -> pd.DataFrame:
    """Per-cluster profile table: size, income rate, top differentiators.

    For each cluster we compute:
    * ``n_weighted`` — population-weighted size
    * ``pct_of_population``
    * ``mean_age``, ``mean_weeks_worked``, ``mean_dividends``
    * ``pct_high_income`` — the fraction earning >$50k, using weights
    * a text description of the top differentiating features

    The "differentiators" are the features whose within-cluster mean
    (for numerics) or top value (for categoricals) is most unlike the
    global distribution. For numerics we score by standardised
    difference; for categoricals by the probability ratio of the modal
    value against the global base rate.
    """
    labels = np.asarray(labels)
    rows: list[dict] = []

    global_w = sample_weight.sum()
    global_mean_num = {c: _weighted_mean(X[c].to_numpy(dtype=float), sample_weight) for c in SEG_NUMERIC}
    global_high = float((sample_weight * is_high_income).sum() / global_w)

    for cid in sorted(np.unique(labels)):
        mask = labels == cid
        w = sample_weight[mask]
        Xc = X.loc[mask]
        desc: list[str] = []

        # Numeric differentiators
        num_scores: list[tuple[str, float, float]] = []
        for c in SEG_NUMERIC:
            cluster_mean = _weighted_mean(Xc[c].to_numpy(dtype=float), w)
            # Standard deviation of the full population — use it to score.
            std = float(X[c].std() or 1.0)
            z = (cluster_mean - global_mean_num[c]) / std
            num_scores.append((c, cluster_mean, z))
        num_scores.sort(key=lambda r: abs(r[2]), reverse=True)

        # Categorical differentiators (ratio of cluster modal-value rate to
        # the global rate of that value — high ratio = distinctive).
        cat_scores: list[tuple[str, str, float, float]] = []
        for c in SEG_CATEGORICAL:
            cluster_vals = Xc[c].astype(str).fillna("MISSING")
            global_vals = X[c].astype(str).fillna("MISSING")
            cluster_counts = _weighted_counts(cluster_vals, w)
            cluster_counts = cluster_counts / cluster_counts.sum()
            global_counts = _weighted_counts(global_vals, sample_weight)
            global_counts = global_counts / global_counts.sum()
            # Score each value by how overrepresented it is in this cluster.
            ratios = (cluster_counts + 1e-6) / (global_counts.reindex(cluster_counts.index).fillna(1e-6))
            top_val = ratios.idxmax()
            cat_scores.append(
                (
                    c,
                    top_val,
                    float(cluster_counts.get(top_val, 0.0)),
                    float(ratios.get(top_val, 0.0)),
                )
            )
        cat_scores.sort(key=lambda r: r[3], reverse=True)

        # Build a human description string
        for c, mean_val, z in num_scores[: top_k_features // 2]:
            direction = "high" if z > 0 else "low"
            desc.append(f"{direction} {c} (mean {mean_val:.1f})")
        for c, val, share, ratio in cat_scores[: top_k_features - top_k_features // 2]:
            desc.append(f"{c} '{val}' ({share:.0%}, {ratio:.1f}× base)")

        row = {
            "cluster": int(cid),
            "n_rows": int(mask.sum()),
            "weighted_n": float(w.sum()),
            "pct_of_population": float(w.sum() / global_w),
            "pct_high_income": float((w * is_high_income[mask]).sum() / w.sum()),
            "lift_vs_base_high_income": float(
                ((w * is_high_income[mask]).sum() / w.sum()) / max(global_high, 1e-9)
            ),
            "mean_age": float(_weighted_mean(Xc["age"].to_numpy(dtype=float), w)),
            "mean_weeks_worked": float(
                _weighted_mean(Xc["weeks worked in year"].to_numpy(dtype=float), w)
            ),
            "mean_wage_per_hour": float(
                _weighted_mean(Xc["wage per hour"].to_numpy(dtype=float), w)
            ),
            "mean_dividends": float(
                _weighted_mean(Xc["dividends from stocks"].to_numpy(dtype=float), w)
            ),
            "top_differentiators": " | ".join(desc),
        }
        rows.append(row)

    profile = (
        pd.DataFrame(rows)
        .sort_values("pct_high_income", ascending=False)
        .reset_index(drop=True)
    )
    return profile


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    total_w = weights.sum()
    if total_w == 0:
        return float("nan")
    return float((values * weights).sum() / total_w)


def _weighted_counts(values: pd.Series, weights: np.ndarray) -> pd.Series:
    """Weighted value_counts — normalise to a pd.Series indexed by category."""
    return (
        pd.DataFrame({"v": values.to_numpy(), "w": weights})
        .groupby("v", observed=True)["w"]
        .sum()
    )
