"""Fit and profile the marketing segmentation.

Run:

    python scripts/03_run_segmentation.py --k 6

Pipeline:

1. Load the full dataset (the segmentation uses all rows because there is
   no leakage concern — it's unsupervised and the classification model
   never sees it this way).
2. Sweep k in [3..8] to pick a sensible cluster count.
3. Fit K-Prototypes with k = recommended (or --k if given).
4. Profile each cluster with survey-weighted statistics and build a
   "persona card" table.
5. Emit three figures:
   * the cost curve for k selection
   * a bar chart of high-income rate by segment
   * a 2D PCA projection of the numerics, coloured by segment
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
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from census_income import config  # noqa: E402
from census_income.data import load_raw  # noqa: E402
from census_income.segmentation import (  # noqa: E402
    SEG_NUMERIC,
    fit_segmentation,
    prepare_segmentation_matrix,
    profile_clusters,
    select_k,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--k",
        type=int,
        default=None,
        help="Cluster count. Default: pick from the k-sweep elbow.",
    )
    p.add_argument(
        "--sample-for-sweep",
        type=int,
        default=20_000,
        help="Rows sampled for the k-sweep (full data used for final fit).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()

    print("[seg] loading data...")
    df = load_raw()
    X = df[config.FEATURES]
    w = df[config.WEIGHT].to_numpy()
    y = df["is_high_income"].to_numpy()

    # ---------------------------------------------------------------- #
    # 1. k sweep
    # ---------------------------------------------------------------- #
    print("[seg] preparing segmentation matrix...")
    prep = prepare_segmentation_matrix(X)

    print(f"[seg] sweeping k on {args.sample_for_sweep} sampled rows...")
    selection = select_k(prep, sample_size=args.sample_for_sweep)
    print(f"[seg] k candidates: {selection.ks}")
    print(f"[seg] costs: {['%.1f' % c for c in selection.costs]}")
    print(f"[seg] recommended k (elbow heuristic): {selection.recommended_k}")

    k = args.k if args.k is not None else selection.recommended_k
    print(f"[seg] using k={k}")

    # ---------------------------------------------------------------- #
    # 2. Final fit
    # ---------------------------------------------------------------- #
    print(f"[seg] fitting K-Prototypes on full data (n={len(X):,})...")
    fit = fit_segmentation(X, k=k, n_init=5)

    # ---------------------------------------------------------------- #
    # 3. Profile
    # ---------------------------------------------------------------- #
    profile = profile_clusters(
        X=X,
        labels=fit.labels,
        sample_weight=w,
        is_high_income=y,
    )
    print("[seg] cluster profile:")
    with pd.option_context("display.max_colwidth", 120, "display.width", 160):
        print(profile.to_string(index=False))

    profile.to_csv(config.METRICS_DIR / "segmentation_profile.csv", index=False)
    (config.METRICS_DIR / "segmentation_profile.json").write_text(
        profile.to_json(orient="records", indent=2)
    )

    # ---------------------------------------------------------------- #
    # 4. Plots
    # ---------------------------------------------------------------- #
    # k sweep plot
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(selection.ks, selection.costs, "o-", color="#3B7DDD")
    ax.axvline(selection.recommended_k, ls="--", c="grey", label=f"elbow @ k={selection.recommended_k}")
    ax.set_xlabel("k")
    ax.set_ylabel("K-Prototypes cost")
    ax.set_title("Cluster cost vs k (lower is tighter)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "seg_k_sweep.png", dpi=160)
    plt.close(fig)

    # income rate by cluster
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    order = profile.sort_values("pct_high_income", ascending=False)
    bars = ax.bar(
        [f"C{int(c)}" for c in order["cluster"]],
        order["pct_high_income"].to_numpy(),
        color="#3B7DDD",
    )
    base = float((w * y).sum() / w.sum())
    ax.axhline(base, c="grey", ls="--", lw=1, label=f"population base rate ({base:.1%})")
    ax.set_ylabel("P(income >$50k)")
    ax.set_title("High-income rate by segment")
    ax.legend()
    for b, pct, n in zip(bars, order["pct_high_income"], order["pct_of_population"]):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.005,
            f"{pct:.1%}\n({n:.0%} of pop)",
            ha="center",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "seg_income_by_cluster.png", dpi=160)
    plt.close(fig)

    # 2D projection — PCA on standardised numerics only, coloured by cluster.
    # A full mixed-space projection would be more faithful but harder to
    # read; the point of this chart is *separation*, not faithfulness.
    numeric = prep.matrix[:, : len(SEG_NUMERIC)].astype(float)
    pca = PCA(n_components=2, random_state=config.RANDOM_STATE)
    proj = pca.fit_transform(numeric)

    rng = np.random.default_rng(config.RANDOM_STATE)
    # Downsample for the scatter so the PNG stays legible and small.
    sample = rng.choice(len(proj), size=min(15_000, len(proj)), replace=False)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    scatter = ax.scatter(
        proj[sample, 0],
        proj[sample, 1],
        c=fit.labels[sample],
        cmap="tab10",
        s=6,
        alpha=0.55,
        linewidths=0,
    )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
    ax.set_title("Segments in 2D PCA of numeric features")
    legend = ax.legend(
        *scatter.legend_elements(),
        title="segment",
        loc="best",
        fontsize=8,
    )
    ax.add_artist(legend)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "seg_pca_scatter.png", dpi=160)
    plt.close(fig)

    # ---------------------------------------------------------------- #
    # 5. Persist model + summary
    # ---------------------------------------------------------------- #
    joblib.dump(
        {
            "model": fit.model,
            "k": fit.k,
            "prep_scaler": fit.prep.numeric_scaler,
            "numeric_cols": fit.prep.numeric_cols,
            "categorical_cols": fit.prep.categorical_cols,
        },
        config.MODELS_DIR / "segmentation.joblib",
    )

    (config.METRICS_DIR / "segmentation_summary.json").write_text(
        json.dumps(
            {
                "runtime_seconds": round(time.perf_counter() - t0, 1),
                "k": fit.k,
                "recommended_k": selection.recommended_k,
                "k_sweep_costs": dict(zip(selection.ks, selection.costs)),
                "n_rows": int(len(X)),
            },
            indent=2,
        )
    )
    print(f"[seg] done in {round(time.perf_counter() - t0, 1)}s")


if __name__ == "__main__":
    main()
