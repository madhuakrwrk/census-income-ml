"""Exploratory data analysis for the census income extract.

Run:

    python scripts/01_run_eda.py

Writes:

* ``artifacts/metrics/eda_summary.json`` — row count, weighted/unweighted
  positive rate, per-feature missingness and cardinality.
* ``artifacts/figures/eda_*.png`` — target distribution by age, education,
  sex, race, plus the survey-weight distribution.

Everything is population-weighted where it makes sense, because that's
the number the client cares about.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from census_income import config  # noqa: E402
from census_income.data import load_raw  # noqa: E402


def main() -> None:
    print("[eda] loading raw data...")
    df = load_raw()
    print(f"[eda] shape: {df.shape}")

    y = df["is_high_income"].to_numpy()
    w = df[config.WEIGHT].to_numpy()

    summary: dict = {
        "n_rows": int(len(df)),
        "n_features": len(config.FEATURES),
        "positive_rate_unweighted": float(y.mean()),
        "positive_rate_weighted": float((w * y).sum() / w.sum()),
        "weight_sum": float(w.sum()),
        "weight_min": float(w.min()),
        "weight_max": float(w.max()),
        "weight_median": float(np.median(w)),
        "features_with_NA": {},
        "features_with_NIU": {},
        "cardinality": {},
    }

    for c in config.FEATURES:
        s = df[c]
        summary["cardinality"][c] = int(s.nunique(dropna=False))
        na_rate = float(s.isna().mean())
        if na_rate > 0:
            summary["features_with_NA"][c] = na_rate
        if s.dtype.name == "category":
            vals = s.astype(str)
            niu = float((vals == "Not in universe").mean())
            if niu > 0:
                summary["features_with_NIU"][c] = niu

    # Save the JSON summary
    out_json = config.METRICS_DIR / "eda_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"[eda] wrote {out_json}")

    # --- figure: target rate by age bucket --------------------------------
    ages = df["age"].to_numpy()
    bins = np.array([0, 18, 25, 35, 45, 55, 65, 120])
    age_bin = pd.cut(ages, bins=bins, right=False, labels=[
        "<18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"
    ])
    by_age = (
        pd.DataFrame({"age_bin": age_bin, "y": y, "w": w})
        .groupby("age_bin", observed=True)
        .apply(lambda g: (g["y"] * g["w"]).sum() / g["w"].sum(), include_groups=False)
    )

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    by_age.plot(kind="bar", ax=ax, color="#3B7DDD", edgecolor="none")
    ax.set_ylabel("P(income > $50k | age bucket), weighted")
    ax.set_xlabel("")
    ax.set_title("High-income rate by age")
    ax.axhline(summary["positive_rate_weighted"], c="grey", ls="--", lw=1, label="overall")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "eda_income_by_age.png", dpi=160)
    plt.close(fig)

    # --- figure: target rate by education ---------------------------------
    by_edu = (
        pd.DataFrame({"edu": df["education"].astype(str), "y": y, "w": w})
        .groupby("edu", observed=True)
        .apply(lambda g: (g["y"] * g["w"]).sum() / g["w"].sum(), include_groups=False)
        .sort_values(ascending=True)
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    by_edu.plot(kind="barh", ax=ax, color="#3B7DDD", edgecolor="none")
    ax.set_xlabel("P(income > $50k), weighted")
    ax.set_title("High-income rate by education")
    ax.axvline(summary["positive_rate_weighted"], c="grey", ls="--", lw=1)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "eda_income_by_education.png", dpi=160)
    plt.close(fig)

    # --- figure: target rate by sex and race -----------------------------
    def _by_group(col: str):
        return (
            pd.DataFrame({"g": df[col].astype(str), "y": y, "w": w})
            .groupby("g", observed=True)
            .apply(lambda g: (g["y"] * g["w"]).sum() / g["w"].sum(), include_groups=False)
            .sort_values(ascending=True)
        )

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    _by_group("sex").plot(kind="barh", ax=axes[0], color="#3B7DDD")
    axes[0].set_title("Income >$50k by sex")
    axes[0].set_xlabel("P(income > $50k)")
    _by_group("race").plot(kind="barh", ax=axes[1], color="#3B7DDD")
    axes[1].set_title("Income >$50k by race")
    axes[1].set_xlabel("P(income > $50k)")
    for ax in axes:
        ax.axvline(summary["positive_rate_weighted"], c="grey", ls="--", lw=1)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "eda_income_by_sex_race.png", dpi=160)
    plt.close(fig)

    # --- figure: survey-weight distribution -------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.hist(np.log10(w + 1), bins=60, color="#3B7DDD", edgecolor="white")
    ax.set_xlabel("log10(survey weight)")
    ax.set_ylabel("count")
    ax.set_title("Distribution of survey weights (log scale)")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "eda_weight_distribution.png", dpi=160)
    plt.close(fig)

    print("[eda] done.")


if __name__ == "__main__":
    main()
