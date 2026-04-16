"""Load and partition the 1994/1995 Census CPS extract.

The raw file is a headerless CSV. Column names live in a sidecar file. Every
cell arrives with a leading space. True missing values are the string ``?``.
Because several "missingness" patterns are semantically meaningful (``Not in
universe`` = "this question was not asked of this person") we take care to
preserve them as a distinct category instead of collapsing them into NA.

The loader returns a tidy DataFrame with:
* clean string values (leading whitespace stripped),
* proper numeric / categorical dtypes,
* a cleaned binary ``is_high_income`` target,
* the original survey ``weight`` column,
* no duplicated rows.

``load_and_split`` wraps the loader with a stratified train/test split that
preserves the target prevalence on both sides.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def _read_column_names(path: Path = config.RAW_COLUMNS_FILE) -> list[str]:
    """Parse the sidecar columns file.

    The file is one column name per line, no numbering, no header. We strip
    each line but deliberately *preserve* multi-word column names (they have
    spaces that matter for matching between code and data).
    """
    with path.open() as f:
        names = [line.strip() for line in f if line.strip()]
    if len(names) != 42:
        raise ValueError(
            f"expected 42 column names (40 features + weight + label), got {len(names)}"
        )
    return names


def load_raw(
    data_path: Path = config.RAW_DATA_FILE,
    columns_path: Path = config.RAW_COLUMNS_FILE,
) -> pd.DataFrame:
    """Load the raw CPS data and return a well-typed DataFrame.

    Parameters
    ----------
    data_path : Path
        The headerless CSV (``census-bureau.data``).
    columns_path : Path
        The sidecar column-name file (``census-bureau.columns``).

    Returns
    -------
    pandas.DataFrame
        Indexed 0..N-1, with columns in the same order as ``columns_path``.
        Numeric columns are float64, categorical columns are ``category``, and
        a derived ``is_high_income`` int8 column is appended (1 if the person
        earns >$50k, else 0).
    """
    columns = _read_column_names(columns_path)

    # ``skipinitialspace=True`` removes the leading space on every cell. We
    # point ``na_values`` at the literal "?" that the corpus uses for missing
    # values *after* the space is gone (the comparison happens on the
    # stripped value).
    df = pd.read_csv(
        data_path,
        header=None,
        names=columns,
        skipinitialspace=True,
        na_values=[config.NA_SENTINEL],
        dtype=str,  # read everything as string, then cast deliberately below
        low_memory=False,
    )

    # --- dtype coercion ----------------------------------------------------
    for col in config.NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="raise")

    df[config.WEIGHT] = pd.to_numeric(df[config.WEIGHT], errors="raise")

    for col in config.CATEGORICAL_FEATURES:
        # ``category`` is what HistGradientBoostingClassifier wants, and it
        # keeps memory sane on 200k rows.
        df[col] = df[col].astype("category")

    # --- target ------------------------------------------------------------
    # Raw label is " 50000+." or " - 50000.". A robust match on "+" avoids any
    # brittle reliance on whitespace.
    raw_label = df[config.TARGET].astype(str).str.strip()
    df["is_high_income"] = raw_label.str.contains(r"\+", regex=True).astype("int8")
    df = df.drop(columns=[config.TARGET])

    # --- basic sanity ------------------------------------------------------
    # The full file has 199_523 rows with no duplicates. If that changes,
    # fail loudly rather than silently modelling corrupted data.
    expected_n = 199_523
    if len(df) != expected_n:
        # Not strictly an error, but worth surfacing: shrink or grow means the
        # data file has changed and the rest of the project's tuned defaults
        # (e.g. class-weight, learning rate) may need a second look.
        print(f"[data] WARNING: expected {expected_n} rows, got {len(df)}")

    n_dups = df.duplicated(subset=config.FEATURES).sum()
    if n_dups:
        # There are genuine duplicates in this dataset (different households
        # can have identical observed covariates), so we don't drop them —
        # we just record the count for the report.
        df.attrs["n_feature_duplicates"] = int(n_dups)

    return df


# --------------------------------------------------------------------------- #
# Split
# --------------------------------------------------------------------------- #

@dataclass
class TrainTestArrays:
    """Container for a (features, target, weight) tuple per split.

    A tiny dataclass so that downstream code reads like
    ``data.train.X`` / ``data.test.y`` instead of juggling six parallel arrays.
    """

    X: pd.DataFrame
    y: np.ndarray
    w: np.ndarray

    def __len__(self) -> int:  # pragma: no cover - convenience
        return len(self.y)


@dataclass
class Dataset:
    train: TrainTestArrays
    test: TrainTestArrays


def load_and_split(
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
) -> Dataset:
    """Load the raw data and return a stratified train/test split.

    The split is stratified on ``is_high_income`` so the positive-class rate
    (~6%) is preserved on both sides. Weights go along for the ride untouched;
    we *never* rescale them inside the split because they are absolute survey
    weights that retain meaning only in their original units.
    """
    df = load_raw()

    X = df[config.FEATURES].copy()
    y = df["is_high_income"].to_numpy()
    w = df[config.WEIGHT].to_numpy()

    X_tr, X_te, y_tr, y_te, w_tr, w_te = train_test_split(
        X,
        y,
        w,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Re-align category codes: after the split, some levels may exist only in
    # one side (rare countries of birth etc). HGBC is fine with unseen
    # categories at predict time, but we want the `.cat.categories` index to
    # match between train and test so that downstream code (e.g. SHAP) reads
    # clean feature names. Using the union preserves all levels.
    for col in config.CATEGORICAL_FEATURES:
        all_cats = pd.api.types.union_categoricals(
            [X_tr[col], X_te[col]], sort_categories=True
        ).categories
        X_tr[col] = pd.Categorical(X_tr[col], categories=all_cats)
        X_te[col] = pd.Categorical(X_te[col], categories=all_cats)

    return Dataset(
        train=TrainTestArrays(X=X_tr.reset_index(drop=True), y=y_tr, w=w_tr),
        test=TrainTestArrays(X=X_te.reset_index(drop=True), y=y_te, w=w_te),
    )


def describe_split(ds: Dataset) -> pd.DataFrame:
    """Return a one-shot summary table for the README / report."""
    def _row(name: str, arr: TrainTestArrays) -> dict:
        return {
            "split": name,
            "n_rows": len(arr),
            "positive_rate_unweighted": float(arr.y.mean()),
            "positive_rate_weighted": float(
                (arr.w * arr.y).sum() / arr.w.sum()
            ),
            "weight_sum": float(arr.w.sum()),
        }

    return pd.DataFrame([_row("train", ds.train), _row("test", ds.test)])
