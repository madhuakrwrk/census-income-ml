"""Project-wide configuration: paths, column metadata, feature roles.

The census extract ships with columns in a sidecar file rather than a header
row, so a lot of downstream pain is avoided by centralising *all* column
knowledge here — name, role (target / weight / numeric / categorical), and the
handful of quirks that the raw data has ("?" for NA, a leading space on every
value, "Not in universe" as a meaningful category, etc.).

Importing code should never hard-code a column name. It should reach into
``FEATURES`` / ``NUMERIC_FEATURES`` / ``CATEGORICAL_FEATURES`` / ``TARGET`` /
``WEIGHT`` from this module, so renaming a column is a one-line change here.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

# Resolve paths relative to the repo root so scripts can be invoked from any
# working directory — this matters a lot when the same code runs from a
# notebook cwd, a script, and a CI runner.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_DATA_FILE = RAW_DIR / "census-bureau.data"
RAW_COLUMNS_FILE = RAW_DIR / "census-bureau.columns"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
FIGURES_DIR = ARTIFACTS_DIR / "figures"

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORT_FIG_DIR = REPORTS_DIR / "figures"

for _p in (MODELS_DIR, METRICS_DIR, FIGURES_DIR, REPORT_FIG_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Column metadata
# --------------------------------------------------------------------------- #

# Target and survey weight — these are *not* features.
TARGET = "label"
WEIGHT = "weight"

# Columns that are genuinely continuous. `detailed industry recode` and
# `detailed occupation recode` look numeric in the raw file but are BLS code
# identifiers with no ordinal meaning, so they belong to CATEGORICAL_FEATURES.
NUMERIC_FEATURES: list[str] = [
    "age",
    "wage per hour",
    "capital gains",
    "capital losses",
    "dividends from stocks",
    "num persons worked for employer",
    "weeks worked in year",
]

# Columns that are categorical — either string-valued in the raw data or small
# integer code systems (occupation recode, own business, veterans benefits,
# year). Using native `pandas.Categorical` dtype lets
# HistGradientBoostingClassifier route them through its categorical split path.
CATEGORICAL_FEATURES: list[str] = [
    "class of worker",
    "detailed industry recode",
    "detailed occupation recode",
    "education",
    "enroll in edu inst last wk",
    "marital stat",
    "major industry code",
    "major occupation code",
    "race",
    "hispanic origin",
    "sex",
    "member of a labor union",
    "reason for unemployment",
    "full or part time employment stat",
    "tax filer stat",
    "region of previous residence",
    "state of previous residence",
    "detailed household and family stat",
    "detailed household summary in household",
    "migration code-change in msa",
    "migration code-change in reg",
    "migration code-move within reg",
    "live in this house 1 year ago",
    "migration prev res in sunbelt",
    "family members under 18",
    "country of birth father",
    "country of birth mother",
    "country of birth self",
    "citizenship",
    "own business or self employed",
    "fill inc questionnaire for veteran's admin",
    "veterans benefits",
    "year",
]

FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Sanity check — if someone edits the lists above and breaks the partition,
# surface it the moment this module is imported rather than deep inside a
# training loop.
assert len(set(FEATURES)) == len(FEATURES), "duplicate feature names"
assert TARGET not in FEATURES and WEIGHT not in FEATURES
assert len(FEATURES) == 40, (
    "the project objective mentions 40 demographic/employment variables; "
    f"config currently lists {len(FEATURES)} — recheck NUMERIC vs CATEGORICAL"
)

# --------------------------------------------------------------------------- #
# Feature semantics (used for reporting & persona generation)
# --------------------------------------------------------------------------- #

# Thematic groupings — these make cluster profiling readable and keep the
# feature-importance sections of the report legible instead of a wall of 40
# column names.
FEATURE_GROUPS: dict[str, list[str]] = {
    "demographics": [
        "age",
        "sex",
        "race",
        "hispanic origin",
        "citizenship",
        "country of birth self",
        "country of birth father",
        "country of birth mother",
    ],
    "household": [
        "marital stat",
        "detailed household and family stat",
        "detailed household summary in household",
        "family members under 18",
    ],
    "education": [
        "education",
        "enroll in edu inst last wk",
    ],
    "employment": [
        "class of worker",
        "detailed industry recode",
        "detailed occupation recode",
        "major industry code",
        "major occupation code",
        "full or part time employment stat",
        "member of a labor union",
        "reason for unemployment",
        "num persons worked for employer",
        "weeks worked in year",
        "own business or self employed",
    ],
    "income_capital": [
        "wage per hour",
        "capital gains",
        "capital losses",
        "dividends from stocks",
        "tax filer stat",
    ],
    "migration": [
        "region of previous residence",
        "state of previous residence",
        "migration code-change in msa",
        "migration code-change in reg",
        "migration code-move within reg",
        "live in this house 1 year ago",
        "migration prev res in sunbelt",
    ],
    "other": [
        "fill inc questionnaire for veteran's admin",
        "veterans benefits",
        "year",
    ],
}

# --------------------------------------------------------------------------- #
# Training constants
# --------------------------------------------------------------------------- #

RANDOM_STATE = 42

# 80/20 stratified split. The dataset is large enough (200k rows) and the
# positive class is small enough (~6%) that a single held-out test set is a
# better choice than a smaller, noisier one — we still use 5-fold CV inside
# train for model selection.
TEST_SIZE = 0.20
N_CV_FOLDS = 5

# In the raw file every value has a leading space ("Not in universe" arrives as
# " Not in universe"), and true missing values are the literal string "?".
NA_SENTINEL = "?"
