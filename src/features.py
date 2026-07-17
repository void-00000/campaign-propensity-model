"""
PHASE 2b - Turn the SQL feature view into a model-ready matrix.

WHAT'S LEFT TO DO AFTER SQL?
The database gave us clean, decoded, bucketed features. But scikit-learn and
XGBoost cannot read the word "student" -- they only do arithmetic. This script
does the last mile: turn text categories into numbers, drop features we proved
are redundant, and hand back a matrix.

WHY THIS IS A MODULE, NOT A SCRIPT
Phase 4 (training) and Phase 7 (the dashboard) both need the exact same feature
matrix, built the exact same way. If each built its own, they would drift apart
and the dashboard would silently score customers differently than the model was
trained to expect. So the logic lives here once, and both import it.

RUN IT DIRECTLY to inspect the output:
    python src/features.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_PROCESSED, get_engine

# The seed for every random operation in this project.
#
# WHY A FIXED SEED? Splitting data into train/test involves random choices.
# Random means different every run -- which would mean your AUC changes each
# time you train, and you could never tell a real improvement from luck.
# Fixing the seed makes the randomness REPEATABLE: same split, same numbers,
# every run, on any machine. That's what "reproducible" means in practice, and
# it's why the number on your resume will still be true tomorrow.
#
# 42 is a convention, not magic. Any fixed integer works.
RANDOM_STATE = 42

# Share of data held back for testing. 20% of 41,188 is ~8,238 customers,
# containing ~928 responders -- enough that our metrics aren't noise.
TEST_SIZE = 0.2

# ---------------------------------------------------------------------------
# FEATURES WE DROP, AND WHY
# ---------------------------------------------------------------------------
# The EDA correlation check caught something I got wrong in the SQL view.
#
# I built euribor_vs_avg as "euribor3m / average euribor". But the average is a
# CONSTANT -- the same number for every row. Dividing a column by a constant
# just rescales it; it does not add one bit of new information. The correlation
# came back at exactly r = 1.000, which is the mathematical proof: these two
# columns are the same column wearing different units.
#
# Same story for confidence_vs_avg (subtract a constant) and employment_vs_avg
# (subtract a constant). All three are perfectly redundant.
#
# They are still useful to HUMANS -- "1.2x the typical rate" is easier to read
# than "4.3", and the dashboard will use them. But feeding a model two copies of
# one feature is pure downside: it destabilises Logistic Regression's
# coefficients (the model can't decide which twin gets the credit, so it splits
# it arbitrarily and both look weak) and it dilutes tree-based importance
# scores across duplicates.
#
# The lesson worth taking to the interview: a "ratio feature" only earns its
# place if the denominator VARIES per row. Ratio against a global constant is
# a rescale, not a feature. is_low_rate_period survives because a threshold is
# genuinely non-linear -- it's not a rescale, it's a step.
REDUNDANT_FEATURES = [
    "euribor_vs_avg",       # r = 1.000 with euribor3m
    "confidence_vs_avg",    # r = 1.000 with cons_conf_idx
    "employment_vs_avg",    # r = 1.000 with nr_employed
]

# Text columns that need converting to numbers via one-hot encoding.
CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "credit_default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
    "recency_bucket",
    "campaign_intensity_bucket",
    "quarter",
    "age_band",
]

TARGET = "target"


def load_feature_view() -> pd.DataFrame:
    """Read the SQL view built in Phase 1."""
    return pd.read_sql("SELECT * FROM bank_marketing_features;", get_engine())


def build_feature_matrix(df: pd.DataFrame | None = None):
    """
    Return (X, y) ready for scikit-learn.

    X = the features (a DataFrame of numbers only)
    y = the target  (a Series of 1s and 0s)

    Those names are a near-universal convention in ML code: capital X because
    it's a 2-D matrix, lowercase y because it's a 1-D vector.
    """
    if df is None:
        df = load_feature_view()

    # ---- 1. Separate the answer from the questions -------------------------
    # The target must NEVER end up inside X. If the model can see the answer
    # among its inputs it will simply read it off and score 100%, having
    # learned nothing. This is the most basic form of leakage, and it's an easy
    # accident: one stray column and your results are a fantasy.
    y = df[TARGET].copy()
    X = df.drop(columns=[TARGET])

    # ---- 2. Drop the mathematically redundant features ---------------------
    X = X.drop(columns=REDUNDANT_FEATURES)

    # ---- 3. One-hot encode the text columns --------------------------------
    # WHAT IS ONE-HOT ENCODING?
    # A model can't read "job = student". So we replace the one text column
    # with one 0/1 column PER category:
    #
    #   job              job_student  job_retired  job_admin.  ...
    #   student     ->        1            0           0
    #   retired     ->        0            1           0
    #
    # Exactly one column is "hot" (1) per row -- hence the name.
    #
    # WHY NOT JUST NUMBER THEM (student=1, retired=2, admin=3)?
    # Because that invents an order that doesn't exist. The model would infer
    # that admin > retired > student, and that retired is the *average* of
    # student and admin. All meaningless. One-hot refuses to imply any ranking.
    #
    # (Where a real order DOES exist we exploit it -- that's exactly what
    # education_years does in the SQL view, mapping basic.4y -> 4, degree -> 16.
    # Order should be encoded when it's real and never invented when it isn't.)
    X = pd.get_dummies(
        X,
        columns=CATEGORICAL_FEATURES,
        # dtype=int8: get_dummies defaults to True/False booleans. Small
        # integers are what both sklearn and XGBoost want, and int8 uses an
        # eighth of the memory of a default int64 for columns that only ever
        # hold 0 or 1.
        dtype="int8",
        # drop_first=False: we keep every category column.
        #
        # The argument for dropping one: with all k columns, any one is exactly
        # 1-minus-the-sum-of-the-others -- perfectly redundant ("the dummy
        # variable trap"), which breaks plain Logistic Regression.
        #
        # The argument for keeping all k, which wins here: our Logistic
        # Regression is REGULARIZED (Phase 4), which handles the redundancy
        # fine. And keeping every column means each category gets its own
        # readable coefficient and its own SHAP value, instead of one category
        # vanishing into the intercept where we can't see it. Interpretability
        # is a headline goal of this project, so we pay the small cost.
        drop_first=False,
    )

    # ---- 4. Force everything numeric ---------------------------------------
    # A defensive check, not decoration. If a text column ever sneaks through
    # -- a new category appears, a name gets misspelled in the list above --
    # we want to fail loudly HERE, with a clear message, rather than get a
    # confusing error from deep inside sklearn ten minutes later.
    non_numeric = X.select_dtypes(exclude=["number"]).columns.tolist()
    if non_numeric:
        raise ValueError(
            f"These columns are still non-numeric and would break training: "
            f"{non_numeric}. Add them to CATEGORICAL_FEATURES."
        )

    return X, y


def get_train_test(random_state: int = RANDOM_STATE):
    """
    Build the features and split them into a training set and a test set.

    Returns (X_train, X_test, y_train, y_test).

    WHY SPLIT AT ALL?
    A model that has seen an example can just memorise it. Scoring a model on
    data it trained on tells you how good its memory is, not whether it learned
    anything general. So we hide 20% of customers from training entirely and
    only score on those. The test set stands in for "customers we haven't
    called yet" -- the situation the model actually faces in production.

    EVERY PHASE FROM HERE CALLS THIS FUNCTION.
    Phases 3, 4, 5, 6 and 7 must all use the identical split. If Phase 4 split
    the data differently than Phase 5, comparing their numbers would be
    meaningless -- you'd be comparing two models on two different exams. One
    function, one seed, one split.
    """
    X, y = build_feature_matrix()

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=random_state,
        # stratify=y is ESSENTIAL with imbalanced data. It forces the split to
        # preserve the 11.27% response rate in BOTH halves.
        #
        # Without it, the split is random -- and with only 4,640 positives, an
        # unlucky draw could hand the test set 9% responders and the training
        # set 12%. Then you're training on one problem and grading on a
        # slightly different one, and the difference shows up as a mysterious
        # score gap you'll waste a day chasing.
        stratify=y,
    )


def main():
    print("Loading feature view from PostgreSQL ...")
    df = load_feature_view()
    print(f"  SQL view: {len(df):,} rows x {len(df.columns)} columns\n")

    X, y = build_feature_matrix(df)

    print(f"Dropped {len(REDUNDANT_FEATURES)} mathematically redundant features:")
    for f in REDUNDANT_FEATURES:
        print(f"  - {f}")

    print(f"\nOne-hot encoded {len(CATEGORICAL_FEATURES)} categorical columns.")
    print(f"\nFINAL FEATURE MATRIX: {X.shape[0]:,} rows x {X.shape[1]} features")
    print(f"TARGET: {y.sum():,} responders / {len(y):,} customers "
          f"({100 * y.mean():.2f}%)")

    # The project requires at least 15 features. Assert it rather than trust it.
    assert X.shape[1] >= 15, f"Only {X.shape[1]} features - need at least 15!"
    print(f"\nRequirement check: {X.shape[1]} features >= 15 minimum. PASS")

    # ---- Print the full feature list ---------------------------------------
    print("\n" + "=" * 70)
    print(f"  ALL {X.shape[1]} FEATURES")
    print("=" * 70)

    # Group by origin so the list is readable rather than an alphabet soup.
    engineered = [
        "was_contacted_before", "days_since_prev_contact", "prev_success",
        "prev_failure", "total_contacts", "prior_contact_ratio",
        "contact_fatigue", "is_cellular", "is_high_season",
        "is_life_stage_edge", "has_any_loan", "education_years",
        "is_low_rate_period",
    ]
    raw_numeric = [
        "age", "campaign", "previous", "emp_var_rate", "cons_price_idx",
        "cons_conf_idx", "euribor3m", "nr_employed",
    ]

    print("\n-- Engineered in SQL (numeric/flags) --")
    for i, col in enumerate(engineered, 1):
        print(f"  {i:2d}. {col}")

    print("\n-- Raw numeric passthrough --")
    for i, col in enumerate(raw_numeric, len(engineered) + 1):
        print(f"  {i:2d}. {col}")

    print("\n-- One-hot encoded categories --")
    onehot = [c for c in X.columns if c not in engineered + raw_numeric]
    for i, col in enumerate(onehot, len(engineered) + len(raw_numeric) + 1):
        print(f"  {i:2d}. {col}")

    # ---- Missing values that survive on purpose ----------------------------
    print("\n--- NULLs remaining (deliberate) ---")
    missing = X.isna().sum()
    missing = missing[missing > 0]
    for col, count in missing.items():
        print(f"  {col:26s} {count:>6,} ({100 * count / len(X):.2f}%)")
    print("\n  These stay NULL on purpose. XGBoost handles missing values")
    print("  natively -- it learns which way to send them at each split.")
    print("  Logistic Regression can't, so Phase 4 imputes them for that model")
    print("  only, inside a Pipeline. Each model gets what it needs.")

    # ---- Save ---------------------------------------------------------------
    out = DATA_PROCESSED / "model_features.csv"
    X.assign(target=y).to_csv(out, index=False)
    print(f"\nSaved feature matrix to: {out}")
    print("(CSV so you can open and inspect it. The pipeline itself rebuilds")
    print(" from PostgreSQL, so this file is a convenience, not a dependency.)")


if __name__ == "__main__":
    main()
