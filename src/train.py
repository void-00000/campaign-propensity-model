"""
PHASE 4 - Train two models on the identical train/test split.

THE TWO MODELS, AND WHY BOTH

  LOGISTIC REGRESSION - the interpretable baseline.
    Despite the name, it's a CLASSIFIER, not a regression. It draws one
    straight boundary through the feature space and asks which side of it you
    fall on. It learns exactly one number (a "coefficient") per feature, and
    that number is the entire model. You can read it, print it, and defend it
    to a regulator.

  XGBOOST - the main model.
    "eXtreme Gradient Boosting". It builds hundreds of small decision trees in
    sequence, where each new tree focuses on the mistakes the previous ones
    made. Slower, far harder to read, and usually more accurate on tabular data
    like ours.

  WHY TRAIN BOTH? Because "XGBoost got 0.80 AUC" means nothing on its own. Good
  compared to what? The baseline is what turns a number into evidence. If
  XGBoost can't beat a straight line, all its complexity is buying you nothing
  and you should ship the straight line. This is the single most-skipped step in
  beginner ML projects.

RUN IT WITH:
    python src/train.py
"""

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import MODELS
from features import RANDOM_STATE, get_train_test


def train_logistic_regression(X_train, y_train) -> Pipeline:
    """
    Train the interpretable baseline.

    It's wrapped in a Pipeline because Logistic Regression needs two things
    done to the data first, and the Pipeline guarantees they happen correctly:

      1. IMPUTE - fill our deliberate NULLs. Logistic Regression has no concept
         of "missing"; it can only do arithmetic, so a NaN just breaks it.

      2. SCALE - put every feature on the same footing. nr_employed is ~5,000
         while our flags are 0 or 1. Since the model penalises large
         coefficients (regularization), an unscaled feature measured in
         thousands gets an unfairly tiny coefficient and dodges the penalty,
         while a 0/1 flag needs a big coefficient and gets punished for it.
         Without scaling you rank features by their UNITS, not their value.

    The Pipeline matters for correctness, not tidiness: .fit() learns the median
    and the mean from TRAINING data only, and .predict() reuses those stored
    values on the test set. The test set is never peeked at. Compute those
    statistics by hand over the full dataset -- a very easy mistake -- and the
    test data has leaked into training with nothing to warn you.
    """
    print("\n[1/2] Training Logistic Regression ...")

    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000,
            # Our Phase 3 decision: weight each class inversely to its
            # frequency, so a responder counts ~7.9x a non-responder.
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])

    pipe.fit(X_train, y_train)

    train_auc = roc_auc_score(y_train, pipe.predict_proba(X_train)[:, 1])
    print(f"      Training AUC: {train_auc:.4f}")

    return pipe


def train_xgboost(X_train, y_train) -> XGBClassifier:
    """
    Train the main model.

    NOTE WHAT'S MISSING: no imputer, no scaler.
      - Missing values: XGBoost handles them natively. At each split it learns
        which direction to send NaNs, treating "missing" as information rather
        than a hole to plug. For days_since_prev_contact -- NULL for 96% of
        customers because they were never contacted -- that's exactly right.
        "Never contacted" is a fact about the customer, not an absence of data.
      - Scaling: trees ask "is x > 5000?". Rescaling x rescales the threshold
        identically, so the tree is unchanged. Scaling here is harmless and
        pointless.

    That's not laziness, it's the honest difference between the algorithms.
    """
    print("\n[2/2] Training XGBoost ...")

    # ---- Handling imbalance the XGBoost way --------------------------------
    # scale_pos_weight is XGBoost's equivalent of class_weight='balanced'.
    # The standard recipe is (negatives / positives) -- so errors on the rare
    # class are penalised ~7.9x harder. Computed from the TRAINING data only;
    # deriving it from the full dataset would peek at the test set.
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    scale_pos_weight = n_neg / n_pos
    print(f"      scale_pos_weight = {n_neg:,} / {n_pos:,} = {scale_pos_weight:.2f}")

    # ---- Early stopping needs its own validation slice ---------------------
    # WHAT IS EARLY STOPPING? XGBoost adds trees one at a time, and each tree
    # fits the training data a little better. Past a point it stops learning
    # real patterns and starts memorising noise -- that's OVERFITTING, and it
    # makes training scores rise while real-world performance falls.
    #
    # Early stopping catches that: watch performance on data the model isn't
    # training on, and stop when it stops improving. It also means we don't
    # have to guess the right number of trees -- the data decides.
    #
    # CRUCIAL: this validation slice is carved out of TRAINING data, never the
    # test set. If we early-stopped on the test set, we'd be using it to make a
    # modelling decision -- and it would no longer be an honest estimate of
    # unseen performance. The test set gets looked at exactly once, in Phase 5.
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )
    print(f"      Fitting on {len(X_fit):,} rows, "
          f"validating on {len(X_val):,} for early stopping")

    model = XGBClassifier(
        # ---- How many trees, and how fast each one learns ----
        n_estimators=1000,       # an upper limit; early stopping picks the real
                                 # number, so this just needs to be generous
        learning_rate=0.05,      # how much each tree corrects the last. Lower =
                                 # slower but steadier. 0.05 is a safe default
                                 # that trades training time for stability.

        # ---- How complex each tree is allowed to get ----
        max_depth=5,             # max questions asked per tree. Deeper trees
                                 # capture more interactions but memorise faster.
                                 # 5 is a reasonable middle for tabular data.
        min_child_weight=5,      # don't create a split that isolates a tiny
                                 # group -- that's usually noise, not signal.

        # ---- Randomness, which fights overfitting ----
        subsample=0.8,           # each tree sees a random 80% of rows
        colsample_bytree=0.8,    # each tree sees a random 80% of features
                                 # Both stop trees from being clones of each
                                 # other and stop any single feature dominating.

        # ---- Regularization: penalties on complexity ----
        reg_lambda=1.0,          # discourage large leaf weights
        gamma=0.1,               # a split must improve things by at least this
                                 # much to be kept at all

        # ---- Imbalance ----
        scale_pos_weight=scale_pos_weight,

        # ---- Bookkeeping ----
        eval_metric="auc",       # what early stopping watches. AUC, not
                                 # accuracy -- Phase 3 showed why.
        early_stopping_rounds=50,  # stop after 50 rounds with no improvement
        random_state=RANDOM_STATE,
        n_jobs=-1,               # use all CPU cores
    )

    model.fit(
        X_fit,
        y_fit,
        eval_set=[(X_val, y_val)],   # the data early stopping watches
        verbose=False,
    )

    print(f"      Early stopping chose {model.best_iteration + 1} trees "
          f"(of a possible 1000)")
    print(f"      Best validation AUC: {model.best_score:.4f}")

    train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    print(f"      Training AUC: {train_auc:.4f}")

    return model


def main():
    print("=" * 78)
    print("  PHASE 4: TRAINING")
    print("=" * 78)

    print("\nLoading the same train/test split every phase uses ...")
    X_train, X_test, y_train, y_test = get_train_test()
    print(f"  Train: {len(X_train):,} customers ({y_train.sum():,} responders)")
    print(f"  Test : {len(X_test):,} customers ({y_test.sum():,} responders)  "
          f"<- untouched until Phase 5")
    print(f"  Features: {X_train.shape[1]}")

    logreg = train_logistic_regression(X_train, y_train)
    xgb = train_xgboost(X_train, y_train)

    # ---- A quick honesty check ---------------------------------------------
    # Full evaluation is Phase 5. But we do one check now: is the model
    # overfitting? Compare its score on data it trained on against data it
    # didn't. A large gap means it memorised rather than learned.
    print("\n" + "=" * 78)
    print("  OVERFITTING CHECK")
    print("=" * 78)
    print("\n  A model always scores better on data it has seen. The question is")
    print("  HOW MUCH better. A big gap means it memorised instead of learning.\n")

    for name, model in [("Logistic Regression", logreg), ("XGBoost", xgb)]:
        tr = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
        te = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        gap = tr - te
        verdict = "healthy" if gap < 0.03 else "some overfitting" if gap < 0.08 else "OVERFIT"
        print(f"  {name:22s} train {tr:.4f} | test {te:.4f} | "
              f"gap {gap:+.4f}  ({verdict})")

    # ---- Save ---------------------------------------------------------------
    # joblib is the standard way to save a fitted scikit-learn object. It
    # preserves the WHOLE pipeline -- imputer, scaler and model together -- so
    # the dashboard in Phase 7 can't accidentally skip a preprocessing step.
    # Saving a bare model and reimplementing its preprocessing elsewhere is a
    # classic way to make production quietly disagree with your test results.
    joblib.dump(logreg, MODELS / "logistic_regression.joblib")
    joblib.dump(xgb, MODELS / "xgboost.joblib")

    # Save the feature order too. The model takes an unlabelled array of 92
    # numbers; if the dashboard ever builds those columns in a different order,
    # every prediction is silently wrong -- no error, just nonsense. Storing
    # the order lets Phase 7 assert it matches.
    joblib.dump(list(X_train.columns), MODELS / "feature_names.joblib")

    print(f"\nSaved to {MODELS}:")
    print("  logistic_regression.joblib   (full pipeline: impute + scale + model)")
    print("  xgboost.joblib               (the trained booster)")
    print("  feature_names.joblib         (the 92 column names, in order)")
    print("\nNext: python src/evaluate.py  (Phase 5 - the real scoring)")


if __name__ == "__main__":
    main()
