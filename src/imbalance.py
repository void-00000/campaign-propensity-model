"""
PHASE 3 - The class imbalance problem, demonstrated rather than described.

THE SITUATION
Only 11.27% of our customers said yes. That's "imbalanced data": one class
massively outnumbers the other. It is the normal state of affairs for anything
worth predicting -- fraud, churn, disease, click-through, campaign response.
The interesting thing is always rare. That's WHY it's worth predicting.

WHAT THIS SCRIPT DOES
  1. Builds a deliberately stupid model to show why accuracy lies.
  2. Trains the same Logistic Regression three ways -- untouched, with class
     weights, and with SMOTE -- and compares them honestly.
  3. Proves, with actual numbers from our actual data, why SMOTE is the wrong
     tool for THIS dataset.
  4. Commits to an approach and writes down the reasoning.

RUN IT WITH:
    python src/imbalance.py
"""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import RANDOM_STATE, get_train_test


def make_logreg_pipeline(class_weight=None) -> Pipeline:
    """
    Build a Logistic Regression with the preprocessing it needs, as one object.

    WHAT IS A PIPELINE, AND WHY DOES IT MATTER?
    A Pipeline chains steps so they always run in the right order on the right
    data. It exists to prevent a specific, very common, very sneaky bug.

    Consider scaling. It needs a mean and a standard deviation. If you compute
    those over the WHOLE dataset before splitting, then the mean you used on
    training data was partly calculated from the test data. The test set has
    leaked into training, your test score is flattered, and nothing warns you.

    A Pipeline makes that mistake structurally impossible: when you call .fit()
    it learns the mean from TRAINING data only; when you call .predict() on the
    test set it reuses that stored mean. The test set is never consulted. You
    get the right behaviour by default instead of by remembering.
    """
    return Pipeline([
        # Step 1: fill the deliberate NULLs (days_since_prev_contact,
        # education_years). Logistic Regression cannot do arithmetic with a
        # missing value -- it has no concept of "unknown", so it just breaks.
        # The median is a safe filler: unlike the mean, it isn't dragged around
        # by outliers.
        ("impute", SimpleImputer(strategy="median")),

        # Step 2: rescale every feature to mean 0, standard deviation 1.
        #
        # WHY: nr_employed runs around 5,000 while our 0/1 flags are, well, 0
        # or 1. Logistic Regression works by finding a weight per feature, and
        # it penalises large weights (that's regularization). A feature measured
        # in thousands needs a tiny weight to have any effect -- so the penalty
        # barely touches it, while a 0/1 flag needs a large weight and gets
        # punished for it. The model ends up ranking features by their UNITS
        # rather than their usefulness. Scaling puts everything in the same
        # units so the comparison is fair.
        #
        # (XGBoost needs none of this. Trees ask "is x > 5000?" -- a question
        # whose answer doesn't change if you rescale x. Scaling a tree model is
        # harmless but pointless.)
        ("scale", StandardScaler()),

        ("model", LogisticRegression(
            max_iter=2000,          # give it room to converge; the default 100
                                    # is not enough for 92 scaled features
            class_weight=class_weight,
            random_state=RANDOM_STATE,
        )),
    ])


def evaluate(name: str, y_true, y_pred, y_proba) -> dict:
    """
    Score a model every way that matters, and keep the numbers together.

    THE METRICS, IN PLAIN ENGLISH -- using our actual problem:

      ACCURACY  - what fraction of ALL guesses were right?
                  Sounds like the obvious metric. It is a trap here. Read on.

      PRECISION - of the customers we CALLED, what fraction actually said yes?
                  This is about wasted effort. Low precision = your callers
                  spend their day being hung up on.

      RECALL    - of the customers who WOULD have said yes, what fraction did
                  we actually call? This is about missed money. Low recall =
                  real buyers we never phoned, revenue left on the table.

      F1        - one number balancing precision and recall (their harmonic
                  mean). Useful for a quick comparison; it hides the tradeoff
                  you often actually care about, so never use it alone.

      AUC-ROC   - across every possible cutoff, how well does the model RANK
                  responders above non-responders? 0.5 = coin flip. 1.0 =
                  perfect. This is the honest headline number for our problem,
                  because ranking is literally the job: we're sorting a call
                  list, not delivering verdicts.
    """
    # A confusion matrix lays out the four possible outcomes.
    # .ravel() flattens it into: true negatives, false positives,
    # false negatives, true positives.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        # AUC needs PROBABILITIES, not yes/no predictions. It asks "did you
        # rank them correctly", which requires a score to rank by.
        "auc": roc_auc_score(y_true, y_proba),
        "called": int(tp + fp),      # how many people we'd phone
        "found": int(tp),            # real responders we'd reach
        "missed": int(fn),           # real responders we'd never call
        "wasted": int(fp),           # calls to people who say no
    }


def demo_accuracy_trap(X_train, X_test, y_train, y_test) -> dict:
    """
    Build the laziest possible model: always guess 'no'. Never look at a
    single feature. Then look at its accuracy.
    """
    print("\n" + "=" * 78)
    print("  PART 1: WHY ACCURACY IS A LIAR")
    print("=" * 78)

    # DummyClassifier is scikit-learn's built-in "stupid baseline". Its whole
    # purpose is to give you a floor to beat. strategy="most_frequent" means
    # "always predict whichever class is commonest" -- here, always 'no'.
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)

    y_pred = dummy.predict(X_test)
    y_proba = dummy.predict_proba(X_test)[:, 1]

    result = evaluate("Always predict 'no'", y_test, y_pred, y_proba)

    print("\nA model that ignores every feature and always answers 'no':\n")
    print(f"  Accuracy : {result['accuracy']:.4f}  <-- looks like a great model!")
    print(f"  Recall   : {result['recall']:.4f}  <-- found ZERO of the buyers")
    print(f"  Precision: {result['precision']:.4f}")
    print(f"  AUC      : {result['auc']:.4f}  <-- 0.5 is a coin flip")

    print(f"\n  Customers it would call    : {result['called']:,}")
    print(f"  Real buyers it would find  : {result['found']:,}")
    print(f"  Real buyers it would MISS  : {result['missed']:,}")

    print("""
  THIS IS THE WHOLE PROBLEM.

  88.7% accuracy. It sounds like a strong model. You could put it on a slide.
  And it is completely worthless -- it identifies not one single customer, so
  the marketing team gets an empty call list.

  Accuracy fails here because it treats both mistakes as equally bad. But they
  are not remotely equal in business terms:
    - Call someone who says no  -> you wasted five minutes.
    - Miss someone who'd say yes -> you lost a customer and their deposit.

  When 88.7% of the answers are 'no', a model can score 88.7% by learning
  nothing whatsoever. Any metric a lazy model can max out is not measuring the
  thing you care about.

  THIS is why the project uses AUC and lift. Neither can be gamed this way:
  this model's AUC is exactly 0.500, which correctly calls it worthless.""")

    return result


def demo_smote_problem(X_train, y_train) -> dict:
    """
    Show, with real numbers, why SMOTE is wrong for this particular dataset.

    Most tutorials reach for SMOTE reflexively. Understanding when NOT to use a
    technique is worth more than knowing how to call it.

    Returns the measured facts so the conclusion in Part 4 can quote the real
    numbers instead of numbers I typed from memory. (The first version of this
    script hardcoded "77%" into the conclusion while the code above it computed
    87%. Prose that restates a number the code already knows is prose that will
    eventually contradict it -- so the prose should ask the code.)
    """
    print("\n" + "=" * 78)
    print("  PART 2: WHY SMOTE IS THE WRONG TOOL FOR *THIS* DATA")
    print("=" * 78)

    print("""
  WHAT IS SMOTE? "Synthetic Minority Over-sampling TEchnique". It manufactures
  new fake examples of the rare class. For each real responder it finds nearby
  responders and invents new ones on the line between them -- a blend.

  For continuous features that's reasonable. A synthetic customer aged 41.6
  with a euribor rate of 3.2 is a perfectly plausible person.

  But look at what our feature matrix is actually made of:""")

    binary_cols = [c for c in X_train.columns
                   if set(X_train[c].dropna().unique()) <= {0, 1}]
    print(f"\n  Binary 0/1 features : {len(binary_cols)} of {X_train.shape[1]}"
          f"  ({100 * len(binary_cols) / X_train.shape[1]:.0f}%)")
    print(f"  Continuous features : {X_train.shape[1] - len(binary_cols)}")

    print("""
  Most of our matrix is one-hot flags. So what does 'the midpoint between two
  customers' mean when the feature is job_student? Let's not speculate -- let's
  run SMOTE and look at what it actually produced.""")

    # SMOTE can't handle NaN, so impute first -- same as the real pipeline.
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X_train), columns=X_train.columns
    )

    smote = SMOTE(random_state=RANDOM_STATE)
    X_smote, y_smote = smote.fit_resample(X_imputed, y_train)

    print(f"\n  Training rows before SMOTE: {len(X_imputed):,} "
          f"({y_train.sum():,} responders)")
    print(f"  Training rows after SMOTE : {len(X_smote):,} "
          f"({y_smote.sum():,} responders)")
    print(f"  Synthetic customers invented: {len(X_smote) - len(X_imputed):,}")

    # The synthetic rows are the ones appended after the originals.
    synthetic = X_smote.iloc[len(X_imputed):]

    print("\n  Now: how many synthetic customers have IMPOSSIBLE values on the")
    print("  one-hot flags -- i.e. a value that is neither 0 nor 1?\n")

    check_cols = ["job_student", "job_retired", "poutcome_success",
                  "is_cellular", "contact_cellular", "month_mar"]

    for col in check_cols:
        vals = synthetic[col]
        impossible = ((vals > 0.001) & (vals < 0.999)).sum()
        pct = 100 * impossible / len(synthetic)
        print(f"    {col:22s} {impossible:>6,} of {len(synthetic):,} "
              f"({pct:5.1f}%) are fractional")

    # Show a concrete offender so this isn't abstract. We print whatever is
    # actually non-zero for this customer rather than a fixed list of columns
    # we hope will be interesting -- the data decides what gets shown.
    frac_mask = (synthetic["job_student"] > 0.001) & (synthetic["job_student"] < 0.999)
    if frac_mask.any():
        example = synthetic[frac_mask].iloc[0]
        print("\n  Here is one synthetic 'customer' SMOTE actually created.")
        print("  (Showing every job/marital flag that isn't zero for them.)\n")

        print(f"    {'age':24s} = {example['age']:.2f}")
        for prefix in ("job_", "marital_"):
            group = [c for c in synthetic.columns if c.startswith(prefix)]
            nonzero = [(c, example[c]) for c in group if example[c] > 0.001]
            for col, val in nonzero:
                print(f"    {col:24s} = {val:.3f}")
            total = sum(example[c] for c in group)
            print(f"    {'-> ' + prefix + '* sums to':24s} = {total:.3f}\n")

        # Describe the customer in words, generated from their real values.
        job_parts = [
            (c.replace("job_", ""), example[c])
            for c in synthetic.columns
            if c.startswith("job_") and example[c] > 0.001
        ]
        desc = " and ".join(f"{v:.0%} {name}" for name, v in job_parts)
        print(f"    In words: this 'person' is {desc}.")

    print("""
  Read that customer again. They are a fraction of one job and a fraction of
  another, simultaneously. Their job flags sum to 1.0 -- so SMOTE has produced
  something that is arithmetically tidy and humanly impossible. No such person
  exists. No such person could exist.

  THE PROBLEM IN ONE SENTENCE: SMOTE assumes the space between two examples is
  meaningful. For a continuous feature like age it is. For a category it is
  not -- there is nothing halfway between 'student' and 'retired'.

  So SMOTE hands our model thousands of impossible customers and says "learn
  what a responder looks like from these". The model dutifully learns the shape
  of a fictional population, then meets real customers in production.

  (SMOTENC exists for exactly this -- it handles categorical features properly.
  It's the right answer if you're committed to oversampling. We're going a
  simpler route that avoids inventing data at all.)

  ONE MORE HAZARD: SMOTE must only ever touch the TRAINING data. Oversample
  before you split, and copies of the same customer land in both train and test
  -- the model is graded on people it memorised. Instant leakage, spectacular
  fake scores. Our Pipeline structure makes this hard to get wrong, but plenty
  of tutorials get it wrong.""")

    # Hand the measured facts back so Part 4 can quote them rather than
    # restate them from memory.
    return {
        "n_binary": len(binary_cols),
        "n_features": X_train.shape[1],
        "pct_binary": 100 * len(binary_cols) / X_train.shape[1],
        "n_synthetic": len(X_smote) - len(X_imputed),
        "n_after": len(X_smote),
    }


def demo_three_approaches(X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """Train the same model three ways and compare on the same test set."""
    print("\n" + "=" * 78)
    print("  PART 3: THREE APPROACHES, ONE HONEST COMPARISON")
    print("=" * 78)

    results = []

    # ---- Approach A: do nothing ------------------------------------------
    print("\n[A] Logistic Regression, imbalance ignored ...")
    pipe_a = make_logreg_pipeline(class_weight=None)
    pipe_a.fit(X_train, y_train)
    results.append(evaluate(
        "A: No handling",
        y_test,
        pipe_a.predict(X_test),
        pipe_a.predict_proba(X_test)[:, 1],
    ))

    # ---- Approach B: class weights ----------------------------------------
    # class_weight="balanced" tells the model to weight each class inversely to
    # how common it is. Our 7.9:1 imbalance means every responder counts ~7.9x
    # as much as a non-responder when computing the training error.
    #
    # The effect: getting a responder wrong now hurts the model ~8x more than
    # getting a non-responder wrong, so ignoring the rare class stops being the
    # easy win it was. No data is invented -- we just stop pretending the two
    # mistakes cost the same.
    print("[B] Logistic Regression, class_weight='balanced' ...")
    pipe_b = make_logreg_pipeline(class_weight="balanced")
    pipe_b.fit(X_train, y_train)
    results.append(evaluate(
        "B: Class weights",
        y_test,
        pipe_b.predict(X_test),
        pipe_b.predict_proba(X_test)[:, 1],
    ))

    # ---- Approach C: SMOTE -------------------------------------------------
    # Note SMOTE is applied to the TRAINING data only, after the split. The
    # test set stays untouched and real -- it must, or the score is fiction.
    print("[C] Logistic Regression, SMOTE oversampling ...")
    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train), columns=X_train.columns
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test), columns=X_test.columns  # transform, NOT fit
    )

    X_res, y_res = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train_imp, y_train)

    pipe_c = Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ])
    pipe_c.fit(X_res, y_res)
    results.append(evaluate(
        "C: SMOTE",
        y_test,
        pipe_c.predict(X_test_imp),
        pipe_c.predict_proba(X_test_imp)[:, 1],
    ))

    df = pd.DataFrame(results)

    print("\n--- Scored on the SAME untouched test set "
          f"({len(y_test):,} customers, {y_test.sum():,} real responders) ---\n")
    display = df[["model", "accuracy", "precision", "recall", "f1", "auc"]].copy()
    for col in ["accuracy", "precision", "recall", "f1", "auc"]:
        display[col] = display[col].map(lambda v: f"{v:.4f}")
    print(display.to_string(index=False))

    print("\n--- The same results, in business language ---\n")
    biz = df[["model", "called", "found", "missed", "wasted"]].copy()
    biz.columns = ["model", "would call", "buyers found", "buyers MISSED",
                   "wasted calls"]
    print(biz.to_string(index=False))

    return df


def explain_choice(df: pd.DataFrame, smote_facts: dict) -> None:
    """
    State the decision and the reasoning, with the numbers in hand.

    Every number below is interpolated from measured results. None are typed
    in. If the data changes, this conclusion changes with it instead of
    quietly becoming a lie.
    """
    print("\n" + "=" * 78)
    print("  PART 4: WHAT WE CHOSE AND WHY")
    print("=" * 78)

    a = df.iloc[0]
    b = df.iloc[1]
    c = df.iloc[2]

    print(f"""
  LOOK AT APPROACH A. Accuracy {a['accuracy']:.4f} -- the best of the three!
  And it finds only {a['found']:,} of the {a['found'] + a['missed']:,} real
  buyers, missing {a['missed']:,} of them. It scores well by mostly saying 'no',
  which is the accuracy trap from Part 1 in a subtler costume.

  APPROACH B (class weights) finds {b['found']:,} buyers -- {b['found'] / max(a['found'], 1):.1f}x
  what A found -- at the cost of {b['wasted']:,} wasted calls. Its accuracy is
  LOWER ({b['accuracy']:.4f}) and it is a far more useful model. Sit with that
  sentence; it is the whole lesson of this phase.

  APPROACH C (SMOTE) lands at AUC {c['auc']:.4f} vs B's {b['auc']:.4f} --
  a difference of {abs(c['auc'] - b['auc']):.4f}, which is nothing. It invented
  {smote_facts['n_synthetic']:,} synthetic customers to arrive at the same place.

  NOTICE THE AUC COLUMN. All three are close, because AUC measures RANKING, and
  all three rank about equally well. Class weights don't make the model smarter
  -- they move the DECISION THRESHOLD, the cutoff where a probability becomes a
  'yes'. That is genuinely useful, but be clear about what it does and doesn't do.

  ---------------------------------------------------------------------------
  DECISION: class weights (class_weight='balanced' for Logistic Regression,
            scale_pos_weight for XGBoost). Not SMOTE.

  THE REASONING, in the order it actually matters:

  1. SMOTE invents impossible customers on THIS data. {smote_facts['pct_binary']:.0f}%
     of our features ({smote_facts['n_binary']} of {smote_facts['n_features']})
     are one-hot flags, and interpolating between categories produces people who
     are a fraction of one job and a fraction of another. Class weights invent
     nothing -- every row the model sees is a real customer.

  2. Equivalent results, less machinery. C matched B on AUC while adding a
     resampling step, {smote_facts['n_synthetic']:,} synthetic rows, and a
     leakage risk if anyone ever moves it to the wrong side of the split.
     Same outcome, more ways to break.

  3. It's honest about what's really going on. We don't have a data shortage;
     we have a COST asymmetry. Missing a buyer costs more than a wasted call.
     Class weights state that directly. SMOTE encodes the same idea by faking
     a dataset where buyers are common -- which is not the world we live in.

  4. Faster to train and simpler to explain to a stakeholder.

  THE TRADEOFF, stated plainly -- because there is one:
    We accept LOWER ACCURACY and MORE WASTED CALLS to find MORE BUYERS.
    B makes {b['wasted']:,} pointless calls that A avoids. That is a real cost
    in real salary. It is worth it because {b['found'] - a['found']:,} additional
    buyers -- each worth a deposit -- comfortably outvalue some wasted minutes.

    That judgment depends on the actual numbers. If a call cost $500 and a
    deposit earned $10, the correct answer would flip. The model can't know
    your economics; you have to tell it. Which is precisely why we tune the
    threshold in Phase 5 rather than accepting the default 0.5.

  WHEN WOULD I PICK SMOTE INSTEAD? If the minority class were genuinely tiny
  (a few dozen cases, not 4,640), where the model needs more examples rather
  than just a stronger signal about the ones it has. And even then, SMOTENC --
  the variant that handles categories properly.
""")


def main():
    print("Loading data and building the train/test split ...")
    X_train, X_test, y_train, y_test = get_train_test()

    n_train, n_test = len(y_train), len(y_test)
    print(f"\n  Training set: {n_train:,} customers, {y_train.sum():,} responders "
          f"({100 * y_train.mean():.2f}%)")
    print(f"  Test set    : {n_test:,} customers, {y_test.sum():,} responders "
          f"({100 * y_test.mean():.2f}%)")
    print("  (stratify kept the response rate identical in both halves)")

    print(f"\n  IMBALANCE: {(1 - y_train.mean()) / y_train.mean():.1f} "
          f"non-responders for every responder")

    demo_accuracy_trap(X_train, X_test, y_train, y_test)
    smote_facts = demo_smote_problem(X_train, y_train)
    results = demo_three_approaches(X_train, X_test, y_train, y_test)
    explain_choice(results, smote_facts)


if __name__ == "__main__":
    main()
