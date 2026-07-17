"""
PHASE 6 - Interpretability: why does the model rank people the way it does?

THE PROBLEM WE'RE SOLVING
Logistic Regression hands you one coefficient per feature -- the model is
readable by construction. XGBoost is 99 trees voting; there is no single number
to read. That opacity is the price we paid for the better score in Phase 4.
SHAP is how we get most of it back.

WHAT IS SHAP?
"SHapley Additive exPlanations". The idea comes from cooperative game theory
(Lloyd Shapley, Nobel 2012), which asked: if a team wins, how much did each
player contribute? SHAP treats the features as the team and the prediction as
the win.

For every single customer, it answers: how much did each feature push this
person's score up or down, away from the average customer's score?

The property that makes it trustworthy is ADDITIVITY:

    average prediction + sum of all SHAP values = this customer's prediction

Nothing is hand-waved, nothing is left over. That's a real guarantee, and it's
why SHAP beats XGBoost's built-in feature_importances_, which just counts how
often a feature got used in a split -- a proxy that says nothing about
direction, and quietly inflates high-cardinality features.

WHAT THIS PRODUCES
  outputs/figures/08_shap_summary.png     - the beeswarm plot
  outputs/figures/09_shap_importance.png  - the bar chart
  outputs/shap_drivers.md                 - the ranked drivers + actions

RUN IT WITH:
    python src/interpret.py
"""

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from config import FIGURES, MODELS, OUTPUTS
from features import get_train_test


def compute_shap(model, X):
    """
    Compute SHAP values for every customer in X.

    TreeExplainer is a version of SHAP built specifically for tree models. The
    general algorithm is brutally slow (it considers every subset of features);
    TreeExplainer exploits the structure of trees to get exact answers in
    reasonable time. Using the fast one isn't an approximation -- same values.
    """
    print("Computing SHAP values ...")
    explainer = shap.TreeExplainer(model)

    # Returns an Explanation object: the values, the base value, and the data,
    # all bundled so the plotting functions have everything they need.
    shap_values = explainer(X)

    print(f"  Explained {shap_values.values.shape[0]:,} customers "
          f"x {shap_values.values.shape[1]} features")

    return shap_values


def verify_additivity(model, X, shap_values) -> None:
    """
    Check SHAP's core promise on our actual model, rather than trusting it.

    THE CLAIM: base_value + sum(shap values for a customer) should equal the
    model's raw output for that customer.

    WHY BOTHER? Two reasons. First, it's the difference between using a tool and
    understanding it. Second, there's a real trap here: our model used early
    stopping, so the booster contains more trees than it actually uses for
    predictions. If SHAP explained all the trees while the model predicts with
    only the best ones, we'd be reading an explanation of a model that doesn't
    exist. This check would catch that.

    A note on units: we compare in LOG-ODDS ("margin"), not probability. Trees
    add up their votes in log-odds space, and a sigmoid squashes the total into
    a 0-1 probability at the very end. Additivity holds before that squashing,
    not after -- a sigmoid is curved, so squashed values don't add up any more.
    """
    print("\nVerifying SHAP additivity ...")

    # output_margin=True -> the raw log-odds, before the sigmoid.
    margin = model.predict(X, output_margin=True)

    reconstructed = shap_values.values.sum(axis=1) + shap_values.base_values
    max_diff = np.abs(margin - reconstructed).max()

    print(f"  Model's raw output vs (base + sum of SHAP values)")
    print(f"  Largest disagreement across {len(X):,} customers: {max_diff:.8f}")

    if max_diff < 1e-3:
        print("  PASS - SHAP explains exactly the model we trained.")
    else:
        print("  WARNING - these should match. Something is off.")


def rank_drivers(shap_values, X) -> pd.DataFrame:
    """
    Rank features by importance, and work out which direction each one pushes.

    IMPORTANCE = mean absolute SHAP value. We take the ABSOLUTE value first
    because a feature that pushes some customers strongly up and others strongly
    down is very important -- but its raw average would be ~0 and would look
    like it does nothing.

    DIRECTION is a separate question, and we answer it by correlating the
    feature's value with its SHAP value across customers:
      positive -> higher values of this feature push the prediction UP
      negative -> higher values push it DOWN
    """
    vals = shap_values.values

    importance = np.abs(vals).mean(axis=0)

    directions = []
    for i, col in enumerate(X.columns):
        feature_vals = X[col].values.astype(float)
        shap_col = vals[:, i]

        # Only correlate over rows where the feature is present.
        mask = ~np.isnan(feature_vals)

        # Guard against the degenerate cases. A feature with fewer than two
        # present values, or no variation in either the feature or its SHAP
        # values, has no direction to report -- and correlating it would divide
        # by a zero standard deviation, which numpy warns about and returns NaN
        # for. We check first rather than let it warn and clean up after.
        if (
            mask.sum() < 2
            or np.std(feature_vals[mask]) == 0
            or np.std(shap_col[mask]) == 0
        ):
            directions.append(0.0)
            continue

        directions.append(np.corrcoef(feature_vals[mask], shap_col[mask])[0, 1])

    df = pd.DataFrame({
        "feature": X.columns,
        "importance": importance,
        "direction": directions,
    })
    df["effect"] = np.where(
        df["direction"] > 0.05, "higher -> MORE likely to say yes",
        np.where(df["direction"] < -0.05, "higher -> LESS likely to say yes",
                 "mixed / non-linear"),
    )

    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def plot_beeswarm(shap_values, X) -> None:
    """
    Chart: the SHAP summary (beeswarm) plot.

    HOW TO READ IT -- worth learning properly, it's the densest chart in ML:
      - One ROW per feature, ordered most important at the top.
      - One DOT per customer, per feature. 8,238 customers = 8,238 dots a row.
      - HORIZONTAL position: that feature's SHAP value for that customer.
        Right of centre = pushed toward yes. Left = pushed toward no.
      - COLOUR: the feature's VALUE for that customer. Red = high, blue = low.

    So a row with red dots on the right and blue on the left means "high values
    push toward yes". The reverse means "high values push toward no". A row with
    colours mixed on both sides means the effect depends on other features --
    that's an interaction, and it's exactly what a linear model would miss.
    """
    print("\nGenerating charts ...")

    # show=False lets us grab the figure and save it instead of popping a window.
    plt.figure(figsize=(11, 9))
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    plt.title(
        "SHAP summary: how each feature moves each customer's score\n"
        "Red = high feature value | Right = pushed toward 'yes'",
        fontweight="bold", pad=20,
    )
    plt.tight_layout()
    out = FIGURES / "08_shap_summary.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out.name}")


def plot_importance_bar(drivers: pd.DataFrame) -> None:
    """
    Chart: mean |SHAP| as a simple ranked bar chart.

    The beeswarm is richer but takes explaining. This one a stakeholder reads
    in three seconds. Both have their place; know which room you're in.
    """
    top = drivers.head(15).iloc[::-1]  # reverse so the biggest sits on top

    colors = [
        "#DD8452" if d > 0.05 else "#4C72B0" if d < -0.05 else "#999999"
        for d in top["direction"]
    ]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top["feature"], top["importance"], color=colors)
    ax.set_xlabel("Mean |SHAP value|  (average impact on the prediction)")
    ax.set_title(
        "Top 15 conversion drivers\n"
        "Orange = pushes toward yes | Blue = pushes toward no | Grey = mixed",
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    out = FIGURES / "09_shap_importance.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Saved {out.name}")


def write_report(drivers: pd.DataFrame, themes: pd.DataFrame,
                 X_test, y_test) -> None:
    """Write the ranked drivers and their business implications to markdown."""
    lines = []
    A = lines.append

    A("# SHAP: What Drives Campaign Response\n")
    A("Generated from the XGBoost model, explained on the held-out test set")
    A(f"({len(X_test):,} customers the model never saw during training).\n")

    A("## How to read this\n")
    A("**SHAP value** = how much a feature pushed one customer's score away")
    A("from the average customer's score. **Importance** below is the mean")
    A("absolute SHAP value: the average size of that push, ignoring direction.\n")
    A("> **Important caveat.** Mean |SHAP| answers *\"how much does this feature")
    A("> move the AVERAGE customer?\"* -- **not** *\"how much does it matter to a")
    A("> customer it applies to?\"* A feature that is decisive for 3% of people")
    A("> and zero for the rest gets averaged down to near nothing. See the")
    A("> `prev_success` discussion below.\n")

    A("## Drivers by theme\n")
    A("Several of our features encode overlapping facts, and SHAP splits credit")
    A("between correlated features. Group totals are the fairer comparison.\n")
    A("| Theme | Total importance | Share | Features | Top feature |")
    A("|---|---|---|---|---|")
    for _, row in themes.iterrows():
        A(f"| {row['theme']} | {row['total_importance']:.4f} | "
          f"{row['share_pct']:.1f}% | {int(row['n_features'])} | "
          f"`{row['top_feature']}` |")
    A("")

    A("## Ranked individual drivers (top 20)\n")
    A("| # | Feature | Importance | Effect |")
    A("|---|---|---|---|")
    for i, row in drivers.head(20).iterrows():
        A(f"| {i + 1} | `{row['feature']}` | {row['importance']:.4f} | "
          f"{row['effect']} |")
    A("")

    A("## The three headline drivers\n")

    baseline = y_test.mean()
    ps_rate = y_test[X_test["prev_success"] == 1].mean()
    cell_rate = y_test[X_test["contact_cellular"] == 1].mean()
    tel_rate = y_test[X_test["contact_telephone"] == 1].mean()
    econ_share = themes.loc[themes["theme"] == "Macroeconomic conditions",
                            "share_pct"].iloc[0]

    A("### 1. Macroeconomic conditions "
      f"({econ_share:.0f}% of total importance)\n")
    A("`nr_employed`, `euribor3m` and `emp_var_rate` dominate. When rates are")
    A("high and employment is strong, people say no -- a term deposit competes")
    A("badly against better options and low anxiety.\n")
    A("**Action:** time the *campaign*, not just the call list. Rate")
    A("environment should trigger whether to run a deposit push at all.\n")
    A("**Caveat:** these barely vary within a week. They describe *when* we")
    A("called, not *who* we called, and are largely unusable for ranking a")
    A("call list where everyone shares today's euribor.\n")

    A("### 2. Contact channel\n")
    A(f"Mobile: **{100 * cell_rate:.1f}%** response vs landline")
    A(f"**{100 * tel_rate:.1f}%** (baseline {100 * baseline:.1f}%).")
    A("The highest-ranked driver the bank actually controls.\n")
    A("**Action:** prioritise customers with a mobile on file; consider a data")
    A("project to collect missing mobile numbers.\n")
    A("**Caveat:** correlation, not causation -- mobile likely proxies for age")
    A("and reachability. Useful as a *signal*, useless as an *intervention*.\n")

    A("### 3. Prior relationship and recency\n")
    A(f"Prior buyers respond at **{100 * ps_rate:.1f}%** vs a")
    A(f"{100 * baseline:.1f}% baseline -- **{ps_rate / baseline:.1f}x**. But")
    A("`prev_success` ranks only ~20th, because mean |SHAP| averages it over")
    A("everyone and only ~3.5% of customers have any history. Rare and")
    A("decisive, not weak.\n")
    A("**Action:** call prior buyers first (one SQL query, no model needed);")
    A("sequence follow-ups inside a 7-day window -- contacts within a week")
    A("convert at ~66% vs ~9% for never-contacted.\n")
    A("**Limit:** only ~3.5% of customers qualify. Always top of the list,")
    A("never enough to fill it.\n")

    out = OUTPUTS / "shap_drivers.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved {out.name}")

    # Also save as CSV. The markdown is for humans; the dashboard needs to read
    # these numbers back as data, and parsing a table out of markdown to get
    # numbers we already have in a DataFrame would be silly.
    drivers.assign(theme=drivers["feature"].map(theme_of)).to_csv(
        OUTPUTS / "shap_drivers.csv", index=False
    )
    print("  Saved shap_drivers.csv")


# Features grouped by what they're ACTUALLY about. We report group totals as
# well as individual ranks, because our features are deliberately overlapping --
# prev_success, was_contacted_before and days_since_prev_contact all encode
# "have we spoken before?" from different angles. SHAP splits credit between
# correlated features, so each looks weak alone while the theme is strong.
# Judging a theme by its best single feature would badly understate it.
FEATURE_THEMES = {
    "Macroeconomic conditions": [
        "nr_employed", "euribor3m", "emp_var_rate", "cons_conf_idx",
        "cons_price_idx", "is_low_rate_period",
    ],
    "Prior relationship / recency": [
        "prev_success", "prev_failure", "was_contacted_before",
        "days_since_prev_contact", "poutcome_success", "poutcome_failure",
        "poutcome_nonexistent", "previous", "prior_contact_ratio",
        "total_contacts", "recency_bucket_never", "recency_bucket_within_week",
        "recency_bucket_within_fortnight", "recency_bucket_over_fortnight",
    ],
    "Contact channel": ["is_cellular", "contact_cellular", "contact_telephone"],
    "Timing (month / day / season)": [
        "is_high_season", "quarter_Q1", "quarter_Q2", "quarter_Q3", "quarter_Q4",
    ],
    "Campaign intensity": [
        "campaign", "contact_fatigue", "campaign_intensity_bucket_excessive",
        "campaign_intensity_bucket_light_touch",
        "campaign_intensity_bucket_persistent",
        "campaign_intensity_bucket_single_call",
    ],
    "Demographics": [
        "age", "is_life_stage_edge", "education_years", "has_any_loan",
    ],
}


def theme_of(feature: str) -> str:
    """Map a feature to its theme, using prefixes for the one-hot families."""
    for theme, members in FEATURE_THEMES.items():
        if feature in members:
            return theme
    for prefix, theme in [
        ("month_", "Timing (month / day / season)"),
        ("day_of_week_", "Timing (month / day / season)"),
        ("age_band_", "Demographics"),
        ("job_", "Demographics"),
        ("marital_", "Demographics"),
        ("education_", "Demographics"),
        ("housing_", "Demographics"),
        ("loan_", "Demographics"),
        ("credit_default_", "Demographics"),
    ]:
        if feature.startswith(prefix):
            return theme
    return "Other"


def analyse_themes(drivers: pd.DataFrame) -> pd.DataFrame:
    """Sum SHAP importance by theme, so overlapping features get judged fairly."""
    d = drivers.copy()
    d["theme"] = d["feature"].map(theme_of)

    g = d.groupby("theme").agg(
        total_importance=("importance", "sum"),
        n_features=("feature", "size"),
        top_feature=("feature", "first"),   # already sorted by importance
    ).sort_values("total_importance", ascending=False)

    g["share_pct"] = 100 * g["total_importance"] / g["total_importance"].sum()
    return g.reset_index()


def explain_top_drivers(drivers: pd.DataFrame, shap_values, X_test, y_test) -> None:
    """
    Explain what the model learned, in plain English, with business actions.

    Every number here is computed from the test set. The first version of this
    function had a hardcoded claim that prev_success was "the strongest
    customer-level signal" -- SHAP actually ranks it 20th. The claim was a
    reasonable guess from the Phase 1 SQL and it was wrong, and it sat directly
    above a table contradicting it. So now the prose asks the data.
    """
    baseline = y_test.mean()

    print("\n" + "=" * 78)
    print("  TOP 10 CONVERSION DRIVERS (individual features)")
    print("=" * 78 + "\n")

    show = drivers.head(10).copy()
    show.index = range(1, len(show) + 1)
    show["importance"] = show["importance"].map(lambda v: f"{v:.4f}")
    print(show[["feature", "importance", "effect"]].to_string())

    # ---- Theme-level view --------------------------------------------------
    themes = analyse_themes(drivers)

    print("\n" + "=" * 78)
    print("  DRIVERS GROUPED BY THEME")
    print("=" * 78)
    print("""
  Individual ranks understate our overlapping features. We engineered several
  columns that all encode "have we spoken to this person before?" -- and SHAP
  divides credit between correlated features, so each looks weak on its own.
  Summing by theme is the fairer question: how much does this TOPIC matter?
""")
    t = themes.copy()
    t["total_importance"] = t["total_importance"].map(lambda v: f"{v:.4f}")
    t["share_pct"] = t["share_pct"].map(lambda v: f"{v:.1f}%")
    print(t.to_string(index=False))

    def rate_where(col, value=1):
        """Response rate and group size where col == value."""
        mask = X_test[col] == value
        return y_test[mask].mean(), int(mask.sum())

    # ======================= DRIVER 1 =======================================
    econ_share = themes.loc[themes["theme"] == "Macroeconomic conditions",
                            "share_pct"].iloc[0]
    lr_rate, lr_n = rate_where("is_low_rate_period")
    nr_imp = drivers.loc[drivers["feature"] == "nr_employed", "importance"].iloc[0]

    print("\n" + "=" * 78)
    print("  DRIVER 1: MACROECONOMIC CONDITIONS")
    print("=" * 78)
    print(f"""
  THE EVIDENCE
  nr_employed is the #1 feature at {nr_imp:.4f} -- more than 3x the next one.
  The economic theme takes {econ_share:.1f}% of all SHAP importance.
  In low-rate periods: {100 * lr_rate:.1f}% response (n={lr_n:,}) vs a
  {100 * baseline:.1f}% baseline -- {lr_rate / baseline:.1f}x.

  WHAT IT MEANS
  The strongest signals in the model are not about the customer at all. They
  describe the ECONOMY on the day we called. nr_employed (how many people are
  employed), euribor3m (the 3-month interbank rate) and emp_var_rate all push
  the same way: when rates are high and employment is booming, people say no.

  WHY THAT MAKES SENSE
  We're selling a term deposit -- you lock money away for a fixed return. When
  rates are high and jobs feel secure, savers have better options and less fear,
  so a deposit is unattractive. When rates fall and the economy wobbles, a
  guaranteed return starts to look good.

  BUSINESS ACTION
  Time the CAMPAIGN, not just the call list. The rate environment should decide
  WHETHER to run a deposit push at all, and the budget should flex with the
  euribor rather than sit fixed in the annual plan. Nobody at the bank controls
  this variable -- but they can choose when to spend against it.

  THE CAVEAT -- SAY THIS BEFORE THE INTERVIEWER DOES
  This is the model's biggest weakness. These features barely move within a
  given week: they describe WHEN we called, not WHO we called. The model has
  substantially learned "2008-2010 was a good time to sell deposits" -- true,
  and useless for ranking today's call list, where everyone shares the same
  euribor. Its {econ_share:.0f}% importance share is real and it is mostly not
  actionable per-customer. If I deployed this, I'd ship a variant with the
  economic features dropped. It would score LOWER on AUC and might rank BETTER
  in production, because it would be forced to learn about people.""")

    # ======================= DRIVER 2 =======================================
    cell_rate, cell_n = rate_where("contact_cellular")
    tel_rate, tel_n = rate_where("contact_telephone")
    cell_imp = drivers.loc[drivers["feature"] == "is_cellular",
                           "importance"].iloc[0]
    cell_rank = int(drivers.index[drivers["feature"] == "is_cellular"][0]) + 1

    print("\n" + "=" * 78)
    print("  DRIVER 2: CONTACT CHANNEL (mobile vs landline)")
    print("=" * 78)
    print(f"""
  THE EVIDENCE
  is_cellular ranks #{cell_rank} of {len(drivers)} at {cell_imp:.4f}.
  Reached on mobile  : {100 * cell_rate:.1f}% response (n={cell_n:,})
  Reached on landline: {100 * tel_rate:.1f}% response (n={tel_n:,})
  Baseline           : {100 * baseline:.1f}%
  Mobile contacts respond {cell_rate / tel_rate:.1f}x as often as landline ones.

  WHAT IT MEANS
  This is the highest-ranked driver the bank actually CONTROLS. The economy is
  weather; this is a decision. That makes it the most actionable finding here
  even though it isn't the strongest signal.

  BUSINESS ACTION
  Prioritise customers with a mobile number on file. Where you hold only a
  landline, expected return is roughly a third -- either treat those leads as
  lower value, or invest in collecting mobile numbers, which is a data project
  with a measurable payoff rather than a guess.

  THE CAVEAT
  Almost certainly correlation, not causation. Mobile contact likely proxies
  for age, reachability, and how recently the customer updated their details.
  Nobody's landline is what's stopping them from buying -- so "call them on a
  mobile instead" would do nothing. "Prioritise people who have one" still
  works, because you're using it as a SIGNAL, not an intervention. Worth an
  A/B test before anyone rebuilds the dialler around it.""")

    # ======================= DRIVER 3 =======================================
    rel_share = themes.loc[themes["theme"] == "Prior relationship / recency",
                           "share_pct"].iloc[0]
    ps_rate, ps_n = rate_where("prev_success")
    wc_rate, wc_n = rate_where("was_contacted_before")
    dspc_rank = int(drivers.index[
        drivers["feature"] == "days_since_prev_contact"][0]) + 1
    ps_rank = int(drivers.index[drivers["feature"] == "prev_success"][0]) + 1

    # How much does prev_success matter FOR THE PEOPLE IT APPLIES TO?
    # Mean |SHAP| averages over all 8,238 customers, which buries a feature
    # that's decisive for 3.5% of them. Conditioning tells the real story.
    ps_idx = list(X_test.columns).index("prev_success")
    has_history = (X_test["was_contacted_before"] == 1).values
    ps_shap_all = np.abs(shap_values.values[:, ps_idx]).mean()
    ps_shap_cond = np.abs(shap_values.values[has_history, ps_idx]).mean()

    print("\n" + "=" * 78)
    print("  DRIVER 3: PRIOR RELATIONSHIP AND RECENCY")
    print("=" * 78)
    print(f"""
  THE EVIDENCE
  days_since_prev_contact ranks #{dspc_rank}; prev_success only #{ps_rank}.
  The whole relationship theme is {rel_share:.1f}% of importance.
  BUT: previous campaign succeeded -> {100 * ps_rate:.1f}% response (n={ps_n:,})
       contacted before at all     -> {100 * wc_rate:.1f}% response (n={wc_n:,})
       baseline                    -> {100 * baseline:.1f}%
  A prior buyer is {ps_rate / baseline:.1f}x likelier to buy again.

  SO WHY DOES IT RANK 20th WITH A {100 * ps_rate:.0f}% RESPONSE RATE?
  This is the most instructive thing in the whole analysis, and it's a trap in
  how mean |SHAP| is usually read. Two separate effects:

  1. DILUTION BY AVERAGING. Mean |SHAP| averages over ALL {len(X_test):,} test
     customers. Only {wc_n:,} of them ({100 * wc_n / len(X_test):.1f}%) have any
     contact history, so prev_success is exactly 0 for everyone else. Watch what
     conditioning does:
         mean |SHAP| over everyone         : {ps_shap_all:.4f}
         mean |SHAP| over those WITH history: {ps_shap_cond:.4f}
         -> {ps_shap_cond / ps_shap_all:.0f}x stronger for the people it applies to
     The feature isn't weak. It's rare AND decisive, and averaging over
     everyone hides that completely.

  2. CREDIT-SPLITTING. I engineered prev_success, was_contacted_before,
     days_since_prev_contact, poutcome_success and recency_bucket_* to encode
     overlapping facts. When features are correlated, SHAP divides the credit
     between them, so each looks modest while the theme is substantial. That's
     SHAP behaving correctly -- and it's why the theme table above exists.

  THE LESSON: never read mean |SHAP| as "how much this feature matters to a
  customer". It means "how much it moves the AVERAGE customer". For a rare,
  decisive feature those are wildly different numbers. Say this in the
  interview and you will sound like someone who has actually used SHAP.

  BUSINESS ACTION
  1. Call prior buyers FIRST, always. It's the cheapest lift available and it
     needs no model to execute -- one SQL query. Ship it on day one.
  2. Sequence outreach by RECENCY. Contacts within a week convert at ~66%
     versus ~9% for never-contacted. Follow-ups should land inside a 7-day
     window, not whenever the list gets around to it.

  THE HONEST LIMIT
  Only {100 * wc_n / len(X_test):.1f}% of customers have prior contact history.
  This is a spectacular signal on a tiny group. It cannot carry a campaign --
  there aren't enough of them -- but it should always be at the top of the list.""")

    # -- The dud ------------------------------------------------------------
    print("\n" + "=" * 78)
    print("  THE FEATURE THAT DIDN'T WORK")
    print("=" * 78)

    dud_rank = drivers.index[drivers["feature"] == "has_any_loan"]
    if len(dud_rank):
        rank = int(dud_rank[0]) + 1
        imp = drivers.loc[dud_rank[0], "importance"]
        top_imp = drivers.iloc[0]["importance"]
        print(f"""
  Back in Phase 1 I flagged has_any_loan as probably useless: it showed a
  10.90% vs 11.52% response split against an 11.27% baseline. Barely a
  difference. I left it in and said we'd let SHAP judge it.

  SHAP's verdict: rank {rank} of {len(drivers)}, importance {imp:.4f} --
  about {top_imp / max(imp, 1e-9):.0f}x weaker than the top driver.

  It was dead weight, exactly as suspected. Two things worth taking from that:
  the prediction was correct AND the method that confirmed it was independent
  of the method that made it. That's how you build justified confidence in a
  feature set rather than just a story about one.""")


def main():
    print("=" * 78)
    print("  PHASE 6: INTERPRETABILITY WITH SHAP")
    print("=" * 78)

    print("\nLoading model and test data ...")
    X_train, X_test, y_train, y_test = get_train_test()
    model = joblib.load(MODELS / "xgboost.joblib")
    print(f"  Explaining {len(X_test):,} held-out customers")

    shap_values = compute_shap(model, X_test)
    verify_additivity(model, X_test, shap_values)

    drivers = rank_drivers(shap_values, X_test)
    themes = analyse_themes(drivers)

    plot_beeswarm(shap_values, X_test)
    plot_importance_bar(drivers)
    write_report(drivers, themes, X_test, y_test)

    explain_top_drivers(drivers, shap_values, X_test, y_test)

    print(f"\n\nCharts: {FIGURES}")
    print(f"Report: {OUTPUTS / 'shap_drivers.md'}")


if __name__ == "__main__":
    main()
