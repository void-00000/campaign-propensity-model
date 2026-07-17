"""
PHASE 7 - The Streamlit dashboard.

WHAT IS STREAMLIT?
A library that turns a plain Python script into a web app. No HTML, no
JavaScript, no web server to configure. You write `st.write("hello")` and a
web page appears. It re-runs your whole script top-to-bottom on every
interaction, which sounds wasteful and is exactly why it stays simple.

WHY THIS EXISTS
A model that lives in a notebook helps nobody. A marketing team can't run
`python src/evaluate.py`. They need a list of names, in priority order, that
they can filter and export. That is the entire product; everything in the six
phases before this was in service of producing this ranked list.

WHAT IT SHOWS
  - Ranked "next best customers to target", scored by the XGBoost model
  - A live campaign simulator: pick a budget, see expected responders and lift
  - The global conversion drivers from Phase 6
  - Per-customer explanations: WHY is this specific person ranked here?

RUN IT WITH:
    streamlit run app/dashboard.py
"""

import sys
from pathlib import Path

# Streamlit runs this file from the project root, so `src` isn't on the import
# path by default. We add it explicitly rather than restructuring the project
# into an installable package -- that would be the "proper" fix and it would be
# three extra files of packaging boilerplate for one import.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from config import MODELS, OUTPUTS
from features import build_feature_matrix, get_train_test, load_feature_view

# ---------------------------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Campaign Propensity Model",
    page_icon="📞",
    layout="wide",
)


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
# @st.cache_data / @st.cache_resource matter a lot here. Streamlit re-runs this
# entire script every time you move a slider. Without caching, every slider
# nudge would re-query PostgreSQL, rebuild 92 features and reload the model --
# several seconds of work, repeated constantly, for data that never changed.
# The cache decorators say "run this once, reuse the result".
#
# cache_data is for data (it copies the result, so a caller can't mutate the
# cache). cache_resource is for things you shouldn't copy, like a model.

@st.cache_resource
def load_model():
    """Load the trained XGBoost model and the feature order it expects."""
    model = joblib.load(MODELS / "xgboost.joblib")
    feature_names = joblib.load(MODELS / "feature_names.joblib")
    return model, feature_names


@st.cache_data
def load_scored_customers():
    """
    Score the held-out test set and return it ready to display.

    WHY THE TEST SET, and not all 41,188 customers?
    Because these 8,238 are the only ones the model has never seen. Scoring
    customers the model trained on would show inflated, dishonest probabilities
    -- the model partly remembers their answers. The test set is our best
    available stand-in for "a fresh list of people we haven't called yet",
    which is the real use case.
    """
    model, feature_names = load_model()

    # The same split every phase uses -- same seed, same rows.
    X_train, X_test, y_train, y_test = get_train_test()

    # SAFETY CHECK, and not a decorative one. The model consumes an unlabelled
    # array of 92 numbers. If the columns arrived in a different order, every
    # prediction would be silently wrong -- no exception, no warning, just
    # confident nonsense. Assert the contract instead of hoping.
    assert list(X_test.columns) == list(feature_names), (
        "Feature order does not match what the model was trained on."
    )

    proba = model.predict_proba(X_test)[:, 1]

    # Pull the human-readable columns from the SQL view. X_test is one-hot
    # encoded -- great for the model, unreadable for a person. Nobody wants a
    # call list with a column called job_blue-collar.
    view = load_feature_view()
    display = view.loc[X_test.index, [
        "age", "job", "marital", "education", "contact", "month",
        "poutcome", "campaign", "days_since_prev_contact", "euribor3m",
    ]].copy()

    display.insert(0, "customer_id", X_test.index)
    display["probability"] = proba
    display["actually_responded"] = y_test.values

    # Rank best-first. This ordering IS the product.
    display = display.sort_values("probability", ascending=False).reset_index(drop=True)
    display.insert(0, "rank", range(1, len(display) + 1))

    # Which decile each customer falls in -- same integer maths as Phase 5.
    display["decile"] = (display.index * 10 // len(display)) + 1

    return display, X_test, y_test


@st.cache_data
def load_drivers():
    """Load the SHAP driver rankings produced by Phase 6."""
    path = OUTPUTS / "shap_drivers.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_resource
def get_explainer():
    """The SHAP explainer, for per-customer 'why is this person ranked here?'."""
    model, _ = load_model()
    return shap.TreeExplainer(model)


# ---------------------------------------------------------------------------
# LOAD EVERYTHING
# ---------------------------------------------------------------------------
try:
    scored, X_test, y_test = load_scored_customers()
    drivers = load_drivers()
except Exception as exc:
    # A dashboard that dies with a raw stack trace teaches the user nothing.
    # Name the things that actually go wrong -- and note that the likely cause
    # differs depending on where this is running, so we show both.
    st.error(f"**Could not load the model or data.**\n\n`{type(exc).__name__}: {exc}`")

    from config import describe_connection

    st.info(
        "**Running locally?**\n\n"
        "1. Is PostgreSQL up? `docker compose up -d`\n"
        "2. Is the model trained? `python src/run_all.py`\n\n"
        "**Deployed on Streamlit Cloud?**\n\n"
        "1. Is `DATABASE_URL` set in the app's *Settings → Secrets*?\n"
        "2. Has the cloud database been loaded? Point `DATABASE_URL` at it "
        "locally and run `python src/load_data.py`, then "
        "`python src/run_sql.py sql/03_feature_extraction.sql`\n"
        "3. Neon free-tier databases sleep when idle — a first request after "
        "a long gap can time out. Try reloading once."
    )
    st.caption(f"Attempted connection: {describe_connection()}")
    st.stop()

baseline_rate = y_test.mean()

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.title("📞 Customer Propensity & Campaign Response")
st.caption(
    f"XGBoost model · AUC 0.82 · scoring {len(scored):,} held-out customers "
    f"the model has never seen"
)

# ---------------------------------------------------------------------------
# SIDEBAR - the campaign controls
# ---------------------------------------------------------------------------
st.sidebar.header("Campaign settings")

budget_pct = st.sidebar.slider(
    "What % of customers can you call?",
    min_value=5,
    max_value=100,
    value=20,
    step=5,
    help="Your call capacity. The model gives you the best customers to fill it.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filter the list")

jobs = st.sidebar.multiselect(
    "Job", options=sorted(scored["job"].unique()), default=[],
    help="Leave empty to include all",
)
contacts = st.sidebar.multiselect(
    "Contact method", options=sorted(scored["contact"].unique()), default=[],
)
age_range = st.sidebar.slider(
    "Age", int(scored["age"].min()), int(scored["age"].max()),
    (int(scored["age"].min()), int(scored["age"].max())),
)

# Apply the filters.
filtered = scored.copy()
if jobs:
    filtered = filtered[filtered["job"].isin(jobs)]
if contacts:
    filtered = filtered[filtered["contact"].isin(contacts)]
filtered = filtered[filtered["age"].between(*age_range)]

# Take the top N% of whatever survived filtering.
n_to_call = max(1, int(len(filtered) * budget_pct / 100))
target_list = filtered.head(n_to_call)

# ---------------------------------------------------------------------------
# KPI ROW
# ---------------------------------------------------------------------------
st.subheader("If you called this list")

# These are measured against actual outcomes, because we have them for the test
# set. In production you would not -- you'd only have the model's estimate.
# Showing what actually happened is what makes this a demonstration of the
# model's value rather than a restatement of its own predictions.
found = int(target_list["actually_responded"].sum())
rate = target_list["actually_responded"].mean() if len(target_list) else 0
lift = rate / baseline_rate if baseline_rate > 0 else 0
total_responders = int(y_test.sum())
captured = found / total_responders if total_responders else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Customers called", f"{len(target_list):,}",
          f"{100 * len(target_list) / len(scored):.0f}% of all")
c2.metric("Responders found", f"{found:,}")
c3.metric("Response rate", f"{100 * rate:.1f}%",
          f"{100 * (rate - baseline_rate):+.1f} pts vs random")
c4.metric("Lift over random", f"{lift:.2f}x")
c5.metric("% of all buyers captured", f"{100 * captured:.1f}%")

if lift >= 1:
    st.success(
        f"**Calling these {len(target_list):,} customers finds {found:,} buyers "
        f"— {lift:.2f}x what you'd get calling {len(target_list):,} people at "
        f"random.** Random targeting would find roughly "
        f"{int(len(target_list) * baseline_rate):,}."
    )
else:
    st.warning(
        "This selection performs no better than random. The filters have "
        "removed the customers the model ranked highly."
    )

# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    ["🎯 Next best customers", "📊 What drives conversion", "🔍 Explain one customer"]
)

# ============================ TAB 1 =========================================
with tab1:
    st.subheader(f"Top {len(target_list):,} customers to call, best first")
    st.caption(
        "Ranked by the model's predicted probability of subscribing. "
        "`actually_responded` is the real outcome — shown here because this is "
        "a demo on historical data. In production that column wouldn't exist yet."
    )

    show = target_list.copy()
    show["probability"] = show["probability"].map(lambda v: f"{100 * v:.1f}%")
    show["days_since_prev_contact"] = show["days_since_prev_contact"].map(
        lambda v: "never" if pd.isna(v) else f"{int(v)} days"
    )
    show["actually_responded"] = show["actually_responded"].map({1: "✅ yes", 0: "— no"})

    st.dataframe(
        show[[
            "rank", "customer_id", "probability", "decile", "age", "job",
            "education", "contact", "month", "poutcome",
            "days_since_prev_contact", "actually_responded",
        ]],
        width="stretch",
        hide_index=True,
        height=420,
    )

    # The export button is the point. A list a team can't take away is a demo,
    # not a tool.
    st.download_button(
        "⬇️ Download this call list as CSV",
        data=target_list.to_csv(index=False).encode("utf-8"),
        file_name=f"call_list_top_{budget_pct}pct.csv",
        mime="text/csv",
    )

    # ---- Decile summary ---------------------------------------------------
    st.markdown("---")
    st.subheader("Response rate by decile")
    st.caption(
        "Decile 1 is the 10% of customers the model likes most. If the model "
        "were worthless, every bar would sit at the baseline."
    )

    dec = scored.groupby("decile").agg(
        customers=("actually_responded", "size"),
        responders=("actually_responded", "sum"),
    )
    dec["response_rate"] = 100 * dec["responders"] / dec["customers"]
    dec["lift"] = dec["response_rate"] / (100 * baseline_rate)

    fig, ax = plt.subplots(figsize=(10, 3.8))
    colors = ["#DD8452" if r > 100 * baseline_rate else "#4C72B0"
              for r in dec["response_rate"]]
    ax.bar(dec.index, dec["response_rate"], color=colors)
    ax.axhline(100 * baseline_rate, color="black", linestyle="--",
               label=f"baseline {100 * baseline_rate:.1f}%")
    for d, r, l in zip(dec.index, dec["response_rate"], dec["lift"]):
        ax.text(d, r + 1, f"{l:.1f}x", ha="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Decile (1 = model's favourites)")
    ax.set_ylabel("Response rate (%)")
    ax.set_xticks(range(1, 11))
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ============================ TAB 2 =========================================
with tab2:
    st.subheader("What drives conversion (SHAP)")

    if drivers is None:
        st.warning("Run `python src/interpret.py` first to generate the drivers.")
    else:
        col1, col2 = st.columns([3, 2])

        with col1:
            st.markdown("**Top 15 features by impact**")
            top = drivers.head(15).iloc[::-1]
            colors = [
                "#DD8452" if d > 0.05 else "#4C72B0" if d < -0.05 else "#999999"
                for d in top["direction"]
            ]
            fig, ax = plt.subplots(figsize=(8, 6.5))
            ax.barh(top["feature"], top["importance"], color=colors)
            ax.set_xlabel("Mean |SHAP value|")
            ax.grid(axis="x", alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption(
                "Orange = pushes toward yes · Blue = pushes toward no · "
                "Grey = mixed/non-linear"
            )

        with col2:
            st.markdown("**By theme**")
            theme = drivers.groupby("theme")["importance"].sum().sort_values(
                ascending=False)
            theme_pct = 100 * theme / theme.sum()
            for name, pct in theme_pct.items():
                st.progress(float(pct / 100), text=f"{name} — {pct:.1f}%")

            st.markdown("---")
            st.info(
                "**Why group them?** Several features encode the same fact "
                "(`prev_success`, `was_contacted_before`, "
                "`days_since_prev_contact` all mean *have we spoken before?*). "
                "SHAP splits credit between correlated features, so each looks "
                "weak alone while the theme is strong."
            )

        st.markdown("---")
        st.markdown("### The three headline drivers")

        d1, d2, d3 = st.columns(3)
        with d1:
            st.markdown("#### 1. The economy")
            st.markdown(
                "`nr_employed`, `euribor3m`, `emp_var_rate` — **54% of all "
                "importance.** High rates and strong employment → people say no. "
                "A term deposit competes badly when savers have better options."
            )
            st.warning(
                "**Caveat:** these describe *when* we called, not *who*. Nearly "
                "constant within a week, so they're weak for ranking today's "
                "list. The model partly learned *'2008–2010 was a good time to "
                "sell deposits'*."
            )
        with d2:
            st.markdown("#### 2. Contact channel")
            st.markdown(
                "Mobile **14.9%** vs landline **5.0%** response. The "
                "highest-ranked driver the bank actually **controls**."
            )
            st.warning(
                "**Caveat:** correlation, not causation. Mobile proxies for age "
                "and reachability. Useful as a *signal*; switching someone's "
                "number changes nothing."
            )
        with d3:
            st.markdown("#### 3. Prior relationship")
            st.markdown(
                "Prior buyers respond at **65.8%** vs an 11.3% baseline — "
                "**5.8x**. Contacts within a week convert at ~66% vs ~9% for "
                "never-contacted."
            )
            st.success(
                "**Action:** call prior buyers first — one SQL query, no model "
                "needed. Sequence follow-ups inside 7 days. Only ~3.5% of "
                "customers qualify, so it tops the list but can't fill it."
            )

# ============================ TAB 3 =========================================
with tab3:
    st.subheader("Why is this customer ranked here?")
    st.caption(
        "SHAP explains one prediction at a time: how much did each feature push "
        "this specific person up or down from the average customer's score?"
    )

    pick = st.selectbox(
        "Choose a customer from the current list",
        options=target_list["rank"].tolist()[:200],
        format_func=lambda r: (
            f"Rank {r} — "
            f"{target_list.loc[target_list['rank'] == r, 'job'].iloc[0]}, "
            f"age {target_list.loc[target_list['rank'] == r, 'age'].iloc[0]}, "
            f"{100 * target_list.loc[target_list['rank'] == r, 'probability'].iloc[0]:.1f}% likely"
        ),
        help="Showing the top 200 of your filtered list",
    )

    row = target_list[target_list["rank"] == pick].iloc[0]
    cust_id = row["customer_id"]

    left, right = st.columns([1, 2])

    with left:
        st.metric("Predicted probability", f"{100 * row['probability']:.1f}%")
        st.metric("Decile", f"{int(row['decile'])}")
        st.metric("Actually responded",
                  "✅ Yes" if row["actually_responded"] == 1 else "— No")
        st.markdown("**Profile**")
        st.write({
            "age": int(row["age"]),
            "job": row["job"],
            "education": row["education"],
            "marital": row["marital"],
            "contact": row["contact"],
            "month": row["month"],
            "previous outcome": row["poutcome"],
            "calls this campaign": int(row["campaign"]),
        })

    with right:
        explainer = get_explainer()

        # Explain this ONE customer. .loc[[cust_id]] with a list keeps it a
        # DataFrame -- a bare .loc[cust_id] returns a Series and SHAP wants 2-D.
        customer_X = X_test.loc[[cust_id]]
        sv = explainer(customer_X)

        contrib = pd.DataFrame({
            "feature": X_test.columns,
            "shap": sv.values[0],
            "value": customer_X.iloc[0].values,
        })
        # Biggest movers, in either direction.
        contrib["abs"] = contrib["shap"].abs()
        top_c = contrib.nlargest(12, "abs").sort_values("shap")

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ["#DD8452" if v > 0 else "#4C72B0" for v in top_c["shap"]]
        ax.barh(
            [f"{f} = {v:.4g}" if not pd.isna(v) else f"{f} = missing"
             for f, v in zip(top_c["feature"], top_c["value"])],
            top_c["shap"],
            color=colors,
        )
        ax.axvline(0, color="black", linewidth=1)
        ax.set_xlabel("SHAP value  (← pushes toward 'no' | pushes toward 'yes' →)")
        ax.set_title(f"Why customer {cust_id} scored {100 * row['probability']:.1f}%",
                     fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Orange bars pushed this customer toward 'yes'; blue bars pushed "
            "toward 'no'. The bars sum (in log-odds) to the gap between this "
            "customer's score and the average customer's score."
        )

# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Model: XGBoost, AUC 0.8156, top-decile lift 4.71x · Features built in "
    "PostgreSQL · `duration` excluded (target leakage) · Imbalance handled with "
    "class weights"
)
