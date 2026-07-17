"""
PHASE 2a - Exploratory Data Analysis (EDA).

WHAT IS EDA?
"Exploratory Data Analysis" is the unglamorous step of LOOKING at your data
before modelling it. Distributions, relationships, oddities. It is not busywork:
almost every serious modelling mistake is a data misunderstanding wearing a
disguise, and looking is how you catch it. We already caught two in SQL (the
pdays sentinel and 'unknown' hiding as clean data). This is the visual pass.

WHAT THIS SCRIPT PRODUCES
  outputs/figures/01_class_balance.png
  outputs/figures/02_numeric_distributions.png
  outputs/figures/03_correlation_heatmap.png
  outputs/figures/04_response_by_category.png
plus a printed summary in your terminal.

RUN IT WITH:
    python src/eda.py
"""

import matplotlib

# Use the "Agg" backend: draw to a file, never try to open a GUI window.
# This must be set BEFORE importing pyplot. Without it, a script run from a
# terminal or a server with no display can hang or crash trying to pop up a
# window that nobody is there to see.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import FIGURES, get_engine

# A consistent look for every chart, set once.
sns.set_theme(style="whitegrid")
# Colour-blind-safe palette. Roughly 1 in 12 men has some form of colour
# vision deficiency, and red/green is the common failure. A chart nobody can
# read isn't a chart.
PALETTE = {"no": "#4C72B0", "yes": "#DD8452"}


def load_features() -> pd.DataFrame:
    """
    Read the feature view we built in SQL (Phase 1) into pandas.

    Note what's happening here: the joining, decoding, bucketing and ratio
    maths all already happened inside PostgreSQL. Python receives a clean,
    finished table. That's the division of labour we want -- and at real scale
    it's the only thing that works, because the raw table wouldn't fit in
    memory but the aggregated result will.
    """
    print("Loading bank_marketing_features from PostgreSQL ...")
    df = pd.read_sql("SELECT * FROM bank_marketing_features;", get_engine())
    print(f"  {len(df):,} rows x {len(df.columns)} columns\n")
    return df


def plot_class_balance(df: pd.DataFrame) -> None:
    """
    Chart 1: how lopsided is our target?

    This is the picture behind Phase 3. Draw it before you fix it.
    """
    counts = df["target"].value_counts().sort_index()
    pct = 100 * counts / counts.sum()

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        ["No (did not subscribe)", "Yes (subscribed)"],
        counts.values,
        color=[PALETTE["no"], PALETTE["yes"]],
    )

    # Put the actual numbers on the bars. A reader shouldn't have to squint at
    # an axis to get the value -- if the number matters, print the number.
    for bar, count, p in zip(bars, counts.values, pct.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 400,
            f"{count:,}\n({p:.2f}%)",
            ha="center",
            fontweight="bold",
        )

    ax.set_ylabel("Number of customers")
    ax.set_title(
        f"Class imbalance: only {pct[1]:.2f}% of customers said yes",
        fontweight="bold",
    )
    ax.set_ylim(0, counts.max() * 1.15)

    fig.tight_layout()
    out = FIGURES / "01_class_balance.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Saved {out.name}")


def plot_numeric_distributions(df: pd.DataFrame) -> None:
    """
    Chart 2: for key numbers, do responders look different from non-responders?

    HOW TO READ THESE: each panel overlays two histograms -- customers who said
    no, and customers who said yes. Where the two shapes SEPARATE, that feature
    carries signal the model can use. Where they sit on top of each other, the
    feature can't tell the two groups apart.

    We plot each group as a percentage of ITSELF (stat="percent", common_norm
    off) rather than as raw counts. With a 9:1 imbalance, raw counts would make
    the 'yes' histogram invisible -- you'd conclude nothing separates, when
    really the bars were just too short to see.
    """
    numeric_cols = [
        "age",
        "campaign",
        "previous",
        "total_contacts",
        "euribor3m",
        "cons_conf_idx",
        "emp_var_rate",
        "nr_employed",
        "education_years",
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 11))

    for ax, col in zip(axes.flatten(), numeric_cols):
        for label, name in [(0, "no"), (1, "yes")]:
            subset = df.loc[df["target"] == label, col].dropna()
            sns.histplot(
                subset,
                ax=ax,
                color=PALETTE[name],
                label=f"{name} (n={len(subset):,})",
                stat="percent",
                common_norm=False,  # each group sums to 100% on its own
                bins=30,
                alpha=0.6,
                edgecolor=None,
            )
        ax.set_title(col, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("% within group")
        ax.legend(fontsize=8)

    fig.suptitle(
        "Where the orange and blue shapes separate, the model has signal",
        fontweight="bold",
        fontsize=13,
    )
    fig.tight_layout()
    out = FIGURES / "02_numeric_distributions.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Saved {out.name}")


def plot_correlation_heatmap(df: pd.DataFrame) -> pd.Series:
    """
    Chart 3: which numbers move together?

    WHAT IS CORRELATION? A number from -1 to +1 describing how two columns move
    together. +1 = perfect lockstep, -1 = perfect opposites, 0 = no linear
    relationship at all.

    We're reading this for two different purposes:
      1. The 'target' row: which features relate to the outcome?
      2. Feature-to-feature: which features duplicate EACH OTHER? Two features
         correlated at 0.95 carry nearly the same information. That's called
         MULTICOLLINEARITY, and it makes Logistic Regression's coefficients
         unstable and untrustworthy -- the model can't tell which of the twins
         deserves the credit, so it splits it arbitrarily.

    THE BIG CAVEAT: correlation only sees STRAIGHT-LINE relationships. Age has
    a U-shaped effect on response (young and old both high, middle low). Its
    correlation with target will be near zero -- and that is NOT the same as
    "age doesn't matter". It means "age doesn't matter LINEARLY". XGBoost will
    find the U-shape that this chart is blind to. Never drop a feature because
    its correlation is low.
    """
    numeric_df = df.select_dtypes(include=["number"])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(
        corr,
        cmap="RdBu_r",  # diverging: red/blue with white at zero
        center=0,       # anchor white at 0 so sign is readable at a glance
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.7, "label": "correlation"},
        ax=ax,
    )
    ax.set_title(
        "Correlation between numeric features (linear relationships only)",
        fontweight="bold",
    )

    fig.tight_layout()
    out = FIGURES / "03_correlation_heatmap.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Saved {out.name}")

    return corr["target"].drop("target").sort_values(key=abs, ascending=False)


def plot_response_by_category(df: pd.DataFrame) -> None:
    """
    Chart 4: response rate across the main categorical features.

    The dashed line is the 11.27% baseline. Bars above it are segments worth
    targeting; bars below it are segments where calling loses money.
    """
    cat_cols = ["job", "education", "month", "poutcome", "age_band", "recency_bucket"]
    baseline = 100 * df["target"].mean()

    fig, axes = plt.subplots(3, 2, figsize=(15, 13))

    for ax, col in zip(axes.flatten(), cat_cols):
        # Response rate per category, biggest first.
        rates = (100 * df.groupby(col)["target"].mean()).sort_values(ascending=False)

        colors = [PALETTE["yes"] if v > baseline else PALETTE["no"] for v in rates.values]
        ax.barh(rates.index.astype(str), rates.values, color=colors)
        ax.axvline(
            baseline,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"baseline {baseline:.2f}%",
        )
        ax.set_title(col, fontweight="bold")
        ax.set_xlabel("response rate (%)")
        ax.legend(fontsize=8)
        ax.invert_yaxis()  # highest rate at the top, natural reading order

    fig.suptitle(
        "Response rate by segment - orange beats the 11.27% baseline, blue trails it",
        fontweight="bold",
        fontsize=13,
    )
    fig.tight_layout()
    out = FIGURES / "04_response_by_category.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Saved {out.name}")


def print_summary(df: pd.DataFrame, target_corr: pd.Series) -> None:
    """Print the findings that matter, in text you can actually read."""
    print("\n" + "=" * 70)
    print("  EDA SUMMARY")
    print("=" * 70)

    n = len(df)
    pos = int(df["target"].sum())
    print(f"\nDataset: {n:,} customers, {len(df.columns) - 1} features")
    print(f"Responders: {pos:,} ({100 * pos / n:.2f}%)")
    print(f"Non-responders: {n - pos:,} ({100 * (n - pos) / n:.2f}%)")
    print(f"Imbalance ratio: {(n - pos) / pos:.1f} non-responders per responder")

    print("\n--- Missing values (only where they exist) ---")
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if len(missing) == 0:
        print("  None.")
    for col, count in missing.items():
        print(f"  {col:28s} {count:>6,} ({100 * count / n:5.2f}%)  <- NULL by design")

    print("\n--- Top 12 features by |correlation| with target ---")
    print("    (linear only -- a low score here does NOT mean 'useless')")
    for col, val in target_corr.head(12).items():
        direction = "higher -> more likely yes" if val > 0 else "higher -> more likely no"
        print(f"  {col:28s} {val:+.3f}   {direction}")

    print("\n--- Highly correlated feature PAIRS (|r| > 0.85) ---")
    print("    (these duplicate each other -- a problem for Logistic Regression)")
    numeric_df = df.select_dtypes(include=["number"]).drop(columns=["target"])
    corr = numeric_df.corr().abs()

    seen = set()
    found = False
    for a in corr.columns:
        for b in corr.columns:
            # Skip self-pairs, and skip (b,a) once we've reported (a,b).
            if a == b or (b, a) in seen:
                continue
            seen.add((a, b))
            if corr.loc[a, b] > 0.85:
                print(f"  {a:26s} <-> {b:26s} r = {corr.loc[a, b]:.3f}")
                found = True
    if not found:
        print("  None.")


def main():
    df = load_features()

    print("Generating charts ...")
    plot_class_balance(df)
    plot_numeric_distributions(df)
    target_corr = plot_correlation_heatmap(df)
    plot_response_by_category(df)

    print_summary(df, target_corr)
    print(f"\nCharts saved to: {FIGURES}")


if __name__ == "__main__":
    main()
