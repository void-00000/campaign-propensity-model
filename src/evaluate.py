"""
PHASE 5 - Evaluate both models properly.

THIS IS THE MOST IMPORTANT SCRIPT IN THE PROJECT. Anyone can call .fit().
Knowing whether the result is any good -- and being able to say so in language
a marketing director understands -- is the actual job.

WHAT THIS PRODUCES
  outputs/metrics.md                     - every number, in one readable file
  outputs/figures/05_roc_curves.png      - ROC curves for both models
  outputs/figures/06_lift_chart.png      - decile lift + cumulative gains
  outputs/figures/07_confusion_matrices.png

RUN IT WITH:
    python src/evaluate.py
"""

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config import FIGURES, MODELS, OUTPUTS
from features import get_train_test

# Consistent colours for the two models across every chart.
COLORS = {"Logistic Regression": "#4C72B0", "XGBoost": "#DD8452"}


# ===========================================================================
# METRICS
# ===========================================================================

def core_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    """
    Compute every headline metric at a given decision threshold.

    WHAT IS A THRESHOLD? The model outputs a PROBABILITY (0.0 to 1.0), not a
    yes/no. To make a call list you must pick a cutoff: "phone everyone above
    0.5". That cutoff is a BUSINESS decision, not a modelling one -- lower it
    to catch more buyers at the cost of more wasted calls, raise it to do the
    reverse. The model ranks; you decide where to cut.
    """
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "threshold": threshold,
        "auc": roc_auc_score(y_true, y_proba),   # threshold-independent
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def decile_analysis(y_true, y_proba, n_bins: int = 10) -> pd.DataFrame:
    """
    THE LIFT TABLE - the single most business-relevant output we produce.

    HOW IT WORKS, and it's simpler than it sounds:
      1. Score every customer with the model.
      2. Sort them best-first.
      3. Chop the sorted list into 10 equal groups ("deciles"). Decile 1 is the
         10% the model likes most; decile 10 is the 10% it likes least.
      4. For each group, count how many actually said yes.

    If the model is worthless, every decile has ~11.27% responders -- sorting
    by a meaningless score just shuffles people. If the model works, decile 1 is
    packed with buyers and decile 10 is nearly empty.

    LIFT = this decile's response rate / the overall 11.27% baseline.
      lift 3.0 in decile 1 -> calling that top 10% finds buyers 3x as often as
      calling at random. Same number of calls, triple the signups.

    This is the number to quote to a marketing team, because it maps directly
    onto their budget. AUC is the statistically honest summary; lift is the one
    that gets the project funded.
    """
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_proba)})

    # Sort best-first. This IS the "who should we call first" list.
    df = df.sort_values("p", ascending=False).reset_index(drop=True)

    # Equal-sized buckets by rank. Integer maths: row 0 -> decile 1, and the
    # last row -> decile 10. We do it by POSITION rather than by probability
    # value, because many customers share near-identical scores and cutting on
    # value would give wildly uneven group sizes.
    df["decile"] = (df.index * n_bins // len(df)) + 1

    baseline_rate = df["y"].mean()
    total_responders = df["y"].sum()

    g = df.groupby("decile").agg(
        customers=("y", "size"),
        responders=("y", "sum"),
    )
    g["response_rate"] = g["responders"] / g["customers"]
    g["lift"] = g["response_rate"] / baseline_rate

    # CUMULATIVE view: "if we call the top N deciles, what do we get?"
    # This is what a campaign actually does -- you work down the list until the
    # budget runs out, you don't call decile 4 in isolation.
    g["cum_customers"] = g["customers"].cumsum()
    g["cum_responders"] = g["responders"].cumsum()
    g["cum_pct_customers"] = g["cum_customers"] / len(df)

    # GAINS / CAPTURE RATE: what share of ALL buyers have we found by here?
    g["cum_pct_responders"] = g["cum_responders"] / total_responders

    # Cumulative lift: buyers captured vs customers called. Always trends
    # toward 1.0 by decile 10 -- call everyone and you find everyone, which is
    # exactly as good as random, because it IS everyone.
    g["cum_lift"] = g["cum_pct_responders"] / g["cum_pct_customers"]

    return g.reset_index()


def threshold_sweep(y_true, y_proba) -> pd.DataFrame:
    """
    Show how precision and recall trade off as the cutoff moves.

    There is no "correct" threshold hiding in the data. There's only the one
    that matches your economics, and only you know those. This table is what
    you'd hand a stakeholder to have that conversation with real numbers
    instead of adjectives.
    """
    rows = []
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        m = core_metrics(y_true, y_proba, threshold=t)
        rows.append({
            "threshold": t,
            "customers_called": m["tp"] + m["fp"],
            "buyers_found": m["tp"],
            "buyers_missed": m["fn"],
            "wasted_calls": m["fp"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
        })
    return pd.DataFrame(rows)


# ===========================================================================
# CHARTS
# ===========================================================================

def plot_roc_curves(results: dict, y_test) -> None:
    """
    Chart: the ROC curve.

    WHAT IS IT? For every possible threshold, plot:
      x-axis: false positive rate -- of the people who'd say no, what fraction
              did we call anyway? (wasted effort)
      y-axis: true positive rate  -- of the people who'd say yes, what fraction
              did we call? (money captured)

    Each point is one threshold. Sliding the threshold traces the curve.

    The diagonal is random guessing: call 30% of everyone at random and you get
    30% of the buyers. Nothing gained. The further ABOVE the diagonal, the more
    the model beats random.

    AUC = the Area Under this Curve. 0.5 = the diagonal = useless. 1.0 =
    perfect. And it has a lovely plain-English meaning: pick one random buyer
    and one random non-buyer -- AUC is the probability the model scores the
    buyer higher. At 0.8156, it gets that pairwise ordering right ~82% of the
    time. That is exactly the skill a call list needs.
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_test, r["proba"])
        ax.plot(fpr, tpr, color=COLORS[name], linewidth=2.5,
                label=f"{name} (AUC = {r['metrics']['auc']:.4f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5,
            label="Random guessing (AUC = 0.500)")

    ax.set_xlabel("False positive rate\n(of non-buyers, the share we'd waste a call on)")
    ax.set_ylabel("True positive rate / Recall\n(of real buyers, the share we'd reach)")
    ax.set_title("ROC curve: further above the diagonal is better",
                 fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    fig.tight_layout()
    out = FIGURES / "05_roc_curves.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Saved {out.name}")


def plot_lift_chart(results: dict, baseline_rate: float) -> None:
    """
    Chart: decile lift (left) and cumulative gains (right).

    THE MONEY CHART. If you show one picture in the interview, show this one.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))

    # ---- LEFT: lift per decile -------------------------------------------
    width = 0.38
    x = np.arange(1, 11)

    for i, (name, r) in enumerate(results.items()):
        d = r["deciles"]
        ax1.bar(x + (i - 0.5) * width, d["lift"], width,
                color=COLORS[name], label=name, alpha=0.9)

    ax1.axhline(1.0, color="black", linestyle="--", linewidth=1.5,
                label="Random targeting (lift = 1.0)")
    ax1.set_xlabel("Decile (1 = the 10% the model likes most)")
    ax1.set_ylabel("Lift over random targeting")
    ax1.set_title("Lift by decile: how much better than random?", fontweight="bold")
    ax1.set_xticks(x)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Label decile 1 -- it's the number people actually quote.
    for i, (name, r) in enumerate(results.items()):
        top = r["deciles"].iloc[0]["lift"]
        ax1.text(1 + (i - 0.5) * width, top + 0.08, f"{top:.2f}x",
                 ha="center", fontweight="bold", fontsize=10)

    # ---- RIGHT: cumulative gains ------------------------------------------
    for name, r in results.items():
        d = r["deciles"]
        # Prepend (0,0): calling nobody finds nobody. Without it the curve
        # starts floating in mid-air and the shape misleads.
        xs = np.concatenate([[0], 100 * d["cum_pct_customers"].values])
        ys = np.concatenate([[0], 100 * d["cum_pct_responders"].values])
        ax2.plot(xs, ys, marker="o", color=COLORS[name], linewidth=2.5, label=name)

    ax2.plot([0, 100], [0, 100], "k--", linewidth=1.5,
             label="Random targeting")

    ax2.set_xlabel("% of customers called (working down the ranked list)")
    ax2.set_ylabel("% of all buyers found")
    ax2.set_title("Cumulative gains: call 30%, find how many?", fontweight="bold")
    ax2.legend(loc="lower right")
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 100)

    # Annotate the 30% mark -- a realistic campaign budget, and a concrete
    # anchor beats an abstract curve.
    xgb_d = results["XGBoost"]["deciles"]
    at30 = 100 * xgb_d.iloc[2]["cum_pct_responders"]
    ax2.annotate(
        f"Call top 30%\n-> find {at30:.0f}% of buyers",
        xy=(30, at30),
        xytext=(45, at30 - 22),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="gray"),
    )

    fig.tight_layout()
    out = FIGURES / "06_lift_chart.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Saved {out.name}")


def plot_confusion_matrices(results: dict) -> None:
    """
    Chart: the confusion matrix for each model.

    WHAT IS IT? A 2x2 grid of every possible outcome:

                        Predicted NO        Predicted YES
      Actually NO    True Negative        False Positive
                     (correctly skipped)  (wasted call)
      Actually YES   False Negative       True Positive
                     (MISSED BUYER)       (found a buyer!)

    The bottom-left cell is the expensive one. Those are real customers who
    would have said yes, and we never phoned them.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, (name, r) in zip(axes, results.items()):
        m = r["metrics"]
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])

        im = ax.imshow(cm, cmap="Blues")

        labels = [
            [f"Correctly skipped\n{m['tn']:,}", f"Wasted call\n{m['fp']:,}"],
            [f"MISSED BUYER\n{m['fn']:,}", f"Found buyer\n{m['tp']:,}"],
        ]
        for i in range(2):
            for j in range(2):
                # White text on dark cells, black on light -- otherwise the
                # biggest number is the one you can't read.
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, labels[i][j], ha="center", va="center",
                        color=color, fontweight="bold", fontsize=11)

        ax.set_xticks([0, 1], ["Predicted: don't call", "Predicted: call"])
        ax.set_yticks([0, 1], ["Actually: no", "Actually: yes"])
        ax.set_title(
            f"{name}\nthreshold {m['threshold']:.2f} | "
            f"precision {m['precision']:.3f} | recall {m['recall']:.3f}",
            fontweight="bold",
        )

    fig.suptitle(
        "Confusion matrices - the bottom-left cell is the expensive mistake",
        fontweight="bold", fontsize=13,
    )
    fig.tight_layout()
    out = FIGURES / "07_confusion_matrices.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Saved {out.name}")


# ===========================================================================
# REPORT
# ===========================================================================

def write_metrics_file(results: dict, y_test, baseline_rate: float,
                       sweep: pd.DataFrame) -> None:
    """Write every number to outputs/metrics.md."""
    xgb = results["XGBoost"]
    lr = results["Logistic Regression"]
    xgb_d = xgb["deciles"]
    lr_d = lr["deciles"]

    top_lift = xgb_d.iloc[0]["lift"]
    top_rate = xgb_d.iloc[0]["response_rate"]

    lines = []
    A = lines.append

    A("# Model Evaluation Results\n")
    A("Bank Marketing campaign-response model. All numbers below are measured")
    A("on the **held-out test set** -- customers neither model saw during")
    A("training.\n")

    A("## Setup\n")
    A(f"- Test set: **{len(y_test):,} customers**, "
      f"**{int(y_test.sum()):,} real responders** ({100 * baseline_rate:.2f}%)")
    A(f"- Baseline response rate (random targeting): **{100 * baseline_rate:.2f}%**")
    A("- Imbalance handling: class weights (`class_weight='balanced'`, "
      "`scale_pos_weight=7.88`)")
    A("- `duration` excluded from features (target leakage -- see "
      "`sql/02_response_by_segment.sql`)\n")

    A("## Headline results\n")
    A("| Metric | Logistic Regression | XGBoost |")
    A("|---|---|---|")
    for key, label in [
        ("auc", "**AUC-ROC**"), ("accuracy", "Accuracy"),
        ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"),
    ]:
        A(f"| {label} | {lr['metrics'][key]:.4f} | {xgb['metrics'][key]:.4f} |")
    A(f"| **Top-decile lift** | {lr_d.iloc[0]['lift']:.2f}x | "
      f"{xgb_d.iloc[0]['lift']:.2f}x |")
    A("")

    A("## Confusion matrices (threshold = 0.50)\n")
    for name, r in results.items():
        m = r["metrics"]
        A(f"### {name}\n")
        A("| | Predicted: don't call | Predicted: call |")
        A("|---|---|---|")
        A(f"| **Actually no** | {m['tn']:,} correctly skipped | "
          f"{m['fp']:,} wasted calls |")
        A(f"| **Actually yes** | {m['fn']:,} MISSED buyers | "
          f"{m['tp']:,} buyers found |")
        A("")

    A("## Lift by decile (XGBoost)\n")
    A("Customers sorted best-first, cut into ten equal groups.\n")
    A("| Decile | Customers | Responders | Response rate | Lift | "
      "Cum. % of buyers found |")
    A("|---|---|---|---|---|---|")
    for _, row in xgb_d.iterrows():
        A(f"| {int(row['decile'])} | {int(row['customers']):,} | "
          f"{int(row['responders']):,} | {100 * row['response_rate']:.2f}% | "
          f"**{row['lift']:.2f}x** | {100 * row['cum_pct_responders']:.1f}% |")
    A("")

    A("## Threshold sweep (XGBoost)\n")
    A("There is no 'correct' threshold in the data -- only the one matching")
    A("your economics. This is the tradeoff, in real numbers.\n")
    A("| Threshold | Customers called | Buyers found | Buyers missed | "
      "Wasted calls | Precision | Recall |")
    A("|---|---|---|---|---|---|---|")
    for _, row in sweep.iterrows():
        A(f"| {row['threshold']:.1f} | {int(row['customers_called']):,} | "
          f"{int(row['buyers_found']):,} | {int(row['buyers_missed']):,} | "
          f"{int(row['wasted_calls']):,} | {row['precision']:.3f} | "
          f"{row['recall']:.3f} |")
    A("")

    A("## What these numbers mean in business terms\n")
    A(f"**Top-decile lift = {top_lift:.2f}x.** Rank all customers by predicted")
    A(f"probability and call only the top 10%. In that group "
      f"{100 * top_rate:.2f}% say yes,")
    A(f"versus {100 * baseline_rate:.2f}% if you called 10% at random. Same")
    A(f"number of calls, **{top_lift:.2f}x the signups**.\n")

    cum30 = xgb_d.iloc[2]["cum_pct_responders"]
    A(f"**Call the top 30%, capture {100 * cum30:.1f}% of all buyers.** Cut")
    A(f"your call volume by 70% and still get {100 * cum30:.0f}% of the")
    A(f"revenue -- because the model put the buyers at the top of the list.\n")

    A("**Why not accuracy?** A model that always predicts 'no' scores 88.7%")
    A("accuracy on this data and finds zero buyers. Accuracy rewards a model")
    A("for ignoring the rare class -- which is the only class we care about.\n")

    out = OUTPUTS / "metrics.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved {out.name}")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 78)
    print("  PHASE 5: EVALUATION")
    print("=" * 78)

    print("\nLoading the same split and the trained models ...")
    X_train, X_test, y_train, y_test = get_train_test()

    models = {
        "Logistic Regression": joblib.load(MODELS / "logistic_regression.joblib"),
        "XGBoost": joblib.load(MODELS / "xgboost.joblib"),
    }

    baseline_rate = y_test.mean()
    print(f"  Test set: {len(y_test):,} customers, {y_test.sum():,} responders "
          f"({100 * baseline_rate:.2f}% baseline)")

    # ---- Score both models -------------------------------------------------
    results = {}
    for name, model in models.items():
        # predict_proba returns two columns: P(no) and P(yes). We want [:, 1],
        # the probability of the class we care about.
        proba = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "proba": proba,
            "metrics": core_metrics(y_test, proba),
            "deciles": decile_analysis(y_test, proba),
        }

    # ---- Print the comparison ----------------------------------------------
    print("\n" + "=" * 78)
    print("  HEADLINE METRICS (test set, threshold 0.50)")
    print("=" * 78 + "\n")

    table = pd.DataFrame({
        name: {
            "AUC-ROC": f"{r['metrics']['auc']:.4f}",
            "Accuracy": f"{r['metrics']['accuracy']:.4f}",
            "Precision": f"{r['metrics']['precision']:.4f}",
            "Recall": f"{r['metrics']['recall']:.4f}",
            "F1": f"{r['metrics']['f1']:.4f}",
            "Top-decile lift": f"{r['deciles'].iloc[0]['lift']:.2f}x",
        }
        for name, r in results.items()
    })
    print(table.to_string())

    print("\n--- Confusion matrices ---")
    for name, r in results.items():
        m = r["metrics"]
        print(f"\n  {name}")
        print(f"    Correctly skipped : {m['tn']:,}")
        print(f"    Wasted calls      : {m['fp']:,}")
        print(f"    MISSED buyers     : {m['fn']:,}   <- the expensive mistake")
        print(f"    Buyers found      : {m['tp']:,}")

    # ---- Lift table --------------------------------------------------------
    print("\n" + "=" * 78)
    print("  LIFT BY DECILE (XGBoost)")
    print("=" * 78)
    print("\n  Sort all customers best-first, cut into 10 equal groups.\n")

    d = results["XGBoost"]["deciles"].copy()
    show = pd.DataFrame({
        "decile": d["decile"].astype(int),
        "customers": d["customers"].astype(int),
        "responders": d["responders"].astype(int),
        "response_rate": (100 * d["response_rate"]).map(lambda v: f"{v:.2f}%"),
        "lift": d["lift"].map(lambda v: f"{v:.2f}x"),
        "cum_% of buyers found": (100 * d["cum_pct_responders"]).map(
            lambda v: f"{v:.1f}%"),
    })
    print(show.to_string(index=False))

    # ---- Threshold sweep ---------------------------------------------------
    sweep = threshold_sweep(y_test, results["XGBoost"]["proba"])
    print("\n" + "=" * 78)
    print("  THRESHOLD SWEEP (XGBoost) - the business tradeoff")
    print("=" * 78 + "\n")
    sweep_show = sweep.copy()
    for c in ["precision", "recall", "f1"]:
        sweep_show[c] = sweep_show[c].map(lambda v: f"{v:.3f}")
    print(sweep_show.to_string(index=False))

    # ---- Charts ------------------------------------------------------------
    print("\nGenerating charts ...")
    plot_roc_curves(results, y_test)
    plot_lift_chart(results, baseline_rate)
    plot_confusion_matrices(results)
    write_metrics_file(results, y_test, baseline_rate, sweep)

    # ---- THE HEADLINE NUMBERS ----------------------------------------------
    xgb_auc = results["XGBoost"]["metrics"]["auc"]
    xgb_d = results["XGBoost"]["deciles"]
    top_lift = xgb_d.iloc[0]["lift"]
    top_rate = xgb_d.iloc[0]["response_rate"]
    cum30 = xgb_d.iloc[2]["cum_pct_responders"]

    print("\n" + "#" * 78)
    print("#" + " " * 76 + "#")
    print("#" + "  THE TWO NUMBERS FOR YOUR RESUME".ljust(76) + "#")
    print("#" + " " * 76 + "#")
    print("#" + f"    Final XGBoost AUC   :  {xgb_auc:.2f}".ljust(76) + "#")
    print("#" + f"    Top-decile lift     :  {top_lift:.2f}x  "
          f"({(top_lift - 1) * 100:.0f}% above baseline)".ljust(76) + "#")
    print("#" + " " * 76 + "#")
    print("#" * 78)

    print(f"""
  In words, for the interview:

    "I built a campaign-response model on 41,188 bank customers that reaches
     {xgb_auc:.2f} AUC on held-out data. Ranking customers by predicted
     probability, the top decile converts at {100 * top_rate:.1f}% versus an
     {100 * baseline_rate:.1f}% baseline -- {top_lift:.2f}x lift. Calling only
     the top 30% of the list captures {100 * cum30:.0f}% of all responders, so
     the team could cut call volume by 70% and keep {100 * cum30:.0f}% of the
     revenue."

  Full details: outputs/metrics.md
""")


if __name__ == "__main__":
    main()
