"""
Run the entire pipeline end-to-end, in order.

    python src/run_all.py

WHY THIS EXISTS
"Reproducible" doesn't mean "I wrote down the steps somewhere". It means
someone else can clone the repo, run one command, and get your exact numbers.
If reproducing your work requires remembering that eda.py comes before
train.py, it isn't reproducible -- it's a memory test.

Every step below is safe to re-run: load_data replaces its table, the SQL uses
CREATE OR REPLACE, and every random operation is seeded. Run this twice and you
get identical results, not duplicated rows.

Takes under a minute (plus the dataset download on the first run). The database
must be running first:
    docker compose up -d
"""

import sys
import time
import traceback

from sqlalchemy import text

import eda
import evaluate
import features
import imbalance
import interpret
import load_data
import run_sql
import train
from config import SQL_DIR, get_engine


def check_database() -> bool:
    """
    Fail fast with a useful message if PostgreSQL isn't up.

    Without this, the first step dies with a connection stack trace 30 lines
    long, and the actual problem -- "you forgot to start Docker" -- is buried
    in the middle of it. Checking the obvious precondition first costs two
    seconds and saves the reader from debugging the wrong thing.
    """
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1;"))
        return True
    except Exception as exc:
        print("\n" + "!" * 78)
        print("  CANNOT REACH POSTGRESQL")
        print("!" * 78)
        print(f"\n  {type(exc).__name__}: {exc}")
        print("\n  Start the database with:")
        print("      docker compose up -d")
        print("\n  Then check it's healthy with:")
        print("      docker compose ps")
        return False


def banner(step: int, total: int, name: str) -> None:
    print("\n\n" + "=" * 78)
    print(f"  STEP {step}/{total}: {name}")
    print("=" * 78)


def main():
    print("=" * 78)
    print("  CAMPAIGN PROPENSITY MODEL - FULL PIPELINE")
    print("=" * 78)

    if not check_database():
        raise SystemExit(1)
    print("\nPostgreSQL is reachable. Starting.\n")

    started = time.time()

    # Each step is (name, callable). Keeping them as a list means the step
    # numbering and the banner text can't drift out of sync with what actually
    # runs -- there's one source of truth for the order.
    steps = [
        ("Load raw CSV into PostgreSQL", load_data.main),
        ("SQL: explore the raw data",
         lambda: run_sql.run_file(SQL_DIR / "01_overview.sql")),
        ("SQL: response rates by segment",
         lambda: run_sql.run_file(SQL_DIR / "02_response_by_segment.sql")),
        ("SQL: build the feature view",
         lambda: run_sql.run_file(SQL_DIR / "03_feature_extraction.sql")),
        ("EDA: distributions, correlations, class balance", eda.main),
        ("Build the model-ready feature matrix", features.main),
        ("Class imbalance: demonstrate and decide", imbalance.main),
        ("Train Logistic Regression and XGBoost", train.main),
        ("Evaluate: AUC, lift, confusion matrices", evaluate.main),
        ("Interpret: SHAP drivers", interpret.main),
    ]

    for i, (name, fn) in enumerate(steps, start=1):
        banner(i, len(steps), name)
        try:
            fn()
        except Exception:
            print("\n" + "!" * 78)
            print(f"  FAILED at step {i}: {name}")
            print("!" * 78 + "\n")
            traceback.print_exc()
            raise SystemExit(1)

    elapsed = time.time() - started

    print("\n\n" + "=" * 78)
    print(f"  PIPELINE COMPLETE in {elapsed / 60:.1f} minutes")
    print("=" * 78)
    print("""
  Everything produced:
    outputs/metrics.md            all evaluation numbers
    outputs/shap_drivers.md       conversion drivers + business actions
    outputs/shap_drivers.csv      the same, as data for the dashboard
    outputs/figures/*.png         9 charts
    outputs/models/*.joblib       the trained models

  Launch the dashboard:
    streamlit run app/dashboard.py
""")


if __name__ == "__main__":
    main()
