"""
PHASE 1 - Get the raw data into PostgreSQL.

WHAT THIS SCRIPT DOES, IN ORDER
  1. Downloads the UCI "Bank Marketing" dataset (a .zip file) -- unless we
     already downloaded it, in which case it skips straight to step 2.
  2. Unzips it and finds bank-additional-full.csv inside.
  3. Reads that CSV into pandas and tidies up the column names.
  4. Writes the whole table into PostgreSQL.
  5. Verifies the load by asking PostgreSQL to count the rows back.

RUN IT WITH:
    python src/load_data.py

This script is SAFE TO RE-RUN. It replaces the table each time rather than
appending, so running it twice does not give you 82,000 rows. Scripts that
you can run repeatedly without corrupting state are called "idempotent", and
they matter because you WILL re-run this while learning.
"""

import io
import zipfile

import pandas as pd
import requests
from sqlalchemy import text

# Import the settings we defined once in Phase 0, rather than repeating them.
from config import DATA_RAW, RAW_TABLE, get_engine

# The UCI Machine Learning Repository's permanent link for dataset #222.
DATA_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"

# Where we'll keep the downloaded zip so we never download it twice.
ZIP_PATH = DATA_RAW / "bank+marketing.zip"

# The specific file we want from inside the zip. The dataset ships several
# versions; we want the full 41k-row, 20-feature one the project calls for.
TARGET_CSV = "bank-additional-full.csv"

# Shown if the download fails, so you know exactly what to do by hand.
MANUAL_INSTRUCTIONS = f"""
------------------------------------------------------------------
AUTOMATIC DOWNLOAD FAILED - here is how to do it manually:

  1. Open this page in your browser:
       https://archive.ics.uci.edu/dataset/222/bank+marketing
  2. Click the "Download" button (top right). You'll get bank+marketing.zip
  3. Save that zip file to exactly this location:
       {ZIP_PATH}
  4. Re-run this script:
       python src/load_data.py

  (Don't unzip it yourself -- the script handles the unzipping.)
------------------------------------------------------------------
"""


def download_zip() -> bool:
    """
    Download the dataset zip if we don't already have it.
    Returns True on success, False if the download failed.
    """
    if ZIP_PATH.exists():
        size_mb = ZIP_PATH.stat().st_size / 1_000_000
        print(f"[1/5] Zip already downloaded ({size_mb:.1f} MB) - skipping download.")
        return True

    print(f"[1/5] Downloading from {DATA_URL} ...")
    try:
        # timeout: don't hang forever if UCI is slow or down.
        response = requests.get(DATA_URL, timeout=60)
        # Raises an error if the server replied 404, 500, etc. Without this,
        # we'd happily save an HTML error page and call it a zip file.
        response.raise_for_status()
        ZIP_PATH.write_bytes(response.content)
        print(f"      Saved {len(response.content) / 1_000_000:.1f} MB to {ZIP_PATH}")
        return True
    except Exception as exc:
        print(f"      Download failed: {type(exc).__name__}: {exc}")
        return False


def extract_csv() -> pd.DataFrame:
    """
    Pull bank-additional-full.csv out of the zip and return it as a DataFrame.

    A wrinkle worth knowing about: this zip has a zip INSIDE it. The outer
    bank+marketing.zip contains bank-additional.zip, which contains the CSV.
    So we may need to unzip twice. We search rather than hardcoding the path,
    because the exact folder layout inside UCI zips has changed over the years.
    """
    print("[2/5] Extracting CSV from zip ...")

    with zipfile.ZipFile(ZIP_PATH) as outer:
        # Case A: the CSV sits directly in the outer zip.
        for name in outer.namelist():
            if name.endswith(TARGET_CSV):
                print(f"      Found {name}")
                with outer.open(name) as f:
                    return _read_csv(f)

        # Case B: the CSV is inside a nested zip. Look for the inner zip,
        # read it into memory, and search that.
        for name in outer.namelist():
            if name.endswith(".zip"):
                # io.BytesIO wraps the raw bytes so zipfile can treat them as
                # a file -- it saves us writing a temp file to disk.
                inner_bytes = io.BytesIO(outer.read(name))
                with zipfile.ZipFile(inner_bytes) as inner:
                    for inner_name in inner.namelist():
                        if inner_name.endswith(TARGET_CSV):
                            print(f"      Found {name} -> {inner_name}")
                            with inner.open(inner_name) as f:
                                return _read_csv(f)

    raise FileNotFoundError(
        f"Could not find {TARGET_CSV} inside {ZIP_PATH}. "
        f"Zip contents: {zipfile.ZipFile(ZIP_PATH).namelist()}"
    )


def _read_csv(file_obj) -> pd.DataFrame:
    """
    Read the CSV with the right settings.

    sep=";"  -- this file is semicolon-separated, not comma-separated. That's
                a European convention. If you forget this, pandas reads every
                row as ONE giant column and nothing works.
    quotechar='"' -- text fields are wrapped in double quotes.
    """
    return pd.read_csv(file_obj, sep=";", quotechar='"')


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make the column names safe for SQL.

    WHY THIS MATTERS: five columns arrive named like "emp.var.rate" and
    "cons.price.idx". In SQL a dot means "table.column", so a column literally
    named emp.var.rate has to be quoted everywhere forever -- painful and
    error-prone. We rename dots to underscores now, once, and every query
    afterwards is clean. Fixing data problems at the boundary, as data enters
    the system, is much cheaper than working around them in every query.
    """
    print("[3/5] Cleaning column names ...")
    original = list(df.columns)

    df.columns = [
        col.strip().lower().replace(".", "_").replace(" ", "_") for col in df.columns
    ]

    renamed = [(o, n) for o, n in zip(original, df.columns) if o != n]
    for old, new in renamed:
        print(f"      {old:20s} -> {new}")
    if not renamed:
        print("      (no changes needed)")

    return df


def load_to_postgres(df: pd.DataFrame) -> None:
    """Write the DataFrame into PostgreSQL as a table."""
    print(f"[4/5] Loading {len(df):,} rows into table '{RAW_TABLE}' ...")
    engine = get_engine()

    # Drop the dependent view first.
    #
    # WHY THIS LINE EXISTS: to_sql(if_exists="replace") runs DROP TABLE. But
    # sql/03_feature_extraction.sql builds a VIEW on top of this table, and
    # PostgreSQL refuses to drop a table something else depends on:
    #
    #   DETAIL: view bank_marketing_features depends on table bank_marketing_raw
    #
    # This is a genuinely sneaky bug, because it CANNOT happen on a fresh
    # database -- the first run works perfectly and only the second one fails.
    # I claimed this script was safe to re-run, and it wasn't; running the full
    # pipeline twice is what caught it. That's the argument for testing the
    # rerun path rather than assuming it.
    #
    # CASCADE = "drop anything depending on this too". Safe here because the
    # view holds no data of its own -- it's a saved query, and step 4 of the
    # pipeline recreates it moments later.
    with engine.begin() as conn:
        conn.execute(text("DROP VIEW IF EXISTS bank_marketing_features CASCADE;"))

    df.to_sql(
        RAW_TABLE,
        engine,
        # "replace" = drop the table if it exists, then recreate it. This is
        # what makes the script safe to re-run. "append" would duplicate rows.
        if_exists="replace",
        # Don't write pandas' row-number index as a database column -- it's
        # meaningless here and would just be noise in every query.
        index=False,
        # Send rows in batches of 1000 in a single statement instead of one
        # INSERT per row. Turns ~41,000 round-trips into ~41.
        chunksize=1000,
        method="multi",
    )
    print("      Load complete.")


def verify(df: pd.DataFrame) -> None:
    """
    Ask PostgreSQL what it actually stored, and compare to what we sent.

    This is the whole point of verification: we don't trust that the load
    worked because the code didn't crash -- we go and check.
    """
    print("[5/5] Verifying load ...")
    engine = get_engine()

    with engine.connect() as conn:
        db_rows = conn.execute(text(f"SELECT COUNT(*) FROM {RAW_TABLE};")).scalar()
        db_cols = conn.execute(
            text(
                "SELECT COUNT(*) FROM information_schema.columns "
                "WHERE table_name = :t;"
            ),
            {"t": RAW_TABLE},
        ).scalar()

    print(f"      Rows    in pandas: {len(df):,}   in PostgreSQL: {db_rows:,}")
    print(f"      Columns in pandas: {len(df.columns)}      in PostgreSQL: {db_cols}")

    # assert = "this must be true; if it isn't, stop immediately and shout".
    # A loud failure now beats a silent one that you debug three phases later.
    assert db_rows == len(df), "Row count mismatch between pandas and PostgreSQL!"
    assert db_cols == len(df.columns), "Column count mismatch!"
    print("      Verified: PostgreSQL contents match the source CSV exactly.")


def main():
    if not download_zip():
        print(MANUAL_INSTRUCTIONS)
        # Exit code 1 tells the shell "this failed", which matters if you ever
        # chain this into an automated pipeline.
        raise SystemExit(1)

    df = extract_csv()
    print(f"      Read {len(df):,} rows x {len(df.columns)} columns")

    df = clean_columns(df)
    load_to_postgres(df)
    verify(df)

    print(f"\nDone. Raw data is in PostgreSQL table '{RAW_TABLE}'.")


if __name__ == "__main__":
    main()
