"""
Central configuration for the project.

WHY THIS FILE EXISTS
Every later script (loading data, running SQL, training the model, the dashboard)
needs to know two things: where the project folders are, and how to reach the
database. Rather than repeat that knowledge in five files -- where it would
eventually drift out of sync -- we define it once, here, and import it.

RUN THIS FILE DIRECTLY to test your database connection:
    python src/config.py
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# --------------------------------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------------------------------
# __file__ is the path of THIS file (src/config.py).
# .resolve() turns it into a full, unambiguous path.
# .parents[1] goes up one level: src/config.py -> src/ -> the project root.
#
# We do this instead of writing "C:\Users\annub\..." because a hardcoded path
# breaks the moment the project is opened on another machine. This way the
# project locates itself, and the scripts work no matter which folder you run
# them from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
SQL_DIR = PROJECT_ROOT / "sql"
OUTPUTS = PROJECT_ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
MODELS = OUTPUTS / "models"   # trained models saved here (Phase 4)

# Make sure these folders exist. exist_ok=True means "don't error if it's
# already there", so this is safe to run repeatedly.
for _folder in (DATA_RAW, DATA_PROCESSED, OUTPUTS, FIGURES, MODELS):
    _folder.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# DATABASE CONNECTION
# --------------------------------------------------------------------------
# load_dotenv() reads the .env file and loads its KEY=VALUE pairs into the
# environment, so os.getenv("POSTGRES_USER") below returns "bank".
load_dotenv(PROJECT_ROOT / ".env")

DB_USER = os.getenv("POSTGRES_USER", "bank")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "bankpass")
DB_NAME = os.getenv("POSTGRES_DB", "bank_campaign")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")

# The name of the table we'll load the raw CSV into (Phase 1).
RAW_TABLE = "bank_marketing_raw"


def _normalise_url(url: str) -> str:
    """
    Take a connection URL from anywhere and make it work with our driver.

    Cloud providers (Neon, Supabase, Heroku) hand you a URL starting with
    'postgresql://'. SQLAlchemy needs to know WHICH driver to use, which is the
    '+psycopg2' part. Same database, same credentials -- we're just naming the
    driver. Providers can't include it because they don't know what language
    you're connecting from.

    We also force SSL. Neon requires an encrypted connection and rejects
    plaintext ones; a local Docker container on your own machine doesn't need it.
    Rather than make you remember that, we add it when connecting to anything
    that isn't localhost.
    """
    # postgres:// is a legacy alias some providers still emit. SQLAlchemy 2.0
    # dropped support for it, so translate it rather than let it fail cryptically.
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)

    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)

    is_local = "@localhost" in url or "@127.0.0.1" in url
    if not is_local and "sslmode=" not in url:
        url += "&sslmode=require" if "?" in url else "?sslmode=require"

    return url


def get_database_url() -> str:
    """
    Work out which database to talk to, checking three places in priority order.

    WHY THREE? Because the same code has to run in three different places:
      1. DATABASE_URL environment variable -- production, CI, or when you want
         to point the local scripts at the cloud database for a one-off load.
      2. Streamlit secrets -- how the deployed dashboard gets its credentials.
         Streamlit Cloud has no .env file; you paste secrets into its UI and
         they arrive as st.secrets.
      3. The local .env file -- your Docker container. The everyday default.

    The alternative is a hardcoded connection string plus a comment saying
    "change this before deploying", which someone always forgets. Config that
    adapts to its environment is config you can't forget to change.
    """
    # 1. Explicit environment variable wins. Nothing overrides an operator
    #    deliberately setting this.
    url = os.getenv("DATABASE_URL")
    if url:
        return _normalise_url(url)

    # 2. Streamlit secrets, but only if we're actually running inside Streamlit.
    #    The import is inside a try because the training scripts don't have
    #    streamlit as a hard requirement, and reading secrets outside a
    #    Streamlit runtime raises rather than returning empty.
    try:
        import streamlit as st

        if "DATABASE_URL" in st.secrets:
            return _normalise_url(st.secrets["DATABASE_URL"])
    except Exception:
        pass

    # 3. Fall back to the local .env pieces -- the Docker container.
    #    Read it as: using the postgresql driver 'psycopg2', log in as DB_USER
    #    with DB_PASSWORD, to the server at DB_HOST:DB_PORT, use database DB_NAME.
    return (
        f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )


# Kept as a module-level name because several scripts import it directly.
DATABASE_URL = get_database_url()


def get_engine():
    """
    Return a SQLAlchemy 'engine' -- the object that manages connections to
    PostgreSQL. Think of it as a phone line to the database that our other
    scripts pick up whenever they need to run a query.
    """
    # Resolved at call time, not import time, so a script that sets
    # DATABASE_URL after importing config still gets the right database.
    return create_engine(
        get_database_url(),
        # Cloud databases drop idle connections; a pooled connection that died
        # quietly produces a baffling error on the next query. pre_ping sends a
        # cheap "are you alive?" first and transparently reconnects if not.
        # Harmless locally, essential against Neon.
        pool_pre_ping=True,
    )


# This block only runs when you execute "python src/config.py" directly.
# It does NOT run when another script imports this file. That's the standard
# Python idiom for "give this module a self-check you can run by hand".
def describe_connection() -> str:
    """
    A human-readable description of the target, with the password stripped out.

    Once you have both a local and a cloud database, "which one am I actually
    talking to?" becomes a real question -- and the most expensive way to answer
    it is to load 41,000 rows into the wrong one. So we say it out loud.
    """
    url = get_database_url()

    # Strip credentials before anything prints or logs this. A password that
    # reaches a terminal tends to reach a screenshot.
    safe = url
    if "@" in url:
        scheme, rest = url.split("://", 1)
        if "@" in rest:
            creds, host = rest.split("@", 1)
            user = creds.split(":")[0]
            safe = f"{scheme}://{user}:***@{host}"

    if "localhost" in url or "127.0.0.1" in url:
        where = "LOCAL Docker container"
    elif "neon.tech" in url:
        where = "NEON (cloud)"
    elif "supabase" in url:
        where = "SUPABASE (cloud)"
    else:
        where = "remote database"

    return f"{where}\n  {safe}"


if __name__ == "__main__":
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Target       : {describe_connection()}")

    try:
        engine = get_engine()
        with engine.connect() as conn:
            # A trivial query whose only job is to prove the round-trip works.
            version = conn.execute(text("SELECT version();")).scalar()
            # Does the data actually exist here? A live connection to an EMPTY
            # database still connects fine -- and that's exactly the confusing
            # state you land in halfway through setting up a second one.
            table_exists = conn.execute(text(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
                "WHERE table_name = :t);"
            ), {"t": RAW_TABLE}).scalar()
            view_exists = conn.execute(text(
                "SELECT EXISTS (SELECT 1 FROM information_schema.views "
                "WHERE table_name = 'bank_marketing_features');"
            )).scalar()
            rows = None
            if table_exists:
                rows = conn.execute(
                    text(f"SELECT COUNT(*) FROM {RAW_TABLE};")
                ).scalar()

        print("\nSUCCESS - PostgreSQL replied:")
        print(f"  {version}")
        print("\nData check:")
        print(f"  table '{RAW_TABLE}'  : "
              f"{f'{rows:,} rows' if table_exists else 'MISSING'}")
        print(f"  view  'bank_marketing_features' : "
              f"{'present' if view_exists else 'MISSING'}")

        if not table_exists or not view_exists:
            print("\n  This database is connected but not loaded. Run:")
            print("      python src/load_data.py")
            print("      python src/run_sql.py sql/03_feature_extraction.sql")
    except Exception as exc:
        print("\nFAILED to connect to PostgreSQL.")
        print(f"  {type(exc).__name__}: {exc}")
        print("\n  If you meant the LOCAL database:  docker compose up -d")
        print("  If you meant a CLOUD database, check DATABASE_URL is set.")
