"""
A tiny helper that runs a .sql file and prints each query's results as a table.

WHY THIS EXISTS
Our analysis lives in real .sql files (in the sql/ folder), not buried inside
Python strings. That's deliberate: a .sql file can be opened in any database
tool, handed to a data engineer, or reviewed on its own. Keeping SQL in SQL
files is how it's done on real teams.

This script is just the thing that runs them and shows you the answers.

RUN IT WITH:
    python src/run_sql.py sql/01_overview.sql
"""

import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from config import SQL_DIR, get_engine


def strip_comments(sql_text: str) -> str:
    """
    Remove '--' comments, leaving only executable SQL.

    In SQL, '--' means "ignore the rest of this line". We need this for two
    reasons: to find where a statement really ends, and to identify what kind
    of statement it is. A statement that begins with a comment still begins
    with SELECT as far as the database is concerned.
    """
    return "\n".join(line.split("--", 1)[0] for line in sql_text.splitlines())


def split_statements(sql_text: str) -> list[str]:
    """
    Split a .sql file into individual queries on the semicolon.

    THE SUBTLETY: we must only split on semicolons that are real SQL, not ones
    sitting inside a comment. (I learned this the direct way -- the first
    version of this function split on every ';' and happily tore a query in
    half because I'd written a semicolon inside an explanatory comment.)

    So we scan line by line, and only treat a ';' as an end-of-statement marker
    if it survives having comments stripped from that line.

    Known limitation, stated rather than hidden: a ';' or '--' inside a quoted
    string literal would still fool this. Our queries contain none, and a
    proper SQL parser is a big dependency to take on for a 20-line helper.
    """
    statements: list[str] = []
    current: list[str] = []

    for line in sql_text.splitlines():
        current.append(line)
        if ";" in strip_comments(line):
            statements.append("\n".join(current).strip())
            current = []

    # Anything after the final semicolon (e.g. a trailing comment block).
    tail = "\n".join(current).strip()
    if tail:
        statements.append(tail)

    # Drop chunks that are pure commentary with no actual SQL in them, such as
    # the banner at the top of each file.
    return [s for s in statements if strip_comments(s).strip().strip(";")]


def extract_title(statement: str) -> str:
    """
    Pull the title out of a query's leading comment.

    We write each query with a header like:
        -- QUERY: Response rate by job
    so the printed output is self-labelling and readable.
    """
    for line in statement.splitlines():
        line = line.strip()
        if line.upper().startswith("-- QUERY:"):
            return line.split(":", 1)[1].strip()
    return "(untitled query)"


def run_file(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"No such SQL file: {path}")

    engine = get_engine()
    statements = split_statements(path.read_text())

    print(f"\n{'=' * 78}")
    print(f"  {path.name}  ({len(statements)} queries)")
    print(f"{'=' * 78}")

    for i, statement in enumerate(statements, start=1):
        title = extract_title(statement)
        print(f"\n--- [{i}] {title} ---\n")

        # Decide whether this statement returns rows. We check the first word
        # of the REAL SQL, ignoring any comment block above it.
        code = strip_comments(statement).strip()
        first_word = code.split()[0].upper()
        returns_rows = first_word in ("SELECT", "WITH", "TABLE", "VALUES")

        if returns_rows:
            df = pd.read_sql(text(statement), engine)
            # to_string() prints every row rather than pandas' truncated view.
            print(df.to_string(index=False))
            print(f"\n({len(df)} rows)")
        else:
            # CREATE VIEW, DROP, etc. return no result set to display.
            # begin() opens a transaction and commits it when the block exits.
            with engine.begin() as conn:
                conn.execute(text(statement))
            print(f"OK - executed ({first_word})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # No argument given: list what's available rather than just erroring.
        print("Usage: python src/run_sql.py <path-to-sql-file>\n")
        print("Available SQL files:")
        for f in sorted(SQL_DIR.glob("*.sql")):
            print(f"  sql/{f.name}")
        raise SystemExit(1)

    run_file(Path(sys.argv[1]))
