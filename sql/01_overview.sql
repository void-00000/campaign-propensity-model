-- ============================================================================
-- 01_OVERVIEW.SQL
-- Purpose: Before analysing anything, find out what we actually have.
--          Shape, target balance, and data-quality problems.
--
-- Run with:  python src/run_sql.py sql/01_overview.sql
-- ============================================================================


-- QUERY: Table structure - what columns exist and what type is each?
-- information_schema is a set of tables PostgreSQL maintains ABOUT your tables.
-- Asking the database to describe itself is the fastest way to orient yourself
-- in an unfamiliar dataset.
SELECT
    ordinal_position AS col_num,
    column_name,
    data_type
FROM information_schema.columns
WHERE table_name = 'bank_marketing_raw'
ORDER BY ordinal_position;


-- QUERY: Class balance - what share of customers actually said yes?
-- This single number drives most of the decisions in this whole project.
--
-- The window function trick: SUM(COUNT(*)) OVER () means "the total count
-- across ALL groups". Without OVER(), COUNT(*) is per-group (per y value);
-- OVER() with empty parentheses says "now sum that across every group".
-- It gives us a percentage-of-total without a second query or a subquery.
SELECT
    y AS response,
    COUNT(*) AS customers,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_of_total
FROM bank_marketing_raw
GROUP BY y
ORDER BY customers DESC;


-- QUERY: Are there any NULL (missing) values?
-- COUNT(*) counts rows; COUNT(column) counts only NON-NULL values in it.
-- So the difference between them is exactly the number of NULLs.
SELECT
    COUNT(*) - COUNT(age)       AS null_age,
    COUNT(*) - COUNT(job)       AS null_job,
    COUNT(*) - COUNT(education) AS null_education,
    COUNT(*) - COUNT(duration)  AS null_duration,
    COUNT(*) - COUNT(euribor3m) AS null_euribor3m,
    COUNT(*) - COUNT(y)         AS null_y
FROM bank_marketing_raw;


-- QUERY: The REAL missing data - columns using the literal string 'unknown'
-- This dataset has almost no NULLs, which looks great until you notice it
-- encodes missing values as the text 'unknown' instead. A naive check would
-- report "no missing data!" and be completely wrong.
--
-- This is the single most useful habit in this file: never trust a clean NULL
-- check. Go look at what the placeholder actually is.
SELECT 'job'       AS column_name, COUNT(*) FILTER (WHERE job = 'unknown')       AS unknown_count, ROUND(100.0 * COUNT(*) FILTER (WHERE job = 'unknown') / COUNT(*), 2) AS unknown_pct FROM bank_marketing_raw
UNION ALL
SELECT 'marital',   COUNT(*) FILTER (WHERE marital = 'unknown'),   ROUND(100.0 * COUNT(*) FILTER (WHERE marital = 'unknown') / COUNT(*), 2) FROM bank_marketing_raw
UNION ALL
SELECT 'education', COUNT(*) FILTER (WHERE education = 'unknown'), ROUND(100.0 * COUNT(*) FILTER (WHERE education = 'unknown') / COUNT(*), 2) FROM bank_marketing_raw
UNION ALL
SELECT 'default',   COUNT(*) FILTER (WHERE "default" = 'unknown'), ROUND(100.0 * COUNT(*) FILTER (WHERE "default" = 'unknown') / COUNT(*), 2) FROM bank_marketing_raw
UNION ALL
SELECT 'housing',   COUNT(*) FILTER (WHERE housing = 'unknown'),   ROUND(100.0 * COUNT(*) FILTER (WHERE housing = 'unknown') / COUNT(*), 2) FROM bank_marketing_raw
UNION ALL
SELECT 'loan',      COUNT(*) FILTER (WHERE loan = 'unknown'),      ROUND(100.0 * COUNT(*) FILTER (WHERE loan = 'unknown') / COUNT(*), 2) FROM bank_marketing_raw
ORDER BY unknown_count DESC;


-- QUERY: Numeric columns - do the ranges look sane?
-- Every dataset has at least one column whose values are impossible. Looking at
-- min/max early is how you find them before they poison your model.
-- Watch the pdays row in particular; it is about to look very strange.
SELECT 'age'      AS metric, MIN(age)::numeric      AS min_val, ROUND(AVG(age), 1)      AS avg_val, MAX(age)::numeric      AS max_val FROM bank_marketing_raw
UNION ALL
SELECT 'duration',           MIN(duration)::numeric, ROUND(AVG(duration), 1),           MAX(duration)::numeric FROM bank_marketing_raw
UNION ALL
SELECT 'campaign',           MIN(campaign)::numeric, ROUND(AVG(campaign), 1),           MAX(campaign)::numeric FROM bank_marketing_raw
UNION ALL
SELECT 'pdays',              MIN(pdays)::numeric,    ROUND(AVG(pdays), 1),              MAX(pdays)::numeric FROM bank_marketing_raw
UNION ALL
SELECT 'previous',           MIN(previous)::numeric, ROUND(AVG(previous), 1),           MAX(previous)::numeric FROM bank_marketing_raw
UNION ALL
SELECT 'euribor3m',          MIN(euribor3m)::numeric, ROUND(AVG(euribor3m)::numeric, 2), MAX(euribor3m)::numeric FROM bank_marketing_raw;


-- QUERY: The pdays mystery - decoding a sentinel value
-- pdays = "days since this customer was last contacted in a PREVIOUS campaign".
-- Its max is 999, which is not a real number of days (that's 2.7 years) - it's
-- a SENTINEL: a magic value meaning "never previously contacted".
--
-- This matters enormously. If we feed 999 to a model as a number, the model
-- thinks these customers were contacted a very long time ago, rather than never
-- at all. Averaging it is meaningless. In Phase 2 we split this into a proper
-- yes/no flag plus a real recency number.
SELECT
    CASE WHEN pdays = 999 THEN 'never contacted before (999)'
         ELSE 'contacted before (real day count)' END AS pdays_meaning,
    COUNT(*) AS customers,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_of_total,
    ROUND(100.0 * COUNT(*) FILTER (WHERE y = 'yes') / COUNT(*), 2) AS response_rate_pct
FROM bank_marketing_raw
GROUP BY 1
ORDER BY customers DESC;
