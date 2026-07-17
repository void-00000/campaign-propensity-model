-- ============================================================================
-- 02_RESPONSE_BY_SEGMENT.SQL
-- Purpose: Find WHO responds. Every query here asks the same question of a
--          different slice: "is this group better or worse than average?"
--
-- THE KEY IDEA - LIFT
-- The overall response rate is 11.27%. That's our "baseline": what you'd get
-- by calling people at random. For any group, lift = that group's response
-- rate / the baseline.
--     lift = 2.0  ->  this group responds twice as often as random calling
--     lift = 1.0  ->  no better than random; targeting them gains nothing
--     lift = 0.5  ->  half as good as random; calling them wastes money
--
-- Lift is the number a marketing manager actually cares about, because it maps
-- straight to money: 2x lift means the same call budget produces twice the
-- signups. We'll compute lift for the MODEL in Phase 5. Doing it by hand here
-- first means that chart won't be a mystery when we get there.
--
-- Run with:  python src/run_sql.py sql/02_response_by_segment.sql
-- ============================================================================


-- QUERY: Response rate and lift by job
-- A CTE ("Common Table Expression") is the WITH block below. Read it as
-- "define a temporary named result, then use it in the main query". It's just
-- a way to name an intermediate step so the final query reads cleanly instead
-- of nesting a subquery inside itself.
--
-- COUNT(*) FILTER (WHERE y = 'yes') is PostgreSQL's conditional count: count
-- rows, but only the ones matching this condition. It's the clean way to get
-- "how many of this group said yes" alongside "how big is this group" in a
-- single pass over the data.
WITH baseline AS (
    SELECT 100.0 * COUNT(*) FILTER (WHERE y = 'yes') / COUNT(*) AS rate
    FROM bank_marketing_raw
)
SELECT
    job,
    COUNT(*) AS contacted,
    COUNT(*) FILTER (WHERE y = 'yes') AS responded,
    ROUND(100.0 * COUNT(*) FILTER (WHERE y = 'yes') / COUNT(*), 2) AS response_rate_pct,
    -- CROSS JOIN baseline gives every row access to that single baseline number.
    ROUND(
        (100.0 * COUNT(*) FILTER (WHERE y = 'yes') / COUNT(*)) / MIN(baseline.rate),
        2
    ) AS lift_vs_baseline
FROM bank_marketing_raw
CROSS JOIN baseline
GROUP BY job
-- HAVING filters GROUPS (after aggregation); WHERE filters ROWS (before it).
-- We drop tiny groups because a 100% response rate from 3 people is noise,
-- not a finding.
HAVING COUNT(*) >= 100
ORDER BY response_rate_pct DESC;


-- QUERY: Response rate and lift by education
WITH baseline AS (
    SELECT 100.0 * COUNT(*) FILTER (WHERE y = 'yes') / COUNT(*) AS rate
    FROM bank_marketing_raw
)
SELECT
    education,
    COUNT(*) AS contacted,
    COUNT(*) FILTER (WHERE y = 'yes') AS responded,
    ROUND(100.0 * COUNT(*) FILTER (WHERE y = 'yes') / COUNT(*), 2) AS response_rate_pct,
    ROUND(
        (100.0 * COUNT(*) FILTER (WHERE y = 'yes') / COUNT(*)) / MIN(baseline.rate),
        2
    ) AS lift_vs_baseline
FROM bank_marketing_raw
CROSS JOIN baseline
GROUP BY education
HAVING COUNT(*) >= 100
ORDER BY response_rate_pct DESC;


-- QUERY: Response rate by contact month
-- Note what this does NOT tell us. A high rate in March could mean "March is a
-- good month to call" OR it could mean "we only called our best leads in March".
-- This is correlation, not causation, and the difference matters if someone
-- tries to act on it. We report the pattern; we don't promise it's the cause.
WITH baseline AS (
    SELECT 100.0 * COUNT(*) FILTER (WHERE y = 'yes') / COUNT(*) AS rate
    FROM bank_marketing_raw
)
SELECT
    month,
    COUNT(*) AS contacted,
    COUNT(*) FILTER (WHERE y = 'yes') AS responded,
    ROUND(100.0 * COUNT(*) FILTER (WHERE y = 'yes') / COUNT(*), 2) AS response_rate_pct,
    ROUND(
        (100.0 * COUNT(*) FILTER (WHERE y = 'yes') / COUNT(*)) / MIN(baseline.rate),
        2
    ) AS lift_vs_baseline
FROM bank_marketing_raw
CROSS JOIN baseline
GROUP BY month
ORDER BY response_rate_pct DESC;


-- QUERY: Response rate by contact method (cellular vs telephone)
WITH baseline AS (
    SELECT 100.0 * COUNT(*) FILTER (WHERE y = 'yes') / COUNT(*) AS rate
    FROM bank_marketing_raw
)
SELECT
    contact AS contact_method,
    COUNT(*) AS contacted,
    COUNT(*) FILTER (WHERE y = 'yes') AS responded,
    ROUND(100.0 * COUNT(*) FILTER (WHERE y = 'yes') / COUNT(*), 2) AS response_rate_pct,
    ROUND(
        (100.0 * COUNT(*) FILTER (WHERE y = 'yes') / COUNT(*)) / MIN(baseline.rate),
        2
    ) AS lift_vs_baseline
FROM bank_marketing_raw
CROSS JOIN baseline
GROUP BY contact
ORDER BY response_rate_pct DESC;


-- QUERY: Response rate by previous campaign outcome - the strongest segment
-- poutcome = how the PREVIOUS marketing campaign ended for this customer.
-- Expect this to be the sharpest split in the whole dataset. There's a plain
-- human reason: someone who bought from you before is far easier to sell to
-- again than a stranger. Watch the lift column.
WITH baseline AS (
    SELECT 100.0 * COUNT(*) FILTER (WHERE y = 'yes') / COUNT(*) AS rate
    FROM bank_marketing_raw
)
SELECT
    poutcome AS previous_outcome,
    COUNT(*) AS contacted,
    COUNT(*) FILTER (WHERE y = 'yes') AS responded,
    ROUND(100.0 * COUNT(*) FILTER (WHERE y = 'yes') / COUNT(*), 2) AS response_rate_pct,
    ROUND(
        (100.0 * COUNT(*) FILTER (WHERE y = 'yes') / COUNT(*)) / MIN(baseline.rate),
        2
    ) AS lift_vs_baseline
FROM bank_marketing_raw
CROSS JOIN baseline
GROUP BY poutcome
ORDER BY response_rate_pct DESC;


-- QUERY: Response rate by age band
-- Age is a number, but response probably isn't a straight line in age -- the
-- young and the retired may both behave unusually. Bucketing into bands with
-- CASE lets us SEE the shape instead of assuming it's linear.
WITH baseline AS (
    SELECT 100.0 * COUNT(*) FILTER (WHERE y = 'yes') / COUNT(*) AS rate
    FROM bank_marketing_raw
)
SELECT
    CASE
        WHEN age < 25 THEN '1. under 25'
        WHEN age < 35 THEN '2. 25-34'
        WHEN age < 45 THEN '3. 35-44'
        WHEN age < 55 THEN '4. 45-54'
        WHEN age < 65 THEN '5. 55-64'
        ELSE               '6. 65+'
    END AS age_band,
    COUNT(*) AS contacted,
    COUNT(*) FILTER (WHERE y = 'yes') AS responded,
    ROUND(100.0 * COUNT(*) FILTER (WHERE y = 'yes') / COUNT(*), 2) AS response_rate_pct,
    ROUND(
        (100.0 * COUNT(*) FILTER (WHERE y = 'yes') / COUNT(*)) / MIN(baseline.rate),
        2
    ) AS lift_vs_baseline
FROM bank_marketing_raw
CROSS JOIN baseline
GROUP BY age_band
ORDER BY age_band;


-- QUERY: Does calling someone repeatedly help? (campaign = calls this campaign)
-- A "diminishing returns" check. If response collapses after N calls, that's
-- directly actionable: stop calling after N and spend the time on fresh leads.
WITH baseline AS (
    SELECT 100.0 * COUNT(*) FILTER (WHERE y = 'yes') / COUNT(*) AS rate
    FROM bank_marketing_raw
)
SELECT
    CASE
        WHEN campaign = 1 THEN '1 call'
        WHEN campaign = 2 THEN '2 calls'
        WHEN campaign = 3 THEN '3 calls'
        WHEN campaign BETWEEN 4 AND 5 THEN '4-5 calls'
        ELSE '6+ calls'
    END AS calls_this_campaign,
    COUNT(*) AS contacted,
    COUNT(*) FILTER (WHERE y = 'yes') AS responded,
    ROUND(100.0 * COUNT(*) FILTER (WHERE y = 'yes') / COUNT(*), 2) AS response_rate_pct,
    ROUND(
        (100.0 * COUNT(*) FILTER (WHERE y = 'yes') / COUNT(*)) / MIN(baseline.rate),
        2
    ) AS lift_vs_baseline
FROM bank_marketing_raw
CROSS JOIN baseline
GROUP BY calls_this_campaign
ORDER BY calls_this_campaign;


-- QUERY: The 'duration' trap - why we must DELETE our most predictive column
-- duration = how many seconds the call lasted. Look at the response rates
-- below: they're extreme. duration is by far the strongest predictor here.
--
-- And we are going to throw it away. Here's why, and this is the single most
-- important idea in the entire project:
--
-- You only know how long a call lasted AFTER you've made the call. But the
-- model's whole job is to decide WHO TO CALL -- before any call happens. At
-- the moment you need the prediction, duration does not exist yet.
--
-- Worse, it partly IS the answer: a 20-minute call means the customer was
-- interested. Predicting "yes" from a long call is circular reasoning.
--
-- This is called TARGET LEAKAGE: a feature that secretly contains information
-- about the outcome, which won't be available when you actually deploy. It
-- produces a model that scores brilliantly in testing and is worthless in
-- production. The UCI dataset's own documentation warns about this column.
--
-- Interviewers love this. Spotting leakage unprompted is the difference
-- between someone who runs models and someone who understands them.
WITH baseline AS (
    SELECT 100.0 * COUNT(*) FILTER (WHERE y = 'yes') / COUNT(*) AS rate
    FROM bank_marketing_raw
)
SELECT
    CASE
        WHEN duration < 60  THEN '1. under 1 min'
        WHEN duration < 180 THEN '2. 1-3 min'
        WHEN duration < 300 THEN '3. 3-5 min'
        WHEN duration < 600 THEN '4. 5-10 min'
        ELSE                     '5. over 10 min'
    END AS call_length,
    COUNT(*) AS contacted,
    COUNT(*) FILTER (WHERE y = 'yes') AS responded,
    ROUND(100.0 * COUNT(*) FILTER (WHERE y = 'yes') / COUNT(*), 2) AS response_rate_pct,
    ROUND(
        (100.0 * COUNT(*) FILTER (WHERE y = 'yes') / COUNT(*)) / MIN(baseline.rate),
        2
    ) AS lift_vs_baseline
FROM bank_marketing_raw
CROSS JOIN baseline
GROUP BY call_length
ORDER BY call_length;
