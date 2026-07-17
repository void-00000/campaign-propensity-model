-- ============================================================================
-- 03_FEATURE_EXTRACTION.SQL
-- Purpose: Turn the raw table into a clean, model-ready feature table.
--
-- THIS IS THE "SQL FOR FEATURE EXTRACTION" FILE. Everything the database can
-- do well -- decoding sentinels, bucketing, flags, ratios, comparisons against
-- an aggregate -- happens here, close to the data. Python then handles what
-- Python does well (one-hot encoding, scaling, model fitting).
--
-- WHAT IS A VIEW?
-- A view is a saved query that behaves like a table. It stores no data of its
-- own -- every time you SELECT from it, PostgreSQL re-runs the query beneath.
-- Two reasons that's the right choice here:
--   1. It can never go stale. Reload the raw table and the view is already
--      correct, because it's just a question, not a copy.
--   2. The feature definitions live in ONE place. The dashboard, the training
--      script, and any analyst all read the same logic. If "recent contact"
--      has to mean the same thing everywhere, define it once.
--
-- Run with:  python src/run_sql.py sql/03_feature_extraction.sql
-- ============================================================================


-- QUERY: Build the model-ready feature view
-- CREATE OR REPLACE means re-running this file is safe -- it overwrites the
-- old definition instead of erroring that the view already exists.
CREATE OR REPLACE VIEW bank_marketing_features AS
WITH stats AS (
    -- One row of dataset-wide averages. We use these below to express a
    -- customer's economic conditions RELATIVE to typical conditions, which is
    -- more meaningful to a model than a raw index number floating in space.
    SELECT
        AVG(euribor3m)     AS avg_euribor,
        AVG(cons_conf_idx) AS avg_conf,
        AVG(nr_employed)   AS avg_employed
    FROM bank_marketing_raw
)
SELECT
    -- ==================== RAW PASS-THROUGH ====================
    -- Kept as-is; genuinely useful and already in a sensible form.
    r.age,
    r.job,
    r.marital,
    r.education,
    r."default"   AS credit_default,   -- "default" is a reserved SQL keyword, so
                                       -- it must be quoted. We rename it here so
                                       -- nothing downstream has to keep quoting it.
    r.housing,
    r.loan,
    r.contact,
    r.month,
    r.day_of_week,
    r.poutcome,
    r.campaign,
    r.previous,

    -- ==================== ENGINEERED: RECENCY ====================
    -- Fixing the pdays sentinel we found in 01_overview.sql.
    --
    -- F1: was_contacted_before -- the yes/no fact hiding inside pdays=999.
    -- Splitting one polluted column into a flag plus a clean number lets the
    -- model learn "have we ever spoken?" separately from "how long ago?".
    -- Those are two different questions and they deserve two features.
    CASE WHEN r.pdays = 999 THEN 0 ELSE 1 END AS was_contacted_before,

    -- F2: days_since_prev_contact -- real recency, NULL when never contacted.
    -- NULL is the honest answer for "how long since a thing that never
    -- happened". XGBoost handles NULLs natively; it learns its own best route
    -- for missing values rather than being fed a fake number like 999 or -1
    -- that it would wrongly treat as "a very long time ago".
    CASE WHEN r.pdays = 999 THEN NULL ELSE r.pdays END AS days_since_prev_contact,

    -- F3: recency_bucket -- recency as coarse bands.
    -- Buckets let a model use a non-linear effect without having to discover
    -- the shape itself. Fewer than 7 days is a hot lead; 15+ days is cooling.
    CASE
        WHEN r.pdays = 999      THEN 'never'
        WHEN r.pdays <= 7       THEN 'within_week'
        WHEN r.pdays <= 14      THEN 'within_fortnight'
        ELSE                         'over_fortnight'
    END AS recency_bucket,

    -- ==================== ENGINEERED: PRIOR RELATIONSHIP ====================
    -- F4: prev_success -- did this customer buy from us LAST time?
    -- Segment query [5] showed poutcome='success' responds at 65% (5.8x lift).
    -- That's the strongest single signal in the dataset, so we give the model
    -- a dedicated 0/1 column instead of hoping it finds it inside poutcome.
    CASE WHEN r.poutcome = 'success' THEN 1 ELSE 0 END AS prev_success,

    -- F5: prev_failure -- rejected us before. Distinct from never-contacted,
    -- and the data says so: 14.2% response vs 8.8%. Counter-intuitive, but
    -- "said no once" still beats "complete stranger" -- they're a known lead.
    CASE WHEN r.poutcome = 'failure' THEN 1 ELSE 0 END AS prev_failure,

    -- F6: total_contacts -- lifetime touches across all campaigns.
    r.campaign + r.previous AS total_contacts,

    -- F7: prior_contact_ratio -- what share of contact happened BEFORE now?
    -- Near 0 = brand new lead. Higher = an established relationship.
    -- The ::numeric cast forces decimal division; without it PostgreSQL does
    -- integer division and 1/3 silently becomes 0. A classic, quiet SQL bug.
    ROUND(
        r.previous::numeric / NULLIF(r.campaign + r.previous, 0),
        4
    ) AS prior_contact_ratio,

    -- ==================== ENGINEERED: CONTACT INTENSITY ====================
    -- F8: contact_fatigue -- query [7] showed response falling from 13.0% at
    -- one call to 5.5% at six or more. Past ~3 calls we're annoying people.
    -- This flag hands the model that threshold directly.
    CASE WHEN r.campaign > 3 THEN 1 ELSE 0 END AS contact_fatigue,

    -- F9: campaign_intensity_bucket -- the same effect, in bands.
    CASE
        WHEN r.campaign = 1        THEN 'single_call'
        WHEN r.campaign <= 3       THEN 'light_touch'
        WHEN r.campaign <= 5       THEN 'persistent'
        ELSE                            'excessive'
    END AS campaign_intensity_bucket,

    -- ==================== ENGINEERED: CHANNEL & TIMING ====================
    -- F10: is_cellular -- query [4]: cellular 14.7% vs landline 5.2% (2.8x
    -- apart). Likely a proxy for age and reachability rather than a magic
    -- property of mobile phones, but the signal is real either way.
    CASE WHEN r.contact = 'cellular' THEN 1 ELSE 0 END AS is_cellular,

    -- F11: is_high_season -- query [3]: mar/dec/sep/oct all clear 43%, while
    -- may sits at 6.4%. Almost certainly reflects WHO was called in those
    -- months rather than the months themselves -- but as a predictor it works.
    CASE WHEN r.month IN ('mar', 'dec', 'sep', 'oct') THEN 1 ELSE 0 END AS is_high_season,

    -- F12: quarter -- coarser seasonality, less prone to overfitting on one
    -- month that happened to go well in this particular dataset.
    CASE
        WHEN r.month IN ('jan', 'feb', 'mar') THEN 'Q1'
        WHEN r.month IN ('apr', 'may', 'jun') THEN 'Q2'
        WHEN r.month IN ('jul', 'aug', 'sep') THEN 'Q3'
        ELSE                                       'Q4'
    END AS quarter,

    -- ==================== ENGINEERED: DEMOGRAPHIC ====================
    -- F13: age_band -- query [6] found a U-shape: under-25s at 24% and 65+ at
    -- 47%, but the 35-54 middle at only 8.7%. A straight line through age
    -- would average that away to nothing. Logistic Regression especially
    -- cannot see a U-shape in a raw number; bands let it.
    CASE
        WHEN r.age < 25 THEN 'under_25'
        WHEN r.age < 35 THEN 'age_25_34'
        WHEN r.age < 45 THEN 'age_35_44'
        WHEN r.age < 55 THEN 'age_45_54'
        WHEN r.age < 65 THEN 'age_55_64'
        ELSE                 'age_65_plus'
    END AS age_band,

    -- F14: is_life_stage_edge -- the two ends of that U-shape as one flag.
    -- Plausible story: students and retirees have time to listen and a reason
    -- to care about a savings product.
    CASE WHEN r.age < 25 OR r.age >= 65 THEN 1 ELSE 0 END AS is_life_stage_edge,

    -- F15: has_any_loan -- existing debt load, housing or personal.
    -- 'unknown' is deliberately NOT counted as yes. We don't invent facts.
    CASE
        WHEN r.housing = 'yes' OR r.loan = 'yes' THEN 1
        ELSE 0
    END AS has_any_loan,

    -- F16: education_years -- ordinal encoding.
    -- education has a real ORDER to it (4y < 6y < 9y < high school < degree),
    -- which one-hot encoding would throw away by treating each level as an
    -- unrelated category. Mapping to approximate years keeps the ordering.
    -- 'unknown' becomes NULL rather than a guessed number -- again, we don't
    -- invent data to fill a hole.
    CASE r.education
        WHEN 'illiterate'          THEN 0
        WHEN 'basic.4y'            THEN 4
        WHEN 'basic.6y'            THEN 6
        WHEN 'basic.9y'            THEN 9
        WHEN 'high.school'         THEN 12
        WHEN 'professional.course' THEN 14
        WHEN 'university.degree'   THEN 16
        ELSE NULL
    END AS education_years,

    -- ==================== ENGINEERED: ECONOMIC CONTEXT ====================
    -- These five columns describe the ECONOMY at call time, not the customer.
    -- They matter because "will you lock money into a term deposit?" depends
    -- heavily on interest rates and whether people feel secure in their jobs.
    r.emp_var_rate,
    r.cons_price_idx,
    r.cons_conf_idx,
    r.euribor3m,
    r.nr_employed,

    -- F17: euribor_vs_avg -- this customer's interest rate divided by the
    -- dataset average. Above 1 = rates were high when we called. Ratios are
    -- easier for both models and humans to reason about than raw index values,
    -- because the units cancel out and 1.0 always means "typical".
    ROUND((r.euribor3m / s.avg_euribor)::numeric, 4) AS euribor_vs_avg,

    -- F18: confidence_vs_avg -- same idea for consumer confidence.
    -- Note the values are negative, so we subtract instead of divide: a ratio
    -- of two negatives would be confusing and would flip sign unhelpfully.
    ROUND((r.cons_conf_idx - s.avg_conf)::numeric, 4) AS confidence_vs_avg,

    -- F19: is_low_rate_period -- rates below average. A term deposit is a bet
    -- on future rates, so the rate environment plausibly drives the decision.
    CASE WHEN r.euribor3m < s.avg_euribor THEN 1 ELSE 0 END AS is_low_rate_period,

    -- F20: employment_vs_avg -- job-market strength relative to typical.
    ROUND((r.nr_employed - s.avg_employed)::numeric, 2) AS employment_vs_avg,

    -- ==================== TARGET ====================
    -- The thing we're predicting, as 1/0 rather than 'yes'/'no'.
    -- Every ML library expects numbers, and by convention the rare/positive
    -- class is 1. Converting once here means no script downstream has to
    -- remember which way round it goes.
    CASE WHEN r.y = 'yes' THEN 1 ELSE 0 END AS target

    -- ==================== DELIBERATELY EXCLUDED ====================
    -- r.duration is NOT here. It's the strongest predictor in the dataset and
    -- it is unusable: you only learn a call's length by making the call, but
    -- this model runs BEFORE the call to decide who to ring. See the leakage
    -- discussion in 02_response_by_segment.sql, query [8].
    --
    -- r.pdays is NOT here either -- it's been decomposed into F1/F2/F3 above,
    -- which say the same thing without the 999 landmine.

FROM bank_marketing_raw r
CROSS JOIN stats s;


-- QUERY: Confirm the view exists and check its feature count
SELECT
    COUNT(*) AS total_columns,
    COUNT(*) - 1 AS feature_columns  -- minus the target itself
FROM information_schema.columns
WHERE table_name = 'bank_marketing_features';


-- QUERY: List every column in the feature view
SELECT
    ordinal_position AS col_num,
    column_name,
    data_type
FROM information_schema.columns
WHERE table_name = 'bank_marketing_features'
ORDER BY ordinal_position;


-- QUERY: Sanity-check the engineered features against the raw data
-- Never trust a feature you haven't looked at. These should match the numbers
-- we saw in 01 and 02 -- if they don't, the view has a bug and we want to know
-- now, not after we've trained a model on it.
SELECT
    COUNT(*) AS rows_in_view,
    SUM(target) AS total_responders,
    ROUND(100.0 * SUM(target) / COUNT(*), 2) AS response_rate_pct,
    SUM(was_contacted_before) AS ever_contacted_before,
    SUM(prev_success) AS prior_buyers,
    SUM(is_cellular) AS reached_on_mobile,
    SUM(is_high_season) AS called_in_high_season,
    COUNT(days_since_prev_contact) AS non_null_recency,
    COUNT(education_years) AS non_null_education_years
FROM bank_marketing_features;


-- QUERY: Do the new engineered flags actually separate responders?
-- A feature is only worth having if the response rate DIFFERS across its
-- values. If a flag shows 11% either way, it's dead weight -- it knows nothing
-- the baseline didn't already know. This query checks each flag earns its seat.
SELECT 'prev_success'        AS feature, prev_success::text        AS value, COUNT(*) AS customers, ROUND(100.0 * AVG(target), 2) AS response_rate_pct FROM bank_marketing_features GROUP BY prev_success
UNION ALL
SELECT 'was_contacted_before', was_contacted_before::text, COUNT(*), ROUND(100.0 * AVG(target), 2) FROM bank_marketing_features GROUP BY was_contacted_before
UNION ALL
SELECT 'contact_fatigue',      contact_fatigue::text,      COUNT(*), ROUND(100.0 * AVG(target), 2) FROM bank_marketing_features GROUP BY contact_fatigue
UNION ALL
SELECT 'is_cellular',          is_cellular::text,          COUNT(*), ROUND(100.0 * AVG(target), 2) FROM bank_marketing_features GROUP BY is_cellular
UNION ALL
SELECT 'is_high_season',       is_high_season::text,       COUNT(*), ROUND(100.0 * AVG(target), 2) FROM bank_marketing_features GROUP BY is_high_season
UNION ALL
SELECT 'is_life_stage_edge',   is_life_stage_edge::text,   COUNT(*), ROUND(100.0 * AVG(target), 2) FROM bank_marketing_features GROUP BY is_life_stage_edge
UNION ALL
SELECT 'has_any_loan',         has_any_loan::text,         COUNT(*), ROUND(100.0 * AVG(target), 2) FROM bank_marketing_features GROUP BY has_any_loan
UNION ALL
SELECT 'is_low_rate_period',   is_low_rate_period::text,   COUNT(*), ROUND(100.0 * AVG(target), 2) FROM bank_marketing_features GROUP BY is_low_rate_period
ORDER BY feature, value;


-- QUERY: Recency bucket - does our decoded pdays feature work?
SELECT
    recency_bucket,
    COUNT(*) AS customers,
    ROUND(100.0 * AVG(target), 2) AS response_rate_pct
FROM bank_marketing_features
GROUP BY recency_bucket
ORDER BY response_rate_pct DESC;
