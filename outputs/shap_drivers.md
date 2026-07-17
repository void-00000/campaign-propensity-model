# SHAP: What Drives Campaign Response

Generated from the XGBoost model, explained on the held-out test set
(8,238 customers the model never saw during training).

## How to read this

**SHAP value** = how much a feature pushed one customer's score away
from the average customer's score. **Importance** below is the mean
absolute SHAP value: the average size of that push, ignoring direction.

> **Important caveat.** Mean |SHAP| answers *"how much does this feature
> move the AVERAGE customer?"* -- **not** *"how much does it matter to a
> customer it applies to?"* A feature that is decisive for 3% of people
> and zero for the rest gets averaged down to near nothing. See the
> `prev_success` discussion below.

## Drivers by theme

Several of our features encode overlapping facts, and SHAP splits credit
between correlated features. Group totals are the fairer comparison.

| Theme | Total importance | Share | Features | Top feature |
|---|---|---|---|---|
| Macroeconomic conditions | 0.9350 | 54.0% | 6 | `nr_employed` |
| Demographics | 0.2668 | 15.4% | 43 | `age` |
| Prior relationship / recency | 0.1759 | 10.2% | 14 | `days_since_prev_contact` |
| Timing (month / day / season) | 0.1639 | 9.5% | 20 | `day_of_week_mon` |
| Contact channel | 0.1507 | 8.7% | 3 | `is_cellular` |
| Campaign intensity | 0.0404 | 2.3% | 6 | `campaign` |

## Ranked individual drivers (top 20)

| # | Feature | Importance | Effect |
|---|---|---|---|
| 1 | `nr_employed` | 0.4626 | higher -> LESS likely to say yes |
| 2 | `is_low_rate_period` | 0.1469 | higher -> MORE likely to say yes |
| 3 | `euribor3m` | 0.1220 | higher -> LESS likely to say yes |
| 4 | `is_cellular` | 0.1139 | higher -> MORE likely to say yes |
| 5 | `emp_var_rate` | 0.1028 | higher -> LESS likely to say yes |
| 6 | `cons_conf_idx` | 0.0589 | higher -> LESS likely to say yes |
| 7 | `age` | 0.0481 | higher -> LESS likely to say yes |
| 8 | `days_since_prev_contact` | 0.0454 | mixed / non-linear |
| 9 | `cons_price_idx` | 0.0418 | higher -> LESS likely to say yes |
| 10 | `credit_default_no` | 0.0390 | higher -> MORE likely to say yes |
| 11 | `contact_cellular` | 0.0369 | higher -> MORE likely to say yes |
| 12 | `day_of_week_mon` | 0.0312 | higher -> LESS likely to say yes |
| 13 | `month_may` | 0.0296 | higher -> LESS likely to say yes |
| 14 | `campaign` | 0.0292 | higher -> LESS likely to say yes |
| 15 | `is_high_season` | 0.0280 | higher -> MORE likely to say yes |
| 16 | `prev_failure` | 0.0274 | higher -> LESS likely to say yes |
| 17 | `was_contacted_before` | 0.0268 | higher -> MORE likely to say yes |
| 18 | `day_of_week_wed` | 0.0249 | higher -> MORE likely to say yes |
| 19 | `total_contacts` | 0.0235 | higher -> LESS likely to say yes |
| 20 | `prev_success` | 0.0189 | higher -> MORE likely to say yes |

## The three headline drivers

### 1. Macroeconomic conditions (54% of total importance)

`nr_employed`, `euribor3m` and `emp_var_rate` dominate. When rates are
high and employment is strong, people say no -- a term deposit competes
badly against better options and low anxiety.

**Action:** time the *campaign*, not just the call list. Rate
environment should trigger whether to run a deposit push at all.

**Caveat:** these barely vary within a week. They describe *when* we
called, not *who* we called, and are largely unusable for ranking a
call list where everyone shares today's euribor.

### 2. Contact channel

Mobile: **14.9%** response vs landline
**5.0%** (baseline 11.3%).
The highest-ranked driver the bank actually controls.

**Action:** prioritise customers with a mobile on file; consider a data
project to collect missing mobile numbers.

**Caveat:** correlation, not causation -- mobile likely proxies for age
and reachability. Useful as a *signal*, useless as an *intervention*.

### 3. Prior relationship and recency

Prior buyers respond at **65.8%** vs a
11.3% baseline -- **5.8x**. But
`prev_success` ranks only ~20th, because mean |SHAP| averages it over
everyone and only ~3.5% of customers have any history. Rare and
decisive, not weak.

**Action:** call prior buyers first (one SQL query, no model needed);
sequence follow-ups inside a 7-day window -- contacts within a week
convert at ~66% vs ~9% for never-contacted.

**Limit:** only ~3.5% of customers qualify. Always top of the list,
never enough to fill it.
