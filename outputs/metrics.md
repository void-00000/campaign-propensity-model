# Model Evaluation Results

Bank Marketing campaign-response model. All numbers below are measured
on the **held-out test set** -- customers neither model saw during
training.

## Setup

- Test set: **8,238 customers**, **928 real responders** (11.26%)
- Baseline response rate (random targeting): **11.26%**
- Imbalance handling: class weights (`class_weight='balanced'`, `scale_pos_weight=7.88`)
- `duration` excluded from features (target leakage -- see `sql/02_response_by_segment.sql`)

## Headline results

| Metric | Logistic Regression | XGBoost |
|---|---|---|
| **AUC-ROC** | 0.8010 | 0.8156 |
| Accuracy | 0.8333 | 0.8418 |
| Precision | 0.3649 | 0.3829 |
| Recall | 0.6476 | 0.6606 |
| F1 | 0.4668 | 0.4848 |
| **Top-decile lift** | 4.50x | 4.71x |

## Confusion matrices (threshold = 0.50)

### Logistic Regression

| | Predicted: don't call | Predicted: call |
|---|---|---|
| **Actually no** | 6,264 correctly skipped | 1,046 wasted calls |
| **Actually yes** | 327 MISSED buyers | 601 buyers found |

### XGBoost

| | Predicted: don't call | Predicted: call |
|---|---|---|
| **Actually no** | 6,322 correctly skipped | 988 wasted calls |
| **Actually yes** | 315 MISSED buyers | 613 buyers found |

## Lift by decile (XGBoost)

Customers sorted best-first, cut into ten equal groups.

| Decile | Customers | Responders | Response rate | Lift | Cum. % of buyers found |
|---|---|---|---|---|---|
| 1 | 824 | 437 | 53.03% | **4.71x** | 47.1% |
| 2 | 824 | 182 | 22.09% | **1.96x** | 66.7% |
| 3 | 824 | 70 | 8.50% | **0.75x** | 74.2% |
| 4 | 824 | 56 | 6.80% | **0.60x** | 80.3% |
| 5 | 823 | 38 | 4.62% | **0.41x** | 84.4% |
| 6 | 824 | 34 | 4.13% | **0.37x** | 88.0% |
| 7 | 824 | 31 | 3.76% | **0.33x** | 91.4% |
| 8 | 824 | 25 | 3.03% | **0.27x** | 94.1% |
| 9 | 824 | 28 | 3.40% | **0.30x** | 97.1% |
| 10 | 823 | 27 | 3.28% | **0.29x** | 100.0% |

## Threshold sweep (XGBoost)

There is no 'correct' threshold in the data -- only the one matching
your economics. This is the tradeoff, in real numbers.

| Threshold | Customers called | Buyers found | Buyers missed | Wasted calls | Precision | Recall |
|---|---|---|---|---|---|---|
| 0.1 | 8,177 | 926 | 2 | 7,251 | 0.113 | 0.998 |
| 0.2 | 7,411 | 901 | 27 | 6,510 | 0.122 | 0.971 |
| 0.3 | 4,535 | 800 | 128 | 3,735 | 0.176 | 0.862 |
| 0.4 | 2,184 | 671 | 257 | 1,513 | 0.307 | 0.723 |
| 0.5 | 1,601 | 613 | 315 | 988 | 0.383 | 0.661 |
| 0.6 | 1,262 | 571 | 357 | 691 | 0.452 | 0.615 |
| 0.7 | 1,079 | 532 | 396 | 547 | 0.493 | 0.573 |
| 0.8 | 664 | 374 | 554 | 290 | 0.563 | 0.403 |
| 0.9 | 197 | 147 | 781 | 50 | 0.746 | 0.158 |

## What these numbers mean in business terms

**Top-decile lift = 4.71x.** Rank all customers by predicted
probability and call only the top 10%. In that group 53.03% say yes,
versus 11.26% if you called 10% at random. Same
number of calls, **4.71x the signups**.

**Call the top 30%, capture 74.2% of all buyers.** Cut
your call volume by 70% and still get 74% of the
revenue -- because the model put the buyers at the top of the list.

**Why not accuracy?** A model that always predicts 'no' scores 88.7%
accuracy on this data and finds zero buyers. Accuracy rewards a model
for ignoring the rare class -- which is the only class we care about.
