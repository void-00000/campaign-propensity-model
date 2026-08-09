# Project Brief — Customer Propensity & Campaign-Response Model

**Prepared for:** resume writer
**Candidate:** Abhay Bansal
**Project type:** Personal / portfolio project (public dataset)

> Every figure in this document is measured from the actual model on a held-out
> test set and is independently verifiable from the linked repository. Nothing
> is rounded up or estimated.

---

## Project title

**Customer Propensity & Campaign-Response Model** — end-to-end machine learning
pipeline predicting term-deposit subscription for bank marketing campaigns.

## One-line description

Built a binary classification model on 41,188 bank customers that ranks
prospects by conversion probability, enabling a marketing team to capture 74% of
responders while calling only 30% of the list.

## Links

| | |
|---|---|
| **Live demo** | https://campaign-propensity-model-djaftj2gummd232pchdnsk.streamlit.app/ |
| **Source code** | https://github.com/abhaybansal0322/campaign-propensity-model |

---

## Headline metrics

| Metric | Value |
|---|---|
| **AUC-ROC (XGBoost, held-out test set)** | **0.82** (precisely 0.8156) |
| **Top-decile lift** | **4.71x** |
| Top-decile response rate | 53.0% vs 11.3% random-targeting baseline |
| Efficiency gain | Call top 30% of list → capture **74.2%** of all responders |
| Baseline model (Logistic Regression) | 0.8010 AUC |
| Precision / Recall / F1 | 0.383 / 0.661 / 0.485 |

## Scale & scope

| | |
|---|---|
| Dataset | 41,188 records — UCI Bank Marketing |
| Features | **92** (20 engineered in SQL, expanded via one-hot encoding) |
| Class imbalance | 11.27% positive class (7.9:1 ratio) |
| Test set | 8,238 held-out customers |
| Models trained | 2 (Logistic Regression baseline + XGBoost) |

## Tech stack

| Category | Technologies |
|---|---|
| Databases | PostgreSQL 16, Neon (cloud Postgres), Docker |
| Languages | Python 3.12, SQL |
| Machine learning | scikit-learn, XGBoost, imbalanced-learn |
| Data engineering | pandas, NumPy, SQLAlchemy |
| Explainability | SHAP |
| Visualization | matplotlib, seaborn |
| Deployment | Streamlit, Streamlit Cloud, Git |

---

## Ready-to-use resume bullets

### Option A — single bullet (tightest)

> Built and deployed an end-to-end customer propensity model (PostgreSQL,
> XGBoost, SHAP, Streamlit) on 41K bank marketing records achieving **0.82 AUC**
> and **4.71x top-decile lift**, enabling 74% of responders to be captured from
> 30% of call volume.

### Option B — three bullets (recommended)

> - Engineered 20 features in **SQL** from 41,188 bank marketing records in
>   PostgreSQL, expanded to 92 model inputs; identified and excluded a
>   target-leakage feature that inflated AUC by ~0.12 but was unavailable at
>   prediction time.
> - Trained **XGBoost** against a Logistic Regression baseline on imbalanced data
>   (11.27% positive class), reaching **0.82 AUC** and **4.71x top-decile lift** —
>   top decile converts at 53.0% vs an 11.3% random-targeting baseline.
> - Deployed a **Streamlit** dashboard serving a ranked call list backed by a
>   cloud PostgreSQL database, with **SHAP** explanations surfacing per-customer
>   conversion drivers.

### Option C — four bullets (if space allows)

Add to Option B:

> - Evaluated SMOTE against class weighting and selected class weights with
>   documented rationale: 87% of features were one-hot encoded, causing SMOTE to
>   generate invalid synthetic records with no AUC benefit.

---

## Notes for the resume writer

**Use "4.71x lift", not "371% above baseline."** They are the same number. The
multiple is standard phrasing in marketing analytics; the percentage reads as
inflated.

**"0.82" is the correct precision.** The true value is 0.8156. Avoid 0.816
(false precision for a resume) and avoid rounding to 0.8 (loses the signal).

**Keep the "held-out" framing** if there is room. "0.82 AUC on held-out data"
signals methodological competence that a bare "0.82 AUC" does not.

**The target-leakage bullet is the differentiator.** Most candidates using this
dataset report ~0.94 AUC because they retain the `duration` feature — call
length, which is only knowable *after* the call the model exists to decide
whether to make. This project's 0.82 is lower **and correct**. If only one
technical detail survives editing, keep that one.

**Do not claim:** production deployment at a company, real revenue impact,
stakeholder management, or team leadership. This is a personal project on a
public dataset. The metrics are real; the context is portfolio.

---

## Optional context (only if the format allows depth)

Technical decisions the candidate can defend in interview:

- **Target leakage identified and removed.** `duration` is the dataset's
  strongest predictor (calls >10 min convert at 48.6%; <1 min at 0.02%) and is
  unusable — it does not exist at prediction time and partly encodes the outcome.
- **Accuracy rejected as a metric.** A model that always predicts "no" scores
  88.7% accuracy and identifies zero buyers. AUC scores it 0.500, correctly.
- **Class weights over SMOTE, with evidence.** SMOTE generated 25,526 synthetic
  records that were categorically invalid (e.g. 52% blue-collar and 48% student
  simultaneously) and produced no AUC gain over class weighting.
- **Baseline model trained deliberately.** XGBoost outperforms Logistic
  Regression by only 0.0146 AUC — a finding only visible because the baseline
  exists, and the reason the candidate can justify the model choice on
  deployment context rather than score alone.
- **Model limitations documented.** Macroeconomic features account for 54% of
  SHAP importance but describe *when* customers were called rather than *who*
  they are — documented as the model's principal weakness rather than presented
  as a strength.
- **Reproducible pipeline.** Single command (`python src/run_all.py`) rebuilds
  every result from raw data in under a minute; dependencies pinned because the
  serialized model artifacts are version-coupled.
