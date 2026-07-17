# Customer Propensity & Campaign-Response Model

Predicts which bank customers will subscribe to a term deposit, so a marketing
team can call the ~10% most likely to say yes instead of calling everyone.

**Headline results — XGBoost, on 8,238 held-out customers:**

| | |
|---|---|
| **AUC-ROC** | **0.82** |
| **Top-decile lift** | **4.71x** |
| Top-decile response rate | 53.0% vs an 11.3% baseline |
| Call the top 30% of the list | capture **74%** of all responders |

Binary classification on imbalanced tabular data (11.27% positive class).
Features engineered in PostgreSQL, models in scikit-learn and XGBoost,
explained with SHAP, delivered through a Streamlit dashboard.

---

## What this project actually does

A bank runs phone campaigns selling term deposits. Calling all 41,188 customers
is expensive and ~89% of them say no. This model ranks customers by their
probability of subscribing, so the team works down a prioritised list.

**The business case, in one line:** call the top 10% of the ranked list and 53%
of them subscribe, versus 11.3% calling at random — the same phone bill for
4.71x the signups.

---

## Results

### Model comparison

| Metric | Logistic Regression | XGBoost |
|---|---|---|
| **AUC-ROC** | 0.8010 | **0.8156** |
| Accuracy | 0.8333 | 0.8418 |
| Precision | 0.3649 | 0.3829 |
| Recall | 0.6476 | 0.6606 |
| F1 | 0.4668 | 0.4848 |
| **Top-decile lift** | 4.50x | **4.71x** |

XGBoost wins by 0.0146 AUC. That's a small margin, and it's *why* the baseline
exists — without it, 0.8156 is a number with nothing to compare against.

### Lift by decile (XGBoost)

| Decile | Customers | Responders | Response rate | Lift | Cumulative % of buyers found |
|---|---|---|---|---|---|
| **1** | 824 | 437 | **53.03%** | **4.71x** | 47.1% |
| 2 | 824 | 182 | 22.09% | 1.96x | 66.7% |
| 3 | 824 | 70 | 8.50% | 0.75x | **74.2%** |
| 4 | 824 | 56 | 6.80% | 0.60x | 80.3% |
| 5–10 | ~4,940 | 183 | ~3.7% | ~0.33x | 100% |

Full numbers: [`outputs/metrics.md`](outputs/metrics.md)

### Charts

![Lift chart](outputs/figures/06_lift_chart.png)
*Decile lift and cumulative gains. Call 30% of the list, find 74% of the buyers.*

![ROC curves](outputs/figures/05_roc_curves.png)
*ROC curves for both models against random guessing.*

![SHAP summary](outputs/figures/08_shap_summary.png)
*SHAP: how each feature moves each customer's score. Red = high value, right = pushed toward "yes".*

> **Dashboard screenshot:** run `streamlit run app/dashboard.py`, then save a
> screenshot to `outputs/figures/10_dashboard.png` and it will render here.

---

## Two decisions that define this project

### 1. `duration` is excluded — target leakage

`duration` (call length in seconds) is the strongest predictor in the dataset.
Calls over 10 minutes convert at 48.6%; calls under a minute convert at 0.02%.
Including it pushes AUC to ~0.94.

**It's unusable.** You only know how long a call lasted *after making the call* —
but this model's job is deciding **who to call**, before any call exists. At
prediction time the feature doesn't exist. Worse, it partly *is* the outcome: a
20-minute call means the customer was interested.

The UCI documentation warns about this column. Our 0.82 is lower than the
tutorials reporting 0.94, and it's real.

### 2. Class weights, not SMOTE

**80 of our 92 features (87%) are one-hot flags.** SMOTE creates synthetic
minority examples by interpolating between real ones — which for categorical
features produces customers who are *52% blue-collar and 48% student
simultaneously*. `src/imbalance.py` runs SMOTE and prints one of these.

It generated 25,526 impossible customers and matched class weights on AUC anyway
(0.7998 vs 0.8010). Class weights invent nothing — every row the model sees is a
real person — and they're honest about the actual problem, which is a **cost
asymmetry** (missing a buyer costs more than a wasted call), not a data shortage.

---

## Tech stack

| Layer | Tool | Used for |
|---|---|---|
| Database | PostgreSQL 16 (Docker) | Raw storage, feature extraction |
| Data | pandas, NumPy | Feature matrix, analysis |
| Models | scikit-learn, XGBoost | Logistic Regression baseline, gradient boosting |
| Imbalance | scikit-learn, imbalanced-learn | Class weights (SMOTE evaluated and rejected) |
| Explainability | SHAP | Conversion drivers, per-customer explanations |
| Charts | matplotlib, seaborn | 9 figures |
| Dashboard | Streamlit | Ranked call list |

---

## Project structure

```
campaign-propensity-model/
├── docker-compose.yml       PostgreSQL 16 container
├── requirements.txt         pinned Python dependencies
├── .env                     DB credentials (git-ignored)
│
├── sql/
│   ├── 01_overview.sql              shape, class balance, data quality
│   ├── 02_response_by_segment.sql   response rates + lift by segment
│   └── 03_feature_extraction.sql    the feature view (20 engineered features)
│
├── src/
│   ├── config.py            paths + DB connection, defined once
│   ├── load_data.py         download UCI data -> PostgreSQL
│   ├── run_sql.py           run a .sql file, print the results
│   ├── eda.py               distributions, correlations, class balance
│   ├── features.py          SQL view -> 92-feature matrix + train/test split
│   ├── imbalance.py         the imbalance problem, demonstrated
│   ├── train.py             Logistic Regression + XGBoost
│   ├── evaluate.py          AUC, lift, confusion matrices, ROC
│   ├── interpret.py         SHAP drivers
│   └── run_all.py           the whole pipeline, one command
│
├── app/
│   └── dashboard.py         Streamlit dashboard
│
└── outputs/
    ├── metrics.md           every evaluation number
    ├── shap_drivers.md      drivers + business actions
    ├── figures/             9 PNGs
    └── models/              trained models (.joblib)
```

---

## Running it

### Prerequisites
- Python 3.11+
- Docker Desktop

### Setup (once)

```powershell
# 1. Start PostgreSQL
docker compose up -d

# 2. Create the virtual environment and install dependencies
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows
# source .venv/bin/activate           # macOS/Linux
pip install -r requirements.txt

# 3. Verify the database connection
python src/config.py
```

### Run the whole pipeline

```powershell
python src/run_all.py
```

Downloads the data, loads PostgreSQL, runs the SQL, builds features, trains both
models, evaluates, and generates SHAP. **Under a minute** on a modern laptop
(add a few seconds on the first run for the dataset download). Every random
operation is seeded, so you get the numbers above exactly — verified by running
it from a cleared `outputs/` folder.

### Launch the dashboard

```powershell
streamlit run app/dashboard.py        # http://localhost:8501
```

### Or run steps individually

```powershell
python src/load_data.py                      # CSV -> PostgreSQL
python src/run_sql.py sql/01_overview.sql    # any .sql file
python src/eda.py                            # charts + summary
python src/features.py                       # feature matrix
python src/imbalance.py                      # the imbalance demonstration
python src/train.py                          # both models
python src/evaluate.py                       # metrics + charts
python src/interpret.py                      # SHAP
```

### Shut down

```powershell
docker compose down      # stops the DB; your data survives
```

---

## Deploying it

The dashboard reads from PostgreSQL. Locally that's the Docker container; in the
cloud it needs a database the internet can reach. `src/config.py` resolves the
connection in priority order — `DATABASE_URL` env var → Streamlit secrets →
local `.env` — so **the same code runs in both places with no edits.**

### 1. Create a free Neon database

1. Sign up at [neon.tech](https://neon.tech) (free tier, GitHub login works).
2. Create a project — any name, pick the region closest to you.
3. Copy the connection string from **Connection Details**. It looks like:
   ```
   postgresql://neondb_owner:PASSWORD@ep-xxx-xxx.us-east-2.aws.neon.tech/neondb?sslmode=require
   ```

### 2. Load your data into it

Point the local scripts at Neon by setting `DATABASE_URL`, then run the loader.
Nothing else changes — the same `load_data.py` that fills Docker fills Neon.

```powershell
$env:DATABASE_URL = "postgresql://neondb_owner:PASS@ep-xxx.neon.tech/neondb?sslmode=require"

python src/config.py                                  # confirm it says NEON
python src/load_data.py                               # ~1-2 min over the network
python src/run_sql.py sql/03_feature_extraction.sql   # rebuild the view
python src/config.py                                  # confirm table + view exist

Remove-Item Env:\DATABASE_URL                         # back to local Docker
```

`python src/config.py` prints which database it's talking to and whether the data
is actually there. Run it before and after — loading 41,000 rows into the wrong
database is an easy and annoying mistake.

### 3. Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. **New app** → pick this repo → main file path: `app/dashboard.py`
3. Open **Advanced settings → Secrets** and paste:
   ```toml
   DATABASE_URL = "postgresql://neondb_owner:PASS@ep-xxx.neon.tech/neondb?sslmode=require"
   ```
4. Deploy. First build takes ~3–5 minutes (installing XGBoost and SHAP).

The trained models are committed to `outputs/models/`, so the cloud app loads
them directly — it never trains anything.

### Deployment notes

- **Never commit your Neon password.** It goes in Streamlit's Secrets UI, not in
  the repo. `.gitignore` blocks `.env` and `.streamlit/secrets.toml`;
  `.streamlit/secrets.toml.example` is committed as the template.
- **Neon free-tier compute sleeps after ~5 minutes idle.** The first request
  after a quiet spell takes a second or two to wake it. Fine for a portfolio
  link; not how you'd run production.
- **Rotate the password if it ever leaks.** Neon lets you reset it from the
  dashboard. If a secret reaches a commit, rotating it is the fix — deleting the
  commit is not, because it's already in the history and possibly in someone's
  clone.
- **`pool_pre_ping=True`** is set in `get_engine()`. Cloud databases drop idle
  connections, and a dead pooled connection produces a baffling error on the
  next query. This checks the connection is alive first and silently reconnects.

---

## Dataset

[UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
(`bank-additional-full.csv`) — 41,188 Portuguese bank customers, 20 features,
binary target. Downloaded automatically by `src/load_data.py`; if that fails the
script prints manual instructions.

Two data traps worth knowing about, both caught in `sql/01_overview.sql`:

- **`pdays = 999` is a sentinel**, not a number. It means "never previously
  contacted" and applies to 96.3% of rows. Averaging it gives a meaningless
  962. We decompose it into a flag plus a real recency value.
- **Missing data is encoded as the string `'unknown'`**, not NULL. A NULL check
  returns perfectly clean — and is completely wrong. `default` is 20.87%
  unknown.

---

## What SHAP found

| Theme | Share of importance |
|---|---|
| **Macroeconomic conditions** | **54.0%** |
| Demographics | 15.4% |
| Prior relationship / recency | 10.2% |
| Timing | 9.5% |
| Contact channel | 8.7% |
| Campaign intensity | 2.3% |

1. **The economy dominates** (`nr_employed`, `euribor3m`). High rates and strong
   employment → people say no. **This is also the model's biggest weakness**:
   these features describe *when* we called, not *who* we called, and barely
   vary within a week. A deployed variant should drop them — it would score
   lower and might rank better.
2. **Contact channel** — mobile 14.9% vs landline 5.0%. The highest-ranked
   driver the bank actually controls.
3. **Prior relationship** — prior buyers respond at 65.8% (5.8x baseline).
   Ranks only ~20th because mean |SHAP| averages it over everyone and only
   ~3.5% of customers have any history. **Rare and decisive, not weak** —
   conditioned on those who have it, it's 8x stronger.

Details and business actions: [`outputs/shap_drivers.md`](outputs/shap_drivers.md)

---

## Honest limitations

- **The economic features are a crutch.** 54% of importance from variables that
  are constant across any given week. The model has partly learned "2008–2010
  was a good time to sell deposits" — true, useless for ranking today's list.
- **Correlation, not causation, throughout.** Mobile contact and the March
  effect are almost certainly proxies. Switching a customer's landline to a
  mobile would not make them buy. Useful as *signals*, not *interventions*.
- **No hyperparameter search.** XGBoost uses sensible defaults with early
  stopping. A tuned model would likely gain a little AUC; that wasn't the point.
- **Single train/test split, not cross-validation.** With 41k rows and a
  stratified split, the test estimate is stable enough. CV would give tighter
  error bars.
- **Historical data.** The dataset is from a 2008–2013 Portuguese bank. The
  economic relationships it learned may not transfer.

---

## Further reading

- [`LEARNING.md`](LEARNING.md) — the concepts, explained in plain English
- [`outputs/metrics.md`](outputs/metrics.md) — every evaluation number
- [`outputs/shap_drivers.md`](outputs/shap_drivers.md) — drivers and actions
