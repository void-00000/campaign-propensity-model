# LEARNING.md — the concepts, in plain English

This is the file to read the night before the interview. Every number in it is
real and comes from this project.

**How to use it:** each of the four big questions has a **30-second answer**
(what you actually say) and a **the full picture** section (what you say when
they follow up). Learn the short one properly. Read the long one enough that a
follow-up doesn't surprise you.

**One rule that matters more than any answer here:** never state a number you
can't explain. An interviewer who senses recitation will dig, and they should.
If you don't know, say "I don't know" and then say what you'd do to find out.
That answer scores better than a confident wrong one, every time.

---

## Q1: Why AUC and lift instead of accuracy?

### The 30-second answer

> "Because only 11% of customers say yes. A model that always predicts 'no'
> scores 88.7% accuracy and finds zero buyers — I actually built that model to
> check, and its AUC is 0.500, which correctly calls it worthless. Accuracy
> rewards ignoring the rare class, which is the only class we care about.
>
> AUC measures *ranking*: given one buyer and one non-buyer, it's the
> probability the model scores the buyer higher. Mine is 0.82. That's the right
> metric because ranking is literally the job — we're sorting a call list, not
> delivering verdicts.
>
> Lift is what I'd show the business. Top decile converts at 53% versus an 11.3%
> baseline — 4.71x. Same phone bill, 4.71x the signups."

### The full picture

**Why accuracy breaks.** Accuracy is "what fraction of guesses were right". It
treats both mistakes as equal. They aren't:

- Call someone who says no → you wasted five minutes.
- Miss someone who'd say yes → you lost a customer and their deposit.

When 88.7% of answers are "no", a model gets 88.7% by learning nothing. **Any
metric a lazy model can max out isn't measuring what you care about.**

It shows up in real models too, not just the dummy. Untouched Logistic
Regression scored the *best* accuracy of everything I trained — 0.9020 — while
missing 718 of 928 buyers. The class-weighted version scored **lower** accuracy
(0.8333) and found 2.9x as many buyers. Lower accuracy, far better model.

**What AUC actually means.** Plot true positive rate against false positive rate
at every possible threshold; AUC is the area under that curve. 0.5 = random,
1.0 = perfect.

The plain-English version is the one to say: **pick one random buyer and one
random non-buyer — AUC is the probability the model scores the buyer higher.**
At 0.8156 it gets that pairwise ordering right ~82% of the time.

Two properties make it right here:
1. **Threshold-independent.** It summarises every possible cutoff, so it doesn't
   bake in a business decision you haven't made yet.
2. **Un-gameable by the majority class.** The always-no model gets exactly 0.500.

**What lift means.** Rank everyone by predicted probability, cut into ten equal
groups. Decile 1 = the 10% the model likes most.

| Decile | Response rate | Lift |
|---|---|---|
| 1 | 53.03% | **4.71x** |
| 2 | 22.09% | 1.96x |
| 3 | 8.50% | 0.75x |
| 10 | 3.28% | 0.29x |

Call 824 people at random → ~93 say yes. Call the 824 the model ranked highest →
**437 say yes.** That's lift, and it maps straight onto budget.

Better still: **call the top 30%, capture 74% of all buyers.** Cut call volume
70%, keep three-quarters of the revenue.

**Say both.** AUC is the statistically honest summary. Lift is what gets the
project funded. Quoting only AUC to a marketing director is a failure to
communicate; quoting only lift to a data scientist looks like you're hiding.

### If they push: "isn't F1 enough?"

> "F1 balances precision and recall into one number, which is useful for a quick
> comparison — I report it. But it hides the tradeoff I actually care about. At
> threshold 0.3 I find 800 of 928 buyers with 3,735 wasted calls; at 0.7 I find
> 532 with 547 wasted. Same model, same AUC, completely different campaigns. F1
> collapses that into one number and throws away the decision."

---

## Q2: How did you handle class imbalance, and why that method?

### The 30-second answer

> "Class weights — `class_weight='balanced'` for Logistic Regression,
> `scale_pos_weight=7.88` for XGBoost. It weights each class inversely to its
> frequency, so with a 7.9:1 imbalance a responder counts ~7.9x as much during
> training. Ignoring the rare class stops being the easy win.
>
> I tested SMOTE against it and rejected it. 87% of my features are one-hot
> flags, and SMOTE interpolates between examples — so it generated customers who
> were 52% blue-collar and 48% student simultaneously. I printed one. It invented
> 25,526 impossible people and matched class weights on AUC anyway: 0.7998 vs
> 0.8010.
>
> The tradeoff is real: I accept 1,046 wasted calls to find 391 more buyers.
> That's only correct because a deposit is worth more than five wasted minutes.
> If a call cost $500 and a deposit earned $10, I'd flip it."

### The full picture

**The situation.** 4,640 responders out of 41,188 — **11.27%**. 7.9
non-responders per responder. That's normal for anything worth predicting:
fraud, churn, disease, click-through. **The interesting thing is always rare —
that's why it's worth predicting.**

**What I compared** (all on the identical test set):

| Approach | Accuracy | Recall | AUC | Buyers found | Buyers missed | Wasted calls |
|---|---|---|---|---|---|---|
| Ignore it | **0.9020** | 0.2263 | 0.8015 | 210 | **718** | 89 |
| **Class weights** | 0.8333 | 0.6476 | 0.8010 | **601** | 327 | 1,046 |
| SMOTE | 0.8303 | 0.6573 | 0.7998 | 610 | 318 | 1,080 |

**Why class weights won:**

1. **SMOTE invents impossible customers on this data.** 80 of 92 features are
   one-hot flags. There's nothing halfway between "student" and "retired". Here
   is one SMOTE actually produced:
   ```
   age                = 26.52
   job_blue-collar    = 0.520
   job_student        = 0.480     -> job_* sums to 1.000
   marital_single     = 0.480
   marital_unknown    = 0.520
   ```
   Arithmetically tidy, humanly impossible. The model learns the shape of a
   fictional population, then meets real customers.

2. **Same result, more machinery.** SMOTE matched on AUC while adding a
   resampling step, 25,526 synthetic rows, and a leakage risk.

3. **It's honest about the real problem.** We don't have a data shortage — we
   have a **cost asymmetry**. Class weights say that directly. SMOTE encodes the
   same idea by faking a world where buyers are common. They aren't.

**The thing most people miss.** Look at the AUC column: 0.8015 / 0.8010 /
0.7998. **Almost identical.** Class weights don't make the model *smarter* —
they move the **decision threshold**, the cutoff where a probability becomes a
"yes". The model's ability to *rank* barely changes.

That's worth understanding precisely, because it means: if you only care about
ranking (and for a call list, you do), imbalance handling matters far less than
people assume. It matters when you need hard yes/no predictions at a sensible
default cutoff.

### If they push: "when WOULD you use SMOTE?"

> "When the minority class is genuinely tiny — a few dozen cases, not 4,640 —
> where the model needs more *examples* rather than a stronger *signal* about
> the ones it has. And even then I'd use SMOTENC, which handles categoricals
> properly instead of interpolating through them. The other thing I'd watch is
> that SMOTE only ever touches training data. Oversample before you split and
> copies of the same customer land in train and test — the model gets graded on
> people it memorised."

---

## Q3: What did SHAP reveal, and what action does it imply?

### The 30-second answer

> "Three things. First, the economy dominates — `nr_employed` and `euribor3m`
> are 54% of total importance. When rates are high and employment is strong,
> people say no, because a term deposit competes badly. That says *time the
> campaign*, not just the call list.
>
> Second, and this is the honest part: that's the model's biggest **weakness**,
> not its best feature. Those variables barely move within a week — they describe
> *when* we called, not *who*. The model partly learned '2008–2010 was a good
> time to sell deposits'. True, and useless for ranking today's list. I'd ship a
> variant without them; it would score lower and might rank better.
>
> Third, the actionable one: prior buyers respond at 65.8% versus an 11.3%
> baseline. Call them first — that needs one SQL query, no model. And sequence
> follow-ups inside 7 days: contacts within a week convert at ~66% vs ~9% for
> never-contacted."

### The full picture

**What SHAP is.** From cooperative game theory — if a team wins, how much did
each player contribute? SHAP treats features as the team. For every customer it
answers: how much did each feature push this person's score away from the
average?

Its guarantee is **additivity**:
```
average prediction + sum of SHAP values = this customer's prediction
```
Nothing hand-waved, nothing left over. **I verified this on our model** — the
largest disagreement across 8,238 customers was 0.0000014. Worth doing: our model
used early stopping, so the booster holds more trees than it predicts with. Had
SHAP explained unused trees, we'd be reading an explanation of a model that
doesn't exist.

**Why not XGBoost's built-in `feature_importances_`?** It counts how often a
feature was used in a split. That says nothing about *direction*, and it inflates
high-cardinality features that simply offer more places to split.

**The drivers by theme:**

| Theme | Share |
|---|---|
| Macroeconomic conditions | **54.0%** |
| Demographics | 15.4% |
| Prior relationship / recency | 10.2% |
| Timing | 9.5% |
| Contact channel | 8.7% |
| Campaign intensity | 2.3% |

**Driver 2 — contact channel.** Mobile 14.9% vs landline 5.0%. The
highest-ranked driver the bank actually **controls**; the economy is weather,
this is a decision.

*Caveat:* correlation, not causation. Mobile proxies for age and reachability.
"Call them on a mobile instead" would do nothing — nobody's landline is stopping
them buying. **"Prioritise people who have one" still works, because you're
using it as a signal, not an intervention.** That distinction is the whole game.

**Driver 3 — and the best thing in this project.**

Prior buyers respond at 65.8%. **So why does `prev_success` rank only 20th of
92?** Two effects, and understanding them is what separates using SHAP from
understanding it:

1. **Dilution by averaging.** Mean |SHAP| averages over all 8,238 customers.
   Only 291 (3.5%) have any contact history — the feature is exactly 0 for
   everyone else.
   ```
   mean |SHAP| over everyone            : 0.0189
   mean |SHAP| over those WITH history  : 0.1429   -> 8x stronger
   ```
   **The feature isn't weak. It's rare AND decisive**, and averaging hides that
   completely.

2. **Credit-splitting.** I engineered `prev_success`, `was_contacted_before`,
   `days_since_prev_contact` and `poutcome_success` to encode overlapping facts.
   SHAP divides credit between correlated features, so each looks modest while
   the theme is substantial. That's SHAP behaving *correctly*.

**Say this and you sound like someone who has actually used SHAP:** never read
mean |SHAP| as "how much this feature matters to a customer". It means "how much
it moves the **average** customer". For a rare, decisive feature those are wildly
different numbers.

### If they push: "so is SHAP telling you what causes conversion?"

> "No, and that's an important limit. SHAP explains **the model**, not the world.
> It tells me what the model used, not what causes anything. If the model learned
> a spurious pattern, SHAP will faithfully explain the spurious pattern. That's
> exactly why I flagged the economic features as a weakness — SHAP told me they
> dominate, and judgment told me that's a problem rather than a finding.
> Anything causal needs an A/B test."

---

## Q4: Logistic Regression vs XGBoost — the tradeoff, and which would you ship?

### The 30-second answer

> "XGBoost, but the case is closer than the numbers suggest and I'd say so.
> 0.8156 vs 0.8010 — it wins by 0.0146 AUC, which is a small gain for a model
> that's dramatically harder to explain and maintain.
>
> The signal in this data is mostly linear, so the straight line already captures
> most of it. I'd ship XGBoost because SHAP gives back the interpretability I
> gave up, and nobody has a right to an explanation for not being phoned — the
> cost of being wrong is five wasted minutes.
>
> If this were credit approval, I'd ship Logistic Regression without hesitating
> and consider 0.0146 a fair price for a model I can defend line by line to a
> regulator. **The deciding factor is the deployment context, not the score.**"

### The full picture

**Logistic Regression.** Draws one straight boundary through the feature space
and asks which side you're on. Learns exactly one coefficient per feature — and
those coefficients *are* the model. Readable, printable, defensible.
(Despite the name, it's a classifier, not a regression.)

**XGBoost.** Builds hundreds of small decision trees in sequence, each focusing
on the previous ones' mistakes. Finds non-linear patterns and interactions
automatically.

**What each needed:**

| | Logistic Regression | XGBoost |
|---|---|---|
| Missing values | Must impute (median) | **Handles natively** |
| Scaling | **Required** | Irrelevant |
| Imbalance | `class_weight='balanced'` | `scale_pos_weight=7.88` |
| Interpretability | Built in | Needs SHAP |
| Training time | Seconds | ~a minute |
| Non-linearity | Only what you engineer | Finds it itself |

Two of those are worth explaining if asked:

- **Why LogReg needs scaling:** `nr_employed` is ~5,000 while our flags are 0/1.
  The model penalises large coefficients, so an unscaled feature in thousands
  needs a tiny coefficient and dodges the penalty, while a 0/1 flag needs a big
  one and gets punished. **Without scaling you rank features by their units, not
  their usefulness.**
- **Why XGBoost doesn't:** trees ask "is x > 5000?". Rescale x and the threshold
  rescales identically. The tree is unchanged.

**Why XGBoost's native NaN handling matters here.**
`days_since_prev_contact` is NULL for 96% of customers **because they were never
contacted**. That's a fact about the customer, not an absence of data. XGBoost
learns which direction to send NaNs at each split — treating "missing" as
information. Imputing a median there tells a small lie. On the SHAP beeswarm you
can see those as grey dots.

**Where XGBoost genuinely earns it.** Age has a **U-shape** — under-25s respond
at 24%, 65+ at 47%, but the 35–54 middle at only 8.7%. Its linear correlation
with the target is near zero, so it didn't even crack the top 12 by correlation.
A straight line averages that U away to nothing. XGBoost finds it effortlessly.

I gave Logistic Regression a fighting chance by engineering `age_band` in SQL —
handing it the non-linearity it can't discover. **That's the real tradeoff:
with XGBoost the model does that work; with Logistic Regression you do.** Part of
why the gap is only 0.0146 is that I did that work.

### If they push: "only 0.015 — is XGBoost even worth it?"

This isn't a trap, it's the actual question. The strong answer:

> "On this data, barely. The signal is mostly linear, so the straight line
> already captures it. I'd still ship XGBoost because SHAP recovers the
> interpretability and the context tolerates a black box — but if the team needed
> to explain decisions, I'd ship the baseline and lose almost nothing. The
> honest summary is that the baseline is doing most of the work here, and the
> only reason I know that is that I built it. If I'd only trained XGBoost, 0.8156
> would be a number with nothing to compare it to."

---

## Things they might ask that aren't on the list

### "Walk me through the project."

> "Bank runs phone campaigns for term deposits; 89% say no, so calling everyone
> wastes money. I loaded 41,188 customers into PostgreSQL, explored it in SQL,
> and built a feature view there — decoding sentinels, bucketing, flags, ratios.
> Python turned that into 92 features, mostly one-hot encoded.
>
> The most important decision was dropping `duration`. It's the strongest
> predictor in the data and it's target leakage — you only know call length after
> the call, but the model runs before it.
>
> Trained Logistic Regression as a baseline and XGBoost as the main model, both
> with class weights for the 11.27% imbalance. XGBoost got 0.82 AUC and 4.71x
> top-decile lift. SHAP said the economy dominates — which I read as the model's
> weakness, not its strength. Then a Streamlit dashboard so the team gets a
> ranked call list."

### "What was the hardest part / what went wrong?"

Pick one and be specific. Real ones from this build:

> "I engineered three ratio features in SQL — things like euribor divided by the
> average euribor. The EDA correlation check came back at exactly r = 1.000 with
> their source columns. I'd divided by a **constant**, which is just a rescale —
> zero new information. The rule I took from it: a ratio feature only earns its
> place if the denominator varies per row."

> "I predicted `prev_success` would be the top customer-level driver, because the
> SQL showed prior buyers converting at 65%. SHAP ranked it 20th. I was wrong
> about the mechanism — mean |SHAP| averages over all customers and only 3.5%
> have any history, so a rare-but-decisive feature gets diluted to nothing.
> Conditioned on the people it applies to, it's 8x stronger. It taught me what
> that metric actually measures."

### "How would you improve it?"

> "Three things, in order. **Drop the economic features** and see what happens —
> they're 54% of importance and mostly describe when we called, not who. AUC
> would fall; ranking might improve. **Tune hyperparameters** — I used sensible
> defaults with early stopping, never ran a search. **Tune the threshold to real
> economics** — I report a sweep, but the right cutoff needs actual cost-per-call
> and value-per-deposit, which only the business has."

### "Is this model fair?"

Do not wing this one.

> "I haven't audited it, and I'd want to before deploying. It uses age, job,
> marital status and education — in the EU, age is a protected characteristic and
> the others proxy for socioeconomic status. For a marketing call list the harm
> is low: the downside of being excluded is not getting a phone call. For
> anything touching credit or pricing, that same feature set would be a serious
> problem and I'd expect a fairness review before it went near production."

---

## Glossary

| Term | Plain English |
|---|---|
| **AUC-ROC** | Given one buyer and one non-buyer, the probability the model scores the buyer higher. 0.5 = random. |
| **Baseline rate** | The response rate from calling at random. Here, 11.27%. |
| **Class imbalance** | One outcome vastly outnumbers the other. 88.73% no, 11.27% yes. |
| **Class weights** | Telling the model a rare-class mistake costs more than a common-class one. |
| **Confusion matrix** | The 2x2 grid of right/wrong by predicted/actual. |
| **Decile** | A tenth of the list. Decile 1 = the model's top 10%. |
| **Early stopping** | Stop adding trees when validation performance stops improving. |
| **Feature** | One input column. We have 92. |
| **Lift** | This group's response rate ÷ baseline. 4.71x = 4.71x better than random. |
| **Multicollinearity** | Features that duplicate each other. Destabilises linear models. |
| **One-hot encoding** | One text column → one 0/1 column per category. |
| **Overfitting** | Memorising the training data instead of learning patterns. |
| **Pipeline** | Chained preprocessing + model, so steps can't leak or run out of order. |
| **Precision** | Of the people we called, what fraction said yes? (wasted effort) |
| **Recall** | Of the people who'd say yes, what fraction did we call? (missed money) |
| **Sentinel value** | A magic number meaning something other than its value. `pdays=999` = "never contacted". |
| **SHAP** | How much each feature pushed one prediction away from average. |
| **SMOTE** | Inventing synthetic minority examples by interpolation. Rejected here. |
| **Stratified split** | Splitting so both halves keep the same class ratio. |
| **Target leakage** | A feature that secretly contains the answer, unavailable at prediction time. `duration`. |
| **Threshold** | The cutoff turning a probability into a yes/no. A business decision. |
