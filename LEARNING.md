# LEARNING.md — every concept in this project, from zero

This file assumes you know **nothing**. Every technical word is defined before
it gets used. Read Part 1 first; the interview answers in Part 8 will make sense
only after you have the vocabulary.

Every number here is real and comes from this project.

## How to use this file

| If you have... | Read |
|---|---|
| 2 hours | The whole thing, in order |
| 30 minutes | Part 1 (vocabulary), then Part 8 (the four answers) |
| 10 minutes, interview is now | The **30-second answers** in Part 8, plus the glossary |

**One rule worth more than any answer here:** never say a number you can't
explain. An interviewer who senses recitation will dig, and they should. "I
don't know, but here's how I'd find out" beats a confident wrong answer every
single time.

---
---

# PART 1 — THE ABSOLUTE BASICS

Start here. Each term below uses only words defined above it.

## What is a model?

A **model** is a function that takes information about something and returns a
prediction about it.

Ours takes facts about a bank customer (age, job, whether we've called them
before) and returns a number between 0 and 1: **how likely is this person to say
yes to our offer?**

That's it. Everything else in this file is detail about how to build one, how to
tell if it's any good, and how to explain what it's doing.

## What is the thing we're predicting?

A Portuguese bank ran phone campaigns selling **term deposits** — a savings
product where you lock money away for a fixed period and get a guaranteed return.

Every customer either subscribed (**yes**) or didn't (**no**). We have 41,188
customers and we know what each one answered. The job: predict that answer
*before* making the call, so the bank calls the people most likely to say yes.

## Feature and target

- A **feature** is one piece of input information. One column. Age is a feature.
  Job is a feature. We have **92** of them.
- The **target** is the thing being predicted — the answer we want. Ours is
  `y`: did this customer subscribe, yes or no?

The whole game is: learn the relationship between the features and the target.

> **Convention you'll see everywhere:** features are called `X` (capital, because
> it's a table — many rows, many columns) and the target is called `y` (lowercase,
> because it's a single column). This isn't a rule, it's a habit the entire field
> shares.

## Binary classification

**Classification** = predicting which category something belongs to.
**Binary** = there are exactly two categories.

Ours is binary classification: yes or no. (If we were predicting *how much money*
someone would deposit, that would be **regression** — predicting a number on a
continuous scale rather than a category.)

## Probability, and why the model doesn't just say "yes"

Our model doesn't output "yes" or "no". It outputs a **probability** — a number
from 0.0 to 1.0 meaning "how confident am I that this person says yes?"

- 0.97 → very likely to subscribe
- 0.03 → very unlikely

**This is deliberate and important.** A probability can be *sorted*. Given 8,238
customers with 8,238 probabilities, you can rank them best-first and hand a
marketing team a prioritised list. A bare "yes"/"no" can't be sorted — you'd just
have two piles, and no idea who to call first within the "yes" pile.

**Our model's real job is ranking, not deciding.** Hold onto that — it's the
reason for half the decisions in this project.

## Training set and test set

To learn, a model needs examples. But here's the trap:

**A model that has seen an example can simply memorise it.** Testing a model on
data it learned from tells you how good its memory is, not whether it learned
anything useful about new people.

So we split the data:

| | Rows | Purpose |
|---|---|---|
| **Training set** | 32,950 (80%) | The model learns from these |
| **Test set** | 8,238 (20%) | Hidden during training. Used once, at the end, to score it |

The test set stands in for *"customers we haven't called yet"* — the situation
the model actually faces in real life.

**The rule:** you look at the test set once, at the very end. Every time you use
it to make a decision, it becomes a little less honest as a measure of
performance, because you've started fitting your choices to it.

## Baseline

The **baseline** is what you'd get with no model at all — by calling people at
random.

**Our baseline is 11.27%.** Out of 41,188 customers, 4,640 said yes. Call people
at random and about 11.27% of them say yes.

Every result in this project is measured against that number. A model is only
worth anything if it beats calling at random.

---
---

# PART 2 — THE FOUR OUTCOMES

Everything in Part 3 is built from this. It's worth being solid here.

## Threshold

The model gives a probability. But to actually *do* something, you need a
decision: call this person, or don't.

So you pick a cutoff. **"Call everyone above 0.5."** That cutoff is called the
**threshold**.

**The threshold is a business decision, not a mathematical one.**
- Lower it → you call more people → you catch more buyers → you waste more calls
- Raise it → you call fewer people → less waste → you miss more buyers

The model ranks. **You** decide where to cut. There is no "correct" threshold
sitting in the data waiting to be discovered.

## The four things that can happen

Once you have a threshold, every customer falls into exactly one of four boxes:

| | We predicted: **don't call** | We predicted: **call** |
|---|---|---|
| **They'd say no** | **True Negative** — correctly skipped | **False Positive** — wasted call |
| **They'd say yes** | **False Negative** — MISSED BUYER | **True Positive** — found a buyer! |

The words decoded:
- **True** = we were right. **False** = we were wrong.
- **Positive** = we predicted "yes/call". **Negative** = we predicted "no/skip".

So **False Negative** = "we said negative (don't call), and we were wrong" =
a real buyer we never phoned.

## Which mistake costs more?

This is the whole point of the exercise, and the two mistakes are **not equal**:

- **False Positive** (wasted call) → you burned five minutes.
- **False Negative** (missed buyer) → you lost a customer and their deposit.

**A missed buyer costs far more than a wasted call.** Every decision in Part 3
follows from that asymmetry.

## Confusion matrix

The **confusion matrix** is just that 2×2 grid with the actual counts filled in.
Nothing more. It's called "confusion" because it shows you exactly where the
model gets confused.

Here's our XGBoost model at threshold 0.5, on the 8,238 test customers:

| | Predicted: don't call | Predicted: call |
|---|---|---|
| **Actually no** | 6,322 correctly skipped | 988 wasted calls |
| **Actually yes** | **315 MISSED buyers** | 613 buyers found |

Read it: of 928 real buyers, we'd find 613 and miss 315. We'd make 1,601 calls
total, of which 988 are to people who say no.

**The bottom-left cell is the expensive one.**

---
---

# PART 3 — METRICS BUILT FROM THOSE FOUR BOXES

Every metric below is just a different way of dividing those four numbers. Once
you see that, they stop being intimidating.

## Accuracy

**Definition:** of all guesses, what fraction were right?

```
accuracy = (True Negatives + True Positives) / everyone
         = (6,322 + 613) / 8,238
         = 0.8418  →  84.18%
```

**This is the metric everyone reaches for first, and on our data it is a trap.**
Part 8, Q1 explains exactly why. Short version: 88.73% of our customers say no,
so a model that blindly answers "no" to everyone scores 88.73% while finding zero
buyers. **Accuracy rewards ignoring the rare answer.**

## Precision

**Definition:** of the people we **called**, what fraction actually said yes?

```
precision = True Positives / (True Positives + False Positives)
          = 613 / (613 + 988)
          = 0.383  →  38.3%
```

**Precision is about wasted effort.** Low precision means your callers spend
their day getting hung up on. Read it as: *"when we call someone, how often are
we right?"*

## Recall

**Definition:** of the people who **would have said yes**, what fraction did we
actually call?

```
recall = True Positives / (True Positives + False Negatives)
       = 613 / (613 + 315)
       = 0.661  →  66.1%
```

**Recall is about missed money.** Low recall means real buyers you never phoned.
Read it as: *"of all the money out there, how much did we go after?"*

> **The one-line way to keep them apart:**
> **Precision** = "when we call, are we right?" (measures waste)
> **Recall** = "of the buyers, did we get them?" (measures misses)
>
> They pull against each other. Call everyone → recall is 100%, precision is
> terrible. Call only your single most confident person → precision might be
> 100%, recall is ~0%.

## F1

**Definition:** one number that balances precision and recall.

```
F1 = 2 × (precision × recall) / (precision + recall)
   = 0.485
```

It's the **harmonic mean** — a type of average that punishes imbalance. If either
precision or recall is terrible, F1 is terrible, even if the other is perfect. A
plain average would let a great score on one hide a disaster on the other.

**Use it for a quick comparison. Never use it alone** — it collapses the
precision/recall tradeoff into one number and throws away the decision you
actually care about.

---
---

# PART 4 — THE METRICS THAT ACTUALLY MATTER HERE

Everything in Part 3 needed a threshold. These don't — and that's the point.

## ROC curve

**ROC** stands for "Receiver Operating Characteristic". The name is a historical
accident (it comes from WWII radar operators deciding whether a blip was a real
plane). **Ignore the name — it tells you nothing.**

Here's what it actually is. Remember the threshold can be anything from 0 to 1.
For **every possible threshold**, plot one dot:

- **x-axis — False Positive Rate:** of the people who'd say no, what fraction did
  we call anyway? (*wasted effort*)
- **y-axis — True Positive Rate:** of the people who'd say yes, what fraction did
  we call? (*money captured* — this is exactly **recall**, same thing, different name)

Slide the threshold from 1.0 down to 0.0 and the dots trace a curve.

**The diagonal line is random guessing.** Call 30% of people at random, you get
30% of the buyers. No skill. **The further the curve bulges above that diagonal,
the better the model.**

## AUC — the headline number

**AUC** = **A**rea **U**nder the **C**urve. Literally: measure the area
underneath that ROC curve.

- **0.5** = the curve *is* the diagonal = worthless, a coin flip
- **1.0** = perfect
- **Our XGBoost: 0.8156**

**The plain-English meaning is the one to memorise:**

> **Pick one random buyer and one random non-buyer. AUC is the probability the
> model gives the buyer the higher score.**

At 0.8156, our model gets that pairwise ordering right about **82% of the time**.

### Why AUC is the right metric for us

1. **It measures ranking** — and ranking is literally our job. We're sorting a
   call list, not passing judgment.
2. **It's threshold-independent.** It summarises performance across *every*
   possible cutoff, so it doesn't bake in a business decision nobody has made yet.
3. **It can't be gamed by the common answer.** The "always say no" model scores
   88.73% accuracy and **exactly 0.500 AUC** — correctly identified as worthless.

## Decile

**Decile** = one tenth. Sort all customers by predicted probability, best first,
then chop the list into 10 equal groups.

- **Decile 1** = the 10% the model likes most (824 customers)
- **Decile 10** = the 10% it likes least

(Same idea as "percentile" from exam results — just tenths instead of hundredths.)

## Lift — the number the business cares about

**Definition:** how much better is this group than random?

```
lift = this group's response rate ÷ the baseline response rate
```

- **lift = 1.0** → no better than random. The model added nothing.
- **lift = 2.0** → twice as good as calling at random.
- **lift = 0.5** → *half* as good as random. Calling these people loses money.

**Our results:**

| Decile | Customers | Responders | Response rate | Lift |
|---|---|---|---|---|
| **1** | 824 | 437 | **53.03%** | **4.71x** |
| 2 | 824 | 182 | 22.09% | 1.96x |
| 3 | 824 | 70 | 8.50% | 0.75x |
| 10 | 823 | 27 | 3.28% | 0.29x |

**Read decile 1 out loud:** call 824 people at random and about 93 say yes. Call
the 824 people the model ranked highest and **437 say yes.** Same phone bill.
**4.71× the signups.**

That's why lift is the number you show a marketing director — it maps straight
onto their budget.

**If the model were worthless**, every decile would sit at ~11.27% and every lift
would be 1.0, because sorting by a meaningless score just shuffles people. The
gap between 53.03% and 3.28% *is* the model's value made visible.

## Cumulative gains (capture rate)

Lift-per-decile answers "how good is decile 4?" But nobody calls decile 4 on its
own — you work **down** the list until budget runs out. So the more useful
question is cumulative:

> **Call the top 30% of the list → you find 74.2% of all the buyers.**

Cut call volume by 70%, keep three-quarters of the revenue. That's a staffing
decision a director can act on this quarter.

> **Why lift changes with how deep you call:** the top decile is 4.71x. Call the
> top 20% and it drops to 3.34x — because you've now included decile 2, which is
> good but not as good as decile 1, and the average dilutes. Both numbers are
> correct. **Quote 4.71x (top decile) but know it falls as you call deeper**, and
> say so if asked.

## AUC vs lift — say both

- **AUC** is the statistically honest summary. It's what a data scientist wants.
- **Lift** is what gets the project funded. It's what a marketing director wants.

Quoting only AUC to a marketing director is a failure to communicate. Quoting
only lift to a data scientist looks like you're hiding something.

---
---

# PART 5 — THE PROBLEMS IN THE DATA

## Class imbalance

**Class** = one of the possible answers. We have two: "yes" and "no".
**Class imbalance** = one class massively outnumbers the other.

**Ours: 11.27% yes, 88.73% no. About 7.9 non-buyers for every buyer.**

This is normal for anything worth predicting — fraud, disease, churn,
click-through. **The interesting thing is always rare. That's *why* it's worth
predicting.** But it breaks accuracy completely (Part 8, Q1).

## Target leakage — the most important concept here

**Definition:** a feature that secretly contains the answer, and that **won't
exist when you actually need to make the prediction.**

**Our example, and the single best thing in this project:**

`duration` = how many seconds the call lasted. It is by far the strongest
predictor in the dataset:

| Call length | Response rate |
|---|---|
| Under 1 min | 0.02% |
| Over 10 min | 48.59% |

Keeping it pushes AUC to about **0.94**. We **deleted it.** Two reasons:

1. **It doesn't exist at prediction time.** You only learn how long a call lasted
   *after making the call*. But the model's entire job is deciding **who to
   call** — before any call happens.
2. **It partly *is* the answer.** A 20-minute call means the customer was
   interested. Predicting "yes" from a long call is circular reasoning.

**Why this matters more than any metric:** a leaky model scores brilliantly in
testing and is worthless in production. Plenty of tutorials on this dataset report
0.94 AUC. They built a model that must know the call's length in order to decide
whether to make the call. **Our 0.82 is lower and it's real.**

## Overfitting

**Definition:** the model **memorises** the training data — including its random
noise — instead of learning patterns that generalise to new people.

**How you spot it:** a big gap between the score on data it trained on and data it
didn't.

```
Logistic Regression   train 0.7959 | test 0.8010 | gap -0.0051  (healthy)
XGBoost               train 0.8297 | test 0.8156 | gap +0.0140  (healthy)
```

A model always does a bit better on data it's seen. **The question is how much
better.** Ours are fine.

> **Why is Logistic Regression's gap *negative* (better on the test set)?** Not a
> bug. A heavily restrained model isn't fitting the training data tightly in the
> first place, and an 8,238-row test set has its own luck. Small negative gaps are
> noise, not achievement.

## Correlation — and its big blind spot

**Definition:** a number from **−1 to +1** describing how two columns move
together, **in a straight line**.

- **+1** = perfect lockstep (one goes up, the other always goes up)
- **−1** = perfect opposites
- **0** = no *linear* relationship

**The blind spot that nearly cost us:** correlation only sees **straight lines**.

Age has a **U-shape** in our data: under-25s respond at 23.97%, over-65s at
47.21%, but the 35–54 middle at only 8.65%. Young and old both high, middle low.

Averaged into a single straight line, that U **cancels itself out to nearly
zero**. Age didn't even make our top 12 by correlation — despite over-65s
responding at more than 4× the baseline.

> **The rule:** low correlation means *"no **linear** relationship"* — a far
> smaller claim than *"useless"*. **Never drop a feature because correlation is
> low.** A common beginner move that would have deleted one of our best predictors.

## Multicollinearity

**Definition:** features that duplicate each other. ("Multi" = many, "collinear" =
lying on the same line.)

**Why it's bad:** if two features carry the same information, a linear model can't
tell which one deserves the credit. It splits it arbitrarily, and both look weak.
The coefficients become unstable and untrustworthy.

**How I got bitten by this:** I engineered three "ratio" features in SQL, like
`euribor_vs_avg` = euribor ÷ the average euribor. The correlation check came back
at **exactly r = 1.000** with the source column.

A correlation of exactly 1.000 is a mathematical proof that two columns are **the
same column in different units**. The average is a **constant** — the same number
for all 41,188 rows — so dividing by it just rescales. Zero new information.

> **The rule I took from it:** a ratio feature only earns its place if the
> **denominator varies per row**. A ratio against a global constant is a rescale
> wearing a feature's costume.

## Sentinel value

**Definition:** a magic number that means something other than its face value.

**Ours:** `pdays` = days since we last contacted this customer. It looked like a
normal number (min 0, max 999, average 962). But **999 isn't 999 days** — it's a
code meaning **"never contacted before"**, and it applies to 96.3% of rows.

That average of 962 is meaningless — it's the average of a *code* and real
measurements mixed together.

Decoded properly, it's one of the strongest signals in the data:

| | Customers | Response rate |
|---|---|---|
| Never contacted | 39,673 | 9.26% |
| Contacted before | 1,515 | **63.83%** |

We split it into a yes/no flag plus a real recency number — two different
questions deserving two features.

## 'unknown' pretending to be clean data

Our NULL check (**NULL** = a database's way of saying "no value here") came back
perfectly clean: zero missing values anywhere.

**That was a trap.** This dataset encodes missing values as the *text string*
`'unknown'`. The `default` column is **20.87% unknown**.

> **The habit:** never trust a clean NULL check. Go look at what the placeholder
> actually is.

---
---

# PART 6 — THE TECHNIQUES

## One-hot encoding

**The problem:** models do arithmetic. They can't read `"student"`.

**The naive fix, which is wrong:** number them — student=1, retired=2, admin=3.
This **invents an order that doesn't exist**. The model would infer admin >
retired, and that retired is the *average* of student and admin. Nonsense.

**One-hot encoding:** replace the one text column with **one 0/1 column per
category**:

| job | → | job_student | job_retired | job_admin |
|---|---|---|---|---|
| student | | **1** | 0 | 0 |
| retired | | 0 | **1** | 0 |

Exactly one column is "hot" (1) per row — hence the name. It refuses to imply any
ranking.

**Where order is real, exploit it.** Education genuinely has an order (basic.4y <
basic.6y < high school < degree). We mapped it to approximate years —
`education_years` — preserving the ordering one-hot would discard.

> **The rule:** encode order when it's real; never invent it when it isn't.

This is why 20 SQL features became **92** columns.

## Scaling (standardisation)

**Definition:** rescaling every feature to have an average of 0 and a similar
spread, so they're all in comparable units.

**Why Logistic Regression needs it:** `nr_employed` is around 5,000 while our
flags are 0 or 1. The model penalises large coefficients (see *regularization*).
So a feature measured in thousands needs a tiny coefficient to have any effect —
and dodges the penalty. A 0/1 flag needs a big coefficient and gets punished for
it.

> **Without scaling, you rank features by their units rather than their
> usefulness.**

**Why XGBoost doesn't need it:** trees ask *"is x > 5000?"*. Rescale x and the
threshold rescales identically. The tree is unchanged. Scaling a tree model is
harmless and pointless.

## Imputation

**Definition:** filling in missing values.

We use the **median** (the middle value when sorted) rather than the **mean**
(the average), because the median isn't dragged around by extreme outliers.

**We only impute for Logistic Regression**, which can't do arithmetic with a
missing value. XGBoost handles missing natively (below).

## Pipeline

**Definition:** an object that chains preprocessing steps and the model together,
so they always run in the right order on the right data.

**It exists to prevent one specific, sneaky bug.** Scaling needs an average.
Compute that average over the *whole* dataset before splitting, and the average
you used on training data was partly calculated **from the test data**. The test
set has leaked into training. Your score is flattered. Nothing warns you.

A Pipeline makes that structurally impossible: `.fit()` learns the average from
**training data only**; `.predict()` reuses that stored value. **You get the right
behaviour by default instead of by remembering.**

## Class weights

**Definition:** telling the model that a mistake on the rare class costs more.

`class_weight='balanced'` weights each class inversely to how common it is. With
our 7.9:1 imbalance, **each buyer counts about 7.9× as much** as a non-buyer when
computing training error. Ignoring the rare class stops being the easy win it was.

XGBoost's version of the same idea is `scale_pos_weight` — we set it to
**7.88** (29,238 non-buyers ÷ 3,712 buyers, computed from training data only).

**No data is invented.** We just stop pretending both mistakes cost the same.

## SMOTE (and why we rejected it)

**SMOTE** = **S**ynthetic **M**inority **O**ver-sampling **TE**chnique. It
manufactures fake examples of the rare class by finding two real buyers and
inventing new ones **on the line between them** — a blend.

For continuous features that's reasonable: a synthetic customer aged 41.6 is a
plausible person.

**But 80 of our 92 features (87%) are one-hot flags** — and there is nothing
halfway between "student" and "retired". I ran it and printed what it actually
produced:

```
age                = 26.52
job_blue-collar    = 0.520
job_student        = 0.480     -> job_* sums to 1.000
marital_single     = 0.480
marital_unknown    = 0.520
```

**This "person" is 52% blue-collar and 48% student. Simultaneously.**
Arithmetically tidy, humanly impossible. It built 25,526 of these and matched
class weights on AUC anyway (0.7998 vs 0.8010).

> **SMOTENC** is the variant that handles categories properly. It's the right
> answer if you're committed to oversampling. We chose not to invent data at all.

## Regularization

**Definition:** a penalty the model pays for having large coefficients —
discouraging it from leaning too hard on any single feature.

**Why:** an unrestrained model will contort itself to fit the training data
perfectly, including its noise. That's overfitting. Regularization keeps it
simpler and more general.

It's also why scaling matters — the penalty is applied to the coefficients, so
the features must be in comparable units for the penalty to be fair.

## Early stopping

**Definition:** stop adding trees when performance on unseen data stops improving.

XGBoost adds trees one at a time; each fits the training data slightly better.
Past a point it stops learning real patterns and starts memorising noise. Early
stopping watches performance on data it isn't training on and stops when it
plateaus. **It also means we don't have to guess the right number of trees — the
data decides.** Ours chose **99 out of a possible 1,000**.

**The discipline that matters:** that watching data — the **validation set** — was
carved out of the **training** data, never the test set. Early-stop on the test
set and you've used it to make a modelling decision, and it's no longer an honest
estimate of unseen performance.

> **Three sets, three jobs:** *training* = learn from. *validation* = tune on.
> *test* = graded on, once, at the very end.

## Stratified split

**Definition:** splitting the data so that both halves keep the **same class
ratio**.

With only 4,640 buyers, an unlucky random split could hand the test set 9% buyers
and training 12%. You'd be training on one problem and grading on a slightly
different one, and the gap would show up as a mysterious score difference you'd
waste a day chasing.

`stratify=y` forces 11.27% in both halves. Ours came out 11.27% train / 11.26%
test.

## Random seed

**Definition:** a fixed number that makes "random" operations repeatable.

Splitting data involves random choices. Random means different every run — your
AUC would change each time and you could never tell a real improvement from luck.
**A fixed seed makes the randomness repeatable:** same split, same numbers, every
run, on any machine. Ours is 42 (a convention, not magic).

**That's what "reproducible" means in practice** — and why the number on your
resume will still be true tomorrow.

## Cross-validation (what we didn't do)

**Definition:** instead of one train/test split, split the data 5 (or 10) ways,
train 5 models, and average the scores.

**Why it's better:** you get error bars — a sense of how much your score wobbles.
**Why we skipped it:** with 41k rows and a stratified split, a single test
estimate is stable enough. Worth knowing the term; be ready to say you'd use it
with less data.

---
---

# PART 7 — THE MODELS

## Logistic Regression

**What it does:** draws **one straight boundary** through the feature space and
asks which side you're on. It learns exactly **one number per feature** — a
**coefficient** — and those coefficients *are* the entire model. You can print
them, read them, and defend them to a regulator.

> **The name is a lie.** Despite "regression", it's a **classifier**. Ignore the
> name; it's a historical artifact.

**Ours scores 0.8010 AUC.** It's our **baseline** — the thing that makes any other
score mean something.

## Decision tree

**What it does:** a flowchart of yes/no questions. *"Was the customer contacted
before? → Is euribor below 2%? → predict yes."* Each question splits the group in
two.

Trees find **non-linear** patterns naturally — a straight line can't express "high
at both ends, low in the middle" but a tree just asks two questions.

## Gradient boosting and XGBoost

**Boosting:** build many small trees **in sequence**, where each new tree focuses
on the mistakes the previous ones made. **Gradient** refers to the maths used to
identify those mistakes.

**XGBoost** = e**X**treme **G**radient **B**oosting. A fast, heavily optimised
implementation. It's the default winner on tabular data (data in rows and
columns, like a spreadsheet — as opposed to images or text).

**Ours: 99 trees, 0.8156 AUC.**

## What each model needed

| | Logistic Regression | XGBoost |
|---|---|---|
| Missing values | Must impute | **Handles natively** |
| Scaling | **Required** | Irrelevant |
| Imbalance | `class_weight='balanced'` | `scale_pos_weight=7.88` |
| Interpretability | Built in | Needs SHAP |
| Non-linearity | Only what you engineer | Finds it itself |

**Why XGBoost's native missing-value handling matters here:**
`days_since_prev_contact` is NULL for 96% of customers **because they were never
contacted**. That's a *fact about the customer*, not an absence of data. XGBoost
learns which direction to send missing values at each split — treating "missing"
as information. Imputing a median there tells a small lie.

## SHAP

**SHAP** = **SH**apley **A**dditive ex**P**lanations.

**Where it comes from:** cooperative game theory. Lloyd Shapley (Nobel Prize,
2012) asked: *if a team wins, how much did each player contribute?* SHAP treats
the **features** as the team and the **prediction** as the win.

**What it tells you:** for every single customer, how much did each feature push
*this person's* score up or down, away from the average customer's score?

**Its guarantee — additivity:**

```
average prediction + sum of all SHAP values = this customer's prediction
```

Nothing hand-waved, nothing left over. **I verified this on our model** — across
8,238 customers, the largest disagreement was **0.0000014**.

> **Why bother verifying?** Our model used early stopping, so the saved object
> contains more trees than it actually predicts with. If SHAP had explained the
> unused trees, we'd be reading an explanation of a model that doesn't exist. The
> check would catch that.

**Why not XGBoost's built-in `feature_importances_`?** It counts how often a
feature was used in a split. That says nothing about **direction**, and it
inflates features that simply offer more places to split.

### Mean |SHAP| — and the trap in reading it

**Mean |SHAP|** (mean *absolute* SHAP) = the average size of a feature's push,
ignoring whether it pushed up or down. We take the absolute value first because a
feature that pushes some people strongly up and others strongly down is very
important — but its raw average would be ~0 and look like it does nothing.

**The trap:** mean |SHAP| averages over **everyone**. It answers *"how much does
this feature move the **average** customer?"* — **not** *"how much does it matter
to a customer it applies to?"*

For a rare-but-decisive feature those are wildly different numbers. Part 8, Q3
has the full story.

---
---

# PART 8 — THE FOUR INTERVIEW QUESTIONS

Now that you have the vocabulary, these should read easily. Each has a
**30-second answer** (what you say) and a **full picture** (for the follow-up).

---

## Q1: Why AUC and lift instead of accuracy?

### The 30-second answer

> "Because only 11% of customers say yes. A model that always predicts 'no' scores
> 88.7% accuracy and finds zero buyers — I actually built that model to check, and
> its AUC is 0.500, which correctly calls it worthless. Accuracy rewards ignoring
> the rare class, which is the only class we care about.
>
> AUC measures *ranking*: given one buyer and one non-buyer, it's the probability
> the model scores the buyer higher. Mine is 0.82. That's the right metric because
> ranking is literally the job — we're sorting a call list, not delivering verdicts.
>
> Lift is what I'd show the business. Top decile converts at 53% versus an 11.3%
> baseline — 4.71x. Same phone bill, 4.71x the signups."

### The full picture

**I built the dumbest possible model** — ignores all 92 features, always answers
"no":

| Metric | Score |
|---|---|
| **Accuracy** | **0.8874** |
| Recall | 0.0000 |
| AUC | 0.5000 |
| Real buyers found | **0 of 928** |

88.74% accuracy. Zero customers identified. The marketing team gets an empty call
list. **Any metric a lazy model can max out isn't measuring what you care about.**

**And it's not just the dummy.** Untouched Logistic Regression scored the *best*
accuracy of anything I trained — 0.9020 — while missing **718 of 928 buyers**. The
class-weighted version scored **lower** accuracy (0.8333) and found **2.9× as
many buyers**.

> **Lower accuracy, far better model. Sit with that sentence — it's the whole
> lesson.**

### If they push: "isn't F1 enough?"

> "F1 balances precision and recall into one number, and I report it. But it hides
> the tradeoff I actually care about. At threshold 0.3 I find 800 of 928 buyers
> with 3,735 wasted calls; at 0.7 I find 532 with 547 wasted. Same model, same
> AUC, completely different campaigns. F1 collapses that into one number and
> throws away the decision."

---

## Q2: How did you handle class imbalance, and why that method?

### The 30-second answer

> "Class weights — `class_weight='balanced'` for Logistic Regression,
> `scale_pos_weight=7.88` for XGBoost. It weights each class inversely to its
> frequency, so with a 7.9:1 imbalance a responder counts about 7.9x as much
> during training.
>
> I tested SMOTE against it and rejected it. 87% of my features are one-hot flags,
> and SMOTE interpolates between examples — so it generated customers who were 52%
> blue-collar and 48% student simultaneously. I printed one. It invented 25,526
> impossible people and matched class weights on AUC anyway: 0.7998 vs 0.8010.
>
> The tradeoff is real: I accept 1,046 wasted calls to find 391 more buyers. That's
> only correct because a deposit is worth more than five wasted minutes. If a call
> cost $500 and a deposit earned $10, I'd flip it."

### The full picture

All three tested on the identical test set:

| Approach | Accuracy | Recall | AUC | Buyers found | Buyers missed | Wasted calls |
|---|---|---|---|---|---|---|
| Ignore it | **0.9020** | 0.2263 | 0.8015 | 210 | **718** | 89 |
| **Class weights** | 0.8333 | 0.6476 | 0.8010 | **601** | 327 | 1,046 |
| SMOTE | 0.8303 | 0.6573 | 0.7998 | 610 | 318 | 1,080 |

**Why class weights won:**

1. **SMOTE invents impossible customers on this data** (see Part 6).
2. **Same result, more machinery** — SMOTE matched on AUC while adding a
   resampling step, 25,526 synthetic rows, and a leakage risk.
3. **It's honest about the real problem.** We don't have a *data shortage* — we
   have a **cost asymmetry**. Class weights say that directly. SMOTE encodes the
   same idea by faking a world where buyers are common. They aren't.

### The thing most people miss

Look at the AUC column: **0.8015 / 0.8010 / 0.7998. Almost identical.**

**Class weights don't make the model smarter — they move the threshold.** The
model's ability to *rank* barely changes.

That's worth understanding precisely: **if you only care about ranking (and for a
call list, you do), imbalance handling matters far less than people assume.** It
matters when you need hard yes/no predictions at a sensible default cutoff.

### If they push: "when WOULD you use SMOTE?"

> "When the minority class is genuinely tiny — a few dozen cases, not 4,640 —
> where the model needs more *examples* rather than a stronger *signal* about the
> ones it has. And even then I'd use SMOTENC, which handles categoricals properly.
> The other thing I'd watch is that SMOTE only ever touches training data.
> Oversample before you split and copies of the same customer land in train and
> test — the model gets graded on people it memorised."

---

## Q3: What did SHAP reveal, and what action does it imply?

### The 30-second answer

> "Three things. First, the economy dominates — `nr_employed` and `euribor3m` are
> 54% of total importance. When rates are high and employment is strong, people
> say no, because a term deposit competes badly. That says *time the campaign*,
> not just the call list.
>
> Second, and this is the honest part: that's the model's biggest **weakness**,
> not its best feature. Those variables barely move within a week — they describe
> *when* we called, not *who*. The model partly learned '2008–2010 was a good time
> to sell deposits'. True, and useless for ranking today's list. I'd ship a variant
> without them; it would score lower and might rank better.
>
> Third, the actionable one: prior buyers respond at 65.8% versus an 11.3%
> baseline. Call them first — that needs one SQL query, no model. And sequence
> follow-ups inside 7 days: contacts within a week convert at ~66% vs ~9% for
> never-contacted."

### The full picture

| Theme | Share of importance |
|---|---|
| **Macroeconomic conditions** | **54.0%** |
| Demographics | 15.4% |
| Prior relationship / recency | 10.2% |
| Timing | 9.5% |
| Contact channel | 8.7% |
| Campaign intensity | 2.3% |

**Driver 2 — contact channel.** Mobile **14.9%** vs landline **5.0%**. The
highest-ranked driver the bank actually **controls** — the economy is weather,
this is a decision.

> **Caveat:** correlation, not causation. Mobile proxies for age and reachability.
> *"Call them on a mobile instead"* would do nothing — nobody's landline is
> stopping them buying. *"Prioritise people who have one"* still works, because
> you're using it as a **signal**, not an **intervention**. That distinction is the
> whole game.

### Driver 3 — and the best thing in this project

Prior buyers respond at **65.8%**. **So why does `prev_success` rank only 20th of
92?** Two effects:

**1. Dilution by averaging.** Mean |SHAP| averages over all 8,238 customers. Only
**291 (3.5%)** have any contact history — the feature is exactly 0 for everyone
else.

```
mean |SHAP| over everyone            : 0.0189
mean |SHAP| over those WITH history  : 0.1429   -> 8x stronger
```

**The feature isn't weak. It's rare AND decisive**, and averaging hides that.

**2. Credit-splitting.** I engineered `prev_success`, `was_contacted_before`,
`days_since_prev_contact` and `poutcome_success` to encode overlapping facts. SHAP
divides credit between correlated features, so each looks modest while the theme
is substantial. **That's SHAP behaving correctly** — and why I report themes too.

> **Say this and you sound like someone who has actually used SHAP:** never read
> mean |SHAP| as "how much this feature matters to a customer". It means "how much
> it moves the **average** customer". For a rare, decisive feature those are wildly
> different numbers.

### If they push: "so is SHAP telling you what causes conversion?"

> "No, and that's an important limit. SHAP explains **the model**, not the world.
> If the model learned a spurious pattern, SHAP will faithfully explain the
> spurious pattern. That's exactly why I flagged the economic features as a
> weakness — SHAP told me they dominate, and judgment told me that's a problem
> rather than a finding. Anything causal needs an A/B test."

---

## Q4: Logistic Regression vs XGBoost — the tradeoff, and which would you ship?

### The 30-second answer

> "XGBoost, but the case is closer than the numbers suggest and I'd say so. 0.8156
> vs 0.8010 — it wins by 0.0146 AUC, which is a small gain for a model that's
> dramatically harder to explain and maintain.
>
> The signal in this data is mostly linear, so the straight line already captures
> most of it. I'd ship XGBoost because SHAP gives back the interpretability I gave
> up, and nobody has a right to an explanation for not being phoned — the cost of
> being wrong is five wasted minutes.
>
> If this were credit approval, I'd ship Logistic Regression without hesitating and
> consider 0.0146 a fair price for a model I can defend line by line to a
> regulator. **The deciding factor is the deployment context, not the score.**"

### The full picture

**Where XGBoost genuinely earns it:** age is **U-shaped** (Part 5). Its linear
correlation with the target is near zero. A straight line averages that U away to
nothing. XGBoost finds it effortlessly.

**But I gave Logistic Regression a fighting chance** by engineering `age_band` in
SQL — handing it the non-linearity it can't discover.

> **That's the real tradeoff: with XGBoost the model does that work; with Logistic
> Regression you do.** Part of why the gap is only 0.0146 is that I did that work.

### If they push: "only 0.015 — is XGBoost even worth it?"

This isn't a trap, it's the actual question.

> "On this data, barely. The signal is mostly linear, so the straight line already
> captures it. I'd still ship XGBoost because SHAP recovers the interpretability
> and the context tolerates a black box — but if the team needed to explain
> decisions, I'd ship the baseline and lose almost nothing. The honest summary is
> that the baseline is doing most of the work here, and **the only reason I know
> that is that I built it.** If I'd only trained XGBoost, 0.8156 would be a number
> with nothing to compare it to."

---
---

# PART 9 — OTHER QUESTIONS THEY MIGHT ASK

## "Walk me through the project."

> "Bank runs phone campaigns for term deposits; 89% say no, so calling everyone
> wastes money. I loaded 41,188 customers into PostgreSQL, explored it in SQL, and
> built a feature view there — decoding sentinels, bucketing, flags, ratios. Python
> turned that into 92 features, mostly one-hot encoded.
>
> The most important decision was dropping `duration`. It's the strongest predictor
> in the data and it's target leakage — you only know call length after the call,
> but the model runs before it.
>
> Trained Logistic Regression as a baseline and XGBoost as the main model, both with
> class weights for the 11.27% imbalance. XGBoost got 0.82 AUC and 4.71x top-decile
> lift. SHAP said the economy dominates — which I read as the model's weakness, not
> its strength. Then a Streamlit dashboard so the team gets a ranked call list."

## "What went wrong / what was hardest?"

Pick one, be specific. All real:

> "I engineered three ratio features in SQL — euribor divided by the average
> euribor, that sort of thing. The correlation check came back at exactly r = 1.000
> with their source columns. I'd divided by a **constant**, which is just a
> rescale — zero new information. The rule I took from it: a ratio feature only
> earns its place if the denominator varies per row."

> "I predicted `prev_success` would be the top customer-level driver, because the
> SQL showed prior buyers converting at 65%. SHAP ranked it 20th. I was wrong about
> the mechanism — mean |SHAP| averages over all customers and only 3.5% have any
> history, so a rare-but-decisive feature gets diluted. Conditioned on the people it
> applies to, it's 8x stronger. It taught me what that metric actually measures."

> "I claimed my data-loading script was safe to re-run, and it wasn't. Dropping the
> raw table failed because a SQL view depended on it — and it could only fail on the
> *second* run, so a fresh database always looked fine. Running the full pipeline
> twice is what caught it."

## "How would you improve it?"

> "Three things, in order. **Drop the economic features** and see what happens —
> they're 54% of importance and mostly describe when we called, not who. AUC would
> fall; ranking might improve. **Tune hyperparameters** — I used sensible defaults
> with early stopping, never ran a search. **Tune the threshold to real economics** —
> I report a sweep, but the right cutoff needs actual cost-per-call and
> value-per-deposit, which only the business has."

## "Is this model fair?"

Do not wing this one.

> "I haven't audited it, and I'd want to before deploying. It uses age, job, marital
> status and education — in the EU, age is a protected characteristic and the others
> proxy for socioeconomic status. For a marketing call list the harm is low: the
> downside of being excluded is not getting a phone call. For anything touching
> credit or pricing, that same feature set would be a serious problem and I'd expect
> a fairness review before it went near production."

---
---

# GLOSSARY — every term, one line each

| Term | Plain English |
|---|---|
| **Accuracy** | Of all guesses, what fraction were right? Useless on imbalanced data. |
| **AUC-ROC** | Given one buyer and one non-buyer, the probability the model scores the buyer higher. 0.5 = random, 1.0 = perfect. Ours: 0.8156. |
| **Baseline** | What you'd get with no model — calling at random. Ours: 11.27%. |
| **Binary classification** | Predicting which of exactly two categories something belongs to. |
| **Boosting** | Building many small trees in sequence, each fixing the last one's mistakes. |
| **Class** | One possible answer. We have two: yes and no. |
| **Class imbalance** | One class vastly outnumbers the other. Ours: 88.73% no, 11.27% yes. |
| **Class weights** | Telling the model a rare-class mistake costs more than a common-class one. |
| **Classifier** | A model that predicts a category (vs. regression, which predicts a number). |
| **Coefficient** | The single number Logistic Regression learns per feature. Its whole model. |
| **Confusion matrix** | The 2×2 grid of right/wrong by predicted/actual. |
| **Correlation** | −1 to +1: how two columns move together **in a straight line**. Blind to curves. |
| **Cross-validation** | Splitting data several ways and averaging, to get error bars. We didn't. |
| **Cumulative gains** | Call the top N% of the list → what share of all buyers do you find? |
| **Decile** | One tenth of the sorted list. Decile 1 = the model's top 10%. |
| **Early stopping** | Stop adding trees when validation performance plateaus. Ours chose 99. |
| **F1** | One number balancing precision and recall (harmonic mean). Never use alone. |
| **False Negative** | A real buyer we never called. **The expensive mistake.** |
| **False Positive** | A call to someone who says no. Wasted five minutes. |
| **Feature** | One input column. We have 92. |
| **Gradient boosting** | Boosting that uses gradients (maths) to find the previous trees' errors. |
| **Harmonic mean** | An average that punishes imbalance — one bad number drags it all down. |
| **Imputation** | Filling in missing values. We use the median. |
| **Lift** | This group's response rate ÷ baseline. 4.71x = 4.71× better than random. |
| **Logistic Regression** | One straight boundary through the data. A classifier, despite the name. |
| **Mean \|SHAP\|** | Average push size, ignoring direction. Measures the **average** customer — not a customer the feature applies to. |
| **Multicollinearity** | Features duplicating each other. Destabilises linear models. |
| **NULL** | A database's way of saying "no value here". |
| **One-hot encoding** | One text column → one 0/1 column per category. Implies no ranking. |
| **Overfitting** | Memorising training data instead of learning general patterns. |
| **Pipeline** | Chained preprocessing + model, so steps can't leak or run out of order. |
| **Precision** | Of the people we **called**, what fraction said yes? (measures waste) |
| **Probability** | 0.0–1.0: the model's confidence. Sortable — which is the whole point. |
| **Random seed** | A fixed number making "random" repeatable. Ours: 42. |
| **Recall** | Of the people who'd say **yes**, what fraction did we call? (measures misses) |
| **Regression** | Predicting a number on a continuous scale (vs. classification). |
| **Regularization** | A penalty for large coefficients, discouraging overfitting. |
| **ROC curve** | Recall vs. false-positive-rate at every threshold. The name means nothing. |
| **Sentinel value** | A magic number meaning something else. `pdays=999` = "never contacted". |
| **SHAP** | How much each feature pushed one prediction away from average. |
| **SMOTE** | Inventing synthetic minority examples by interpolation. Rejected here. |
| **SMOTENC** | SMOTE that handles categorical features properly. |
| **Stratified split** | Splitting so both halves keep the same class ratio. |
| **Tabular data** | Data in rows and columns, like a spreadsheet. XGBoost's home turf. |
| **Target** | The thing being predicted. Ours: did they subscribe? |
| **Target leakage** | A feature secretly containing the answer, unavailable at prediction time. `duration`. |
| **Term deposit** | Savings product: lock money away for a fixed period, guaranteed return. |
| **Test set** | Data hidden during training. Graded on once, at the very end. |
| **Threshold** | The cutoff turning a probability into yes/no. **A business decision.** |
| **Training set** | The data the model learns from. |
| **True Negative** | Correctly skipped someone who'd say no. |
| **True Positive** | Correctly called someone who says yes. |
| **Validation set** | Carved from training data, used for tuning (e.g. early stopping). |
| **X and y** | Convention: `X` = the features table, `y` = the target column. |
| **XGBoost** | eXtreme Gradient Boosting. Fast boosted trees. Ours: 99 trees, 0.8156 AUC. |
