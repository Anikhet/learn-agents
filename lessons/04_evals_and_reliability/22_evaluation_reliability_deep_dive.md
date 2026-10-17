# 22. Evaluation Reliability Deep Dive

*Track A (agent engineering) with Track D (interviews). Course addition; extends the book's Chapter 19 (Evaluation and Monitoring). Prerequisites: 19. Evaluation and Monitoring. Companion: 37. Trajectory Evals and Statistics.*

Lesson 19 built an eval harness. This lesson asks the uncomfortable question that follows: how do you know the eval is telling the truth? A grader is a measurement instrument. Instruments have bias, noise, and drift. If you ship on a number you have not checked, you are not doing eval-driven development; you are doing vibes with extra steps.

## The problem

The billing agent's release gate is an LLM judge that scores each answer 0 to 1 against a rubric. Last sprint the average was 0.81. This sprint it is 0.86. The team is ready to ship.

Then someone runs the judge twice on the same outputs and gets 0.86 and 0.79. Someone else swaps the order of two answers in a pairwise comparison and the judge changes its mind. A third person notices that three of the fifty eval questions appear verbatim in a public benchmark the base model was almost certainly trained on.

```text
what the dashboard says            what is actually true
──────────────────────────         ────────────────────────────────────────
0.81 -> 0.86, ship it              judge noise alone spans 0.79 - 0.86
judge agrees with humans 85%       kappa is 0.4; most agreement is chance
A beats B in 70% of pairs          A is listed first in 70% of pairs
50 cases                           3 are contaminated, 20 are near-duplicates
"87% confident"                    right 60% of the time when it says 87%
```

Every one of those is a way an eval lies. The cure for each is a specific measurement you make about the eval itself. This lesson builds those measurements.

### The layers of trust

```text
   Does the grader agree with humans?           inter-rater agreement (kappa)
              │
   Does the grader agree with itself?           position bias, run-to-run variance
              │
   Do its confidences mean anything?            calibration
              │
   Is the metric the right shape?               pass@k vs pass^k, per-case vs aggregate
              │
   Is the sample big enough?                    confidence intervals, minimum n
              │
   Is the dataset clean?                        contamination, duplicates, leakage
              │
   ── only now ──►  trust the number
```

You do not need all six layers for every eval. You need to know which ones you skipped.

## Build it without a framework

No model is needed for most of this lesson: we are measuring graders, so the graders' outputs are the data. Where a judge appears it is simulated with seeded randomness so the effects are visible and reproducible. A live cell at the end runs the position-bias check against a real model.

### Layer 1: does the judge agree with humans?

Raw agreement is misleading when labels are imbalanced. If 80% of answers are good and the judge says "good" to everything, it agrees with humans 80% of the time while carrying zero information. Cohen's kappa corrects for the agreement you would get by chance:

```text
kappa = (p_observed - p_chance) / (1 - p_chance)

p_observed = fraction of items where both raters agree
p_chance   = sum over labels of P(rater1 = label) * P(rater2 = label)

worked example, 100 items, labels good/bad:
  humans: 80 good, 20 bad     judge: 90 good, 10 bad     both agree on 82
  p_observed = 0.82
  p_chance   = 0.8*0.9 + 0.2*0.1 = 0.74
  kappa      = (0.82 - 0.74) / (1 - 0.74) = 0.31        <- "fair", not "85% accurate"
```

Rules of thumb people quote (Landis and Koch, 1977): below 0.2 slight, 0.2 to 0.4 fair, 0.4 to 0.6 moderate, 0.6 to 0.8 substantial, above 0.8 almost perfect. Treat these as vocabulary, not thresholds. What matters is whether the judge is reliable enough for the decision you are making with it.

```{code-cell} python
import numpy as np

rng = np.random.default_rng(0)


def cohens_kappa(a, b):
    """Chance-corrected agreement between two label sequences."""
    a, b = np.asarray(a), np.asarray(b)
    labels = np.unique(np.concatenate([a, b]))
    p_obs = np.mean(a == b)
    p_chance = sum(np.mean(a == l) * np.mean(b == l) for l in labels)
    return float((p_obs - p_chance) / (1 - p_chance)) if p_chance < 1 else 1.0


# 200 answers; humans say 80% are good.
human = rng.random(200) < 0.80

# Judge A: a lazy judge that says "good" 95% of the time regardless.
judge_lazy = rng.random(200) < 0.95
# Judge B: a real judge that flips the human label 10% of the time.
judge_real = np.where(rng.random(200) < 0.10, ~human, human)

for name, judge in (("lazy", judge_lazy), ("real", judge_real)):
    agreement = np.mean(judge == human)
    print(f"judge {name:5s} raw agreement={agreement:.2f}  kappa={cohens_kappa(human, judge):.2f}")

k_lazy, k_real = cohens_kappa(human, judge_lazy), cohens_kappa(human, judge_real)
assert np.mean(judge_lazy == human) > 0.65                # looks respectable
assert k_lazy < 0.15 and k_real > 0.6                      # kappa exposes it
```

The lazy judge's raw agreement is above 70%. Its kappa is near zero. If your release gate is a judge you have only checked with raw agreement, you may be gating on a coin that always comes up heads.

How many human labels do you need to calibrate a judge? Enough to estimate kappa with a usable interval; in practice a few hundred items, stratified so that "bad" answers are over-represented. Labeling 200 outputs is a day of work and it is the best day you will spend on the eval.

### Layer 2: does the judge agree with itself?

Pairwise judging ("which of A or B is better?") is popular because humans find relative judgments easier than absolute scores, and so do models. It has a well-documented failure: position bias. Many judges prefer whichever answer is presented first (or last), independent of content.

The fix is mechanical. Ask twice with the positions swapped. If the verdicts disagree, the judge did not have an opinion; record a tie or discard the pair. Only count a win when both orderings agree.

```text
ask(A, B) -> "first"     ask(B, A) -> "first"     inconsistent: position, not content -> tie
ask(A, B) -> "first"     ask(B, A) -> "second"    consistent: A wins
```

```{code-cell} python
def make_judge(position_bias, rng):
    """Simulated pairwise judge. `position_bias` is P(pick first slot regardless of content)."""
    def judge(first, second):
        if rng.random() < position_bias:
            return "first"
        return "first" if first["quality"] > second["quality"] else "second"
    return judge


def pairwise_winrate(judge, pairs, swap=False):
    """Fraction of decided pairs where A wins. With swap=True only consistent verdicts count."""
    wins = ties = 0
    for a, b in pairs:
        v1 = judge(a, b)
        if not swap:
            wins += v1 == "first"
            continue
        v2 = judge(b, a)
        a_wins_1 = v1 == "first"
        a_wins_2 = v2 == "second"
        if a_wins_1 == a_wins_2:
            wins += a_wins_1
        else:
            ties += 1
    decided = len(pairs) - ties
    return wins / decided, ties / len(pairs)          # win rate among decided pairs


rng = np.random.default_rng(1)
# Ground truth: A and B are equally good on average. True win rate should be ~0.5.
pairs = [({"quality": rng.random()}, {"quality": rng.random()}) for _ in range(400)]
true_rate = np.mean([a["quality"] > b["quality"] for a, b in pairs])

biased = make_judge(position_bias=0.4, rng=np.random.default_rng(2))
naive_rate, _ = pairwise_winrate(biased, pairs)
swapped_rate, tie_rate = pairwise_winrate(make_judge(0.4, np.random.default_rng(2)), pairs, swap=True)

print(f"true A win rate       {true_rate:.2f}")
print(f"naive (A first)       {naive_rate:.2f}   <- inflated by position")
print(f"swap-and-agree        {swapped_rate:.2f}   ties discarded: {tie_rate:.2f}")
assert naive_rate > true_rate + 0.10          # bias is visible
assert abs(swapped_rate - true_rate) < 0.10   # swapping removes it
```

The swap costs two judge calls instead of one and turns a chunk of pairs into ties. That is the correct price. A tie is an honest answer; a biased win is a lie.

Other judge biases to know by name: verbosity bias (longer answers score higher), self-preference (a model rates its own family's style higher), and anchoring on the reference (if the reference is wrong the judge follows it). Position swapping fixes only the first. The rest are why Layer 1 exists.

### Layer 3: calibration, and the right shape of metric

A judge that outputs a confidence should be right about as often as it claims. Bucket the confidences, compare each bucket's stated confidence to its observed accuracy, and weight the gaps by bucket size. That is expected calibration error (ECE):

```text
ECE = sum over buckets  (n_bucket / N) * | mean_confidence_bucket - accuracy_bucket |

bucket 0.9-1.0:  120 items, mean confidence 0.94, accuracy 0.70   gap 0.24
bucket 0.7-0.9:   60 items, mean confidence 0.80, accuracy 0.75   gap 0.05
...
```

An overconfident judge has high stated confidence and lower accuracy. Its scores still rank answers, but you cannot read "0.9" as "90% likely correct," so do not put a threshold on it without checking.

The second half of this cell is about the shape of the aggregate. When each case is run k times:

```text
pass@k   = fraction of cases where at least one of k runs passed     "can it do this at all?"
pass^k   = fraction of cases where all k runs passed                  "can I rely on it?"
avg@k    = mean per-case pass rate                                    the expected single-run score
```

For a coding assistant with a human reviewing, pass@k is fine. For an agent that issues refunds unattended, pass^k is the metric, and it falls fast: a case that passes 90% of the time passes all of 5 runs only 59% of the time.

```{code-cell} python
def expected_calibration_error(confidence, correct, bins=5):
    edges = np.linspace(0, 1, bins + 1)
    ece, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lo) & (confidence < hi if hi < 1 else confidence <= hi)
        if not mask.any():
            continue
        conf, acc = confidence[mask].mean(), correct[mask].mean()
        ece += mask.mean() * abs(conf - acc)
        rows.append((lo, hi, int(mask.sum()), round(conf, 2), round(acc, 2)))
    return ece, rows


rng = np.random.default_rng(3)
n = 1000
truth = rng.random(n) < 0.7
# Overconfident judge: says 0.85-1.0 almost always, is right 70% of the time.
conf = np.clip(0.85 + 0.15 * rng.random(n), 0, 1)
correct = rng.random(n) < 0.70
ece, rows = expected_calibration_error(conf, correct)
for lo, hi, cnt, c, a in rows:
    print(f"  conf [{lo:.1f},{hi:.1f})  n={cnt:4d}  stated={c:.2f}  actual={a:.2f}")
print(f"ECE = {ece:.2f}  (0 is perfect)")
assert ece > 0.15

# pass@k versus pass^k on the same per-case data.
def pass_metrics(results):
    """results: cases x k boolean matrix."""
    return {"avg@k": results.mean(), "pass@k": results.any(axis=1).mean(),
            "pass^k": results.all(axis=1).mean()}


k, cases = 5, 200
per_case_p = np.full(cases, 0.9)                       # every case passes 90% of the time
results = rng.random((cases, k)) < per_case_p[:, None]
m = pass_metrics(results)
print({key: round(v, 2) for key, v in m.items()})
print(f"theory: pass^5 at p=0.9 is 0.9**5 = {0.9**5:.2f}")
assert m["pass@k"] > 0.97 and abs(m["pass^k"] - 0.9**5) < 0.08
```

The gap between pass@5 (near 1.0) and pass^5 (near 0.59) on the same runs is the gap between a demo and a product. Report the one that matches how the agent will be used.

### Layer 4: variance across runs and how many samples you need

A pass rate measured on n cases is an estimate with an interval around it. Roughly, the standard error of a proportion is `sqrt(p(1-p)/n)`, so a 95% interval is about `±1.96 * sqrt(p(1-p)/n)`:

```text
p = 0.8, n = 20     ±0.175    "80%" means somewhere in 62% - 98%
p = 0.8, n = 50     ±0.111
p = 0.8, n = 200    ±0.055
p = 0.8, n = 1000   ±0.025
```

To detect a change from p1 to p2 between two versions with 80% power at 5% significance, an approximate minimum n per arm is:

```text
n ≈ (z_alpha + z_power)^2 * (p1(1-p1) + p2(1-p2)) / (p2 - p1)^2
  with z_alpha = 1.96 (two-sided 5%), z_power = 0.84 (80%)

detect 0.80 -> 0.85:   (2.8)^2 * (0.16 + 0.1275) / 0.0025  ≈ 902 cases per arm
detect 0.80 -> 0.90:   (2.8)^2 * (0.16 + 0.09)   / 0.01    ≈ 196 cases per arm
```

That is the number most teams do not want to hear. A 5-point improvement on a 50-case dataset is indistinguishable from noise. 37. Trajectory Evals and Statistics shows how paired comparisons and repeated runs bring that number down; here the point is to see the spread directly.

```{code-cell} python
rng = np.random.default_rng(4)
TRUE_P = 0.80


def observed_pass_rates(n_cases, trials=2000):
    """Rerun 'the same eval' many times; how much does the headline number move?"""
    return rng.binomial(n_cases, TRUE_P, size=trials) / n_cases


print(f"true pass rate {TRUE_P:.2f}; spread of the measured number across reruns:")
spreads = {}
for n in (20, 50, 200, 1000):
    rates = observed_pass_rates(n)
    lo, hi = np.percentile(rates, [2.5, 97.5])
    spreads[n] = hi - lo
    print(f"  n={n:5d}   95% of reruns land in [{lo:.2f}, {hi:.2f}]   width {hi - lo:.2f}")
assert spreads[20] > 0.3 and spreads[1000] < 0.06


def min_n_per_arm(p1, p2, z_alpha=1.96, z_power=0.84):
    """Approximate two-proportion sample size (normal approximation)."""
    return int(np.ceil((z_alpha + z_power) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (p2 - p1) ** 2))


print("\ncases per arm to detect a change (80% power, 5% two-sided):")
for p2 in (0.82, 0.85, 0.90, 0.95):
    print(f"  0.80 -> {p2:.2f}: {min_n_per_arm(0.80, p2):5d}")
assert min_n_per_arm(0.80, 0.85) > 800 and min_n_per_arm(0.80, 0.90) < 250
```

Two consequences. First, small evals are still useful: they catch large regressions and they surface individual failures to read. They just cannot certify small improvements. Second, when you cannot afford more cases, run each case more times and pair the comparison; that attacks the variance from a different direction.

### Layer 5: is the dataset clean?

Contamination means the model has seen your eval inputs (or their answers) during training. Public benchmarks are the obvious case, but it also happens internally: a support transcript that was used for fine-tuning and is also in the eval set. The signature is a score that is too good and does not transfer.

Duplicates are the quieter problem. Twenty near-identical "what does Acme owe" cases inflate n without adding information, and make the interval from Layer 4 look tighter than it is.

A cheap detector for both: n-gram overlap between each eval input and a reference corpus (a training set, a public benchmark dump, or the rest of the eval set itself).

```{code-cell} python
import re


def ngrams(text, n=8):
    words = re.findall(r"\w+", text.lower())
    return {" ".join(words[i:i + n]) for i in range(max(0, len(words) - n + 1))}


def overlap(text, corpus_ngrams, n=8):
    grams = ngrams(text, n)
    return len(grams & corpus_ngrams) / len(grams) if grams else 0.0


TRAINING_LIKE = [
    "Why is Acme complaining about their invoice and what should we tell them about the open balance",
    "Which plan is Globex on and has their latest invoice been paid according to Stripe",
]
EVAL_INPUTS = [
    "Why is Acme complaining about their invoice and what should we tell them about the open balance",  # verbatim
    "Which plan is Globex on and has their latest invoice been paid according to Stripe records",     # near copy
    "Customer Initech says they were charged twice in March; find both charges and propose a fix",     # fresh
    "Customer Initech says they were charged twice in March; find both charges and propose a fix",     # duplicate
]

corpus = set().union(*(ngrams(t) for t in TRAINING_LIKE))
flags = []
for i, text in enumerate(EVAL_INPUTS):
    contamination = overlap(text, corpus)
    others = set().union(*(ngrams(t) for j, t in enumerate(EVAL_INPUTS) if j != i))
    duplication = overlap(text, others)
    flag = "CONTAMINATED" if contamination > 0.5 else "DUPLICATE" if duplication > 0.9 else "ok"
    flags.append(flag)
    print(f"case {i}: contamination={contamination:.2f} duplication={duplication:.2f} -> {flag}")

assert flags == ["CONTAMINATED", "CONTAMINATED", "DUPLICATE", "DUPLICATE"]
clean = [t for t, f in zip(EVAL_INPUTS, flags) if f == "ok"]
print(f"{len(clean)} of {len(EVAL_INPUTS)} cases usable as-is")
```

The two Initech cases are flagged as duplicates of each other; keep one. The two contaminated cases should be rewritten, not deleted, because the scenario is still worth testing. Change the entities, the phrasing, and the numbers so the model has to do the work rather than recall it.

## How LangGraph does it

Reliability of the eval is not something LangGraph or LangSmith can do for you, but the structure they impose helps at each layer:

| Layer | Where it shows up |
|---|---|
| Human agreement | LangSmith annotation queues let humans label sampled runs; export the labels and compute kappa against your judge's feedback on the same runs. |
| Position bias | Pairwise evaluators in the current LangSmith docs take two experiment outputs; run the swap yourself by calling the evaluator both ways and reconciling. |
| Repeated runs | `client.evaluate(..., num_repetitions=k)` runs each example k times, which is what pass@k and pass^k need. |
| Variance | Experiment comparison shows per-example results; compute intervals from the exported feedback rather than trusting a single aggregate. |
| Contamination | Nothing automatic. Keep eval datasets private and versioned; tag examples with their provenance. |

The key mapping back to the from-scratch code: LangSmith feedback is a table of `(run_id, key, score, comment)`. Every measurement in this lesson is a query over that table joined with human labels. If you can export it, you can compute kappa, ECE, pass^k, and intervals with the functions above.

```{code-cell} python live
import os
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

model = init_chat_model(os.environ.get("COURSE_MODEL", "openai:gpt-5.6"))

A = "Invoice in_1001 for Acme is open with a balance of $1,200."
B = "Acme owes $1,200 on in_1001, which is still open. Let me know if you want it refunded."
QUESTION = "What does Acme owe on in_1001?"


def ask(first, second):
    prompt = (f"Question: {QUESTION}\n\nAnswer 1: {first}\n\nAnswer 2: {second}\n\n"
              "Which answer is better for a support reply? Reply with exactly '1' or '2'.")
    return model.invoke([HumanMessage(prompt)]).content.strip()[:1]


v1, v2 = ask(A, B), ask(B, A)
consistent = (v1 == "1") == (v2 == "2")
print(f"A-first verdict={v1}  B-first verdict={v2}  consistent={consistent}")
if consistent:
    print("winner:", "A" if v1 == "1" else "B")
else:
    print("verdict flips with position: record a tie")
```

Run the live cell a few times. Even strong models flip on close pairs. That is not a bug in the model; it is a reason to swap.

## What goes wrong in production

1. **The judge is trusted because it agreed with the team once.** Symptom: release gate scores drift upward while customer complaints do not fall. Cause: no ongoing human calibration; the judge was checked on launch data and prompts changed since. Fix: label a fresh stratified sample every release cycle, compute kappa, and alert if it drops. Treat the judge as a model with its own eval.

2. **Pairwise wins are position artifacts.** Symptom: the new version wins 65% of pairs; users prefer the old one. Cause: the new version was always in slot A. Fix: swap and require agreement, randomize slot assignment, and report the tie rate. A high tie rate is itself information: the versions are close.

3. **Threshold on an uncalibrated score.** Symptom: "reject if judge score < 0.7" blocks good answers and passes bad ones. Cause: the judge's 0.7 does not mean 70% of anything. Fix: measure ECE; set thresholds from observed precision at each score bucket rather than from the nominal number.

4. **Shipping unattended automation on pass@k.** Symptom: the eval says 98%, the refund agent double-refunds one customer in fifty. Cause: pass@k measured "can it ever succeed"; the product needed "does it always succeed." Fix: report pass^k for unattended actions and put a human gate on the ones that cannot reach it (13. Human in the Loop).

5. **A 3-point win on 40 cases.** Symptom: improvements ship and then vanish. Cause: the change was inside the noise band. Fix: compute the interval before celebrating; use repeated runs and paired tests; when the eval is too small to decide, say so and gather more before shipping.

6. **Contaminated or duplicated datasets.** Symptom: excellent scores that do not match production. Cause: eval inputs overlap with training or fine-tuning data, or the dataset is twenty rephrasings of five scenarios. Fix: overlap detection at dataset build time, provenance tags, and deduplication before computing any interval.

7. **The judge sees a wrong reference.** Symptom: correct answers marked wrong in a cluster. Cause: reference answers were written quickly and some are wrong; the judge anchors on them. Fix: audit references on every disagreement between judge and a second grader; disagreements are the cheapest place to find bad references.

## What changed by 2026

- **Judge evaluation is now part of the eval.** Teams that use LLM judges report agreement with human labels and swap positions by default; current LangSmith docs describe pairwise evaluators and repeated runs (`num_repetitions`) as standard features, which is the infrastructure for pass@k, pass^k, and position checks.
- **Separate evaluator roles are recommended by vendors.** Anthropic's harness-design guidance for long-running work argues that a generator grading its own output is unreliable and recommends a distinct evaluator with gradable rubrics and "define done before coding" contracts. That is Layer 1 applied to the agent's own self-checks.
- **Reasoning text is not evidence.** Anthropic's reasoning-faithfulness research found that models often do not verbalize the factors that drove an answer. A judge that grades the chain of thought is grading a story. Grade the outcome and the trajectory.
- **pass^k entered the vocabulary.** Reliability metrics for agents (all-of-k rather than any-of-k) are now commonly reported alongside single-run accuracy, especially for tool-using agents acting without review.
- **Datasets are treated as versioned private assets.** Public benchmark numbers are read as sanity checks; release decisions rest on private datasets built from traffic, with provenance recorded per example.
- **Tool descriptions are under eval too.** Anthropic's tool-writing guidance recommends iterating on tool descriptions against agent evals. That makes reliability of the eval doubly important: an unreliable eval will steer tool design in random directions.

## Interview questions

**Q1. Your LLM judge agrees with human labels 85% of the time. Is it good?**

Strong: I cannot say until I know the base rate. If 80% of answers are good, a judge that always says "good" agrees 80% of the time and knows nothing. I compute Cohen's kappa, which subtracts chance agreement; 85% raw agreement on that base rate might be a kappa near 0.3, which is not reliable enough for a release gate. I also look at agreement separately on the "bad" class, because catching bad answers is the job. Then I decide what kappa is acceptable for the decision, and I re-measure it every cycle because judges drift when prompts change.

Weak: "85% is pretty high, so yes."

**Q2. What is position bias and how do you handle it?**

Strong: In pairwise judging, models tend to prefer an answer based on where it appears rather than what it says. I ask twice with the positions swapped and only count a win when both orderings agree; disagreements become ties. I also randomize which version goes first across pairs and report the tie rate. It doubles judge cost, which is cheap compared to shipping on an artifact. Related biases, like verbosity and self-preference, need human calibration rather than swapping.

Weak: "Tell the judge in the prompt to ignore order."

**Q3. Explain pass@k versus pass^k and when each is the right metric.**

Strong: Run each case k times. pass@k is the fraction of cases where at least one run succeeded; it measures capability, and it is right when a human will pick the good attempt or when a retry is cheap. pass^k is the fraction where all k runs succeeded; it measures reliability, and it is right when the agent acts unattended, like issuing a refund. They diverge fast: a case that passes 90% of the time has pass^5 around 59%. For agent products I report both and gate on pass^k for autonomous actions.

Weak: "pass@k is accuracy with k attempts."

**Q4. How many eval cases do you need?**

Strong: It depends on the effect size you need to detect. A pass rate on n cases has an interval about ±1.96·sqrt(p(1-p)/n); at p=0.8 and n=50 that is ±11 points, so a 5-point improvement is noise. To detect 0.80 to 0.85 with 80% power you need on the order of 900 cases per arm; 0.80 to 0.90 needs about 200. When I cannot get that many cases, I run each case several times and use a paired comparison, which reduces variance, and I am explicit that a small eval can catch large regressions but cannot certify small wins.

Weak: "Around 50 is usually enough."

**Q5. What is calibration and why does it matter for a judge?**

Strong: A calibrated judge is right about as often as it says it is; when it outputs 0.9, roughly 90% of those answers are actually good. I measure it with expected calibration error: bucket by stated confidence, compare to observed accuracy, weight by bucket size. LLM judges are often overconfident. It matters because thresholds and cost-of-error decisions assume the number means something; if 0.9 really means 65%, a threshold at 0.85 lets through far more bad answers than planned. If a judge is uncalibrated but ranks well, I still use it for ranking and set thresholds from observed precision.

Weak: "Calibration is fine-tuning the judge."

**Q6. How would you detect that an eval set is contaminated?**

Strong: First by symptom: scores that are much better than production, or a model that answers questions it has no tool access to answer. Then mechanically: n-gram overlap between eval inputs and any corpus the model might have trained on, including public benchmarks and our own fine-tuning data. I also deduplicate the eval set against itself, because near-duplicates inflate n and fake precision. Contaminated cases get rewritten with new entities and phrasing rather than deleted, since the scenario still matters. And I tag every example with its provenance so this is checkable later.

Weak: "Use a private test set."

**Q7. Two judges disagree on 15% of items. What do you do with those items?**

Strong: Treat them as the most valuable items in the set. Disagreements concentrate the bad references, the ambiguous rubric lines, and the genuinely hard cases. I send them to humans first, fix the references or rubric where that is the cause, and keep the truly ambiguous ones labeled as such so they do not silently count as failures or successes. Agreement rate between judges also tells me how much of the score is noise, which feeds into the interval I report.

Weak: "Average the two judges."

## Build it yourself

Build a `JudgeAudit` that takes a table of `(item_id, judge_score, judge_confidence, human_label)` rows and produces a reliability report.

1. Compute kappa between the judge's binarized score (threshold 0.5) and the human label, overall and on the "bad" class only.
2. Compute ECE on `judge_confidence` and recommend a threshold: the lowest confidence bucket whose observed precision is at least 0.9.
3. Accept an optional second judge's scores and list the items where the two disagree, sorted by confidence descending.
4. Given per-item repeated runs, report avg@k, pass@k, and pass^k, and the minimum n to detect a 5-point improvement from the observed pass rate.

Acceptance checks:

- Feeding the "lazy judge" data from this lesson yields kappa below 0.15 while raw agreement is above 0.7.
- On perfectly calibrated synthetic data (confidence equals the true probability of being correct), ECE is below 0.03.
- With two identical judges the disagreement list is empty; flipping 10% of one judge's labels produces a list of roughly 10% of items.
- For an observed pass rate of 0.8, the minimum-n recommendation is between 850 and 950 cases per arm.

## Key takeaways

- A grader is a measurement instrument; measure its bias and noise before trusting its number.
- Use Cohen's kappa, not raw agreement, to compare a judge with humans; imbalanced labels make raw agreement meaningless.
- Swap positions in pairwise judging and count only consistent verdicts; disagreements are ties.
- Check calibration before putting a threshold on a judge's score; overconfident judges are common.
- Report pass^k for unattended actions and pass@k for assisted ones; they diverge quickly.
- A pass rate has an interval; small evals catch large regressions but cannot certify small wins, and detecting a 5-point change needs hundreds of cases.
- Detect contamination and duplicates before computing anything, and rewrite rather than delete contaminated scenarios.
