# 30. Fine-tuning, Routing, and Prompt Optimization: change the model or change the system

*Track C (retrieval + production AI) with Track D (interviews). Course addition beyond the book; it extends Chapter 16 (Resource-Aware Optimization) and Chapter 9 (Learning and Adaptation). Prerequisites: 16. Resource-Aware Optimization, 19. Evaluation and Monitoring, 24. Structured Output. Companions: 22. Evaluation Reliability Deep Dive, 44. Decoding, reasoning models, and structured output.*

## The problem

The support agent's first step is triage: classify each incoming ticket so it reaches the right tools and the right team. It runs on a frontier model. It works. Then the numbers arrive:

```text
tickets per day             50,000
accuracy on the eval set    91%
cost                        the largest line item in the agent's budget
p95 latency                 2.8 s, and triage blocks everything after it
```

Four people propose four fixes:

```text
PM:          "Fine-tune a model on our tickets."
Eng lead:    "Rewrite the prompt, it's from last year."
Infra:       "Send the easy tickets to a small model."
Data team:   "It keeps misclassifying the new SSO product. Give it the docs."
```

Each proposal changes a different thing:

```text
                 ┌────────────────────── what you change ──────────────────────┐
  prompt         │ instructions and examples        minutes, free, reversible  │
  retrieval      │ knowledge in the context         days, cheap, stays fresh   │
  routing        │ WHICH model answers              days, cuts cost/latency    │
  fine-tuning    │ the model's weights              weeks, data + evals, sticky│
                 └──────────────────────────────────────────────────────────────┘
```

The naive move is to pick the most impressive one. Fine-tuning sounds like the serious engineering answer. It is usually the last one you should try, and "it keeps misclassifying the new SSO product" is a knowledge problem that fine-tuning handles badly: the product will change again next month, and the weights will not.

The engineering move is to diagnose first. Each lever fixes a different kind of failure:

| Symptom on the eval set | Kind of problem | First lever |
|---|---|---|
| Wrong about facts that change (products, prices, policies) | knowledge | retrieval (14. Knowledge Retrieval) |
| Ignores a rule, wrong format, inconsistent labels | instruction | prompt, schema (24. Structured Output) |
| Right answers, too slow or too expensive | cost / latency | routing, caching, then distillation |
| Right with long prompts and many examples; you need it short and cheap | behavior at scale | fine-tuning (SFT, distillation) |
| Wrong even with a perfect prompt and perfect context | capability | bigger model, reasoning effort, or RL fine-tuning |

Everything in this lesson depends on one prerequisite: an eval set. Without it, none of these levers can be compared, and every change is a guess (19. Evaluation and Monitoring).

## Build it without a framework

Three mechanisms, each in plain Python: a prompt optimization loop, a routing cascade with its cost/quality curve, and the training objectives that interviewers ask about. The scripted model helper sits at the top of the first cell because later cells use it; the first cell uses a deterministic stand-in so the *optimization loop* is the object of study.

### Prompt optimization is search against an eval set

A prompt optimizer is a loop: propose an edit, score it on labelled data, keep it if it helps. The proposer is usually another model reading the failures. The scorer is your eval. The part people skip is the held-out split, and that is where it goes wrong.

Our stand-in "model" follows whichever instructions appear in its prompt, in order. That makes it easy to see the two properties real models share: instructions interact, and their order matters.

```{code-cell} python
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage


class ScriptedChatModel(GenericFakeChatModel):
    """Replays scripted turns so the control flow, not the model, is under study."""

    def bind_tools(self, tools, **kwargs):
        return self  # a real model would attach tool schemas here


def scripted(*turns):
    """Each turn is either final text or a list of (tool_name, args) tool calls."""
    messages = []
    for turn in turns:
        if isinstance(turn, str):
            messages.append(AIMessage(content=turn))
        else:
            messages.append(AIMessage(content="", tool_calls=[
                {"name": name, "args": args, "id": f"call_{i}", "type": "tool_call"}
                for i, (name, args) in enumerate(turn)]))
    return ScriptedChatModel(messages=iter(messages))


# Labelled tickets, split BEFORE any optimization so dev stays unseen.
TRAIN = [
    ("I was charged twice for invoice in_1042", "billing"),
    ("Please refund my last payment", "billing"),
    ("Cannot log in, password reset email never arrives", "account"),
    ("2FA code rejected on every attempt", "account"),
    ("API returns error 500 on /v1/charges", "technical"),
    ("Webhook timeout when paying an invoice", "technical"),
    ("What are your support hours?", "general"),
    ("Invoice PDF shows the wrong VAT number", "billing"),
    ("Card declined when upgrading my plan", "billing"),
]
DEV = [
    ("Double charge on my card this morning", "billing"),
    ("Locked out after changing my password", "account"),
    ("Timeout from the API when I pay by card", "technical"),
    ("Do you have an office in Berlin?", "general"),
    ("Refund still not showing after 10 days", "billing"),
    ("SSO login loops back to the sign-in page", "account"),
]

# Candidate instructions, as a proposer model might suggest after reading failures.
CANDIDATES = {
    "money":     ("Anything mentioning money, payment, or paying is billing.",
                  ("pay", "charge", "refund", "invoice"), "billing"),
    "access":    ("Login, password, 2FA, SSO, or lockout problems are account issues.",
                  ("log in", "login", "password", "2fa", "sso", "locked"), "account"),
    "technical": ("API errors, timeouts, and webhooks are technical, even during payment.",
                  ("api", "error", "timeout", "webhook"), "technical"),
    "billing":   ("Charges, double charges, refunds, and invoices are billing.",
                  ("charge", "refund", "invoice"), "billing"),
    "card_hack": ("Any ticket that mentions a card is billing.", ("card",), "billing"),
}


def simulated_model(prompt_rules, ticket):
    """Deterministic stand-in for an LLM: follows the instructions present, in order."""
    text = ticket.lower()
    for key in prompt_rules:
        _, keywords, label = CANDIDATES[key]
        if any(k in text for k in keywords):
            return label
    return "general"


def accuracy(prompt_rules, data):
    return sum(simulated_model(prompt_rules, t) == y for t, y in data) / len(data)


def optimize(max_rules=4):
    """Greedy prompt search: try each edit, keep it only if train improves and dev does not drop."""
    prompt, banned = [], set()
    history = [("baseline", accuracy([], TRAIN), accuracy([], DEV))]
    while len(prompt) < max_rules:
        best = None
        for key in CANDIDATES:
            if key in prompt or key in banned:
                continue
            for pos in range(len(prompt) + 1):          # instruction ORDER is part of the search
                trial = prompt[:pos] + [key] + prompt[pos:]
                score = accuracy(trial, TRAIN)
                if best is None or score > best[0]:
                    best = (score, trial, key)
        kept_train, kept_dev = accuracy(prompt, TRAIN), accuracy(prompt, DEV)
        if best is None or best[0] <= kept_train:
            break                                       # no edit helps on train
        dev = accuracy(best[1], DEV)
        if dev < kept_dev:                              # fits train, hurts unseen data
            banned.add(best[2])
            history.append((f"reject {best[2]} (overfit)", best[0], dev))
            continue
        prompt = best[1]
        history.append((f"+{best[2]}", best[0], dev))
    return prompt, history


prompt, history = optimize()
for step, tr, dv in history:
    print(f"{step:32s} train={tr:.2f}  dev={dv:.2f}")
print("final instruction order:", prompt)
print("\n".join(f"- {CANDIDATES[k][0]}" for k in prompt))
assert "card_hack" not in prompt and any("reject card_hack" in h[0] for h in history)
assert accuracy(prompt, DEV) > accuracy([], DEV) + 0.5
assert prompt.index("technical") < prompt.index("money" if "money" in prompt else "billing")
```

Read the history line by line.

The first three edits are general rules and they improve both splits. The optimizer also discovered an *ordering* fact: the "technical" rule must come before the "money" rule, otherwise "Webhook timeout when paying an invoice" gets labelled billing. Humans writing prompts miss this kind of interaction all the time.

The fourth candidate, `card_hack`, is the lesson. It lifts train accuracy to 100% by memorizing one training ticket, and it breaks a dev ticket ("Timeout from the API when I pay by card"). An optimizer that looks only at train would have shipped it. That is overfitting, and it happens to prompts exactly as it happens to weights.

One more subtlety: because we used dev to *accept or reject* edits, dev is no longer a clean estimate. In practice you keep three splits: train for proposing, dev for selection, and a test set you look at once before shipping. DSPy (Khattab et al., 2023) built a framework around this idea, treating prompts and few-shot demonstrations as parameters optimized against a metric. Anthropic's tool-writing guidance describes the same loop applied to tool descriptions: run the eval, have a model read the failing transcripts, revise, rerun.

### Routing: the cost/quality curve

A router sends each request to the cheapest option that will get it right. The standard shape is a **cascade** (FrugalGPT, Chen et al., 2023): try the small model, and escalate to the frontier model when a confidence signal is low. RouteLLM (Ong et al., 2024) trains the router itself on preference data instead of using a threshold.

The numbers below are simulated, not benchmarks. The shape is what matters.

```{code-cell} python
import numpy as np

rng = np.random.default_rng(7)
N = 10_000
difficulty = rng.beta(2, 5, N)                    # most support tickets are easy

# Per-ticket correctness of each option (1 = right). Illustrative, not benchmarks.
small_ok = rng.random(N) < np.clip(1.0 - 1.2 * difficulty ** 2, 0, 1)
frontier_ok = rng.random(N) < np.clip(0.99 - 0.15 * difficulty, 0, 1)
tuned_ok = rng.random(N) < np.clip(1.0 - 0.6 * difficulty ** 2, 0, 1)   # small model after distillation

# The router's signal is noisy: confidence correlates with difficulty but is not it.
confidence = np.clip(1 - difficulty + rng.normal(0, 0.08, N), 0, 1)

COST = {"small": 0.02, "frontier": 1.00}          # relative cost per ticket


def cascade(threshold, cheap_ok):
    """Cheap model answers; below the confidence threshold, escalate to the frontier model."""
    escalate = confidence < threshold
    ok = np.where(escalate, frontier_ok, cheap_ok)
    # An escalated ticket pays for BOTH calls: the cheap attempt already happened.
    cost = COST["small"] + escalate * COST["frontier"]
    return ok.mean(), cost.mean(), escalate.mean()


print(f"{'policy':24s} {'accuracy':>8s} {'cost':>6s} {'escalated':>9s}")
print(f"{'all small':24s} {small_ok.mean():8.3f} {COST['small']:6.2f} {0:9.0%}")
print(f"{'all frontier':24s} {frontier_ok.mean():8.3f} {COST['frontier']:6.2f} {1:9.0%}")
curve = []
for t in np.round(np.linspace(0, 1, 21), 2):
    acc, cost, esc = cascade(t, small_ok)
    curve.append((t, acc, cost))
    if t in (0.5, 0.7, 0.8, 0.9):
        print(f"{f'cascade t={t:.1f}':24s} {acc:8.3f} {cost:6.2f} {esc:9.0%}")

# Operating point: the cheapest threshold within 1 point of all-frontier accuracy.
target = frontier_ok.mean() - 0.01
t_star, acc_star, cost_star = min((c for c in curve if c[1] >= target), key=lambda c: c[2])
acc_d, cost_d, esc_d = cascade(t_star, tuned_ok)
print(f"\noperating point t={t_star:.1f}: accuracy {acc_star:.3f} at {cost_star:.0%} of frontier cost")
print(f"same threshold, distilled small model: accuracy {acc_d:.3f}, cost {cost_d:.2f}")

costs = [c[2] for c in curve]
assert all(a <= b for a, b in zip(costs, costs[1:]))       # higher threshold never cheaper
assert cost_star < 0.6 * COST["frontier"] and acc_star >= target
assert acc_d > acc_star                                      # better cheap model, same cost
```

Three things to notice.

**The curve, not the point.** Sweeping the threshold traces a cost/quality curve from "all small" to "all frontier." Picking a point on it is a product decision: one accuracy point for half the cost may be right for triage and wrong for refunds.

**Escalation double-pays.** An escalated ticket costs the small call *and* the frontier call. If most traffic escalates, the cascade is more expensive than sending everything to the frontier model. The `cost` line in `cascade()` is the bug most back-of-the-envelope router estimates contain.

**The router is only as good as its signal.** Here confidence is difficulty plus noise. In production, self-reported confidence ("rate your confidence 0 to 1") is poorly calibrated. Better signals are cheaper to trust: a verifier that checks the output (does the SQL run, does the invoice id exist), agreement between two cheap samples, token log-probabilities where the provider exposes them, or a small trained classifier. Measure calibration on your eval set before you rely on any of them.

The last line shows where fine-tuning enters: a small model **distilled** on the frontier model's outputs for this task moves the whole curve up. Same threshold, same cost, higher accuracy. That is the most common reason teams fine-tune in 2026: not to add knowledge, but to make a cheap model good at one narrow job.

### The training objectives, in numpy

Interviewers ask "explain pre-training versus supervised fine-tuning versus preference optimization" (reported at Meta and Scale AI) and "GRPO versus PPO." You do not need to train a model to answer well, but you should know what each objective optimizes.

```text
stage              data                             objective                              teaches
-------------------------------------------------------------------------------------------------------------
pre-training       trillions of web/code tokens     next-token cross-entropy               language, knowledge, skills
SFT                prompt -> ideal response pairs   cross-entropy on the response tokens   format, behavior, instruction following
RLHF (PPO)         prompts + reward model trained   maximize reward - beta * KL(pi||ref)   preferences humans can judge but not write
                   on human preference pairs
DPO                (prompt, chosen, rejected)       -log sigmoid(beta * margin)            same goal, no reward model, no RL loop
RL w/ verifiable   prompts + a checker              group-relative advantage (GRPO)        reasoning, tool use, anything checkable
rewards            (tests, exact answers)
```

The DPO margin, from "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (Rafailov et al., 2023):

```text
margin = [log pi(y_w|x) - log pi_ref(y_w|x)] - [log pi(y_l|x) - log pi_ref(y_l|x)]
loss   = -log sigmoid(beta * margin)

worked example (beta = 0.1):
  policy == reference          margin = 0                        loss = log 2 = 0.693
  chosen -12 -> -10,           margin = (2) - (-3) = 5            loss = -log sigmoid(0.5) = 0.474
  rejected -12 -> -15
```

GRPO, introduced in DeepSeekMath (Shao et al., 2024) and used for DeepSeek-R1 (2025), removes PPO's learned value function (the critic). It samples a group of G responses per prompt, scores each with a reward (often a rule-based checker), and uses the group itself as the baseline:

```text
A_i = (r_i - mean(r_1..r_G)) / std(r_1..r_G)

worked example, G = 8, rewards [1,0,0,1,0,0,0,1]:
  mean = 0.375, std = 0.484
  correct samples:   (1 - 0.375) / 0.484 = +1.29
  incorrect samples: (0 - 0.375) / 0.484 = -0.77
```

```{code-cell} python
def log_softmax(z):
    z = np.asarray(z, dtype=float)
    return z - (z.max() + np.log(np.exp(z - z.max()).sum()))


# --- SFT: cross-entropy on the demonstrated token (here one step, 4-token vocab) ---
vocab = ["billing", "account", "technical", "general"]
logits = np.array([1.0, 0.5, 2.0, -1.0])        # model currently prefers "technical"
sft_loss = -log_softmax(logits)[vocab.index("billing")]


# --- DPO: widen the gap between chosen and rejected, measured against a frozen reference ---
def dpo_loss(pol_chosen, pol_rejected, ref_chosen, ref_rejected, beta=0.1):
    """Inputs are sequence log-probs log pi(y|x). Loss = -log sigmoid(beta * margin)."""
    margin = (pol_chosen - ref_chosen) - (pol_rejected - ref_rejected)
    return np.log1p(np.exp(-beta * margin)), margin


loss_start, m0 = dpo_loss(-12.0, -12.0, -12.0, -12.0)      # policy == reference
loss_after, m1 = dpo_loss(-10.0, -15.0, -12.0, -12.0)      # chosen up, rejected down


# --- GRPO: sample a group per prompt, score with a verifier, normalize within the group ---
rewards = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])   # e.g. "did the SQL return the right rows?"
advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)


# --- Distillation: match the teacher's softened distribution, not just its top label ---
def kl(p, q):
    return float(np.sum(p * (np.log(p) - np.log(q))))


T = 2.0
teacher = np.exp(log_softmax(np.array([4.0, 1.0, 2.5, -2.0]) / T))
student = np.exp(log_softmax(logits / T))

print(f"SFT loss on 'billing':          {sft_loss:.3f}")
print(f"DPO loss at margin {m0:+.0f}:          {loss_start:.3f}  (= log 2)")
print(f"DPO loss at margin {m1:+.0f}:          {loss_after:.3f}")
print(f"GRPO advantages:                {np.round(advantages, 2)}")
print(f"distillation KL(teacher||student) at T={T}: {kl(teacher, student):.3f}")
assert abs(loss_start - np.log(2)) < 1e-9 and loss_after < loss_start
assert abs(advantages.mean()) < 1e-9 and advantages[0] > 0 > advantages[1]
assert sft_loss > -log_softmax(logits)[vocab.index("technical")]
```

Four things to say in an interview about that cell. SFT pushes probability onto the demonstrated answer and nothing else. DPO never trains a reward model; the reference model's log-probabilities play that role, and `beta` controls how far the policy may drift from it. GRPO's advantages sum to zero within a group, so "better than your siblings" is the whole signal, and no critic network is needed. Distillation matches the teacher's full softened distribution (Hinton et al., 2015, used a temperature for exactly this), which carries more information per example than the top label alone; in LLM practice the common form is simpler, supervised training on the teacher's generated outputs.

## How LangGraph does it

Routing is a graph problem: a conditional edge is a router. Fine-tuning and prompt optimization happen offline, outside the agent graph; what the graph needs is the ability to choose a model per step and to survive a provider outage.

### A cascade as a graph

The cascade from `cascade()` becomes two nodes and two conditional edges. A rule-based pre-router sends known-hard tickets (chargebacks, legal) straight to the frontier model, because paying for a small attempt you will always discard is waste. The small model's structured output includes a confidence field, and a conditional edge escalates on it. A reducer sums cost across nodes, so every run carries its own bill.

```{code-cell} python
import operator
from typing import Annotated, Literal, TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END


class RoutedTriage(BaseModel):
    category: Literal["billing", "account", "technical", "general"]
    confidence: float = Field(ge=0, le=1, description="Probability the category is right.")


class RouteState(TypedDict):
    ticket: str
    triage: RoutedTriage | None
    path: Annotated[list[str], operator.add]     # audit trail of which models ran
    cost: Annotated[float, operator.add]         # summed across nodes by the reducer


ALWAYS_FRONTIER = ("chargeback", "legal", "lawyer")   # known-hard: skip the cheap attempt
THRESHOLD = 0.7


def build_router(small_llm, frontier_llm):
    small = small_llm.with_structured_output(RoutedTriage)
    frontier = frontier_llm.with_structured_output(RoutedTriage)

    def pre_route(state: RouteState):
        hard = any(w in state["ticket"].lower() for w in ALWAYS_FRONTIER)
        return "frontier" if hard else "small"

    def small_node(state: RouteState):
        return {"triage": small.invoke(state["ticket"]), "path": ["small"], "cost": COST["small"]}

    def frontier_node(state: RouteState):
        return {"triage": frontier.invoke(state["ticket"]), "path": ["frontier"], "cost": COST["frontier"]}

    def escalate(state: RouteState):
        return "frontier" if state["triage"].confidence < THRESHOLD else END

    g = StateGraph(RouteState)
    g.add_node("small", small_node)
    g.add_node("frontier", frontier_node)
    g.add_conditional_edges(START, pre_route, ["small", "frontier"])
    g.add_conditional_edges("small", escalate, ["frontier", END])
    g.add_edge("frontier", END)
    return g.compile()


small_llm = scripted(
    [("RoutedTriage", {"category": "billing", "confidence": 0.95})],     # ticket 1: confident
    [("RoutedTriage", {"category": "general", "confidence": 0.41})],     # ticket 2: unsure
)
frontier_llm = scripted(
    [("RoutedTriage", {"category": "technical", "confidence": 0.9})],    # ticket 2 escalated
    [("RoutedTriage", {"category": "billing", "confidence": 0.88})],     # ticket 3 pre-routed
)
router = build_router(small_llm, frontier_llm)

runs = []
for ticket in ["Refund my duplicate charge on in_1042",
               "Payments page hangs after the new SDK release",
               "We are filing a chargeback with our bank"]:
    out = router.invoke({"ticket": ticket, "triage": None, "path": [], "cost": 0.0})
    runs.append(out)
    print(f"{ticket[:44]:46s} {'->'.join(out['path']):16s} {out['triage'].category:10s} cost={out['cost']:.2f}")

assert [r["path"] for r in runs] == [["small"], ["small", "frontier"], ["frontier"]]
assert runs[1]["triage"].category == "technical" and abs(runs[1]["cost"] - 1.02) < 1e-9
```

Map it back:

| From scratch | LangGraph |
|---|---|
| `escalate = confidence < threshold` | the `escalate` conditional edge after the `small` node |
| cost of both calls on escalation | `cost: Annotated[float, operator.add]`, summed by the reducer |
| a known-hard rule before the cascade | `add_conditional_edges(START, pre_route, ...)` |
| which model answered | `path`, an append-only audit trail for tracing and billing |

The graph also makes the cascade observable: in LangSmith each run shows which nodes executed, so "what fraction of traffic escalates, and why" is a query rather than a guess.

### Per-call model choice inside one agent

Inside a `create_agent` loop, routing is middleware. A `wrap_model_call` hook sees each model request and can swap the model before the call. `ModelFallbackMiddleware` catches a failed call and retries it on the next model in its list. Order the middleware so routing chooses first and fallback wraps the chosen call.

```{code-cell} python
from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware, wrap_model_call
from langchain_core.messages import HumanMessage


class DownModel(ScriptedChatModel):
    """A provider outage: every call fails."""

    def _generate(self, *args, **kwargs):
        raise ConnectionError("503 from provider")


small_chat = scripted("Our support hours are 9-18 CET, Monday to Friday.")
frontier_chat = DownModel(messages=iter([]))
backup_chat = scripted("Section 7.2 caps liability at 12 months of fees; flag for legal review.")
routed_to = []


@wrap_model_call
def route_by_task(request, handler):
    """Per-call model choice: the same agent, a different model for different work."""
    text = request.messages[-1].content.lower()
    heavy = any(w in text for w in ("contract", "indemnity", "liability"))
    routed_to.append("frontier" if heavy else "small")
    return handler(request.override(model=frontier_chat if heavy else small_chat))


agent = create_agent(
    small_chat,
    tools=[],
    # Order matters: routing picks a model, then fallback catches that model's outage.
    middleware=[route_by_task, ModelFallbackMiddleware(backup_chat)],
)
easy = agent.invoke({"messages": [HumanMessage("What are your support hours?")]})
hard = agent.invoke({"messages": [HumanMessage("Summarize the liability clause in Acme's contract.")]})
print("routed:", routed_to)
print("easy ->", easy["messages"][-1].content)
print("hard ->", hard["messages"][-1].content)
assert routed_to == ["small", "frontier"]
assert "12 months" in hard["messages"][-1].content      # served by the fallback after the 503
```

Two different concerns, deliberately separated. The router decides by *task* (a contract question needs the stronger model). The fallback reacts to *availability* (that model's provider returned 503). Mixing them, for example by treating "the small model gave a bad answer" as an outage, hides quality regressions inside your failover metrics.

### The live version

The live cell runs the optimized instructions from the first cell against a real model on the dev set. Expect the improvement to be smaller than in the simulation; real models already follow many of these rules. That gap is itself the lesson: measure before you optimize.

```{code-cell} python live
import os
from langchain.chat_models import init_chat_model

llm = init_chat_model(os.environ.get("COURSE_MODEL", "openai:gpt-5.6"))
classifier = llm.with_structured_output(RoutedTriage)
base = "Classify the support ticket as billing, account, technical, or general."
optimized = base + "\n" + "\n".join(f"- {CANDIDATES[k][0]}" for k in prompt)


def live_accuracy(instructions, data):
    hits = [classifier.invoke(f"{instructions}\n\nTicket: {t}").category == y for t, y in data]
    return sum(hits) / len(hits)


print(f"dev accuracy, base prompt:      {live_accuracy(base, DEV):.2f}")
print(f"dev accuracy, optimized prompt: {live_accuracy(optimized, DEV):.2f}")
```

## What goes wrong in production

1. **Fine-tuning for knowledge.** Symptom: the tuned model states last quarter's prices confidently. Cause: facts were baked into weights that do not update. Fix: retrieval for anything that changes; fine-tune for format, style, and narrow behavior.

2. **Overfit prompts.** Symptom: the prompt optimizer reports 100% and production accuracy drops. Cause: edits memorized the eval set, and the same set was used for selection and reporting. Fix: train, dev, and test splits; look at test once; prefer general rules over example-specific ones; refresh the eval set from production failures.

3. **Regression after fine-tuning.** Symptom: the tuned triage model is better at triage and worse at following the JSON schema or refusing unsafe requests. Cause: narrow training shifted general behavior. Fix: run the full regression suite, including safety and format checks, on every tuned checkpoint; parameter-efficient methods such as LoRA (Hu et al., 2021) reduce but do not remove this.

4. **Cascades that cost more than the frontier model.** Symptom: the bill goes up after introducing routing. Cause: most traffic escalates, and each escalation pays twice. Fix: plot the curve on real traffic, pre-route known-hard classes, and alert on escalation rate.

5. **Miscalibrated confidence.** Symptom: the small model is "95% confident" on the tickets it gets wrong. Cause: self-reported confidence is not a probability. Fix: use a verifier or agreement signal, and measure calibration (reliability curve) on the eval set before trusting a threshold.

6. **Silent model drift.** Symptom: accuracy slides with no code change. Cause: a provider updated a model behind an alias, or traffic shifted to a new product. Fix: pin model versions, run evals on a schedule against production samples, and version prompts with the same review and rollback discipline as code.

7. **Preference data that teaches the wrong thing.** Symptom: after DPO on thumbs-up/down data, answers get longer and more agreeable but not more correct. Cause: user preferences reward confidence and length. Fix: curate pairs with graders that check correctness; control for length; hold out a correctness eval.

## What changed by 2026

- **Prompt optimization became an eval-driven loop, not an art.** Teams run optimizers (DSPy-style, or a model reading failure transcripts) against versioned eval sets, and treat prompts as versioned artifacts with regression gates. Anthropic's tool guidance applies the same loop to tool descriptions.
- **Reasoning models changed what prompts need.** Current OpenAI reasoning guidance says to state the task, constraints, output format, and what counts as done, and to treat reasoning effort as a tuning knob; it advises against chain-of-thought prefixes. Reasoning effort is now a routing dimension too: the same model at low effort for triage, high effort for a hard reconciliation.
- **RL fine-tuning with verifiable rewards went mainstream.** After DeepSeek-R1 showed GRPO with rule-based rewards producing strong reasoning, RL against checkers (tests, exact answers, graders) became a standard post-training stage, and some providers now describe reinforcement fine-tuning with custom graders as a product. Check your provider's current docs for which models and methods are offered.
- **Distillation is the default reason to fine-tune.** The common pattern: prove the task on a frontier model, log its outputs, filter them with graders, and train a small model on the result for a narrow, high-volume step.
- **Routing moved into the stack.** In LangChain 1.x, per-call model choice and fallback are middleware (`wrap_model_call`, `ModelFallbackMiddleware`), and LLM gateways handle provider failover, budgets, and caching below the agent. The router, the fallback, and the gateway are three separate layers.

## Interview questions

**Q1. Explain the difference between pre-training, supervised fine-tuning, and preference optimization. (reported: Meta, Scale AI)**

Strong: Pre-training is next-token prediction on a huge unlabelled corpus; it is where the model gets language, knowledge, and most of its skills. Supervised fine-tuning trains on prompt and ideal-response pairs with the same cross-entropy loss, but only on the response tokens; it teaches format and behavior, like following instructions. Preference optimization uses comparisons instead of ideal answers: classic RLHF trains a reward model on human preference pairs and optimizes the policy against it with PPO and a KL penalty to a reference model; DPO skips the reward model and optimizes the pairs directly with a logistic loss on the log-probability margin. Increasingly there is a fourth stage, RL with verifiable rewards, for reasoning and tool use.

Weak: "Pre-training is training and fine-tuning is training more on your data."

**Q2. When do you fine-tune versus improve the prompt versus add retrieval?**

Strong: I diagnose the failure on an eval set first. If it is wrong about facts that change, that is retrieval, because weights do not update when the price list does. If it ignores rules or formats inconsistently, that is prompt and schema work, which is cheap and reversible. If it is right but too slow or expensive, I route or cache first, and fine-tune a small model when a high-volume narrow task needs a long prompt to work. If it fails even with perfect context, that is capability: a stronger model, more reasoning effort, or RL fine-tuning against a checker. Fine-tuning comes last because it needs data, evals, and a retraining pipeline you now own.

Weak: "Fine-tune when prompting doesn't work."

**Q3. How does DPO differ from RLHF with PPO?**

Strong: RLHF trains a separate reward model on preference pairs, then runs an RL loop that samples from the policy, scores samples with the reward model, and updates with PPO while a KL term keeps it close to a reference model. DPO shows the same objective can be optimized directly on the pairs: the loss is the negative log-sigmoid of beta times the margin between how much the policy raised the chosen answer's log-probability and how much it raised the rejected one's, both relative to the reference. No reward model, no sampling during training, and it is much simpler to run. The trade-off is that it only learns from the fixed pairs you have, whereas online RL keeps exploring.

Weak: "DPO is a newer, better version of RLHF."

**Q4. What is GRPO and why did it matter for reasoning models?**

Strong: GRPO is a PPO variant from the DeepSeekMath work that drops the learned value function. For each prompt it samples a group of responses, scores them, often with a rule-based checker like "is the final answer correct," and uses each response's reward minus the group mean, divided by the group standard deviation, as its advantage. That removes a whole critic network and fits verifiable tasks well. DeepSeek-R1 used it with rule-based rewards and showed reasoning behaviors like self-verification emerging from RL. For an agent engineer the practical point is that anything you can check automatically, you can in principle train against.

Weak: "It's the algorithm DeepSeek used."

**Q5. Design model routing for a high-volume support agent. (related: LLM gateway design, reported at Perplexity and Palantir)**

Strong: I start from the eval set and plot the cost/quality curve for all-small, all-frontier, and a cascade at several thresholds, remembering that escalated requests pay for both calls. I pre-route classes that are known to need the strong model, so we do not waste the cheap attempt. The escalation signal is a verifier or agreement check rather than self-reported confidence, and I measure its calibration. Operationally, routing by task sits in the agent, failover by availability sits in a fallback layer or gateway, and I track escalation rate, cost per ticket, and accuracy per route as first-class metrics.

Weak: "Send simple questions to the small model."

**Q6. What is distillation, and when would you use it in an agent product?**

Strong: Distillation trains a smaller student model to reproduce a larger teacher's behavior. Classically the student matches the teacher's softened probability distribution; with LLMs the common form is supervised fine-tuning on outputs the teacher generated, filtered by graders. I use it for a narrow, high-volume step that works on a frontier model but costs too much, like ticket triage or extraction: log the frontier outputs, filter for correctness, train the small model, then put it at the cheap end of the cascade. It moves the whole cost/quality curve up rather than picking a different point on it.

Weak: "Making a model smaller."

**Q7. How do you manage prompt changes and optimization without regressing production?**

Strong: Prompts are versioned artifacts with an owner, a changelog, and a regression gate. Every change, whether hand-written or produced by an optimizer, runs against a fixed eval suite with train, dev, and test splits, and ships only if it improves the target metric without dropping format, safety, or cost metrics. Optimizers get held-out data so they cannot memorize the eval. In production I pin model versions, sample live traffic into the eval set, and keep one-click rollback to the previous prompt version.

Weak: "We test it on a few examples before deploying."

## Build it yourself

Extend the optimizer and the router:

1. Add a `TEST` split of four tickets and report test accuracy only once, after `optimize()` finishes.
2. Add a second kind of edit: choosing up to two few-shot examples from `TRAIN` to include in the prompt. Model them in `simulated_model` as exact-match lookups before the rules.
3. Replace the router's confidence with a verifier: a function that returns `False` when the small model's category disagrees with a keyword check, and escalate on `False`.
4. Add a third route to the LangGraph cascade: tickets over 400 characters go to a `"long_context"` node with its own cost.

Acceptance checks:

- The few-shot edit that copies a training ticket verbatim improves train accuracy but is rejected when it does not improve dev.
- Test accuracy is computed exactly once and is printed next to dev accuracy.
- With the verifier signal, the escalation rate on the simulated data is lower than with confidence at the same accuracy, or the cell prints why not.
- A 500-character ticket's `path` is `["long_context"]`, and its `cost` equals the long-context cost exactly.

## Key takeaways

- Prompts, retrieval, routing, and fine-tuning change different things; diagnose the failure on an eval set before choosing a lever.
- Prompt optimization is search against labelled data, and it overfits exactly like training does, so keep train, dev, and test splits.
- A cascade's value is a curve, not a point; escalations pay twice and the router is only as good as its confidence signal.
- Pre-training teaches knowledge, SFT teaches behavior, preference optimization (RLHF, DPO) teaches what people prefer, and RL with verifiable rewards (GRPO) teaches what a checker can confirm.
- Fine-tune for narrow, high-volume behavior and distillation, never for facts that change.
- In LangGraph a router is a conditional edge; in `create_agent` it is `wrap_model_call` middleware, with fallback as a separate layer for outages.
- Version prompts and pin models like code, with regression gates and rollback.
