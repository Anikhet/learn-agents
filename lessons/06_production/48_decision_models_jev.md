# 48. Decision Models: Jev and Typed Probabilistic Answers

*Track A (agent engineering). Book: Chapter 2 (Routing) and Chapter 18 (Guardrails/Safety Patterns). Prerequisites: 02. Routing; 24. Structured Output; 13. Human in the Loop.*

Most of the decisions an agent makes are not essays. "Which team owns this ticket?" "Is this refund consistent with policy?" "How angry is this customer?" Each answer is one label, one number, or one yes/no. Yet the default way to get them is to ask a language model to *write* the answer as tokens and then parse what it wrote. This lesson is about a different kind of model built only for those decisions. TypeSafe AI's Jev returns typed answers with probability distributions instead of text. We build the client and the decision policy from scratch against a scripted stand-in, wire it into LangGraph as a router and as a pre-action guard, calibrate its confidence, and finish with the live API, the Python SDK, and the Pydantic AI integration.

## The problem

The support agent's triage step runs on every inbound ticket. Today it looks like lesson 02's LLM router:

```text
ticket ──► LLM: "Classify into billing/technical/sales. Return JSON
                 with route and confidence."
                 │
                 ▼
          '{"route": "billing", "confidence": 0.9}'   ◄── generated text
                 │
                 ▼
          json.loads(...) ──► handler
```

This works, and it has three problems that get worse at volume.

**1. The confidence is made up.** The `0.9` is a token the model wrote because the prompt asked for one. It was not computed from anything. Lesson 02 called it "advisory". In practice LLM self-reported confidence clusters around 0.8 to 0.95 whether the model is right or wrong. You cannot put a threshold on a number that does not move.

**2. You pay for generation you do not need.** The answer is one of three labels. The model still decodes token by token, through a JSON wrapper, often after a reasoning preamble. Latency is hundreds of milliseconds to seconds, and cost scales with output tokens. For a decision that runs on every ticket, before every tool call, and on every retrieved document, that is the dominant cost of the system.

**3. The output can be malformed.** `"Billing"` with a capital B. A fourth label nobody defined. Trailing prose after the JSON. Lesson 24 showed how constrained decoding fixes the format. It does not fix problems 1 and 2.

A **decision model** removes the generation step. You give it the state and a set of typed questions. It returns, for each question, a probability distribution over the allowed answers:

```text
you:    state   = "My card was charged twice and nobody replied in 3 days."
        questions = { route: choice[billing, shipping, technical],
                      urgency: score[low, medium, high],
                      needs_human: noul }

model:  route       → billing   p = {billing: .91, shipping: .02, technical: .07}
        urgency     → 1.6       p = {0: .05, 1: .30, 2: .65}
        needs_human → 0.88

your software: route to billing, priority high, flag for a person
```

It cannot answer with a label that isn't in the list, because the output space *is* the list. The probabilities come from the model's output distribution, not from words it wrote. The model decides and your software acts: the model supplies calibrated odds and your code owns the thresholds.

TypeSafe AI calls this a **System One** model, after Kahneman's fast/intuitive System 1 versus the slow, deliberate System 2. Jev entered limited early access in September 2026. The vendor claims speedups and cost reductions of one to two orders of magnitude over LLMs on classification-style tasks. Those are vendor numbers and this lesson does not depend on them. The architecture is the lesson.

## Build it without a framework

### The wire format

The API has one endpoint. Per the TypeSafe API reference:

```text
POST https://api.typesafe.ai/v1/systemone
Authorization: Bearer $TYPESAFE_API_KEY

{ "model": "jev-latest",
  "state": <string | object | array>,
  "questions": { "<your id>": <Question>, ... } }
```

There are exactly three question types:

| Type | Asks | `criteria` | Answer fields |
|---|---|---|---|
| `noul` | "Is this statement true?" | optional `{"true": ..., "false": ...}` | `noul`: probability of yes, 0 to 1 |
| `choice` | "Which one of these?" | required map of option → description (≤ 255 options) | `choice`, `probabilities`, `confidence` |
| `score` | "How much, on this ordered scale?" | required list, low → high (2 to 10 levels) | `score` (expected level), `legend`, `probabilities`, `confidence` |

Three details matter for correct use:

- **The question id is not sent to the model.** `"route"` is only your key for reading the answer. The full question has to be in `instructions`.
- **A `noul` value near 0.5 means "undecided", not "somewhat".** Use `score` for degree. A noul of 0.5 is the model telling you it does not know.
- **`score` is a probability-weighted mean of level indices.** With p = {0: .05, 1: .30, 2: .65} the score is 0×.05 + 1×.30 + 2×.65 = 1.6. It can fall between levels. Read `probabilities` when the shape matters: a bimodal {0: .5, 2: .5} also averages to 1.0.

`confidence` is derived from the distribution. The docs publish the statistic as:

```text
confidence = (n · p_max − 1) / (n − 1)        n = number of options

uniform (no idea):      n=3, p_max=1/3  →  (1 − 1)/2    = 0.0
certain:                n=3, p_max=1.0  →  (3 − 1)/2    = 1.0
docs quickstart:        n=3, p_max=0.85 →  (2.55 − 1)/2 = 0.775  (shown as 0.78)
```

It rescales the top probability so that 0 means "no better than guessing" whatever the number of options. A p_max of 0.5 is fairly confident with 10 options (0.44) and useless with 2 (0.0).

We cannot call the real API in an offline cell. `ScriptedJev` below stands in for it, the same way the scripted chat model stands in for an LLM in other lessons. It validates requests the way the API reference describes and builds responses in the documented shape. The probabilities come from a script. What we are studying is the control flow around the model, not the model. The live cells at the end call the real thing.

```{code-cell} python
import json


class JevAPIError(Exception):
    def __init__(self, status, message):
        super().__init__(f"{status}: {message}")
        self.status = status


def confidence(probs):
    """TypeSafe's published statistic: 0 at uniform, 1 when all mass is on one option."""
    n = len(probs)
    return max(0.0, min(1.0, (n * max(probs.values()) - 1) / (n - 1)))


def validate(body):
    """The checks the API reference documents. The real API answers 422 and names the field."""
    for key in ("state", "model", "questions"):
        if key not in body:
            raise JevAPIError(422, f"missing field: {key}")
    for qid, q in body["questions"].items():
        kind = q.get("type")
        if kind not in ("noul", "choice", "score"):
            raise JevAPIError(422, f"questions.{qid}.type: {kind!r}")
        if "instructions" not in q:
            raise JevAPIError(422, f"questions.{qid}.instructions: required")
        if kind == "choice" and len(q.get("criteria", {})) > 255:
            raise JevAPIError(422, f"questions.{qid}.criteria: max 255 options")
        if kind == "score" and not 2 <= len(q.get("criteria", [])) <= 10:
            raise JevAPIError(422, f"questions.{qid}.criteria: 2 to 10 levels")


class ScriptedJev:
    """Stands in for POST /v1/systemone. Each call replays one scripted distribution per question.

    Script values: a float for noul, {option: p} for choice, [p_level0, p_level1, ...] for score.
    """

    def __init__(self, *turns):
        self.turns = iter(turns)
        self.calls = []

    def system_one(self, body):
        validate(body)
        self.calls.append(body)
        script = next(self.turns)
        answers = {}
        for qid, q in body["questions"].items():
            p = script[qid]
            if q["type"] == "noul":
                answers[qid] = {"type": "noul", "noul": p}
            elif q["type"] == "choice":
                assert set(p) == set(q["criteria"]), "script must cover exactly the options"
                answers[qid] = {"type": "choice", "choice": max(p, key=p.get),
                                "confidence": confidence(p), "probabilities": p}
            else:
                probs = {str(i): v for i, v in enumerate(p)}
                answers[qid] = {"type": "score", "score": sum(i * v for i, v in enumerate(p)),
                                "confidence": confidence(probs), "probabilities": probs,
                                "legend": {str(i): c for i, c in enumerate(q["criteria"])}}
        return {"model": "jev-scripted", "answers": answers,
                "usage": {"input_tokens": len(json.dumps(body)) // 4, "output_tokens": 0}}


TRIAGE_QUESTIONS = {
    "route": {"type": "choice", "instructions": "Which team should handle this ticket?",
              "criteria": {"billing": "payments, invoices, charges, refunds",
                           "technical": "bugs, errors, integrations",
                           "sales": "pricing, plans, seats"}},
    "urgency": {"type": "score", "instructions": "How urgent is this ticket?",
                "criteria": ["can wait a week", "should be handled today", "customer is blocked or losing money"]},
    "needs_human": {"type": "noul",
                    "instructions": "This ticket needs a human agent rather than an automated reply.",
                    "criteria": {"true": "money, legal, or an unanswered complaint",
                                 "false": "a routine question a bot can close"}},
}

jev = ScriptedJev({"route": {"billing": 0.85, "technical": 0.15, "sales": 0.0},
                   "urgency": [0.05, 0.30, 0.65], "needs_human": 0.88})
resp = jev.system_one({"model": "jev-latest", "questions": TRIAGE_QUESTIONS,
                       "state": "My card was charged twice and nobody has replied in three days."})
print(json.dumps(resp["answers"], indent=1)[:600])

assert resp["answers"]["route"]["choice"] == "billing"
assert abs(resp["answers"]["route"]["confidence"] - 0.775) < 1e-9       # matches the docs example
assert abs(resp["answers"]["urgency"]["score"] - 1.6) < 1e-9            # weighted mean of levels
assert confidence({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}) < 1e-9          # uniform means no information
try:
    jev.system_one({"model": "jev-latest", "state": "x",
                    "questions": {"q": {"type": "score", "instructions": "?", "criteria": ["only one"]}}})
except JevAPIError as e:
    print("rejected:", e)
    assert e.status == 422
```

### Parse into types, and check the invariants

The model's output space is typed, but the payload still crosses a network. Parse it into types at the boundary and check the invariants the docs promise: probabilities sum to 1, `choice` is the argmax, and every question you asked has an answer. If one fails, it is a client bug, a version change, or a proxy that changed the payload. Raise, don't route.

```{code-cell} python
from typing import Annotated, Literal, Union
from pydantic import BaseModel, Field, TypeAdapter, model_validator


def _sums_to_one(probs):
    if abs(sum(probs.values()) - 1.0) > 1e-3:
        raise ValueError(f"probabilities sum to {sum(probs.values()):.4f}")


class NoulAnswer(BaseModel):
    type: Literal["noul"]
    noul: float = Field(ge=0, le=1)


class ChoiceAnswer(BaseModel):
    type: Literal["choice"]
    choice: str
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[str, float]

    @model_validator(mode="after")
    def consistent(self):
        _sums_to_one(self.probabilities)
        if self.probabilities[self.choice] < max(self.probabilities.values()):
            raise ValueError("choice is not the argmax")
        return self


class ScoreAnswer(BaseModel):
    type: Literal["score"]
    score: float
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[str, float]
    legend: dict[str, str]

    @model_validator(mode="after")
    def consistent(self):
        _sums_to_one(self.probabilities)
        return self

    def top_level(self):
        """The single most likely level. `score` is a mean and can hide a bimodal split."""
        return self.legend[max(self.probabilities, key=self.probabilities.get)]


Answer = TypeAdapter(Annotated[Union[NoulAnswer, ChoiceAnswer, ScoreAnswer], Field(discriminator="type")])


def parse_answers(response, questions):
    missing = set(questions) - set(response["answers"])
    if missing:
        raise ValueError(f"no answer for {sorted(missing)}")
    return {qid: Answer.validate_python(a) for qid, a in response["answers"].items()}


answers = parse_answers(resp, TRIAGE_QUESTIONS)
print({k: type(v).__name__ for k, v in answers.items()})
print("urgency top level:", answers["urgency"].top_level())
assert isinstance(answers["route"], ChoiceAnswer) and answers["route"].choice == "billing"
assert answers["urgency"].top_level() == "customer is blocked or losing money"

broken = {"answers": {**resp["answers"], "route": {**resp["answers"]["route"], "choice": "sales"}}}
try:
    parse_answers(broken, TRIAGE_QUESTIONS)
except ValueError as e:
    print("caught:", str(e).splitlines()[0])
```

### The decision policy is your code

Jev gives you distributions. What happens next is a **policy**, and the policy belongs to you. It is plain code, it is versioned, and it gets tested. For triage:

```text
needs_human.noul ≥ 0.80            → human queue            (safety first, no model can override)
route.confidence ≥ 0.60            → that team's handler
0.35 < needs_human.noul < 0.65     → treated as "undecided", same as low confidence
otherwise                          → escalate to an LLM (slow path) → if still unsure, clarify
```

This is lesson 02's cascade with a better middle tier. Rules stay in front because they're free. The decision model handles the bulk because it's fast and gives real probabilities. The LLM only sees the remainder, where reading and reasoning actually help.

```text
ticket ──► rules ──hit──► handler
             │miss
             ▼
            Jev ──confident──► handler / human
             │unsure
             ▼
            LLM ──confident──► handler
             │unsure
             ▼
          clarify (no side effects)
```

```{code-cell} python
POLICY = {"human_at": 0.80, "route_at": 0.60, "undecided": (0.35, 0.65), "llm_at": 0.70}


def jev_triage(jev, text):
    body = {"model": "jev-latest", "state": text, "questions": TRIAGE_QUESTIONS}
    return parse_answers(jev.system_one(body), TRIAGE_QUESTIONS)


def decide(text, jev, llm, policy=POLICY):
    """Returns (destination, trace). The trace records which tier decided and why."""
    trace = []
    a = jev_triage(jev, text)
    human, route = a["needs_human"].noul, a["route"]
    trace.append(("jev", route.choice, round(route.confidence, 2), round(human, 2)))

    if human >= policy["human_at"]:
        return "human", trace
    lo, hi = policy["undecided"]
    undecided = lo < human < hi
    if route.confidence >= policy["route_at"] and not undecided:
        return route.choice, trace

    # Slow path: only the tickets Jev could not settle pay for an LLM call.
    raw = json.loads(llm.invoke(f"Route to billing/technical/sales. JSON route+confidence.\n\n{text}").content)
    trace.append(("llm", raw["route"], raw["confidence"]))
    if raw["route"] in TRIAGE_QUESTIONS["route"]["criteria"] and raw["confidence"] >= policy["llm_at"]:
        return raw["route"], trace
    trace.append(("fallback", "clarify"))
    return "clarify", trace


print("policy:", POLICY)
assert POLICY["undecided"][0] < 0.5 < POLICY["undecided"][1]
```

The LLM tier needs the course's scripted chat model:

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


cases = [
    # Clear billing ticket: Jev decides, the LLM is never called (empty script would raise).
    ("Invoice #4411 charged us for 12 seats, we have 10.",
     {"route": {"billing": 0.93, "technical": 0.04, "sales": 0.03}, "urgency": [0.6, 0.3, 0.1], "needs_human": 0.12},
     scripted()),
    # Double charge + ignored complaint: needs_human wins regardless of route.
    ("Charged twice, no reply in three days, I'm disputing with my bank.",
     {"route": {"billing": 0.90, "technical": 0.05, "sales": 0.05}, "urgency": [0.0, 0.1, 0.9], "needs_human": 0.95},
     scripted()),
    # "Upgrade broke my export": billing vs technical vs sales, Jev splits, LLM settles it.
    ("Since upgrading our plan the CSV export returns a 500.",
     {"route": {"billing": 0.20, "technical": 0.45, "sales": 0.35}, "urgency": [0.2, 0.6, 0.2], "needs_human": 0.30},
     scripted('{"route": "technical", "confidence": 0.85}')),
    # Vague: Jev unsure, LLM unsure -> clarify, never act.
    ("hey can you look at the thing from before",
     {"route": {"billing": 0.34, "technical": 0.33, "sales": 0.33}, "urgency": [0.4, 0.4, 0.2], "needs_human": 0.50},
     scripted('{"route": "sales", "confidence": 0.40}')),
]

results = []
for text, script, llm in cases:
    dest, trace = decide(text, ScriptedJev(script), llm)
    results.append(dest)
    print(f"{dest:9} <- {trace}")

assert results == ["billing", "human", "technical", "clarify"]
```

The third case is the design working as intended. Jev's route distribution was {technical: .45, sales: .35, billing: .20}. That is a real split: the ticket mentions an upgrade and a crash. Confidence was (3 × 0.45 − 1)/2 = 0.175, far below the bar, so the ticket went to the slow path. With an LLM router alone you would have paid for the LLM on all four tickets. Here, one in four reached it. The fraction that reaches the LLM tier is the number to track in production.

### A decision model as a pre-action guard

Triage is the obvious use. The more valuable one is **guarding actions**. Before an agent executes a side-effecting tool call, ask a `noul` about the proposed action. The latency budget for a check that runs before every tool call is tight, and that's where a millisecond-class model fits and an LLM-as-judge doesn't.

`state` can be an object. Instructions can refer to its fields by backticked path, so the guard sees the request and the proposed action as separate fields:

```{code-cell} python
def guard_action(jev, ticket, proposed, threshold=0.90):
    """Ask whether a proposed tool call follows from the request. Below threshold -> human approval."""
    body = {
        "model": "jev-latest",
        "state": {"ticket": ticket, "proposed_action": proposed,
                  "policy": "Refunds only for duplicate or erroneous charges, max $500 without approval."},
        "questions": {"consistent": {
            "type": "noul",
            "instructions": "`proposed_action` is what the customer in `ticket` asked for and is allowed by `policy`.",
        }},
    }
    p = parse_answers(jev.system_one(body), body["questions"])["consistent"].noul
    return ("execute" if p >= threshold else "needs_approval"), p


ok = guard_action(ScriptedJev({"consistent": 0.97}),
                  "Charged twice for invoice in_9F2 ($120).",
                  {"tool": "issue_refund", "args": {"invoice": "in_9F2", "amount": 120}})
bad = guard_action(ScriptedJev({"consistent": 0.08}),
                   "Charged twice for invoice in_9F2 ($120).",
                   {"tool": "issue_refund", "args": {"invoice": "in_9F2", "amount": 1200}})
print(ok, bad)
assert ok[0] == "execute" and bad[0] == "needs_approval"
```

The threshold is 0.90, not 0.5. The costs are asymmetric. A false "consistent" moves money, and a false "inconsistent" costs a human thirty seconds. The TypeSafe docs suggest a high bar like 0.9 before destructive actions as an example. Your real number comes from calibration, later in this lesson.

## How LangGraph does it

The from-scratch pieces map directly onto graph primitives:

```text
from scratch                          LangGraph
------------                          ---------
jev_triage(...) + decide(...)         `triage` node writing state["route"], state["trace"]
dest -> handler                       add_conditional_edges("triage", pick, [...])
guard_action -> "needs_approval"      `guard` node calling interrupt(...) (13. Human in the Loop)
approval arrives later                Command(resume=...) on the same thread_id
```

The decision model lives inside ordinary nodes. LangGraph doesn't know or care that the model returns distributions and not messages. What LangGraph adds is the **pause**: when the guard is unsure, `interrupt` checkpoints the graph and returns control to the caller. A person approves or rejects hours later, and `Command(resume=...)` continues the run from that node. Without the checkpointer, "needs approval" is a dead end. With it, it is a durable wait (see 36. Durable Agent Orchestration).

```{code-cell} python
import operator
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command


class TicketState(TypedDict, total=False):
    text: str
    route: str
    proposed: dict
    verdict: str
    p_consistent: float
    trace: Annotated[list, operator.add]
    outcome: str


def build(jev_triage_client, jev_guard_client, llm):
    def triage(state: TicketState):
        dest, trace = decide(state["text"], jev_triage_client, llm)
        return {"route": dest, "trace": trace}

    def billing(state: TicketState):
        # A real billing handler would be an agent that proposes this call. Here it is fixed.
        return {"proposed": {"tool": "issue_refund", "args": {"invoice": "in_9F2", "amount": 1200}}}

    def guard(state: TicketState):
        # Score in its own node. A resumed node re-runs from its first line, so the model
        # call must not live in the same node as interrupt() or it runs twice.
        verdict, p = guard_action(jev_guard_client, state["text"], state["proposed"])
        return {"verdict": verdict, "p_consistent": p}

    def approve(state: TicketState):
        decision = interrupt({"proposed": state["proposed"], "p_consistent": state["p_consistent"]})
        return {"verdict": "execute" if decision == "approve" else "blocked",
                "trace": [("reviewer", decision)]}

    def execute(state: TicketState):
        if state["verdict"] != "execute":
            return {"outcome": "refund blocked by reviewer"}
        return {"outcome": f"executed {state['proposed']['tool']}"}

    def simple(name):
        return lambda state: {"outcome": f"{name} handled"}

    b = StateGraph(TicketState)
    b.add_node("triage", triage)
    b.add_node("billing", billing)
    b.add_node("guard", guard)
    b.add_node("approve", approve)
    b.add_node("execute", execute)
    for n in ("technical", "sales", "human", "clarify"):
        b.add_node(n, simple(n))
        b.add_edge(n, END)
    b.add_edge(START, "triage")
    b.add_conditional_edges("triage", lambda s: s["route"],
                            ["billing", "technical", "sales", "human", "clarify"])
    b.add_edge("billing", "guard")
    b.add_conditional_edges("guard", lambda s: "execute" if s["verdict"] == "execute" else "approve",
                            ["execute", "approve"])
    b.add_edge("approve", "execute")
    b.add_edge("execute", END)
    return b.compile(checkpointer=InMemorySaver())


graph = build(
    ScriptedJev({"route": {"billing": 0.93, "technical": 0.04, "sales": 0.03},
                 "urgency": [0.2, 0.5, 0.3], "needs_human": 0.20}),
    ScriptedJev({"consistent": 0.08}),        # $1200 refund on a $120 double charge
    scripted(),
)
config = {"configurable": {"thread_id": "ticket-4411"}}
first = graph.invoke({"text": "Charged twice for invoice in_9F2 ($120)."}, config)
pending = first["__interrupt__"][0].value
print("paused for approval:", pending)
assert first["route"] == "billing" and pending["p_consistent"] < 0.9

final = graph.invoke(Command(resume="reject"), config)
print(final["outcome"], final["trace"])
assert final["outcome"] == "refund blocked by reviewer"
assert len(graph.get_state(config).values["trace"]) == 2   # triage + reviewer: Jev guard ran once
```

```text
START ──► triage ──┬──► billing ──► guard ──(p ≥ .9)────────────────────► execute ──► END
     (Jev choice + │                  │                                      ▲
      noul, LLM on ├──► technical     └─(p < .9)──► approve ── interrupt ···─┘
      low conf)    ├──► sales                       (human decides, resume)
                   ├──► human
                   └──► clarify
```

The split between `guard` and `approve` is not cosmetic. The first version of this graph had the Jev call and `interrupt()` in the same node, and on resume the scripted client raised `StopIteration`. When a run resumes, LangGraph re-executes the interrupted node **from its first line**. `interrupt()` returns the resume value the second time through, but everything above it runs again. Against a real API that means a second paid call, and a second *different* probability that the reviewer never saw. The rule: anything non-idempotent (a model call, a write) goes in a node *before* the one that interrupts.

### Pydantic AI: a decision model behind an agent interface

Pydantic AI supports Jev as a model provider. Its docs describe `TypeSafeModel` as a `DecisionModel`: you declare an `output_type` (a `bool`, an `Enum`, or a Pydantic model whose field docstrings are the questions) and the integration turns it into `noul`/`choice`/`score` questions. It is the most compact way to see the idea that *the output type is the question*. Install with `pip install "pydantic-ai-slim[typesafe]"`. This is a live cell:

```{code-cell} python live
from enum import Enum
from pydantic import BaseModel, ConfigDict
from pydantic_ai import Agent
from pydantic_ai.models.typesafe import TypeSafeModel


class Team(Enum):
    billing = "billing"
    """Payments, invoices, charges, refunds."""
    technical = "technical"
    """Bugs, errors, integrations."""
    sales = "sales"
    """Pricing, plans, seats."""


class Triage(BaseModel):
    """Triage an inbound support ticket."""
    model_config = ConfigDict(use_attribute_docstrings=True)

    team: Team
    """Which team should handle this ticket?"""
    needs_human: bool
    """This ticket needs a human agent rather than an automated reply."""


agent = Agent(TypeSafeModel("jev-latest"), output_type=Triage)
result = agent.run_sync("My card was charged twice and nobody has replied in three days.")
print(result.output)
# Per-field confidence, per the Pydantic AI TypeSafe page:
print(result.response.provider_details.get("confidence"))
```

The same docs show `FallbackModel('typesafe:jev-latest', <an LLM>)` for handing off to a language model. Note that it is a *failure* fallback (errors, unavailability), not a *confidence* fallback. The confidence cascade in `decide` is still your code.

### The raw API and the official SDK

The raw HTTP call uses only the standard library:

```{code-cell} python live
import json, os, urllib.request

body = {"model": "jev-latest", "questions": TRIAGE_QUESTIONS,
        "state": "My card was charged twice and nobody has replied in three days."}
req = urllib.request.Request(
    "https://api.typesafe.ai/v1/systemone", data=json.dumps(body).encode(),
    headers={"Authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}",
             "Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=10) as r:
    live = json.load(r)
print(live["model"], live["usage"])
answers = parse_answers(live, TRIAGE_QUESTIONS)   # the same parser as the offline cells
print({k: v.model_dump() for k, v in answers.items()})
```

And the official Python SDK (`pip install typesafe-sdk`), per the quickstart, which reads `TYPESAFE_API_KEY` and defaults to `jev-latest`:

```{code-cell} python live
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

with TypeSafeClient() as client:
    response = client.system_one(
        state="Since upgrading our plan the CSV export returns a 500.",
        questions={
            "route": Choice("Which team should handle this ticket?",
                            criteria={"billing": "payments, invoices", "technical": "bugs, errors",
                                      "sales": "pricing, plans"}),
            "urgency": Score("How urgent is this ticket?",
                             criteria=["can wait", "today", "customer is blocked"]),
            "needs_human": Noul("This ticket needs a human agent."),
        },
    )
print(response.answers["route"].choice, response.answers["route"].probabilities)
print(response.answers["urgency"].score, response.answers["needs_human"].noul)
```

The SDK's exact constructor signatures may differ from this sketch. Check the quickstart for the installed version. The HTTP cell above is the stable contract.

## What goes wrong in production

1. **Treating `confidence` as accuracy.** Symptom: you set `route_at = 0.9` expecting 90% accuracy and measure 78%. Cause: confidence is a rescaled top probability. It only means accuracy if the model is *calibrated on your data*. Fix: measure it. Build a labeled set from real tickets, bin predictions by confidence, and compare each bin's predicted confidence with its actual accuracy. Pick the threshold from the curve, not from intuition. The next cell shows how.

2. **Reading `noul` 0.5 as "half yes".** Symptom: a guard that approves at > 0.5 lets through actions the model had no opinion on. Cause: noul is a probability, and 0.5 is maximum uncertainty. Fix: treat a band around 0.5 as "undecided" and route it to the slow path or a person, as `decide` does.

3. **Questions written for the id, not the model.** Symptom: `{"is_spam": {"type": "noul", "instructions": "spam?"}}` performs badly. Cause: the key `is_spam` is never sent to the model, so the model sees a one-word question with no definition. Fix: write the full statement in `instructions` and define the poles in `criteria`. Treat question text like tool descriptions (40. Tool Design): it is a prompt, and it gets versioned and evaluated.

4. **Overlapping options.** Symptom: probability mass splits between `billing` and `sales` on every plan-change ticket, and confidence stays low. Cause: the options overlap. The model is honestly uncertain because the taxonomy is. Fix: define options by *which handler is needed*, as lesson 02 did. A persistently low-confidence cluster is a signal about your taxonomy, not about the model.

5. **Silent model upgrades.** Symptom: the route distribution shifts overnight. Cause: `jev-latest` is an alias, and the response's `model` field shows a new version. Fix: pin a versioned model id in production. Log `response["model"]` on every call. Re-run the calibration eval before moving the pin, exactly like an LLM upgrade (see 22. Evaluation Reliability).

6. **No budget for the slow path.** Symptom: a new product launch doubles ambiguous tickets, the LLM tier saturates, and triage latency spikes. Cause: the cascade's cost model assumed a stable escalation rate. Fix: emit the escalation rate as a metric and alert on it. Put backpressure on the slow path: a queue with a timeout that falls back to `clarify`, never a blocking wait.

7. **Retries without idempotency.** Symptom: a 529 (overloaded) mid-triage and a retry that routes the same ticket twice. Cause: triage has a side effect (enqueue to a team) and the retry repeats it. Fix: the decision call is safe to retry, and the SDKs retry 429/529 with backoff. The *effect* must be idempotent: key the enqueue on ticket id. For state too large for the context window (the API rejects it with a token-limit error), truncate deliberately in the harness. Never let the client pick what to cut.

8. **Using a decision model where you need reading.** Symptom: poor accuracy on "does clause 14.2 contradict the new DPA?". Cause: that question needs multi-step reasoning over long text. System One is the wrong system. Fix: decision models are for fast judgments over bounded state. Reasoning questions go to a reasoning model, and the cascade exists so you can do both.

### Calibrate before you pick thresholds

This cell shows the calibration workflow on a synthetic eval set. The synthetic model is deliberately overconfident: its true accuracy is below its stated confidence. We measure expected calibration error (ECE), then choose the lowest threshold that meets a target precision. That maximizes the tickets handled on the fast path without breaking the quality bar.

```{code-cell} python
import numpy as np

rng = np.random.default_rng(7)
n = 2000
conf = rng.uniform(0, 1, n)
correct = rng.random(n) < (0.30 + 0.60 * conf)       # true accuracy lags stated confidence at the top


def reliability(conf, correct, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    rows, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.any():
            gap = abs(conf[m].mean() - correct[m].mean())
            ece += m.mean() * gap
            rows.append((round(lo, 1), round(conf[m].mean(), 2), round(correct[m].mean(), 2), int(m.sum())))
    return rows, ece


def threshold_for(conf, correct, target_precision):
    """Lowest threshold whose accepted set meets the precision target -> maximum fast-path coverage."""
    for t in np.round(np.arange(0.0, 1.0, 0.01), 2):
        accepted = conf >= t
        if accepted.sum() >= 50 and correct[accepted].mean() >= target_precision:
            return float(t), float(accepted.mean()), float(correct[accepted].mean())
    return None


rows, ece = reliability(conf, correct)
for r in rows[-4:]:
    print("bin %.1f  stated %.2f  actual %.2f  n=%d" % r)
t, coverage, precision = threshold_for(conf, correct, target_precision=0.85)
print(f"ECE={ece:.3f}  threshold={t}  fast-path coverage={coverage:.0%}  precision={precision:.2f}")

assert ece > 0.05                    # this model's confidence overstates accuracy
assert precision >= 0.85 and t > 0.6 # the naive 0.6 would miss the target
```

On this synthetic model, stated 0.95 means about 0.88 actual. A threshold that looks conservative at 0.6 does not meet an 85% precision target. The measured threshold (0.88) does, but it keeps only about 12% of traffic on the fast path. That is what miscalibration costs: the cascade still meets the quality bar, but most tickets pay for the slow path. With a well-calibrated model the same target would admit far more. Redo this per question and per model version. Don't do it once for "Jev".

## What changed by 2026

- **Decision models became a separate model category.** In 2026 the idea that a model can return typed decisions with distributions, rather than generated text, shipped as a product. Jev entered limited early access in September 2026, with a hosted API only and no open weights. It is early. Treat vendor speed and cost figures as claims to verify on your own workload.
- **Frameworks gained a slot for them.** Current Pydantic AI docs describe a `DecisionModel` base class and a TypeSafe provider, where the agent's `output_type` becomes the question set. LiteLLM, Vercel's AI SDK, and Cloudflare also document Jev integrations. In LangGraph there's no special primitive and none is needed. A decision model is a call inside a node, and conditional edges and `interrupt` do the rest.
- **Cascades got a principled middle tier.** Lesson 02's cascade (rules → classifier → LLM) always had a weak middle: training and hosting your own classifier. A hosted decision model fills that slot without a training loop, and it returns probabilities you can threshold, which LLM self-reports never gave you.
- **Guards moved closer to the action.** Pre-action checks used to be expensive LLM-as-judge calls, so teams ran them only on the riskiest tools. Fast typed judgments make it practical to check every side-effecting tool call. This is the same trajectory as the guardrails in 18. Guardrails and the tool-layer policies in 31. Prompt Injection.
- **Calibration became a first-class eval.** Once thresholds drive routing and approval, "is 0.9 really 0.9?" is a production question. Reliability curves and ECE sit alongside accuracy in the eval suite (19. Evaluation and Monitoring).

## Interview questions

**Q1. Why not just ask an LLM for a label and a confidence score?**

Strong: Three reasons: the confidence, the cost, and the format. An LLM's self-reported confidence is a generated token and is not derived from its output distribution, so it tends to cluster high whether the model is right or wrong. You can't threshold it reliably. Generation also costs latency and output tokens that a one-label decision doesn't need, and at the triage or guard layer that runs on every request, it dominates the bill. Constrained decoding fixes the format but not the other two. A decision model returns a distribution over only the allowed answers. The confidence comes from that distribution, and I still check calibration before I trust any threshold.

Weak: "LLMs hallucinate, decision models don't."

**Q2. Where would you put a decision model in an agent architecture?**

Strong: Anywhere the output is a label, a level, or a yes/no, and the call runs often enough that latency and cost matter. There are three places. First, the triage router in front of specialized handlers, as the middle tier of a cascade after rules and before an LLM. Second, a pre-action guard that checks every side-effecting tool call against the request and policy, with an `interrupt` for approval when the probability is below a calibrated bar. Third, filters on retrieved documents or tool outputs, for example "is this chunk relevant" or "does this output contain an instruction". I would not use it for anything that needs multi-step reasoning over long text.

Weak: "Replace the LLM with it."

**Q3. How do you choose the confidence threshold?**

Strong: Empirically, per question and per model version. I build a labeled set from real traffic, including the boundary cases. I bin predictions by confidence and compare predicted confidence with actual accuracy in each bin, which gives a reliability curve and ECE. Then I set a target precision for the fast path from the cost of an error, and pick the lowest threshold that meets it, which maximizes coverage. The target differs by action: routing to the wrong team is cheap, refunding the wrong amount is not. I pin the model version and re-run this before any upgrade.

Weak: "0.8 seems reasonable."

**Q4. A `noul` guard returns 0.52 for a refund. What does your system do?**

Strong: It treats 0.52 as "the model doesn't know", not "slightly yes". Noul is a probability of yes, so the region around 0.5 is maximum uncertainty. For a destructive action the approve threshold sits far above that, around 0.9 depending on calibration. So this goes to human approval: the graph interrupts, checkpoints, and waits for a resume. Whatever the reviewer decides becomes a labeled example for the next calibration pass.

Weak: "0.52 is above 0.5 so it passes."

**Q5. Your triage confidence dropped across the board last Tuesday. How do you debug it?**

Strong: First I check whether the model changed. The response carries the versioned model id, so I compare it before and after. If we were on the `latest` alias, a silent upgrade is the first suspect. Next I check whether the input changed: a new product, a new ticket source, or a new state format from an upstream service. Then I check whether the question changed, since instructions and option descriptions are prompts and they should be versioned. Low confidence concentrated on two options usually means the taxonomy overlaps for a new kind of ticket, which is a taxonomy fix, not a model fix. Meanwhile the cascade should be absorbing it through the slow path, so I also check that the escalation budget held.

Weak: "Retrain the model."

**Q6. What are the trade-offs of a hosted decision model versus a classifier you train yourself?**

Strong: A hosted model needs no labeled training set to start. You describe the options in natural language and change them without retraining. The costs are a network hop, a vendor dependency, data leaving your boundary, and a model version you don't control. A self-trained classifier is cheaper per call and fully under your control, but it needs labeled data, a training pipeline, and retraining whenever the taxonomy changes. I'd start with the hosted model to find the taxonomy and collect labels from escalations. I'd move to an in-house classifier when volume, latency, or data residency justifies it, keeping the same policy code and calibration eval.

Weak: "Hosted is always easier."

## Build it yourself

Extend the cascade with a **per-question calibrated policy** and a **retrieval filter**:

1. Add a `relevant` noul question and use it to filter five retrieved Notion chunks before they reach the handler's context (34. Context Engineering). Keep chunks with `noul ≥ t_rel`.
2. Replace the constants in `POLICY` with thresholds computed by `threshold_for` from a labeled set for each question (`route`, `needs_human`, `relevant`), using different precision targets per question.
3. Add a `ScriptedJev` failure mode that raises `JevAPIError(529, ...)` on the first call. Retry with backoff, and make the "enqueue to team" step idempotent on ticket id.
4. Record every `decide` trace as a row `(ticket_id, tier, choice, confidence, final_label)` so that escalations become new calibration data.

Acceptance checks:

- `assert` that no chunk with `noul < t_rel` reaches the handler's context.
- `assert` that the `needs_human` threshold's measured precision ≥ its target on the labeled set, and that its target is stricter than `route`'s.
- `assert` that a 529 on the first attempt produces exactly one enqueue after retry.
- `assert` that the fraction of tickets reaching the LLM tier is computed from the trace log and is below a budget you set.

## Key takeaways

- A decision model answers typed questions (yes/no, choice, ordered score) with probability distributions instead of generated text.
- Its confidence is derived from the output distribution, which makes it thresholdable in a way an LLM's self-reported confidence is not.
- The model supplies the odds and your code owns the policy: thresholds, undecided bands, and escalation are versioned and tested.
- The best place for it is the middle of a cascade (rules → decision model → LLM → clarify) and as a guard in front of every side-effecting tool call.
- A `noul` near 0.5 means "doesn't know", and it should route to the slow path or a person, never to action.
- Calibrate per question and per pinned model version, and pick thresholds from the reliability curve and the cost of an error.
- Question text is a prompt: the id is never sent, so the full question and its options' definitions go in `instructions` and `criteria`.
