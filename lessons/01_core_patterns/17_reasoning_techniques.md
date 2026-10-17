# 17. Reasoning Techniques

*Track A (agent engineering). Book: Chapter 17 (Reasoning Techniques). Prerequisites: 00. What an agent actually is, 04. Reflection, 06. Planning.*

A single forward pass of a language model is a fixed amount of computation. It cannot pause, check a sub-result, or back up when an early guess turns out wrong. Reasoning techniques are ways of buying more computation, and more structure, for problems where one pass is not enough. Chain-of-thought, self-consistency, tree-of-thought, ReAct, and debate are all the same idea wearing different clothes: spend more tokens or more samples, and arrange them so that errors get caught.

This lesson builds each technique as control flow you can read, then shows where the reasoning models of 2026 changed the advice. The most important change is this: the model now does a lot of its own thinking, but that written thinking is not a faithful record of why it answered the way it did. Your job moved from prompting the reasoning to verifying the outcome.

## The problem

The running example again. A Slack message arrives:

> "Acme says their September invoice is wrong. It should be $4,800 and it shows $6,200. Why?"

The support agent has tools for the CRM, Stripe, and SQL. Suppose it answers in one pass:

```text
model: "The invoice is wrong because Acme was charged for 31 seats instead of 24.
        24 seats at $200 is $4,800, so the extra 7 seats explain the $1,400 difference."
```

Fluent, specific, and wrong. Stripe shows 24 seats. The real cause is a duplicated one-time onboarding line item of $1,400. The model pattern-matched "seat count" because that is the most common invoice complaint in its training data, then generated arithmetic that fit its story.

Three different things failed:

```text
1. No decomposition.   The question hides sub-questions: what is on the invoice,
                       what should be on it, which line differs.
2. No alternatives.    The first hypothesis (seat count) was the only one considered.
3. No verification.    The arithmetic "checked out" against the model's own guess,
                       not against Stripe.
```

Every technique in this lesson attacks one of those three:

| Technique | Attacks | Mechanism | Cost |
|---|---|---|---|
| Chain-of-thought (CoT) | 1 | Emit intermediate steps before the answer | 1 call, more output tokens |
| Self-consistency | 2 | Sample N answers, vote or verify | N calls |
| Tree-of-thought (ToT) | 2 | Propose branches, score, prune, expand best | Many short calls |
| ReAct | 1 and 3 | Interleave reasoning with tool observations | 1 call per step |
| Debate / multi-agent | 2 and 3 | Independent proposers, a judge | 2 to 4 calls per round |
| Verification | 3 | Check the claim against ground truth or a test | 1 tool call or 1 grader |

The book frames these as techniques for making an agent's thought process more robust. The systems view is simpler: they are all ways of trading tokens for reliability, and each one has a control-flow shape you can draw.

```text
CoT             prompt -> [step, step, step, answer]

Self-consist.   prompt -> answer_1 ┐
                prompt -> answer_2 ├-> vote / verify -> answer
                prompt -> answer_3 ┘

ToT             root -> propose k -> score -> keep best b -> propose k ... -> answer

ReAct           think -> act -> observe -> think -> act -> observe -> answer

Debate          proposer A ┐
                proposer B ├-> judge (with evidence) -> answer
```

### Why "think step by step" is no longer the lever

For several years the standard advice was to append "let's think step by step" or to show few-shot worked examples. That worked because the base models did not reason unless the prompt shaped the output into reasoning. Modern reasoning models are trained, with reinforcement learning on verifiable outcomes, to produce long internal chains of thought before answering. They already decompose. Telling them to do so is at best redundant and at worst harmful: the vendors' current guidance is to state the task, constraints, output format, and definition of done, and to leave the how to the model.

The second change matters more for anyone building agents. Anthropic's 2025 study "Reasoning Models Don't Always Say What They Think" gave models hints that changed their answers, then checked whether the visible chain-of-thought mentioned the hint. Claude 3.7 Sonnet verbalized it about 25 percent of the time and DeepSeek R1 about 39 percent. In a reward-hacking setup the models exploited the hack in over 99 percent of cases and admitted it in under 2 percent. The reasoning text is useful signal. It is not an audit log. If you need to know that the invoice diagnosis is right, you check Stripe, not the prose.

So the design rule for 2026 is: structure the loop, verify the outcome, and treat the model's stated reasoning as one more input rather than as evidence.

## Build it without a framework

Every offline cell uses a scripted stand-in for the model so the control flow is the object of study. The script replays fixed turns; the shape of the loop is real. A `live` cell at the end runs the same shape against a real model.

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


print("scripted model ready")
```

### Self-consistency: vote, then verify

Self-consistency samples the same question several times at nonzero temperature and takes the majority answer. The intuition: there are many ways to be wrong and usually one way to be right, so wrong answers spread out and the right one concentrates.

Below, `sample_diagnosis` stands in for a model call at temperature 0.7. It is right 60 percent of the time and wrong in three different ways. Watch what happens to agreement as `k` grows, and notice the second half: a vote is only a proxy for correctness. When a cheap ground-truth check exists, use it instead of, or on top of, the vote.

```{code-cell} python
import random
from collections import Counter

STRIPE_INVOICE = {  # what the tool would return; the ground truth for verification
    "lines": [
        {"desc": "24 seats x $200", "amount": 4800},
        {"desc": "Onboarding (one-time)", "amount": 1400},
        {"desc": "Onboarding (one-time)", "amount": 1400},
    ],
    "total": 7600,
    "expected_total": 4800 + 1400,  # onboarding was legitimately billed once
}

CANDIDATES = {  # each is a (diagnosis, claimed_overcharge) the model might produce
    "duplicate onboarding line": 1400,
    "7 extra seats": 1400,
    "tax applied twice": 480,
    "wrong plan tier": 2400,
}


def sample_diagnosis(rng):
    """Stand-in for one sampled model answer: 60% right, 40% spread over wrong answers."""
    return rng.choices(list(CANDIDATES), weights=[60, 20, 10, 10])[0]


def self_consistency(k, seed=7):
    rng = random.Random(seed)
    votes = Counter(sample_diagnosis(rng) for _ in range(k))
    best, n = votes.most_common(1)[0]
    return best, n / k, dict(votes)


for k in (1, 5, 15):
    best, agreement, votes = self_consistency(k)
    print(f"k={k:>2}  answer={best!r:30s} agreement={agreement:.0%}  votes={votes}")


def verify(diagnosis, invoice):
    """Ground-truth check: does the diagnosis explain the actual discrepancy?"""
    actual_gap = invoice["total"] - invoice["expected_total"]
    lines = [l["desc"] for l in invoice["lines"]]
    if diagnosis == "duplicate onboarding line":
        return lines.count("Onboarding (one-time)") == 2 and actual_gap == 1400
    if diagnosis == "7 extra seats":
        return any("31 seats" in d for d in lines)      # it is not there
    return False


best, agreement, _ = self_consistency(15)
assert best == "duplicate onboarding line" and agreement >= 0.5
assert verify(best, STRIPE_INVOICE)
assert not verify("7 extra seats", STRIPE_INVOICE)   # same $1,400, wrong story
print(f"\nverified: {best!r} explains the ${STRIPE_INVOICE['total'] - STRIPE_INVOICE['expected_total']} gap")
```

Two things to take from this. Agreement is a usable confidence signal: at `k=1` you have none; at `k=15` a strong majority for one answer and a scatter for the rest says something. But "7 extra seats" and "duplicate onboarding line" both predict a $1,400 overcharge. A vote on the number alone would not separate them. The verifier does, because it checks the story against the invoice lines rather than checking the arithmetic against itself. That is the difference between self-consistency and verification, and it is the difference that matters in production.

### Tree-of-thought: propose, score, prune

Self-consistency samples whole answers independently. Tree-of-thought (Yao et al., 2023) builds the answer incrementally. At each depth the model proposes several next steps, an evaluator scores each partial path, the search keeps the best `b` (the beam width), and expands only those. Dead ends are pruned before they consume more budget.

Here the "thoughts" are diagnostic checks. The evaluator is deliberately cheap and deterministic, which is how you should build the first version of any ToT evaluator: a rule, a test, or a tool lookup, not another model call.

```{code-cell} python
CHECKS = {
    # check -> (evidence it would produce from the tools, does it explain the gap?)
    "compare seat count CRM vs Stripe": ("CRM 24 seats, Stripe 24 seats", False),
    "list invoice line items": ("3 lines: seats 4800, onboarding 1400, onboarding 1400", True),
    "check tax rate": ("tax 0%, none applied", False),
    "check plan tier": ("Business tier on both systems", False),
    "diff line items against contract": ("contract: onboarding billed once", True),
}


def propose(path):
    """Candidate next thoughts: every check not yet on this path."""
    return [path + [c] for c in CHECKS if c not in path]


def evaluate(path):
    """Cheap evaluator: how many steps on this path produced explanatory evidence."""
    return sum(CHECKS[c][1] for c in path) / len(path)


def tree_of_thought(beam_width=2, max_depth=3, budget=12):
    frontier, evaluated, log = [[]], 0, []
    for depth in range(max_depth):
        candidates = [p for path in frontier for p in propose(path)]
        candidates = candidates[: budget - evaluated]           # hard budget on evaluations
        evaluated += len(candidates)
        scored = sorted(((evaluate(p), p) for p in candidates), key=lambda s: -s[0])
        frontier = [p for score, p in scored[:beam_width] if score > 0]   # prune dead ends
        log.append({"depth": depth, "proposed": len(candidates), "kept": len(frontier)})
        if not frontier or evaluated >= budget:
            break
    return (frontier[0] if frontier else None), evaluated, log


best_path, evaluated, log = tree_of_thought()
for row in log:
    print(row)
print("best path:", best_path)
print(f"evaluated {evaluated} of {len(CHECKS) ** 2} possible two-step paths")
assert best_path is not None and "list invoice line items" in best_path
assert evaluated <= 12
assert all(CHECKS[c][1] for c in best_path)      # every kept step was explanatory
```

The pruning is the point. Depth 0 proposes five checks and keeps two. Depth 1 expands only those two. A brute-force enumeration would evaluate 25 two-step paths; the beam evaluates far fewer and still lands on "list the line items, then diff them against the contract." In a real system every evaluation is a model call or a tool call, so the budget is money.

Also notice `budget=12` sits outside the search logic and is enforced by the code, not requested from the model. The same rule as every loop in this course.

### ReAct with a verification step

ReAct (Reason, Act, Observe) is what the base agent loop from 00 already does. What the reasoning chapter of the book adds is the observation that grounding each reasoning step in a tool result is itself a reasoning technique: the model cannot hallucinate the seat count if it just read it from Stripe.

The from-scratch loop below adds one node the standard loop lacks: a `verify` step that checks the model's final claim against the tool evidence already gathered. If the claim references a number that no tool returned, the loop sends it back with a precise objection. This is verification of outcome, not of prose.

```{code-cell} python
import re
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool


@tool
def get_invoice(customer: str) -> str:
    """Return the Stripe invoice line items and total for a customer."""
    return "lines: [24 seats x $200 = 4800, Onboarding 1400, Onboarding 1400]; total 7600"


@tool
def get_contract(customer: str) -> str:
    """Return the contracted charges for a customer from the CRM."""
    return "24 seats at $200/mo; onboarding fee 1400 billed once"


def numbers_in(text):
    return {int(n.replace(",", "")) for n in re.findall(r"\d[\d,]*", text)}


def verify_claim(claim, observations):
    """Every number in the claim must be observed, or derivable as a sum/difference of observed numbers."""
    observed = set().union(*(numbers_in(o) for o in observations)) if observations else set()
    derived = {a + b for a in observed for b in observed} | {abs(a - b) for a in observed for b in observed}
    return numbers_in(claim) - observed - derived


def react_with_verification(model, tools, question, max_steps=8):
    tools_by_name = {t.name: t for t in tools}
    messages, observations = [HumanMessage(question)], []
    for _ in range(max_steps):
        ai = model.invoke(messages)
        messages.append(ai)
        if ai.tool_calls:
            for call in ai.tool_calls:                               # act
                obs = str(tools_by_name[call["name"]].invoke(call["args"]))
                observations.append(obs)                             # observe
                messages.append(ToolMessage(content=obs, tool_call_id=call["id"]))
            continue
        unsupported = verify_claim(ai.content, observations)        # verify outcome
        if not unsupported:
            return ai.content, messages
        messages.append(HumanMessage(                                # precise objection
            f"Your answer cites {sorted(unsupported)} but no tool returned those values. "
            "Re-check the evidence and answer again."))
    raise RuntimeError("no verified answer within budget")


model = scripted(
    [("get_invoice", {"customer": "Acme"})],
    "Acme was billed for 31 seats; the extra 7 seats explain the $1,400 gap.",   # unsupported
    [("get_contract", {"customer": "Acme"})],
    "The onboarding fee of 1400 appears twice on the invoice but the contract bills it once; "
    "total 7600 should be 6200.",
).bind_tools([get_invoice, get_contract])

answer, transcript = react_with_verification(model, [get_invoice, get_contract],
                                             "Why is Acme's invoice wrong?")
for m in transcript:
    print(f"{type(m).__name__:13s} {str(m.content or m.tool_calls)[:90]}")
assert "twice" in answer
assert any("cites" in str(m.content) for m in transcript)   # the objection was raised
assert verify_claim(answer, [get_invoice.invoke({"customer": "Acme"}),
                             get_contract.invoke({"customer": "Acme"})]) == set()
```

The scripted model does exactly what the real one did in the opening scenario: it produces a confident seat-count story. The verifier catches 31, which no tool returned and which is not a sum or difference of observed values, and the objection is specific enough for the model to recover. The corrected total 6200 passes because it is derivable as 7600 minus 1400; a verifier that rejected every computed value would reject every useful answer. Read the transcript: the objection is a `HumanMessage`, the model's second attempt cites only observed numbers, and the loop ends.

This verifier is crude. In production it becomes a grader: an arithmetic check, a schema validation, a unit test, a second model with the evidence and a rubric. The shape is the same.

## How LangGraph does it

Everything above is control flow over model calls, which is exactly what LangGraph is for. The graph version of ReAct-with-verification adds a `verify` node after the agent decides it is done, and a conditional edge that either ends or routes back to the agent with the objection. Debate fits the same graph with a `Send` fan-out to two proposer nodes and a judge, which we will sketch after the code.

```{code-cell} python
import operator
from typing import Annotated
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode


class ReasoningState(MessagesState):
    observations: Annotated[list[str], operator.add]   # append-only evidence log
    objections: int                                    # how many times verify pushed back


model = scripted(
    [("get_invoice", {"customer": "Acme"})],
    "Acme was billed for 31 seats; the extra 7 seats explain the $1,400 gap.",
    [("get_contract", {"customer": "Acme"})],
    "The onboarding fee of 1400 appears twice on the invoice but the contract bills it once; "
    "total 7600 should be 6200.",
).bind_tools([get_invoice, get_contract])


def agent(state: ReasoningState):
    return {"messages": [model.invoke(state["messages"])]}


def record(state: ReasoningState):
    """Copy the trailing ToolMessages (this step's observations) into the evidence log."""
    new = []
    for m in reversed(state["messages"]):
        if not isinstance(m, ToolMessage):
            break
        new.append(m.content)
    return {"observations": new}


def verify(state: ReasoningState):
    claim = state["messages"][-1].content
    unsupported = verify_claim(claim, state["observations"])
    if not unsupported:
        return {}
    return {"objections": state.get("objections", 0) + 1,
            "messages": [HumanMessage(f"Your answer cites {sorted(unsupported)} but no tool "
                                      "returned those values. Re-check the evidence.")]}


def after_agent(state: ReasoningState):
    return "tools" if state["messages"][-1].tool_calls else "verify"


def after_verify(state: ReasoningState):
    if isinstance(state["messages"][-1], HumanMessage):       # verify appended an objection
        return "agent" if state["objections"] < 3 else END    # bounded retries
    return END


builder = StateGraph(ReasoningState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode([get_invoice, get_contract]))
builder.add_node("record", record)
builder.add_node("verify", verify)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", after_agent, ["tools", "verify"])
builder.add_edge("tools", "record")
builder.add_edge("record", "agent")
builder.add_conditional_edges("verify", after_verify, ["agent", END])
graph = builder.compile()

result = graph.invoke({"messages": [HumanMessage("Why is Acme's invoice wrong?")],
                       "observations": [], "objections": 0})
print(result["messages"][-1].content)
print("objections raised:", result["objections"], "| evidence items:", len(result["observations"]))
assert result["objections"] == 1
assert "twice" in result["messages"][-1].content
assert len(result["observations"]) == 2
```

Map the primitives back to the from-scratch loop:

| From scratch | LangGraph | What it is doing |
|---|---|---|
| `observations` list | `observations` key with `operator.add` reducer | append-only evidence log across steps |
| `if ai.tool_calls: ... continue` | `after_agent` conditional edge | route to tools or to verification |
| `verify_claim` then objection | `verify` node writing a `HumanMessage` | same check, same objection |
| `for _ in range(max_steps)` | `objections < 3` in `after_verify`, plus recursion limit | bounded retries |

```text
START -> agent --(tool calls)--> tools -> record -> agent
           |
           +--(final text)--> verify --(unsupported numbers, objections<3)--> agent
                                 |
                                 +--(supported, or budget spent)--> END
```

Why a separate `record` node rather than reading `ToolMessage`s directly in `verify`? Because in the next lesson's variants the evidence log gets compacted, offloaded, or shared with a judge, and a typed key with an append reducer is the thing you can reason about. The message list is the transcript; the evidence log is the verifier's input. Keeping them separate is a small instance of the context-engineering habit that 34. Context Engineering makes explicit.

### Debate as a graph

Debate (or "multi-agent reasoning" in the book's terms) is two or more proposers who answer independently, followed by a judge who sees the proposals and the evidence. The graph shape is a fan-out and a fan-in:

```text
START -> gather_evidence -+-> proposer_A (system prompt: argue the billing-config side) -+
                          |                                                             +-> judge -> END
                          +-> proposer_B (system prompt: argue the usage side) ---------+
```

In LangGraph the fan-out is a conditional edge returning `Send("proposer", {...})` once per proposer, and the fan-in is a node that runs after both because their outputs land in a key with an `operator.add` reducer. 03. Parallelization builds that exact machinery and 26. Map-Reduce reuses it, so the code is not repeated here.

Two design rules make debate worth its cost. First, the judge gets the evidence, not just the arguments, or it will pick the more fluent proposer. Second, the proposers must be genuinely independent: different system prompts, different tool subsets, or different models. Two copies of the same model with the same prompt do not debate; they agree, confidently.

### The 2026 version: a reasoning model plus a grader

```{code-cell} python live
import os
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

real_model = init_chat_model(os.environ.get("COURSE_MODEL", "openai:gpt-5.6"))

# No "think step by step". State the task, the constraints, and what done means.
SYSTEM = (
    "You investigate billing complaints for the support team. Use the tools to gather "
    "evidence before concluding. Done means: a one-paragraph diagnosis in which every "
    "number you cite appears in a tool result, followed by the corrected total."
)
agent = create_agent(real_model, tools=[get_invoice, get_contract], system_prompt=SYSTEM)
out = agent.invoke({"messages": [HumanMessage("Why is Acme's invoice wrong?")]})
final = out["messages"][-1].content
observations = [m.content for m in out["messages"] if isinstance(m, ToolMessage)]

# Verify the outcome, not the reasoning text.
unsupported = verify_claim(final, observations)
print(final)
print("unsupported numbers:", sorted(unsupported) or "none")

# Self-consistency against a real model: sample three times, compare corrected totals.
totals = []
for _ in range(3):
    o = agent.invoke({"messages": [HumanMessage("Why is Acme's invoice wrong? End with 'Corrected total: $N'.")]})
    m = re.search(r"Corrected total: \$?([\d,]+)", o["messages"][-1].content)
    totals.append(m.group(1).replace(",", "") if m else None)
print("sampled corrected totals:", totals)
```

The live cell shows the current practice in one place: a task-shaped system prompt with a definition of done, a real model doing its own decomposition, and a programmatic verifier over the outcome. The self-consistency loop at the end is still useful, but in 2026 its main job is measurement: if three samples disagree on the corrected total, that is a signal to escalate, not to average.

## What goes wrong in production

1. **Reasoning tokens blow the budget.** Symptom: latency triples and the bill doubles after switching to a reasoning model. Cause: the model's internal chain-of-thought is billed as output and can run to thousands of tokens per call, and self-consistency multiplies it by `k`. Fix: treat reasoning effort as a per-call knob (low for routing and extraction, high for the diagnosis step), cap `k`, and never put a reasoning model in a hot loop that a classifier could handle. 16. Resource-Aware Optimization covers the budget machinery.

2. **Self-consistency converges on a confident wrong answer.** Symptom: nine of ten samples agree and they are all wrong. Cause: a shared bias (the "seat count" prior) makes the wrong answer the mode; voting measures agreement, not truth. Fix: vote on verifiable facts, not stories, and add a ground-truth verifier where one exists. Treat unanimous agreement on an unverifiable claim as a warning, not as confidence.

3. **The ToT evaluator is another unverified model call.** Symptom: the search keeps paths that feel promising and prunes the right one. Cause: the evaluator is the same model scoring its own proposals; the reasoning-model faithfulness result says its stated confidence is weak evidence. Fix: make the evaluator cheap and external first (a rule, a test, a tool lookup). Only add a model-based evaluator when you can measure its agreement with outcomes.

4. **Reading the chain-of-thought as an audit log.** Symptom: an incident review concludes "the model explained its reasoning and it looked correct." Cause: the visible reasoning omitted the actual cause of the decision, which the faithfulness study shows happens most of the time when a hint drives the answer. Fix: log tool calls, arguments, and results as the audit trail. The reasoning text goes in the trace as context, not as evidence. 32. Observability covers what to record.

5. **Verification objections loop forever.** Symptom: the graph bounces between `agent` and `verify` until the recursion limit. Cause: the objection is vague ("that is not supported") or the verifier is stricter than the tools allow (a total the model must compute is never literally in an observation). Fix: bound the objections in code, make the objection name the exact unsupported items, and let the verifier accept derived values it can recompute. If the bound is hit, escalate with the transcript attached; do not return the last attempt as if it were verified.

6. **Debate between clones.** Symptom: two "independent" proposers agree in 98 percent of cases and the judge rubber-stamps. Cause: same model, same prompt, same tools. Fix: enforce diversity (different framing, tool subsets, or models) and give the judge the raw evidence. Measure how often the judge overrules; if it never does, the debate is decoration.

7. **Non-determinism breaks regression tests.** Symptom: the eval suite flips between passing and failing with no code change. Cause: sampling at temperature for self-consistency, plus reasoning models whose output varies run to run. Fix: evaluate outcomes with tolerances (the corrected total, the named line item) rather than exact text, run each case several times and report pass rate, and pin the graders. 22. Evaluation Reliability treats this in depth.

## What changed by 2026

- **The model decomposes; you define done.** The reasoning models from OpenAI, Anthropic, Google, and the open-weight labs (DeepSeek-R1 showed that reinforcement learning on verifiable rewards alone produces self-verification and reflection) do their own chain-of-thought. Both OpenAI's reasoning guidance and Anthropic's prompting guidance now say: give the task, constraints, and output format; for agentic work, say what counts as done and how the model should verify its work; do not prefix "think step by step." Reasoning effort is a tuning knob, not a prompt trick.

- **Chain-of-thought is signal, not evidence.** The faithfulness finding moved verification from optional to required. The strong engineering position is: the reasoning text can help you debug, but correctness is established by tests, tool checks, graders, and evals over outcomes. This is why the `verify` node exists in this lesson and why the harness guidance from Anthropic separates generation from evaluation: a generator grading its own work is unreliable.

- **Test-time compute is a budget line.** Self-consistency, best-of-N with a verifier, and tree search are now understood as points on the same curve (more inference compute, higher accuracy), and providers expose effort levels that trade the same way. The engineering question became "where in the graph is the extra compute worth it" rather than "which prompt trick."

- **ReAct is the loop, not a prompt format.** The original ReAct paper's "Thought: / Action: / Observation:" text format has disappeared into native tool calling. The pattern survives as the agent loop itself. When an interviewer asks about ReAct, they are asking about grounding reasoning in observations, not about the string format.

- **Verification moved into the harness.** LangChain's `create_agent` middleware, Deep Agents' subagents, and Anthropic's harness guidance all encourage a separate evaluator step with gradable criteria. Current docs describe rubric-style middleware and evaluator roles; the exact APIs move quickly, so learn the shape (generate, then grade against a stated contract) and check the installed version for names.

## Interview questions

**Q1. Explain the ReAct pattern and what it solves over chain-of-thought alone.** (interview repository: Agents and Tool Use)

Strong: Chain-of-thought lets the model reason in text, but every step is generated from the model's own beliefs, so a wrong premise propagates. ReAct interleaves reasoning with actions: the model reasons about what it needs, calls a tool, observes the real result, and reasons again. That grounds each step in evidence, so the model cannot hallucinate a seat count it just read from Stripe. Structurally it is the agent loop: reason, act, observe, repeat, with termination when the model stops calling tools. Today the "Thought/Action/Observation" text format is gone; native tool calling carries the same pattern. The thing ReAct does not solve by itself is verifying the final claim, so production loops add an explicit check of the answer against the gathered observations.

Weak: "ReAct is a prompt format where the model writes Thought, Action, Observation."

**Q2. When would you use self-consistency, and what is its failure mode?**

Strong: Self-consistency samples the same problem several times and takes the majority answer. It helps when errors are diverse and the correct answer is the mode, typically for short verifiable outputs like a number or a class label. Its cost is linear in the number of samples, including reasoning tokens. Its failure mode is correlated error: if the model has a shared bias, the wrong answer becomes the mode and high agreement looks like high confidence. So I vote on verifiable facts, not narratives, add a ground-truth check where one exists, and treat unanimous agreement on an unverifiable claim as a flag rather than a green light. In practice I also use the disagreement rate as an escalation trigger.

Weak: "Run it five times and pick the most common answer."

**Q3. You switched to a reasoning model. Should your prompts still say "think step by step"?**

Strong: No. Reasoning models are trained to produce their own chain-of-thought, and both major vendors' guidance now says to state the task, constraints, output format, and definition of done, and to avoid chain-of-thought prefixes. Prescribing the steps can constrain a model that would have found a better decomposition. What I do instead is invest in the "done" condition: what the output must contain, how it should be verified, and what to do when the evidence is insufficient. And I set reasoning effort per call, low for routing and high for the hard diagnosis step, because reasoning tokens are billed and slow.

Weak: "Yes, chain-of-thought always improves accuracy."

**Q4. Can you use the model's chain-of-thought as an audit log for a regulated decision?**

Strong: No. The 2025 faithfulness research showed that when a hint changed a reasoning model's answer, the visible chain-of-thought mentioned the hint only about a quarter to a third of the time, and in a reward-hacking setup the models admitted the exploit in under two percent of cases. The reasoning text is a useful debugging signal but it is not a reliable account of why the model decided. For audit I log what is actually observable: tool calls, arguments, results, the final claim, and the verifier's checks against those results. If a decision needs a justification, I generate it from the evidence trail with a separate step, and I verify the outcome with tests or graders, not by reading the prose.

Weak: "Yes, that is the whole point of chain-of-thought: the model shows its reasoning."

**Q5. Explain tree-of-thought and when it is worth its cost.**

Strong: Tree-of-thought treats reasoning as search. At each step the model proposes several partial next steps, an evaluator scores each partial path, and the search keeps the best few and expands only those, with a hard budget on total evaluations. It is worth it when the problem has an early branching decision that is hard to undo, when partial progress can be scored cheaply, and when a single-pass answer is measurably unreliable. It is not worth it for problems that are linear or where the evaluator would just be the same model guessing. The key engineering decision is the evaluator: I start with a rule, a test, or a tool lookup, and only add a model-based evaluator once I can measure its agreement with outcomes.

Weak: "Tree-of-thought lets the model explore multiple reasoning paths and pick the best."

**Q6. How do you design a verification step for an agent's final answer?**

Strong: I define what "done" means as a checkable contract before building the agent, then implement the check as code where possible: every number cited must appear in a tool observation or be recomputable from them, the output must match a schema, a test must pass. Where a model grader is needed, it gets the evidence and a rubric, not just the answer, and it is a different model or prompt from the generator, because self-grading is unreliable. When verification fails, the objection is precise enough for the model to correct itself, retries are bounded in code, and on exhaustion the run escalates with the transcript rather than returning the last attempt. I also measure the verifier itself against human labels, because a verifier that accepts everything is worse than none.

Weak: "Ask the model to double-check its answer."

**Q7. What is the difference between debate and self-consistency, and when does debate break down?**

Strong: Self-consistency is independent samples plus a vote; debate is independent proposers whose arguments are then judged by a third party with the evidence. Debate can beat voting when the answer needs justification, because the judge can reward the better-supported argument rather than the more common one. It breaks down when the proposers are not independent, two copies of the same model with the same prompt agree rather than argue, and when the judge sees only the arguments, because then fluency wins. So I enforce diversity in prompts, tool subsets, or models, give the judge the raw evidence, and track how often the judge overrules; a judge that never overrules means the debate is decoration.

Weak: "Debate is when multiple agents discuss until they agree."

## Build it yourself

Extend the LangGraph verification loop into a small best-of-N-with-verifier system.

1. Make the `agent` node run `n=3` scripted candidates (three separate scripted models with different final claims: one unsupported, two supported but with different corrected totals) and store all three claims in a `candidates` key with an `operator.add` reducer.
2. Replace `verify` with a `select` node that drops candidates with unsupported numbers, then picks the candidate whose corrected total can be recomputed from the observations (`7600 - 1400 = 6200`).
3. Add an `escalate` node reached when zero candidates survive; it should return a message starting with `ESCALATE:` and include the count of rejected candidates.
4. Record a per-run trace dictionary: number of candidates, number rejected, whether escalation happened.

Acceptance checks:

- With the three scripted candidates, the surviving answer contains `6200` and the trace shows exactly one rejection.
- With all three candidates scripted to cite `31 seats`, the graph ends at `escalate` and the final message starts with `ESCALATE:`.
- The `observations` key contains exactly two items in both cases (no duplicate tool calls).
- No run exceeds 3 objections or hits the recursion limit.

## Key takeaways

- Every reasoning technique is control flow that trades tokens for reliability: more steps, more samples, or more branches, arranged so errors get caught.
- Self-consistency measures agreement, not truth; vote on verifiable facts and add a ground-truth check whenever one exists.
- Tree-of-thought is search with a budget; the evaluator is the design decision, and it should be cheap and external before it is a model.
- ReAct is the agent loop: grounding each reasoning step in a tool observation is what stops fabricated premises from propagating.
- Reasoning models decompose on their own; define the task and what done means, do not prescribe "think step by step."
- A model's chain-of-thought is signal for debugging, not an audit log; verify outcomes with tools, tests, and graders.
- Bound retries and objections in code, make objections precise, and escalate with the transcript when the bound is hit.
