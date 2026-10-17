# 11. Goal Setting and Monitoring

*Track A (agent engineering). Book: Chapter 11 (Goal Setting and Monitoring). Prerequisites: 00. What an agent actually is, 06. Planning, 12. Exception Handling and Recovery.*

An agent loop has one built-in exit: the model stops calling tools. That exit means "the model thinks it is finished." It does not mean the work is done. This lesson is about the gap between those two statements. We turn a vague request into a goal the software can check, record evidence from tools rather than claims from the model, measure progress after every step, detect stalls and drift, and stop for the right reason with the right report. The book frames this as goal setting and monitoring; production teams call it "defining done," and in 2026 it is the most repeated piece of vendor guidance for agentic work.

## The problem

The ops lead posts in Slack:

> "Acme says they were charged twice for September. Sort it out."

Run the plain agent loop from 00. What an agent actually is. Three different runs, three different failures:

```text
Run 1: premature done
  search_slack("Acme charged twice")      -> complaint found
  stripe_invoice("Acme")                  -> ch_1 $120, ch_2 $120 (duplicate)
  model: "I found the duplicate charge ch_2. Resolved."        <- no refund, no reply
  loop exits: no tool calls. Status: "success".

Run 2: stall
  search_slack("Acme")  search_slack("Acme")  search_slack("Acme") ... x 40
  loop exits: recursion limit. Status: "error". Cost: 40 model calls, zero information.

Run 3: drift
  ...investigates Acme's plan, then its usage, then writes a churn-risk memo...
  model: "Here is a detailed analysis of Acme's account health."
  loop exits: no tool calls. Status: "success". The customer still paid twice.
```

All three runs are "successful" loops. Run 1 stopped too early, Run 2 never made progress, and Run 3 did real work on the wrong thing. The interview repository has a question for exactly this, reported at Cognition: "Your agent drifts after a long run and confidently works on the wrong thing. Diagnose it." The diagnosis is almost always the same: the only definition of done was the model's opinion.

The fix is to give the software its own view of the goal:

```text
                 ┌──────────────── GOAL SPEC (written before the run) ───────────────┐
                 │ objective     resolve Acme's double charge                        │
                 │ criteria      root cause found; duplicate refunded exactly;       │
                 │               customer told in #acme-support with the refund id   │
                 │ constraints   refund <= $500 without approval; only the duplicate │
                 │ done          all criteria pass AND no constraint violated        │
                 │ verification  each criterion checked from tool evidence           │
                 │ budgets       12 steps, 3 steps without progress, 2 false "done"s │
                 └───────────────────────────────────────────────────────────────────┘
                                             │
   agent ──► tools ──► evidence ──► GOAL CHECK ──► done      (verified)
     ▲                                  │     ├──► stalled   (no new evidence in k steps)
     │                                  │     ├──► violated  (constraint broken: stop now)
     └──── feedback: "unmet: X, Y" ◄────┘     └──► budget    (steps / time / tokens / $)
```

The model still chooses the actions. The software decides whether the goal is met.

## Build it without a framework

### A goal the software can check

A goal spec has five parts, and each one answers a question the loop will otherwise answer badly:

| Part | Question | Without it |
|---|---|---|
| Objective | What outcome does the user want? | Drift toward adjacent, easier work |
| Success criteria | What observable facts mean success? | Premature "done" |
| Constraints | What must never happen on the way? | Success by forbidden means |
| Definition of done | Which combination of criteria ends the run? | Endless polishing, or stopping at 2 of 3 |
| Verification | How is each criterion checked, by whom? | The model grades its own homework |

The key design choice is where criteria read from. They read from **evidence**: structured facts recorded by tools when they execute. Not from the model's final message. A refund criterion checks that a refund tool returned a refund id for the right charge and amount. It does not check that the model wrote the word "refunded."

```{code-cell} python
from dataclasses import dataclass, field
from typing import Callable

Evidence = dict   # facts recorded by tools, e.g. {"duplicate_charge": {...}, "refund": {...}}


@dataclass(frozen=True)
class Criterion:
    name: str
    check: Callable[[Evidence], bool]     # pure function of evidence: testable, replayable


@dataclass(frozen=True)
class Constraint:
    name: str
    violated: Callable[[Evidence], bool]


@dataclass(frozen=True)
class GoalSpec:
    objective: str
    criteria: tuple
    constraints: tuple
    max_steps: int = 12
    stall_after: int = 3          # consecutive steps that add no new evidence
    max_false_done: int = 2       # times the model may claim done while criteria fail

    def evaluate(self, ev: Evidence) -> dict:
        met = [c.name for c in self.criteria if c.check(ev)]
        return {
            "met": met,
            "unmet": [c.name for c in self.criteria if c.name not in met],
            "violated": [c.name for c in self.constraints if c.violated(ev)],
            "progress": len(met) / len(self.criteria),
        }


def _dup(ev):
    return ev.get("duplicate_charge", {})


acme_goal = GoalSpec(
    objective="Resolve Acme's September double charge",
    criteria=(
        Criterion("root_cause_identified", lambda ev: "duplicate_charge" in ev),
        Criterion("duplicate_refunded_exactly", lambda ev: "refund" in ev
                  and ev["refund"]["charge"] == _dup(ev).get("id")
                  and ev["refund"]["amount"] == _dup(ev).get("amount")),
        Criterion("customer_notified_with_refund_id", lambda ev: "notified" in ev
                  and ev["notified"]["channel"] == "#acme-support"
                  and ev.get("refund", {}).get("id", "?") in ev["notified"]["text"]),
    ),
    constraints=(
        Constraint("refund_over_500_without_approval",
                   lambda ev: ev.get("refund", {}).get("amount", 0) > 50_000),
        Constraint("refunded_a_non_duplicate_charge",
                   lambda ev: "refund" in ev and ev["refund"]["charge"] != _dup(ev).get("id")),
    ),
)

snapshots = {
    "start": {},
    "after diagnosis": {"duplicate_charge": {"id": "ch_2", "amount": 12_000}},
    "wrong refund": {"duplicate_charge": {"id": "ch_2", "amount": 12_000},
                     "refund": {"id": "re_9", "charge": "ch_1", "amount": 12_000}},
    "all done": {"duplicate_charge": {"id": "ch_2", "amount": 12_000},
                 "refund": {"id": "re_1", "charge": "ch_2", "amount": 12_000},
                 "notified": {"channel": "#acme-support", "text": "Refunded ch_2 (re_1)."}},
}
for name, ev in snapshots.items():
    r = acme_goal.evaluate(ev)
    print(f"{name:16s} progress={r['progress']:.2f} unmet={r['unmet']} violated={r['violated']}")

assert acme_goal.evaluate({})["progress"] == 0
assert acme_goal.evaluate(snapshots["after diagnosis"])["met"] == ["root_cause_identified"]
assert acme_goal.evaluate(snapshots["wrong refund"])["violated"] == ["refunded_a_non_duplicate_charge"]
assert acme_goal.evaluate(snapshots["all done"])["unmet"] == []
```

Notice the "wrong refund" row. Progress would count it as partial success if we only counted criteria. The constraint catches it as a violation. That distinction matters: criteria say what to achieve, constraints say what must never happen on the way, and a violated constraint ends the run immediately, because the next step cannot undo a refund.

### A monitored loop

Now the loop. Compared with the loop in 00, three things change:

```text
1. tools return (text for the model, evidence for the monitor)   two audiences, two channels
2. after every step the monitor evaluates the goal               progress is measured, not guessed
3. termination has named reasons                                 done / violated / stalled /
                                                                 false_done_limit / budget
```

The scripted model below stands in for a real model so the control flow is the object of study. The script deliberately claims "done" too early, the way real models do, and a second script loops.

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


from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool


# content_and_artifact: the string goes to the model, the artifact goes to the monitor.
@tool(response_format="content_and_artifact")
def search_slack(query: str):
    """Search Slack messages. Returns matching messages."""
    return "#acme-support: 'We were charged twice for September!'", {"complaint": "double charge"}


@tool(response_format="content_and_artifact")
def stripe_invoice(customer: str):
    """List a customer's recent Stripe charges."""
    return ("ch_1 $120.00 Sep invoice; ch_2 $120.00 Sep invoice (same invoice id)",
            {"duplicate_charge": {"id": "ch_2", "amount": 12_000}})


@tool(response_format="content_and_artifact")
def stripe_refund(charge_id: str, amount_cents: int):
    """Refund a Stripe charge. Amount in cents."""
    return (f"refund re_1 created for {charge_id}",
            {"refund": {"id": "re_1", "charge": charge_id, "amount": amount_cents}})


@tool(response_format="content_and_artifact")
def post_slack(channel: str, text: str):
    """Post a message to a Slack channel."""
    return f"posted to {channel}", {"notified": {"channel": channel, "text": text}}


TOOLS = {t.name: t for t in [search_slack, stripe_invoice, stripe_refund, post_slack]}


def run_monitored(model, goal: GoalSpec, request: str):
    messages, evidence, trace = [HumanMessage(request)], {}, []
    no_new, false_done = 0, 0
    for step in range(1, goal.max_steps + 1):
        ai = model.invoke(messages)
        messages.append(ai)
        before = set(evidence)
        for call in ai.tool_calls:
            msg = TOOLS[call["name"]].invoke({**call, "type": "tool_call"})   # -> ToolMessage
            messages.append(msg)
            evidence.update(msg.artifact or {})
        report = goal.evaluate(evidence)
        no_new = 0 if set(evidence) - before else no_new + 1
        trace.append((step, [c["name"] for c in ai.tool_calls] or "claims done",
                      report["progress"], no_new))

        if report["violated"]:
            return "violated", report, trace          # irreversible: stop, escalate
        if not report["unmet"]:
            return "done", report, trace              # verified by evidence, not by the model
        if not ai.tool_calls:                         # the model thinks it is done; it is not
            false_done += 1
            if false_done >= goal.max_false_done:
                return "false_done_limit", report, trace
            messages.append(HumanMessage(
                f"Not done. Unmet criteria: {report['unmet']}. Continue toward: {goal.objective}."))
        if no_new >= goal.stall_after:
            return "stalled", report, trace
    return "budget_exhausted", report, trace


good_model = scripted(
    [("search_slack", {"query": "Acme charged twice"})],
    [("stripe_invoice", {"customer": "Acme"})],
    "I found the duplicate charge ch_2. Resolved.",                     # premature
    [("stripe_refund", {"charge_id": "ch_2", "amount_cents": 12_000})],
    [("post_slack", {"channel": "#acme-support", "text": "Refunded duplicate ch_2 (re_1)."})],
)
status, report, trace = run_monitored(good_model, acme_goal, "Acme was charged twice. Sort it out.")
for row in trace:
    print(row)
print("->", status, report["met"])
assert status == "done" and trace[2][1] == "claims done" and len(trace) == 5

looping_model = scripted(*[[("search_slack", {"query": "Acme"})]] * 4)
status, report, trace = run_monitored(looping_model, acme_goal, "Acme was charged twice.")
print("->", status, "after", len(trace), "steps; progress", report["progress"])
assert status == "stalled" and len(trace) == 4    # 1 useful search + 3 with nothing new
```

Two things to notice in the trace.

First, step 3. The model said "Resolved." The monitor evaluated the evidence, found two unmet criteria, and fed them back as a message. The model then did the remaining work. Without the check, Run 1 from the problem section is what ships.

Second, the looping run ends after four steps with the reason `stalled`, not after the recursion limit with an error. The step budget is a *safety net*; the stall detector is the *monitor*. A stalled run produces a report ("progress 0.00, no new evidence for 3 steps, unmet: all three") that a human or a replanner can act on.

### What counts as progress

"No new evidence" is the simplest stall signal that works. Real systems combine several:

```text
signal                         catches                           cost
─────────────────────────────  ────────────────────────────────  ─────────────
criteria met, over time        slow or zero goal progress        free (you have it)
new evidence keys              loops that learn nothing          free
repeated call fingerprints     identical tool+args retried       hash per call
off-goal tool use              drift (tools unrelated to goal)   a relevance map per goal
token / $ per criterion met    expensive progress                budget ledger (16.)
LLM judge on the plan          subtle drift in long tasks        one extra model call
```

Stall and drift get different responses. A stall usually means the current approach is stuck: replan once with the stall report as input (06. Planning), and escalate if the replan also stalls. Drift means the agent lost the objective: re-inject the goal spec and the unmet criteria, which long-running harnesses do anyway after compaction. A violation means stop, because a constraint exists precisely for things you cannot take back.

## How LangGraph does it

In LangGraph the monitor becomes a node. The graph looks like the agent graph from 00 with one addition: every path back to the model goes through `goal_check`, and `goal_check` alone decides whether the run ends.

```text
START ──► agent ──(tool calls)──► tools ──► goal_check ──(continue)──► agent
            │                                   │
            └──(no tool calls: "done?")─────────┘──(done | stalled | violated | ...)──► END
```

The mapping to the from-scratch loop:

| `run_monitored` | LangGraph |
|---|---|
| `evidence` dict updated from `msg.artifact` | state key `evidence` with a merge reducer, filled in `goal_check` from `ToolMessage.artifact` |
| `no_new`, `false_done` counters | state keys, returned as updates by `goal_check` |
| `goal.evaluate(evidence)` | the body of `goal_check` |
| `return "stalled", ...` | `goal_check` sets `status`; a conditional edge routes to `END` |
| feedback `HumanMessage` | `goal_check` appends it to `messages` |
| `range(goal.max_steps)` | a `steps` counter in state; the graph's `recursion_limit` (1000 by default in current docs) stays as a backstop |

The goal spec lives outside state because it is configuration, not something nodes change. Evidence lives in state because it must survive checkpoints: if the run pauses for approval or crashes, the resumed run must know that the refund already happened.

```{code-cell} python
import operator
from typing import Annotated
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode


def merge(left: dict, right: dict) -> dict:
    return {**left, **right}


class GoalState(MessagesState):
    evidence: Annotated[dict, merge]              # facts from tools, merged step by step
    progress: Annotated[list, operator.add]       # progress history: an append-only log
    steps: int
    no_new: int
    false_done: int
    status: str


def build_monitored_graph(model, goal: GoalSpec):
    def agent(state: GoalState):
        return {"messages": [model.invoke(state["messages"])], "steps": state.get("steps", 0) + 1}

    def after_agent(state: GoalState):
        return "tools" if state["messages"][-1].tool_calls else "goal_check"

    def goal_check(state: GoalState):
        # Harvest evidence from ToolMessages since the last AI turn, like `evidence.update`.
        new = {}
        for m in reversed(state["messages"]):
            if not isinstance(m, ToolMessage):
                break
            new.update(m.artifact or {})
        evidence = {**state.get("evidence", {}), **new}
        report = goal.evaluate(evidence)
        no_new = 0 if set(new) - set(state.get("evidence", {})) else state.get("no_new", 0) + 1
        update = {"evidence": new, "progress": [report["progress"]], "no_new": no_new}
        last = state["messages"][-1]
        claimed_done = isinstance(last, AIMessage) and not last.tool_calls   # arrived from agent

        if report["violated"]:
            return {**update, "status": f"violated:{report['violated'][0]}"}
        if not report["unmet"]:
            return {**update, "status": "done"}
        if claimed_done:
            false_done = state.get("false_done", 0) + 1
            if false_done >= goal.max_false_done:
                return {**update, "false_done": false_done, "status": "false_done_limit"}
            update |= {"false_done": false_done, "messages": [HumanMessage(
                f"Not done. Unmet criteria: {report['unmet']}. Objective: {goal.objective}.")]}
        if no_new >= goal.stall_after:
            return {**update, "status": "stalled"}
        if state.get("steps", 0) >= goal.max_steps:
            return {**update, "status": "budget_exhausted"}
        return {**update, "status": "running"}

    def route(state: GoalState):
        return "agent" if state["status"] == "running" else END

    b = StateGraph(GoalState)
    b.add_node("agent", agent)
    b.add_node("tools", ToolNode(list(TOOLS.values())))
    b.add_node("goal_check", goal_check)
    b.add_edge(START, "agent")
    b.add_conditional_edges("agent", after_agent, ["tools", "goal_check"])
    b.add_edge("tools", "goal_check")
    b.add_conditional_edges("goal_check", route, ["agent", END])
    return b.compile()


request = {"messages": [HumanMessage("Acme was charged twice for September. Sort it out.")]}

good = build_monitored_graph(scripted(
    [("search_slack", {"query": "Acme charged twice"})],
    [("stripe_invoice", {"customer": "Acme"})],
    "I found the duplicate charge ch_2. Resolved.",
    [("stripe_refund", {"charge_id": "ch_2", "amount_cents": 12_000})],
    [("post_slack", {"channel": "#acme-support", "text": "Refunded duplicate ch_2 (re_1)."})],
), acme_goal).invoke(request)
print("good run:", good["status"], "progress history", good["progress"])
assert good["status"] == "done" and good["progress"][-1] == 1.0
assert any(isinstance(m, HumanMessage) and "Unmet" in m.content for m in good["messages"])

wrong = build_monitored_graph(scripted(
    [("stripe_invoice", {"customer": "Acme"})],
    [("stripe_refund", {"charge_id": "ch_1", "amount_cents": 12_000})],   # the ORIGINAL charge
), acme_goal).invoke(request)
print("wrong refund:", wrong["status"], "after", wrong["steps"], "model calls")
assert wrong["status"] == "violated:refunded_a_non_duplicate_charge" and wrong["steps"] == 2

stuck = build_monitored_graph(scripted(*[[("search_slack", {"query": "Acme"})]] * 4),
                              acme_goal).invoke(request)
print("looping run:", stuck["status"], "progress history", stuck["progress"])
assert stuck["status"] == "stalled" and stuck["steps"] == 4
```

Three runs, three named outcomes. The good run finished only when the evidence said so, even though the model tried to stop at step 3. The wrong-refund run stopped immediately after the violating call; the graph did not give the model another turn to "fix" an irreversible action. The looping run ended as `stalled` after four model calls instead of burning the recursion limit.

Two primitives do the real work. The **reducer** on `evidence` (`merge`) and on `progress` (`operator.add`) is what `evidence.update(...)` and `trace.append(...)` were in the plain loop; declaring them in the state type is what lets a checkpointer persist and restore them. The **conditional edge** out of `goal_check` is the `return status` statements; making it the only path back to `agent` guarantees the monitor runs on every iteration. If you use `create_agent` instead of a custom graph, the same check fits in an `after_model` or `after_agent` middleware hook, and `ModelCallLimitMiddleware` and `ToolCallLimitMiddleware` provide the budget backstops.

### Where the goal spec comes from

Someone has to write the criteria. Three sources, in order of preference:

```text
1. the product        the ticket type "billing: double charge" has a fixed checklist
2. a planner step     a model drafts criteria from the request; a human or a rule
                      approves them before execution (a "contract" for the run)
3. the user           ask one clarifying question when the request has no checkable outcome
```

Anthropic's harness-design write-up for long-running application work uses the second pattern under the name "sprint contracts": the planner and evaluator agree on what done means before any code is written, and a separate evaluator grades the result against a rubric, because a generator praising its own work is unreliable. The live cell has the model draft a goal spec as structured output, which you would then validate before running.

```{code-cell} python live
import os
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model


class DraftGoal(BaseModel):
    objective: str
    success_criteria: list[str] = Field(description="Observable facts a tool result can confirm")
    constraints: list[str] = Field(description="Things that must never happen during the task")
    verification: list[str] = Field(description="For each criterion, which tool result proves it")


model = init_chat_model(os.environ.get("COURSE_MODEL", "openai:gpt-5.6"))
draft = model.with_structured_output(DraftGoal).invoke(
    "Write a goal contract for this support request. Tools available: search_slack, "
    "stripe_invoice, stripe_refund, post_slack. Request: 'Acme says they were charged "
    "twice for September. Sort it out.' Define what counts as done and how to verify it.")
print(draft.model_dump_json(indent=2))

live = build_monitored_graph(model.bind_tools(list(TOOLS.values())), acme_goal).invoke(
    {"messages": [HumanMessage("Acme was charged twice for September. Refund the duplicate "
                               "and tell them in #acme-support with the refund id.")]})
print(live["status"], live["progress"])
```

## What goes wrong in production

1. **The model is the verifier.** Symptom: high "success" rate on dashboards, angry customers. Cause: done means "the model stopped calling tools" or "the model said resolved." Fix: criteria evaluated from tool evidence by code, or by a separate evaluator with a rubric. Self-assessment can be a signal, never the gate.

2. **Proxy criteria.** Symptom: every ticket closes with a Slack reply, and half the replies are wrong. Cause: the criterion was "a message was posted," which is easy to measure and easy to satisfy. Fix: criteria that encode the user's actual outcome (the refund exists, for the right charge, for the right amount), and a periodic human review of "done" runs to find criteria that are being gamed. Goodhart's law applies to agents faster than to people.

3. **Only a step budget.** Symptom: stuck runs consume the entire recursion limit and end in an exception with no report. Cause: the step budget is the only terminator. Fix: stall detection on progress, named termination reasons, and a structured final report (met, unmet, evidence, reason) for every non-success exit so a human or replanner can continue.

4. **Evidence lost on resume.** Symptom: after a crash or an approval pause, the agent refunds again. Cause: progress was tracked in a local variable, not in checkpointed state. Fix: evidence and counters in graph state with reducers, persisted by the checkpointer; side-effecting tools idempotent (36. Durable Agent Orchestration).

5. **Constraints checked at the end.** Symptom: the final report says "violated: refunded the wrong charge," three steps after it happened. Cause: constraints evaluated only on completion. Fix: evaluate constraints after every tool step, and for the most dangerous ones evaluate *before* execution in the policy layer (39. Agent Identity and Governance). The monitor catches what the policy layer could not predict.

6. **Drift after compaction.** Symptom: a long task's later steps optimize for something adjacent. Cause: the goal was stated once in the first message and was summarized away. Fix: keep the goal spec and current unmet criteria in a pinned position (system prompt or a progress file) that compaction never removes, and re-inject unmet criteria with every feedback message.

7. **False-done loops.** Symptom: the model says "done," the monitor says "not done," forever. Cause: a criterion the model cannot satisfy with its tools, or feedback that does not say what is missing. Fix: specific feedback naming unmet criteria, a cap on false-done claims, and escalation that reports which criterion was unreachable.

## What changed by 2026

- **"Define done" became standard vendor guidance.** OpenAI's reasoning-model guidance tells you, for agentic work, to define what counts as done and how the model should verify its work, and to give explicit stop conditions instead of chain-of-thought prefixes. With reasoning models, the goal spec replaced "think step by step" as the prompt component that most affects results.
- **Generation and evaluation are separated.** Anthropic's harness-design guidance for long-running work uses a planner, a generator, and an evaluator, with contracts that define done before work starts, because self-evaluation is unreliable. A model's written reasoning is not a faithful audit log either, so verification leans on tests, tool results, and graders rather than on reading the model's explanation.
- **Progress lives in files and state, not in the transcript.** Long-running harnesses keep a feature list with pass/fail status and a progress file, plus git commits, so a fresh session can see exactly which criteria are met. Deep Agents' optional todo-list middleware is a lightweight version of the same idea.
- **Budgets and limits moved into middleware.** LangChain's `create_agent` ships model-call and tool-call limit middleware, and LangGraph's recursion limit is a high backstop (1000 super-steps by default in current docs) rather than the primary terminator. Goal checks run as nodes or `after_model` hooks.
- **Goal verification feeds evals.** The same criteria functions that gate a run in production become the grader in trajectory evals (37. Trajectory Evals and Statistics). A goal spec you can check at runtime is a test you can run offline.

## Interview questions

**Q1. How do you know when an agent is done?**

Strong: Not when the model stops calling tools; that only means the model thinks it is done. Before the run, I write a goal spec with checkable success criteria, constraints, and a definition of done. Tools record structured evidence as they execute, and a goal-check step after every iteration evaluates the criteria against that evidence. The run ends as "done" only when all required criteria pass and no constraint is violated. If the model claims done early, the check feeds back the unmet criteria; if it keeps claiming done, I cap it and escalate.

Weak: "When it returns a final answer."

**Q2. Your agent drifts after a long run and confidently works on the wrong thing. Diagnose it.** (reported at Cognition)

Strong: I would look at three things. First, whether the goal was ever stated as checkable criteria, or only as a first message that compaction later summarized away. Second, whether anything measured progress toward the goal during the run, or only step counts. Third, whether the drift started after a context reset or a large tool result displaced the objective. The fixes are a pinned goal spec and unmet-criteria list that survive compaction, a per-step goal check that notices when actions stop advancing any criterion, and off-goal detection that flags tool use unrelated to the objective. I would then turn the drifting trace into an eval case.

Weak: "Use a bigger context window."

**Q3. How do you detect that an agent is stuck?**

Strong: Measure progress per step and stop on its absence, not only on a step budget. The cheapest signals are no new evidence for k consecutive steps, repeated identical tool-call fingerprints, and criteria-met not increasing. When a stall is detected, the loop ends with a named reason and a report, then replans once with the stall report as input, and escalates if that stalls too. The step budget and recursion limit remain as backstops.

Weak: "Set max_iterations."

**Q4. What is the difference between a success criterion and a constraint?**

Strong: A criterion describes the outcome to reach, like "the duplicate charge is refunded exactly." A constraint describes something that must never happen on the way, like "never refund a charge that is not the duplicate." They are checked differently: criteria determine when the run can end, while a violated constraint ends the run immediately and escalates, because constraints usually guard irreversible actions. Criteria can be partially met; constraints cannot be partially violated.

Weak: "They're both requirements."

**Q5. Why not let the model verify its own work?**

Strong: Because self-evaluation is biased toward approving what the model just produced, and the model's stated reasoning is not a reliable record of what actually happened. Verification should come from something independent: tool evidence checked by code, tests that run, or a separate evaluator with a rubric and a fresh context. The model's self-check is still useful as a cheap early signal, but it is not the gate.

Weak: "Ask it to double-check."

**Q6. How do you implement goal monitoring in LangGraph?**

Strong: Keep the goal spec as configuration and put evidence, progress history, counters, and status in the graph state with reducers, so the checkpointer persists them. Tools return structured artifacts alongside their text. A `goal_check` node runs after every tool step and after every model turn without tool calls; it harvests evidence, evaluates criteria and constraints, updates counters, and sets a status. A conditional edge out of `goal_check` is the only route back to the agent, so no iteration skips the check, and every non-running status routes to END with a report.

Weak: "Use the recursion limit."

**Q7. Where do the success criteria come from?**

Strong: Best case, from the product: a ticket type has a known checklist. Otherwise a planner step drafts criteria and verification methods as structured output, and a rule or a human approves them before execution, which is a contract for the run. If the request has no checkable outcome, the agent asks one clarifying question. In every case the criteria should encode the user's real outcome, not an easy proxy like "a reply was sent."

Weak: "The model figures out what the user wants."

## Build it yourself

Extend `build_monitored_graph`:

1. Add a repeated-call detector: fingerprint each tool call as `name + sorted args`, and end the run with status `looping` when the same fingerprint appears three times, even if other evidence arrives.
2. Add a `replan` node. On the first `stalled`, route to `replan`, which appends a message containing the stall report and the unmet criteria and resets `no_new`; on the second stall, end with `escalated`.
3. Add an `off_goal` signal: give the goal spec a set of relevant tools, and count tool calls outside it; after two off-goal calls, inject the objective and unmet criteria again.
4. Make every terminal status produce a `final_report` state key with `status`, `met`, `unmet`, `violated`, and the evidence keys.

Acceptance checks:

- A script that calls `search_slack("Acme")` forever ends with `looping` after the third identical call.
- A script that stalls, receives the replan message, then completes the task ends with `done`, and exactly one replan message is in `messages`.
- A script that stalls twice ends with `escalated` and a `final_report` whose `unmet` lists all three criteria.
- Two consecutive calls to a tool outside the relevant set produce exactly one re-injected objective message.
- The happy-path run still ends with `done` and `progress[-1] == 1.0`.

## Key takeaways

- The model stopping is not the goal being met; the software needs its own definition of done.
- A goal spec has an objective, checkable success criteria, constraints, a definition of done, verification methods, and budgets.
- Criteria read structured evidence recorded by tools, not the model's claims about what it did.
- Monitor after every step: progress, stalls, drift, and constraint violations each get a named termination reason and a report.
- Step budgets and recursion limits are backstops; stall detection and goal checks are the monitor.
- In LangGraph, a `goal_check` node with reducers for evidence and progress, and a conditional edge that is the only way back to the agent, makes monitoring durable and unskippable.
- Define done before the run, verify with something other than the generator, and reuse the same criteria as eval graders.
