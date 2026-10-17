# 06. Planning

*Track A (agent engineering). Book: Chapter 6 (Planning). Prerequisites: 00. What an agent actually is; 01. Prompt Chaining; 04. Reflection.*

A ReAct loop decides one step at a time. That works until the task is long enough that "one step at a time" turns into wandering: the agent re-checks Slack three times, forgets it already pulled the invoice, and runs out of budget before it drafts the reply. Planning is the fix, but not in the way people usually describe it. The value of a plan is not that the model "thinks ahead." The value is that the plan becomes **explicit state**: something your software can validate, display, checkpoint, compare against observations, and revise. This lesson builds plan-and-execute with replanning in plain Python, rebuilds it as a planner/executor/replanner graph in LangGraph, looks at the todo-list tool that harnesses use instead, and then spends real time on when planning makes an agent worse.

## The problem

The ops agent gets this in Slack:

> "Acme says they were double-charged for September. Figure out what happened and get it fixed."

A plain ReAct loop (00. What an agent actually is) handles it like this on a good day:

```text
model:        "I need search_slack('acme double charge')"
application:  search_slack(...) -> 3 messages, one mentions invoice in_0931
model:        "I need stripe_charges('cus_acme', '2026-09')"
application:  stripe_charges(...) -> two charges of $49.00, same day
model:        "The customer was double-charged. I'll tell them."
```

And on a bad day:

```text
step 1   search_slack        ok
step 2   search_slack        (again, different phrasing)
step 3   crm_lookup          ok
step 4   search_notion       "refund policy" (not needed yet)
step 5   search_slack        (again)
...
step 14  budget exhausted, no refund ticket, no reply drafted
```

Nothing in the ReAct loop records *what the agent intends to do* or *what is already done*. The only memory is a flat transcript, and the model must reconstruct its progress from raw observations every turn. That is where it drifts. The interview repository has this exact question from Cognition: "Your agent drifts after a long run and confidently works on the wrong thing. Diagnose it." Missing explicit plan state is one of the first answers.

The naive fix is "ask the model to write a plan first." That produces a paragraph at the top of the transcript:

```text
Plan: 1) check Slack 2) check the CRM 3) check Stripe 4) reply.
```

That is still text. Your code cannot tell which steps are done, cannot reject a step that calls a nonexistent tool, and cannot notice when step 3 reveals something the plan did not anticipate.

What we want instead:

```text
                ┌──────────┐
 request ──────►│ planner  │── plan (validated, structured) ──┐
                └──────────┘                                  │
                                                              ▼
                ┌──────────┐   step    ┌──────────┐   obs   ┌──────────┐
                │replanner │◄──────────│ monitor  │◄────────│ executor │
                └────┬─────┘ surprise  └────┬─────┘         └────▲─────┘
                     │                      │ no surprise        │
                     │ revised remaining    └────────────────────┘
                     │ steps                     next step
                     └──────────────────────────────────────────►
                                      (when plan is empty) ──► finalize
```

Three roles. The **planner** turns a goal into a structured list of steps. The **executor** runs one step at a time. The **monitor** compares each observation with what the plan assumed, and when they disagree the **replanner** revises the remaining steps. The plan lives in state, not in prose.

### Receding horizon: why replanning is the point

Control engineers solved this decades ago with **model predictive control**, also called receding-horizon control. You plan the whole trajectory, commit only to the first action, observe the result, and plan again from the new state. You never trust a plan further than the next observation.

```text
t=0   plan: [a1, a2, a3, a4]   execute a1   observe
t=1   plan: [a2', a3', a4']    execute a2'  observe      (plan revised with new info)
t=2   plan: [a3'', a4'']       execute a3'' observe
```

For agents the pure version (replan after every step) is expensive, because every replan is a model call with the whole history. The practical version is: replan **on surprise**. A cheap deterministic monitor checks each observation, and only a flagged observation triggers a model call. That is the design we build.

### When planning is the wrong tool

Before building it, the honest caveat. The book presents planning as a core pattern, and Anthropic's "Building effective agents" guidance says to start with the simplest architecture that works. Both are right. A plan has costs:

| Cost | Why it hurts |
|---|---|
| An extra model call up front | Latency and tokens before any useful work |
| Commitment to a guess | The plan is written before any observation; early steps often assume facts that turn out false |
| Plan-following bias | Models tend to finish the plan even when the goal is already met or the plan is wrong |
| Bookkeeping | Status updates cost turns and tokens |

If the task is two or three tool calls, or the path is fixed (a workflow), do not add a planner. If the path depends on what you discover and the task is long, explicit plan state pays for itself.

## Build it without a framework

The model is scripted. The script stands in for a real model so the *control flow* (plan, execute, monitor, replan, finalize) is the object of study, and a live cell at the end shows the real thing. Everything else, including the tools, is deterministic Python.

First, the plan as a data structure. Each step names a tool and its arguments, carries a status, and says what it depends on. A validator runs before any step executes. This is the most important line of defense in the whole pattern: a plan that references a tool that does not exist, or depends on a step that comes later, is rejected before it costs anything.

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


import json
from typing import Literal
from pydantic import BaseModel, Field


class Step(BaseModel):
    id: str
    tool: str
    args: dict = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    done_when: str = ""                      # human-readable completion criterion
    status: Literal["pending", "done", "skipped"] = "pending"


class Plan(BaseModel):
    goal: str
    steps: list[Step]


# Deterministic stand-ins for Slack, CRM, and Stripe. `flags` is how a tool
# reports something the plan may not have anticipated.
TICKETS_OPENED = []


def search_slack(query):
    return {"hits": ["#acme-support: 'we see two $49 charges on Sep 3'"], "flags": []}


def crm_lookup(account):
    return {"account": "acme", "customer_id": "cus_acme", "owner": "dana", "flags": []}


def stripe_charges(customer, month):
    charges = [{"id": "ch_1", "amount": 4900, "date": "2026-09-03", "status": "succeeded"},
               {"id": "ch_2", "amount": 4900, "date": "2026-09-03", "status": "succeeded"}]
    return {"charges": charges, "flags": ["duplicate_charge:ch_2"]}


def refund_status(charge_id):
    return {"charge_id": charge_id, "refunded": False, "flags": []}


def open_refund_ticket(charge_id, amount, idempotency_key):
    # Side effect: guarded by an idempotency key so a retried step cannot double-refund.
    if idempotency_key not in {t["key"] for t in TICKETS_OPENED}:
        TICKETS_OPENED.append({"key": idempotency_key, "charge_id": charge_id, "amount": amount})
    return {"ticket": f"RF-{len(TICKETS_OPENED)}", "flags": []}


TOOLS = {f.__name__: f for f in
         [search_slack, crm_lookup, stripe_charges, refund_status, open_refund_ticket]}


def validate_plan(plan: Plan, completed: set[str] = frozenset()) -> list[str]:
    """Reject plans your software cannot execute. Runs before any step does."""
    errors, seen = [], set(completed)
    for step in plan.steps:
        if step.tool not in TOOLS:
            errors.append(f"{step.id}: unknown tool {step.tool!r}")
        for dep in step.depends_on:
            if dep not in seen:
                errors.append(f"{step.id}: depends on {dep!r}, which is not earlier in the plan")
        seen.add(step.id)
    if len(plan.steps) > 8:
        errors.append("plan too long: split the goal or ask the user")
    return errors


bad = Plan(goal="x", steps=[Step(id="s1", tool="stripe_charges", depends_on=["s2"]),
                            Step(id="s2", tool="delete_customer")])
print(validate_plan(bad))
assert len(validate_plan(bad)) == 2
```

Two design decisions are hiding in that cell.

First, `depends_on` and `done_when` exist even though the executor runs steps in order. Dependencies enable parallel execution later (03. Parallelization); `done_when` is what a human or an evaluator checks.

Second, the tool reports `flags`. The monitor does not ask a model "did anything surprising happen?" It reads a structured field. Cheap deterministic checks decide *when* to spend a model call; the model decides *what* to do about it.

Now the loop. The planner is one model call that returns JSON. The executor walks the steps. After each step the monitor looks at flags. On a flag, and only then, the replanner gets the goal, the completed steps with their observations, and the remaining steps, and returns a new list of remaining steps. The replanner's output goes through the same validator. The final answer is one more model call over the recorded observations.

```{code-cell} python
from langchain_core.messages import HumanMessage, SystemMessage

PLAN_JSON = json.dumps({"goal": "Resolve Acme double-charge report", "steps": [
    {"id": "s1", "tool": "search_slack", "args": {"query": "acme double charge"},
     "done_when": "customer report located"},
    {"id": "s2", "tool": "crm_lookup", "args": {"account": "acme"},
     "done_when": "Stripe customer id known"},
    {"id": "s3", "tool": "stripe_charges", "args": {"customer": "cus_acme", "month": "2026-09"},
     "depends_on": ["s2"], "done_when": "September charges listed"}]})

REPLAN_JSON = json.dumps({"goal": "Resolve Acme double-charge report", "steps": [
    {"id": "s4", "tool": "refund_status", "args": {"charge_id": "ch_2"},
     "depends_on": ["s3"], "done_when": "know whether ch_2 was already refunded"},
    {"id": "s5", "tool": "open_refund_ticket", "args": {"charge_id": "ch_2", "amount": 4900},
     "depends_on": ["s4"], "done_when": "refund ticket exists"}]})

FINAL = ("Acme was charged twice on Sep 3 (ch_1, ch_2, $49 each). ch_2 had not been "
         "refunded; refund ticket RF-1 is open. Draft reply ready for Dana.")


def plan_and_execute(model, request, max_replans=2, run_id="run-1"):
    trace = []                                           # (step_id, observation)
    plan = Plan.model_validate_json(model.invoke(
        [SystemMessage("Return a JSON plan."), HumanMessage(request)]).content)
    assert not validate_plan(plan), validate_plan(plan)
    remaining, replans = list(plan.steps), 0

    while remaining:
        step = remaining.pop(0)                          # receding horizon: commit to one step
        args = dict(step.args)
        if step.tool == "open_refund_ticket":            # side effects get a stable key
            args["idempotency_key"] = f"{run_id}:{step.id}"
        obs = TOOLS[step.tool](**args)
        step.status = "done"
        trace.append((step.id, obs))

        if obs["flags"] and replans < max_replans:       # monitor: replan on surprise only
            replans += 1
            revised = Plan.model_validate_json(model.invoke([
                SystemMessage("Revise the REMAINING steps given what happened."),
                HumanMessage(json.dumps({"goal": plan.goal, "done": trace,
                                         "remaining": [s.model_dump() for s in remaining]}))
            ]).content)
            errors = validate_plan(revised, completed={sid for sid, _ in trace})
            assert not errors, errors
            remaining = list(revised.steps)

    answer = model.invoke([SystemMessage("Answer from these observations only."),
                           HumanMessage(json.dumps(trace))]).content
    return {"answer": answer, "trace": trace, "replans": replans}


TICKETS_OPENED.clear()
model = scripted(PLAN_JSON, REPLAN_JSON, FINAL)   # exactly three model calls
result = plan_and_execute(model, "Acme says they were double-charged for September.")
for sid, obs in result["trace"]:
    print(sid, {k: v for k, v in obs.items() if k != "charges"})
print(result["answer"])
assert [sid for sid, _ in result["trace"]] == ["s1", "s2", "s3", "s4", "s5"]
assert result["replans"] == 1
assert TICKETS_OPENED == [{"key": "run-1:s5", "charge_id": "ch_2", "amount": 4900}]
```

The initial plan ended at "list the charges." It could not have planned a refund, because nobody knew there was a duplicate until `stripe_charges` ran. The monitor saw `duplicate_charge:ch_2`, the replanner added two steps, the validator checked them against the completed steps (so `depends_on: ["s3"]` is legal), and execution continued. Three model calls total. A pure receding-horizon loop would have made six.

The model never called a tool. It wrote a plan; your code validated it, ran it, attached idempotency keys, and decided when to replan. The model chooses, your software executes. Planning just moves the choosing into a data structure you can inspect.

In many real systems each step is itself a small ReAct agent ("find the customer's Stripe id" might take two or three calls). That changes nothing important: the step still has a completion criterion, reports a structured observation, and code updates the plan state.

## How LangGraph does it

The from-scratch version already has the three pieces of state a graph needs:

```text
remaining      the plan that has not run yet       replace on every write
trace          (step, observation) pairs           append-only log
replans        counter for the replan budget       replace
```

In LangGraph those become state keys with reducers. `plan` has no reducer, so every write replaces it, which is exactly what "the replanner revises the remaining steps" means. `past_steps` uses `operator.add`, so each executor run appends. The `while remaining` loop and the `if obs["flags"]` check become a conditional edge.

```{code-cell} python
import operator
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


class PlanState(TypedDict):
    request: str
    goal: str
    plan: list[dict]                                   # replace: the replanner overwrites it
    past_steps: Annotated[list[tuple], operator.add]   # append: the executor logs to it
    replans: int
    response: str


def planner(state: PlanState):
    plan = Plan.model_validate_json(model.invoke([HumanMessage(state["request"])]).content)
    assert not validate_plan(plan)
    return {"goal": plan.goal, "plan": [s.model_dump() for s in plan.steps], "replans": 0}


def executor(state: PlanState, config):
    step, rest = Step(**state["plan"][0]), state["plan"][1:]
    args = dict(step.args)
    if step.tool == "open_refund_ticket":
        # thread_id + step id is stable across retries and resumes of this run
        args["idempotency_key"] = f"{config['configurable']['thread_id']}:{step.id}"
    obs = TOOLS[step.tool](**args)
    return {"plan": rest, "past_steps": [(step.id, obs)]}


def replanner(state: PlanState):
    revised = Plan.model_validate_json(model.invoke([HumanMessage(json.dumps(
        {"goal": state["goal"], "done": state["past_steps"], "remaining": state["plan"]}))]).content)
    assert not validate_plan(revised, completed={sid for sid, _ in state["past_steps"]})
    return {"plan": [s.model_dump() for s in revised.steps], "replans": state["replans"] + 1}


def finalize(state: PlanState):
    return {"response": model.invoke([HumanMessage(json.dumps(state["past_steps"]))]).content}


def after_execute(state: PlanState):
    """The monitor. Same test as the `if obs['flags']` line in plan_and_execute."""
    _, last_obs = state["past_steps"][-1]
    if last_obs["flags"] and state["replans"] < 2:
        return "replanner"
    return "executor" if state["plan"] else "finalize"


def after_replan(state: PlanState):
    return "executor" if state["plan"] else "finalize"


builder = StateGraph(PlanState)
builder.add_node("planner", planner)
builder.add_node("executor", executor)
builder.add_node("replanner", replanner)
builder.add_node("finalize", finalize)
builder.add_edge(START, "planner")
builder.add_edge("planner", "executor")
builder.add_conditional_edges("executor", after_execute, ["executor", "replanner", "finalize"])
builder.add_conditional_edges("replanner", after_replan, ["executor", "finalize"])
builder.add_edge("finalize", END)
graph = builder.compile(checkpointer=InMemorySaver())

TICKETS_OPENED.clear()
model = scripted(PLAN_JSON, REPLAN_JSON, FINAL)
config = {"configurable": {"thread_id": "acme-dup-42"}}
out = graph.invoke({"request": "Acme says they were double-charged.", "past_steps": []}, config)

print([sid for sid, _ in out["past_steps"]], "replans:", out["replans"])
print(out["response"])
history = list(graph.get_state_history(config))
print(f"{len(history)} checkpoints; plan length over time:",
      [len(h.values.get("plan", [])) for h in reversed(history) if "plan" in h.values])
assert [sid for sid, _ in out["past_steps"]] == ["s1", "s2", "s3", "s4", "s5"]
assert out["replans"] == 1 and out["plan"] == []
assert TICKETS_OPENED[0]["key"] == "acme-dup-42:s5"
```

Map it back:

| In `plan_and_execute` | In the graph |
|---|---|
| `remaining = list(plan.steps)` | `plan` key, replace semantics |
| `trace.append(...)` | `past_steps` with the `operator.add` reducer |
| `remaining.pop(0)` + tool call | the `executor` node, one step per super-step |
| `if obs["flags"] and replans < max_replans` | the `after_execute` conditional edge |
| `remaining = list(revised.steps)` | the `replanner` node writing `plan` |
| `while remaining` exit | the `"finalize"` branch of the edge |
| nothing | a checkpoint after every node |

The last row is what the graph buys you. The printed plan lengths are the plan *over time*, read back from checkpoints: 3, 2, 1, 0, then 2 after the replan, then 1, 0. If the process dies after `s4`, resuming the thread restarts at the executor with `plan == [s5]`. The from-scratch loop would restart from the planner and re-run everything. And because the idempotency key is built from `thread_id` and the step id rather than a random value, a resumed `s5` produces the same key and the ticket system deduplicates it (36. Durable Agent Orchestration goes deeper).

The plan in state is also what a "review the plan" UI edits: an `interrupt()` before the executor (13. Human in the Loop) lets a human approve or change `plan` before anything runs.

### The harness version: a todo-list tool

Graph-shaped plan-and-execute puts planning in *your* code. Harnesses like Claude Code and Deep Agents took the opposite approach: give the model a tool, commonly called `write_todos`, whose only effect is to overwrite a `todos` list in agent state. The model plans by calling the tool, marks items `in_progress` and `completed` as it goes, and revises the list whenever it wants. There is no separate planner node and no separate replanner. The plan is still explicit state; the model owns the updates.

LangChain ships this as `TodoListMiddleware` for `create_agent`:

```{code-cell} python
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.tools import tool


@tool
def stripe_charges_tool(customer: str, month: str) -> str:
    """List a customer's Stripe charges for a month (YYYY-MM)."""
    return json.dumps(stripe_charges(customer, month))


todo_model = scripted(
    [("write_todos", {"todos": [
        {"content": "List Acme's September charges", "status": "in_progress"},
        {"content": "Check whether the duplicate was refunded", "status": "pending"}]})],
    [("stripe_charges_tool", {"customer": "cus_acme", "month": "2026-09"})],
    [("write_todos", {"todos": [
        {"content": "List Acme's September charges", "status": "completed"},
        {"content": "Check whether the duplicate was refunded", "status": "in_progress"},
        {"content": "Open refund ticket for ch_2", "status": "pending"}]})],
    "ch_2 is a duplicate. Next: refund check, then ticket.",
)
agent = create_agent(todo_model, tools=[stripe_charges_tool], middleware=[TodoListMiddleware()])
out = agent.invoke({"messages": [HumanMessage("Acme says they were double-charged.")]})
for t in out["todos"]:
    print(f"[{t['status']:>11}] {t['content']}")
assert [t["status"] for t in out["todos"]] == ["completed", "in_progress", "pending"]
assert len(out["todos"]) == 3        # the model added a step after seeing the data
```

Underneath, the middleware does two things you already built. It adds a tool whose handler writes a `Command(update={"todos": ...})` to state, which is our `remaining = list(revised.steps)` done by the model instead of a replanner node. And it adds a system-prompt section telling the model when to use the list and when not to (its default text says to skip it for tasks under about three steps). That is the whole mechanism. No validator, no monitor. That is the trade.

| | Graph plan-and-execute | Todo-list tool |
|---|---|---|
| Who decides when to replan | your monitor | the model |
| Plan validation | your code, before execution | none by default |
| Cost | model calls only at plan and surprise | a tool call per status update |
| Best for | known domains, consequential actions, audit | open-ended coding and research |

Deep Agents is where this pattern was most visible. Current Deep Agents docs describe the todo middleware as **opt-in** starting with the 0.7 line, as part of making the default harness leaner; earlier versions, including the 0.5.x installed in this course, enabled it by default. Treat that as a signal of the direction: a todo list is a tool you add when tasks are long, not a default for every agent. Here is the live version with a real model. Check the Deep Agents docs for the exact opt-in parameter on the version you install.

```{code-cell} python live
import os
from langchain.chat_models import init_chat_model

real = init_chat_model(os.environ.get("COURSE_MODEL", "openai:gpt-5.6"))
live_agent = create_agent(real, tools=[stripe_charges_tool], middleware=[TodoListMiddleware()],
                          system_prompt="You resolve billing disputes. Plan with todos when "
                                        "the task has more than three steps.")
out = live_agent.invoke({"messages": [HumanMessage(
    "Acme says they were double-charged for September. Investigate and propose a fix.")]})
print(out.get("todos"))
print(out["messages"][-1].content)

# Deep Agents: the same idea inside a full harness (files, subagents, context management).
from deepagents import create_deep_agent
deep = create_deep_agent(model=real, tools=[stripe_charges_tool],
                         system_prompt="Plan before acting on multi-step billing tasks.")
print(deep.invoke({"messages": [{"role": "user", "content": "Investigate Acme's duplicate charge."}]})["messages"][-1].content)

# And the from-scratch planner against the real model: ask for JSON matching Plan.
planner_model = real.with_structured_output(Plan)
print(planner_model.invoke("Plan how to resolve Acme's double-charge report using tools: "
                           + ", ".join(TOOLS)).model_dump_json(indent=2))
```

## What goes wrong in production

1. **The plan references things that do not exist.** Symptom: step 3 calls `stripe_refund` (the tool is `refund_status`) and the run dies mid-way. Cause: the plan was never validated. Fix: validate before execution against the live tool registry, argument schemas, and dependency order, and send validation errors back to the planner once. Structured output with a `Plan` schema (24. Structured Output) removes the parse failures but not the semantic ones.

2. **Plan-following bias.** Symptom: step 2 found that Acme already got a refund yesterday, and the agent opens a second refund ticket anyway because it was step 5. Cause: no monitor, or a monitor that only looks for errors, not for "goal already met." Fix: the monitor checks the goal's completion criterion after every step, not only failures, and every step carries a `done_when`. The cheapest version is a deterministic predicate on structured observations.

3. **Replan storms.** Symptom: cost triples on messy accounts; traces show a replan after every step. Cause: the surprise signal is too sensitive, or the replanner keeps producing plans that immediately trigger again. Fix: a replan budget (we used two), then escalate to a human or finish with what you have. Log every replan with its trigger so you can tune the monitor from traces.

4. **Stale plans after resume.** Symptom: a run paused for approval over the weekend resumes and executes a plan written against Friday's data. Cause: the plan is state, and state survives; the world did not wait. Fix: on resume, re-run the read steps whose data has a freshness bound, or force a replan when the pause exceeds a threshold.

5. **Duplicate side effects on retry.** Symptom: two refund tickets. Cause: the executor timed out after the ticket system accepted the request, and the retry opened another. Fix: idempotency keys derived from the run and step ids, as in both implementations above. Never use a random UUID generated inside the step, because a re-executed step generates a new one.

6. **The plan is too fine or too coarse.** Symptom: fine-grained plans spend more tokens updating status than doing work; coarse plans ("investigate billing") cannot be checked. Fix: one step per verifiable outcome, usually one to three tool calls each, with a plan length cap. A plan longer than about eight steps usually means the goal should be split or clarified.

7. **Old plans leak into context.** Symptom: after five replans the transcript holds five plans and the model follows an old one. Fix: keep one current plan in a replaced state key and render only that into the prompt (34. Context Engineering).

## What changed by 2026

- **Reasoning models moved the break-even.** Models that reason before answering do a lot of short-horizon planning internally, so explicit planners for three-step tasks mostly add latency. Explicit plan state still matters for long tasks, for audit, and for anything a human must approve. Do not add "first make a plan, think step by step" prompt scaffolding by default; state the goal, the constraints, and what "done" means, and verify outcomes rather than reading the model's reasoning text, which is not a faithful log of why it acted.
- **Plans as tools, not prompts.** Harnesses standardized on a todo-list tool that writes plan state. LangChain's `TodoListMiddleware` is the framework version. Its own default prompt tells the model to skip the list for trivial tasks.
- **Leaner defaults.** Current Deep Agents docs describe the todo middleware becoming opt-in in the 0.7 line, alongside a much smaller default prompt. The direction is clear across harnesses: fewer always-on components, add planning when the task needs it.
- **Planner / generator / evaluator.** Anthropic's harness-design write-up for long-running application work describes a planner that expands the goal, a generator that builds, and a separate evaluator, with "done" agreed before work starts. That is plan-and-execute with a stronger monitor, and it argues for separating the component that checks from the component that does.
- **Durable plans.** With checkpointers as a default, the plan survives crashes and pauses. That shifted the failure mode from "lost progress" to "stale plan," which is why freshness checks on resume are now part of the pattern.

## Interview questions

**Q1. What is plan-and-execute, and how is it different from ReAct?**

Strong: ReAct decides one action at a time from the transcript; its only memory of intent is the log. Plan-and-execute first produces an explicit, structured list of steps, then an executor runs them one at a time, and a replanner revises the remaining steps when observations contradict the plan. The practical difference is that the plan is state my code can validate, checkpoint, render to a human, and compare against observations. It also saves model calls when steps are mechanical, since the executor does not need the big model for each one. The cost is an up-front guess that is often wrong, so replanning is not optional.

Weak: "Plan-and-execute thinks ahead and ReAct doesn't."

**Q2. When should an agent replan?**

Strong: Conceptually, after every observation; that is receding-horizon control, where you commit only to the next action. In practice that is too expensive, so I replan on surprise: a deterministic monitor checks each structured observation for flags, failed preconditions, or the goal already being met, and only then calls the model. I cap replans per run and escalate when the cap is hit. And every replan output goes through the same validator as the first plan.

Weak: "When a step fails."

**Q3. Your agent drifts after a long run and confidently works on the wrong thing. Diagnose it.**

Strong: I would look at the trace for three things. First, whether there is explicit goal and plan state at all, or whether the model is reconstructing progress from a long flat transcript each turn; that is the most common cause. Second, whether the context was compacted in a way that dropped the goal or the constraints. Third, whether a replan changed the plan without anything checking that the new plan still served the original goal. Fixes are a plan in state rendered fresh every turn, a goal and done criterion pinned outside compaction, a monitor that checks progress against the goal, and a step budget that forces a checkpoint with a human.

Weak: "Use a bigger context window."

**Q4. How do you make plan execution safe when steps have side effects?**

Strong: Validate the whole plan before any step runs, including tool names, arguments, and a policy check on consequential tools. Put approval gates in front of consequential steps, ideally showing the human the plan, not just the single call. Give every side-effecting step an idempotency key derived from the run and step ids so retries and resumes are safe. And checkpoint after every step so a crash resumes at the failed step rather than replaying earlier writes.

Weak: "Ask the model to double-check before acting."

**Q5. When does planning hurt?**

Strong: On short tasks, where the planner call adds latency and the model's own reasoning already plans implicitly. On tasks where the first observation changes everything, because the up-front plan is mostly wrong and you pay to write and then rewrite it. And when there is no monitor, because models tend to keep following a plan after it stopped making sense. I start without a planner and add plan state when traces show the agent losing track of progress or the task needs human review of intent.

Weak: "Planning always improves reliability."

**Q6. Compare a planner node in a graph with a todo-list tool in a harness.**

Strong: A planner node puts planning under my control: I decide when plans are written and revised, I validate them, and I only pay for model calls at planning and at surprises. A todo-list tool gives the model a tool that overwrites plan state; the model decides when to plan and replan, which suits open-ended coding or research but gives up validation and costs a tool call per status update. Both make the plan explicit state. For billing or ops actions I prefer the graph; for exploratory work I prefer the tool, and I would still add a completion check outside the model.

Weak: "They're the same thing."

**Q7. How granular should plan steps be?**

Strong: One step per verifiable outcome. A step should have a completion criterion my code or an evaluator can check, which usually means one to three tool calls. Finer than that and the agent spends its budget on bookkeeping; coarser and nothing can tell whether the step succeeded. A cap on plan length is a useful forcing function: if the planner needs more than about eight steps, the goal probably needs splitting or clarification.

Weak: "As detailed as possible."

## Build it yourself

Extend `plan_and_execute` (or the graph) with a **goal-met monitor** and **parallel steps**:

1. Add a `goal_met(trace) -> bool` predicate. For the Acme case it returns true when any observation shows `refunded: True` for the duplicate charge. When it fires, skip remaining steps (mark them `skipped`) and finalize.
2. Change `refund_status` so it can be configured to return `refunded: True`. With that configuration, the run must not call `open_refund_ticket`.
3. Execute steps whose `depends_on` are all satisfied in the same batch, using a thread pool in the from-scratch version or `Send` in the graph. `search_slack` and `crm_lookup` have no dependencies and must run in the same batch.
4. Add a hard cap: if the replanner is called a third time, stop and return a result with `status: "needs_human"` and the current plan.

Acceptance checks:

- With `refunded: True`, `TICKETS_OPENED` is empty and the trace contains a `skipped` step.
- `s1` and `s2` appear in the same batch in the trace.
- A replanner scripted to always return a flagged step ends with `status == "needs_human"` after exactly two replans.
- Running the graph twice with the same `thread_id` never produces two tickets with the same key.

## Key takeaways

- The value of a plan is that it is explicit state your code can validate, checkpoint, display, and compare against observations.
- Plan, commit to one step, observe, replan: receding-horizon control, made affordable by replanning only on surprise.
- Validate every plan and every revision against the real tool registry and dependency order before anything runs.
- In LangGraph the plan is a replace-semantics key, the executed steps are an append-only log, and the monitor is a conditional edge.
- Harnesses expose planning as a todo-list tool the model updates; current Deep Agents docs make it opt-in, which says planning is for long tasks, not every task.
- Side-effecting steps need idempotency keys derived from run and step ids, because retries and resumes will re-run them.
- Planning hurts on short tasks and without a monitor; add it when traces show the agent losing track.
