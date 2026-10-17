# 20. Prioritization

*Track A (agent engineering). Book: Chapter 20 (Prioritization). Prerequisites: 00. What an agent actually is, 06. Planning, 11. Goal Setting and Monitoring.*

An agent that only ever has one task does not need to prioritize. The moment it has two, it is making a scheduling decision, and if it makes that decision implicitly (whichever task appeared first, whichever the model mentioned last) it will make it badly. Prioritization is the part of agent design where you decide, in code, what "most important next" means, and where you keep the queue of work as first-class state rather than as a paragraph in the transcript.

This lesson builds a scoring function, a queue with aging and preemption, and a scheduler loop. Then it puts the same machinery in a LangGraph graph whose queue survives across invocations, which is what a continuously running agent needs.

## The problem

Monday morning. The ops agent's inbox holds:

```text
A. Slack: "Acme's invoice is wrong, they're threatening to churn"        (customer, $48k ARR)
B. Stripe webhook: 14 payment failures in the last hour                  (many customers)
C. Notion: quarterly docs cleanup task assigned two weeks ago            (nobody is waiting)
D. CRM: Beta Corp renewal call notes need a summary by 2pm               (deadline)
E. SQL: nightly usage report job failed                                  (blocks D partly)
```

A naive agent works the list top to bottom, because the model was handed the list and lists have an order. It spends twenty minutes on Acme's invoice, then discovers the Stripe failures, which turn out to be the *cause* of Acme's complaint and of thirteen others. Meanwhile D's 2pm deadline passes. C never gets done, this week or any week, because something always arrives that feels more urgent.

```text
naive:   A ────────────────► B ───────► D (late) ──► E ──► C (never)
better:  E (unblocks D) ► B (root cause, wide blast radius) ► D (deadline) ► A (now trivial) ► C (aged in)
```

The naive version fails in four distinct ways:

1. **No scoring.** "Urgent" and "important" were never separated, so the loud item won.
2. **No dependency awareness.** E blocks D; nothing knew that.
3. **No preemption.** New information (the webhook storm) arrived mid-task and could not interrupt.
4. **No fairness.** Low-urgency strategic work starved.

The book describes prioritization as ranking tasks by value, urgency, dependencies, and cost, and as something that must update as new information arrives. The systems reader already knows this problem: it is a scheduler. Priority queues, aging to prevent starvation, preemption, and admission control are fifty-year-old ideas. The agent-specific twist is that the *scoring inputs* often come from the model (how important does this look? what is the expected value of fixing it?) while the *scheduling decision* must be made by code, deterministically, with a log.

```text
model:        "This webhook storm probably explains the Acme complaint; importance 5, blast radius 14 accounts"
application:  score(B) = 9.1  -> promote B above A; log the reason
model:        (works on B)
```

The model estimates. Your software decides.

### What goes into a score

A useful score has five inputs, and it matters that they stay separate until the last step so you can inspect each one:

| Input | Question it answers | Typical source |
|---|---|---|
| importance | How much does it matter if this is never done? | policy, account tier, model estimate |
| urgency | How fast does the value decay? Is there a deadline? | deadline, SLA, age |
| expected value | What do we gain if it succeeds, times the probability it succeeds? | model estimate, historical data |
| cost | Tokens, wall-clock, human attention, dollars | estimate from similar tasks |
| risk | What breaks if the action is wrong or irreversible? | tool metadata (side effects), policy |

A common shape is `(importance × urgency + expected_value) / cost`, with risk acting as a gate rather than a term: a high-risk action does not get a lower score, it gets routed to approval (13. Human in the Loop). Whatever formula you choose, three properties matter more than the formula:

```text
1. Monotone:     raising urgency never lowers the score.
2. Explainable:  every score can be decomposed into its inputs for the log.
3. Dynamic:      scores are recomputed when inputs change, not frozen at enqueue time.
```

## Build it without a framework

The scripted model helper is copied verbatim so the LangGraph section can use it. The scheduler itself needs no model.

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

### A task, a score, and a queue

Tasks are plain data. The score is a pure function of the task and the current time, so it can be recomputed every tick. The queue is a list we re-sort on each pick; a heap would be faster but a heap cannot be re-scored in place, and re-scoring is the whole point when priorities are dynamic.

```{code-cell} python
from dataclasses import dataclass, field


@dataclass
class Task:
    id: str
    title: str
    importance: int            # 1..5, how much it matters if never done
    expected_value: float      # 0..10, gain x probability of success
    cost: float                # relative effort; 1.0 = a typical task
    risk: str = "low"          # low | medium | high  (high routes to approval)
    deadline: int | None = None   # tick by which it must be done
    blocked_by: set[str] = field(default_factory=set)
    created: int = 0
    status: str = "pending"


def urgency(task, now):
    """1.0 baseline; rises toward the deadline; aging adds a slow ramp so nothing starves."""
    u = 1.0
    if task.deadline is not None:
        remaining = max(task.deadline - now, 0)
        u += 3.0 / (remaining + 1)               # 4.0 at the deadline, 1.75 with 3 ticks left
    u += 0.1 * (now - task.created)              # aging: +0.1 per tick waiting
    return u


def score(task, now):
    """(importance x urgency + expected value) / cost. Risk gates, it does not score."""
    return (task.importance * urgency(task, now) + task.expected_value) / task.cost


def explain(task, now):
    return {"id": task.id, "importance": task.importance, "urgency": round(urgency(task, now), 2),
            "ev": task.expected_value, "cost": task.cost, "score": round(score(task, now), 2)}


INBOX = [
    Task("A", "Acme invoice complaint", importance=4, expected_value=6, cost=2.0),
    Task("B", "Stripe webhook failure storm", importance=5, expected_value=9, cost=1.5, risk="medium"),
    Task("C", "Quarterly docs cleanup", importance=2, expected_value=2, cost=1.0),
    Task("D", "Summarize Beta Corp renewal notes", importance=3, expected_value=4, cost=1.0,
         deadline=4, blocked_by={"E"}),
    Task("E", "Rerun failed nightly usage report", importance=3, expected_value=3, cost=1.0),
]


def runnable(tasks, done):
    return [t for t in tasks if t.status == "pending" and t.blocked_by <= done]


def pick_next(tasks, done, now):
    candidates = runnable(tasks, done)
    return max(candidates, key=lambda t: score(t, now)) if candidates else None


ranked = sorted(INBOX, key=lambda t: -score(t, 0))
for t in ranked:
    print(explain(t, 0), "" if not t.blocked_by else f"blocked by {t.blocked_by}")

first = pick_next(INBOX, done=set(), now=0)
print("\nfirst pick:", first.id, first.title)
assert first.id == "B"                         # highest score among runnable tasks
assert "D" not in {t.id for t in runnable(INBOX, set())}   # blocked until E is done
```

Read the ranking, not just the winner. B wins because it combines the highest importance with the highest expected value at a moderate cost. D has the deadline but is blocked, so it is not even a candidate. E unblocks D, which the score does not yet know; we will fix that in the scheduler by letting a blocked task's urgency flow to its blocker.

### The scheduler loop: aging, preemption, and a log

A continuously running agent is a loop over ticks. Each tick: admit any new tasks, re-score everything, pick, work for one tick, log the decision. Two additions make it a real scheduler rather than a sort:

- **Blocker inheritance.** A task that blocks a deadline task inherits that deadline's urgency. Otherwise cheap prerequisites sit behind flashy work.
- **Preemption.** If a newly admitted task outscores the task in progress by a margin, the current task is parked (its progress preserved) and the new one runs. The margin prevents thrashing.

```{code-cell} python
def effective_score(task, tasks, now):
    """A blocker inherits the best score of the tasks waiting on it."""
    own = score(task, now)
    dependents = [t for t in tasks if task.id in t.blocked_by and t.status == "pending"]
    return max([own] + [score(t, now) for t in dependents])


def run_scheduler(tasks, arrivals, ticks=12, preempt_margin=1.5):
    """arrivals: {tick: [Task, ...]} admitted at that tick. Returns the decision log."""
    tasks, done, log, current = list(tasks), set(), [], None
    for now in range(ticks):
        for t in arrivals.get(now, []):                    # admission
            t.created = now
            tasks.append(t)
        candidates = runnable(tasks, done)
        if not candidates:
            break
        best = max(candidates, key=lambda t: effective_score(t, tasks, now))
        if current and current.status == "pending" and best is not current:
            gap = effective_score(best, tasks, now) - effective_score(current, tasks, now)
            if gap < preempt_margin:
                best = current                             # not worth the context switch
            else:
                log.append({"tick": now, "event": "preempt", "parked": current.id, "for": best.id,
                            "gap": round(gap, 2)})
        current = best
        log.append({"tick": now, "event": "work", "task": current.id,
                    "score": round(effective_score(current, tasks, now), 2)})
        current.status = "done"                            # one tick of work completes a task here
        done.add(current.id)
    return log, done


import copy
tasks = copy.deepcopy(INBOX)
arrivals = {1: [Task("F", "P0: checkout is down for all EU customers", importance=5,
                     expected_value=10, cost=1.0, risk="high")]}
log, done = run_scheduler(tasks, arrivals)
for row in log:
    print(row)

order = [r["task"] for r in log if r["event"] == "work"]
print("\norder:", order)
assert order[0] == "B"
assert order.index("E") < order.index("D")            # blocker before the deadline task
assert "C" in done                                     # low-urgency work eventually ran
assert order.index("F") == 1                           # the P0 was admitted at tick 1 and ran at once
```

Every row of the log is a decision with its score attached. That is the audit trail. When a stakeholder asks "why did the agent spend Monday morning on webhooks instead of my account," the answer is a row, not a guess.

Notice what aging does for C. Its raw score is the lowest in the inbox, but `urgency` grows by 0.1 per waiting tick. In a longer run with a steady stream of arrivals, that ramp is what guarantees C eventually outranks the newest medium-priority item. Without it, strategic work is starved forever and nobody notices because nothing is "wrong" at any single tick.

### Risk is a gate, not a term

F was marked `risk="high"`. In this toy scheduler it ran anyway. In the real agent, high-risk tasks are routed to an approval step before any side-effecting tool runs, and the queue keeps them in a `waiting_approval` status that does not consume the worker. That is the boundary between this lesson and 13. Human in the Loop: prioritization decides *what next*; approval decides *whether at all*.

## How LangGraph does it

The from-scratch scheduler kept its queue in a Python list inside one function call. A continuously running agent cannot do that: it is invoked many times (by a cron, by a webhook, by a user), and the queue must survive between invocations, across process restarts, and through crashes mid-task. That is exactly what a LangGraph checkpointer gives you: the state, including the queue, persisted per `thread_id` after every step.

Three primitives carry the design:

```text
State reducer     tasks: merge-by-id, so a node can upsert one task without rewriting the queue
Command(goto=)    the pick node routes to "work" or END based on the queue, and updates state in the same step
Checkpointer      the queue is durable; invoke #2 sees invoke #1's leftovers
```

The `work` node calls the model. The scripted model stands in for a real one: for most tasks it says "done"; for the Stripe task it discovers a new task and calls an `enqueue_task` tool, which is how work generates work in a real system.

```{code-cell} python
from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver


def merge_tasks(existing: list[dict], updates: list[dict]) -> list[dict]:
    """Upsert by id. A node returns only the tasks it touched."""
    by_id = {t["id"]: t for t in existing}
    for t in updates:
        by_id[t["id"]] = {**by_id.get(t["id"], {}), **t}
    return list(by_id.values())


class SchedulerState(TypedDict):
    tasks: Annotated[list[dict], merge_tasks]
    current: str | None
    log: Annotated[list[dict], lambda a, b: a + b]
    tick: int


def task_score(t, now):
    return score(Task(**{k: v for k, v in t.items() if k != "status"}), now) if t["status"] == "pending" else -1


def pick(state: SchedulerState) -> Command:
    """Deterministic scheduling decision. The model never picks."""
    done = {t["id"] for t in state["tasks"] if t["status"] == "done"}
    candidates = [t for t in state["tasks"]
                  if t["status"] == "pending" and set(t.get("blocked_by", ())) <= done]
    if not candidates:
        return Command(update={"current": None}, goto=END)
    best = max(candidates, key=lambda t: task_score(t, state["tick"]))
    return Command(update={"current": best["id"],
                           "log": [{"tick": state["tick"], "picked": best["id"],
                                    "score": round(task_score(best, state["tick"]), 2)}]},
                   goto="work")


@tool
def enqueue_task(id: str, title: str, importance: int, expected_value: float, cost: float) -> str:
    """Add a newly discovered task to the queue."""
    return f"enqueued {id}"


model = scripted(
    # B takes two turns: it discovers follow-up work, then finishes.
    [("enqueue_task", {"id": "G", "title": "Notify 14 customers whose payments failed",
                       "importance": 4, "expected_value": 7, "cost": 1.0})],
    "Investigated the Stripe failures: an expired API key on the EU gateway. Rotated it.",
    "Notified 14 customers with the corrected charge date.",     # G outscores E, so it runs next
    "Report rerun; usage table refreshed.",                       # E
    "Renewal notes summarized and posted to CRM.",                # D, unblocked by E
    "Acme invoice explained: onboarding fee billed twice; credit issued.",   # A
    "Docs cleanup complete.",                                     # C, arrives in invocation 2
).bind_tools([enqueue_task])


def work(state: SchedulerState):
    """One unit of work on the current task. Discovered tasks are enqueued, not narrated."""
    task = next(t for t in state["tasks"] if t["id"] == state["current"])
    updates, done_now = [], False
    while not done_now:                                   # a task may need several model turns
        ai = model.invoke([HumanMessage(f"Work on: {task['title']}")])
        for call in ai.tool_calls:
            if call["name"] == "enqueue_task":
                updates.append({**call["args"], "status": "pending", "created": state["tick"]})
        done_now = not ai.tool_calls
    updates.append({"id": task["id"], "status": "done"})
    return {"tasks": updates, "tick": state["tick"] + 1}


builder = StateGraph(SchedulerState)
builder.add_node("pick", pick)
builder.add_node("work", work)
builder.add_edge(START, "pick")
builder.add_edge("work", "pick")
graph = builder.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "ops-queue-monday"}}

initial = [{"id": t.id, "title": t.title, "importance": t.importance, "expected_value": t.expected_value,
            "cost": t.cost, "deadline": t.deadline, "blocked_by": list(t.blocked_by),
            "created": 0, "status": "pending"} for t in INBOX if t.id in {"A", "B", "D", "E"}]

# Invocation 1: the morning batch. C is deliberately left out; it arrives later.
result = graph.invoke({"tasks": initial, "current": None, "log": [], "tick": 0}, config)
order = [r["picked"] for r in result["log"]]
print("invocation 1 order:", order)
assert order == ["B", "G", "E", "D", "A"]      # G was discovered during B and outscored E

# Invocation 2, later: a new task arrives. The queue and the tick counter came from the checkpoint.
result = graph.invoke({"tasks": [{"id": "C", "title": "Quarterly docs cleanup", "importance": 2,
                                  "expected_value": 2, "cost": 1.0, "created": result["tick"],
                                  "status": "pending"}]}, config)
print("invocation 2 order:", [r["picked"] for r in result["log"][len(order):]])
print("all done:", sorted(t["id"] for t in result["tasks"] if t["status"] == "done"))
assert {t["id"] for t in result["tasks"] if t["status"] == "done"} == {"A", "B", "C", "D", "E", "G"}
assert len(graph.get_state(config).values["log"]) == 6
```

Map it back:

| From scratch | LangGraph | Underneath |
|---|---|---|
| `tasks` list mutated in place | `tasks` key with `merge_tasks` reducer | each node returns only its deltas; the reducer upserts by id |
| `pick_next` + `if candidates` | `pick` node returning `Command(goto=...)` | routing and state update in one step, no separate conditional edge |
| `for now in range(ticks)` | `work -> pick` edge plus `goto=END` when the queue is empty | the loop is the graph cycle |
| function-local state | checkpointer keyed by `thread_id` | queue survives between invocations and across restarts |
| `arrivals` dict | a second `invoke` with new tasks | admission is just another state update through the reducer |

The second invocation is the important one. Nothing was passed but the new task. The queue, the tick, and the log all came from the checkpoint, and the graph resumed scheduling as if it had never stopped. That is what "continuously running agent" means in practice: not a process that never exits, but a state that never resets.

Two further LangGraph fits worth knowing without code here. `interrupt()` inside `work` turns the high-risk gate into a real approval pause: the task stays at the head of the queue, the graph parks at the checkpoint, and `Command(resume=...)` continues it hours later (13. Human in the Loop). And a `Store` (08. Memory Management) is where you keep the *scoring policy* (account tiers, SLA tables, learned cost estimates) so that it is shared across threads while the queue stays per-thread.

### Letting the model estimate, structurally

The scripted model above never scored anything; the scores came from the task fields. In production those fields are often missing when a task arrives ("Acme is unhappy" carries no importance integer). The right pattern is structured output: the model fills in the *inputs* to the score with a typed schema, your code computes the score, and the log records both.

```{code-cell} python live
import os
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model


class TriageEstimate(BaseModel):
    importance: int = Field(ge=1, le=5, description="Consequence if never done")
    expected_value: float = Field(ge=0, le=10, description="Gain x probability of success")
    cost: float = Field(gt=0, description="Relative effort; 1.0 is a typical task")
    risk: str = Field(description="low, medium, or high; high means irreversible or customer-visible side effects")
    rationale: str


real_model = init_chat_model(os.environ.get("COURSE_MODEL", "openai:gpt-5.6"))
triage = real_model.with_structured_output(TriageEstimate)

raw_items = [
    "Slack from CSM: Acme ($48k ARR) says their invoice is wrong and mentioned churn.",
    "Stripe webhook: 14 payment_failed events in the last hour, all EU gateway.",
    "Notion: quarterly docs cleanup, assigned two weeks ago, no one waiting on it.",
]
estimated = []
for i, item in enumerate(raw_items):
    est = triage.invoke(f"Estimate triage inputs for this ops item. Be conservative.\n\n{item}")
    task = Task(id=str(i), title=item[:40], importance=est.importance,
                expected_value=est.expected_value, cost=est.cost, risk=est.risk)
    estimated.append((task, est.rationale))

for task, why in sorted(estimated, key=lambda p: -score(p[0], 0)):
    print(f"{score(task, 0):5.2f}  {task.title:42s} risk={task.risk:6s} | {why[:70]}")
```

The model wrote the rationale; the code wrote the ranking. If the model's estimate is wrong you can see which input it got wrong, correct the schema description, and re-run the eval set. That is impossible when the model is asked "which task should I do first?" and answers in prose.

## What goes wrong in production

1. **Starvation of strategic work.** Symptom: the docs cleanup, the eval-set expansion, and the monitoring improvements never happen; the agent is always busy. Cause: a pure priority queue with no aging. Fix: an aging term in urgency, or a reserved share of ticks for the lowest quartile, and a metric for "oldest pending task age" with an alert.

2. **Thrashing on preemption.** Symptom: five tasks each half-done, none finished, context churned on every switch. Cause: every new arrival preempts the current task. Fix: a preemption margin (as above), and preemption only at safe points: between tool calls, never in the middle of a multi-step side effect. Record parked progress in state so the resumed task does not restart from zero.

3. **Scores frozen at enqueue.** Symptom: a task whose deadline is now in five minutes sits behind tasks scored yesterday. Cause: the score was computed once and stored. Fix: store inputs, compute scores at pick time. If you cache scores for speed, invalidate on every tick and on every input change.

4. **The model picks.** Symptom: the same inbox produces different orders on different runs and no one can explain any of them. Cause: the prompt says "decide what to work on first" and the transcript is the queue. Fix: the model estimates inputs with a typed schema, code computes and logs the score, and the queue is a state key with a reducer. Determinism and explainability come from that split.

5. **Duplicate tasks after retries.** Symptom: the same "notify 14 customers" task appears three times and customers get three emails. Cause: a `work` node that enqueued a discovered task, then crashed before its checkpoint, was retried and enqueued again. Fix: deterministic task ids (hash of source event and intent), an upsert reducer as above, and idempotency keys on the side-effecting tools. 36. Durable Agent Orchestration goes deeper.

6. **Blocked tasks that block forever.** Symptom: the deadline task never runs because its blocker failed and was marked `failed`, not `done`. Cause: dependency resolution only checks `done`. Fix: a `failed` blocker either unblocks with a note ("prerequisite failed, proceed with degraded data") or escalates the dependent task to a human. Never leave it silently pending.

7. **Risk treated as a score penalty.** Symptom: a high-risk refund action scores low, waits, and then runs at 3am when the queue is empty. Cause: risk was a divisor instead of a gate. Fix: high-risk tasks go to a `waiting_approval` status and an `interrupt()`; they never run unattended regardless of score.

## What changed by 2026

- **Queues became state, not prose.** The early pattern was to keep a to-do list in the model's context and hope it followed it. The harness pattern now is an explicit todo or task list held in state or in a file, updated through tools, and rendered back to the model when needed. LangChain's `TodoListMiddleware` and Deep Agents' planning tools implement exactly this; current Deep Agents docs describe the todo tool as opt-in so you only pay its context cost when the task warrants it. The mechanism is the reducer-backed `tasks` key you built above.

- **Durable execution made "continuously running" cheap.** With a checkpointer keyed by thread, a scheduler does not need a daemon; it needs a durable state and something that invokes it (a cron, a webhook, an event). LangGraph's durability modes and the hosted runtimes from LangChain, OpenAI's Agents API, and Anthropic's Managed Agents all frame the agent as a durable session plus a stateless loop. Prioritization lives in the session's state.

- **The model estimates, structured output enforces the shape.** Provider-native structured outputs and constrained decoding mean the triage schema in the live cell is guaranteed to parse. The model still gets the *value* wrong sometimes, which is why the estimate is an input to a logged deterministic score rather than the decision.

- **Attention is a budget too.** Anthropic's context-engineering guidance treats the context window as a finite resource that degrades as it fills. That reframes prioritization: it is not only which task to do next but which information deserves the model's attention on this step. 34. Context Engineering is that lesson.

- **Preemption got a protocol shape.** Long-running tasks over MCP (through its Tasks extension) and A2A carry task states like working, input-required, and canceled, so an orchestrating agent can park or cancel a remote task instead of only its own. Current docs describe these as evolving; the scheduler logic is unchanged, the task handle just points across a network.

## Interview questions

**Q1. How does an agent decide what to work on next when it has many pending tasks?**

Strong: I separate estimation from decision. The model, or policy tables, estimate the inputs: importance, urgency including deadlines, expected value, cost, and risk. Code computes a score, something like importance times urgency plus expected value over cost, and picks the maximum among tasks whose dependencies are met. Scores are recomputed at pick time, not stored, because urgency changes. Risk is a gate, not a term: high-risk actions route to approval regardless of score. Every pick is logged with its decomposed inputs so the order is explainable. The queue is a state key with an upsert reducer, persisted by a checkpointer, so it survives across invocations.

Weak: "I put the tasks in the prompt and ask the model to prioritize."

**Q2. How do you prevent low-priority but important work from starving?**

Strong: Aging. Urgency includes a term that grows with time waiting, so anything pending long enough eventually outranks fresh medium-priority arrivals. I also track the age of the oldest pending task as a metric with an alert, because starvation is invisible at any single tick. In some systems I reserve a fraction of capacity for the lowest quartile. The failure I am guarding against is that "always busy" looks healthy while eval expansion and documentation never happen.

Weak: "Add a priority field and sort by it."

**Q3. A P0 arrives while the agent is halfway through another task. What happens?**

Strong: Preemption, but with rules. New arrivals are re-scored against the current task; if the new one wins by more than a margin, the current task is parked with its progress recorded in state and the new one runs. The margin prevents thrashing. Preemption only happens at safe points, between tool calls and never in the middle of a multi-step side effect, because a half-done refund is worse than a late one. The parked task keeps its partial results so resuming does not repeat work or repeat side effects.

Weak: "The agent finishes what it is doing first."

**Q4. Why should the model not make the final scheduling decision?**

Strong: Three reasons. Determinism: the same inbox should produce the same order, and model prose does not. Explainability: a logged score decomposed into inputs answers "why this first" in a way "the model felt it was urgent" cannot. Enforceability: policy constraints like "high-risk actions always go to approval" and "deadline tasks never wait behind unblocked low-value work" must hold every time, and a prompt is a suggestion. The model is excellent at estimating the inputs from messy text, so I use it there with a typed schema, and the code owns the decision.

Weak: "The model is smart enough to prioritize."

**Q5. How do you handle dependencies between tasks in the queue?**

Strong: Each task carries a `blocked_by` set. Only tasks whose blockers are done are candidates. Then I let urgency flow upstream: a blocker inherits the best score of the tasks waiting on it, otherwise a cheap prerequisite sits behind flashy work while the deadline task it unblocks goes late. Failed blockers need an explicit policy, either unblock with a degraded-data note or escalate, because a dependent task silently pending forever is the worst outcome. Dependency edges and statuses live in state with the tasks so they survive restarts.

Weak: "Do the tasks in the order they were created."

**Q6. Design the queue for an agent that runs continuously for weeks.**

Strong: The queue is state, not a process. Tasks with deterministic ids live in a state key with an upsert reducer, persisted by a checkpointer per thread so any invocation, from a cron, a webhook, or a user, sees the same queue. Work generates work through an enqueue tool, never through prose. Side-effecting tools carry idempotency keys because a retry after a crash will re-run the node. Scores are computed at pick time from stored inputs. High-risk tasks pause on an interrupt and resume with a decision. Scoring policy that is shared across threads, like account tiers and SLA tables, lives in a store. Metrics: oldest pending age, preemption rate, tasks completed per hour, approval wait time.

Weak: "A while-true loop that polls a database."

**Q7. Your agent drifts after a long run and confidently works on the wrong thing. Diagnose it.** (interview repository: Agents and Tool Use)

Strong: I would look at the scheduling log first, because drift is usually a prioritization failure that looks like a reasoning failure. Common causes: the goal or the queue lived only in the transcript and was compacted or crowded out, so the model reconstructed a plausible but wrong task; scores were frozen at enqueue so stale urgency dominated; a discovered subtask was enqueued at the parent's priority and became the main event; or the model was allowed to pick and its pick was never checked against the goal. Fixes: keep the goal and the queue in state and re-inject them each step, recompute scores at pick time, have a goal-monitoring node compare the current task to the original objective, and alert when the agent has worked on a task with no logged score.

Weak: "Add a reminder to the prompt to stay focused."

## Build it yourself

Extend the LangGraph scheduler into an approval-aware, preemptive queue.

1. Add `risk` to the task dicts and a `gate` node between `pick` and `work`. When the picked task has `risk == "high"`, the node calls `interrupt({"task": task})` and, on resume with `{"approve": True}`, proceeds to `work`; on `{"approve": False}` it marks the task `rejected` and returns to `pick`.
2. Add a `parked` status and a `progress` field. Make `work` return after one model turn instead of looping, and make `pick` implement the preemption margin from the from-scratch scheduler.
3. Add a `failed` status: when a blocker is `failed`, its dependents become runnable but receive a `degraded` flag in state.
4. Record `oldest_pending_age` in the log on every pick.

Acceptance checks:

- Invoking with a high-risk task pauses at `gate`; `graph.get_state(config).next` contains `"gate"`, and `Command(resume={"approve": False})` results in that task having status `rejected` and never calling the model.
- A task that arrives with a score exceeding the current task's by the margin is picked next and the previous task has status `parked` with nonzero `progress`.
- A dependent of a `failed` blocker completes with `degraded == True`.
- Across two invocations the total number of model turns equals the number of scripted turns (no task is worked twice).

## Key takeaways

- Prioritization is scheduling: score by importance, urgency, expected value, and cost; treat risk as a gate that routes to approval.
- The model estimates the inputs, ideally through a typed schema; deterministic code makes and logs the decision.
- Recompute scores at pick time; frozen scores are the most common cause of missed deadlines.
- Aging prevents starvation, a preemption margin prevents thrashing, and blocker inheritance keeps cheap prerequisites from hiding behind flashy work.
- The queue is state with an upsert reducer, persisted by a checkpointer, so a continuously running agent is a durable state, not a daemon.
- Work generates work through an enqueue tool with deterministic ids and idempotent side effects, never through prose.
- Every pick gets a log row with its decomposed score; that row is the answer to "why did the agent do that first."
