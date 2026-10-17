# 36. Durable Agent Orchestration

*Track A (agent engineering), with Track D (infra) overlap. Course addition; extends the book's Chapter 12 (Exception Handling and Recovery) and Chapter 13 (Human-in-the-Loop). Prerequisites: 00. What an agent actually is, 12. Exception Handling and Recovery, 13. Human in the Loop.*

An agent run that fails halfway through has already *done things*: moved money, posted to Slack, changed a CRM record. Restarting from the top repeats them. Abandoning the run leaves them half-finished. This lesson is about the third option: resume exactly where the run stopped, without repeating any effect that already happened. Distributed systems call this durable execution, and it is the difference between a demo agent and one you can let touch Stripe.

## The problem

The support/ops agent gets a ticket from Acme: "We were charged $120 twice on invoice in_acme_0142." The plan is short:

```text
1. fetch_invoice     Stripe read           (safe to repeat)
2. request_approval  post to Slack, wait   (a human may answer in 3 days)
3. issue_refund      Stripe write          (money moves)
4. post_slack        Slack write           (tell the account team)
5. update_crm        CRM write             (close the ticket)
```

The naive implementation is a Python function that calls these in order, holding everything in local variables. Now put it on real infrastructure, where pods get rescheduled, deploys roll, and a human takes a long weekend before approving.

```text
worker A                                      Stripe
────────                                      ──────
fetch_invoice ────────────────────────────►   GET invoice      ok
request_approval ... (waiting 3 days) ...
                   deploy restarts worker A   (local variables gone)
issue_refund  ────────────────────────────►   POST refund      ok, re_001 created
              ◄── response in flight ───
      ✗ OOM kill here                         (money has moved)

worker B picks up the ticket, knows nothing, starts at step 1
fetch_invoice → request_approval (asks the human AGAIN)
issue_refund  ────────────────────────────►   POST refund      ok, re_002 created
                                              Acme is refunded $240
```

Two separate failures are hiding in that picture.

**Lost progress.** The run's state lived in process memory. When the process died, so did the knowledge that step 1 and step 2 were done. Worker B redoes them, including asking the human a second time.

**Duplicated effects.** Even if you *had* saved progress after each step, there is a window between "Stripe executed the refund" and "we recorded that it did." A crash in that window is indistinguishable, from the worker's point of view, from a crash *before* the call. The only safe move is to call again, and calling again refunds twice.

Checkpointing solves the first. It cannot solve the second, however often you checkpoint, because the window always exists. **Idempotency** solves it: make the second call harmless. You need both, and interviewers probe exactly the seam between them.

```text
                 checkpoint after step 2          checkpoint after step 3
                          │                                │
  ─── step 2 done ────────┼──── call Stripe ──── Stripe ok ─┼─── step 4 ───
                          │     ◄──────── crash window ────►│
                          │  a crash anywhere in here means │
                          │  step 3 re-runs on resume       │
```

This is the core fact of the lesson:

```text
Execution is at-least-once.   Any step can run more than once.
Effects must be exactly-once. The world must see each effect once.
Idempotency is the bridge.    Same key, same result, no second effect.
```

"Exactly-once execution" does not exist across a network boundary. "Exactly-once *effects*" does, and it is built from at-least-once execution plus an idempotent receiver.

## Build it without a framework

Everything here is plain Python. The "Stripe" is a fake with the one property that matters: server-side idempotency keys. Real payment APIs, Stripe among them, accept an `Idempotency-Key` header and return the original result when a key is reused. A checkpointer is just a durable map from run id to the last committed state.

### The crash-and-resume runner

We will run the same three-step refund three times, with three different idempotency-key strategies, and crash every time in the worst possible place: after Stripe has executed the refund but before our checkpoint is written.

```{code-cell} python
import json
import uuid


class Crash(Exception):
    """Stands in for SIGKILL, OOM, a rolling deploy, or a lost pod."""


class FakeStripe:
    """Payments API with server-side idempotency: same key -> same result, no new effect."""

    def __init__(self):
        self.refunds = []            # money that actually moved
        self._by_key = {}

    def create_refund(self, invoice_id, amount_cents, idempotency_key=None):
        if idempotency_key is not None and idempotency_key in self._by_key:
            prior = self._by_key[idempotency_key]
            if (prior["invoice_id"], prior["amount_cents"]) != (invoice_id, amount_cents):
                raise ValueError("idempotency key reused with different parameters")
            return prior             # replay: original result, no second refund
        refund = {"id": f"re_{len(self.refunds) + 1:03d}",
                  "invoice_id": invoice_id, "amount_cents": amount_cents}
        self.refunds.append(refund)
        if idempotency_key is not None:
            self._by_key[idempotency_key] = refund
        return refund


class Journal:
    """A checkpointer is a durable map: run_id -> last committed state (serialized)."""

    def __init__(self):
        self.rows = {}

    def save(self, run_id, state):
        self.rows[run_id] = json.dumps(state)

    def load(self, run_id):
        raw = self.rows.get(run_id)
        return json.loads(raw) if raw else None


def refund_key(strategy, run_id, state):
    if strategy == "none":
        return None
    if strategy == "random":
        return str(uuid.uuid4())     # the classic bug: a fresh key on every attempt
    # Derived only from durable inputs, so every attempt computes the same key.
    return f"{run_id}:refund:{state['invoice_id']}"


def make_steps(stripe, strategy):
    def fetch_invoice(state, run_id):
        return {**state, "invoice_id": "in_acme_0142", "overcharge_cents": 12_000}

    def issue_refund(state, run_id):
        key = refund_key(strategy, run_id, state)
        refund = stripe.create_refund(state["invoice_id"], state["overcharge_cents"],
                                      idempotency_key=key)
        return {**state, "refund_id": refund["id"]}

    def post_slack(state, run_id):
        return {**state, "slack": f"Refund {state['refund_id']} issued to Acme"}

    return [("fetch_invoice", fetch_invoice), ("issue_refund", issue_refund),
            ("post_slack", post_slack)]


def run(run_id, steps, journal, crash_after=None):
    state = journal.load(run_id) or {"done": []}      # resume, or start fresh
    for name, step in steps:
        if name in state["done"]:
            continue                                   # committed before the crash
        state = step(state, run_id)                    # the effect happens here
        if name == crash_after:
            raise Crash(f"died after {name} ran, before its checkpoint")
        state["done"] = state["done"] + [name]
        journal.save(run_id, state)                    # commit point
    return state


def crash_then_resume(strategy):
    stripe, journal = FakeStripe(), Journal()
    steps = make_steps(stripe, strategy)
    try:
        run("ticket-8812", steps, journal, crash_after="issue_refund")
    except Crash:
        pass
    final = run("ticket-8812", steps, journal)         # a new worker, same journal
    return len(stripe.refunds), final


results = {s: crash_then_resume(s) for s in ["none", "random", "deterministic"]}
for strategy, (n, final) in results.items():
    print(f"{strategy:13s} refunds issued: {n}   final: {final['slack']}")

assert results["none"][0] == 2            # checkpointing alone does not prevent the double refund
assert results["random"][0] == 2          # a key that changes per attempt is no key at all
assert results["deterministic"][0] == 1   # same key on retry -> Stripe replays the original
assert results["deterministic"][1]["refund_id"] == "re_001"
assert results["deterministic"][1]["done"] == ["fetch_invoice", "issue_refund", "post_slack"]
```

Three lessons are packed into those asserts.

1. **Checkpointing gave us resume.** On the second run `fetch_invoice` did not execute again.
2. **Checkpointing did not give us exactly-once.** With no key, the refund step re-ran and moved money twice.
3. **The key must be deterministic across attempts.** A `uuid4()` generated *inside* the step is a new key per attempt, which is no key at all. Derive it from something durable *before* the attempt started (run id plus business intent, "refund invoice X"), or generate it in an earlier step and checkpoint it.

Notice the `ValueError` branch in `FakeStripe`. Real providers refuse a reused key with different parameters, because that is almost always a bug: the retry computed a different amount. If a model computes the amount in the same step that issues the refund, a retry can compute a different number. Decide in one step, checkpoint, and act in the next step from the checkpointed value.

### When you cannot undo: sagas and compensation

Idempotency handles "did it happen once?" It does not handle "step 3 of 5 failed permanently; now what?" A database transaction would roll back. External systems do not share a transaction. Stripe, the CRM, and Notion will not two-phase-commit with your agent.

The distributed-systems answer is the **saga** (Garcia-Molina and Salem, "Sagas", 1987): a sequence of local steps, each paired with a **compensating action** that semantically undoes it. If a later step fails, run the compensations for the completed steps in reverse order.

```text
forward:      crm.set_plan ──► stripe.credit_note ──► notion.provision ✗
compensate:   crm.restore  ◄── stripe.void_note   ◄──┘
```

Compensation is not rollback. The credit note existed for a few seconds, appears in Stripe's audit log, and may have triggered a webhook. Compensation brings the business state back to consistent; it does not erase history.

```{code-cell} python
class Systems:
    """The three external systems a plan upgrade touches."""

    def __init__(self):
        self.crm = {"acme": {"plan": "team"}}
        self.credit_notes = {}          # id -> status
        self.notion_pages = []
        self.log = []


def run_saga(sys_, steps):
    """Run steps; on failure, run completed steps' compensations in reverse."""
    completed = []
    for name, action, compensate in steps:
        try:
            result = action()
            completed.append((name, compensate, result))
            sys_.log.append(f"do   {name}")
        except Exception as exc:
            sys_.log.append(f"FAIL {name}: {exc}")
            for done_name, comp, res in reversed(completed):
                if comp is None:
                    sys_.log.append(f"skip {done_name} (no compensation: pivot step)")
                    continue
                comp(res)
                sys_.log.append(f"undo {done_name}")
            return "compensated"
    return "committed"


s = Systems()


def set_plan():
    previous = s.crm["acme"]["plan"]
    s.crm["acme"]["plan"] = "enterprise"
    return previous                        # what the compensation needs to restore


def restore_plan(previous):
    s.crm["acme"]["plan"] = previous       # idempotent: setting a value twice is harmless


def create_credit_note():
    note_id = "cn_0001"
    s.credit_notes[note_id] = "issued"
    return note_id


def void_credit_note(note_id):
    s.credit_notes[note_id] = "void"       # idempotent: voiding a void note is a no-op


def provision_notion():
    raise ConnectionError("Notion 503 after 5 retries")


status = run_saga(s, [
    ("crm.set_plan", set_plan, restore_plan),
    ("stripe.credit_note", create_credit_note, void_credit_note),
    ("notion.provision", provision_notion, None),
])
print(status)
print("\n".join(s.log))

assert status == "compensated"
assert s.crm["acme"]["plan"] == "team"                 # business state restored
assert s.credit_notes == {"cn_0001": "void"}           # not deleted: voided, auditable
assert [l.split()[0] for l in s.log] == ["do", "do", "FAIL", "undo", "undo"]
assert s.log[-2:] == ["undo stripe.credit_note", "undo crm.set_plan"]   # reverse order
```

Three design rules come out of this.

- **Compensations must themselves be idempotent and retryable.** A saga coordinator can crash in the middle of compensating, and it will run the compensation again on recovery.
- **Order steps so the irreversible one goes last.** A Slack message to a customer or an email cannot be unsent. Saga literature calls the point of no return the *pivot*: everything before it is compensatable, everything after it must be retryable until it succeeds. Put `post_slack` after the refund is confirmed, not before.
- **Capture what compensation needs at the time of the action.** `set_plan` returns the previous plan. Looking it up later would read a value someone else may have changed.

## How LangGraph does it

LangGraph's durable execution is the `Journal` above, done carefully. Compile a graph with a **checkpointer** and give each run a `thread_id`. After every **super-step** (one round of node executions), LangGraph serializes the channel values, which nodes run next, and any pending writes into a checkpoint row keyed by `(thread_id, checkpoint_id)`. On `invoke(None, config)` for the same thread, it loads the latest checkpoint and continues from the nodes listed in `next`.

```text
our Journal                       LangGraph
─────────────                     ───────────────────────────────────
run_id                            config["configurable"]["thread_id"]
journal.save after each step      checkpoint after each super-step
state["done"]                     snapshot.next  (what still has to run)
run(run_id, ...) again            graph.invoke(None, config)
json.dumps                        a serializer (`JsonPlusSerializer` by default)
dict in memory                    InMemorySaver / SqliteSaver / PostgresSaver
```

The granularity of "do not redo" is the node. A node that finished and was checkpointed will not run again. A node that was running when the process died *will* run again from its first line. So the design rule is: **one side effect per node, and every side-effecting node is idempotent.**

The cell below is the refund scenario done for real: the graph crashes inside `issue_refund` after Stripe has executed the refund, a "new process" compiles the graph again against the same checkpointer, and the run resumes.

```{code-cell} python
import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


class RefundState(TypedDict, total=False):
    ticket: str
    invoice_id: str
    overcharge_cents: int
    refund_id: str
    audit: Annotated[list, operator.add]    # append-only log across nodes


stripe = FakeStripe()
node_runs = {"fetch_invoice": 0, "issue_refund": 0, "post_slack": 0}
chaos = {"crash_in": "issue_refund"}         # fault injection, fires once


def fetch_invoice(state: RefundState):
    node_runs["fetch_invoice"] += 1
    return {"invoice_id": "in_acme_0142", "overcharge_cents": 12_000,
            "audit": ["fetched in_acme_0142"]}


def issue_refund(state: RefundState):
    node_runs["issue_refund"] += 1
    # Business-intent key: stable across retries, resumes, and even a new thread for the same ticket.
    key = f"{state['ticket']}:refund:{state['invoice_id']}"
    refund = stripe.create_refund(state["invoice_id"], state["overcharge_cents"],
                                  idempotency_key=key)
    if chaos["crash_in"] == "issue_refund":
        chaos["crash_in"] = None
        raise Crash("worker died after Stripe returned, before the node returned")
    return {"refund_id": refund["id"], "audit": [f"refund {refund['id']}"]}


def post_slack(state: RefundState):
    node_runs["post_slack"] += 1
    return {"audit": [f"slack: refund {state['refund_id']} issued"]}


def build_graph():
    b = StateGraph(RefundState)
    b.add_node("fetch_invoice", fetch_invoice)
    b.add_node("issue_refund", issue_refund)
    b.add_node("post_slack", post_slack)
    b.add_edge(START, "fetch_invoice")
    b.add_edge("fetch_invoice", "issue_refund")
    b.add_edge("issue_refund", "post_slack")
    b.add_edge("post_slack", END)
    return b


saver = InMemorySaver()                       # in production: PostgresSaver, a real database
config = {"configurable": {"thread_id": "ticket-8812"}}

graph = build_graph().compile(checkpointer=saver)
try:
    graph.invoke({"ticket": "ticket-8812", "audit": []}, config, durability="sync")
except Crash as exc:
    print("crashed:", exc)

snapshot = graph.get_state(config)
print("pending node after crash:", snapshot.next)
print("checkpointed values:     ", {k: v for k, v in snapshot.values.items() if k != "audit"})
assert snapshot.next == ("issue_refund",)          # fetch_invoice is committed; the refund is not
assert "refund_id" not in snapshot.values          # the checkpoint never saw the refund
assert len(stripe.refunds) == 1                    # ...but Stripe did

# "New process": a fresh graph object, same durable checkpointer, same thread_id.
graph_after_restart = build_graph().compile(checkpointer=saver)
final = graph_after_restart.invoke(None, config)   # None = no new input, continue the thread

print("node runs:", node_runs)
print("audit:    ", final["audit"])
print("refunds at Stripe:", [r["id"] for r in stripe.refunds])

assert node_runs == {"fetch_invoice": 1, "issue_refund": 2, "post_slack": 1}
assert len(stripe.refunds) == 1                    # re-executed node, single effect
assert final["refund_id"] == "re_001"
assert final["audit"] == ["fetched in_acme_0142", "refund re_001", "slack: refund re_001 issued"]
```

Read the asserts as a sentence: the read ran once, the refund *node* ran twice, the refund *effect* happened once, and the Slack post ran once after the resume. That is at-least-once execution turned into exactly-once effects by an idempotency key, which is precisely the from-scratch `deterministic` strategy with LangGraph doing the journaling.

In an incident, `snapshot.next` tells an operator where a thread is stuck, and `snapshot.tasks` carries the error, so a crashed thread is distinguishable from a waiting one.

LangGraph also stores **pending writes**. When several nodes run in parallel in one super-step and one of them fails, the writes of the siblings that succeeded are saved against the checkpoint. On resume only the failed node re-runs; the successful siblings are not repeated. Our `Journal` did not have that, and adding it is the Build-it-yourself exercise.

### Durability modes: when the checkpoint is written

The `durability` argument controls *when* the checkpoint write happens relative to the next step. Current LangGraph docs describe three modes, and the installed version defaults to `"async"`:

| Mode | Checkpoint write | Cost | What a hard crash can lose |
|---|---|---|---|
| `"sync"` | before the next step starts | slowest; one blocking write per step | nothing that completed |
| `"async"` | in the background while the next step runs | small overhead | the last step, if the process dies before the write lands |
| `"exit"` | only when the graph exits (finish, error, or interrupt) | fastest | every step since the run started |

Pick by what redoing a step costs: `"sync"` for a refund agent with slow, money-touching steps; `"exit"` for a short read-only graph where a rerun costs cents. Idempotency is required in every mode, because no mode closes the crash window inside a node.

### Time travel and forks

Every checkpoint is kept, not just the latest. `get_state_history(config)` walks them newest-first. Passing a historical checkpoint's config to `invoke` **replays** from that point: the nodes after it execute again, model calls included. `update_state(config, values, as_node=...)` writes a *new* checkpoint that branches off an old one, which is how you correct a bad intermediate value and re-run from there.

```text
cp0 ─► cp1 (fetched) ─► cp2 (refunded) ─► cp3 (slack posted)      original timeline
         │
         └─► cp1' (update_state: overcharge 9000) ─► ...            fork
```

The trap is that replay and forks re-execute side effects. Time travel rewinds *your* state. It does not rewind Stripe.

```{code-cell} python
history = list(graph_after_restart.get_state_history(config))
print("checkpoints, newest first:")
for snap in history:
    print(f"  step={snap.metadata.get('step'):>2}  next={snap.next}")

before_refund = next(s for s in history if s.next == ("issue_refund",))

# 1) Replay from the checkpoint before the refund: the node re-runs, Stripe dedupes.
replayed = graph_after_restart.invoke(None, before_refund.config)
assert replayed["refund_id"] == "re_001"
assert len(stripe.refunds) == 1
assert node_runs["issue_refund"] == 3              # the node really did run again

# 2) Fork with a different amount: same business key, different parameters.
fork_config = graph_after_restart.update_state(before_refund.config,
                                               {"overcharge_cents": 9_000},
                                               as_node="fetch_invoice")
try:
    graph_after_restart.invoke(None, fork_config)
    fork_outcome = "second refund issued"
except ValueError as exc:
    fork_outcome = f"blocked by provider: {exc}"
print("fork:", fork_outcome)
assert fork_outcome.startswith("blocked")          # the key protected us from a silent second refund
assert len(stripe.refunds) == 1

# 3) Durability modes, measured by how many checkpoints a clean run leaves behind.
counts = {}
for mode in ["sync", "async", "exit"]:
    cfg = {"configurable": {"thread_id": f"mode-{mode}"}}
    build_graph().compile(checkpointer=saver).invoke(
        {"ticket": f"mode-{mode}", "audit": []}, cfg, durability=mode)
    counts[mode] = len(list(saver.list(cfg)))
print("checkpoints per clean run:", counts)
assert counts["sync"] == counts["async"] > counts["exit"] == 1
```

The fork is the important result. Someone "fixed" the amount in the past; without a key the fork would have silently refunded again. With a business-intent key the provider refuses, which makes the right action explicit: a changed decision about money needs a new, reviewed key or a compensation, not a replay.

### Interrupt, and resume days later

Human approval is where durability stops being about crashes and becomes about time. `interrupt(payload)` inside a node saves a checkpoint and ends the invocation. Nothing is running while the human thinks: no thread, no process, no open connection. Days later, any worker compiles the graph, calls `invoke(Command(resume=decision), config)`, and the run continues. (13. Human in the Loop covers approval UX; here we care about the mechanics.)

The mechanic that bites people: **on resume, the interrupted node runs again from its first line.** `interrupt()` returns the resume value the second time instead of pausing. Any side effect *before* the `interrupt()` call executes twice.

```{code-cell} python
from langgraph.types import Command, interrupt


class FakeSlack:
    def __init__(self):
        self.messages = {}                      # dedupe key -> text

    def post(self, key, text):
        self.messages.setdefault(key, text)     # idempotent by client-supplied key
        return key


slack = FakeSlack()
approval_runs = {"n": 0}


def request_approval(state: RefundState):
    approval_runs["n"] += 1
    # This line executes on the first run AND on resume. The key makes it harmless.
    slack.post(f"{state['ticket']}:approval-request",
               f"Approve refund of ${state['overcharge_cents'] / 100:.2f} on {state['invoice_id']}?")
    decision = interrupt({"action": "refund", "invoice_id": state["invoice_id"],
                          "amount_cents": state["overcharge_cents"]})
    if not decision["approved"]:
        return {"audit": [f"rejected by {decision['approver']}"]}
    return {"audit": [f"approved by {decision['approver']}"]}


def build_approval_graph():
    b = StateGraph(RefundState)
    b.add_node("fetch_invoice", fetch_invoice)
    b.add_node("request_approval", request_approval)
    b.add_node("issue_refund", issue_refund)
    b.add_edge(START, "fetch_invoice")
    b.add_edge("fetch_invoice", "request_approval")
    b.add_edge("request_approval", "issue_refund")
    b.add_edge("issue_refund", END)
    return b


stripe = FakeStripe()                           # fresh ledger for this scenario
approval_saver = InMemorySaver()
cfg = {"configurable": {"thread_id": "ticket-9001"}}

paused = build_approval_graph().compile(checkpointer=approval_saver).invoke(
    {"ticket": "ticket-9001", "audit": []}, cfg)
print("paused with:", paused["__interrupt__"][0].value)
assert len(stripe.refunds) == 0                 # nothing irreversible before approval

# ... three days pass; the original process is long gone ...
later = build_approval_graph().compile(checkpointer=approval_saver)
assert later.get_state(cfg).next == ("request_approval",)
done = later.invoke(Command(resume={"approved": True, "approver": "maria@ops"}), cfg)

print("audit:", done["audit"])
print("approval node runs:", approval_runs["n"], "| slack messages:", len(slack.messages))
assert approval_runs["n"] == 2                  # the node re-ran from the top on resume
assert len(slack.messages) == 1                 # but the pre-interrupt post happened once
assert len(stripe.refunds) == 1 and done["refund_id"] == "re_001"
assert done["audit"][-2:] == ["approved by maria@ops", "refund re_001"]
```

Three rules follow. Put side effects *after* the `interrupt()`, or in a separate node, whenever you can. If one must come before, make it idempotent, as the Slack post is here. And keep the code between the node's start and the `interrupt()` deterministic, because a resumed node is matched to its saved resume value by the order of `interrupt()` calls; branching differently on resume confuses that matching.

### The functional API, and the other durable runtimes

LangGraph also has a functional API (`@entrypoint` and `@task` in `langgraph.func`). Its durable-execution guide recommends wrapping side effects and non-deterministic operations in `@task`, because a task's result is checkpointed and returned from the checkpoint on resume instead of being re-executed. That moves the "do not redo" granularity from the node down to the call.

That design is exactly how the dedicated durable-execution engines work, and it is worth knowing both because interviewers ask "why not just use Temporal?"

```text
                LangGraph                  Temporal                     DBOS
unit of replay  node / @task               activity                     @DBOS.step
state store     checkpointer (Postgres..)  Temporal server event        your Postgres
                                           history
recovery        load latest checkpoint,    replay workflow code         re-run workflow; completed
                run `next` nodes           against history; completed   steps return recorded
                                           activities return recorded   outputs
                                           results
constraint      one effect per node;       workflow code must be        workflow code must be
                idempotent nodes           deterministic (no raw time,  deterministic; steps are
                                           random, I/O in workflow)     at-least-once
fit             agent state, interrupts,   long business processes,     durable workflows as a
                streaming, time travel     many services, huge scale    library, no extra server
```

The shape of a Temporal-style agent, as a sketch (not run here):

```python
@workflow.defn
class RefundWorkflow:
    @workflow.run
    async def run(self, ticket: str) -> str:
        invoice = await workflow.execute_activity(fetch_invoice, ticket, ...)
        approved = await workflow.wait_condition(lambda: self.approved)   # can wait for days
        # Activities are at-least-once: the activity still sends an idempotency key.
        return await workflow.execute_activity(issue_refund, invoice, ...)
```

The engines disagree on mechanics and agree on the rule: execution may repeat, so effects carry keys. A reasonable split: LangGraph when the durable thing is the *agent* (message state, interrupts, streaming, time travel); Temporal or DBOS when it is a *business process* that calls an agent in one of its steps.

## What goes wrong in production

1. **Idempotency key generated inside the retried step.** Symptom: duplicate refunds only after crashes, never in tests. Cause: `uuid4()` or a timestamp inside the node yields a new key per attempt. Fix: derive the key from durable inputs (ticket, invoice, intent) or generate it in an earlier step and checkpoint it. Add a chaos test that kills every side-effecting node once, like the cell above.

2. **The provider's dedupe window is shorter than your run.** Symptom: an approval that took five days leads to a duplicate charge on a resume. Cause: provider idempotency keys expire (Stripe documents that keys can be pruned once they are at least 24 hours old), and your resume came later. Fix: keep your own **effect ledger** (a table of `key -> external id, status`) written in the same transaction as your business state, and check it before calling out. The provider's dedupe is a second line, not the first.

3. **Side effect before `interrupt()`.** Symptom: every approval request appears twice in Slack. Cause: the node re-runs from the top on resume. Fix: move effects after the interrupt or into their own node, or key them.

4. **Model decides and acts in the same node.** Symptom: a resumed run fails with "idempotency key reused with different parameters," or worse, with no key, refunds a different amount. Cause: the node calls the model (non-deterministic) and then the API; the retry gets a different answer. Fix: split into `decide_refund` (model output checkpointed) and `issue_refund` (pure function of checkpointed state).

5. **`InMemorySaver` in production.** Symptom: every deploy loses every paused approval. Cause: the checkpointer was process memory. Fix: a database-backed checkpointer (usually Postgres) with backups and retention. Checkpoints hold full state, including customer data: encrypt at rest and set TTLs.

6. **Code changed under a paused thread.** Symptom: a thread paused last week fails on resume with a missing node or a key error. Cause: the checkpoint's `next` names a node you renamed, or the state schema gained a required field. Fix: treat graph topology and state schema as a versioned API. Keep old node names routable, make new fields optional, and drain or migrate paused threads before removing nodes. Temporal has explicit workflow versioning for the same reason.

7. **Compensation that fails silently.** Symptom: an aborted upgrade leaves the customer on the enterprise plan. Cause: the coordinator crashed mid-compensation, or a compensation errored and was swallowed. Fix: checkpoint compensation progress, retry until success, page a human when one cannot succeed.

8. **Replay as a debugging tool on production threads.** Symptom: an engineer "replays from step 3 to see what happens" and a customer gets a second email. Fix: replay and fork against sandboxed or recording tools. The outside world does not rewind.

## What changed by 2026

- **Durable execution became a baseline expectation.** LangChain's docs describe LangGraph as a runtime for long-running, stateful agents with durable execution and compare it to Temporal and Inngest. Design reviews moved from "should we checkpoint?" to "which durability mode, which store, where are the keys?"
- **Explicit durability modes.** LangGraph exposes `durability="sync" | "async" | "exit"` per invocation (the older `checkpoint_during` flag is deprecated in the installed version), making the latency-versus-recovery trade-off a visible choice instead of a hidden default.
- **Agent frameworks plug into workflow engines.** Pydantic AI documents durable-execution integrations with Temporal, DBOS, Prefect, and Restate. The direction is converging: the agent loop runs inside a durable workflow, model calls and tool calls become recorded steps.
- **Harnesses assume checkpointers.** Deep Agents and `create_agent` with human-in-the-loop middleware require a checkpointer for approvals, because an approval is an interrupt and an interrupt is a checkpoint.
- **Fault-tolerance policies moved into the runtime.** Per-node `RetryPolicy` is standard in LangGraph; LangChain has written about further per-node timeout and error-handler policies, so check the current docs for exact names before relying on them. None of these remove the need for idempotent effects; they make retries more frequent, which makes idempotency more important.

## Interview questions

**Q1. How do you make a long-running agent survive a crash halfway through?** *(OpenAI, Cognition style)*

Strong: Two separate mechanisms. First, checkpoint state after every step to a durable store, keyed by a run or thread id, so a new worker can load the latest checkpoint and continue from the next step instead of restarting; in LangGraph that is a Postgres-backed checkpointer and `invoke(None, config)` on the same thread. Second, accept that the step that was running at the crash will run again, because there is always a window between an external effect and the checkpoint that records it. So every side-effecting step carries an idempotency key derived from durable inputs, and the external system or my own effect ledger dedupes. I would prove it with a chaos test that kills each side-effecting node once and asserts the effect count.

Weak: "Save the state to a database and restart from there."

**Q2. Is exactly-once execution possible? What do you actually guarantee?**

Strong: Not across a network boundary; a lost response is indistinguishable from a failed request, so the caller has to retry, which means at-least-once execution. What you can guarantee is exactly-once *effects*: at-least-once delivery plus an idempotent receiver. The receiver stores the key with the result and returns the stored result on a repeat, and it rejects a repeated key with different parameters. That is how payment APIs work, and it is how I make an agent's tools safe to retry.

Weak: "Use a transaction" or "Temporal gives you exactly-once."

**Q3. Your agent refunded a customer twice. Walk me through the likely causes.**

Strong: I would look for five things. No idempotency key on the refund call. A key generated inside the retried step, so each attempt had a new key. A key that expired at the provider before a long-paused run resumed. The refund issued before an `interrupt()` in the same node, so resume re-ran it. Or a time-travel replay or fork run against production tools. Then I check the trace for how many times the node executed versus how many refunds exist, and the fix is a business-intent key plus my own effect ledger checked before the call.

Weak: "The model made a mistake."

**Q4. How do you make an agent's actions reversible, or at least auditable?** *(Palantir)*

Strong: Most external actions cannot be rolled back, so I use sagas: every step is paired with a compensating action, such as void the credit note or restore the previous plan, captured at the time of the action with whatever it needs. On failure, compensations run in reverse order and are themselves idempotent and retried until they succeed. Irreversible steps like customer emails go last, after the pivot. For audit, every effect is written to an append-only ledger with the key, the approver, and the external id, and the checkpoint history gives the full state at each step.

Weak: "Keep logs so we can undo things manually."

**Q5. What happens to code inside a LangGraph node when you resume after an `interrupt()`?**

Strong: The node runs again from its first line. The checkpoint is at the node boundary, not at the `interrupt()` call, so on resume LangGraph re-executes the node and the `interrupt()` call returns the resume value instead of pausing. Anything before it runs twice: a Slack message, a model call, a database write. So I put side effects after the interrupt or in their own node, key any that must come before, and keep the pre-interrupt code deterministic because multiple interrupts in a node are matched to resume values by order.

Weak: "It continues from the line after `interrupt()`."

**Q6. Compare LangGraph's durability modes. Which would you use for a payments agent?**

Strong: `sync` writes the checkpoint before the next step starts, so a crash loses nothing that completed, at the cost of a blocking write per step. `async` writes in the background while the next step runs, which is cheaper but can lose the last step on a hard kill. `exit` writes only when the graph finishes, errors, or interrupts, which is fastest but offers no mid-run recovery. For payments I use `sync`, because re-running a step means another model call and another effect attempt, and I still require idempotency because no mode covers a crash inside a node.

Weak: "Always use the default."

**Q7. When would you use Temporal or DBOS instead of LangGraph's checkpointer?**

Strong: When the durable thing is a business process rather than an agent's state: many services, long timers, strict workflow versioning, high volume, where Temporal's event-history replay and operational tooling are the point. DBOS fits when I want durable workflows as a library on the Postgres I already run. LangGraph fits when the durable thing is the agent: message state, approval interrupts, streaming, time travel. Often both: a Temporal workflow owns order-to-cash and calls a LangGraph agent as one activity. Activities and nodes are both at-least-once, so idempotency keys stay mandatory.

Weak: "Temporal is more enterprise."

## Build it yourself

Extend the from-scratch `run` into a durable runner with **parallel steps and pending writes**, and a local **effect ledger**.

1. Allow a step to be a list of step functions that run "in parallel" (sequentially is fine) as one super-step. Each successful sub-step's output is recorded as a pending write in the journal immediately, before the super-step commits.
2. On resume, sub-steps with a pending write are not re-run; their saved output is merged instead.
3. Add an `EffectLedger` table in the same `Journal` (key -> external id). `issue_refund` checks the ledger before calling `FakeStripe`, and records the result after. Give `FakeStripe` a `forget_keys()` method that simulates provider key expiry.
4. Add a `durability` parameter to `run` with `"sync"` and `"exit"` behaviors.

Acceptance checks:

- A super-step with `[lookup_crm, lookup_stripe]` where `lookup_stripe` crashes once: after resume, `lookup_crm` ran exactly once.
- After `stripe.forget_keys()` and a crash-then-resume in `issue_refund`, `len(stripe.refunds) == 1` because the ledger short-circuited the call.
- With `durability="exit"` and a crash in step 3, resume starts from step 1; with `"sync"` it starts from step 3.
- A compensation that fails once is retried, and the saga ends `"compensated"` with every compensation applied exactly once.

## Key takeaways

- Checkpointing solves lost progress; it cannot solve duplicated effects, because there is always a window between an effect and the checkpoint that records it.
- Execution is at-least-once; exactly-once effects come from idempotency keys derived from durable inputs, never generated inside the retried step.
- In LangGraph the unit of "do not redo" is the node, so put one side effect in each node and make it idempotent.
- On resume after `interrupt()`, the node re-runs from its first line; effects before the interrupt run twice unless keyed.
- Pick the durability mode by what a redone step costs: `sync` for money and long steps, `exit` for cheap read-only graphs.
- Time travel rewinds your state, not the world; replays and forks re-execute side effects.
- When a step cannot be undone, use a saga: idempotent compensations in reverse order, with irreversible actions placed after the pivot.
