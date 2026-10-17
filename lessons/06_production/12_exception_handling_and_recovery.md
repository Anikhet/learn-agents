# 12. Exception Handling and Recovery

*Track A (agent engineering). Book: Chapter 12 (Exception Handling and Recovery). Prerequisites: 00. What an agent actually is, 05. Tool Use, 06. Planning.*

An agent that never fails is an agent that has never been deployed. Tools time out. APIs return 503. The model emits arguments that fail validation. A credential expires halfway through a run. The question is never whether failures happen but what the software around the model does in the next hundred milliseconds. This lesson builds that software: a taxonomy that decides which failures to retry, a retry that does not stampede the downstream service, fallbacks and replanning when retries are exhausted, and escalation when the machine should stop guessing. Then it shows how LangGraph's `RetryPolicy`, `error_handler`, and `ToolNode` error handling implement the same ideas.

## The problem

The running example: the support agent receives "Why is Acme complaining about their invoice?" It plans to look up the account in the CRM, fetch the Stripe invoice, search Slack, and draft a reply. Here is what happens in a real week:

```text
Monday     stripe.get_invoice  -> 503 Service Unavailable (for 4 seconds)
Tuesday    crm.lookup          -> 401 Unauthorized (token rotated overnight)
Wednesday  slack.search        -> 429 Too Many Requests
Thursday   model emits         -> get_invoice(invoice_id="inv 4471 maybe?")  (fails schema)
Friday     sql.query           -> hangs 30 s, returns 200 rows out of 800, then times out
```

The naive agent loop treats all five identically:

```python
try:
    result = tool.invoke(args)
except Exception as exc:
    raise   # the run dies, the user sees a stack trace
```

Or slightly less naively, it treats all five identically the other way:

```python
for attempt in range(5):
    try:
        return tool.invoke(args)
    except Exception:
        time.sleep(1)
```

Both are wrong for the same reason: they do not know what kind of failure they have. Retrying Monday's 503 is right. Retrying Tuesday's 401 five times is pointless and burns four seconds. Retrying Wednesday's 429 with a fixed one-second sleep makes the rate limit worse. Thursday's failure is not the tool's fault at all; the model needs to see the validation message and try again with corrected arguments. Friday is the most dangerous: a timeout on a partial result is not the same as "no result," and if that query had been a write, retrying it could apply the write twice.

```text
                    ┌──────────────┐
   tool call  ───►  │  exception   │
                    └──────┬───────┘
                           │  what kind?
        ┌──────────┬───────┴────────┬─────────────┬────────────┐
        ▼          ▼                ▼             ▼            ▼
    transient   permission     validation     timeout      partial
   (503, 429,   (401, 403)    (bad args,     (unknown     (some rows,
    conn reset)                bad schema)    outcome)     some pages)
        │          │                │             │            │
     retry w/   do not retry;   feed error     retry ONLY   keep what you
     backoff    escalate or     to the model   if idempotent have; mark
     + jitter   switch tool     for repair     else escalate incomplete
```

The book's chapter on exception handling makes the point that recovery is a design concern, not an afterthought: logging, retries, fallbacks, graceful degradation, and escalation are separate mechanisms that answer different questions. The distributed-systems reader already knows most of this from building services. What is new with agents is that one participant in the recovery loop is a model, and the model is a legitimate recovery mechanism. An error message, fed back as an observation, often lets the model pick a different tool or reformulate a query. That is a capability an ordinary service does not have, and it is also a trap, because a model will happily retry a forbidden action forever if you let it.

## Build it without a framework

Every offline cell uses the scripted model from STYLE.md. The script replays fixed turns so the recovery logic, not the model's judgment, is what we test. A live cell at the end runs the same graph against a real model.

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

### The taxonomy is code, not a comment

The first thing to build is a classifier from exception to error class. Everything downstream (retry, fallback, escalate) reads the class, never the raw exception. Eight classes cover almost everything an agent meets:

| Class | Examples | Retry? | Who recovers |
|---|---|---|---|
| `transient` | 503, 502, connection reset, 429 | yes, with backoff | software |
| `timeout` | deadline exceeded, no response | only if the call is idempotent | software, then model |
| `permission` | 401, 403, policy block | never | human or config |
| `validation` | bad arguments, schema mismatch, enum miss | no; repair | model |
| `not_found` | unknown id, empty result | no; replan | model |
| `model` | malformed tool call, refused, context overflow | sometimes (re-ask, compact) | software + model |
| `partial` | truncated pages, some shards failed | keep and mark | software + model |
| `budget` | step, token, or cost limit reached | never | escalate |

The next cell implements the classifier and a retry loop with exponential backoff and full jitter. Jitter matters: if fifty agents hit the same 503 at once and all retry after exactly one second, the downstream service sees the same spike again. Randomizing the wait spreads them out. We inject the sleep function and a seeded random source so the cell is deterministic; in production you pass `time.sleep` and `random.random`.

```{code-cell} python
import random
from dataclasses import dataclass, field


class Transient(Exception): ...
class RateLimited(Transient): ...
class Timeout(Exception): ...
class Forbidden(Exception): ...
class NotFound(Exception): ...
class ValidationFailed(Exception): ...


def classify(exc: Exception) -> str:
    """Exception -> error class. Everything downstream reads the class."""
    if isinstance(exc, Transient):        return "transient"
    if isinstance(exc, Timeout):          return "timeout"
    if isinstance(exc, Forbidden):        return "permission"
    if isinstance(exc, NotFound):         return "not_found"
    if isinstance(exc, ValidationFailed): return "validation"
    return "unknown"           # unknown is treated as non-retryable on purpose


@dataclass
class RetryPolicyFromScratch:
    max_attempts: int = 4
    base: float = 0.5          # seconds
    cap: float = 8.0
    retry_on: tuple = ("transient",)
    sleep: callable = lambda s: None          # injected; time.sleep in prod
    rng: random.Random = field(default_factory=lambda: random.Random(0))
    waits: list = field(default_factory=list)  # recorded for inspection

    def run(self, fn, *, idempotent: bool):
        # A timeout on a non-idempotent call is NOT retryable: the effect may have landed.
        retry_on = set(self.retry_on) | ({"timeout"} if idempotent else set())
        for attempt in range(1, self.max_attempts + 1):
            try:
                return fn()
            except Exception as exc:
                kind = classify(exc)
                if kind not in retry_on or attempt == self.max_attempts:
                    raise
                # exponential backoff with full jitter: U(0, min(cap, base * 2^attempt))
                wait = self.rng.uniform(0, min(self.cap, self.base * 2 ** attempt))
                self.waits.append(round(wait, 3))
                self.sleep(wait)


# A Stripe stand-in that is unavailable for its first two calls.
class FlakyStripe:
    def __init__(self): self.calls = 0
    def get_invoice(self, invoice_id):
        self.calls += 1
        if self.calls <= 2:
            raise Transient("503 Service Unavailable")
        return {"id": invoice_id, "amount_due": 4400, "status": "open"}


stripe = FlakyStripe()
policy = RetryPolicyFromScratch()
invoice = policy.run(lambda: stripe.get_invoice("in_4471"), idempotent=True)
print("invoice:", invoice, "| attempts:", stripe.calls, "| waits:", policy.waits)
assert stripe.calls == 3 and invoice["status"] == "open"
assert len(policy.waits) == 2 and policy.waits[0] <= 1.0 and policy.waits[1] <= 2.0

# The same policy refuses to retry a permission error: attempt count stays at 1.
attempts = {"n": 0}
def forbidden():
    attempts["n"] += 1
    raise Forbidden("401 token expired")
try:
    policy.run(forbidden, idempotent=True)
except Forbidden:
    pass
assert attempts["n"] == 1

# And refuses to retry a timeout when the call is a write.
attempts["n"] = 0
def slow_refund():
    attempts["n"] += 1
    raise Timeout("deadline exceeded after 30 s")
try:
    policy.run(slow_refund, idempotent=False)
except Timeout:
    pass
assert attempts["n"] == 1
print("permission not retried, non-idempotent timeout not retried: OK")
```

Two decisions in that cell are worth saying aloud in an interview. First, `unknown` is non-retryable. If you do not recognize the failure, retrying is a guess, and guesses with side effects are how you double-charge a customer. Second, the retry loop needs to know whether the call is idempotent. That is a property of the tool, so it belongs on the tool's registration, which the next cell does.

### Errors become observations, with a recovery policy per class

Now the agent loop. Each tool carries metadata: `idempotent` and `fallback`. When a tool fails after retries, the loop consults a recovery policy keyed by error class:

```text
transient (exhausted)  -> try fallback tool if any, else tell the model
permission             -> tell the model, and flag for escalation
validation             -> tell the model exactly what was wrong (repair)
not_found              -> tell the model (it may replan)
timeout (write)        -> STOP: escalate, outcome unknown
budget                 -> STOP: escalate with partial results
```

```{code-cell} python
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool


@tool
def crm_lookup(company: str) -> dict:
    """Look up a customer account in the CRM by company name."""
    raise Forbidden("403: CRM token lacks scope accounts:read")


@tool
def crm_search_notion(company: str) -> dict:
    """Fallback: find the customer record in the Notion accounts database."""
    return {"company": company, "stripe_customer": "cus_9f2", "plan": "enterprise"}


@tool
def stripe_get_invoice(invoice_id: str) -> dict:
    """Fetch a Stripe invoice by id such as in_1234."""
    if not invoice_id.startswith("in_"):
        raise ValidationFailed(f"invoice_id must look like in_XXXX, got {invoice_id!r}")
    return {"id": invoice_id, "amount_due": 4400, "status": "open", "attempts": 3}


TOOLS = {
    crm_lookup.name:        {"tool": crm_lookup,        "idempotent": True,  "fallback": "crm_search_notion"},
    crm_search_notion.name: {"tool": crm_search_notion, "idempotent": True,  "fallback": None},
    stripe_get_invoice.name:{"tool": stripe_get_invoice,"idempotent": True,  "fallback": None},
}


def execute_with_recovery(name, args, policy, trace):
    """Retry -> fallback -> observation. Returns (text, escalate_flag)."""
    spec = TOOLS[name]
    try:
        result = policy.run(lambda: spec["tool"].invoke(args), idempotent=spec["idempotent"])
        return str(result), False
    except Exception as exc:
        kind = classify(exc)
        trace.append((name, kind))
        if kind in ("transient", "permission") and spec["fallback"]:
            fb = spec["fallback"]
            trace.append(("fallback", fb))
            fb_result, _ = execute_with_recovery(fb, args, policy, trace)
            return (f"{name} failed ({kind}: {exc}); used {fb} instead -> {fb_result}",
                    kind == "permission")          # still flag permission problems for a human
        if kind == "validation":
            return f"{name} rejected the arguments. {exc}. Fix the arguments and call again.", False
        if kind == "timeout" and not spec["idempotent"]:
            return f"{name} timed out and may have executed. STOP and escalate.", True
        return f"{name} failed ({kind}): {exc}", kind == "permission"


def run_agent(model, user_text, max_steps=6):
    messages = [HumanMessage(user_text)]
    trace, escalations = [], []
    for _ in range(max_steps):
        ai = model.invoke(messages)
        messages.append(ai)
        if not ai.tool_calls:
            return messages, trace, escalations
        for call in ai.tool_calls:
            text, escalate = execute_with_recovery(call["name"], call["args"],
                                                  RetryPolicyFromScratch(), trace)
            if escalate:
                escalations.append(text)
            messages.append(ToolMessage(content=text, tool_call_id=call["id"]))
    raise RuntimeError("step budget exhausted")   # class: budget -> caller escalates


model = scripted(
    [("crm_lookup", {"company": "Acme"})],                     # 403 -> fallback to Notion
    [("stripe_get_invoice", {"invoice_id": "4471"})],          # validation -> model repairs
    [("stripe_get_invoice", {"invoice_id": "in_4471"})],       # success
    "Acme's invoice in_4471 is open for $44.00 after 3 failed payment attempts.",
)
messages, trace, escalations = run_agent(model, "Why is Acme complaining about their invoice?")

for m in messages:
    print(f"{type(m).__name__:12s} {(m.content or m.tool_calls)!s:.110s}")
print("trace:", trace)
print("escalations:", escalations)
assert trace == [("crm_lookup", "permission"), ("fallback", "crm_search_notion"),
                 ("stripe_get_invoice", "validation")]
assert len(escalations) == 1 and "403" in escalations[0]
assert "in_4471" in messages[-1].content
```

Notice what the model saw and what it did not. It saw a clear validation message and repaired its own arguments; that is the model as a recovery mechanism. It saw that the CRM was forbidden and that a fallback was used; it did not get to decide whether to retry the forbidden call, because the software refused. And the permission failure was recorded for a human even though the run succeeded, because a rotated token is an operational problem that will bite the next run too.

## How LangGraph does it

LangGraph splits the same machinery across three primitives, each mapping to something in the from-scratch code:

| From scratch | LangGraph | What it does underneath |
|---|---|---|
| `RetryPolicyFromScratch.run` | `RetryPolicy` on `add_node(..., retry_policy=)` | The Pregel executor wraps the node call in a retry loop: `initial_interval * backoff_factor ** attempt`, capped by `max_interval`, with optional jitter, up to `max_attempts`, only when `retry_on` says yes |
| `classify` + "unknown is not retryable" | `RetryPolicy(retry_on=...)` | Default `retry_on` retries connection errors and 5xx from `httpx`/`requests`; it explicitly refuses `ValueError`, `TypeError`, `LookupError`, `RuntimeError`, and friends. Your own exception types must be listed |
| the `except` in `execute_with_recovery` | `add_node(..., error_handler=)` | When retries are exhausted the handler receives the state plus a `NodeError` (which node, which exception) and returns an update or a `Command`, so the graph continues instead of raising |
| errors -> `ToolMessage` | `ToolNode(handle_tool_errors=...)` | Catches tool exceptions and writes a `ToolMessage` with the error text, so the model sees it as an observation |
| `max_steps` | `recursion_limit` in config | The step bound; exceeding it raises `GraphRecursionError` (class: budget) |

The next cell builds a small graph: a flaky Stripe node with a `RetryPolicy` that only retries `Transient`, an `error_handler` that records a degraded result when retries are exhausted, and a `ToolNode` that turns tool exceptions into observations. We keep `initial_interval` tiny so the cell is fast.

```{code-cell} python
from typing import Annotated, TypedDict
from operator import add
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.types import RetryPolicy, Command
from langgraph.errors import NodeError
from langgraph.graph.message import add_messages


class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    invoice: dict
    degraded: Annotated[list[str], add]
    stripe_calls: int


counters = {"stripe": 0}

def fetch_invoice(state: State):
    counters["stripe"] += 1
    if counters["stripe"] <= 2:
        raise Transient("503 Service Unavailable")     # first two attempts fail
    return {"invoice": {"id": "in_4471", "status": "open"}, "stripe_calls": counters["stripe"]}

def invoice_failed(state: State, error: NodeError):
    # Retries exhausted (or non-retryable): degrade instead of crashing the run.
    return Command(update={"invoice": {},
                           "degraded": [f"{error.node}: {type(error.error).__name__}: {error.error}"]})

@tool
def slack_search(query: str) -> str:
    """Search Slack messages."""
    raise Forbidden("403: bot not in channel #acme-support")

def call_model(state: State):
    # The scripted model asks for Slack once, then answers.
    return {"messages": [model.invoke(state["messages"])]}

def after_model(state: State):
    return "tools" if state["messages"][-1].tool_calls else END

model = scripted([("slack_search", {"query": "Acme invoice"})],
                 "Invoice in_4471 is open; Slack was unavailable to the bot.")

builder = StateGraph(State)
builder.add_node("fetch_invoice", fetch_invoice,
                 retry_policy=RetryPolicy(max_attempts=4, initial_interval=0.001,
                                          backoff_factor=2.0, jitter=True, retry_on=Transient),
                 error_handler=invoice_failed)
builder.add_node("model", call_model)
builder.add_node("tools", ToolNode([slack_search],
                 handle_tool_errors=lambda e: f"{type(e).__name__}: {e}. Do not retry; report it."))
builder.add_edge(START, "fetch_invoice")
builder.add_edge("fetch_invoice", "model")
builder.add_conditional_edges("model", after_model, {"tools": "tools", END: END})
builder.add_edge("tools", "model")
graph = builder.compile()

out = graph.invoke({"messages": [HumanMessage("Why is Acme complaining?")]},
                   config={"recursion_limit": 10})
tool_msgs = [m for m in out["messages"] if isinstance(m, ToolMessage)]
print("stripe attempts:", out["stripe_calls"], "| invoice:", out["invoice"])
print("tool observation:", tool_msgs[0].content)
print("degraded:", out.get("degraded", []))
assert out["stripe_calls"] == 3                       # retried twice, succeeded on the third
assert "Forbidden" in tool_msgs[0].content            # permission error became an observation
assert out.get("degraded", []) == []                  # handler not needed on this run

# Now exhaust the retries on a one-node graph: the error_handler degrades instead of raising.
counters["stripe"] = -10
single = StateGraph(State)
single.add_node("fetch_invoice", fetch_invoice,
                retry_policy=RetryPolicy(max_attempts=2, initial_interval=0.001, retry_on=Transient),
                error_handler=invoice_failed)
single.add_edge(START, "fetch_invoice"); single.add_edge("fetch_invoice", END)
degraded = single.compile().invoke({})
print("after exhaustion:", degraded["degraded"])
assert degraded["invoice"] == {} and "Transient" in degraded["degraded"][0]
```

Two details of `RetryPolicy` trip people up. The default `retry_on` treats `ValueError`, `RuntimeError`, `LookupError`, and `OSError` as non-retryable, which is the right default for exactly the reason our `classify` returns `unknown` for anything unrecognized. If your tool wraps a 503 in a `RuntimeError`, the node will not retry; either raise a distinct exception type or pass a callable `retry_on`. And `RetryPolicy` retries the whole node, so a node that makes a model call and then a write will repeat the model call too. Keep retried nodes small and single-purpose, which the from-scratch version made obvious: the retry wrapped one tool invocation, not the whole step.

A live cell shows the same graph against a real model, where the model has to read the Forbidden observation and decide on its own not to call Slack again.

```{code-cell} python live
import os
from langchain.chat_models import init_chat_model

live_model = init_chat_model(os.environ.get("COURSE_MODEL", "openai:gpt-5.6")).bind_tools([slack_search])

def call_live(state: State):
    return {"messages": [live_model.invoke(state["messages"])]}

live_builder = StateGraph(State)
live_builder.add_node("model", call_live)
live_builder.add_node("tools", ToolNode([slack_search],
                      handle_tool_errors=lambda e: f"{type(e).__name__}: {e}. Do not retry."))
live_builder.add_edge(START, "model")
live_builder.add_conditional_edges("model", after_model, {"tools": "tools", END: END})
live_builder.add_edge("tools", "model")
live_out = live_builder.compile().invoke(
    {"messages": [HumanMessage("Search Slack for Acme invoice complaints and summarize.")]},
    config={"recursion_limit": 8})
print(live_out["messages"][-1].content)
print("slack calls:", sum(len(m.tool_calls) for m in live_out["messages"] if isinstance(m, AIMessage)))
```

## What goes wrong in production

1. **Retrying a non-idempotent write after a timeout.** Symptom: duplicate refunds, duplicate Slack posts. Cause: the retry loop does not know the tool is a write, and a timeout is treated like a connection error. Fix: idempotency is a property registered on the tool; timeouts on writes are never retried blindly. Give every write an idempotency key (see 36. Durable Agent Orchestration) so a retry is safe, then and only then mark it retryable.

2. **Retry storms.** Symptom: a downstream 503 lasts three times longer than it should. Cause: fixed-interval retries across many concurrent agents synchronize into waves. Fix: exponential backoff with jitter (`RetryPolicy(jitter=True)` does this), a cap on total retry time, and a circuit breaker that stops calling a tool that has failed N times in a window and reports it as unavailable instead.

3. **The model retries what the software refused.** Symptom: the trace shows the same forbidden call ten times. Cause: the error text says "failed" but not "do not retry," and the model's instinct is to try again. Fix: actionable error text that names the class and the next step ("403; this credential cannot read accounts; use another source or report"). Anthropic's tool-design guidance calls this out: error messages should tell the model what to do, not just what happened. Also cap repeated identical calls in code.

4. **Swallowed errors.** Symptom: the final answer is confident and wrong; the CRM lookup silently returned `{}`. Cause: a tool caught its exception and returned an empty result to "be safe." Fix: errors must reach the observation as errors. Empty is a value; failure is a failure. Partial results are labelled partial ("200 of 800 rows; query timed out").

5. **Retrying the wrong unit.** Symptom: a retry re-runs a model call, doubling cost and sometimes producing a different tool call than the one that failed. Cause: the retry policy wraps a node that does several things. Fix: one side effect per retried node. In LangGraph, put the model call and the tool execution in separate nodes; put `RetryPolicy` on the tool node only.

6. **Validation errors handled by software instead of the model.** Symptom: a hard-coded "argument fixer" grows into a second parser. Cause: engineers patch the arguments rather than returning the validation message. Fix: schema validation errors are the model's to repair; feed the exact pydantic message back and re-ask, with a bounded repair count (see 24. Structured Output).

7. **No escalation path.** Symptom: the agent times out after 40 minutes of trying. Cause: every recovery branch loops back to the model; none ends in a human. Fix: `permission`, `budget`, and non-idempotent `timeout` classes terminate the loop with a structured escalation record (what was attempted, what is known, what is unknown). Route it to a queue a human reads (see 13. Human in the Loop).

8. **Losing the partial work on failure.** Symptom: a crash at step 9 of 10 means starting from step 1. Cause: state lives only in the process. Fix: checkpoint every step so recovery resumes rather than restarts; that is what LangGraph's checkpointer and 36. Durable Agent Orchestration are for.

## What changed by 2026

The mechanics of retry and backoff have not changed in a decade. What changed is where recovery lives and who participates in it.

**Recovery moved into the harness.** Anthropic's engineering guidance on managed agents describes a session that is an append-only event log, a stateless harness loop, and an interchangeable sandbox, with the explicit design rule that sandbox crashes surface to the model as tool errors rather than as harness failures. That is the "errors become observations" pattern promoted from a `try/except` in your loop to an architectural invariant. The OpenAI Agents API public beta, as its current docs describe it, takes on orchestration, context compaction, and recovery on the hosted side. In both cases the application developer's job shifts from writing retry loops to declaring policies.

**Tool errors are part of tool design.** The "Writing effective tools for agents" guidance (2025) treats error text as a first-class part of the tool contract: actionable, specific, and written so the model can act on it. Teams now evaluate tool error paths in their agent evals, not just the happy path, tracking error counts alongside accuracy and tool-call counts.

**Partial results are the norm, not the exception.** With MCP's current revision, tool listings and results carry cache metadata and servers are expected to paginate and truncate. Long-running tool work is exposed through the Tasks extension, where a result can be `input_required` rather than complete. An agent harness therefore has to treat "not finished yet" and "finished with some of it" as ordinary states with their own handling, which is the `partial` class in our taxonomy.

**Reasoning models change the repair loop.** Models that reason before acting are better at reading a validation error and correcting arguments in one pass, which reduces repair iterations. They do not change the rule that the repair budget is bounded in code. And the visible reasoning text is not a reliable record of why a retry was chosen; verify outcomes, not explanations.

**LangGraph made the primitives declarative.** In the installed LangGraph 1.2, `add_node` accepts `retry_policy`, `error_handler`, `cache_policy`, and `timeout`. That is the taxonomy from this lesson expressed as node configuration: which exceptions to retry, what to do when retries are exhausted, and how long a node may run.

## Interview questions

**Q1. A tool call fails inside an agent loop. Walk me through what should happen.**

Strong: First the exception must not escape the loop; the loop catches it and classifies it. The class decides the response. Transient failures such as 5xx or connection resets are retried with exponential backoff and jitter, bounded in attempts and total time. Permission failures are never retried; they are recorded for escalation and the model is told the source is unavailable. Validation failures are returned to the model verbatim so it can repair its arguments, with a bounded repair count. Timeouts are retried only if the call is idempotent, because the effect may already have landed. Whatever the outcome, the model receives a precise observation with a suggested next step, and if the class is one that a machine should not guess about, the loop terminates with a structured escalation.

Weak: "Wrap it in try/except and retry a few times."

**Q2. Why is a timeout different from a 503?**

Strong: A 503 tells you the request was not processed, so a retry is safe. A timeout tells you nothing about the outcome; the server may have processed the request and the response was lost. For a read that does not matter, retry. For a write it is the classic at-least-once problem: a retry can apply the effect twice. So the rule is that timeouts are retryable only when the operation is idempotent, which usually means the write carries an idempotency key the server deduplicates on. Without that key, a timed-out write ends the run with an escalation that says the outcome is unknown.

Weak: "Both are network errors, so retry both."

**Q3. Why add jitter to backoff?**

Strong: Because failures are correlated. When a dependency goes down, every agent instance that was calling it fails at the same moment, and if they all wait exactly one second they all retry at the same moment, recreating the spike that caused the failure. Randomizing each wait within the backoff window spreads the retries over time so the dependency can recover. Full jitter, a uniform draw between zero and the exponential cap, is the standard choice. LangGraph's `RetryPolicy` has `jitter=True` for this reason.

Weak: "To make it look more natural."

**Q4. How do you stop the model from retrying something the software has refused?**

Strong: Three layers. The error text names the class and the next action, for example "403: this credential cannot read accounts; do not retry; use another source or report." The software tracks repeated identical tool calls and refuses after a small count with a message saying so. And permission failures are also recorded as escalations so a human fixes the root cause. The model is a good recovery mechanism for validation and not-found errors, where reformulating helps, and a bad one for permission errors, where nothing it does can help.

Weak: "Tell it in the system prompt not to retry."

**Q5. What is the role of fallbacks and replanning versus retries?**

Strong: Retries handle the same action failing for a transient reason. Fallbacks handle the same goal through a different means, such as searching Notion when the CRM is forbidden, or using a cheaper model when the primary is rate limited. Replanning handles the case where the failure changes what is possible, so the plan itself needs revision; a not-found on the invoice id means the whole approach of "check the invoice" needs a new first step. Retries are software's job. Fallbacks can be declared per tool. Replanning is the model's job, given a precise observation of what failed.

Weak: "If it fails, try a different tool."

**Q6. What does LangGraph's RetryPolicy do, and what would you warn a teammate about?**

Strong: It wraps a node's execution in a retry loop with exponential backoff, a cap, optional jitter, and a `retry_on` predicate. Two warnings. The default predicate deliberately refuses common Python exceptions like `ValueError` and `RuntimeError`, so if your tool wraps a transient failure in one of those, it will not retry; raise a distinct type or pass your own predicate. And the unit of retry is the whole node, so a node that makes a model call and then a write will repeat the model call on retry, costing money and possibly changing the action. Keep retried nodes single-purpose and use `error_handler` for what happens when retries are exhausted.

Weak: "It retries the node when it fails."

**Q7. How do you handle partial results?**

Strong: Treat them as a distinct class, not as success or failure. Keep what was returned, mark it incomplete with the reason and the coverage, for example "200 of 800 rows; query timed out at 30 s," and let the model decide whether that is enough for the goal or whether to narrow the query. Never let a truncated result pass as complete, because the model will summarize it confidently. For paginated tools, make pagination explicit in the schema so the model knows there is more.

Weak: "Return whatever you got."

## Build it yourself

Extend the from-scratch loop in this lesson with a circuit breaker and a repair budget:

1. Add a `CircuitBreaker` that tracks failures per tool name; after 3 failures within one run the tool is "open" and any call returns the observation `"<tool> is unavailable (circuit open); do not retry"` without invoking it.
2. Bound validation repairs: after 2 consecutive `validation` failures on the same tool, return an observation that asks the model to try a different approach, and record an escalation.
3. Add a `partial` class: a `sql_query` tool that returns `{"rows": [...], "complete": False, "reason": "timeout"}`; the loop must label the observation `PARTIAL:` in the `ToolMessage`.
4. Return a summary object from `run_agent`: counts per error class, fallbacks used, escalations.

Acceptance checks:

- A scripted model that calls a permanently failing transient tool 5 times produces exactly 3 real invocations and 2 circuit-open observations.
- A model that sends bad arguments 3 times gets 2 validation observations and one "try a different approach" observation; the escalation list has one entry.
- The `sql_query` observation starts with `PARTIAL:` and the final answer cell asserts the model's answer mentions incompleteness.
- The summary's error counts sum to the number of `ToolMessage`s that contain an error.

## Key takeaways

- Classify before you recover: every response (retry, fallback, repair, escalate) is a function of the error class, never of the raw exception.
- Retry only transient failures, with exponential backoff and jitter, bounded by attempts and total time; unknown means non-retryable.
- A timeout is not a failure to execute; retry it only when the operation is idempotent.
- Errors become observations with actionable text; the model is a good recovery mechanism for validation and not-found, and a dangerous one for permission.
- Permission, budget, and unknown-outcome writes end the loop with a structured escalation to a human.
- LangGraph's `RetryPolicy`, `error_handler`, and `ToolNode(handle_tool_errors=)` are the taxonomy expressed as node configuration; keep retried nodes single-purpose.
- Partial results are a first-class state: keep them, label them, and never let them pass as complete.
