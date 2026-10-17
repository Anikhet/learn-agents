# 32. Observability and Tracing

*Track A (agent engineering) with Track D (infra). Course addition; supports the book's Chapter 19 (Evaluation and Monitoring). Prerequisites: 00. What an agent actually is, 05. Tool Use. Feeds 19. Evaluation and Monitoring and 37. Trajectory Evals and Statistics.*

You cannot evaluate, debug, or bill for what you did not record. An agent run is a tree of model calls and tool calls, each with inputs, outputs, timing, and cost. A trace is that tree, written down. This lesson builds a tracer from scratch so you know what every field means, then hooks the same tracer into LangGraph through callbacks, then shows LangSmith doing the same job remotely.

## The problem

A customer writes to support: "your bot told me my invoice was $300 when it is $1,200." The engineer on call opens the logs:

```text
2026-09-27 14:02:11 INFO  agent started
2026-09-27 14:02:14 INFO  tool called
2026-09-27 14:02:15 INFO  tool called
2026-09-27 14:02:19 INFO  agent finished
```

Four lines, no answer. Which tools? With which arguments? What did Stripe return? What did the model see when it wrote "$300"? Did it read the wrong invoice, or read the right one and misreport it? How much did this run cost, and is this happening to other customers right now?

Plain logging fails for agents because it flattens a tree into lines and drops the relationships. An agent run has structure:

```text
run: "what does Acme owe on in_1001?"                       4.1 s   $0.018
├── model call #1                                            1.2 s   $0.006
│     in:  [system, human]                    412 tokens
│     out: tool_call get_invoice(in_1002)      18 tokens    <- wrong id chosen here
├── tool get_invoice(invoice_id="in_1002")                   0.3 s
│     out: {"customer": "Globex", "amount": 300}
├── model call #2                                            1.9 s   $0.009
│     in:  [system, human, ai, tool]           470 tokens
│     out: "Acme owes $300."                    9 tokens
└── (end)
```

With that tree the bug is obvious in ten seconds: the model picked the wrong invoice id at model call #1, and everything downstream was faithfully wrong. Without it you are guessing.

### Vocabulary

Every tracing system, from OpenTelemetry to LangSmith, uses roughly the same three nouns:

| Term | Meaning | In the tree above |
|---|---|---|
| Trace (run) | One end-to-end request, identified by a trace id. | the whole tree |
| Span | One unit of work inside a trace, with a start, end, parent, and attributes. | each model call, each tool call |
| Attributes (metadata) | Key-value facts recorded on a span. | tokens, cost, arguments, result size |

LangSmith calls the whole tree a "trace" and each node a "run"; OpenTelemetry says "trace" and "span." Same structure.

### What to record, per span

```text
every span         name, kind (model / tool / chain), start, end, parent id, status, error
model call         model id, input messages (redacted), output, prompt tokens, completion tokens,
                   cached tokens, cost, stop reason, temperature, tool schemas sent
tool call          tool name, arguments (redacted), result (truncated), result size, error class,
                   retries, idempotency key
run level          user/tenant id, session/thread id, agent version, prompt version, total cost,
                   total latency, step count, final outcome, user feedback
```

The list is longer than most people's first tracer and shorter than what you eventually want. The rule: if you would need it to answer "why did this run do that" or "what did this run cost and who pays," record it.

## Build it without a framework

The tracer needs three things: a way to open and close spans, a way to know the current parent span without threading it through every function signature, and a store. `contextvars` solves the second one and is the same trick OpenTelemetry uses. A context manager gives you the first.

```{code-cell} python
import contextvars
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class Span:
    name: str
    kind: str                      # "model" | "tool" | "chain"
    trace_id: str
    span_id: str
    parent_id: str | None
    start: float
    end: float | None = None
    status: str = "ok"
    attributes: dict = field(default_factory=dict)

    @property
    def duration_ms(self):
        return round((self.end - self.start) * 1000, 1) if self.end else None


_current_span = contextvars.ContextVar("current_span", default=None)


class Tracer:
    """Minimal OpenTelemetry-style tracer. Spans nest via a context variable."""

    def __init__(self, clock=time.perf_counter):
        self.spans: list[Span] = []
        self.clock = clock

    @contextmanager
    def span(self, name, kind="chain", **attributes):
        parent = _current_span.get()
        span = Span(name=name, kind=kind,
                    trace_id=parent.trace_id if parent else uuid.uuid4().hex,
                    span_id=uuid.uuid4().hex[:12],
                    parent_id=parent.span_id if parent else None,
                    start=self.clock(), attributes=dict(attributes))
        token = _current_span.set(span)
        try:
            yield span                      # caller adds attributes as it learns them
        except Exception as exc:            # errors are recorded, then re-raised
            span.status = "error"
            span.attributes["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            span.end = self.clock()
            _current_span.reset(token)
            self.spans.append(span)

    def tree(self, trace_id):
        """Render one trace as an indented tree, children in start order."""
        by_parent = {}
        for s in sorted(self.spans, key=lambda s: s.start):
            if s.trace_id == trace_id:
                by_parent.setdefault(s.parent_id, []).append(s)
        lines = []

        def walk(parent, depth):
            for s in by_parent.get(parent, []):
                cost = s.attributes.get("cost_usd")
                lines.append(f"{'  ' * depth}{s.kind:5s} {s.name:22s} {s.duration_ms:7.1f} ms"
                             + (f"  ${cost:.4f}" if cost is not None else "")
                             + ("  ERROR" if s.status == "error" else ""))
                walk(s.span_id, depth + 1)
        walk(None, 0)
        return "\n".join(lines)


tracer = Tracer()
with tracer.span("outer") as outer:
    with tracer.span("inner", kind="tool", tool="get_invoice") as inner:
        pass
    try:
        with tracer.span("failing", kind="tool"):
            raise TimeoutError("stripe took too long")
    except TimeoutError:
        pass

print(tracer.tree(outer.trace_id))
assert inner.parent_id == outer.span_id and inner.trace_id == outer.trace_id
assert [s.status for s in tracer.spans] == ["ok", "error", "ok"]  # spans append on close
```

Read the `span` method carefully. The context variable is set on entry and reset on exit, so any code running inside the block, no matter how deep, sees this span as its parent. That is why you do not have to pass a `parent` argument around. Errors are recorded on the span and re-raised: tracing observes; it never swallows.

### Instrumenting the agent loop

Now wrap the lesson-00 loop. The model is scripted so the control flow is under study; a real model plugs into the same function. Two details matter for production and are easy to forget:

- **Cost is derived, not observed.** Providers return token counts; you multiply by a price table you maintain. Record both the tokens and the derived cost, and record which price table version you used.
- **Redaction happens before storage.** A trace is a copy of customer data. Emails, card numbers, and API keys must not reach the trace store. Redact at the boundary where the span is written.

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


import re
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

PRICE_PER_1K = {"prompt": 0.0025, "completion": 0.010}      # a price table you version and update
PII_PATTERNS = [(re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "<email>"),
                (re.compile(r"\b(?:\d[ -]?){13,16}\b"), "<card>"),
                (re.compile(r"sk_(live|test)_\w+"), "<secret>")]


def redact(text):
    for pattern, token in PII_PATTERNS:
        text = pattern.sub(token, text)
    return text


def approx_tokens(text):
    return max(1, len(text) // 4)            # stand-in for provider usage metadata


@tool
def get_invoice(invoice_id: str) -> dict:
    """Fetch a Stripe invoice."""
    return {"in_1001": {"customer": "Acme", "amount": 1200, "email": "billing@acme.example"},
            "in_1002": {"customer": "Globex", "amount": 300}}.get(invoice_id, {"error": "not found"})


@tool
def get_account(customer: str) -> dict:
    """Fetch a CRM account."""
    return {"Acme": {"plan": "enterprise", "owner": "jo@acme.example"}}.get(customer, {"error": "not found"})


TOOLS = {t.name: t for t in (get_invoice, get_account)}


def traced_agent(tracer, model, user_text, user_id, max_steps=6):
    with tracer.span("agent_run", user_id=user_id, agent_version="v3", input=redact(user_text)) as run:
        messages = [HumanMessage(user_text)]
        for step in range(max_steps):
            with tracer.span(f"model_call_{step + 1}", kind="model", model="scripted") as ms:
                prompt_text = " ".join(str(m.content) for m in messages)
                ai = model.invoke(messages)
                p_tok, c_tok = approx_tokens(prompt_text), approx_tokens(ai.content or str(ai.tool_calls))
                ms.attributes.update(prompt_tokens=p_tok, completion_tokens=c_tok,
                                     cost_usd=(p_tok * PRICE_PER_1K["prompt"] + c_tok * PRICE_PER_1K["completion"]) / 1000,
                                     output=redact(ai.content) or [c["name"] for c in ai.tool_calls])
            messages.append(ai)
            if not ai.tool_calls:
                run.attributes.update(steps=step + 1, outcome="answered", output=redact(ai.content))
                break
            for call in ai.tool_calls:
                with tracer.span(call["name"], kind="tool", args=call["args"]) as ts:
                    obs = str(TOOLS[call["name"]].invoke(call["args"]))
                    ts.attributes.update(result=redact(obs)[:200], result_chars=len(obs))
                messages.append(ToolMessage(content=obs, tool_call_id=call["id"]))
        else:
            run.attributes.update(steps=max_steps, outcome="step_budget_exceeded")
        # Roll derived costs up to the run so billing reads one number.
        run.attributes["cost_usd"] = sum(s.attributes.get("cost_usd", 0) for s in tracer.spans
                                         if s.trace_id == run.trace_id and s.kind == "model")
        return messages, run


tracer = Tracer()
model = scripted([("get_account", {"customer": "Acme"})],
                 [("get_invoice", {"invoice_id": "in_1001"})],
                 "Acme (enterprise) owes $1,200 on in_1001; contact billing@acme.example.").bind_tools(list(TOOLS.values()))
messages, run = traced_agent(tracer, model, "What does Acme owe? My email is jo@acme.example", user_id="u_42")

print(tracer.tree(run.trace_id))
print("run attributes:", {k: v for k, v in run.attributes.items() if k != "cost_usd"})
tool_spans = [s for s in tracer.spans if s.kind == "tool"]
assert [s.name for s in tool_spans] == ["get_account", "get_invoice"]
assert "@" not in run.attributes["input"] and "@" not in run.attributes["output"]   # redacted
assert all("@" not in str(s.attributes.get("result")) for s in tool_spans)
assert run.attributes["cost_usd"] > 0 and run.attributes["steps"] == 3
```

The final `cost_usd` on the run is a rollup of its model spans. Note that the *raw* tool result still goes back to the model (the model needs the email to do its job); only the *recorded* copy is redacted. Tracing changes what you store, never what the agent sees.

### Querying traces: attribution

A trace store is only useful if you can ask it questions. The two most common: "where does the time go" and "where does the money go," broken down by span name, tool, user, or agent version. Here the store is a list; in production it is a database with the same columns.

```{code-cell} python
from collections import defaultdict

# A second, slower and more expensive run for contrast.
slow = scripted([("get_invoice", {"invoice_id": "in_1002"})],
                [("get_invoice", {"invoice_id": "in_1001"})],
                [("get_account", {"customer": "Acme"})],
                "Acme owes $1,200 on in_1001.").bind_tools(list(TOOLS.values()))
traced_agent(tracer, slow, "Acme invoice?", user_id="u_7")


def attribute(spans, by):
    """Sum duration and cost grouped by a span attribute or field."""
    totals = defaultdict(lambda: {"count": 0, "ms": 0.0, "cost_usd": 0.0})
    for s in spans:
        key = getattr(s, by, None) or s.attributes.get(by)
        if key is None:
            continue
        t = totals[key]
        t["count"] += 1
        t["ms"] += s.duration_ms
        t["cost_usd"] += s.attributes.get("cost_usd", 0.0)
    return dict(totals)


runs = [s for s in tracer.spans if s.name == "agent_run"]
print("cost per user:", {u: round(v["cost_usd"], 4) for u, v in attribute(runs, "user_id").items()})
print("tool call counts:", {k: v["count"] for k, v in attribute([s for s in tracer.spans if s.kind == "tool"], "name").items()})
print("steps per run:", [r.attributes["steps"] for r in runs])

by_user = attribute(runs, "user_id")
assert by_user["u_7"]["cost_usd"] > by_user["u_42"]["cost_usd"]       # more model calls, more cost
assert attribute([s for s in tracer.spans if s.kind == "tool"], "name")["get_invoice"]["count"] == 3
```

These are the queries behind every agent dashboard: cost per tenant for billing, latency per span kind for optimization, tool-call counts per run for detecting loops. The `RollingMonitor` in 19. Evaluation and Monitoring reads exactly these fields.

## How LangGraph does it

LangGraph does not have its own tracer. It has something more general: every node, model call, and tool call fires **callbacks** through LangChain's callback system, and a tracer is just a callback handler that turns those events into spans. LangSmith's tracer is one such handler. Yours can be another.

The callback events map onto the spans you built by hand:

```text
from-scratch span                  LangChain callback events
─────────────────────────────      ───────────────────────────────────────────────────
tracer.span("agent_run")           on_chain_start / on_chain_end  (name "LangGraph")
tracer.span("agent", "chain")      on_chain_start / on_chain_end  (one per node)
tracer.span("model_call", "model") on_chat_model_start / on_llm_end (usage_metadata on the message)
tracer.span("get_invoice", "tool") on_tool_start / on_tool_end / on_tool_error
parent_id via contextvar           run_id / parent_run_id passed to every event
```

The handler below feeds LangGraph's events into the same `Tracer`. Since callbacks are not nested Python blocks, the handler opens spans manually and closes them on the matching end event, keyed by `run_id`.

```{code-cell} python
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode


class SpanCallbackHandler(BaseCallbackHandler):
    """Turns LangChain/LangGraph callback events into Tracer spans."""

    def __init__(self, tracer):
        self.tracer, self.open = tracer, {}

    def _start(self, run_id, parent_run_id, name, kind, **attrs):
        parent = self.open.get(parent_run_id)
        span = Span(name=name, kind=kind,
                    trace_id=parent.trace_id if parent else uuid.uuid4().hex,
                    span_id=uuid.uuid4().hex[:12], parent_id=parent.span_id if parent else None,
                    start=self.tracer.clock(), attributes=attrs)
        self.open[run_id] = span

    def _end(self, run_id, status="ok", **attrs):
        span = self.open.pop(run_id, None)
        if span:
            span.end, span.status = self.tracer.clock(), status
            span.attributes.update(attrs)
            self.tracer.spans.append(span)

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, name=None, **kw):
        self._start(run_id, parent_run_id, name or "chain", "chain")

    def on_chain_end(self, outputs, *, run_id, **kw):
        self._end(run_id)

    def on_chain_error(self, error, *, run_id, **kw):
        self._end(run_id, status="error", error=repr(error))

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kw):
        prompt_chars = sum(len(str(m.content)) for m in messages[0])
        self._start(run_id, parent_run_id, "chat_model", "model", prompt_tokens=approx_tokens(" " * prompt_chars))

    def on_llm_end(self, response, *, run_id, **kw):
        msg = response.generations[0][0].message
        usage = msg.usage_metadata or {}                      # real providers fill this in
        c_tok = usage.get("output_tokens") or approx_tokens(msg.content or str(msg.tool_calls))
        p_tok = usage.get("input_tokens") or self.open[run_id].attributes["prompt_tokens"]
        self._end(run_id, completion_tokens=c_tok, prompt_tokens=p_tok,
                  cost_usd=(p_tok * PRICE_PER_1K["prompt"] + c_tok * PRICE_PER_1K["completion"]) / 1000)

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, inputs=None, **kw):
        self._start(run_id, parent_run_id, serialized.get("name", "tool"), "tool", args=inputs)

    def on_tool_end(self, output, *, run_id, **kw):
        self._end(run_id, result=redact(str(getattr(output, "content", output)))[:200])

    def on_tool_error(self, error, *, run_id, **kw):
        self._end(run_id, status="error", error=repr(error))


tools = list(TOOLS.values())
graph_model = scripted([("get_invoice", {"invoice_id": "in_1001"})],
                       "Acme owes $1,200 on in_1001.").bind_tools(tools)


def call_model(state: MessagesState):
    return {"messages": [graph_model.invoke(state["messages"])]}


builder = StateGraph(MessagesState)
builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", lambda s: "tools" if s["messages"][-1].tool_calls else END, ["tools", END])
builder.add_edge("tools", "agent")
graph = builder.compile()

lg_tracer = Tracer()
handler = SpanCallbackHandler(lg_tracer)
result = graph.invoke({"messages": [HumanMessage("What does Acme owe on in_1001?")]},
                      config={"callbacks": [handler], "metadata": {"user_id": "u_42"}})

root = next(s for s in lg_tracer.spans if s.parent_id is None)
print(lg_tracer.tree(root.trace_id))
kinds = [s.kind for s in lg_tracer.spans]
assert kinds.count("model") == 2 and kinds.count("tool") == 1
assert all(s.trace_id == root.trace_id for s in lg_tracer.spans)      # one tree, one trace id
assert not handler.open                                               # every span was closed
```

Look at the rendered tree: `LangGraph` at the root, then the `agent` node containing a `chat_model` span, then the `tools` node containing `get_invoice`, then `agent` again. LangGraph gave you node boundaries for free; the callback system gave you parent ids; your handler decided what to record. With a real model, `usage_metadata` on the AI message carries the true token counts and the handler uses them instead of the approximation.

Passing `metadata` in the config is how you attach run-level attributes (user id, tenant, agent version). LangSmith reads the same field, which is why the remote version below needs almost no code.

### LangSmith: the same tree, hosted

With `LANGSMITH_TRACING=true` and an API key in the environment, every LangChain and LangGraph call is traced automatically by a handler that does what `SpanCallbackHandler` does, plus upload. Code outside the framework gets a `@traceable` decorator, which is `tracer.span` as a decorator.

```{code-cell} python remote
import os
from langsmith import traceable, Client
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

# In your shell, not the notebook:  LANGSMITH_TRACING=true  LANGSMITH_API_KEY=...  LANGSMITH_PROJECT=learn-agents
model = init_chat_model(os.environ.get("COURSE_MODEL", "openai:gpt-5.6"))
agent = create_agent(model, tools=[get_invoice, get_account],
                     system_prompt="Answer billing questions using the tools. Never invent amounts.")


@traceable(name="support_request", run_type="chain")       # a span around non-framework code
def handle(question: str, user_id: str) -> str:
    result = agent.invoke({"messages": [HumanMessage(question)]},
                          config={"metadata": {"user_id": user_id, "agent_version": "v3"},
                                  "tags": ["billing", "notebook"]})
    return result["messages"][-1].content


print(handle("What does Acme owe on in_1001?", user_id="u_42"))

# Read back what was recorded: the same fields our Tracer stored, now queryable remotely.
client = Client()
runs = list(client.list_runs(project_name=os.environ.get("LANGSMITH_PROJECT", "learn-agents"),
                             filter='eq(metadata_key, "user_id")', limit=5))
for r in runs:
    print(r.run_type, r.name, r.total_tokens, r.total_cost, r.latency)
```

Redaction on LangSmith is configured with input/output processors (the current SDK docs describe `hide_inputs`/`hide_outputs` hooks on the client). The principle is the one you built: redact before the span leaves the process.

## What goes wrong in production

1. **PII in the trace store.** Symptom: a compliance review finds customer emails and card fragments in the tracing SaaS. Cause: tool results were recorded raw. Fix: redact at the span boundary, as `redact()` does, with an allowlist of fields for structured results; treat the trace store as a data store subject to retention and access policy.

2. **Tokens are recorded, cost is not.** Symptom: finance asks what the agent costs per tenant and nobody can answer without a spreadsheet. Cause: cost is derived and the price table lives nowhere. Fix: compute cost at span-close time from a versioned price table, roll it up to the run, and tag runs with tenant and agent version so attribution is a group-by.

3. **Trace volume becomes the cost.** Symptom: the tracing bill rivals the model bill. Cause: every 30 KB tool result is stored in full on every run. Fix: truncate results (record size separately), sample low-value traces (keep 100% of errors and slow runs, 5% of the rest), and keep full payloads only for a short retention window.

4. **Spans without parents.** Symptom: the UI shows model calls floating outside any run. Cause: work moved to a thread or async task without propagating context. Fix: `contextvars` propagate into `asyncio` tasks automatically but not into raw threads; copy the context (`contextvars.copy_context().run`) or pass the parent explicitly. In LangChain, pass `config` through to child calls so `parent_run_id` is set.

5. **Tracing changes behavior.** Symptom: a bug appears only with tracing off, or latency doubles with it on. Cause: the tracer does synchronous network uploads inside the request, or an exception in the handler propagates into the agent. Fix: handlers must be non-blocking and must never raise into user code; batch uploads in a background thread; a tracer failure is logged, not surfaced.

6. **No correlation with the outside world.** Symptom: a Slack message id, a Stripe request id, and a trace id that cannot be joined. Cause: ids from external systems were not recorded on spans. Fix: record external request ids as span attributes and put your trace id in outgoing request headers. This is the same discipline as distributed tracing across microservices.

7. **Traces nobody reads.** Symptom: full instrumentation, and the same bugs recur. Cause: no path from trace to eval case. Fix: make "open the failing trace, add it as a dataset example" the first step of every bug report (19. Evaluation and Monitoring).

## What changed by 2026

- **Tracing is the substrate for evals, not a debugging extra.** Graders in 19 and 37 read tool calls and node transitions from traces. Anthropic's tool-design guidance recommends tracking accuracy, runtime, tool-call count, tokens, and errors together; all five are span attributes.
- **OpenTelemetry became the lingua franca.** The current MCP specification revision deprecates its own logging primitive in favor of logging to stderr or OpenTelemetry, and LangSmith, the OpenAI Agents SDK, and most agent frameworks can export or ingest OpenTelemetry spans. Building your tracer around trace/span/attribute means it will interoperate.
- **Harnesses ship tracing built in.** The OpenAI Agents SDK lists tracing as a core component; LangSmith traces LangGraph automatically; Anthropic's managed-agents design describes an append-only durable event log as the session record. The event log and the trace are converging into one artifact.
- **Cost and cache attribution matter more.** Prompt caching means the same prompt costs different amounts depending on cache hits; providers report cached tokens separately and a tracer should record them. Cost per tenant now has a caching dimension.
- **Reasoning tokens complicate the picture.** Reasoning models bill for thinking tokens you may not see. Record the provider's reported usage rather than estimating from visible text, and do not treat visible reasoning as the audit record; Anthropic's faithfulness research shows it often is not.
- **Redaction and retention are table stakes.** Tracing vendors expose input/output redaction hooks and retention controls because agent traces are customer data. Design for it from the first span.

## Interview questions

**Q1. What should a trace of an agent run contain?**

Strong: A tree, not a log. One root span per request with user or tenant id, agent and prompt version, total latency, total cost, step count, and outcome. Under it, a span per model call with model id, redacted input and output, prompt, completion and cached token counts, derived cost, and stop reason; a span per tool call with name, redacted arguments, truncated result plus its size, error class, and retry count; and node spans for orchestration boundaries. Parent ids link them into the tree so I can see what the model saw when it made each decision. Everything a grader or a billing query needs should be a field, not something parsed from text.

Weak: "Log the prompts and responses."

**Q2. How do you attribute cost in a multi-step agent?**

Strong: Cost is derived at span-close time from provider token counts and a versioned price table, including cached tokens which are priced differently. Each model span carries its own cost; the run rolls them up; the run carries tenant, user, and agent version. Attribution is then a group-by over spans: cost per tenant for billing, per agent version for regressions, per tool for finding expensive loops. I also record tool-call count and step count per run because rising step count is the earliest cost signal.

Weak: "Add up the tokens at the end."

**Q3. How do you keep PII out of traces without breaking the agent?**

Strong: Separate what the agent sees from what the tracer stores. The model still gets the raw tool result because it needs it; the recorded copy is redacted at the span boundary with pattern rules and an allowlist for structured fields. The trace store is treated as a data store with retention and access control. For LangSmith, the SDK exposes hide-inputs and hide-outputs hooks for the same purpose. I verify with an eval that scans stored spans for known PII patterns.

Weak: "Don't log user messages."

**Q4. How does LangGraph or LangChain tracing work under the hood?**

Strong: Through the callback system. Every chain, node, model call, and tool call fires start and end events carrying a run id and parent run id. A tracer is a callback handler that opens a span on start and closes it on end, using the ids to build the tree. LangSmith's tracer is one such handler that also uploads; you can write your own in fifty lines. Run-level metadata and tags are passed in the invoke config and land on the root span. Nothing in the graph runtime is tracing-specific.

Weak: "You set an environment variable and it traces."

**Q5. A span shows up with no parent. What happened?**

Strong: Context did not propagate. The parent is carried in a context variable, which flows into asyncio tasks but not into raw threads or process pools, and in LangChain flows through the config object. So either someone spawned a thread without copying the context, or a child call was made without passing config through. The fix is to copy the context when handing work to a thread, or pass config explicitly; and to alert on orphan spans, because they usually mean a code path lost its audit trail.

Weak: "The tracer dropped it."

**Q6. Tracing everything is expensive. What do you sample?**

Strong: Keep the metadata for every run because it is small and the dashboards depend on it. Keep full payloads for every run that errored, exceeded a latency or cost threshold, or received negative user feedback, plus a fixed sample of the rest. Truncate large tool results and record their size instead. Set a short retention window for payloads and a longer one for metadata. The goal is that any run you would want to investigate still has its full trace.

Weak: "Sample 10% of everything."

## Build it yourself

Extend `Tracer` and `SpanCallbackHandler`:

1. Add an `exporter` hook to `Tracer` that is called with each span on close, and implement a `JsonlExporter` that writes redacted spans to a file in the scratch directory as JSON lines.
2. Add `cached_tokens` to model spans and a third price in `PRICE_PER_1K` for cached input; cost must use it.
3. Add sampling: a `should_keep_payloads(run_span)` predicate that keeps payloads for error or slow runs and drops the `input`/`output`/`result` attributes otherwise, while keeping all timing and cost fields.
4. Add an `orphans()` method that returns spans whose `parent_id` is not None but does not match any recorded span.

Acceptance checks:

- After a traced run, the JSONL file has one line per span and no line contains an `@` character or a 13-16 digit number.
- A model span with `cached_tokens=400, prompt_tokens=500` costs less than one with `cached_tokens=0, prompt_tokens=500`.
- With sampling on, a fast successful run keeps `duration_ms` and `cost_usd` but has no `output` attribute; a run whose tool raised keeps everything.
- Running a tool inside a raw `threading.Thread` without copying context produces exactly one orphan; wrapping it with `contextvars.copy_context().run` produces zero.

## Key takeaways

- A trace is a tree of spans; plain logs flatten the tree and lose what you need most, the parent-child relationships.
- Record per model call the tokens, derived cost, and redacted input/output; per tool call the arguments, result size, and error class; per run the tenant, version, steps, and outcome.
- `contextvars` plus a context manager gives you nesting without threading a parent through every function; that is the OpenTelemetry pattern.
- Cost is derived from tokens and a versioned price table; roll it up to the run and tag runs so attribution is a group-by.
- Redact at the span boundary; the agent sees raw data, the trace store never does.
- LangGraph tracing is the callback system: a handler that turns start/end events with run ids into spans; LangSmith is one such handler with upload.
- Traces exist to be read: every failing trace should become an eval case.
