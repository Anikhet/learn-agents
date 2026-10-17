# D. Building an Agent: the enterprise support agent, end to end

*Track A (agent engineering). Book: Appendix D (Building an Agent), drawing on Chapters 2, 5, 8, 13, 14, 19. Prerequisites: 00. Course Orientation, 02. Routing, 05. Tool Use, 13. Human in the Loop, 14. Knowledge Retrieval, 32. Observability and Tracing.*

Every earlier lesson isolated one pattern. This one assembles them. The question is the one the course has asked since lesson 00:

> "Why is Acme complaining about their invoice?"

By the end you will have a LangGraph agent that routes the question, retrieves policy from Notion, plans, calls Slack, the CRM and Stripe, pauses for a human before refunding money, resumes from a checkpoint, writes a memory note, emits trace spans, and passes an eval suite. All of it runs offline with the scripted model, so the control flow is what you study. A live cell at the end points the same tools at a real model.

## The problem

The support lead wants one thing: type the question into Slack and get a correct, actioned answer. Behind that sentence sit six systems and one irreversible action.

```text
"Why is Acme complaining about their invoice?"
            │
            ▼
   ┌─────────────────┐     Slack     "CFO says March invoice is wrong, 48 seats"
   │  support agent  │───► CRM       contracted_seats = 40
   │                 │───► Stripe    in_1042: seats_billed = 48, $480
   │                 │───► Notion    "disputed true-ups are refunded at contract price"
   │                 │───► Stripe    refund(in_1042, $80)   ◄── money moves here
   └─────────────────┘
            │
            ▼
   "Billed 48, contracted 40, refunded 8 seats as re_1."
```

The naive build is one `create_agent` call with all five tools registered and a system prompt that says "be careful with refunds." It demos beautifully. Then:

- A user asks "how do I reset my password?" and the agent searches Stripe for twenty steps because nothing routed it away from billing.
- The agent refunds $480 instead of $80 because it never read the true-up policy; the policy lived in Notion and nobody put it in the prompt.
- The refund call times out, the loop retries, and Acme is refunded twice.
- The process restarts mid-run and starts over from step one, calling Slack and the CRM again.
- The support lead asks "who approved that refund?" and the answer is a 30 KB transcript with no timestamps.

Each failure has a lesson in this course that fixes it. The engineering problem of this appendix is putting the fixes together without making the graph unreadable.

```text
failure                          fix                       lesson
──────────────────────────────── ───────────────────────── ────────
wrong domain, wandering loop     routing node              02
policy not in context            RAG node before the loop  14
double refund                    idempotency key + gate    05, 13, 36
restart from scratch             checkpointer per step     28, 36
"who approved this?"             trace spans + audit log   32
forgot last month's dispute      memory store              08
```

### The shape we are aiming for

```text
START
  │
  ▼
router ──(not billing)──► escalate ──► END
  │ (billing)
  ▼
retrieve  (Notion policy + memory notes into the system message)
  │
  ▼
planner   (one model call: the steps, as text)
  │
  ▼
agent ◄──────────────────────────────┐
  │                                  │
  ├──(no tool calls)──► memory ──► END
  │                                  │
  ├──(plain tools)────► tools ───────┘
  │                                  │
  └──(refund)──► approval ─(yes)─► tools
                    │
                    └───(no)──► agent   (with "denied" observations)
```

The approval node is the only place the graph stops and waits. That is deliberate. A pause is expensive (a human has to act), so it guards exactly one thing: the side effect that moves money.

## Build it without a framework

First the fixtures. Four systems, each a dict, each a `@tool` with the same shape as the real API. The refund tool takes an idempotency key because the real Stripe API does, and because the double-refund failure above is the one you cannot apologize your way out of.

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


from langchain_core.tools import tool

# Fixtures standing in for four SaaS systems. Same shape as the real APIs, no network.
SLACK = [
    {"channel": "#support", "user": "dana",
     "text": "Acme's CFO says the March invoice is wrong, they were billed for 48 seats"},
    {"channel": "#sales", "user": "raj", "text": "Acme renewed at 40 seats in Feb, contract attached"},
    {"channel": "#random", "user": "lee", "text": "anyone want lunch"},
]
CRM = {"acme": {"account_id": "ACME-001", "plan": "enterprise", "contracted_seats": 40,
                "owner": "raj", "contact": "cfo@acme.example"}}
INVOICES = {"in_1042": {"customer": "ACME-001", "period": "2026-03", "seats_billed": 48,
                        "unit_cents": 1000, "amount_cents": 48000, "status": "paid"}}
REFUNDS = {}                      # idempotency_key -> refund record
NOTION = [
    {"title": "Billing policy: seat true-ups", "text": "Seat true-ups are billed only for seats "
     "active over 30 days. Disputed true-ups are refunded at the contracted unit price."},
    {"title": "Refund approval policy", "text": "Refunds above $50 require approval from an ops "
     "lead. Always use an idempotency key."},
    {"title": "Onboarding checklist",
     "text": "Create the Slack channel, add the CSM, send the kickoff deck."},
]


@tool
def slack_search(query: str) -> list[str]:
    """Search Slack messages. Returns matching messages as 'channel user: text' strings."""
    words = set(query.lower().split())
    return [f"{m['channel']} {m['user']}: {m['text']}" for m in SLACK
            if words & set(m["text"].lower().split())]


@tool
def crm_lookup(company: str) -> dict:
    """Look up an account in the CRM by company name."""
    return CRM.get(company.lower(), {"error": f"no account named {company!r}"})


@tool
def stripe_get_invoice(invoice_id: str) -> dict:
    """Fetch one Stripe invoice by id."""
    return INVOICES.get(invoice_id, {"error": f"no invoice {invoice_id!r}"})


@tool
def stripe_refund(invoice_id: str, amount_cents: int, idempotency_key: str) -> dict:
    """Refund part of an invoice. Needs ops approval above $50. Same key -> same refund, never two."""
    if idempotency_key in REFUNDS:            # a retry must not move money twice
        return {**REFUNDS[idempotency_key], "replayed": True}
    REFUNDS[idempotency_key] = {"refund_id": f"re_{len(REFUNDS) + 1}", "invoice_id": invoice_id,
                                "amount_cents": amount_cents, "status": "succeeded"}
    return REFUNDS[idempotency_key]


TOOLS = [slack_search, crm_lookup, stripe_get_invoice, stripe_refund]
SENSITIVE = {"stripe_refund"}       # the capability boundary lives in code, not the prompt
print([t.name for t in TOOLS])
assert stripe_refund.args["idempotency_key"]["type"] == "string"
```

Now the agent as one function. Read it top to bottom: it is lesson 00's `run_agent` with four things bolted on in the order they are needed.

```text
route      one cheap model call; wrong domain exits before any tool runs
retrieve   deterministic; policy text and memory notes go into the system message
plan       one model call; the plan is text the model reads back on every step
loop       reason / act / observe, with a gate in front of SENSITIVE tools
memory     after the answer, one note per customer for next time
```

The `approve` argument is a callback. In this cell it is a lambda. In production it is a human, and the function has to *stop* until the human answers. That is the one thing this version cannot do, and it is the reason the LangGraph version exists.

```{code-cell} python
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

QUESTION = "Why is Acme complaining about their invoice?"


def retrieve(question, pages=NOTION, k=2):
    """Keyword-overlap retriever. 14. Knowledge Retrieval replaces this with BM25 + embeddings."""
    q = set(question.lower().split())
    scored = sorted(pages, key=lambda p: -len(q & set(p["text"].lower().split())))
    return [p for p in scored[:k] if q & set(p["text"].lower().split())]


def build_context(docs, notes, plan):
    return SystemMessage(
        "You are the enterprise support agent.\n"
        "Policy pages:\n" + "\n".join(f"- {d['title']}: {d['text']}" for d in docs)
        + "\nCustomer notes:\n" + "\n".join(f"- {n}" for n in notes)
        + f"\nPlan:\n{plan}")


def run_support_agent(model, question, approve, memory, max_steps=8):
    tools_by_name = {t.name: t for t in TOOLS}
    trace = []
    route = model.invoke([SystemMessage("Classify: billing | technical | general. One word."),
                          HumanMessage(question)]).content.strip().lower()
    trace.append(("route", route))
    if route != "billing":
        return {"route": route, "answer": "Escalated to a human: not a billing question.",
                "trace": trace}

    docs = retrieve(question + " invoice refund seats")
    notes = memory.get("acme", [])
    plan = model.invoke([HumanMessage(f"Plan the steps to answer: {question}")]).content
    messages = [build_context(docs, notes, plan), HumanMessage(question)]

    for step in range(max_steps):
        ai = model.invoke(messages)
        messages.append(ai)
        if not ai.tool_calls:
            break
        for call in ai.tool_calls:
            if call["name"] in SENSITIVE and not approve(call):     # gate before the side effect
                trace.append(("denied", call["name"]))
                messages.append(ToolMessage(content="Denied by approver.", tool_call_id=call["id"]))
                continue
            try:
                obs = tools_by_name[call["name"]].invoke(call["args"])
            except Exception as exc:
                obs = f"error: {type(exc).__name__}: {exc}"
            trace.append(("tool", call["name"]))
            messages.append(ToolMessage(content=str(obs), tool_call_id=call["id"]))
    else:
        raise RuntimeError("step budget exhausted")

    answer = messages[-1].content
    memory.setdefault("acme", []).append(f"{INVOICES['in_1042']['period']}: {answer[:60]}")
    return {"route": route, "plan": plan, "docs": docs, "answer": answer, "trace": trace}


def happy_path_script():
    """Exactly the turns the agent will request: route, plan, three tool rounds, answer."""
    return scripted(
        "billing",
        "1. Find the complaint in Slack. 2. Check contracted seats in CRM. "
        "3. Compare with the Stripe invoice. 4. Propose a refund for the difference.",
        [("slack_search", {"query": "Acme invoice"}), ("crm_lookup", {"company": "Acme"})],
        [("stripe_get_invoice", {"invoice_id": "in_1042"})],
        [("stripe_refund", {"invoice_id": "in_1042", "amount_cents": 8000,
                            "idempotency_key": "in_1042-trueup-2026-03"})],
        "Acme was billed for 48 seats but contracted 40. Per the true-up policy the 8 extra seats "
        "($80.00) were refunded as re_1. I have told #support.",
    ).bind_tools(TOOLS)


REFUNDS.clear()
memory = {}
result = run_support_agent(happy_path_script(), QUESTION, approve=lambda call: True, memory=memory)
for kind, name in result["trace"]:
    print(f"{kind:7s} {name}")
print(result["answer"])
assert result["route"] == "billing"
assert [n for k, n in result["trace"] if k == "tool"] == [
    "slack_search", "crm_lookup", "stripe_get_invoice", "stripe_refund"]
assert len(REFUNDS) == 1 and REFUNDS["in_1042-trueup-2026-03"]["amount_cents"] == 8000
assert result["docs"][0]["title"].startswith("Billing policy")
assert memory["acme"], "memory note written"
```

Three things to notice before we move on.

The script has exactly six turns because the agent makes exactly six model calls: one to route, one to plan, three tool rounds, one answer. If you add a node that calls the model, add a turn. That discipline is annoying and useful: it forces you to know how many times your agent talks to the model, which is also the number you will be billed for.

The gate is in front of the tool call, not inside the tool. `stripe_refund` does not know about approvals. The loop does. That means the same tool can be registered in a back-office agent where refunds under $50 need no approval, and the gate policy changes without touching the tool.

And `approve` is synchronous. If the human is asleep, this function blocks a worker for eight hours or the process dies and the transcript dies with it. Fixing that is the whole reason for the next section.

## How LangGraph does it

The from-scratch version is one function with the state in local variables. LangGraph asks you to name that state, split the function into nodes, and let the runtime own the transitions. In exchange it gives you three things the function could not have: a checkpoint after every node, an `interrupt()` that suspends the run across processes, and a `Store` that outlives the thread.

Map the local variables to a state schema first:

| In `run_support_agent` | In `SupportState` | Reducer |
|---|---|---|
| `route` | `route: str` | replace |
| `docs`, `plan` | `docs: list`, `plan: str` | replace |
| `messages` | `messages: Annotated[list, add_messages]` | append by id |
| `trace` | `trace: Annotated[list, operator.add]` | append |
| `memory` dict | the graph `Store`, namespaced by customer | outside the state |

Two reducers matter. `add_messages` appends and de-duplicates by message id, which is what a transcript needs. `operator.add` on `trace` makes it an append-only log: several nodes can write to it in one step and nothing is lost. Everything else replaces.

The approval node uses `interrupt()`. Underneath, `interrupt` raises a special exception. The runtime catches it, writes a checkpoint with the node marked as pending, and returns to the caller with an `__interrupt__` key in the output. When you call `invoke(Command(resume=...))` on the same `thread_id`, the runtime re-runs the pending node from the top, and this time `interrupt()` returns the resume value instead of raising. Lesson 13 covers the details; F. Under the Hood rebuilds the mechanism.

That re-run rule has a consequence: anything in the approval node *before* `interrupt()` runs twice. Keep it pure. Read state, ask, decide. The side effect happens in `tools`, one node later.

```{code-cell} python
import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command, interrupt


class SupportState(TypedDict):
    question: str
    customer: str
    route: str
    docs: list
    plan: str
    messages: Annotated[list, add_messages]     # append, matched by id
    trace: Annotated[list, operator.add]        # append-only audit log


def build_graph(model, tracer=None):
    tool_node = ToolNode(TOOLS)

    def router(state: SupportState):
        route = model.invoke([SystemMessage("Classify: billing | technical | general. One word."),
                              HumanMessage(state["question"])]).content.strip().lower()
        return {"route": route, "trace": [("route", route)]}

    def retrieve_node(state: SupportState, *, store):
        docs = retrieve(state["question"] + " invoice refund seats")
        notes = [item.value["text"] for item in store.search(("customers", state["customer"]))]
        return {"docs": docs, "trace": [("retrieve", f"{len(docs)} docs, {len(notes)} notes")],
                "messages": [build_context(docs, notes, "(plan pending)")]}

    def planner(state: SupportState):
        plan = model.invoke([HumanMessage(f"Plan the steps to answer: {state['question']}")]).content
        return {"plan": plan, "messages": [HumanMessage(state["question"])],
                "trace": [("plan", plan[:40])]}

    def agent(state: SupportState):
        return {"messages": [model.invoke(state["messages"])]}

    def after_agent(state: SupportState):
        last = state["messages"][-1]
        if not last.tool_calls:
            return "memory"
        if any(c["name"] in SENSITIVE for c in last.tool_calls):
            return "approval"
        return "tools"

    def approval(state: SupportState):
        calls = [c for c in state["messages"][-1].tool_calls if c["name"] in SENSITIVE]
        decision = interrupt({"action": calls[0]["name"], "args": calls[0]["args"],
                              "question": "Approve this side effect?"})
        if decision.get("approved"):
            return Command(goto="tools", update={"trace": [("approved", decision.get("by", "?"))]})
        denials = [ToolMessage(content=f"Denied by {decision.get('by', 'approver')}.",
                               tool_call_id=c["id"]) for c in state["messages"][-1].tool_calls]
        return Command(goto="agent", update={"messages": denials,
                                             "trace": [("denied", calls[0]["name"])]})

    def tools(state: SupportState):
        out = tool_node.invoke(state)
        return {**out, "trace": [("tool", c["name"]) for c in state["messages"][-1].tool_calls]}

    def memory(state: SupportState, *, store):
        answer = state["messages"][-1].content
        existing = store.search(("customers", state["customer"]))
        store.put(("customers", state["customer"]), f"note-{len(existing) + 1}",
                  {"text": f"{state['question']} -> {answer[:60]}"})
        return {"trace": [("memory", "note written")]}

    def escalate(state: SupportState):
        return {"messages": [AIMessage("Escalated to a human: not a billing question.")],
                "trace": [("escalate", state["route"])]}

    nodes = {"router": router, "retrieve": retrieve_node, "planner": planner, "agent": agent,
             "approval": approval, "tools": tools, "memory": memory, "escalate": escalate}
    builder = StateGraph(SupportState)
    for name, fn in nodes.items():
        builder.add_node(name, tracer(name, fn) if tracer else fn)
    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router", lambda s: "retrieve" if s["route"] == "billing" else "escalate",
        ["retrieve", "escalate"])
    builder.add_edge("retrieve", "planner")
    builder.add_edge("planner", "agent")
    builder.add_conditional_edges("agent", after_agent, ["tools", "approval", "memory"])
    builder.add_edge("tools", "agent")
    builder.add_edge("memory", END)
    builder.add_edge("escalate", END)
    return builder.compile(checkpointer=InMemorySaver(), store=InMemoryStore())


REFUNDS.clear()
graph = build_graph(happy_path_script())
config = {"configurable": {"thread_id": "acme-2026-03"}}
first = graph.invoke({"question": QUESTION, "customer": "acme"}, config)

paused = graph.get_state(config)
print("paused before:", paused.next, "->", paused.tasks[0].interrupts[0].value["action"])
assert paused.next == ("approval",)
assert REFUNDS == {}, "no money moved while waiting for a human"

# Hours later, in a different process, the ops lead clicks Approve:
final = graph.invoke(Command(resume={"approved": True, "by": "ops-lead"}), config)
for kind, detail in final["trace"]:
    print(f"{kind:9s} {detail}")
print(final["messages"][-1].content)
assert len(REFUNDS) == 1
assert ("approved", "ops-lead") in final["trace"]
assert graph.get_state(config).next == ()
history = list(graph.get_state_history(config))
print("checkpoints on thread:", len(history))
assert len(history) >= 8, "one checkpoint per superstep, so the run can resume anywhere"
notes = graph.store.search(("customers", "acme"))
assert len(notes) == 1 and notes[0].value["text"].startswith(QUESTION)
```

Walk the trace against the from-scratch version. Same route, same three tool rounds, same refund. The differences are all in what happens *between* nodes:

- After every node the checkpointer wrote the full state under `thread_id`. Fourteen checkpoints for one question. Kill the process after `stripe_get_invoice` and `invoke` on the same thread continues from the `agent` node; Slack and the CRM are not called again.
- `first` returned with `__interrupt__` and the refund not executed. The `paused.next == ("approval",)` line is what a UI polls to render an Approve button. The interrupt value carries the action and its arguments so the human sees exactly what they are approving.
- The `Command(goto=...)` returned by `approval` is a node choosing its successor. It is the same thing a conditional edge does, written from inside the node because the choice depends on data that only exists there (the resume value).
- `store.search` in `retrieve_node` ran before there were any notes and found zero. Ask a second question on a new thread and it finds one. The store is keyed by customer, not by thread; that is the difference between conversation memory and long-term memory (08. Memory Management).

### Tracing hooks and the denial path

Production needs to answer "what did node X do at 14:02 and how long did it take?" LangSmith or OpenTelemetry answers that for you. Here is the mechanism in twenty lines: wrap every node in a span recorder. Note that the interrupt shows up as a span with status `GraphInterrupt`, followed later by a second span for the same node with status `ok`. That is the re-run rule made visible.

```{code-cell} python
import inspect
import time
import uuid


class Tracer:
    """Minimal span recorder. LangSmith or OpenTelemetry replaces it in production."""

    def __init__(self):
        self.run_id = uuid.uuid4().hex[:8]
        self.spans = []

    def __call__(self, name, fn):
        wants_store = "store" in inspect.signature(fn).parameters

        def wrapped(state, *, store=None):
            started = time.perf_counter()
            status = "ok"
            try:
                return fn(state, store=store) if wants_store else fn(state)
            except BaseException as exc:          # interrupt propagates as an exception
                status = type(exc).__name__
                raise
            finally:
                self.spans.append({"run": self.run_id, "node": name, "status": status,
                                   "ms": round((time.perf_counter() - started) * 1000, 2)})
        wrapped.__signature__ = inspect.signature(fn)   # so LangGraph still injects `store`
        return wrapped


def denial_script():
    """Same first five turns; the final turn has to handle the denial."""
    return scripted(
        "billing",
        "1. Slack. 2. CRM. 3. Stripe. 4. Refund the difference.",
        [("slack_search", {"query": "Acme invoice"}), ("crm_lookup", {"company": "Acme"})],
        [("stripe_get_invoice", {"invoice_id": "in_1042"})],
        [("stripe_refund", {"invoice_id": "in_1042", "amount_cents": 8000,
                            "idempotency_key": "in_1042-trueup-2026-03"})],
        "Acme was billed for 48 seats against 40 contracted. The refund was not approved; "
        "I have opened a ticket for the ops lead instead.",
    ).bind_tools(TOOLS)


REFUNDS.clear()
tracer = Tracer()
graph = build_graph(denial_script(), tracer=tracer)
config = {"configurable": {"thread_id": "acme-denied"}}
for update in graph.stream({"question": QUESTION, "customer": "acme"}, config, stream_mode="updates"):
    print("update from:", next(iter(update)))          # what a UI would stream to the user
final = graph.invoke(Command(resume={"approved": False, "by": "ops-lead"}), config)

for s in tracer.spans:
    print(f"{s['run']} {s['node']:9s} {s['status']:14s} {s['ms']:>6} ms")
assert REFUNDS == {}, "denied means no money moved"
assert ("denied", "stripe_refund") in final["trace"]
approval_spans = [s["status"] for s in tracer.spans if s["node"] == "approval"]
assert approval_spans[0] == "GraphInterrupt" and approval_spans[-1] == "ok"
assert final["messages"][-1].content.startswith("Acme was billed")
```

### Eval cases

An agent without evals is a demo. The eval harness below runs the assembled graph, not its parts, because integration bugs (router says billing, gate never fires) are the ones unit tests miss. Each case is a script, an optional resume decision, and named checks. Lesson 19 grows this into a versioned dataset with judges; 37 adds trajectory metrics. The shape is the same.

```{code-cell} python
def run_case(case):
    REFUNDS.clear()
    g = build_graph(case["script"]())
    cfg = {"configurable": {"thread_id": f"eval-{case['name']}"}}
    out = g.invoke({"question": case["question"], "customer": "acme"}, cfg)
    if "__interrupt__" in out:
        out = g.invoke(Command(resume=case["resume"]), cfg)
    return out


def tools_in(out):
    return [d for k, d in out["trace"] if k == "tool"]


CASES = [
    {"name": "billing-approved", "question": QUESTION, "script": happy_path_script,
     "resume": {"approved": True, "by": "ops-lead"},
     "checks": {
         "routed to billing": lambda o: o["route"] == "billing",
         "policy retrieved": lambda o: any("true-up" in d["text"] for d in o["docs"]),
         "reads before writes": lambda o: tools_in(o).index("stripe_refund") == 3,
         "approval before refund": lambda o: [k for k, _ in o["trace"]].index("approved")
                                              < [k for k, _ in o["trace"]].index("memory"),
         "exactly one refund": lambda o: len(REFUNDS) == 1,
         "answer cites the seat delta": lambda o: "48" in o["messages"][-1].content
                                                  and "40" in o["messages"][-1].content}},
    {"name": "billing-denied", "question": QUESTION, "script": denial_script,
     "resume": {"approved": False, "by": "ops-lead"},
     "checks": {
         "no refund on denial": lambda o: REFUNDS == {},
         "model told about denial": lambda o: any(
             isinstance(m, ToolMessage) and "Denied" in m.content for m in o["messages"]),
         "still answers": lambda o: "not approved" in o["messages"][-1].content}},
    {"name": "not-billing", "question": "How do I reset my password?",
     "script": lambda: scripted("technical").bind_tools(TOOLS), "resume": None,
     "checks": {
         "routed away": lambda o: o["route"] == "technical",
         "zero tool calls": lambda o: tools_in(o) == [],
         "escalated": lambda o: "Escalated" in o["messages"][-1].content}},
]

scoreboard = []
for case in CASES:
    out = run_case(case)
    results = {name: bool(check(out)) for name, check in case["checks"].items()}
    scoreboard.append((case["name"], results))
    print(f"{case['name']:17s} {sum(results.values())}/{len(results)}  "
          + ", ".join(k for k, v in results.items() if not v))

# Idempotency is a property of the tool, so test the tool directly as well.
REFUNDS.clear()
a = stripe_refund.invoke({"invoice_id": "in_1042", "amount_cents": 8000, "idempotency_key": "k1"})
b = stripe_refund.invoke({"invoice_id": "in_1042", "amount_cents": 8000, "idempotency_key": "k1"})
print("replayed:", b.get("replayed"), "refunds:", len(REFUNDS))
assert a["refund_id"] == b["refund_id"] and b["replayed"] and len(REFUNDS) == 1
assert all(all(r.values()) for _, r in scoreboard), "every eval check must pass"
```

### The same agent against a real model

With a real model the scripts disappear and the tools stay. Note what is *not* registered: `stripe_refund`. In an unattended live run the safest gate is to not hand the model the capability at all. The graph above is how you hand it over with a human in front of it.

```{code-cell} python live
import os
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

real_model = init_chat_model(os.environ.get("COURSE_MODEL", "openai:gpt-5.6"))
read_only = [slack_search, crm_lookup, stripe_get_invoice]
docs = retrieve(QUESTION + " invoice refund seats")
system = build_context(docs, notes=[], plan="Investigate, then recommend; do not refund.")
agent = create_agent(real_model, tools=read_only, system_prompt=system.content)
out = agent.invoke({"messages": [HumanMessage(QUESTION)]})
print(out["messages"][-1].content)

# And the full graph, real model, human gate intact:
live_graph = build_graph(real_model.bind_tools(TOOLS))
cfg = {"configurable": {"thread_id": "live-acme"}}
step = live_graph.invoke({"question": QUESTION, "customer": "acme"}, cfg)
print("paused for:", live_graph.get_state(cfg).next)
```

## What goes wrong in production

1. **The gate is on the prompt, not the graph.** Symptom: "never refund without approval" in the system prompt, and a refund goes out. Cause: the prompt is advice; the tool is a capability. Fix: `SENSITIVE` is checked in code (`after_agent`), and the tool is not registered where no gate exists. The model chooses, your software executes.

2. **Approval resumes but the refund runs twice.** Symptom: the ops lead double-clicks Approve, or the resume request is retried by a proxy. Cause: two resumes on the same checkpoint both reach `tools`. Fix: the idempotency key on `stripe_refund`; LangGraph also rejects a resume on a thread that is no longer waiting, but the key is the guarantee. Test it directly, as the eval cell does.

3. **The plan goes stale.** Symptom: the planner said "refund the difference" but the invoice turned out to be correct, and the agent refunds anyway. Cause: a text plan is a suggestion the model reads back, not a program. Fix: make the refund argument come from observed data, and add a verification step or a Pydantic-validated `RefundProposal` that the gate shows to the human (06. Planning, 18. Guardrails). Trust outcomes, not the plan.

4. **Checkpoints grow without bound.** Symptom: the checkpoint store hits gigabytes; `get_state_history` takes seconds. Cause: full state after every superstep, including every 30 KB tool result. Fix: trim old tool output in the `agent` node (34. Context Engineering), offload large results to the Store or files and keep references in state, and set a retention policy on the checkpointer.

5. **Memory poisons the next run.** Symptom: next month the agent asserts "Acme always disputes true-ups" and refunds unprompted. Cause: the memory node stored a raw answer and the retrieve node injected it as fact. Fix: store structured, provenance-tagged notes, label them in the system message as history not instruction, and never let a memory note change the gate policy.

6. **The trace has no correlation id.** Symptom: three concurrent Acme threads interleave in the logs. Cause: spans without `run_id` or `thread_id`. Fix: the `Tracer` stamps every span with a run id; production tracers also carry `thread_id`, `checkpoint_id`, model, tokens, and cost per span (32. Observability).

7. **Router drift.** Symptom: after a model upgrade, "invoice" questions route to `general` 8% of the time. Cause: routing is a model decision and the model changed. Fix: the `not-billing` and `billing-approved` eval cases run in CI on every model or prompt change; add a confidence threshold with a fallback to a human (02. Routing).

## What changed by 2026

- **Harness, not loop.** The graph in this lesson is a small harness: it owns context assembly, planning, tool execution, permission gating, checkpointing, memory, and tracing around one model loop. Anthropic's harness guidance and Deep Agents (built on LangGraph) package these same parts. The value of building it once by hand is being able to read a harness and know which node is which.
- **Context engineering replaces prompt stuffing.** The retrieve node puts two policy pages in the system message, not all of Notion. Anthropic's context-engineering guidance calls this just-in-time retrieval via lightweight identifiers, alongside compaction and structured note-taking; the memory node is the smallest possible note-taking step.
- **Tool design is evaluated, not admired.** The tool descriptions here say when to use the tool and what it needs (`idempotency_key`). Anthropic's tool-design guidance goes further: namespace by service (`stripe_get_invoice`), return semantic fields rather than opaque ids, keep result sizes bounded, and build tool evals from real multi-step workflows. The eval cell is the seed of that.
- **Durable execution is the baseline.** Interrupt and resume across processes, one checkpoint per superstep, and idempotent side effects are now expected of any agent that touches money. LangGraph's persistence and human-in-the-loop guides describe exactly this pattern; 36. Durable Agent Orchestration covers the production checkpointers.
- **Reasoning models change the planner, not the gate.** With a reasoning model the explicit planner node often disappears; the model plans internally. The approval gate does not move, because a model's stated reasoning is not a faithful audit record. Verify the outcome (the refund amount against the observed seat delta), not the prose.
- **Protocols move the tools out of the process.** The four stubs here would be MCP servers in production, and the host would merge their tool lists into the registry the model sees. The gate, the checkpointer, and the eval cases are unchanged by that move.

## Interview questions

**Q1. Walk me through how you would build an agent that investigates a billing complaint and can issue a refund.**

Strong: I would start with the capability boundary: which tools are read-only and which move money. Reads go straight into a tool node; the refund goes behind an approval gate enforced in code, not in the prompt. Before the loop I would add a router so non-billing questions exit early, and a retrieval step that puts the relevant policy pages into context. The loop itself is standard reason-act-observe with a step budget. I would run it on a graph runtime with a checkpointer so the approval can pause for hours and resume from the same state, and I would give the refund tool an idempotency key so a retried resume cannot double-refund. Then I would write eval cases for the assembled graph: happy path, denial, non-billing, and idempotency.

Weak: "Give the model Slack, CRM and Stripe tools and a good system prompt."

**Q2. Where exactly does the approval check live, and why there?**

Strong: In the routing function after the model node, before any tool executes. The model emits a tool call; my code inspects the name against a sensitive set and routes to an approval node instead of the tool node. It lives there because that is the last point where the action is still a request rather than an effect. Putting it inside the tool couples policy to the integration; putting it in the prompt makes it optional. The approval node itself is pure: it only reads state and asks, because the runtime re-runs it on resume.

Weak: "The tool checks a flag before refunding."

**Q3. The process dies while waiting for approval. What happens?**

Strong: Nothing is lost, because the runtime wrote a checkpoint before the interrupt with the approval node marked pending. When any process calls invoke with a resume command on the same thread id, the runtime loads that checkpoint, re-runs the approval node, and the interrupt call returns the human's decision. The earlier tool calls are not repeated because their results are in the checkpointed messages. The one thing I have to design for is the node re-running from the top, so no side effects before the interrupt call.

Weak: "We would have to restart the run."

**Q4. How do you prevent a double refund?**

Strong: Defense in depth. The tool takes an idempotency key derived from the invoice and the dispute, and the backing API deduplicates on it; a replay returns the original refund. Above that, the gate ensures only an approved call reaches the tool, and the runtime refuses to resume a thread that is not waiting. I test the idempotency property directly on the tool, not just through the agent, because it is a property of the tool.

Weak: "Retry only once."

**Q5. What do you put in the eval suite for this agent?**

Strong: Cases that exercise the assembled graph, because integration failures are the common ones. Trajectory checks: the router picked billing, reads happened before the write, approval preceded the refund, exactly one refund. Outcome checks: the answer cites the observed seat delta. Negative cases: a denial produces no refund and the model is told; a non-billing question makes zero tool calls. And a direct idempotency test on the tool. These run in CI on every prompt or model change, and production traces that fail get added as new cases.

Weak: "Check that the final answer looks right."

**Q6. Why a graph runtime instead of a function with a while loop here?**

Strong: The function version works until it needs to wait for a human. A synchronous callback blocks a worker or dies with the process. The graph gives me per-step checkpoints, an interrupt that suspends across processes, a store for cross-thread memory, and streaming of intermediate updates to the UI, and each of those is one line at the call site. For a read-only agent with no pause I would keep the function.

Weak: "LangGraph is what everyone uses."

**Q7. How does long-term memory differ from the message history in this design?**

Strong: The message history is thread state: it is checkpointed with the run and ends with it. Long-term memory is in the store, namespaced by customer, written by a node after the answer and read by the retrieval node at the start of the next thread. I keep it structured and labeled as history in the system message so the model does not treat a past note as an instruction, and it never influences the gate policy.

Weak: "We keep the whole conversation in a vector database."

## Build it yourself

Extend the graph in three ways:

1. Add a `verify` node between `agent` and `approval` that recomputes the refund amount from `INVOICES` and `CRM` (`(seats_billed - contracted_seats) * unit_cents`) and rejects the proposed refund if it disagrees, routing back to `agent` with a `ToolMessage` explaining the mismatch.
2. Add a per-thread budget: a `cost` field with an `operator.add` reducer, incremented by each node that calls the model, and a routing check that escalates when it exceeds a limit.
3. Make the memory note structured: `{"period", "issue", "resolution", "approved_by"}`, and have `retrieve_node` render it as a labeled "History" block.

Acceptance checks:

- A script that proposes `amount_cents=48000` is rejected by `verify`, `REFUNDS` stays empty, and the trace contains `("verify", "mismatch")`.
- The happy path still refunds exactly 8000 cents once and `verify` records `("verify", "ok")`.
- With the budget set to 2 model calls, the run ends in `escalate` before any tool executes.
- A second thread for the same customer sees `1 notes` in the retrieve trace and the note contains `approved_by`.
- All original eval cases still pass.

## Key takeaways

- The capstone agent is lesson 00's loop with a router, a retrieval step, a planner, a gate, a checkpointer, a store, and a tracer bolted on in that order.
- Put the approval gate in the routing function after the model node: the last point where the action is still a request.
- `interrupt()` re-runs its node on resume, so keep everything before it pure and put the side effect one node later.
- One checkpoint per superstep is what makes "resume hours later in another process" a one-liner.
- Idempotency is a property of the tool; test it on the tool, and rely on it even with a gate in front.
- Long-term memory lives in the store keyed by customer, labeled as history, never as instruction.
- Evaluate the assembled graph with trajectory and outcome checks, and turn production failures into new cases.
