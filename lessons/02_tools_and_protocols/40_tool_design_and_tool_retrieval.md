# 40. Tool Design and Tool Retrieval: how many tools is too many?

*Track A (agent engineering). Book: Chapter 5 (Tool Use), Chapter 2 (Routing), Chapter 14 (Knowledge Retrieval). Prerequisites: 05. Tool Use and Function Calling, 10. Model Context Protocol, 34. Context Engineering.*

In 05. Tool Use you learned that a tool is a function plus a schema, and that the model chooses while your software executes. That lesson had three tools. This lesson is about what happens when you have eighty, which is not a hypothetical: connect the support agent to Slack, Notion, Stripe, the CRM, SQL, GitHub, a calendar, and the web through MCP servers and you are past eighty before anyone has designed anything.

Two questions follow. First, what does a *good* tool look like when there are many of them? Second, what do you do when the tool set no longer fits in the model's attention, let alone its context window? The first question is tool design. The second is tool retrieval. They are the same problem seen from two sides: the model can only pick well from a small, clear, non-overlapping menu.

## The problem

The support agent from 00. Course Orientation gets asked:

> "Why is Acme complaining about their invoice?"

The naive integration wires every endpoint of every connected system into the tool list. Here is what the model sees before it has read a single word of the question:

```text
system prompt                 1,200 tokens
80 tool schemas               ~14,000 tokens   <- before any work has happened
conversation so far              300 tokens
tool results (growing)             0 tokens
```

Every turn of the loop re-sends those 14,000 tokens. On a ten-turn task, tool schemas alone cost 140,000 input tokens. Prompt caching softens the bill but not the second problem: **selection accuracy**.

With three tools the model rarely picks the wrong one. With eighty, several of which overlap (`stripe_search_invoices`, `stripe_list_invoices`, `sql_query` against the billing table, `crm_get_account` which also has an invoices field), the model's choice becomes noisy. It calls `sql_query` with a guessed table name, gets an error, tries `crm_get_account`, gets a UUID it cannot use, and then finally finds `stripe_get_invoice`. Three wasted turns, each carrying 14,000 tokens of schemas.

```text
                 few tools                       many tools
                 ─────────                       ──────────
context cost     small, constant                 large, paid every turn
selection        near-perfect                    noisy; overlaps confuse the model
error recovery   one retry                       cascades of "wrong tool, try again"
latency          one round trip                  several, each slower (longer prefill)
```

Anthropic's "Writing effective tools for agents" (2025) puts the design side plainly: the tool set is the agent-computer interface, and it should be designed for the agent the way a UI is designed for a person. Its context-engineering guidance adds the other half: keep tool sets minimal and non-overlapping, because context is finite and attention degrades as it fills.

So the lesson has two parts. Design each tool so the model can use it correctly on the first try. Then, when the set is still too large, retrieve the relevant subset per turn instead of sending all of it.

### The design rules, in one table

| Rule | Bad | Good | Why |
|---|---|---|---|
| Namespace by service and resource | `search`, `get` | `stripe_search_invoices`, `slack_search_messages` | Disambiguates 80 similar verbs |
| Consolidate multi-step chores | `list_customers` then `get_customer` then `list_invoices` | `stripe_find_invoice(customer_name, status)` | One call replaces a fragile three-call dance |
| Return semantic fields | `{"id": "cus_8f3a..."}` | `{"customer": "Acme Corp", "invoice": "INV-2044", "status": "overdue"}` | The model reasons over names, not UUIDs |
| Offer response formats | always full JSON | `format="concise" \| "detailed"` | Concise for scanning, detailed for acting |
| Paginate and truncate | dump 400 rows | `page`, `page_size`, hard cap on result tokens | Protects the context window |
| Actionable errors | `Error 400` | `"status must be one of: open, paid, overdue"` | The model can fix the call without guessing |
| Descriptions as onboarding | `"Searches invoices."` | when to use, when not to, what it returns, an example | The description is the only training the model gets |
| Unambiguous parameters | `user` | `user_id` or `user_email` | Removes a coin flip |

Claude Code caps tool results at 25,000 tokens by default. That number is a good reminder that truncation is a platform-level decision, not a nicety.

## Build it without a framework

First the scripted model helper, copied verbatim from the style guide, so every offline cell runs without network or keys.

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

### Eighty tools and what they cost

We generate the tool set the way an MCP host would assemble it: eight services, five resources each, a search and a get for every resource. The descriptions follow the onboarding rule, so each one says what it returns and when to use it.

```{code-cell} python
import json
from langchain_core.tools import StructuredTool

SERVICES = {
    "slack":    ["message", "channel", "thread", "user", "reaction"],
    "notion":   ["page", "database", "block", "comment", "workspace"],
    "stripe":   ["invoice", "customer", "subscription", "refund", "payment"],
    "crm":      ["account", "contact", "deal", "ticket", "note"],
    "sql":      ["table", "schema", "query_result", "index", "view"],
    "web":      ["page", "search_result", "news_article", "document", "site"],
    "github":   ["issue", "pull_request", "commit", "repository", "release"],
    "calendar": ["event", "attendee", "room", "availability", "reminder"],
}
HINTS = {  # extra vocabulary the model (and our retriever) will match on
    "invoice": "billing, amount due, overdue, complaint about a charge",
    "refund": "money back, reversed charge, dispute",
    "ticket": "support case, escalation, complaint history",
    "message": "conversation, what someone said, customer chatter",
    "event": "meeting, schedule, when",
    "issue": "bug report, feature request",
}


def make_tool(service, resource, action):
    name = f"{service}_{action}_{resource}"
    hint = HINTS.get(resource, "")
    if action == "search":
        desc = (f"Search {resource} records in {service.title()} by free text. "
                f"Returns up to page_size matches with human-readable fields, not IDs. "
                f"Use when the user asks about a {resource} but you do not have its id. "
                f"Related vocabulary: {hint}.")
    else:
        desc = (f"Get one {resource} from {service.title()} by its id, with full detail. "
                f"Use after a search has given you the id. Related vocabulary: {hint}.")

    def fn(query: str, page_size: int = 5) -> str:
        return f"{name}: results for {query!r} (page_size={page_size})"

    return StructuredTool.from_function(fn, name=name, description=desc)


ALL_TOOLS = [make_tool(s, r, a) for s, rs in SERVICES.items() for r in rs for a in ("search", "get")]
TOOLS_BY_NAME = {t.name: t.name and t for t in ALL_TOOLS}


def schema_tokens(tools):
    """Rough token estimate: 4 characters per token over the JSON the provider would see."""
    payload = json.dumps([{"name": t.name, "description": t.description, "parameters": t.args}
                          for t in tools])
    return len(payload) // 4


full = schema_tokens(ALL_TOOLS)
five = schema_tokens(ALL_TOOLS[:5])
print(f"{len(ALL_TOOLS)} tools -> ~{full:,} tokens per turn; 5 tools -> ~{five:,} tokens per turn")
print(f"ten-turn task: {full * 10:,} vs {five * 10:,} schema tokens")
assert len(ALL_TOOLS) == 80
assert full > 10 * five
```

The number is the argument. Whatever the exact tokenizer says, the schemas are a fixed tax paid on every turn, and the tax grows linearly with the tool count while the useful work per turn does not.

### One well-designed tool, in code

Before retrieving anything, make each tool worth retrieving. This cell shows four of the rules at once: semantic fields instead of UUIDs, concise versus detailed formats, pagination with a hard token cap, and errors that tell the model what to do next.

```{code-cell} python
from typing import Literal

INVOICES = [  # stand-in for Stripe; note the semantic fields alongside the ids
    {"id": "in_8f3a", "customer": "Acme Corp", "number": "INV-2044", "status": "overdue",
     "amount_due": 12400, "currency": "usd", "line_items": ["Seats x40", "Overage 12k events"]},
    {"id": "in_9c21", "customer": "Acme Corp", "number": "INV-2031", "status": "paid",
     "amount_due": 0, "currency": "usd", "line_items": ["Seats x40"]},
    {"id": "in_1d77", "customer": "Globex", "number": "INV-2050", "status": "open",
     "amount_due": 800, "currency": "usd", "line_items": ["Seats x4"]},
]
MAX_RESULT_TOKENS = 200   # far below Claude Code's 25,000; small so the truncation is visible
VALID_STATUS = ("open", "paid", "overdue", "any")


class ToolError(ValueError):
    """Raised with text the MODEL can act on, not a stack trace."""


def stripe_find_invoices(customer: str, status: str = "any",
                         format: Literal["concise", "detailed"] = "concise",
                         page: int = 1, page_size: int = 2) -> dict:
    """Find invoices for a customer by name. Consolidates list_customers + list_invoices."""
    if status not in VALID_STATUS:
        raise ToolError(f"status={status!r} is invalid. Use one of {VALID_STATUS}. "
                        f"You searched customer={customer!r}; retry with a valid status.")
    hits = [i for i in INVOICES if i["customer"].lower() == customer.lower()
            and (status == "any" or i["status"] == status)]
    if not hits:
        known = sorted({i["customer"] for i in INVOICES})
        raise ToolError(f"No invoices for customer={customer!r}. Known customers: {known}. "
                        f"Check the spelling or use crm_search_account first.")
    start = (page - 1) * page_size
    page_hits = hits[start:start + page_size]
    if format == "concise":
        rows = [{"number": i["number"], "status": i["status"], "amount_due": i["amount_due"]}
                for i in page_hits]
    else:
        rows = page_hits
    body = json.dumps(rows)
    truncated = len(body) // 4 > MAX_RESULT_TOKENS
    if truncated:
        body = body[: MAX_RESULT_TOKENS * 4] + "...[truncated; ask for a smaller page_size]"
    return {"results": body, "page": page, "total": len(hits),
            "has_more": start + page_size < len(hits), "truncated": truncated}


concise = stripe_find_invoices("Acme Corp", status="overdue")
detailed = stripe_find_invoices("Acme Corp", format="detailed", page_size=5)
print("concise :", concise["results"])
print("detailed:", detailed["results"][:90], "...")
assert "INV-2044" in concise["results"] and "line_items" not in concise["results"]
assert "line_items" in detailed["results"]

for bad in ({"customer": "Acme Corp", "status": "late"}, {"customer": "Acme Inc"}):
    try:
        stripe_find_invoices(**bad)
    except ToolError as exc:
        print("actionable error:", str(exc)[:80], "...")
        assert "retry" in str(exc) or "crm_search_account" in str(exc)
```

Read the error strings as if you were the model. Each one names what was wrong, what the valid options are, and what to do next. That is the difference between a loop that recovers in one turn and a loop that flails for four.

### Tool retrieval from scratch

Now the second half. When the set is still too large after design, do not send it all. Embed each tool's description once, offline. At each turn, embed the current request, score every tool by cosine similarity, and bind only the top-k to the model. The model sees five tools, not eighty.

A real system uses a sentence-embedding model. Offline we use a deterministic stand-in: TF-IDF vectors over a shared vocabulary. The mechanism (vectors, cosine, top-k) is identical; only the quality of the vectors differs.

```text
offline:   for each tool:  vec[tool] = embed(name + description)          (80 vectors, once)

per turn:  q = embed(last user message + recent tool results)
           scores = cosine(q, vec[tool])  for all tools                    (80 dot products)
           active = top_k(scores, k=5)
           model.bind_tools(active)                                         (5 schemas, not 80)
```

```{code-cell} python
import math
import re
import numpy as np

STOP = {"the", "a", "an", "in", "by", "of", "to", "and", "or", "with", "for", "its", "you",
        "not", "do", "have", "when", "use", "after", "one", "up", "user", "about", "is", "are"}


def tokens(text):
    """Lowercase words with a crude stemmer so 'complaining' matches 'complaint'."""
    out = []
    for w in re.findall(r"[a-z]+", text.lower()):
        for suffix in ("ing", "ed", "es", "s"):
            if w.endswith(suffix) and len(w) - len(suffix) >= 4:
                w = w[: -len(suffix)]
                break
        if w not in STOP:
            out.append(w)
    return out


class ToolIndex:
    """Embeds tool descriptions once; returns the top-k tools for a query."""

    def __init__(self, tools):
        self.tools = list(tools)
        docs = [tokens(f"{t.name.replace('_', ' ')} {t.description}") for t in self.tools]
        self.vocab = {w: i for i, w in enumerate(sorted({w for d in docs for w in d}))}
        df = np.zeros(len(self.vocab))
        for d in docs:
            for w in set(d):
                df[self.vocab[w]] += 1
        self.idf = np.log((1 + len(docs)) / (1 + df)) + 1
        self.matrix = np.stack([self._embed(d) for d in docs])          # 80 x |vocab|

    def _embed(self, toks):
        v = np.zeros(len(self.vocab))
        for w in toks:
            if w in self.vocab:
                v[self.vocab[w]] += 1
        v *= self.idf
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def retrieve(self, query, k=5, always=()):
        q = self._embed(tokens(query))
        scores = self.matrix @ q                                        # cosine: both unit-norm
        order = np.argsort(-scores, kind="stable")
        chosen = [self.tools[i] for i in order[:k]]
        for name in always:                                             # pinned tools never drop out
            if name not in {t.name for t in chosen}:
                chosen.append(TOOLS_BY_NAME[name])
        return chosen, {self.tools[i].name: round(float(scores[i]), 3) for i in order[:k]}


index = ToolIndex(ALL_TOOLS)
active, scores = index.retrieve("Why is Acme complaining about their invoice?", k=5)
print(f"model sees {len(active)} of {len(ALL_TOOLS)} tools:")
for name, s in scores.items():
    print(f"  {s:.3f}  {name}")
names = [t.name for t in active]
assert len(active) == 5
assert "stripe_search_invoice" in names and "stripe_get_invoice" in names
assert not any(n.startswith("calendar") for n in names)
print(f"schema tokens this turn: ~{schema_tokens(active):,} instead of ~{schema_tokens(ALL_TOOLS):,}")
```

Three details matter in production and are already visible here:

- **The query is not just the user message.** After a tool result mentions a Slack thread id, the next turn's query should include that result so `slack_get_thread` rises. Retrieval is per turn, on the current state.
- **Some tools are pinned.** A `finish` tool or a `ask_human` tool must always be present. The `always` argument does that.
- **k is a budget, not a truth.** Too small and the right tool is missing; too large and you are back to the original problem. Measure recall@k on real tasks and pick k accordingly.

### Evaluating tools like an agent would

Anthropic's guidance says to test tools with agent evals built from realistic multi-step tasks, tracking accuracy, tool-call count, tokens, and errors, not just unit tests on the function. The offline version of that idea is a labeled set of requests and the tool that should be chosen. It tells you two things: how good your retriever is (recall@k) and how good your *descriptions* are, because a tool that never gets retrieved for its own use case has a bad description.

```{code-cell} python
EVAL_SET = [
    ("Why is Acme complaining about their invoice?",        "stripe_search_invoice"),
    ("Did we ever refund Globex for the double charge?",     "stripe_search_refund"),
    ("What did the customer say in Slack yesterday?",        "slack_search_message"),
    ("Open a bug report for the export failure",            "github_search_issue"),
    ("When is the next meeting with Acme scheduled?",       "calendar_search_event"),
    ("Show me the escalation history for this support case", "crm_search_ticket"),
    ("Which table holds subscription renewals?",            "sql_search_table"),
    ("Find the Notion page with the runbook",               "notion_search_page"),
]


def evaluate(index, k):
    hits_at_1 = hits_at_k = 0
    for query, expected in EVAL_SET:
        chosen, _ = index.retrieve(query, k=k)
        names = [t.name for t in chosen]
        hits_at_1 += names[0] == expected
        hits_at_k += expected in names
    n = len(EVAL_SET)
    return {"k": k, "top1": hits_at_1 / n, f"recall@{k}": hits_at_k / n,
            "schema_tokens": schema_tokens(index.retrieve(EVAL_SET[0][0], k=k)[0])}


for k in (1, 3, 5, 10):
    print(evaluate(index, k))
report = evaluate(index, 5)
assert report["recall@5"] >= 0.875, report
assert evaluate(index, 10)["recall@10"] >= report["recall@5"]
```

When a case fails at k=5, the fix is usually not a better embedding model. It is a better description: add the vocabulary users actually use. That is the "descriptions as onboarding" rule closing the loop with measurement.

## How LangGraph does it

LangGraph has no built-in tool retriever, and it does not need one. Tool selection is a node that runs before the agent node and writes the chosen tool names into state. The agent node reads that list and binds only those tools. `ToolNode` still knows all eighty, because execution is not the problem; the model's menu is.

```text
START
  │
  ▼
select_tools ──► agent ──(no tool calls)──► END
  ▲               │
  │               │ (tool calls)
  │               ▼
  └──────────── tools
```

The loop goes back through `select_tools`, not straight to `agent`, so that the menu is recomputed from the latest tool results. Compare with the from-scratch version: `retrieve()` is the node, `model.bind_tools(active)` is the one line in the agent node, and `MessagesState` is the transcript we were passing around by hand.

```{code-cell} python
from typing import Annotated
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

BOUND_LOG = []   # what the model was actually offered on each turn


class RecordingScriptedModel(ScriptedChatModel):
    def bind_tools(self, tools, **kwargs):
        BOUND_LOG.append(sorted(t.name for t in tools))
        return self


class AgentState(MessagesState):
    active_tools: list[str]          # replaced each turn, not appended


def select_tools(state: AgentState):
    """Retrieval node: query = last human message + the most recent tool result."""
    query_parts = [m.content for m in state["messages"] if isinstance(m, HumanMessage)][-1:]
    query_parts += [m.content for m in state["messages"] if isinstance(m, ToolMessage)][-1:]
    chosen, _ = index.retrieve(" ".join(query_parts), k=5)
    return {"active_tools": [t.name for t in chosen]}


def build_graph(model):
    def agent(state: AgentState):
        subset = [TOOLS_BY_NAME[n] for n in state["active_tools"]]
        return {"messages": [model.bind_tools(subset).invoke(state["messages"])]}

    def route(state: AgentState):
        return "tools" if state["messages"][-1].tool_calls else END

    b = StateGraph(AgentState)
    b.add_node("select_tools", select_tools)
    b.add_node("agent", agent)
    b.add_node("tools", ToolNode(ALL_TOOLS))             # executor knows everything
    b.add_edge(START, "select_tools")
    b.add_edge("select_tools", "agent")
    b.add_conditional_edges("agent", route, ["tools", END])
    b.add_edge("tools", "select_tools")                  # re-select on the new evidence
    return b.compile()


model = RecordingScriptedModel(messages=iter([
    AIMessage(content="", tool_calls=[{"name": "stripe_search_invoice", "id": "c0", "type": "tool_call",
                                       "args": {"query": "Acme overdue", "page_size": 3}}]),
    AIMessage(content="Acme's INV-2044 is overdue for $12,400 because of an overage line item."),
]))
graph = build_graph(model)
out = graph.invoke({"messages": [HumanMessage("Why is Acme complaining about their invoice?")],
                    "active_tools": []})

for i, offered in enumerate(BOUND_LOG):
    print(f"turn {i}: model offered {len(offered)} of {len(ALL_TOOLS)} tools -> {offered}")
called = [c["name"] for m in out["messages"] if isinstance(m, AIMessage) for c in m.tool_calls]
assert all(len(offered) == 5 for offered in BOUND_LOG)
assert called == ["stripe_search_invoice"] and called[0] in BOUND_LOG[0]
assert out["messages"][-1].content.startswith("Acme's INV-2044")
```

What each primitive is doing underneath:

| From-scratch | LangGraph | Underneath |
|---|---|---|
| `index.retrieve(query)` | `select_tools` node | Plain function; returns a state update |
| `active_tools` variable | `active_tools: list[str]` in state, no reducer | Replaced per turn, which is what a menu needs |
| `model.bind_tools(active)` | one line inside `agent` | Binding is per invocation, so the schema set can change every turn |
| `tools_by_name[...]` executor | `ToolNode(ALL_TOOLS)` | Execution is separate from selection; the executor can know everything |
| the `for` loop | `tools -> select_tools` edge | Re-selection on new evidence |

### The framework versions: two middlewares

LangChain 1.x ships the same idea as `create_agent` middleware, and it is worth knowing both variants because they make different tradeoffs:

| Middleware | Who selects | Cost per turn | When to use |
|---|---|---|---|
| `LLMToolSelectorMiddleware(model=, max_tools=, always_include=)` | A (usually small, cheap) model reads the query and the full tool list, returns a subset | One extra model call over all 80 schemas | Tool set fits in a cheap model's context; you want semantic judgment, not lexical match |
| `ProviderToolSearchMiddleware(searchable_tools=)` | The provider's own tool-search feature; the model calls a search tool to pull in schemas on demand | Provider-dependent; schemas are fetched, not pre-sent | Provider supports it; the tool set is too big even for the selector model |
| Our `select_tools` node | An embedding index | 80 dot products | Cheapest and fully under your control; quality bounded by descriptions |

`always_include` is our `always=` pin list under a different name. `max_tools` is our `k`. The mechanism under all three is the one you built by hand: a step before the model call that narrows what gets bound.

```{code-cell} python live
import os
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import LLMToolSelectorMiddleware

real = init_chat_model(os.environ.get("COURSE_MODEL", "openai:gpt-5.6"))

# Variant 1: our retrieval node, real model.
live_graph = build_graph(real)
res = live_graph.invoke({"messages": [HumanMessage("Did we ever refund Globex for the double charge?")],
                        "active_tools": []})
print(res["messages"][-1].content)
print("offered per turn:", BOUND_LOG[-3:])

# Variant 2: LangChain's selector middleware; a model trims the menu before each call.
selector = LLMToolSelectorMiddleware(model=real, max_tools=5, always_include=[])
agent = create_agent(real, tools=ALL_TOOLS, middleware=[selector])
out = agent.invoke({"messages": [HumanMessage("Did we ever refund Globex for the double charge?")]})
print(out["messages"][-1].content)
```

Where MCP fits: the host merges `tools/list` from every server into one registry. Today that registry is what your retriever indexes. The current MCP roadmap lists progressive tool discovery, meaning servers would let a host ask for tools relevant to a task rather than the full list. Until that lands, tool retrieval is a host-side layer exactly like the one above, and the cacheable, deterministically ordered `tools/list` responses in the current spec revision are what make indexing the registry cheap.

## What goes wrong in production

1. **The right tool is not in the top-k.** Symptom: the agent says "I do not have a way to check refunds" while `stripe_search_refund` exists. Cause: the description lacks the words users use, or k is too small. Fix: run the retrieval eval on real traffic, add vocabulary to descriptions, and pin a small `list_more_tools(topic)` escape hatch so the model can ask for a second menu.

2. **The menu changes and the model loses a tool it was mid-way through using.** Symptom: turn 3 calls `slack_get_thread`, which was offered on turn 2 but not turn 3, and the provider rejects the call. Cause: re-selecting per turn without hysteresis. Fix: keep tools called in the last N turns in the active set, and treat an unknown-tool call as an actionable error ("tool not available this turn; it may be offered if you restate the goal").

3. **Overlapping tools survive design review.** Symptom: three different tools can return an invoice, and the model alternates between them. Cause: each MCP server was designed alone. Fix: consolidate at the host with a thin facade tool, or exclude the redundant ones from the registry. Retrieval cannot fix overlap; it only hides it.

4. **Results flood the context.** Symptom: a 40 KB `sql_search_table` result on turn 2 degrades every later turn. Cause: no pagination, no cap. Fix: hard-cap result tokens in the executor, return `has_more`, and write big results to a file or store with a handle (34. Context Engineering).

5. **Errors are stack traces.** Symptom: the model retries the same bad argument four times. Cause: the tool raised a raw exception and the loop stringified it. Fix: catch at the tool boundary and rewrite into "what was wrong, what is valid, what to do next," as `stripe_find_invoices` does.

6. **Retrieval adds latency you did not budget.** Symptom: p95 grows by 300 ms per turn. Cause: an embedding API call per turn. Fix: cache the query embedding for identical text, use a small local embedding model, or select tools only when the task changes (a cheap classifier decides whether to re-select).

7. **Cache misses from a shifting schema set.** Symptom: prompt cache hit rate collapses after enabling retrieval. Cause: tool schemas sit at the front of the prompt, so any change invalidates the cached prefix. Fix: order the prefix as stable system prompt, then a stable pinned tool block, then the volatile retrieved block; and keep the retrieved set stable across turns when nothing changed.

## What changed by 2026

- **Tool design became a documented discipline.** Anthropic's "Writing effective tools for agents" (2025) turned folk wisdom into rules: namespacing, consolidation, semantic results, response formats, pagination, actionable errors, onboarding-quality descriptions, and evals built from real multi-step workflows. Interviewers now expect you to cite them.
- **Context engineering absorbed tool selection.** The same guidance that describes compaction and just-in-time retrieval of documents treats the tool list as part of the context budget: minimal, non-overlapping, and loaded when relevant. A tool menu is context.
- **MCP made the problem universal.** Any host that connects several servers has the 80-tool problem by default. The current spec revision makes `tools/list` cacheable with `ttlMs` and asks servers to return tools in deterministic order, which helps prompt caching; the published roadmap names progressive tool discovery as a direction. Until then, retrieval lives in the host.
- **Harnesses treat the tool list as dynamic.** LangChain's `create_agent` ships `LLMToolSelectorMiddleware` and `ProviderToolSearchMiddleware`; Deep Agents' current release notes describe cutting default-turn input tokens by roughly two thirds partly by trimming tool-description tokens, which is the same budget argument this lesson makes. Exact mechanisms vary and change; the design is the node above.
- **Skills separated "how" from "what".** Tools are what the agent can do; skills (41. Skills) are procedural knowledge about how to do it well, loaded on demand with the same progressive-disclosure idea.

## Interview questions

**Q1. How many tools is too many, and how do you design tool schemas an LLM can actually use correctly?** (The interview repository tags this one Anthropic and Cognition.)

Strong: There is no fixed number; the limit is where selection accuracy drops or schema tokens dominate the turn, and you find it by measurement. Two things break as the count grows. Context cost: every turn re-sends all schemas, roughly ten thousand or more tokens, before any work starts. Selection accuracy: with overlapping tools the model picks wrong, which costs extra turns that each pay the full tax. I fix design first: namespace by service and resource, consolidate multi-call chores into one tool, remove overlap, write descriptions like onboarding docs. If the set is still large, I add a retrieval layer: embed descriptions once, embed the current request each turn, bind the top-k plus a few pinned tools. I measure recall@k on labeled real tasks and tune k and descriptions from the failures.

Weak: "Just give the model all the tools; modern context windows are big."

**Q2. What makes a tool description good?**

Strong: It reads like onboarding for a new hire: what the tool returns, when to use it, when not to and what to use instead, and one example. Parameter names are unambiguous, `user_id` not `user`. The description carries the vocabulary users actually use, because that is what both the model and any retriever match on. I validate descriptions with an agent eval: if the tool is never selected for its own use case, the description is wrong, not the model.

Weak: "A one-line docstring that says what the function does."

**Q3. How do you evaluate tools?**

Strong: Not with unit tests alone. Build a set of realistic multi-step tasks drawn from real traffic, run the agent, and track task accuracy, number of tool calls, tokens consumed, and error rate per tool. Read the transcripts: repeated retries on one tool mean its errors are not actionable; skipped tools mean bad descriptions; long results mean missing pagination. Offline, a labeled query-to-tool set gives a quick recall@k for the retrieval layer. Then iterate on descriptions and consolidation and re-run.

Weak: "Check that each function returns the right value."

**Q4. Why return semantic fields instead of ids?**

Strong: The model reasons in language. `customer: Acme Corp, status: overdue` lets it answer the user directly and choose the next tool sensibly; `cus_8f3a` forces another lookup and invites hallucinated ids. I return both when the id is needed for a follow-up call, and I offer a concise format for scanning and a detailed one for acting so the model controls its own context spend.

Weak: "Return the raw API JSON; the model can figure it out."

**Q5. How does tool retrieval interact with prompt caching?**

Strong: Badly if done naively. Schemas sit early in the prompt, so a menu that changes every turn invalidates the cached prefix. I structure the prefix as stable system prompt, then pinned tools, then the retrieved set, and I add hysteresis so the retrieved set only changes when the task changes. That keeps most turns on a cache hit while still paying for five schemas instead of eighty. MCP's cacheable, deterministically ordered `tools/list` helps on the registry side.

Weak: "Caching is the provider's problem."

**Q6. Where does this live in LangGraph?**

Strong: As a node before the agent node. `select_tools` reads the latest human message and last tool result, retrieves top-k, writes names into a state key that is replaced each turn, and the agent node binds only that subset. `ToolNode` keeps the full tool list because executing is not the bottleneck, choosing is. The `tools` edge loops back to `select_tools`, not `agent`, so the menu reflects new evidence. It is one extra node and one state key.

Weak: "LangGraph handles tool selection automatically."

**Q7. What is the relationship between MCP and this problem?**

Strong: Function calling is the model-to-application interface for choosing a tool; MCP is the application-to-server interface for discovering and invoking them. The host merges every server's `tools/list` into one registry, which is exactly how you end up with eighty tools. Today the host is responsible for trimming that registry per turn; the MCP roadmap names progressive tool discovery, which would move part of that responsibility to the protocol. Either way the design rules for individual tools do not change.

Weak: "MCP solves tool overload."

## Build it yourself

Extend the retrieval graph:

1. Add hysteresis: any tool called in the last two turns stays in `active_tools` even if it drops out of the top-k.
2. Add a pinned `list_more_tools(topic: str)` tool that returns the names and descriptions of the next five best matches for `topic`, and make the agent node bind whatever it returns on the following turn.
3. Add a per-tool result cap in a wrapper around `ToolNode` that truncates any `ToolMessage` over 2,000 characters and appends `[truncated; has_more]`.
4. Extend `EVAL_SET` with four multi-step cases where the second step depends on the first result, and report recall@5 for step two using the tool-result text in the query.

Acceptance checks:

- After a `slack_search_message` call on turn 1, `slack_search_message` is present in `BOUND_LOG[1]` even when its score is outside the top 5.
- Calling `list_more_tools("calendar")` results in at least one `calendar_*` tool in the next turn's bound set, and the bound set never exceeds 10 tools.
- A tool that returns 10,000 characters yields a `ToolMessage` of at most 2,030 characters ending in `has_more]`.
- Step-two recall@5 on the new cases is at least 0.75, and the report prints which descriptions you changed to get there.

## Key takeaways

- Tool schemas are context: every extra tool is a fixed token tax on every turn and a hit to selection accuracy.
- Design first: namespace, consolidate, return semantic fields, offer concise and detailed formats, paginate and truncate, make errors actionable, write descriptions as onboarding.
- When the set is still large, retrieve tools per turn: embed descriptions once, embed the current request, bind top-k plus pinned tools.
- Evaluate tools with agent-style evals on real tasks and with recall@k for the retrieval layer; failures usually point at descriptions, not models.
- In LangGraph, selection is a node before the agent node with a replaced-per-turn state key; the executor can still know every tool.
- Keep the retrieved set stable when the task has not changed, or you trade schema tokens for prompt-cache misses.
- MCP makes the 80-tool problem the default; the host owns the fix today, and progressive tool discovery is on the protocol's roadmap.
