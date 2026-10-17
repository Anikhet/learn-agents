# 05. Tool Use and Function Calling

*Track A (agent engineering). Book: Chapter 5 (Tool Use). Prerequisite: 00. Course Orientation (the loop).*

The orientation lesson (00) built the loop. This lesson is about the thing the loop is built around: the tool call. A tool call looks like a small detail, a JSON blob with a name and some arguments. It is not a small detail. The tool schema is the only channel through which the model learns what it can do, and the tool result is the only channel through which it learns what happened. Everything an agent gets right or wrong about the outside world passes through those two documents.

By the end you should be able to explain what the provider actually does with a schema, why a `tool_call_id` exists, how parallel calls are matched to results, why an error is an observation and not an exception, what separates structured output from function calling, and what `ToolNode` is doing on your behalf.

## The problem

The support agent gets this message in Slack:

> "Acme says invoice INV-2291 is wrong. Can you check it against their plan and tell me what happened?"

To answer, the agent needs the Stripe invoice, the CRM record for Acme, and probably the plan catalog in Notion. The naive approach is to expose everything the backend can do as one tool:

```python
@tool
def run(command: str) -> str:
    """Run any backend command."""
    ...
```

This is easy to write and impossible to operate. The model has to invent the command syntax. You cannot validate the arguments because there is no structure to validate. You cannot authorize the call because you do not know what it does until it runs. When it fails you cannot tell the model what went wrong because you do not know what it meant. And when three commands are needed, the model issues them one at a time, waiting on each round trip.

The next naive approach exposes the raw API surface: `stripe_get`, `stripe_post`, `crm_query` with a raw SQL string. Now there is structure, but the model is doing the API's job: remembering endpoint paths, building filter syntax, paginating. Every one of those is a place to make a mistake that your code cannot catch.

What we actually want is a contract:

```text
                    schema (what the model sees)
                    ┌───────────────────────────────────┐
model ──chooses──►  │ name: get_invoice                 │
                    │ description: when and why to use  │
                    │ parameters: invoice_id: string    │
                    └───────────────────────────────────┘
                                   │
                                   │  your code validates, authorizes, executes
                                   ▼
                    ┌───────────────────────────────────┐
                    │ def get_invoice(invoice_id): ...  │  (what your program runs)
                    └───────────────────────────────────┘
                                   │
                                   ▼
                    result (what the model sees next)
```

A tool is a function plus a schema, and the schema is the contract. The model never sees the function body. Your code never sees the model's reasoning. They meet only at the schema on the way in and the result on the way out.

### What the provider does with your schema

This is the part most tutorials skip, and it is the part that explains almost every tool-selection failure.

When you call `model.bind_tools([...])` and then `invoke`, the client library serializes each tool into the provider's JSON format (for OpenAI-style APIs it is `{"type": "function", "function": {"name", "description", "parameters"}}`). The provider then renders those schemas into the prompt. The exact template is provider-specific and mostly undocumented, but the shape is always the same: a block of text near the system prompt that lists each tool's name, description, and parameter schema, followed by instructions on how to emit a call.

```text
system prompt
  │
  ├─ your instructions
  │
  ├─ [tools block, rendered by the provider]
  │    get_invoice: Fetch one Stripe invoice by id. Use when the user
  │                 names an invoice or asks about a charge...
  │      invoice_id (string, required)
  │    crm_lookup_account: ...
  │
  └─ (how to call a tool, in the provider's format)
conversation
  │
  └─ ... the model generates a special token sequence, then JSON
```

Three consequences fall out of this:

1. **Tool schemas cost tokens on every call.** Fifty tools with long descriptions can be several thousand tokens, prepended to every turn. Prompt caching helps only if the block is byte-identical between calls, which is why stable, deterministically ordered tool lists matter.
2. **The model chooses a tool by reading text.** It has no other information. If two descriptions overlap, the choice is a coin flip. If a parameter is called `user`, the model has to guess whether that means an id, an email, or a name.
3. **The output is constrained generation.** Most providers switch to a grammar or a fine-tuned format when the model decides to call a tool, so the JSON is syntactically valid. It is not guaranteed to be semantically valid: a string where an integer was expected, or an invoice id that does not exist. That validation is your job.

The book's Chapter 5 states the pattern: the model decides *whether* a tool is needed and *which* one, produces the arguments, and the application executes and returns the result. Everything below is the machinery for doing that well.

## Build it without a framework

We use LangChain's `@tool` and message types because LangGraph uses them, but every mechanism here is visible. Offline cells use the scripted model from STYLE.md so the control flow is under study; a live cell at the end runs the same tools against a real model.

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

### The schema is the contract

Define three tools for the running example. Then look at exactly what leaves your process when you bind them: the OpenAI-format schema that `langchain_core` derives from the signature and docstring. We also render that schema into plain text the way a provider template roughly would, so you can count the tokens you are paying for.

```{code-cell} python
import json
from typing import Literal
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool

INVOICES = {"INV-2291": {"customer": "acme", "amount_cents": 129900, "plan": "team",
                         "seats": 40, "status": "open"}}
ACCOUNTS = {"acme": {"plan": "team", "seats": 25, "owner": "j.park@acme.example"}}
CATALOG = {"team": {"per_seat_cents": 3000}, "enterprise": {"per_seat_cents": 5000}}


@tool
def stripe_get_invoice(invoice_id: str) -> dict:
    """Fetch one Stripe invoice by its id (format INV-####).

    Use when the user names an invoice or asks why they were charged.
    Do not use to search; you need the exact id first."""
    if invoice_id not in INVOICES:
        raise KeyError(f"no invoice {invoice_id}; ids look like INV-2291")
    return INVOICES[invoice_id]


@tool
def crm_get_account(account_slug: str) -> dict:
    """Fetch the CRM record for a customer by its lowercase slug, e.g. 'acme'.

    Returns the contracted plan and seat count. Use to compare what was
    sold against what was billed."""
    return ACCOUNTS[account_slug]


@tool
def notion_get_plan(plan: Literal["team", "enterprise"]) -> dict:
    """Fetch per-seat pricing for a plan from the Notion pricing page."""
    return CATALOG[plan]


tools = [stripe_get_invoice, crm_get_account, notion_get_plan]
wire = [convert_to_openai_tool(t) for t in tools]
print(json.dumps(wire[0], indent=2))


def render_tools_for_prompt(schemas):
    """A stand-in for the provider's hidden template. Real templates differ,
    but every one of them turns the schema into prompt text like this."""
    lines = []
    for s in schemas:
        fn = s["function"]
        lines.append(f"{fn['name']}: {fn['description']}")
        for name, prop in fn["parameters"]["properties"].items():
            req = "required" if name in fn["parameters"].get("required", []) else "optional"
            kind = prop.get("type") or f"enum{prop.get('enum')}"
            lines.append(f"  - {name} ({kind}, {req})")
    return "\n".join(lines)


prompt_text = render_tools_for_prompt(wire)
print(prompt_text)
approx_tokens = len(prompt_text) // 4
print(f"~{approx_tokens} tokens of tool schema on EVERY model call")
assert wire[0]["function"]["parameters"]["required"] == ["invoice_id"]
assert "enum" in wire[2]["function"]["parameters"]["properties"]["plan"]
```

Notice what made it into the schema and what did not. The `Literal` became an `enum`, so the model cannot invent a plan name. The docstring became the description, including the "Do not use to search" line, which is the sentence that stops the model from calling `stripe_get_invoice` with a customer name. The function body is absent. That is the contract.

### Parallel calls, ids, and errors as observations

The model's first move on the Acme question is to fetch the invoice and the account at once. Modern models emit several tool calls in one turn. Your code must execute them (ideally concurrently), then return one `ToolMessage` per call, each carrying the `tool_call_id` of the request it answers. The ids exist because results can come back in any order and the provider matches them by id, not by position. If a result is missing or an id is wrong, the provider rejects the next request outright.

One of the calls below fails. We do not let the exception escape. It becomes an observation with `status="error"` so the model can correct itself on the next turn.

```{code-cell} python
from concurrent.futures import ThreadPoolExecutor
from langchain_core.messages import HumanMessage, ToolMessage

tools_by_name = {t.name: t for t in tools}


def execute_tool_calls(tool_calls):
    """Run every call in the turn concurrently; never let an exception escape."""
    def run_one(call):
        tool_fn = tools_by_name.get(call["name"])
        if tool_fn is None:
            return ToolMessage(content=f"error: unknown tool {call['name']}; "
                                       f"available: {sorted(tools_by_name)}",
                               tool_call_id=call["id"], status="error")
        try:
            result = tool_fn.invoke(call["args"])
            return ToolMessage(content=json.dumps(result), tool_call_id=call["id"])
        except Exception as exc:                        # validation, KeyError, network...
            return ToolMessage(content=f"error: {type(exc).__name__}: {exc}",
                               tool_call_id=call["id"], status="error")

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(run_one, tool_calls))
    # Results are appended in request order regardless of completion order;
    # the ids, not the positions, are what the provider uses to match them.
    return results


def run_agent(model, user_text, max_steps=6):
    messages = [HumanMessage(user_text)]
    for _ in range(max_steps):
        ai = model.invoke(messages)
        messages.append(ai)
        if not ai.tool_calls:
            return messages
        messages.extend(execute_tool_calls(ai.tool_calls))
    raise RuntimeError("step budget exhausted")


model = scripted(
    # turn 1: two calls in parallel; the invoice id is wrong (typo)
    [("stripe_get_invoice", {"invoice_id": "INV-2219"}),
     ("crm_get_account", {"account_slug": "acme"})],
    # turn 2: the model reads the error and retries with the right id
    [("stripe_get_invoice", {"invoice_id": "INV-2291"})],
    # turn 3: it needs the price to explain the discrepancy
    [("notion_get_plan", {"plan": "team"})],
    "INV-2291 bills 40 team seats at $30 = $1,299, but the CRM contract is for "
    "25 seats ($750). The invoice overcounts by 15 seats.",
).bind_tools(tools)

history = run_agent(model, "Acme says INV-2291 is wrong. Check it against their plan.")
for m in history:
    tag = getattr(m, "status", "") or ""
    print(f"{type(m).__name__:12s} {tag:6s} {(m.content or str(m.tool_calls))[:90]}")

tool_msgs = [m for m in history if isinstance(m, ToolMessage)]
assert tool_msgs[0].status == "error" and "INV-2291" in tool_msgs[0].content
assert tool_msgs[1].tool_call_id == "call_1"          # matched by id, not position
assert history[-1].content.startswith("INV-2291 bills 40")
```

Look at the first error message. It says `no invoice INV-2219; ids look like INV-2291`. That sentence was written for the model, not for a human log. Anthropic's tool-writing guidance calls this actionable error text: the error should tell the model what to do next, not just that something failed. A bare `KeyError: 'INV-2219'` would have cost another round trip while the model guessed.

### Structured output is function calling pointed at yourself

The Acme answer above is prose. The Slack bot that posts it, and the dashboard that counts billing discrepancies, want a typed record. There are two ways to get it, and they are the same mechanism.

**Function calling** binds schemas the model may call, executes whatever it calls, and feeds the result back. **Structured output** binds exactly one schema, forces the model to "call" it, and then does *not* execute anything: the arguments *are* the answer. LangChain's `with_structured_output` does this under the hood for most providers (some providers also offer a native JSON-schema response mode, which is the same idea implemented in the decoder instead of the tool channel).

```{code-cell} python
from pydantic import BaseModel, Field, ValidationError


class BillingFinding(BaseModel):
    """Final structured verdict on an invoice dispute."""
    invoice_id: str = Field(pattern=r"^INV-\d{4}$")
    billed_seats: int
    contracted_seats: int
    overbilled_cents: int = Field(ge=0)
    recommended_action: Literal["credit_note", "no_action", "escalate"]


def extract(model, messages, schema: type[BaseModel]):
    """Structured output by hand: bind one 'tool' whose args are the answer.
    Nothing is executed. Validation failure is fed back once, then raised."""
    bound = model.bind_tools([schema], tool_choice=schema.__name__)
    for attempt in range(2):
        ai = bound.invoke(messages)
        call = ai.tool_calls[0]
        assert call["name"] == schema.__name__
        try:
            return schema.model_validate(call["args"])
        except ValidationError as exc:
            messages = messages + [ai, ToolMessage(content=f"error: {exc.errors()[0]['msg']}",
                                                  tool_call_id=call["id"], status="error")]
    raise RuntimeError("model could not produce a valid record")


extractor = scripted(
    # first attempt: negative overbilling, violates ge=0
    [("BillingFinding", {"invoice_id": "INV-2291", "billed_seats": 40,
                         "contracted_seats": 25, "overbilled_cents": -45000,
                         "recommended_action": "credit_note"})],
    # corrected
    [("BillingFinding", {"invoice_id": "INV-2291", "billed_seats": 40,
                         "contracted_seats": 25, "overbilled_cents": 45000,
                         "recommended_action": "credit_note"})],
)
finding = extract(extractor, history, BillingFinding)
print(finding.model_dump())
assert finding.overbilled_cents == 45000
assert finding.recommended_action == "credit_note"
```

The decision rule:

| You want | Use | Result flows |
|---|---|---|
| The model to *do* something in the world | function calling | back to the model as an observation |
| The model to *say* something in a fixed shape | structured output | to your code as the answer |
| Both: act, then report | function calling for tools, structured output for the final turn | tool results to the model, final record to you |

### When should the agent answer directly?

Not every message needs a tool. "Thanks, that fixed it" needs no Stripe call. The model decides this by reading the tool descriptions against the user turn, which is exactly why descriptions say *when not to use* a tool. You can also make it a hard rule in code: a cheap classifier or a routing node that sends chit-chat to a no-tools model and operational requests to the tool-equipped one (02. Routing). The book's rule is the right one: when the information is already in the context or the weights and no side effect is needed, calling a tool adds latency, cost, and a new failure surface for nothing.

## How LangGraph does it

The loop above, with parallel execution and errors turned into `ToolMessage`s, is what `langgraph.prebuilt.ToolNode` implements. Under the hood it reads the last `AIMessage`'s `tool_calls`, runs them in a thread pool (the same `executor.map` shape as our `execute_tool_calls`), and returns a list of `ToolMessage`s with matching ids.

One behavior is worth knowing precisely, because it bites. In the installed LangGraph (1.2.x) the default `handle_tool_errors` handler catches only *invocation* errors: unknown tool names and argument validation failures. It re-raises everything else. So a `RuntimeError` from a flaky Stripe client crashes the graph unless you pass `handle_tool_errors=True` (turn every exception into an error message) or a custom callable that decides per exception type. This is the right default for a framework, since silently swallowing a bug is worse than crashing, but you must choose deliberately.

```{code-cell} python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

flaky_calls = {"n": 0}


@tool
def stripe_get_invoice_flaky(invoice_id: str) -> dict:
    """Fetch one Stripe invoice by id. (Simulates one transient upstream failure.)"""
    flaky_calls["n"] += 1
    if flaky_calls["n"] == 1:
        raise RuntimeError("stripe: 503 Service Unavailable, retry after 1s")
    return INVOICES[invoice_id]


def classify_tool_error(exc: Exception) -> str:
    """Decide per error what the MODEL should be told. Retry policy for the
    transient class lives in the node's RetryPolicy or your tool, not here."""
    if isinstance(exc, KeyError):
        return f"error: not found: {exc}. Check the id format and try once more."
    if isinstance(exc, RuntimeError) and "503" in str(exc):
        return "error: upstream temporarily unavailable. Retry the same call once."
    raise exc  # programming errors should still crash loudly


graph_tools = [stripe_get_invoice_flaky, crm_get_account, notion_get_plan]
graph_model = scripted(
    [("stripe_get_invoice_flaky", {"invoice_id": "INV-2291"}),
     ("crm_get_account", {"account_slug": "acme"})],
    [("stripe_get_invoice_flaky", {"invoice_id": "INV-2291"})],   # retry after 503
    "Invoice overcounts seats: 40 billed vs 25 contracted.",
).bind_tools(graph_tools)


def agent(state: MessagesState):
    return {"messages": [graph_model.invoke(state["messages"])]}


def route(state: MessagesState):
    return "tools" if state["messages"][-1].tool_calls else END


builder = StateGraph(MessagesState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(graph_tools, handle_tool_errors=classify_tool_error))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", route, ["tools", END])
builder.add_edge("tools", "agent")
graph = builder.compile()

out = graph.invoke({"messages": [HumanMessage("Check INV-2291 against Acme's contract.")]})
tool_msgs = [m for m in out["messages"] if isinstance(m, ToolMessage)]
for m in tool_msgs:
    print(f"{m.tool_call_id:8s} {m.status:8s} {m.content[:80]}")
assert tool_msgs[0].status == "error" and "Retry" in tool_msgs[0].content
assert tool_msgs[1].status == "success"                 # crm call ran despite the sibling's failure
assert json.loads(tool_msgs[2].content)["seats"] == 40
assert out["messages"][-1].content.startswith("Invoice overcounts")
```

Map it back:

| From-scratch | `ToolNode` |
|---|---|
| `tools_by_name` dict | built from the `tools` list; unknown names produce an error message listing valid tools |
| `ThreadPoolExecutor.map` | the node's internal executor; one turn's calls run concurrently |
| `ToolMessage(..., tool_call_id=call["id"])` | same; ids preserved in request order |
| `except Exception` → observation | `handle_tool_errors` (default: validation errors only; `True`: everything; callable: your policy) |
| Pydantic validation in `tool.invoke` | same; `@tool` builds an `args_schema` and validates before calling your function |

The part `ToolNode` does *not* do is retry. For transient failures you attach `RetryPolicy` to the node (36. Durable Agent Orchestration) or handle it inside the tool. Keep the two concerns separate: retry policy is about *whether to try again*; the error message is about *what to tell the model*.

### The same tools, real model

```{code-cell} python live
import os
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

real_model = init_chat_model(os.environ.get("COURSE_MODEL", "openai:gpt-5.6"))
live_agent = create_agent(
    real_model,
    tools=[stripe_get_invoice, crm_get_account, notion_get_plan],
    system_prompt=("You are a billing-ops assistant. Answer directly if no lookup is "
                   "needed. Fetch invoice and account in parallel when both are needed."),
)
out = live_agent.invoke({"messages": [HumanMessage(
    "Acme says INV-2291 is wrong. Check it against their contract and say what happened.")]})
for m in out["messages"]:
    print(f"{type(m).__name__:12s} {(m.content or str(m.tool_calls))[:110]}")

# The final turn as a typed record, using the framework's structured-output path.
finding = real_model.with_structured_output(BillingFinding).invoke(out["messages"])
print(finding)
```

Watch the live transcript for whether the model fetched invoice and account in one turn. If it did not, the fix is usually in the system prompt or the descriptions, not in the code.

## What goes wrong in production

1. **Overlapping tools.** Symptom: the model calls `search_tickets` when it should call `get_ticket`, or alternates between them. Cause: two descriptions that read the same. Fix: make each description say when to use it *and* when not to; consolidate tools that are always used together (`schedule_event` instead of `list_calendars` + `find_slot` + `create_event`); run tool-selection evals over real transcripts.

2. **Ambiguous parameter names.** Symptom: `user` receives an email in one call and a name in the next. Cause: the schema does not say. Fix: `user_id`, `user_email`; enums for closed sets; docstrings with an example value. The schema is documentation for a reader who has no other documentation.

3. **Results flood the context.** Symptom: quality drops after a `list_invoices` call returns 400 rows. Cause: tool results go straight into `messages` forever. Fix: paginate, cap result size (Claude Code caps tool results at 25,000 tokens), offer `response_format="concise"|"detailed"`, and return semantic fields (customer name, plan) instead of opaque UUIDs the model then has to resolve with another call. See 34. Context Engineering.

4. **Retries on non-idempotent tools.** Symptom: a credit note issued twice. Cause: `issue_credit` timed out, the loop retried, both succeeded. Fix: every side-effecting tool takes an idempotency key derived from the `tool_call_id` (or a hash of the arguments plus thread id); the server deduplicates. Reads can retry freely; writes only with a key.

5. **Exceptions treated as crashes, or crashes treated as observations.** Symptom A: one flaky 503 kills a 20-step run. Symptom B: a `TypeError` in your own tool is fed to the model as "please fix your mistakes" and the model loops trying. Fix: classify. Validation and not-found go to the model as observations. Transient upstream errors go to a retry policy. Programming errors raise.

6. **Prompt rules mistaken for authorization.** Symptom: "only refund if approved" in the description; the model refunds. Cause: a description influences selection; it cannot deny execution. Fix: authorization lives in the tool body or a gate before it (13. Human in the Loop, 18. Guardrails). If a tool must never run in an environment, do not register it there.

7. **Schema drift.** Symptom: tool-selection accuracy quietly degrades after a "harmless" description edit. Cause: descriptions are prompt text; editing one is a prompt change. Fix: version the tool set, diff schemas in CI, and rerun tool-use evals on every change.

8. **Non-deterministic tool ordering.** Symptom: prompt-cache hit rate collapses. Cause: the tool list is built from a dict or a live registry whose order changes between calls. Fix: sort tools deterministically; bind the same list every turn.

## What changed by 2026

- **Tool design became an evaluated discipline.** Anthropic's "Writing effective tools for agents" turned folk wisdom into a method: build evals from real multi-step workflows, measure accuracy, runtime, call count, tokens, and errors, then let an agent critique the tools' descriptions against transcripts. Namespacing (`stripe_get_invoice`, `crm_get_account`) is now the default because it disambiguates by prefix before the model reads a word of the description.
- **Fewer, broader, better-described tools.** The guidance moved away from mirroring API endpoints one-to-one toward tools shaped around what the agent actually does. `search_logs` beats `read_logs` because the agent never wants the whole log.
- **Tool retrieval and progressive discovery.** With hundreds of tools available (especially via MCP servers, 10. Model Context Protocol), binding all of them is neither affordable nor accurate. The pattern is to retrieve a small relevant subset per turn, or expose a `search_tools` meta-tool. The MCP roadmap lists progressive tool discovery as a direction; the installed stack lets you implement it yourself by filtering before `bind_tools`.
- **Parallel calls are the norm.** Models now routinely emit several calls per turn. `ToolNode` and the OpenAI Agents SDK both execute them concurrently; your tools must be safe to run side by side.
- **Results are context-engineered, not just returned.** Concise-by-default formats, pagination, and offloading big results to files or a store are standard. The tool result is prompt text; treat writing it with the same care as the description.
- **Reasoning models changed the prompting, not the contract.** You no longer instruct "think step by step" before a tool decision; you state the goal, constraints, and what counts as done. Reasoning traces are not a faithful audit log; verify tool arguments and outcomes in code.
- **Errors as first-class results.** `ToolMessage.status="error"` exists in the installed `langchain_core` and `ToolNode` uses it. The convention that a failed tool is an observation, not an exception, is now framework-level.

## Interview questions

**Q1. Walk me through what happens between `bind_tools` and the model's first tool call.**

Strong: The client serializes each tool into the provider's JSON function format: name, description, and a JSON-schema `parameters` object derived from the signature. The provider renders that into prompt text using its own template, so the schemas cost tokens on every request. The model reads that text alongside the conversation and either generates a normal answer or a tool-call sequence; providers constrain that sequence so the JSON parses. The client turns it back into a `tool_calls` list with an id per call. My code then validates the arguments against the schema, decides whether to execute, runs it, and appends a `ToolMessage` with the matching id. The model never runs anything; it only proposes.

Weak: "You register functions and the model calls them."

**Q2. Why does `tool_call_id` exist?**

Strong: Because a single model turn can contain several tool calls, and results can be produced in any order or concurrently. The id is how each result is matched back to the request that produced it. Providers validate this strictly: every call must have exactly one result with its id before the next model turn, and an extra or missing id rejects the request. In practice I also reuse the id as an idempotency key for side-effecting tools so a retried call cannot double-execute.

Weak: "It's just a unique identifier for logging."

**Q3. What is the difference between structured output and function calling?**

Strong: They are the same mechanism used for two purposes. Function calling binds tools the model may call; whatever it calls, my code executes and feeds the result back as an observation. Structured output binds one schema, forces the model to call it, and treats the arguments as the answer; nothing is executed. Some providers implement structured output natively in the decoder, but the contract is identical. I use function calling when the model needs to act, structured output when I need a typed final record, and often both in one agent: tools during the loop, a schema on the last turn.

Weak: "Structured output is when you ask for JSON."

**Q4. A tool raises an exception mid-loop. What should happen?**

Strong: It depends on the class of error, so the first job is classification. Argument validation and not-found errors go back to the model as an error observation with actionable text, because the model can fix them. Transient upstream errors go to a retry policy with backoff, and only for idempotent operations or ones protected by an idempotency key. Programming errors in my own tool should raise and fail the run, because feeding a `TypeError` to the model just makes it loop. In LangGraph, `ToolNode`'s default handler only catches invocation errors and re-raises the rest, so I pass a callable that encodes this policy.

Weak: "Catch everything and send it back to the model."

**Q5. How do you decide whether the agent should call a tool or answer directly?**

Strong: The model makes that decision by reading the tool descriptions against the conversation, so the first lever is the description: say what the tool is for and when not to use it. The second lever is the system prompt: state that lookups are only for questions that need fresh or external data. The third is architectural: route obvious conversational turns to a no-tools path before the agent sees them, which is cheaper and removes a failure surface. I measure it with an eval set that includes turns where the right answer is no tool call at all.

Weak: "The model figures it out."

**Q6. What makes a good tool for an agent, concretely?**

Strong: Narrow purpose with a namespaced name; a description written like onboarding a new hire, including when not to use it; unambiguous, typed parameters with enums for closed sets and example values in the doc; a result that is concise by default, returns semantic fields rather than raw ids, and is paginated or capped so it cannot flood the context; error messages that tell the model what to do next; and idempotency for anything that writes. Then I evaluate the tool set on real multi-step tasks, not just unit tests, because the failure mode is selection, not execution.

Weak: "Good docstrings and type hints."

**Q7. How would you expose a system with 300 API endpoints to an agent?**

Strong: Not as 300 tools. Binding them all costs thousands of tokens per turn and tanks selection accuracy. I would design a small set of task-shaped tools that cover the real workflows, consolidate calls that always happen together, and put the long tail behind a `search_tools` or retrieval step that binds only the relevant subset for the current turn. I would also ask whether some of it should be exposed as resources the agent reads rather than tools it calls. The guidance from the tool-writing literature is fewer, broader, well-described tools, validated by agent evals.

Weak: "Generate tools from the OpenAPI spec."

## Build it yourself

Extend `execute_tool_calls` and the tool set:

1. Add `stripe_issue_credit(invoice_id, amount_cents, idempotency_key)`. Store issued credits in a dict keyed by `idempotency_key`; a second call with the same key returns the original result without creating a new credit.
2. Make `execute_tool_calls` derive the idempotency key from `tool_call_id` when the model omits it.
3. Add a `max_result_chars` cap: any `ToolMessage` content longer than the cap is truncated with a trailing `... [truncated; use pagination]` note.
4. Add a `tools_used` counter to the loop and stop with a clear error after 12 tool calls total.

Acceptance checks:

- Two scripted turns issuing the same credit with the same `tool_call_id` produce exactly one entry in the credit store.
- A tool returning a 10,000-character string yields a `ToolMessage` of at most `max_result_chars + len(note)`.
- A run that issues 13 tool calls raises before the 13th executes.
- The happy-path final message is unchanged.

## Key takeaways

- A tool is a function plus a schema; the model sees only the schema and the result, so both are prompt text and must be written as such.
- The provider renders schemas into the prompt on every call; tool count and description length are a token cost and a selection-accuracy cost.
- `tool_call_id` matches results to requests when a turn issues several calls; it doubles as a natural idempotency key.
- A failing tool is an observation, not a crash, but only after classification: validation errors to the model, transient errors to a retry policy, bugs raised.
- Structured output is function calling with one forced schema and no execution; use it for the final record, not for actions.
- `ToolNode` is the from-scratch executor with concurrency, id matching, and a pluggable error policy whose default catches only invocation errors.
- Descriptions steer selection; they never authorize execution. Authorization lives in code.
