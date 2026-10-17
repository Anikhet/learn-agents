# 34. Context Engineering

*Track A (agent engineering). Book: course addition (extends Chapter 8, Memory Management, and Chapter 16, Resource-Aware Optimization). Prerequisites: 00. What an agent actually is; 06. Planning; 08. Memory Management.*

Every step of an agent is one model call, and the model knows only what is in that call's context. So the question that decides whether a long-running agent works is not "how do I word the prompt?" but "what exactly does the model see at this step, and why?" That is context engineering. This lesson builds the four mechanisms that answer it, from scratch: a per-step context builder with just-in-time retrieval, offloading of large tool results, a compactor that cannot lose decisions or constraints, and a subagent that isolates a messy sub-task. Then it maps each one onto LangChain middleware and Deep Agents.

## The problem

The ops agent is on step 38 of this request:

> "Acme's renewal is Friday. Figure out every open billing and support issue for them, fix what we can, and prepare a briefing for the account owner."

By now the transcript contains:

```text
system prompt + tool schemas              3,000 tokens
user request                                 60
14 Slack search results                   21,000
full Stripe charge export (1 call)        18,000
3 Notion policy pages                      9,500
SQL usage query result                     6,000
the model's own intermediate messages      4,500
                                         ───────
                                          62,060 tokens, and growing ~2,000 per step
```

The naive approach, "append everything, the window is big enough," fails before the window is full:

```text
step  5   answers are sharp
step 20   re-runs a Slack search it already ran
step 30   forgets the constraint "do not issue refunds over $500 without approval"
step 38   briefing cites a Notion policy page that was superseded at step 12
step 45   context limit exceeded, run fails, 45 steps of work lost
```

Three different things are going wrong, and they need different fixes.

**Attention is a budget, not a bucket.** A transformer attends over every token in context, and the cost grows with length (naive attention is quadratic in sequence length; see 43. Attention, KV Cache, GQA, MoE). More important for us: quality degrades with length and with position. "Lost in the Middle" (Liu et al., 2023) showed a U-shaped curve, where information at the start and end of a long context is used better than information in the middle. Anthropic's context-engineering guidance calls the broader effect context rot: as the context fills, the model's ability to recall and use any particular piece of it declines. Every irrelevant token is spending attention you needed for a relevant one.

**Stale and superseded content stays authoritative.** The old policy page is still in context, word for word. The model has no way to know a later message superseded it unless something removes or marks it.

**Constraints are just more tokens.** The $500 refund rule is one sentence at step 2. By step 30 it is surrounded by 50,000 tokens of search results. Nothing distinguishes it as load-bearing.

The fix is to stop treating context as an append-only log and start treating it as a **view you compute for each step**:

```text
                    ┌────────────────────────────── durable state (outside the window) ───┐
                    │ transcript log   notes (decisions, constraints, open issues)         │
                    │ plan/todos       offloaded tool results (files)   long-term memory    │
                    └───────────────────────────────┬───────────────────────────────────────┘
                                                    │ select, compress, order
                                                    ▼
                                    ┌──────────── context for step N ─────────────┐
                                    │ stable prefix: system prompt, tool schemas  │
                                    │ pinned: goal, constraints, notes, plan      │
                                    │ just-in-time: only what step N needs        │
                                    │ recent turns, verbatim                      │
                                    │ current instruction (goal restated last)    │
                                    └─────────────────────────────────────────────┘
                                                    │
                                                    ▼
                                                  model
```

The transcript is still the audit log. It is no longer the thing you send.

### The four moves

Anthropic's "Effective context engineering for AI agents" names the techniques; LangChain's docs group them similarly. In this course's vocabulary:

| Move | What it does | Example in the Acme run |
|---|---|---|
| Select (just-in-time) | Keep lightweight identifiers in context; load content only when a step needs it | `notion:refund-policy-v3` as a reference, fetched when drafting the refund |
| Offload | Put large results outside the window; leave a preview and a path | the 18,000-token Stripe export becomes a 10-line preview plus `/tool_results/call_7.txt` |
| Compress (compaction) | Replace old turns with a summary that provably keeps decisions, constraints, open issues | steps 1 to 30 become a 600-token briefing |
| Isolate (subagents) | Run a messy sub-task in its own context; return only a condensed result | a research subagent reads 12 Slack threads and returns 400 tokens |

Plus a fifth that runs through all of them: **structured note-taking**. The agent writes decisions, constraints, and open issues into a structured record outside the transcript, which survives every compaction because it is state, not conversation.

## Build it without a framework

The model is scripted in every cell below. The script stands in for a real model so the context mechanics (what is sent, what is dropped, what survives) are the object of study; a live cell at the end runs a real model. Token counts use a simple four-characters-per-token estimate, which is close to what `count_tokens_approximately` in `langchain_core` does.

### 1. A context builder with just-in-time retrieval

The builder takes everything the agent *could* show the model and produces what it *will* show for this step. Four rules, in order: pinned items always go in; references are listed by identifier; content is fetched only for references selected for this step, by relevance per token within the remaining budget; and the layout puts the stable prefix first (so the provider's prompt cache can reuse it; see 16. Resource-Aware Optimization) and restates the goal last (so it sits at the high-attention end).

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


import hashlib


def tok(text: str) -> int:
    return max(1, len(text) // 4)


SYSTEM = ("You are the ops agent for Acme Corp's account team. Tools: search_slack, "
          "stripe_charges, crm_lookup, read_ref. Refunds over $500 need human approval. ") * 4
NOTES = {"goal": "Resolve Acme's open billing/support issues before Friday's renewal; brief Dana.",
         "constraints": ["Refunds over $500 need human approval",
                         "Do not email the customer directly"],
         "decisions": ["D1: ch_2 is a duplicate charge ($49); refund it"],
         "open_issues": ["O1: SSO outage ticket #4471 has no owner"]}

FETCHES = []                                  # proves which references were actually loaded


def make_ref(ref_id, body, relevance):
    def fetch():
        FETCHES.append(ref_id)
        return body
    return {"id": ref_id, "tokens": tok(body), "relevance": relevance, "fetch": fetch}


REFS = [make_ref("notion:refund-policy-v3", "Refunds: duplicate charges refunded in full... " * 40, 0.9),
        make_ref("notion:refund-policy-v2", "(superseded) Refunds require a ticket... " * 40, 0.1),
        make_ref("notion:billing-schema", "Table invoices(id, customer, amount)... " * 120, 0.2),
        make_ref("slack:acme-sso-thread", "SSO failing since Tuesday for all admins... " * 30, 0.7)]


def build_context(step_instruction, recent, refs, notes, budget):
    pinned = [("system", SYSTEM), ("notes", repr(notes))]
    tail = [("recent", "\n".join(recent)), ("instruction", f"GOAL: {notes['goal']}\nNOW: {step_instruction}")]
    index = ("references", "Available (load with read_ref): " + ", ".join(r["id"] for r in refs))
    used = sum(tok(t) for _, t in pinned + tail + [index])
    assert used <= budget, "pinned content alone exceeds the budget: compact first"

    loaded = []
    for r in sorted(refs, key=lambda r: r["relevance"] / r["tokens"], reverse=True):
        if r["relevance"] >= 0.5 and used + r["tokens"] <= budget:   # relevant AND fits
            loaded.append((f"ref {r['id']}", r["fetch"]()))
            used += r["tokens"]
    sections = pinned + [index] + loaded + tail                         # stable prefix first
    return sections, used


step1, used1 = build_context("Draft the refund for ch_2.", ["tool: ch_2 confirmed duplicate"],
                             REFS, NOTES, budget=2_000)
step2, used2 = build_context("Assign an owner to SSO ticket #4471.", ["tool: refund queued"],
                             REFS, NOTES, budget=2_000)
prefix = lambda secs: hashlib.sha256(secs[0][1].encode()).hexdigest()[:12]

for name, text in step1:
    print(f"{name:32s} {tok(text):5d} tokens")
print("fetched:", FETCHES, "| used:", used1, "| stable prefix:", prefix(step1) == prefix(step2))
assert used1 <= 2_000 and used2 <= 2_000
assert "notion:billing-schema" not in FETCHES and "notion:refund-policy-v2" not in FETCHES
assert step1[-1][1].startswith("GOAL:") and "Refunds over $500" in step1[1][1]
assert prefix(step1) == prefix(step2)
```

The superseded policy page and the billing schema are *listed* but never *fetched*: the model knows they exist and can ask with `read_ref`, which is Anthropic's "lightweight identifiers" pattern. In production, relevance comes from retrieval (14. Knowledge Retrieval) and metadata (a `superseded` flag should simply exclude a document). And the constraints live in pinned `NOTES`, not in recent turns where they would scroll away.

### 2. Offloading large tool results

A tool result is the biggest single threat to the budget, because the tool author decided its size, not you. The fix is a wrapper at the tool-execution boundary: if a result is over a threshold, write it to a file (here a dict standing in for a filesystem or object store), put a preview and the path in the `ToolMessage`, and give the model tools to page through or search the file.

```{code-cell} python
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

FILES = {}
OFFLOAD_OVER = 500                            # tokens


def offload(call_id: str, result: str, preview_lines: int = 10) -> str:
    if tok(result) <= OFFLOAD_OVER:
        return result
    path = f"/tool_results/{call_id}.txt"
    FILES[path] = result
    lines = result.splitlines()
    return ("\n".join(lines[:preview_lines])
            + f"\n[... {len(lines)} lines, ~{tok(result)} tokens. Full result saved to {path}. "
              f"Use grep_file(path, pattern) or read_file(path, offset, limit).]")


@tool
def stripe_charges(customer: str) -> str:
    """Export all charges for a customer, one per line."""
    rows = [f"ch_{i:04d} 2026-0{1 + i % 9}-{1 + i % 27:02d} 4900 succeeded" for i in range(1, 400)]
    rows[241] = "ch_0242 2026-09-03 4900 succeeded DUPLICATE_OF ch_0241"
    return "\n".join(rows)


@tool
def grep_file(path: str, pattern: str) -> str:
    """Return lines in an offloaded file that contain the pattern."""
    return "\n".join(l for l in FILES[path].splitlines() if pattern in l)[:2000] or "no match"


@tool
def read_file(path: str, offset: int = 0, limit: int = 20) -> str:
    """Read `limit` lines of an offloaded file starting at `offset`."""
    return "\n".join(FILES[path].splitlines()[offset:offset + limit])


tools = {t.name: t for t in [stripe_charges, grep_file, read_file]}
model = scripted([("stripe_charges", {"customer": "cus_acme"})],
                 [("grep_file", {"path": "/tool_results/call_0.txt", "pattern": "DUPLICATE"})],
                 "ch_0242 duplicates ch_0241 on 2026-09-03.")

messages = [HumanMessage("Find any duplicate charge for Acme.")]
while True:
    ai = model.invoke(messages)
    messages.append(ai)
    if not ai.tool_calls:
        break
    for call in ai.tool_calls:
        raw = tools[call["name"]].invoke(call["args"])
        messages.append(ToolMessage(offload(call["id"], raw), tool_call_id=call["id"]))

raw_size = tok(stripe_charges.invoke({"customer": "cus_acme"}))
in_context = sum(tok(str(m.content)) for m in messages)
print(f"raw export ~{raw_size} tokens; whole transcript in context ~{in_context} tokens")
print(messages[-2].content)
assert in_context < raw_size / 5
assert "/tool_results/call_0.txt" in FILES
assert "ch_0242" in messages[-1].content
```

The model found the duplicate by searching a file it never saw in full, the same shape as a coding agent running `grep` instead of reading a repository into context. The preview must show *what kind* of data is in the file, and the pointer must name the tools that read it.

### 3. Compaction that cannot lose what matters

Compaction replaces old turns with a summary. The risk is obvious: a model-written summary can drop the one constraint that mattered. Anthropic's guidance is to preserve architectural decisions, unresolved bugs, and implementation details while discarding redundant tool output, and to tune the summarizer for recall first. We go one step further and make it **checked**: structured notes are the source of truth for decisions, constraints, and open issues, and after the model summarizes, code verifies that every note item survived and re-inserts any that did not.

The other correctness rule is mechanical and easy to get wrong: never cut between an `AIMessage` with tool calls and its `ToolMessage`s. Provider APIs reject a `ToolMessage` whose call is missing. The split point must move back to a turn boundary.

```{code-cell} python
from langchain_core.messages import SystemMessage


def build_history():
    h = [SystemMessage("You are the Acme ops agent."), HumanMessage(NOTES["goal"])]
    for i in range(12):
        h.append(AIMessage(content="", tool_calls=[{"name": "search_slack", "args": {"q": f"acme {i}"},
                                                   "id": f"c{i}", "type": "tool_call"}]))
        h.append(ToolMessage(f"result {i}: " + "slack message text " * 60, tool_call_id=f"c{i}"))
    return h


def safe_split(history, keep_last):
    """Index where the verbatim tail starts, moved back so no ToolMessage is orphaned."""
    i = max(2, len(history) - keep_last)
    while i > 2 and isinstance(history[i], ToolMessage):
        i -= 1
    return i


def note_items(notes):
    return notes["constraints"] + notes["decisions"] + notes["open_issues"]


def compact(history, notes, summarizer, keep_last=3):
    head, cut = history[:2], safe_split(history, keep_last)          # system + original request
    old, recent = history[2:cut], history[cut:]
    summary = summarizer.invoke([SystemMessage(
        "Summarize for continuation. Preserve decisions, constraints, open issues, next step."),
        *old]).content
    missing = [item for item in note_items(notes) if item not in summary]
    if missing:                                    # the check a plain summarizer does not have
        summary += "\nPINNED (restored by code):\n" + "\n".join(f"- {m}" for m in missing)
    return head + [HumanMessage(f"[Compacted history of {len(old)} messages]\n{summary}")] + recent, missing


# The scripted summarizer drops the $500 constraint, as real summarizers sometimes do.
lossy = scripted("Searched Slack 11 times. D1: ch_2 is a duplicate charge ($49); refund it. "
                 "Do not email the customer directly. O1: SSO outage ticket #4471 has no owner. "
                 "Next: queue refund, find SSO owner.")
history = build_history()
compacted, restored = compact(history, NOTES, lossy)
count = lambda msgs: sum(tok(str(m.content)) for m in msgs)

print(f"{len(history)} msgs / ~{count(history)} tokens -> {len(compacted)} msgs / ~{count(compacted)} tokens")
print("restored by code:", restored)
print("after summary:", [type(m).__name__ for m in compacted[3:]])
assert restored == ["Refunds over $500 need human approval"]
assert all(item in compacted[2].content for item in note_items(NOTES))
assert isinstance(compacted[3], AIMessage) and compacted[3].tool_calls   # tail starts on a turn boundary
assert count(compacted) < count(history) / 3
```

`keep_last=3` would have started the tail on a `ToolMessage`; `safe_split` moved it back one message so the tail begins with the `AIMessage` that made the call. And the summarizer silently dropped the refund-approval constraint; the check restored it.

The notes come from two places: the agent writing them through a tool (`record_decision`, `write_todos`, or a `NOTES.md` file, as in Anthropic's harness write-ups), and your code writing them from structured events (an approval granted, a ticket opened). The second is more reliable for constraints.

### 4. Subagents for context isolation

Some sub-tasks are messy by nature: read twelve Slack threads to find out what the customer is actually upset about. Doing that in the main context leaves 20,000 tokens of chatter behind. A subagent runs the sub-task in its **own** message list and returns only a condensed report. The main agent sees one tool call and one short result. Anthropic's guidance describes subagents returning condensed summaries on the order of 1,000 to 2,000 tokens from much larger explorations.

```{code-cell} python
THREADS = {f"t{i}": f"thread {i}: " + ("customer frustrated about SSO lockout and a double charge. " * 50)
           for i in range(1, 5)}


@tool
def read_thread(thread_id: str) -> str:
    """Read one Slack thread in full."""
    return THREADS[thread_id]


def research_subagent(question: str) -> tuple[str, int]:
    """Runs with its own context. Returns (report, tokens the subagent consumed)."""
    sub_model = scripted([("read_thread", {"thread_id": t}) for t in THREADS],
                         "Acme sentiment: two drivers. (1) SSO lockout since Tuesday, all admins, "
                         "4 threads. (2) Duplicate $49 charge on Sep 3. No churn threat stated.")
    sub_messages = [SystemMessage("Research only. Return <= 150 words."), HumanMessage(question)]
    while True:
        ai = sub_model.invoke(sub_messages)
        sub_messages.append(ai)
        if not ai.tool_calls:
            return ai.content, count(sub_messages)
        for c in ai.tool_calls:
            sub_messages.append(ToolMessage(read_thread.invoke(c["args"]), tool_call_id=c["id"]))


main_messages = [HumanMessage("Brief Dana on Acme's mood before the renewal call.")]
report, sub_tokens = research_subagent("What is Acme upset about? Read the Slack threads.")
main_messages.append(AIMessage(content="", tool_calls=[
    {"name": "research", "args": {"q": "acme mood"}, "id": "r1", "type": "tool_call"}]))
main_messages.append(ToolMessage(report, tool_call_id="r1"))

print(f"subagent consumed ~{sub_tokens} tokens; main context grew by ~{tok(report)} tokens")
print(report)
assert sub_tokens > 10 * tok(report)
assert "SSO" in report and "Duplicate" in report
```

Isolation is also a correctness tool: the subagent's dead ends never reach the main context, so they cannot bias later steps. The cost is lost detail: the main agent cannot ask about something the report omitted. Give subagents narrow questions and a required report shape (07. Multi-Agent Collaboration).

## How LangGraph does it

In LangChain 1.x these mechanisms are **middleware** on `create_agent`: hooks that run before each model call and can rewrite the request or the state. Two are built in and map one-to-one onto what we just wrote.

`ContextEditingMiddleware` with `ClearToolUsesEdit` is a crude offloader: once the context passes a token trigger, it replaces older tool results with a placeholder in the request sent to the model, keeping the most recent few. `SummarizationMiddleware` is the compactor: past a trigger (a token count, a message count, or a fraction of the model's window), it summarizes older messages with a model you choose and keeps a tail, and it keeps AI/tool message pairs together, like our `safe_split`.

```{code-cell} python
from langchain.agents import create_agent
from langchain.agents.middleware import (AgentMiddleware, ClearToolUsesEdit,
                                         ContextEditingMiddleware, SummarizationMiddleware)

SEEN = []


class RecordRequests(AgentMiddleware):
    """Records what each model call actually received. The view, not the log."""

    def wrap_model_call(self, request, handler):
        SEEN.append([(type(m).__name__, tok(str(m.content))) for m in request.messages])
        return handler(request)


@tool
def search_slack(q: str) -> str:
    """Search Slack messages about a customer."""
    return f"[{q}] " + "slack-hit " * 400


agent_model = scripted([("search_slack", {"q": "acme billing"})],
                       [("search_slack", {"q": "acme sso"})],
                       [("search_slack", {"q": "acme renewal"})],
                       "Briefing ready.")
summary_model = scripted("SESSION INTENT: brief Dana on Acme. DECISIONS: D1 refund ch_2. "
                         "CONSTRAINTS: refunds over $500 need approval. NEXT: SSO owner.")

agent = create_agent(agent_model, tools=[search_slack], middleware=[
    SummarizationMiddleware(model=summary_model, trigger=("tokens", 2_500), keep=("messages", 3)),
    ContextEditingMiddleware(edits=[ClearToolUsesEdit(trigger=1_500, keep=1)]),
    RecordRequests(),
])
out = agent.invoke({"messages": [HumanMessage("Brief Dana on Acme before the renewal.")]})

for i, req in enumerate(SEEN):
    print(f"model call {i}: {req}")
print("state[0]:", str(out["messages"][0].content)[:90], "...")
assert any(size < 10 for name, size in SEEN[2] if name == "ToolMessage")   # older result cleared in the view
assert "DECISIONS: D1" in str(out["messages"][0].content)                   # old turns compacted in state
assert out["messages"][-1].content == "Briefing ready."
```

On the third call, the first tool result arrives as a two-token `[cleared]` placeholder while the newest one is intact: that is `ClearToolUsesEdit` editing the *request*. The checkpointed transcript still holds the full result, which is the log/view split from the problem section. On the fourth call the token trigger fired, and the state's first message became the summary: `SummarizationMiddleware` rewrote *state*, not just the view. Context edits are reversible; summarization is permanent for that thread.

Mapped to the from-scratch code:

| From scratch | Framework |
|---|---|
| `build_context` (select, order, pin) | your own middleware (`wrap_model_call` / `before_model`), or a custom node before the model node in a LangGraph graph |
| `offload` + `grep_file` / `read_file` | Deep Agents' filesystem backend and its automatic large-result offloading; `ClearToolUsesEdit` is the lossy cousin |
| `compact` + `safe_split` | `SummarizationMiddleware` (its default prompt asks for intent, summary, artifacts, next steps) |
| the notes check after summarizing | not built in: add it as middleware, or keep constraints in state that is never summarized |
| `research_subagent` | Deep Agents' `task` tool with subagent specs, or a subgraph with its own state |
| `NOTES` in state | a custom state key, `write_todos` from `TodoListMiddleware`, or files in the agent's filesystem |

In a hand-built LangGraph graph, the same things are nodes and state keys: a `notes` key with replace semantics, a `compact` node on a conditional edge when the token count crosses a threshold, and `RemoveMessage` updates to delete old messages from the `add_messages` list.

**Deep Agents** packages all of this as a harness. Current Deep Agents docs (the 0.7 line, which this course pins) describe summarization triggering at a large fraction of the model's input window, tool results above a size threshold being written to a file and replaced by a short preview, oversized old tool inputs being truncated, and subagents via a built-in `task` tool that return a single final report. The live cell shows both the middleware stack with a real model and a Deep Agent; exact parameter names differ across Deep Agents versions, so treat the Deep Agents call as a sketch.

```{code-cell} python live
import os
from langchain.chat_models import init_chat_model

real = init_chat_model(os.environ.get("COURSE_MODEL", "openai:gpt-5.6"))
live = create_agent(real, tools=[search_slack, stripe_charges, grep_file, read_file], middleware=[
    SummarizationMiddleware(model=real, trigger=("fraction", 0.7), keep=("messages", 20)),
    ContextEditingMiddleware(edits=[ClearToolUsesEdit(trigger=30_000, keep=3)]),
])
print(live.invoke({"messages": [HumanMessage("Find Acme's duplicate charge and summarize Slack mood.")]}
                  )["messages"][-1].content)

from deepagents import create_deep_agent
deep = create_deep_agent(
    model=real, tools=[search_slack, stripe_charges],
    system_prompt="You are the Acme ops agent. Keep decisions and constraints in /notes.md.",
    subagents=[{"name": "slack-researcher", "description": "Reads many Slack threads, returns a short report.",
                "system_prompt": "Research only. Return at most 200 words.", "tools": [search_slack]}])
print(deep.invoke({"messages": [{"role": "user", "content": "Brief Dana on Acme before renewal."}]}
                  )["messages"][-1].content)
```

## What goes wrong in production

1. **Summaries that drop constraints.** Symptom: after a long run, the agent issues a $700 refund without approval. Cause: compaction summarized away a sentence from step 2. Fix: constraints live in pinned structured state, never only in conversation; verify every note item survives compaction; and enforce the rule in the tool layer anyway (18. Guardrails), because context is never enforcement.

2. **Orphaned tool messages.** Symptom: a 400 error from the provider right after compaction or trimming. Cause: the cut landed between an `AIMessage` with tool calls and its `ToolMessage`s. Fix: split only on turn boundaries; test compaction with histories that end mid-tool-call.

3. **Prompt-cache misses from a moving prefix.** Symptom: cost per step doubles after a "harmless" change. Cause: a timestamp, a per-step counter, or reordered tools at the top of the context invalidates the cached prefix on every call. Fix: stable content first and byte-identical across steps (system prompt, tool schemas in deterministic order), volatile content last. MCP's current spec asks servers to list tools in deterministic order for the same reason.

4. **Offloaded data the model cannot find.** Symptom: the agent re-runs the expensive export instead of reading the file. Cause: the preview did not say what was in the file or how to read it. Fix: previews that show structure (header, first rows, row count) and name the read and search tools; evaluate on tasks that require the offloaded data.

5. **Just-in-time retrieval that retrieves stale truth.** Symptom: the briefing cites the superseded policy. Cause: retrieval ranked by similarity with no notion of validity. Fix: metadata filters (superseded, effective dates, permissions) before relevance; one current version per document id.

6. **Subagent reports that omit the key detail.** Symptom: the main agent confidently states "no churn risk" because the subagent's summary said so, but a thread contained a cancellation threat. Cause: an unconstrained report format. Fix: a required report schema (findings, evidence ids, open questions, confidence), and the ability to re-query the subagent or read the cited evidence.

7. **Compaction thrash.** Symptom: every other step triggers a summarization call. Cause: the trigger sits too close to the post-compaction size. Fix: hysteresis (trigger high, compact well below it) and offload big results so compaction is rare.

## What changed by 2026

- **"Context engineering" became the name of the job.** Anthropic's guidance frames it as curating the smallest high-signal context for each step, with just-in-time retrieval, compaction, structured note-taking, and subagents as the core long-horizon techniques.
- **Harnesses own context by default.** LangChain ships context middleware (`SummarizationMiddleware`, `ContextEditingMiddleware`, `TodoListMiddleware`, filesystem and subagent middleware) for `create_agent`, and Deep Agents turns them on as a package. Current Deep Agents docs also report a much smaller default prompt and tool-description footprint in the 0.7 line: the harness's own tokens are part of the budget too.
- **Offloading to files is standard.** Coding harnesses treat the filesystem as extended memory: large outputs go to files, the model greps and pages through them. Anthropic's tool-design guidance recommends truncation, pagination, and concise response formats at the tool level, and reports that Claude Code caps tool results at 25,000 tokens by default.
- **Fresh context with handoff artifacts, sometimes over compaction.** Anthropic's write-up on harness design for long-running application development reports that, for their setup, resetting to a fresh context with a structured handoff artifact worked better than compacting in place. The general lesson: the notes and the progress file are the durable state; the context window is disposable.

## Interview questions

**Q1. What is context engineering, and how is it different from prompt engineering?**

Strong: Prompt engineering is about wording the instructions. Context engineering is deciding what the model sees at each step of a multi-step run: which instructions, tools, memories, retrieved documents, prior results, and in what form and order. For an agent, the context is recomputed every step from durable state, and the transcript becomes an audit log rather than the thing you send. The core techniques are just-in-time retrieval, offloading large results, compaction, structured notes, and subagent isolation, all aimed at the smallest high-signal context for the current step.

Weak: "It's prompt engineering for agents."

**Q2. Your agent works for 20 steps and then degrades. What do you check?**

Strong: First, the context size and composition per step from the trace: usually it is stuffed with old tool results. Second, whether the goal and constraints are still prominent or buried in the middle. Third, whether stale or superseded content is still present and being treated as current. Fourth, whether compaction ran and what it dropped. The fixes are offloading large results with previews, pinning goal and constraints as structured state rendered every step, compacting with checked preservation of decisions and open issues, and isolating messy sub-tasks in subagents.

Weak: "Switch to a model with a longer context window."

**Q3. How would you implement compaction safely?**

Strong: Trigger on a token threshold with hysteresis so it does not thrash. Split only on turn boundaries so no tool result loses its call. Keep the system prompt, the original request, and a verbatim tail of recent turns. Summarize the middle with a prompt that asks for decisions, constraints, open issues, artifacts, and next step, tuned for recall. Then verify: decisions, constraints, and open issues also live in structured notes, and code checks they all appear in the compacted context and restores any that are missing. I log restorations as a quality signal for the summarizer.

Weak: "Ask the model to summarize the conversation when it gets long."

**Q4. A tool returns 50,000 tokens. What do you do?**

Strong: Fix it at the tool if I own it: pagination, filters, a concise response mode, semantic fields instead of raw dumps. At the harness boundary, anything over a threshold is written to a file or object store and replaced with a preview that shows its structure, its size, and the path, plus read and search tools the model can use. The model then pulls only the lines it needs. I make sure evals cover tasks that require the offloaded data, because a preview the model cannot navigate just causes re-runs.

Weak: "Truncate it to the first 2,000 tokens."

**Q5. Design memory and context for a long-running agent: what do you store, where, and how do you retrieve it?**

Strong: I separate the log from the view. The full transcript is persisted by the checkpointer for audit and recovery. Working state (goal, constraints, decisions, open issues, plan) is structured state, rendered into every step's context. Large artifacts go to files and enter context through previews and read tools. Cross-session facts go to a long-term store with namespaces, retrieved just in time by relevance and filtered by validity and permissions. Each step's context is built from these by a function I can test: stable prefix first for caching, pinned state, just-in-time content, recent turns, goal restated last.

Weak: "Put everything in a vector database."

**Q6. When do you use a subagent for context reasons, and what does it cost?**

Strong: When a sub-task needs a lot of reading or trial-and-error whose details the main agent does not need: researching many threads, searching a codebase, exploring a data set. The subagent burns its own context and returns a condensed report, so the main context stays clean and unbiased by dead ends. The costs are extra latency and tokens, lost detail if the report omits something, and no shared state unless I pass it. So I give subagents narrow questions, the relevant constraints, and a required report schema with evidence ids.

Weak: "Subagents are better because they're specialized."

**Q7. Why does the order of content in the context matter?**

Strong: Two reasons. Prompt caching reuses computation for an identical prefix, so stable content (system prompt, tool schemas in a fixed order) goes first and volatile content last, or every step pays full price. And models use information at the start and end of long contexts better than the middle, the "Lost in the Middle" result, so the goal and the current instruction go at the end, near the question, and the bulky retrieved material sits in between.

Weak: "It doesn't, the model reads all of it."

## Build it yourself

Combine the four mechanisms into one `ContextManager` class that the loop from 00. What an agent actually is calls before every model invocation:

1. `ContextManager.render(state) -> list[BaseMessage]` builds the per-step context: stable system prefix, pinned notes, reference index, just-in-time refs, recent turns, goal last.
2. Tool results pass through `offload` automatically; the manager exposes `grep_file` and `read_file` as tools.
3. When the rendered context exceeds 80% of the budget, it compacts to 40% with `compact`, including the notes check.
4. Add a `record_note(kind, text)` tool the model can call; notes it records must survive compaction.

Acceptance checks:

- Over a scripted 30-step run with a 300-line tool result every third step, no rendered context exceeds the budget.
- The first system message is byte-identical across all 30 renders.
- A note recorded at step 3 is present in the rendered context at step 30.
- No rendered context starts its tail with a `ToolMessage` whose call is missing.
- Compaction runs at most three times (hysteresis works).

## Key takeaways

- The model sees only the context of the current call, so each step's context should be computed from durable state, not appended to forever.
- Attention is a budget: long, noisy contexts degrade recall, and the middle of a long context is used least well.
- Keep references as identifiers and load content just in time; offload large tool results to files with a preview and read tools.
- Compaction must split on turn boundaries and verifiably preserve decisions, constraints, and open issues, which should also live in structured notes.
- Subagents isolate messy sub-tasks and return condensed reports, trading detail and latency for a clean main context.
- In LangChain these are middleware: `ContextEditingMiddleware` edits the request view, `SummarizationMiddleware` rewrites state, and Deep Agents bundles both with filesystem offloading and subagents.
- Put stable content first for prompt caching and the goal last for attention, and never treat context as enforcement.
