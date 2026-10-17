# 28. The Production Agent Harness: everything around the loop

*Track A (agent engineering) with Track D (infra + interviews). Course addition beyond the book; it assembles Chapters 5 (Tool Use), 6 (Planning), 8 (Memory Management), 13 (Human in the Loop), and 18 (Guardrails) into one runtime. Prerequisites: 00. What an agent actually is, 34. Context Engineering, 13. Human in the Loop. Companions: 33. Sandboxed Agent Execution, 36. Durable Agent Orchestration, 41. Skills.*

## The problem

In 00. What an agent actually is you built the loop in twenty lines. It answered "compute 231 × 482" in two tool calls. Now give the same loop a real ops task:

> "Acme says they were double charged on invoice in_1042. Find the duplicate, refund it, and post a summary in #billing."

Run it and watch it fail, one piece at a time:

```text
step  what the naive loop does                         what breaks
----------------------------------------------------------------------------------------
 2    stripe_search_charges("acme") -> 200 charges     16 KB of JSON enters the transcript and
                                                       stays there for every later model call
 9    transcript is 60 KB                              quality drops; the model re-searches
                                                       things it already found (context rot)
 14   model forgets step 3 of its own plan             the plan only existed as prose in an
                                                       old message
 15   stripe_refund(ch_042, 12000)                     money moved with no human approval
 16   worker is redeployed mid-run                     the run restarts from step 1 and
                                                       refunds AGAIN
 --   customer asks "why did it do that?"              nothing was recorded but the final text
```

None of these failures is about the model being unintelligent. Each is missing software around the model. That software has a name now: the **harness**.

```text
                           ┌───────────────────── HARNESS ─────────────────────┐
                           │                                                    │
  user task ─────────────► │   context      planning     permissions   budgets  │
                           │   manager      (todos)      (allow/ask/   (steps,  │
                           │   (offload,                  deny)         tokens, │
                           │    compact)                                $)      │
                           │        │           │            │            │     │
                           │        ▼           ▼            ▼            ▼     │
                           │   ┌──────────── THE LOOP (lesson 00) ───────────┐  │
                           │   │  model ──► tool calls ──► results ──► model │  │
                           │   └─────────────────────────────────────────────┘  │
                           │        │           │            │            │     │
                           │   checkpoints   subagents    sandbox      tracing  │
                           │   (resume)      (isolated    (where code  (every   │
                           │                  context)     runs)        step)   │
                           └────────────────────────────────────────────────────┘
```

LangChain's docs put the equation plainly: **Agent = Model + Harness**. The model supplies judgment. The harness supplies everything that makes judgment safe to act on over hours instead of seconds.

### What a harness owns

| Component | Question it answers | Failure without it |
|---|---|---|
| Loop | What runs next, and when do we stop? | runs forever, or stops early |
| Context manager | What does the model see this turn? | overflow, context rot, re-work |
| Tools | What can the model ask for? | wrong granularity, flooding results |
| Planning | Where does the plan live? | forgotten steps after long runs |
| Permissions | Which calls run, pause, or are refused? | unapproved side effects |
| Sandbox | Where does model-written code execute? | code runs on your host with your credentials |
| Checkpoints | Where do we resume after a crash? | restarts, duplicate side effects |
| Subagents | How is a big task split with clean context? | one transcript holding everything |
| Tracing | What happened, step by step? | undebuggable, unauditable |

Every row is a lesson elsewhere in this course. This lesson is about how they fit together into one runtime, and about knowing which harness to pick versus which to build.

## Build it without a framework

We build a minimal harness that owns the loop, context, planning, permissions, checkpoints, subagents, and a trace. Only the sandbox is left out (see 33. Sandboxed Agent Execution). As always, the scripted model replays fixed turns so the control flow is what we study. The same harness shape runs against a real model in the live cell at the end.

### The environment: tools with real consequences

Three tools from the running example. One returns a large result. One moves money and carries an idempotency key. The `SIDE_EFFECTS` list is our view of the outside world, so we can assert "exactly once."

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
from langchain_core.tools import tool

# Fake systems of record. The side-effect log is how we check "exactly once".
SIDE_EFFECTS = []
CHARGES = [{"id": f"ch_{i:03d}", "customer": "acme", "amount_cents": 12_000 if i in (41, 42) else 900 + i,
            "invoice": "in_1042" if i in (41, 42) else f"in_{2000 + i}"} for i in range(200)]


@tool
def stripe_search_charges(customer: str) -> str:
    """List every charge for a customer, one JSON object per line. Large output."""
    return "\n".join(json.dumps(c) for c in CHARGES if c["customer"] == customer)


@tool
def stripe_refund(charge_id: str, amount_cents: int, idempotency_key: str) -> str:
    """Refund a charge. Side-effecting; requires human approval."""
    if any(e[0] == "refund" and e[3] == idempotency_key for e in SIDE_EFFECTS):
        return f"already refunded (idempotency_key={idempotency_key})"
    SIDE_EFFECTS.append(("refund", charge_id, amount_cents, idempotency_key))
    return f"refunded {amount_cents} on {charge_id}"


@tool
def slack_post(channel: str, text: str) -> str:
    """Post a message to a Slack channel."""
    SIDE_EFFECTS.append(("slack", channel, text, None))
    return "posted"


TOOLS = {t.name: t for t in [stripe_search_charges, stripe_refund, slack_post]}
raw = stripe_search_charges.invoke({"customer": "acme"})
print(f"{len(TOOLS)} tools; one search returns {len(raw):,} chars (~{len(raw)//4:,} tokens)")
assert len(raw) > 10_000
```

Four thousand tokens from one call. If the loop appends that to the transcript, every later model call pays for it and the model has to find two relevant lines in two hundred.

### Context and durability primitives

Three small functions carry most of the weight.

**Offload.** A large result is written to a virtual filesystem, and the model receives a path and a preview. It can search the file later with a tool. This is the "just-in-time retrieval via lightweight identifiers" idea from Anthropic's context-engineering guidance, applied to tool output.

**Compact.** When the transcript is over budget, old turns are replaced by a summary that keeps the task, the tools used, and the notes. The subtle bug here is the cut point. If you drop an `AIMessage` with tool calls but keep its `ToolMessage` results, provider APIs reject the request because the result has no matching call. The cut must move forward past orphaned results.

**Checkpoint.** The whole state serializes to JSON. Not "the messages plus some Python objects that happened to be in memory." Everything needed to resume on a different machine.

```{code-cell} python
from langchain_core.messages import (HumanMessage, SystemMessage, ToolMessage,
                                     messages_from_dict, messages_to_dict)

POLICY = {"stripe_search_charges": "allow", "slack_post": "allow", "stripe_refund": "ask"}
OFFLOAD_CHARS = 2_000        # tool results bigger than this go to the filesystem
CONTEXT_BUDGET_CHARS = 6_000  # compact the transcript above this size


def offload(call_id, result, fs):
    """Large results become a file plus a preview; the model can read_file later."""
    if len(result) <= OFFLOAD_CHARS:
        return result
    path = f"/tool_results/{call_id}.json"
    fs[path] = result
    return f"[{len(result):,} chars saved to {path}] preview: {result[:200]}..."


def size(messages):
    return sum(len(str(m.content)) + len(json.dumps(getattr(m, "tool_calls", []))) for m in messages)


def compact(messages, notes, keep_last=4):
    """Replace old turns with a summary. Never split an AI tool call from its results."""
    if size(messages) <= CONTEXT_BUDGET_CHARS or len(messages) <= keep_last + 1:
        return messages, False
    cut = len(messages) - keep_last
    while cut < len(messages) and isinstance(messages[cut], ToolMessage):
        cut += 1                     # an orphaned ToolMessage is rejected by provider APIs
    old, recent = messages[1:cut], messages[cut:]
    tools_used = [c["name"] for m in old for c in getattr(m, "tool_calls", [])]
    summary = HumanMessage(
        f"[compacted {len(old)} messages] task: {messages[0].content} | "
        f"tools used so far: {tools_used} | notes: {notes}")
    return [messages[0], summary, *recent], True


def save_checkpoint(state):
    """Everything needed to resume, as plain JSON (prod: Postgres, keyed by thread_id)."""
    return json.dumps({**state, "messages": messages_to_dict(state["messages"])})


def load_checkpoint(blob):
    state = json.loads(blob)
    state["messages"] = messages_from_dict(state["messages"])
    return state


demo_fs = {}
preview = offload("call_x", raw, demo_fs)
print(preview[:90])
msgs = [HumanMessage("task")] + [AIMessage("", tool_calls=[{"name": "t", "args": {}, "id": f"c{i}", "type": "tool_call"}])
        if i % 2 == 0 else ToolMessage("r" * 1500, tool_call_id=f"c{i-1}") for i in range(8)]
compacted, did = compact(msgs, notes="none")
print(f"compacted {len(msgs)} -> {len(compacted)} messages; first kept after summary: {type(compacted[2]).__name__}")
assert "/tool_results/call_x.json" in demo_fs and len(preview) < 300
assert did and not isinstance(compacted[2], ToolMessage)
```

### The harness loop

Now the loop itself. Compare it line by line with `run_agent` from lesson 00. The skeleton is identical: invoke the model, run tool calls, feed results back, stop when there are no calls. Everything added is harness.

```{code-cell} python
SYSTEM = "You are the billing ops agent. Keep your todo list current. Refunds need approval."


def pending_calls(state):
    """Tool calls from the last AI turn that have no result yet (the resume point)."""
    msgs = state["messages"]
    turns = [k for k, m in enumerate(msgs) if isinstance(m, AIMessage) and m.tool_calls]
    if not turns:
        return []
    answered = sum(1 for m in msgs[turns[-1] + 1:] if isinstance(m, ToolMessage))
    return msgs[turns[-1]].tool_calls[answered:]


class Harness:
    """Loop + context + planning + permissions + checkpoints + trace, nothing hidden."""

    def __init__(self, model, tools, policy, store, max_steps=12):
        self.model, self.tools, self.policy = model, tools, policy
        self.store, self.max_steps = store, max_steps   # store: thread_id -> JSON blob

    def new_state(self, task):
        return {"messages": [HumanMessage(task)], "todos": [], "fs": {}, "trace": [],
                "status": "running", "step": 0}

    # --- built-in harness tools: the model plans and reads files through these ---
    def builtin(self, state, name, args):
        if name == "write_todos":
            state["todos"] = args["todos"]           # the plan lives in STATE, not the transcript
            return f"todos updated ({len(args['todos'])} items)"
        if name == "grep_file":
            lines = state["fs"].get(args["path"], "").splitlines()
            return "\n".join(ln for ln in lines if args["pattern"] in ln) or "no matches"
        return None

    def act(self, state, decision):
        """Run the not-yet-answered calls of the last AI turn. Returns False to pause."""
        for call in pending_calls(state):
            name, args = call["name"], call["args"]
            out = self.builtin(state, name, args)
            if out is None:
                rule = self.policy.get(name, "deny")        # default-deny unknown tools
                if rule == "deny" or name not in self.tools:
                    out = f"error: tool {name} is not permitted"
                elif rule == "ask":
                    if decision is None:                     # pause; the checkpoint holds the call
                        state["status"] = "awaiting_approval"
                        state["trace"].append(("pause", name, args))
                        return False
                    out = ("rejected by human reviewer; do not retry" if decision == "reject"
                           else self.tools[name].invoke(args))
                    decision = None                          # one decision covers one call
                else:
                    out = self.tools[name].invoke(args)
            out = offload(f"step{state['step']}_{call['id']}", str(out), state["fs"])
            state["messages"].append(ToolMessage(out, tool_call_id=call["id"]))
            state["trace"].append(("tool", name, len(out)))
        return True

    def run(self, state, thread_id, decision=None):
        state["status"] = "running"
        while True:
            if pending_calls(state):
                ok = self.act(state, decision)
                decision = None
                self.store[thread_id] = save_checkpoint(state)   # checkpoint every step
                if not ok:
                    return state
            if state["step"] >= self.max_steps:
                state["status"] = "budget_exhausted"
                return state
            state["messages"], did = compact(state["messages"], notes=state["todos"])
            if did:
                state["trace"].append(("compact", len(state["messages"])))
            system = SystemMessage(f"{SYSTEM}\nCurrent todos: {state['todos']}")  # re-injected every turn
            ai = self.model.invoke([system, *state["messages"]])
            state["messages"].append(ai)
            state["step"] += 1
            state["trace"].append(("model", [c["name"] for c in ai.tool_calls]))
            self.store[thread_id] = save_checkpoint(state)
            if not ai.tool_calls:
                state["status"] = "done"
                return state


STORE = {}
first_process = Harness(scripted(
    [("write_todos", {"todos": ["find duplicate charge", "refund duplicate", "post to #billing"]})],
    [("stripe_search_charges", {"customer": "acme"})],
    [("grep_file", {"path": "/tool_results/step2_call_0.json", "pattern": "in_1042"})],
    [("stripe_refund", {"charge_id": "ch_042", "amount_cents": 12000, "idempotency_key": "acme-in_1042-dup"})],
), TOOLS, POLICY, STORE)

state = first_process.run(first_process.new_state("Acme was double charged on in_1042. Fix it."), "acme-1")
print("status:", state["status"], "| todos:", state["todos"])
assert state["status"] == "awaiting_approval" and SIDE_EFFECTS == []

# The process dies here. A different worker, with a fresh model client, resumes from the store.
del first_process, state
second_process = Harness(scripted(
    [("slack_post", {"channel": "#billing", "text": "Refunded duplicate charge ch_042 (in_1042)."})],
    "Refunded the duplicate $120.00 charge on in_1042 and posted to #billing.",
), TOOLS, POLICY, STORE)
state = second_process.run(load_checkpoint(STORE["acme-1"]), "acme-1", decision="approve")

for event in state["trace"]:
    print(" ", event)
print("final:", state["messages"][-1].content)
assert state["status"] == "done"
assert [e[0] for e in SIDE_EFFECTS] == ["refund", "slack"]      # refund exactly once
assert any(p.startswith("/tool_results/") for p in state["fs"])   # big result never hit the context
assert max(len(str(m.content)) for m in state["messages"]) < OFFLOAD_CHARS
```

Read the trace, because it is the whole lesson in twelve lines:

```text
model   write_todos                 the plan is written to STATE, not left in prose
tool    write_todos
model   stripe_search_charges
tool    stripe_search_charges  268  16 KB result became a 268-char preview + a file
model   grep_file                   just-in-time retrieval from the offloaded file
tool    grep_file              163
model   stripe_refund
pause   stripe_refund               policy "ask": checkpoint written, process exits
   ...  process dies; another worker loads the checkpoint ...
tool    stripe_refund               executed once, after approval
model   slack_post
tool    slack_post
model   (no calls)                  done
```

Four design decisions are worth saying out loud in an interview.

**The plan is state.** `write_todos` does nothing clever. It writes a list into `state["todos"]`, and the harness re-injects that list into the system message every turn. Compaction can delete every old message and the plan survives. That is why Claude Code and Deep Agents both expose a todo tool: it is structured note-taking with a guaranteed place in context.

**Permissions live in code with default-deny.** An unknown tool name gets `"deny"`. The model cannot talk its way past `self.policy`. `"ask"` is not a prompt instruction; it is the loop refusing to proceed.

**Pausing is just returning.** When approval is needed the harness writes a checkpoint and returns. There is no thread blocked waiting for a human. Resuming is "load the checkpoint, find the tool calls that have no result yet, continue." `pending_calls` is the resume point, and it is exactly what LangGraph's checkpointer plus `interrupt` compute for you.

**Idempotency is the tool's job, not the harness's.** The checkpoint is written *after* each step, so a crash between "Stripe executed the refund" and "checkpoint saved" would replay the call on resume. The `idempotency_key` makes that replay harmless. Durable execution does not remove the need for idempotent side effects; it makes it mandatory (36. Durable Agent Orchestration).

### Subagents: context isolation as a tool

The last component is delegation. A subagent is a harness running inside a tool call. It gets a fresh transcript, its own filesystem, a narrower tool set, and it returns one condensed report. The parent never sees the 16 KB search result or the child's intermediate steps.

```{code-cell} python
READ_ONLY = {"stripe_search_charges": TOOLS["stripe_search_charges"]}


class HarnessWithSubagents(Harness):
    """Adds a `task` tool: a child harness with its own context and fewer permissions."""

    def __init__(self, *args, child_model, **kwargs):
        super().__init__(*args, **kwargs)
        self.child_model = child_model

    def builtin(self, state, name, args):
        if name != "task":
            return super().builtin(state, name, args)
        child = Harness(self.child_model, READ_ONLY, {"stripe_search_charges": "allow"},
                        store={}, max_steps=5)
        result = child.run(child.new_state(args["description"]), thread_id="child")
        state["trace"].append(("subagent", result["step"], size(result["messages"])))
        return result["messages"][-1].content      # only the condensed report crosses back


SIDE_EFFECTS.clear()
child_model = scripted(
    [("stripe_search_charges", {"customer": "acme"})],
    [("grep_file", {"path": "/tool_results/step1_call_0.json", "pattern": "in_1042"})],
    "Duplicate: ch_041 and ch_042 both charge 12000 cents on in_1042. Refund ch_042.",
)
parent = HarnessWithSubagents(scripted(
    [("task", {"description": "Find duplicate Acme charges on in_1042. Report ids only."})],
    "Research found the duplicate (ch_042). Awaiting refund approval next.",
), TOOLS, POLICY, {}, child_model=child_model)
state = parent.run(parent.new_state("Investigate Acme's double charge."), "acme-2")

sub = next(e for e in state["trace"] if e[0] == "subagent")
print(f"child: {sub[1]} model calls, {sub[2]:,} chars of context")
print(f"parent: {size(state['messages']):,} chars of context")
print("report:", state["messages"][2].content)
assert state["fs"] == {}                          # the 16 KB result lives only in the child
assert size(state["messages"]) < sub[2] / 2
assert SIDE_EFFECTS == []                         # the child had no write tools at all
```

Two properties matter. First, **context isolation**: the parent's context is under half of what the child used, and it would be far smaller with a real research task. Anthropic's guidance describes subagents returning condensed summaries of roughly 1,000 to 2,000 tokens for this reason. Second, **permission narrowing**: the child physically cannot refund anything, because `stripe_refund` is not in its tool map. Delegation is also a least-privilege boundary.

The cost is real. Every subagent is at least one extra model call, and its report is lossy by design. Use subagents when the work produces a lot of intermediate context the parent does not need (search, reading, exploration), not as an org-chart metaphor.

## How LangGraph does it

LangChain 1.x's `create_agent` is described in its docs as "a minimal, highly configurable agent harness." It is the loop from lesson 00 compiled to a LangGraph graph, plus **middleware**: hooks that run before the model, after the model, or wrapped around each tool call. Each harness component we built becomes one middleware.

| From scratch | `create_agent` |
|---|---|
| `offload()` inside `act()` | a `@wrap_tool_call` middleware around every tool execution |
| `compact()` before each model call | `SummarizationMiddleware` (a `before_model` hook) |
| `write_todos` + re-injection | `TodoListMiddleware` |
| `policy` with `"ask"` + checkpoint + return | `HumanInTheLoopMiddleware` + a checkpointer + `interrupt` |
| `max_steps` | `ModelCallLimitMiddleware`, `ToolCallLimitMiddleware`, and the graph's recursion limit (1000 super-steps by default in current docs) |
| `save_checkpoint` / `load_checkpoint` | the checkpointer (`InMemorySaver`, Postgres, ...) keyed by `thread_id` |
| `task` subagent tool | Deep Agents' `task` tool (below) |
| `state["trace"]` | LangSmith tracing (32. Observability and Tracing) |

```{code-cell} python
from langchain.agents import create_agent
from langchain.agents.middleware import (HumanInTheLoopMiddleware, ModelCallLimitMiddleware,
                                         ToolCallLimitMiddleware, wrap_tool_call)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

SIDE_EFFECTS.clear()


@wrap_tool_call
def offload_large_results(request, handler):
    """Same job as offload(): wraps every tool execution, like our act() did."""
    result = handler(request)
    if isinstance(result, ToolMessage) and len(str(result.content)) > OFFLOAD_CHARS:
        return ToolMessage(f"[{len(result.content):,} chars offloaded] preview: {result.content[:200]}...",
                           tool_call_id=result.tool_call_id, name=result.name)
    return result


model = scripted(
    [("stripe_search_charges", {"customer": "acme"})],
    [("stripe_refund", {"charge_id": "ch_042", "amount_cents": 12000, "idempotency_key": "acme-in_1042-dup"})],
    "Refunded the duplicate charge on in_1042.",
)
agent = create_agent(
    model,
    tools=list(TOOLS.values()),
    system_prompt=SYSTEM,
    middleware=[
        offload_large_results,                                   # context
        HumanInTheLoopMiddleware(interrupt_on={"stripe_refund": True}),  # permissions
        ModelCallLimitMiddleware(run_limit=12),                  # budgets
        ToolCallLimitMiddleware(tool_name="stripe_refund", run_limit=1),
    ],
    checkpointer=InMemorySaver(),                                # durability
)
config = {"configurable": {"thread_id": "acme-3"}}

paused = agent.invoke({"messages": [HumanMessage("Acme was double charged on in_1042. Fix it.")]}, config)
request = paused["__interrupt__"][0].value["action_requests"][0]
print("paused for:", request["name"], request["args"]["charge_id"])
assert SIDE_EFFECTS == []

done = agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config)
print("final:", done["messages"][-1].content)
print("graph nodes:", [n for n in agent.get_graph().nodes if n not in ("__start__", "__end__")])
assert [e[0] for e in SIDE_EFFECTS] == ["refund"]
assert max(len(str(m.content)) for m in done["messages"]) < OFFLOAD_CHARS
```

Look at the printed node list. The middleware did not disappear into a black box; each hook is a node in the compiled graph (`HumanInTheLoopMiddleware.after_model`, `ModelCallLimitMiddleware.before_model`, and so on). When the model's turn contains a `stripe_refund` call, the `after_model` hook calls `interrupt`, the checkpointer saves the state, and `invoke` returns with `__interrupt__`. `Command(resume=...)` loads that checkpoint and continues. That is our `awaiting_approval` return plus `load_checkpoint`, with a production storage backend.

### Deep Agents: the batteries-included harness

Deep Agents is LangChain's harness built on `create_agent` and LangGraph. Its docs describe it as "the same core tool calling loop as other agent frameworks," plus planning, a filesystem for context management, and subagents. In other words: the middleware stack you just saw, preconfigured.

Current Deep Agents 0.7 docs describe:

- **A filesystem abstraction** with tools such as `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, and `execute`, backed by pluggable backends (in-state, local disk, a LangGraph store, a sandbox, or a composite that routes paths like `/memories/` to long-term storage).
- **Automatic context management**: summarization near the model's input limit, and tool results above roughly 20,000 tokens offloaded to a file with a short preview. That is `offload()` and `compact()`.
- **Subagents through a built-in `task` tool**, defined as dicts (name, description, system prompt, tools, model), which can run in parallel and return one final report. That is `HarnessWithSubagents`.
- **Human-in-the-loop** via `interrupt_on`, requiring a checkpointer.
- **Skills** loaded by progressive disclosure (41. Skills) and AGENTS.md-style memory files.
- **Sandbox backends** for code execution (33. Sandboxed Agent Execution).

The 0.7 release notes frame it as "a leaner, more configurable harness by default": the todo middleware became opt-in, the base system prompt became empty, and default-turn input tokens dropped by about 65%. That is the Anthropic finding below, shipped as a release: as models get better, the harness should do less.

This course pins the `deepagents` 0.7 line. The snippet below runs only as a live cell because it calls a real model, and exact argument names should be checked against the docs for your installed version.

```{code-cell} python live
import os
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent   # optional package; argument names per current docs

llm = init_chat_model(os.environ.get("COURSE_MODEL", "openai:gpt-5.6"))
researcher = {
    "name": "charge-researcher",
    "description": "Finds duplicate or disputed charges for a customer. Read-only.",
    "system_prompt": "Search charges, identify duplicates, return charge ids and amounts only.",
    "tools": [stripe_search_charges],
}
deep = create_deep_agent(
    model=llm,
    tools=list(TOOLS.values()),
    system_prompt=SYSTEM,
    subagents=[researcher],
    interrupt_on={"stripe_refund": True},
    checkpointer=InMemorySaver(),
)
cfg = {"configurable": {"thread_id": "acme-live"}}
out = deep.invoke({"messages": [{"role": "user", "content":
                   "Acme was double charged on in_1042. Find the duplicate and refund it."}]}, cfg)
print("interrupt:", out.get("__interrupt__"))

# The OpenAI Agents SDK harness, same tools, different defaults (tracing, sessions, handoffs).
from agents import Agent, Runner, function_tool

@function_tool
def search_charges(customer: str) -> str:
    """List every charge for a customer, one JSON object per line."""
    return stripe_search_charges.invoke({"customer": customer})


oa_agent = Agent(name="billing-ops", instructions=SYSTEM, tools=[search_charges])
print(Runner.run_sync(oa_agent, "How many charges does acme have on invoice in_1042?").final_output)
```

The OpenAI Agents SDK is a harness too, with a different center of gravity: handoffs between agents, guardrails, sessions for conversation state, tracing on by default, and, per its current docs, sandbox agents with persistent workspaces. Claude Code is the harness most engineers have used without thinking of it as one: a loop, a todo tool, permission modes, compaction, subagents, hooks, and skills. The components are the same; the defaults differ.

### Which harness, or none?

```text
Is the path known in advance?
  yes ──► workflow graph (StateGraph). No harness needed.
  no  ──► Is the task short (minutes, < ~20 tool calls, one context window)?
            yes ──► create_agent with the middleware you need
            no  ──► long-horizon: files, subagents, compaction, sandbox
                      ├── LangGraph shop / need custom graph ──► Deep Agents
                      ├── OpenAI-centric, handoffs            ──► OpenAI Agents SDK
                      └── coding/ops on a repo or machine      ──► Claude Agent SDK / Claude Code
```

The one wrong answer is building all of this from scratch in production because the framework "hides things." You now know what it hides. Build from scratch to learn and to evaluate; adopt a harness to ship, and replace one component at a time when a measured failure demands it.

## What goes wrong in production

1. **Resume replays side effects.** Symptom: a customer gets two refunds after a deploy. Cause: the crash happened after the tool ran and before the checkpoint was saved, so the resumed run re-executed it. With LangGraph `interrupt`, the interrupted node also re-runs from the top on resume. Fix: idempotency keys on every side-effecting tool, and no side effects before an `interrupt` in the same node.

2. **Compaction deletes the thing that mattered.** Symptom: after a long run, the agent re-asks for an approval it already received, or contradicts an earlier decision. Cause: the summary kept "tools used" but dropped decisions and open issues. Fix: summaries that explicitly carry decisions, constraints, and open items; durable notes (todos, a progress file) outside the transcript; and an eval that runs long tasks past the compaction threshold.

3. **Context resets without a handoff.** Symptom: a fresh-context session "starts over" and redoes work. Cause: the reset threw away state that lived only in messages. Fix: Anthropic's long-running-harness pattern: a progress file, a feature list with pass/fail, and version-control commits as the durable handoff artifacts each new session reads first.

4. **Permission policy by prompt.** Symptom: an agent posts to a customer-facing Slack channel it was "told" not to use. Cause: the rule was in the system prompt. Fix: enforce in the harness (policy map, default-deny, argument-level rules such as allowed channels), and keep credentials out of anything the model-written code can read.

5. **Harness drift from the model.** Symptom: a model upgrade makes the agent slower and no better. Cause: scaffolding written for a weaker model (forced planning steps, verbose prompts, mandatory reflection) now gets in the way. Fix: ablate harness components against your eval suite on every model change and remove what no longer earns its tokens.

6. **Checkpoints that are not really state.** Symptom: resume works in tests and fails in production. Cause: something the loop needed (a file handle, an in-memory cache, a client session) was not in the serialized state. Fix: make the checkpoint the single source of truth, and test resume on a different process as we did above.

7. **No per-step trace.** Symptom: "why did it refund?" has no answer. Cause: only the final message was logged. Fix: trace every model call, tool call, approval decision, and compaction event with the thread id, and keep the approval record as an audit artifact (39. Agent Identity and Governance).

## What changed by 2026

- **"Harness" became the unit of design.** LangChain's own docs now describe three layers: LangGraph as the runtime, LangChain's `create_agent` as a minimal harness, and Deep Agents as a batteries-included harness in the same category as the Claude Agent SDK. Interview questions moved from "what is ReAct" to "what does your harness own."
- **Anthropic published harness-design findings.** Its long-running application work uses a **planner / generator / evaluator** split: a planner expands the request into a spec, the generator builds in sprints against a "sprint contract" that defines done before coding, and a separate evaluator grades against rubrics, because a model praising its own work is unreliable. It reports that **fresh context resets with handoff artifacts** worked better than compaction for those long runs, and that harness components should be **removed as models improve**. Its managed-agents design splits a durable, append-only **session** log, a stateless **harness** loop, and an interchangeable **sandbox**, with credentials never entering the sandbox.
- **Harnesses got leaner.** Deep Agents 0.7 made planning opt-in and emptied its base prompt. The direction across vendors is fewer default components and more configuration, because stronger models need less scaffolding and every default costs tokens on every turn.
- **Hosted harnesses arrived.** OpenAI describes an Agents API (in beta) where it runs orchestration, compaction, and recovery server-side, alongside the open-source Agents SDK. The build-versus-buy line for harnesses now looks like the one for databases.
- **Reasoning models moved planning inside the model.** Explicit "think step by step" scaffolding is not a harness feature anymore. The harness's job is to state the goal, the constraints, and what counts as done, and then verify outcomes with tests and graders rather than trusting the model's narrated reasoning.

## Interview questions

**Q1. What is an agent harness, and what does it own?**

Strong: A harness is everything around the model that turns a tool-calling loop into something you can run for hours in production. It owns the loop and its termination, what goes into context each turn, the tool registry, where the plan lives, permissions for each call, the sandbox where generated code runs, checkpoints for resume, subagents for context isolation, and tracing. LangChain's framing is "agent equals model plus harness." When I evaluate one I ask where each of those lives and whether I can replace it.

Weak: "It's the framework you use to build agents."

**Q2. Your agent works for five steps and degrades after forty. What do you change in the harness? (reported: Cognition)**

Strong: That is context rot, so I look at what the model sees at step forty. First, offload large tool results to files with a preview and let the model search them. Second, move the plan and key decisions out of the transcript into state that is re-injected every turn, like a todo list or a notes file. Third, add compaction that preserves decisions and open issues, or for very long runs a context reset with a handoff artifact. Fourth, push context-heavy exploration into subagents that return short reports. Then I rerun long-task evals to confirm it held.

Weak: "Use a model with a bigger context window."

**Q3. How do you make an agent resumable after a crash without duplicating side effects? (reported: OpenAI, Palantir)**

Strong: Checkpoint the full state after every step, keyed by a thread id, in durable storage, and make the checkpoint the only source of truth. On resume, find the tool calls from the last model turn that have no recorded result and run only those. The gap between "tool executed" and "checkpoint saved" means some calls may run twice, so every side-effecting tool takes an idempotency key derived from the task, not a random one. In LangGraph, that is a checkpointer plus interrupt, and the rule that the interrupted node re-runs from the top.

Weak: "Save the conversation to a database."

**Q4. When would you use a subagent, and what does it cost?**

Strong: When a piece of work generates lots of intermediate context the parent does not need, like searching two hundred charges or reading a codebase. The subagent gets a clean context and narrower tools and returns a short report, so the parent's context stays small and the child cannot perform writes it does not need. The cost is extra model calls, latency, and a lossy handoff, so I use it for context-heavy work rather than splitting every task into roles, and I budget it.

Weak: "Multiple agents are better than one because they specialize."

**Q5. Why separate the evaluator from the generator in a long-running harness?**

Strong: Because a model grading its own output is unreliable; it tends to approve its own work. Anthropic's harness-design write-up splits planner, generator, and evaluator, agrees a contract that defines done before building, and has the evaluator grade against explicit rubrics and actual checks, like running the app. The general principle is to verify outcomes with an independent check, not with the generator's own narration, which is also why reading chain-of-thought is not verification.

Weak: "Ask the model to double-check its answer."

**Q6. Build a harness or adopt one?**

Strong: Adopt one to ship, and understand it well enough to replace pieces. The components are standard and the edge cases, like tool-call pairing during compaction, resume semantics, and approval payloads, are where bugs live. I would start with `create_agent` and only the middleware I need, move to Deep Agents or another batteries-included harness for long-horizon work, and build a custom component only when an eval shows a specific failure the default cannot fix. Building one from scratch once is how I learned what to look for.

Weak: "Always build your own so you have control."

**Q7. A new model is 20% better on your evals. What do you do with the harness?**

Strong: Re-run the eval suite with harness components ablated one at a time: forced planning, reflection passes, verbose system prompt sections, aggressive compaction. Scaffolding that helped a weaker model can cost tokens and latency or even hurt a stronger one. Anthropic's guidance and the Deep Agents 0.7 defaults both point the same way: remove components as models improve. I keep what still measurably helps and delete the rest.

Weak: "Swap the model name and ship."

## Build it yourself

Add a generator/evaluator gate and a context reset to the from-scratch `Harness`:

1. Add an `evaluator` callable that inspects the final state and returns `(passed, feedback)`. It must check the outside world (`SIDE_EFFECTS`), not the model's final message.
2. If the evaluator fails, append the feedback as a `HumanMessage` and continue the loop, at most twice.
3. Add `reset_with_handoff(state)`: write a `/progress.md` file to `state["fs"]` containing todos and completed tool calls, then replace the transcript with the original task plus the contents of that file.
4. Add a `"deny"` rule for `slack_post` to a channel outside an allowlist, enforced by argument inspection.

Acceptance checks:

- A scripted run that claims "Refunded" without calling `stripe_refund` is caught by the evaluator, and the next model turn receives the feedback.
- After two failed evaluations the status is `"evaluation_failed"`, never `"done"`.
- After `reset_with_handoff`, `len(state["messages"]) == 2` and `/progress.md` lists the tools already run.
- `slack_post(channel="#general", ...)` produces an error `ToolMessage` and no entry in `SIDE_EFFECTS`.
- Resuming the reset state in a new `Harness` instance from its JSON checkpoint completes the task with the refund recorded exactly once.

## Key takeaways

- Agent equals model plus harness: the loop is twenty lines, and the harness is everything that makes it survive production.
- A harness owns loop, context, tools, planning, permissions, sandbox, checkpoints, subagents, and tracing; know where each lives in any harness you adopt.
- Keep plans and decisions in state that is re-injected every turn, and offload large results to files, so compaction and resets cannot erase them.
- Pausing for approval is checkpoint-and-return; resuming is running the tool calls that have no result yet, and idempotency keys make the replay safe.
- Subagents are context isolation and least privilege, paid for in extra calls.
- `create_agent` middleware and Deep Agents are the same components you built, so choose by the task's horizon and replace pieces only when evals demand it.
- As models improve, the right harness gets smaller: ablate scaffolding on every model change.
